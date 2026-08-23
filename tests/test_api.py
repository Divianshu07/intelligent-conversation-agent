from __future__ import annotations

from fastapi.testclient import TestClient

from app.api import create_app
from app.config import Settings


def make_client(tmp_path) -> TestClient:
    settings = Settings(
        database_path=str(tmp_path / "contexts.db"),
        team_name="Test Team",
        team_members="Ada, Grace",
        model="test-model",
        approach="test foundation",
        contact_email="test@example.com",
        version="test-1.0.0",
        submitted_at="2026-04-26T08:00:00Z",
    )
    return TestClient(create_app(settings))


def context_payload(scope: str, context_id: str, version: int, payload: dict) -> dict:
    return {
        "scope": scope,
        "context_id": context_id,
        "version": version,
        "payload": payload,
        "delivered_at": "2026-04-26T10:00:00Z",
    }


def test_health_endpoint_starts_with_empty_contexts(tmp_path):
    with make_client(tmp_path) as client:
        response = client.get("/v1/healthz")
    assert response.status_code == 200
    assert response.json()["status"] == "ok"
    assert response.json()["contexts_loaded"] == {"category": 0, "merchant": 0, "customer": 0, "trigger": 0}


def test_metadata_endpoint_uses_configuration(tmp_path):
    with make_client(tmp_path) as client:
        response = client.get("/v1/metadata")
    assert response.status_code == 200
    assert response.json()["team_name"] == "Test Team"
    assert response.json()["team_members"] == ["Ada", "Grace"]


def test_context_is_inserted_and_retrievable_internally(tmp_path):
    with make_client(tmp_path) as client:
        response = client.post("/v1/context", json=context_payload("merchant", "m_001", 1, {"name": "Clinic"}))
        assert response.status_code == 200
        stored = client.app.state.store.get("merchant", "m_001")
    assert stored is not None
    assert stored.version == 1
    assert stored.payload == {"name": "Clinic"}


def test_newer_context_version_replaces_previous_value(tmp_path):
    with make_client(tmp_path) as client:
        client.post("/v1/context", json=context_payload("merchant", "m_001", 1, {"views": 100}))
        response = client.post("/v1/context", json=context_payload("merchant", "m_001", 2, {"views": 200}))
        stored = client.app.state.store.get("merchant", "m_001")
    assert response.status_code == 200
    assert stored.version == 2
    assert stored.payload == {"views": 200}


def test_older_context_version_is_rejected_without_overwrite(tmp_path):
    with make_client(tmp_path) as client:
        client.post("/v1/context", json=context_payload("merchant", "m_001", 3, {"views": 300}))
        response = client.post("/v1/context", json=context_payload("merchant", "m_001", 2, {"views": 200}))
        stored = client.app.state.store.get("merchant", "m_001")
    assert response.status_code == 409
    assert response.json() == {"accepted": False, "ack_id": None, "stored_at": None, "reason": "stale_version", "current_version": 3, "details": None}
    assert stored.version == 3
    assert stored.payload == {"views": 300}


def test_context_counts_cover_all_scopes(tmp_path):
    with make_client(tmp_path) as client:
        for scope in ("category", "merchant", "customer", "trigger"):
            response = client.post("/v1/context", json=context_payload(scope, f"{scope}_1", 1, {"scope": scope}))
            assert response.status_code == 200
        health = client.get("/v1/healthz")
    assert health.json()["contexts_loaded"] == {"category": 1, "merchant": 1, "customer": 1, "trigger": 1}
