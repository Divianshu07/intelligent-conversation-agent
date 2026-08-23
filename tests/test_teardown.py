from __future__ import annotations

from fastapi.testclient import TestClient

from app.api import create_app
from app.config import Settings


def make_client(tmp_path) -> TestClient:
    settings = Settings(database_path=str(tmp_path / "contexts.db"))
    return TestClient(create_app(settings))


def context_payload(scope: str, context_id: str, version: int, payload: dict) -> dict:
    return {
        "scope": scope,
        "context_id": context_id,
        "version": version,
        "payload": payload,
        "delivered_at": "2026-04-26T10:00:00Z",
    }


def reply_payload(conversation_id: str, message: str, turn_number: int) -> dict:
    return {
        "conversation_id": conversation_id,
        "merchant_id": "m_001",
        "customer_id": None,
        "from_role": "merchant",
        "message": message,
        "received_at": "2026-04-26T10:45:00Z",
        "turn_number": turn_number,
    }


def test_teardown_succeeds(tmp_path):
    with make_client(tmp_path) as client:
        response = client.post("/v1/teardown")
    assert response.status_code == 200
    assert response.json()["status"] == "ok"


def test_teardown_removes_previously_stored_contexts(tmp_path):
    with make_client(tmp_path) as client:
        for scope in ("category", "merchant", "customer", "trigger"):
            client.post("/v1/context", json=context_payload(scope, f"{scope}_1", 1, {"scope": scope}))
        store = client.app.state.store

        response = client.post("/v1/teardown")
        assert response.status_code == 200

        for scope in ("category", "merchant", "customer", "trigger"):
            assert store.get(scope, f"{scope}_1") is None


def test_teardown_removes_conversation_state(tmp_path):
    with make_client(tmp_path) as client:
        client.post("/v1/reply", json=reply_payload("conv_teardown", "Hello, following up.", 1))
        store = client.app.state.store
        assert store.get("conversation", "conv_teardown") is not None

        response = client.post("/v1/teardown")
        assert response.status_code == 200

        assert store.get("conversation", "conv_teardown") is None


def test_healthz_remains_safe_after_teardown(tmp_path):
    with make_client(tmp_path) as client:
        client.post("/v1/context", json=context_payload("merchant", "m_1", 1, {"name": "Test"}))
        client.post("/v1/teardown")

        health = client.get("/v1/healthz")

    assert health.status_code == 200
    assert health.json()["status"] == "ok"
    assert health.json()["contexts_loaded"] == {"category": 0, "merchant": 0, "customer": 0, "trigger": 0}


def test_teardown_on_already_empty_store_does_not_crash(tmp_path):
    with make_client(tmp_path) as client:
        first = client.post("/v1/teardown")
        second = client.post("/v1/teardown")
    assert first.status_code == 200
    assert second.status_code == 200
    assert second.json()["status"] == "ok"
