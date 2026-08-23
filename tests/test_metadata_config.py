from __future__ import annotations

from fastapi.testclient import TestClient

from app.api import create_app
from app.config import Settings

PLACEHOLDER_STRINGS = (
    "Team Placeholder",
    "Your Name",
    "not_configured",
    "composer not implemented",
    "team@example.com",
)


def make_client(tmp_path, **overrides) -> TestClient:
    settings = Settings(database_path=str(tmp_path / "contexts.db"), **overrides)
    return TestClient(create_app(settings))


def test_metadata_has_no_placeholder_values_by_default(tmp_path):
    # No env overrides at all (matching an unconfigured deployment / no
    # GEMINI_API_KEY) — this is the exact scenario the old defaults leaked.
    with make_client(tmp_path) as client:
        response = client.get("/v1/metadata")

    assert response.status_code == 200
    body = response.json()

    for value in (body["team_name"], body["contact_email"], body["submitted_at"]):
        assert value in (None, "")

    assert body["team_members"] == []

    for field_name in ("model", "approach"):
        for placeholder in PLACEHOLDER_STRINGS:
            assert placeholder not in body[field_name]

    # The approach must no longer claim the composer isn't implemented.
    assert "not implemented" not in body["approach"].lower()


def test_metadata_reflects_configured_values(tmp_path):
    with make_client(
        tmp_path,
        team_name="Team Alpha",
        team_members="Alice, Bob",
        contact_email="alice@example.com",
        submitted_at="2026-08-23T09:00:00Z",
    ) as client:
        response = client.get("/v1/metadata")

    assert response.status_code == 200
    body = response.json()
    assert body["team_name"] == "Team Alpha"
    assert body["team_members"] == ["Alice", "Bob"]
    assert body["contact_email"] == "alice@example.com"
    assert body["submitted_at"] == "2026-08-23T09:00:00Z"


def test_metadata_model_reflects_gemini_model_without_api_key(tmp_path):
    # No GEMINI_API_KEY configured -> the reported model must still be a
    # truthful, non-placeholder value (the configured/default Gemini model),
    # not "not_configured".
    with make_client(tmp_path, gemini_model="gemini-2.5-flash") as client:
        response = client.get("/v1/metadata")

    assert response.status_code == 200
    assert response.json()["model"] == "gemini-2.5-flash"


def test_metadata_model_reflects_explicit_override_when_set(tmp_path):
    with make_client(tmp_path, model="claude-opus-4-8", gemini_model="gemini-2.5-flash") as client:
        response = client.get("/v1/metadata")

    assert response.status_code == 200
    assert response.json()["model"] == "claude-opus-4-8"


def test_metadata_never_exposes_gemini_api_key(tmp_path):
    with make_client(tmp_path, gemini_api_key="super-secret-key") as client:
        response = client.get("/v1/metadata")

    assert response.status_code == 200
    assert "super-secret-key" not in response.text
    assert "gemini_api_key" not in response.json()
