from __future__ import annotations

from fastapi.testclient import TestClient

from app.api import create_app
from app.config import Settings
from app.llm.gemini import GeminiProvider
from app.prompt_builder import TRIGGER_INSTRUCTIONS
from app.trigger_router import ROUTING_RULES


def test_every_routable_trigger_kind_has_a_prompt_instruction():
    # PromptBuilder.build() does TRIGGER_INSTRUCTIONS[brief.trigger_kind]
    # unguarded, outside AIComposer's try/except. If a kind is routable
    # (present in ROUTING_RULES, i.e. TriggerRouter will happily build a
    # MessageBrief for it) but missing here, the live Gemini path raises an
    # uncaught KeyError for every trigger of that kind -- silently dropped
    # by TickService's broad except, with no fabrication risk but also no
    # message ever sent for a kind that may have real, gradeable facts
    # attached (e.g. research_digest, the brief's own flagship example).
    missing = set(ROUTING_RULES) - set(TRIGGER_INSTRUCTIONS)
    assert not missing, f"trigger kinds routable but missing a prompt instruction: {sorted(missing)}"


def make_client(tmp_path, **settings_overrides) -> TestClient:
    settings = Settings(database_path=str(tmp_path / "contexts.db"), **settings_overrides)
    return TestClient(create_app(settings))


def context_payload(scope: str, context_id: str, version: int, payload: dict) -> dict:
    return {
        "scope": scope,
        "context_id": context_id,
        "version": version,
        "payload": payload,
        "delivered_at": "2026-04-26T10:00:00Z",
    }


def seed_perf_dip_trigger(client: TestClient) -> str:
    """Minimal category/merchant/trigger set the deterministic fallback can send for."""
    client.post("/v1/context", json=context_payload("category", "salons", 1, {
        "slug": "salons",
        "voice": {"tone": "friendly"},
    }))
    client.post("/v1/context", json=context_payload("merchant", "m_test", 1, {
        "category_slug": "salons",
        "identity": {"name": "Test Salon", "city": "Pune", "locality": "Kothrud", "languages": ["en"]},
        "offers": [],
        "performance": {},
        "signals": [],
    }))
    client.post("/v1/context", json=context_payload("trigger", "trig_test_1", 1, {
        "kind": "perf_dip",
        "urgency": 5,
        "suppression_key": "perf_dip:m_test:2026-W01",
        "merchant_id": "m_test",
        "scope": "merchant",
        "payload": {"metric": "calls", "delta_pct": -0.3, "window": "7d"},
    }))
    return "trig_test_1"


def test_tick_works_without_gemini_key(tmp_path):
    with make_client(tmp_path) as client:
        # No GEMINI_API_KEY configured -> the live composer must have no
        # provider, i.e. it will use the deterministic fallback.
        assert client.app.state.composer._provider is None

        trigger_id = seed_perf_dip_trigger(client)
        response = client.post("/v1/tick", json={"now": "2026-04-26T10:00:00Z", "available_triggers": [trigger_id]})

    assert response.status_code == 200
    actions = response.json()["actions"]
    assert len(actions) == 1
    assert actions[0]["body"]
    assert actions[0]["trigger_id"] == trigger_id


def test_live_path_wires_configured_gemini_provider(tmp_path):
    with make_client(
        tmp_path,
        gemini_api_key="fake-test-key",
        gemini_model="gemini-test-model",
    ) as client:
        composer = client.app.state.composer
        assert isinstance(composer._provider, GeminiProvider)
        assert composer._provider.available is True
        # The configured model setting must reach the live provider.
        assert composer._provider._model == "gemini-test-model"


def test_tick_still_works_with_missing_context_and_no_key(tmp_path):
    # Safety: an unknown trigger id must never crash /v1/tick, with or
    # without a Gemini key configured.
    with make_client(tmp_path) as client:
        response = client.post("/v1/tick", json={"now": "2026-04-26T10:00:00Z", "available_triggers": ["does_not_exist"]})
    assert response.status_code == 200
    assert response.json()["actions"] == []
