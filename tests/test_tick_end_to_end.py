from __future__ import annotations

from fastapi.testclient import TestClient

from app.api import create_app
from app.composer import AIComposer
from app.config import Settings
from app.llm.gemini import GeminiUnavailableError

# --- Fixed pieces of ROUTING_RULES data used to build valid ComposerOutput
# candidates for the fake provider (mirrors app/trigger_router.py exactly,
# so the fake responses are indistinguishable from a real successful Gemini
# call as far as OutputValidator is concerned).
PERF_DIP_OBJECTIVE = "acknowledge the supplied decline and invite investigation or action"
PERF_DIP_CTA = "open_ended"
CUSTOMER_LAPSED_HARD_OBJECTIVE = "win back the customer"
CUSTOMER_LAPSED_HARD_CTA = "binary_yes_no"


class FakeStructuredLLM:
    """Deterministic stand-in for GeminiProvider. Never makes a real network
    call; either returns a scripted structured response or raises, so tests
    can exercise both the "successful LLM" and "provider failure -> fallback"
    paths through the real AIComposer/OutputValidator."""

    def __init__(self, response: dict | None = None, exc: Exception | None = None) -> None:
        self._response = response
        self._exc = exc
        self.call_count = 0

    def generate(self, system_prompt: str, prompt: str, response_schema: dict) -> dict:
        self.call_count += 1
        if self._exc is not None:
            raise self._exc
        assert self._response is not None
        return self._response


def make_client(tmp_path) -> TestClient:
    settings = Settings(database_path=str(tmp_path / "contexts.db"))
    return TestClient(create_app(settings))


def use_fake_provider(client: TestClient, **fake_kwargs) -> FakeStructuredLLM:
    fake = FakeStructuredLLM(**fake_kwargs)
    client.app.state.composer = AIComposer(provider=fake)
    return fake


def context_payload(scope: str, context_id: str, version: int, payload: dict) -> dict:
    return {
        "scope": scope,
        "context_id": context_id,
        "version": version,
        "payload": payload,
        "delivered_at": "2026-04-26T10:00:00Z",
    }


def seed_category(client: TestClient, slug: str = "salons") -> None:
    response = client.post("/v1/context", json=context_payload("category", slug, 1, {
        "slug": slug,
        "voice": {"tone": "friendly"},
    }))
    assert response.status_code == 200


def seed_merchant(client: TestClient, merchant_id: str, category_slug: str = "salons") -> None:
    response = client.post("/v1/context", json=context_payload("merchant", merchant_id, 1, {
        "category_slug": category_slug,
        "identity": {"name": "Test Merchant", "city": "Pune", "locality": "Kothrud", "languages": ["en"]},
        "offers": [],
        "performance": {},
        "signals": [],
    }))
    assert response.status_code == 200


def seed_customer(client: TestClient, customer_id: str, *, promotional_consent: bool) -> None:
    response = client.post("/v1/context", json=context_payload("customer", customer_id, 1, {
        "identity": {"name": "Test Customer", "language_pref": "en"},
        "state": {},
        "relationship": {},
        "preferences": {"channel": "whatsapp"},
        "consent": {"scope": ["promotional_offers"] if promotional_consent else []},
    }))
    assert response.status_code == 200


def seed_perf_dip_trigger(client: TestClient, trigger_id: str, merchant_id: str) -> None:
    response = client.post("/v1/context", json=context_payload("trigger", trigger_id, 1, {
        "kind": "perf_dip",
        "urgency": 5,
        "suppression_key": f"perf_dip:{merchant_id}:2026-W01",
        "merchant_id": merchant_id,
        "scope": "merchant",
        "payload": {"metric": "calls", "delta_pct": -0.3, "window": "7d"},
    }))
    assert response.status_code == 200


def seed_customer_lapsed_trigger(client: TestClient, trigger_id: str, merchant_id: str, customer_id: str) -> None:
    response = client.post("/v1/context", json=context_payload("trigger", trigger_id, 1, {
        "kind": "customer_lapsed_hard",
        "urgency": 3,
        "suppression_key": f"winback:{customer_id}",
        "merchant_id": merchant_id,
        "customer_id": customer_id,
        "scope": "customer",
        "payload": {"days_since_last_visit": 57, "previous_focus": "weight_loss"},
    }))
    assert response.status_code == 200


def tick(client: TestClient, trigger_ids: list[str]):
    return client.post("/v1/tick", json={"now": "2026-04-26T10:00:00Z", "available_triggers": trigger_ids})


# --- 1. A merchant trigger that should produce a send action, composed by a
#        (fake) successful LLM call and passing through the real validator.
def test_merchant_trigger_produces_send_via_successful_llm(tmp_path):
    with make_client(tmp_path) as client:
        seed_category(client)
        seed_merchant(client, "m_merchant_send")
        seed_perf_dip_trigger(client, "trig_merchant_send", "m_merchant_send")

        fake = use_fake_provider(
            client,
            response={
                "message": "Test Merchant, calls are down 30% over 7d. Want help figuring out what changed?",
                "audience": "merchant",
                "language": "en",
                "objective": PERF_DIP_OBJECTIVE,
                "cta_type": PERF_DIP_CTA,
                "action": "send",
                "facts_used": ["merchant_name", "metric", "delta_pct", "window"],
                "confidence": 0.9,
                "should_send": True,
            },
        )

        response = tick(client, ["trig_merchant_send"])

    assert response.status_code == 200
    actions = response.json()["actions"]
    assert len(actions) == 1
    assert actions[0]["trigger_id"] == "trig_merchant_send"
    assert actions[0]["send_as"] == "vera"
    assert "30%" in actions[0]["body"]
    assert fake.call_count == 1  # the live path actually invoked the provider


# --- 2. A customer trigger with the appropriate consent, composed by a
#        (fake) successful LLM call and passing through the real validator.
def test_customer_trigger_with_consent_produces_send_via_successful_llm(tmp_path):
    with make_client(tmp_path) as client:
        seed_category(client)
        seed_merchant(client, "m_customer_send")
        seed_customer(client, "c_customer_send", promotional_consent=True)
        seed_customer_lapsed_trigger(client, "trig_customer_send", "m_customer_send", "c_customer_send")

        fake = use_fake_provider(
            client,
            response={
                "message": "Hi Test Customer, it's been 57 days since your last visit. Want to come back in?",
                "audience": "customer",
                "language": "en",
                "objective": CUSTOMER_LAPSED_HARD_OBJECTIVE,
                "cta_type": CUSTOMER_LAPSED_HARD_CTA,
                "action": "send",
                "facts_used": ["customer_name", "days_since_last_visit"],
                "confidence": 0.9,
                "should_send": True,
            },
        )

        response = tick(client, ["trig_customer_send"])

    assert response.status_code == 200
    actions = response.json()["actions"]
    assert len(actions) == 1
    assert actions[0]["customer_id"] == "c_customer_send"
    assert actions[0]["send_as"] == "merchant_on_behalf"
    assert "Test Customer" in actions[0]["body"]
    assert fake.call_count == 1


# --- 3. Missing required facts: the (fake) LLM references a fact outside the
#        brief's known_facts. The real OutputValidator must reject it and the
#        endpoint must safely wait (no fabricated send), not crash.
def test_llm_hallucinated_fact_is_rejected_and_safely_waits(tmp_path):
    with make_client(tmp_path) as client:
        seed_category(client)
        seed_merchant(client, "m_hallucination")
        seed_perf_dip_trigger(client, "trig_hallucination", "m_hallucination")

        use_fake_provider(
            client,
            response={
                "message": "Test Merchant, calls are down 30% over 7d, driven by a new competitor nearby.",
                "audience": "merchant",
                "language": "en",
                "objective": PERF_DIP_OBJECTIVE,
                "cta_type": PERF_DIP_CTA,
                "action": "send",
                # "competitor_name" is not in this brief's known_facts at all.
                "facts_used": ["merchant_name", "metric", "delta_pct", "window", "competitor_name"],
                "confidence": 0.9,
                "should_send": True,
            },
        )

        response = tick(client, ["trig_hallucination"])

    assert response.status_code == 200
    assert response.json()["actions"] == []


# --- 4. An unknown trigger ID must not crash the endpoint.
def test_unknown_trigger_id_does_not_crash(tmp_path):
    with make_client(tmp_path) as client:
        response = tick(client, ["does_not_exist"])
    assert response.status_code == 200
    assert response.json()["actions"] == []


# --- 5. Multiple available triggers in one request: a sendable one, a
#        hallucination-rejected one, and an unknown one, all together.
def test_multiple_available_triggers_in_one_request(tmp_path):
    with make_client(tmp_path) as client:
        seed_category(client)
        seed_merchant(client, "m_multi")
        seed_perf_dip_trigger(client, "trig_multi_send", "m_multi")
        seed_perf_dip_trigger(client, "trig_multi_wait", "m_multi")

        fake = FakeStructuredLLM(response=None)

        def generate(system_prompt: str, prompt: str, response_schema: dict) -> dict:
            fake.call_count += 1
            if "trig_multi_send" in prompt or fake.call_count == 1:
                return {
                    "message": "Test Merchant, calls are down 30% over 7d. Want help figuring out what changed?",
                    "audience": "merchant",
                    "language": "en",
                    "objective": PERF_DIP_OBJECTIVE,
                    "cta_type": PERF_DIP_CTA,
                    "action": "send",
                    "facts_used": ["merchant_name", "metric", "delta_pct", "window"],
                    "confidence": 0.9,
                    "should_send": True,
                }
            return {
                "message": "Test Merchant, calls are down 30%, likely due to a rival opening up.",
                "audience": "merchant",
                "language": "en",
                "objective": PERF_DIP_OBJECTIVE,
                "cta_type": PERF_DIP_CTA,
                "action": "send",
                "facts_used": ["merchant_name", "metric", "delta_pct", "window", "competitor_name"],
                "confidence": 0.9,
                "should_send": True,
            }

        fake.generate = generate  # type: ignore[assignment]
        client.app.state.composer = AIComposer(provider=fake)

        response = tick(client, ["trig_multi_send", "trig_multi_wait", "does_not_exist"])

    assert response.status_code == 200
    actions = response.json()["actions"]
    assert len(actions) == 1
    assert actions[0]["trigger_id"] == "trig_multi_send"


# --- 6b. Anti-repetition: the same trigger (same suppression_key) showing
#        up as "available" in a later tick must not produce a second,
#        verbatim-duplicate send. This mirrors the judge harness, which
#        re-lists still-active triggers across consecutive 5-min ticks.
def test_repeated_tick_with_same_trigger_does_not_resend(tmp_path):
    with make_client(tmp_path) as client:
        seed_category(client)
        seed_merchant(client, "m_repeat")
        seed_perf_dip_trigger(client, "trig_repeat", "m_repeat")

        fake = use_fake_provider(
            client,
            response={
                "message": "Test Merchant, calls are down 30% over 7d. Want help figuring out what changed?",
                "audience": "merchant",
                "language": "en",
                "objective": PERF_DIP_OBJECTIVE,
                "cta_type": PERF_DIP_CTA,
                "action": "send",
                "facts_used": ["merchant_name", "metric", "delta_pct", "window"],
                "confidence": 0.9,
                "should_send": True,
            },
        )

        first = tick(client, ["trig_repeat"])
        second = tick(client, ["trig_repeat"])
        third = tick(client, ["trig_repeat"])

    assert first.status_code == second.status_code == third.status_code == 200
    first_actions = first.json()["actions"]
    assert len(first_actions) == 1
    assert first_actions[0]["trigger_id"] == "trig_repeat"

    # No further sends for the same suppression_key on later ticks.
    assert second.json()["actions"] == []
    assert third.json()["actions"] == []
    # The composer/LLM is only ever invoked once — the second and third
    # ticks are suppressed before composition, not merely deduped after.
    assert fake.call_count == 1


# --- 6c. Different triggers (distinct suppression_keys) for different
#        merchants must still both send in the same tick — dedup must be
#        scoped to the suppression_key, not global.
def test_distinct_suppression_keys_both_send_independently(tmp_path):
    with make_client(tmp_path) as client:
        seed_category(client)
        seed_merchant(client, "m_repeat_a")
        seed_merchant(client, "m_repeat_b")
        seed_perf_dip_trigger(client, "trig_repeat_a", "m_repeat_a")
        seed_perf_dip_trigger(client, "trig_repeat_b", "m_repeat_b")

        use_fake_provider(
            client,
            response={
                "message": "Test Merchant, calls are down 30% over 7d. Want help figuring out what changed?",
                "audience": "merchant",
                "language": "en",
                "objective": PERF_DIP_OBJECTIVE,
                "cta_type": PERF_DIP_CTA,
                "action": "send",
                "facts_used": ["merchant_name", "metric", "delta_pct", "window"],
                "confidence": 0.9,
                "should_send": True,
            },
        )

        response = tick(client, ["trig_repeat_a", "trig_repeat_b"])

    assert response.status_code == 200
    actions = response.json()["actions"]
    assert {action["trigger_id"] for action in actions} == {"trig_repeat_a", "trig_repeat_b"}


# --- 6d. Teardown must clear suppression state so a fresh test run isn't
#        permanently blocked from resending a previously-suppressed key.
def test_teardown_clears_suppression_state(tmp_path):
    with make_client(tmp_path) as client:
        seed_category(client)
        seed_merchant(client, "m_repeat_teardown")
        seed_perf_dip_trigger(client, "trig_repeat_teardown", "m_repeat_teardown")

        use_fake_provider(
            client,
            response={
                "message": "Test Merchant, calls are down 30% over 7d. Want help figuring out what changed?",
                "audience": "merchant",
                "language": "en",
                "objective": PERF_DIP_OBJECTIVE,
                "cta_type": PERF_DIP_CTA,
                "action": "send",
                "facts_used": ["merchant_name", "metric", "delta_pct", "window"],
                "confidence": 0.9,
                "should_send": True,
            },
        )

        first = tick(client, ["trig_repeat_teardown"])
        assert len(first.json()["actions"]) == 1

        suppressed = tick(client, ["trig_repeat_teardown"])
        assert suppressed.json()["actions"] == []

        teardown_response = client.post("/v1/teardown")
        assert teardown_response.status_code == 200

        # Re-seed context (teardown wipes context too) and confirm the same
        # suppression_key can send again after a clean restart.
        seed_category(client)
        seed_merchant(client, "m_repeat_teardown")
        seed_perf_dip_trigger(client, "trig_repeat_teardown", "m_repeat_teardown")

        after_teardown = tick(client, ["trig_repeat_teardown"])

    assert len(after_teardown.json()["actions"]) == 1


# --- 6. A composer/provider failure must safely fall back to the
#        deterministic FallbackComposer, not crash or produce nothing useful.
def test_provider_failure_falls_back_to_deterministic_composer(tmp_path):
    with make_client(tmp_path) as client:
        seed_category(client)
        seed_merchant(client, "m_fallback")
        seed_perf_dip_trigger(client, "trig_fallback", "m_fallback")

        fake = use_fake_provider(client, exc=GeminiUnavailableError("simulated Gemini outage"))

        response = tick(client, ["trig_fallback"])

    assert response.status_code == 200
    actions = response.json()["actions"]
    assert len(actions) == 1
    # Matches FallbackComposer's deterministic perf_dip phrasing exactly.
    assert "calls are down 30% over 7d" in actions[0]["body"]
    assert fake.call_count == 1
