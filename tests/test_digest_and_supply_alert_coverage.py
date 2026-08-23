from __future__ import annotations

import pytest

from app.composer import AIComposer
from app.context_resolver import ContextResolver
from app.context_store import ContextStore
from app.trigger_router import TriggerRouter


def build_brief(kind: str, trigger_payload: dict):
    """Seed a minimal category/merchant/trigger set and build the
    MessageBrief the live TickService path would produce for one trigger."""
    store = ContextStore(":memory:")
    store.put("category", "cat1", 1, {"slug": "general", "voice": {"tone": "friendly"}})
    store.put(
        "merchant",
        "m1",
        1,
        {
            "category_slug": "cat1",
            "identity": {"name": "Test Merchant", "city": "Pune", "locality": "Kothrud", "languages": ["en"]},
            "offers": [],
            "performance": {},
            "signals": [],
        },
    )
    store.put(
        "trigger",
        "trig1",
        1,
        {
            "kind": kind,
            "urgency": 1,
            "suppression_key": f"{kind}:m1",
            "merchant_id": "m1",
            "scope": "merchant",
            "payload": trigger_payload,
        },
    )

    resolver = ContextResolver(store)
    router = TriggerRouter()
    brief = router.build_brief(resolver.resolve("trig1"), request_id="trig1")
    store.close()
    return brief


def compose(brief):
    # provider=None -> the exact live-path fallback used when Gemini is
    # unavailable / no GEMINI_API_KEY is configured.
    return AIComposer(provider=None).compose(brief)


def test_research_digest_produces_grounded_send():
    brief = build_brief("research_digest", {"category": "dentists", "top_item_id": "d_2026W17_jida_fluoride"})
    output = compose(brief)
    assert output.should_send is True
    assert output.action == "send"
    assert "dentists" in output.message
    assert "d_2026W17_jida_fluoride" in output.message


def test_supply_alert_produces_grounded_send():
    brief = build_brief(
        "supply_alert",
        {
            "alert_id": "d_2026W17_atorvastatin_recall",
            "molecule": "atorvastatin",
            "affected_batches": ["AT2024-1102", "AT2024-1108"],
            "manufacturer": "MfrZ",
        },
    )
    output = compose(brief)
    assert output.should_send is True
    assert output.action == "send"
    assert "atorvastatin" in output.message
    assert "AT2024-1102" in output.message
    assert "AT2024-1108" in output.message
    assert "MfrZ" in output.message
    # No medical advice/recommendation beyond relaying the supplied alert.
    lowered = output.message.lower()
    for forbidden in ("stop taking", "consult a doctor", "safe to", "should switch"):
        assert forbidden not in lowered


def test_supply_alert_without_manufacturer_still_sends():
    brief = build_brief(
        "supply_alert",
        {"molecule": "atorvastatin", "affected_batches": ["AT2024-1102"]},
    )
    output = compose(brief)
    assert output.should_send is True
    assert "atorvastatin" in output.message
    assert "AT2024-1102" in output.message


@pytest.mark.parametrize(
    "kind,payload",
    [
        ("research_digest", {"category": "dentists"}),  # missing top_item_id
        ("research_digest", {"top_item_id": "d_1"}),  # missing category
        ("supply_alert", {"molecule": "atorvastatin"}),  # missing affected_batches
        ("supply_alert", {"affected_batches": ["AT2024-1102"]}),  # missing molecule
        ("supply_alert", {"molecule": "atorvastatin", "affected_batches": []}),  # empty batches
    ],
)
def test_missing_required_fields_fall_back_to_wait(kind, payload):
    brief = build_brief(kind, payload)
    output = compose(brief)
    assert output.should_send is False
    assert output.action == "wait"
