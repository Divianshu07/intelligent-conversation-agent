from __future__ import annotations

import importlib.util
import random
from pathlib import Path

import pytest

from app.context_resolver import ContextResolver
from app.context_store import ContextStore
from app.trigger_router import TriggerRouter, normalize_language


ROOT = Path(__file__).resolve().parents[1]
GENERATOR_PATH = ROOT / "dataset" / "generate_dataset.py"


def load_expanded_dataset() -> tuple[dict, dict, dict, dict, list[dict]]:
    spec = importlib.util.spec_from_file_location("dataset_generator", GENERATOR_PATH)
    assert spec and spec.loader
    generator = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(generator)
    categories, merchants, customers, triggers = generator.load_seeds(ROOT / "dataset")
    random_source = random.Random(generator.SEED)
    merchants = generator.expand_merchants(merchants, random_source)
    customers = generator.expand_customers(customers, merchants, random_source)
    triggers = generator.expand_triggers(triggers, merchants, customers, random_source)
    pairs_by_kind: dict[str, list[dict]] = {}
    for trigger in triggers:
        pairs_by_kind.setdefault(trigger["kind"], []).append(trigger)
    canonical_triggers = [trigger for _, values in sorted(pairs_by_kind.items()) for trigger in values[:2]][:30]
    pairs = [
        {
            "test_id": f"T{index:02d}",
            "trigger_id": trigger["id"],
            "merchant_id": trigger["merchant_id"],
            "customer_id": trigger.get("customer_id"),
        }
        for index, trigger in enumerate(canonical_triggers, start=1)
    ]
    return (
        categories,
        {item["merchant_id"]: item for item in merchants},
        {item["customer_id"]: item for item in customers},
        {item["id"]: item for item in triggers},
        pairs,
    )


@pytest.fixture()
def router_and_pairs():
    categories, merchants, customers, triggers, pairs = load_expanded_dataset()
    store = ContextStore(":memory:")
    for context_id, payload in categories.items():
        store.put("category", context_id, 1, payload)
    for context_id, payload in merchants.items():
        store.put("merchant", context_id, 1, payload)
    for context_id, payload in customers.items():
        store.put("customer", context_id, 1, payload)
    for context_id, payload in triggers.items():
        store.put("trigger", context_id, 1, payload)
    yield TriggerRouter(), ContextResolver(store), {pair["test_id"]: pair for pair in pairs}
    store.close()


def brief_for(router_and_pairs, test_id: str):
    router, resolver, pairs = router_and_pairs
    pair = pairs[test_id]
    return router.build_brief(resolver.resolve(pair["trigger_id"]), request_id=test_id)


@pytest.mark.parametrize(
    ("test_id", "kind", "known_fact", "forbidden"),
    [
        ("T01", "active_planning_intent", "intent_topic", "package_pricing_if_missing"),
        ("T03", "appointment_tomorrow", "metric_or_topic", "appointment_time_if_missing"),
        ("T05", "category_seasonal", "trends", "inventory_not_supplied"),
        ("T07", "chronic_refill_due", "molecule_list", "medical_advice"),
        ("T09", "competitor_opened", "competitor_name", "competitor_ratings_not_supplied"),
        ("T10", "competitor_opened", "metric_or_topic", "competitor_performance_not_supplied"),
        ("T20", "gbp_unverified", "verification_path", "guaranteed_uplift"),
        ("T22", "milestone_reached", "milestone_value", "milestone_reached_when_only_imminent"),
        ("T24", "perf_dip", "delta_pct", "cause_of_decline"),
        ("T26", "perf_spike", "likely_driver", "proven_cause"),
        ("T28", "recall_due", "available_slots", "medical_outcome"),
        ("T30", "regulation_change", "deadline_iso", "legal_conclusion"),
    ],
)
def test_canonical_routes_preserve_facts_and_safety_guards(router_and_pairs, test_id, kind, known_fact, forbidden):
    brief = brief_for(router_and_pairs, test_id)
    assert brief.request_id == test_id
    assert brief.trigger_kind == kind
    assert known_fact in brief.known_facts
    assert forbidden in brief.forbidden_assumptions
    assert "fabricated_price" not in brief.known_facts


def test_placeholder_trigger_cannot_gain_fabricated_details(router_and_pairs):
    brief = brief_for(router_and_pairs, "T25")
    assert brief.known_facts["trigger_payload"] == {"placeholder": True, "metric_or_topic": "perf_dip"}
    assert "metric" not in brief.known_facts
    assert "delta_pct" not in brief.known_facts
    assert "likely_driver" not in brief.known_facts
    assert "price" not in brief.known_facts
    assert {"fabricated_metric", "fabricated_percentage", "fabricated_cause", "fabricated_price"}.issubset(brief.forbidden_assumptions)


def test_imminent_milestone_remains_imminent(router_and_pairs):
    brief = brief_for(router_and_pairs, "T22")
    assert brief.known_facts["metric"] == "review_count"
    assert brief.known_facts["value_now"] == 145
    assert brief.known_facts["milestone_value"] == 150
    assert brief.known_facts["is_imminent"] is True
    assert "milestone_reached" not in brief.known_facts


def test_customer_briefs_normalize_language_and_expose_only_explicit_consent(router_and_pairs):
    refill = brief_for(router_and_pairs, "T07")
    recall = brief_for(router_and_pairs, "T28")
    appointment_placeholder = brief_for(router_and_pairs, "T03")
    assert refill.audience == "customer"
    assert refill.language_preference == "hi"
    assert refill.consent_state and refill.consent_state.refill_consent is True
    assert recall.language_preference == "hi-en"
    assert recall.consent_state and recall.consent_state.recall_consent is True
    assert appointment_placeholder.consent_state and appointment_placeholder.consent_state.appointment_reminder_consent is False
    assert appointment_placeholder.consent_state.whatsapp_consent is None


def test_resolver_uses_latest_context_version(router_and_pairs):
    router, resolver, pairs = router_and_pairs
    pair = pairs["T24"]
    resolver._store.put("merchant", pair["merchant_id"], 2, {"merchant_id": pair["merchant_id"], "category_slug": "dentists", "identity": {"name": "Updated Bharat"}})
    brief = router.build_brief(resolver.resolve(pair["trigger_id"]))
    assert brief.merchant.version == 2
    assert brief.known_facts["merchant_name"] == "Updated Bharat"


@pytest.mark.parametrize(
    ("value", "expected"),
    [
        ("hi-en mix", "hi-en"),
        (["en", "hi"], "hi-en"),
        ("english", "en"),
        (["ta"], "other"),
        (None, "unknown"),
    ],
)
def test_language_normalization(value, expected):
    assert normalize_language(value) == expected
