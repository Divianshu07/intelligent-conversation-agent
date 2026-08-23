from __future__ import annotations

import random

import pytest

from app.composer import AIComposer
from app.context_resolver import ContextResolver
from app.context_store import ContextStore
from app.models import ComposerOutput
from app.prompt_builder import PromptBuilder
from tests.test_context_intelligence import load_expanded_dataset


class MockLLM:
    def __init__(self, response):
        self.response = response
        self.calls = 0

    def generate(self, system_prompt, prompt, response_schema):
        self.calls += 1
        if isinstance(self.response, Exception):
            raise self.response
        return self.response


@pytest.fixture()
def briefs():
    categories, merchants, customers, triggers, pairs = load_expanded_dataset()
    store = ContextStore(":memory:")
    for scope, collection, key in (
        ("category", categories, None),
        ("merchant", merchants, None),
        ("customer", customers, None),
        ("trigger", triggers, None),
    ):
        for context_id, payload in collection.items():
            store.put(scope, context_id, 1, payload)
    resolver = ContextResolver(store)
    pair_map = {pair["test_id"]: pair for pair in pairs}
    from app.trigger_router import TriggerRouter

    router = TriggerRouter()
    yield {test_id: router.build_brief(resolver.resolve(pair["trigger_id"]), test_id) for test_id, pair in pair_map.items()}
    store.close()


def response_for(brief, message: str, facts_used: list[str], **overrides):
    payload = {
        "message": message,
        "audience": brief.audience,
        "language": brief.language_preference,
        "objective": brief.recommended_objective,
        "cta_type": brief.recommended_cta_type,
        "action": "send",
        "facts_used": facts_used,
        "confidence": 0.9,
        "should_send": True,
        "suppression_reason": None,
    }
    payload.update(overrides)
    return payload


def test_valid_perf_dip_response_is_accepted(briefs):
    brief = briefs["T24"]
    llm = MockLLM(response_for(brief, "Bharat, calls are down 50% over 7d versus the baseline. Want me to help review what changed?", ["merchant_name", "metric", "delta_pct", "window", "vs_baseline"]))
    output = AIComposer(provider=llm).compose(brief)
    assert output.should_send is True
    assert output.action == "send"
    assert "50%" in output.message
    assert output.message.count("?") == 1


def test_placeholder_perf_dip_with_invented_percentage_is_rejected(briefs):
    brief = briefs["T25"]
    llm = MockLLM(response_for(brief, "Your calls are down 30%. Try a Haircut @ ₹99 offer?", ["metric_or_topic"]))
    output = AIComposer(provider=llm).compose(brief)
    assert output.should_send is False
    assert output.action == "wait"
    assert "placeholder trigger" in (output.suppression_reason or "")


def test_imminent_milestone_cannot_be_presented_as_achieved(briefs):
    brief = briefs["T22"]
    llm = MockLLM(response_for(brief, "Mylari, congratulations — 150 reviews achieved! Want a celebration post?", ["merchant_name", "milestone_value"]))
    output = AIComposer(provider=llm).compose(brief)
    assert output.should_send is False
    assert "imminent milestone" in (output.suppression_reason or "")


def test_recall_can_use_only_supplied_slots(briefs):
    brief = briefs["T28"]
    llm = MockLLM(response_for(brief, "Hi Priya, your 6 month cleaning is due. Wed 5 Nov, 6pm or Thu 6 Nov, 5pm — which works for you?", ["customer_name", "service_due", "available_slots"]))
    output = AIComposer(provider=llm).compose(brief)
    assert output.should_send is True
    assert output.action == "send"


def test_recall_rejects_unsupported_slot(briefs):
    brief = briefs["T28"]
    llm = MockLLM(response_for(brief, "Hi Priya, Fri 7 Nov, 7pm is available. Does that work?", ["customer_name"]))
    output = AIComposer(provider=llm).compose(brief)
    assert output.should_send is False
    assert "unsupported appointment slot" in (output.suppression_reason or "")


def test_regulation_allows_deadline_but_rejects_legal_conclusion(briefs):
    brief = briefs["T30"]
    valid = MockLLM(response_for(brief, "Dr. Meera, the supplied DCI item has a 2026-12-15 deadline. Want its summary?", ["merchant_name", "top_item_id", "deadline_iso"]))
    assert AIComposer(provider=valid).compose(brief).should_send is True

    invalid = MockLLM(response_for(brief, "Dr. Meera, you must comply by 2026-12-15.", ["merchant_name", "deadline_iso"]))
    output = AIComposer(provider=invalid).compose(brief)
    assert output.should_send is False
    assert "legal" in (output.suppression_reason or "")


def test_customer_message_cannot_claim_unknown_whatsapp_consent(briefs):
    brief = briefs["T03"]
    llm = MockLLM(response_for(brief, "Hi Aditya, you opted in to WhatsApp reminders. Your appointment is tomorrow.", ["customer_name", "metric_or_topic"]))
    output = AIComposer(provider=llm).compose(brief)
    assert output.should_send is False
    assert "unknown WhatsApp consent" in (output.suppression_reason or "")


def test_empty_or_invalid_model_response_uses_safe_fallback(briefs):
    brief = briefs["T25"]
    output = AIComposer(provider=MockLLM({})).compose(brief)
    assert output.should_send is False
    assert output.action == "wait"
    assert output.suppression_reason == "Trigger details are incomplete; no safe message will be sent."


def test_prompt_uses_message_brief_not_raw_context_snapshot(briefs):
    brief = briefs["T24"]
    _, prompt = PromptBuilder().build(brief)
    assert "ChIJ_ANDHERI_DENTIST_002" not in prompt
    assert '"merchant": {' not in prompt
    assert '"known_facts"' in prompt
