from __future__ import annotations

from collections.abc import Iterable
from typing import Any

from app.context_resolver import ResolvedContexts
from app.models import ConsentState, ContextSnapshot, MessageBrief, NormalizedLanguage


ROUTING_RULES: dict[str, tuple[str, str, tuple[str, ...]]] = {
    "active_planning_intent": ("help merchant move an existing planning conversation forward", "planning_continuation", ("package_pricing_if_missing", "operational_details_if_missing")),
    "appointment_tomorrow": ("appointment reminder", "appointment_confirmation_if_details_available", ("appointment_time_if_missing", "service_if_missing", "price_if_missing", "staff_member_if_missing", "booking_confirmation_if_missing")),
    "category_seasonal": ("connect supplied seasonal trend data to a merchant-relevant action", "open_ended", ("inventory_not_supplied", "demand_not_supplied")),
    "cde_opportunity": ("make merchant aware of the supplied professional-development opportunity", "open_ended", ("registration_details_if_missing", "membership_status_if_missing", "event_schedule_if_missing")),
    "chronic_refill_due": ("refill or delivery reminder", "confirm_or_decline", ("medical_advice", "dose_if_missing", "price_if_missing", "availability_if_missing", "savings_if_missing")),
    "competitor_opened": ("make merchant aware of the supplied competitive signal", "open_ended", ("competitor_ratings_not_supplied", "competitor_motives_not_supplied", "competitor_performance_not_supplied", "business_impact_not_supplied")),
    "curious_ask_due": ("create a useful curiosity-driven merchant conversation", "open_ended", ("unknown_demand_trend",)),
    "customer_lapsed_hard": ("win back the customer", "binary_yes_no", ("new_class_if_missing", "trial_if_missing", "price_if_missing", "schedule_if_missing")),
    "customer_lapsed_soft": ("gently re-engage the customer", "low_pressure_open_ended", ("lapse_reason_if_missing", "service_due_if_missing", "offer_if_missing")),
    "dormant_with_vera": ("reopen the merchant conversation", "open_ended", ("reason_for_nonresponse",)),
    "festival_upcoming": ("explore a relevant seasonal opportunity", "open_ended", ("festival_details_if_missing", "campaign_if_missing", "capacity_if_missing")),
    "gbp_unverified": ("encourage Google Business Profile verification", "binary_yes_no", ("guaranteed_uplift", "completion_time_if_missing", "verification_availability_if_missing")),
    "ipl_match_today": ("connect supplied match information to a relevant merchant opportunity", "open_ended", ("increased_covers_not_supplied", "increased_orders_not_supplied", "demand_impact_not_supplied")),
    "milestone_reached": ("acknowledge a reached milestone or prepare for an imminent milestone", "open_ended", ("milestone_value_if_missing", "reward_if_missing", "reviewer_identity_if_missing", "milestone_reached_when_only_imminent")),
    "perf_dip": ("acknowledge the supplied decline and invite investigation or action", "open_ended", ("cause_of_decline", "proven_intervention", "competitor_information_not_supplied")),
    "perf_spike": ("acknowledge positive movement and explore what to build on", "open_ended", ("proven_cause", "spike_amount_if_missing")),
    "recall_due": ("service recall", "slot_selection_if_slots_available", ("medical_outcome", "availability_if_missing", "service_if_missing", "date_if_missing")),
    "regulation_change": ("make merchant aware of the supplied regulatory or digest item", "open_ended", ("legal_conclusion", "merchant_compliance_status_if_missing", "remediation_complete_if_missing")),
    "research_digest": ("make merchant aware of the supplied category research digest item", "open_ended", ("digest_conclusion_not_supplied", "category_comparison_not_supplied")),
    "renewal_due": ("remind merchant of an upcoming subscription renewal", "confirm_or_decline", ("renewal_outcome_guarantee", "plan_change_recommendation")),
    "review_theme_emerged": ("make merchant aware of a recurring review theme", "open_ended", ("root_cause_not_supplied", "customer_identity_not_supplied")),
    "seasonal_perf_dip": ("reassure merchant that a flagged dip is an expected seasonal pattern", "open_ended", ("cause_of_decline", "proven_intervention")),
    "supply_alert": ("make merchant aware of the supplied supply/recall alert", "open_ended", ("medical_advice", "recommended_action_not_supplied", "affected_batch_completeness_not_supplied")),
    "trial_followup": ("follow up after a completed trial with the next-session options", "slot_selection_if_slots_available", ("trial_outcome_not_supplied", "medical_or_fitness_advice", "availability_if_missing")),
    "winback_eligible": ("open a reactivation conversation after a subscription lapse", "open_ended", ("guaranteed_uplift", "renewal_price_if_missing")),
    "wedding_package_followup": ("follow up on a bridal/wedding package journey with the next program step", "open_ended", ("date_confirmation_not_supplied", "price_if_missing")),
}


def normalize_language(value: str | Iterable[str] | None) -> NormalizedLanguage:
    """Map input language labels to the challenge's restricted language categories."""
    if value is None:
        return "unknown"
    values = [value] if isinstance(value, str) else list(value)
    normalized = {str(item).strip().lower() for item in values if str(item).strip()}
    if not normalized:
        return "unknown"
    if any(item in {"hi-en", "hi-en mix", "hindi-english", "hinglish"} for item in normalized):
        return "hi-en"
    if "hi" in normalized and "en" in normalized:
        return "hi-en"
    if "hi" in normalized or "hindi" in normalized:
        return "hi"
    if "en" in normalized or "english" in normalized:
        return "en"
    return "other"


class TriggerRouter:
    """Deterministically converts resolved contexts into a safe future-composer brief."""

    def build_brief(self, contexts: ResolvedContexts, request_id: str | None = None) -> MessageBrief:
        trigger = contexts.trigger.payload
        kind = str(trigger.get("kind", ""))
        if kind not in ROUTING_RULES:
            raise ValueError(f"Unsupported trigger kind: {kind!r}")
        objective, cta_type, base_forbidden = ROUTING_RULES[kind]
        audience = "customer" if trigger.get("scope") == "customer" else "merchant"
        trigger_payload = trigger.get("payload") if isinstance(trigger.get("payload"), dict) else {}
        forbidden = list(base_forbidden)
        if trigger_payload.get("placeholder") is True:
            forbidden.extend(("trigger_details_not_supplied", "fabricated_metric", "fabricated_percentage", "fabricated_cause", "fabricated_price"))

        language, consent = self._customer_details(contexts, audience)
        if audience == "merchant":
            language = normalize_language(contexts.merchant.payload.get("identity", {}).get("languages"))

        return MessageBrief(
            request_id=request_id,
            audience=audience,
            category=self._snapshot(contexts.category),
            merchant=self._snapshot(contexts.merchant),
            customer=self._snapshot(contexts.customer) if contexts.customer else None,
            trigger_id=contexts.trigger.context_id,
            trigger_kind=kind,
            trigger_urgency=int(trigger.get("urgency", 0)),
            suppression_key=str(trigger.get("suppression_key", "")),
            known_facts=self._known_facts(contexts, trigger_payload),
            forbidden_assumptions=forbidden,
            recommended_objective=objective,
            recommended_cta_type=cta_type,
            language_preference=language,
            consent_state=consent,
        )

    @staticmethod
    def _snapshot(context) -> ContextSnapshot:
        return ContextSnapshot(context_id=context.context_id, version=context.version, payload=context.payload)

    @staticmethod
    def _known_facts(contexts: ResolvedContexts, trigger_payload: dict[str, Any]) -> dict[str, Any]:
        merchant = contexts.merchant.payload
        identity = merchant.get("identity", {})
        category = contexts.category.payload
        facts: dict[str, Any] = {
            "category_slug": category.get("slug"),
            "category_tone": category.get("voice", {}).get("tone"),
            "merchant_name": identity.get("name"),
            "merchant_city": identity.get("city"),
            "merchant_locality": identity.get("locality"),
            "merchant_languages": identity.get("languages", []),
            "active_offers": [offer.get("title") for offer in merchant.get("offers", []) if offer.get("status") == "active"],
            "merchant_performance": merchant.get("performance", {}),
            "merchant_signals": merchant.get("signals", []),
            "trigger_payload": trigger_payload,
        }
        for key, value in trigger_payload.items():
            if key != "placeholder":
                facts[key] = value
        if contexts.customer:
            customer = contexts.customer.payload
            facts.update(
                {
                    "customer_name": customer.get("identity", {}).get("name"),
                    "customer_language_preference": customer.get("identity", {}).get("language_pref"),
                    "customer_state": customer.get("state"),
                    "customer_relationship": customer.get("relationship", {}),
                    "customer_preferences": customer.get("preferences", {}),
                }
            )
        return facts

    @staticmethod
    def _customer_details(contexts: ResolvedContexts, audience: str) -> tuple[NormalizedLanguage, ConsentState | None]:
        if audience != "customer" or contexts.customer is None:
            return "unknown", None
        customer = contexts.customer.payload
        consent = customer.get("consent", {})
        scopes = consent.get("scope", []) if isinstance(consent.get("scope", []), list) else []
        scope_set = set(scopes)
        preferences = customer.get("preferences", {})
        return (
            normalize_language(customer.get("identity", {}).get("language_pref")),
            ConsentState(
                scopes=scopes,
                promotional_consent=bool(scope_set & {"promotional_offers", "winback_offers"}),
                appointment_reminder_consent="appointment_reminders" in scope_set,
                recall_consent="recall_reminders" in scope_set or "recall_alerts" in scope_set,
                refill_consent="refill_reminders" in scope_set,
                delivery_notification_consent="delivery_notifications" in scope_set,
                whatsapp_consent=None,
                channel=preferences.get("channel"),
            ),
        )
