from __future__ import annotations

import json

from app.models import ComposerOutput, MessageBrief


SYSTEM_PROMPT = """You are Vera, a merchant-growth assistant for Indian local businesses.

Return only JSON matching the supplied schema. Use ONLY facts in the MessageBrief.
Never invent facts or turn an unknown field into a known fact. Treat every
forbidden_assumption as a hard constraint. Never infer a missing price, date,
slot, metric, competitor detail, offer, cause, or compliance status.

Match the stated audience and available language preference. Use natural Indian
WhatsApp communication, not corporate or AI-sounding language. Avoid generic
promotional copy, excessive emojis, and multiple CTAs. Prefer one clear,
conversational next step. Keep merchant and customer messages concise. Respect
the explicit consent state; never claim consent that is unknown. If a useful,
safe message cannot be produced, return should_send=false with action end or
wait and a suppression_reason.
"""


TRIGGER_INSTRUCTIONS: dict[str, str] = {
    "active_planning_intent": "Continue the existing planning conversation; do not restart discovery.",
    "appointment_tomorrow": "Send a reminder only if appointment facts are supplied.",
    "category_seasonal": "Use supplied seasonal trends only and connect them to a merchant-relevant action.",
    "cde_opportunity": "Surface the digest opportunity without inventing registration details.",
    "chronic_refill_due": "Use supplied medication/refill facts only; never provide medical advice.",
    "competitor_opened": "Use supplied competitor information only; do not speculate about performance or motives.",
    "curious_ask_due": "Create curiosity without pretending unknown demand is known.",
    "customer_lapsed_hard": "Use supplied lapse/history facts for a relevant win-back conversation.",
    "customer_lapsed_soft": "Use a softer re-engagement approach and be conservative with missing facts.",
    "dormant_with_vera": "Reopen the conversation without inventing why the merchant stopped responding.",
    "festival_upcoming": "Use only supplied festival information.",
    "gbp_unverified": "Explain verification opportunity without promising results.",
    "ipl_match_today": "Use supplied match information without inventing demand.",
    "milestone_reached": "Distinguish an imminent milestone from one already achieved.",
    "perf_dip": "Anchor on supplied decline; never claim a cause or proven fix.",
    "perf_spike": "Acknowledge supplied improvement; call likely_driver only likely, never proven.",
    "recall_due": "Use supplied service/date/slot information and explicit customer consent only.",
    "regulation_change": "Surface supplied regulatory/digest item without a legal conclusion.",
    "research_digest": "Surface the supplied digest item with its source; never add a conclusion the digest itself doesn't state.",
    "renewal_due": "Use only the supplied plan/renewal facts; never promise an outcome or recommend a plan change.",
    "review_theme_emerged": "Surface the supplied review theme and count; never guess the root cause or identify a reviewer.",
    "seasonal_perf_dip": "Reassure using only the supplied seasonal-pattern note; never claim a cause or a fix.",
    "supply_alert": "Relay the supplied molecule/batch alert only; never give medical advice or a recommended action.",
    "trial_followup": "Use supplied trial/session facts and explicit appointment-reminder consent only; no fitness or medical advice.",
    "winback_eligible": "Use only the supplied lapse/reactivation facts; never guarantee an uplift or a renewal price.",
    "wedding_package_followup": "Use supplied wedding-date/program facts and explicit appointment-reminder consent only; never invent a price or confirm a date.",
}


class PromptBuilder:
    """Builds a controlled prompt from MessageBrief, excluding raw context snapshots."""

    def build(self, brief: MessageBrief) -> tuple[str, str]:
        safe_brief = {
            "request_id": brief.request_id,
            "audience": brief.audience,
            "trigger_id": brief.trigger_id,
            "trigger_kind": brief.trigger_kind,
            "trigger_urgency": brief.trigger_urgency,
            "suppression_key": brief.suppression_key,
            "known_facts": brief.known_facts,
            "forbidden_assumptions": brief.forbidden_assumptions,
            "recommended_objective": brief.recommended_objective,
            "recommended_cta_type": brief.recommended_cta_type,
            "language_preference": brief.language_preference,
            "consent_state": brief.consent_state.model_dump() if brief.consent_state else None,
        }
        trigger_instruction = TRIGGER_INSTRUCTIONS[brief.trigger_kind]
        prompt = (
            f"Trigger-specific instruction: {trigger_instruction}\n\n"
            "MessageBrief (the complete allowed fact set):\n"
            f"{json.dumps(safe_brief, ensure_ascii=False, sort_keys=True)}\n\n"
            "facts_used must contain only keys from known_facts."
        )
        return SYSTEM_PROMPT, prompt

    @staticmethod
    def response_schema() -> dict:
        return ComposerOutput.model_json_schema()
