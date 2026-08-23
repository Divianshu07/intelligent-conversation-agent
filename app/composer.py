from __future__ import annotations

import logging
import re
from typing import Any

from pydantic import ValidationError

from app.config import Settings
from app.llm.gemini import GeminiError, GeminiProvider, StructuredLLM
from app.models import ComposerOutput, MessageBrief
from app.output_validator import OutputValidator
from app.prompt_builder import PromptBuilder


logger = logging.getLogger(__name__)

_WEEK_CODE_PATTERN = re.compile(r"^\d{4}W\d{1,2}$")

# The only ask_template value present anywhere in the trigger dataset today;
# falls back to a mechanical underscore->space rendering for any other value
# so nothing here silently invents meaning for an unseen template.
_ASK_TEMPLATE_QUESTIONS = {
    "what_service_in_demand_this_week": "what's a service that's been in demand this week",
}


def _readable_topic_from_id(item_id: Any) -> str | None:
    """Extract a human-readable topic from a digest-style id such as
    "d_2026W17_dci_radiograph" -> "DCI radiograph". Only unpacks the literal
    tokens already present in the id (short all-letter tokens are
    upper-cased as a typographic acronym convention, nothing is expanded or
    interpreted). Returns None -- rather than a misleading guess -- if the
    id doesn't match the expected "d_<weekcode>_<topic...>" shape."""
    if not item_id:
        return None
    parts = str(item_id).split("_")
    if len(parts) < 3 or parts[0] != "d" or not _WEEK_CODE_PATTERN.match(parts[1]):
        return None
    topic_parts = [part for part in parts[2:] if part]
    if not topic_parts:
        return None
    rendered = [part.upper() if len(part) <= 4 and part.isalpha() else part for part in topic_parts]
    return " ".join(rendered)


def _readable_metric(metric: str) -> str:
    """Underscore->space rendering with a light, generic pluralization for
    "*_count" metric names (e.g. "review_count" -> "reviews") so milestone
    copy reads naturally. Falls back to a plain space-joined rendering for
    any metric name that doesn't match that shape -- no new information is
    added, only grammar."""
    readable = str(metric).replace("_", " ")
    if readable.endswith(" count"):
        noun = readable[: -len(" count")]
        if noun and not noun.endswith("s"):
            return f"{noun}s"
    return readable


class FallbackComposer:
    """Deterministic, low-risk behavior when Gemini is unavailable."""

    def compose(self, brief: MessageBrief, reason: str) -> ComposerOutput:
        facts = brief.known_facts
        payload = facts.get("trigger_payload", {})

        if not isinstance(payload, dict) or payload.get("placeholder") is True:
            return self._wait(
                brief,
                "Trigger details are incomplete; no safe message will be sent.",
            )

        if brief.trigger_kind == "perf_dip":
            metric = payload.get("metric", "performance")
            delta_pct = abs(payload.get("delta_pct", 0)) * 100
            window = payload.get("window", "recent period")

            return self._send(
                brief,
                f"{facts.get('merchant_name')}, {metric} are down "
                f"{delta_pct:g}% over {window}. Want me to help review what changed?",
                ["merchant_name", "metric", "delta_pct", "window"],
            )

        if (
            brief.trigger_kind == "recall_due"
            and brief.consent_state
            and brief.consent_state.recall_consent
        ):
            slots = [
                slot.get("label")
                for slot in payload.get("available_slots", [])
                if isinstance(slot, dict) and slot.get("label")
            ]

            if slots:
                return self._send(
                    brief,
                    f"Hi {facts.get('customer_name')}, your "
                    f"{payload.get('service_due', '').replace('_', ' ')} is due. "
                    f"Available slots: {' or '.join(slots)}. Which works for you?",
                    ["customer_name", "service_due", "available_slots"],
                )

        if brief.trigger_kind == "regulation_change":
            topic = _readable_topic_from_id(payload.get("top_item_id"))
            topic_clause = f" {topic}" if topic else ""
            return self._send(
                brief,
                f"{facts.get('merchant_name')}, there's a{topic_clause} "
                f"compliance update in your category, with a deadline of "
                f"{payload.get('deadline_iso')}. Want a quick summary of "
                f"what's changing?",
                ["merchant_name", "top_item_id", "deadline_iso"],
            )

        if brief.trigger_kind == "perf_spike":
            metric = payload.get("metric")
            delta_pct = payload.get("delta_pct")
            window = payload.get("window")
            if metric and isinstance(delta_pct, (int, float)) and window:
                return self._send(
                    brief,
                    f"{facts.get('merchant_name')}, {metric} are up "
                    f"{abs(delta_pct) * 100:g}% over {window}. Want to explore "
                    f"what's driving it?",
                    ["merchant_name", "metric", "delta_pct", "window"],
                )

        if brief.trigger_kind == "festival_upcoming":
            festival = payload.get("festival")
            date = payload.get("date")
            if festival and date:
                return self._send(
                    brief,
                    f"{facts.get('merchant_name')}, {festival} is coming up on "
                    f"{date}. Want to plan something for it?",
                    ["merchant_name", "festival", "date"],
                )

        if brief.trigger_kind == "active_planning_intent":
            intent_topic = payload.get("intent_topic")
            if intent_topic:
                topic_readable = str(intent_topic).replace("_", " ")
                return self._send(
                    brief,
                    f"Following up on {topic_readable} — happy to help figure "
                    f"out the details whenever you're ready.",
                    ["merchant_name", "intent_topic"],
                )

        if brief.trigger_kind == "milestone_reached":
            metric = payload.get("metric")
            milestone_value = payload.get("milestone_value")
            is_imminent = payload.get("is_imminent")
            if metric and milestone_value is not None and isinstance(is_imminent, bool):
                metric_readable = _readable_metric(metric)
                value_now = payload.get("value_now")
                facts_used = ["merchant_name", "metric", "milestone_value"]
                if is_imminent:
                    gap = milestone_value - value_now if isinstance(value_now, (int, float)) else None
                    if gap is not None and gap > 0:
                        # Must never be presented as already achieved.
                        message = (
                            f"{facts.get('merchant_name')}, you're at "
                            f"{value_now} {metric_readable} — just {gap} "
                            f"away from {milestone_value}! Want a nudge to "
                            f"help close the gap?"
                        )
                        facts_used.append("value_now")
                    else:
                        # Must never be presented as already achieved.
                        message = (
                            f"{facts.get('merchant_name')}, you're almost at "
                            f"{milestone_value} {metric_readable} — nearly there!"
                        )
                else:
                    message = (
                        f"{facts.get('merchant_name')}, congratulations on "
                        f"reaching {milestone_value} {metric_readable}!"
                    )
                return self._send(
                    brief,
                    message,
                    facts_used,
                )

        if (
            brief.trigger_kind == "customer_lapsed_hard"
            and brief.consent_state
            and brief.consent_state.promotional_consent
        ):
            days_since_last_visit = payload.get("days_since_last_visit")
            if facts.get("customer_name") and days_since_last_visit is not None:
                return self._send(
                    brief,
                    f"Hi {facts.get('customer_name')}, it's been "
                    f"{days_since_last_visit} days since your last visit. "
                    f"Want to come back in?",
                    ["customer_name", "days_since_last_visit"],
                )

        if brief.trigger_kind == "gbp_unverified":
            verification_path = payload.get("verification_path")
            if verification_path:
                path_readable = str(verification_path).replace("_", " ")
                return self._send(
                    brief,
                    f"{facts.get('merchant_name')}, your Google Business "
                    f"Profile isn't verified yet. Want help getting it "
                    f"verified via {path_readable}?",
                    ["merchant_name", "verification_path"],
                )

        if brief.trigger_kind == "category_seasonal":
            season = payload.get("season")
            if season:
                season_readable = str(season).replace("_", " ")
                facts_used = ["merchant_name", "season"]
                trend_clause = ""
                trends = payload.get("trends")
                if isinstance(trends, list) and trends:
                    trend_bits = []
                    for trend in trends[:3]:
                        trend_str = str(trend)
                        if "_demand_" in trend_str:
                            item, _, delta = trend_str.partition("_demand_")
                            trend_bits.append(f"{item.replace('_', ' ')} {delta}%")
                        else:
                            trend_bits.append(trend_str.replace("_", " "))
                    if trend_bits:
                        trend_clause = f" — {', '.join(trend_bits)}"
                        facts_used.append("trends")
                return self._send(
                    brief,
                    f"{facts.get('merchant_name')}, {season_readable} trends "
                    f"are shifting in your category{trend_clause}. Want a "
                    f"quick look at what's changing?",
                    facts_used,
                )

        if brief.trigger_kind == "cde_opportunity":
            digest_item_id = payload.get("digest_item_id")
            credits = payload.get("credits")
            fee = payload.get("fee")
            if digest_item_id and credits is not None:
                fee_readable = str(fee).replace("_", " ") if fee else "fee not supplied"
                topic = _readable_topic_from_id(digest_item_id)
                topic_clause = f" ({topic})" if topic else ""
                return self._send(
                    brief,
                    f"{facts.get('merchant_name')}, there's a "
                    f"professional-development opportunity{topic_clause} "
                    f"worth {credits} credits ({fee_readable}). Want the "
                    f"details?",
                    ["merchant_name", "credits", "fee", "digest_item_id"],
                )

        if brief.trigger_kind == "competitor_opened":
            competitor_name = payload.get("competitor_name")
            if competitor_name:
                distance_km = payload.get("distance_km")
                opened_date = payload.get("opened_date")
                distance_clause = f" {distance_km}km away" if distance_km is not None else ""
                date_clause = f" on {opened_date}" if opened_date else ""
                return self._send(
                    brief,
                    f"{facts.get('merchant_name')}, {competitor_name} opened"
                    f"{distance_clause}{date_clause}. Want to talk through "
                    f"your response?",
                    ["merchant_name", "competitor_name", "distance_km", "opened_date"],
                )

        if brief.trigger_kind == "dormant_with_vera":
            days_since_last_merchant_message = payload.get("days_since_last_merchant_message")
            last_topic = payload.get("last_topic")
            if days_since_last_merchant_message is not None and last_topic:
                topic_readable = str(last_topic).replace("_", " ")
                return self._send(
                    brief,
                    f"{facts.get('merchant_name')}, it's been "
                    f"{days_since_last_merchant_message} days since we last "
                    f"spoke about {topic_readable}. Want to pick that back up?",
                    ["merchant_name", "days_since_last_merchant_message", "last_topic"],
                )

        if brief.trigger_kind == "curious_ask_due":
            ask_template = payload.get("ask_template")
            if ask_template:
                question_readable = _ASK_TEMPLATE_QUESTIONS.get(
                    str(ask_template), str(ask_template).replace("_", " ")
                )
                return self._send(
                    brief,
                    f"Quick one — {question_readable}?",
                    ["merchant_name", "ask_template"],
                )

        if brief.trigger_kind == "renewal_due":
            plan = payload.get("plan")
            days_remaining = payload.get("days_remaining")
            renewal_amount = payload.get("renewal_amount")
            if plan and days_remaining is not None and renewal_amount is not None:
                return self._send(
                    brief,
                    f"{facts.get('merchant_name')}, your {plan} plan renews in "
                    f"{days_remaining} days (₹{renewal_amount}). Want help "
                    f"sorting the renewal?",
                    ["merchant_name", "plan", "days_remaining", "renewal_amount"],
                )

        if brief.trigger_kind == "review_theme_emerged":
            theme = payload.get("theme")
            occurrences_30d = payload.get("occurrences_30d")
            if theme and occurrences_30d is not None:
                theme_readable = str(theme).replace("_", " ")
                trend_clause = f" ({payload.get('trend')})" if payload.get("trend") else ""
                return self._send(
                    brief,
                    f"{facts.get('merchant_name')}, {occurrences_30d} reviews "
                    f"in the last 30 days mention {theme_readable}{trend_clause}. "
                    f"Want to see the details?",
                    ["merchant_name", "theme", "occurrences_30d", "trend"],
                )

        if brief.trigger_kind == "seasonal_perf_dip":
            metric = payload.get("metric")
            delta_pct = payload.get("delta_pct")
            window = payload.get("window")
            if (
                metric
                and isinstance(delta_pct, (int, float))
                and window
                and payload.get("is_expected_seasonal") is True
            ):
                season_note = payload.get("season_note")
                note_clause = f" ({str(season_note).replace('_', ' ')})" if season_note else ""
                return self._send(
                    brief,
                    f"{facts.get('merchant_name')}, {metric} are down "
                    f"{abs(delta_pct) * 100:g}% over {window} — this lines up "
                    f"with the usual seasonal pattern{note_clause}, nothing "
                    f"unusual to flag.",
                    ["merchant_name", "metric", "delta_pct", "window", "season_note"],
                )

        if (
            brief.trigger_kind == "trial_followup"
            and brief.consent_state
            and brief.consent_state.appointment_reminder_consent
        ):
            trial_date = payload.get("trial_date")
            slots = [
                slot.get("label")
                for slot in payload.get("next_session_options", [])
                if isinstance(slot, dict) and slot.get("label")
            ]
            if trial_date and slots and facts.get("customer_name"):
                return self._send(
                    brief,
                    f"Hi {facts.get('customer_name')}, hope the {trial_date} "
                    f"trial went well! Next session options: "
                    f"{' or '.join(slots)}. Want to book one?",
                    ["customer_name", "trial_date", "next_session_options"],
                )

        if brief.trigger_kind == "winback_eligible":
            days_since_expiry = payload.get("days_since_expiry")
            lapsed_customers_added_since_expiry = payload.get("lapsed_customers_added_since_expiry")
            if days_since_expiry is not None and lapsed_customers_added_since_expiry is not None:
                return self._send(
                    brief,
                    f"{facts.get('merchant_name')}, it's been "
                    f"{days_since_expiry} days since your subscription lapsed, "
                    f"and {lapsed_customers_added_since_expiry} customers have "
                    f"churned since. Want to talk about reactivating?",
                    ["merchant_name", "days_since_expiry", "lapsed_customers_added_since_expiry"],
                )

        if (
            brief.trigger_kind == "wedding_package_followup"
            and brief.consent_state
            and brief.consent_state.appointment_reminder_consent
        ):
            wedding_date = payload.get("wedding_date")
            days_to_wedding = payload.get("days_to_wedding")
            next_step_window_open = payload.get("next_step_window_open")
            if wedding_date and days_to_wedding is not None and next_step_window_open and facts.get("customer_name"):
                step_readable = str(next_step_window_open).replace("_", " ")
                return self._send(
                    brief,
                    f"Hi {facts.get('customer_name')}, {days_to_wedding} days "
                    f"to go until your big day ({wedding_date})! Ready to "
                    f"start the {step_readable}?",
                    ["customer_name", "wedding_date", "days_to_wedding", "next_step_window_open"],
                )

        if brief.trigger_kind == "research_digest":
            category = payload.get("category")
            top_item_id = payload.get("top_item_id")
            if category and top_item_id:
                return self._send(
                    brief,
                    f"{facts.get('merchant_name')}, there's a new research "
                    f"digest item for {category}: {top_item_id}. Want the "
                    f"summary?",
                    ["merchant_name", "category", "top_item_id"],
                )

        if brief.trigger_kind == "supply_alert":
            molecule = payload.get("molecule")
            affected_batches = payload.get("affected_batches")
            if molecule and isinstance(affected_batches, list) and affected_batches:
                manufacturer = payload.get("manufacturer")
                manufacturer_clause = f" ({manufacturer})" if manufacturer else ""
                return self._send(
                    brief,
                    f"{facts.get('merchant_name')}, there's a supply alert "
                    f"for {molecule}{manufacturer_clause} affecting batches: "
                    f"{', '.join(affected_batches)}. Want the full alert "
                    f"details?",
                    ["merchant_name", "molecule", "affected_batches", "manufacturer"],
                )

        if (
            brief.trigger_kind == "chronic_refill_due"
            and brief.consent_state
            and brief.consent_state.refill_consent
        ):
            molecule_list = payload.get("molecule_list")
            stock_runs_out_iso = payload.get("stock_runs_out_iso")
            if (
                facts.get("customer_name")
                and isinstance(molecule_list, list)
                and molecule_list
                and stock_runs_out_iso
            ):
                stock_out_date = str(stock_runs_out_iso).split("T")[0]
                delivery_address_saved = payload.get("delivery_address_saved")
                delivery_clause = (
                    "We have your delivery address saved. " if delivery_address_saved else ""
                )
                return self._send(
                    brief,
                    f"Hi {facts.get('customer_name')}, your "
                    f"{', '.join(molecule_list)} refill is running low — "
                    f"stock is expected to run out around {stock_out_date}. "
                    f"{delivery_clause}Want me to arrange it?",
                    ["customer_name", "molecule_list", "stock_runs_out_iso", "delivery_address_saved"],
                )

        if brief.trigger_kind == "ipl_match_today":
            match = payload.get("match")
            venue = payload.get("venue")
            if match and venue:
                city = payload.get("city")
                city_clause = f", {city}" if city else ""
                return self._send(
                    brief,
                    f"{facts.get('merchant_name')}, {match} is on today at "
                    f"{venue}{city_clause}. Want to plan something for it?",
                    ["merchant_name", "match", "venue", "city"],
                )

        if (
            brief.trigger_kind == "appointment_tomorrow"
            and brief.consent_state
            and brief.consent_state.appointment_reminder_consent
        ):
            appointment_time = payload.get("appointment_time")
            service = payload.get("service")
            if appointment_time and service and facts.get("customer_name"):
                staff_member = payload.get("staff_member")
                staff_clause = f" with {staff_member}" if staff_member else ""
                return self._send(
                    brief,
                    f"Hi {facts.get('customer_name')}, reminder: your "
                    f"{service} appointment{staff_clause} is tomorrow at "
                    f"{appointment_time}. See you then?",
                    ["customer_name", "appointment_time", "service", "staff_member"],
                )

        return self._wait(brief, reason)

    @staticmethod
    def _send(
        brief: MessageBrief,
        message: str,
        facts_used: list[str],
    ) -> ComposerOutput:
        return ComposerOutput(
            message=message,
            audience=brief.audience,
            language=brief.language_preference,
            objective=brief.recommended_objective,
            cta_type=brief.recommended_cta_type,
            action="send",
            facts_used=facts_used,
            confidence=0.3,
            should_send=True,
        )

    @staticmethod
    def _wait(
        brief: MessageBrief,
        reason: str,
    ) -> ComposerOutput:
        return ComposerOutput(
            message="",
            audience=brief.audience,
            language=brief.language_preference,
            objective=brief.recommended_objective,
            cta_type="none",
            action="wait",
            facts_used=[],
            confidence=0,
            should_send=False,
            suppression_reason=reason,
        )


class AIComposer:
    """MessageBrief -> optional Gemini structured output -> validator -> safe fallback."""

    def __init__(
        self,
        provider: StructuredLLM | None = None,
        prompt_builder: PromptBuilder | None = None,
        validator: OutputValidator | None = None,
        fallback: FallbackComposer | None = None,
        settings: Settings | None = None,
    ) -> None:
        runtime_settings = settings or Settings()

        # Gemini is optional.
        # If no provider is supplied, the deterministic fallback is used.
        self._provider = provider

        self._prompt_builder = prompt_builder or PromptBuilder()
        self._validator = validator or OutputValidator()
        self._fallback = fallback or FallbackComposer()
        self._debug_log_prompts = runtime_settings.debug_log_prompts

    def compose(self, brief: MessageBrief) -> ComposerOutput:
        # No LLM configured -> use deterministic fallback directly.
        if self._provider is None:
            return self._fallback.compose(
                brief,
                "No LLM provider configured; using deterministic fallback.",
            )

        system_prompt, prompt = self._prompt_builder.build(brief)

        if self._debug_log_prompts:
            logger.debug(
                "Gemini composer request: system=%s prompt=%s",
                system_prompt,
                prompt,
            )

        try:
            raw_response: dict[str, Any] = self._provider.generate(
                system_prompt,
                prompt,
                self._prompt_builder.response_schema(),
            )

            candidate = ComposerOutput.model_validate(raw_response)

        except (
            GeminiError,
            ValidationError,
            ValueError,
            TypeError,
        ) as exc:
            return self._fallback.compose(
                brief,
                f"Gemini unavailable or returned invalid structured output: {exc}",
            )

        result = self._validator.validate(candidate, brief)

        if not result.valid:
            return result.output

        return result.output