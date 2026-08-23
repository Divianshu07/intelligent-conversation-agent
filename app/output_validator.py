from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Any

from app.models import ComposerOutput, MessageBrief


@dataclass(frozen=True)
class ValidationResult:
    valid: bool
    output: ComposerOutput
    reasons: list[str]


class OutputValidator:
    """Conservative post-model validation; invalid output is never sendable."""

    max_message_length = 500

    def validate(self, candidate: ComposerOutput, brief: MessageBrief) -> ValidationResult:
        reasons: list[str] = []
        if candidate.audience != brief.audience:
            reasons.append("audience does not match MessageBrief")
        if candidate.language != brief.language_preference:
            reasons.append("language does not match normalized language preference")
        if candidate.should_send and (candidate.action != "send" or not candidate.message.strip()):
            reasons.append("sendable output requires action=send and a non-empty message")
        if not candidate.should_send and candidate.action == "send":
            reasons.append("non-send output cannot use action=send")
        if len(candidate.message) > self.max_message_length:
            reasons.append("message exceeds safe length")
        if candidate.message.count("?") > 1:
            reasons.append("message contains multiple primary CTAs")
        self._validate_fact_references(candidate, brief, reasons)
        self._validate_claims(candidate.message, brief, reasons)
        if reasons:
            return ValidationResult(False, self.safe_failure("; ".join(reasons)), reasons)
        return ValidationResult(True, candidate, [])

    @staticmethod
    def safe_failure(reason: str) -> ComposerOutput:
        return ComposerOutput(
            message="",
            audience="merchant",
            language="unknown",
            objective="suppress unsafe composer output",
            cta_type="none",
            action="wait",
            facts_used=[],
            confidence=0,
            should_send=False,
            suppression_reason=reason,
        )

    @staticmethod
    def _validate_fact_references(candidate: ComposerOutput, brief: MessageBrief, reasons: list[str]) -> None:
        known_keys = set(brief.known_facts)
        forbidden = set(brief.forbidden_assumptions)
        for fact in candidate.facts_used:
            if fact not in known_keys:
                reasons.append(f"facts_used references unavailable fact: {fact}")
            if fact in forbidden:
                reasons.append(f"facts_used references forbidden assumption: {fact}")

    def _validate_claims(self, message: str, brief: MessageBrief, reasons: list[str]) -> None:
        lowered = message.lower()
        payload = brief.known_facts.get("trigger_payload", {})
        if not isinstance(payload, dict):
            payload = {}

        if payload.get("placeholder") is True and re.search(r"\b\d+(?:\.\d+)?%\b|₹\s*\d", message):
            reasons.append("placeholder trigger contains fabricated percentage or price")

        if brief.trigger_kind == "milestone_reached" and payload.get("is_imminent") is True:
            value = payload.get("milestone_value")
            if value is not None and re.search(rf"\b{re.escape(str(value))}\s+reviews?\s+(achieved|reached)", lowered):
                reasons.append("imminent milestone represented as achieved")

        if brief.trigger_kind == "recall_due":
            allowed_slots = {slot.get("label", "").lower() for slot in payload.get("available_slots", []) if isinstance(slot, dict)}
            mentioned_slots = re.findall(r"\b[A-Z][a-z]{2}\s+\d{1,2}\s+[A-Z][a-z]{2},\s*\d{1,2}(?:am|pm)\b", message)
            if any(slot.lower() not in allowed_slots for slot in mentioned_slots):
                reasons.append("message contains an unsupported appointment slot")

        if brief.trigger_kind == "regulation_change" and re.search(r"\b(compliant|non-compliant|legal(?:ly)? required|you must comply)\b", lowered):
            reasons.append("message makes an unsupported compliance or legal claim")

        if brief.trigger_kind == "competitor_opened" and "competitor_name" not in payload and "competitor" in lowered:
            reasons.append("message introduces competitor information absent from trigger")

        self._validate_percentage(message, payload, reasons)
        self._validate_price(message, brief, reasons)
        self._validate_iso_date(message, payload, reasons)

        if brief.audience == "customer" and brief.consent_state and brief.consent_state.whatsapp_consent is None:
            if re.search(r"\b(opted in|your consent|you consented|permission to message)\b", lowered):
                reasons.append("customer message claims unknown WhatsApp consent")

    @staticmethod
    def _validate_percentage(message: str, payload: dict[str, Any], reasons: list[str]) -> None:
        percentages = re.findall(r"(?<!\d)(\d+(?:\.\d+)?)%", message)
        allowed: set[str] = set()
        if isinstance(payload.get("delta_pct"), (int, float)):
            allowed.add(str(abs(payload["delta_pct"]) * 100).rstrip("0").rstrip("."))
        for value in payload.values():
            if isinstance(value, str):
                allowed.update(re.findall(r"(?:_|\b)[+-]?(\d+(?:\.\d+)?)", value))
        if any(percent not in allowed for percent in percentages):
            reasons.append("message contains an unsupported percentage")

    @staticmethod
    def _validate_price(message: str, brief: MessageBrief, reasons: list[str]) -> None:
        mentioned = re.findall(r"₹\s*([\d,]+)", message)
        if not mentioned:
            return
        allowed = set()
        for offer in brief.known_facts.get("active_offers", []):
            allowed.update(re.findall(r"₹\s*([\d,]+)", str(offer)))
        allowed.update(re.findall(r"₹\s*([\d,]+)", str(brief.known_facts.get("trigger_payload", {}))))
        if any(value not in allowed for value in mentioned):
            reasons.append("message contains an unsupported price")

    @staticmethod
    def _validate_iso_date(message: str, payload: dict[str, Any], reasons: list[str]) -> None:
        mentioned = re.findall(r"\b\d{4}-\d{2}-\d{2}\b", message)
        if not mentioned:
            return
        allowed = set(re.findall(r"\d{4}-\d{2}-\d{2}", str(payload)))
        if any(value not in allowed for value in mentioned):
            reasons.append("message contains an unsupported date")
