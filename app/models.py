from __future__ import annotations

from datetime import datetime
from typing import Any, Literal

from pydantic import BaseModel, Field


ContextScope = Literal["category", "merchant", "customer", "trigger"]
NormalizedLanguage = Literal["en", "hi", "hi-en", "other", "unknown"]


class ContextRequest(BaseModel):
    scope: ContextScope
    context_id: str = Field(min_length=1)
    version: int = Field(ge=1)
    payload: dict[str, Any]
    delivered_at: datetime


class ContextResponse(BaseModel):
    accepted: bool
    ack_id: str | None = None
    stored_at: datetime | None = None
    reason: str | None = None
    current_version: int | None = None
    details: str | None = None


class TickRequest(BaseModel):
    now: datetime
    available_triggers: list[str] = Field(default_factory=list)


class TickAction(BaseModel):
    conversation_id: str
    merchant_id: str
    customer_id: str | None = None
    send_as: Literal["vera", "merchant_on_behalf"]
    trigger_id: str
    template_name: str
    template_params: list[str]
    body: str = Field(min_length=1)
    cta: str
    suppression_key: str
    rationale: str = Field(min_length=1)


class TickResponse(BaseModel):
    actions: list[TickAction] = Field(default_factory=list)


class ReplyRequest(BaseModel):
    conversation_id: str = Field(min_length=1)
    merchant_id: str | None = None
    customer_id: str | None = None
    from_role: Literal["merchant", "customer"]
    message: str
    received_at: datetime
    turn_number: int = Field(ge=1)


class ReplyResponse(BaseModel):
    action: Literal["send", "wait", "end"]
    body: str | None = None
    cta: str | None = None
    wait_seconds: int | None = Field(default=None, ge=1)
    rationale: str = Field(min_length=1)


class HealthResponse(BaseModel):
    status: Literal["ok"]
    uptime_seconds: int = Field(ge=0)
    contexts_loaded: dict[ContextScope, int]


class TeardownResponse(BaseModel):
    """Response for the optional end-of-test cleanup call."""

    status: Literal["ok"]
    cleared: bool = True


class MetadataResponse(BaseModel):
    team_name: str | None
    team_members: list[str]
    model: str
    approach: str
    contact_email: str | None
    version: str
    submitted_at: str | None


class ContextSnapshot(BaseModel):
    """Latest persisted context used to form a deterministic message brief."""

    context_id: str
    version: int
    payload: dict[str, Any]


class ConsentState(BaseModel):
    """Explicit consent only; absent scope is never interpreted as consent."""

    scopes: list[str] = Field(default_factory=list)
    promotional_consent: bool = False
    appointment_reminder_consent: bool = False
    recall_consent: bool = False
    refill_consent: bool = False
    delivery_notification_consent: bool = False
    whatsapp_consent: bool | None = None
    channel: str | None = None


class MessageBrief(BaseModel):
    """Structured input for a future composer; contains no generated copy."""

    request_id: str | None = None
    audience: Literal["merchant", "customer"]
    category: ContextSnapshot
    merchant: ContextSnapshot
    customer: ContextSnapshot | None = None
    trigger_id: str
    trigger_kind: str
    trigger_urgency: int
    suppression_key: str
    known_facts: dict[str, Any]
    forbidden_assumptions: list[str]
    recommended_objective: str
    recommended_cta_type: str
    language_preference: NormalizedLanguage
    consent_state: ConsentState | None = None


class ComposerOutput(BaseModel):
    """Validated structured response from the future-composer boundary."""

    message: str = ""
    audience: Literal["merchant", "customer"]
    language: NormalizedLanguage
    objective: str
    cta_type: str
    action: Literal["send", "end", "wait"]
    facts_used: list[str] = Field(default_factory=list)
    confidence: float = Field(ge=0, le=1)
    should_send: bool
    suppression_reason: str | None = None

    def to_rationale(self) -> str:
        """Human-readable rationale for the judge-facing `rationale` field
        (challenge-testing-brief.md: "the rationale ... is included in the
        scoring rubric"). For a decline, the suppression_reason is already
        specific and honest. For a send, `facts_used` is itself a
        pre-validated subset of the brief's known_facts (every fallback
        template only ever lists the facts it actually referenced, and
        OutputValidator rejects any LLM output that lists a fact outside
        known_facts) -- so naming those facts here adds real traceability
        for the judge without introducing anything not already grounded."""
        if self.suppression_reason:
            return self.suppression_reason
        if self.facts_used:
            return f"{self.objective} Grounded on: {', '.join(self.facts_used)}."
        return self.objective
