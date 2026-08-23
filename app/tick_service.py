from __future__ import annotations

from app.composer import AIComposer
from app.context_resolver import ContextResolutionError, ContextResolver
from app.context_store import ContextStore
from app.models import TickAction, TickRequest, TickResponse
from app.trigger_router import TriggerRouter

# Reuses ContextStore under a dedicated scope (like ReplyService's
# "conversation" scope) rather than introducing a second persistence
# mechanism. Once a suppression_key has produced a sent action, it is
# recorded here so later ticks that still list the same (unexpired) trigger
# as "available" don't recompose and resend the same message verbatim.
# This is the anti-repetition contract the testing brief calls for:
# suppression_key exists "for dedup", and re-sending an identical body in
# the same conversation is an explicitly penalized anti-pattern.
SUPPRESSION_SCOPE = "suppression"


class TickService:
    """Orchestrates context resolution, routing, composition, and tick output."""

    def __init__(
        self,
        store: ContextStore,
        composer: AIComposer | None = None,
        resolver: ContextResolver | None = None,
        router: TriggerRouter | None = None,
    ) -> None:
        self._store = store
        self._resolver = resolver or ContextResolver(store)
        self._router = router or TriggerRouter()
        self._composer = composer or AIComposer()

    def tick(self, request: TickRequest) -> TickResponse:
        actions: list[TickAction] = []

        for trigger_id in request.available_triggers:
            try:
                contexts = self._resolver.resolve(trigger_id)
                brief = self._router.build_brief(
                    contexts,
                    request_id=trigger_id,
                )

                if brief.suppression_key and self._store.get(
                    SUPPRESSION_SCOPE, brief.suppression_key
                ) is not None:
                    # Already sent for this suppression_key; the trigger is
                    # still listed as "available" (e.g. it hasn't expired
                    # yet) but re-sending would be a verbatim repeat.
                    continue

                output = self._composer.compose(brief)

                if not output.should_send or output.action != "send":
                    continue

                trigger_payload = contexts.trigger.payload

                merchant_id = str(
                    trigger_payload.get("merchant_id")
                    or contexts.merchant.context_id
                )

                customer_id = trigger_payload.get("customer_id")
                if customer_id is not None:
                    customer_id = str(customer_id)

                conversation_id = str(
                    trigger_payload.get("conversation_id")
                    or f"conv_{merchant_id}"
                )

                send_as = (
                    "merchant_on_behalf"
                    if output.audience == "customer"
                    else "vera"
                )

                actions.append(
                    TickAction(
                        conversation_id=conversation_id,
                        merchant_id=merchant_id,
                        customer_id=customer_id,
                        send_as=send_as,
                        trigger_id=contexts.trigger.context_id,
                        template_name=f"ai_{contexts.trigger.payload.get('kind', 'general')}",
                        template_params=output.facts_used,
                        body=output.message,
                        cta=output.cta_type,
                        suppression_key=brief.suppression_key,
                        rationale=output.to_rationale(),
                    )
                )

                if brief.suppression_key:
                    self._store.put(
                        SUPPRESSION_SCOPE,
                        brief.suppression_key,
                        1,
                        {"trigger_id": contexts.trigger.context_id, "body": output.message},
                    )

            except (ContextResolutionError, ValueError, KeyError, TypeError):
                # A malformed or incomplete trigger must never produce
                # a fabricated outbound message.
                continue

        return TickResponse(actions=actions)