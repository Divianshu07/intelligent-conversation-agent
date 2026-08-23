from __future__ import annotations

from dataclasses import dataclass

from app.context_store import ContextStore, StoredContext


class ContextResolutionError(ValueError):
    """Raised when a mandatory context required by a trigger is unavailable."""


@dataclass(frozen=True)
class ResolvedContexts:
    category: StoredContext
    merchant: StoredContext
    trigger: StoredContext
    customer: StoredContext | None


class ContextResolver:
    """Resolves the latest stored four-context set for one trigger."""

    def __init__(self, store: ContextStore) -> None:
        self._store = store

    def resolve(
        self,
        trigger_id: str,
        merchant_id: str | None = None,
        customer_id: str | None = None,
    ) -> ResolvedContexts:
        trigger = self._required("trigger", trigger_id)
        trigger_payload = trigger.payload
        resolved_merchant_id = merchant_id or trigger_payload.get("merchant_id")
        if not resolved_merchant_id:
            raise ContextResolutionError(f"Trigger {trigger_id!r} does not identify a merchant.")

        merchant = self._required("merchant", str(resolved_merchant_id))
        category_slug = merchant.payload.get("category_slug")
        if not category_slug:
            raise ContextResolutionError(f"Merchant {merchant.context_id!r} does not identify a category.")
        category = self._required("category", str(category_slug))

        resolved_customer_id = customer_id or trigger_payload.get("customer_id")
        customer = self._store.get("customer", str(resolved_customer_id)) if resolved_customer_id else None
        return ResolvedContexts(category=category, merchant=merchant, trigger=trigger, customer=customer)

    def _required(self, scope: str, context_id: str) -> StoredContext:
        context = self._store.get(scope, context_id)  # type: ignore[arg-type]
        if context is None:
            raise ContextResolutionError(f"Missing {scope} context {context_id!r}.")
        return context
