#!/usr/bin/env python3
"""Generate submission.jsonl for the 30 canonical (merchant, trigger) test pairs.

Required by challenge-brief.md §7.2:
    "submission.jsonl (30 lines, one per test pair)"
    {"test_id": "T01", "body": "...", "cta": "...", "send_as": "...",
     "suppression_key": "...", "rationale": "..."}

This script does not fabricate anything. It:

  1. Deterministically expands the seed dataset into the full 50 merchant /
     200 customer / 100 trigger dataset, plus the canonical 30
     ``test_pairs`` -- using dataset/generate_dataset.py's own unmodified
     functions and its fixed SEED, so this is byte-for-byte the same
     dataset + pair selection every participant/run gets.
  2. Loads every context into the project's real ``ContextStore``.
  3. For each of the 30 canonical pairs, resolves contexts and composes a
     message through the project's real pipeline: ``ContextResolver`` ->
     ``TriggerRouter`` -> ``AIComposer`` -- the exact same code path
     ``/v1/tick`` uses in production (see app/tick_service.py).
  4. Writes one JSONL line per pair, faithfully reporting whatever the
     composer actually produced -- including a "no message" (wait) result
     when the composer has no deterministic, non-fabricating way to
     compose one for that trigger kind. Nothing is invented here that the
     composer itself didn't produce.

No network calls: this environment has no GEMINI_API_KEY configured, so
AIComposer (constructed here with no provider) transparently uses its
deterministic FallbackComposer -- already covered by tests/test_composer.py
and tests/test_fallback_trigger_coverage.py. If a provider *is* configured
in another environment, this script would go through the real Gemini path
instead, since it uses the exact same AIComposer class.

Usage:
    python scripts/generate_submission.py [--out submission.jsonl]
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import random
import sys
import tempfile
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
DATASET_DIR = REPO_ROOT / "dataset"

sys.path.insert(0, str(REPO_ROOT))

from app.composer import AIComposer  # noqa: E402
from app.context_resolver import ContextResolutionError, ContextResolver  # noqa: E402
from app.context_store import ContextStore  # noqa: E402
from app.trigger_router import TriggerRouter  # noqa: E402


def _load_generator_module():
    """Import dataset/generate_dataset.py by path, without modifying it."""
    spec = importlib.util.spec_from_file_location(
        "generate_dataset", DATASET_DIR / "generate_dataset.py"
    )
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def build_dataset() -> tuple[dict, list[dict], list[dict], list[dict], list[dict]]:
    """Deterministically expand the seed dataset + canonical 30 test pairs.

    Reuses dataset/generate_dataset.py's own functions verbatim (same fixed
    SEED) so this always matches what `python dataset/generate_dataset.py`
    would produce on disk -- it's just built in-memory here.
    """
    gen = _load_generator_module()
    rnd = random.Random(gen.SEED)

    categories, m_seeds, c_seeds, t_seeds = gen.load_seeds(DATASET_DIR)
    merchants = gen.expand_merchants(m_seeds, rnd)
    customers = gen.expand_customers(c_seeds, merchants, rnd)
    triggers = gen.expand_triggers(t_seeds, merchants, customers, rnd)

    # write_test_pairs's selection is a pure function of `triggers`'s order
    # (grouped by kind, first two per kind) -- it takes `rnd` but never
    # consumes it. Writing to a throwaway temp dir and reading the result
    # back guarantees this script's pair selection can never drift from the
    # canonical generator's, without duplicating its selection logic here.
    with tempfile.TemporaryDirectory() as tmp:
        tmp_path = Path(tmp)
        gen.write_test_pairs(tmp_path, triggers, rnd)
        pairs = json.loads((tmp_path / "test_pairs.json").read_text())["pairs"]

    return categories, merchants, customers, triggers, pairs


def load_store(categories: dict, merchants: list[dict], customers: list[dict], triggers: list[dict]) -> ContextStore:
    store = ContextStore(":memory:")
    for slug, payload in categories.items():
        store.put("category", slug, 1, payload)
    for merchant in merchants:
        store.put("merchant", merchant["merchant_id"], 1, merchant)
    for customer in customers:
        store.put("customer", customer["customer_id"], 1, customer)
    for trigger in triggers:
        store.put("trigger", trigger["id"], 1, trigger)
    return store


def compose_submission_lines(
    store: ContextStore, pairs: list[dict]
) -> list[dict]:
    """Mirrors app/tick_service.py's action-building exactly, but emits one
    line per canonical pair regardless of send/wait outcome, since
    submission.jsonl requires exactly 30 lines (challenge-brief.md §7.2)."""
    resolver = ContextResolver(store)
    router = TriggerRouter()
    composer = AIComposer()  # no provider configured -> deterministic fallback

    lines: list[dict] = []
    for pair in pairs:
        test_id = pair["test_id"]
        trigger_id = pair["trigger_id"]

        try:
            contexts = resolver.resolve(
                trigger_id,
                merchant_id=pair.get("merchant_id"),
                customer_id=pair.get("customer_id"),
            )
            brief = router.build_brief(contexts, request_id=trigger_id)
            output = composer.compose(brief)
        except (ContextResolutionError, ValueError, KeyError, TypeError) as exc:
            # A malformed/unresolvable pair must never produce a fabricated
            # message -- report an explicit, honest "wait" line instead.
            lines.append(
                {
                    "test_id": test_id,
                    "body": "",
                    "cta": "none",
                    "send_as": "vera",
                    "suppression_key": "",
                    "rationale": f"Could not resolve required context: {exc}",
                }
            )
            continue

        send_as = "merchant_on_behalf" if output.audience == "customer" else "vera"
        rationale = output.to_rationale()

        lines.append(
            {
                "test_id": test_id,
                "body": output.message,
                "cta": output.cta_type,
                "send_as": send_as,
                "suppression_key": brief.suppression_key,
                "rationale": rationale,
            }
        )

    return lines


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--out",
        default=str(REPO_ROOT / "submission.jsonl"),
        help="Output path for submission.jsonl (default: repo root)",
    )
    args = parser.parse_args()

    categories, merchants, customers, triggers, pairs = build_dataset()
    store = load_store(categories, merchants, customers, triggers)
    lines = compose_submission_lines(store, pairs)
    store.close()

    out_path = Path(args.out)
    with open(out_path, "w", encoding="utf-8") as fh:
        for line in lines:
            fh.write(json.dumps(line, ensure_ascii=False) + "\n")

    sent = sum(1 for line in lines if line["body"])
    waited = len(lines) - sent
    print(f"Wrote {len(lines)} lines to {out_path} ({sent} send, {waited} wait/no-message).")


if __name__ == "__main__":
    main()
