from __future__ import annotations

import importlib.util
import json
import re
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parent.parent
SUBMISSION_PATH = REPO_ROOT / "submission.jsonl"

REQUIRED_KEYS = {"test_id", "body", "cta", "send_as", "suppression_key", "rationale"}
VALID_SEND_AS = {"vera", "merchant_on_behalf"}
TEST_ID_PATTERN = re.compile(r"^T\d{2}$")


def _load_generator_script():
    """Import scripts/generate_submission.py by path (mirrors how that
    script itself loads dataset/generate_dataset.py), so tests exercise the
    exact same code that produced the committed submission.jsonl."""
    spec = importlib.util.spec_from_file_location(
        "generate_submission", REPO_ROOT / "scripts" / "generate_submission.py"
    )
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


@pytest.fixture(scope="module")
def submission_lines() -> list[dict]:
    assert SUBMISSION_PATH.exists(), "submission.jsonl must exist at the repo root"
    with open(SUBMISSION_PATH, encoding="utf-8") as fh:
        raw_lines = [line for line in fh.read().splitlines() if line.strip()]
    return [json.loads(line) for line in raw_lines]


# --- 1. Exactly 30 lines, per challenge-brief.md §7.2: "submission.jsonl
#        (30 lines, one per test pair)".
def test_submission_has_exactly_30_lines(submission_lines):
    assert len(submission_lines) == 30


# --- 2. Every line is valid JSON (already implied by the fixture parsing
#        successfully) with exactly the required keys, no more, no fewer.
def test_every_line_has_exactly_the_required_keys(submission_lines):
    for entry in submission_lines:
        assert set(entry.keys()) == REQUIRED_KEYS


# --- 3. test_id values are unique and cover T01..T30 exactly.
def test_ids_are_unique_and_cover_t01_to_t30(submission_lines):
    ids = [entry["test_id"] for entry in submission_lines]
    assert len(ids) == len(set(ids)), "duplicate test_id values found"
    assert sorted(ids) == [f"T{i:02d}" for i in range(1, 31)]
    for test_id in ids:
        assert TEST_ID_PATTERN.match(test_id)


# --- 4. send_as is always one of the two contract values.
def test_send_as_is_a_valid_value(submission_lines):
    for entry in submission_lines:
        assert entry["send_as"] in VALID_SEND_AS


# --- 5. rationale and cta are always non-empty strings (every line must be
#        explainable, even a "no message" line).
def test_rationale_and_cta_are_non_empty(submission_lines):
    for entry in submission_lines:
        assert isinstance(entry["rationale"], str) and entry["rationale"].strip()
        assert isinstance(entry["cta"], str) and entry["cta"].strip()


# --- 6. Internal consistency: an empty body must mean no CTA was offered.
#        A bot can't have "no message" but still present a call-to-action.
def test_empty_body_implies_no_cta(submission_lines):
    for entry in submission_lines:
        if not entry["body"]:
            assert entry["cta"] == "none", entry["test_id"]


# --- 7. No two non-empty bodies are byte-for-byte identical -- the same
#        anti-repetition principle the harness penalizes at runtime
#        (challenge-brief.md §11) should hold across the static submission.
def test_no_verbatim_duplicate_bodies(submission_lines):
    bodies = [entry["body"] for entry in submission_lines if entry["body"]]
    assert len(bodies) == len(set(bodies))


# --- 8. Anti-fabrication: for every "send" line, every word from the
#        composed body that looks like a private compound-copy phrase
#        must be traceable back to the composer's own facts_used mechanism.
#        We can't re-derive natural language easily, but we CAN assert the
#        stronger, cheaper invariant already enforced by OutputValidator
#        end-to-end: re-running the exact same generation pipeline produces
#        byte-identical output (determinism, no hidden randomness/LLM calls
#        snuck into a "grounded" claim).
def test_generation_is_fully_deterministic():
    generator = _load_generator_script()
    categories, merchants, customers, triggers, pairs = generator.build_dataset()
    store_a = generator.load_store(categories, merchants, customers, triggers)
    lines_a = generator.compose_submission_lines(store_a, pairs)
    store_a.close()

    categories2, merchants2, customers2, triggers2, pairs2 = generator.build_dataset()
    store_b = generator.load_store(categories2, merchants2, customers2, triggers2)
    lines_b = generator.compose_submission_lines(store_b, pairs2)
    store_b.close()

    assert lines_a == lines_b
    assert pairs == pairs2


# --- 9. The committed submission.jsonl matches what the generator script
#        currently produces -- i.e. it's not stale / hand-edited out of
#        sync with the pipeline that's supposed to have produced it.
def test_committed_submission_matches_generator_output(submission_lines):
    generator = _load_generator_script()
    categories, merchants, customers, triggers, pairs = generator.build_dataset()
    store = generator.load_store(categories, merchants, customers, triggers)
    fresh_lines = generator.compose_submission_lines(store, pairs)
    store.close()

    assert submission_lines == fresh_lines


# --- 10. Every suppression_key that appears is either empty (only for the
#        unresolved-context safety path) or exactly matches the real
#        trigger's own suppression_key from the dataset -- never invented.
def test_suppression_keys_trace_back_to_real_triggers(submission_lines):
    generator = _load_generator_script()
    categories, merchants, customers, triggers, pairs = generator.build_dataset()
    triggers_by_id = {trigger["id"]: trigger for trigger in triggers}
    pairs_by_test_id = {pair["test_id"]: pair for pair in pairs}

    for entry in submission_lines:
        pair = pairs_by_test_id[entry["test_id"]]
        trigger = triggers_by_id[pair["trigger_id"]]
        if entry["suppression_key"] == "":
            continue
        assert entry["suppression_key"] == trigger["suppression_key"]
