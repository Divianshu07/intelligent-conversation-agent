# Magicpin AI Challenge — Vera Rebuild

A FastAPI bot implementing the 5-endpoint judge-harness contract
(`challenge-testing-brief.md`) and the 4-context composition model
(`challenge-brief.md`) for the magicpin AI Challenge.

## Approach

Every inbound trigger flows through four deterministic stages before any
copy is generated, so nothing reaches the merchant/customer without being
traceable back to real stored data:

1. **`ContextResolver`** — loads the category/merchant/trigger/(customer)
   contexts a trigger needs from a versioned SQLite `ContextStore`. Missing
   or unresolvable context raises immediately; it never falls through to a
   guess.
2. **`TriggerRouter`** — maps the trigger's `kind` to an objective, a CTA
   shape, and an explicit list of *forbidden assumptions* (facts the
   composer must not invent), and assembles a `MessageBrief` containing
   only facts actually present in the stored contexts.
3. **`AIComposer`** — sends the brief to Gemini (structured output) when
   `GEMINI_API_KEY` is configured; otherwise (and always, in this
   evaluation environment) falls back to `FallbackComposer`, a rule-based,
   per-trigger-kind template engine that only sends when every fact its
   template needs is actually present — otherwise it returns `wait` rather
   than fabricate.
4. **`OutputValidator`** — even a "successful" Gemini response is rejected
   (safe fallback substituted) if it references a fact outside the
   brief's `known_facts`, i.e. hallucination is structurally blocked, not
   just prompted against.

On top of composition: `/v1/tick` now **deduplicates by `suppression_key`**
so a trigger that's still "available" across several 5-minute ticks isn't
resent verbatim (anti-repetition, challenge-brief.md §11); `/v1/reply`
detects auto-replies (same message repeated 3×), hostile/opt-out language,
and commitment language ("let's do it") to switch from qualification to
action immediately (the intent-handoff failure the brief calls out).
Detected auto-replies end the conversation with a short, non-fabricated
closing line (`action="end"`) rather than going silent — matching the
brief's own Pattern B "gold standard" (§9) and the testing brief's Phase 4
"must detect and exit gracefully" replay requirement (§4). `/v1/teardown`
wipes all state (including the new suppression ledger) for a clean re-run.

Every `send` rationale (from `/v1/tick` and `submission.jsonl`) states the
objective plus the exact `known_facts` keys the message is grounded on
(e.g. `"...invite investigation or action Grounded on: merchant_name,
metric, delta_pct, window."`), since the testing brief scores rationale
quality and `facts_used` is already a pre-validated subset of the brief's
known facts — nothing is added that wasn't already checked.

`PromptBuilder`'s per-`kind` trigger instructions (used only on the live
Gemini path) now cover all 26 routable trigger kinds, not just the 18 that
happen to appear in the canonical 30. The 8 that were missing
(`research_digest`, `renewal_due`, `review_theme_emerged`,
`seasonal_perf_dip`, `supply_alert`, `trial_followup`, `winback_eligible`,
`wedding_package_followup`) each have at least one fully-grounded seed
trigger in the dataset; before this fix, a live run with `GEMINI_API_KEY`
configured would have silently dropped every trigger of those kinds (an
uncaught `KeyError` in `PromptBuilder.build`, swallowed by `TickService`'s
broad exception handler) — including `research_digest`, the brief's own
flagship example (Appendix A). `tests/test_tick_gemini_wiring.py` now
asserts this parity so it can't regress.

## Tradeoffs

- **Restraint over reach.** The fallback composer only sends when its
  template's required fields are all present. Given this environment has
  no LLM key, a meaningful fraction of trigger kinds/pairs legitimately
  produce `wait` rather than a plausible-but-invented message — see
  `submission.jsonl` below. With a live LLM (still passing through
  `OutputValidator`) that fraction would likely shrink for pairs with rich
  context, but never sends without a validated grounding.
- **Rule-based, not learned, routing.** `TriggerRouter`'s per-kind
  objective/CTA table is easy to audit and extend, but every new trigger
  `kind` needs an explicit routing rule and (for the no-LLM path) an
  explicit `FallbackComposer` template — it doesn't generalize to unseen
  kinds by itself.
- **SQLite over an external store.** Simple, durable across process
  restarts within a test window, versioned by `(scope, context_id)` — but
  single-process only, which matches the challenge's scale (one bot, one
  test window) and isn't intended to scale beyond it.

## What additional context would have helped most

- A live `GEMINI_API_KEY` for this evaluation run — an LLM path with the
  same validator would cover new trigger kinds without adding a bespoke
  `FallbackComposer` template for every one. `chronic_refill_due`,
  `ipl_match_today`, and `appointment_tomorrow` now have grounded
  fallback templates (gated on `refill_consent` /
  `appointment_reminder_consent` where the kind is customer-facing); a
  new kind still needs a Gemini fallback or a new template until one is
  added, per `customer_lapsed_soft`, which remains unsupported — every
  `customer_lapsed_soft` trigger the dataset generator produces is
  `payload: {"placeholder": true}` with zero real facts ever attached, so
  no template (fallback or Gemini) could compose one without inventing a
  lapse reason, service, or offer; this is a genuine dataset gap, not an
  implementation gap.
- A canonical mapping from consent-scope labels to the challenge's
  four named consent scopes — the seed dataset uses labels like
  `"winback_offers"` and `"renewal_reminders"`. `"winback_offers"` is now
  treated as equivalent to `"promotional_offers"` for
  `customer_lapsed_hard`; other non-standard scope labels
  (`"renewal_reminders"`, `"bridal_package_followup"`, etc.) still fall
  outside the four named consent scopes and are conservatively not
  mapped.

## Submission deliverables

### `submission.jsonl`

30 lines, one per canonical `(merchant, trigger[, customer])` test pair, as
required by `challenge-brief.md` §7.2. Each line has exactly:
`test_id`, `body`, `cta`, `send_as`, `suppression_key`, `rationale`.

Generated by `scripts/generate_submission.py`, which:

1. Deterministically expands the seed dataset (`dataset/*_seed.json`,
   `dataset/categories/`) into the full 50-merchant / 200-customer /
   100-trigger dataset and the canonical 30 test pairs, by calling
   `dataset/generate_dataset.py`'s own functions unmodified (fixed seed —
   same dataset and same 30 pairs every run).
2. Loads every context into the project's real `ContextStore`.
3. Composes each pair through the project's actual pipeline
   (`ContextResolver` → `TriggerRouter` → `AIComposer`) — the exact code
   path `/v1/tick` uses — with no network calls (no `GEMINI_API_KEY` here,
   so `AIComposer` uses its deterministic `FallbackComposer`).
4. Writes one line per pair, faithfully reporting whatever the composer
   actually produced, including an honest `wait` (`body: ""`, `cta:
   "none"`) when no safe, non-fabricated message could be composed.

Regenerate with:

```bash
python scripts/generate_submission.py
```

**Result on this dataset: 17 of 30 pairs send a grounded message; 13 wait.**
All 13 `wait` lines are honest, non-fabrication declines — never a crash
or an invented fact:

- **Placeholder trigger data (13 pairs)** — the dataset generator marks
  ~75 of the 100 expanded triggers as `payload: {"placeholder": true}`
  with no real facts attached; roughly a third of the canonical pairs draw
  one of these. This includes both `appointment_tomorrow` pairs and one of
  the two `chronic_refill_due` pairs, whose `FallbackComposer` templates
  now exist but still have no real facts to compose from, as well as
  `customer_lapsed_soft` (2 pairs), which has no `FallbackComposer`
  template yet — see "What additional context would have helped most"
  above. There is nothing true to say about any of these without
  inventing detail, so the composer correctly declines.

`chronic_refill_due`, `ipl_match_today`, and `appointment_tomorrow` each
now have a grounded `FallbackComposer` template, and `"winback_offers"`
consent is recognized for `customer_lapsed_hard` — three canonical pairs
(`chronic_refill_due` for a real refill, `ipl_match_today`, and
`customer_lapsed_hard` for a `winback_offers`-consented customer) moved
from `wait` to a grounded `send` as a result.

`tests/test_submission.py` validates the file's schema, uniqueness,
internal consistency (e.g. an empty body never carries a CTA), full
determinism of the generator, and that every non-empty `suppression_key`
traces back to a real trigger's own field — never invented.

### `bot.py`

ASGI entry point (`from app.api import app`) exposing all 5 harness
endpoints: `/v1/context`, `/v1/tick`, `/v1/reply`, `/v1/healthz`,
`/v1/metadata`, plus the optional `/v1/teardown`.

## Setup

Create and activate a virtual environment:

```powershell
py -m venv .venv
.\.venv\Scripts\Activate.ps1
```

Install dependencies:

```powershell
pip install -r requirements.txt
```

Optionally copy `.env.example` to `.env` and fill in team metadata / a
`GEMINI_API_KEY` (the bot runs correctly without one, via the deterministic
fallback composer).

## Run locally

```powershell
uvicorn bot:app --host 0.0.0.0 --port 8080
```

Verify the health endpoint from another terminal:

```powershell
Invoke-RestMethod http://127.0.0.1:8080/v1/healthz
```

## Tests

```powershell
pytest
```

SQLite state defaults to `data/context_store.db`; set `MAGICPIN_DATABASE_PATH`
in `.env` to use another local path.
