# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## What this is

MerchantShield is a defensive merchant-onboarding risk investigator built for the Razorpay AI
Buildathon. It detects groups of merchant applications that may be controlled together (fraud
rings), explains the shared evidence, and hands high-risk or uncertain cases to a **human
reviewer** — it never autonomously blacklists or rejects a merchant. Every identity, document,
link, and metric in this repo is synthetic; results are not production Razorpay performance.

Full rules and required architecture live in `AGENTS.md` — read it before making structural
changes. Key constraints from there:

- Hybrid of deterministic rules + tabular ML + graph analysis + LLM reasoning + human review.
- A system-level sparse mixture of experts (routing between existing components), **not** a
  trained neural MoE and not multiple chatbots debating each other.
- At most **two** LLM agents ever: an investigator and an evidence-grounding reviewer. Currently
  only the investigator is active — grounding is deterministic (see below).
- Deterministic code owns the numerical risk score and permitted actions. LLM output can raise a
  case to human review but can never lower risk, set the score, or approve/reject/blacklist.
- Missing or unavailable evidence is never treated as clean; it always forces review.
- Never commit API keys, PII, uploaded documents (`data/uploads/`), the SQLite DB
  (`merchantshield.db`), or `.venv`.

## Commands

```bash
python -m pip install -r requirements.txt   # install (or let ./run.sh create .venv)
./run.sh                                    # API :8000 + Streamlit reviewer UI :8501
./run.sh --api-only                         # API only
python -m pytest -q                         # full test suite (use .venv/bin/python if no venv active)
python -m pytest tests/test_risk_scorer.py                       # one file
python -m pytest tests/test_risk_scorer.py::test_midband_goes_to_review  # one test
python scripts/acceptance.py                # Phase 6 end-to-end acceptance checks (throwaway DB, no API key needed)
```

Tests always run against an isolated temp SQLite DB and force `ANTHROPIC_API_KEY=""`
(heuristic-only mode) via `tests/conftest.py`, regardless of your local `.env`.

There is no lint/typecheck command configured in this repo (no ruff/mypy config present).

## Architecture

The system is layered so that **numerical risk and permitted actions never come from an LLM**.
Read `README.md`'s Architecture section and mermaid diagram for the full picture; summary:

```
applications → evidence graph → [rules expert, graph ML expert] → deterministic sparse router
                                                                   → tabular expert (if uncertain/disagreement)
                                                                   → LangGraph workflow
                                                                       → bounded investigator (LLM, optional)
                                                                       → citation/grounding validator
                                                                       → human review
                                                                           → simulated/Razorpay-test gateway (only on verified approval)
```

| Concern | Owns | Location |
|---|---|---|
| Evidence linking | Builds ring candidates from shared hashed infrastructure (address, device, IP, bank, etc.) | `merchantshield/analysis/evidence_graph.py` |
| Rules expert | Obvious deterministic consistency checks; may only trigger review | `merchantshield/agent/experts.py` |
| Graph ML expert | Fitted on the training split only; **owns the primary numerical risk score** | `merchantshield/evaluation/ring_models.py` (`GraphMLBaseline`) |
| Tabular expert | Selectively routed second opinion under uncertainty/disagreement | `merchantshield/evaluation/ring_models.py` (`TabularOnlyBaseline`) |
| Sparse router + policy | Deterministic expert selection, timeouts/failure isolation, permitted-action policy | `merchantshield/agent/router.py` (`RoutedMoESystem`, `RoutedMoEConfig`) |
| LangGraph workflow | Case state, conditional routing, investigation, pause/resume via `interrupt()` | `merchantshield/workflow/graph.py`, `merchantshield/workflow/state.py` |
| Investigator (the one LLM agent) | Evidence-bound explanation, legitimate alternatives, abstention; read-only tools only | `merchantshield/investigation/investigator.py`, `investigation/tools.py`, `investigation/contracts.py` |
| Grounding validator | Deterministically rejects fabricated/undisclosed evidence IDs → forces `insufficient_grounding` review | `merchantshield/investigation/grounding.py` |
| Review persistence | Append-only case events, human vs. automated decisions, optimistic concurrency, reason codes | `merchantshield/review/service.py`, `review/store.py`, `db/` |
| Gateway | Onboarding hand-off; only ever called after a recorded human `approve_onboarding` | `merchantshield/gateway/base.py`, `gateway/simulated.py`, `gateway/razorpay.py` |
| Safe API | The mounted, safe API surface | `merchantshield/api/case_routes.py`, `api/demo_routes.py` |
| Legacy API | Old **automatic-decision** API — isolated test fixture only, deliberately **not mounted** on the real app (`legacy_app` in `main.py`) because it can auto-reject | `merchantshield/api/routes.py` |
| Reviewer UI | Decision-first Streamlit app | `merchantshield/ui/streamlit_app.py` |
| Runtime assembly | Wires models/workflow/store/gateway into one process-wide singleton (`get_runtime()`) | `merchantshield/runtime.py` |
| Legacy per-application pipeline | Older synchronous scorer (`OnboardingGate`), used by `pipeline.py`/older routes, distinct from the LangGraph workflow above | `merchantshield/pipeline.py` |

Two overlapping systems exist by design: the **legacy `OnboardingGate` pipeline**
(`pipeline.py`, `api/routes.py`) which scores and can auto-approve/reject a single application in
isolation, and the current **LangGraph-based `MerchantShieldWorkflow`** (`workflow/graph.py`,
`runtime.py`) which is ring-aware, sparse-MoE-routed, and routes high-risk/uncertain cases to a
human. Only the latter is mounted in `main.py`'s default `app`; treat the legacy path as frozen
compatibility surface, not a place for new features.

### Modes, always visible in the UI

- `LLM_ENABLED` / `LLM_DISABLED` (`SCRIPTED` in tests) — set by whether `ANTHROPIC_API_KEY` is
  present (`merchantshield/config.py`). Without a key, document vision and LLM explanations are
  skipped but deterministic rules/scoring still run fully.
- `SIMULATED` / `RAZORPAY_TEST` gateway mode (`merchantshield/gateway/`) — simulated by default;
  the Razorpay Partner test adapter activates only with explicit env config, and a live
  (`rzp_live_...`) key is refused.
- `data: SIMULATED` — every demo/eval dataset is synthetic; never presented as production metrics.

### Evaluation datasets (do not conflate)

Several frozen synthetic datasets and reports coexist under `data/evaluation/` — each backs a
different narrative in `README.md` and a different `?dataset=` query param on
`GET /api/v1/evaluation/report`. When changing evaluation code, check which dataset/report pair
you're touching:

- `synthetic_merchants.jsonl` / `latest_report.json` — original 104-row demo dataset (`dataset.py`), `?dataset=demo`.
- `expanded_merchants.jsonl` / `expanded_report.json` — 2,002-application benchmark (`expanded_dataset.py`), default and `?dataset=expanded`.
- `performance_merchants.jsonl` / `performance_plan.json` / `performance_report.json` — merchant/peer routed model experiment (`evaluation/performance.py`), `?dataset=performance`; this is the **current validation page** default.
- `relationships/` — rejected "wider-context" experiment (`evaluation/relationships.py`), `?dataset=relationships`.

Rebuilding any of these (`scripts/build_expanded_benchmark.py`,
`scripts/improve_risk_models.py`, `scripts/recover_hard_rings.py`) locks train/val/test splits
and checksums before final testing — final-test reports must never be overwritten, and new
experiments need a genuinely new sealed split, not reuse of a prior "final" test. See `README.md`
for the exact reproduction commands and locked numbers if you need to reproduce or extend one of
these.

### Safety invariants to preserve in any change

- An LLM recommendation can raise a case to review; it can never lower one, set the numeric
  score, or trigger an onboarding action.
- Fabricated or undisclosed evidence citations are rejected deterministically
  (`investigation/grounding.py`) and fail safe to human review — don't relax this to "trust the
  LLM's citation."
- Case mutations go through version checks and produce an append-only audit trail
  (`review/store.py`, `review/contracts.py`) — don't overwrite prior events.
- Only a recorded human `approve_onboarding` may reach a gateway (`gateway/base.py`).
- `frontend/` is currently an empty placeholder directory — no frontend code exists there yet.
