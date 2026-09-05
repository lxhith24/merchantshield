# MerchantShield build plan

## Locked outcome

Deliver one complete defensive vertical slice: detect coordinated merchant
onboarding rings, explain the supplied evidence, and route risky or uncertain
cases to a human. Do not claim to identify every shell company or infer hidden
beneficial ownership.

## Runtime architecture

| Layer | Responsibility | Decision authority |
|---|---|---|
| Normalization + evidence graph | Link applications using hashed shared infrastructure | None |
| Rules expert | Catch obvious consistency failures | May trigger review only |
| Graph model expert | Produce the primary numerical ring-risk score | Numerical risk authority |
| Tabular expert | Provide a selectively routed second view | May trigger review only |
| LLM investigator | Explain evidence and consider legitimate alternatives | Recommend or abstain only |
| Evidence validator | Reject nonexistent citations and invalid structured output | Grounding authority |
| Deterministic policy | Permit onboarding continuation or require review | Action authority |
| Human reviewer | Resolve a held case | Final high-risk authority |

Agent count: **one LLM agent**. Rules, graph, tabular, validation, and policy are
experts/components, not autonomous conversational agents.

## Milestones

### Phase 1 — Evaluation foundation (complete)

- 104 labelled synthetic applications across five cohorts.
- Fraud-ring/connected-component-safe train/test split.
- Rules, clustering, graph, tabular, graph-ML, and static-hybrid comparisons.
- Precision, recall, F1, PR-AUC, FPR, ring recall, review load, and FP cost.

### Phase 2 — Stable contracts and sparse MoE (complete)

- Typed expert results with evidence, confidence, abstention/error, and timing.
- Concurrent mandatory rules + graph execution.
- Conditional tabular routing for uncertainty/disagreement.
- Bounded expert timeout and failure isolation.
- No rejection action; unavailable primary evidence always routes to a human.
- Routed-MoE benchmark and routing-usage report.

### Phase 3 — LangGraph and grounded investigator (next)

- Graph state for candidate, expert outputs, route trace, investigation, policy,
  and review status.
- Conditional edges for optional expert and investigator execution.
- One bounded LLM investigator with read-only evidence tools.
- Strict JSON response containing recommendation, alternatives, evidence IDs,
  uncertainty, and abstention.
- Deterministic citation validator. A nonexistent evidence ID invalidates the
  narrative and forces `insufficient_grounding` human review.
- LLM-disabled, timeout, invalid-JSON, and hallucinated-citation tests.

### Phase 4 — Review persistence and API

- Append-only case events and separate automated versus human decisions.
- Reviewer identity, reason codes, claim/resolve endpoints, and optimistic
  concurrency protection.
- Idempotency keys and resumable review state.

### Phase 5 — Demonstration surface and Razorpay adapter

- Streamlit ring graph, evidence panel, expert route trace, and review queue.
- Frozen deterministic demo scenarios including a legitimate shared-office
  group and an evasive ring.
- Simulated onboarding gateway by default; Razorpay Partner test adapter only
  when account access exists.
- Clear `SIMULATED`, `RAZORPAY_TEST`, and `LLM_DISABLED` labels.

### Phase 6 — Acceptance and recording

- Full test suite and clean install/start/seed/reset flow.
- Threshold report on frozen held-out data plus explicit synthetic disclaimer.
- Demonstrate one LLM citation failure and graceful human-review recovery.
- Record the five-minute flow only after the dataset and outputs are frozen.

## Stop conditions

- Do not add voice, vector RAG, MCP, document-forgery vision, live Aadhaar/GST/MCA
  scraping, or post-activation transaction monitoring before the core demo works.
- Do not let an LLM calculate risk, lower graph risk, create a blocklist, or call
  a merchant onboarding API directly.
- Do not show real identities, API keys, or production-looking synthetic metrics.
