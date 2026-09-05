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

### Phase 3 — LangGraph and bounded investigation agent (next; friend-owned)

Phase 3 turns the tested Phase 2 detector/router into a real investigative
agent. It must not replace or retrain Phase 2 components. Its input boundary is
`CandidateAssessment` plus the associated `RingCandidate` and redacted merchant
applications.

#### Goal

Determine whether the observed relationships are more consistent with a
coordinated abuse ring or a legitimate shared-infrastructure explanation, and
identify the next evidence a human needs. The agent recommends or abstains; it
does not calculate risk or own the final action.

#### Suggested new modules

```text
merchantshield/investigation/
├── __init__.py
├── contracts.py       # hypotheses, recommendation, request and grounding models
├── tools.py           # allow-listed read-only evidence tools
├── investigator.py    # bounded provider-agnostic investigation loop
├── grounding.py       # deterministic evidence-ID authorization/validation
└── memory.py          # structured prior-case evidence lookup

merchantshield/workflow/
├── __init__.py
├── state.py           # serializable LangGraph case state
└── graph.py           # nodes, edges, checkpoint, interrupt and resume

tests/
├── test_investigation_tools.py
├── test_investigator.py
├── test_grounding.py
└── test_workflow_graph.py
```

#### Required graph state

- `case_id` and idempotency key;
- candidate and redacted merchant inputs;
- Phase 2 `CandidateAssessment` and `ExpertResult` values;
- available evidence-ID catalogue;
- investigation hypotheses and tool observations;
- structured investigator output;
- grounding status and grounding errors;
- requested information and supplied follow-up evidence;
- route trace, failure state, review status and step count.

State must be serializable and safe to checkpoint. Do not place API keys, raw
uploaded documents, full Aadhaar numbers, or unredacted logs in graph state.

#### Read-only investigator tools

| Tool | Contract |
|---|---|
| `get_candidate_summary(candidate_id)` | Topology, primary score, component size, expert status |
| `get_relationship_evidence(candidate_id, filters)` | Authorized redacted evidence records and IDs |
| `get_member_profile(candidate_id, member_id)` | One redacted application profile |
| `compare_submission_timeline(candidate_id)` | Submission span, bursts and ordering |
| `get_prior_review_history(evidence_ids)` | Structured prior outcomes for related hashed infrastructure |
| `inspect_document_consistency(candidate_id, member_id)` | Phase 3.5 status or explicit `unavailable` |

Tools must validate candidate/member scope, return structured objects, record a
trace entry, and never mutate a merchant decision. No arbitrary filesystem,
network, SQL, shell, MCP or unrestricted retrieval tool is permitted.

#### Investigation loop

1. Receive only cases marked `investigator_required` by Phase 2.
2. Form a coordinated-ring hypothesis and at least one plausible legitimate
   alternative.
3. Select the next read-only tool based on unresolved uncertainty.
4. Add the returned observation to state.
5. Repeat while information value remains and execution budget permits.
6. Return a structured recommendation, request information, or abstain.
7. Validate every citation against the authorized evidence catalogue.
8. Route invalid grounding, timeout, budget exhaustion and provider failure to
   human review.

Default demonstration budgets:

- maximum 5 investigation steps;
- maximum 6 tool calls;
- maximum 15 seconds wall-clock time;
- configurable model-token budget;
- no recursive delegation and no second LLM agent.

#### Structured output

The investigator response must include:

- `recommendation`: `continue_onboarding`, `request_information`,
  `human_review`, or `abstain`;
- `ring_hypothesis` and supporting evidence IDs;
- at least one `legitimate_alternative` and supporting/contradicting evidence;
- `missing_evidence`;
- `requested_information` when applicable;
- `confidence` describing the recommendation, not a replacement risk score;
- `cited_evidence_ids`;
- concise reviewer narrative.

#### Grounding and recovery

- A citation is valid only if it exists in the case evidence catalogue and the
  current tool authorization scope.
- A nonexistent citation invalidates the narrative and sets
  `insufficient_grounding`.
- Invalid JSON, timeout, provider error, tool error and exhausted budget are
  explicit states, never clean results.
- LLM-disabled mode must use a deterministic fake investigator for tests and a
  clearly labelled `LLM_DISABLED` abstention in the demonstration.
- The deliberate failure demo must inject a nonexistent evidence ID, catch it,
  preserve the trace and route the case to a human.

#### Information request and resume

When decisive evidence is missing, the agent may create a structured request
for items such as an accountant authorization, franchise agreement, explanation
of a shared settlement account, or clearer synthetic document. LangGraph must
checkpoint and interrupt the case. Supplying follow-up evidence resumes the same
case and preserves the earlier trace; it must not silently start a new case.

#### Structured case memory

Store exact prior outcomes for hashed infrastructure, for example reviewed
device/address/account hashes and whether a human resolved the associated case
as a confirmed ring, legitimate accountant cluster, franchise, coworking group,
or inconclusive. This is structured lookup, not free-form LLM memory and not
vector RAG. Prior memory may inform investigation but may not directly overwrite
the primary graph score.

#### Phase 3 definition of done

- Existing 52 tests still pass.
- New workflow tests require no real API key.
- Low-risk cases skip the investigator.
- Review cases demonstrate selective tool choice, not unconditional fan-out.
- Every narrative citation is validated.
- Request-information cases pause and resume with the same case ID.
- Missing evidence, timeout, invalid JSON and hallucinated citations fail safe.
- LLM output never changes graph risk or directly approves/rejects a merchant.
- A deterministic end-to-end scenario produces a complete auditable trace.

### Phase 3.5 — Synthetic paper-document slice

- Generate clearly marked, non-functional synthetic document bundles.
- Include clean, photocopy, blur, glare, skew, crop, low-contrast,
  transliteration, old-address and mismatch variants.
- Keep every source document and its variants in one dataset partition.
- Implement quality assessment, preprocessing, OCR abstraction and deterministic
  application-field comparison behind the shared expert/evidence contracts.
- Report document metrics separately from ring-detection metrics.
- Treat `missing`, `unreadable` and `verification_unavailable` as unknown/review,
  never clean.

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
