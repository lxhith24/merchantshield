# MerchantShield — Buildathon submission brief

## 30-second explanation

MerchantShield is a defensive merchant-onboarding review desk. It looks for
clusters of applications that may be controlled by the same operator, explains
which shared signals created the concern, and routes the case to a human. It
is a decision-support system, not an autonomous blacklist.

Every identity, document, relationship and metric in this repository is
synthetic. Nothing here is a claim about Razorpay production performance.

## The problem

Shell merchants can look individually plausible while sharing infrastructure:
the same device, network, settlement account, address or submission pattern.
Checking one merchant at a time misses the relationship. Treating every shared
office, kiosk or family business as fraud creates expensive false positives.

The product question is therefore:

> “Which applications appear related, what evidence supports that relationship,
> and what should a trained reviewer verify next?”

## What is implemented

1. A synthetic onboarding population contains ordinary legitimate merchants,
   legitimate shared-infrastructure groups, obvious shell rings, evasive shell
   rings and borderline merchants.
2. An evidence graph creates weighted links from exact or normalized shared
   attributes and groups connected applications into candidate rings.
3. Deterministic rules check identity/document consistency, missingness and
   timing signals.
4. A graph expert scores relationship topology; a tabular expert scores
   merchant-level signals.
5. A deterministic sparse router always runs rules and graph analysis and calls
   the tabular specialist when uncertainty, missing evidence or disagreement
   makes a second view useful.
6. LangGraph owns workflow state, bounded investigation, pause/resume and the
   audit trace. It does not make the risk policy itself.
7. The optional investigator LLM writes an evidence-bound explanation. A
   grounding validator rejects citations that are not present in the case.
8. A human reviewer can verify the relationship, request information, approve
   onboarding or hold the case. Only recorded human approval can reach the
   simulated gateway.

## Architecture and ownership

See the presentation-ready diagram at
[`merchantshield-architecture.svg`](merchantshield-architecture.svg).

| Layer | Responsibility | Code |
|---|---|---|
| Input | Synthetic application fields or a transient CSV batch | `merchantshield/demo/live_check.py`, `merchantshield/ui/streamlit_app.py` |
| API | Validation and safe case/evaluation endpoints | `merchantshield/api/` |
| Graph | Shared-attribute edges, components, pair strength and time context | `merchantshield/analysis/evidence_graph.py` |
| Rules | Numerical consistency and identity signals | `merchantshield/agent/experts.py`, `merchantshield/analysis/` |
| ML experts | Graph, tabular and hybrid fitted baselines | `merchantshield/evaluation/ring_models.py`, `merchantshield/evaluation/performance.py` |
| Router/policy | Expert routing, score ownership and permitted actions | `merchantshield/agent/router.py` |
| Orchestration | State, bounded loops, interruptions and resume | `merchantshield/workflow/` |
| Investigation | Evidence-bound explanation and abstention | `merchantshield/investigation/` |
| Review | Versioned case lifecycle and append-only events | `merchantshield/review/`, `merchantshield/db/` |
| UI | Review queue, transient batch check and validation view | `merchantshield/ui/streamlit_app.py` |

The system is a system-level sparse mixture of experts. It is not a newly
trained neural MoE and it is not multiple chatbots debating one another.

## Primary metrics to present

The recommended headline is the fresh validation-tuned report in
`data/evaluation/performance_report.json`. Models and thresholds were selected
on a separate validation partition, then evaluated once on an untouched test
partition. The split is by connected component/ring group.

**Population:** 2,417 train / 813 validation / 841 final-test applications;
49 fraud rings in the final test. False-positive cost is a configurable review
unit, default 1; it is not rupee-denominated.

| System | Precision | Recall | PR-AUC | FPR | Ring recall | Review rate | FP count | FP cost |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| Rules only | 0.504 | 0.598 | 0.485 | 0.336 | 0.939 | 0.577 | 180 | 265 |
| Clustering only | 0.462 | 0.964 | 0.549 | 0.641 | 1.000 | 0.845 | 343 | 405 |
| Graph only | 0.464 | 0.820 | 0.566 | 0.542 | 0.918 | 0.756 | 290 | 365 |
| Tabular only | 0.377 | 0.830 | 0.634 | 0.785 | 1.000 | 0.880 | 420 | 461 |
| Graph ML | 0.539 | 0.794 | 0.743 | 0.389 | 0.898 | 0.683 | 208 | 307 |
| Hybrid | 0.609 | 0.824 | 0.836 | 0.303 | 0.878 | 0.650 | 162 | 269 |
| Routed MoE | **0.612** | **0.824** | **0.835** | **0.299** | **0.878** | **0.648** | **160** | **267** |

The routed model produced 160 false risk flags in the 841-row test. Its
precision/recall trade-off is deliberate: a higher threshold would reduce
legitimate reviews but could miss more fraud. These are screening metrics, not
confirmed fraud decisions.

## Larger and harder evidence

The expanded benchmark (`data/evaluation/expanded_report.json`) contains 2,002
synthetic applications and 35 held-out fraud rings. Its primary 600-row test
exposes the weakness of the original fixed policy:

| System | Precision | Recall | PR-AUC | FPR | Ring recall | Review rate | FP count | FP cost |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| Rules only | 0.562 | 0.617 | 0.538 | 0.292 | 0.914 | 0.595 | 109 | 198 |
| Clustering only | 0.496 | 0.978 | 0.582 | 0.606 | 1.000 | 0.842 | 226 | 278 |
| Graph only | 0.508 | 0.863 | 0.633 | 0.509 | 0.971 | 0.780 | 190 | 256 |
| Tabular only | 0.377 | 0.947 | 0.658 | 0.952 | 1.000 | 0.968 | 355 | 363 |
| Graph ML | 0.393 | 0.991 | 0.723 | 0.930 | 1.000 | 0.960 | 347 | 349 |
| Hybrid | 0.396 | 0.987 | 0.795 | 0.914 | 1.000 | 0.958 | 341 | 348 |
| Routed MoE | 0.393 | 0.991 | 0.723 | 0.930 | 1.000 | 0.992 | 347 | 368 |

This result should be shown if a judge asks about robustness. It is honest
evidence that more synthetic data alone does not guarantee reliable precision.

The hard-ring recovery experiment (`data/evaluation/relationships/report.json`)
also failed its adoption checks: wider relationship context reached 0.590
precision, 0.885 recall, 0.836 PR-AUC, 0.367 FPR and 0.890 ring recall on 1,192
held-out applications. It remains shadow evidence; it did not replace the
review model.

For completeness, its full baseline table is:

| System | Precision | Recall | PR-AUC | FPR | Ring recall | Review rate | FP count | FP cost |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| Rules only | 0.548 | 0.665 | 0.535 | 0.327 | 0.932 | 0.605 | 244 | 383 |
| Clustering only | 0.483 | 0.966 | 0.583 | 0.617 | 1.000 | 0.828 | 461 | 544 |
| Graph only | 0.477 | 0.787 | 0.560 | 0.514 | 0.932 | 0.742 | 384 | 490 |
| Tabular only | 0.410 | 0.863 | 0.696 | 0.740 | 0.986 | 0.900 | 553 | 658 |
| Graph ML | 0.560 | 0.912 | 0.795 | 0.427 | 0.890 | 0.720 | 319 | 437 |
| Hybrid | 0.587 | 0.885 | 0.841 | 0.371 | 0.890 | 0.687 | 277 | 408 |
| Routed MoE | 0.590 | 0.885 | 0.836 | 0.367 | 0.890 | 0.685 | 274 | 405 |

## What the employee sees in the demo

The shortest credible flow is:

1. Open **Review next case** and show the five linked synthetic merchants.
2. Point to the strongest shared signal, coverage, submission window and the
   legitimate alternative explanation.
3. Open the evidence trace only if asked; it shows the actual routed workflow,
   not a staged animation.
4. Choose **Confirm suspicious ring** and then **Confirm and finish**. Explain
   that this records a human hold for enhanced review, not an automatic ban.
5. Open **Check merchants** to upload multiple synthetic CSVs. The batch is
   compared both against the submitted files and the frozen synthetic reference
   population; the input is transient and not persisted.
6. Open **Model validation** to show the component-safe split, ablation table,
   false-positive counts, review rate and configured cost.

For the false-positive story, use **Shared kiosk: a false positive we do not
hide**. Clear it with a verified legitimate explanation and explain why shared
infrastructure is a review signal, not proof of fraud.

## Judge questions — direct answers

**Is it genuinely learning?** Yes, the graph-ML, tabular and hybrid rows fit
models on training partitions. Rules and graph-only rows are intentionally
non-learning controls. The same held-out applications are used for the layer
comparison, so the judge can see what each evidence source adds.

**Why is precision lower on the harder test?** Legitimate shared offices,
kiosks and family businesses overlap with shell-ring signals. The system is
conservative and sends uncertain cases to review. The expanded test is retained
as a failure mode, not hidden.

**How would real onboarding data enter?** A production adapter would map fields
from the KYC/onboarding event stream, tokenize sensitive identifiers, enforce
retention/RBAC and attach independent document-verification results. The demo
CSV is only a synthetic version of that event contract.

**Why LangGraph if the default runs without an LLM?** LangGraph coordinates
state, bounded investigation, human interruption and resume. The workflow must
still be deterministic and auditable when the optional narrator is disabled.

**How much is rules versus trained ML?** The ablation table answers this:
rules-only is the deterministic floor, graph/tabular rows isolate each fitted
view, hybrid combines them, and routed MoE chooses the specialist only when
uncertainty or disagreement warrants it.

**Is the demo fixed?** The scenarios are reproducible fixtures, but the path is
real: API calls, graph construction, routing, workflow trace, persistence and
human resolution all execute. A reviewer can also upload a new synthetic batch.

## Explicit limits

- Synthetic identities and documents only; no real KYC authentication.
- No production identity, access-control, retention or deployment hardening.
- No autonomous blacklist or permanent rejection.
- Streamlit Cloud storage and the demo SQLite database are ephemeral.
- The live reviewer desk uses the frozen demonstration model; the newer
  performance model is reported as read-only validation/shadow evidence.
- Shared infrastructure can be legitimate, and a new ring's first merchant may
  not yet have a cross-application link.

## Reproducibility and safety

```bash
./run.sh
.venv/bin/python -m pytest -q
.venv/bin/python scripts/acceptance.py
```

The checked-in benchmark is reproducible from its seed. The evaluation split
keeps ring IDs and connected components together. The full test suite currently
passes 246 tests, and the acceptance harness reports 12/12 checks passed.
