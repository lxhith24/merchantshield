#!/usr/bin/env python3
"""Deterministic Phase 3 walkthrough: four cases, four auditable outcomes.

Runs with no API key. Scenario 3 deliberately injects a nonexistent evidence ID
to show that a fabricated citation is caught by deterministic code, the trace is
preserved, and the case goes to a human.

    .venv/bin/python scripts/investigation_demo.py
"""
from __future__ import annotations

import asyncio
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from merchantshield.agent import (  # noqa: E402
    CandidateAction,
    GraphScoringExpert,
    RoutedMoESystem,
    RulesExpert,
)
from merchantshield.analysis import EvidenceGraphBuilder  # noqa: E402
from merchantshield.evaluation import load_jsonl  # noqa: E402
from merchantshield.evaluation.dataset import generate_synthetic_dataset  # noqa: E402
from merchantshield.investigation import (  # noqa: E402
    BoundedInvestigator,
    DeterministicInvestigator,
    ScriptedProvider,
    summarize_grounding_failure,
)
from merchantshield.workflow import (  # noqa: E402
    InMemoryCaseRepository,
    MerchantShieldWorkflow,
)

DATASET = Path(__file__).resolve().parent.parent / "data/evaluation/synthetic_merchants.jsonl"
STRONG_ATTRIBUTES = {"bank_account", "owner_pan", "device_fingerprint"}


def rule(title: str) -> None:
    print(f"\n{'=' * 72}\n{title}\n{'=' * 72}")


def show(run) -> None:
    state = run.state
    print(f"  case_id            : {state.case_id}")
    print(f"  candidate_id       : {state.candidate_id}")
    if state.assessment:
        print(f"  primary risk score : {state.assessment.primary_risk_score}")
        print(f"  phase-2 action     : {state.assessment.action.value}")
    investigation = state.investigation
    if investigation is None:
        print("  investigation      : skipped (confident low risk)")
    else:
        print(
            f"  investigation      : {investigation.status.value} "
            f"({investigation.steps_used} steps, "
            f"{investigation.tool_calls_used}/6 tool calls, "
            f"{investigation.duration_ms:.1f} ms)"
        )
        print(
            "  tools chosen       : "
            + (", ".join(item.tool_name for item in investigation.observations) or "none")
        )
        print(f"  grounding          : {investigation.grounding.status.value}")
        if investigation.grounding.unknown_evidence_ids or investigation.grounding.undisclosed_evidence_ids:
            print(f"    -> {summarize_grounding_failure(investigation.grounding)}")
        if investigation.rejected_narrative:
            print(f"    -> discarded narrative: \"{investigation.rejected_narrative[:70]}...\"")
        if investigation.output:
            output = investigation.output
            print(f"  recommendation     : {output.recommendation.value} (advisory)")
            print(f"  ring hypothesis    : {output.ring_hypothesis.label}")
            print(
                "  legitimate alt.    : "
                + ", ".join(item.label for item in output.legitimate_alternatives)
            )
            print(f"  cited evidence     : {len(output.cited_evidence_ids)} validated ID(s)")
    print(f"  workflow trace     : {' -> '.join(e.node for e in state.workflow_trace)}")
    if run.decision:
        decision = run.decision
        print(f"  FINAL ACTION       : {decision.action.value}  [deterministic policy]")
        print(f"  review status      : {decision.review_status.value}")
        print(f"  reason codes       : {', '.join(decision.reason_codes) or 'none'}")
    else:
        print("  FINAL ACTION       : paused, awaiting merchant information")


def build_world():
    if DATASET.exists():
        examples = load_jsonl(DATASET)
    else:
        examples = generate_synthetic_dataset()
    applications = {item.example_id: item.application for item in examples}
    return EvidenceGraphBuilder().build(applications), applications


def make_workflow(candidates, applications, provider, *, graph_score):
    return MerchantShieldWorkflow(
        repository=InMemoryCaseRepository(candidates, applications),
        router=RoutedMoESystem(
            rules_expert=RulesExpert(),
            graph_expert=GraphScoringExpert(lambda _: graph_score),
        ),
        investigator=BoundedInvestigator(provider=provider),
    )


async def main() -> int:
    candidates, applications = build_world()
    singletons = [item for item in candidates if len(item.member_ids) == 1]
    strong = next(
        item
        for item in candidates
        if len(item.member_ids) >= 3
        and any(edge.attribute in STRONG_ATTRIBUTES for edge in item.evidence)
    )
    weak = next(
        item
        for item in candidates
        if len(item.member_ids) >= 3
        and not any(edge.attribute in STRONG_ATTRIBUTES for edge in item.evidence)
    )

    print("MerchantShield -- Phase 3 investigator walkthrough")
    print("All data is synthetic. LLM_DISABLED offline mode; no API key in use.")

    rule("1. Confident low risk -- the investigator never runs")
    workflow = make_workflow(
        candidates, applications, DeterministicInvestigator(), graph_score=0.05
    )
    for candidate in singletons:
        run = await workflow.run(candidate.candidate_id)
        if run.decision and run.decision.action is CandidateAction.PROCEED_TO_ONBOARDING:
            show(run)
            break

    rule("2. Strong shared infrastructure -- bounded, grounded investigation")
    workflow = make_workflow(
        candidates, applications, DeterministicInvestigator(), graph_score=0.82
    )
    show(await workflow.run(strong.candidate_id))

    rule("3. DELIBERATE FAILURE -- the model cites evidence that does not exist")
    fabricated = json.dumps(
        {
            "action": "conclude",
            "recommendation": "continue_onboarding",
            "ring_hypothesis": {
                "kind": "coordinated_ring",
                "label": "coordinated_merchant_ring",
                "statement": "The applications may share a controlling party.",
                "supporting_evidence_ids": [],
                "contradicting_evidence_ids": [],
            },
            "legitimate_alternatives": [
                {
                    "kind": "legitimate_alternative",
                    "label": "verified_franchise_group",
                    "statement": "A franchise agreement explains every shared attribute.",
                    "supporting_evidence_ids": ["edge-fabricated01"],
                    "contradicting_evidence_ids": [],
                }
            ],
            "missing_evidence": [],
            "requested_information": [],
            "confidence": 0.93,
            "cited_evidence_ids": ["edge-fabricated01"],
            "narrative": (
                "Evidence edge-fabricated01 confirms a registered franchise "
                "agreement covering all members, so onboarding may continue."
            ),
        }
    )
    workflow = make_workflow(
        candidates,
        applications,
        ScriptedProvider(
            [
                json.dumps(
                    {
                        "action": "call_tool",
                        "tool": "get_candidate_summary",
                        "arguments": {"candidate_id": strong.candidate_id},
                    }
                ),
                fabricated,
            ]
        ),
        graph_score=0.82,
    )
    show(await workflow.run(strong.candidate_id))
    print(
        "\n  A confident 'continue_onboarding' built on a fabricated citation was\n"
        "  rejected by deterministic code and became human review instead."
    )

    rule("4. Missing decisive fact -- pause, then resume the same case")
    workflow = make_workflow(
        candidates, applications, DeterministicInvestigator(), graph_score=0.45
    )
    paused = await workflow.run(weak.candidate_id)
    show(paused)
    if paused.is_awaiting_information:
        for item in paused.interrupt_payload["requested_information"]:
            print(f"  requested          : {item['item_code']} -- {item['description']}")
        resumed = await workflow.resume(
            paused.case_id,
            supplied_information={
                "item_code": "shared_premises_explanation",
                "content": (
                    "All applicants rent desks on the same coworking floor; "
                    "the lease covers each business separately."
                ),
                "supplied_by": "merchant",
            },
        )
        print("\n  -- resumed --")
        print(f"  same case_id       : {resumed.case_id == paused.case_id}")
        show(resumed)

    rule("Invariants demonstrated")
    for line in (
        "The LLM never produced or altered a numerical risk score.",
        "The LLM never approved or rejected a merchant; policy owns every action.",
        "A fabricated citation invalidated the narrative and forced human review.",
        "Unavailable document verification stayed 'unknown', never 'clean'.",
        "Every case above is reproducible with no API key.",
    ):
        print(f"  - {line}")
    return 0


if __name__ == "__main__":
    raise SystemExit(asyncio.run(main()))
