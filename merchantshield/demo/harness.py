"""Deterministic demonstration harness for two paths the frozen data avoids.

Both runs below are honest about what they pin. They build a *throwaway*
workflow over the same repository as the live runtime, never touch the case
store, and are exposed under `/api/v1/demo/*` so nothing here can be mistaken
for a production decision path.

Why a harness is needed at all: on the frozen synthetic world the fitted graph
model is confident about every group, so neither a fabricated citation nor an
information request occurs by itself. Phase 6 requires both to be demonstrable
with no API key, reproducibly.
"""
from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Any, Dict, Optional, Tuple

from ..agent.experts import GraphScoringExpert, RulesExpert
from ..agent.router import RoutedMoEConfig, RoutedMoESystem
from ..investigation.investigator import (
    BoundedInvestigator,
    DeterministicInvestigator,
    ScriptedProvider,
)
from ..workflow.graph import MerchantShieldWorkflow, WorkflowRun

#: An evidence ID that provably does not exist for any candidate. The demo
#: depends on it never being real, so it is deliberately not hash-shaped.
FABRICATED_EVIDENCE_ID = "edge-fabricated01"

#: Pinned graph risk for the information-request harness. Uncertain enough to
#: require an investigator, weak enough that one question is the right move.
PINNED_UNCERTAIN_RISK = 0.45

SUPPLIED_EXPLANATION = (
    "All four applicants rent desks on the same coworking floor. The building "
    "lease names each business separately and the office network is shared by "
    "every tenant."
)


@dataclass(frozen=True)
class HarnessRun:
    """One harness outcome, plus the disclosure a surface must render."""

    name: str
    runs: Tuple[WorkflowRun, ...]
    pinned_primary_risk_score: Optional[float]
    disclosure: str

    def to_dict(self) -> Dict[str, Any]:
        return {
            "harness": self.name,
            "persisted": False,
            "pinned_primary_risk_score": self.pinned_primary_risk_score,
            "disclosure": self.disclosure,
            "runs": [_run_payload(run) for run in self.runs],
        }


def _workflow(runtime, *, investigator, pinned_risk: Optional[float]) -> MerchantShieldWorkflow:
    """A throwaway workflow sharing the live repository but nothing mutable."""
    if pinned_risk is None:
        router = runtime.workflow.router
    else:
        router = RoutedMoESystem(
            rules_expert=RulesExpert(),
            graph_expert=GraphScoringExpert(lambda _: pinned_risk),
            config=RoutedMoEConfig(),
        )
    return MerchantShieldWorkflow(
        repository=runtime.workflow.repository,
        router=router,
        investigator=investigator,
        memory=runtime.workflow.memory,
    )


def _fabricated_response(candidate_id: str) -> str:
    """A confident, well-formed, entirely ungrounded conclusion."""
    return json.dumps(
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
                    "statement": (
                        "A franchise agreement explains every shared attribute."
                    ),
                    "supporting_evidence_ids": [FABRICATED_EVIDENCE_ID],
                    "contradicting_evidence_ids": [],
                }
            ],
            "missing_evidence": [],
            "requested_information": [],
            "confidence": 0.93,
            "cited_evidence_ids": [FABRICATED_EVIDENCE_ID],
            "narrative": (
                f"Evidence {FABRICATED_EVIDENCE_ID} confirms a registered "
                "franchise agreement covering all members, so onboarding may "
                "continue."
            ),
        }
    )


async def run_grounding_failure(runtime, candidate_id: str) -> HarnessRun:
    """Script a fabricated citation and let deterministic code reject it."""
    provider = ScriptedProvider(
        [
            json.dumps(
                {
                    "action": "call_tool",
                    "tool": "get_candidate_summary",
                    "arguments": {"candidate_id": candidate_id},
                }
            ),
            _fabricated_response(candidate_id),
        ]
    )
    workflow = _workflow(
        runtime, investigator=BoundedInvestigator(provider=provider), pinned_risk=None
    )
    run = await workflow.run(candidate_id, case_id=f"demo-grounding-{candidate_id}")
    return HarnessRun(
        name="grounding_failure",
        runs=(run,),
        pinned_primary_risk_score=None,
        disclosure=(
            "The risk score is the real fitted graph score. Only the model "
            "response is scripted, so the fabricated citation is reproducible "
            "with no API key. This run is not persisted as a case."
        ),
    )


async def run_information_request(runtime, candidate_id: str) -> HarnessRun:
    """Pause on an ambiguous weak-link group, then resume the identical case."""
    workflow = _workflow(
        runtime,
        investigator=BoundedInvestigator(provider=DeterministicInvestigator()),
        pinned_risk=PINNED_UNCERTAIN_RISK,
    )
    case_id = f"demo-information-{candidate_id}"
    paused = await workflow.run(candidate_id, case_id=case_id)
    runs = [paused]
    if paused.is_awaiting_information:
        runs.append(
            await workflow.resume(
                paused.case_id,
                supplied_information={
                    "item_code": "shared_premises_explanation",
                    "content": SUPPLIED_EXPLANATION,
                    "supplied_by": "merchant",
                },
            )
        )
    return HarnessRun(
        name="information_request",
        runs=tuple(runs),
        pinned_primary_risk_score=PINNED_UNCERTAIN_RISK,
        disclosure=(
            f"Graph risk is pinned to {PINNED_UNCERTAIN_RISK} for this run so "
            "the uncertain-links path is reachable on frozen data, where the "
            "fitted model scores this group confidently low. Everything after "
            "the score -- routing, investigation, interrupt and resume -- is "
            "the real workflow. This run is not persisted as a case."
        ),
    )


HARNESSES = {
    "grounding_failure": run_grounding_failure,
    "information_request": run_information_request,
}


def _run_payload(run: WorkflowRun) -> Dict[str, Any]:
    state = run.state
    investigation = state.investigation
    return {
        "case_id": run.case_id,
        "candidate_id": state.candidate_id,
        "awaiting_information": run.is_awaiting_information,
        "interrupt_payload": dict(run.interrupt_payload or {}) or None,
        "primary_risk_score": (
            state.assessment.primary_risk_score if state.assessment else None
        ),
        "expert_results": (
            [item.model_dump(mode="json") for item in state.assessment.expert_results]
            if state.assessment
            else []
        ),
        "route_trace": (
            [item.model_dump(mode="json") for item in state.assessment.route_trace]
            if state.assessment
            else []
        ),
        "investigation": (
            investigation.model_dump(mode="json") if investigation else None
        ),
        "effective_recommendation": (
            investigation.effective_recommendation.value if investigation else None
        ),
        "workflow_trace": [item.model_dump(mode="json") for item in state.workflow_trace],
        "decision": (
            run.decision.model_dump(mode="json") if run.decision is not None else None
        ),
    }
