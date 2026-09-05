"""LangGraph state machine: assess, selectively investigate, pause, decide.

Phase 2's `RoutedMoESystem` is wrapped as a single node. Its rules, graph score,
optional-tabular routing and policy are not reimplemented here.

The graph's contract is narrow and deliberate: an LLM recommendation can raise a
case to human review but can never lower one, and it can never produce or alter
the numerical risk score.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Mapping, Optional, Protocol, Tuple

from langgraph.checkpoint.memory import MemorySaver
from langgraph.checkpoint.serde.jsonplus import JsonPlusSerializer
from langgraph.graph import END, START, StateGraph
from langgraph.types import Command, interrupt

from ..agent.contracts import (
    CandidateAction,
    CandidateAssessment,
    ExecutionMetadata,
    ExpertCost,
    ExpertEvidence,
    ExpertResult,
    ExpertStatus,
    RouteEvent,
)
from ..agent.experts import CandidateContext
from ..agent.router import RoutedMoESystem
from ..analysis.evidence_graph import RingCandidate
from ..investigation.contracts import (
    GroundingReport,
    GroundingStatus,
    Hypothesis,
    HypothesisKind,
    InformationRequestItem,
    InvestigationResult,
    InvestigationStatus,
    InvestigatorOutput,
    Recommendation,
    ToolCallStatus,
    ToolObservation,
)
from ..investigation.investigator import BoundedInvestigator
from ..investigation.memory import CaseMemory
from ..models.merchant import MerchantApplication
from .state import (
    WORKFLOW_VERSION,
    CaseDecision,
    CaseState,
    ReviewStatus,
    SuppliedInformation,
    WorkflowEvent,
    case_id_for,
)


# Every type that legitimately crosses a checkpoint boundary. The allow-list is
# explicit so a future serializer change cannot silently start accepting
# arbitrary pickled objects out of stored case state.
CHECKPOINT_TYPES = (
    CandidateAction,
    CandidateAssessment,
    ExecutionMetadata,
    ExpertCost,
    ExpertEvidence,
    ExpertResult,
    ExpertStatus,
    RouteEvent,
    GroundingReport,
    GroundingStatus,
    Hypothesis,
    HypothesisKind,
    InformationRequestItem,
    InvestigationResult,
    InvestigationStatus,
    InvestigatorOutput,
    Recommendation,
    ToolCallStatus,
    ToolObservation,
    CaseDecision,
    CaseState,
    ReviewStatus,
    SuppliedInformation,
    WorkflowEvent,
)


def build_checkpointer() -> MemorySaver:
    """In-memory checkpointer restricted to MerchantShield's own case types."""
    return MemorySaver(
        serde=JsonPlusSerializer(allowed_msgpack_modules=CHECKPOINT_TYPES)
    )


class CaseRepository(Protocol):
    """Resolves the heavy, non-checkpointed objects a case refers to by ID."""

    def get_candidate(self, candidate_id: str) -> RingCandidate: ...

    def get_applications(
        self, candidate_id: str
    ) -> Mapping[str, MerchantApplication]: ...


class InMemoryCaseRepository:
    """Repository backed by one already-built evidence graph."""

    def __init__(
        self,
        candidates: Tuple[RingCandidate, ...],
        applications: Mapping[str, MerchantApplication],
    ) -> None:
        self._candidates = {item.candidate_id: item for item in candidates}
        self._applications = dict(applications)

    def get_candidate(self, candidate_id: str) -> RingCandidate:
        try:
            return self._candidates[candidate_id]
        except KeyError:
            raise KeyError(f"unknown candidate: {candidate_id}") from None

    def get_applications(
        self, candidate_id: str
    ) -> Mapping[str, MerchantApplication]:
        candidate = self.get_candidate(candidate_id)
        return {
            member_id: self._applications[member_id]
            for member_id in candidate.member_ids
        }


@dataclass(frozen=True)
class WorkflowConfig:
    max_information_requests: int = 1
    policy_version: str = "ring-policy-v1"

    def __post_init__(self) -> None:
        if self.max_information_requests < 0:
            raise ValueError("max_information_requests must not be negative")


@dataclass(frozen=True)
class WorkflowRun:
    """Result of one graph execution: either a decision or a pending request."""

    case_id: str
    state: CaseState
    interrupt_payload: Optional[Mapping[str, object]] = None

    @property
    def is_awaiting_information(self) -> bool:
        return self.interrupt_payload is not None

    @property
    def decision(self) -> Optional[CaseDecision]:
        return self.state.decision


class MerchantShieldWorkflow:
    """Compiled LangGraph app plus the run/resume surface Phase 4 will call."""

    def __init__(
        self,
        *,
        repository: CaseRepository,
        router: RoutedMoESystem,
        investigator: BoundedInvestigator,
        memory: Optional[CaseMemory] = None,
        config: WorkflowConfig = WorkflowConfig(),
        checkpointer: Optional[MemorySaver] = None,
    ) -> None:
        self.repository = repository
        self.router = router
        self.investigator = investigator
        self.memory = memory or CaseMemory()
        self.config = config
        self.checkpointer = checkpointer or build_checkpointer()
        self.app = self._build()

    # -- graph -----------------------------------------------------------

    def _build(self):
        graph = StateGraph(CaseState)
        graph.add_node("assess", self._assess)
        graph.add_node("investigate", self._investigate)
        graph.add_node("await_information", self._await_information)
        graph.add_node("finalize", self._finalize)

        graph.add_edge(START, "assess")
        graph.add_conditional_edges(
            "assess",
            self._route_after_assessment,
            {"investigate": "investigate", "finalize": "finalize"},
        )
        graph.add_conditional_edges(
            "investigate",
            self._route_after_investigation,
            {"await_information": "await_information", "finalize": "finalize"},
        )
        graph.add_edge("await_information", "investigate")
        graph.add_edge("finalize", END)
        return graph.compile(checkpointer=self.checkpointer)

    # -- nodes -----------------------------------------------------------

    async def _assess(self, state: CaseState) -> Dict[str, object]:
        candidate = self.repository.get_candidate(state.candidate_id)
        applications = self.repository.get_applications(state.candidate_id)
        assessment = await self.router.assess(
            CandidateContext(candidate=candidate, applications=applications)
        )
        return {
            "assessment": assessment,
            "workflow_trace": state.workflow_trace
            + (
                WorkflowEvent(
                    node="assess",
                    detail=(
                        f"Routed-MoE assessment complete: action="
                        f"{assessment.action.value}, experts="
                        f"{len(assessment.expert_results)}."
                    ),
                ),
            ),
        }

    async def _investigate(self, state: CaseState) -> Dict[str, object]:
        assert state.assessment is not None
        candidate = self.repository.get_candidate(state.candidate_id)
        applications = self.repository.get_applications(state.candidate_id)
        investigation = await self.investigator.investigate(
            candidate=candidate,
            applications=applications,
            assessment=state.assessment,
            memory=self.memory,
            supplied_information=state.investigator_context(),
        )
        return {
            "investigation": investigation,
            "workflow_trace": state.workflow_trace
            + (
                WorkflowEvent(
                    node="investigate",
                    detail=(
                        f"Investigation {investigation.status.value} after "
                        f"{investigation.steps_used} step(s) and "
                        f"{investigation.tool_calls_used} tool call(s); "
                        f"grounding={investigation.grounding.status.value}."
                    ),
                ),
            ),
        }

    async def _await_information(self, state: CaseState) -> Dict[str, object]:
        """Checkpoint and pause. The same case resumes; it never restarts."""
        assert state.investigation is not None and state.investigation.output is not None
        requested = state.investigation.output.requested_information
        supplied = interrupt(
            {
                "case_id": state.case_id,
                "candidate_id": state.candidate_id,
                "reason": "investigator_requested_information",
                "requested_information": [
                    item.model_dump(mode="json") for item in requested
                ],
            }
        )
        return {
            "supplied_information": state.supplied_information
            + _coerce_supplied(supplied),
            "information_request_rounds": state.information_request_rounds + 1,
            "workflow_trace": state.workflow_trace
            + (
                WorkflowEvent(
                    node="await_information",
                    detail=(
                        f"Case paused for {len(requested)} requested item(s) and "
                        "resumed with supplied evidence."
                    ),
                ),
            ),
        }

    async def _finalize(self, state: CaseState) -> Dict[str, object]:
        assert state.assessment is not None
        decision = self._apply_policy(state)
        return {
            "decision": decision,
            "workflow_trace": state.workflow_trace
            + (
                WorkflowEvent(
                    node="finalize",
                    detail=(
                        f"Deterministic policy: action={decision.action.value}, "
                        f"review_status={decision.review_status.value}."
                    ),
                ),
            ),
        }

    # -- edges -----------------------------------------------------------

    @staticmethod
    def _route_after_assessment(state: CaseState) -> str:
        assessment = state.assessment
        if assessment is not None and assessment.investigator_required:
            return "investigate"
        return "finalize"

    def _route_after_investigation(self, state: CaseState) -> str:
        investigation = state.investigation
        if (
            investigation is not None
            and investigation.status is InvestigationStatus.AWAITING_INFORMATION
            and state.information_request_rounds < self.config.max_information_requests
        ):
            return "await_information"
        return "finalize"

    # -- deterministic policy --------------------------------------------

    def _apply_policy(self, state: CaseState) -> CaseDecision:
        assessment: CandidateAssessment = state.assessment  # type: ignore[assignment]
        investigation = state.investigation
        action = assessment.action
        reasons: List[str] = list(assessment.reason_codes)
        recommendation: Optional[Recommendation] = None

        if investigation is not None:
            recommendation = investigation.effective_recommendation
            if investigation.grounding.status is GroundingStatus.INSUFFICIENT_GROUNDING:
                reasons.append("INSUFFICIENT_GROUNDING")
            elif investigation.status is InvestigationStatus.FAILED:
                reasons.append("INVESTIGATOR_UNAVAILABLE")
            elif investigation.status is InvestigationStatus.ABSTAINED:
                reasons.append("INVESTIGATOR_ABSTAINED")
            elif investigation.status is InvestigationStatus.AWAITING_INFORMATION:
                reasons.append("INFORMATION_REQUESTED")
            if recommendation is Recommendation.HUMAN_REVIEW:
                reasons.append("INVESTIGATOR_RECOMMENDS_REVIEW")
            # An LLM opinion may only ever raise the outcome. A
            # `continue_onboarding` recommendation on a case Phase 2 already
            # flagged changes nothing.
            if (
                action is CandidateAction.PROCEED_TO_ONBOARDING
                and recommendation
                in (Recommendation.HUMAN_REVIEW, Recommendation.REQUEST_INFORMATION)
            ):
                action = CandidateAction.HUMAN_REVIEW

        if action is CandidateAction.HUMAN_REVIEW and not reasons:
            reasons.append("REVIEW_REQUIRED")

        if action is CandidateAction.PROCEED_TO_ONBOARDING:
            review_status = ReviewStatus.CLEARED
        elif (
            investigation is not None
            and investigation.status is InvestigationStatus.AWAITING_INFORMATION
        ):
            review_status = ReviewStatus.AWAITING_MERCHANT_INFORMATION
        else:
            review_status = ReviewStatus.PENDING_HUMAN_REVIEW

        return CaseDecision(
            case_id=state.case_id,
            candidate_id=state.candidate_id,
            action=action,
            review_status=review_status,
            reason_codes=tuple(dict.fromkeys(reasons)),
            # Always the graph expert's number, never the investigator's.
            primary_risk_score=assessment.primary_risk_score,
            investigator_recommendation=recommendation,
            policy_version=self.config.policy_version,
            workflow_version=WORKFLOW_VERSION,
        )

    # -- public surface --------------------------------------------------

    async def run(
        self, candidate_id: str, *, case_id: Optional[str] = None
    ) -> WorkflowRun:
        resolved = case_id or case_id_for(candidate_id)
        raw = await self.app.ainvoke(
            {"case_id": resolved, "candidate_id": candidate_id},
            config=self._config(resolved),
        )
        return self._to_run(resolved, raw)

    async def resume(
        self, case_id: str, *, supplied_information: object
    ) -> WorkflowRun:
        """Resume the *same* case with follow-up evidence, preserving its trace."""
        raw = await self.app.ainvoke(
            Command(resume=supplied_information), config=self._config(case_id)
        )
        run = self._to_run(case_id, raw)
        if run.state.case_id != case_id:
            raise RuntimeError("resume must not start a new case")
        return run

    async def get_state(self, case_id: str) -> CaseState:
        snapshot = await self.app.aget_state(self._config(case_id))
        return CaseState.model_validate(snapshot.values)

    @staticmethod
    def _config(case_id: str) -> Dict[str, object]:
        return {"configurable": {"thread_id": case_id}}

    @staticmethod
    def _to_run(case_id: str, raw: Mapping[str, object]) -> WorkflowRun:
        interrupts = raw.get("__interrupt__") or ()
        payload = None
        if interrupts:
            value = getattr(interrupts[0], "value", None)
            payload = value if isinstance(value, Mapping) else {"value": value}
        state = CaseState.model_validate(
            {key: value for key, value in raw.items() if key != "__interrupt__"}
        )
        return WorkflowRun(case_id=case_id, state=state, interrupt_payload=payload)


def _coerce_supplied(supplied: object) -> Tuple[SuppliedInformation, ...]:
    """Accept a single item, a list, or plain text from the resuming caller."""
    if supplied is None:
        return ()
    if isinstance(supplied, SuppliedInformation):
        return (supplied,)
    if isinstance(supplied, Mapping):
        return (SuppliedInformation.model_validate(supplied),)
    if isinstance(supplied, str):
        return (
            SuppliedInformation(item_code="free_text_response", content=supplied),
        )
    if isinstance(supplied, (list, tuple)):
        items: List[SuppliedInformation] = []
        for entry in supplied:
            items.extend(_coerce_supplied(entry))
        return tuple(items)
    raise TypeError(f"unsupported supplied information type: {type(supplied).__name__}")
