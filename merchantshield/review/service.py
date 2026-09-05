"""The Phase 4 review service: run a case, persist it, hand it to a human.

This is the only place where the LangGraph workflow and the durable store meet.
It preserves the Phase 3 invariants: the automated decision is written exactly
as the deterministic policy produced it, and nothing a human does can rewrite
it -- a reviewer's resolution is stored beside it, not over it.
"""
from __future__ import annotations

from datetime import datetime, timezone
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple

from ..analysis.evidence_graph import RingCandidate
from ..gateway import (
    GatewayMode,
    HandoffRequest,
    OnboardingGateway,
    SimulatedOnboardingGateway,
)
from ..investigation.memory import CaseMemory, PriorCaseRecord
from ..workflow.graph import MerchantShieldWorkflow, WorkflowRun
from ..workflow.state import CaseState, ReviewStatus, case_id_for
from .contracts import (
    SYSTEM_ACTOR,
    ActorKind,
    AutomatedDecisionView,
    CaseEventType,
    CaseStatus,
    CaseSummary,
    CaseView,
    HumanDecision,
    HumanDecisionView,
    RequestedInformationView,
    ResolutionRequest,
)
from .store import CaseStore, ConcurrentModification

EventTuple = Tuple[CaseEventType, str, ActorKind, Mapping[str, Any]]


class CaseAlreadyOpen(RuntimeError):
    """A case for this candidate exists under a different idempotency key."""


class NotAwaitingInformation(RuntimeError):
    """Follow-up evidence was supplied for a case that is not paused for it."""


class HandoffNotPermitted(RuntimeError):
    """A hand-off was attempted without a recorded human approval."""


class ReviewService:
    """Open, progress and resolve ring cases against durable storage."""

    def __init__(
        self,
        *,
        workflow: MerchantShieldWorkflow,
        store: CaseStore,
        candidates: Mapping[str, RingCandidate] | None = None,
        gateway: Optional[OnboardingGateway] = None,
    ) -> None:
        self.workflow = workflow
        self.store = store
        self._candidates = dict(candidates or {})
        self.gateway = gateway or SimulatedOnboardingGateway()

    # -- automated side ---------------------------------------------------

    async def open_case(
        self, candidate_id: str, *, idempotency_key: Optional[str] = None
    ) -> CaseView:
        """Assess a candidate and persist the resulting case.

        Idempotent in two directions: a repeated key returns the stored case
        without re-running the workflow, and a candidate that already has a
        case is never assessed twice into two competing records.
        """
        case_id = case_id_for(candidate_id)
        key = idempotency_key or f"auto:{case_id}"

        stored = self.store.find_by_idempotency_key(key)
        if stored is not None:
            return stored
        if self.store.exists(case_id):
            existing = self.store.get(case_id)
            if existing.idempotency_key != key:
                raise CaseAlreadyOpen(
                    f"case {case_id} already exists under key "
                    f"'{existing.idempotency_key}'"
                )
            return existing

        run = await self.workflow.run(candidate_id, case_id=case_id)
        automated, status, requested = _interpret(run)
        events: List[EventTuple] = [
            (
                CaseEventType.CASE_OPENED,
                SYSTEM_ACTOR,
                ActorKind.SYSTEM,
                {"candidate_id": candidate_id, "idempotency_key": key},
            )
        ]
        events.extend(_progress_events(run))
        return self.store.open_case(
            case_id=case_id,
            candidate_id=candidate_id,
            idempotency_key=key,
            status=status,
            automated=automated,
            member_count=self._member_count(candidate_id),
            requested_information=requested,
            case_state=run.state.model_dump(mode="json"),
            events=events,
        )

    async def supply_information(
        self,
        case_id: str,
        *,
        expected_version: int,
        items: Sequence[Mapping[str, Any]],
        supplied_by: str,
    ) -> CaseView:
        """Resume the *same* paused case with follow-up evidence."""
        current = self.store.get(case_id)
        if current.status is not CaseStatus.AWAITING_INFORMATION:
            raise NotAwaitingInformation(
                f"case {case_id} is {current.status.value}, not awaiting information"
            )
        if current.version != expected_version:
            raise ConcurrentModification(case_id, expected_version, current.version)

        payload = [
            {**dict(item), "supplied_by": item.get("supplied_by", supplied_by)}
            for item in items
        ]
        run = await self.workflow.resume(case_id, supplied_information=payload)
        if run.state.case_id != case_id:
            raise RuntimeError("resume must not start a new case")

        automated, status, requested = _interpret(run)
        events: List[EventTuple] = [
            (
                CaseEventType.INFORMATION_SUPPLIED,
                supplied_by,
                ActorKind.HUMAN,
                {"items": payload},
            )
        ]
        events.extend(_progress_events(run))
        return self.store.record_automated_progress(
            case_id=case_id,
            expected_version=expected_version,
            status=status,
            automated=automated,
            requested_information=requested,
            case_state=run.state.model_dump(mode="json"),
            events=events,
        )

    # -- human side -------------------------------------------------------

    def claim(self, case_id: str, reviewer_id: str, expected_version: int) -> CaseView:
        return self.store.claim(case_id, reviewer_id, expected_version)

    def release(self, case_id: str, reviewer_id: str, expected_version: int) -> CaseView:
        return self.store.release(case_id, reviewer_id, expected_version)

    def resolve(
        self,
        case_id: str,
        *,
        request: ResolutionRequest,
        expected_version: int,
    ) -> CaseView:
        """Record the human decision and teach case memory what was resolved."""
        current = self.store.get(case_id)
        human = HumanDecisionView(
            reviewer_id=request.reviewer_id,
            decision=request.decision,
            outcome=request.outcome,
            reason_codes=request.reason_codes,
            notes=request.notes,
            resolved_at=datetime.now(timezone.utc),
        )
        prior = self._prior_records(current.candidate_id, human)
        return self.store.resolve(
            case_id=case_id,
            expected_version=expected_version,
            human=human,
            prior_records=prior,
        )

    def hand_off(self, case_id: str, *, expected_version: int, actor: str) -> CaseView:
        """Submit an approved group to the onboarding gateway.

        The gateway is unreachable from any other path. A case that a human did
        not explicitly approve cannot be handed off, whatever the models said.
        """
        current = self.store.get(case_id)
        if current.human is None or current.human.decision is not HumanDecision.APPROVE_ONBOARDING:
            raise HandoffNotPermitted(
                f"case {case_id} has no recorded human approval; "
                "only an approved case may be handed off"
            )
        if current.handoff is not None:
            return current
        candidate = self._candidates.get(current.candidate_id)
        member_ids = tuple(candidate.member_ids) if candidate else (current.candidate_id,)
        receipt = self.gateway.submit(
            HandoffRequest(
                case_id=case_id,
                candidate_id=current.candidate_id,
                member_ids=member_ids,
                approved_by=current.human.reviewer_id,
                reason_codes=tuple(code.value for code in current.human.reason_codes),
                notes=current.human.notes,
            )
        )
        return self.store.record_handoff(
            case_id=case_id,
            expected_version=expected_version,
            reference=receipt.reference,
            mode=receipt.mode.value,
            accepted_at=receipt.accepted_at,
            actor=actor,
            detail={key: str(value) for key, value in receipt.detail.items()},
        )

    # -- reads ------------------------------------------------------------

    def get(self, case_id: str) -> CaseView:
        return self.store.get(case_id)

    def list_cases(self, **kwargs: Any) -> Tuple[int, Tuple[CaseSummary, ...]]:
        return self.store.list_cases(**kwargs)

    def queue_counts(self) -> Mapping[str, int]:
        return self.store.queue_counts()

    def memory(self) -> CaseMemory:
        """A memory view over everything humans have resolved so far."""
        return CaseMemory(self.store.prior_records())

    # -- internals --------------------------------------------------------

    def _member_count(self, candidate_id: str) -> int:
        candidate = self._candidates.get(candidate_id)
        if candidate is not None:
            return len(candidate.member_ids)
        try:
            return len(self.workflow.repository.get_candidate(candidate_id).member_ids)
        except Exception:
            return 0

    def _prior_records(
        self, candidate_id: str, human: HumanDecisionView
    ) -> Tuple[PriorCaseRecord, ...]:
        """Write one memory record per hashed attribute the case rested on.

        Only hashes are stored. An inconclusive resolution is still recorded --
        'we looked and could not tell' is information a later reviewer needs.
        """
        try:
            candidate = self.workflow.repository.get_candidate(candidate_id)
        except Exception:
            return ()
        seen: Dict[Tuple[str, str], PriorCaseRecord] = {}
        for edge in candidate.evidence:
            key = (edge.shared_value_hash, edge.attribute)
            if key in seen:
                continue
            seen[key] = PriorCaseRecord(
                infrastructure_hash=edge.shared_value_hash,
                attribute=edge.attribute,
                outcome=human.outcome,
                resolved_at=human.resolved_at,
                reviewer_reference=human.reviewer_id,
                note=(human.notes or "")[:400],
            )
        return tuple(seen.values())


# ------------------------------------------------------------------ helpers


def _interpret(
    run: WorkflowRun,
) -> Tuple[AutomatedDecisionView, CaseStatus, Tuple[RequestedInformationView, ...]]:
    """Translate one workflow run into the durable automated record."""
    state: CaseState = run.state
    grounding = (
        state.investigation.grounding.status if state.investigation else None
    )

    if run.is_awaiting_information:
        payload = run.interrupt_payload or {}
        requested = tuple(
            RequestedInformationView.model_validate(item)
            for item in payload.get("requested_information", ())
        )
        assessment = state.assessment
        if assessment is None:
            raise RuntimeError("a paused case must already carry an assessment")
        return (
            AutomatedDecisionView(
                action=assessment.action,
                review_status=ReviewStatus.AWAITING_MERCHANT_INFORMATION,
                reason_codes=tuple(assessment.reason_codes),
                primary_risk_score=assessment.primary_risk_score,
                investigator_recommendation=(
                    state.investigation.effective_recommendation
                    if state.investigation
                    else None
                ),
                grounding_status=grounding,
                policy_version=assessment.policy_version,
                workflow_version=state.workflow_version,
            ),
            CaseStatus.AWAITING_INFORMATION,
            requested,
        )

    decision = state.decision
    if decision is None:
        raise RuntimeError("a finished workflow run must carry a decision")
    status = (
        CaseStatus.CLEARED
        if decision.review_status is ReviewStatus.CLEARED
        else CaseStatus.PENDING_REVIEW
    )
    return (
        AutomatedDecisionView(
            action=decision.action,
            review_status=decision.review_status,
            reason_codes=decision.reason_codes,
            primary_risk_score=decision.primary_risk_score,
            investigator_recommendation=decision.investigator_recommendation,
            grounding_status=grounding,
            policy_version=decision.policy_version,
            workflow_version=decision.workflow_version,
        ),
        status,
        (),
    )


def _progress_events(run: WorkflowRun) -> Tuple[EventTuple, ...]:
    """The system-side audit entries for one workflow run."""
    state = run.state
    events: List[EventTuple] = []
    investigation = state.investigation
    if investigation is not None:
        events.append(
            (
                CaseEventType.INVESTIGATION_RECORDED,
                SYSTEM_ACTOR,
                ActorKind.SYSTEM,
                {
                    "status": investigation.status.value,
                    "grounding": investigation.grounding.status.value,
                    "steps_used": investigation.steps_used,
                    "tool_calls_used": investigation.tool_calls_used,
                    "tools": [item.tool_name for item in investigation.observations],
                    "provider": investigation.provider_name,
                    "recommendation": investigation.effective_recommendation.value,
                    "grounding_error": investigation.grounding.error_code,
                    "error_code": investigation.error_code,
                },
            )
        )
    if run.is_awaiting_information:
        payload = run.interrupt_payload or {}
        events.append(
            (
                CaseEventType.INFORMATION_REQUESTED,
                SYSTEM_ACTOR,
                ActorKind.SYSTEM,
                {"requested_information": list(payload.get("requested_information", ()))},
            )
        )
    elif state.decision is not None:
        events.append(
            (
                CaseEventType.AUTOMATED_DECISION_RECORDED,
                SYSTEM_ACTOR,
                ActorKind.SYSTEM,
                {
                    "action": state.decision.action.value,
                    "review_status": state.decision.review_status.value,
                    "reason_codes": list(state.decision.reason_codes),
                    "primary_risk_score": state.decision.primary_risk_score,
                    "policy_version": state.decision.policy_version,
                    "trace": [event.node for event in state.workflow_trace],
                },
            )
        )
    return tuple(events)
