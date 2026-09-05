"""Typed contracts for case persistence and human review.

The separation this module enforces is the point of Phase 4: an *automated
decision* is produced by the deterministic policy in `workflow.graph` and is
immutable; a *human decision* is produced by a named reviewer and is the only
thing that can approve onboarding. Neither one overwrites the other -- both are
kept, side by side, on the same case.
"""
from __future__ import annotations

from datetime import datetime
from enum import Enum
from typing import Any, Mapping, Optional, Tuple

from pydantic import BaseModel, Field, model_validator

from ..agent.contracts import CandidateAction
from ..investigation.contracts import GroundingStatus, Recommendation
from ..investigation.memory import PriorOutcome
from ..workflow.state import ReviewStatus


class CaseStatus(str, Enum):
    """Where a case sits in the review lifecycle."""

    CLEARED = "cleared"
    PENDING_REVIEW = "pending_review"
    CLAIMED = "claimed"
    AWAITING_INFORMATION = "awaiting_information"
    RESOLVED = "resolved"

    @property
    def is_terminal(self) -> bool:
        return self in (CaseStatus.CLEARED, CaseStatus.RESOLVED)


class HumanDecision(str, Enum):
    """The final actions a human reviewer may take. Only a human may approve."""

    APPROVE_ONBOARDING = "approve_onboarding"
    KEEP_ON_HOLD = "keep_on_hold"
    ESCALATE = "escalate"


class ReviewReasonCode(str, Enum):
    """Controlled reason vocabulary, so resolutions are analysable, not prose."""

    # Reasons that support keeping a case held or escalating it.
    SHARED_SETTLEMENT_ACCOUNT = "shared_settlement_account"
    SHARED_OWNER_IDENTIFIER = "shared_owner_identifier"
    SHARED_DEVICE_OR_NETWORK = "shared_device_or_network"
    SHARED_REGISTERED_ADDRESS = "shared_registered_address"
    COORDINATED_SUBMISSION_TIMING = "coordinated_submission_timing"
    APPLICATION_FIELDS_INCONSISTENT = "application_fields_inconsistent"
    MERCHANT_UNRESPONSIVE = "merchant_unresponsive"
    INSUFFICIENT_EVIDENCE = "insufficient_evidence"

    # Reasons that can clear a case. An approval requires at least one of these.
    VERIFIED_FRANCHISE_AGREEMENT = "verified_franchise_agreement"
    VERIFIED_ACCOUNTANT_RELATIONSHIP = "verified_accountant_relationship"
    VERIFIED_COWORKING_TENANCY = "verified_coworking_tenancy"
    VERIFIED_FAMILY_BUSINESS = "verified_family_business"
    LINKS_EXPLAINED_BY_SUPPLIED_EVIDENCE = "links_explained_by_supplied_evidence"


CLEARING_REASON_CODES: frozenset[ReviewReasonCode] = frozenset(
    {
        ReviewReasonCode.VERIFIED_FRANCHISE_AGREEMENT,
        ReviewReasonCode.VERIFIED_ACCOUNTANT_RELATIONSHIP,
        ReviewReasonCode.VERIFIED_COWORKING_TENANCY,
        ReviewReasonCode.VERIFIED_FAMILY_BUSINESS,
        ReviewReasonCode.LINKS_EXPLAINED_BY_SUPPLIED_EVIDENCE,
    }
)


class CaseEventType(str, Enum):
    """Every append-only event kind. The log is the audit trail."""

    CASE_OPENED = "case_opened"
    AUTOMATED_DECISION_RECORDED = "automated_decision_recorded"
    INVESTIGATION_RECORDED = "investigation_recorded"
    INFORMATION_REQUESTED = "information_requested"
    INFORMATION_SUPPLIED = "information_supplied"
    CASE_CLAIMED = "case_claimed"
    CASE_RELEASED = "case_released"
    HUMAN_DECISION_RECORDED = "human_decision_recorded"
    ONBOARDING_HANDOFF_RECORDED = "onboarding_handoff_recorded"


class ActorKind(str, Enum):
    SYSTEM = "system"
    HUMAN = "human"


SYSTEM_ACTOR = "merchantshield_workflow"


class CaseEvent(BaseModel):
    """One immutable entry in a case's audit log."""

    case_id: str = Field(..., min_length=2, max_length=100)
    sequence: int = Field(..., ge=1)
    event_type: CaseEventType
    actor: str = Field(..., min_length=2, max_length=80)
    actor_kind: ActorKind
    payload: Mapping[str, Any] = Field(default_factory=dict)
    recorded_at: datetime


class AutomatedDecisionView(BaseModel):
    """The deterministic outcome, exactly as the workflow produced it."""

    action: CandidateAction
    review_status: ReviewStatus
    reason_codes: Tuple[str, ...] = ()
    primary_risk_score: Optional[float] = Field(None, ge=0.0, le=1.0)
    investigator_recommendation: Optional[Recommendation] = None
    grounding_status: Optional[GroundingStatus] = None
    policy_version: Optional[str] = None
    workflow_version: Optional[str] = None


class HumanDecisionView(BaseModel):
    """A named reviewer's resolution. Absent until somebody actually resolves."""

    reviewer_id: str = Field(..., min_length=2, max_length=80)
    decision: HumanDecision
    outcome: PriorOutcome
    reason_codes: Tuple[ReviewReasonCode, ...] = Field(..., min_length=1)
    notes: str = Field("", max_length=2000)
    resolved_at: datetime

    @model_validator(mode="after")
    def validate_decision(self) -> "HumanDecisionView":
        return _validate_resolution(self.decision, self.outcome, self.reason_codes, self)


class ResolutionRequest(BaseModel):
    """What a reviewer must supply to close a case."""

    reviewer_id: str = Field(..., min_length=2, max_length=80)
    decision: HumanDecision
    outcome: PriorOutcome
    reason_codes: Tuple[ReviewReasonCode, ...] = Field(..., min_length=1)
    notes: str = Field("", max_length=2000)

    @model_validator(mode="after")
    def validate_request(self) -> "ResolutionRequest":
        return _validate_resolution(self.decision, self.outcome, self.reason_codes, self)


def _validate_resolution(
    decision: HumanDecision,
    outcome: PriorOutcome,
    reason_codes: Tuple[ReviewReasonCode, ...],
    obj: Any,
) -> Any:
    """An approval must state which verified fact cleared the case.

    Without this, 'approved, inconclusive, no clearing reason' would be a
    legal resolution -- and the prior-outcome memory would learn nothing.
    """
    if decision is HumanDecision.APPROVE_ONBOARDING:
        if outcome is PriorOutcome.CONFIRMED_RING:
            raise ValueError("a confirmed ring may not be approved for onboarding")
        if not any(code in CLEARING_REASON_CODES for code in reason_codes):
            raise ValueError(
                "approve_onboarding requires at least one verified clearing reason code"
            )
    if decision is HumanDecision.KEEP_ON_HOLD and any(
        code in CLEARING_REASON_CODES for code in reason_codes
    ):
        raise ValueError("keep_on_hold may not cite a clearing reason code")
    return obj


class HandoffView(BaseModel):
    """Where an approved group was handed off, and under which labelled mode."""

    reference: str = Field(..., min_length=4, max_length=120)
    mode: str = Field(..., min_length=4, max_length=32)
    accepted_at: datetime


class RequestedInformationView(BaseModel):
    item_code: str = Field(..., min_length=2, max_length=60)
    description: str = Field(..., min_length=4, max_length=400)
    member_id: Optional[str] = Field(None, max_length=100)


class CaseSummary(BaseModel):
    """Queue row. Enough to triage without loading the whole case."""

    case_id: str
    candidate_id: str
    status: CaseStatus
    version: int = Field(..., ge=1)
    automated: AutomatedDecisionView
    claimed_by: Optional[str] = None
    claimed_at: Optional[datetime] = None
    resolved_by: Optional[str] = None
    resolved_at: Optional[datetime] = None
    member_count: int = Field(0, ge=0)
    created_at: datetime
    updated_at: datetime


class CaseView(BaseModel):
    """The full durable case record: both decisions, state and audit log."""

    case_id: str
    candidate_id: str
    status: CaseStatus
    version: int = Field(..., ge=1)
    idempotency_key: str
    automated: AutomatedDecisionView
    human: Optional[HumanDecisionView] = None
    requested_information: Tuple[RequestedInformationView, ...] = ()
    claimed_by: Optional[str] = None
    claimed_at: Optional[datetime] = None
    handoff: Optional[HandoffView] = None
    case_state: Mapping[str, Any] = Field(default_factory=dict)
    events: Tuple[CaseEvent, ...] = ()
    created_at: datetime
    updated_at: datetime

    @property
    def is_awaiting_information(self) -> bool:
        return self.status is CaseStatus.AWAITING_INFORMATION
