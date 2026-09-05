"""Stable contracts shared by experts, routing, policy and the future UI."""
from __future__ import annotations

from enum import Enum
from typing import Optional, Tuple

from pydantic import BaseModel, Field, model_validator


class ExpertStatus(str, Enum):
    COMPLETED = "completed"
    ABSTAINED = "abstained"
    FAILED = "failed"


class ExpertCost(str, Enum):
    CHEAP = "cheap"
    MEDIUM = "medium"
    EXPENSIVE = "expensive"


class CandidateAction(str, Enum):
    """The only actions permitted before a human review decision."""

    PROCEED_TO_ONBOARDING = "proceed_to_onboarding"
    HUMAN_REVIEW = "human_review"


class ExpertEvidence(BaseModel):
    """One redacted, addressable fact that an investigator may cite."""

    evidence_id: str = Field(..., min_length=4, max_length=100)
    evidence_type: str = Field(..., min_length=2, max_length=80)
    summary: str = Field(..., min_length=2, max_length=500)
    source: str = Field(..., min_length=2, max_length=80)
    severity: float = Field(..., ge=0.0, le=1.0)


class ExecutionMetadata(BaseModel):
    implementation_version: str = Field(..., min_length=1, max_length=80)
    route_reason: str = Field(..., min_length=1, max_length=300)
    duration_ms: float = Field(..., ge=0.0)
    cost_class: ExpertCost


class ExpertResult(BaseModel):
    """One expert's output, including explicit abstention and failure states."""

    expert_name: str = Field(..., min_length=2, max_length=80)
    candidate_id: str = Field(..., min_length=2, max_length=100)
    status: ExpertStatus
    risk_score: Optional[float] = Field(None, ge=0.0, le=1.0)
    confidence: Optional[float] = Field(None, ge=0.0, le=1.0)
    evidence: Tuple[ExpertEvidence, ...] = ()
    rationale: str = Field(..., min_length=1, max_length=1000)
    error_code: Optional[str] = Field(None, max_length=80)
    metadata: ExecutionMetadata

    @model_validator(mode="after")
    def validate_state(self) -> "ExpertResult":
        if self.status is ExpertStatus.COMPLETED:
            if self.risk_score is None or self.confidence is None:
                raise ValueError("completed experts require risk_score and confidence")
            if self.error_code is not None:
                raise ValueError("completed experts may not carry an error_code")
        else:
            if self.risk_score is not None:
                raise ValueError("failed or abstained experts may not imply a risk score")
            if self.status is ExpertStatus.FAILED and not self.error_code:
                raise ValueError("failed experts require an error_code")
        return self


class RouteEvent(BaseModel):
    stage: str = Field(..., min_length=1, max_length=80)
    selected_experts: Tuple[str, ...]
    reason: str = Field(..., min_length=1, max_length=500)


class CandidateAssessment(BaseModel):
    """Auditable routed-MoE outcome; it deliberately has no reject action."""

    candidate_id: str
    primary_risk_score: Optional[float] = Field(None, ge=0.0, le=1.0)
    action: CandidateAction
    reason_codes: Tuple[str, ...]
    investigator_required: bool
    expert_results: Tuple[ExpertResult, ...]
    route_trace: Tuple[RouteEvent, ...]
    policy_version: str

    @model_validator(mode="after")
    def validate_safe_action(self) -> "CandidateAssessment":
        if self.primary_risk_score is None and self.action is not CandidateAction.HUMAN_REVIEW:
            raise ValueError("missing primary risk must route to human review")
        if self.action is CandidateAction.HUMAN_REVIEW and not self.reason_codes:
            raise ValueError("human review requires at least one reason code")
        return self
