"""Serializable LangGraph state for one merchant-ring case.

State holds IDs, typed expert output and the investigation record. It never
holds API keys, raw uploaded documents, unredacted identifiers, or the
application objects themselves -- those are resolved from a repository outside
the checkpoint, so a checkpoint is safe to persist.
"""
from __future__ import annotations

import hashlib
from enum import Enum
from typing import Mapping, Optional, Tuple

from pydantic import BaseModel, Field

from ..agent.contracts import CandidateAction, CandidateAssessment
from ..investigation.contracts import InvestigationResult, Recommendation

WORKFLOW_VERSION = "ring-workflow-v1"


class ReviewStatus(str, Enum):
    CLEARED = "cleared"
    PENDING_HUMAN_REVIEW = "pending_human_review"
    AWAITING_MERCHANT_INFORMATION = "awaiting_merchant_information"


class WorkflowEvent(BaseModel):
    """One node transition, kept for the reviewer's audit trail."""

    node: str = Field(..., min_length=2, max_length=80)
    detail: str = Field(..., min_length=2, max_length=400)


class SuppliedInformation(BaseModel):
    """Follow-up evidence a merchant or reviewer provided after an interrupt."""

    item_code: str = Field(..., min_length=2, max_length=60)
    content: str = Field(..., min_length=1, max_length=1000)
    supplied_by: str = Field("reviewer", min_length=2, max_length=80)


class CaseDecision(BaseModel):
    """The deterministic outcome. The only component allowed to name an action.

    `investigator_recommendation` is recorded for the reviewer but is advisory:
    it is never the source of `action`.
    """

    case_id: str = Field(..., min_length=2, max_length=100)
    candidate_id: str = Field(..., min_length=2, max_length=100)
    action: CandidateAction
    review_status: ReviewStatus
    reason_codes: Tuple[str, ...]
    primary_risk_score: Optional[float] = Field(None, ge=0.0, le=1.0)
    investigator_recommendation: Optional[Recommendation] = None
    policy_version: str = Field(..., min_length=2, max_length=80)
    workflow_version: str = WORKFLOW_VERSION


class CaseState(BaseModel):
    """The full checkpointable case."""

    case_id: str = Field(..., min_length=2, max_length=100)
    candidate_id: str = Field(..., min_length=2, max_length=100)
    assessment: Optional[CandidateAssessment] = None
    investigation: Optional[InvestigationResult] = None
    supplied_information: Tuple[SuppliedInformation, ...] = ()
    information_request_rounds: int = 0
    decision: Optional[CaseDecision] = None
    workflow_trace: Tuple[WorkflowEvent, ...] = ()
    workflow_version: str = WORKFLOW_VERSION

    def investigator_context(self) -> Tuple[Mapping[str, str], ...]:
        """Supplied evidence in the shape the investigator's brief expects."""
        return tuple(
            {
                "item_code": item.item_code,
                "content": item.content,
                "supplied_by": item.supplied_by,
            }
            for item in self.supplied_information
        )


def case_id_for(candidate_id: str) -> str:
    """Stable case identity, so a resumed case is provably the same case."""
    digest = hashlib.sha256(candidate_id.encode("utf-8")).hexdigest()[:12]
    return f"case-{digest}"
