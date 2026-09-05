"""Risk assessment models — signals, clusters, and the final gate decision."""
from __future__ import annotations

from datetime import datetime, timezone
from enum import Enum
from typing import List, Optional

from pydantic import BaseModel, Field


class Decision(str, Enum):
    APPROVED = "approved"
    UNDER_REVIEW = "under_review"
    REJECTED = "rejected"


class RiskSignal(BaseModel):
    """One piece of evidence produced by an analysis module.

    `value` is the severity in [0,1]; `weight` lets a module say "this one
    matters more than my other signals" during fusion.
    """

    name: str
    value: float = Field(..., ge=0.0, le=1.0)
    weight: float = Field(1.0, gt=0.0)
    explanation: str
    source: str  # document_intelligence | synthetic_identity | application_clustering


class ClusterInfo(BaseModel):
    """Fraud-ring linkage for one application."""

    cluster_id: Optional[str] = None
    cluster_size: int = 1  # includes the application under assessment
    shared_attributes: List[str] = Field(default_factory=list)
    related_application_ids: List[str] = Field(default_factory=list)


class RiskAssessment(BaseModel):
    """Complete, auditable assessment for a single merchant application."""

    application_id: str

    # Per-module scores (0 = clean, 1 = certainly fraudulent)
    document_risk_score: float = Field(0.0, ge=0.0, le=1.0)
    synthetic_identity_score: float = Field(0.0, ge=0.0, le=1.0)
    clustering_risk_score: float = Field(0.0, ge=0.0, le=1.0)

    risk_signals: List[RiskSignal] = Field(default_factory=list)
    cluster_info: Optional[ClusterInfo] = None

    final_risk_score: float = Field(0.0, ge=0.0, le=1.0)

    decision: Decision = Decision.UNDER_REVIEW
    decision_reason: str = ""
    human_readable_explanation: str = ""

    # Thresholds in force when this decision was made (audit trail)
    auto_approve_threshold: float = 0.25
    auto_reject_threshold: float = 0.75

    # "full" when LLM analysis ran, "heuristic_only" otherwise
    analysis_mode: str = "full"

    assessed_at: datetime = Field(
        default_factory=lambda: datetime.now(timezone.utc)
    )

    @property
    def top_signals(self) -> List[RiskSignal]:
        return sorted(self.risk_signals, key=lambda s: s.value, reverse=True)[:5]
