"""Typed contracts for the bounded, evidence-grounded investigator.

Nothing in this module may carry a risk score. The investigator explains the
evidence Phase 2 already produced; the graph expert remains the only numerical
authority and the deterministic policy remains the only action authority.
"""
from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import Mapping, Optional, Tuple

from pydantic import BaseModel, Field, model_validator


class Recommendation(str, Enum):
    """The only opinions an investigator may express. None of them is an action."""

    CONTINUE_ONBOARDING = "continue_onboarding"
    REQUEST_INFORMATION = "request_information"
    HUMAN_REVIEW = "human_review"
    ABSTAIN = "abstain"


class HypothesisKind(str, Enum):
    COORDINATED_RING = "coordinated_ring"
    LEGITIMATE_ALTERNATIVE = "legitimate_alternative"


class GroundingStatus(str, Enum):
    NOT_EVALUATED = "not_evaluated"
    GROUNDED = "grounded"
    INSUFFICIENT_GROUNDING = "insufficient_grounding"


class InvestigationStatus(str, Enum):
    COMPLETED = "completed"
    ABSTAINED = "abstained"
    AWAITING_INFORMATION = "awaiting_information"
    FAILED = "failed"


class ToolCallStatus(str, Enum):
    OK = "ok"
    ERROR = "error"


@dataclass(frozen=True)
class InvestigationBudget:
    """Operational limits for one investigation. Not a model-quality claim."""

    max_steps: int = 5
    max_tool_calls: int = 6
    max_seconds: float = 15.0
    max_output_tokens: int = 2048

    def __post_init__(self) -> None:
        if self.max_steps < 1:
            raise ValueError("max_steps must be at least 1")
        if self.max_tool_calls < 1:
            raise ValueError("max_tool_calls must be at least 1")
        if self.max_seconds <= 0:
            raise ValueError("max_seconds must be positive")
        if self.max_output_tokens < 256:
            raise ValueError("max_output_tokens must be at least 256")

    def to_dict(self) -> dict:
        return {
            "max_steps": self.max_steps,
            "max_tool_calls": self.max_tool_calls,
            "max_seconds": self.max_seconds,
            "max_output_tokens": self.max_output_tokens,
        }


class Hypothesis(BaseModel):
    """One explanation of the observed links, with the evidence it rests on."""

    kind: HypothesisKind
    label: str = Field(..., min_length=2, max_length=80)
    statement: str = Field(..., min_length=4, max_length=600)
    supporting_evidence_ids: Tuple[str, ...] = ()
    contradicting_evidence_ids: Tuple[str, ...] = ()

    @property
    def referenced_evidence_ids(self) -> Tuple[str, ...]:
        return tuple(
            dict.fromkeys(
                self.supporting_evidence_ids + self.contradicting_evidence_ids
            )
        )


class InformationRequestItem(BaseModel):
    """One decisive fact the merchant or reviewer could supply to resolve a case."""

    item_code: str = Field(..., min_length=2, max_length=60)
    description: str = Field(..., min_length=4, max_length=400)
    member_id: Optional[str] = Field(None, max_length=100)


class InvestigatorOutput(BaseModel):
    """The investigator's structured opinion. Deliberately has no risk score."""

    recommendation: Recommendation
    ring_hypothesis: Hypothesis
    legitimate_alternatives: Tuple[Hypothesis, ...] = Field(..., min_length=1)
    missing_evidence: Tuple[str, ...] = ()
    requested_information: Tuple[InformationRequestItem, ...] = ()
    confidence: float = Field(..., ge=0.0, le=1.0)
    cited_evidence_ids: Tuple[str, ...] = ()
    narrative: str = Field(..., min_length=4, max_length=1500)

    @model_validator(mode="after")
    def validate_output(self) -> "InvestigatorOutput":
        if self.ring_hypothesis.kind is not HypothesisKind.COORDINATED_RING:
            raise ValueError("ring_hypothesis must be a coordinated_ring hypothesis")
        if any(
            item.kind is not HypothesisKind.LEGITIMATE_ALTERNATIVE
            for item in self.legitimate_alternatives
        ):
            raise ValueError("legitimate_alternatives must all be legitimate_alternative")
        if self.recommendation is Recommendation.REQUEST_INFORMATION and not self.requested_information:
            raise ValueError("request_information requires at least one requested item")
        # Every ID a hypothesis leans on must also appear in the citation list, so
        # the grounding validator only ever has to check one place.
        cited = set(self.cited_evidence_ids)
        for hypothesis in (self.ring_hypothesis, *self.legitimate_alternatives):
            missing = [
                item for item in hypothesis.referenced_evidence_ids if item not in cited
            ]
            if missing:
                raise ValueError(
                    f"hypothesis '{hypothesis.label}' references uncited evidence: "
                    f"{sorted(missing)}"
                )
        return self


class ToolObservation(BaseModel):
    """One auditable read-only tool call and what it disclosed."""

    step: int = Field(..., ge=1)
    tool_name: str = Field(..., min_length=2, max_length=80)
    arguments: Mapping[str, str] = Field(default_factory=dict)
    status: ToolCallStatus
    summary: str = Field(..., min_length=2, max_length=800)
    disclosed_evidence_ids: Tuple[str, ...] = ()
    error_code: Optional[str] = Field(None, max_length=80)
    duration_ms: float = Field(..., ge=0.0)

    @model_validator(mode="after")
    def validate_observation(self) -> "ToolObservation":
        if self.status is ToolCallStatus.ERROR and not self.error_code:
            raise ValueError("failed tool calls require an error_code")
        if self.status is ToolCallStatus.ERROR and self.disclosed_evidence_ids:
            raise ValueError("failed tool calls may not disclose evidence")
        return self


class GroundingReport(BaseModel):
    """Deterministic verdict on whether the narrative's citations are real."""

    status: GroundingStatus
    validated_evidence_ids: Tuple[str, ...] = ()
    unknown_evidence_ids: Tuple[str, ...] = ()
    undisclosed_evidence_ids: Tuple[str, ...] = ()
    error_code: Optional[str] = Field(None, max_length=80)

    @model_validator(mode="after")
    def validate_report(self) -> "GroundingReport":
        failed = bool(self.unknown_evidence_ids or self.undisclosed_evidence_ids)
        if failed and self.status is not GroundingStatus.INSUFFICIENT_GROUNDING:
            raise ValueError("invalid citations must produce insufficient_grounding")
        if self.status is GroundingStatus.INSUFFICIENT_GROUNDING and not self.error_code:
            raise ValueError("insufficient_grounding requires an error_code")
        return self

    @classmethod
    def not_evaluated(cls) -> "GroundingReport":
        return cls(status=GroundingStatus.NOT_EVALUATED)


class InvestigationResult(BaseModel):
    """Complete, checkpointable record of one bounded investigation."""

    candidate_id: str = Field(..., min_length=2, max_length=100)
    status: InvestigationStatus
    output: Optional[InvestigatorOutput] = None
    grounding: GroundingReport = Field(default_factory=GroundingReport.not_evaluated)
    observations: Tuple[ToolObservation, ...] = ()
    steps_used: int = Field(..., ge=0)
    tool_calls_used: int = Field(..., ge=0)
    duration_ms: float = Field(..., ge=0.0)
    provider_name: str = Field(..., min_length=2, max_length=80)
    provider_version: str = Field(..., min_length=1, max_length=80)
    budget: Mapping[str, float] = Field(default_factory=dict)
    error_code: Optional[str] = Field(None, max_length=80)
    # A narrative that failed grounding is preserved for the reviewer's audit
    # trail but is deliberately kept out of `output`, so no downstream code can
    # act on an ungrounded opinion.
    rejected_narrative: Optional[str] = Field(None, max_length=1500)

    @model_validator(mode="after")
    def validate_result(self) -> "InvestigationResult":
        if self.status is InvestigationStatus.COMPLETED:
            if self.output is None:
                raise ValueError("completed investigations require structured output")
            if self.error_code is not None:
                raise ValueError("completed investigations may not carry an error_code")
            if self.grounding.status is not GroundingStatus.GROUNDED:
                raise ValueError("completed investigations require grounded citations")
        if self.status is InvestigationStatus.FAILED:
            if not self.error_code:
                raise ValueError("failed investigations require an error_code")
            if self.output is not None:
                raise ValueError("failed investigations may not present clean output")
        if self.status is InvestigationStatus.AWAITING_INFORMATION:
            if self.output is None or not self.output.requested_information:
                raise ValueError("awaiting_information requires a structured request")
        if self.status is InvestigationStatus.ABSTAINED and self.output is not None:
            if self.output.recommendation is not Recommendation.ABSTAIN:
                raise ValueError("abstained investigations may only carry an abstention")
        return self

    @property
    def effective_recommendation(self) -> Recommendation:
        """What the system may act on, after grounding and failure are applied.

        A narrative that failed grounding is discarded entirely; it never
        becomes a `continue_onboarding` opinion.
        """
        if self.status is InvestigationStatus.FAILED:
            return Recommendation.HUMAN_REVIEW
        if self.grounding.status is GroundingStatus.INSUFFICIENT_GROUNDING:
            return Recommendation.HUMAN_REVIEW
        if self.output is None:
            return Recommendation.ABSTAIN
        return self.output.recommendation
