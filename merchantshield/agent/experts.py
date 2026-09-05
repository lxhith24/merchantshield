"""Concrete non-LLM experts used by the sparse router."""
from __future__ import annotations

import hashlib
import inspect
import time
from dataclasses import dataclass
from typing import Awaitable, Callable, Dict, Mapping, Protocol, Union

from ..analysis.evidence_graph import RingCandidate
from ..analysis.synthetic_identity import SyntheticIdentityDetector
from ..models.merchant import MerchantApplication
from .contracts import (
    ExecutionMetadata,
    ExpertCost,
    ExpertEvidence,
    ExpertResult,
    ExpertStatus,
)


@dataclass(frozen=True)
class CandidateContext:
    candidate: RingCandidate
    applications: Mapping[str, MerchantApplication]

    def __post_init__(self) -> None:
        missing = set(self.candidate.member_ids) - set(self.applications)
        if missing:
            raise ValueError(f"candidate applications are missing: {sorted(missing)}")


class CandidateExpert(Protocol):
    name: str
    cost_class: ExpertCost
    implementation_version: str

    async def evaluate(
        self, context: CandidateContext, *, route_reason: str
    ) -> ExpertResult: ...


class RulesExpert:
    """Cheap per-application consistency rules aggregated at candidate level."""

    name = "rules"
    cost_class = ExpertCost.CHEAP
    implementation_version = "identity-rules-v1"

    def __init__(self) -> None:
        self.detector = SyntheticIdentityDetector()

    async def evaluate(
        self, context: CandidateContext, *, route_reason: str
    ) -> ExpertResult:
        started = time.perf_counter()
        maximum = 0.0
        evidence = []
        for member_id in sorted(context.candidate.member_ids):
            result = await self.detector.analyze(context.applications[member_id])
            maximum = max(maximum, float(result["synthetic_score"]))
            for signal in result["signals"]:
                evidence.append(
                    ExpertEvidence(
                        evidence_id=_rule_evidence_id(
                            context.candidate.candidate_id, member_id, signal.name
                        ),
                        evidence_type=signal.name,
                        summary=(
                            f"Deterministic consistency rule '{signal.name}' fired "
                            f"for member {member_id}."
                        ),
                        source=self.name,
                        severity=float(signal.value),
                    )
                )
        evidence.sort(key=lambda item: (-item.severity, item.evidence_id))
        return ExpertResult(
            expert_name=self.name,
            candidate_id=context.candidate.candidate_id,
            status=ExpertStatus.COMPLETED,
            risk_score=round(maximum, 6),
            confidence=0.95,
            evidence=tuple(evidence[:20]),
            rationale=(
                "Candidate score is the maximum successfully measured member-level "
                "consistency score; evidence text is redacted."
            ),
            metadata=_metadata(
                self, route_reason, started
            ),
        )


class GraphScoringExpert:
    """Primary candidate-level graph model behind a narrow callable contract."""

    name = "graph_model"
    cost_class = ExpertCost.MEDIUM
    implementation_version = "graph-logistic-v1"

    def __init__(self, scorer: Callable[[RingCandidate], float]) -> None:
        self.scorer = scorer

    async def evaluate(
        self, context: CandidateContext, *, route_reason: str
    ) -> ExpertResult:
        started = time.perf_counter()
        score = float(self.scorer(context.candidate))
        strongest = sorted(
            context.candidate.evidence,
            key=lambda item: (-item.effective_weight, item.evidence_id),
        )[:20]
        evidence = tuple(
            ExpertEvidence(
                evidence_id=item.evidence_id,
                evidence_type=item.attribute,
                summary=item.explanation,
                source=self.name,
                severity=item.effective_weight,
            )
            for item in strongest
        )
        return ExpertResult(
            expert_name=self.name,
            candidate_id=context.candidate.candidate_id,
            status=ExpertStatus.COMPLETED,
            risk_score=round(score, 6),
            confidence=_graph_confidence(context.candidate),
            evidence=evidence,
            rationale=(
                "Interpretable logistic model over redacted graph topology and "
                "shared-infrastructure features."
            ),
            metadata=_metadata(self, route_reason, started),
        )


TabularScores = Mapping[str, float]
TabularScorer = Callable[
    [Mapping[str, MerchantApplication]],
    Union[TabularScores, Awaitable[TabularScores]],
]


class TabularScoringExpert:
    """Optional application-level model, invoked only by the sparse router."""

    name = "tabular_model"
    cost_class = ExpertCost.MEDIUM
    implementation_version = "tabular-logistic-v1"

    def __init__(self, scorer: TabularScorer) -> None:
        self.scorer = scorer

    async def evaluate(
        self, context: CandidateContext, *, route_reason: str
    ) -> ExpertResult:
        started = time.perf_counter()
        member_apps = {
            member_id: context.applications[member_id]
            for member_id in context.candidate.member_ids
        }
        pending = self.scorer(member_apps)
        scores = await pending if inspect.isawaitable(pending) else pending
        if set(scores) != set(member_apps):
            raise ValueError("tabular scorer must return one score per candidate member")
        score = max((float(value) for value in scores.values()), default=0.0)
        return ExpertResult(
            expert_name=self.name,
            candidate_id=context.candidate.candidate_id,
            status=ExpertStatus.COMPLETED,
            risk_score=round(score, 6),
            confidence=0.75,
            evidence=(),
            rationale=(
                "Maximum member-level probability from the tabular model; this "
                "expert may trigger review but does not replace graph risk."
            ),
            metadata=_metadata(self, route_reason, started),
        )


def _rule_evidence_id(candidate_id: str, member_id: str, signal_name: str) -> str:
    digest = hashlib.sha256(
        f"{candidate_id}|{member_id}|{signal_name}".encode("utf-8")
    ).hexdigest()[:12]
    return f"rule-{digest}"


def _graph_confidence(candidate: RingCandidate) -> float:
    if not candidate.evidence:
        return 0.70
    corroboration = candidate.features.corroborated_pair_ratio
    density = candidate.features.density
    return round(min(0.98, 0.78 + 0.12 * corroboration + 0.08 * density), 6)


def _metadata(expert: CandidateExpert, route_reason: str, started: float) -> ExecutionMetadata:
    return ExecutionMetadata(
        implementation_version=expert.implementation_version,
        route_reason=route_reason,
        duration_ms=round(max(0.0, (time.perf_counter() - started) * 1000.0), 3),
        cost_class=expert.cost_class,
    )
