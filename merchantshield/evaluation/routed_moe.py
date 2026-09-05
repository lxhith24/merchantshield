"""Held-out evaluation adapter for the production-shaped sparse router."""
from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Dict, Sequence, Tuple

from ..agent import (
    CandidateAction,
    CandidateAssessment,
    CandidateContext,
    GraphScoringExpert,
    RoutedMoESystem,
    RulesExpert,
    TabularScoringExpert,
)
from ..analysis.evidence_graph import EvidenceGraphBuilder
from .dataset import SyntheticExample
from .ring_models import GraphMLBaseline, TabularOnlyBaseline


@dataclass(frozen=True)
class RoutingSummary:
    candidate_count: int
    expert_execution_count: int
    tabular_invocation_count: int
    human_review_candidate_count: int
    investigator_request_count: int

    def to_dict(self) -> dict:
        return asdict(self)


class RoutedMoEBaseline:
    """Fit offline models, then exercise the same routing contract as runtime."""

    name = "routed_moe"

    def __init__(self, *, seed: int = 20250904) -> None:
        self.builder = EvidenceGraphBuilder()
        self.graph_model = GraphMLBaseline(seed=seed)
        self.tabular_model = TabularOnlyBaseline(seed=seed)

    async def fit(self, examples: Sequence[SyntheticExample]) -> None:
        self.graph_model.fit(examples)
        await self.tabular_model.fit(examples)

    async def score(
        self, examples: Sequence[SyntheticExample]
    ) -> Tuple[Dict[str, float], Dict[str, CandidateAssessment], RoutingSummary]:
        applications = {item.example_id: item.application for item in examples}
        candidates = self.builder.build(applications)
        system = RoutedMoESystem(
            rules_expert=RulesExpert(),
            graph_expert=GraphScoringExpert(self.graph_model.score_candidate),
            tabular_expert=TabularScoringExpert(
                self.tabular_model.score_applications
            ),
        )
        scores: Dict[str, float] = {}
        assessments: Dict[str, CandidateAssessment] = {}
        for candidate in candidates:
            assessment = await system.assess(
                CandidateContext(candidate=candidate, applications=applications)
            )
            assessments[candidate.candidate_id] = assessment
            if assessment.primary_risk_score is None:
                raise RuntimeError("evaluation graph expert unexpectedly failed")
            for member_id in candidate.member_ids:
                scores[member_id] = assessment.primary_risk_score

        results = [
            result
            for assessment in assessments.values()
            for result in assessment.expert_results
        ]
        summary = RoutingSummary(
            candidate_count=len(assessments),
            expert_execution_count=len(results),
            tabular_invocation_count=sum(
                result.expert_name == "tabular_model" for result in results
            ),
            human_review_candidate_count=sum(
                assessment.action is CandidateAction.HUMAN_REVIEW
                for assessment in assessments.values()
            ),
            investigator_request_count=sum(
                assessment.investigator_required
                for assessment in assessments.values()
            ),
        )
        return scores, assessments, summary
