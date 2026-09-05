"""Deterministic sparse routing, failure isolation and permitted-action policy."""
from __future__ import annotations

import asyncio
import time
from dataclasses import dataclass
from typing import Optional, Sequence, Tuple

from .contracts import (
    CandidateAction,
    CandidateAssessment,
    ExecutionMetadata,
    ExpertCost,
    ExpertResult,
    ExpertStatus,
    RouteEvent,
)
from .experts import CandidateContext, CandidateExpert


@dataclass(frozen=True)
class RoutedMoEConfig:
    review_threshold: float = 0.25
    uncertainty_low: float = 0.20
    uncertainty_high: float = 0.75
    disagreement_threshold: float = 0.40
    expert_timeout_seconds: float = 2.0
    policy_version: str = "ring-policy-v1"

    def __post_init__(self) -> None:
        if not 0.0 <= self.review_threshold <= 1.0:
            raise ValueError("review_threshold must be between 0 and 1")
        if not 0.0 <= self.uncertainty_low < self.uncertainty_high <= 1.0:
            raise ValueError("uncertainty band must be ordered within [0, 1]")
        if not 0.0 <= self.disagreement_threshold <= 1.0:
            raise ValueError("disagreement_threshold must be between 0 and 1")
        if self.expert_timeout_seconds <= 0:
            raise ValueError("expert_timeout_seconds must be positive")


class RoutedMoESystem:
    """Run cheap experts first, then selectively invoke additional expertise.

    The graph model owns the numeric risk score. Other experts may cause a case
    to be reviewed, but cannot lower graph risk or autonomously reject anyone.
    """

    def __init__(
        self,
        *,
        rules_expert: CandidateExpert,
        graph_expert: CandidateExpert,
        tabular_expert: Optional[CandidateExpert] = None,
        config: RoutedMoEConfig = RoutedMoEConfig(),
    ) -> None:
        self.rules_expert = rules_expert
        self.graph_expert = graph_expert
        self.tabular_expert = tabular_expert
        self.config = config

    async def assess(self, context: CandidateContext) -> CandidateAssessment:
        trace = [
            RouteEvent(
                stage="initial_fanout",
                selected_experts=(self.rules_expert.name, self.graph_expert.name),
                reason="Always run cheap consistency rules and the primary graph model.",
            )
        ]
        initial = await asyncio.gather(
            self._run_safely(
                self.rules_expert,
                context,
                "mandatory cheap consistency screening",
            ),
            self._run_safely(
                self.graph_expert,
                context,
                "mandatory primary ring-risk scoring",
            ),
        )
        results = list(initial)
        by_name = {result.expert_name: result for result in results}
        graph = by_name.get(self.graph_expert.name)
        rules = by_name.get(self.rules_expert.name)

        optional_reason = self._tabular_route_reason(graph, rules)
        if optional_reason and self.tabular_expert is not None:
            trace.append(
                RouteEvent(
                    stage="conditional_fanout",
                    selected_experts=(self.tabular_expert.name,),
                    reason=optional_reason,
                )
            )
            results.append(
                await self._run_safely(
                    self.tabular_expert,
                    context,
                    optional_reason,
                )
            )
        else:
            trace.append(
                RouteEvent(
                    stage="conditional_fanout",
                    selected_experts=(),
                    reason=(
                        optional_reason + " Optional expert is unavailable."
                        if optional_reason
                        else "Primary and rules outputs do not require tabular escalation."
                    ),
                )
            )

        assessment = self._apply_policy(context, results, trace)
        return assessment

    def _tabular_route_reason(
        self, graph: Optional[ExpertResult], rules: Optional[ExpertResult]
    ) -> Optional[str]:
        if graph is None or graph.status is not ExpertStatus.COMPLETED:
            return "Primary graph output is unavailable; obtain an independent fallback view."
        assert graph.risk_score is not None
        uncertain = self.config.uncertainty_low < graph.risk_score < self.config.uncertainty_high
        disagreement = (
            rules is not None
            and rules.status is ExpertStatus.COMPLETED
            and rules.risk_score is not None
            and abs(graph.risk_score - rules.risk_score)
            >= self.config.disagreement_threshold
        )
        if uncertain and disagreement:
            return "Graph risk is uncertain and materially disagrees with consistency rules."
        if uncertain:
            return "Graph risk lies inside the configured uncertainty band."
        if disagreement:
            return "Graph risk materially disagrees with consistency rules."
        return None

    def _apply_policy(
        self,
        context: CandidateContext,
        results: Sequence[ExpertResult],
        trace: Sequence[RouteEvent],
    ) -> CandidateAssessment:
        by_name = {result.expert_name: result for result in results}
        graph = by_name.get(self.graph_expert.name)
        rules = by_name.get(self.rules_expert.name)
        tabular = by_name.get(self.tabular_expert.name) if self.tabular_expert else None
        reasons = []
        primary_score = None

        if graph is None or graph.status is not ExpertStatus.COMPLETED:
            reasons.append("PRIMARY_EXPERT_UNAVAILABLE")
        else:
            primary_score = graph.risk_score
            assert primary_score is not None
            if primary_score > self.config.review_threshold:
                reasons.append("GRAPH_RISK_ABOVE_REVIEW_THRESHOLD")
            if self.config.uncertainty_low < primary_score < self.config.uncertainty_high:
                reasons.append("GRAPH_RISK_UNCERTAIN")

        if self._material_disagreement(graph, rules):
            reasons.append("RULES_GRAPH_DISAGREEMENT")
        if self._material_disagreement(graph, tabular):
            reasons.append("TABULAR_GRAPH_DISAGREEMENT")
        if any(result.status is ExpertStatus.FAILED for result in results):
            reasons.append("EXPERT_FAILURE")

        action = (
            CandidateAction.HUMAN_REVIEW
            if reasons
            else CandidateAction.PROCEED_TO_ONBOARDING
        )
        final_trace = tuple(trace) + (
            RouteEvent(
                stage="deterministic_policy",
                selected_experts=(),
                reason=(
                    "Route to human review; LLM may recommend or abstain but cannot decide."
                    if action is CandidateAction.HUMAN_REVIEW
                    else "No review trigger fired; candidate may continue to onboarding."
                ),
            ),
        )
        return CandidateAssessment(
            candidate_id=context.candidate.candidate_id,
            primary_risk_score=primary_score,
            action=action,
            reason_codes=tuple(dict.fromkeys(reasons)),
            investigator_required=action is CandidateAction.HUMAN_REVIEW,
            expert_results=tuple(results),
            route_trace=final_trace,
            policy_version=self.config.policy_version,
        )

    def _material_disagreement(
        self,
        primary: Optional[ExpertResult],
        other: Optional[ExpertResult],
    ) -> bool:
        return bool(
            primary
            and other
            and primary.status is ExpertStatus.COMPLETED
            and other.status is ExpertStatus.COMPLETED
            and primary.risk_score is not None
            and other.risk_score is not None
            and abs(primary.risk_score - other.risk_score)
            >= self.config.disagreement_threshold
        )

    async def _run_safely(
        self,
        expert: CandidateExpert,
        context: CandidateContext,
        route_reason: str,
    ) -> ExpertResult:
        started = time.perf_counter()
        try:
            return await asyncio.wait_for(
                expert.evaluate(context, route_reason=route_reason),
                timeout=self.config.expert_timeout_seconds,
            )
        except asyncio.TimeoutError:
            code = "EXPERT_TIMEOUT"
            rationale = "Expert exceeded its bounded execution time."
        except Exception as exc:  # failure isolation is the point of this boundary
            code = "EXPERT_EXECUTION_FAILED"
            rationale = f"Expert failed safely: {type(exc).__name__}."
        return ExpertResult(
            expert_name=expert.name,
            candidate_id=context.candidate.candidate_id,
            status=ExpertStatus.FAILED,
            risk_score=None,
            confidence=None,
            evidence=(),
            rationale=rationale,
            error_code=code,
            metadata=ExecutionMetadata(
                implementation_version=expert.implementation_version,
                route_reason=route_reason,
                duration_ms=round(max(0.0, (time.perf_counter() - started) * 1000.0), 3),
                cost_class=getattr(expert, "cost_class", ExpertCost.EXPENSIVE),
            ),
        )
