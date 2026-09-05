"""Sparse routing, deterministic policy and failure isolation."""
from __future__ import annotations

from merchantshield.agent import (
    CandidateAction,
    CandidateContext,
    ExpertCost,
    GraphScoringExpert,
    RoutedMoESystem,
    RulesExpert,
    TabularScoringExpert,
)
from merchantshield.analysis import EvidenceGraphBuilder
from merchantshield.evaluation import (
    RoutedMoEBaseline,
    generate_synthetic_dataset,
    group_train_test_split,
)


def _singleton_context(seed: int = 800) -> CandidateContext:
    example = generate_synthetic_dataset(seed=seed)[0]
    applications = {example.example_id: example.application}
    candidate = EvidenceGraphBuilder().build(applications)[0]
    return CandidateContext(candidate=candidate, applications=applications)


async def test_sparse_router_skips_optional_model_for_confident_low_risk():
    context = _singleton_context()
    calls = 0

    async def tabular(applications):
        nonlocal calls
        calls += 1
        return {item: 0.0 for item in applications}

    system = RoutedMoESystem(
        rules_expert=RulesExpert(),
        graph_expert=GraphScoringExpert(lambda _: 0.05),
        tabular_expert=TabularScoringExpert(tabular),
    )
    assessment = await system.assess(context)

    assert assessment.action is CandidateAction.PROCEED_TO_ONBOARDING
    assert assessment.primary_risk_score == 0.05
    assert not assessment.investigator_required
    assert calls == 0
    assert {result.expert_name for result in assessment.expert_results} == {
        "rules",
        "graph_model",
    }


async def test_uncertain_graph_invokes_optional_expert_and_human_review():
    context = _singleton_context(seed=801)

    async def tabular(applications):
        return {item: 0.20 for item in applications}

    system = RoutedMoESystem(
        rules_expert=RulesExpert(),
        graph_expert=GraphScoringExpert(lambda _: 0.50),
        tabular_expert=TabularScoringExpert(tabular),
    )
    assessment = await system.assess(context)

    assert assessment.action is CandidateAction.HUMAN_REVIEW
    assert assessment.primary_risk_score == 0.50
    assert assessment.investigator_required
    assert "GRAPH_RISK_UNCERTAIN" in assessment.reason_codes
    assert any(
        result.expert_name == "tabular_model"
        for result in assessment.expert_results
    )


async def test_primary_expert_failure_is_unknown_and_routes_to_review():
    context = _singleton_context(seed=802)

    class BrokenGraphExpert:
        name = "graph_model"
        cost_class = ExpertCost.MEDIUM
        implementation_version = "broken-test-v1"

        async def evaluate(self, context, *, route_reason):
            raise RuntimeError("simulated model failure")

    async def tabular(applications):
        return {item: 0.01 for item in applications}

    system = RoutedMoESystem(
        rules_expert=RulesExpert(),
        graph_expert=BrokenGraphExpert(),
        tabular_expert=TabularScoringExpert(tabular),
    )
    assessment = await system.assess(context)

    assert assessment.primary_risk_score is None
    assert assessment.action is CandidateAction.HUMAN_REVIEW
    assert "PRIMARY_EXPERT_UNAVAILABLE" in assessment.reason_codes
    assert "EXPERT_FAILURE" in assessment.reason_codes
    graph = next(
        result for result in assessment.expert_results
        if result.expert_name == "graph_model"
    )
    assert graph.risk_score is None
    assert graph.error_code == "EXPERT_EXECUTION_FAILED"


async def test_routed_moe_scores_held_out_data_and_reports_sparse_usage():
    split = group_train_test_split(
        generate_synthetic_dataset(seed=803), seed=804
    )
    baseline = RoutedMoEBaseline(seed=805)
    await baseline.fit(split.train)
    scores, assessments, summary = await baseline.score(split.test)

    assert set(scores) == {item.example_id for item in split.test}
    assert assessments
    assert summary.candidate_count == len(assessments)
    assert summary.expert_execution_count >= 2 * summary.candidate_count
    assert summary.tabular_invocation_count < summary.candidate_count
    assert summary.investigator_request_count == summary.human_review_candidate_count
