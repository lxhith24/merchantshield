"""Evaluation metrics, operational costs, and baseline execution."""
from __future__ import annotations

import pytest

from merchantshield.evaluation import (
    ClusteringOnlyBaseline,
    Cohort,
    CostConfig,
    RulesOnlyBaseline,
    SyntheticExample,
    evaluate_baselines,
    evaluate_scores,
    generate_synthetic_dataset,
)


def test_metrics_include_screening_ring_workflow_and_cost_measures():
    source = generate_synthetic_dataset(seed=3)[:4]
    examples = [
        SyntheticExample(
            example_id=f"metric-{index}",
            application=row.application,
            is_shell=index < 2,
            cohort=Cohort.EVASIVE_SHELL_RING if index < 2 else Cohort.LEGITIMATE,
            ring_id="metric-ring" if index < 2 else None,
        )
        for index, row in enumerate(source)
    ]
    scores = {
        "metric-0": 0.90,
        "metric-1": 0.10,
        "metric-2": 0.50,
        "metric-3": 0.00,
    }
    report = evaluate_scores(
        examples,
        scores,
        costs=CostConfig(false_review=2.0, false_reject=9.0),
    )
    assert report.precision == 0.5
    assert report.recall == 0.5
    assert report.f1 == 0.5
    assert report.pr_auc == pytest.approx(0.833333, abs=1e-6)
    assert report.false_positive_rate == 0.5
    assert report.ring_recall == 1.0
    assert report.manual_review_rate == 0.25
    assert report.false_positive_cost == 2.0
    assert report.false_positive_count == 1

    reject_scores = {**scores, "metric-2": 0.80}
    reject_report = evaluate_scores(
        examples,
        reject_scores,
        costs=CostConfig(false_review=2.0, false_reject=9.0),
    )
    assert reject_report.manual_review_rate == 0.0
    assert reject_report.false_positive_cost == 9.0


@pytest.mark.asyncio
async def test_rules_and_clustering_baselines_are_distinct_and_label_blind():
    examples = generate_synthetic_dataset(seed=31)
    rules = await RulesOnlyBaseline().score(examples)
    clusters = ClusteringOnlyBaseline().score(examples)
    assert set(rules) == set(clusters) == {row.example_id for row in examples}
    assert all(0.0 <= value <= 1.0 for value in rules.values())
    assert all(0.0 <= value <= 1.0 for value in clusters.values())

    evasive_ids = {
        row.example_id for row in examples if row.cohort is Cohort.EVASIVE_SHELL_RING
    }
    assert max(clusters[item] for item in evasive_ids) > max(
        rules[item] for item in evasive_ids
    )


@pytest.mark.asyncio
async def test_held_out_runner_reports_both_baselines_and_disclaimer():
    evaluation = await evaluate_baselines(
        generate_synthetic_dataset(seed=41),
        seed=42,
        costs=CostConfig(false_review=3.0, false_reject=11.0),
    )
    assert set(evaluation.reports) == {
        "rules_only",
        "clustering_only",
        "graph_only",
        "tabular_only",
        "graph_ml",
        "hybrid",
        "routed_moe",
    }
    assert evaluation.split.train
    assert evaluation.split.test
    assert evaluation.manifest.dataset_version == "synthetic-rings-v2"
    assert evaluation.manifest.feature_schema_version == "evidence-graph-v1"
    assert evaluation.manifest.routing_policy_version == "ring-policy-v1"
    assert set(evaluation.manifest.train_ids).isdisjoint(evaluation.manifest.test_ids)
    assert evaluation.manifest.model_names == tuple(evaluation.reports)
    assert evaluation.routing_summary.candidate_count > 0
    assert "not production Razorpay performance" in evaluation.disclaimer
    for report in evaluation.reports.values():
        assert report.sample_count == len(evaluation.split.test)
        assert 0.0 <= report.pr_auc <= 1.0


def test_metrics_reject_missing_predictions():
    examples = generate_synthetic_dataset(seed=5)[:2]
    with pytest.raises(ValueError, match="scores must match examples"):
        evaluate_scores(examples, {examples[0].example_id: 0.1})
