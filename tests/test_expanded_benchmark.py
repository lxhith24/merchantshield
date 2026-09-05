"""Expanded fixtures, training isolation, safe workload and report contracts."""
from collections import defaultdict
from dataclasses import replace
import hashlib
import json
from pathlib import Path

import pytest

from merchantshield.evaluation.dataset import Cohort, generate_synthetic_dataset, load_jsonl
from merchantshield.evaluation.expanded_dataset import generate_expanded_dataset
from merchantshield.evaluation.metrics import CostConfig, evaluate_scores
from merchantshield.evaluation.ring_models import GraphMLBaseline, HybridRingBaseline, TabularOnlyBaseline
from merchantshield.evaluation.runner import evaluate_baselines
from merchantshield.evaluation.split import connected_component_groups, group_train_test_split

ROOT = Path(__file__).resolve().parents[1]


@pytest.fixture(scope="module")
def expanded():
    return generate_expanded_dataset()


def test_expansion_is_reproducible_diverse_and_format_safe(expanded):
    assert 1900 < len(expanded) < 2300
    assert [row.to_dict() for row in expanded] == [row.to_dict() for row in generate_expanded_dataset()]
    assert expanded[0].to_dict() != generate_expanded_dataset(seed=19)[0].to_dict()
    assert set(row.cohort for row in expanded) == set(Cohort)
    assert len({row.ring_id for row in expanded if row.ring_id}) == 120
    assert len({row.example_id for row in expanded}) == len(expanded)
    groups = defaultdict(list)
    for row in expanded:
        groups[row.component_id].append(row)
        assert str(row.application.owner_email).endswith(".example")
    assert len({len(members) for members in groups.values()}) >= 8
    assert any(len({row.is_shell for row in members}) == 2 for members in groups.values())
    assert {row.is_shell for row in expanded if row.application.device_fingerprint is None} == {False, True}
    assert {row.is_shell for row in expanded if row.application.ip_address is None} == {False, True}
    assert len(generate_expanded_dataset(scale=2)) > len(expanded) * 1.8


@pytest.mark.parametrize("scale", [0, -1, 11, 1.5, True])
def test_invalid_scale_is_rejected(scale):
    with pytest.raises(ValueError, match="scale"):
        generate_expanded_dataset(scale=scale)


def test_expanded_groups_and_rings_never_cross_partitions(expanded):
    groups = connected_component_groups(expanded)
    for seed in (20250904, 20250905, 20250906):
        split = group_train_test_split(expanded, seed=seed)
        assert {groups[row.example_id] for row in split.train}.isdisjoint(groups[row.example_id] for row in split.test)
        assert {row.component_id for row in split.train}.isdisjoint(row.component_id for row in split.test)
        assert {row.ring_id for row in split.train if row.ring_id}.isdisjoint(row.ring_id for row in split.test if row.ring_id)
        assert len({row.ring_id for row in split.test if row.ring_id}) >= 30
        assert set(row.cohort for row in split.test) == set(Cohort)


def test_explicit_component_protects_disconnected_and_mixed_label_members():
    rows = generate_synthetic_dataset()[:4]
    rows = [replace(row, component_id="declared-component" if index < 2 else row.component_id)
            for index, row in enumerate(rows)]
    groups = connected_component_groups(rows)
    assert groups[rows[0].example_id] == groups[rows[1].example_id]
    assert groups[rows[0].example_id] != groups[rows[2].example_id]


def test_review_only_counts_high_risk_and_missing_evidence_without_rejection():
    rows = generate_synthetic_dataset()[:3]  # three labelled negatives
    scores = {row.example_id: score for row, score in zip(rows, (.9, .5, .01))}
    result = evaluate_scores(rows, scores, review_only=True,
                            additional_review_ids={rows[2].example_id},
                            costs=CostConfig(false_review=7, false_reject=999))
    assert result.manual_review_rate == 1
    assert result.false_positive_cost == 21
    assert result.false_positive_count == 2  # scores vs workload are distinct
    assert result.false_positive_rate == pytest.approx(2 / 3, abs=1e-6)


@pytest.mark.parametrize("cost", [float("nan"), float("inf"), -1])
def test_invalid_review_cost_rejected(cost):
    with pytest.raises(ValueError, match="finite and non-negative"):
        CostConfig(false_review=cost)


def test_additional_review_ids_are_validated():
    rows = generate_synthetic_dataset()[:1]
    with pytest.raises(ValueError, match="belong to examples"):
        evaluate_scores(rows, {rows[0].example_id: .1}, review_only=True, additional_review_ids={"absent"})
    with pytest.raises(ValueError, match="require review_only"):
        evaluate_scores(rows, {rows[0].example_id: .1}, additional_review_ids={rows[0].example_id})


@pytest.mark.asyncio
async def test_training_is_held_out_and_saved_results_are_reproducible(expanded, monkeypatch):
    fitted_ids = []
    original_graph = GraphMLBaseline.fit
    original_tabular = TabularOnlyBaseline.fit
    original_hybrid = HybridRingBaseline.fit

    def graph_fit(self, examples):
        fitted_ids.append({row.example_id for row in examples})
        return original_graph(self, examples)

    async def tabular_fit(self, examples):
        fitted_ids.append({row.example_id for row in examples})
        return await original_tabular(self, examples)

    async def hybrid_fit(self, examples):
        fitted_ids.append({row.example_id for row in examples})
        return await original_hybrid(self, examples)

    monkeypatch.setattr(GraphMLBaseline, "fit", graph_fit)
    monkeypatch.setattr(TabularOnlyBaseline, "fit", tabular_fit)
    monkeypatch.setattr(HybridRingBaseline, "fit", hybrid_fit)
    result = await evaluate_baselines(expanded, review_only=True)
    assert len(fitted_ids) == 5
    assert all(ids == set(result.manifest.train_ids) for ids in fitted_ids)
    assert all(ids.isdisjoint(result.manifest.test_ids) for ids in fitted_ids)
    report = json.loads((ROOT / "data/evaluation/expanded_report.json").read_text())
    assert {name: metric.to_dict() for name, metric in result.reports.items()} == report["baselines"]
    assert sum(next(iter(metrics.values())).sample_count for metrics in result.cohort_reports.values()) == len(result.split.test)
    assert report["metric_policy"]["auto_rejection"] is False
    assert len(report["runs"]) == 3
    assert report["generation"]["dataset_sha256"] == hashlib.sha256((ROOT / "data/evaluation/expanded_merchants.jsonl").read_bytes()).hexdigest()
    assert [row.to_dict() for row in load_jsonl(ROOT / "data/evaluation/expanded_merchants.jsonl")] == [row.to_dict() for row in expanded]
    for run in report["runs"]:
        assert set(run["manifest"]["train_ids"]).isdisjoint(run["manifest"]["test_ids"])
    for name, metrics in report["stability"].items():
        assert metrics["precision"]["min"] == min(run["baselines"][name]["precision"] for run in report["runs"])
