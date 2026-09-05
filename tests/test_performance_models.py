"""Honest tuning separation, feature contracts and safe inference."""
from dataclasses import replace
import json
from pathlib import Path

import numpy as np
import pytest

from merchantshield.evaluation.dataset import generate_synthetic_dataset, load_jsonl
from merchantshield.evaluation.performance import (
    MerchantPeerModel, extract_features, select_threshold, screening_decision,
)
from scripts.improve_risk_models import split_rows, digest, test_locked as evaluate_locked_test
from merchantshield.evaluation.validated import load_validated_model

ROOT = Path(__file__).resolve().parents[1]


@pytest.mark.asyncio
async def test_features_have_no_labels_and_ignore_input_order_and_ids():
    rows = generate_synthetic_dataset()[:40]
    original = await extract_features({row.example_id: row.application for row in rows})
    reordered = await extract_features({row.example_id: row.application for row in reversed(rows)})
    assert original.ids == reordered.ids
    np.testing.assert_array_equal(original.combined, reordered.combined)
    renamed = await extract_features({"renamed-" + row.example_id: row.application for row in rows})
    np.testing.assert_array_equal(original.combined, renamed.combined)
    assert np.isfinite(original.combined).all()


@pytest.mark.parametrize("score", [None, float("nan"), float("inf"), -1, 2])
def test_unavailable_risk_cannot_proceed(score):
    assert screening_decision(score, .4)["action"] == "human_review"


def test_threshold_boundary_and_missing_evidence_are_distinct_from_fraud():
    assert screening_decision(.4, .4)["action"] == "proceed_to_onboarding"
    assert screening_decision(.400001, .4)["action"] == "human_review"
    missing = screening_decision(.01, .4, missing_evidence=True)
    assert missing["action"] == "human_review"
    assert missing["fraud_flag"] is False
    assert missing["reason_codes"] == ["MISSING_SESSION_EVIDENCE"]


@pytest.mark.parametrize("threshold", [0, 1, float("nan")])
def test_invalid_threshold_rejected(threshold):
    with pytest.raises(ValueError, match="threshold"):
        screening_decision(.5, threshold)


def test_threshold_selection_is_deterministic_and_meets_floors():
    rows = generate_synthetic_dataset()[:10]
    rows = [replace(row, is_shell=index < 5, ring_id=f"ring-{index}" if index < 5 else None) for index, row in enumerate(rows)]
    scores = {row.example_id: .8 if row.is_shell else .2 for row in rows}
    left = select_threshold(rows, scores)
    assert left == select_threshold(rows, scores)
    assert left["metrics"]["precision"] == 1
    assert left["metrics"]["recall"] == 1
    with pytest.raises(ValueError, match="no validation threshold"):
        select_threshold(rows, {row.example_id: 0. for row in rows})


def test_fresh_three_way_split_separates_every_ring_and_component():
    rows = load_jsonl(ROOT / "data/evaluation/performance_merchants.jsonl")
    partitions, components = split_rows(rows)
    group_sets = [{components[row.example_id] for row in values} for values in partitions.values()]
    for left in range(3):
        for right in range(left + 1, 3):
            assert group_sets[left].isdisjoint(group_sets[right])
    assert sum(map(len, partitions.values())) == len(rows)


@pytest.mark.asyncio
async def test_real_model_is_fitted_only_on_passed_training_ids_and_sparse_calls_skip_rows():
    rows = generate_synthetic_dataset()
    partitions, _ = split_rows(rows)
    train = partitions["train"]
    batch = await extract_features({row.example_id: row.application for row in train})
    model = MerchantPeerModel(leaf=5).fit(batch, {row.example_id: row.is_shell for row in train})
    assert model.training_ids == {row.example_id for row in train}
    test = partitions["test"]
    assert model.training_ids.isdisjoint(row.example_id for row in test)
    apps = {row.example_id: row.application for row in test}
    result = await model.screen_applications(apps, threshold=.4)
    assert set(result["applications"]) == set(apps)
    assert all(item["action"] in ("human_review", "proceed_to_onboarding") for item in result["applications"].values())
    # Guaranteed low-confidence gate case: avoids relying on a lucky fitted score.
    class LowGraph:
        def predict_proba(self, values):
            return np.tile([.99, .01], (len(values), 1))
    class NeverSpecialist:
        def predict_proba(self, values):
            raise AssertionError("specialist must not run for confident complete rows")
    batch = replace(batch, missing_ids=set(), rule_scores=np.zeros(len(batch.ids)))
    model.graph, model.hybrid = LowGraph(), NeverSpecialist()
    scores, calls = model.score(batch)
    assert calls == 0 and set(scores.values()) == {.01}


@pytest.mark.asyncio
async def test_locked_test_refuses_tampering_or_overwrite(tmp_path):
    plan = {"code_hash": "changed"}
    plan["lock_hash"] = digest(plan)
    (tmp_path / "performance_plan.json").write_text(json.dumps(plan))
    with pytest.raises(ValueError, match="scoring code changed"):
        await evaluate_locked_test(tmp_path)
    (tmp_path / "performance_report.json").write_text("{}")
    with pytest.raises(ValueError, match="refusing to silently overwrite"):
        await evaluate_locked_test(tmp_path)


@pytest.mark.asyncio
async def test_saved_challenger_reproduces_locked_test_without_refitting_on_test():
    directory = ROOT / "data/evaluation"
    plan = json.loads((directory / "performance_plan.json").read_text())
    report = json.loads((directory / "performance_report.json").read_text())
    model, threshold = await load_validated_model(directory)
    assert model.training_ids == set(plan["ids"]["train"])
    assert model.training_ids.isdisjoint(plan["ids"]["validation"] + plan["ids"]["test"])
    rows = load_jsonl(directory / "performance_merchants.jsonl")
    test_ids = set(plan["ids"]["test"])
    test = [row for row in rows if row.example_id in test_ids]
    from merchantshield.evaluation.performance import measure
    batch = await extract_features({row.example_id: row.application for row in test})
    scores, _ = model.score(batch)
    assert measure(test, scores, threshold).to_dict() == report["baselines"]["routed_moe"]
    # Selection depends on validation objective only, not reported test metrics.
    best = min(plan["trials"], key=lambda trial: (trial["objective"], trial["metrics"]["false_positive_rate"], trial["depth"], trial["leaf"]))
    assert plan["selected"] == {"depth": best["depth"], "leaf": best["leaf"]}
