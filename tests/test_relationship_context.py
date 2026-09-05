"""Wider relationship context cannot turn missing/oversized evidence into clean."""
from dataclasses import replace
import json
from pathlib import Path

import numpy as np
import pytest

from merchantshield.evaluation.dataset import generate_synthetic_dataset
from merchantshield.evaluation.relationships import (
    RelationshipRiskModel, extract_relationship_features, relationship_evidence_requests,
)


def chain_applications(size):
    rows = generate_synthetic_dataset()[:size]
    return {row.example_id: row.application.model_copy(update={
        "device_fingerprint": f"pair-{index // 2}",
        "bank_account": str(980000000000 + (index + 1) // 2),
        "owner_email": f"contact@unique-{index}.example",
        "owner_pan": f"ABCPZ{index + 1000:04d}A",
        "registered_address": f"Office uniquetoken{index}longtoken Pune 411001",
        "ip_address": f"fd00:abcd::{index:x}",
    }) for index, row in enumerate(rows)}


@pytest.mark.asyncio
async def test_weak_multihop_links_are_seen_without_ring_labels():
    apps = chain_applications(8)
    features = await extract_relationship_features(apps)
    assert set(features.context_sizes) == {8}
    assert not features.context_capped_ids
    assert np.isfinite(features.combined).all()
    reordered = await extract_relationship_features(dict(reversed(list(apps.items()))))
    np.testing.assert_array_equal(features.combined, reordered.combined)
    renamed = await extract_relationship_features({"renamed-" + key: value for key, value in apps.items()})
    np.testing.assert_array_equal(features.combined, renamed.combined)


@pytest.mark.asyncio
async def test_missing_values_do_not_join_unrelated_applications():
    rows = generate_synthetic_dataset()[:10]
    apps = {row.example_id: row.application.model_copy(update={"device_fingerprint": None, "ip_address": None}) for row in rows}
    features = await extract_relationship_features(apps)
    assert set(features.context_sizes) == {1}
    assert features.missing_ids == set(apps)


@pytest.mark.asyncio
async def test_oversized_relationship_context_routes_to_review_even_at_low_risk():
    apps = chain_applications(55)
    batch = await extract_relationship_features(apps)
    assert batch.context_capped_ids == set(apps)
    assert batch.missing_ids == set(apps)
    class LowModel(RelationshipRiskModel):
        def score(self, features, **kwargs):
            return {key: .001 for key in features.ids}, 0
    result = await LowModel().screen_applications(apps, threshold=.2)
    assert all(row["action"] == "human_review" for row in result["applications"].values())
    assert all("RELATIONSHIP_CONTEXT_LIMIT" in row["reason_codes"] for row in result["applications"].values())


def test_relationship_checklists_do_not_assert_verification_or_guilt():
    rows = generate_synthetic_dataset()
    ordinary = {row.example_id: row.application for row in rows[:4]}
    assert relationship_evidence_requests(ordinary) == []
    shared = {row.example_id: row.application for row in rows[34:38]}
    requests = relationship_evidence_requests(shared)
    assert {row["attribute"] for row in requests} >= {"device_fingerprint", "ip_address"}
    assert all(row["status"] == "UNVERIFIED" for row in requests)
    assert any("alone is not fraud evidence" in row["request"] for row in requests)


@pytest.mark.asyncio
async def test_relationship_fit_keeps_merchant_labels_separate():
    rows = generate_synthetic_dataset()
    batch = await extract_relationship_features({row.example_id: row.application for row in rows})
    model = RelationshipRiskModel(group_weight=True, leaf=5).fit(batch, {row.example_id: row.is_shell for row in rows})
    assert model.training_ids == set(batch.ids)
    scores, calls = model.score(batch)
    assert set(scores) == set(batch.ids)
    assert 0 <= calls <= len(rows)
    assert all(0 <= score <= 1 for score in scores.values())
    with pytest.raises(ValueError, match="aligned training"):
        model.fit(batch, {key: False for key in batch.ids})


@pytest.mark.parametrize("blend", [0, 1.1, float("nan")])
def test_invalid_blend_rejected(blend):
    with pytest.raises(ValueError, match="blend"):
        RelationshipRiskModel(blend=blend)


@pytest.mark.asyncio
async def test_locked_relationship_study_has_no_leakage_and_retains_failed_checks():
    from scripts.recover_hard_rings import load_locked, feature_batch
    from merchantshield.evaluation.performance import measure
    directory = Path(__file__).resolve().parents[1] / "data/evaluation/relationships"
    plan, partitions = await load_locked(directory, require_report=True)
    report = json.loads((directory / "report.json").read_text())
    assigned = {}
    from merchantshield.evaluation.split import connected_component_groups
    all_rows = [row for partition in partitions.values() for row in partition]
    groups = connected_component_groups(all_rows)
    for name, rows in partitions.items():
        for row in rows:
            group = groups[row.example_id]
            assert assigned.setdefault(group, name) == name
    train, test = partitions["train"], partitions["test"]
    model = RelationshipRiskModel(**plan["selected"], seed=plan["seed"])
    model.fit(await feature_batch(train), {row.example_id: row.is_shell for row in train})
    assert model.training_ids.isdisjoint(plan["ids"]["test"] + plan["ids"]["validation"])
    scores, _ = model.score(await feature_batch(test))
    threshold = plan["thresholds"]["routed_moe"]["threshold"]
    assert measure(test, scores, threshold).to_dict() == report["baselines"]["routed_moe"]
    assert report["adoption_passed"] == all(report["adoption_checks"].values())
    assert not report["adoption_passed"]


@pytest.mark.asyncio
async def test_relationship_test_cannot_be_overwritten(tmp_path):
    from scripts.recover_hard_rings import test_once
    (tmp_path / "report.json").write_text("{}")
    with pytest.raises(ValueError, match="refusing to overwrite"):
        await test_once(tmp_path)
