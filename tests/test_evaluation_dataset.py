"""Synthetic benchmark generation and leakage-safe splitting."""
from __future__ import annotations

from collections import Counter

from merchantshield.evaluation import (
    Cohort,
    connected_component_groups,
    generate_synthetic_dataset,
    group_train_test_split,
    load_jsonl,
    write_jsonl,
)


def test_dataset_is_reproducible_and_contains_every_required_cohort(tmp_path):
    first = generate_synthetic_dataset(seed=17)
    second = generate_synthetic_dataset(seed=17)
    assert [row.to_dict() for row in first] == [row.to_dict() for row in second]

    counts = Counter(row.cohort for row in first)
    assert set(counts) == set(Cohort)
    assert all(count > 0 for count in counts.values())
    assert any(row.is_shell for row in first)
    assert any(not row.is_shell for row in first)
    assert all(row.application.owner_email for row in first)

    path = tmp_path / "benchmark.jsonl"
    write_jsonl(first, path)
    loaded = load_jsonl(path)
    assert [row.to_dict() for row in loaded] == [row.to_dict() for row in first]


def test_group_split_keeps_rings_and_connected_components_whole():
    examples = generate_synthetic_dataset(seed=23)
    split = group_train_test_split(examples, test_fraction=0.30, seed=91)
    train_ids = {row.example_id for row in split.train}
    test_ids = {row.example_id for row in split.test}
    assert train_ids.isdisjoint(test_ids)
    assert train_ids | test_ids == {row.example_id for row in examples}

    components = connected_component_groups(examples)
    train_components = {components[item] for item in train_ids}
    test_components = {components[item] for item in test_ids}
    assert train_components.isdisjoint(test_components)

    for ring_id in {row.ring_id for row in examples if row.ring_id}:
        partitions = {
            "train" if row.example_id in train_ids else "test"
            for row in examples
            if row.ring_id == ring_id
        }
        assert len(partitions) == 1


def test_split_is_seed_reproducible():
    examples = generate_synthetic_dataset(seed=9)
    left = group_train_test_split(examples, seed=12)
    right = group_train_test_split(examples, seed=12)
    assert [row.example_id for row in left.train] == [row.example_id for row in right.train]
    assert [row.example_id for row in left.test] == [row.example_id for row in right.test]
