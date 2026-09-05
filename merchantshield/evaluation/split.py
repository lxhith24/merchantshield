"""Leakage-safe train/test splitting for linked merchant applications."""
from __future__ import annotations

import random
from dataclasses import dataclass
from typing import Dict, List, Sequence

from ..analysis.application_clustering import ApplicationClusterer
from .dataset import SyntheticExample


@dataclass(frozen=True)
class DatasetSplit:
    train: List[SyntheticExample]
    test: List[SyntheticExample]


class _UnionFind:
    def __init__(self, ids: Sequence[str]) -> None:
        self.parent = {item: item for item in ids}

    def find(self, item: str) -> str:
        while self.parent[item] != item:
            self.parent[item] = self.parent[self.parent[item]]
            item = self.parent[item]
        return item

    def union(self, left: str, right: str) -> None:
        left_root, right_root = self.find(left), self.find(right)
        if left_root != right_root:
            self.parent[max(left_root, right_root)] = min(left_root, right_root)


def connected_component_groups(examples: Sequence[SyntheticExample]) -> Dict[str, str]:
    """Return example -> component using every production clustering key.

    Explicit ring and component IDs are unioned as additional safeguards,
    including when missing observations disconnect a labelled group. A
    ring, or any larger connected component containing it, cannot straddle the
    train/test boundary.
    """
    ids = [e.example_id for e in examples]
    if len(ids) != len(set(ids)):
        raise ValueError("example_id values must be unique")

    union_find = _UnionFind(ids)
    clusterer = ApplicationClusterer()
    seen_keys: Dict[tuple[str, str], str] = {}
    seen_rings: Dict[str, str] = {}
    seen_components: Dict[str, str] = {}

    for example in examples:
        if example.component_id:
            if example.component_id in seen_components:
                union_find.union(example.example_id, seen_components[example.component_id])
            else:
                seen_components[example.component_id] = example.example_id
        for attr, value in clusterer.extract_keys(example.application).items():
            key = (attr, value)
            if key in seen_keys:
                union_find.union(example.example_id, seen_keys[key])
            else:
                seen_keys[key] = example.example_id
        if example.ring_id:
            if example.ring_id in seen_rings:
                union_find.union(example.example_id, seen_rings[example.ring_id])
            else:
                seen_rings[example.ring_id] = example.example_id

    return {example_id: union_find.find(example_id) for example_id in ids}


def group_train_test_split(
    examples: Sequence[SyntheticExample],
    test_fraction: float = 0.30,
    seed: int = 20250904,
) -> DatasetSplit:
    """Cohort-stratified split that assigns whole connected components."""
    if not 0.0 < test_fraction < 1.0:
        raise ValueError("test_fraction must be between 0 and 1")
    if not examples:
        raise ValueError("examples must not be empty")

    components = connected_component_groups(examples)
    by_component: Dict[str, List[SyntheticExample]] = {}
    for example in examples:
        by_component.setdefault(components[example.example_id], []).append(example)

    by_cohort: Dict[str, List[str]] = {}
    for component, members in by_component.items():
        cohorts = {member.cohort.value for member in members}
        # Cross-cohort connected components are kept intact under a mixed bucket.
        cohort_key = next(iter(cohorts)) if len(cohorts) == 1 else "__mixed__"
        by_cohort.setdefault(cohort_key, []).append(component)

    rng = random.Random(seed)
    test_components: set[str] = set()
    for cohort in sorted(by_cohort):
        groups = sorted(by_cohort[cohort])
        rng.shuffle(groups)
        count = max(1, round(len(groups) * test_fraction)) if len(groups) > 1 else 0
        test_components.update(groups[:count])

    train = [e for e in examples if components[e.example_id] not in test_components]
    test = [e for e in examples if components[e.example_id] in test_components]
    if not train or not test:
        raise ValueError("split produced an empty partition; provide more independent groups")
    return DatasetSplit(train=train, test=test)
