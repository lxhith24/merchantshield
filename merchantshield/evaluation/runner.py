"""Held-out baseline evaluation runner."""
from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Dict, Sequence, Tuple

from .baselines import ClusteringOnlyBaseline, RulesOnlyBaseline
from .dataset import SyntheticExample
from .metrics import CostConfig, EvaluationMetrics, evaluate_scores
from .ring_models import (
    GraphMLBaseline,
    GraphOnlyBaseline,
    HybridRingBaseline,
    TabularOnlyBaseline,
)
from .routed_moe import RoutedMoEBaseline, RoutingSummary
from .split import DatasetSplit, group_train_test_split
from ..agent import CandidateAction


DATASET_VERSION = "synthetic-rings-v2"
FEATURE_SCHEMA_VERSION = "evidence-graph-v1"
ROUTING_POLICY_VERSION = "ring-policy-v1"


@dataclass(frozen=True)
class EvaluationManifest:
    dataset_version: str
    feature_schema_version: str
    routing_policy_version: str
    seed: int
    test_fraction: float
    train_ids: Tuple[str, ...]
    test_ids: Tuple[str, ...]
    model_names: Tuple[str, ...]
    false_review_cost: float
    false_reject_cost: float

    def to_dict(self) -> dict:
        return asdict(self)


@dataclass(frozen=True)
class BaselineEvaluation:
    split: DatasetSplit
    manifest: EvaluationManifest
    reports: Dict[str, EvaluationMetrics]
    routing_summary: RoutingSummary
    cohort_reports: Dict[str, Dict[str, EvaluationMetrics]]
    disclaimer: str = (
        "Synthetic benchmark results; they are not production Razorpay performance."
    )


async def evaluate_baselines(
    examples: Sequence[SyntheticExample],
    *,
    test_fraction: float = 0.30,
    seed: int = 20250904,
    costs: CostConfig = CostConfig(),
    dataset_version: str = DATASET_VERSION,
    review_only: bool = False,
) -> BaselineEvaluation:
    split = group_train_test_split(examples, test_fraction=test_fraction, seed=seed)
    rules_scores = await RulesOnlyBaseline().score(split.test)
    clustering_scores = ClusteringOnlyBaseline().score(split.test)
    graph_scores = GraphOnlyBaseline().score(split.test)

    tabular = TabularOnlyBaseline(seed=seed)
    await tabular.fit(split.train)
    tabular_scores = await tabular.score(split.test)

    graph_ml = GraphMLBaseline(seed=seed)
    graph_ml.fit(split.train)
    graph_ml_scores = graph_ml.score(split.test)

    hybrid = HybridRingBaseline(seed=seed)
    await hybrid.fit(split.train)
    hybrid_scores = await hybrid.score(split.test)
    routed_moe = RoutedMoEBaseline(seed=seed)
    await routed_moe.fit(split.train)
    routed_scores, assessments, routing_summary = await routed_moe.score(split.test)
    all_scores = {
        "rules_only": rules_scores, "clustering_only": clustering_scores,
        "graph_only": graph_scores, "tabular_only": tabular_scores,
        "graph_ml": graph_ml_scores, "hybrid": hybrid_scores,
        "routed_moe": routed_scores,
    }
    missing = {item.example_id for item in split.test if
               not item.application.device_fingerprint or not item.application.ip_address}
    candidates = routed_moe.builder.build({item.example_id: item.application for item in split.test}) if review_only else ()
    routed_review = {member for candidate in candidates
                     if assessments[candidate.candidate_id].action is CandidateAction.HUMAN_REVIEW
                     for member in candidate.member_ids}

    def measure(name, subset):
        ids = {item.example_id for item in subset}
        extra = missing | (routed_review if name == "routed_moe" else set())
        return evaluate_scores(subset, {key: all_scores[name][key] for key in ids},
                               costs=costs, review_only=review_only,
                               additional_review_ids=extra & ids if review_only else frozenset())

    reports = {name: measure(name, split.test) for name in all_scores}
    cohort_reports = {
        cohort: {name: measure(name, [item for item in split.test if item.cohort.value == cohort])
                 for name in all_scores}
        for cohort in sorted({item.cohort.value for item in split.test})
    }
    return BaselineEvaluation(
        split=split,
        manifest=EvaluationManifest(
            dataset_version=dataset_version,
            feature_schema_version=FEATURE_SCHEMA_VERSION,
            routing_policy_version=ROUTING_POLICY_VERSION,
            seed=seed,
            test_fraction=test_fraction,
            train_ids=tuple(sorted(item.example_id for item in split.train)),
            test_ids=tuple(sorted(item.example_id for item in split.test)),
            model_names=tuple(reports),
            false_review_cost=costs.false_review,
            false_reject_cost=costs.false_reject,
        ),
        reports=reports,
        routing_summary=routing_summary,
        cohort_reports=cohort_reports,
    )
