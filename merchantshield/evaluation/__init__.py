"""Offline evaluation tools for MerchantShield's synthetic benchmark."""

from .baselines import ClusteringOnlyBaseline, RulesOnlyBaseline
from .dataset import (
    Cohort,
    SyntheticExample,
    generate_synthetic_dataset,
    load_jsonl,
    write_jsonl,
)
from .metrics import CostConfig, EvaluationMetrics, evaluate_scores
from .ring_models import (
    GraphMLBaseline,
    GraphOnlyBaseline,
    HybridRingBaseline,
    TabularOnlyBaseline,
)
from .runner import BaselineEvaluation, EvaluationManifest, evaluate_baselines
from .routed_moe import RoutedMoEBaseline, RoutingSummary
from .split import DatasetSplit, connected_component_groups, group_train_test_split

__all__ = [
    "BaselineEvaluation",
    "ClusteringOnlyBaseline",
    "Cohort",
    "CostConfig",
    "DatasetSplit",
    "EvaluationMetrics",
    "EvaluationManifest",
    "GraphMLBaseline",
    "GraphOnlyBaseline",
    "HybridRingBaseline",
    "RulesOnlyBaseline",
    "RoutedMoEBaseline",
    "RoutingSummary",
    "SyntheticExample",
    "TabularOnlyBaseline",
    "connected_component_groups",
    "evaluate_baselines",
    "evaluate_scores",
    "generate_synthetic_dataset",
    "group_train_test_split",
    "load_jsonl",
    "write_jsonl",
]
