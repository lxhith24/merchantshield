"""Dependency-free classification, ring, workflow and cost metrics."""
from __future__ import annotations

from dataclasses import asdict, dataclass
import math
from typing import Dict, Sequence, AbstractSet

from .dataset import SyntheticExample


@dataclass(frozen=True)
class CostConfig:
    false_review: float = 1.0
    false_reject: float = 5.0

    def __post_init__(self) -> None:
        if any(not math.isfinite(value) or value < 0 for value in (self.false_review, self.false_reject)):
            raise ValueError("false-positive costs must be finite and non-negative")


@dataclass(frozen=True)
class EvaluationMetrics:
    sample_count: int
    positive_count: int
    precision: float
    recall: float
    f1: float
    pr_auc: float
    false_positive_rate: float
    ring_recall: float
    manual_review_rate: float
    false_positive_cost: float
    false_positive_count: int

    def to_dict(self) -> dict:
        return asdict(self)


def evaluate_scores(
    examples: Sequence[SyntheticExample],
    scores: Dict[str, float],
    *,
    auto_approve_threshold: float = 0.25,
    auto_reject_threshold: float = 0.75,
    costs: CostConfig = CostConfig(),
    review_only: bool = False,
    additional_review_ids: AbstractSet[str] = frozenset(),
) -> EvaluationMetrics:
    """Evaluate screening scores.

    Review and reject decisions count as positive fraud-screening flags for
    precision/recall. ``manual_review_rate`` counts only the middle band.
    ``false_positive_cost`` distinguishes unnecessary review from wrongful
    auto-rejection using ``CostConfig``.

    Expanded benchmarks use ``review_only=True``: ALL screening flags are
    reviewed, never auto-rejected. Extra review triggers (evidence gaps or
    router disagreement) affect workload/cost, not score precision/recall.
    The default preserves the historical demo's hypothetical cost comparison.
    """
    if not 0 <= auto_approve_threshold < auto_reject_threshold <= 1:
        raise ValueError("thresholds must satisfy 0 <= approve < reject <= 1")
    if not examples:
        raise ValueError("examples must not be empty")
    expected = {example.example_id for example in examples}
    if not additional_review_ids <= expected:
        raise ValueError("additional review IDs must belong to examples")
    if additional_review_ids and not review_only:
        raise ValueError("additional review IDs require review_only")
    if set(scores) != expected:
        missing = sorted(expected - set(scores))
        extra = sorted(set(scores) - expected)
        raise ValueError(f"scores must match examples; missing={missing}, extra={extra}")
    if any(not 0.0 <= score <= 1.0 for score in scores.values()):
        raise ValueError("scores must be between 0 and 1")

    labels = {example.example_id: example.is_shell for example in examples}
    flagged = {item for item, score in scores.items() if score > auto_approve_threshold}
    actual = {item for item, label in labels.items() if label}
    negatives = expected - actual
    true_positive = len(flagged & actual)
    false_positive_ids = flagged & negatives

    precision = _divide(true_positive, len(flagged))
    recall = _divide(true_positive, len(actual))
    false_positive_rate = _divide(len(false_positive_ids), len(negatives))
    review_ids = {
        item for item, score in scores.items()
        if auto_approve_threshold < score < auto_reject_threshold
    }
    reject_ids = {item for item, score in scores.items() if score >= auto_reject_threshold}
    if review_only:
        review_ids = flagged | set(additional_review_ids)
        reject_ids = set()
    false_positive_cost = (
        len(review_ids & negatives) * costs.false_review
        + len(reject_ids & negatives) * costs.false_reject
    )

    rings: Dict[str, set[str]] = {}
    for example in examples:
        if example.is_shell and example.ring_id:
            rings.setdefault(example.ring_id, set()).add(example.example_id)
    detected_rings = sum(bool(members & flagged) for members in rings.values())

    return EvaluationMetrics(
        sample_count=len(examples),
        positive_count=len(actual),
        precision=round(precision, 6),
        recall=round(recall, 6),
        f1=round(_divide(2 * precision * recall, precision + recall), 6),
        pr_auc=round(_average_precision(examples, scores), 6),
        false_positive_rate=round(false_positive_rate, 6),
        ring_recall=round(_divide(detected_rings, len(rings)), 6),
        manual_review_rate=round(len(review_ids) / len(examples), 6),
        false_positive_cost=round(false_positive_cost, 6),
        false_positive_count=len(false_positive_ids),
    )


def _average_precision(
    examples: Sequence[SyntheticExample], scores: Dict[str, float]
) -> float:
    positives = sum(example.is_shell for example in examples)
    if positives == 0:
        return 0.0

    # Group equal scores so results do not depend on example ordering.
    ranked: Dict[float, list[bool]] = {}
    for example in examples:
        ranked.setdefault(scores[example.example_id], []).append(example.is_shell)

    true_positive = 0
    predicted = 0
    previous_recall = 0.0
    area = 0.0
    for score in sorted(ranked, reverse=True):
        labels = ranked[score]
        true_positive += sum(labels)
        predicted += len(labels)
        recall = true_positive / positives
        precision = true_positive / predicted
        area += (recall - previous_recall) * precision
        previous_recall = recall
    return area


def _divide(numerator: int, denominator: int) -> float:
    return numerator / denominator if denominator else 0.0
