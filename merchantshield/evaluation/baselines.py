"""Rules-only and graph-linkage-only evaluation baselines."""
from __future__ import annotations

from typing import Dict, List, Sequence

from ..analysis.application_clustering import (
    ATTRIBUTE_SEVERITY,
    CRITICAL_SIZE,
    HIGH_RISK_SIZE,
    SUSPICIOUS_SIZE,
    ApplicationClusterer,
)
from ..analysis.synthetic_identity import SyntheticIdentityDetector
from .dataset import SyntheticExample


class RulesOnlyBaseline:
    name = "rules_only"

    def __init__(self) -> None:
        self.detector = SyntheticIdentityDetector()

    async def score(self, examples: Sequence[SyntheticExample]) -> Dict[str, float]:
        scores: Dict[str, float] = {}
        for example in examples:
            result = await self.detector.analyze(example.application)
            scores[example.example_id] = float(result["synthetic_score"])
        return scores


class ClusteringOnlyBaseline:
    """Batch graph baseline with no identity labels or prior decisions.

    Unlike the online clusterer, this held-out evaluator sees the complete test
    batch symmetrically. It therefore measures graph signal quality without an
    arbitrary submission-order advantage.
    """

    name = "clustering_only"

    def __init__(self) -> None:
        self.clusterer = ApplicationClusterer()

    def score(self, examples: Sequence[SyntheticExample]) -> Dict[str, float]:
        keys = {
            example.example_id: self.clusterer.extract_keys(example.application)
            for example in examples
        }
        index: Dict[tuple[str, str], List[str]] = {}
        for example_id, app_keys in keys.items():
            for attr, value in app_keys.items():
                index.setdefault((attr, value), []).append(example_id)

        scores: Dict[str, float] = {}
        for example in examples:
            matches: Dict[str, int] = {}
            related: set[str] = set()
            for attr, value in keys[example.example_id].items():
                peers = set(index[(attr, value)]) - {example.example_id}
                if peers:
                    matches[attr] = len(peers)
                    related.update(peers)
            scores[example.example_id] = self._score(matches, len(related))
        return scores

    @staticmethod
    def _score(matches: Dict[str, int], related_count: int) -> float:
        if not matches:
            return 0.0
        signals: List[tuple[float, float]] = []
        for attr, count in matches.items():
            severity = ATTRIBUTE_SEVERITY.get(attr, 0.3)
            value = min(1.0, severity * (1 + 0.12 * (count - 1)))
            weight = 1.4 if severity >= 0.7 else 1.0
            signals.append((value, weight))

        size = related_count + 1
        if size >= CRITICAL_SIZE:
            signals.append((0.92, 1.6))
        elif size >= HIGH_RISK_SIZE:
            signals.append((0.74, 1.3))
        elif size >= SUSPICIOUS_SIZE:
            signals.append((0.52, 1.1))

        weighted = sum(value * weight for value, weight in signals) / sum(
            weight for _, weight in signals
        )
        corroboration = min(0.15, 0.05 * max(0, len(matches) - 1))
        return round(min(1.0, weighted + corroboration), 4)
