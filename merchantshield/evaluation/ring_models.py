"""Interpretable graph, tabular, and hybrid merchant-ring baselines."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Mapping, Sequence, Tuple

from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from ..analysis.evidence_graph import EvidenceGraphBuilder, RingCandidate
from ..analysis.synthetic_identity import SyntheticIdentityDetector
from ..models.merchant import MerchantApplication
from .dataset import SyntheticExample


@dataclass(frozen=True)
class ApplicationFeatureRow:
    synthetic_score: float
    signal_count: float
    maximum_signal: float
    mean_signal: float
    high_risk_category: float
    has_gst_number: float
    has_website: float

    NAMES = (
        "synthetic_score",
        "signal_count",
        "maximum_signal",
        "mean_signal",
        "high_risk_category",
        "has_gst_number",
        "has_website",
    )

    def as_vector(self) -> Tuple[float, ...]:
        return (
            self.synthetic_score,
            self.signal_count,
            self.maximum_signal,
            self.mean_signal,
            self.high_risk_category,
            self.has_gst_number,
            self.has_website,
        )


class _LogisticRiskModel:
    def __init__(self, *, seed: int = 20250904) -> None:
        self.pipeline = Pipeline(
            steps=[
                ("scale", StandardScaler()),
                (
                    "model",
                    LogisticRegression(
                        class_weight="balanced",
                        max_iter=1000,
                        random_state=seed,
                        solver="liblinear",
                    ),
                ),
            ]
        )
        self._fitted = False

    def fit(self, vectors: Sequence[Sequence[float]], labels: Sequence[bool]) -> None:
        if not vectors or len(vectors) != len(labels):
            raise ValueError("training vectors and labels must be non-empty and aligned")
        if len(set(labels)) < 2:
            raise ValueError("training data must contain positive and negative labels")
        self.pipeline.fit(vectors, labels)
        self._fitted = True

    def predict(self, vectors: Sequence[Sequence[float]]) -> Sequence[float]:
        if not self._fitted:
            raise RuntimeError("model must be fitted before scoring")
        if not vectors:
            return []
        probabilities = self.pipeline.predict_proba(vectors)
        return [round(float(row[1]), 6) for row in probabilities]


class GraphOnlyBaseline:
    """Deterministic weighted-graph score with no labels or fitted model."""

    name = "graph_only"

    def __init__(self, builder: EvidenceGraphBuilder | None = None) -> None:
        self.builder = builder or EvidenceGraphBuilder()

    def score(self, examples: Sequence[SyntheticExample]) -> Dict[str, float]:
        candidates = self.builder.build(_applications(examples))
        scores: Dict[str, float] = {}
        for candidate in candidates:
            score = self._candidate_score(candidate)
            scores.update({member: score for member in candidate.member_ids})
        return scores

    @staticmethod
    def _candidate_score(candidate: RingCandidate) -> float:
        features = candidate.features
        if features.edge_pair_count == 0:
            return 0.0
        high_value_links = max(
            features.bank_account_ratio,
            features.owner_pan_ratio,
            features.device_fingerprint_ratio,
        )
        score = (
            0.30 * features.max_pair_strength
            + 0.15 * features.mean_pair_strength
            + 0.15 * features.corroborated_pair_ratio
            + 0.10 * features.density
            + 0.15 * features.submission_velocity
            + 0.15 * high_value_links
        )
        return round(max(0.0, min(1.0, score)), 6)


class TabularOnlyBaseline:
    """Application-level ML without cross-application graph features."""

    name = "tabular_only"

    def __init__(self, *, seed: int = 20250904) -> None:
        self.model = _LogisticRiskModel(seed=seed)
        self.detector = SyntheticIdentityDetector()

    async def fit(self, examples: Sequence[SyntheticExample]) -> None:
        rows = await self._feature_rows(examples)
        self.model.fit(
            [rows[item.example_id].as_vector() for item in examples],
            [item.is_shell for item in examples],
        )

    async def score(self, examples: Sequence[SyntheticExample]) -> Dict[str, float]:
        return await self.score_applications(_applications(examples))

    async def score_applications(
        self, applications: Mapping[str, MerchantApplication]
    ) -> Dict[str, float]:
        """Score runtime applications without requiring benchmark labels."""
        rows = await self._application_feature_rows(applications)
        ordered_ids = tuple(applications)
        scores = self.model.predict(
            [rows[item].as_vector() for item in ordered_ids]
        )
        return {item: score for item, score in zip(ordered_ids, scores, strict=True)}

    async def _feature_rows(
        self, examples: Sequence[SyntheticExample]
    ) -> Dict[str, ApplicationFeatureRow]:
        return await self._application_feature_rows(_applications(examples))

    async def _application_feature_rows(
        self, applications: Mapping[str, MerchantApplication]
    ) -> Dict[str, ApplicationFeatureRow]:
        rows: Dict[str, ApplicationFeatureRow] = {}
        for application_id, application in applications.items():
            result = await self.detector.analyze(application)
            signals = list(result["signals"])
            signal_values = [float(signal.value) for signal in signals]
            rows[application_id] = ApplicationFeatureRow(
                synthetic_score=float(result["synthetic_score"]),
                signal_count=float(len(signals)),
                maximum_signal=max(signal_values, default=0.0),
                mean_signal=(sum(signal_values) / len(signal_values))
                if signal_values
                else 0.0,
                high_risk_category=float(application.is_high_risk_category),
                has_gst_number=float(bool(application.gst_number)),
                has_website=float(bool(application.website_url)),
            )
        return rows


class GraphMLBaseline:
    """Logistic regression trained only on cluster-level graph features."""

    name = "graph_ml"

    def __init__(self, *, seed: int = 20250904) -> None:
        self.builder = EvidenceGraphBuilder()
        self.model = _LogisticRiskModel(seed=seed)

    def fit(self, examples: Sequence[SyntheticExample]) -> None:
        candidates = self.builder.build(_applications(examples))
        labels = _labels(examples)
        self.model.fit(
            [candidate.features.as_vector() for candidate in candidates],
            [_candidate_label(candidate, labels) for candidate in candidates],
        )

    def score(self, examples: Sequence[SyntheticExample]) -> Dict[str, float]:
        candidates = self.builder.build(_applications(examples))
        predictions = [self.score_candidate(candidate) for candidate in candidates]
        return _expand_candidate_scores(candidates, predictions)

    def score_candidate(self, candidate: RingCandidate) -> float:
        """Score one already-built runtime candidate without any labels."""
        return float(self.model.predict([candidate.features.as_vector()])[0])


class HybridRingBaseline:
    """Cluster model combining graph and aggregate application-level evidence."""

    name = "hybrid"

    def __init__(self, *, seed: int = 20250904) -> None:
        self.builder = EvidenceGraphBuilder()
        self.model = _LogisticRiskModel(seed=seed)
        self.tabular = TabularOnlyBaseline(seed=seed)

    async def fit(self, examples: Sequence[SyntheticExample]) -> None:
        candidates = self.builder.build(_applications(examples))
        app_rows = await self.tabular._feature_rows(examples)
        labels = _labels(examples)
        self.model.fit(
            [self._candidate_vector(candidate, app_rows) for candidate in candidates],
            [_candidate_label(candidate, labels) for candidate in candidates],
        )

    async def score(self, examples: Sequence[SyntheticExample]) -> Dict[str, float]:
        candidates = self.builder.build(_applications(examples))
        app_rows = await self.tabular._feature_rows(examples)
        predictions = self.model.predict(
            [self._candidate_vector(candidate, app_rows) for candidate in candidates]
        )
        return _expand_candidate_scores(candidates, predictions)

    @staticmethod
    def _candidate_vector(
        candidate: RingCandidate,
        app_rows: Mapping[str, ApplicationFeatureRow],
    ) -> Tuple[float, ...]:
        member_vectors = [app_rows[item].as_vector() for item in candidate.member_ids]
        column_count = len(ApplicationFeatureRow.NAMES)
        means = tuple(
            sum(row[column] for row in member_vectors) / len(member_vectors)
            for column in range(column_count)
        )
        maximums = tuple(
            max(row[column] for row in member_vectors) for column in range(column_count)
        )
        return (*candidate.features.as_vector(), *means, *maximums)


def _applications(
    examples: Sequence[SyntheticExample],
) -> Dict[str, MerchantApplication]:
    return {item.example_id: item.application for item in examples}


def _labels(examples: Sequence[SyntheticExample]) -> Dict[str, bool]:
    return {item.example_id: item.is_shell for item in examples}


def _candidate_label(candidate: RingCandidate, labels: Mapping[str, bool]) -> bool:
    member_labels = {labels[item] for item in candidate.member_ids}
    if len(member_labels) > 1:
        # A candidate can contain innocent and fraudulent merchants in reality.
        # Any positive member makes it a review-worthy ring candidate.
        return True
    return member_labels.pop()


def _expand_candidate_scores(
    candidates: Sequence[RingCandidate], predictions: Sequence[float]
) -> Dict[str, float]:
    scores: Dict[str, float] = {}
    for candidate, score in zip(candidates, predictions, strict=True):
        for member in candidate.member_ids:
            scores[member] = float(score)
    return scores
