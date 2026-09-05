"""Label-blind merchant/peer features and a validation-tuned sparse classifier.

Candidate groups are evidence, not guilt: each merchant receives its own score.
No identifier strings, declared components, ring IDs or cohort labels are inputs.
"""
from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Mapping

import numpy as np
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from ..analysis.application_clustering import ApplicationClusterer, ATTRIBUTE_SEVERITY
from ..analysis.evidence_graph import EvidenceGraphBuilder
from ..analysis.synthetic_identity import SyntheticIdentityDetector
from ..models.merchant import MerchantApplication
from .metrics import CostConfig, evaluate_scores

VERSION = "merchant-peer-sparse-v1"
SIGNALS = (
    "disposable_email_domain", "sequential_numeric_email", "placeholder_email",
    "unaffiliated_free_email", "invalid_mobile_series", "repeated_digit_phone",
    "digit_run_phone", "sequential_phone", "invalid_pan_holder_type",
    "pan_surname_mismatch", "keyboard_pattern_pan", "placeholder_address",
    "missing_pincode", "incomplete_address", "unrecognised_locality",
    "placeholder_business_name", "generic_business_name", "high_risk_category",
    "repeated_digit_account", "sequential_account", "gst_pan_mismatch",
)


@dataclass
class FeatureBatch:
    ids: tuple[str, ...]
    graph: np.ndarray
    identity: np.ndarray
    combined: np.ndarray
    rule_scores: np.ndarray
    missing_ids: set[str]


async def extract_features(applications: Mapping[str, MerchantApplication]) -> FeatureBatch:
    if not applications:
        raise ValueError("applications must not be empty")
    ids = tuple(sorted(applications))
    index = {key: position for position, key in enumerate(ids)}
    detector = SyntheticIdentityDetector()
    identity, risk = [], []
    for key in ids:
        app = applications[key]
        result = await detector.analyze(app)
        signals = {signal.name: signal.value for signal in result["signals"]}
        risk.append(result["synthetic_score"])
        identity.append([*(signals.get(name, 0.) for name in SIGNALS),
                         float(bool(app.website_url)), float(bool(app.gst_number)),
                         float(not app.device_fingerprint), float(not app.ip_address)])
    identity = np.asarray(identity, dtype=float)
    candidates = EvidenceGraphBuilder().build(applications)
    candidate_features = {member: candidate.features.as_vector()
                          for candidate in candidates for member in candidate.member_ids}
    clusterer = ApplicationClusterer()
    keys = {key: clusterer.extract_keys(applications[key]) for key in ids}
    inverted = {}
    for key, values in keys.items():
        for attribute, value in values.items():
            inverted.setdefault((attribute, value), set()).add(key)
    graph, peers = [], []
    for key in ids:
        neighbours = set()
        counts = []
        for attribute in ATTRIBUTE_SEVERITY:
            members = inverted.get((attribute, keys[key].get(attribute)), set())
            # Suppress common public infrastructure; missing is not a link.
            members = members - {key} if len(members) <= 50 else set()
            counts.append(math.log1p(len(members)))
            neighbours.update(members)
        stamps = [applications[item].submitted_at.timestamp() for item in neighbours | {key}]
        span_days = (max(stamps) - min(stamps)) / 86400
        graph.append([*candidate_features[key], *counts, math.log1p(span_days)])
        # Peer evidence survives weak links excluded from candidate construction.
        peer_rows = identity[[index[item] for item in sorted(neighbours)]] if neighbours else np.zeros((1, identity.shape[1]))
        peers.append([*peer_rows.mean(axis=0), *peer_rows.max(axis=0)])
    graph = np.asarray(graph, dtype=float)
    return FeatureBatch(ids, graph, identity, np.column_stack([graph, identity, peers]),
                        np.asarray(risk), {key for key in ids if not applications[key].device_fingerprint or not applications[key].ip_address})


class MerchantPeerModel:
    """Cheap graph expert first; learned combined evidence for uncertain rows.

    Numerical models own every score. This is a system-level sparse ensemble,
    not a neural MoE or an LLM. Fit uses individual labels, not group-wide guilt.
    """
    def __init__(self, *, depth=2, leaf=20, seed=20260905):
        self.graph = make_pipeline(StandardScaler(), LogisticRegression(max_iter=1000, solver="liblinear", random_state=seed))
        self.tabular = GradientBoostingClassifier(n_estimators=120, max_depth=depth, min_samples_leaf=leaf, learning_rate=.05, random_state=seed)
        self.hybrid = GradientBoostingClassifier(n_estimators=120, max_depth=depth, min_samples_leaf=leaf, learning_rate=.05, random_state=seed)
        self.training_ids = frozenset()

    def fit(self, features: FeatureBatch, labels: Mapping[str, bool]):
        if set(labels) != set(features.ids) or len(set(labels.values())) < 2:
            raise ValueError("aligned training labels from both classes required")
        y = [labels[key] for key in features.ids]
        self.graph.fit(features.graph, y)
        self.tabular.fit(features.identity, y)
        self.hybrid.fit(features.combined, y)
        self.training_ids = frozenset(features.ids)
        return self

    def score(self, features: FeatureBatch, *, system="routed_moe"):
        if not self.training_ids:
            raise RuntimeError("model must be fitted before scoring")
        routed = np.zeros(len(features.ids), dtype=bool)
        if system == "graph_ml":
            scores = self.graph.predict_proba(features.graph)[:, 1]
        elif system == "tabular_only":
            scores = self.tabular.predict_proba(features.identity)[:, 1]
        elif system == "hybrid":
            scores = self.hybrid.predict_proba(features.combined)[:, 1]
            routed[:] = True
        elif system == "routed_moe":
            scores = self.graph.predict_proba(features.graph)[:, 1]
            # Fixed gate, not selected on the test: ambiguous graph, material
            # rule disagreement, or incomplete session evidence invokes hybrid.
            routed = ((scores > .08) & (scores < .92)) | (abs(scores - features.rule_scores) > .4)
            routed |= np.array([key in features.missing_ids for key in features.ids])
            if routed.any():
                scores[routed] = self.hybrid.predict_proba(features.combined[routed])[:, 1]
        else:
            raise ValueError(f"unknown system: {system}")
        return {key: round(float(value), 6) for key, value in zip(features.ids, scores)}, int(routed.sum())

    async def screen_applications(self, applications, *, threshold):
        """Reusable inference path: no labels, no automatic rejection, no I/O."""
        batch = await extract_features(applications)
        scores, calls = self.score(batch)
        return {"applications": {
            key: screening_decision(scores[key], threshold, missing_evidence=key in batch.missing_ids)
            for key in batch.ids}, "specialist_applications": calls,
            "model_version": VERSION}


def screening_decision(score, threshold, *, missing_evidence=False):
    if not math.isfinite(threshold) or not 0 < threshold < 1:
        raise ValueError("threshold must be finite and within (0, 1)")
    reasons = []
    if score is None or not math.isfinite(score) or not 0 <= score <= 1:
        score = None
        reasons.append("RISK_UNAVAILABLE")
    elif score > threshold:
        reasons.append("RISK_ABOVE_VALIDATED_THRESHOLD")
    if missing_evidence:
        reasons.append("MISSING_SESSION_EVIDENCE")
    return {"risk_score": score, "threshold": threshold,
            "action": "human_review" if reasons else "proceed_to_onboarding",
            "reason_codes": reasons, "fraud_flag": score is not None and score > threshold}


def measure(rows, scores, threshold, *, false_review_cost=1.):
    missing = {row.example_id for row in rows if not row.application.device_fingerprint or not row.application.ip_address}
    return evaluate_scores(rows, scores, auto_approve_threshold=threshold,
                           auto_reject_threshold=1., review_only=True,
                           additional_review_ids=missing, costs=CostConfig(false_review=false_review_cost))


def select_threshold(rows, scores, *, min_recall=.80, min_ring_recall=.90, false_review_cost=1., missed_fraud_cost=1.):
    """Validation ONLY: minimum review cost subject to recall protections.

    The missed-fraud cost and recall floor are explicit development assumptions,
    not a Razorpay business policy. Never takes test predictions as an input.
    """
    if not 0 < min_recall <= 1 or not 0 < min_ring_recall <= 1:
        raise ValueError("recall floors must be within (0, 1]")
    if not math.isfinite(missed_fraud_cost) or missed_fraud_cost < 0:
        raise ValueError("missed fraud cost must be finite and non-negative")
    options = []
    for threshold in (i / 100 for i in range(1, 96)):
        metrics = measure(rows, scores, threshold, false_review_cost=false_review_cost)
        if metrics.recall >= min_recall and metrics.ring_recall >= min_ring_recall:
            missed = sum(row.is_shell and scores[row.example_id] <= threshold for row in rows)
            objective = metrics.false_positive_cost + missed_fraud_cost * missed
            options.append((objective, metrics.false_positive_rate, -metrics.precision, threshold, metrics))
    if not options:
        raise ValueError("no validation threshold meets recall safeguards")
    objective, _, _, threshold, metrics = min(options, key=lambda item: item[:4])
    return {"threshold": threshold, "objective": objective, "metrics": metrics.to_dict()}
