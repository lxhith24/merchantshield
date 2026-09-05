"""Bounded, label-blind wider relationship context for difficult ring cases."""
from dataclasses import dataclass
import math

import numpy as np
from sklearn.ensemble import GradientBoostingClassifier

from ..analysis.application_clustering import ApplicationClusterer, ATTRIBUTE_SEVERITY
from .performance import FeatureBatch, MerchantPeerModel, extract_features, screening_decision

VERSION = "relationship-context-v2"
MAX_CONTEXT = 50


@dataclass
class RelationshipBatch(FeatureBatch):
    context_sizes: np.ndarray
    context_capped_ids: set[str]
    prior_features: FeatureBatch


async def extract_relationship_features(applications):
    base = await extract_features(applications)
    clusterer = ApplicationClusterer()
    keys = {key: clusterer.extract_keys(applications[key]) for key in base.ids}
    inverted = {}
    for key, values in keys.items():
        for attribute, value in values.items():
            inverted.setdefault((attribute, value), set()).add(key)
    adjacency = {key: set() for key in base.ids}
    for members in inverted.values():
        if 1 < len(members) <= MAX_CONTEXT:
            for key in members:
                adjacency[key].update(members - {key})
    index = {key: position for position, key in enumerate(base.ids)}
    contexts, capped = {}, set()
    for root in base.ids:
        if root in contexts:
            continue
        found, pending = {root}, [root]
        while pending and len(found) <= MAX_CONTEXT:
            fresh = adjacency[pending.pop()] - found
            found.update(fresh)
            pending.extend(sorted(fresh))
        if len(found) > MAX_CONTEXT:
            # Do not use an arbitrary truncated neighbourhood as clean evidence.
            contexts[root] = {root}
            capped.add(root)
        else:
            for key in found:
                contexts[key] = found
    graph_context, identity_context, sizes = [], [], []
    for key in base.ids:
        members = sorted(contexts[key])
        sizes.append(len(members))
        stamps = sorted(applications[item].submitted_at.timestamp() for item in members)
        gaps = np.diff(stamps) / 86400 if len(stamps) > 1 else np.array([0.])
        mean_gap = float(gaps.mean())
        spans = [math.log1p(float(gaps.min())), math.log1p(float(np.median(gaps))),
                 math.log1p(float(gaps.max())), float(gaps.std()) / (1 + mean_gap)]
        diversity = []
        for attribute in ATTRIBUTE_SEVERITY:
            values = [keys[item][attribute] for item in members if attribute in keys[item]]
            diversity.extend((len(set(values)) / len(members), len(values) / len(members)))
        internal_edges = sum(len(adjacency[item] & set(members)) for item in members) / 2
        possible = len(members) * (len(members) - 1) / 2
        graph_context.append([math.log1p(len(members)), internal_edges / possible if possible else 0.,
                              *spans, *diversity, float(key in capped)])
        peer_matrix = base.identity[[index[item] for item in members]]
        identity_context.append([*peer_matrix.mean(axis=0), *peer_matrix.max(axis=0),
                                 *(peer_matrix > 0).mean(axis=0)])
    return RelationshipBatch(base.ids, np.column_stack([base.graph, graph_context]),
        base.identity, np.column_stack([base.combined, graph_context, identity_context]),
        base.rule_scores, base.missing_ids | capped, np.asarray(sizes), capped, base)


class RelationshipRiskModel(MerchantPeerModel):
    """Specialist using wider context; no declared ring membership at inference."""
    def __init__(self, *, depth=3, leaf=25, group_weight=False, blend=1., seed=20260907):
        if not 0 < blend <= 1:
            raise ValueError("blend must be within (0, 1]")
        super().__init__(depth=depth, leaf=leaf, seed=seed)
        self.hybrid = GradientBoostingClassifier(n_estimators=160, max_depth=depth,
            min_samples_leaf=leaf, learning_rate=.05, random_state=seed)
        self.group_weight = group_weight
        self.blend = blend
        self.prior = MerchantPeerModel(depth=3, leaf=40, seed=seed) if blend < 1 else None

    def fit(self, features, labels):
        if set(labels) != set(features.ids) or len(set(labels.values())) < 2:
            raise ValueError("aligned training labels from both classes required")
        y = [labels[key] for key in features.ids]
        weights = 1 / np.sqrt(features.context_sizes) if self.group_weight else np.ones(len(y))
        self.graph.fit(features.graph, y, logisticregression__sample_weight=weights)
        self.tabular.fit(features.identity, y, sample_weight=weights)
        self.hybrid.fit(features.combined, y, sample_weight=weights)
        self.training_ids = frozenset(features.ids)
        if self.prior is not None:
            self.prior.fit(features.prior_features, labels)
        return self

    def score(self, features, *, system="routed_moe"):
        scores, calls = super().score(features, system=system)
        if self.prior is not None and system in ("hybrid", "routed_moe"):
            old, old_calls = self.prior.score(features.prior_features, system=system)
            scores = {key: round(self.blend * score + (1 - self.blend) * old[key], 6) for key, score in scores.items()}
            calls += old_calls  # prediction evaluations, not unique applications
        return scores, calls

    async def screen_applications(self, applications, *, threshold):
        batch = await extract_relationship_features(applications)
        scores, calls = self.score(batch)
        decisions = {}
        for key in batch.ids:
            decision = screening_decision(scores[key], threshold, missing_evidence=key in batch.missing_ids)
            if key in batch.context_capped_ids:
                decision["reason_codes"].append("RELATIONSHIP_CONTEXT_LIMIT")
            decisions[key] = decision
        return {"applications": decisions, "specialist_prediction_evaluations": calls, "model_version": VERSION}


def relationship_evidence_requests(applications):
    """Read-only review checklist, not verification or an automatic clearance.

    A shared attribute establishes a relationship, not guilt. Only a person
    checking independently supplied evidence can resolve the relationship.
    """
    clusterer = ApplicationClusterer()
    counts = {}
    for app in applications.values():
        for attribute, value in clusterer.extract_keys(app).items():
            counts.setdefault(attribute, {}).setdefault(value, 0)
            counts[attribute][value] += 1
    questions = {
        "bank_account": "Verify settlement-account ownership and authorization for each separate business.",
        "owner_pan": "Verify the declared ownership relationship and authority to represent each business.",
        "device_fingerprint": "Verify whether an authorized accountant, franchise operator or onboarding intermediary used this device.",
        "address_hash": "Verify tenancy or the disclosed shared-office relationship for each business.",
        "ip_address": "Check whether this is an office, public or intermediary network; network sharing alone is not fraud evidence.",
    }
    return [{"attribute": attribute, "request": question, "status": "UNVERIFIED"}
            for attribute, question in questions.items() if any(count > 1 for count in counts.get(attribute, {}).values())]
