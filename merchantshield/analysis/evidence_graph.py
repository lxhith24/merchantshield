"""Deterministic, privacy-preserving relationship graph for merchant rings.

The graph does not decide that a merchant is fraudulent. It creates candidate
groups and auditable evidence edges for downstream scoring and human review.
Raw shared identifier values are deliberately excluded from graph outputs.
"""
from __future__ import annotations

import hashlib
import math
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from itertools import combinations
from typing import ClassVar, Dict, Iterable, Mapping, Sequence, Tuple

import networkx as nx

from ..models.merchant import MerchantApplication
from .application_clustering import ATTRIBUTE_LABEL, ATTRIBUTE_SEVERITY, ApplicationClusterer


@dataclass(frozen=True)
class EvidenceEdge:
    """One shared-attribute observation between two applications."""

    evidence_id: str
    left_id: str
    right_id: str
    attribute: str
    severity: float
    effective_weight: float
    attribute_frequency: int
    shared_value_hash: str
    explanation: str

    def to_dict(self) -> dict:
        return asdict(self)


@dataclass(frozen=True)
class ClusterFeatures:
    """Small, interpretable feature vector for one candidate component."""

    component_size: float
    edge_pair_count: float
    density: float
    mean_pair_strength: float
    max_pair_strength: float
    strong_evidence_ratio: float
    corroborated_pair_ratio: float
    unique_attribute_ratio: float
    submission_velocity: float
    bank_account_ratio: float
    owner_pan_ratio: float
    device_fingerprint_ratio: float
    address_hash_ratio: float
    ip_address_ratio: float
    phone_series_ratio: float
    email_domain_ratio: float

    NAMES: ClassVar[Tuple[str, ...]] = (
        "component_size",
        "edge_pair_count",
        "density",
        "mean_pair_strength",
        "max_pair_strength",
        "strong_evidence_ratio",
        "corroborated_pair_ratio",
        "unique_attribute_ratio",
        "submission_velocity",
        "bank_account_ratio",
        "owner_pan_ratio",
        "device_fingerprint_ratio",
        "address_hash_ratio",
        "ip_address_ratio",
        "phone_series_ratio",
        "email_domain_ratio",
    )

    def as_vector(self) -> Tuple[float, ...]:
        values = asdict(self)
        return tuple(float(values[name]) for name in self.NAMES)

    def to_dict(self) -> dict:
        return asdict(self)


@dataclass(frozen=True)
class RingCandidate:
    """One connected candidate group, including singleton applications."""

    candidate_id: str
    member_ids: Tuple[str, ...]
    evidence: Tuple[EvidenceEdge, ...]
    pair_strengths: Tuple[Tuple[str, str, float], ...]
    features: ClusterFeatures

    def to_dict(self) -> dict:
        return {
            "candidate_id": self.candidate_id,
            "member_ids": list(self.member_ids),
            "evidence": [edge.to_dict() for edge in self.evidence],
            "pair_strengths": [
                {"left_id": left, "right_id": right, "strength": strength}
                for left, right, strength in self.pair_strengths
            ],
            "features": self.features.to_dict(),
        }


class EvidenceGraphBuilder:
    """Build candidate components from shared onboarding attributes.

    Common, weak attributes receive a frequency discount. This keeps a public
    email domain or shared office IP from becoming decisive merely because it
    connects many otherwise unrelated merchants.
    """

    def __init__(
        self,
        *,
        minimum_evidence_weight: float = 0.10,
        minimum_pair_strength: float = 0.35,
        maximum_attribute_frequency: int = 50,
    ) -> None:
        if not 0.0 <= minimum_evidence_weight <= 1.0:
            raise ValueError("minimum_evidence_weight must be between 0 and 1")
        if not 0.0 <= minimum_pair_strength <= 1.0:
            raise ValueError("minimum_pair_strength must be between 0 and 1")
        if maximum_attribute_frequency < 2:
            raise ValueError("maximum_attribute_frequency must be at least 2")
        self.minimum_evidence_weight = minimum_evidence_weight
        self.minimum_pair_strength = minimum_pair_strength
        self.maximum_attribute_frequency = maximum_attribute_frequency
        self.clusterer = ApplicationClusterer()

    def build(
        self, applications: Mapping[str, MerchantApplication]
    ) -> Tuple[RingCandidate, ...]:
        if not applications:
            return tuple()
        if any(not application_id for application_id in applications):
            raise ValueError("application IDs must be non-empty")

        ordered_ids = tuple(sorted(applications))
        keys = {
            application_id: self.clusterer.extract_keys(applications[application_id])
            for application_id in ordered_ids
        }
        inverted: Dict[tuple[str, str], list[str]] = {}
        for application_id, app_keys in keys.items():
            for attribute, value in app_keys.items():
                inverted.setdefault((attribute, value), []).append(application_id)

        evidence_by_pair: Dict[tuple[str, str], list[EvidenceEdge]] = {}
        for (attribute, raw_value), members in sorted(inverted.items()):
            unique_members = sorted(set(members))
            frequency = len(unique_members)
            if frequency < 2 or frequency > self.maximum_attribute_frequency:
                continue
            severity = float(ATTRIBUTE_SEVERITY.get(attribute, 0.25))
            effective_weight = self._frequency_adjusted_weight(severity, frequency)
            if effective_weight < self.minimum_evidence_weight:
                continue
            value_hash = self._value_hash(attribute, raw_value)
            for left_id, right_id in combinations(unique_members, 2):
                evidence_id = self._evidence_id(
                    left_id, right_id, attribute, value_hash
                )
                label = ATTRIBUTE_LABEL.get(attribute, attribute.replace("_", " "))
                evidence_by_pair.setdefault((left_id, right_id), []).append(
                    EvidenceEdge(
                        evidence_id=evidence_id,
                        left_id=left_id,
                        right_id=right_id,
                        attribute=attribute,
                        severity=round(severity, 6),
                        effective_weight=round(effective_weight, 6),
                        attribute_frequency=frequency,
                        shared_value_hash=value_hash,
                        explanation=f"Applications share {label}.",
                    )
                )

        graph = nx.Graph()
        graph.add_nodes_from(ordered_ids)
        pair_strengths: Dict[tuple[str, str], float] = {}
        for pair, edges in sorted(evidence_by_pair.items()):
            strength = self._noisy_or(edge.effective_weight for edge in edges)
            pair_strengths[pair] = strength
            if strength >= self.minimum_pair_strength:
                graph.add_edge(*pair, strength=strength)

        candidates = []
        for members in sorted(
            (tuple(sorted(component)) for component in nx.connected_components(graph)),
            key=lambda item: item,
        ):
            member_set = set(members)
            component_evidence = tuple(
                sorted(
                    (
                        edge
                        for pair, edges in evidence_by_pair.items()
                        if pair[0] in member_set and pair[1] in member_set
                        for edge in edges
                    ),
                    key=lambda edge: edge.evidence_id,
                )
            )
            component_pairs = tuple(
                (left, right, pair_strengths[(left, right)])
                for left, right in sorted(pair_strengths)
                if left in member_set
                and right in member_set
                and pair_strengths[(left, right)] >= self.minimum_pair_strength
            )
            candidates.append(
                RingCandidate(
                    candidate_id=self._candidate_id(members),
                    member_ids=members,
                    evidence=component_evidence,
                    pair_strengths=component_pairs,
                    features=self._features(
                        members,
                        component_evidence,
                        component_pairs,
                        applications,
                    ),
                )
            )
        return tuple(candidates)

    @staticmethod
    def _frequency_adjusted_weight(severity: float, frequency: int) -> float:
        if frequency <= 2:
            return severity
        if severity >= 0.70:
            denominator = math.sqrt(1.0 + 0.20 * (frequency - 2))
        else:
            denominator = math.sqrt(frequency - 1)
        return severity / denominator

    @staticmethod
    def _noisy_or(weights: Iterable[float]) -> float:
        remaining = 1.0
        for weight in weights:
            remaining *= 1.0 - max(0.0, min(1.0, float(weight)))
        return round(1.0 - remaining, 6)

    @staticmethod
    def _value_hash(attribute: str, raw_value: str) -> str:
        return hashlib.sha256(
            f"merchantshield:{attribute}:{raw_value}".encode("utf-8")
        ).hexdigest()[:16]

    @staticmethod
    def _evidence_id(
        left_id: str, right_id: str, attribute: str, value_hash: str
    ) -> str:
        digest = hashlib.sha256(
            f"{left_id}|{right_id}|{attribute}|{value_hash}".encode("utf-8")
        ).hexdigest()[:12]
        return f"edge-{digest}"

    @staticmethod
    def _candidate_id(member_ids: Sequence[str]) -> str:
        digest = hashlib.sha256("|".join(member_ids).encode("utf-8")).hexdigest()[:12]
        return f"candidate-{digest}"

    def _features(
        self,
        members: Sequence[str],
        evidence: Sequence[EvidenceEdge],
        pair_strengths: Sequence[Tuple[str, str, float]],
        applications: Mapping[str, MerchantApplication],
    ) -> ClusterFeatures:
        size = len(members)
        possible_pairs = size * (size - 1) / 2
        pair_count = len(pair_strengths)
        strengths = [strength for _, _, strength in pair_strengths]
        evidence_pairs: Dict[tuple[str, str], set[str]] = {}
        attribute_pairs: Dict[str, set[tuple[str, str]]] = {}
        for edge in evidence:
            pair = (edge.left_id, edge.right_id)
            evidence_pairs.setdefault(pair, set()).add(edge.attribute)
            attribute_pairs.setdefault(edge.attribute, set()).add(pair)

        strong_count = sum(edge.severity >= 0.70 for edge in evidence)
        corroborated = sum(len(attributes) >= 2 for attributes in evidence_pairs.values())
        timestamps = [self._as_aware(applications[item].submitted_at) for item in members]
        span_hours = (
            (max(timestamps) - min(timestamps)).total_seconds() / 3600.0
            if len(timestamps) > 1
            else 0.0
        )
        velocity = 1.0 if len(timestamps) == 1 else 1.0 / (1.0 + span_hours / 24.0)

        def pair_ratio(attribute: str) -> float:
            return len(attribute_pairs.get(attribute, set())) / possible_pairs if possible_pairs else 0.0

        return ClusterFeatures(
            component_size=float(size),
            edge_pair_count=float(pair_count),
            density=round(pair_count / possible_pairs, 6) if possible_pairs else 0.0,
            mean_pair_strength=round(sum(strengths) / len(strengths), 6) if strengths else 0.0,
            max_pair_strength=round(max(strengths), 6) if strengths else 0.0,
            strong_evidence_ratio=round(strong_count / len(evidence), 6) if evidence else 0.0,
            corroborated_pair_ratio=round(corroborated / len(evidence_pairs), 6)
            if evidence_pairs
            else 0.0,
            unique_attribute_ratio=round(
                len(attribute_pairs) / len(ATTRIBUTE_SEVERITY), 6
            ),
            submission_velocity=round(velocity, 6),
            bank_account_ratio=round(pair_ratio("bank_account"), 6),
            owner_pan_ratio=round(pair_ratio("owner_pan"), 6),
            device_fingerprint_ratio=round(pair_ratio("device_fingerprint"), 6),
            address_hash_ratio=round(pair_ratio("address_hash"), 6),
            ip_address_ratio=round(pair_ratio("ip_address"), 6),
            phone_series_ratio=round(pair_ratio("phone_series"), 6),
            email_domain_ratio=round(pair_ratio("email_domain"), 6),
        )

    @staticmethod
    def _as_aware(value: datetime) -> datetime:
        if value.tzinfo is None:
            return value.replace(tzinfo=timezone.utc)
        return value
