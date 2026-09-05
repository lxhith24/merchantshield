"""The allow-listed, read-only evidence tools available to the investigator.

Every tool validates candidate/member scope, returns a JSON-safe structured
object, and discloses only redacted values. There is deliberately no
filesystem, network, SQL, shell, MCP or unrestricted retrieval tool here, and
no tool can mutate a merchant decision.
"""
from __future__ import annotations

import hashlib
from datetime import datetime, timezone
from typing import Dict, List, Mapping, Optional, Sequence, Tuple

from ..agent.contracts import CandidateAssessment
from ..analysis.application_clustering import (
    ATTRIBUTE_LABEL,
    ATTRIBUTE_SEVERITY,
    ApplicationClusterer,
)
from ..analysis.evidence_graph import RingCandidate
from ..models.merchant import MerchantApplication
from .memory import CaseMemory

TOOL_NAMES: Tuple[str, ...] = (
    "get_candidate_summary",
    "get_relationship_evidence",
    "get_member_profile",
    "compare_submission_timeline",
    "get_prior_review_history",
    "inspect_document_consistency",
)

TOOL_DESCRIPTIONS: Mapping[str, str] = {
    "get_candidate_summary": (
        "Read component topology, the primary graph score, component size and "
        "per-expert status. Arguments: candidate_id."
    ),
    "get_relationship_evidence": (
        "List redacted shared-attribute records and their evidence IDs. "
        "Arguments: candidate_id, optional attribute, optional min_severity."
    ),
    "get_member_profile": (
        "Read one redacted merchant profile (hashed identifiers only). "
        "Arguments: candidate_id, member_id."
    ),
    "compare_submission_timeline": (
        "Distinguish a bulk submission burst from long-term shared "
        "infrastructure. Arguments: candidate_id."
    ),
    "get_prior_review_history": (
        "Read resolved human outcomes for the hashed infrastructure behind "
        "given evidence IDs. Arguments: candidate_id, evidence_ids."
    ),
    "inspect_document_consistency": (
        "Read document-consistency status for one member, or an explicit "
        "unavailable result. Arguments: candidate_id, member_id."
    ),
}


class ToolError(Exception):
    """A scope or argument violation. Never silently treated as clean evidence."""

    def __init__(self, code: str, message: str) -> None:
        super().__init__(message)
        self.code = code
        self.message = message


class EvidenceCatalogue:
    """Every evidence ID that genuinely belongs to this case.

    A citation outside this catalogue is a fabrication. A citation inside it
    that was never disclosed by a tool call is unauthorized: the investigator
    could not have read it.
    """

    def __init__(self, records: Mapping[str, Mapping[str, object]]) -> None:
        self._records = dict(records)

    def __contains__(self, evidence_id: object) -> bool:
        return evidence_id in self._records

    def __len__(self) -> int:
        return len(self._records)

    @property
    def evidence_ids(self) -> Tuple[str, ...]:
        return tuple(sorted(self._records))

    def get(self, evidence_id: str) -> Optional[Mapping[str, object]]:
        return self._records.get(evidence_id)

    def infrastructure_hashes(self, evidence_ids: Sequence[str]) -> Tuple[str, ...]:
        hashes = []
        for evidence_id in evidence_ids:
            record = self._records.get(evidence_id)
            if record and record.get("shared_value_hash"):
                hashes.append(str(record["shared_value_hash"]))
        return tuple(dict.fromkeys(hashes))


def build_evidence_catalogue(
    candidate: RingCandidate, assessment: CandidateAssessment
) -> EvidenceCatalogue:
    """Union of graph edges and every expert's emitted evidence for this case."""
    records: Dict[str, Mapping[str, object]] = {}
    for edge in candidate.evidence:
        records[edge.evidence_id] = {
            "evidence_id": edge.evidence_id,
            "attribute": edge.attribute,
            "label": ATTRIBUTE_LABEL.get(edge.attribute, edge.attribute),
            "left_id": edge.left_id,
            "right_id": edge.right_id,
            "severity": edge.severity,
            "effective_weight": edge.effective_weight,
            "attribute_frequency": edge.attribute_frequency,
            "shared_value_hash": edge.shared_value_hash,
            "explanation": edge.explanation,
            "source": "evidence_graph",
        }
    for result in assessment.expert_results:
        for item in result.evidence:
            records.setdefault(
                item.evidence_id,
                {
                    "evidence_id": item.evidence_id,
                    "attribute": item.evidence_type,
                    "label": item.evidence_type.replace("_", " "),
                    "severity": item.severity,
                    "effective_weight": item.severity,
                    "explanation": item.summary,
                    "source": result.expert_name,
                    "shared_value_hash": None,
                },
            )
    return EvidenceCatalogue(records)


class InvestigationToolbox:
    """Bound to exactly one candidate; every call is scope-checked."""

    def __init__(
        self,
        *,
        candidate: RingCandidate,
        applications: Mapping[str, MerchantApplication],
        assessment: CandidateAssessment,
        catalogue: Optional[EvidenceCatalogue] = None,
        memory: Optional[CaseMemory] = None,
        supplied_information: Sequence[Mapping[str, str]] = (),
    ) -> None:
        missing = set(candidate.member_ids) - set(applications)
        if missing:
            raise ValueError(f"toolbox applications are missing: {sorted(missing)}")
        self.candidate = candidate
        self.applications = applications
        self.assessment = assessment
        self.catalogue = catalogue or build_evidence_catalogue(candidate, assessment)
        self.memory = memory or CaseMemory()
        self.supplied_information = tuple(supplied_information)
        self._clusterer = ApplicationClusterer()

    # -- scope guards ----------------------------------------------------

    def _check_candidate(self, candidate_id: str) -> None:
        if candidate_id != self.candidate.candidate_id:
            raise ToolError(
                "CANDIDATE_OUT_OF_SCOPE",
                "This investigation may only read its own candidate.",
            )

    def _check_member(self, member_id: str) -> None:
        if member_id not in self.candidate.member_ids:
            raise ToolError(
                "MEMBER_OUT_OF_SCOPE",
                "Requested member is not part of this candidate component.",
            )

    # -- tools -----------------------------------------------------------

    def get_candidate_summary(self, candidate_id: str) -> Mapping[str, object]:
        self._check_candidate(candidate_id)
        features = self.candidate.features
        return {
            "candidate_id": self.candidate.candidate_id,
            "component_size": len(self.candidate.member_ids),
            "member_ids": list(self.candidate.member_ids),
            "linked_pair_count": len(self.candidate.pair_strengths),
            "density": features.density,
            "max_pair_strength": features.max_pair_strength,
            "corroborated_pair_ratio": features.corroborated_pair_ratio,
            "distinct_attribute_count": len(
                {edge.attribute for edge in self.candidate.evidence}
            ),
            "primary_risk_score": self.assessment.primary_risk_score,
            "policy_action": self.assessment.action.value,
            "reason_codes": list(self.assessment.reason_codes),
            "expert_status": {
                result.expert_name: {
                    "status": result.status.value,
                    "confidence": result.confidence,
                    "error_code": result.error_code,
                }
                for result in self.assessment.expert_results
            },
            "supplied_information_count": len(self.supplied_information),
        }

    def get_relationship_evidence(
        self,
        candidate_id: str,
        *,
        attribute: Optional[str] = None,
        min_severity: float = 0.0,
        limit: int = 25,
    ) -> Mapping[str, object]:
        self._check_candidate(candidate_id)
        if attribute is not None and attribute not in ATTRIBUTE_SEVERITY:
            raise ToolError(
                "UNKNOWN_ATTRIBUTE",
                f"Unknown attribute filter; valid values are {sorted(ATTRIBUTE_SEVERITY)}.",
            )
        if not 0.0 <= float(min_severity) <= 1.0:
            raise ToolError("INVALID_FILTER", "min_severity must be between 0 and 1.")
        selected = [
            edge
            for edge in self.candidate.evidence
            if (attribute is None or edge.attribute == attribute)
            and edge.severity >= float(min_severity)
        ]
        selected.sort(key=lambda edge: (-edge.effective_weight, edge.evidence_id))
        records = [
            {
                "evidence_id": edge.evidence_id,
                "attribute": edge.attribute,
                "label": ATTRIBUTE_LABEL.get(edge.attribute, edge.attribute),
                "between": [edge.left_id, edge.right_id],
                "severity": edge.severity,
                "effective_weight": edge.effective_weight,
                "attribute_frequency": edge.attribute_frequency,
                "shared_value_hash": edge.shared_value_hash,
                "explanation": edge.explanation,
            }
            for edge in selected[: max(1, int(limit))]
        ]
        return {
            "candidate_id": self.candidate.candidate_id,
            "filter": {"attribute": attribute, "min_severity": float(min_severity)},
            "total_matching": len(selected),
            "returned": len(records),
            "evidence": records,
        }

    def get_member_profile(
        self, candidate_id: str, member_id: str
    ) -> Mapping[str, object]:
        self._check_candidate(candidate_id)
        self._check_member(member_id)
        application = self.applications[member_id]
        hashes = attribute_hashes(application)
        return {
            "candidate_id": self.candidate.candidate_id,
            "member_id": member_id,
            "business_type": application.business_type.value,
            "high_risk_category": application.is_high_risk_category,
            "pan_holder_type": application.pan_state_independent_holder_type,
            "has_gst": application.gst_number is not None,
            "has_website": bool(application.website_url),
            "submitted_at": _as_aware(application.submitted_at).isoformat(),
            "hashed_attributes": dict(hashes),
            "shared_with_component": sorted(
                attribute
                for attribute, value in hashes.items()
                if any(
                    edge.shared_value_hash == value
                    for edge in self.candidate.evidence
                    if member_id in (edge.left_id, edge.right_id)
                )
            ),
            "redaction_note": (
                "Raw PAN, bank account, phone, email and address are never "
                "disclosed to the investigator."
            ),
        }

    def compare_submission_timeline(self, candidate_id: str) -> Mapping[str, object]:
        self._check_candidate(candidate_id)
        ordered = sorted(
            (
                (_as_aware(self.applications[member_id].submitted_at), member_id)
                for member_id in self.candidate.member_ids
            ),
            key=lambda item: (item[0], item[1]),
        )
        timestamps = [item[0] for item in ordered]
        span_hours = (
            (timestamps[-1] - timestamps[0]).total_seconds() / 3600.0
            if len(timestamps) > 1
            else 0.0
        )
        largest_burst = _largest_burst(timestamps, window_hours=24.0)
        return {
            "candidate_id": self.candidate.candidate_id,
            "submission_count": len(ordered),
            "span_hours": round(span_hours, 3),
            "span_days": round(span_hours / 24.0, 3),
            "largest_burst_within_24h": largest_burst,
            "ordering": [
                {"member_id": member_id, "submitted_at": moment.isoformat()}
                for moment, member_id in ordered
            ],
            "interpretation_note": (
                "A tight burst suggests coordinated batch submission; a long "
                "span is more consistent with durable shared infrastructure."
            ),
        }

    def get_prior_review_history(
        self, candidate_id: str, evidence_ids: Sequence[str]
    ) -> Mapping[str, object]:
        self._check_candidate(candidate_id)
        requested = tuple(dict.fromkeys(evidence_ids or ()))
        if not requested:
            raise ToolError("INVALID_FILTER", "evidence_ids must not be empty.")
        unknown = [item for item in requested if item not in self.catalogue]
        known = [item for item in requested if item in self.catalogue]
        hashes = self.catalogue.infrastructure_hashes(known)
        records = self.memory.lookup(hashes)
        return {
            "candidate_id": self.candidate.candidate_id,
            "requested_evidence_ids": list(requested),
            "unknown_evidence_ids": unknown,
            "infrastructure_hash_count": len(hashes),
            "outcome_counts": dict(self.memory.summarize(hashes)),
            "records": [record.to_dict() for record in records],
            "note": (
                "Prior outcomes are structured human decisions. They may inform "
                "the narrative but never change the graph risk score."
            ),
        }

    def inspect_document_consistency(
        self, candidate_id: str, member_id: str
    ) -> Mapping[str, object]:
        self._check_candidate(candidate_id)
        self._check_member(member_id)
        # Phase 3.5 has not generated the synthetic document corpus. Unavailable
        # verification is reported as unknown; it is never reported as clean.
        return {
            "candidate_id": self.candidate.candidate_id,
            "member_id": member_id,
            "status": "verification_unavailable",
            "treated_as": "unknown",
            "reason": (
                "The Phase 3.5 synthetic document corpus is not generated, so "
                "no document consistency evidence exists for this member."
            ),
        }

    # -- dispatch --------------------------------------------------------

    def call(self, tool_name: str, arguments: Mapping[str, object]) -> Mapping[str, object]:
        """Dispatch by name, rejecting anything outside the allow list."""
        if tool_name not in TOOL_NAMES:
            raise ToolError(
                "TOOL_NOT_ALLOWED",
                f"'{tool_name}' is not an allow-listed read-only evidence tool.",
            )
        candidate_id = str(arguments.get("candidate_id", self.candidate.candidate_id))
        if tool_name == "get_candidate_summary":
            return self.get_candidate_summary(candidate_id)
        if tool_name == "get_relationship_evidence":
            raw_severity = arguments.get("min_severity", 0.0)
            try:
                min_severity = float(raw_severity if raw_severity is not None else 0.0)
            except (TypeError, ValueError):
                raise ToolError("INVALID_FILTER", "min_severity must be numeric.")
            attribute = arguments.get("attribute")
            return self.get_relationship_evidence(
                candidate_id,
                attribute=str(attribute) if attribute else None,
                min_severity=min_severity,
            )
        if tool_name == "get_member_profile":
            member_id = arguments.get("member_id")
            if not member_id:
                raise ToolError("INVALID_FILTER", "member_id is required.")
            return self.get_member_profile(candidate_id, str(member_id))
        if tool_name == "compare_submission_timeline":
            return self.compare_submission_timeline(candidate_id)
        if tool_name == "get_prior_review_history":
            raw_ids = arguments.get("evidence_ids") or ()
            if isinstance(raw_ids, str):
                raw_ids = [raw_ids]
            return self.get_prior_review_history(
                candidate_id, [str(item) for item in raw_ids]
            )
        member_id = arguments.get("member_id")
        if not member_id:
            raise ToolError("INVALID_FILTER", "member_id is required.")
        return self.inspect_document_consistency(candidate_id, str(member_id))


def disclosed_evidence_ids(payload: object) -> Tuple[str, ...]:
    """Collect every evidence ID a tool result actually showed the investigator."""
    found: List[str] = []

    def walk(node: object, key: Optional[str] = None) -> None:
        if isinstance(node, Mapping):
            for child_key, child in node.items():
                walk(child, str(child_key))
        elif isinstance(node, (list, tuple)):
            for child in node:
                walk(child, key)
        elif isinstance(node, str) and key in {"evidence_id", "evidence_ids"}:
            found.append(node)

    walk(payload)
    return tuple(dict.fromkeys(found))


def attribute_hashes(application: MerchantApplication) -> Mapping[str, str]:
    """Hash an application's linkage keys with the evidence graph's own scheme."""
    keys = ApplicationClusterer().extract_keys(application)
    return {
        attribute: hashlib.sha256(
            f"merchantshield:{attribute}:{value}".encode("utf-8")
        ).hexdigest()[:16]
        for attribute, value in sorted(keys.items())
    }


def _largest_burst(timestamps: Sequence[datetime], *, window_hours: float) -> int:
    if not timestamps:
        return 0
    ordered = sorted(timestamps)
    window = window_hours * 3600.0
    best = 1
    start = 0
    for end in range(len(ordered)):
        while (ordered[end] - ordered[start]).total_seconds() > window:
            start += 1
        best = max(best, end - start + 1)
    return best


def _as_aware(value: datetime) -> datetime:
    if value.tzinfo is None:
        return value.replace(tzinfo=timezone.utc)
    return value.astimezone(timezone.utc)
