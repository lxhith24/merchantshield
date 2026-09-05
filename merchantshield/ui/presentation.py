"""Plain-language case presentation for the reviewer surface.

These helpers deliberately translate model-shaped payloads into the small set
of facts an operations reviewer needs. They never score or decide a case.
"""
from __future__ import annotations

from datetime import datetime
from typing import Any, Dict, List, Mapping, Sequence

from .ring_graph import attribute_label


ATTRIBUTE_ORDER = {
    "bank_account": 0,
    "owner_pan": 1,
    "device_fingerprint": 2,
    "address_hash": 3,
    "ip_address": 4,
    "phone_series": 5,
    "email_domain": 6,
}


def reviewable_cases(
    cases: Sequence[Mapping[str, Any]], reviewer: str
) -> List[Mapping[str, Any]]:
    """Prioritise available cases without sending someone to another's claim."""
    eligible = [
        row for row in cases
        if row.get("status") in {"pending_review", "claimed", "awaiting_information"}
        and (not row.get("claimed_by") or row.get("claimed_by") == reviewer)
    ]
    # An unavailable score must remain visible and go to review, not sort as clean.
    def priority(row):
        score = (row.get("automated") or {}).get("primary_risk_score")
        return (score is None, float(score) if score is not None else 0)

    return sorted(eligible, key=priority, reverse=True)


def short_merchant_id(member_id: str) -> str:
    """Return a stable display ID without inventing a new identifier."""
    tail = str(member_id).rsplit("-", 1)[-1].upper()
    return f"MS-{tail[:6]}"


def priority_label(case: Mapping[str, Any]) -> str:
    action = str((case.get("automated") or {}).get("action") or "")
    score = (case.get("automated") or {}).get("primary_risk_score")
    if action == "proceed_to_onboarding":
        return "Cleared"
    if score is not None and float(score) >= 0.75:
        return "High priority"
    return "Needs review"


def grouped_signals(case: Mapping[str, Any]) -> List[Dict[str, Any]]:
    """Collapse pairwise graph edges into one practical row per similarity."""
    candidate = case.get("candidate") or {}
    total = len(candidate.get("member_ids") or case.get("members") or ())
    grouped: Dict[str, Dict[str, Any]] = {}
    cited = set(_cited_ids(case))
    for edge in candidate.get("evidence", ()):
        attribute = str(edge.get("attribute", "unknown"))
        row = grouped.setdefault(
            attribute,
            {
                "attribute": attribute,
                "label": attribute_label(attribute),
                "members": set(),
                "pair_count": 0,
                "weight": 0.0,
                "cited": False,
            },
        )
        row["members"].update((str(edge.get("left_id")), str(edge.get("right_id"))))
        row["pair_count"] += 1
        row["weight"] = max(row["weight"], float(edge.get("effective_weight", 0.0)))
        row["cited"] = row["cited"] or edge.get("evidence_id") in cited

    results: List[Dict[str, Any]] = []
    for attribute, row in grouped.items():
        count = len(row.pop("members"))
        row["member_count"] = count
        row["total_members"] = total
        row["coverage"] = f"{count} of {total}" if total else str(count)
        row["sentence"] = f"{row['label']} shared by {row['coverage']} merchants"
        results.append(row)
    return sorted(
        results,
        key=lambda item: (
            ATTRIBUTE_ORDER.get(item["attribute"], 99),
            -item["member_count"],
            -item["weight"],
        ),
    )


def strongest_signal(case: Mapping[str, Any]) -> str:
    rows = grouped_signals(case)
    return rows[0]["label"] if rows else "No shared signal"


def submission_window(members: Sequence[Mapping[str, Any]]) -> str:
    values = []
    for member in members:
        raw = member.get("submitted_at")
        if not raw:
            continue
        try:
            values.append(datetime.fromisoformat(str(raw).replace("Z", "+00:00")))
        except ValueError:
            continue
    if len(values) < 2:
        return "Single submission"
    seconds = max(0.0, (max(values) - min(values)).total_seconds())
    if seconds < 60:
        return "Under 1 minute"
    if seconds < 3600:
        return f"{round(seconds / 60):d} minutes"
    if seconds < 172800:
        return f"{seconds / 3600:.1f} hours"
    return f"{seconds / 86400:.1f} days"


def ring_reason_codes(case: Mapping[str, Any]) -> List[str]:
    attributes = {item["attribute"] for item in grouped_signals(case)}
    reasons: List[str] = []
    if "bank_account" in attributes:
        reasons.append("shared_settlement_account")
    if "owner_pan" in attributes:
        reasons.append("shared_owner_identifier")
    if attributes & {"device_fingerprint", "ip_address"}:
        reasons.append("shared_device_or_network")
    if "address_hash" in attributes:
        reasons.append("shared_registered_address")
    return reasons or ["insufficient_evidence"]


def _cited_ids(case: Mapping[str, Any]) -> Sequence[str]:
    investigation = (case.get("case_state") or {}).get("investigation") or {}
    output = investigation.get("output") or {}
    return tuple(output.get("cited_evidence_ids", ()))
