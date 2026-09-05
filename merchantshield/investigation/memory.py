"""Structured prior-case memory keyed by hashed shared infrastructure.

This is an exact lookup over resolved human decisions, not free-form LLM recall
and not vector retrieval. Prior outcomes may inform an investigation; they may
never overwrite the graph expert's numerical score.
"""
from __future__ import annotations

from datetime import datetime, timezone
from typing import Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

from enum import Enum

from pydantic import BaseModel, Field


class PriorOutcome(str, Enum):
    """How a human actually resolved an earlier case on the same infrastructure."""

    CONFIRMED_RING = "confirmed_ring"
    LEGITIMATE_ACCOUNTANT = "legitimate_accountant"
    LEGITIMATE_FRANCHISE = "legitimate_franchise"
    LEGITIMATE_COWORKING = "legitimate_coworking"
    LEGITIMATE_FAMILY = "legitimate_family"
    INCONCLUSIVE = "inconclusive"


class PriorCaseRecord(BaseModel):
    """One resolved case, addressed only by hashed infrastructure."""

    infrastructure_hash: str = Field(..., min_length=4, max_length=64)
    attribute: str = Field(..., min_length=2, max_length=80)
    outcome: PriorOutcome
    resolved_at: datetime
    reviewer_reference: str = Field(..., min_length=2, max_length=80)
    note: str = Field("", max_length=400)

    def to_dict(self) -> dict:
        return {
            "infrastructure_hash": self.infrastructure_hash,
            "attribute": self.attribute,
            "outcome": self.outcome.value,
            "resolved_at": self.resolved_at.astimezone(timezone.utc).isoformat(),
            "reviewer_reference": self.reviewer_reference,
            "note": self.note,
        }


class CaseMemory:
    """In-process structured store of prior human outcomes.

    Phase 4 replaces the backing dictionary with append-only persistence; the
    lookup contract used by the investigator stays the same.
    """

    def __init__(self, records: Optional[Iterable[PriorCaseRecord]] = None) -> None:
        self._by_hash: Dict[str, List[PriorCaseRecord]] = {}
        for record in records or ():
            self.record(record)

    def record(self, record: PriorCaseRecord) -> None:
        bucket = self._by_hash.setdefault(record.infrastructure_hash, [])
        bucket.append(record)
        bucket.sort(key=lambda item: (item.resolved_at, item.reviewer_reference))

    def lookup(self, infrastructure_hashes: Sequence[str]) -> Tuple[PriorCaseRecord, ...]:
        """Exact-match lookup; unknown hashes simply return nothing."""
        found: List[PriorCaseRecord] = []
        for value in dict.fromkeys(infrastructure_hashes):
            found.extend(self._by_hash.get(value, ()))
        return tuple(found)

    def summarize(
        self, infrastructure_hashes: Sequence[str]
    ) -> Mapping[str, int]:
        """Outcome counts, so a narrative can cite history without over-reading it."""
        counts: Dict[str, int] = {}
        for record in self.lookup(infrastructure_hashes):
            counts[record.outcome.value] = counts.get(record.outcome.value, 0) + 1
        return dict(sorted(counts.items()))

    def __len__(self) -> int:
        return sum(len(bucket) for bucket in self._by_hash.values())
