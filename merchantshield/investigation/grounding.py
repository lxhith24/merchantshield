"""Deterministic citation validation. This is the grounding authority.

An LLM narrative is only as trustworthy as its citations, so validation happens
in plain code with no model involvement. Two distinct failures are separated
because they mean different things:

* an unknown ID is a fabrication -- the evidence does not exist for this case;
* an undisclosed ID is unauthorized -- the evidence is real, but no tool call
  ever showed it to the investigator, so the narrative cannot have read it.

Either failure invalidates the whole narrative and forces human review.
"""
from __future__ import annotations

from typing import Iterable, Sequence, Tuple

from .contracts import GroundingReport, GroundingStatus, InvestigatorOutput
from .tools import EvidenceCatalogue


class CitationValidator:
    """Reject nonexistent or unauthorized evidence IDs before a human sees them."""

    def __init__(self, *, require_disclosure: bool = True) -> None:
        self.require_disclosure = require_disclosure

    def validate(
        self,
        output: InvestigatorOutput,
        *,
        catalogue: EvidenceCatalogue,
        disclosed_evidence_ids: Iterable[str] = (),
    ) -> GroundingReport:
        disclosed = set(disclosed_evidence_ids)
        cited = tuple(dict.fromkeys(output.cited_evidence_ids))

        unknown = tuple(item for item in cited if item not in catalogue)
        undisclosed = tuple(
            item
            for item in cited
            if item in catalogue and self.require_disclosure and item not in disclosed
        )
        validated = tuple(
            item for item in cited if item not in unknown and item not in undisclosed
        )

        if unknown or undisclosed:
            return GroundingReport(
                status=GroundingStatus.INSUFFICIENT_GROUNDING,
                validated_evidence_ids=validated,
                unknown_evidence_ids=unknown,
                undisclosed_evidence_ids=undisclosed,
                error_code=(
                    "HALLUCINATED_EVIDENCE_ID"
                    if unknown
                    else "UNAUTHORIZED_EVIDENCE_ID"
                ),
            )
        return GroundingReport(
            status=GroundingStatus.GROUNDED,
            validated_evidence_ids=validated,
        )


def summarize_grounding_failure(report: GroundingReport) -> str:
    """Reviewer-facing explanation of why a narrative was discarded."""
    parts = []
    if report.unknown_evidence_ids:
        parts.append(
            "cited evidence that does not exist for this case: "
            + ", ".join(report.unknown_evidence_ids)
        )
    if report.undisclosed_evidence_ids:
        parts.append(
            "cited evidence that was never disclosed by a tool call: "
            + ", ".join(report.undisclosed_evidence_ids)
        )
    if not parts:
        return "Citations validated against the authorized evidence catalogue."
    return "Narrative discarded because it " + "; and ".join(parts) + "."
