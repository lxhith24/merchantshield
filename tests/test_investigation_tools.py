"""Read-only evidence tools: scope, redaction, and the evidence catalogue."""
from __future__ import annotations

import json
from datetime import datetime, timezone

import pytest

from merchantshield.investigation import (
    CaseMemory,
    InvestigationToolbox,
    PriorCaseRecord,
    PriorOutcome,
    ToolError,
    build_evidence_catalogue,
    disclosed_evidence_ids,
)
from merchantshield.investigation.tools import TOOL_NAMES
from tests.conftest import make_ring_case


async def _toolbox(**kwargs):
    candidate, applications, assessment = await make_ring_case()
    return (
        InvestigationToolbox(
            candidate=candidate,
            applications=applications,
            assessment=assessment,
            **kwargs,
        ),
        candidate,
        applications,
        assessment,
    )


async def test_catalogue_covers_graph_edges_and_expert_evidence():
    _, candidate, _, assessment = await _toolbox()
    catalogue = build_evidence_catalogue(candidate, assessment)

    for edge in candidate.evidence:
        assert edge.evidence_id in catalogue
    for result in assessment.expert_results:
        for item in result.evidence:
            assert item.evidence_id in catalogue
    assert len(catalogue) >= len(candidate.evidence)


async def test_tools_reject_out_of_scope_candidate_and_member():
    toolbox, candidate, _, _ = await _toolbox()

    with pytest.raises(ToolError) as candidate_error:
        toolbox.get_candidate_summary("candidate-not-mine")
    assert candidate_error.value.code == "CANDIDATE_OUT_OF_SCOPE"

    with pytest.raises(ToolError) as member_error:
        toolbox.get_member_profile(candidate.candidate_id, "syn-9999")
    assert member_error.value.code == "MEMBER_OUT_OF_SCOPE"


async def test_unlisted_tool_is_rejected():
    toolbox, candidate, _, _ = await _toolbox()

    with pytest.raises(ToolError) as error:
        toolbox.call("read_file", {"candidate_id": candidate.candidate_id})
    assert error.value.code == "TOOL_NOT_ALLOWED"
    assert set(TOOL_NAMES) == {
        "get_candidate_summary",
        "get_relationship_evidence",
        "get_member_profile",
        "compare_submission_timeline",
        "get_prior_review_history",
        "inspect_document_consistency",
    }


async def test_member_profile_never_discloses_raw_identifiers():
    toolbox, candidate, applications, _ = await _toolbox()
    member_id = candidate.member_ids[0]
    application = applications[member_id]

    profile = toolbox.get_member_profile(candidate.candidate_id, member_id)
    serialized = json.dumps(profile)

    for secret in (
        application.owner_pan,
        application.bank_account,
        str(application.owner_email),
        application.owner_phone,
        application.registered_address,
        application.owner_name,
    ):
        assert secret not in serialized
    assert profile["hashed_attributes"]
    assert profile["business_type"] == application.business_type.value


async def test_relationship_evidence_filters_and_stays_within_catalogue():
    toolbox, candidate, _, assessment = await _toolbox()
    catalogue = build_evidence_catalogue(candidate, assessment)

    everything = toolbox.get_relationship_evidence(candidate.candidate_id)
    assert everything["evidence"]
    assert all(item["evidence_id"] in catalogue for item in everything["evidence"])

    filtered = toolbox.get_relationship_evidence(
        candidate.candidate_id, min_severity=0.80
    )
    assert all(item["severity"] >= 0.80 for item in filtered["evidence"])
    assert filtered["total_matching"] <= everything["total_matching"]

    with pytest.raises(ToolError) as error:
        toolbox.get_relationship_evidence(candidate.candidate_id, attribute="nonsense")
    assert error.value.code == "UNKNOWN_ATTRIBUTE"


async def test_timeline_separates_burst_from_long_span():
    toolbox, candidate, _, _ = await _toolbox()

    timeline = toolbox.compare_submission_timeline(candidate.candidate_id)

    assert timeline["submission_count"] == len(candidate.member_ids)
    assert timeline["span_hours"] >= 0.0
    assert 1 <= timeline["largest_burst_within_24h"] <= len(candidate.member_ids)
    assert [entry["member_id"] for entry in timeline["ordering"]]


async def test_prior_review_history_reads_structured_human_outcomes():
    candidate, applications, assessment = await make_ring_case()
    catalogue = build_evidence_catalogue(candidate, assessment)
    edge = candidate.evidence[0]
    memory = CaseMemory(
        [
            PriorCaseRecord(
                infrastructure_hash=edge.shared_value_hash,
                attribute=edge.attribute,
                outcome=PriorOutcome.LEGITIMATE_COWORKING,
                resolved_at=datetime(2025, 6, 1, tzinfo=timezone.utc),
                reviewer_reference="reviewer-7",
            )
        ]
    )
    toolbox = InvestigationToolbox(
        candidate=candidate,
        applications=applications,
        assessment=assessment,
        catalogue=catalogue,
        memory=memory,
    )

    history = toolbox.get_prior_review_history(
        candidate.candidate_id, [edge.evidence_id, "edge-doesnotexist"]
    )

    assert history["outcome_counts"] == {"legitimate_coworking": 1}
    assert history["unknown_evidence_ids"] == ["edge-doesnotexist"]
    assert history["records"][0]["reviewer_reference"] == "reviewer-7"


async def test_missing_documents_are_unavailable_never_clean():
    toolbox, candidate, _, _ = await _toolbox()

    result = toolbox.inspect_document_consistency(
        candidate.candidate_id, candidate.member_ids[0]
    )

    assert result["status"] == "verification_unavailable"
    assert result["treated_as"] == "unknown"


async def test_disclosed_evidence_ids_are_collected_from_tool_output():
    toolbox, candidate, _, _ = await _toolbox()

    payload = toolbox.get_relationship_evidence(candidate.candidate_id)
    ids = disclosed_evidence_ids(payload)

    assert ids
    assert set(ids) == {item["evidence_id"] for item in payload["evidence"]}
    assert disclosed_evidence_ids({"nothing": "here"}) == ()
