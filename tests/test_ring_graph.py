"""The ring drawing: deterministic, privacy-preserving, and honest about links."""
from __future__ import annotations

import base64
import re

from merchantshield.analysis.evidence_graph import EvidenceGraphBuilder
from merchantshield.evaluation.dataset import generate_synthetic_dataset
from merchantshield.ui.ring_graph import attribute_label, render_ring_svg, svg_data_uri


def a_linked_candidate(min_size: int = 3):
    examples = generate_synthetic_dataset()
    applications = {item.example_id: item.application for item in examples}
    candidates = EvidenceGraphBuilder().build(applications)
    candidate = next(
        item
        for item in candidates
        if len(item.member_ids) >= min_size
        and any(edge.attribute == "bank_account" for edge in item.evidence)
    )
    members = [
        {
            "member_id": member_id,
            "business_name": applications[member_id].business_name,
        }
        for member_id in candidate.member_ids
    ]
    return candidate, applications, members


def test_the_same_candidate_always_draws_identically():
    candidate, _, members = a_linked_candidate()
    first = render_ring_svg(candidate.to_dict(), members=members)
    second = render_ring_svg(candidate.to_dict(), members=members)
    assert first == second
    assert first.startswith("<svg")


def test_every_member_and_link_is_drawn():
    candidate, _, members = a_linked_candidate()
    svg = render_ring_svg(candidate.to_dict(), members=members)

    assert svg.count("<circle") == len(candidate.member_ids)
    pairs = {
        tuple(sorted((edge.left_id, edge.right_id))) for edge in candidate.evidence
    }
    assert svg.count("<line") >= len(pairs)
    assert attribute_label("bank_account") in svg


def test_no_raw_identifier_reaches_the_drawing():
    """Only hashed values and business names are disclosed by the API."""
    candidate, applications, members = a_linked_candidate()
    svg = render_ring_svg(candidate.to_dict(), members=members)

    for member_id in candidate.member_ids:
        application = applications[member_id]
        for secret in (
            application.owner_pan,
            application.bank_account,
            application.owner_phone,
            application.owner_email,
        ):
            assert secret not in svg


def test_highlighting_dims_the_links_the_investigator_did_not_cite():
    candidate, _, members = a_linked_candidate()
    cited = [candidate.evidence[0].evidence_id]

    plain = render_ring_svg(candidate.to_dict(), members=members)
    highlighted = render_ring_svg(
        candidate.to_dict(), members=members, highlight_evidence_ids=cited
    )

    assert 'stroke-opacity="0.18"' not in plain
    assert 'stroke-opacity="0.18"' in highlighted
    # The cited link stays at full strength.
    assert 'stroke-opacity="0.95"' in highlighted


def test_a_singleton_says_there_is_nothing_to_corroborate():
    examples = generate_synthetic_dataset()
    applications = {item.example_id: item.application for item in examples}
    candidates = EvidenceGraphBuilder().build(applications)
    singleton = next(item for item in candidates if len(item.member_ids) == 1)

    svg = render_ring_svg(singleton.to_dict())

    assert svg.count("<circle") == 1
    assert "<line" not in svg
    assert "nothing to corroborate" in svg


def test_business_names_are_escaped_not_injected():
    candidate, _, _ = a_linked_candidate()
    hostile = [
        {
            "member_id": candidate.member_ids[0],
            "business_name": "<script>alert(1)</script>",
        }
    ]
    svg = render_ring_svg(candidate.to_dict(), members=hostile)
    assert "<script>" not in svg
    assert "&lt;script&gt;" in svg


def test_empty_candidates_render_a_message_rather_than_failing():
    svg = render_ring_svg({"member_ids": []})
    assert "No members" in svg
    assert re.match(r"^<svg", svg)


def test_svg_data_uri_preserves_the_deterministic_graph():
    svg = '<svg role="img"><text>MS-0001</text></svg>'

    uri = svg_data_uri(svg)

    assert uri.startswith("data:image/svg+xml;base64,")
    encoded = uri.split(",", 1)[1]
    assert base64.b64decode(encoded).decode("utf-8") == svg
