"""Weighted evidence graph contracts and hard-negative behaviour."""
from __future__ import annotations

import json

from merchantshield.analysis import EvidenceGraphBuilder
from merchantshield.evaluation import Cohort, generate_synthetic_dataset


def _candidate_containing(candidates, example_id):
    return next(item for item in candidates if example_id in item.member_ids)


def test_graph_is_deterministic_complete_and_privacy_preserving():
    examples = generate_synthetic_dataset(seed=101)
    applications = {row.example_id: row.application for row in examples}
    builder = EvidenceGraphBuilder()

    first = builder.build(applications)
    second = builder.build(applications)
    assert [item.to_dict() for item in first] == [item.to_dict() for item in second]

    all_members = [member for candidate in first for member in candidate.member_ids]
    assert sorted(all_members) == sorted(applications)
    assert len(all_members) == len(set(all_members))

    obvious = next(row for row in examples if row.cohort is Cohort.OBVIOUS_SHELL_RING)
    candidate = _candidate_containing(first, obvious.example_id)
    assert len(candidate.member_ids) >= 3
    assert candidate.evidence
    assert len({edge.evidence_id for edge in candidate.evidence}) == len(candidate.evidence)

    serialized = json.dumps(candidate.to_dict(), sort_keys=True)
    assert obvious.application.bank_account not in serialized
    assert obvious.application.owner_pan not in serialized
    assert all(edge.shared_value_hash for edge in candidate.evidence)


def test_velocity_separates_bulk_rings_from_legitimate_shared_infrastructure():
    examples = generate_synthetic_dataset(seed=202)
    candidates = EvidenceGraphBuilder().build(
        {row.example_id: row.application for row in examples}
    )
    evasive = next(row for row in examples if row.cohort is Cohort.EVASIVE_SHELL_RING)
    hard_negative = next(
        row
        for row in examples
        if row.cohort is Cohort.LEGITIMATE_SHARED_INFRA
        and row.component_id == "legit-shared-03"
    )
    evasive_candidate = _candidate_containing(candidates, evasive.example_id)
    negative_candidate = _candidate_containing(candidates, hard_negative.example_id)

    assert evasive_candidate.features.edge_pair_count > 0
    assert negative_candidate.features.edge_pair_count > 0
    assert evasive_candidate.features.submission_velocity > 0.9
    assert negative_candidate.features.submission_velocity < 0.1


def test_common_consumer_email_domains_do_not_create_graph_components():
    examples = [
        row for row in generate_synthetic_dataset(seed=303)
        if row.cohort is Cohort.LEGITIMATE
    ][:3]
    applications = {
        row.example_id: row.application.model_copy(
            update={"owner_email": f"independent{index}@gmail.com"}
        )
        for index, row in enumerate(examples)
    }
    candidates = EvidenceGraphBuilder().build(applications)
    assert all(len(candidate.member_ids) == 1 for candidate in candidates)
