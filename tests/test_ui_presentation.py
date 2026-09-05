from merchantshield.ui.presentation import (
    reviewable_cases,
    grouped_signals,
    priority_label,
    ring_reason_codes,
    short_merchant_id,
    submission_window,
)
from merchantshield.ui.streamlit_app import _confusion_counts, reviewer_greeting
from pathlib import Path


def sample_case():
    return {
        "automated": {"action": "human_review", "primary_risk_score": 0.91},
        "candidate": {
            "member_ids": ["syn-0001", "syn-0002", "syn-0003"],
            "evidence": [
                {
                    "left_id": "syn-0001",
                    "right_id": "syn-0002",
                    "attribute": "device_fingerprint",
                    "evidence_id": "edge-one",
                    "effective_weight": 0.72,
                },
                {
                    "left_id": "syn-0002",
                    "right_id": "syn-0003",
                    "attribute": "device_fingerprint",
                    "evidence_id": "edge-two",
                    "effective_weight": 0.70,
                },
                {
                    "left_id": "syn-0001",
                    "right_id": "syn-0003",
                    "attribute": "ip_address",
                    "evidence_id": "edge-three",
                    "effective_weight": 0.20,
                },
            ],
        },
        "case_state": {
            "investigation": {"output": {"cited_evidence_ids": ["edge-one"]}}
        },
    }


def test_pairwise_edges_become_one_row_per_similarity():
    rows = grouped_signals(sample_case())
    assert [row["attribute"] for row in rows] == ["device_fingerprint", "ip_address"]
    assert rows[0]["coverage"] == "3 of 3"
    assert rows[0]["pair_count"] == 2
    assert rows[0]["cited"] is True


def test_reviewer_labels_are_plain_and_stable():
    assert priority_label(sample_case()) == "High priority"
    assert short_merchant_id("syn-0071") == "MS-0071"
    assert ring_reason_codes(sample_case()) == ["shared_device_or_network"]


def test_reviewer_greeting_uses_the_entered_name():
    assert reviewer_greeting("Johan") == "Hey, Johan"
    assert reviewer_greeting("  Johan   Nil  ") == "Hey, Johan Nil"
    assert reviewer_greeting("reviewer_demo") == "Reviewer"
    assert reviewer_greeting("") == "Reviewer"


def test_confusion_counts_make_the_operating_point_concrete():
    counts = _confusion_counts(
        {
            "sample_count": 29,
            "positive_count": 10,
            "recall": 0.6,
            "false_positive_count": 4,
        }
    )
    assert counts == {
        "true positives": 6,
        "false negatives": 4,
        "false positives": 4,
        "true negatives": 15,
    }


def test_submission_window_uses_human_units():
    members = [
        {"submitted_at": "2025-01-01T10:00:00+00:00"},
        {"submitted_at": "2025-01-01T10:18:00+00:00"},
    ]
    assert submission_window(members) == "18 minutes"


def test_queue_excludes_resolved_and_other_reviewers_claims():
    rows = [
        {"case_id": "available", "status": "pending_review"},
        {"case_id": "mine", "status": "claimed", "claimed_by": "alice"},
        {"case_id": "theirs", "status": "claimed", "claimed_by": "bob"},
        {"case_id": "done", "status": "resolved"},
    ]
    assert [r["case_id"] for r in reviewable_cases(rows, "alice")] == ["available", "mine"]


def test_unknown_risk_stays_visible_ahead_of_scored_cases():
    rows = [
        {"case_id": "low", "status": "pending_review", "automated": {"primary_risk_score": .2}},
        {"case_id": "high", "status": "pending_review", "automated": {"primary_risk_score": .95}},
        {"case_id": "unknown", "status": "pending_review", "automated": {"primary_risk_score": None}},
    ]
    assert [r["case_id"] for r in reviewable_cases(rows, "alice")] == ["unknown", "high", "low"]


def test_interface_uses_the_local_openclaw_inspired_design_tokens():
    css = (Path(__file__).resolve().parents[1] / "merchantshield/ui/theme.css").read_text()
    assert "--paper:#080808" in css
    assert "--accent:#EE725C" in css
    assert "ui-monospace" in css
    assert ".signal-field" in css
    assert "@keyframes signal-scan" in css
    assert "prefers-reduced-motion" in css
