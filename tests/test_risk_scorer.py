"""Risk fusion and decision gating, including the missing-measurement rule."""
from __future__ import annotations

import pytest

from merchantshield.analysis import RiskScorer
from merchantshield.models.document import DocumentAnalysis, DocumentType
from merchantshield.models.risk_assessment import ClusterInfo, Decision, RiskSignal

pytestmark = pytest.mark.asyncio


def empty_cluster() -> dict:
    return {
        "clustering_risk_score": 0.0,
        "cluster_info": ClusterInfo(cluster_size=1),
        "signals": [],
    }


async def test_clean_application_is_approved():
    a = await RiskScorer().score(
        "app1", [], {"synthetic_score": 0.0, "signals": []}, empty_cluster(), "Clean Co"
    )
    assert a.decision is Decision.APPROVED
    assert a.final_risk_score == 0.0
    assert a.analysis_mode == "heuristic_only"


async def test_midband_goes_to_review():
    synth = {
        "synthetic_score": 0.45,
        "signals": [RiskSignal(
            name="unaffiliated_free_email", value=0.45,
            explanation="free inbox", source="synthetic_identity",
        )],
    }
    a = await RiskScorer().score("app2", [], synth, empty_cluster(), "Grey Co")
    assert a.decision is Decision.UNDER_REVIEW
    assert 0.25 < a.final_risk_score < 0.75


async def test_decisive_signal_rejects_regardless_of_average():
    synth = {
        "synthetic_score": 0.30,
        "signals": [RiskSignal(
            name="repeated_digit_account", value=0.95, weight=1.5,
            explanation="account is one repeated digit", source="synthetic_identity",
        )],
    }
    a = await RiskScorer().score("app3", [], synth, empty_cluster(), "Fake Co")
    assert a.decision is Decision.REJECTED
    assert "Decisive signal" in a.decision_reason


async def test_large_cluster_forces_rejection():
    cluster = {
        "clustering_risk_score": 0.6,
        "cluster_info": ClusterInfo(
            cluster_id="cl_x", cluster_size=9,
            shared_attributes=["bank_account"],
            related_application_ids=[f"m{i}" for i in range(8)],
        ),
        "signals": [],
    }
    a = await RiskScorer().score(
        "app4", [], {"synthetic_score": 0.1, "signals": []}, cluster, "Ring Co"
    )
    assert a.decision is Decision.REJECTED


async def test_skipped_documents_do_not_count_as_clean():
    """A document that could not be analysed must not dilute the score toward 0."""
    skipped = DocumentAnalysis(
        document_type=DocumentType.PAN_CARD, file_path="/tmp/x.jpg",
        document_risk_score=0.0, analysis_skipped=True, skip_reason="no API key",
    )
    synth = {
        "synthetic_score": 0.60,
        "signals": [RiskSignal(
            name="s", value=0.60, explanation="e", source="synthetic_identity"
        )],
    }
    with_skip = await RiskScorer().score("a", [skipped], synth, empty_cluster(), "Co")
    without = await RiskScorer().score("b", [], synth, empty_cluster(), "Co")
    # Identical: the skipped document's weight is redistributed, not scored 0.
    assert with_skip.final_risk_score == without.final_risk_score
    assert "not analysed" in with_skip.human_readable_explanation


async def test_measured_document_raises_score():
    tampered = DocumentAnalysis(
        document_type=DocumentType.PAN_CARD, file_path="/tmp/x.jpg",
        tampering_detected=True, tampering_confidence=0.9,
        tampering_reasons=["font mismatch on the PAN line"],
        document_risk_score=0.8,
    )
    synth = {"synthetic_score": 0.1, "signals": []}
    a = await RiskScorer().score("app6", [tampered], synth, empty_cluster(), "Doc Co")
    assert a.document_risk_score == 0.8
    assert any(s.name.startswith("tampering_") for s in a.risk_signals)
    assert a.final_risk_score > 0.25


async def test_explanation_is_always_present():
    a = await RiskScorer().score(
        "app7", [], {"synthetic_score": 0.0, "signals": []}, empty_cluster(), "Quiet Co"
    )
    assert a.human_readable_explanation
    assert "Quiet Co" in a.human_readable_explanation


async def test_thresholds_recorded_for_audit():
    a = await RiskScorer().score(
        "app8", [], {"synthetic_score": 0.0, "signals": []}, empty_cluster(), "Audit Co"
    )
    assert a.auto_approve_threshold == 0.25
    assert a.auto_reject_threshold == 0.75
    assert a.assessed_at is not None
