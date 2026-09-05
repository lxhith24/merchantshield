"""Phase 4 REST surface: queue, claim, resolve, audit log, concurrency."""
from __future__ import annotations

import pytest
from fastapi.testclient import TestClient

from merchantshield.main import app
from merchantshield.runtime import reset_runtime

client = TestClient(app)


@pytest.fixture(autouse=True)
def fresh_runtime():
    """Each test gets a runtime bound to this test's empty database."""
    reset_runtime()
    yield
    reset_runtime()


def a_review_candidate() -> str:
    """The largest candidate group; it always requires human review."""
    body = client.get("/api/v1/candidates", params={"min_members": 3}).json()
    assert body["total"] > 0
    return body["candidates"][0]["candidate_id"]


def open_case(candidate_id: str) -> dict:
    response = client.post("/api/v1/cases", json={"candidate_id": candidate_id})
    assert response.status_code == 201, response.text
    return response.json()


def test_candidates_are_labelled_simulated():
    body = client.get("/api/v1/candidates").json()
    assert body["labels"]["data"] == "SIMULATED"
    assert body["labels"]["gateway"] == "SIMULATED"
    assert body["labels"]["investigator"] == "LLM_DISABLED"


def test_opening_a_case_returns_both_decision_slots():
    case = open_case(a_review_candidate())
    assert case["status"] == "pending_review"
    assert case["automated"]["action"] == "human_review"
    assert case["automated"]["reason_codes"]
    assert case["human"] is None
    assert case["version"] == 1
    assert case["candidate"]["member_count"] >= 3
    assert case["members"]
    # Redacted profiles only: no identifier values cross the wire.
    member = case["members"][0]
    assert set(member) == {
        "member_id",
        "available",
        "business_name",
        "business_type",
        "owner_name",
        "registered_address",
        "submitted_at",
        "website_url",
    }


def test_replaying_an_open_request_returns_the_stored_case():
    candidate_id = a_review_candidate()
    first = open_case(candidate_id)
    again = client.post("/api/v1/cases", json={"candidate_id": candidate_id})
    assert again.status_code == 200
    assert again.json()["version"] == first["version"]
    assert len(again.json()["events"]) == len(first["events"])


def test_a_conflicting_idempotency_key_is_refused():
    candidate_id = a_review_candidate()
    open_case(candidate_id)
    response = client.post(
        "/api/v1/cases",
        json={"candidate_id": candidate_id, "idempotency_key": "some-other-key"},
    )
    assert response.status_code == 409


def test_unknown_case_is_404():
    assert client.get("/api/v1/cases/case-missing").status_code == 404


def test_claim_release_and_resolve_flow():
    case = open_case(a_review_candidate())
    case_id = case["case_id"]

    claimed = client.post(
        f"/api/v1/cases/{case_id}/claim",
        json={"reviewer_id": "reviewer_a", "expected_version": case["version"]},
    )
    assert claimed.status_code == 200
    assert claimed.json()["status"] == "claimed"
    assert claimed.json()["claimed_by"] == "reviewer_a"

    released = client.post(
        f"/api/v1/cases/{case_id}/release",
        json={"reviewer_id": "reviewer_a", "expected_version": claimed.json()["version"]},
    )
    assert released.status_code == 200
    assert released.json()["status"] == "pending_review"

    resolved = client.post(
        f"/api/v1/cases/{case_id}/resolve",
        json={
            "reviewer_id": "reviewer_b",
            "expected_version": released.json()["version"],
            "decision": "keep_on_hold",
            "outcome": "confirmed_ring",
            "reason_codes": ["shared_settlement_account"],
            "notes": "One settlement account across the group.",
        },
    )
    assert resolved.status_code == 200
    body = resolved.json()
    assert body["status"] == "resolved"
    assert body["human"]["reviewer_id"] == "reviewer_b"
    # The automated verdict is untouched by the human decision.
    assert body["automated"]["action"] == case["automated"]["action"]
    assert body["automated"]["reason_codes"] == case["automated"]["reason_codes"]


def test_a_stale_version_returns_409_with_the_current_version():
    case = open_case(a_review_candidate())
    case_id = case["case_id"]
    client.post(
        f"/api/v1/cases/{case_id}/claim",
        json={"reviewer_id": "reviewer_a", "expected_version": case["version"]},
    )
    conflict = client.post(
        f"/api/v1/cases/{case_id}/claim",
        json={"reviewer_id": "reviewer_b", "expected_version": case["version"]},
    )
    assert conflict.status_code == 409
    detail = conflict.json()["detail"]
    assert detail["error"] == "concurrent_modification"
    assert detail["current_version"] == case["version"] + 1


def test_an_approval_without_a_clearing_reason_is_rejected():
    case = open_case(a_review_candidate())
    response = client.post(
        f"/api/v1/cases/{case['case_id']}/resolve",
        json={
            "reviewer_id": "reviewer_a",
            "expected_version": case["version"],
            "decision": "approve_onboarding",
            "outcome": "inconclusive",
            "reason_codes": ["insufficient_evidence"],
        },
    )
    assert response.status_code == 422
    assert "clearing reason code" in response.json()["detail"]


def test_the_audit_log_is_ordered_and_attributed():
    case = open_case(a_review_candidate())
    case_id = case["case_id"]
    client.post(
        f"/api/v1/cases/{case_id}/resolve",
        json={
            "reviewer_id": "reviewer_a",
            "expected_version": case["version"],
            "decision": "keep_on_hold",
            "outcome": "confirmed_ring",
            "reason_codes": ["shared_owner_identifier"],
        },
    )
    events = client.get(f"/api/v1/cases/{case_id}/events").json()["events"]
    assert [event["sequence"] for event in events] == list(range(1, len(events) + 1))
    assert events[0]["event_type"] == "case_opened"
    assert events[-1]["event_type"] == "human_decision_recorded"
    assert events[-1]["actor_kind"] == "human"
    assert events[0]["actor_kind"] == "system"


def test_the_queue_and_counters_reflect_persisted_cases():
    candidate_id = a_review_candidate()
    open_case(candidate_id)

    queue = client.get("/api/v1/cases", params={"status": "pending_review"}).json()
    assert queue["total"] == 1
    assert queue["cases"][0]["candidate_id"] == candidate_id

    stats = client.get("/api/v1/cases/stats").json()
    assert stats["counts"]["pending_review"] == 1
    assert stats["open_for_review"] == 1
    assert stats["labels"]["data"] == "SIMULATED"


def test_the_review_vocabulary_is_published_for_the_ui():
    body = client.get("/api/v1/reference/review-vocabulary").json()
    assert "approve_onboarding" in body["decisions"]
    assert "confirmed_ring" in body["outcomes"]
    assert set(body["clearing_reason_codes"]).issubset(set(body["reason_codes"]))
    assert "approve_onboarding requires" in body["rule"]


def test_safe_demo_does_not_mount_the_legacy_auto_rejection_api():
    response = client.post("/api/v1/merchants/assess")
    assert response.status_code == 404


def test_safe_demo_lifespan_starts_cleanly():
    with TestClient(app) as live_client:
        response = live_client.get("/health")
    assert response.status_code == 200
