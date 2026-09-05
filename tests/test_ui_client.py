"""The reviewer surface's API client, including how it loses a version race.

The client is exercised against the real ASGI app, so these tests also prove
the demonstration UI can only do things the API genuinely permits.
"""
from __future__ import annotations

import pytest
from fastapi.testclient import TestClient

from merchantshield.main import app
from merchantshield.runtime import reset_runtime
from merchantshield.ui.client import ApiError, MerchantShieldClient, VersionConflict


@pytest.fixture(autouse=True)
def fresh_runtime():
    reset_runtime()
    yield
    reset_runtime()


@pytest.fixture
def api():
    """A client wired straight to the app, with no socket in between."""
    with MerchantShieldClient(
        "http://testserver", http_client=TestClient(app)
    ) as client:
        yield client


def a_review_case(api: MerchantShieldClient) -> dict:
    scenario = next(
        item for item in api.scenarios() if item["key"] == "evasive_ring"
    )
    return api.open_case(scenario["candidate_id"])


def test_status_and_scenarios_round_trip(api):
    assert api.status()["labels"]["data"] == "SIMULATED"
    keys = {item["key"] for item in api.scenarios()}
    assert {"evasive_ring", "fabricated_citation"} <= keys


def test_opening_the_same_candidate_twice_returns_one_case(api):
    first = a_review_case(api)
    second = api.open_case(first["candidate_id"])
    assert second["case_id"] == first["case_id"]
    assert second["version"] == first["version"]


def test_claim_and_resolve_move_the_case_through_the_lifecycle(api):
    case = a_review_case(api)
    claimed = api.claim(case["case_id"], "reviewer_a", case["version"])
    assert claimed["status"] == "claimed"
    assert claimed["claimed_by"] == "reviewer_a"

    resolved = api.resolve(
        claimed["case_id"],
        reviewer_id="reviewer_a",
        version=claimed["version"],
        decision="escalate",
        outcome="confirmed_ring",
        reason_codes=["shared_device_or_network"],
        notes="Five applications, one device.",
    )
    assert resolved["status"] == "resolved"
    assert resolved["human"]["reviewer_id"] == "reviewer_a"
    # The automated decision was kept beside the human one, not replaced.
    assert resolved["automated"]["action"] == "human_review"


def test_a_stale_version_raises_version_conflict_with_both_numbers(api):
    case = a_review_case(api)
    stale = case["version"]
    api.claim(case["case_id"], "reviewer_a", stale)

    with pytest.raises(VersionConflict) as excinfo:
        api.claim(case["case_id"], "reviewer_b", stale)

    assert excinfo.value.expected_version == stale
    assert excinfo.value.current_version == stale + 1


def test_an_approval_without_a_clearing_reason_is_refused(api):
    case = a_review_case(api)
    claimed = api.claim(case["case_id"], "reviewer_a", case["version"])

    with pytest.raises(ApiError) as excinfo:
        api.resolve(
            claimed["case_id"],
            reviewer_id="reviewer_a",
            version=claimed["version"],
            decision="approve_onboarding",
            outcome="confirmed_ring",
            reason_codes=["insufficient_evidence"],
        )
    assert excinfo.value.status_code == 422


def test_the_client_cannot_hand_off_a_case_no_human_approved(api):
    case = a_review_case(api)
    with pytest.raises(ApiError) as excinfo:
        api.hand_off(case["case_id"], actor="reviewer_a", version=case["version"])
    assert excinfo.value.status_code == 409


def test_a_missing_case_surfaces_as_an_api_error(api):
    with pytest.raises(ApiError) as excinfo:
        api.case("case-does-not-exist")
    assert excinfo.value.status_code == 404


def test_the_harness_is_reachable_through_the_client(api):
    result = api.run_harness("fabricated_citation")
    assert result["persisted"] is False
    assert result["runs"][0]["decision"]["action"] == "human_review"


def test_an_approved_case_can_be_handed_to_the_simulated_gateway(api):
    """The only route to a gateway: a named human recorded an approval."""
    scenario = next(
        item
        for item in api.scenarios()
        if item["key"] == "shared_kiosk_false_positive"
    )
    case = api.open_case(scenario["candidate_id"])
    claimed = api.claim(case["case_id"], "reviewer_a", case["version"])

    resolved = api.resolve(
        claimed["case_id"],
        reviewer_id="reviewer_a",
        version=claimed["version"],
        decision="approve_onboarding",
        outcome="legitimate_coworking",
        reason_codes=["verified_coworking_tenancy"],
        notes="Kiosk tenancy confirmed against the building lease.",
    )
    assert resolved["human"]["decision"] == "approve_onboarding"
    assert resolved["handoff"] is None

    handed = api.hand_off(
        resolved["case_id"], actor="reviewer_a", version=resolved["version"]
    )
    assert handed["handoff"]["mode"] == "SIMULATED"
    assert handed["handoff"]["reference"]
    # The automated human_review decision is still on the record beside it.
    assert handed["automated"]["action"] == "human_review"
