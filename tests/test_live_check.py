"""Live synthetic merchant checking through ingestion, workflow, API and UI client."""
from __future__ import annotations

from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from merchantshield.demo.live_check import (
    LIVE_CHECK_EXAMPLES,
    MAX_LIVE_CHECK_ROWS,
    parse_live_check_csv,
    run_live_check,
)
from merchantshield.main import app
from merchantshield.runtime import get_runtime, reset_runtime
from merchantshield.ui.client import ApiError, MerchantShieldClient


@pytest.fixture(autouse=True)
def fresh_runtime():
    reset_runtime()
    yield
    reset_runtime()


@pytest.fixture
def api():
    with MerchantShieldClient(
        "http://testserver", http_client=TestClient(app)
    ) as client:
        yield client


def test_examples_are_valid_and_use_only_new_merchant_ids():
    for example in LIVE_CHECK_EXAMPLES.values():
        applications = parse_live_check_csv(example["csv"])
        assert applications
        assert all(member_id.startswith("LIVE-") for member_id in applications)


def test_ingestion_requires_complete_relationship_evidence():
    text = LIVE_CHECK_EXAMPLES["new_ring"]["csv"].replace(
        "device-demo-3321", "", 1
    )
    with pytest.raises(ValueError, match="missing evidence is not clean"):
        parse_live_check_csv(text)


@pytest.mark.parametrize("account", ["HASH_123456789", "123", "1234567890123456789"])
def test_ingestion_refuses_non_numeric_or_out_of_range_bank_accounts(account):
    text = LIVE_CHECK_EXAMPLES["known_ring_match"]["csv"].replace(
        "111111111111", account
    )
    with pytest.raises(ValueError, match="bank_account must contain 9–18 digits"):
        parse_live_check_csv(text)


def test_ingestion_accepts_the_expanded_batch_limit():
    header, row = LIVE_CHECK_EXAMPLES["known_ring_match"]["csv"].strip().splitlines()
    rows = [
        row.replace("LIVE-701", f"LIVE-{700 + number}")
        for number in range(1, MAX_LIVE_CHECK_ROWS + 1)
    ]
    applications = parse_live_check_csv("\n".join([header, *rows]))
    assert len(applications) == MAX_LIVE_CHECK_ROWS


def test_ingestion_enforces_the_per_request_merchant_limit():
    header, row = LIVE_CHECK_EXAMPLES["known_ring_match"]["csv"].strip().splitlines()
    rows = [row.replace("LIVE-701", f"LIVE-{700 + number}") for number in range(1, MAX_LIVE_CHECK_ROWS + 2)]
    with pytest.raises(ValueError, match="merchant demo limit"):
        parse_live_check_csv("\n".join([header, *rows]))


@pytest.mark.asyncio
async def test_one_submitted_merchant_can_join_a_synthetic_reference_ring():
    runtime = await get_runtime()
    submitted = parse_live_check_csv(LIVE_CHECK_EXAMPLES["known_ring_match"]["csv"])
    result = await run_live_check(submitted, runtime=runtime)

    assert result["persisted"] is False
    assert result["gateway_called"] is False
    assert len(result["results"]) == 1
    match = result["results"][0]
    assert match["finding"] == "linked_to_synthetic_reference"
    assert match["submitted_member_ids"] == ["LIVE-701"]
    assert len(match["reference_member_ids"]) == 5
    assert match["action"] == "human_review"


@pytest.mark.asyncio
async def test_new_linked_batch_and_independent_control_take_different_paths():
    runtime = await get_runtime()
    linked = await run_live_check(
        parse_live_check_csv(LIVE_CHECK_EXAMPLES["new_ring"]["csv"]),
        runtime=runtime,
    )
    independent = await run_live_check(
        parse_live_check_csv(LIVE_CHECK_EXAMPLES["independent"]["csv"]),
        runtime=runtime,
    )

    assert [(row["finding"], row["action"]) for row in linked["results"]] == [
        ("new_linked_group", "human_review")
    ]
    assert len(independent["results"]) == 3
    assert {
        (row["finding"], row["action"]) for row in independent["results"]
    } == {("no_ring_link_found", "proceed_to_onboarding")}


@pytest.mark.asyncio
async def test_six_file_video_fixture_forms_one_explainable_review_group():
    fixture_dir = Path(__file__).resolve().parents[1] / "demo_uploads/six_file_ring"
    csv_files = sorted(fixture_dir.glob("*.csv"))
    lines = [path.read_text(encoding="utf-8").strip().splitlines() for path in csv_files]
    combined = "\n".join([lines[0][0], *(file_lines[1] for file_lines in lines)])

    result = await run_live_check(parse_live_check_csv(combined), runtime=await get_runtime())

    assert len(csv_files) == 6
    assert result["submitted_count"] == 6
    assert result["result_count"] == 1
    group = result["results"][0]
    assert group["finding"] == "new_linked_group"
    assert group["action"] == "human_review"
    assert len(group["submitted_member_ids"]) == 6
    assert {
        edge["attribute"] for edge in group["candidate"]["evidence"]
    } == {"address_hash", "bank_account", "device_fingerprint", "ip_address"}


def test_api_returns_real_trace_labels_and_safety_disclaimer(api):
    before = api.list_cases(limit=200)["total"]
    examples = api.live_check_examples()["examples"]
    response = api.live_check(examples["new_ring"]["csv"])

    assert response["labels"]["data"] == "SIMULATED"
    assert response["persisted"] is False
    assert response["gateway_called"] is False
    assert response["results"][0]["route_trace"]
    assert response["results"][0]["workflow_trace"]
    assert "does not authenticate documents" in response["scope_note"]
    assert api.list_cases(limit=200)["total"] == before


def test_api_rejects_malformed_pasted_details(api):
    with pytest.raises(ApiError) as excinfo:
        api.live_check("merchant_id,business_name\nLIVE-1,Incomplete")
    assert excinfo.value.status_code == 422
