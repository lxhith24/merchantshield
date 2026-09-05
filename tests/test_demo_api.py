"""Phase 5 demonstration and evaluation endpoints."""
from __future__ import annotations

import pytest
from fastapi.testclient import TestClient

from merchantshield.demo import SCENARIOS
from merchantshield.main import app
from merchantshield.review.contracts import CaseStatus
from merchantshield.runtime import reset_runtime

client = TestClient(app)


@pytest.fixture(autouse=True)
def fresh_runtime():
    reset_runtime()
    yield
    reset_runtime()


def test_meta_status_publishes_the_labels_a_surface_must_display():
    body = client.get("/api/v1/meta/status").json()
    assert body["labels"] == {
        "data": "SIMULATED",
        "gateway": "SIMULATED",
        "investigator": "LLM_DISABLED",
    }
    assert "synthetic" in body["disclaimer"].lower()
    assert body["candidate_count"] > 0


def test_every_scenario_is_listed_and_carries_its_candidate():
    body = client.get("/api/v1/demo/scenarios").json()
    assert len(body["scenarios"]) == len(SCENARIOS)
    for scenario in body["scenarios"]:
        assert scenario["candidate_id"].startswith("candidate-")
        assert scenario["member_count"] >= 1
        assert scenario["headline"]
    assert body["labels"]["data"] == "SIMULATED"


def test_an_unknown_scenario_is_404():
    assert client.get("/api/v1/demo/scenarios/nope").status_code == 404
    assert client.post("/api/v1/demo/harness/nope").status_code == 404


def test_a_queue_scenario_refuses_to_run_as_a_harness():
    response = client.post("/api/v1/demo/harness/obvious_ring")
    assert response.status_code == 400
    assert "POST /api/v1/cases" in response.json()["detail"]


def test_the_grounding_harness_returns_a_rejected_citation_and_persists_nothing():
    response = client.post("/api/v1/demo/harness/fabricated_citation")
    assert response.status_code == 200
    body = response.json()

    assert body["persisted"] is False
    run = body["runs"][0]
    assert run["investigation"]["grounding"]["status"] == "insufficient_grounding"
    assert run["decision"]["action"] == "human_review"

    # Nothing reached the durable queue.
    queue = client.get("/api/v1/cases").json()
    assert queue["total"] == 0


def test_the_information_harness_discloses_the_pinned_score():
    body = client.post("/api/v1/demo/harness/information_request_and_resume").json()
    assert body["pinned_primary_risk_score"] == pytest.approx(0.45)
    assert "pinned" in body["disclosure"]

    paused, resumed = body["runs"]
    assert paused["awaiting_information"] is True
    assert resumed["case_id"] == paused["case_id"]
    assert resumed["decision"] is not None


def test_a_queue_scenario_opens_as_a_real_case():
    scenario = client.get("/api/v1/demo/scenarios/evasive_ring").json()
    response = client.post(
        "/api/v1/cases", json={"candidate_id": scenario["candidate_id"]}
    )
    assert response.status_code == 201
    case = response.json()
    assert case["status"] == CaseStatus.PENDING_REVIEW.value
    assert case["automated"]["action"] == "human_review"


def test_the_evaluation_report_omits_id_lists_and_keeps_the_disclaimer():
    body = client.get("/api/v1/evaluation/report").json()
    assert "routed_moe" in body["baselines"]
    assert "rules_only" in body["baselines"]
    assert "test_ids" not in body["manifest"]
    assert "train_ids" not in body["manifest"]
    assert "not production" in body["disclaimer"]
    assert body["routing_summary"]["candidate_count"] > 0
    assert body["train_count"] == 1402
    assert body["test_count"] == 600
    assert body["partitions"]["test"]["fraud_rings"] == 35
    assert "runs" not in body
    assert "original 104-row" in body["evaluation_scope"]


def test_original_demo_report_remains_available():
    body = client.get("/api/v1/evaluation/report?dataset=demo").json()
    assert body["train_count"] == 75
    assert body["test_count"] == 29
    assert client.get("/api/v1/evaluation/report?dataset=unknown").status_code == 422


def test_missing_expanded_report_does_not_silently_show_old_results(tmp_path, monkeypatch):
    from merchantshield.api import demo_routes
    monkeypatch.setattr(demo_routes, "EXPANDED_REPORT_PATH", tmp_path / "missing.json")
    response = client.get("/api/v1/evaluation/report")
    assert response.status_code == 404
    assert "build_expanded_benchmark.py" in response.json()["detail"]


def test_performance_report_exposes_fair_comparison_and_unmet_ring_target():
    response = client.get("/api/v1/evaluation/report?dataset=performance")
    assert response.status_code == 200
    body = response.json()
    assert (body["train_count"], body["validation_count"], body["test_count"]) == (2417, 813, 841)
    comparison = body["comparison"]
    assert comparison["routed_moe"]["precision"] > comparison["legacy_graph_tuned"]["precision"]
    assert comparison["routed_moe"]["recall"] < comparison["legacy_graph_fixed"]["recall"]
    assert comparison["routed_moe"]["ring_recall"] < body["validation_targets"]["ring_recall"]
    assert "not a production" in body["evaluation_scope"]


def test_challenger_scoring_cannot_change_case_or_onboarding():
    candidate_id = client.get("/api/v1/demo/scenarios/obvious_ring").json()["candidate_id"]
    before = client.get("/api/v1/cases").json()
    result = client.get(f"/api/v1/evaluation/challenger/{candidate_id}")
    assert result.status_code == 200
    body = result.json()
    assert body["mode"] == "SHADOW_ONLY"
    assert body["case_updated"] is body["gateway_called"] is False
    assert body["applications"]
    assert all(row["action"] in ("human_review", "proceed_to_onboarding") for row in body["applications"].values())
    assert before == client.get("/api/v1/cases").json()
    assert client.get("/api/v1/evaluation/challenger/absent").status_code == 404


def test_unavailable_challenger_fails_closed(monkeypatch):
    from merchantshield.evaluation import validated
    async def unavailable(*args):
        raise ValueError("invalid manifest")
    monkeypatch.setattr(validated, "load_validated_model", unavailable)
    candidate_id = client.get("/api/v1/demo/scenarios/obvious_ring").json()["candidate_id"]
    response = client.get(f"/api/v1/evaluation/challenger/{candidate_id}")
    assert response.status_code == 503


def test_relationship_report_keeps_failed_adoption_visible():
    response = client.get("/api/v1/evaluation/report?dataset=relationships")
    assert response.status_code == 200
    report = response.json()
    assert report["test_count"] == 1192
    assert not report["adoption_passed"]
    assert not report["adoption_checks"]["false_positive_rate_no_worse_than_v1"]
    assert "prior_v1_retuned" in report["comparison"]
    assert "evasive_shell_ring" in report["cohort_reports"]


def test_relationship_shadow_score_requests_verification_without_case_changes():
    candidate_id = client.get("/api/v1/demo/scenarios/obvious_ring").json()["candidate_id"]
    before = client.get("/api/v1/cases").json()
    response = client.get(f"/api/v1/evaluation/challenger/{candidate_id}?profile=relationships")
    assert response.status_code == 200
    body = response.json()
    assert body["model_version"] == "relationship-context-v2"
    assert body["mode"] == "SHADOW_ONLY"
    assert body["verification_requests"]
    assert all(item["status"] == "UNVERIFIED" for item in body["verification_requests"])
    assert body["case_updated"] is body["gateway_called"] is False
    assert client.get("/api/v1/cases").json() == before
