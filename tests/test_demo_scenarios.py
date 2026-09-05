"""Every frozen scenario is checked against the real runtime.

The point of these tests is that a demonstration cannot drift away from the
system: if the pipeline stops producing what a scenario claims, the suite goes
red before a recording does.
"""
from __future__ import annotations

import pytest

from merchantshield.demo import (
    SCENARIOS,
    ScenarioKind,
    ScenarioNotFound,
    resolve_scenario,
    resolve_scenarios,
    run_grounding_failure,
    run_information_request,
)
from merchantshield.demo.harness import FABRICATED_EVIDENCE_ID
from merchantshield.runtime import build_runtime

QUEUE_SCENARIOS = [item.key for item in SCENARIOS if item.kind is ScenarioKind.QUEUE]


@pytest.fixture
async def runtime():
    """A freshly assembled runtime per test; `fresh_db` has already run."""
    return await build_runtime()


async def test_every_scenario_resolves_to_a_whole_candidate_group(runtime):
    resolved = resolve_scenarios(runtime.world)
    assert len(resolved) == len(SCENARIOS)
    for item in resolved:
        assert item.candidate_id in runtime.world.by_id
        assert item.member_ids


async def test_an_unknown_scenario_key_is_refused(runtime):
    with pytest.raises(ScenarioNotFound):
        resolve_scenario(runtime.world, "no-such-scenario")


@pytest.mark.parametrize("key", QUEUE_SCENARIOS)
async def test_queue_scenario_matches_what_the_pipeline_actually_does(runtime, key):
    resolved = resolve_scenario(runtime.world, key)
    scenario = resolved.scenario

    run = await runtime.workflow.run(resolved.candidate_id)

    assert run.decision is not None
    assert run.decision.action is scenario.expected_action
    assert run.decision.review_status.value == scenario.expected_review_status

    ran = tuple(item.expert_name for item in run.state.assessment.expert_results)
    assert ran == scenario.expected_experts

    investigation = run.state.investigation
    actual = "skipped" if investigation is None else investigation.status.value
    assert actual == scenario.expected_investigator_status


async def test_the_held_out_flag_matches_the_frozen_split(runtime):
    evasive = resolve_scenario(runtime.world, "evasive_ring")
    assert evasive.held_out is True
    assert evasive.ground_truth == "shell_ring"

    office = resolve_scenario(runtime.world, "legitimate_shared_office")
    assert office.ground_truth == "legitimate"


async def test_the_declared_false_positive_really_is_one(runtime):
    """The shared-kiosk scenario claims to be an honest false positive."""
    resolved = resolve_scenario(runtime.world, "shared_kiosk_false_positive")
    assert resolved.ground_truth == "legitimate"

    run = await runtime.workflow.run(resolved.candidate_id)
    assert run.decision.action.value == "human_review"


async def test_the_sparse_router_really_skips_the_tabular_expert(runtime):
    """Two scenarios differ precisely in whether the third expert is bought."""
    obvious = resolve_scenario(runtime.world, "obvious_ring")
    evasive = resolve_scenario(runtime.world, "evasive_ring")

    obvious_run = await runtime.workflow.run(obvious.candidate_id)
    evasive_run = await runtime.workflow.run(evasive.candidate_id)

    obvious_experts = {
        item.expert_name for item in obvious_run.state.assessment.expert_results
    }
    evasive_experts = {
        item.expert_name for item in evasive_run.state.assessment.expert_results
    }
    assert "tabular_model" not in obvious_experts
    assert "tabular_model" in evasive_experts


async def test_evasive_ring_narrative_matches_its_actual_device_evidence(runtime):
    resolved = resolve_scenario(runtime.world, "evasive_ring")
    run = await runtime.workflow.run(resolved.candidate_id)

    output = run.state.investigation.output
    assert output is not None
    assert "device fingerprint" in output.narrative
    assert "settlement" not in output.narrative
    assert "owner identity" not in output.narrative
    assert "shared application device" in output.missing_evidence[0]


async def test_grounding_failure_harness_rejects_the_fabricated_citation(runtime):
    resolved = resolve_scenario(runtime.world, "fabricated_citation")
    result = await run_grounding_failure(runtime, resolved.candidate_id)
    payload = result.to_dict()

    assert payload["persisted"] is False
    assert payload["pinned_primary_risk_score"] is None

    run = payload["runs"][0]
    grounding = run["investigation"]["grounding"]
    assert grounding["status"] == "insufficient_grounding"
    assert FABRICATED_EVIDENCE_ID in grounding["unknown_evidence_ids"]

    # The confident 'continue_onboarding' never survives into structured output.
    assert run["investigation"]["output"] is None
    assert FABRICATED_EVIDENCE_ID in run["investigation"]["rejected_narrative"]
    assert run["effective_recommendation"] == "human_review"
    assert run["decision"]["action"] == "human_review"
    assert "INSUFFICIENT_GROUNDING" in run["decision"]["reason_codes"]


async def test_the_grounding_harness_does_not_persist_a_case(runtime):
    resolved = resolve_scenario(runtime.world, "fabricated_citation")
    before = runtime.service.queue_counts()

    await run_grounding_failure(runtime, resolved.candidate_id)

    assert runtime.service.queue_counts() == before


async def test_information_request_harness_pauses_then_resumes_one_case(runtime):
    resolved = resolve_scenario(runtime.world, "information_request_and_resume")
    result = await run_information_request(runtime, resolved.candidate_id)
    payload = result.to_dict()

    assert payload["pinned_primary_risk_score"] == pytest.approx(0.45)
    assert "pinned" in payload["disclosure"]

    paused, resumed = payload["runs"]
    assert paused["awaiting_information"] is True
    assert paused["decision"] is None
    assert paused["investigation"]["status"] == "awaiting_information"
    assert paused["interrupt_payload"]["requested_information"]

    assert resumed["case_id"] == paused["case_id"]
    assert resumed["awaiting_information"] is False
    assert resumed["decision"] is not None
