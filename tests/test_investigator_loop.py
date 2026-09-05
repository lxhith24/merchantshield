"""Bounded investigation: budgets, grounding, and safe failure states."""
from __future__ import annotations

import asyncio
import json

from merchantshield.investigation import (
    BoundedInvestigator,
    DeterministicInvestigator,
    DisabledProvider,
    GroundingStatus,
    InvestigationBudget,
    InvestigationStatus,
    InvestigatorOutput,
    Recommendation,
    ScriptedProvider,
    ToolCallStatus,
    build_evidence_catalogue,
)
from merchantshield.investigation.tools import TOOL_NAMES
from tests.conftest import make_ring_case


def _conclude(cited, *, recommendation="human_review", requested=()) -> str:
    payload = {
        "action": "conclude",
        "recommendation": recommendation,
        "ring_hypothesis": {
            "kind": "coordinated_ring",
            "label": "coordinated_merchant_ring",
            "statement": "One party controls these nominally separate merchants.",
            "supporting_evidence_ids": list(cited),
            "contradicting_evidence_ids": [],
        },
        "legitimate_alternatives": [
            {
                "kind": "legitimate_alternative",
                "label": "shared_accountant",
                "statement": "An accountant filed on behalf of unrelated clients.",
                "supporting_evidence_ids": [],
                "contradicting_evidence_ids": [],
            }
        ],
        "missing_evidence": [],
        "requested_information": list(requested),
        "confidence": 0.7,
        "cited_evidence_ids": list(cited),
        "narrative": "Members reuse onboarding infrastructure across applications.",
    }
    return json.dumps(payload)


def _call(tool: str, **arguments) -> str:
    return json.dumps({"action": "call_tool", "tool": tool, "arguments": arguments})


async def _run(provider, *, budget=None, **case_kwargs):
    candidate, applications, assessment = await make_ring_case(**case_kwargs)
    investigator = BoundedInvestigator(
        provider=provider, budget=budget or InvestigationBudget()
    )
    result = await investigator.investigate(
        candidate=candidate, applications=applications, assessment=assessment
    )
    return result, candidate, assessment


# -- happy path ---------------------------------------------------------


async def test_deterministic_investigator_selects_tools_rather_than_fanning_out():
    result, _, _ = await _run(DeterministicInvestigator())

    assert result.status in {
        InvestigationStatus.COMPLETED,
        InvestigationStatus.AWAITING_INFORMATION,
    }
    assert result.grounding.status is GroundingStatus.GROUNDED
    called = [item.tool_name for item in result.observations]
    assert called[0] == "get_candidate_summary"
    assert len(called) == len(set(called)), "no tool should be called twice"
    assert 0 < len(called) < len(TOOL_NAMES), "selective choice, not full fan-out"
    assert result.output is not None
    assert result.output.legitimate_alternatives


async def test_output_carries_no_risk_score_and_stays_within_budget():
    budget = InvestigationBudget(max_steps=5, max_tool_calls=6, max_seconds=15.0)
    result, _, assessment = await _run(DeterministicInvestigator(), budget=budget)

    assert "risk_score" not in InvestigatorOutput.model_fields
    assert result.steps_used <= budget.max_steps
    assert result.tool_calls_used <= budget.max_tool_calls
    assert result.duration_ms < budget.max_seconds * 1000.0
    # The investigation neither reproduces nor perturbs the graph expert's score.
    assert assessment.primary_risk_score == 0.80


async def test_every_cited_id_is_validated_against_disclosed_evidence():
    result, candidate, assessment = await _run(DeterministicInvestigator())
    catalogue = build_evidence_catalogue(candidate, assessment)
    disclosed = {
        item
        for observation in result.observations
        for item in observation.disclosed_evidence_ids
    }

    assert result.output is not None
    for evidence_id in result.output.cited_evidence_ids:
        assert evidence_id in catalogue
        assert evidence_id in disclosed
    assert set(result.grounding.validated_evidence_ids) == set(
        result.output.cited_evidence_ids
    )


# -- grounding failures -------------------------------------------------


async def test_hallucinated_evidence_id_invalidates_the_narrative():
    result, _, _ = await _run(ScriptedProvider([_conclude(["edge-000000000000"])]))

    assert result.status is InvestigationStatus.FAILED
    assert result.grounding.status is GroundingStatus.INSUFFICIENT_GROUNDING
    assert result.grounding.unknown_evidence_ids == ("edge-000000000000",)
    assert result.error_code == "HALLUCINATED_EVIDENCE_ID"
    # The narrative is preserved for audit but kept out of actionable output.
    assert result.output is None
    assert result.rejected_narrative
    assert result.effective_recommendation is Recommendation.HUMAN_REVIEW


async def test_real_but_undisclosed_evidence_id_is_unauthorized():
    candidate, applications, assessment = await make_ring_case()
    real_id = candidate.evidence[0].evidence_id
    investigator = BoundedInvestigator(provider=ScriptedProvider([_conclude([real_id])]))

    result = await investigator.investigate(
        candidate=candidate, applications=applications, assessment=assessment
    )

    assert result.status is InvestigationStatus.FAILED
    assert result.error_code == "UNAUTHORIZED_EVIDENCE_ID"
    assert result.grounding.undisclosed_evidence_ids == (real_id,)


async def test_citation_is_accepted_once_a_tool_actually_disclosed_it():
    candidate, applications, assessment = await make_ring_case()
    real_id = candidate.evidence[0].evidence_id
    provider = ScriptedProvider(
        [
            _call("get_relationship_evidence", candidate_id=candidate.candidate_id),
            _conclude([real_id]),
        ]
    )
    investigator = BoundedInvestigator(provider=provider)

    result = await investigator.investigate(
        candidate=candidate, applications=applications, assessment=assessment
    )

    assert result.status is InvestigationStatus.COMPLETED
    assert result.grounding.status is GroundingStatus.GROUNDED
    assert result.grounding.validated_evidence_ids == (real_id,)


# -- provider and protocol failures -------------------------------------


async def test_llm_disabled_mode_abstains_and_is_labelled():
    result, _, _ = await _run(DisabledProvider())

    assert result.status is InvestigationStatus.ABSTAINED
    assert result.error_code == "LLM_DISABLED"
    assert result.output is None
    assert result.effective_recommendation is Recommendation.ABSTAIN


async def test_invalid_json_is_an_explicit_failure_not_a_clean_result():
    result, _, _ = await _run(ScriptedProvider(["I think this merchant is fine."]))

    assert result.status is InvestigationStatus.FAILED
    assert result.error_code == "INVALID_JSON_RESPONSE"
    assert result.output is None
    assert result.effective_recommendation is Recommendation.HUMAN_REVIEW


async def test_schema_violating_conclusion_is_rejected():
    incomplete = json.dumps(
        {
            "action": "conclude",
            "recommendation": "continue_onboarding",
            "confidence": 0.9,
            "narrative": "Looks fine to me.",
        }
    )
    result, _, _ = await _run(ScriptedProvider([incomplete]))

    assert result.status is InvestigationStatus.FAILED
    assert result.error_code == "INVALID_OUTPUT_SCHEMA"


async def test_unknown_action_fails_safe():
    result, _, _ = await _run(ScriptedProvider([json.dumps({"action": "delete_case"})]))

    assert result.status is InvestigationStatus.FAILED
    assert result.error_code == "UNKNOWN_ACTION"


async def test_provider_exception_is_isolated():
    result, _, _ = await _run(ScriptedProvider([]))

    assert result.status is InvestigationStatus.FAILED
    assert result.error_code == "PROVIDER_ERROR"


async def test_investigation_timeout_fails_safe():
    class SlowProvider:
        name = "slow"
        version = "slow-v1"

        async def propose(self, *, system, transcript, max_tokens):
            await asyncio.sleep(5.0)
            return _conclude([])

    result, _, _ = await _run(
        SlowProvider(), budget=InvestigationBudget(max_seconds=0.05)
    )

    assert result.status is InvestigationStatus.FAILED
    assert result.error_code == "INVESTIGATION_TIMEOUT"
    assert result.effective_recommendation is Recommendation.HUMAN_REVIEW


# -- budgets ------------------------------------------------------------


async def test_tool_budget_exhaustion_is_explicit():
    candidate, applications, assessment = await make_ring_case()
    looping = [
        _call("get_candidate_summary", candidate_id=candidate.candidate_id)
    ] * 6
    investigator = BoundedInvestigator(
        provider=ScriptedProvider(looping),
        budget=InvestigationBudget(max_steps=5, max_tool_calls=2),
    )

    result = await investigator.investigate(
        candidate=candidate, applications=applications, assessment=assessment
    )

    assert result.status is InvestigationStatus.FAILED
    assert result.error_code == "TOOL_BUDGET_EXHAUSTED"
    assert result.tool_calls_used == 2


async def test_step_budget_exhaustion_is_explicit():
    candidate, applications, assessment = await make_ring_case()
    investigator = BoundedInvestigator(
        provider=ScriptedProvider(
            [_call("get_candidate_summary", candidate_id=candidate.candidate_id)] * 4
        ),
        budget=InvestigationBudget(max_steps=2, max_tool_calls=6),
    )

    result = await investigator.investigate(
        candidate=candidate, applications=applications, assessment=assessment
    )

    assert result.status is InvestigationStatus.FAILED
    assert result.error_code == "STEP_BUDGET_EXHAUSTED"
    assert result.steps_used == 2


async def test_out_of_scope_tool_call_is_recorded_and_recoverable():
    candidate, applications, assessment = await make_ring_case()
    real_id = candidate.evidence[0].evidence_id
    provider = ScriptedProvider(
        [
            _call("get_member_profile", candidate_id=candidate.candidate_id,
                  member_id="syn-9999"),
            _call("get_relationship_evidence", candidate_id=candidate.candidate_id),
            _conclude([real_id]),
        ]
    )
    investigator = BoundedInvestigator(provider=provider)

    result = await investigator.investigate(
        candidate=candidate, applications=applications, assessment=assessment
    )

    rejected = result.observations[0]
    assert rejected.status is ToolCallStatus.ERROR
    assert rejected.error_code == "MEMBER_OUT_OF_SCOPE"
    assert rejected.disclosed_evidence_ids == ()
    assert result.status is InvestigationStatus.COMPLETED


async def test_request_information_pauses_with_a_structured_request():
    requested = [
        {
            "item_code": "franchise_agreement",
            "description": "Provide the franchise agreement for this brand.",
            "member_id": None,
        }
    ]
    candidate, applications, assessment = await make_ring_case()
    real_id = candidate.evidence[0].evidence_id
    provider = ScriptedProvider(
        [
            _call("get_relationship_evidence", candidate_id=candidate.candidate_id),
            _conclude(
                [real_id], recommendation="request_information", requested=requested
            ),
        ]
    )

    result = await BoundedInvestigator(provider=provider).investigate(
        candidate=candidate, applications=applications, assessment=assessment
    )

    assert result.status is InvestigationStatus.AWAITING_INFORMATION
    assert result.output is not None
    assert result.output.requested_information[0].item_code == "franchise_agreement"
