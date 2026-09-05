"""LangGraph case lifecycle: routing, pause/resume, and policy authority."""
from __future__ import annotations

import json

from merchantshield.agent import (
    CandidateAction,
    ExpertCost,
    GraphScoringExpert,
    RoutedMoESystem,
    RulesExpert,
)
from merchantshield.analysis import EvidenceGraphBuilder
from merchantshield.evaluation import generate_synthetic_dataset
from merchantshield.investigation import (
    BoundedInvestigator,
    DeterministicInvestigator,
    DisabledProvider,
    InvestigationStatus,
    Recommendation,
    ScriptedProvider,
)
from merchantshield.workflow import (
    CaseState,
    InMemoryCaseRepository,
    MerchantShieldWorkflow,
    ReviewStatus,
    WorkflowConfig,
    case_id_for,
)
from tests.test_investigator_loop import _call, _conclude


def _world(seed: int = 20250904):
    examples = generate_synthetic_dataset(seed=seed)
    applications = {item.example_id: item.application for item in examples}
    candidates = EvidenceGraphBuilder().build(applications)
    return candidates, applications


def _workflow(candidates, applications, provider, *, graph_score=0.80, config=None):
    return MerchantShieldWorkflow(
        repository=InMemoryCaseRepository(candidates, applications),
        router=RoutedMoESystem(
            rules_expert=RulesExpert(),
            graph_expert=GraphScoringExpert(lambda _: graph_score),
        ),
        investigator=BoundedInvestigator(provider=provider),
        config=config or WorkflowConfig(),
    )


def _multi(candidates):
    return next(item for item in candidates if len(item.member_ids) >= 3)


# Candidates linked only by weak attributes make the investigator ask for more
# information, which pauses the graph. Tests that need a terminal run pick a
# component carrying a strong shared attribute instead.
_STRONG_ATTRIBUTES = {"bank_account", "owner_pan", "device_fingerprint"}


def _strong_multi(candidates):
    return next(
        item
        for item in candidates
        if len(item.member_ids) >= 3
        and any(edge.attribute in _STRONG_ATTRIBUTES for edge in item.evidence)
    )


# -- routing ------------------------------------------------------------


async def test_confident_low_risk_case_skips_the_investigator():
    candidates, applications = _world()
    workflow = _workflow(
        candidates, applications, DeterministicInvestigator(), graph_score=0.05
    )
    singletons = [item for item in candidates if len(item.member_ids) == 1]

    cleared = None
    for candidate in singletons:
        run = await workflow.run(candidate.candidate_id)
        if run.decision.action is CandidateAction.PROCEED_TO_ONBOARDING:
            cleared = run
            break

    assert cleared is not None, "expected at least one confidently clean candidate"
    assert cleared.state.investigation is None
    assert cleared.decision.review_status is ReviewStatus.CLEARED
    assert [event.node for event in cleared.state.workflow_trace] == [
        "assess",
        "finalize",
    ]


async def test_review_case_runs_a_bounded_investigation_with_a_full_trace():
    candidates, applications = _world()
    workflow = _workflow(candidates, applications, DeterministicInvestigator())

    run = await workflow.run(_strong_multi(candidates).candidate_id)

    assert run.decision.action is CandidateAction.HUMAN_REVIEW
    assert run.state.investigation is not None
    assert run.state.investigation.observations
    nodes = [event.node for event in run.state.workflow_trace]
    assert nodes[0] == "assess" and nodes[-1] == "finalize"
    assert "investigate" in nodes
    # The whole case is auditable: experts, route trace, tools, grounding, policy.
    assert run.state.assessment.route_trace
    assert run.state.investigation.grounding.status.value
    assert run.decision.reason_codes


# -- policy authority ---------------------------------------------------


async def test_investigator_cannot_downgrade_a_flagged_case():
    candidates, applications = _world()
    candidate = _multi(candidates)
    real_id = candidate.evidence[0].evidence_id
    provider = ScriptedProvider(
        [
            _call("get_relationship_evidence", candidate_id=candidate.candidate_id),
            _conclude([real_id], recommendation="continue_onboarding"),
        ]
    )
    workflow = _workflow(candidates, applications, provider)

    run = await workflow.run(candidate.candidate_id)

    assert run.state.investigation.status is InvestigationStatus.COMPLETED
    assert (
        run.decision.investigator_recommendation is Recommendation.CONTINUE_ONBOARDING
    )
    # Recorded as an opinion, but the action is unchanged.
    assert run.decision.action is CandidateAction.HUMAN_REVIEW
    assert run.decision.review_status is ReviewStatus.PENDING_HUMAN_REVIEW


async def test_investigator_never_alters_the_primary_risk_score():
    candidates, applications = _world()
    candidate = _strong_multi(candidates)
    workflow = _workflow(
        candidates, applications, DeterministicInvestigator(), graph_score=0.66
    )

    run = await workflow.run(candidate.candidate_id)

    assert run.decision.primary_risk_score == 0.66
    assert run.state.assessment.primary_risk_score == 0.66


async def test_hallucinated_citation_routes_to_a_human_with_the_trace_intact():
    candidates, applications = _world()
    candidate = _multi(candidates)
    provider = ScriptedProvider(
        [
            _call("get_candidate_summary", candidate_id=candidate.candidate_id),
            _conclude(["edge-deadbeef0000"], recommendation="continue_onboarding"),
        ]
    )
    workflow = _workflow(candidates, applications, provider)

    run = await workflow.run(candidate.candidate_id)

    assert run.state.investigation.error_code == "HALLUCINATED_EVIDENCE_ID"
    assert "INSUFFICIENT_GROUNDING" in run.decision.reason_codes
    assert run.decision.action is CandidateAction.HUMAN_REVIEW
    assert run.decision.investigator_recommendation is Recommendation.HUMAN_REVIEW
    # The failed attempt is still fully auditable.
    assert run.state.investigation.observations
    assert run.state.investigation.rejected_narrative


async def test_llm_disabled_case_is_labelled_and_still_reviewed():
    candidates, applications = _world()
    workflow = _workflow(candidates, applications, DisabledProvider())

    run = await workflow.run(_multi(candidates).candidate_id)

    assert run.state.investigation.error_code == "LLM_DISABLED"
    assert "INVESTIGATOR_ABSTAINED" in run.decision.reason_codes
    assert run.decision.action is CandidateAction.HUMAN_REVIEW


async def test_primary_expert_failure_still_reaches_a_decision():
    candidates, applications = _world()
    candidate = _strong_multi(candidates)

    class BrokenGraphExpert:
        name = "graph_model"
        cost_class = ExpertCost.MEDIUM
        implementation_version = "broken-test-v1"

        async def evaluate(self, context, *, route_reason):
            raise RuntimeError("simulated model failure")

    workflow = MerchantShieldWorkflow(
        repository=InMemoryCaseRepository(candidates, applications),
        router=RoutedMoESystem(
            rules_expert=RulesExpert(), graph_expert=BrokenGraphExpert()
        ),
        investigator=BoundedInvestigator(provider=DeterministicInvestigator()),
    )

    run = await workflow.run(candidate.candidate_id)

    assert run.decision.primary_risk_score is None
    assert run.decision.action is CandidateAction.HUMAN_REVIEW
    assert "PRIMARY_EXPERT_UNAVAILABLE" in run.decision.reason_codes


# -- interrupt and resume -----------------------------------------------


async def test_information_request_pauses_and_resumes_the_same_case():
    candidates, applications = _world()
    candidate = _multi(candidates)
    real_id = candidate.evidence[0].evidence_id
    requested = [
        {
            "item_code": "shared_premises_explanation",
            "description": "Explain the shared registered address.",
            "member_id": None,
        }
    ]
    provider = ScriptedProvider(
        [
            _call("get_relationship_evidence", candidate_id=candidate.candidate_id),
            _conclude(
                [real_id], recommendation="request_information", requested=requested
            ),
            _call("compare_submission_timeline", candidate_id=candidate.candidate_id),
            _conclude([real_id], recommendation="human_review"),
        ]
    )
    workflow = _workflow(candidates, applications, provider)

    paused = await workflow.run(candidate.candidate_id)

    assert paused.is_awaiting_information
    assert paused.decision is None
    assert paused.interrupt_payload["case_id"] == case_id_for(candidate.candidate_id)
    assert (
        paused.interrupt_payload["requested_information"][0]["item_code"]
        == "shared_premises_explanation"
    )

    resumed = await workflow.resume(
        paused.case_id,
        supplied_information={
            "item_code": "shared_premises_explanation",
            "content": "All applicants rent desks on one coworking floor.",
            "supplied_by": "merchant",
        },
    )

    # Same case, not a new one, and the earlier trace survived the interrupt.
    assert resumed.case_id == paused.case_id
    assert resumed.state.case_id == paused.state.case_id
    assert [event.node for event in resumed.state.workflow_trace] == [
        "assess",
        "investigate",
        "await_information",
        "investigate",
        "finalize",
    ]
    assert resumed.state.information_request_rounds == 1
    assert resumed.state.supplied_information[0].supplied_by == "merchant"
    assert resumed.decision.action is CandidateAction.HUMAN_REVIEW


async def test_information_requests_are_capped_so_a_case_always_terminates():
    candidates, applications = _world()
    candidate = _multi(candidates)
    real_id = candidate.evidence[0].evidence_id
    requested = [
        {"item_code": "again", "description": "Another decisive fact.", "member_id": None}
    ]
    asking_forever = [
        _call("get_relationship_evidence", candidate_id=candidate.candidate_id),
        _conclude([real_id], recommendation="request_information", requested=requested),
    ] * 4
    workflow = _workflow(
        candidates,
        applications,
        ScriptedProvider(asking_forever),
        config=WorkflowConfig(max_information_requests=1),
    )

    paused = await workflow.run(candidate.candidate_id)
    resumed = await workflow.resume(paused.case_id, supplied_information="Explanation.")

    assert not resumed.is_awaiting_information
    assert resumed.decision is not None
    assert "INFORMATION_REQUESTED" in resumed.decision.reason_codes
    assert resumed.decision.review_status is ReviewStatus.AWAITING_MERCHANT_INFORMATION


# -- checkpoint safety --------------------------------------------------


async def test_case_state_round_trips_through_json_and_hides_raw_identifiers():
    candidates, applications = _world()
    candidate = _multi(candidates)
    workflow = _workflow(candidates, applications, DeterministicInvestigator())

    run = await workflow.run(candidate.candidate_id)
    encoded = run.state.model_dump_json()
    restored = CaseState.model_validate(json.loads(encoded))

    assert restored.case_id == run.state.case_id
    assert restored.decision == run.state.decision
    assert restored.investigation.status is run.state.investigation.status

    for member_id in candidate.member_ids:
        application = applications[member_id]
        for secret in (
            application.owner_pan,
            application.bank_account,
            application.owner_phone,
            application.registered_address,
        ):
            assert secret not in encoded


async def test_stored_state_is_retrievable_by_case_id():
    candidates, applications = _world()
    candidate = _strong_multi(candidates)
    workflow = _workflow(candidates, applications, DeterministicInvestigator())

    run = await workflow.run(candidate.candidate_id)
    stored = await workflow.get_state(run.case_id)

    assert stored.case_id == run.case_id
    assert stored.candidate_id == candidate.candidate_id
    assert stored.decision is not None
