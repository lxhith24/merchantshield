"""Phase 5: the onboarding hand-off boundary is human-gated and clearly labelled."""
from __future__ import annotations

import pytest

from merchantshield.gateway import (
    GatewayMode,
    GatewayUnavailable,
    HandoffRequest,
    RazorpayTestGateway,
    SimulatedOnboardingGateway,
    select_gateway,
)
from merchantshield.investigation.memory import PriorOutcome
from merchantshield.review import (
    CaseEventType,
    HandoffNotPermitted,
    HumanDecision,
    ResolutionRequest,
    ReviewReasonCode,
)
from merchantshield.review.contracts import CaseStatus
from merchantshield.runtime import build_runtime, load_world

@pytest.fixture(scope="module")
def world():
    return load_world()


def a_request(case_id: str = "case-abc123") -> HandoffRequest:
    return HandoffRequest(
        case_id=case_id,
        candidate_id="cand-1",
        member_ids=("m1", "m2"),
        approved_by="reviewer_a",
        reason_codes=("verified_franchise_agreement",),
    )


def test_the_default_gateway_is_simulated(monkeypatch):
    monkeypatch.delenv("RAZORPAY_PARTNER_TEST_ENABLED", raising=False)
    assert select_gateway().mode is GatewayMode.SIMULATED


def test_the_simulated_gateway_is_idempotent_per_case():
    gateway = SimulatedOnboardingGateway()
    first = gateway.submit(a_request())
    second = gateway.submit(a_request())
    assert first.reference == second.reference
    assert first.is_simulated
    assert len(gateway.receipts()) == 1


def test_an_unconfigured_razorpay_adapter_refuses_rather_than_falling_back():
    gateway = RazorpayTestGateway(key_id="", key_secret="", enabled=False)
    assert not gateway.is_available
    with pytest.raises(GatewayUnavailable, match="not usable"):
        gateway.submit(a_request())


def test_a_live_razorpay_key_is_not_accepted():
    gateway = RazorpayTestGateway(
        key_id="rzp_live_abc", key_secret="secret", enabled=True
    )
    assert not gateway.is_available
    assert "not a test key" in gateway.unavailable_reason()


def test_configured_test_credentials_still_do_not_invent_an_activation():
    gateway = RazorpayTestGateway(
        key_id="rzp_test_abc", key_secret="secret", enabled=True
    )
    assert gateway.is_available
    with pytest.raises(GatewayUnavailable, match="not provisioned|not attempted"):
        gateway.submit(a_request())


async def review_case(world):
    runtime = await build_runtime(world=world, force_offline_investigator=True)
    for candidate in sorted(
        runtime.world.candidates, key=lambda item: -len(item.member_ids)
    ):
        view = await runtime.service.open_case(candidate.candidate_id)
        if view.status is CaseStatus.PENDING_REVIEW:
            return runtime, view
    raise AssertionError("expected a review case")


@pytest.mark.asyncio
async def test_an_unapproved_case_cannot_reach_the_gateway(world):
    runtime, view = await review_case(world)
    with pytest.raises(HandoffNotPermitted):
        runtime.service.hand_off(view.case_id, expected_version=view.version, actor="ops")

    held = runtime.service.resolve(
        view.case_id,
        request=ResolutionRequest(
            reviewer_id="reviewer_a",
            decision=HumanDecision.KEEP_ON_HOLD,
            outcome=PriorOutcome.CONFIRMED_RING,
            reason_codes=(ReviewReasonCode.SHARED_SETTLEMENT_ACCOUNT,),
        ),
        expected_version=view.version,
    )
    with pytest.raises(HandoffNotPermitted):
        runtime.service.hand_off(
            view.case_id, expected_version=held.version, actor="ops"
        )


@pytest.mark.asyncio
async def test_an_approved_case_is_handed_off_and_receipted(world):
    runtime, view = await review_case(world)
    approved = runtime.service.resolve(
        view.case_id,
        request=ResolutionRequest(
            reviewer_id="reviewer_a",
            decision=HumanDecision.APPROVE_ONBOARDING,
            outcome=PriorOutcome.LEGITIMATE_FRANCHISE,
            reason_codes=(ReviewReasonCode.VERIFIED_FRANCHISE_AGREEMENT,),
            notes="Franchise agreement verified.",
        ),
        expected_version=view.version,
    )
    handed = runtime.service.hand_off(
        view.case_id, expected_version=approved.version, actor="reviewer_a"
    )

    assert handed.handoff is not None
    assert handed.handoff.mode == "SIMULATED"
    assert handed.handoff.reference.startswith("sim_")
    assert any(
        event.event_type is CaseEventType.ONBOARDING_HANDOFF_RECORDED
        for event in handed.events
    )
    # A repeated hand-off returns the same receipt rather than activating twice.
    again = runtime.service.hand_off(
        view.case_id, expected_version=handed.version, actor="reviewer_a"
    )
    assert again.handoff.reference == handed.handoff.reference
    assert again.version == handed.version
