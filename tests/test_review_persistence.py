"""Phase 4: append-only events, split decisions, concurrency and idempotency."""
from __future__ import annotations

from datetime import datetime, timezone

import pytest
from sqlalchemy import select

from merchantshield.agent.contracts import CandidateAction
from merchantshield.db.database import SessionLocal
from merchantshield.db.models import AppendOnlyViolation, DBCase, DBCaseEvent
from merchantshield.investigation.memory import PriorOutcome
from merchantshield.review import (
    ActorKind,
    CaseEventType,
    CaseNotFound,
    CaseStatus,
    CaseStore,
    ConcurrentModification,
    HumanDecision,
    InvalidTransition,
    ResolutionRequest,
    ReviewReasonCode,
)
from merchantshield.runtime import build_runtime, load_world
from merchantshield.workflow.state import ReviewStatus

pytestmark = pytest.mark.asyncio

HOLD_REASON = (ReviewReasonCode.SHARED_SETTLEMENT_ACCOUNT,)
CLEAR_REASON = (ReviewReasonCode.VERIFIED_FRANCHISE_AGREEMENT,)


@pytest.fixture(scope="module")
def world():
    return load_world()


async def runtime_with(world):
    return await build_runtime(world=world, force_offline_investigator=True)


async def open_review_case(world):
    """Open the highest-risk candidate; it is always a human-review case."""
    runtime = await runtime_with(world)
    for candidate in sorted(
        runtime.world.candidates, key=lambda item: -len(item.member_ids)
    ):
        view = await runtime.service.open_case(candidate.candidate_id)
        if view.status is CaseStatus.PENDING_REVIEW:
            return runtime, view
    raise AssertionError("expected at least one case to require human review")


# ------------------------------------------------------------------ opening


async def test_opening_a_case_persists_the_automated_decision(world):
    runtime, view = await open_review_case(world)

    assert view.status is CaseStatus.PENDING_REVIEW
    assert view.automated.action is CandidateAction.HUMAN_REVIEW
    assert view.automated.review_status is ReviewStatus.PENDING_HUMAN_REVIEW
    assert view.automated.reason_codes
    assert view.automated.policy_version
    assert view.human is None
    assert view.version == 1

    # The durable record survives a fresh store built on the same database.
    reloaded = CaseStore().get(view.case_id)
    assert reloaded.automated.model_dump() == view.automated.model_dump()


async def test_reopening_the_same_candidate_is_idempotent(world):
    runtime, first = await open_review_case(world)
    second = await runtime.service.open_case(first.candidate_id)

    assert second.case_id == first.case_id
    assert second.version == first.version
    assert len(second.events) == len(first.events)

    with SessionLocal() as session:
        assert (
            session.execute(
                select(DBCase).where(DBCase.candidate_id == first.candidate_id)
            )
            .scalars()
            .all()
            .__len__()
            == 1
        )


async def test_a_second_idempotency_key_for_one_candidate_is_refused(world):
    from merchantshield.review.service import CaseAlreadyOpen

    runtime, first = await open_review_case(world)
    with pytest.raises(CaseAlreadyOpen):
        await runtime.service.open_case(first.candidate_id, idempotency_key="different")


# ------------------------------------------------------------------- events


async def test_the_event_log_is_append_only(world):
    runtime, view = await open_review_case(world)
    assert [event.sequence for event in view.events] == list(
        range(1, len(view.events) + 1)
    )
    assert view.events[0].event_type is CaseEventType.CASE_OPENED
    assert all(event.actor_kind is ActorKind.SYSTEM for event in view.events)

    with SessionLocal() as session:
        row = session.execute(
            select(DBCaseEvent).where(DBCaseEvent.case_id == view.case_id)
        ).scalars().first()
        row.payload = {"tampered": True}
        with pytest.raises(AppendOnlyViolation):
            session.flush()
        session.rollback()

        row = session.execute(
            select(DBCaseEvent).where(DBCaseEvent.case_id == view.case_id)
        ).scalars().first()
        session.delete(row)
        with pytest.raises(AppendOnlyViolation):
            session.flush()
        session.rollback()


async def test_resolution_records_the_automated_decision_it_overrode(world):
    runtime, view = await open_review_case(world)
    claimed = runtime.service.claim(view.case_id, "reviewer_a", view.version)
    resolved = runtime.service.resolve(
        view.case_id,
        request=ResolutionRequest(
            reviewer_id="reviewer_a",
            decision=HumanDecision.APPROVE_ONBOARDING,
            outcome=PriorOutcome.LEGITIMATE_FRANCHISE,
            reason_codes=CLEAR_REASON,
            notes="Franchise agreement supplied and verified.",
        ),
        expected_version=claimed.version,
    )
    decision_event = next(
        event
        for event in resolved.events
        if event.event_type is CaseEventType.HUMAN_DECISION_RECORDED
    )
    assert decision_event.payload["automated_action"] == "human_review"
    assert decision_event.payload["automated_reason_codes"]
    assert decision_event.actor_kind is ActorKind.HUMAN


# ------------------------------------------------- separated decision records


async def test_a_human_approval_does_not_rewrite_the_automated_decision(world):
    runtime, view = await open_review_case(world)
    claimed = runtime.service.claim(view.case_id, "reviewer_a", view.version)
    resolved = runtime.service.resolve(
        view.case_id,
        request=ResolutionRequest(
            reviewer_id="reviewer_a",
            decision=HumanDecision.APPROVE_ONBOARDING,
            outcome=PriorOutcome.LEGITIMATE_COWORKING,
            reason_codes=(ReviewReasonCode.VERIFIED_COWORKING_TENANCY,),
        ),
        expected_version=claimed.version,
    )

    assert resolved.status is CaseStatus.RESOLVED
    assert resolved.human is not None
    assert resolved.human.decision is HumanDecision.APPROVE_ONBOARDING
    # The machine's verdict is still exactly what it was.
    assert resolved.automated.action is CandidateAction.HUMAN_REVIEW
    assert resolved.automated.reason_codes == view.automated.reason_codes


async def test_an_approval_requires_a_verified_clearing_reason(world):
    with pytest.raises(ValueError, match="clearing reason code"):
        ResolutionRequest(
            reviewer_id="reviewer_a",
            decision=HumanDecision.APPROVE_ONBOARDING,
            outcome=PriorOutcome.INCONCLUSIVE,
            reason_codes=(ReviewReasonCode.INSUFFICIENT_EVIDENCE,),
        )


async def test_a_confirmed_ring_may_not_be_approved(world):
    with pytest.raises(ValueError, match="confirmed ring"):
        ResolutionRequest(
            reviewer_id="reviewer_a",
            decision=HumanDecision.APPROVE_ONBOARDING,
            outcome=PriorOutcome.CONFIRMED_RING,
            reason_codes=CLEAR_REASON,
        )


async def test_a_cleared_case_cannot_be_resolved_by_a_human(world):
    runtime = await runtime_with(world)
    cleared = None
    for candidate in runtime.world.candidates:
        view = await runtime.service.open_case(candidate.candidate_id)
        if view.status is CaseStatus.CLEARED:
            cleared = view
            break
    assert cleared is not None
    with pytest.raises(InvalidTransition, match="cleared automatically"):
        runtime.service.resolve(
            cleared.case_id,
            request=ResolutionRequest(
                reviewer_id="reviewer_a",
                decision=HumanDecision.KEEP_ON_HOLD,
                outcome=PriorOutcome.INCONCLUSIVE,
                reason_codes=(ReviewReasonCode.INSUFFICIENT_EVIDENCE,),
            ),
            expected_version=cleared.version,
        )


# --------------------------------------------------------------- concurrency


async def test_a_stale_version_is_rejected_not_merged(world):
    runtime, view = await open_review_case(world)
    runtime.service.claim(view.case_id, "reviewer_a", view.version)

    with pytest.raises(ConcurrentModification) as excinfo:
        runtime.service.claim(view.case_id, "reviewer_b", view.version)
    assert excinfo.value.expected_version == view.version
    assert excinfo.value.actual_version == view.version + 1


async def test_two_reviewers_cannot_hold_the_same_case(world):
    runtime, view = await open_review_case(world)
    claimed = runtime.service.claim(view.case_id, "reviewer_a", view.version)
    with pytest.raises(InvalidTransition, match="already claimed"):
        runtime.service.claim(view.case_id, "reviewer_b", claimed.version)


async def test_releasing_returns_a_case_to_the_queue(world):
    runtime, view = await open_review_case(world)
    claimed = runtime.service.claim(view.case_id, "reviewer_a", view.version)
    released = runtime.service.release(view.case_id, "reviewer_a", claimed.version)

    assert released.status is CaseStatus.PENDING_REVIEW
    assert released.claimed_by is None
    reclaimed = runtime.service.claim(view.case_id, "reviewer_b", released.version)
    assert reclaimed.claimed_by == "reviewer_b"


async def test_a_reviewer_cannot_resolve_a_case_another_reviewer_holds(world):
    runtime, view = await open_review_case(world)
    claimed = runtime.service.claim(view.case_id, "reviewer_a", view.version)
    with pytest.raises(InvalidTransition, match="claimed by"):
        runtime.service.resolve(
            view.case_id,
            request=ResolutionRequest(
                reviewer_id="reviewer_b",
                decision=HumanDecision.KEEP_ON_HOLD,
                outcome=PriorOutcome.CONFIRMED_RING,
                reason_codes=HOLD_REASON,
            ),
            expected_version=claimed.version,
        )


async def test_a_resolved_case_cannot_be_resolved_twice(world):
    runtime, view = await open_review_case(world)
    resolved = runtime.service.resolve(
        view.case_id,
        request=ResolutionRequest(
            reviewer_id="reviewer_a",
            decision=HumanDecision.KEEP_ON_HOLD,
            outcome=PriorOutcome.CONFIRMED_RING,
            reason_codes=HOLD_REASON,
        ),
        expected_version=view.version,
    )
    with pytest.raises(InvalidTransition, match="already resolved"):
        runtime.service.resolve(
            view.case_id,
            request=ResolutionRequest(
                reviewer_id="reviewer_a",
                decision=HumanDecision.KEEP_ON_HOLD,
                outcome=PriorOutcome.CONFIRMED_RING,
                reason_codes=HOLD_REASON,
            ),
            expected_version=resolved.version,
        )


# ------------------------------------------------------------- case memory


async def test_a_resolution_teaches_durable_case_memory(world):
    runtime, view = await open_review_case(world)
    assert len(runtime.service.memory()) == 0

    runtime.service.resolve(
        view.case_id,
        request=ResolutionRequest(
            reviewer_id="reviewer_a",
            decision=HumanDecision.KEEP_ON_HOLD,
            outcome=PriorOutcome.CONFIRMED_RING,
            reason_codes=HOLD_REASON,
            notes="Confirmed shared settlement account.",
        ),
        expected_version=view.version,
    )

    memory = runtime.service.memory()
    assert len(memory) > 0
    candidate = runtime.world.by_id[view.candidate_id]
    hashes = [edge.shared_value_hash for edge in candidate.evidence]
    found = memory.lookup(hashes)
    assert found
    assert all(record.outcome is PriorOutcome.CONFIRMED_RING for record in found)
    # Only hashed infrastructure is stored, never a raw identifier.
    assert all(
        record.infrastructure_hash in set(hashes) for record in found
    )
    # A freshly built runtime sees the same history.
    reloaded = await runtime_with(world)
    assert len(reloaded.service.memory()) == len(memory)


async def test_missing_cases_are_reported_not_invented(world):
    runtime = await runtime_with(world)
    with pytest.raises(CaseNotFound):
        runtime.service.get("case-does-not-exist")


# ------------------------------------------------------------------- queue


async def test_the_queue_lists_and_filters_cases(world):
    runtime = await runtime_with(world)
    for candidate in runtime.world.candidates[:15]:
        await runtime.service.open_case(candidate.candidate_id)

    total, rows = runtime.service.list_cases(status=CaseStatus.PENDING_REVIEW)
    assert total == len(rows)
    assert all(row.status is CaseStatus.PENDING_REVIEW for row in rows)
    # Highest risk first, so a reviewer starts where it matters.
    scores = [row.automated.primary_risk_score or 0.0 for row in rows]
    assert scores == sorted(scores, reverse=True)

    counts = runtime.service.queue_counts()
    assert counts[CaseStatus.PENDING_REVIEW.value] == total
    assert sum(counts.values()) == 15
