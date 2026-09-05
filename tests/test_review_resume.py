"""Phase 4: a paused case survives persistence and resumes as the same case."""
from __future__ import annotations

import pytest

from merchantshield.agent import (
    GraphScoringExpert,
    RoutedMoESystem,
    RulesExpert,
)
from merchantshield.investigation import BoundedInvestigator, DeterministicInvestigator
from merchantshield.review import (
    CaseEventType,
    CaseStatus,
    CaseStore,
    ConcurrentModification,
)
from merchantshield.review.service import NotAwaitingInformation
from merchantshield.runtime import build_runtime, load_world

pytestmark = pytest.mark.asyncio

# The deterministic investigator asks for information when the links are real
# but weak; 0.45 sits inside that band for every multi-member candidate.
UNCERTAIN_SCORE = 0.45


@pytest.fixture(scope="module")
def world():
    return load_world()


async def paused_case(world):
    runtime = await build_runtime(
        world=world,
        force_offline_investigator=True,
        router=RoutedMoESystem(
            rules_expert=RulesExpert(),
            graph_expert=GraphScoringExpert(lambda _: UNCERTAIN_SCORE),
        ),
        investigator=BoundedInvestigator(provider=DeterministicInvestigator()),
    )
    for candidate in sorted(
        runtime.world.candidates, key=lambda item: -len(item.member_ids)
    ):
        view = await runtime.service.open_case(candidate.candidate_id)
        if view.status is CaseStatus.AWAITING_INFORMATION:
            return runtime, view
    pytest.skip("no candidate produced an information request in this dataset")


async def test_a_paused_case_is_persisted_with_its_request(world):
    runtime, view = await paused_case(world)

    assert view.status is CaseStatus.AWAITING_INFORMATION
    assert view.requested_information
    assert view.human is None
    assert any(
        event.event_type is CaseEventType.INFORMATION_REQUESTED for event in view.events
    )

    # The request survives a completely separate store instance.
    reloaded = CaseStore().get(view.case_id)
    assert [item.item_code for item in reloaded.requested_information] == [
        item.item_code for item in view.requested_information
    ]


async def test_supplying_information_resumes_the_same_case(world):
    runtime, view = await paused_case(world)
    item_code = view.requested_information[0].item_code

    resumed = await runtime.service.supply_information(
        view.case_id,
        expected_version=view.version,
        items=[
            {
                "item_code": item_code,
                "content": "All applicants rent desks on one coworking floor.",
            }
        ],
        supplied_by="merchant",
    )

    assert resumed.case_id == view.case_id
    assert resumed.status is not CaseStatus.AWAITING_INFORMATION
    assert resumed.version == view.version + 1
    # The earlier trace is preserved, not replaced by a new case's trace.
    types = [event.event_type for event in resumed.events]
    assert types.index(CaseEventType.INFORMATION_REQUESTED) < types.index(
        CaseEventType.INFORMATION_SUPPLIED
    )
    assert CaseEventType.CASE_OPENED in types
    supplied = next(
        event
        for event in resumed.events
        if event.event_type is CaseEventType.INFORMATION_SUPPLIED
    )
    assert supplied.payload["items"][0]["item_code"] == item_code
    assert supplied.payload["items"][0]["supplied_by"] == "merchant"


async def test_supplying_information_needs_the_current_version(world):
    runtime, view = await paused_case(world)
    with pytest.raises(ConcurrentModification):
        await runtime.service.supply_information(
            view.case_id,
            expected_version=view.version + 5,
            items=[{"item_code": "x_code", "content": "late"}],
            supplied_by="merchant",
        )


async def test_information_cannot_be_supplied_to_a_case_that_did_not_ask(world):
    runtime = await build_runtime(world=world, force_offline_investigator=True)
    candidate = runtime.world.candidates[0]
    view = await runtime.service.open_case(candidate.candidate_id)
    with pytest.raises(NotAwaitingInformation):
        await runtime.service.supply_information(
            view.case_id,
            expected_version=view.version,
            items=[{"item_code": "unsolicited", "content": "here you go"}],
            supplied_by="merchant",
        )


async def test_failed_resume_does_not_commit_a_partial_information_event(
    world, monkeypatch
):
    runtime, view = await paused_case(world)

    async def fail_resume(*args, **kwargs):
        raise RuntimeError("injected resume failure")

    monkeypatch.setattr(runtime.workflow, "resume", fail_resume)
    with pytest.raises(RuntimeError, match="injected resume failure"):
        await runtime.service.supply_information(
            view.case_id,
            expected_version=view.version,
            items=[{"item_code": "shared_premises", "content": "evidence"}],
            supplied_by="merchant",
        )

    unchanged = runtime.store.get(view.case_id)
    assert unchanged.version == view.version
    assert all(
        event.event_type is not CaseEventType.INFORMATION_SUPPLIED
        for event in unchanged.events
    )
