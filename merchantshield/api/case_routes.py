"""Case and review REST surface.

Concurrency is explicit here: every mutating endpoint requires the version the
caller read, and a mismatch returns 409 with the current version rather than
silently winning. Nothing in this router can approve a merchant on its own --
approval only exists as a recorded human decision.
"""
from __future__ import annotations

from typing import Any, Dict, List, Optional

from fastapi import APIRouter, Body, HTTPException, Query, Response
from pydantic import BaseModel, Field, ValidationError

from ..investigation.memory import PriorOutcome
from ..review.contracts import (
    CaseStatus,
    HumanDecision,
    ResolutionRequest,
    ReviewReasonCode,
)
from ..gateway import GatewayUnavailable
from ..review.service import (
    CaseAlreadyOpen,
    HandoffNotPermitted,
    NotAwaitingInformation,
)
from ..review.store import CaseNotFound, ConcurrentModification, InvalidTransition
from ..runtime import get_runtime

router = APIRouter(prefix="/api/v1", tags=["cases"])


# ------------------------------------------------------------------ payloads


class OpenCaseRequest(BaseModel):
    candidate_id: str = Field(..., min_length=2, max_length=100)
    idempotency_key: Optional[str] = Field(None, min_length=4, max_length=120)


class ClaimRequest(BaseModel):
    reviewer_id: str = Field(..., min_length=2, max_length=80)
    expected_version: int = Field(..., ge=1)


class SuppliedItem(BaseModel):
    item_code: str = Field(..., min_length=2, max_length=60)
    content: str = Field(..., min_length=1, max_length=1000)
    supplied_by: Optional[str] = Field(None, min_length=2, max_length=80)


class SupplyInformationRequest(BaseModel):
    expected_version: int = Field(..., ge=1)
    supplied_by: str = Field("reviewer", min_length=2, max_length=80)
    items: List[SuppliedItem] = Field(..., min_length=1, max_length=10)


class HandoffRequestBody(BaseModel):
    actor: str = Field(..., min_length=2, max_length=80)
    expected_version: int = Field(..., ge=1)


class ResolveRequest(BaseModel):
    reviewer_id: str = Field(..., min_length=2, max_length=80)
    expected_version: int = Field(..., ge=1)
    decision: HumanDecision
    outcome: PriorOutcome
    reason_codes: List[ReviewReasonCode] = Field(..., min_length=1, max_length=8)
    notes: str = Field("", max_length=2000)


# ------------------------------------------------------------------- routes


@router.get("/reference/review-vocabulary", summary="Allowed decisions and reason codes")
def review_vocabulary() -> Dict[str, Any]:
    """The controlled vocabulary a reviewer UI must offer. No free-text verdicts."""
    from ..review.contracts import CLEARING_REASON_CODES

    return {
        "decisions": [item.value for item in HumanDecision],
        "outcomes": [item.value for item in PriorOutcome],
        "reason_codes": [item.value for item in ReviewReasonCode],
        "clearing_reason_codes": sorted(item.value for item in CLEARING_REASON_CODES),
        "statuses": [item.value for item in CaseStatus],
        "rule": (
            "approve_onboarding requires at least one clearing reason code and "
            "may not be paired with a confirmed_ring outcome"
        ),
    }


@router.get("/candidates", summary="Candidate ring groups available for assessment")
async def list_candidates(
    min_members: int = Query(1, ge=1, le=50),
    limit: int = Query(100, ge=1, le=500),
) -> Dict[str, Any]:
    runtime = await get_runtime()
    rows = [
        {
            "candidate_id": candidate.candidate_id,
            "member_count": len(candidate.member_ids),
            "evidence_count": len(candidate.evidence),
            "attributes": sorted({edge.attribute for edge in candidate.evidence}),
            "has_case": runtime.store.exists(_case_id(candidate.candidate_id)),
        }
        for candidate in runtime.world.candidates
        if len(candidate.member_ids) >= min_members
    ]
    rows.sort(key=lambda item: (-item["member_count"], item["candidate_id"]))
    return {
        "total": len(rows),
        "candidates": rows[:limit],
        "labels": runtime.labels,
    }


@router.post("/cases", status_code=201, summary="Open (or return) a case for a candidate")
async def open_case(
    response: Response, payload: OpenCaseRequest = Body(...)
) -> Dict[str, Any]:
    runtime = await get_runtime()
    existed = runtime.store.exists(_case_id(payload.candidate_id))
    try:
        view = await runtime.service.open_case(
            payload.candidate_id, idempotency_key=payload.idempotency_key
        )
    except KeyError as exc:
        raise HTTPException(status_code=404, detail=f"Unknown candidate: {exc}") from exc
    except CaseAlreadyOpen as exc:
        raise HTTPException(status_code=409, detail=str(exc)) from exc
    # A replayed request returns the stored case, not a second assessment.
    response.status_code = 200 if existed else 201
    return _case_payload(view, runtime)


@router.get("/cases", summary="Review queue")
async def list_cases(
    status: Optional[CaseStatus] = Query(None),
    claimed_by: Optional[str] = Query(None, min_length=2, max_length=80),
    unclaimed_only: bool = Query(False),
    min_risk: Optional[float] = Query(None, ge=0.0, le=1.0),
    limit: int = Query(50, ge=1, le=500),
    offset: int = Query(0, ge=0),
) -> Dict[str, Any]:
    runtime = await get_runtime()
    total, rows = runtime.service.list_cases(
        status=status,
        claimed_by=claimed_by,
        unclaimed_only=unclaimed_only,
        min_risk=min_risk,
        limit=limit,
        offset=offset,
    )
    return {
        "total": total,
        "count": len(rows),
        "cases": [row.model_dump(mode="json") for row in rows],
        "labels": runtime.labels,
    }


@router.get("/cases/stats", summary="Queue counters")
async def case_stats() -> Dict[str, Any]:
    runtime = await get_runtime()
    counts = dict(runtime.service.queue_counts())
    return {
        "counts": counts,
        "open_for_review": counts.get(CaseStatus.PENDING_REVIEW.value, 0)
        + counts.get(CaseStatus.CLAIMED.value, 0),
        "prior_outcomes_recorded": len(runtime.service.memory()),
        "labels": runtime.labels,
    }


@router.get("/cases/{case_id}", summary="Full case record")
async def get_case(case_id: str) -> Dict[str, Any]:
    runtime = await get_runtime()
    return _case_payload(_require(runtime, case_id), runtime)


@router.get("/cases/{case_id}/events", summary="Append-only audit log")
async def case_events(case_id: str) -> Dict[str, Any]:
    runtime = await get_runtime()
    view = _require(runtime, case_id)
    return {
        "case_id": view.case_id,
        "count": len(view.events),
        "events": [event.model_dump(mode="json") for event in view.events],
    }


@router.post("/cases/{case_id}/claim", summary="Claim a case for one reviewer")
async def claim_case(case_id: str, payload: ClaimRequest = Body(...)) -> Dict[str, Any]:
    runtime = await get_runtime()
    try:
        view = runtime.service.claim(case_id, payload.reviewer_id, payload.expected_version)
    except CaseNotFound as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc
    except ConcurrentModification as exc:
        raise _conflict(exc)
    except InvalidTransition as exc:
        raise HTTPException(status_code=409, detail=str(exc)) from exc
    return _case_payload(view, runtime)


@router.post("/cases/{case_id}/release", summary="Return a case to the queue")
async def release_case(case_id: str, payload: ClaimRequest = Body(...)) -> Dict[str, Any]:
    runtime = await get_runtime()
    try:
        view = runtime.service.release(
            case_id, payload.reviewer_id, payload.expected_version
        )
    except CaseNotFound as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc
    except ConcurrentModification as exc:
        raise _conflict(exc)
    except InvalidTransition as exc:
        raise HTTPException(status_code=409, detail=str(exc)) from exc
    return _case_payload(view, runtime)


@router.post("/cases/{case_id}/information", summary="Supply requested evidence and resume")
async def supply_information(
    case_id: str, payload: SupplyInformationRequest = Body(...)
) -> Dict[str, Any]:
    runtime = await get_runtime()
    try:
        view = await runtime.service.supply_information(
            case_id,
            expected_version=payload.expected_version,
            items=[item.model_dump(exclude_none=True) for item in payload.items],
            supplied_by=payload.supplied_by,
        )
    except CaseNotFound as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc
    except ConcurrentModification as exc:
        raise _conflict(exc)
    except NotAwaitingInformation as exc:
        raise HTTPException(status_code=409, detail=str(exc)) from exc
    return _case_payload(view, runtime)


@router.post("/cases/{case_id}/resolve", summary="Record the final human decision")
async def resolve_case(case_id: str, payload: ResolveRequest = Body(...)) -> Dict[str, Any]:
    runtime = await get_runtime()
    try:
        request = ResolutionRequest(
            reviewer_id=payload.reviewer_id,
            decision=payload.decision,
            outcome=payload.outcome,
            reason_codes=tuple(payload.reason_codes),
            notes=payload.notes,
        )
    except ValidationError as exc:
        raise HTTPException(status_code=422, detail=_readable(exc)) from exc
    try:
        view = runtime.service.resolve(
            case_id, request=request, expected_version=payload.expected_version
        )
    except CaseNotFound as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc
    except ConcurrentModification as exc:
        raise _conflict(exc)
    except InvalidTransition as exc:
        raise HTTPException(status_code=409, detail=str(exc)) from exc
    return _case_payload(view, runtime)


@router.post("/cases/{case_id}/handoff", summary="Hand an approved group to the gateway")
async def hand_off_case(
    case_id: str, payload: HandoffRequestBody = Body(...)
) -> Dict[str, Any]:
    """Only a case a human approved can reach a gateway. Never automatic."""
    runtime = await get_runtime()
    try:
        view = runtime.service.hand_off(
            case_id, expected_version=payload.expected_version, actor=payload.actor
        )
    except CaseNotFound as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc
    except ConcurrentModification as exc:
        raise _conflict(exc)
    except HandoffNotPermitted as exc:
        raise HTTPException(status_code=409, detail=str(exc)) from exc
    except GatewayUnavailable as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from exc
    return _case_payload(view, runtime)


# ------------------------------------------------------------------ helpers


def _case_id(candidate_id: str) -> str:
    from ..workflow.state import case_id_for

    return case_id_for(candidate_id)


def _require(runtime, case_id: str):
    try:
        return runtime.service.get(case_id)
    except CaseNotFound as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc


def _conflict(exc: ConcurrentModification) -> HTTPException:
    return HTTPException(
        status_code=409,
        detail={
            "error": "concurrent_modification",
            "message": str(exc),
            "expected_version": exc.expected_version,
            "current_version": exc.actual_version,
        },
    )


def _readable(exc: ValidationError) -> str:
    parts = []
    for err in exc.errors():
        field = ".".join(str(part) for part in err["loc"])
        parts.append(f"{field}: {err['msg']}" if field else err["msg"])
    return "; ".join(parts)


def _case_payload(view, runtime) -> Dict[str, Any]:
    """One case, with the ring context a reviewer needs beside the decisions."""
    payload = view.model_dump(mode="json")
    candidate = runtime.world.by_id.get(view.candidate_id)
    if candidate is not None:
        payload["candidate"] = {
            "candidate_id": candidate.candidate_id,
            "member_ids": list(candidate.member_ids),
            "member_count": len(candidate.member_ids),
            "evidence": [edge.to_dict() for edge in candidate.evidence],
            "pair_strengths": [
                {"left_id": left, "right_id": right, "strength": strength}
                for left, right, strength in candidate.pair_strengths
            ],
            "features": candidate.features.to_dict(),
        }
        payload["members"] = [
            _redacted_member(member_id, runtime.world.applications.get(member_id))
            for member_id in candidate.member_ids
        ]
    payload["labels"] = runtime.labels
    return payload


def _redacted_member(member_id: str, application) -> Dict[str, Any]:
    """Reviewer-visible profile. Identifier values stay out of the response."""
    if application is None:
        return {"member_id": member_id, "available": False}
    return {
        "member_id": member_id,
        "available": True,
        "business_name": application.business_name,
        "business_type": application.business_type.value,
        "owner_name": application.owner_name,
        "registered_address": application.registered_address,
        "submitted_at": application.submitted_at.isoformat()
        if getattr(application, "submitted_at", None)
        else None,
        "website_url": application.website_url,
    }
