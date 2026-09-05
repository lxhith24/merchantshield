"""Durable, append-only case persistence with optimistic concurrency.

Every mutation follows the same shape: read the case, check the caller's
`expected_version`, write the new values and the audit event in one
transaction, bump the version. A stale version is rejected, never merged.
"""
from __future__ import annotations

from datetime import datetime, timezone
from typing import Any, Callable, Iterable, List, Mapping, Optional, Sequence, Tuple

from sqlalchemy import func, select, update
from sqlalchemy.exc import IntegrityError
from sqlalchemy.orm import Session

from ..db.database import SessionLocal
from ..db.models import DBCase, DBCaseEvent, DBPriorOutcome
from ..investigation.memory import PriorCaseRecord, PriorOutcome
from .contracts import (
    ActorKind,
    AutomatedDecisionView,
    CaseEvent,
    CaseEventType,
    CaseStatus,
    CaseSummary,
    CaseView,
    HandoffView,
    HumanDecisionView,
    RequestedInformationView,
)


class CaseNotFound(LookupError):
    """The requested case does not exist."""


class ConcurrentModification(RuntimeError):
    """The caller's `expected_version` no longer matches the stored case.

    Carries the current version so a UI can re-read and retry deliberately.
    """

    def __init__(self, case_id: str, expected: int, actual: int) -> None:
        super().__init__(
            f"case {case_id} was modified concurrently: "
            f"expected version {expected}, found {actual}"
        )
        self.case_id = case_id
        self.expected_version = expected
        self.actual_version = actual


class InvalidTransition(RuntimeError):
    """The requested lifecycle transition is not legal from the current status."""


def _utcnow() -> datetime:
    return datetime.now(timezone.utc)


def _aware(value: Optional[datetime]) -> Optional[datetime]:
    """SQLite hands back naive datetimes; case records are always UTC."""
    if value is None:
        return None
    return value if value.tzinfo else value.replace(tzinfo=timezone.utc)


class CaseStore:
    """SQL-backed store for cases, their events, and prior human outcomes."""

    def __init__(self, session_factory: Callable[[], Session] = SessionLocal) -> None:
        self._session_factory = session_factory

    # -- reads ------------------------------------------------------------

    def get(self, case_id: str) -> CaseView:
        with self._session() as session:
            return self._view(session, self._require(session, case_id))

    def find_by_idempotency_key(self, key: str) -> Optional[CaseView]:
        with self._session() as session:
            row = session.execute(
                select(DBCase).where(DBCase.idempotency_key == key)
            ).scalar_one_or_none()
            return self._view(session, row) if row is not None else None

    def exists(self, case_id: str) -> bool:
        with self._session() as session:
            return session.get(DBCase, case_id) is not None

    def list_cases(
        self,
        *,
        status: Optional[CaseStatus] = None,
        claimed_by: Optional[str] = None,
        unclaimed_only: bool = False,
        min_risk: Optional[float] = None,
        limit: int = 50,
        offset: int = 0,
    ) -> Tuple[int, Tuple[CaseSummary, ...]]:
        with self._session() as session:
            stmt = select(DBCase)
            if status is not None:
                stmt = stmt.where(DBCase.status == status.value)
            if claimed_by is not None:
                stmt = stmt.where(DBCase.claimed_by == claimed_by)
            if unclaimed_only:
                stmt = stmt.where(DBCase.claimed_by.is_(None))
            if min_risk is not None:
                stmt = stmt.where(DBCase.primary_risk_score >= min_risk)

            total = session.execute(
                select(func.count()).select_from(stmt.subquery())
            ).scalar_one()
            rows = (
                session.execute(
                    stmt.order_by(
                        DBCase.primary_risk_score.desc().nullsfirst(),
                        DBCase.created_at.asc(),
                    )
                    .limit(limit)
                    .offset(offset)
                )
                .scalars()
                .all()
            )
            return int(total), tuple(self._summary(row) for row in rows)

    def events(self, case_id: str) -> Tuple[CaseEvent, ...]:
        with self._session() as session:
            self._require(session, case_id)
            return self._events(session, case_id)

    def queue_counts(self) -> Mapping[str, int]:
        with self._session() as session:
            rows = session.execute(
                select(DBCase.status, func.count(DBCase.case_id)).group_by(DBCase.status)
            ).all()
            counts = {status.value: 0 for status in CaseStatus}
            counts.update({str(status): int(count) for status, count in rows})
            return counts

    # -- writes -----------------------------------------------------------

    def open_case(
        self,
        *,
        case_id: str,
        candidate_id: str,
        idempotency_key: str,
        status: CaseStatus,
        automated: AutomatedDecisionView,
        member_count: int = 0,
        requested_information: Sequence[RequestedInformationView] = (),
        case_state: Mapping[str, Any],
        events: Sequence[Tuple[CaseEventType, str, ActorKind, Mapping[str, Any]]],
    ) -> CaseView:
        """Create a case. Idempotent: a repeated key returns the stored case."""
        with self._session() as session:
            existing = session.execute(
                select(DBCase).where(DBCase.idempotency_key == idempotency_key)
            ).scalar_one_or_none()
            if existing is not None:
                return self._view(session, existing)

            row = DBCase(
                case_id=case_id,
                candidate_id=candidate_id,
                idempotency_key=idempotency_key,
                status=status.value,
                version=1,
                member_count=int(member_count),
                automated_action=automated.action.value,
                automated_review_status=automated.review_status.value,
                automated_reason_codes=list(automated.reason_codes),
                primary_risk_score=automated.primary_risk_score,
                investigator_recommendation=(
                    automated.investigator_recommendation.value
                    if automated.investigator_recommendation
                    else None
                ),
                grounding_status=(
                    automated.grounding_status.value if automated.grounding_status else None
                ),
                policy_version=automated.policy_version,
                workflow_version=automated.workflow_version,
                requested_information=[
                    item.model_dump(mode="json") for item in requested_information
                ],
                case_state=dict(case_state),
            )
            session.add(row)
            try:
                session.flush()
            except IntegrityError:
                session.rollback()
                stored = session.execute(
                    select(DBCase).where(DBCase.idempotency_key == idempotency_key)
                ).scalar_one_or_none()
                if stored is None:
                    raise
                return self._view(session, stored)

            for event_type, actor, actor_kind, payload in events:
                self._append(session, case_id, event_type, actor, actor_kind, payload)
            session.commit()
            return self._view(session, self._require(session, case_id))

    def record_automated_progress(
        self,
        *,
        case_id: str,
        expected_version: int,
        status: CaseStatus,
        automated: AutomatedDecisionView,
        requested_information: Sequence[RequestedInformationView],
        case_state: Mapping[str, Any],
        events: Sequence[Tuple[CaseEventType, str, ActorKind, Mapping[str, Any]]],
    ) -> CaseView:
        """Persist a further workflow run (for example, after a resume)."""
        with self._session() as session:
            row = self._require(session, case_id)
            self._check_version(row, expected_version)
            row.status = status.value
            row.automated_action = automated.action.value
            row.automated_review_status = automated.review_status.value
            row.automated_reason_codes = list(automated.reason_codes)
            row.primary_risk_score = automated.primary_risk_score
            row.investigator_recommendation = (
                automated.investigator_recommendation.value
                if automated.investigator_recommendation
                else None
            )
            row.grounding_status = (
                automated.grounding_status.value if automated.grounding_status else None
            )
            row.policy_version = automated.policy_version
            row.workflow_version = automated.workflow_version
            row.requested_information = [
                item.model_dump(mode="json") for item in requested_information
            ]
            row.case_state = dict(case_state)
            for event_type, actor, actor_kind, payload in events:
                self._append(session, case_id, event_type, actor, actor_kind, payload)
            self._bump(session, row)
            session.commit()
            return self._view(session, self._require(session, case_id))

    def claim(self, case_id: str, reviewer_id: str, expected_version: int) -> CaseView:
        with self._session() as session:
            row = self._require(session, case_id)
            self._check_version(row, expected_version)
            status = CaseStatus(row.status)
            if status.is_terminal:
                raise InvalidTransition(f"case {case_id} is already {status.value}")
            if row.claimed_by is not None and row.claimed_by != reviewer_id:
                raise InvalidTransition(
                    f"case {case_id} is already claimed by {row.claimed_by}"
                )
            row.claimed_by = reviewer_id
            row.claimed_at = _utcnow()
            if status is CaseStatus.PENDING_REVIEW:
                row.status = CaseStatus.CLAIMED.value
            self._append(
                session,
                case_id,
                CaseEventType.CASE_CLAIMED,
                reviewer_id,
                ActorKind.HUMAN,
                {"previous_status": status.value},
            )
            self._bump(session, row)
            session.commit()
            return self._view(session, self._require(session, case_id))

    def release(self, case_id: str, reviewer_id: str, expected_version: int) -> CaseView:
        with self._session() as session:
            row = self._require(session, case_id)
            self._check_version(row, expected_version)
            if row.claimed_by is None:
                raise InvalidTransition(f"case {case_id} is not claimed")
            if row.claimed_by != reviewer_id:
                raise InvalidTransition(
                    f"case {case_id} is claimed by {row.claimed_by}, not {reviewer_id}"
                )
            row.claimed_by = None
            row.claimed_at = None
            if CaseStatus(row.status) is CaseStatus.CLAIMED:
                row.status = CaseStatus.PENDING_REVIEW.value
            self._append(
                session,
                case_id,
                CaseEventType.CASE_RELEASED,
                reviewer_id,
                ActorKind.HUMAN,
                {},
            )
            self._bump(session, row)
            session.commit()
            return self._view(session, self._require(session, case_id))

    def resolve(
        self,
        *,
        case_id: str,
        expected_version: int,
        human: HumanDecisionView,
        prior_records: Sequence[PriorCaseRecord] = (),
    ) -> CaseView:
        with self._session() as session:
            row = self._require(session, case_id)
            self._check_version(row, expected_version)
            status = CaseStatus(row.status)
            if status is CaseStatus.RESOLVED:
                raise InvalidTransition(f"case {case_id} is already resolved")
            if status is CaseStatus.CLEARED:
                raise InvalidTransition(
                    f"case {case_id} cleared automatically and needs no human decision"
                )
            if row.claimed_by is not None and row.claimed_by != human.reviewer_id:
                raise InvalidTransition(
                    f"case {case_id} is claimed by {row.claimed_by}, "
                    f"not {human.reviewer_id}"
                )
            row.status = CaseStatus.RESOLVED.value
            row.resolved_by = human.reviewer_id
            row.resolved_at = human.resolved_at
            row.human_decision = human.decision.value
            row.human_outcome = human.outcome.value
            row.human_reason_codes = [code.value for code in human.reason_codes]
            row.human_notes = human.notes or None
            self._append(
                session,
                case_id,
                CaseEventType.HUMAN_DECISION_RECORDED,
                human.reviewer_id,
                ActorKind.HUMAN,
                {
                    "decision": human.decision.value,
                    "outcome": human.outcome.value,
                    "reason_codes": [code.value for code in human.reason_codes],
                    "notes": human.notes,
                    # Kept beside the human decision so the audit log shows what
                    # the automated system had said at the moment of override.
                    "automated_action": row.automated_action,
                    "automated_reason_codes": list(row.automated_reason_codes or []),
                },
            )
            for record in prior_records:
                session.add(
                    DBPriorOutcome(
                        infrastructure_hash=record.infrastructure_hash,
                        attribute=record.attribute,
                        outcome=record.outcome.value,
                        reviewer_reference=record.reviewer_reference,
                        note=record.note,
                        resolved_at=record.resolved_at,
                    )
                )
            self._bump(session, row)
            session.commit()
            return self._view(session, self._require(session, case_id))

    def record_handoff(
        self,
        *,
        case_id: str,
        expected_version: int,
        reference: str,
        mode: str,
        accepted_at: datetime,
        actor: str,
        detail: Mapping[str, Any],
    ) -> CaseView:
        """Store the receipt for an approved group's onboarding hand-off."""
        with self._session() as session:
            row = self._require(session, case_id)
            self._check_version(row, expected_version)
            row.handoff_reference = reference
            row.handoff_mode = mode
            row.handoff_at = accepted_at
            self._append(
                session,
                case_id,
                CaseEventType.ONBOARDING_HANDOFF_RECORDED,
                actor,
                ActorKind.HUMAN,
                {"reference": reference, "mode": mode, **dict(detail)},
            )
            self._bump(session, row)
            session.commit()
            return self._view(session, self._require(session, case_id))

    def record_supplied_information(
        self,
        *,
        case_id: str,
        expected_version: int,
        actor: str,
        items: Sequence[Mapping[str, Any]],
    ) -> None:
        """Log follow-up evidence without changing the case's version.

        Called inside the same logical operation as the workflow resume, which
        does the version bump. Kept separate so the log records what arrived
        even if the resume itself then fails.
        """
        with self._session() as session:
            row = self._require(session, case_id)
            self._check_version(row, expected_version)
            self._append(
                session,
                case_id,
                CaseEventType.INFORMATION_SUPPLIED,
                actor,
                ActorKind.HUMAN,
                {"items": [dict(item) for item in items]},
            )
            session.commit()

    # -- prior-outcome memory ---------------------------------------------

    def prior_records(self) -> Tuple[PriorCaseRecord, ...]:
        with self._session() as session:
            rows = (
                session.execute(select(DBPriorOutcome).order_by(DBPriorOutcome.resolved_at))
                .scalars()
                .all()
            )
            return tuple(
                PriorCaseRecord(
                    infrastructure_hash=row.infrastructure_hash,
                    attribute=row.attribute,
                    outcome=PriorOutcome(row.outcome),
                    resolved_at=_aware(row.resolved_at),
                    reviewer_reference=row.reviewer_reference,
                    note=row.note or "",
                )
                for row in rows
            )

    def add_prior_records(self, records: Iterable[PriorCaseRecord]) -> int:
        with self._session() as session:
            added = 0
            for record in records:
                session.add(
                    DBPriorOutcome(
                        infrastructure_hash=record.infrastructure_hash,
                        attribute=record.attribute,
                        outcome=record.outcome.value,
                        reviewer_reference=record.reviewer_reference,
                        note=record.note,
                        resolved_at=record.resolved_at,
                    )
                )
                added += 1
            session.commit()
            return added

    # -- internals ---------------------------------------------------------

    def _session(self) -> Session:
        return self._session_factory()

    @staticmethod
    def _require(session: Session, case_id: str) -> DBCase:
        row = session.get(DBCase, case_id)
        if row is None:
            raise CaseNotFound(f"unknown case: {case_id}")
        return row

    @staticmethod
    def _check_version(row: DBCase, expected_version: int) -> None:
        if int(row.version) != int(expected_version):
            raise ConcurrentModification(row.case_id, expected_version, int(row.version))

    @staticmethod
    def _bump(session: Session, row: DBCase) -> None:
        """Increment the concurrency token conditionally, at the SQL level.

        Doing this as a guarded UPDATE (not `row.version += 1`) means two
        transactions racing on the same version cannot both succeed even if
        they both passed the in-Python check.
        """
        result = session.execute(
            update(DBCase)
            .where(DBCase.case_id == row.case_id, DBCase.version == row.version)
            .values(version=row.version + 1, updated_at=_utcnow())
        )
        if result.rowcount != 1:
            session.rollback()
            raise ConcurrentModification(row.case_id, int(row.version), -1)

    @staticmethod
    def _append(
        session: Session,
        case_id: str,
        event_type: CaseEventType,
        actor: str,
        actor_kind: ActorKind,
        payload: Mapping[str, Any],
    ) -> None:
        next_sequence = (
            session.execute(
                select(func.coalesce(func.max(DBCaseEvent.sequence), 0)).where(
                    DBCaseEvent.case_id == case_id
                )
            ).scalar_one()
            + 1
        )
        session.add(
            DBCaseEvent(
                case_id=case_id,
                sequence=int(next_sequence),
                event_type=event_type.value,
                actor=actor,
                actor_kind=actor_kind.value,
                payload=dict(payload),
            )
        )
        session.flush()

    @staticmethod
    def _events(session: Session, case_id: str) -> Tuple[CaseEvent, ...]:
        rows = (
            session.execute(
                select(DBCaseEvent)
                .where(DBCaseEvent.case_id == case_id)
                .order_by(DBCaseEvent.sequence)
            )
            .scalars()
            .all()
        )
        return tuple(
            CaseEvent(
                case_id=row.case_id,
                sequence=row.sequence,
                event_type=CaseEventType(row.event_type),
                actor=row.actor,
                actor_kind=ActorKind(row.actor_kind),
                payload=row.payload or {},
                recorded_at=_aware(row.recorded_at),
            )
            for row in rows
        )

    @staticmethod
    def _automated(row: DBCase) -> AutomatedDecisionView:
        return AutomatedDecisionView(
            action=row.automated_action,
            review_status=row.automated_review_status,
            reason_codes=tuple(row.automated_reason_codes or ()),
            primary_risk_score=row.primary_risk_score,
            investigator_recommendation=row.investigator_recommendation,
            grounding_status=row.grounding_status,
            policy_version=row.policy_version,
            workflow_version=row.workflow_version,
        )

    @classmethod
    def _summary(cls, row: DBCase) -> CaseSummary:
        return CaseSummary(
            case_id=row.case_id,
            candidate_id=row.candidate_id,
            status=CaseStatus(row.status),
            version=int(row.version),
            automated=cls._automated(row),
            claimed_by=row.claimed_by,
            claimed_at=_aware(row.claimed_at),
            resolved_by=row.resolved_by,
            resolved_at=_aware(row.resolved_at),
            member_count=int(row.member_count or 0),
            created_at=_aware(row.created_at),
            updated_at=_aware(row.updated_at),
        )

    @classmethod
    def _view(cls, session: Session, row: DBCase) -> CaseView:
        human = None
        if row.human_decision is not None:
            human = HumanDecisionView(
                reviewer_id=row.resolved_by,
                decision=row.human_decision,
                outcome=row.human_outcome,
                reason_codes=tuple(row.human_reason_codes or ()),
                notes=row.human_notes or "",
                resolved_at=_aware(row.resolved_at),
            )
        return CaseView(
            case_id=row.case_id,
            candidate_id=row.candidate_id,
            status=CaseStatus(row.status),
            version=int(row.version),
            idempotency_key=row.idempotency_key,
            automated=cls._automated(row),
            human=human,
            requested_information=tuple(
                RequestedInformationView.model_validate(item)
                for item in (row.requested_information or ())
            ),
            claimed_by=row.claimed_by,
            claimed_at=_aware(row.claimed_at),
            handoff=(
                HandoffView(
                    reference=row.handoff_reference,
                    mode=row.handoff_mode,
                    accepted_at=_aware(row.handoff_at),
                )
                if row.handoff_reference
                else None
            ),
            case_state=row.case_state or {},
            events=cls._events(session, row.case_id),
            created_at=_aware(row.created_at),
            updated_at=_aware(row.updated_at),
        )
