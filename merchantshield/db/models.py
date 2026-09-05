"""SQLAlchemy ORM tables backing the onboarding gate."""
from __future__ import annotations

import uuid
from datetime import datetime, timezone

from sqlalchemy import (
    Boolean,
    DateTime,
    Float,
    ForeignKey,
    Integer,
    JSON,
    String,
    Text,
    UniqueConstraint,
    event,
)
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column, relationship


class Base(DeclarativeBase):
    pass


def _uuid() -> str:
    return str(uuid.uuid4())


def _utcnow() -> datetime:
    return datetime.now(timezone.utc)


class DBMerchant(Base):
    """A submitted merchant application and its current gate status."""

    __tablename__ = "merchants"

    id: Mapped[str] = mapped_column(String(36), primary_key=True, default=_uuid)

    business_name: Mapped[str] = mapped_column(String(200), nullable=False)
    business_type: Mapped[str] = mapped_column(String(32), nullable=False)
    website_url: Mapped[str | None] = mapped_column(String(500))

    owner_name: Mapped[str] = mapped_column(String(120), nullable=False)
    owner_pan: Mapped[str] = mapped_column(String(10), nullable=False, index=True)
    owner_email: Mapped[str] = mapped_column(String(255), nullable=False, index=True)
    owner_phone: Mapped[str] = mapped_column(String(16), nullable=False, index=True)

    gst_number: Mapped[str | None] = mapped_column(String(15), index=True)

    bank_account: Mapped[str] = mapped_column(String(18), nullable=False, index=True)
    bank_ifsc: Mapped[str] = mapped_column(String(11), nullable=False)

    registered_address: Mapped[str] = mapped_column(Text, nullable=False)
    address_hash: Mapped[str | None] = mapped_column(String(16), index=True)

    device_fingerprint: Mapped[str | None] = mapped_column(String(128), index=True)
    ip_address: Mapped[str | None] = mapped_column(String(45), index=True)

    status: Mapped[str] = mapped_column(String(20), default="under_review", index=True)
    risk_score: Mapped[float | None] = mapped_column(Float)

    reviewer_notes: Mapped[str | None] = mapped_column(Text)
    reviewed_at: Mapped[datetime | None] = mapped_column(DateTime)

    created_at: Mapped[datetime] = mapped_column(DateTime, default=_utcnow, index=True)
    updated_at: Mapped[datetime] = mapped_column(
        DateTime, default=_utcnow, onupdate=_utcnow
    )

    documents: Mapped[list["DBDocument"]] = relationship(
        back_populates="merchant", cascade="all, delete-orphan"
    )
    assessment: Mapped["DBRiskAssessment | None"] = relationship(
        back_populates="merchant", uselist=False, cascade="all, delete-orphan"
    )


class DBDocument(Base):
    """One uploaded KYC document plus its analysis output."""

    __tablename__ = "documents"

    id: Mapped[str] = mapped_column(String(36), primary_key=True, default=_uuid)
    merchant_id: Mapped[str] = mapped_column(
        String(36), ForeignKey("merchants.id", ondelete="CASCADE"), index=True
    )

    document_type: Mapped[str] = mapped_column(String(40), nullable=False)
    file_path: Mapped[str] = mapped_column(String(500), nullable=False)

    extracted_fields: Mapped[dict] = mapped_column(JSON, default=dict)
    tampering_detected: Mapped[bool] = mapped_column(Boolean, default=False)
    tampering_confidence: Mapped[float] = mapped_column(Float, default=0.0)
    tampering_reasons: Mapped[list] = mapped_column(JSON, default=list)
    is_template_forgery: Mapped[bool] = mapped_column(Boolean, default=False)
    template_match_score: Mapped[float] = mapped_column(Float, default=0.0)
    field_mismatches: Mapped[list] = mapped_column(JSON, default=list)
    document_risk_score: Mapped[float] = mapped_column(Float, default=0.0)
    analysis_skipped: Mapped[bool] = mapped_column(Boolean, default=False)
    skip_reason: Mapped[str | None] = mapped_column(String(255))

    created_at: Mapped[datetime] = mapped_column(DateTime, default=_utcnow)

    merchant: Mapped[DBMerchant] = relationship(back_populates="documents")


class DBRiskAssessment(Base):
    """The auditable decision record for one application."""

    __tablename__ = "risk_assessments"

    id: Mapped[str] = mapped_column(String(36), primary_key=True, default=_uuid)
    merchant_id: Mapped[str] = mapped_column(
        String(36),
        ForeignKey("merchants.id", ondelete="CASCADE"),
        unique=True,
        index=True,
    )

    document_risk_score: Mapped[float] = mapped_column(Float, default=0.0)
    synthetic_identity_score: Mapped[float] = mapped_column(Float, default=0.0)
    clustering_risk_score: Mapped[float] = mapped_column(Float, default=0.0)
    final_risk_score: Mapped[float] = mapped_column(Float, default=0.0)

    risk_signals: Mapped[list] = mapped_column(JSON, default=list)
    cluster_info: Mapped[dict | None] = mapped_column(JSON)

    decision: Mapped[str] = mapped_column(String(20), default="under_review")
    decision_reason: Mapped[str | None] = mapped_column(Text)
    human_readable_explanation: Mapped[str | None] = mapped_column(Text)

    auto_approve_threshold: Mapped[float] = mapped_column(Float, default=0.25)
    auto_reject_threshold: Mapped[float] = mapped_column(Float, default=0.75)
    analysis_mode: Mapped[str] = mapped_column(String(20), default="full")

    assessed_at: Mapped[datetime] = mapped_column(DateTime, default=_utcnow)

    merchant: Mapped[DBMerchant] = relationship(back_populates="assessment")


class DBApplicationCluster(Base):
    """Inverted index: one row per shared attribute value -> applications using it.

    This is what turns "each application looks fine alone" into a visible ring.
    """

    __tablename__ = "application_clusters"

    id: Mapped[str] = mapped_column(String(36), primary_key=True, default=_uuid)

    cluster_type: Mapped[str] = mapped_column(String(32), nullable=False, index=True)
    cluster_key: Mapped[str] = mapped_column(String(255), nullable=False, index=True)

    merchant_ids: Mapped[list] = mapped_column(JSON, default=list)
    cluster_size: Mapped[int] = mapped_column(Integer, default=0)

    created_at: Mapped[datetime] = mapped_column(DateTime, default=_utcnow)
    updated_at: Mapped[datetime] = mapped_column(
        DateTime, default=_utcnow, onupdate=_utcnow
    )


# ------------------------------------------------------------------ Phase 4
# Ring-case review persistence. These tables are the durable record behind the
# LangGraph workflow: `DBCase` holds current case state and both decisions,
# `DBCaseEvent` is an append-only audit log, and `DBPriorOutcome` is the
# durable backing store for structured case memory.


class AppendOnlyViolation(RuntimeError):
    """Raised when code tries to mutate or delete an append-only audit row."""


class DBCase(Base):
    """One ring case: its automated outcome and, separately, its human outcome.

    `version` is the optimistic-concurrency token. Every mutating operation
    supplies the version it read; a mismatch is rejected rather than merged, so
    two reviewers cannot silently overwrite each other.
    """

    __tablename__ = "cases"

    case_id: Mapped[str] = mapped_column(String(100), primary_key=True)
    candidate_id: Mapped[str] = mapped_column(String(100), nullable=False, index=True)
    idempotency_key: Mapped[str] = mapped_column(
        String(120), nullable=False, unique=True, index=True
    )

    status: Mapped[str] = mapped_column(String(32), nullable=False, index=True)
    version: Mapped[int] = mapped_column(Integer, nullable=False, default=1)
    member_count: Mapped[int] = mapped_column(Integer, nullable=False, default=0)

    # --- automated side: produced by deterministic policy, never by a human ---
    automated_action: Mapped[str] = mapped_column(String(40), nullable=False)
    automated_review_status: Mapped[str] = mapped_column(String(40), nullable=False)
    automated_reason_codes: Mapped[list] = mapped_column(JSON, default=list)
    primary_risk_score: Mapped[float | None] = mapped_column(Float)
    investigator_recommendation: Mapped[str | None] = mapped_column(String(40))
    grounding_status: Mapped[str | None] = mapped_column(String(40))
    policy_version: Mapped[str | None] = mapped_column(String(80))
    workflow_version: Mapped[str | None] = mapped_column(String(80))

    # --- human side: never written by the workflow -------------------------
    claimed_by: Mapped[str | None] = mapped_column(String(80), index=True)
    claimed_at: Mapped[datetime | None] = mapped_column(DateTime)
    resolved_by: Mapped[str | None] = mapped_column(String(80))
    resolved_at: Mapped[datetime | None] = mapped_column(DateTime)
    human_decision: Mapped[str | None] = mapped_column(String(40))
    human_outcome: Mapped[str | None] = mapped_column(String(40))
    human_reason_codes: Mapped[list] = mapped_column(JSON, default=list)
    human_notes: Mapped[str | None] = mapped_column(Text)

    # --- onboarding hand-off: only ever written after a human approval -----
    handoff_reference: Mapped[str | None] = mapped_column(String(120))
    handoff_mode: Mapped[str | None] = mapped_column(String(32))
    handoff_at: Mapped[datetime | None] = mapped_column(DateTime)

    # --- resumable workflow state -----------------------------------------
    requested_information: Mapped[list] = mapped_column(JSON, default=list)
    case_state: Mapped[dict] = mapped_column(JSON, default=dict)

    created_at: Mapped[datetime] = mapped_column(DateTime, default=_utcnow, index=True)
    updated_at: Mapped[datetime] = mapped_column(
        DateTime, default=_utcnow, onupdate=_utcnow
    )

    events: Mapped[list["DBCaseEvent"]] = relationship(
        back_populates="case",
        cascade="all, delete-orphan",
        order_by="DBCaseEvent.sequence",
    )


class DBCaseEvent(Base):
    """One append-only case event. Rows are never updated or deleted."""

    __tablename__ = "case_events"
    __table_args__ = (UniqueConstraint("case_id", "sequence", name="uq_case_sequence"),)

    id: Mapped[str] = mapped_column(String(36), primary_key=True, default=_uuid)
    case_id: Mapped[str] = mapped_column(
        String(100), ForeignKey("cases.case_id", ondelete="CASCADE"), index=True
    )
    sequence: Mapped[int] = mapped_column(Integer, nullable=False)

    event_type: Mapped[str] = mapped_column(String(48), nullable=False, index=True)
    actor: Mapped[str] = mapped_column(String(80), nullable=False)
    actor_kind: Mapped[str] = mapped_column(String(16), nullable=False)
    payload: Mapped[dict] = mapped_column(JSON, default=dict)

    recorded_at: Mapped[datetime] = mapped_column(DateTime, default=_utcnow, index=True)

    case: Mapped[DBCase] = relationship(back_populates="events")


class DBPriorOutcome(Base):
    """Durable structured case memory: how a human resolved shared infrastructure.

    Exact lookup by hashed attribute value only. No raw identifier is stored,
    and nothing here may overwrite the graph expert's numerical score.
    """

    __tablename__ = "prior_outcomes"

    id: Mapped[str] = mapped_column(String(36), primary_key=True, default=_uuid)
    infrastructure_hash: Mapped[str] = mapped_column(String(64), nullable=False, index=True)
    attribute: Mapped[str] = mapped_column(String(80), nullable=False)
    outcome: Mapped[str] = mapped_column(String(40), nullable=False)
    reviewer_reference: Mapped[str] = mapped_column(String(80), nullable=False)
    note: Mapped[str] = mapped_column(Text, default="")
    resolved_at: Mapped[datetime] = mapped_column(DateTime, default=_utcnow, index=True)


@event.listens_for(DBCaseEvent, "before_update", propagate=True)
def _block_event_update(mapper, connection, target) -> None:  # pragma: no cover - guard
    raise AppendOnlyViolation("case events are append-only and cannot be updated")


@event.listens_for(DBCaseEvent, "before_delete", propagate=True)
def _block_event_delete(mapper, connection, target) -> None:
    raise AppendOnlyViolation("case events are append-only and cannot be deleted")
