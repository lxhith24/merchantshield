"""REST API — the pre-activation gate surface."""
from __future__ import annotations

from datetime import datetime, timezone
from typing import Any, Dict, List, Optional

from fastapi import APIRouter, Depends, File, Form, HTTPException, Query, UploadFile
from pydantic import ValidationError
from sqlalchemy import func, select
from sqlalchemy.orm import Session

from ..config import get_settings
from ..db.database import get_db
from ..db.models import DBDocument, DBMerchant, DBRiskAssessment
from ..models.document import DocumentType
from ..models.merchant import BusinessType, MerchantApplication
from ..pipeline import OnboardingGate

router = APIRouter(prefix="/api/v1", tags=["onboarding"])
gate = OnboardingGate()


# --------------------------------------------------------------------- assess
@router.post("/merchants/assess", summary="Risk-score a merchant before activation")
async def assess_merchant(
    business_name: str = Form(...),
    business_type: str = Form(...),
    owner_name: str = Form(...),
    owner_pan: str = Form(...),
    owner_email: str = Form(...),
    owner_phone: str = Form(...),
    bank_account: str = Form(...),
    bank_ifsc: str = Form(...),
    registered_address: str = Form(...),
    gst_number: Optional[str] = Form(None),
    website_url: Optional[str] = Form(None),
    device_fingerprint: Optional[str] = Form(None),
    ip_address: Optional[str] = Form(None),
    pan_card: Optional[UploadFile] = File(None),
    aadhaar: Optional[UploadFile] = File(None),
    gst_certificate: Optional[UploadFile] = File(None),
    bank_statement: Optional[UploadFile] = File(None),
    address_proof: Optional[UploadFile] = File(None),
    db: Session = Depends(get_db),
) -> Dict[str, Any]:
    """Full pipeline: documents + identity + clustering -> fused decision."""
    try:
        application = MerchantApplication(
            business_name=business_name,
            business_type=BusinessType(business_type.lower()),
            website_url=website_url or None,
            owner_name=owner_name,
            owner_pan=owner_pan,
            owner_email=owner_email,
            owner_phone=owner_phone,
            gst_number=gst_number or None,
            bank_account=bank_account,
            bank_ifsc=bank_ifsc,
            registered_address=registered_address,
            device_fingerprint=device_fingerprint or None,
            ip_address=ip_address or None,
        )
    except ValueError as exc:
        # Pydantic validation and BusinessType coercion both land here.
        raise HTTPException(status_code=422, detail=_readable_validation(exc)) from exc

    uploads = [
        (DocumentType.PAN_CARD, pan_card),
        (DocumentType.AADHAAR, aadhaar),
        (DocumentType.GST_CERTIFICATE, gst_certificate),
        (DocumentType.BANK_STATEMENT, bank_statement),
        (DocumentType.ADDRESS_PROOF, address_proof),
    ]
    supplied = [(t, f) for t, f in uploads if f is not None and f.filename]

    try:
        assessment, app_id = await gate.assess(application, db, supplied)
    except Exception as exc:
        db.rollback()
        raise HTTPException(status_code=500, detail=f"Assessment failed: {exc}") from exc

    return {
        "application_id": app_id,
        "decision": assessment.decision.value,
        "risk_score": assessment.final_risk_score,
        "risk_breakdown": {
            "document_intelligence": assessment.document_risk_score,
            "synthetic_identity": assessment.synthetic_identity_score,
            "application_clustering": assessment.clustering_risk_score,
        },
        "decision_reason": assessment.decision_reason,
        "explanation": assessment.human_readable_explanation,
        "cluster": assessment.cluster_info.model_dump() if assessment.cluster_info else None,
        "top_signals": [s.model_dump() for s in assessment.top_signals],
        "signal_count": len(assessment.risk_signals),
        "documents_analysed": len(supplied),
        "analysis_mode": assessment.analysis_mode,
    }


# ----------------------------------------------------------------------- list
@router.get("/merchants", summary="List applications")
def list_merchants(
    status: Optional[str] = Query(None, pattern="^(approved|under_review|rejected)$"),
    min_risk: Optional[float] = Query(None, ge=0.0, le=1.0),
    limit: int = Query(50, ge=1, le=500),
    offset: int = Query(0, ge=0),
    db: Session = Depends(get_db),
) -> Dict[str, Any]:
    stmt = select(DBMerchant)
    if status:
        stmt = stmt.where(DBMerchant.status == status)
    if min_risk is not None:
        stmt = stmt.where(DBMerchant.risk_score >= min_risk)

    total = db.execute(
        select(func.count()).select_from(stmt.subquery())
    ).scalar_one()

    rows = db.execute(
        stmt.order_by(DBMerchant.created_at.desc()).limit(limit).offset(offset)
    ).scalars().all()

    return {
        "total": total,
        "count": len(rows),
        "merchants": [_merchant_summary(m) for m in rows],
    }


# ----------------------------------------------------------------------- stats
@router.get("/stats", summary="Queue counters for the dashboard")
def stats(db: Session = Depends(get_db)) -> Dict[str, Any]:
    counts = dict(
        db.execute(
            select(DBMerchant.status, func.count(DBMerchant.id)).group_by(DBMerchant.status)
        ).all()
    )
    total = sum(counts.values())
    avg = db.execute(select(func.avg(DBMerchant.risk_score))).scalar() or 0.0
    settings = get_settings()
    return {
        "total": total,
        "approved": counts.get("approved", 0),
        "under_review": counts.get("under_review", 0),
        "rejected": counts.get("rejected", 0),
        "average_risk_score": round(float(avg), 4),
        "analysis_mode": settings.mode,
    }


# ---------------------------------------------------------------------- detail
@router.get("/merchants/{application_id}", summary="Full assessment record")
def get_merchant(application_id: str, db: Session = Depends(get_db)) -> Dict[str, Any]:
    merchant = db.get(DBMerchant, application_id)
    if merchant is None:
        raise HTTPException(status_code=404, detail="Application not found")

    assessment = db.execute(
        select(DBRiskAssessment).where(DBRiskAssessment.merchant_id == application_id)
    ).scalar_one_or_none()

    documents = db.execute(
        select(DBDocument).where(DBDocument.merchant_id == application_id)
    ).scalars().all()

    # Resolve linked applications to names so the reviewer sees the ring.
    related: List[Dict[str, Any]] = []
    if assessment and assessment.cluster_info:
        ids = assessment.cluster_info.get("related_application_ids") or []
        if ids:
            rows = db.execute(
                select(DBMerchant).where(DBMerchant.id.in_(ids))
            ).scalars().all()
            related = [
                {
                    "id": r.id,
                    "business_name": r.business_name,
                    "owner_name": r.owner_name,
                    "status": r.status,
                    "risk_score": r.risk_score,
                }
                for r in rows
            ]

    return {
        "merchant": {
            **_merchant_summary(merchant),
            "owner_pan": merchant.owner_pan,
            "owner_email": merchant.owner_email,
            "owner_phone": merchant.owner_phone,
            "gst_number": merchant.gst_number,
            "bank_account": merchant.bank_account,
            "bank_ifsc": merchant.bank_ifsc,
            "registered_address": merchant.registered_address,
            "device_fingerprint": merchant.device_fingerprint,
            "ip_address": merchant.ip_address,
            "website_url": merchant.website_url,
            "reviewer_notes": merchant.reviewer_notes,
        },
        "assessment": _assessment_payload(assessment),
        "documents": [
            {
                "document_type": d.document_type,
                "document_risk_score": d.document_risk_score,
                "tampering_detected": d.tampering_detected,
                "tampering_reasons": d.tampering_reasons,
                "is_template_forgery": d.is_template_forgery,
                "field_mismatches": d.field_mismatches,
                "extracted_fields": d.extracted_fields,
                "analysis_skipped": d.analysis_skipped,
                "skip_reason": d.skip_reason,
            }
            for d in documents
        ],
        "related_applications": related,
    }


# ---------------------------------------------------------------------- review
@router.post("/merchants/{application_id}/review", summary="Record a manual decision")
def submit_review(
    application_id: str,
    decision: str = Form(..., pattern="^(approved|rejected)$"),
    reviewer_notes: str = Form(""),
    db: Session = Depends(get_db),
) -> Dict[str, Any]:
    merchant = db.get(DBMerchant, application_id)
    if merchant is None:
        raise HTTPException(status_code=404, detail="Application not found")
    if merchant.status != "under_review":
        raise HTTPException(
            status_code=409,
            detail=f"Application is '{merchant.status}', not awaiting review.",
        )

    merchant.status = decision
    merchant.reviewer_notes = reviewer_notes or None
    merchant.reviewed_at = datetime.now(timezone.utc)

    assessment = db.execute(
        select(DBRiskAssessment).where(DBRiskAssessment.merchant_id == application_id)
    ).scalar_one_or_none()
    if assessment:
        assessment.decision = decision
        assessment.decision_reason = (
            f"Manual review: {decision}."
            + (f" {reviewer_notes}" if reviewer_notes else "")
        )
    db.commit()

    return {
        "application_id": application_id,
        "status": decision,
        "reviewer_notes": reviewer_notes,
        "reviewed_at": merchant.reviewed_at.isoformat(),
    }


# --------------------------------------------------------------------- helpers
def _merchant_summary(m: DBMerchant) -> Dict[str, Any]:
    return {
        "id": m.id,
        "business_name": m.business_name,
        "business_type": m.business_type,
        "owner_name": m.owner_name,
        "status": m.status,
        "risk_score": m.risk_score,
        "created_at": m.created_at.isoformat() if m.created_at else None,
    }


def _assessment_payload(a: Optional[DBRiskAssessment]) -> Optional[Dict[str, Any]]:
    if a is None:
        return None
    return {
        "final_risk_score": a.final_risk_score,
        "risk_breakdown": {
            "document_intelligence": a.document_risk_score,
            "synthetic_identity": a.synthetic_identity_score,
            "application_clustering": a.clustering_risk_score,
        },
        "decision": a.decision,
        "decision_reason": a.decision_reason,
        "explanation": a.human_readable_explanation,
        "signals": sorted(
            a.risk_signals or [], key=lambda s: s.get("value", 0), reverse=True
        ),
        "cluster_info": a.cluster_info,
        "thresholds": {
            "auto_approve": a.auto_approve_threshold,
            "auto_reject": a.auto_reject_threshold,
        },
        "analysis_mode": a.analysis_mode,
        "assessed_at": a.assessed_at.isoformat() if a.assessed_at else None,
    }


def _readable_validation(exc: Exception) -> str:
    if isinstance(exc, ValidationError):
        parts = []
        for err in exc.errors():
            field = ".".join(str(p) for p in err["loc"])
            parts.append(f"{field}: {err['msg']}")
        return "; ".join(parts)
    return str(exc)
