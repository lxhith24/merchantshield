"""Orchestration: run the four pillars over one application and persist the result.

Kept separate from the HTTP layer so the same pipeline is callable from the
seeder, tests, and any future queue worker.
"""
from __future__ import annotations

import asyncio
import os
import uuid
from datetime import datetime, timezone
from typing import Dict, List, Optional, Sequence, Tuple

from fastapi import UploadFile
from sqlalchemy.orm import Session

from .analysis import (
    ApplicationClusterer,
    DocumentIntelligence,
    RiskScorer,
    SyntheticIdentityDetector,
)
from .config import get_settings
from .db.models import DBDocument, DBMerchant, DBRiskAssessment
from .models.document import DocumentAnalysis, DocumentType
from .models.merchant import MerchantApplication
from .models.risk_assessment import RiskAssessment

# Which claimed form fields each document type should be cross-checked against.
_CLAIM_MAP = {
    DocumentType.PAN_CARD: lambda a: {"pan_number": a.owner_pan, "name": a.owner_name},
    DocumentType.AADHAAR: lambda a: {"name": a.owner_name, "address": a.registered_address},
    DocumentType.GST_CERTIFICATE: lambda a: {
        "gstin": a.gst_number, "legal_name": a.owner_name,
        "trade_name": a.business_name, "address": a.registered_address,
    },
    DocumentType.BANK_STATEMENT: lambda a: {
        "account_number": a.bank_account, "ifsc": a.bank_ifsc, "name": a.owner_name,
    },
    DocumentType.ADDRESS_PROOF: lambda a: {"address": a.registered_address, "name": a.owner_name},
    DocumentType.BUSINESS_REGISTRATION: lambda a: {
        "business_name": a.business_name, "owner": a.owner_name,
    },
}


class OnboardingGate:
    """The pre-activation gate. One call per merchant application."""

    def __init__(self) -> None:
        self.settings = get_settings()
        self.documents = DocumentIntelligence()
        self.identity = SyntheticIdentityDetector()
        self.clustering = ApplicationClusterer()
        self.scorer = RiskScorer()

    async def assess(
        self,
        application: MerchantApplication,
        db: Session,
        uploads: Optional[Sequence[Tuple[DocumentType, UploadFile]]] = None,
        application_id: Optional[str] = None,
    ) -> Tuple[RiskAssessment, str]:
        """Score one application, persist everything, return the assessment."""
        app_id = application_id or str(uuid.uuid4())

        saved = await self._persist_uploads(app_id, uploads or [])

        # Documents are I/O-bound against the vision API; identity and
        # clustering are local. Run the document batch concurrently.
        doc_task = asyncio.gather(*[
            self.documents.analyze_document(
                document_type=doc_type,
                file_path=path,
                claimed_fields=_CLAIM_MAP[doc_type](application),
            )
            for doc_type, path in saved
        ]) if saved else None

        identity_result = await self.identity.analyze(application)
        cluster_result = await self.clustering.analyze(application, db, exclude_id=app_id)
        doc_results: List[DocumentAnalysis] = list(await doc_task) if doc_task else []

        assessment = await self.scorer.score(
            application_id=app_id,
            document_results=doc_results,
            synthetic_result=identity_result,
            clustering_result=cluster_result,
            business_name=application.business_name,
        )

        self._persist(app_id, application, assessment, doc_results, db)
        db.commit()
        return assessment, app_id

    # ---------------------------------------------------------------- uploads
    async def _persist_uploads(
        self, app_id: str, uploads: Sequence[Tuple[DocumentType, UploadFile]]
    ) -> List[Tuple[DocumentType, str]]:
        if not uploads:
            return []
        target_dir = os.path.join(self.settings.upload_dir, app_id)
        os.makedirs(target_dir, exist_ok=True)

        saved: List[Tuple[DocumentType, str]] = []
        for doc_type, upload in uploads:
            if not upload or not upload.filename:
                continue
            safe = os.path.basename(upload.filename).replace("..", "_")
            path = os.path.join(target_dir, f"{doc_type.value}_{safe}")
            content = await upload.read()
            if not content:
                continue
            with open(path, "wb") as fh:
                fh.write(content)
            saved.append((doc_type, path))
        return saved

    # -------------------------------------------------------------- persistence
    @staticmethod
    def _persist(
        app_id: str,
        application: MerchantApplication,
        assessment: RiskAssessment,
        documents: Sequence[DocumentAnalysis],
        db: Session,
    ) -> None:
        clusterer = ApplicationClusterer()

        db.add(DBMerchant(
            id=app_id,
            business_name=application.business_name,
            business_type=application.business_type.value,
            website_url=application.website_url,
            owner_name=application.owner_name,
            owner_pan=application.owner_pan,
            owner_email=str(application.owner_email),
            owner_phone=application.owner_phone,
            gst_number=application.gst_number,
            bank_account=application.bank_account,
            bank_ifsc=application.bank_ifsc,
            registered_address=application.registered_address,
            address_hash=clusterer.address_hash(application.registered_address),
            device_fingerprint=application.device_fingerprint,
            ip_address=application.ip_address,
            status=assessment.decision.value,
            risk_score=assessment.final_risk_score,
            created_at=datetime.now(timezone.utc),
        ))
        db.flush()

        for doc in documents:
            db.add(DBDocument(
                merchant_id=app_id,
                document_type=doc.document_type.value,
                file_path=doc.file_path,
                extracted_fields=doc.extracted_fields,
                tampering_detected=doc.tampering_detected,
                tampering_confidence=doc.tampering_confidence,
                tampering_reasons=doc.tampering_reasons,
                is_template_forgery=doc.is_template_forgery,
                template_match_score=doc.template_match_score,
                field_mismatches=doc.field_mismatches,
                document_risk_score=doc.document_risk_score,
                analysis_skipped=doc.analysis_skipped,
                skip_reason=doc.skip_reason or None,
            ))

        db.add(DBRiskAssessment(
            merchant_id=app_id,
            document_risk_score=assessment.document_risk_score,
            synthetic_identity_score=assessment.synthetic_identity_score,
            clustering_risk_score=assessment.clustering_risk_score,
            final_risk_score=assessment.final_risk_score,
            risk_signals=[s.model_dump() for s in assessment.risk_signals],
            cluster_info=assessment.cluster_info.model_dump()
            if assessment.cluster_info else None,
            decision=assessment.decision.value,
            decision_reason=assessment.decision_reason,
            human_readable_explanation=assessment.human_readable_explanation,
            auto_approve_threshold=assessment.auto_approve_threshold,
            auto_reject_threshold=assessment.auto_reject_threshold,
            analysis_mode=assessment.analysis_mode,
        ))

        # Index last so this application is discoverable by the NEXT one.
        clusterer.index_application(app_id, clusterer.extract_keys(application), db)
