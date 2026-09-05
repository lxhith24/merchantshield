"""Document models — KYC artifacts and the result of analysing them."""
from __future__ import annotations

from datetime import datetime, timezone
from enum import Enum
from typing import Any, Dict, List

from pydantic import BaseModel, Field


class DocumentType(str, Enum):
    PAN_CARD = "pan_card"
    AADHAAR = "aadhaar"
    GST_CERTIFICATE = "gst_certificate"
    BANK_STATEMENT = "bank_statement"
    ADDRESS_PROOF = "address_proof"
    BUSINESS_REGISTRATION = "business_registration"


class DocumentAnalysis(BaseModel):
    """Outcome of running one uploaded document through document intelligence."""

    document_type: DocumentType
    file_path: str

    # Fields the model actually read off the document
    extracted_fields: Dict[str, Any] = Field(default_factory=dict)

    # Tampering / manipulation
    tampering_detected: bool = False
    tampering_confidence: float = Field(0.0, ge=0.0, le=1.0)
    tampering_reasons: List[str] = Field(default_factory=list)

    # Template forgery (generated from a fraud kit rather than issued)
    is_template_forgery: bool = False
    template_match_score: float = Field(0.0, ge=0.0, le=1.0)

    # Claimed-vs-extracted disagreements
    field_mismatches: List[str] = Field(default_factory=list)

    # Fused per-document risk, 0 = clean, 1 = certainly fraudulent
    document_risk_score: float = Field(0.0, ge=0.0, le=1.0)

    # True when analysis could not run (no API key / unreadable file)
    analysis_skipped: bool = False
    skip_reason: str = ""

    analyzed_at: datetime = Field(
        default_factory=lambda: datetime.now(timezone.utc)
    )
