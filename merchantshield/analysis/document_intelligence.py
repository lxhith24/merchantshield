"""Document intelligence — LLM vision analysis of submitted KYC documents.

Checks each uploaded document for three things a field-by-field validator
cannot see:
  1. Tampering — editing artifacts, font/kerning breaks, cloned regions.
  2. Template forgery — generated from a fraud kit rather than issued.
  3. Claimed-vs-extracted disagreement — the form says one thing, the
     document says another.

Degrades gracefully: with no ANTHROPIC_API_KEY the module returns a neutral,
explicitly-skipped analysis rather than a fabricated score.
"""
from __future__ import annotations

import asyncio
import base64
import json
import os
from typing import Any, Dict, Optional

from ..config import get_settings
from ..models.document import DocumentAnalysis, DocumentType

# Per-document-type verification guidance handed to the vision model.
_DOC_GUIDANCE: Dict[DocumentType, str] = {
    DocumentType.PAN_CARD: (
        "PAN card checks: PAN must match AAAAA9999A. The 4th character encodes "
        "holder type (P=individual). The 5th character must be the surname's "
        "first letter. Look for the Income Tax Department emblem, the "
        "government hologram, consistent card-stock texture, and a photo whose "
        "lighting matches the card surface."
    ),
    DocumentType.AADHAAR: (
        "Aadhaar checks: 12 digits, usually grouped XXXX XXXX XXXX. A genuine "
        "card carries a scannable QR code and the UIDAI logo. Verify the "
        "printed digits use UIDAI's typeface and that the photo has not been "
        "spliced in."
    ),
    DocumentType.GST_CERTIFICATE: (
        "GST certificate checks: GSTIN is 2-digit state code + 10-char PAN + "
        "entity digit + 'Z' + checksum. Confirm the legal name, trade name, "
        "principal place of business, and registration date are internally "
        "consistent and that the GSTIN's embedded PAN matches the declared PAN."
    ),
    DocumentType.BANK_STATEMENT: (
        "Bank statement checks: verify the bank letterhead/logo, that the "
        "running balance arithmetic is self-consistent line to line, that "
        "dates run in order, and that the account number format matches the "
        "issuing bank. Fabricated statements very often fail the running-balance "
        "arithmetic or use one uniform font for every row."
    ),
    DocumentType.ADDRESS_PROOF: (
        "Address proof checks: the issuing utility/authority must be "
        "identifiable, the bill date recent, and the printed address must "
        "match the declared registered address."
    ),
    DocumentType.BUSINESS_REGISTRATION: (
        "Registration checks: verify the registrar's seal, the CIN/LLPIN "
        "format, incorporation date, and that the listed directors/partners "
        "include the applicant."
    ),
}

_MEDIA_TYPES = {
    "jpg": "image/jpeg", "jpeg": "image/jpeg", "png": "image/png",
    "webp": "image/webp", "gif": "image/gif", "pdf": "application/pdf",
}

_SCHEMA = """{
  "extracted_fields": {"<field>": "<value read off the document>"},
  "tampering_detected": true|false,
  "tampering_confidence": 0.0-1.0,
  "tampering_reasons": ["specific visual evidence, not generic statements"],
  "is_template_forgery": true|false,
  "template_match_score": 0.0-1.0,
  "field_mismatches": ["claimed X = 'a' but document shows 'b'"],
  "overall_risk_assessment": "low"|"medium"|"high"|"critical",
  "risk_explanation": "one or two sentences"
}"""


class DocumentIntelligence:
    """Vision-model document forensics with a no-key fallback path."""

    def __init__(self) -> None:
        self.settings = get_settings()
        self._client = None
        if self.settings.llm_enabled:
            from anthropic import AsyncAnthropic

            self._client = AsyncAnthropic(api_key=self.settings.anthropic_api_key)

    @property
    def enabled(self) -> bool:
        return self._client is not None

    async def analyze_document(
        self,
        document_type: DocumentType,
        file_path: str,
        claimed_fields: Optional[Dict[str, Any]] = None,
    ) -> DocumentAnalysis:
        claimed_fields = {k: v for k, v in (claimed_fields or {}).items() if v}

        if not self.enabled:
            return self._skipped(
                document_type, file_path,
                "No ANTHROPIC_API_KEY configured — document vision analysis skipped.",
            )

        if not os.path.exists(file_path):
            return self._skipped(
                document_type, file_path, f"File not found: {file_path}"
            )

        try:
            payload = await asyncio.to_thread(self._encode, file_path)
        except Exception as exc:  # unreadable upload
            return self._skipped(
                document_type, file_path, f"Could not read upload: {exc}"
            )

        media_type, b64 = payload
        block_type = "document" if media_type == "application/pdf" else "image"

        try:
            resp = await self._client.messages.create(
                model=self.settings.model,
                max_tokens=1600,
                messages=[{
                    "role": "user",
                    "content": [
                        {
                            "type": block_type,
                            "source": {
                                "type": "base64",
                                "media_type": media_type,
                                "data": b64,
                            },
                        },
                        {"type": "text", "text": self._prompt(document_type, claimed_fields)},
                    ],
                }],
            )
        except Exception as exc:
            # An API failure must not silently read as "document is clean".
            return self._skipped(
                document_type, file_path, f"Vision API call failed: {exc}"
            )

        text = "".join(
            block.text for block in resp.content if getattr(block, "type", "") == "text"
        )
        return self._parse(text, document_type, file_path)

    # ---------------------------------------------------------------- helpers
    @staticmethod
    def _encode(file_path: str) -> tuple[str, str]:
        ext = file_path.rsplit(".", 1)[-1].lower()
        media_type = _MEDIA_TYPES.get(ext, "image/jpeg")
        with open(file_path, "rb") as fh:
            return media_type, base64.standard_b64encode(fh.read()).decode()

    def _prompt(self, doc_type: DocumentType, claimed: Dict[str, Any]) -> str:
        guidance = _DOC_GUIDANCE.get(doc_type, "")
        claimed_block = (
            json.dumps(claimed, indent=2) if claimed else "(none supplied)"
        )
        return f"""You are a KYC document forensics analyst at a payment aggregator. \
Examine this {doc_type.value.replace('_', ' ')} for a merchant onboarding application.

FIELDS CLAIMED ON THE APPLICATION FORM:
{claimed_block}

{guidance}

Carry out four checks:

1. FIELD EXTRACTION — read every legible field off the document as-is.

2. TAMPERING — look for concrete manipulation evidence: mismatched fonts or \
kerning within one line, pixelation or halos around text, cloned background \
patches, inconsistent JPEG blocking between regions, text baselines that do not \
share the document's grid, lighting or shadow that disagrees with the rest of \
the surface. Cite what you actually see; do not speculate.

3. TEMPLATE FORGERY — decide whether this looks generated from a blank template \
rather than issued: absent security features, unnaturally clean print, layout \
proportions that differ from the genuine article, filler-style typography.

4. FIELD CONSISTENCY — compare what the document says against the claimed \
fields above and list every disagreement explicitly.

Be conservative: a low-resolution or poorly-lit photo is NOT by itself evidence \
of tampering. Only raise tampering_detected when you can name the artifact.

Reply with JSON only, matching exactly this schema:
{_SCHEMA}"""

    def _parse(
        self, text: str, doc_type: DocumentType, file_path: str
    ) -> DocumentAnalysis:
        try:
            start, end = text.find("{"), text.rfind("}") + 1
            if start < 0 or end <= start:
                raise ValueError("no JSON object in model response")
            data = json.loads(text[start:end])
        except Exception as exc:
            return self._skipped(
                doc_type, file_path, f"Could not parse model response: {exc}"
            )

        analysis = DocumentAnalysis(
            document_type=doc_type,
            file_path=file_path,
            extracted_fields=data.get("extracted_fields") or {},
            tampering_detected=bool(data.get("tampering_detected")),
            tampering_confidence=_clamp(data.get("tampering_confidence")),
            tampering_reasons=[str(r) for r in (data.get("tampering_reasons") or [])],
            is_template_forgery=bool(data.get("is_template_forgery")),
            template_match_score=_clamp(data.get("template_match_score")),
            field_mismatches=[str(m) for m in (data.get("field_mismatches") or [])],
        )
        analysis.document_risk_score = self._score(analysis, data)
        return analysis

    @staticmethod
    def _score(a: DocumentAnalysis, raw: Dict[str, Any]) -> float:
        score = 0.0
        if a.tampering_detected:
            score += 0.42 * max(a.tampering_confidence, 0.5)
        if a.is_template_forgery:
            score += 0.33 * max(a.template_match_score, 0.5)
        score += min(0.20, 0.10 * len(a.field_mismatches))
        score += {"low": 0.0, "medium": 0.08, "high": 0.18, "critical": 0.30}.get(
            str(raw.get("overall_risk_assessment", "low")).lower(), 0.0
        )
        return round(min(1.0, score), 4)

    @staticmethod
    def _skipped(
        doc_type: DocumentType, file_path: str, reason: str
    ) -> DocumentAnalysis:
        """Neutral result. Never invents a risk score it could not measure."""
        return DocumentAnalysis(
            document_type=doc_type,
            file_path=file_path,
            document_risk_score=0.0,
            analysis_skipped=True,
            skip_reason=reason,
        )


def _clamp(v: Any) -> float:
    try:
        return max(0.0, min(1.0, float(v)))
    except (TypeError, ValueError):
        return 0.0
