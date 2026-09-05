"""Merchant application model — the payload arriving at the onboarding gate."""
from __future__ import annotations

from datetime import datetime, timezone
from enum import Enum
from typing import Optional

from pydantic import BaseModel, EmailStr, Field, field_validator

# Indian identifier formats
PAN_REGEX = r"^[A-Z]{5}[0-9]{4}[A-Z]$"
GSTIN_REGEX = r"^[0-9]{2}[A-Z]{5}[0-9]{4}[A-Z][0-9A-Z]Z[0-9A-Z]$"
IFSC_REGEX = r"^[A-Z]{4}0[A-Z0-9]{6}$"
PHONE_REGEX = r"^\+91[6-9][0-9]{9}$"


class BusinessType(str, Enum):
    RETAIL = "retail"
    ECOMMERCE = "ecommerce"
    SERVICES = "services"
    FOOD = "food"
    TRAVEL = "travel"
    GAMING = "gaming"
    CRYPTO = "crypto"
    OTHER = "other"


# Business categories that carry elevated inherent laundering / chargeback risk.
HIGH_RISK_CATEGORIES = {BusinessType.GAMING, BusinessType.CRYPTO, BusinessType.TRAVEL}


class MerchantApplication(BaseModel):
    """A merchant's onboarding submission, pre-activation."""

    # --- Business identity ---
    business_name: str = Field(..., min_length=2, max_length=200)
    business_type: BusinessType
    website_url: Optional[str] = Field(None, max_length=500)

    # --- Signatory / owner ---
    owner_name: str = Field(..., min_length=2, max_length=120)
    owner_pan: str = Field(..., pattern=PAN_REGEX)
    owner_email: EmailStr
    owner_phone: str = Field(..., pattern=PHONE_REGEX)

    # --- Registration ---
    gst_number: Optional[str] = Field(None, pattern=GSTIN_REGEX)

    # --- Settlement banking ---
    bank_account: str = Field(..., min_length=9, max_length=18)
    bank_ifsc: str = Field(..., pattern=IFSC_REGEX)

    # --- Address ---
    registered_address: str = Field(..., min_length=5, max_length=500)

    # --- Session fingerprints used for ring clustering ---
    device_fingerprint: Optional[str] = Field(None, max_length=128)
    ip_address: Optional[str] = Field(None, max_length=45)

    submitted_at: datetime = Field(
        default_factory=lambda: datetime.now(timezone.utc)
    )

    @field_validator("owner_pan", "gst_number", "bank_ifsc", mode="before")
    @classmethod
    def _upper(cls, v: Optional[str]) -> Optional[str]:
        return v.strip().upper() if isinstance(v, str) and v.strip() else None

    @field_validator("bank_account", mode="before")
    @classmethod
    def _digits_only(cls, v: str) -> str:
        if isinstance(v, str):
            cleaned = "".join(ch for ch in v if ch.isdigit())
            if not cleaned:
                raise ValueError("bank_account must contain digits")
            return cleaned
        return v

    @property
    def is_high_risk_category(self) -> bool:
        return self.business_type in HIGH_RISK_CATEGORIES

    @property
    def pan_state_independent_holder_type(self) -> str:
        """4th char of PAN encodes holder type: P=individual, C=company, etc."""
        return self.owner_pan[3] if len(self.owner_pan) >= 4 else "?"
