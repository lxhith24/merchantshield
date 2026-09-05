"""Synthetic-identity detection.

A synthetic identity is assembled from individually-valid fragments — a
well-formed PAN, a real-looking mobile number, a plausible address — that never
co-occur in genuine records. Field-by-field validation passes; the combination
is fabricated.

This module is deterministic and runs without any API key, so it is the
backbone of heuristic-only mode.
"""
from __future__ import annotations

import re
from typing import Dict, List

from ..models.merchant import MerchantApplication
from ..models.risk_assessment import RiskSignal

SOURCE = "synthetic_identity"

# Disposable / temporary mail providers — near-conclusive fraud signal on a
# regulated financial onboarding form.
DISPOSABLE_DOMAINS = {
    "tempmail.com", "temp-mail.org", "guerrillamail.com", "10minutemail.com",
    "mailinator.com", "yopmail.com", "throwaway.email", "fakeinbox.com",
    "trashmail.com", "sharklasers.com", "getnada.com", "dispostable.com",
    "maildrop.cc", "mintemail.com", "tempinbox.com", "emailondeck.com",
}

FREE_CONSUMER_DOMAINS = {
    "gmail.com", "yahoo.com", "yahoo.co.in", "outlook.com", "hotmail.com",
    "rediffmail.com", "protonmail.com", "icloud.com", "live.com", "aol.com",
}

PLACEHOLDER_TOKENS = {
    "test", "testing", "fake", "dummy", "sample", "demo", "example",
    "asdf", "qwerty", "abcd", "xyz", "temp", "xxx", "aaa",
}

# Valid Indian mobile numbers start 6-9 after the country code.
VALID_MOBILE_FIRST_DIGITS = {"6", "7", "8", "9"}

INDIAN_STATE_HINTS = {
    "andhra", "arunachal", "assam", "bihar", "chhattisgarh", "goa", "gujarat",
    "haryana", "himachal", "jharkhand", "karnataka", "kerala", "madhya",
    "maharashtra", "manipur", "meghalaya", "mizoram", "nagaland", "odisha",
    "punjab", "rajasthan", "sikkim", "tamil", "telangana", "tripura",
    "uttar", "bengal", "delhi", "mumbai", "bangalore", "bengaluru", "chennai",
    "kolkata", "hyderabad", "pune", "ahmedabad", "jaipur", "lucknow", "noida",
    "gurgaon", "gurugram", "surat", "kanpur", "nagpur", "indore", "bhopal",
}


class SyntheticIdentityDetector:
    """Scores how likely an applicant identity is fabricated rather than real."""

    async def analyze(self, application: MerchantApplication) -> Dict:
        signals: List[RiskSignal] = []

        signals += self._check_email(application)
        signals += self._check_phone(application)
        signals += self._check_pan(application)
        signals += self._check_address(application)
        signals += self._check_business_name(application)
        signals += self._check_cross_field(application)

        score = self._fuse(signals)
        return {
            "synthetic_score": score,
            "signals": signals,
            "risk_level": _level(score),
        }

    # ------------------------------------------------------------------ email
    def _check_email(self, app: MerchantApplication) -> List[RiskSignal]:
        out: List[RiskSignal] = []
        email = str(app.owner_email).lower()
        local, _, domain = email.partition("@")

        if domain in DISPOSABLE_DOMAINS:
            out.append(RiskSignal(
                name="disposable_email_domain", value=0.82, weight=1.6,
                explanation=(
                    f"Email uses disposable provider '{domain}' — a throwaway "
                    f"inbox on a regulated onboarding form."
                ),
                source=SOURCE,
            ))

        # e.g. rahul12345@ — mass-registered account pattern
        if re.fullmatch(r"[a-z]+\d{4,}", local):
            out.append(RiskSignal(
                name="sequential_numeric_email", value=0.45,
                explanation=(
                    f"Email local part '{local}' follows an auto-generated "
                    f"name+digits pattern typical of bulk-created accounts."
                ),
                source=SOURCE,
            ))

        if any(tok in local for tok in PLACEHOLDER_TOKENS):
            out.append(RiskSignal(
                name="placeholder_email", value=0.7, weight=1.3,
                explanation=f"Email local part '{local}' contains placeholder text.",
                source=SOURCE,
            ))

        # A real business rarely runs settlement off a free consumer inbox that
        # shares no token with the business name.
        if domain in FREE_CONSUMER_DOMAINS:
            biz_tokens = {t for t in re.findall(r"[a-z]{4,}", app.business_name.lower())}
            if biz_tokens and not any(t[:5] in local for t in biz_tokens):
                out.append(RiskSignal(
                    name="unaffiliated_free_email", value=0.22,
                    explanation=(
                        "Business uses a free consumer email unrelated to the "
                        "business name (no custom domain)."
                    ),
                    source=SOURCE,
                ))
        return out

    # ------------------------------------------------------------------ phone
    def _check_phone(self, app: MerchantApplication) -> List[RiskSignal]:
        out: List[RiskSignal] = []
        digits = re.sub(r"\D", "", app.owner_phone)
        national = digits[2:] if digits.startswith("91") and len(digits) == 12 else digits

        if len(national) == 10 and national[0] not in VALID_MOBILE_FIRST_DIGITS:
            out.append(RiskSignal(
                name="invalid_mobile_series", value=0.9, weight=1.5,
                explanation=(
                    f"Mobile number begins with '{national[0]}' — not a valid "
                    f"Indian mobile series (must be 6-9)."
                ),
                source=SOURCE,
            ))

        if re.search(r"(\d)\1{5,}", national):
            out.append(RiskSignal(
                name="repeated_digit_phone", value=0.85, weight=1.4,
                explanation=(
                    f"Mobile number '{national}' contains a long run of one "
                    f"repeated digit — not a real allocated number."
                ),
                source=SOURCE,
            ))
        elif re.search(r"(\d)\1{3,}", national):
            out.append(RiskSignal(
                name="digit_run_phone", value=0.4,
                explanation=f"Mobile number '{national}' contains a suspicious digit run.",
                source=SOURCE,
            ))

        if national in {"1234567890", "9876543210", "0123456789"} or \
                _is_sequential(national):
            out.append(RiskSignal(
                name="sequential_phone", value=0.6,
                explanation=f"Mobile number '{national}' is a sequential placeholder.",
                source=SOURCE,
            ))
        return out

    # -------------------------------------------------------------------- PAN
    def _check_pan(self, app: MerchantApplication) -> List[RiskSignal]:
        out: List[RiskSignal] = []
        pan = app.owner_pan

        # PAN[3] = holder type. 'P' = individual proprietor.
        holder = pan[3]
        if holder not in "PCHFATBLJG":
            out.append(RiskSignal(
                name="invalid_pan_holder_type", value=0.8, weight=1.4,
                explanation=(
                    f"PAN 4th character '{holder}' is not a valid holder-type "
                    f"code — PAN is structurally invalid."
                ),
                source=SOURCE,
            ))

        # PAN[4] must equal the first letter of the surname for individuals.
        if holder == "P":
            parts = [p for p in re.split(r"\s+", app.owner_name.strip().upper()) if p]
            if parts:
                surname_initial = parts[-1][0]
                if pan[4] != surname_initial:
                    out.append(RiskSignal(
                        name="pan_surname_mismatch", value=0.65, weight=1.3,
                        explanation=(
                            f"PAN 5th character '{pan[4]}' does not match the "
                            f"surname initial '{surname_initial}' of "
                            f"'{app.owner_name}' — PAN and name belong to "
                            f"different people."
                        ),
                        source=SOURCE,
                    ))

        if any(tok in pan.lower() for tok in ("qwert", "asdfg", "abcde", "aaaaa")):
            out.append(RiskSignal(
                name="keyboard_pattern_pan", value=0.75, weight=1.3,
                explanation=f"PAN '{pan}' follows a keyboard/alphabet pattern.",
                source=SOURCE,
            ))
        return out

    # ---------------------------------------------------------------- address
    def _check_address(self, app: MerchantApplication) -> List[RiskSignal]:
        out: List[RiskSignal] = []
        addr = app.registered_address.strip()
        low = addr.lower()

        if any(re.search(rf"\b{tok}\b", low) for tok in PLACEHOLDER_TOKENS):
            out.append(RiskSignal(
                name="placeholder_address", value=0.8, weight=1.4,
                explanation=f"Registered address '{addr}' contains placeholder text.",
                source=SOURCE,
            ))

        if not re.search(r"\b\d{6}\b", addr):
            out.append(RiskSignal(
                name="missing_pincode", value=0.35,
                explanation="Registered address has no 6-digit PIN code.",
                source=SOURCE,
            ))

        if len(addr) < 20:
            out.append(RiskSignal(
                name="incomplete_address", value=0.45,
                explanation=(
                    f"Registered address is only {len(addr)} characters — too "
                    f"short to be a real deliverable address."
                ),
                source=SOURCE,
            ))

        if not any(hint in low for hint in INDIAN_STATE_HINTS):
            out.append(RiskSignal(
                name="unrecognised_locality", value=0.25,
                explanation="Address names no recognisable Indian city or state.",
                source=SOURCE,
            ))
        return out

    # ---------------------------------------------------------- business name
    def _check_business_name(self, app: MerchantApplication) -> List[RiskSignal]:
        out: List[RiskSignal] = []
        low = app.business_name.lower()

        if any(re.search(rf"\b{tok}\b", low) for tok in PLACEHOLDER_TOKENS):
            out.append(RiskSignal(
                name="placeholder_business_name", value=0.8, weight=1.4,
                explanation=(
                    f"Business name '{app.business_name}' contains placeholder "
                    f"text — not a trading name."
                ),
                source=SOURCE,
            ))

        if len(low.split()) == 1 and low in {
            "store", "shop", "enterprise", "enterprises", "trading", "services", "traders"
        }:
            out.append(RiskSignal(
                name="generic_business_name", value=0.3,
                explanation=(
                    f"Business name '{app.business_name}' is a bare generic "
                    f"noun with no distinguishing identity."
                ),
                source=SOURCE,
            ))

        if app.is_high_risk_category:
            out.append(RiskSignal(
                name="high_risk_category", value=0.35,
                explanation=(
                    f"Business category '{app.business_type.value}' carries "
                    f"elevated laundering and chargeback exposure."
                ),
                source=SOURCE,
            ))
        return out

    # ------------------------------------------------------------ cross-field
    def _check_cross_field(self, app: MerchantApplication) -> List[RiskSignal]:
        out: List[RiskSignal] = []

        acct = app.bank_account
        if re.fullmatch(r"(\d)\1+", acct):
            out.append(RiskSignal(
                name="repeated_digit_account", value=0.88, weight=1.5,
                explanation=(
                    f"Settlement account '{acct}' is a single repeated digit — "
                    f"not a real bank account."
                ),
                source=SOURCE,
            ))
        elif _is_sequential(acct):
            out.append(RiskSignal(
                name="sequential_account", value=0.6,
                explanation=f"Settlement account '{acct}' is a sequential placeholder.",
                source=SOURCE,
            ))

        # A GST number embeds the holder's PAN in positions 2..11.
        if app.gst_number:
            embedded_pan = app.gst_number[2:12]
            if embedded_pan != app.owner_pan:
                out.append(RiskSignal(
                    name="gst_pan_mismatch", value=0.85, weight=1.5,
                    explanation=(
                        f"GSTIN embeds PAN '{embedded_pan}' but the application "
                        f"declares '{app.owner_pan}' — the GST registration "
                        f"belongs to a different entity."
                    ),
                    source=SOURCE,
                ))
        return out

    # ----------------------------------------------------------------- fusion
    def _fuse(self, signals: List[RiskSignal]) -> float:
        if not signals:
            return 0.0
        total_w = sum(s.weight for s in signals)
        weighted = sum(s.value * s.weight for s in signals) / total_w
        # Several independent tells are worse than one loud one.
        corroboration = min(0.18, 0.045 * max(0, len(signals) - 1))
        return round(min(1.0, weighted + corroboration), 4)


def _is_sequential(digits: str) -> bool:
    """True for runs like 123456789 or 987654321 (length >= 6)."""
    if len(digits) < 6 or not digits.isdigit():
        return False
    asc = all(int(digits[i + 1]) - int(digits[i]) == 1 for i in range(len(digits) - 1))
    desc = all(int(digits[i]) - int(digits[i + 1]) == 1 for i in range(len(digits) - 1))
    return asc or desc


def _level(score: float) -> str:
    if score < 0.3:
        return "low"
    if score < 0.5:
        return "medium"
    if score < 0.7:
        return "high"
    return "critical"
