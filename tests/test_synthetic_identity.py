"""Synthetic-identity detector: it must stay quiet on real merchants and fire
specifically — not generically — on fabricated ones."""
from __future__ import annotations

import pytest

from merchantshield.analysis import SyntheticIdentityDetector
from tests.conftest import make_application

pytestmark = pytest.mark.asyncio


def names(result) -> set[str]:
    return {s.name for s in result["signals"]}


async def test_clean_application_scores_zero(clean_application):
    result = await SyntheticIdentityDetector().analyze(clean_application)
    assert result["synthetic_score"] == 0.0
    assert result["signals"] == []
    assert result["risk_level"] == "low"


async def test_disposable_email_is_flagged():
    app = make_application(owner_email="best@tempmail.com")
    result = await SyntheticIdentityDetector().analyze(app)
    assert "disposable_email_domain" in names(result)
    assert result["synthetic_score"] > 0.5


async def test_pan_surname_mismatch_detected():
    # PAN 5th char 'S' implies surname starting with S; owner surname is Kumar.
    app = make_application(owner_name="Rahul Kumar", owner_pan="BKAPS1234F")
    result = await SyntheticIdentityDetector().analyze(app)
    assert "pan_surname_mismatch" in names(result)


async def test_pan_matching_surname_does_not_flag():
    app = make_application(owner_name="Rahul Kumar", owner_pan="BKAPK1234F")
    result = await SyntheticIdentityDetector().analyze(app)
    assert "pan_surname_mismatch" not in names(result)


async def test_repeated_digit_account_and_phone():
    app = make_application(bank_account="11111111111", owner_phone="+916111111111")
    result = await SyntheticIdentityDetector().analyze(app)
    fired = names(result)
    assert "repeated_digit_account" in fired
    assert "repeated_digit_phone" in fired
    assert result["risk_level"] in {"high", "critical"}


async def test_placeholder_tokens_across_fields():
    app = make_application(
        business_name="Test Store Enterprise",
        registered_address="Test Address",
        owner_email="john.test123@yahoo.com",
    )
    result = await SyntheticIdentityDetector().analyze(app)
    fired = names(result)
    assert "placeholder_business_name" in fired
    assert "placeholder_address" in fired
    assert "placeholder_email" in fired


async def test_gst_pan_mismatch_detected():
    # GSTIN embeds a PAN in positions 2..11 that differs from the declared PAN.
    app = make_application(owner_pan="BKAPS1234F", gst_number="29AAAAA0000A1Z5")
    result = await SyntheticIdentityDetector().analyze(app)
    assert "gst_pan_mismatch" in names(result)


async def test_gst_matching_pan_does_not_flag():
    app = make_application(owner_pan="BKAPS1234F", gst_number="29BKAPS1234F1Z5")
    result = await SyntheticIdentityDetector().analyze(app)
    assert "gst_pan_mismatch" not in names(result)


async def test_score_is_bounded():
    app = make_application(
        business_name="Test Fake Dummy",
        owner_name="John Test",
        owner_pan="QWERT1111Z",
        owner_email="test@tempmail.com",
        owner_phone="+916111111111",
        bank_account="11111111111",
        registered_address="Test Addr",
    )
    result = await SyntheticIdentityDetector().analyze(app)
    assert 0.0 <= result["synthetic_score"] <= 1.0
    assert result["risk_level"] == "critical"
