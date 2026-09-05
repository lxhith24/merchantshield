#!/usr/bin/env python3
"""Seed the demo dataset by running real applications through the real gate.

Nothing here is hard-coded: every risk score, decision, and explanation below
is produced by the actual pipeline, so the seeded database is a truthful
demonstration rather than a mock-up.

    python scripts/seed_demo_data.py [--reset]
"""
from __future__ import annotations

import argparse
import asyncio
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from merchantshield.db.database import SessionLocal, engine, init_db  # noqa: E402
from merchantshield.db.models import Base  # noqa: E402
from merchantshield.models.merchant import BusinessType, MerchantApplication  # noqa: E402
from merchantshield.pipeline import OnboardingGate  # noqa: E402

# ---------------------------------------------------------------- scenarios
# Ordered deliberately: the ring members are submitted in sequence so the
# clustering module has prior applications to link against, exactly as it
# would in production.
SCENARIOS = [
    # --- Genuine merchants: should sail through ---
    dict(
        business_name="Priya's Boutique", business_type=BusinessType.RETAIL,
        owner_name="Priya Sharma", owner_pan="BKAPS1234F",
        owner_email="priya@priyaboutique.in", owner_phone="+919845012345",
        gst_number="29BKAPS1234F1Z5",
        bank_account="50100123456789", bank_ifsc="HDFC0001234",
        registered_address="Shop 12, MG Road, Bangalore 560001",
        device_fingerprint="fp_legit_a1", ip_address="103.21.58.9",
    ),
    dict(
        business_name="Anand Traders", business_type=BusinessType.ECOMMERCE,
        owner_name="Ramesh Anand", owner_pan="AXTPA5643L",
        owner_email="ramesh@anandtraders.co.in", owner_phone="+919920145678",
        bank_account="00341200098765", bank_ifsc="ICIC0000034",
        registered_address="204 Linking Road, Bandra West, Mumbai 400050",
        device_fingerprint="fp_legit_b7", ip_address="49.36.112.44",
    ),
    dict(
        business_name="Kerala Spice Kitchen", business_type=BusinessType.FOOD,
        owner_name="Deepa Menon", owner_pan="CQZPM7781N",
        owner_email="deepa@keralaspice.in", owner_phone="+919747882211",
        bank_account="67219004455321", bank_ifsc="SBIN0006721",
        registered_address="18 Marine Drive, Kochi, Kerala 682031",
        device_fingerprint="fp_legit_c3", ip_address="117.196.4.88",
    ),

    # --- Fraud ring: three shells sharing device, IP, and settlement account ---
    dict(
        business_name="Best Electronics Hub", business_type=BusinessType.RETAIL,
        owner_name="Amit Verma", owner_pan="DEFAV9876H",
        owner_email="best.electronics@tempmail.com", owner_phone="+917777666655",
        bank_account="98765432100", bank_ifsc="ICIC0001234",
        registered_address="123 Industrial Area Phase 2, Noida 201301",
        device_fingerprint="fp_ring_01", ip_address="45.67.89.101",
    ),
    dict(
        business_name="Super Mobile World", business_type=BusinessType.RETAIL,
        owner_name="Vikram Singh", owner_pan="GHIVS4321J",
        owner_email="supermobile@guerrillamail.com", owner_phone="+917777666656",
        bank_account="98765432100", bank_ifsc="ICIC0001234",
        registered_address="124 Industrial Area Phase 2, Noida 201301",
        device_fingerprint="fp_ring_01", ip_address="45.67.89.101",
    ),
    dict(
        business_name="Prime Gadget Store", business_type=BusinessType.RETAIL,
        owner_name="Rohit Yadav", owner_pan="JKLRY8765K",
        owner_email="primegadget@mailinator.com", owner_phone="+917777666657",
        bank_account="98765432100", bank_ifsc="ICIC0001234",
        registered_address="125 Industrial Area Phase 2, Noida 201301",
        device_fingerprint="fp_ring_01", ip_address="45.67.89.101",
    ),

    # --- Synthetic identity: valid formats, fabricated person ---
    dict(
        business_name="Test Store Enterprise", business_type=BusinessType.SERVICES,
        owner_name="John Test", owner_pan="QWERT1111Z",
        owner_email="john.test123456@yahoo.com", owner_phone="+916111111111",
        bank_account="11111111111", bank_ifsc="UTIB0001111",
        registered_address="Test Address, Sector 1",
        device_fingerprint="fp_synth_z9", ip_address="8.8.8.8",
    ),

    # --- Borderline: real-looking but several soft tells; belongs in review ---
    dict(
        business_name="QuickMart Trading", business_type=BusinessType.ECOMMERCE,
        owner_name="Rahul Kumar", owner_pan="ABCPS5678G",
        owner_email="rahul12345@gmail.com", owner_phone="+919999888877",
        bank_account="12345678901", bank_ifsc="SBIN0005678",
        registered_address="Flat 5B, Sunrise Apartments, Dwarka, Delhi",
        device_fingerprint="fp_grey_k2", ip_address="157.32.88.201",
    ),

    # --- High-risk category with a GST that belongs to someone else ---
    dict(
        business_name="Apex Crypto Exchange", business_type=BusinessType.CRYPTO,
        owner_name="Sanjay Mehta", owner_pan="LMNPM4433Q",
        owner_email="ops@apexcrypto.io", owner_phone="+918800223344",
        gst_number="07AAAAA0000A1Z5",  # embedded PAN != declared PAN
        bank_account="44556677889900", bank_ifsc="KKBK0000123",
        registered_address="Tower B, Cyber City, Gurugram, Haryana 122002",
        device_fingerprint="fp_crypto_m4", ip_address="182.71.9.14",
    ),
]


async def seed(reset: bool) -> None:
    if reset:
        Base.metadata.drop_all(bind=engine)
    init_db()

    gate = OnboardingGate()
    db = SessionLocal()
    results = []

    try:
        for spec in SCENARIOS:
            application = MerchantApplication(**spec)
            assessment, app_id = await gate.assess(application, db)
            results.append((application.business_name, assessment))
    finally:
        db.close()

    # ------------------------------------------------------------- report
    width = max(len(n) for n, _ in results)
    print(f"\n  Seeded {len(results)} applications through the live pipeline\n")
    print(f"  {'MERCHANT'.ljust(width)}  {'DECISION':<13} {'RISK':>5}  BREAKDOWN")
    print(f"  {'-' * width}  {'-' * 13} {'-' * 5}  {'-' * 34}")
    for name, a in results:
        breakdown = (
            f"doc {a.document_risk_score:.2f} | "
            f"id {a.synthetic_identity_score:.2f} | "
            f"ring {a.clustering_risk_score:.2f}"
        )
        print(
            f"  {name.ljust(width)}  {a.decision.value:<13} "
            f"{a.final_risk_score:>5.2f}  {breakdown}"
        )

    counts: dict[str, int] = {}
    for _, a in results:
        counts[a.decision.value] = counts.get(a.decision.value, 0) + 1
    summary = "  ".join(f"{k}={v}" for k, v in sorted(counts.items()))
    print(f"\n  Totals: {summary}")

    mode = results[0][1].analysis_mode if results else "unknown"
    print(f"  Analysis mode: {mode}")
    if mode == "heuristic_only":
        print(
            "  (Set ANTHROPIC_API_KEY in .env to enable document vision "
            "analysis and LLM-written reviewer notes.)"
        )
    print()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--reset", action="store_true", help="drop all tables before seeding"
    )
    asyncio.run(seed(parser.parse_args().reset))
