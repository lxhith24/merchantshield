"""End-to-end API tests against the real FastAPI app via TestClient."""
from __future__ import annotations

import pytest
from fastapi.testclient import TestClient

from merchantshield.main import legacy_app

client = TestClient(legacy_app)

LEGIT = {
    "business_name": "Priya's Boutique", "business_type": "retail",
    "owner_name": "Priya Sharma", "owner_pan": "BKAPS1234F",
    "owner_email": "priya@priyaboutique.in", "owner_phone": "+919845012345",
    "bank_account": "50100123456789", "bank_ifsc": "HDFC0001234",
    "registered_address": "Shop 12, MG Road, Bangalore 560001",
    "device_fingerprint": "fp_legit", "ip_address": "103.21.58.9",
}


def ring_member(n: int) -> dict:
    return {
        **LEGIT,
        "business_name": f"Ring Shop {n}",
        "owner_name": f"Ring Owner{n}",
        "owner_pan": f"DEFA{chr(65 + n)}9876H",
        "owner_email": f"ring{n}@tempmail.com",
        "owner_phone": f"+91777766665{n}",
        "bank_account": "98765432100",
        "device_fingerprint": "fp_ring_shared",
        "ip_address": "45.67.89.101",
        "registered_address": f"{100 + n} Industrial Area Phase 2, Noida 201301",
    }


def test_health():
    r = client.get("/health")
    assert r.status_code == 200
    assert r.json()["status"] == "healthy"


def test_legitimate_merchant_is_approved():
    r = client.post("/api/v1/merchants/assess", data=LEGIT)
    assert r.status_code == 200, r.text
    body = r.json()
    assert body["decision"] == "approved"
    assert body["risk_score"] < 0.25
    assert body["application_id"]
    assert set(body["risk_breakdown"]) == {
        "document_intelligence", "synthetic_identity", "application_clustering"
    }


def test_invalid_pan_returns_422_with_readable_message():
    bad = {**LEGIT, "owner_pan": "NOTAPAN"}
    r = client.post("/api/v1/merchants/assess", data=bad)
    assert r.status_code == 422
    assert "owner_pan" in r.json()["detail"]


def test_invalid_business_type_returns_422():
    bad = {**LEGIT, "business_type": "banana"}
    r = client.post("/api/v1/merchants/assess", data=bad)
    assert r.status_code == 422


def test_fraud_ring_is_caught_on_the_second_application():
    first = client.post("/api/v1/merchants/assess", data=ring_member(1)).json()
    second = client.post("/api/v1/merchants/assess", data=ring_member(2)).json()

    # First member has no prior to link against.
    assert first["cluster"]["cluster_size"] == 1
    # Second links to the first and scores materially higher.
    assert second["cluster"]["cluster_size"] == 2
    assert "bank_account" in second["cluster"]["shared_attributes"]
    assert second["risk_score"] > first["risk_score"]
    assert second["decision"] == "rejected"


def test_detail_endpoint_resolves_related_applications():
    client.post("/api/v1/merchants/assess", data=ring_member(1))
    second = client.post("/api/v1/merchants/assess", data=ring_member(2)).json()

    r = client.get(f"/api/v1/merchants/{second['application_id']}")
    assert r.status_code == 200
    body = r.json()
    assert body["assessment"]["explanation"]
    assert len(body["related_applications"]) == 1
    assert body["related_applications"][0]["business_name"] == "Ring Shop 1"


def test_detail_404_for_unknown_id():
    assert client.get("/api/v1/merchants/does-not-exist").status_code == 404


def test_list_and_filter():
    client.post("/api/v1/merchants/assess", data=LEGIT)
    client.post("/api/v1/merchants/assess", data=ring_member(1))
    client.post("/api/v1/merchants/assess", data=ring_member(2))

    every = client.get("/api/v1/merchants").json()
    assert every["total"] == 3

    approved = client.get("/api/v1/merchants?status=approved").json()
    assert all(m["status"] == "approved" for m in approved["merchants"])

    risky = client.get("/api/v1/merchants?min_risk=0.7").json()
    assert all(m["risk_score"] >= 0.7 for m in risky["merchants"])


def test_stats_counts_match_submissions():
    client.post("/api/v1/merchants/assess", data=LEGIT)
    client.post("/api/v1/merchants/assess", data=ring_member(1))

    s = client.get("/api/v1/stats").json()
    assert s["total"] == 2
    assert s["approved"] + s["under_review"] + s["rejected"] == 2
    assert s["analysis_mode"] == "heuristic_only"


def test_manual_review_flow():
    body = client.post("/api/v1/merchants/assess", data=ring_member(1)).json()
    assert body["decision"] == "under_review"

    ok = client.post(
        f"/api/v1/merchants/{body['application_id']}/review",
        data={"decision": "rejected", "reviewer_notes": "Disposable email; no premises."},
    )
    assert ok.status_code == 200
    assert ok.json()["status"] == "rejected"

    # Second review on a now-decided application must conflict, not silently pass.
    again = client.post(
        f"/api/v1/merchants/{body['application_id']}/review",
        data={"decision": "approved"},
    )
    assert again.status_code == 409


def test_review_rejects_invalid_decision_value():
    body = client.post("/api/v1/merchants/assess", data=ring_member(1)).json()
    r = client.post(
        f"/api/v1/merchants/{body['application_id']}/review",
        data={"decision": "maybe"},
    )
    assert r.status_code == 422
