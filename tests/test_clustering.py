"""Clustering: the module that turns 'each looks fine alone' into a visible ring."""
from __future__ import annotations

import pytest

from merchantshield.analysis import ApplicationClusterer
from merchantshield.db.models import DBMerchant
from tests.conftest import make_application

pytestmark = pytest.mark.asyncio


def _persist(db, clusterer, app, merchant_id, status="under_review", risk=0.4):
    db.add(DBMerchant(
        id=merchant_id,
        business_name=app.business_name,
        business_type=app.business_type.value,
        owner_name=app.owner_name,
        owner_pan=app.owner_pan,
        owner_email=str(app.owner_email),
        owner_phone=app.owner_phone,
        bank_account=app.bank_account,
        bank_ifsc=app.bank_ifsc,
        registered_address=app.registered_address,
        address_hash=clusterer.address_hash(app.registered_address),
        device_fingerprint=app.device_fingerprint,
        ip_address=app.ip_address,
        status=status,
        risk_score=risk,
    ))
    db.flush()
    clusterer.index_application(merchant_id, clusterer.extract_keys(app), db)
    db.commit()


async def test_first_application_has_no_cluster(db, clean_application):
    result = await ApplicationClusterer().analyze(clean_application, db)
    assert result["clustering_risk_score"] == 0.0
    assert result["cluster_info"].cluster_size == 1
    assert result["related_application_count"] == 0


async def test_shared_bank_account_links_applications(db):
    c = ApplicationClusterer()
    first = make_application(
        business_name="Best Electronics", owner_pan="DEFAV9876H",
        owner_email="a@x.com", bank_account="98765432100",
        device_fingerprint="fp_a", ip_address="1.1.1.1",
        registered_address="123 Industrial Area, Noida 201301",
    )
    _persist(db, c, first, "m1")

    second = make_application(
        business_name="Super Mobile", owner_pan="GHIVS4321J",
        owner_email="b@y.com", bank_account="98765432100",  # same account
        device_fingerprint="fp_b", ip_address="2.2.2.2",
        registered_address="900 Other Road, Chennai 600001",
    )
    result = await c.analyze(second, db)

    assert result["cluster_info"].cluster_size == 2
    assert "bank_account" in result["cluster_info"].shared_attributes
    assert "m1" in result["cluster_info"].related_application_ids
    assert result["clustering_risk_score"] > 0.5


async def test_risk_escalates_with_cluster_size(db):
    c = ApplicationClusterer()
    scores = []
    for i in range(4):
        app = make_application(
            business_name=f"Ring Member {i}",
            owner_pan=f"DEFA{chr(65+i)}9876H",
            owner_email=f"m{i}@ring.com",
            bank_account="98765432100",
            device_fingerprint="fp_ring",
            ip_address="45.67.89.101",
            registered_address=f"{100+i} Industrial Area Phase 2, Noida 201301",
        )
        result = await c.analyze(app, db)
        scores.append(result["clustering_risk_score"])
        _persist(db, c, app, f"ring{i}")

    assert scores[0] == 0.0                      # first sees nobody
    assert scores[1] > 0.0                       # second links to first
    assert scores[3] >= scores[2] >= scores[1]   # monotonic escalation


async def test_cluster_containing_rejected_merchant_raises_risk(db):
    c = ApplicationClusterer()
    bad = make_application(
        business_name="Known Bad", owner_pan="DEFAV9876H", owner_email="bad@x.com",
        bank_account="55555555555", device_fingerprint="fp_bad", ip_address="9.9.9.9",
    )
    _persist(db, c, bad, "rejected1", status="rejected", risk=0.95)

    linked = make_application(
        business_name="Linked Applicant", owner_pan="GHIVS4321J",
        owner_email="new@x.com", bank_account="55555555555",
        device_fingerprint="fp_new", ip_address="8.8.8.8",
    )
    result = await c.analyze(linked, db)
    assert "cluster_contains_rejected" in {s.name for s in result["signals"]}


async def test_address_hash_normalises_variants():
    c = ApplicationClusterer()
    a = c.address_hash("12 M.G. Road, Bangalore 560001")
    b = c.address_hash("12 MG Road Bangalore 560001")
    assert a == b
    assert a != c.address_hash("77 Anna Salai, Chennai 600002")


async def test_exclude_id_prevents_self_match(db, clean_application):
    c = ApplicationClusterer()
    _persist(db, c, clean_application, "self1")
    result = await c.analyze(clean_application, db, exclude_id="self1")
    assert result["cluster_info"].cluster_size == 1
    assert result["clustering_risk_score"] == 0.0
