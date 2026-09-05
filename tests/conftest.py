"""Shared pytest fixtures — every test runs against an isolated temp database."""
from __future__ import annotations

import os
import tempfile

import pytest

# Point the app at a throwaway DB *before* any merchantshield import binds it.
_TMP_DB = os.path.join(tempfile.mkdtemp(prefix="mshield_test_"), "test.db")
os.environ["DATABASE_URL"] = f"sqlite:///{_TMP_DB}"
os.environ["UPLOAD_DIR"] = tempfile.mkdtemp(prefix="mshield_uploads_")
os.environ.setdefault("ANTHROPIC_API_KEY", "")  # force heuristic-only in tests

from merchantshield.db.database import SessionLocal, engine, init_db  # noqa: E402
from merchantshield.db.models import Base  # noqa: E402
from merchantshield.models.merchant import BusinessType, MerchantApplication  # noqa: E402


@pytest.fixture(autouse=True)
def fresh_db():
    """Drop and recreate every table between tests so clusters never leak."""
    Base.metadata.drop_all(bind=engine)
    init_db()
    yield
    Base.metadata.drop_all(bind=engine)


@pytest.fixture
def db():
    session = SessionLocal()
    try:
        yield session
    finally:
        session.close()


def make_application(**overrides) -> MerchantApplication:
    """A clean, legitimate baseline application; override fields per test."""
    base = dict(
        business_name="Priya's Boutique",
        business_type=BusinessType.RETAIL,
        owner_name="Priya Sharma",
        owner_pan="BKAPS1234F",
        owner_email="priya@priyaboutique.in",
        owner_phone="+919845012345",
        bank_account="50100123456789",
        bank_ifsc="HDFC0001234",
        registered_address="Shop 12, MG Road, Bangalore 560001",
        device_fingerprint="fp_clean_a1",
        ip_address="103.21.58.9",
    )
    base.update(overrides)
    return MerchantApplication(**base)


@pytest.fixture
def clean_application() -> MerchantApplication:
    return make_application()


async def make_ring_case(
    *,
    seed: int = 20250904,
    min_size: int = 2,
    graph_score: float = 0.80,
    rules_expert=None,
):
    """Build one candidate component plus its Phase 2 assessment.

    Phase 3 tests need a real `CandidateAssessment` to investigate, but they
    must not depend on the fitted models' exact numbers, so the graph score is
    injected.
    """
    from merchantshield.agent import (
        CandidateContext,
        GraphScoringExpert,
        RoutedMoESystem,
        RulesExpert,
    )
    from merchantshield.analysis import EvidenceGraphBuilder
    from merchantshield.evaluation import generate_synthetic_dataset

    examples = generate_synthetic_dataset(seed=seed)
    applications = {item.example_id: item.application for item in examples}
    candidates = EvidenceGraphBuilder().build(applications)
    candidate = next(
        item for item in candidates if len(item.member_ids) >= min_size
    )
    scoped = {
        member_id: applications[member_id] for member_id in candidate.member_ids
    }
    system = RoutedMoESystem(
        rules_expert=rules_expert or RulesExpert(),
        graph_expert=GraphScoringExpert(lambda _: graph_score),
    )
    assessment = await system.assess(
        CandidateContext(candidate=candidate, applications=scoped)
    )
    return candidate, scoped, assessment
