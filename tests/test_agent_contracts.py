"""Typed expert-state invariants."""
from __future__ import annotations

import pytest
from pydantic import ValidationError

from merchantshield.agent import (
    ExecutionMetadata,
    ExpertCost,
    ExpertResult,
    ExpertStatus,
)


def _metadata() -> ExecutionMetadata:
    return ExecutionMetadata(
        implementation_version="test-v1",
        route_reason="contract test",
        duration_ms=1.0,
        cost_class=ExpertCost.CHEAP,
    )


def test_completed_expert_requires_score_and_confidence():
    with pytest.raises(ValidationError, match="risk_score and confidence"):
        ExpertResult(
            expert_name="test_expert",
            candidate_id="candidate-test",
            status=ExpertStatus.COMPLETED,
            rationale="invalid completed state",
            metadata=_metadata(),
        )


def test_failed_expert_cannot_silently_look_clean():
    failed = ExpertResult(
        expert_name="test_expert",
        candidate_id="candidate-test",
        status=ExpertStatus.FAILED,
        risk_score=None,
        confidence=None,
        rationale="bounded failure",
        error_code="TEST_FAILURE",
        metadata=_metadata(),
    )
    assert failed.risk_score is None
    assert failed.status is ExpertStatus.FAILED

    with pytest.raises(ValidationError, match="may not imply a risk score"):
        ExpertResult(
            expert_name="test_expert",
            candidate_id="candidate-test",
            status=ExpertStatus.FAILED,
            risk_score=0.0,
            rationale="must not become clean",
            error_code="TEST_FAILURE",
            metadata=_metadata(),
        )
