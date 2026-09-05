"""The onboarding hand-off boundary.

Nothing upstream of a human decision may reach this module. A gateway is called
only after a named reviewer has recorded `approve_onboarding`; the investigator,
the router and the policy have no reference to it at all.
"""
from __future__ import annotations

from datetime import datetime, timezone
from enum import Enum
from typing import Mapping, Optional, Protocol

from pydantic import BaseModel, Field


class GatewayMode(str, Enum):
    """Which gateway actually handled a hand-off. Always shown in the UI."""

    SIMULATED = "SIMULATED"
    RAZORPAY_TEST = "RAZORPAY_TEST"


class GatewayUnavailable(RuntimeError):
    """The requested gateway is not configured. Never a silent fallback."""


class HandoffRequest(BaseModel):
    """The minimum an onboarding gateway needs. No identifiers, no documents."""

    case_id: str = Field(..., min_length=2, max_length=100)
    candidate_id: str = Field(..., min_length=2, max_length=100)
    member_ids: tuple[str, ...] = Field(..., min_length=1)
    approved_by: str = Field(..., min_length=2, max_length=80)
    reason_codes: tuple[str, ...] = Field(..., min_length=1)
    notes: str = Field("", max_length=2000)


class HandoffReceipt(BaseModel):
    """Proof that a hand-off happened, and under which mode."""

    reference: str = Field(..., min_length=4, max_length=120)
    mode: GatewayMode
    case_id: str
    member_ids: tuple[str, ...]
    accepted_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
    detail: Mapping[str, str] = Field(default_factory=dict)

    @property
    def is_simulated(self) -> bool:
        return self.mode is GatewayMode.SIMULATED


class OnboardingGateway(Protocol):
    """Accepts an approved merchant group for activation."""

    mode: GatewayMode

    def submit(self, request: HandoffRequest) -> HandoffReceipt: ...
