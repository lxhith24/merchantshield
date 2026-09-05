"""Razorpay Partner *test-mode* adapter.

This adapter exists so the boundary is real rather than hypothetical, but it
refuses to run unless test-mode Partner credentials are explicitly configured.
There is no production mode and no fallback: an unconfigured adapter raises
`GatewayUnavailable` and the caller must stay on the simulated gateway.
"""
from __future__ import annotations

import os
from typing import Optional

from .base import GatewayMode, GatewayUnavailable, HandoffReceipt, HandoffRequest

TEST_KEY_PREFIX = "rzp_test_"


class RazorpayTestGateway:
    """Partner sandbox adapter, enabled only by explicit test credentials."""

    mode = GatewayMode.RAZORPAY_TEST

    def __init__(
        self,
        *,
        key_id: Optional[str] = None,
        key_secret: Optional[str] = None,
        enabled: Optional[bool] = None,
    ) -> None:
        self.key_id = (key_id if key_id is not None else os.getenv("RAZORPAY_KEY_ID", "")).strip()
        self.key_secret = (
            key_secret if key_secret is not None else os.getenv("RAZORPAY_KEY_SECRET", "")
        ).strip()
        self.enabled = (
            enabled
            if enabled is not None
            else os.getenv("RAZORPAY_PARTNER_TEST_ENABLED", "").strip().lower()
            in {"1", "true", "yes"}
        )

    @property
    def is_available(self) -> bool:
        """Configured, and configured with a *test* key. Live keys are refused."""
        return bool(
            self.enabled
            and self.key_id.startswith(TEST_KEY_PREFIX)
            and self.key_secret
        )

    def unavailable_reason(self) -> str:
        if not self.enabled:
            return "RAZORPAY_PARTNER_TEST_ENABLED is not set"
        if not self.key_id:
            return "RAZORPAY_KEY_ID is not set"
        if not self.key_id.startswith(TEST_KEY_PREFIX):
            return f"RAZORPAY_KEY_ID is not a test key (expected {TEST_KEY_PREFIX}...)"
        if not self.key_secret:
            return "RAZORPAY_KEY_SECRET is not set"
        return "adapter is available"

    def submit(self, request: HandoffRequest) -> HandoffReceipt:
        if not self.is_available:
            raise GatewayUnavailable(
                f"Razorpay Partner test adapter is not usable: {self.unavailable_reason()}. "
                "Staying on the simulated gateway."
            )
        # Deliberately not implemented against the live Partner API: this
        # repository has no Partner account, and shipping an untested network
        # call would make a claim the demonstration cannot support.
        raise GatewayUnavailable(
            "Razorpay Partner test credentials are present, but no Partner "
            "sub-merchant account is provisioned for this repository, so the "
            "call is not attempted."
        )
