"""Onboarding hand-off gateways. Reached only after a human approval."""

from .base import (
    GatewayMode,
    GatewayUnavailable,
    HandoffReceipt,
    HandoffRequest,
    OnboardingGateway,
)
from .razorpay import RazorpayTestGateway
from .simulated import SimulatedOnboardingGateway

__all__ = [
    "GatewayMode",
    "GatewayUnavailable",
    "HandoffReceipt",
    "HandoffRequest",
    "OnboardingGateway",
    "RazorpayTestGateway",
    "SimulatedOnboardingGateway",
]


def select_gateway() -> OnboardingGateway:
    """Simulated unless a Razorpay *test* Partner adapter is fully configured."""
    razorpay = RazorpayTestGateway()
    if razorpay.is_available:
        return razorpay
    return SimulatedOnboardingGateway()
