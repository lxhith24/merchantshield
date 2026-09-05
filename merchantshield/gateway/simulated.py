"""Default gateway: records the hand-off and calls nothing external."""
from __future__ import annotations

import hashlib
from typing import Dict, List

from .base import GatewayMode, HandoffReceipt, HandoffRequest


class SimulatedOnboardingGateway:
    """In-process gateway. It is the default, and it never leaves the machine.

    The receipt reference is derived from the case, so a repeated hand-off for
    the same case is visibly the same hand-off rather than a new activation.
    """

    mode = GatewayMode.SIMULATED

    def __init__(self) -> None:
        self._receipts: Dict[str, HandoffReceipt] = {}

    def submit(self, request: HandoffRequest) -> HandoffReceipt:
        existing = self._receipts.get(request.case_id)
        if existing is not None:
            return existing
        digest = hashlib.sha256(request.case_id.encode("utf-8")).hexdigest()[:10]
        receipt = HandoffReceipt(
            reference=f"sim_{digest}",
            mode=self.mode,
            case_id=request.case_id,
            member_ids=request.member_ids,
            detail={
                "gateway": "simulated",
                "network_calls": "0",
                "approved_by": request.approved_by,
            },
        )
        self._receipts[request.case_id] = receipt
        return receipt

    def receipts(self) -> List[HandoffReceipt]:
        return list(self._receipts.values())
