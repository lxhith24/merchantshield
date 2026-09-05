"""Application clustering — exposing fraud rings that submit in bulk.

Each application in a ring looks fine in isolation. The ring becomes visible
only when you link applications by what they involuntarily share: the device
they were typed on, the IP they arrived from, the settlement account the money
ultimately lands in, the address block, the phone batch.

Implemented as an inverted index (attribute value -> merchant ids) in
`application_clusters`, so lookup is a handful of indexed reads rather than an
O(n^2) graph walk.
"""
from __future__ import annotations

import hashlib
import re
from datetime import datetime, timedelta, timezone
from typing import Dict, List, Optional, Set

from sqlalchemy import select
from sqlalchemy.orm import Session

from ..db.models import DBApplicationCluster, DBMerchant
from ..models.merchant import MerchantApplication
from ..models.risk_assessment import ClusterInfo, RiskSignal

SOURCE = "application_clustering"

# How incriminating is it for two applications to share this attribute?
# A shared settlement account is near-conclusive; a shared IP could be a shared
# office, a café, or carrier-grade NAT.
ATTRIBUTE_SEVERITY: Dict[str, float] = {
    "bank_account": 0.88,
    "device_fingerprint": 0.72,
    "address_hash": 0.55,
    "owner_pan": 0.85,
    "ip_address": 0.38,
    "phone_series": 0.30,
    "email_domain": 0.20,
}

ATTRIBUTE_LABEL: Dict[str, str] = {
    "bank_account": "settlement bank account",
    "device_fingerprint": "device fingerprint",
    "address_hash": "registered address",
    "owner_pan": "owner PAN",
    "ip_address": "submission IP address",
    "phone_series": "phone number series",
    "email_domain": "email domain",
}

SUSPICIOUS_SIZE = 3
HIGH_RISK_SIZE = 5
CRITICAL_SIZE = 8

_ADDRESS_NOISE = re.compile(r"[^a-z0-9 ]+")
_ADDRESS_ABBREV = {
    "street": "st", "road": "rd", "apartment": "apt", "building": "bldg",
    "floor": "flr", "opposite": "opp", "near": "nr", "phase": "ph",
    "sector": "sec", "block": "blk", "number": "no", "shop": "shp",
}


class ApplicationClusterer:
    """Links an incoming application to prior applications sharing attributes."""

    def __init__(self, lookback_days: int = 90) -> None:
        self.lookback_days = lookback_days

    # ------------------------------------------------------------------ keys
    def extract_keys(self, app: MerchantApplication) -> Dict[str, str]:
        """The attribute values this application will be indexed under."""
        keys: Dict[str, str] = {
            "bank_account": app.bank_account,
            "owner_pan": app.owner_pan,
            "address_hash": self.address_hash(app.registered_address),
        }
        if app.device_fingerprint:
            keys["device_fingerprint"] = app.device_fingerprint
        if app.ip_address:
            keys["ip_address"] = app.ip_address

        digits = re.sub(r"\D", "", app.owner_phone)
        national = digits[2:] if digits.startswith("91") and len(digits) == 12 else digits
        if len(national) >= 6:
            # First 6 digits identify the operator allocation block; a ring
            # buying SIMs in one batch shares it.
            keys["phone_series"] = national[:6]

        domain = str(app.owner_email).lower().partition("@")[2]
        if domain:
            keys["email_domain"] = domain
        return keys

    @staticmethod
    def address_hash(address: str) -> str:
        """Normalised hash so '12 M.G. Road' and '12 MG Rd' collide.

        Periods are deleted rather than replaced with space, so an initialism
        like 'M.G.' normalises to 'mg' and matches the unpunctuated spelling.
        """
        low = address.lower().replace(".", "")
        low = _ADDRESS_NOISE.sub(" ", low)
        tokens = [_ADDRESS_ABBREV.get(t, t) for t in low.split()]
        # Drop the house/shop number so adjacent units in one building cluster.
        tokens = [t for t in tokens if not (t.isdigit() and len(t) <= 3)]
        return hashlib.sha256(" ".join(sorted(tokens)).encode()).hexdigest()[:16]

    # --------------------------------------------------------------- analysis
    async def analyze(
        self,
        app: MerchantApplication,
        db: Session,
        exclude_id: Optional[str] = None,
    ) -> Dict:
        keys = self.extract_keys(app)
        cutoff = datetime.now(timezone.utc) - timedelta(days=self.lookback_days)

        # attribute -> set of prior merchant ids sharing that exact value
        matches: Dict[str, Set[str]] = {}
        for attr, value in keys.items():
            found = self._lookup(db, attr, value, cutoff, exclude_id)
            if found:
                matches[attr] = found

        related: Set[str] = set()
        for ids in matches.values():
            related |= ids

        cluster_info = self._build_cluster(matches, related, keys)
        signals = self._signals(matches, related, db)
        score = self._score(matches, related, signals)

        return {
            "clustering_risk_score": score,
            "cluster_info": cluster_info,
            "signals": signals,
            "related_application_count": len(related),
        }

    def _lookup(
        self,
        db: Session,
        attr: str,
        value: str,
        cutoff: datetime,
        exclude_id: Optional[str],
    ) -> Set[str]:
        row = db.execute(
            select(DBApplicationCluster).where(
                DBApplicationCluster.cluster_type == attr,
                DBApplicationCluster.cluster_key == value,
            )
        ).scalar_one_or_none()
        if not row or not row.merchant_ids:
            return set()

        candidates = [m for m in row.merchant_ids if m != exclude_id]
        if not candidates:
            return set()

        # Honour the lookback window: only count applications still in scope.
        recent = db.execute(
            select(DBMerchant.id).where(
                DBMerchant.id.in_(candidates), DBMerchant.created_at >= cutoff
            )
        ).scalars().all()
        return set(recent)

    def _build_cluster(
        self,
        matches: Dict[str, Set[str]],
        related: Set[str],
        keys: Dict[str, str],
    ) -> ClusterInfo:
        if not related:
            return ClusterInfo(cluster_size=1)

        shared = sorted(matches, key=lambda a: -ATTRIBUTE_SEVERITY.get(a, 0.0))
        # Stable id derived from the strongest shared attribute's value.
        strongest = shared[0]
        cluster_id = "cl_" + hashlib.sha256(
            f"{strongest}:{keys[strongest]}".encode()
        ).hexdigest()[:12]

        return ClusterInfo(
            cluster_id=cluster_id,
            cluster_size=len(related) + 1,
            shared_attributes=shared,
            related_application_ids=sorted(related),
        )

    def _signals(
        self, matches: Dict[str, Set[str]], related: Set[str], db: Session
    ) -> List[RiskSignal]:
        if not related:
            return []

        signals: List[RiskSignal] = []
        size = len(related) + 1

        for attr, ids in sorted(
            matches.items(), key=lambda kv: -ATTRIBUTE_SEVERITY.get(kv[0], 0.0)
        ):
            severity = ATTRIBUTE_SEVERITY.get(attr, 0.3)
            label = ATTRIBUTE_LABEL.get(attr, attr)
            n = len(ids)
            # Sharing with many is worse than sharing with one.
            scaled = min(1.0, severity * (1 + 0.12 * (n - 1)))
            signals.append(RiskSignal(
                name=f"shared_{attr}",
                value=round(scaled, 4),
                weight=1.4 if severity >= 0.7 else 1.0,
                explanation=(
                    f"Shares {label} with {n} other application"
                    f"{'s' if n != 1 else ''} submitted in the last "
                    f"{self.lookback_days} days."
                ),
                source=SOURCE,
            ))

        if size >= CRITICAL_SIZE:
            band, val, wt = "critical", 0.92, 1.6
        elif size >= HIGH_RISK_SIZE:
            band, val, wt = "high", 0.74, 1.3
        elif size >= SUSPICIOUS_SIZE:
            band, val, wt = "suspicious", 0.52, 1.1
        else:
            band, val, wt = "", 0.0, 1.0

        if band:
            signals.append(RiskSignal(
                name=f"{band}_cluster_size",
                value=val, weight=wt,
                explanation=(
                    f"Application sits in a cluster of {size} linked "
                    f"applications — consistent with a fraud ring submitting "
                    f"in bulk."
                ),
                source=SOURCE,
            ))

        # A cluster that already contains rejected merchants is far worse.
        rejected = db.execute(
            select(DBMerchant.id).where(
                DBMerchant.id.in_(related), DBMerchant.status == "rejected"
            )
        ).scalars().all()
        if rejected:
            signals.append(RiskSignal(
                name="cluster_contains_rejected",
                value=min(1.0, 0.80 + 0.05 * len(rejected)),
                weight=1.7,
                explanation=(
                    f"{len(rejected)} application(s) in this cluster were "
                    f"already rejected — the applicant is linked to known "
                    f"blocked merchants."
                ),
                source=SOURCE,
            ))
        return signals

    def _score(
        self,
        matches: Dict[str, Set[str]],
        related: Set[str],
        signals: List[RiskSignal],
    ) -> float:
        if not related or not signals:
            return 0.0
        total_w = sum(s.weight for s in signals)
        weighted = sum(s.value * s.weight for s in signals) / total_w
        # Independent attribute types corroborating each other raises confidence.
        corroboration = min(0.15, 0.05 * max(0, len(matches) - 1))
        return round(min(1.0, weighted + corroboration), 4)

    # ---------------------------------------------------------------- indexing
    def index_application(
        self, merchant_id: str, keys: Dict[str, str], db: Session
    ) -> None:
        """Add this application to the inverted index. Caller commits."""
        for attr, value in keys.items():
            row = db.execute(
                select(DBApplicationCluster).where(
                    DBApplicationCluster.cluster_type == attr,
                    DBApplicationCluster.cluster_key == value,
                )
            ).scalar_one_or_none()

            if row is None:
                db.add(DBApplicationCluster(
                    cluster_type=attr,
                    cluster_key=value,
                    merchant_ids=[merchant_id],
                    cluster_size=1,
                ))
            elif merchant_id not in row.merchant_ids:
                # JSON columns need reassignment to be marked dirty.
                row.merchant_ids = [*row.merchant_ids, merchant_id]
                row.cluster_size = len(row.merchant_ids)
