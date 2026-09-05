"""Safe, transient checks for newly submitted synthetic merchant details.

The live-check surface is deliberately separate from the frozen evaluation
dataset and the review queue. It compares a small submitted batch with itself
and with the existing synthetic demonstration population, runs the real router
and bounded investigator, and returns redacted evidence suitable for the UI.
Nothing is fitted, persisted, approved, rejected, or sent to a gateway.
"""
from __future__ import annotations

import csv
import hashlib
import io
import ipaddress
import re
from typing import TYPE_CHECKING, Any, Dict, List, Mapping, Sequence, Tuple

from ..analysis.evidence_graph import EvidenceGraphBuilder, RingCandidate
from ..models.merchant import BusinessType, MerchantApplication
from ..workflow.graph import InMemoryCaseRepository, MerchantShieldWorkflow, WorkflowRun

if TYPE_CHECKING:
    from ..runtime import MerchantShieldRuntime


MAX_LIVE_CHECK_BYTES = 1_000_000
# A single transient run can cover a work queue while remaining bounded for
# the local demonstration and the API request body.
MAX_LIVE_CHECK_ROWS = 100

REQUIRED_COLUMNS = (
    "merchant_id",
    "business_name",
    "owner_name",
    "bank_account",
    "device_fingerprint",
    "ip_address",
    "registered_address",
    "submitted_at",
)

_MERCHANT_ID = re.compile(r"^[A-Za-z0-9_-]{2,40}$")
_BANK_ACCOUNT = re.compile(r"^[0-9]{9,18}$")
_LETTERS = "ABCDEFGHIJKLMNOPQRSTUVWXYZ"
_ALNUM = "ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789"
_PAN_HOLDER_TYPES = "PCHFATBLJG"

LIVE_CHECK_EXAMPLES: Dict[str, Dict[str, str]] = {
    "known_ring_match": {
        "label": "known_ring_match.csv",
        "description": "One new application shares strong infrastructure with a stored synthetic ring.",
        "csv": """merchant_id,business_name,owner_name,bank_account,device_fingerprint,ip_address,registered_address,submitted_at
LIVE-701,Northstar Digital Supply,Anaya Rao,111111111111,obvious-device-0,198.51.100.71,"Test Address Block A, Noida 201301",2025-01-01T00:51:00Z
""",
    },
    "new_ring": {
        "label": "alpha_ring.csv",
        "description": "Three unseen applications share a settlement account and two share a device.",
        "csv": """merchant_id,business_name,owner_name,bank_account,device_fingerprint,ip_address,registered_address,submitted_at
LIVE-801,Prime Mobile Traders,Aarav Sharma,9921678201,device-demo-3321,198.51.100.11,"12 MG Road, Bengaluru, Karnataka 560001",2026-01-01T10:00:00Z
LIVE-802,Sunrise Electronics Hub,Vihaan Iyer,9921678201,device-demo-3321,198.51.100.27,"45 Brigade Road, Bengaluru, Karnataka 560025",2026-01-01T10:00:15Z
LIVE-803,Metro Gadget Bazaar,Kabir Nair,9921678201,device-demo-7788,198.51.100.42,"9 Residency Road, Bengaluru, Karnataka 560025",2026-01-01T10:00:45Z
""",
    },
    "independent": {
        "label": "beta_clean.csv",
        "description": "Three unseen applications have distinct infrastructure and separated submission times.",
        "csv": """merchant_id,business_name,owner_name,bank_account,device_fingerprint,ip_address,registered_address,submitted_at
LIVE-901,Coastal Home Decor,Meera Pillai,1002003001,device-demo-5510,203.0.113.10,"88 Marine Drive, Kochi, Kerala 682001",2026-01-05T09:00:00Z
LIVE-902,Northside Bakery Supplies,Rohan Das,2003004002,device-demo-6620,203.0.113.24,"17 Park Street, Kolkata, West Bengal 700016",2026-01-20T14:30:00Z
LIVE-903,Highland Auto Parts,Ishaan Verma,3004005003,device-demo-7733,203.0.113.38,"203 Ring Road, Nagpur, Maharashtra 440001",2026-02-10T11:15:00Z
""",
    },
}


def _digest(seed: str) -> bytes:
    return hashlib.sha256(seed.encode("utf-8")).digest()


def _synthetic_pan(merchant_id: str, owner_name: str) -> str:
    digest = _digest(f"pan:{merchant_id}")
    holder = _PAN_HOLDER_TYPES[digest[3] % len(_PAN_HOLDER_TYPES)]
    surname = [part for part in owner_name.strip().upper().split() if part]
    fifth = surname[-1][0] if holder == "P" and surname else _LETTERS[digest[4] % 26]
    prefix = "".join(_LETTERS[byte % 26] for byte in digest[:3])
    digits = "".join(str(byte % 10) for byte in digest[5:9])
    return f"{prefix}{holder}{fifth}{digits}{_LETTERS[digest[9] % 26]}"


def _synthetic_ifsc(merchant_id: str) -> str:
    digest = _digest(f"ifsc:{merchant_id}")
    bank = "".join(_LETTERS[byte % 26] for byte in digest[:4])
    branch = "".join(_ALNUM[byte % 36] for byte in digest[4:10])
    return f"{bank}0{branch}"


def _synthetic_phone(merchant_id: str) -> str:
    digest = _digest(f"phone:{merchant_id}")
    lead = "6789"[digest[0] % 4]
    return "+91" + lead + "".join(str(byte % 10) for byte in digest[1:10])


def _clean(value: object) -> str:
    return str(value).strip() if value is not None else ""


def parse_live_check_csv(csv_text: str) -> Dict[str, MerchantApplication]:
    """Validate a bounded CSV and create synthetic-only application models."""
    encoded = csv_text.encode("utf-8")
    if len(encoded) > MAX_LIVE_CHECK_BYTES:
        raise ValueError(f"CSV exceeds the {MAX_LIVE_CHECK_BYTES // 1_000_000} MB demo limit")
    reader = csv.DictReader(io.StringIO(csv_text.lstrip("\ufeff")))
    headers = tuple(reader.fieldnames or ())
    missing = [column for column in REQUIRED_COLUMNS if column not in headers]
    if missing:
        raise ValueError("CSV is missing required columns: " + ", ".join(missing))
    rows = list(reader)
    if not rows:
        raise ValueError("CSV has no merchant rows")
    if len(rows) > MAX_LIVE_CHECK_ROWS:
        raise ValueError(f"CSV exceeds the {MAX_LIVE_CHECK_ROWS}-merchant demo limit")

    applications: Dict[str, MerchantApplication] = {}
    for index, row in enumerate(rows, start=2):
        merchant_id = _clean(row.get("merchant_id"))
        if not _MERCHANT_ID.fullmatch(merchant_id):
            raise ValueError(
                f"row {index}: merchant_id must be 2–40 letters, numbers, hyphens or underscores"
            )
        if merchant_id in applications:
            raise ValueError(f"row {index}: duplicate merchant_id {merchant_id}")
        bank_account = _clean(row.get("bank_account"))
        if not _BANK_ACCOUNT.fullmatch(bank_account):
            raise ValueError(f"row {index}: bank_account must contain 9–18 digits")
        device = _clean(row.get("device_fingerprint"))
        ip = _clean(row.get("ip_address"))
        if not device:
            raise ValueError(f"row {index}: device_fingerprint is required; missing evidence is not clean")
        try:
            ipaddress.ip_address(ip)
        except ValueError as exc:
            raise ValueError(f"row {index}: ip_address is invalid") from exc
        owner_name = _clean(row.get("owner_name"))
        slug = merchant_id.lower()
        try:
            applications[merchant_id] = MerchantApplication(
                business_name=_clean(row.get("business_name")),
                business_type=BusinessType.RETAIL,
                owner_name=owner_name,
                owner_pan=_synthetic_pan(merchant_id, owner_name),
                owner_email=f"{slug}@{slug}.example",
                owner_phone=_synthetic_phone(merchant_id),
                bank_account=bank_account,
                bank_ifsc=_synthetic_ifsc(merchant_id),
                registered_address=_clean(row.get("registered_address")),
                device_fingerprint=device,
                ip_address=ip,
                submitted_at=_clean(row.get("submitted_at")),
            )
        except ValueError as exc:
            raise ValueError(f"row {index}: invalid merchant details: {exc}") from exc
    return applications


def _member_payload(
    member_id: str,
    application: MerchantApplication,
    submitted_ids: set[str],
) -> Dict[str, Any]:
    return {
        "member_id": member_id,
        "business_name": application.business_name,
        "business_type": application.business_type.value,
        "submitted_at": application.submitted_at.isoformat(),
        "source": "submitted" if member_id in submitted_ids else "synthetic_reference",
    }


def _run_payload(
    run: WorkflowRun,
    candidate: RingCandidate,
    applications: Mapping[str, MerchantApplication],
    submitted_ids: set[str],
) -> Dict[str, Any]:
    state = run.state
    assessment = state.assessment
    submitted_members = sorted(submitted_ids.intersection(candidate.member_ids))
    reference_members = sorted(set(candidate.member_ids) - submitted_ids)
    if reference_members:
        finding = "linked_to_synthetic_reference"
    elif len(candidate.member_ids) > 1:
        finding = "new_linked_group"
    else:
        finding = "no_ring_link_found"
    return {
        "finding": finding,
        "candidate": candidate.to_dict(),
        "members": [
            _member_payload(member_id, applications[member_id], submitted_ids)
            for member_id in candidate.member_ids
        ],
        "submitted_member_ids": submitted_members,
        "reference_member_ids": reference_members,
        "primary_risk_score": assessment.primary_risk_score if assessment else None,
        "action": assessment.action.value if assessment else "human_review",
        "reason_codes": list(assessment.reason_codes) if assessment else ["ASSESSMENT_UNAVAILABLE"],
        "route_trace": (
            [item.model_dump(mode="json") for item in assessment.route_trace]
            if assessment else []
        ),
        "investigation": state.investigation.model_dump(mode="json") if state.investigation else None,
        "decision": state.decision.model_dump(mode="json") if state.decision else None,
        "workflow_trace": [item.model_dump(mode="json") for item in state.workflow_trace],
        "awaiting_information": run.is_awaiting_information,
    }


async def run_live_check(
    submitted: Mapping[str, MerchantApplication],
    *,
    runtime: "MerchantShieldRuntime",
) -> Dict[str, Any]:
    """Compare submitted merchants with each other and the synthetic reference world."""
    duplicate_ids = sorted(set(submitted).intersection(runtime.world.applications))
    if duplicate_ids:
        raise ValueError("merchant_id already exists in the synthetic reference set: " + ", ".join(duplicate_ids))
    combined = dict(runtime.world.applications)
    combined.update(submitted)
    candidates = EvidenceGraphBuilder().build(combined)
    submitted_ids = set(submitted)
    relevant = tuple(
        candidate for candidate in candidates
        if submitted_ids.intersection(candidate.member_ids)
    )
    repository = InMemoryCaseRepository(relevant, combined)
    workflow = MerchantShieldWorkflow(
        repository=repository,
        router=runtime.workflow.router,
        investigator=runtime.workflow.investigator,
        memory=runtime.workflow.memory,
    )
    runs: List[Dict[str, Any]] = []
    for candidate in relevant:
        run = await workflow.run(candidate.candidate_id, case_id=f"live-{candidate.candidate_id}")
        runs.append(_run_payload(run, candidate, combined, submitted_ids))
    runs.sort(key=lambda item: (item["finding"] == "no_ring_link_found", -len(item["candidate"]["member_ids"])))
    return {
        "submitted_count": len(submitted),
        "reference_count": len(runtime.world.applications),
        "result_count": len(runs),
        "persisted": False,
        "gateway_called": False,
        "results": runs,
    }
