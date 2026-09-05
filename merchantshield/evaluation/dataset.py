"""Reproducible, labelled, entirely synthetic merchant evaluation data."""
from __future__ import annotations

import json
import random
import string
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from enum import Enum
from pathlib import Path
from typing import Iterable, List, Optional, Sequence

from ..models.merchant import BusinessType, MerchantApplication


class Cohort(str, Enum):
    LEGITIMATE = "legitimate"
    LEGITIMATE_SHARED_INFRA = "legitimate_shared_infrastructure"
    OBVIOUS_SHELL_RING = "obvious_shell_ring"
    EVASIVE_SHELL_RING = "evasive_shell_ring"
    BORDERLINE = "borderline"


@dataclass(frozen=True)
class SyntheticExample:
    """One benchmark row plus labels that are never exposed to a baseline."""

    example_id: str
    application: MerchantApplication
    is_shell: bool
    cohort: Cohort
    ring_id: Optional[str] = None
    component_id: Optional[str] = None

    def to_dict(self) -> dict:
        return {
            "example_id": self.example_id,
            "label": {
                "is_shell": self.is_shell,
                "cohort": self.cohort.value,
                "ring_id": self.ring_id,
                "component_id": self.component_id,
            },
            "application": self.application.model_dump(mode="json"),
        }

    @classmethod
    def from_dict(cls, data: dict) -> "SyntheticExample":
        label = data["label"]
        return cls(
            example_id=data["example_id"],
            application=MerchantApplication.model_validate(data["application"]),
            is_shell=bool(label["is_shell"]),
            cohort=Cohort(label["cohort"]),
            ring_id=label.get("ring_id"),
            component_id=label.get("component_id"),
        )


_SURNAMES = (
    "Sharma", "Patel", "Reddy", "Nair", "Mehta", "Singh", "Gupta",
    "Iyer", "Joshi", "Kapoor", "Das", "Verma", "Yadav", "Menon",
)
_CITIES = (
    ("Bangalore", "Karnataka", 560001),
    ("Mumbai", "Maharashtra", 400001),
    ("Chennai", "Tamil Nadu", 600001),
    ("Hyderabad", "Telangana", 500001),
    ("Pune", "Maharashtra", 411001),
    ("Jaipur", "Rajasthan", 302001),
    ("Kochi", "Kerala", 682001),
    ("Noida", "Uttar Pradesh", 201301),
)
_BUSINESS_TYPES = (
    BusinessType.RETAIL,
    BusinessType.ECOMMERCE,
    BusinessType.SERVICES,
    BusinessType.FOOD,
)
_DISPOSABLE_DOMAINS = (
    "tempmail.com", "guerrillamail.com", "mailinator.com", "yopmail.com",
)
_CONSUMER_DOMAINS = (
    "gmail.com", "yahoo.com", "outlook.com", "hotmail.com",
    "rediffmail.com", "protonmail.com", "icloud.com", "live.com", "aol.com",
)
_BASE_TIME = datetime(2025, 1, 1, tzinfo=timezone.utc)


def generate_synthetic_dataset(seed: int = 20250904) -> List[SyntheticExample]:
    """Build the fixed-size benchmark deterministically from ``seed``.

    The identities are invented and the domains are reserved or disposable.
    The five cohorts deliberately include hard negatives: shared infrastructure
    and borderline legitimate merchants should not be mistaken for shell rings.
    """
    rng = random.Random(seed)
    rows: List[SyntheticExample] = []
    serial = 0

    def add(
        cohort: Cohort,
        is_shell: bool,
        *,
        ring_id: Optional[str] = None,
        component_id: Optional[str] = None,
        **overrides,
    ) -> None:
        nonlocal serial
        app = _base_application(serial, rng)
        app.update(overrides)
        app.setdefault("submitted_at", _BASE_TIME + timedelta(minutes=serial))
        rows.append(SyntheticExample(
            example_id=f"syn-{serial:04d}",
            application=MerchantApplication(**app),
            is_shell=is_shell,
            cohort=cohort,
            ring_id=ring_id,
            component_id=component_id or f"singleton-{serial:04d}",
        ))
        serial += 1

    # Ordinary legitimate merchants: unique infrastructure and consistent data.
    for _ in range(30):
        add(Cohort.LEGITIMATE, False)

    # Legitimate shared-infrastructure groups are deliberately varied. They
    # create hard negatives without copying one fixed fraud-generation recipe.
    for group in range(4):
        component = f"legit-shared-{group:02d}"
        city, state, pin = _CITIES[group]
        shared_owner_name = "Franchise Owner Patel"
        shared_owner_pan = _pan(800 + group, "P", rng)
        for member in range(4):
            common = {
                "component_id": component,
                "business_name": f"Independent Market {group}-{member}",
                "owner_email": f"owner{member}@market-hub-{group}.example",
                # Legitimate groups accumulate over weeks, unlike bulk rings.
                "submitted_at": _BASE_TIME
                + timedelta(days=10 + group * 40 + member * 9),
            }
            if group == 0:  # co-working address and office network
                common.update(
                    registered_address=f"{20 + member} Commerce Hub, {city}, {state} {pin}",
                    ip_address="198.51.100.20",
                )
            elif group == 1:  # accountant filing from one managed device
                common.update(
                    device_fingerprint="accountant-device-01",
                    ip_address="198.51.100.21",
                )
            elif group == 2:  # legitimate companies under one disclosed owner
                common.update(
                    owner_name=shared_owner_name,
                    owner_pan=shared_owner_pan,
                )
            else:  # intentionally difficult shared kiosk/address/network group
                common.update(
                    registered_address=f"{20 + member} Commerce Hub, {city}, {state} {pin}",
                    device_fingerprint="shared-kiosk-03",
                    ip_address="198.51.100.23",
                )
            add(
                Cohort.LEGITIMATE_SHARED_INFRA,
                False,
                **common,
            )

    # Obvious shells contain strong per-application tells as well as ring links.
    for ring in range(4):
        ring_id = f"obvious-ring-{ring:02d}"
        repeated = str(ring + 1)
        phone_digit = str(6 + ring)
        for member in range(5):
            surname = _SURNAMES[(ring * 5 + member) % len(_SURNAMES)]
            add(
                Cohort.OBVIOUS_SHELL_RING,
                True,
                ring_id=ring_id,
                component_id=ring_id,
                business_name=f"Test Trading {ring}-{member}",
                owner_name=f"Synthetic {surname}",
                owner_pan=_pan(serial, "Z", rng),  # deliberately mismatched
                owner_email=f"shell{member}@{_DISPOSABLE_DOMAINS[ring]}",
                owner_phone=f"+91{phone_digit * 10}",
                bank_account=repeated * 12,
                registered_address=(
                    f"Test Address Block {string.ascii_uppercase[ring]}, Noida 201301"
                ),
                device_fingerprint=f"obvious-device-{ring}",
                ip_address=f"203.0.113.{10 + ring}",
            )

    # Evasive shells look clean individually; graph linkage is their principal
    # evidence. Accounts and owners remain unique while device/address/IP recur.
    for ring in range(4):
        ring_id = f"evasive-ring-{ring:02d}"
        city, state, pin = _CITIES[ring + 4]
        for member in range(5):
            add(
                Cohort.EVASIVE_SHELL_RING,
                True,
                ring_id=ring_id,
                component_id=ring_id,
                business_name=f"Northstar Commerce {ring}-{member}",
                owner_email=f"accounts{member}@northstar-{ring}.example",
                registered_address=f"{300 + member} Logistics Park, {city}, {state} {pin}",
                device_fingerprint=f"evasive-device-{ring}",
                ip_address=f"192.0.2.{40 + ring}",
            )

    # Borderline but legitimate: soft identity rules should cause some reviews,
    # allowing false-positive cost and manual-review load to be measured.
    for member in range(18):
        domain = _CONSUMER_DOMAINS[member % len(_CONSUMER_DOMAINS)]
        add(
            Cohort.BORDERLINE,
            False,
            component_id=f"borderline-domain-{domain}",
            business_name=f"Local Ventures {member}",
            business_type=(BusinessType.TRAVEL if member % 3 == 0 else BusinessType.SERVICES),
            owner_email=f"personal{1000 + member}@{domain}",
            registered_address=(
                f"Unit {member + 1}, {domain.partition('.')[0].title()} Market Road, Delhi"
            ),
        )

    return rows


def write_jsonl(examples: Iterable[SyntheticExample], path: str | Path) -> None:
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    with target.open("w", encoding="utf-8") as handle:
        for example in examples:
            handle.write(json.dumps(example.to_dict(), sort_keys=True) + "\n")


def load_jsonl(path: str | Path) -> List[SyntheticExample]:
    with Path(path).open(encoding="utf-8") as handle:
        return [SyntheticExample.from_dict(json.loads(line)) for line in handle if line.strip()]


def _base_application(index: int, rng: random.Random) -> dict:
    surname = _SURNAMES[index % len(_SURNAMES)]
    city, state, pin = _CITIES[index % len(_CITIES)]
    token = _token(rng, 7)
    return {
        "business_name": f"{surname} {token.title()} Enterprises",
        "business_type": _BUSINESS_TYPES[index % len(_BUSINESS_TYPES)],
        "website_url": f"https://merchant-{token}.example",
        "owner_name": f"Owner{index} {surname}",
        "owner_pan": _pan(index, surname[0], rng),
        "owner_email": f"owner{index}@merchant-{token}.example",
        "owner_phone": _phone(index),
        "bank_account": f"{470000000000 + index * 7919:012d}",
        "bank_ifsc": f"MSHD0{index:06d}",
        "registered_address": (
            f"{100 + index} {token.title()} Market Road, {city}, {state} {pin}"
        ),
        "device_fingerprint": f"device-{token}-{index}",
        "ip_address": f"10.{(index // 250) % 250}.{(index // 25) % 250}.{index % 250 + 1}",
    }


def _pan(index: int, surname_initial: str, rng: random.Random) -> str:
    prefix = "".join(rng.choice(string.ascii_uppercase) for _ in range(3))
    checksum = string.ascii_uppercase[index % 26]
    return f"{prefix}P{surname_initial.upper()}{(1000 + index) % 10000:04d}{checksum}"


def _phone(index: int) -> str:
    first = 6 + index % 4
    # Keep the six-digit allocation prefix unique outside deliberately linked
    # rings, otherwise the production clusterer's phone-series key would join
    # unrelated synthetic components and compromise the held-out split.
    allocation = 23_456 + index * 37
    subscriber = (7919 * (index + 1)) % 10_000
    return f"+91{first}{allocation:05d}{subscriber:04d}"


def _token(rng: random.Random, length: int) -> str:
    return "".join(rng.choice(string.ascii_lowercase) for _ in range(length))
