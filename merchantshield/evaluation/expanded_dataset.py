"""Larger defensive benchmark fixtures; no real identities or external calls.

The frozen 104-row demo remains separate. These are deliberately overlapping
synthetic populations, not a simulator of real criminal behaviour.
"""
from __future__ import annotations

import random
from datetime import timedelta

from ..models.merchant import BusinessType, MerchantApplication
from .dataset import Cohort, SyntheticExample, _BASE_TIME, _base_application, _token

EXPANDED_DATASET_VERSION = "synthetic-rings-v3-expanded"


def generate_expanded_dataset(seed: int = 20250904, *, scale: int = 1) -> list[SyntheticExample]:
    """About 2,000 rows / 120 fraud rings per scale, bounded to ten scales.

    Both labels can share accounts, owners, devices, addresses or networks;
    neither group size nor submission speed is a perfect label shortcut.
    Labelled components include partial links and occasional innocent members.
    Missing optional evidence is explicit, never filled with a clean value.
    """
    if isinstance(scale, bool) or not isinstance(scale, int) or not 1 <= scale <= 10:
        raise ValueError("scale must be an integer between 1 and 10")
    rng = random.Random(seed)
    specifications = []
    for cohort, groups in (
        (Cohort.LEGITIMATE, 400),
        (Cohort.LEGITIMATE_SHARED_INFRA, 100),
        (Cohort.OBVIOUS_SHELL_RING, 60),
        (Cohort.EVASIVE_SHELL_RING, 60),
        (Cohort.BORDERLINE, 100),
    ):
        specifications.extend([cohort] * groups * scale)
    rng.shuffle(specifications)
    rows = []
    patterns = ("office", "device", "owner", "settlement", "partial", "network")
    for group_index, cohort in enumerate(specifications):
        shell = cohort in (Cohort.OBVIOUS_SHELL_RING, Cohort.EVASIVE_SHELL_RING)
        size = (1 if cohort is Cohort.LEGITIMATE else
                rng.randint(1, 3) if cohort is Cohort.BORDERLINE else rng.randint(3, 10))
        pattern = rng.choice(patterns)
        component = f"expanded-group-{group_index:05d}-{pattern}"
        ring = f"expanded-ring-{group_index:05d}" if shell else None
        token = _token(rng, 12)
        start = _BASE_TIME + timedelta(days=rng.randint(0, 300))
        # Fast legitimate filings and slow labelled rings are intentional.
        fast = rng.random() < (0.65 if shell else 0.30)
        step = rng.choice((2, 12, 50, 180)) if fast else rng.randint(1440, 20160)
        common_owner = _base_application(30000 + group_index, rng)
        mixed = shell and group_index % 8 == 0
        for member in range(size):
            index = len(rows)
            app = _base_application(index, rng)
            # Format-safe unique phone allocation, including at larger scales.
            app["owner_phone"] = f"+916{index + 10000:05d}{rng.randint(1000, 9999):04d}"
            # No serial-number email rule shortcut after the 1,000th row.
            app["owner_email"] = f"contact@merchant-{index:05d}-{token}.example"
            app["submitted_at"] = start + timedelta(minutes=member * step)
            app["business_type"] = rng.choice(list(BusinessType)) if rng.random() < .2 else app["business_type"]
            if size > 1:
                if pattern == "office":
                    app["registered_address"] = f"{member + 1} {token} Commerce Park, Pune Maharashtra 411001"
                    app["ip_address"] = f"fd00::{group_index + 1:x}"
                elif pattern == "device":
                    app["device_fingerprint"] = f"fixture-device-{token}"
                    if rng.random() < .7:
                        app["ip_address"] = f"fd00::{group_index + 1:x}"
                elif pattern == "owner":
                    app["owner_pan"] = common_owner["owner_pan"]
                    app["owner_name"] = common_owner["owner_name"]
                elif pattern == "settlement":
                    app["bank_account"] = common_owner["bank_account"]
                elif pattern == "partial":
                    # Alternating pair links form a sparse chain, not a clique.
                    app["device_fingerprint"] = f"fixture-pair-{token}-{member // 2}"
                    app["bank_account"] = str(880000000000 + group_index * 20 + (member + 1) // 2)
                else:
                    app["ip_address"] = f"fd00::{group_index + 1:x}"
                if rng.random() < .35:
                    app["owner_email"] = f"contact{member}@office-{token}.example"

            positive = shell and not (mixed and member == size - 1)
            row_cohort = Cohort.LEGITIMATE_SHARED_INFRA if shell and not positive else cohort
            noise = .75 if cohort is Cohort.OBVIOUS_SHELL_RING and positive else .3 if cohort is Cohort.BORDERLINE else .06
            if rng.random() < noise:
                app["owner_pan"] = app["owner_pan"][:4] + "Z" + app["owner_pan"][5:]
            if rng.random() < noise:
                app["business_name"] = "Test " + app["business_name"]
            if rng.random() < noise:
                app["registered_address"] = f"Unit {token}, Pune"
            if rng.random() < .55:
                app["gst_number"] = f"27{app['owner_pan']}1Z5"
            for field, probability in (("device_fingerprint", .15), ("ip_address", .15), ("website_url", .25)):
                if rng.random() < probability:
                    app[field] = None
            rows.append(SyntheticExample(
                example_id=f"expanded-{index:05d}",
                application=MerchantApplication(**app),
                is_shell=positive,
                cohort=row_cohort,
                ring_id=ring if positive else None,
                component_id=component,
            ))
    return rows
