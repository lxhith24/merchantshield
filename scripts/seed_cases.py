#!/usr/bin/env python3
"""Open a ring case for every candidate group so the review queue is populated.

Every score, reason code and investigation below is produced by the real
Phase 2/3 pipeline; nothing is hard-coded. Safe to re-run: opening a case is
idempotent per candidate.

    .venv/bin/python scripts/seed_cases.py [--reset] [--min-members N]
"""
from __future__ import annotations

import argparse
import asyncio
import os
import sys
from collections import Counter

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from merchantshield.db.database import engine, init_db  # noqa: E402
from merchantshield.db.models import Base  # noqa: E402
from merchantshield.review.contracts import CaseStatus  # noqa: E402
from merchantshield.runtime import build_runtime  # noqa: E402


async def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--reset", action="store_true", help="drop case tables first")
    parser.add_argument("--min-members", type=int, default=1)
    args = parser.parse_args()

    if args.reset:
        Base.metadata.drop_all(bind=engine)
    init_db()

    runtime = await build_runtime()
    print(f"labels: {runtime.labels}")
    print(f"candidates in frozen synthetic world: {len(runtime.world.candidates)}")

    statuses: Counter[str] = Counter()
    for candidate in runtime.world.candidates:
        if len(candidate.member_ids) < args.min_members:
            continue
        view = await runtime.service.open_case(candidate.candidate_id)
        statuses[view.status.value] += 1

    print("\nseeded cases by status:")
    for status in CaseStatus:
        if statuses.get(status.value):
            print(f"  {status.value:<24} {statuses[status.value]}")

    total, rows = runtime.service.list_cases(status=CaseStatus.PENDING_REVIEW, limit=5)
    print(f"\ntop of the review queue ({total} awaiting a human):")
    for row in rows:
        print(
            f"  {row.case_id}  risk={row.automated.primary_risk_score:.3f}  "
            f"members={row.member_count}  "
            f"reasons={','.join(row.automated.reason_codes)}"
        )
    print("\nAll data is synthetic. Start the API with ./run.sh and open /docs.")
    return 0


if __name__ == "__main__":
    raise SystemExit(asyncio.run(main()))
