#!/usr/bin/env python3
"""Generate the labelled synthetic benchmark and print held-out baselines."""
from __future__ import annotations

import argparse
import asyncio
import json
import os
import sys
from pathlib import Path

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from merchantshield.evaluation import (  # noqa: E402
    CostConfig,
    evaluate_baselines,
    generate_synthetic_dataset,
    write_jsonl,
)


async def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", default="data/evaluation/synthetic_merchants.jsonl")
    parser.add_argument(
        "--report-output", default="data/evaluation/latest_report.json"
    )
    parser.add_argument("--seed", type=int, default=20250904)
    parser.add_argument("--test-fraction", type=float, default=0.30)
    parser.add_argument("--false-review-cost", type=float, default=1.0)
    parser.add_argument("--false-reject-cost", type=float, default=5.0)
    args = parser.parse_args()

    examples = generate_synthetic_dataset(seed=args.seed)
    write_jsonl(examples, args.output)
    evaluation = await evaluate_baselines(
        examples,
        test_fraction=args.test_fraction,
        seed=args.seed,
        costs=CostConfig(
            false_review=args.false_review_cost,
            false_reject=args.false_reject_cost,
        ),
    )
    output = {
        "dataset": args.output,
        "seed": args.seed,
        "train_count": len(evaluation.split.train),
        "test_count": len(evaluation.split.test),
        "disclaimer": evaluation.disclaimer,
        "manifest": evaluation.manifest.to_dict(),
        "baselines": {name: report.to_dict() for name, report in evaluation.reports.items()},
        "routing_summary": evaluation.routing_summary.to_dict(),
    }
    rendered = json.dumps(output, indent=2, sort_keys=True) + "\n"
    report_path = Path(args.report_output)
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(rendered, encoding="utf-8")
    print(rendered, end="")


if __name__ == "__main__":
    asyncio.run(main())
