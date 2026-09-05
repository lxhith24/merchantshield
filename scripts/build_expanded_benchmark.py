#!/usr/bin/env python3
"""Rebuild the larger offline benchmark without changing the frozen demo."""
from __future__ import annotations

import argparse
import asyncio
from collections import Counter
import hashlib
import json
from pathlib import Path
import statistics
import sys

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from merchantshield.evaluation.dataset import write_jsonl
from merchantshield.evaluation.expanded_dataset import EXPANDED_DATASET_VERSION, generate_expanded_dataset
from merchantshield.evaluation.metrics import CostConfig
from merchantshield.evaluation.runner import evaluate_baselines
from merchantshield.evaluation.split import connected_component_groups


def partition_summary(rows, components):
    return {
        "applications": len(rows),
        "fraud_rings": len({row.ring_id for row in rows if row.ring_id}),
        "connected_components": len({components[row.example_id] for row in rows}),
        "cohorts": dict(sorted(Counter(row.cohort.value for row in rows).items())),
        "missing_device_or_ip": sum(not row.application.device_fingerprint or not row.application.ip_address for row in rows),
    }


async def build_benchmark(*, seed=20250904, scale=1, split_seeds=(20250904, 20250905, 20250906), false_review_cost=1.0):
    if len(set(split_seeds)) != len(split_seeds) or not split_seeds:
        raise ValueError("provide distinct non-empty split seeds")
    examples = generate_expanded_dataset(seed, scale=scale)
    components = connected_component_groups(examples)
    runs = []
    for split_seed in split_seeds:
        result = await evaluate_baselines(examples, seed=split_seed,
            costs=CostConfig(false_review=false_review_cost), review_only=True,
            dataset_version=EXPANDED_DATASET_VERSION)
        train_groups = {components[row.example_id] for row in result.split.train}
        test_groups = {components[row.example_id] for row in result.split.test}
        if train_groups & test_groups:
            raise ValueError("connected components leaked across partitions")
        runs.append({
            "seed": split_seed,
            "train_count": len(result.split.train),
            "test_count": len(result.split.test),
            "partitions": {"train": partition_summary(result.split.train, components),
                           "test": partition_summary(result.split.test, components)},
            "manifest": result.manifest.to_dict(),
            "baselines": {name: metric.to_dict() for name, metric in result.reports.items()},
            "cohort_reports": {cohort: {name: metric.to_dict() for name, metric in metrics.items()}
                               for cohort, metrics in result.cohort_reports.items()},
            "routing_summary": result.routing_summary.to_dict(),
        })
    stability = {}
    for name in runs[0]["baselines"]:
        stability[name] = {}
        for metric in ("precision", "recall", "pr_auc", "false_positive_rate", "ring_recall", "manual_review_rate", "false_positive_cost"):
            values = [run["baselines"][name][metric] for run in runs]
            stability[name][metric] = {"mean": round(statistics.mean(values), 6),
                "min": min(values), "max": max(values), "std": round(statistics.pstdev(values), 6)}
    encoded = "".join(json.dumps(row.to_dict(), sort_keys=True) + "\n" for row in examples)
    report = {
        **runs[0],
        "benchmark_name": "Expanded synthetic benchmark",
        "disclaimer": "Synthetic benchmark results; they are not production Razorpay performance.",
        "generation": {"version": EXPANDED_DATASET_VERSION, "seed": seed, "scale": scale,
                       "dataset_sha256": hashlib.sha256(encoded.encode()).hexdigest()},
        "dataset_summary": partition_summary(examples, components),
        "evaluation_scope": "Existing tabular, graph and hybrid models are fitted afresh on each training partition. The interactive demo keeps its original 104-row data and models; expanded models are not promoted to the review desk.",
        "metric_policy": {
            "name": "human-review-only-with-evidence-gap-guard-v1",
            "screening_threshold": 0.25,
            "precision_recall": "Score > 0.25 is a screening flag, not a confirmed fraud decision.",
            "pr_auc": "Average precision (step-wise precision-recall area).",
            "ring_recall": "A labelled ring is detected when at least one of its positive members is flagged; this does not imply full ring recovery.",
            "manual_review_rate": "Application-level workload: all screening flags plus missing device/IP. Routed MoE also includes actual router uncertainty/disagreement triggers. The evidence-gap guard is a benchmark policy, not a change to the live router.",
            "false_positive_cost": "Legitimate applications sent to review multiplied by configured cost units per review. No autonomous rejection cost is incurred.",
            "false_review_cost": false_review_cost,
            "auto_rejection": False,
        },
        "split_seeds": list(split_seeds),
        "stability": stability,
        "runs": runs,
        "limitations": [
            "All splits reuse one synthetic population and scenario families. Seed ranges are sensitivity checks, not independent trials or confidence intervals.",
            "No threshold or model hyperparameter was selected from these held-out results. Future tuning needs a separate grouped validation partition and a new untouched test.",
            "Evaluation sees the complete held-out batch, not a time-ordered stream. Detection latency and first-merchant performance are not measured.",
            "Synthetic prevalence and labels are design choices. No real-world performance, calibration, identity verification or production readiness is established.",
            "Review rate and cost use a safer definition than the frozen demo report and are not directly comparable to its historical middle-band-only figures.",
        ],
    }
    return examples, report


async def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seed", type=int, default=20250904)
    parser.add_argument("--scale", type=int, default=1)
    parser.add_argument("--split-seeds", type=int, nargs="+", default=[20250904, 20250905, 20250906])
    parser.add_argument("--false-review-cost", type=float, default=1.0)
    parser.add_argument("--output-dir", type=Path, default=Path("data/evaluation"))
    args = parser.parse_args()
    rows, report = await build_benchmark(seed=args.seed, scale=args.scale,
        split_seeds=tuple(args.split_seeds), false_review_cost=args.false_review_cost)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    dataset_path = args.output_dir / "expanded_merchants.jsonl"
    report_path = args.output_dir / "expanded_report.json"
    write_jsonl(rows, dataset_path)
    report["dataset"] = str(dataset_path)
    report_path.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps({"dataset": str(dataset_path), "report": str(report_path),
        "partitions": report["partitions"], "baselines": report["baselines"]}, indent=2))


if __name__ == "__main__":
    asyncio.run(main())
