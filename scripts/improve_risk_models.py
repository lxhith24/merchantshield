#!/usr/bin/env python3
"""Tune on validation, lock the choice, then evaluate a fresh grouped test once.

Run --stage tune first. Inspect validation results before --stage test.
The expanded generator and its labels are unchanged from the earlier benchmark.
"""
from __future__ import annotations

import argparse
import asyncio
from dataclasses import replace
import hashlib
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from merchantshield.evaluation.dataset import load_jsonl, write_jsonl
from merchantshield.evaluation.expanded_dataset import generate_expanded_dataset
from merchantshield.evaluation.performance import MerchantPeerModel, extract_features, measure, select_threshold, VERSION
from merchantshield.evaluation.baselines import RulesOnlyBaseline, ClusteringOnlyBaseline
from merchantshield.evaluation.ring_models import GraphMLBaseline, GraphOnlyBaseline, HybridRingBaseline
from merchantshield.evaluation.split import connected_component_groups, group_train_test_split
from scripts.build_expanded_benchmark import partition_summary

SEED = 20260905
SYSTEMS = ("rules_only", "clustering_only", "graph_only", "tabular_only", "graph_ml", "hybrid", "routed_moe")


def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True).encode()).hexdigest()


def code_hash():
    files = ("merchantshield/evaluation/performance.py", "merchantshield/evaluation/expanded_dataset.py",
             "merchantshield/evaluation/split.py", "merchantshield/evaluation/metrics.py",
             "merchantshield/evaluation/ring_models.py", "merchantshield/analysis/evidence_graph.py",
             "merchantshield/analysis/synthetic_identity.py", "merchantshield/analysis/application_clustering.py",
             "scripts/improve_risk_models.py")
    return digest({file: hashlib.sha256((ROOT / file).read_bytes()).hexdigest() for file in files})


async def features(rows):
    return await extract_features({row.example_id: row.application for row in rows})


def split_rows(rows):
    outer = group_train_test_split(rows, test_fraction=.2, seed=SEED)
    inner = group_train_test_split(outer.train, test_fraction=.25, seed=SEED + 1)
    partitions = {"train": inner.train, "validation": inner.test, "test": outer.test}
    groups = connected_component_groups(rows)
    assigned = {}
    for name, members in partitions.items():
        for row in members:
            group = groups[row.example_id]
            if group in assigned and assigned[group] != name:
                raise ValueError("component crossed partitions")
            assigned[group] = name
    return partitions, groups


async def baseline_scores(rows):
    return {"rules_only": await RulesOnlyBaseline().score(rows),
            "clustering_only": ClusteringOnlyBaseline().score(rows),
            "graph_only": GraphOnlyBaseline().score(rows)}


async def tune(output, *, false_review_cost=1., missed_fraud_cost=1.):
    if (output / "performance_report.json").exists():
        raise ValueError("final test already evaluated here; use a new protocol/population for further tuning")
    output.mkdir(parents=True, exist_ok=True)
    rows = [replace(row, example_id="perf-" + row.example_id,
                    ring_id="perf-" + row.ring_id if row.ring_id else None,
                    component_id="perf-" + row.component_id) for row in generate_expanded_dataset(SEED, scale=2)]
    partitions, groups = split_rows(rows)
    train, validation = partitions["train"], partitions["validation"]
    train_features, val_features = await features(train), await features(validation)
    labels = {row.example_id: row.is_shell for row in train}
    trials = []
    for depth, leaf in ((2, 20), (2, 40), (3, 20), (3, 40)):
        model = MerchantPeerModel(depth=depth, leaf=leaf, seed=SEED).fit(train_features, labels)
        scores, calls = model.score(val_features)
        selected = select_threshold(validation, scores, false_review_cost=false_review_cost, missed_fraud_cost=missed_fraud_cost)
        trials.append({"depth": depth, "leaf": leaf, **selected, "specialist_applications": calls})
    best = min(trials, key=lambda trial: (trial["objective"], trial["metrics"]["false_positive_rate"], trial["depth"], trial["leaf"]))
    model = MerchantPeerModel(depth=best["depth"], leaf=best["leaf"], seed=SEED).fit(train_features, labels)
    validation_scores = await baseline_scores(validation)
    for name in SYSTEMS[3:]:
        validation_scores[name], _ = model.score(val_features, system=name)
    thresholds = {}
    for name, scores in validation_scores.items():
        if name in SYSTEMS[:3]:
            thresholds[name] = {"threshold": .25, "selection": "unchanged deterministic baseline",
                                "metrics": measure(validation, scores, .25, false_review_cost=false_review_cost).to_dict()}
        else:
            thresholds[name] = select_threshold(validation, scores, false_review_cost=false_review_cost, missed_fraud_cost=missed_fraud_cost)
    # Include the old graph model with BOTH fixed and validation-selected
    # thresholds, isolating model improvement from moving the operating point.
    old = GraphMLBaseline(seed=SEED)
    old.fit(train)
    thresholds["legacy_graph_tuned"] = select_threshold(validation, old.score(validation), false_review_cost=false_review_cost, missed_fraud_cost=missed_fraud_cost)
    old_hybrid = HybridRingBaseline(seed=SEED)
    await old_hybrid.fit(train)
    thresholds["legacy_hybrid_tuned"] = select_threshold(validation, await old_hybrid.score(validation), false_review_cost=false_review_cost, missed_fraud_cost=missed_fraud_cost)
    plan = {"version": VERSION, "seed": SEED, "scale": 2, "code_hash": code_hash(),
            "dataset_hash": digest([row.to_dict() for row in rows]),
            "partitions": {name: partition_summary(members, groups) for name, members in partitions.items()},
            "ids": {name: [row.example_id for row in members] for name, members in partitions.items()},
            "selected": {"depth": best["depth"], "leaf": best["leaf"]},
            "trials": trials, "thresholds": thresholds,
            "policy": {"min_validation_recall": .80, "min_validation_ring_recall": .90,
                       "false_review_cost": false_review_cost, "missed_fraud_cost": missed_fraud_cost},
            "scope": "Fresh generator seed; model and thresholds chosen ONLY on validation. No final-test scores computed during tuning. Same synthetic recipe/families as the prior expanded benchmark."}
    plan["lock_hash"] = digest(plan)
    write_jsonl(rows, output / "performance_merchants.jsonl")
    (output / "performance_plan.json").write_text(json.dumps(plan, indent=2, sort_keys=True) + "\n")
    print(json.dumps({"validation_trials": trials, "selected": plan["selected"], "thresholds": thresholds,
                      "partitions": plan["partitions"], "lock_hash": plan["lock_hash"]}, indent=2))


async def test_locked(output):
    target = output / "performance_report.json"
    if target.exists():
        raise ValueError("final report exists; refusing to silently overwrite the evaluated test")
    plan = json.loads((output / "performance_plan.json").read_text())
    lock_hash = plan.pop("lock_hash")
    if digest(plan) != lock_hash or code_hash() != plan["code_hash"]:
        raise ValueError("locked plan or scoring code changed; test evaluation refused")
    rows = load_jsonl(output / "performance_merchants.jsonl")
    if digest([row.to_dict() for row in rows]) != plan["dataset_hash"]:
        raise ValueError("dataset changed after tuning")
    by_id = {row.example_id: row for row in rows}
    train = [by_id[key] for key in plan["ids"]["train"]]
    test = [by_id[key] for key in plan["ids"]["test"]]
    train_features, test_features = await features(train), await features(test)
    model = MerchantPeerModel(**plan["selected"], seed=SEED).fit(train_features, {row.example_id: row.is_shell for row in train})
    predictions = await baseline_scores(test)
    calls = {}
    for name in SYSTEMS[3:]:
        predictions[name], calls[name] = model.score(test_features, system=name)
    old = GraphMLBaseline(seed=SEED)
    old.fit(train)
    predictions["legacy_graph_fixed"] = predictions["legacy_graph_tuned"] = old.score(test)
    old_hybrid = HybridRingBaseline(seed=SEED)
    await old_hybrid.fit(train)
    predictions["legacy_hybrid_tuned"] = await old_hybrid.score(test)
    thresholds = {name: selection["threshold"] for name, selection in plan["thresholds"].items()}
    thresholds["legacy_graph_fixed"] = .25
    cost = plan["policy"]["false_review_cost"]
    metrics = {name: measure(test, scores, thresholds[name], false_review_cost=cost).to_dict() for name, scores in predictions.items()}
    cohort_reports = {}
    for cohort in sorted({row.cohort.value for row in test}):
        subset = [row for row in test if row.cohort.value == cohort]
        cohort_reports[cohort] = {name: measure(subset, {row.example_id: scores[row.example_id] for row in subset}, thresholds[name], false_review_cost=cost).to_dict()
                                  for name, scores in predictions.items()}
    report = {
        "benchmark_name": "Validation-tuned risk models · fresh held-out test",
        "disclaimer": "Synthetic benchmark results; they are not production Razorpay performance.",
        "seed": SEED, "train_count": len(train), "validation_count": len(plan["ids"]["validation"]), "test_count": len(test),
        "partitions": plan["partitions"], "manifest": {"dataset_version": "synthetic-rings-v3-expanded", "feature_schema_version": VERSION, "lock_hash": lock_hash},
        "thresholds": thresholds, "baselines": {name: metrics[name] for name in SYSTEMS},
        "comparison": {name: metrics[name] for name in ("legacy_graph_fixed", "legacy_graph_tuned", "legacy_hybrid_tuned", "routed_moe")},
        "cohort_reports": cohort_reports,
        "routing_summary": {"application_count": len(test), "specialist_applications": calls["routed_moe"],
                            "specialist_rate": round(calls["routed_moe"] / len(test), 6)},
        "evaluation_scope": "New merchant-level models and alert thresholds were selected on a separate validation partition before this test was scored. The interactive review desk still uses its frozen demonstration model; this challenger is not a production deployment.",
        "metric_policy": {"name": "validation-selected-human-review-only-v2", "false_review_cost": cost, "auto_rejection": False,
            "screening": "Precision and recall count risk flags above each validation-selected threshold, not confirmed fraud. Ring recall counts at least one positive member flagged.",
            "workload": "Review includes risk flags AND missing device/IP; no uncertainty cases are counted as clean approvals. Cost is unnecessary application reviews times configured units, not rupees.",
            "comparison": "Old and new models are trained on the same new training partition and tested on the identical final batch. Threshold-tuned legacy controls isolate scoring improvements from threshold changes."},
        "limitations": [
            "Same synthetic generator families across partitions, not unseen real-world fraud patterns. No production performance claim.",
            "The 80% merchant-recall / 90% ring-recall validation floors and cost ratio are explicit development assumptions, not approved business policy.",
            "Final-test results were not used to choose the model or threshold. Further tuning requires a fresh sealed test protocol.",
            "Batch evaluation does not measure detection latency. Missing device/IP still consumes human review capacity.",
            "Some legitimate and labelled fraudulent merchants have indistinguishable available evidence; additional verified evidence is needed to resolve them."]}
    target.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    print(json.dumps({"comparison": report["comparison"], "routing_summary": report["routing_summary"], "counts": report["partitions"]}, indent=2))


async def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--stage", choices=("tune", "test"), required=True)
    parser.add_argument("--output-dir", type=Path, default=ROOT / "data/evaluation")
    parser.add_argument("--false-review-cost", type=float, default=1.)
    parser.add_argument("--missed-fraud-cost", type=float, default=1.)
    args = parser.parse_args()
    if args.stage == "tune":
        await tune(args.output_dir, false_review_cost=args.false_review_cost, missed_fraud_cost=args.missed_fraud_cost)
    else:
        await test_locked(args.output_dir)


if __name__ == "__main__":
    asyncio.run(main())
