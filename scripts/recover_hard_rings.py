#!/usr/bin/env python3
"""A separate locked experiment for difficult shared-infrastructure cases."""
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
from merchantshield.evaluation.performance import MerchantPeerModel, extract_features, measure, select_threshold
from merchantshield.evaluation.relationships import RelationshipRiskModel, extract_relationship_features, VERSION
from merchantshield.evaluation.split import connected_component_groups, group_train_test_split
from scripts.build_expanded_benchmark import partition_summary
from scripts.improve_risk_models import baseline_scores, code_hash, digest

SEED = 20260907
SYSTEMS = ("rules_only", "clustering_only", "graph_only", "tabular_only", "graph_ml", "hybrid", "routed_moe")


def study_hash():
    return digest({"prior": code_hash(), **{name: hashlib.sha256((ROOT / name).read_bytes()).hexdigest()
        for name in ("merchantshield/evaluation/relationships.py", "scripts/recover_hard_rings.py")}})


def split_population(rows):
    outer = group_train_test_split(rows, test_fraction=.2, seed=SEED)
    inner = group_train_test_split(outer.train, test_fraction=.25, seed=SEED + 1)
    partitions = {"train": inner.train, "validation": inner.test, "test": outer.test}
    groups = connected_component_groups(rows)
    owner = {}
    for name, members in partitions.items():
        for row in members:
            group = groups[row.example_id]
            if group in owner and owner[group] != name:
                raise ValueError("connected component crossed partitions")
            owner[group] = name
    return partitions, groups


async def feature_batch(rows, *, relationship=True):
    apps = {row.example_id: row.application for row in rows}
    return await (extract_relationship_features(apps) if relationship else extract_features(apps))


def choose(rows, scores):
    # Same floors/cost assumptions as v1, so improvement is not obtained by
    # relaxing safety targets. Every choice uses validation predictions only.
    return select_threshold(rows, scores, min_recall=.8, min_ring_recall=.9)


async def tune(directory):
    if (directory / "report.json").exists():
        raise ValueError("test already evaluated; do not tune this population again")
    rows = [replace(row, example_id="r2-" + row.example_id,
        component_id="r2-" + row.component_id, ring_id="r2-" + row.ring_id if row.ring_id else None)
        for row in generate_expanded_dataset(SEED, scale=3)]
    partitions, groups = split_population(rows)
    train, validation = partitions["train"], partitions["validation"]
    train_batch, val_batch = await feature_batch(train), await feature_batch(validation)
    labels = {row.example_id: row.is_shell for row in train}
    trials = []
    for depth, leaf in ((2, 15), (2, 35), (3, 15), (3, 35)):
        for group_weight in (False, True):
            config = {"depth": depth, "leaf": leaf, "group_weight": group_weight}
            model = RelationshipRiskModel(**config, seed=SEED).fit(train_batch, labels)
            scores, calls = model.score(val_batch)
            trials.append({"config": config, **choose(validation, scores), "specialist_applications": calls})
    best = min(trials, key=lambda item: (item["objective"], item["metrics"]["false_positive_rate"], digest(item["config"])))
    # A small validation-only blend search tests complementary evidence instead
    # of forcing the wider-context model to replace the stronger old decisions.
    for blend in (.25, .5, .75):
        config = {**best["config"], "blend": blend}
        model = RelationshipRiskModel(**config, seed=SEED).fit(train_batch, labels)
        scores, calls = model.score(val_batch)
        trials.append({"config": config, **choose(validation, scores), "specialist_prediction_evaluations": calls})
    best = min(trials, key=lambda item: (item["objective"], item["metrics"]["false_positive_rate"], digest(item["config"])))
    selected = RelationshipRiskModel(**best["config"], seed=SEED).fit(train_batch, labels)
    thresholds = {}
    for name in SYSTEMS:
        if name in SYSTEMS[:3]:
            scores = (await baseline_scores(validation))[name]
            thresholds[name] = {"threshold": .25, "metrics": measure(validation, scores, .25).to_dict()}
        else:
            scores, _ = selected.score(val_batch, system=name)
            thresholds[name] = choose(validation, scores)
    old_train, old_val = await feature_batch(train, relationship=False), await feature_batch(validation, relationship=False)
    old = MerchantPeerModel(depth=3, leaf=40, seed=SEED).fit(old_train, labels)
    old_scores, _ = old.score(old_val)
    thresholds["prior_v1_retuned"] = choose(validation, old_scores)
    thresholds["prior_v1_fixed"] = {"threshold": .2, "metrics": measure(validation, old_scores, .2).to_dict()}
    plan = {"version": VERSION, "seed": SEED, "scale": 3, "source_hash": study_hash(),
        "dataset_hash": digest([row.to_dict() for row in rows]),
        "selected": best["config"], "trials": trials, "thresholds": thresholds,
        "ids": {name: [row.example_id for row in members] for name, members in partitions.items()},
        "partitions": {name: partition_summary(members, groups) for name, members in partitions.items()},
        "policy": {"min_validation_recall": .8, "min_validation_ring_recall": .9,
                   "false_review_cost": 1., "missed_fraud_cost": 1.},
        "note": "Previous test is development evidence; only this fresh grouped test is final. Generator recipe/labels unchanged. No final predictions computed in tune stage."}
    plan["lock_hash"] = digest(plan)
    directory.mkdir(parents=True, exist_ok=True)
    write_jsonl(rows, directory / "merchants.jsonl")
    (directory / "plan.json").write_text(json.dumps(plan, indent=2, sort_keys=True) + "\n")
    print(json.dumps({"trials": trials, "selected": best, "prior_validation": thresholds["prior_v1_retuned"], "partitions": plan["partitions"]}, indent=2))


async def load_locked(directory, *, require_report=False):
    plan = json.loads((directory / "plan.json").read_text())
    lock = plan.pop("lock_hash")
    if digest(plan) != lock or study_hash() != plan["source_hash"] or plan["version"] != VERSION:
        raise ValueError("locked code or plan changed")
    rows = load_jsonl(directory / "merchants.jsonl")
    if digest([row.to_dict() for row in rows]) != plan["dataset_hash"]:
        raise ValueError("dataset changed after locking")
    partitions, _ = split_population(rows)
    if plan["ids"] != {name: [row.example_id for row in members] for name, members in partitions.items()}:
        raise ValueError("partition manifest does not match grouped split")
    if require_report:
        report = json.loads((directory / "report.json").read_text())
        if report["manifest"]["lock_hash"] != lock:
            raise ValueError("evaluated report does not match lock")
    plan["lock_hash"] = lock
    return plan, partitions


async def test_once(directory):
    if (directory / "report.json").exists():
        raise ValueError("refusing to overwrite the evaluated final test")
    plan, partitions = await load_locked(directory)
    train, test = partitions["train"], partitions["test"]
    model = RelationshipRiskModel(**plan["selected"], seed=SEED).fit(await feature_batch(train), {row.example_id: row.is_shell for row in train})
    batch = await feature_batch(test)
    if batch.context_capped_ids:
        raise ValueError("this benchmark metric adapter requires uncapped contexts; inference still fails safe")
    predictions = await baseline_scores(test)
    calls = {}
    for name in SYSTEMS[3:]:
        predictions[name], calls[name] = model.score(batch, system=name)
    old = MerchantPeerModel(depth=3, leaf=40, seed=SEED).fit(await feature_batch(train, relationship=False), {row.example_id: row.is_shell for row in train})
    old_scores, _ = old.score(await feature_batch(test, relationship=False))
    predictions["prior_v1_retuned"] = predictions["prior_v1_fixed"] = old_scores
    thresholds = {name: item["threshold"] for name, item in plan["thresholds"].items()}
    metrics = {name: measure(test, scores, thresholds[name]).to_dict() for name, scores in predictions.items()}
    cohorts = {}
    for cohort in sorted({row.cohort.value for row in test}):
        subset = [row for row in test if row.cohort.value == cohort]
        cohorts[cohort] = {name: measure(subset, {row.example_id: scores[row.example_id] for row in subset}, thresholds[name]).to_dict() for name, scores in predictions.items()}
    # Predefined adoption check: ring recovery must not simply buy more false alarms.
    new, old_metrics = metrics["routed_moe"], metrics["prior_v1_retuned"]
    checks = {"merchant_recall_at_least_80pct": new["recall"] >= .8,
              "ring_recall_at_least_90pct": new["ring_recall"] >= .9,
              "false_positive_rate_no_worse_than_v1": new["false_positive_rate"] <= old_metrics["false_positive_rate"],
              "precision_no_worse_than_v1": new["precision"] >= old_metrics["precision"]}
    report = {"benchmark_name": "Hard-ring recovery · relationship context experiment",
        "disclaimer": "Synthetic benchmark results; they are not production Razorpay performance.",
        "seed": SEED, "train_count": len(train), "validation_count": len(partitions["validation"]), "test_count": len(test),
        "manifest": {"dataset_version": "synthetic-rings-v3-expanded", "feature_schema_version": VERSION, "lock_hash": plan["lock_hash"]},
        "partitions": plan["partitions"], "thresholds": thresholds,
        "baselines": {name: metrics[name] for name in SYSTEMS},
        "comparison": {name: metrics[name] for name in ("prior_v1_fixed", "prior_v1_retuned", "routed_moe")},
        "cohort_reports": cohorts, "adoption_checks": checks, "adoption_passed": all(checks.values()),
        "routing_summary": {"application_count": len(test), "specialist_prediction_evaluations": calls["routed_moe"], "context_capped_applications": len(batch.context_capped_ids)},
        "evaluation_scope": "New wider-context model versus v1 on the identical fresh test. Both fit only the same training partition; thresholds chosen on separate validation. Shadow experiment only; existing cases and the live review model are unchanged.",
        "metric_policy": {"false_review_cost": 1., "auto_rejection": False,
            "screening": "Risk flags, not confirmed fraud; ring recall requires at least one positive member flagged.",
            "review": "All risk flags plus missing device/IP require review. Workload/cost remain application-level, not case-level.",
            "verification": "Shared infrastructure is not proof of fraud. Relationship checklists request independent verification; they do not assert that verification happened."},
        "limitations": ["Synthetic scenario families and prevalence are unchanged; no production performance claim.",
            "The six previously missed rings were used only as development evidence, never counted as the new untouched test.",
            "Scores cannot prove whether an accountant, settlement or shared-office relationship is authorized; that needs independent evidence and human review.",
            "Context expansion is capped at 50 merchants; over-limit context fails safe to review during inference.",
            "No further selection is permitted using this final test. A failed adoption check keeps the challenger in shadow mode."]}
    (directory / "report.json").write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    print(json.dumps({"comparison": report["comparison"], "checks": checks, "cohorts": {name: values["routed_moe"] for name, values in cohorts.items()}}, indent=2))


async def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--stage", choices=("tune", "test"), required=True)
    parser.add_argument("--output-dir", type=Path, default=ROOT / "data/evaluation/relationships")
    args = parser.parse_args()
    await (tune(args.output_dir) if args.stage == "tune" else test_once(args.output_dir))


if __name__ == "__main__":
    asyncio.run(main())
