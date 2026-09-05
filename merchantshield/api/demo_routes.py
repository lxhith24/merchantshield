"""Demonstration and evaluation surface.

Everything here is read-only with respect to the case store. The harness
endpoint runs a throwaway workflow and returns its trace; it cannot open,
mutate or resolve a case, and every response carries the `SIMULATED` /
gateway / investigator labels so no screenshot can be mistaken for production.
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, Literal

from fastapi import APIRouter, HTTPException

from ..demo import HARNESSES, ScenarioNotFound, resolve_scenario, resolve_scenarios
from ..runtime import get_runtime

router = APIRouter(prefix="/api/v1", tags=["demonstration"])

REPORT_PATH = (
    Path(__file__).resolve().parent.parent.parent / "data/evaluation/latest_report.json"
)
EXPANDED_REPORT_PATH = REPORT_PATH.with_name("expanded_report.json")
PERFORMANCE_REPORT_PATH = REPORT_PATH.with_name("performance_report.json")
RELATIONSHIP_REPORT_PATH = REPORT_PATH.parent / "relationships/report.json"

DISCLAIMER = (
    "All identities, documents, links and metrics in MerchantShield are "
    "synthetic. They are not production Razorpay performance."
)


@router.get("/meta/status", summary="Labels and versions every surface must display")
async def meta_status() -> Dict[str, Any]:
    runtime = await get_runtime()
    from ..workflow.state import WORKFLOW_VERSION

    return {
        "labels": runtime.labels,
        "disclaimer": DISCLAIMER,
        "workflow_version": WORKFLOW_VERSION,
        "candidate_count": len(runtime.world.candidates),
        "application_count": len(runtime.world.applications),
        "evaluation_report_available": EXPANDED_REPORT_PATH.exists(),
    }


@router.get("/demo/scenarios", summary="Frozen demonstration scenarios")
async def list_scenarios() -> Dict[str, Any]:
    runtime = await get_runtime()
    return {
        "scenarios": [item.to_dict() for item in resolve_scenarios(runtime.world)],
        "labels": runtime.labels,
        "disclaimer": DISCLAIMER,
    }


@router.get("/demo/scenarios/{key}", summary="One frozen demonstration scenario")
async def get_scenario(key: str) -> Dict[str, Any]:
    runtime = await get_runtime()
    try:
        resolved = resolve_scenario(runtime.world, key)
    except ScenarioNotFound as exc:
        raise HTTPException(status_code=404, detail=f"Unknown scenario: {key}") from exc
    payload = resolved.to_dict()
    payload["labels"] = runtime.labels
    return payload


@router.post("/demo/harness/{key}", summary="Run a deterministic harness scenario")
async def run_harness(key: str) -> Dict[str, Any]:
    """Exercise a failure path the frozen data does not reach on its own.

    Nothing is persisted. The response states exactly what was pinned or
    scripted, so the demonstration never overstates what the system did.
    """
    runtime = await get_runtime()
    try:
        resolved = resolve_scenario(runtime.world, key)
    except ScenarioNotFound as exc:
        raise HTTPException(status_code=404, detail=f"Unknown scenario: {key}") from exc
    harness = HARNESSES.get(resolved.scenario.harness or "")
    if harness is None:
        raise HTTPException(
            status_code=400,
            detail=(
                f"Scenario '{key}' is a queue scenario. Open it as a case via "
                "POST /api/v1/cases instead."
            ),
        )
    result = await harness(runtime, resolved.candidate_id)
    payload = result.to_dict()
    payload["scenario"] = resolved.to_dict()
    payload["labels"] = runtime.labels
    payload["disclaimer"] = DISCLAIMER
    return payload


@router.get("/evaluation/report", summary="Frozen held-out evaluation report")
async def evaluation_report(dataset: Literal["relationships", "performance", "expanded", "demo"] = "expanded") -> Dict[str, Any]:
    runtime = await get_runtime()
    report_path = {"relationships": RELATIONSHIP_REPORT_PATH, "performance": PERFORMANCE_REPORT_PATH, "expanded": EXPANDED_REPORT_PATH, "demo": REPORT_PATH}[dataset]
    if not report_path.exists():
        raise HTTPException(
            status_code=404,
            detail=(
                "No evaluation report. Generate one with "
                + {"relationships": "scripts/recover_hard_rings.py", "performance": "scripts/improve_risk_models.py", "expanded": "scripts/build_expanded_benchmark.py", "demo": "scripts/generate_evaluation_dataset.py"}[dataset]
            ),
        )
    report = json.loads(report_path.read_text(encoding="utf-8"))
    # The manifest's ID lists are long and are not what a reviewer reads.
    manifest = {
        key: value
        for key, value in report.get("manifest", {}).items()
        if key not in ("train_ids", "test_ids")
    }
    validation_targets = {}
    if dataset in ("performance", "relationships"):
        plan = json.loads(report_path.with_name("plan.json" if dataset == "relationships" else "performance_plan.json").read_text(encoding="utf-8"))
        validation_targets = {"merchant_recall": plan["policy"]["min_validation_recall"],
                              "ring_recall": plan["policy"]["min_validation_ring_recall"]}
    return {
        "baselines": report.get("baselines", {}),
        "routing_summary": report.get("routing_summary", {}),
        "manifest": manifest,
        "train_count": report.get("train_count"),
        "test_count": report.get("test_count"),
        "validation_count": report.get("validation_count"),
        "seed": report.get("seed"),
        "disclaimer": report.get("disclaimer", DISCLAIMER),
        "labels": runtime.labels,
        "benchmark_name": report.get("benchmark_name", "Frozen demonstration benchmark"),
        "partitions": report.get("partitions", {}),
        "dataset_summary": report.get("dataset_summary", {}),
        "metric_policy": report.get("metric_policy", {}),
        "evaluation_scope": report.get("evaluation_scope", "Frozen 104-row demonstration benchmark."),
        "stability": report.get("stability", {}),
        "split_seeds": report.get("split_seeds", []),
        "cohort_reports": report.get("cohort_reports", {}),
        "limitations": report.get("limitations", []),
        "comparison": report.get("comparison", {}),
        "thresholds": report.get("thresholds", {}),
        "validation_targets": validation_targets,
        "adoption_checks": report.get("adoption_checks", {}),
        "adoption_passed": report.get("adoption_passed"),
    }


@router.get("/evaluation/challenger/{candidate_id}", summary="Read-only shadow scoring; cannot change a case or onboarding")
async def challenger_score(candidate_id: str, profile: Literal["performance", "relationships"] = "performance") -> Dict[str, Any]:
    from ..evaluation.validated import load_validated_model, load_relationship_model
    from ..evaluation.relationships import relationship_evidence_requests

    runtime = await get_runtime()
    if candidate_id not in runtime.world.by_id:
        raise HTTPException(status_code=404, detail="Unknown candidate")
    try:
        model, threshold = await (load_relationship_model(RELATIONSHIP_REPORT_PATH.parent)
                                  if profile == "relationships" else load_validated_model(PERFORMANCE_REPORT_PATH.parent))
        result = await model.screen_applications(runtime.world.applications, threshold=threshold)
    except (FileNotFoundError, ValueError, KeyError) as exc:
        raise HTTPException(status_code=503, detail="Validated challenger unavailable; no approval or case update performed.") from exc
    candidate = runtime.world.by_id[candidate_id]
    return {"candidate_id": candidate_id, "mode": "SHADOW_ONLY", "disclaimer": DISCLAIMER,
            "model_version": result["model_version"],
            "applications": {key: result["applications"][key] for key in candidate.member_ids},
            "verification_requests": relationship_evidence_requests({key: runtime.world.applications[key] for key in candidate.member_ids}),
            "case_updated": False, "gateway_called": False}
