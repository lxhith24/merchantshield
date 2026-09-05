#!/usr/bin/env python3
"""Phase 6 acceptance: every claim MerchantShield makes, checked in one run.

Runs against a throwaway database with no API key, so the result is the same on
a clean checkout as it is here. Exits non-zero the moment a check fails, which
is the point: the buildathon recording should only be made after this passes.

    .venv/bin/python scripts/acceptance.py
"""
from __future__ import annotations

import argparse
import asyncio
import json
import os
import sys
import tempfile
from pathlib import Path
from typing import Any, Callable, List, Tuple

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

REPORT_PATH = ROOT / "data/evaluation/latest_report.json"
DATASET_PATH = ROOT / "data/evaluation/synthetic_merchants.jsonl"

GREEN, RED, DIM, BOLD, RESET = "\033[32m", "\033[31m", "\033[2m", "\033[1m", "\033[0m"


class Failure(AssertionError):
    """One acceptance check did not hold."""


class Checklist:
    """Runs checks in order, records outcomes, never stops at the first failure."""

    def __init__(self) -> None:
        self.results: List[Tuple[str, bool, str]] = []

    async def check(self, name: str, fn: Callable[[], Any]) -> Any:
        try:
            value = fn()
            if asyncio.iscoroutine(value):
                value = await value
        except Exception as exc:
            self.results.append((name, False, f"{type(exc).__name__}: {exc}"))
            print(f"  {RED}FAIL{RESET}  {name}\n        {exc}")
            return None
        detail = value if isinstance(value, str) else ""
        self.results.append((name, True, detail))
        print(f"  {GREEN}PASS{RESET}  {name}" + (f"  {DIM}{detail}{RESET}" if detail else ""))
        return value

    @property
    def failed(self) -> int:
        return sum(1 for _, ok, _ in self.results if not ok)


def rule(title: str) -> None:
    print(f"\n{BOLD}{title}{RESET}\n{'-' * len(title)}")


def require(condition: bool, message: str) -> None:
    if not condition:
        raise Failure(message)


# ------------------------------------------------------------------- checks


def check_frozen_dataset() -> str:
    from merchantshield.evaluation.dataset import generate_synthetic_dataset, load_jsonl

    require(DATASET_PATH.exists(), f"missing dataset at {DATASET_PATH}")
    stored = [item.to_dict() for item in load_jsonl(DATASET_PATH)]
    regenerated = [item.to_dict() for item in generate_synthetic_dataset(seed=20250904)]
    require(
        stored == regenerated,
        "the checked-in dataset does not match a regeneration from seed 20250904",
    )
    return f"{len(stored)} applications reproduce exactly from the seed"


def check_split_is_leakage_safe() -> str:
    from merchantshield.evaluation.dataset import load_jsonl
    from merchantshield.evaluation.split import group_train_test_split

    examples = load_jsonl(DATASET_PATH)
    split = group_train_test_split(examples, seed=20250904)

    def groups(rows):
        return {item.ring_id or item.component_id for item in rows}

    overlap = groups(split.train) & groups(split.test)
    require(not overlap, f"rings/components cross the split: {sorted(overlap)}")
    return f"{len(split.train)} train / {len(split.test)} test, no shared group"


def check_evaluation_report() -> str:
    require(REPORT_PATH.exists(), f"missing report at {REPORT_PATH}")
    report = json.loads(REPORT_PATH.read_text(encoding="utf-8"))
    for name in ("rules_only", "clustering_only", "graph_only", "tabular_only",
                 "graph_ml", "hybrid", "routed_moe"):
        require(name in report["baselines"], f"report is missing baseline '{name}'")
    require("not production" in report["disclaimer"], "report lost its disclaimer")
    routed = report["baselines"]["routed_moe"]
    require(routed["ring_recall"] == 1.0, "routed MoE no longer recalls every ring")
    return (
        f"routed MoE: precision {routed['precision']:.3f}, recall "
        f"{routed['recall']:.3f}, FPR {routed['false_positive_rate']:.3f}"
    )


async def check_scenarios(runtime) -> str:
    from merchantshield.demo import SCENARIOS, ScenarioKind, resolve_scenario

    checked = 0
    for scenario in SCENARIOS:
        if scenario.kind is not ScenarioKind.QUEUE:
            continue
        resolved = resolve_scenario(runtime.world, scenario.key)
        run = await runtime.workflow.run(resolved.candidate_id)
        require(
            run.decision is not None
            and run.decision.action is scenario.expected_action,
            f"scenario '{scenario.key}' no longer produces {scenario.expected_action}",
        )
        experts = tuple(
            item.expert_name for item in run.state.assessment.expert_results
        )
        require(
            experts == scenario.expected_experts,
            f"scenario '{scenario.key}' routed {experts}, expected "
            f"{scenario.expected_experts}",
        )
        checked += 1
    return f"{checked} frozen queue scenarios still behave as documented"


async def check_citation_failure(runtime) -> str:
    """Phase 6 requires one demonstrated hallucination and its recovery."""
    from merchantshield.demo import resolve_scenario, run_grounding_failure
    from merchantshield.demo.harness import FABRICATED_EVIDENCE_ID

    resolved = resolve_scenario(runtime.world, "fabricated_citation")
    result = await run_grounding_failure(runtime, resolved.candidate_id)
    run = result.runs[0]
    investigation = run.state.investigation

    require(investigation is not None, "no investigation was recorded")
    require(
        FABRICATED_EVIDENCE_ID in investigation.grounding.unknown_evidence_ids,
        "the fabricated evidence ID was not caught",
    )
    require(
        investigation.output is None,
        "an ungrounded narrative survived into structured output",
    )
    require(
        FABRICATED_EVIDENCE_ID in (investigation.rejected_narrative or ""),
        "the rejected narrative was not preserved for audit",
    )
    require(
        run.decision is not None and run.decision.action.value == "human_review",
        "a fabricated citation did not force human review",
    )
    require(
        "INSUFFICIENT_GROUNDING" in run.decision.reason_codes,
        "the grounding failure is not visible in the reason codes",
    )
    return "confident 'continue_onboarding' on a fake ID became human review"


async def check_information_request(runtime) -> str:
    from merchantshield.demo import resolve_scenario, run_information_request

    resolved = resolve_scenario(runtime.world, "information_request_and_resume")
    result = await run_information_request(runtime, resolved.candidate_id)
    require(len(result.runs) == 2, "the case never paused for information")
    paused, resumed = result.runs
    require(paused.is_awaiting_information, "the workflow did not interrupt")
    require(paused.decision is None, "a paused case must not carry a decision")
    require(resumed.case_id == paused.case_id, "the resumed case has a new identity")
    require(resumed.decision is not None, "the resumed case never decided")
    return f"paused and resumed the identical case {resumed.case_id}"


async def check_case_lifecycle(runtime) -> str:
    """Open, claim, resolve and hand off one case through the real service."""
    from merchantshield.demo import resolve_scenario
    from merchantshield.review.contracts import (
        HumanDecision,
        ResolutionRequest,
        ReviewReasonCode,
    )
    from merchantshield.investigation.memory import PriorOutcome

    resolved = resolve_scenario(runtime.world, "shared_kiosk_false_positive")
    case = await runtime.service.open_case(resolved.candidate_id)
    require(case.status.value == "pending_review", "the case did not reach a human")

    claimed = runtime.service.claim(case.case_id, "acceptance_reviewer", case.version)
    final = runtime.service.resolve(
        claimed.case_id,
        request=ResolutionRequest(
            reviewer_id="acceptance_reviewer",
            decision=HumanDecision.APPROVE_ONBOARDING,
            outcome=PriorOutcome.LEGITIMATE_COWORKING,
            reason_codes=(ReviewReasonCode.VERIFIED_COWORKING_TENANCY,),
            notes="Coworking tenancy verified against the building lease.",
        ),
        expected_version=claimed.version,
    )
    require(final.status.value == "resolved", "the case did not resolve")
    require(
        final.automated.action.value == "human_review",
        "the human decision overwrote the automated one",
    )

    handed = runtime.service.hand_off(
        final.case_id, expected_version=final.version, actor="acceptance_reviewer"
    )
    require(handed.handoff is not None, "no hand-off receipt was recorded")
    require(handed.handoff.mode == "SIMULATED", "the gateway was not the simulated one")

    sequences = [event.sequence for event in handed.events]
    require(sequences == sorted(sequences), "the audit log is not ordered")
    require(len(set(sequences)) == len(sequences), "the audit log has duplicates")
    return f"{len(sequences)} append-only events, hand-off {handed.handoff.reference}"


async def _open_scenario_case(runtime, key: str):
    """Each check owns its own case, so check order can never matter."""
    from merchantshield.demo import resolve_scenario

    resolved = resolve_scenario(runtime.world, key)
    return await runtime.service.open_case(resolved.candidate_id)


async def check_concurrency(runtime) -> str:
    from merchantshield.review.store import ConcurrentModification

    case = await _open_scenario_case(runtime, "obvious_ring")
    stale = case.version

    runtime.service.claim(case.case_id, "reviewer_one", stale)
    try:
        runtime.service.claim(case.case_id, "reviewer_two", stale)
    except ConcurrentModification as exc:
        return f"a stale version {exc.expected_version} lost to {exc.actual_version}"
    raise Failure("a stale write was accepted")


async def check_no_gateway_without_approval(runtime) -> str:
    from merchantshield.review.service import HandoffNotPermitted

    case = await _open_scenario_case(runtime, "evasive_ring")
    require(case.human is None, "the fixture case is already resolved")
    try:
        runtime.service.hand_off(
            case.case_id, expected_version=case.version, actor="acceptance_reviewer"
        )
    except HandoffNotPermitted:
        return "a case without a recorded human approval cannot reach a gateway"
    raise Failure("an unapproved case reached the gateway")


def check_razorpay_adapter_refuses() -> str:
    from merchantshield.gateway import GatewayUnavailable, RazorpayTestGateway
    from merchantshield.gateway.base import HandoffRequest

    gateway = RazorpayTestGateway()
    require(not gateway.is_available, "the Razorpay adapter claims to be configured")
    try:
        gateway.submit(
            HandoffRequest(
                case_id="case-acceptance",
                candidate_id="candidate-acceptance",
                member_ids=("syn-0000",),
                approved_by="acceptance_reviewer",
                reason_codes=("verified_coworking_tenancy",),
            )
        )
    except GatewayUnavailable:
        return "refuses without explicit test credentials; never falls back silently"
    raise Failure("the Razorpay adapter did not refuse")


def check_labels(runtime) -> str:
    labels = runtime.labels
    require(labels["data"] == "SIMULATED", "the data label is not SIMULATED")
    require(
        labels["gateway"] in ("SIMULATED", "RAZORPAY_TEST"),
        f"unexpected gateway label {labels['gateway']}",
    )
    require(
        labels["investigator"] == "LLM_DISABLED",
        "acceptance must run with no API key, in LLM_DISABLED mode",
    )
    return ", ".join(f"{key}={value}" for key, value in labels.items())


def check_no_secrets_in_source() -> str:
    """Nothing that looks like a live key may sit in tracked source."""
    import re

    # Prefixes alone are not credentials -- `tests/test_gateway.py` proves the
    # adapter refuses a live-mode key, and `.env.example` documents its shape.
    # Only a prefix followed by key-length material is treated as a secret.
    pattern = re.compile(
        r"rzp_live_[A-Za-z0-9]{10,}|sk-ant-[A-Za-z0-9_-]{20,}|AKIA[0-9A-Z]{16}"
    )
    skip = {".venv", "__pycache__", ".git", ".pytest_cache", "data", "node_modules"}
    scanned = 0
    for path in ROOT.rglob("*"):
        if not path.is_file() or path.suffix not in {".py", ".sh", ".md", ".txt", ".html"}:
            continue
        if any(part in skip for part in path.parts):
            continue
        scanned += 1
        text = path.read_text(encoding="utf-8", errors="ignore")
        match = pattern.search(text)
        if match:
            relative = path.relative_to(ROOT)
            raise Failure(f"possible credential in {relative}: {match.group()[:12]}…")
    return f"{scanned} source files scanned, no key-shaped strings"


# --------------------------------------------------------------------- main


async def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--keep-db",
        action="store_true",
        help="run against the project database instead of a throwaway one",
    )
    args = parser.parse_args()

    if not args.keep_db:
        tmp = tempfile.mkdtemp(prefix="merchantshield_acceptance_")
        os.environ["DATABASE_URL"] = f"sqlite:///{tmp}/acceptance.db"
    os.environ["ANTHROPIC_API_KEY"] = ""  # acceptance never calls a model

    from merchantshield.db.database import init_db
    from merchantshield.runtime import build_runtime

    init_db()

    print(f"{BOLD}MerchantShield — Phase 6 acceptance{RESET}")
    print(f"{DIM}All data is synthetic. No API key is used. Database: "
          f"{os.environ.get('DATABASE_URL', 'project default')}{RESET}")

    checklist = Checklist()

    rule("1. Frozen data and evaluation")
    await checklist.check("dataset is reproducible from its seed", check_frozen_dataset)
    await checklist.check("train/test split is leakage-safe", check_split_is_leakage_safe)
    await checklist.check("held-out report is present and complete", check_evaluation_report)

    runtime = await build_runtime()

    rule("2. Labels and safety boundaries")
    await checklist.check("every surface label is set", lambda: check_labels(runtime))
    await checklist.check("Razorpay adapter refuses", check_razorpay_adapter_refuses)
    await checklist.check("no key-shaped strings in source", check_no_secrets_in_source)

    rule("3. Frozen demonstration scenarios")
    await checklist.check("queue scenarios behave as documented", lambda: check_scenarios(runtime))
    await checklist.check(
        "a fabricated citation is caught and recovers",
        lambda: check_citation_failure(runtime),
    )
    await checklist.check(
        "a paused case resumes under the same ID",
        lambda: check_information_request(runtime),
    )

    rule("4. Review, persistence and hand-off")
    await checklist.check("full case lifecycle", lambda: check_case_lifecycle(runtime))
    await checklist.check("optimistic concurrency holds", lambda: check_concurrency(runtime))
    await checklist.check(
        "no gateway without a human approval",
        lambda: check_no_gateway_without_approval(runtime),
    )

    rule("Result")
    passed = len(checklist.results) - checklist.failed
    if checklist.failed:
        print(f"  {RED}{checklist.failed} of {len(checklist.results)} checks failed."
              f"{RESET} Do not record the demonstration yet.")
        return 1
    print(f"  {GREEN}{passed}/{passed} checks passed.{RESET}")
    print(f"  {DIM}Synthetic benchmark results. Not production Razorpay "
          f"performance.{RESET}")
    return 0


if __name__ == "__main__":
    raise SystemExit(asyncio.run(main()))
