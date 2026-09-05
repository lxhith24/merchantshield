"""Calm, decision-first reviewer surface for MerchantShield.

The default experience answers one question: do these merchants appear to be
controlled together? Technical traces remain available, but never compete with
the evidence or the human decision.

    .venv/bin/streamlit run merchantshield/ui/streamlit_app.py
"""
from __future__ import annotations

import html
import os
import sys
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple

import streamlit as st

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from merchantshield.ui.client import ApiError, MerchantShieldClient, VersionConflict  # noqa: E402
from merchantshield.ui.presentation import (  # noqa: E402
    reviewable_cases,
    grouped_signals,
    priority_label,
    ring_reason_codes,
    short_merchant_id,
    strongest_signal,
    submission_window,
)
from merchantshield.ui.ring_graph import render_ring_svg, svg_data_uri  # noqa: E402


API_BASE_URL = os.environ.get("MERCHANTSHIELD_API_URL", "http://127.0.0.1:8000")

CLEARING_OPTIONS: Mapping[str, Tuple[str, str]] = {
    "Verified coworking tenancy": (
        "legitimate_coworking",
        "verified_coworking_tenancy",
    ),
    "Verified franchise relationship": (
        "legitimate_franchise",
        "verified_franchise_agreement",
    ),
    "Verified accountant relationship": (
        "legitimate_accountant",
        "verified_accountant_relationship",
    ),
    "Verified family business": (
        "legitimate_family",
        "verified_family_business",
    ),
    "Links explained by supplied evidence": (
        "inconclusive",
        "links_explained_by_supplied_evidence",
    ),
}


@st.cache_resource(show_spinner=False)
def get_client(base_url: str) -> MerchantShieldClient:
    return MerchantShieldClient(base_url)


def inject_style() -> None:
    stylesheet = Path(__file__).with_name("theme.css").read_text(encoding="utf-8")
    st.html(f"<style>{stylesheet}</style>")


def api_call(function, *args, quiet: bool = False, **kwargs):
    try:
        return function(*args, **kwargs)
    except VersionConflict as exc:
        st.warning(
            f"This case changed in another session (version {exc.current_version}). "
            "It has been refreshed; review it before deciding."
        )
        return None
    except ApiError as exc:
        if not quiet:
            st.error(str(exc.detail))
        return None
    except Exception as exc:
        if not quiet:
            st.error(f"MerchantShield is unavailable: {exc}")
        return None


def render_mode(labels: Mapping[str, str]) -> None:
    investigator = {
        "LLM_DISABLED": "Offline investigator",
        "LLM_ENABLED": "AI investigator",
        "SCRIPTED": "Scripted investigator",
    }.get(labels.get("investigator"), "Investigator unavailable")
    gateway = "Simulated gateway" if labels.get("gateway") == "SIMULATED" else str(labels.get("gateway", "Gateway unknown"))
    st.html(
        '<div class="mode-strip"><span class="status-dot"></span>Synthetic demonstration'
        f'<span class="mode-secondary"> · {html.escape(investigator)} · {html.escape(gateway)}</span></div>'
    )


def humanize(value: object) -> str:
    return str(value or "").replace("_", " ").strip().capitalize()


def review_home(client: MerchantShieldClient) -> None:
    st.html('<div class="hero-kicker">THE REVIEW DESK</div>')
    st.title("Review with clarity.")
    st.html('<div class="hero-copy">A clear view of the merchants, the evidence, and what needs your judgment.</div>')

    payload = api_call(client.list_cases, limit=200)
    if payload is None:
        st.info("The review queue is unavailable. Refresh once the API is running.")
        return
    cases = reviewable_cases(payload.get("cases", []), st.session_state["reviewer_id"].strip())
    high = sum(priority_label(row) == "High priority" for row in cases)
    waiting = sum(row.get("status") == "awaiting_information" for row in cases)
    st.html(
        '<div class="queue-counts">'
        f'<span><b>{len(cases)}</b> available reviews</span>'
        f'<span><b>{high}</b> high priority</span>'
        f'<span><b>{waiting}</b> awaiting evidence</span></div>'
    )

    if cases:
        next_case = cases[0]
        with st.container(key="featured", border=True):
            details, visual = st.columns([1, 1.12], gap="large", vertical_alignment="center")
            case = api_call(client.case, next_case["case_id"])
            with details:
                st.html('<div class="hero-kicker">NEXT FOR YOUR REVIEW</div>')
                st.subheader(f"{next_case.get('member_count', 0)} linked merchants")
                if case:
                    signals = grouped_signals(case)
                    if signals:
                        st.html('<div class="featured-copy">' + html.escape(signals[0]["sentence"]) + ".</div>")
                    st.caption("Shared infrastructure can have a legitimate explanation. Review the evidence before deciding.")
                st.html(f'<div class="case-reference">{html.escape(next_case["case_id"])}</div>')
                if st.button("Review next case →", type="primary", key="review_next"):
                    open_for_reviewer(client, next_case)
            with visual:
                if case and case.get("candidate"):
                    st.image(svg_data_uri(render_ring_svg(case["candidate"], members=case.get("members", []), width=700, height=400)), width="stretch")
                else:
                    st.caption("The connection map will appear when case evidence is available.")

        if len(cases) > 1:
            with st.expander(f"More in your queue · {len(cases) - 1}"):
                for row in cases[1:]:
                    label, status, action = st.columns([3, 2, 1], vertical_alignment="center")
                    with label:
                        st.markdown(f"**{row.get('member_count', 0)} linked merchants**")
                        st.caption(row["case_id"])
                    with status:
                        st.caption(humanize(row.get("status")))
                    with action:
                        if st.button("Review →", key=f"open-{row['case_id']}"):
                            open_for_reviewer(client, row)
    else:
        with st.container(border=True):
            st.subheader("You're all caught up.")
            st.write("No cases are available for this reviewer. Open a synthetic demonstration below.")

    with st.expander("Try a demonstration case"):
        st.caption("Explore a frozen synthetic scenario through the same investigation flow.")
        scenarios = api_call(client.scenarios) or []
        choices = [item for item in scenarios if item.get("kind") == "queue"]
        if choices:
            default = next((i for i, item in enumerate(choices) if item.get("key") == "evasive_ring"), 0)
            selected = st.selectbox("Scenario", choices, index=default, format_func=lambda item: item["title"])
            st.caption(selected["headline"])
            if selected.get("honest_note"):
                st.caption(selected["honest_note"])
            if st.button("Open demonstration case", type="primary"):
                case = api_call(client.open_case, selected["candidate_id"])
                if case:
                    open_for_reviewer(client, case)


def open_for_reviewer(client: MerchantShieldClient, case: Mapping[str, Any]) -> None:
    reviewer = st.session_state["reviewer_id"].strip()
    if len(reviewer) < 2:
        st.warning("Enter a reviewer name of at least two characters in Session to start a review.")
        return
    current: Optional[Mapping[str, Any]] = case
    if case.get("status") == "pending_review" and reviewer:
        current = api_call(
            client.claim,
            case["case_id"],
            reviewer,
            int(case["version"]),
        )
    if current:
        st.session_state["case_id"] = current["case_id"]
        st.session_state.pop("pending_action", None)
        st.rerun()


def review_case(client: MerchantShieldClient, case_id: str) -> None:
    case = api_call(client.case, case_id)
    if case is None:
        return
    reviewer = st.session_state["reviewer_id"].strip()
    signals = grouped_signals(case)
    members = list(case.get("members") or [])
    count = len(members)
    automated = case.get("automated") or {}
    priority = priority_label(case)

    with st.container(key="case_heading"):
        back, badge = st.columns([1, 4], vertical_alignment="center")
        with back:
            if st.button("← Review queue", type="tertiary"):
                st.session_state.pop("case_id", None)
                st.session_state.pop("pending_action", None)
                st.rerun()
        with badge:
            chip_class = "clear" if priority == "Cleared" else ""
            st.html(f'<span class="case-chip {chip_class}">{html.escape(priority)}</span>')
        st.title(
            f"{count} linked merchant{'s' if count != 1 else ''}."
            if automated.get("action") == "human_review"
            else "No coordinated-control concern found."
        )
    st.html(
        '<div class="hero-copy">Shared signals, one investigation. '
        'Confirm the relationship before making a decision.</div>'
    )

    top_signal = signals[0] if signals else None
    coverage = top_signal["coverage"] if top_signal else "—"
    st.html(
        '<div class="fact-grid">'
        + _fact("Case", case["case_id"].replace("case-", "").upper())
        + _fact("Strongest similarity", strongest_signal(case))
        + _fact("Similarity coverage", coverage)
        + _fact("Submission window", submission_window(members))
        + "</div>"
    )

    evidence_region = st.container(key="case_evidence")
    graph_column, evidence_column = evidence_region.columns([1.65, 1], gap="large")
    with graph_column:
        candidate = case.get("candidate") or {}
        if candidate:
            graph = render_ring_svg(candidate, members=members, width=700, height=430)
            st.image(svg_data_uri(graph), width="stretch")
        else:
            st.info("No graph is available for this case.")
    with evidence_column:
        st.subheader("The shared signals")
        if signals:
            for signal in signals[:4]:
                cited = " · cited" if signal["cited"] else ""
                st.html(
                    '<div class="signal">'
                    f'<div class="signal-title">{html.escape(signal["label"])}</div>'
                    f'<div class="signal-detail">Shared by {html.escape(signal["coverage"])} merchants'
                    f'{html.escape(cited)}</div></div>'
                )
        else:
            st.caption("No shared identifier has been corroborated.")
        output = _investigator_output(case)
        alternatives = (output or {}).get("legitimate_alternatives") or []
        if alternatives:
            st.html('<div class="alternative"><div class="hero-kicker">ALSO CONSIDER</div>'
                    + html.escape(humanize(alternatives[0].get("label"))) + '</div>')

    render_investigator_summary(case)
    render_decision(client, case, reviewer)
    render_members(members)
    render_technical_details(client, case)


def render_investigator_summary(case: Mapping[str, Any]) -> None:
    investigation = (case.get("case_state") or {}).get("investigation") or {}
    output = investigation.get("output")
    st.divider()
    st.subheader("The assessment")
    if not investigation:
        st.write("The investigator was not needed for this case.")
        return
    if not output:
        st.write("The investigator abstained or failed safely. A human remains responsible.")
        return
    st.write(output.get("narrative", ""))
    grounding = investigation.get("grounding") or {}
    if grounding.get("status") == "grounded":
        st.caption(
            f"Citations verified: {len(grounding.get('validated_evidence_ids', ())) or 0} "
            "evidence references existed and were disclosed to the investigator."
        )
    missing = output.get("missing_evidence") or []
    if missing:
        st.info("Still worth verifying: " + "; ".join(str(item) for item in missing))


def render_decision(
    client: MerchantShieldClient, case: Mapping[str, Any], reviewer: str
) -> None:
    st.divider()
    st.subheader("Your decision")
    status = case.get("status")
    human = case.get("human")
    if human:
        st.success(
            f"Recorded by {human['reviewer_id']}: {humanize(human['decision'])}. "
            f"Outcome: {humanize(human['outcome'])}."
        )
        if human.get("decision") == "approve_onboarding" and not case.get("handoff"):
            if st.button("Send to simulated onboarding", type="primary"):
                if api_call(
                    client.hand_off,
                    case["case_id"],
                    actor=reviewer,
                    version=case["version"],
                ):
                    st.rerun()
        elif case.get("handoff"):
            st.caption(f"Simulated handoff reference: {case['handoff']['reference']}")
        return
    if status == "cleared":
        st.success("Cleared by deterministic policy. No human decision was fabricated.")
        return
    if status == "awaiting_information":
        render_information_request(client, case, reviewer)
        return
    if status == "pending_review":
        if st.button("Start review", type="primary"):
            open_for_reviewer(client, case)
        return
    if case.get("claimed_by") != reviewer:
        st.warning(f"This case is currently held by {case.get('claimed_by')}.")
        return

    action_columns = st.columns(3)
    if action_columns[0].button(
        "Confirm suspicious ring", type="primary", use_container_width=True
    ):
        st.session_state["pending_action"] = "ring"
        st.rerun()
    if action_columns[1].button("Escalate for evidence", use_container_width=True):
        st.session_state["pending_action"] = "evidence"
        st.rerun()
    if action_columns[2].button("Clear connection", use_container_width=True):
        st.session_state["pending_action"] = "clear"
        st.rerun()

    pending = st.session_state.get("pending_action")
    if pending == "ring":
        st.warning(
            "This keeps the connected applications on hold and records a confirmed "
            "ring for enhanced review. It does not permanently blacklist them."
        )
        confirm, cancel = st.columns([1, 1])
        if confirm.button("Confirm and finish", type="primary", use_container_width=True):
            resolve_case(
                client,
                case,
                reviewer,
                decision="keep_on_hold",
                outcome="confirmed_ring",
                reasons=ring_reason_codes(case),
                notes="Linked onboarding signals confirmed for enhanced review.",
            )
        if cancel.button("Cancel", use_container_width=True):
            st.session_state.pop("pending_action", None)
            st.rerun()
    elif pending == "evidence":
        st.info(
            "This closes the current review as inconclusive and escalates it for "
            "additional verification."
        )
        if st.button("Confirm escalation", type="primary"):
            resolve_case(
                client,
                case,
                reviewer,
                decision="escalate",
                outcome="inconclusive",
                reasons=["insufficient_evidence"],
                notes="Additional relationship evidence is required.",
            )
    elif pending == "clear":
        explanation = st.selectbox("Verified explanation", list(CLEARING_OPTIONS))
        outcome, reason = CLEARING_OPTIONS[explanation]
        st.caption("Clearing requires a verified explanation; shared infrastructure alone is not enough.")
        if st.button("Confirm legitimate connection", type="primary"):
            resolve_case(
                client,
                case,
                reviewer,
                decision="approve_onboarding",
                outcome=outcome,
                reasons=[reason],
                notes=explanation,
            )


def resolve_case(
    client: MerchantShieldClient,
    case: Mapping[str, Any],
    reviewer: str,
    *,
    decision: str,
    outcome: str,
    reasons: Sequence[str],
    notes: str,
) -> None:
    result = api_call(
        client.resolve,
        case["case_id"],
        reviewer_id=reviewer,
        version=case["version"],
        decision=decision,
        outcome=outcome,
        reason_codes=reasons,
        notes=notes,
    )
    if result:
        st.session_state.pop("pending_action", None)
        st.rerun()


def render_information_request(
    client: MerchantShieldClient, case: Mapping[str, Any], reviewer: str
) -> None:
    requested = case.get("requested_information") or []
    st.info("This case is paused for one decisive fact.")
    with st.form(f"information-{case['case_id']}"):
        answers = []
        for index, item in enumerate(requested):
            answers.append(
                (
                    item["item_code"],
                    st.text_area(item["description"], key=f"answer-{index}"),
                )
            )
        if st.form_submit_button("Supply evidence and resume", type="primary"):
            items = [
                {"item_code": code, "content": value.strip(), "supplied_by": reviewer}
                for code, value in answers
                if value.strip()
            ]
            if not items:
                st.warning("Add the requested evidence before resuming.")
            elif api_call(
                client.supply_information,
                case["case_id"],
                version=case["version"],
                items=items,
                supplied_by=reviewer,
            ):
                st.rerun()


def render_members(members: Sequence[Mapping[str, Any]]) -> None:
    with st.expander(f"Merchant identities ({len(members)})"):
        for member in members:
            name = html.escape(str(member.get("business_name") or "Unnamed merchant"))
            display_id = html.escape(short_merchant_id(str(member.get("member_id") or "")))
            business_type = html.escape(humanize(member.get("business_type")))
            st.html(
                '<div class="merchant-row"><div>'
                f'<div class="merchant-name">{name}</div>'
                f'<div class="merchant-id">{business_type}</div></div>'
                f'<div class="merchant-id">{display_id}</div></div>'
            )


def render_technical_details(client: MerchantShieldClient, case: Mapping[str, Any]) -> None:
    with st.expander("How MerchantShield reached this result"):
        automated = case.get("automated") or {}
        st.caption(
            "For auditors and technical judges. These details are not required to decide the case."
        )
        columns = st.columns(3)
        columns[0].metric("Graph score", _score(automated.get("primary_risk_score")))
        columns[1].metric("Policy action", humanize(automated.get("action")))
        columns[2].metric("Case version", str(case.get("version", "—")))
        assessment = (case.get("case_state") or {}).get("assessment") or {}
        for event in assessment.get("route_trace", ()):
            selected = ", ".join(event.get("selected_experts") or ()) or "none"
            st.write(f"{humanize(event.get('stage'))}: {selected} — {event.get('reason', '')}")
        signals = grouped_signals(case)
        if signals:
            st.dataframe(
                [
                    {
                        "similarity": row["label"],
                        "coverage": row["coverage"],
                        "pairs": row["pair_count"],
                        "used by investigator": row["cited"],
                    }
                    for row in signals
                ],
                hide_index=True,
                width="stretch",
            )
        events = api_call(client.events, case["case_id"], quiet=True)
        if events:
            st.caption("Append-only audit trail")
            st.dataframe(
                [
                    {
                        "#": item["sequence"],
                        "event": humanize(item["event_type"]),
                        "actor": item["actor"],
                        "time": item["recorded_at"][:19].replace("T", " "),
                    }
                    for item in events.get("events", ())
                ],
                hide_index=True,
                width="stretch",
            )


def model_validation(client: MerchantShieldClient) -> None:
    st.html('<div class="hero-kicker">For validation and judging</div>')
    st.title("The evidence behind the system.")
    report_choice = st.selectbox("Evaluation report", ("Current scoring comparison", "Hard-ring experiment"), key="evaluation_report_choice")
    report = (api_call(client.evaluation_report, "relationships") if report_choice == "Hard-ring experiment"
              else api_call(client.evaluation_report))
    if not report:
        return
    st.caption(report["disclaimer"])
    if report.get("benchmark_name"):
        st.subheader(report["benchmark_name"])
    st.html(
        '<div class="fact-grid">'
        + _fact("Training applications", str(report["train_count"]))
        + _fact("Held-out applications", str(report["test_count"]))
        + _fact("Split seed", str(report["seed"]))
        + _fact("Leakage control", "Ring / component")
        + "</div>"
    )
    partitions = report.get("partitions", {})
    if partitions:
        if report.get("validation_count"):
            st.caption(f"{report['validation_count']} separate validation applications · "
                       f"{partitions['test']['fraud_rings']} untouched test rings. Model and thresholds locked before testing.")
        else:
            st.caption(
                f"{partitions['train']['fraud_rings']} training rings · "
                f"{partitions['test']['fraud_rings']} held-out rings · "
                f"{len(report.get('split_seeds', []))} fixed split seeds checked."
            )
        st.info(report["evaluation_scope"])
    if report.get("comparison"):
        targets = report.get("validation_targets", {})
        measured = report["comparison"]["routed_moe"]
        if report.get("adoption_checks"):
            failed = [humanize(name) for name, passed in report["adoption_checks"].items() if not passed]
            if failed:
                st.warning("Shadow experiment only—not adopted. Failed checks: " + "; ".join(failed) + ".")
            else:
                st.info("The predefined experimental checks passed. This is still synthetic evidence, not permission to automate merchant approval or rejection.")
        if targets and (measured["recall"] < targets["merchant_recall"] or measured["ring_recall"] < targets["ring_recall"]):
            st.warning(f"Not promoted to live reviews: final-test merchant recall is {measured['recall']:.1%} "
                       f"and ring recall is {measured['ring_recall']:.1%}. Development targets were "
                       f"{targets['merchant_recall']:.0%} and {targets['ring_recall']:.0%}, respectively.")
        st.subheader("Before and after · identical test applications")
        st.dataframe([
            {"system": humanize(name), "precision": metric["precision"],
             "recall": metric["recall"], "false-positive rate": metric["false_positive_rate"],
             "ring recall": metric["ring_recall"], "review rate": metric["manual_review_rate"],
             "false-positive cost": metric["false_positive_cost"]}
            for name, metric in report["comparison"].items()
        ], hide_index=True, width="stretch")
        st.caption("The tuned legacy controls show what threshold changes alone achieve. Scores are risk flags—not verified fraud decisions.")
        st.subheader("All systems")
        if report.get("adoption_checks"):
            with st.expander("Hard cases · evasive rings and legitimate shared infrastructure"):
                st.dataframe([
                    {"cohort": humanize(cohort), "system": humanize(system),
                     "applications": values["sample_count"], "recall": values["recall"],
                     "false-positive rate": values["false_positive_rate"],
                     "ring recall": values["ring_recall"]}
                    for cohort, systems in report.get("cohort_reports", {}).items()
                    if cohort in ("evasive_shell_ring", "legitimate_shared_infrastructure")
                    for system, values in systems.items()
                    if system in ("prior_v1_retuned", "routed_moe")
                ], hide_index=True, width="stretch")
                st.caption("An unexplained shared relationship needs verification. It is neither proof of fraud nor proof that the merchant is legitimate.")
    baselines = report.get("baselines") or {}
    order = (
        "rules_only",
        "clustering_only",
        "graph_only",
        "tabular_only",
        "graph_ml",
        "hybrid",
        "routed_moe",
    )
    st.dataframe(
        [
            {
                "system": humanize(name),
                "precision": values["precision"],
                "recall": values["recall"],
                "PR-AUC": values["pr_auc"],
                "false-positive rate": values["false_positive_rate"],
                "ring recall": values["ring_recall"],
                "manual review rate": values["manual_review_rate"],
                "false-positive cost": values["false_positive_cost"],
            }
            for name in order
            if (values := baselines.get(name))
        ],
        hide_index=True,
        width="stretch",
    )
    st.caption(
        "These synthetic results validate the pipeline and evaluation method, not production Razorpay performance."
    )
    if report.get("metric_policy"):
        st.caption("Review rate includes high-risk applications and missing device/IP evidence. Cost is measured in configured review units, not rupees.")
        with st.expander("Test coverage and limitations"):
            st.dataframe([
                {"cohort": humanize(cohort), "training": count,
                 "held out": partitions["test"]["cohorts"].get(cohort, 0)}
                for cohort, count in partitions["train"]["cohorts"].items()
            ], hide_index=True, width="stretch")
            for explanation in report["metric_policy"].values():
                if isinstance(explanation, str):
                    st.caption(explanation)
            for limitation in report.get("limitations", []):
                st.caption(limitation)
        with st.expander("Thresholds and split checks" if report.get("thresholds") else "Results across split seeds"):
            if report.get("thresholds"):
                st.caption("One fresh final test; alert thresholds below were chosen only on validation data. No repeated-test tuning.")
                st.json(report["thresholds"])
            if report.get("stability"):
                st.caption("Ranges show split sensitivity, not confidence intervals. The table above uses the first preselected split.")
                st.dataframe([
                    {"system": humanize(name),
                     "precision range": f"{metrics['precision']['min']:.3f}–{metrics['precision']['max']:.3f}",
                     "recall range": f"{metrics['recall']['min']:.3f}–{metrics['recall']['max']:.3f}",
                     "false-positive rate range": f"{metrics['false_positive_rate']['min']:.3f}–{metrics['false_positive_rate']['max']:.3f}"}
                    for name, metrics in report.get("stability", {}).items()
                ], hide_index=True, width="stretch")


def _investigator_output(case: Mapping[str, Any]) -> Optional[Mapping[str, Any]]:
    investigation = (case.get("case_state") or {}).get("investigation") or {}
    output = investigation.get("output")
    return output if isinstance(output, Mapping) else None


def _fact(label: str, value: str) -> str:
    return (
        '<div class="fact">'
        f'<div class="fact-label">{html.escape(label)}</div>'
        f'<div class="fact-value">{html.escape(value)}</div></div>'
    )


def _score(value: object) -> str:
    return "Unknown" if value is None else f"{float(value):.3f}"


def main() -> None:
    st.set_page_config(page_title="MerchantShield · Review desk", page_icon="◈", layout="wide")
    inject_style()
    st.session_state.setdefault("surface", "Review")
    st.session_state.setdefault("reviewer_id", "reviewer_demo")

    with st.container(key="masthead"):
        brand, review, validation, settings = st.columns([4, 1.2, 1.5, 1], vertical_alignment="center")
        with brand:
            st.html('<div class="wordmark"><span class="brand-mark">◈</span> MerchantShield<span class="brand-divider">/</span><span class="brand-detail">Review desk</span></div>')
        with review:
            if st.button("Review", key="nav_review", type="tertiary", use_container_width=True):
                st.session_state["surface"] = "Review"
                st.session_state.pop("case_id", None)
                st.session_state.pop("pending_action", None)
        with validation:
            if st.button("Model validation", key="nav_validation", type="tertiary", use_container_width=True):
                st.session_state["surface"] = "Model validation"
        with settings:
            with st.popover("Session", use_container_width=True):
                st.text_input("Reviewer", key="reviewer_id")
                st.caption("Local demonstration identity. This is not an authenticated account.")

    client = get_client(API_BASE_URL)
    status = api_call(client.status)
    if status is None:
        st.info("Start the application with ./run.sh, keep the terminal open, then reload this page.")
        st.stop()
    render_mode(status["labels"])
    if st.session_state["surface"] == "Model validation":
        model_validation(client)
    elif st.session_state.get("case_id"):
        review_case(client, st.session_state["case_id"])
    else:
        review_home(client)

    st.html('<div class="page-footer"><span>MerchantShield · Merchant operations</span>'
            '<span>Synthetic data. Human decisions.</span></div>')


if __name__ == "__main__":
    main()
