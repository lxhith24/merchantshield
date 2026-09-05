"""Calm, decision-first reviewer surface for MerchantShield.

The default experience answers one question: do these merchants appear to be
controlled together? Technical traces remain available, but never compete with
the evidence or the human decision.

    .venv/bin/streamlit run merchantshield/ui/streamlit_app.py
"""
from __future__ import annotations

import html
import csv
import io
import os
import sys
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple

import streamlit as st

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from merchantshield.ui.client import ApiError, MerchantShieldClient, VersionConflict  # noqa: E402
from merchantshield.demo.live_check import (  # noqa: E402
    MAX_LIVE_CHECK_BYTES,
    MAX_LIVE_CHECK_ROWS,
    REQUIRED_COLUMNS,
)
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
        '<div class="mode-strip"><span class="engine-pill"><span class="status-dot"></span>Engine ready</span>'
        '<span>Synthetic demonstration</span>'
        f'<span class="mode-secondary">· {html.escape(investigator)} · {html.escape(gateway)}</span></div>'
    )


def humanize(value: object) -> str:
    return str(value or "").replace("_", " ").strip().capitalize()


def reviewer_greeting(value: object) -> str:
    """Return the friendly session heading for a named local reviewer."""
    name = " ".join(str(value or "").split())
    if not name or name.casefold() == "reviewer_demo":
        return "Reviewer"
    return f"Hey, {name}"


def combine_live_check_uploads(uploaded_files: Sequence[Any]) -> str:
    """Merge multiple UTF-8 CSV uploads into one bounded transient check."""
    rows: List[Dict[str, str]] = []
    total_bytes = 0
    for file_number, uploaded in enumerate(uploaded_files, start=1):
        raw = uploaded.getvalue()
        if not isinstance(raw, bytes):
            raw = str(raw).encode("utf-8")
        total_bytes += len(raw)
        if total_bytes > MAX_LIVE_CHECK_BYTES:
            raise ValueError(
                f"Combined CSV uploads exceed the {MAX_LIVE_CHECK_BYTES // 1_000_000} MB demo limit"
            )
        try:
            text = raw.decode("utf-8-sig")
        except UnicodeDecodeError as exc:
            raise ValueError(f"File {file_number} must be a UTF-8 CSV") from exc
        reader = csv.DictReader(io.StringIO(text))
        headers = tuple(reader.fieldnames or ())
        missing = [column for column in REQUIRED_COLUMNS if column not in headers]
        if missing:
            raise ValueError(
                f"File {file_number} is missing required columns: {', '.join(missing)}"
            )
        rows.extend(
            {column: row.get(column, "") for column in REQUIRED_COLUMNS}
            for row in reader
        )

    if not rows:
        raise ValueError("CSV uploads have no merchant rows")
    if len(rows) > MAX_LIVE_CHECK_ROWS:
        raise ValueError(
            f"Combined CSV uploads exceed the {MAX_LIVE_CHECK_ROWS}-merchant demo limit"
        )

    output = io.StringIO(newline="")
    writer = csv.DictWriter(
        output,
        fieldnames=REQUIRED_COLUMNS,
        lineterminator="\n",
    )
    writer.writeheader()
    writer.writerows(rows)
    return output.getvalue()


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


def live_merchant_check(client: MerchantShieldClient) -> None:
    """Paste or upload new synthetic onboarding fields and run the real pipeline."""
    st.html(
        '<div class="live-hero">'
        '<div class="signal-field" aria-hidden="true">'
        '<div class="signal-stars"></div>'
        '<div class="signal-column col-left"></div><div class="signal-column col-mid-left"></div>'
        '<div class="signal-column col-mid-right"></div><div class="signal-column col-right"></div>'
        '<div class="signal-ascii ascii-top">:::::::-----==++×××××###888888888888###×××××++==-----:::::::</div>'
        '<div class="signal-ascii ascii-ring">....::::---==++××XX##88##XX××++==---::::....::::---==++××XX##88##XX××++==---::::....</div>'
        '<div class="signal-ascii ascii-mid">::::---==++××××++++====----::::....::::----====++++××××++==---::::</div>'
        '<div class="signal-ascii ascii-lower">::::---==++××××++==---::::....::::---==++××××++==---::::....::::---==++××××++==---::::</div>'
        '<div class="signal-ascii ascii-base">....::::----====++++××××××++++====----::::....::::----====++++××××××++++====----::::....</div>'
        '<div class="signal-scanline"></div>'
        '<div class="signal-mascot"><span class="mascot-face"><i></i><i></i></span><span class="mascot-foot foot-a"></span><span class="mascot-foot foot-b"></span></div>'
        '</div><div class="hero-content">'
        '<div class="hero-kicker">LIVE · SYNTHETIC · TRACEABLE</div>'
        '<h1>Find the links <em>hidden</em><br>in merchant data.</h1>'
        '<p>Drop new onboarding records. MerchantShield builds the evidence graph, routes the '
        'right experts, and shows exactly why a connection needs attention.</p>'
        '<div class="hero-proof">Your data. Your evidence. Human decisions.</div>'
        '</div></div>'
    )

    examples_payload = api_call(client.live_check_examples) or {}
    examples = examples_payload.get("examples") or {}
    if "live_check_csv" not in st.session_state:
        default = examples.get("new_ring") or {}
        st.session_state["live_check_csv"] = default.get("csv", "")

    with st.container(key="live_ingest", border=True):
        st.subheader("Batch CSV check")
        st.caption("Upload one or more synthetic CSV files, or paste rows. Up to 1 MB and 100 merchants per demo run.")
        uploaded = st.file_uploader(
            "Upload synthetic CSV files",
            type=("csv",),
            max_upload_size=1,
            accept_multiple_files=True,
            help="The file is sent for this check only and is not stored.",
            label_visibility="collapsed",
        )
        st.html('<div class="fixture-label">Quick load</div>')
        example_columns = st.columns(3)
        example_order = ("known_ring_match", "new_ring", "independent")
        for column, key in zip(example_columns, example_order):
            example = examples.get(key) or {}
            with column:
                if st.button(
                    example.get("label", humanize(key)),
                    key=f"live-example-{key}",
                    help=example.get("description"),
                    use_container_width=True,
                ):
                    st.session_state["live_check_csv"] = example.get("csv", "")
                    st.session_state.pop("live_check_result", None)
                    st.rerun()
        with st.expander("Paste CSV details instead"):
            st.text_area(
                "Merchant details",
                key="live_check_csv",
                height=220,
                label_visibility="collapsed",
            )
            st.caption(
                "Required: merchant ID, business and owner names, settlement account, device, "
                "IP address, registered address and submission time."
            )
        _, action = st.columns([2.9, 1], vertical_alignment="center")
        with action:
            if st.button("Run live check", type="primary", use_container_width=True):
                csv_text = st.session_state["live_check_csv"]
                if uploaded:
                    try:
                        csv_text = combine_live_check_uploads(uploaded)
                    except ValueError as exc:
                        st.error(str(exc))
                        csv_text = ""
                if csv_text:
                    with st.spinner("Building the evidence graph and running the investigation..."):
                        result = api_call(client.live_check, csv_text)
                    if result:
                        st.session_state["live_check_result"] = result

    result = st.session_state.get("live_check_result")
    if result:
        render_live_check_result(result)


def render_live_check_result(payload: Mapping[str, Any]) -> None:
    """Render only evidence and traces returned by the API—no simulated animation."""
    results = list(payload.get("results") or [])
    linked = [row for row in results if row.get("finding") != "no_ring_link_found"]
    reference_ids = {
        member_id
        for row in results
        for member_id in row.get("reference_member_ids", ())
    }
    st.divider()
    st.html('<div class="hero-kicker">ASSESSMENT COMPLETE · NOT SAVED</div>')
    if linked:
        st.html(
            '<div class="result-summary"><div><div class="result-eyebrow">Risk assessment</div>'
            '<div class="result-title">Potential linked group</div>'
            f'<div class="result-copy">{len(linked)} connected candidate group(s) found. '
            'This is a risk signal, not a fraud verdict.</div></div>'
            '<span class="risk-pill high">Human review</span></div>'
        )
    else:
        st.html(
            '<div class="result-summary"><div><div class="result-eyebrow">Risk assessment</div>'
            '<div class="result-title">No ring connection found</div>'
            '<div class="result-copy">No submitted merchant linked to the checked reference population. '
            'This does not authenticate documents or prove legitimacy.</div></div>'
            '<span class="risk-pill low">No ring alert</span></div>'
        )
    st.html(
        '<div class="fact-grid">'
        + _fact("Candidate groups", str(payload.get("result_count", len(results))))
        + _fact("Submitted merchants", str(payload.get("submitted_count", 0)))
        + _fact("Linked groups", str(len(linked)))
        + _fact("Reference matches", str(len(reference_ids)))
        + "</div>"
    )
    st.caption(str(payload.get("scope_note") or ""))

    for index, row in enumerate(results, start=1):
        candidate = row.get("candidate") or {}
        members = row.get("members") or []
        finding = row.get("finding")
        if finding == "linked_to_synthetic_reference":
            title = "Matches an existing synthetic ring"
        elif finding == "new_linked_group":
            title = "New linked merchant group"
        else:
            names = [
                member.get("business_name", "Merchant")
                for member in members
                if member.get("source") == "submitted"
            ]
            title = f"No ring link · {names[0] if names else f'Merchant {index}'}"
        with st.container(key=f"live-result-{index}", border=True):
            heading, status = st.columns([3, 1], vertical_alignment="center")
            heading.subheader(title)
            status.caption("Review required" if row.get("action") == "human_review" else "No ring alert")
            st.image(
                svg_data_uri(render_ring_svg(candidate, members=members, width=820, height=420)),
                width="stretch",
            )
            submitted_count = len(row.get("submitted_member_ids") or ())
            reference_count = len(row.get("reference_member_ids") or ())
            st.caption(
                f"{submitted_count} submitted merchant(s) · {reference_count} synthetic reference match(es)"
            )
            case_like = {
                "candidate": candidate,
                "case_state": {"investigation": row.get("investigation")},
            }
            signals = grouped_signals(case_like)
            if signals:
                st.dataframe(
                    [
                        {
                            "similarity": item["label"],
                            "coverage": item["coverage"],
                            "linked pairs": item["pair_count"],
                        }
                        for item in signals
                    ],
                    hide_index=True,
                    width="stretch",
                )
            else:
                st.caption("No shared infrastructure crossed the graph-link threshold.")
            if row.get("investigation"):
                render_investigator_summary(case_like)
            with st.expander("Verified execution trace"):
                st.caption("These are returned by the backend workflow; they are not a staged animation.")
                for event in row.get("route_trace", ()):
                    experts = ", ".join(event.get("selected_experts") or ()) or "policy"
                    st.write(f"{humanize(event.get('stage'))} · {experts}: {event.get('reason', '')}")
                for event in row.get("workflow_trace", ()):
                    st.write(f"{humanize(event.get('node'))}: {event.get('detail', '')}")
                grounding = (row.get("investigation") or {}).get("grounding") or {}
                if grounding:
                    st.write(f"Evidence grounding: {humanize(grounding.get('status'))}")


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
             "false positives": metric.get("false_positive_count", 0),
             "false-positive cost": metric["false_positive_cost"]}
            for name, metric in report["comparison"].items()
        ], hide_index=True, width="stretch")
        st.caption("The tuned legacy controls show what threshold changes alone achieve. Scores are risk flags—not verified fraud decisions.")
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
    if baselines:
        st.subheader("Layer comparison · same held-out applications")
        st.caption("This is the model ablation: each row adds a different evidence source while keeping the test merchants and split fixed.")
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
                "false positives": values.get("false_positive_count", 0),
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
    with st.expander("What each layer contributes"):
        st.dataframe(
            [
                {"system": label, "evidence used": description}
                for label, description in (
                    ("Rules only", "Deterministic identity, document and velocity checks; no fitted model."),
                    ("Clustering only", "Shared-attribute groups; useful for discovery, but no fraud labels."),
                    ("Graph only", "Relationship strength, corroboration and component shape; no fitted labels."),
                    ("Tabular only", "A fitted merchant-level model on identity and onboarding signals."),
                    ("Graph ML", "A fitted model on graph topology and relationship features."),
                    ("Hybrid", "A fitted model combining graph, tabular and peer signals."),
                    ("Routed MoE", "Rules always run; graph and tabular experts are routed by uncertainty or disagreement."),
                )
            ],
            hide_index=True,
            width="stretch",
        )
        st.caption("The optional investigator/reviewer LLM explains cited evidence only. It never owns the numeric score or a blacklist/reject action.")
    routed_metric = baselines.get("routed_moe")
    if routed_metric:
        with st.expander("Why precision and recall move together"):
            st.caption("The operating threshold is chosen on validation data, then held fixed for this test. Raising it usually reduces false positives but can miss more rings.")
            st.dataframe([_confusion_counts(routed_metric)], hide_index=True, width="stretch")
            st.caption("Counts are reconstructed from the report's rounded recall plus its exact false-positive count; use the rates above for headline claims.")
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


def _confusion_counts(metric: Mapping[str, Any]) -> Mapping[str, int]:
    """Return an interpretable confusion summary for one held-out report row."""
    sample_count = max(0, int(metric.get("sample_count") or 0))
    positive_count = max(0, int(metric.get("positive_count") or 0))
    recall = max(0.0, min(1.0, float(metric.get("recall") or 0.0)))
    true_positive = min(positive_count, max(0, round(recall * positive_count)))
    false_negative = max(0, positive_count - true_positive)
    false_positive = max(0, int(metric.get("false_positive_count") or 0))
    negative_count = max(0, sample_count - positive_count)
    true_negative = max(0, negative_count - false_positive)
    return {
        "true positives": true_positive,
        "false negatives": false_negative,
        "false positives": false_positive,
        "true negatives": true_negative,
    }


def main() -> None:
    st.set_page_config(page_title="MerchantShield · Review desk", page_icon="◈", layout="wide")
    inject_style()
    st.session_state.setdefault("surface", "Review")
    st.session_state.setdefault("reviewer_id", "reviewer_demo")

    with st.container(key="masthead"):
        brand, review, live_check, validation, settings = st.columns([3.7, 1.05, 1.35, 1.45, .9], vertical_alignment="center")
        with brand:
            st.html('<div class="wordmark"><span class="brand-mark">◈</span> MerchantShield<span class="brand-divider">/</span><span class="brand-detail">Review desk</span></div>')
        with review:
            if st.button("Review", key="nav_review", type="tertiary", use_container_width=True):
                st.session_state["surface"] = "Review"
                st.session_state.pop("case_id", None)
                st.session_state.pop("pending_action", None)
        with live_check:
            if st.button("Check merchants", key="nav_live_check", type="tertiary", use_container_width=True):
                st.session_state["surface"] = "Check merchants"
                st.session_state.pop("case_id", None)
        with validation:
            if st.button("Model validation", key="nav_validation", type="tertiary", use_container_width=True):
                st.session_state["surface"] = "Model validation"
        with settings:
            with st.popover("Session", use_container_width=True):
                reviewer_name = st.text_input(
                    "Reviewer",
                    key="reviewer_id",
                    label_visibility="collapsed",
                )
                st.html(
                    '<div class="session-greeting" aria-live="polite">'
                    f"{html.escape(reviewer_greeting(reviewer_name))}</div>"
                )
                st.caption("Local demonstration identity. This is not an authenticated account.")

    client = get_client(API_BASE_URL)
    status = api_call(client.status)
    if status is None:
        st.info("Start the application with ./run.sh, keep the terminal open, then reload this page.")
        st.stop()
    render_mode(status["labels"])
    if st.session_state["surface"] == "Model validation":
        model_validation(client)
    elif st.session_state["surface"] == "Check merchants":
        live_merchant_check(client)
    elif st.session_state.get("case_id"):
        review_case(client, st.session_state["case_id"])
    else:
        review_home(client)

    st.html('<div class="page-footer"><span>MerchantShield · Merchant operations</span>'
            '<span>Synthetic data. Human decisions.</span></div>')


if __name__ == "__main__":
    main()
