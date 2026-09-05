"""Exercise the review desk as a user, with no network or persistent writes."""
from copy import deepcopy
import csv
import io
import json
from pathlib import Path
import pytest
from unittest.mock import patch

from streamlit.testing.v1 import AppTest

from merchantshield.ui.client import ApiError
from merchantshield.ui import streamlit_app as ui


class DemoClient:
    def __init__(self):
        self.claims = []
        self.resolutions = []
        self.live_checks = []
        self.record = {
            "case_id": "case-preview", "version": 1, "status": "pending_review",
            "member_count": 2, "members": [],
            "automated": {"action": "human_review", "primary_risk_score": .9},
            "candidate": {"member_ids": ["syn-01", "syn-02"], "evidence": []},
        }

    def status(self):
        return {"labels": {"data": "SIMULATED", "gateway": "SIMULATED", "investigator": "LLM_DISABLED"}}

    def list_cases(self, **kwargs):
        return {"cases": [deepcopy(self.record)]}

    def case(self, case_id):
        return deepcopy(self.record)

    def scenarios(self):
        return []

    def events(self, case_id):
        return {"events": []}

    def live_check_examples(self):
        return {
            "examples": {
                "known_ring_match": {"label": "known_ring_match.csv", "description": "Known match", "csv": "known"},
                "new_ring": {"label": "alpha_ring.csv", "description": "Linked batch", "csv": "alpha"},
                "independent": {"label": "beta_clean.csv", "description": "Clean batch", "csv": "beta"},
            }
        }

    def live_check(self, csv_text):
        self.live_checks.append(csv_text)
        return {
            "submitted_count": 3,
            "result_count": 0,
            "results": [],
            "scope_note": "Synthetic transient comparison.",
        }

    def claim(self, case_id, reviewer, version):
        self.claims.append((case_id, reviewer, version))
        self.record.update(status="claimed", claimed_by=reviewer, version=2)
        return deepcopy(self.record)

    def resolve(self, case_id, **kwargs):
        self.resolutions.append(kwargs)
        self.record.update(status="resolved", version=3, human=kwargs)
        return deepcopy(self.record)

    def evaluation_report(self):
        return {
            "disclaimer": "Synthetic benchmark only.", "train_count": 75,
            "test_count": 29, "seed": 20250904, "baselines": {},
        }


def app():
    return AppTest.from_string(
        "from merchantshield.ui.streamlit_app import main\nmain()"
    )


def button(at, label):
    return next(b for b in at.button if b.label == label)


def test_preview_is_read_only_and_hold_requires_explicit_confirmation():
    client = DemoClient()
    with patch.object(ui, "get_client", return_value=client):
        at = app().run()
        assert not at.exception
        assert client.claims == client.resolutions == []
        at.button(key="review_next").click().run()
        assert not at.exception
        assert len(client.claims) == 1
        button(at, "Confirm suspicious ring").click().run()
        assert client.resolutions == []
        button(at, "Confirm and finish").click().run()
        assert not at.exception
        assert client.resolutions[0]["decision"] == "keep_on_hold"
        assert client.resolutions[0]["outcome"] == "confirmed_ring"


def test_top_navigation_returns_to_queue_without_resolving():
    client = DemoClient()
    with patch.object(ui, "get_client", return_value=client):
        at = app().run()
        at.button(key="review_next").click().run()
        at.button(key="nav_validation").click().run()
        assert not at.exception
        assert at.title[0].value == "The evidence behind the system."
        at.button(key="nav_review").click().run()
        assert not at.exception
        assert at.button(key="review_next")
        assert client.resolutions == []


def test_live_check_navigation_loads_a_fixture_and_calls_the_real_client_method():
    client = DemoClient()
    with patch.object(ui, "get_client", return_value=client):
        at = app().run()
        at.button(key="nav_live_check").click().run()
        assert not at.exception
        assert button(at, "alpha_ring.csv")
        at.button(key="live-example-new_ring").click().run()
        button(at, "Run live check").click().run()
        assert not at.exception
        assert client.live_checks == ["alpha"]
        assert any("Synthetic transient comparison" in item.value for item in at.caption)


def test_expanded_validation_shows_scope_and_seed_ranges():
    client = DemoClient()
    report = json.loads((Path(__file__).resolve().parents[1] / "data/evaluation/expanded_report.json").read_text())
    with patch.object(ui, "get_client", return_value=client), patch.object(client, "evaluation_report", return_value=report):
        at = app().run()
        at.button(key="nav_validation").click().run()
        assert not at.exception
        assert at.subheader[0].value == "Expanded synthetic benchmark"
        assert "not promoted" in at.info[0].value
        assert any("35 held-out rings" in caption.value for caption in at.caption)
        # Benchmark, hard-case cohort, layer ablation, routed confusion summary,
        # and split-sensitivity tables are all rendered as evidence panels.
        assert len(at.dataframe) >= 5
        assert any("Layer comparison" in heading.value for heading in at.subheader)
        assert client.claims == client.resolutions == []


def test_performance_validation_discloses_tradeoff_and_no_promotion():
    client = DemoClient()
    report = json.loads((Path(__file__).resolve().parents[1] / "data/evaluation/performance_report.json").read_text())
    report["validation_targets"] = {"merchant_recall": .8, "ring_recall": .9}
    with patch.object(ui, "get_client", return_value=client), patch.object(client, "evaluation_report", return_value=report):
        at = app().run()
        at.button(key="nav_validation").click().run()
        assert not at.exception
        assert "Not promoted" in at.warning[0].value
        assert any("813 separate validation" in caption.value for caption in at.caption)
        assert any("Before and after" in heading.value for heading in at.subheader)
        assert client.claims == client.resolutions == []


def test_failed_relationship_experiment_is_optional_and_clearly_marked():
    client = DemoClient()
    report = json.loads((Path(__file__).resolve().parents[1] / "data/evaluation/relationships/report.json").read_text())
    report["validation_targets"] = {"merchant_recall": .8, "ring_recall": .9}
    with patch.object(ui, "get_client", return_value=client), patch.object(client, "evaluation_report", return_value=report) as fetch:
        at = app().run()
        at.button(key="nav_validation").click().run()
        assert at.selectbox(key="evaluation_report_choice").value == "Current scoring comparison"
        at.selectbox(key="evaluation_report_choice").select("Hard-ring experiment").run()
        assert not at.exception
        fetch.assert_called_with("relationships")
        assert any("not adopted" in item.value for item in at.warning)
        assert any("Hard cases" in item.label for item in at.expander)
        assert client.claims == client.resolutions == []


def test_unavailable_queue_does_not_claim_everything_is_clear():
    client = DemoClient()
    with patch.object(ui, "get_client", return_value=client), patch.object(
        client, "list_cases", side_effect=ApiError(503, "Service unavailable")
    ):
        at = app().run()
        assert not at.exception
        assert any("queue is unavailable" in item.value for item in at.info)
        assert not any("caught up" in item.value for item in at.subheader)


def test_empty_reviewer_cannot_start_a_claim():
    client = DemoClient()
    with patch.object(ui, "get_client", return_value=client):
        at = app().run()
        at.text_input(key="reviewer_id").set_value("").run()
        at.button(key="review_next").click().run()
        assert not at.exception
        assert client.claims == []
        assert any("reviewer name" in item.value for item in at.warning)


def test_named_reviewer_sees_a_friendly_session_greeting():
    client = DemoClient()
    with patch.object(ui, "get_client", return_value=client):
        at = app().run()
        at.text_input(key="reviewer_id").set_value("Johan").run()
        assert not at.exception
        assert any("Hey, Johan" in item.value for item in at.get("html"))


def test_multiple_csv_uploads_are_combined_into_one_batch():
    header = "merchant_id,business_name,owner_name,bank_account,device_fingerprint,ip_address,registered_address,submitted_at"

    class Upload:
        def __init__(self, text):
            self.text = text

        def getvalue(self):
            return self.text.encode("utf-8")

    merged = ui.combine_live_check_uploads(
        [
            Upload(header + "\nA-001,One Shop,A One,111111111,device-a,198.51.100.1,Address A,2026-01-01T00:00:00Z\n"),
            Upload(header + "\nA-002,Two Shop,A Two,111111111,device-a,198.51.100.2,Address B,2026-01-01T00:00:10Z\n"),
            Upload(header + "\nA-003,Three Shop,A Three,111111111,device-b,198.51.100.3,Address C,2026-01-01T00:00:20Z\n"),
        ]
    )
    rows = list(csv.DictReader(io.StringIO(merged)))
    assert [row["merchant_id"] for row in rows] == ["A-001", "A-002", "A-003"]


def test_multiple_csv_uploads_keep_the_combined_merchant_limit():
    class Upload:
        def __init__(self, text):
            self.text = text

        def getvalue(self):
            return self.text.encode("utf-8")

    header = "merchant_id,business_name,owner_name,bank_account,device_fingerprint,ip_address,registered_address,submitted_at"
    row = "A-001,One Shop,A One,111111111,device-a,198.51.100.1,Address A,2026-01-01T00:00:00Z"
    with pytest.raises(ValueError, match="100-merchant demo limit"):
        ui.combine_live_check_uploads([Upload(header + "\n" + "\n".join(row.replace("A-001", f"A-{n:03d}") for n in range(101)))])
