"""Exercise the review desk as a user, with no network or persistent writes."""
from copy import deepcopy
import json
from pathlib import Path
from unittest.mock import patch

from streamlit.testing.v1 import AppTest

from merchantshield.ui.client import ApiError
from merchantshield.ui import streamlit_app as ui


class DemoClient:
    def __init__(self):
        self.claims = []
        self.resolutions = []
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
        assert len(at.dataframe) == 3
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
