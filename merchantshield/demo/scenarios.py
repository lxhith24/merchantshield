"""Frozen demonstration scenarios over the synthetic world.

A scenario is a *claim about behaviour*, not a script: it names a labelled
group in the frozen dataset and states the outcome the real pipeline produces
for it. `tests/test_demo_scenarios.py` runs the actual runtime against every
expectation here, so a scenario that stops being true fails the suite instead
of quietly misleading a demonstration.

Nothing in this module scores, decides, or persists anything.
"""
from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import Any, Dict, Mapping, Optional, Tuple

from ..agent.contracts import CandidateAction
from ..evaluation.dataset import Cohort


class ScenarioKind(str, Enum):
    """How a scenario is exercised."""

    #: A real candidate group. Open it as a case through the normal API.
    QUEUE = "queue"
    #: A deterministic harness run that is never persisted as a case, used to
    #: reach a failure path the fitted models do not reach on frozen data.
    HARNESS = "harness"


@dataclass(frozen=True)
class DemoScenario:
    """One frozen scenario, addressed by its dataset component label."""

    key: str
    title: str
    kind: ScenarioKind
    component_label: str
    cohort: Cohort
    headline: str
    what_to_watch: Tuple[str, ...]
    expected_action: Optional[CandidateAction] = None
    expected_review_status: Optional[str] = None
    expected_experts: Tuple[str, ...] = ()
    expected_investigator_status: str = "skipped"
    harness: Optional[str] = None
    honest_note: str = ""

    def to_dict(self) -> Dict[str, Any]:
        return {
            "key": self.key,
            "title": self.title,
            "kind": self.kind.value,
            "component_label": self.component_label,
            "cohort": self.cohort.value,
            "headline": self.headline,
            "what_to_watch": list(self.what_to_watch),
            "expected_action": self.expected_action.value if self.expected_action else None,
            "expected_review_status": self.expected_review_status,
            "expected_experts": list(self.expected_experts),
            "expected_investigator_status": self.expected_investigator_status,
            "harness": self.harness,
            "honest_note": self.honest_note,
        }


@dataclass(frozen=True)
class ResolvedScenario:
    """A scenario bound to the candidate it addresses in the loaded world."""

    scenario: DemoScenario
    candidate_id: str
    member_ids: Tuple[str, ...]
    shared_attributes: Tuple[str, ...]
    ground_truth: str
    held_out: bool

    def to_dict(self) -> Dict[str, Any]:
        payload = self.scenario.to_dict()
        payload.update(
            candidate_id=self.candidate_id,
            member_ids=list(self.member_ids),
            member_count=len(self.member_ids),
            shared_attributes=list(self.shared_attributes),
            ground_truth=self.ground_truth,
            held_out=self.held_out,
        )
        return payload


SCENARIOS: Tuple[DemoScenario, ...] = (
    DemoScenario(
        key="clean_solo_merchant",
        title="Ordinary merchant, no links",
        kind=ScenarioKind.QUEUE,
        component_label="singleton-0000",
        cohort=Cohort.LEGITIMATE,
        headline="A single applicant with no shared infrastructure clears without an LLM.",
        what_to_watch=(
            "Only the two mandatory experts run; the tabular expert is not routed.",
            "The investigator never runs, so a clean case costs no model tokens.",
        ),
        expected_action=CandidateAction.PROCEED_TO_ONBOARDING,
        expected_review_status="cleared",
        expected_experts=("rules", "graph_model"),
        expected_investigator_status="skipped",
    ),
    DemoScenario(
        key="legitimate_shared_office",
        title="Legitimate shared office (hard negative)",
        kind=ScenarioKind.QUEUE,
        component_label="legit-shared-00",
        cohort=Cohort.LEGITIMATE_SHARED_INFRA,
        headline=(
            "Four independent merchants at one coworking address on one office "
            "network are not treated as a ring."
        ),
        what_to_watch=(
            "Address and IP links alone stay below the review threshold.",
            "Submissions are weeks apart, unlike a bulk-created ring.",
        ),
        expected_action=CandidateAction.PROCEED_TO_ONBOARDING,
        expected_review_status="cleared",
        expected_experts=("rules", "graph_model"),
        expected_investigator_status="skipped",
    ),
    DemoScenario(
        key="obvious_ring",
        title="Obvious shell ring",
        kind=ScenarioKind.QUEUE,
        component_label="obvious-ring-00",
        cohort=Cohort.OBVIOUS_SHELL_RING,
        headline=(
            "Five applications settling to one bank account, from one device, "
            "at one address."
        ),
        what_to_watch=(
            "The rules expert and the graph expert agree, so no second opinion "
            "is bought: the tabular expert is not routed.",
            "The action is human review. There is no automatic rejection.",
        ),
        expected_action=CandidateAction.HUMAN_REVIEW,
        expected_review_status="pending_human_review",
        expected_experts=("rules", "graph_model"),
        expected_investigator_status="completed",
    ),
    DemoScenario(
        key="evasive_ring",
        title="Evasive ring on held-out data",
        kind=ScenarioKind.QUEUE,
        component_label="evasive-ring-01",
        cohort=Cohort.EVASIVE_SHELL_RING,
        headline=(
            "Every application looks clean on its own. Only the graph links "
            "them, and this ring was never in the training split."
        ),
        what_to_watch=(
            "The per-application rules expert scores this low; the graph expert "
            "scores it high.",
            "That disagreement is what routes the expensive tabular expert in.",
            "This component is held out, so no model was fitted on it.",
        ),
        expected_action=CandidateAction.HUMAN_REVIEW,
        expected_review_status="pending_human_review",
        expected_experts=("rules", "graph_model", "tabular_model"),
        expected_investigator_status="completed",
    ),
    DemoScenario(
        key="shared_kiosk_false_positive",
        title="Shared kiosk: a false positive we do not hide",
        kind=ScenarioKind.QUEUE,
        component_label="legit-shared-03",
        cohort=Cohort.LEGITIMATE_SHARED_INFRA,
        headline=(
            "Four legitimate merchants sharing a kiosk device, address and "
            "network are sent to a human. That is a false positive."
        ),
        what_to_watch=(
            "A shared device is strong evidence, and this group is legitimate.",
            "The cost of being wrong here is one review, not a rejection.",
            "A reviewer can clear it with a verified-tenancy reason code.",
        ),
        expected_action=CandidateAction.HUMAN_REVIEW,
        expected_review_status="pending_human_review",
        expected_experts=("rules", "graph_model", "tabular_model"),
        expected_investigator_status="completed",
        honest_note=(
            "This group is labelled legitimate. It is included deliberately: "
            "the false-positive rate on the held-out set is 0.211, which is a "
            "review gate, not an automatic-rejection system."
        ),
    ),
    DemoScenario(
        key="fabricated_citation",
        title="The model cites evidence that does not exist",
        kind=ScenarioKind.HARNESS,
        component_label="obvious-ring-00",
        cohort=Cohort.OBVIOUS_SHELL_RING,
        headline=(
            "A scripted model returns a confident 'continue onboarding' built "
            "on an evidence ID that was never disclosed to it."
        ),
        what_to_watch=(
            "Deterministic code, not a prompt, rejects the citation.",
            "The whole narrative is discarded and kept only for audit.",
            "The case becomes human review; it can never read as clean.",
        ),
        harness="grounding_failure",
        expected_investigator_status="completed",
        honest_note=(
            "This run uses a scripted provider so the failure is reproducible "
            "with no API key. It is not persisted as a case."
        ),
    ),
    DemoScenario(
        key="information_request_and_resume",
        title="Pause for a decisive fact, then resume the same case",
        kind=ScenarioKind.HARNESS,
        component_label="legit-shared-00",
        cohort=Cohort.LEGITIMATE_SHARED_INFRA,
        headline=(
            "Weak links plus a tight submission window is genuinely ambiguous, "
            "so the investigator asks one question instead of guessing."
        ),
        what_to_watch=(
            "The workflow interrupts and the case is durably awaiting information.",
            "Supplying the evidence resumes the identical case ID.",
        ),
        harness="information_request",
        expected_investigator_status="awaiting_information",
        honest_note=(
            "The graph risk is pinned for this harness run so the uncertain-links "
            "path is reachable on frozen data, where the fitted model scores this "
            "group confidently low. The pinned value is reported in the response."
        ),
    ),
)


SCENARIOS_BY_KEY: Mapping[str, DemoScenario] = {item.key: item for item in SCENARIOS}


class ScenarioNotFound(KeyError):
    """No scenario with that key, or its component is not in the loaded world."""


def component_index(world) -> Mapping[str, str]:
    """Map each dataset component/ring label to the candidate that contains it.

    A label only resolves when every member of the candidate carries it, so a
    scenario can never silently point at a merged, differently-shaped group.
    """
    cohort_by_id = {item.example_id: item for item in world.examples}
    index: Dict[str, str] = {}
    for candidate in world.candidates:
        labels = {
            cohort_by_id[member].component_id
            for member in candidate.member_ids
            if member in cohort_by_id
        }
        if len(labels) == 1:
            index[labels.pop()] = candidate.candidate_id
    return index


def resolve_scenario(world, key: str) -> ResolvedScenario:
    """Bind one scenario to the candidate it addresses in `world`."""
    try:
        scenario = SCENARIOS_BY_KEY[key]
    except KeyError as exc:  # pragma: no cover - trivial
        raise ScenarioNotFound(key) from exc
    index = component_index(world)
    candidate_id = index.get(scenario.component_label)
    if candidate_id is None:
        raise ScenarioNotFound(
            f"scenario '{key}' expects component '{scenario.component_label}', "
            "which is not a whole candidate in this dataset"
        )
    candidate = world.by_id[candidate_id]
    return ResolvedScenario(
        scenario=scenario,
        candidate_id=candidate_id,
        member_ids=tuple(candidate.member_ids),
        shared_attributes=tuple(sorted({edge.attribute for edge in candidate.evidence})),
        ground_truth=world.label_for(candidate_id),
        held_out=world.is_held_out(candidate_id),
    )


def resolve_scenarios(world) -> Tuple[ResolvedScenario, ...]:
    """Every frozen scenario, in demonstration order."""
    return tuple(resolve_scenario(world, item.key) for item in SCENARIOS)
