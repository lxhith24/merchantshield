"""One assembled MerchantShield runtime: models, workflow, store and service.

The demonstration world is deliberately frozen and synthetic. Scoring models
are fitted on the training split only and then serve every candidate, so the
numbers a reviewer sees on a held-out ring were never trained on that ring.
"""
from __future__ import annotations

import asyncio
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Mapping, Optional, Sequence, Tuple

from .agent import (
    CandidateContext,
    GraphScoringExpert,
    RoutedMoEConfig,
    RoutedMoESystem,
    RulesExpert,
    TabularScoringExpert,
)
from .analysis.evidence_graph import EvidenceGraphBuilder, RingCandidate
from .config import get_settings
from .evaluation.dataset import (
    SyntheticExample,
    generate_synthetic_dataset,
    load_jsonl,
)
from .evaluation.ring_models import GraphMLBaseline, TabularOnlyBaseline
from .evaluation.split import group_train_test_split
from .investigation import (
    BoundedInvestigator,
    CaseMemory,
    DeterministicInvestigator,
    build_investigator_provider,
)
from .models.merchant import MerchantApplication
from .gateway import OnboardingGateway, select_gateway
from .review.service import ReviewService
from .review.store import CaseStore
from .workflow import InMemoryCaseRepository, MerchantShieldWorkflow

DATASET_PATH = Path(__file__).resolve().parent.parent / "data/evaluation/synthetic_merchants.jsonl"
DATASET_SEED = 20250904


@dataclass(frozen=True)
class DemoWorld:
    """The frozen synthetic population the demonstration runs against."""

    examples: Tuple[SyntheticExample, ...]
    applications: Mapping[str, MerchantApplication]
    candidates: Tuple[RingCandidate, ...]
    training_example_ids: Tuple[str, ...]

    @property
    def by_id(self) -> Mapping[str, RingCandidate]:
        return {item.candidate_id: item for item in self.candidates}

    def label_for(self, candidate_id: str) -> str:
        """Ground-truth label, for the evaluation view only -- never for scoring."""
        candidate = self.by_id[candidate_id]
        labels = {item.example_id: item for item in self.examples}
        members = [labels[member] for member in candidate.member_ids if member in labels]
        if any(item.is_shell for item in members):
            return "shell_ring"
        return "legitimate"

    def is_held_out(self, candidate_id: str) -> bool:
        candidate = self.by_id[candidate_id]
        training = set(self.training_example_ids)
        return not any(member in training for member in candidate.member_ids)


def load_world(*, dataset_path: Optional[Path] = None) -> DemoWorld:
    """Load the frozen dataset if it exists, else regenerate it deterministically."""
    path = dataset_path or DATASET_PATH
    examples = load_jsonl(path) if path.exists() else generate_synthetic_dataset(
        seed=DATASET_SEED
    )
    applications = {item.example_id: item.application for item in examples}
    candidates = EvidenceGraphBuilder().build(applications)
    split = group_train_test_split(examples, seed=DATASET_SEED)
    return DemoWorld(
        examples=tuple(examples),
        applications=applications,
        candidates=candidates,
        training_example_ids=tuple(item.example_id for item in split.train),
    )


class MerchantShieldRuntime:
    """Everything the API and the demo UI need, assembled once."""

    def __init__(
        self,
        *,
        world: DemoWorld,
        service: ReviewService,
        workflow: MerchantShieldWorkflow,
        store: CaseStore,
        investigator_mode: str,
        gateway_mode: str = "SIMULATED",
    ) -> None:
        self.world = world
        self.service = service
        self.workflow = workflow
        self.store = store
        self.investigator_mode = investigator_mode
        self.gateway_mode = gateway_mode

    @property
    def labels(self) -> Mapping[str, str]:
        """The banners every surface must display, so nothing looks production-real."""
        return {
            "data": "SIMULATED",
            "gateway": self.gateway_mode,
            "investigator": self.investigator_mode,
        }


async def build_runtime(
    *,
    world: Optional[DemoWorld] = None,
    store: Optional[CaseStore] = None,
    force_offline_investigator: bool = False,
    router: Optional[RoutedMoESystem] = None,
    investigator: Optional[BoundedInvestigator] = None,
    gateway: Optional[OnboardingGateway] = None,
) -> MerchantShieldRuntime:
    """Fit the scoring models and assemble the runtime. Safe with no API key.

    `router` and `investigator` exist so tests and the scripted demonstration
    can pin a component without reaching inside the assembled runtime.
    """
    resolved_world = world or load_world()
    training = [
        item
        for item in resolved_world.examples
        if item.example_id in set(resolved_world.training_example_ids)
    ]

    graph_model = GraphMLBaseline(seed=DATASET_SEED)
    graph_model.fit(training)
    tabular_model = TabularOnlyBaseline(seed=DATASET_SEED)
    await tabular_model.fit(training)

    resolved_router = router or RoutedMoESystem(
        rules_expert=RulesExpert(),
        graph_expert=GraphScoringExpert(graph_model.score_candidate),
        tabular_expert=TabularScoringExpert(tabular_model.score_applications),
        config=RoutedMoEConfig(),
    )

    settings = get_settings()
    if investigator is not None:
        resolved_investigator = investigator
        mode = "LLM_DISABLED" if force_offline_investigator else "SCRIPTED"
    elif force_offline_investigator or not settings.llm_enabled:
        resolved_investigator = BoundedInvestigator(provider=DeterministicInvestigator())
        mode = "LLM_DISABLED"
    else:
        resolved_investigator = BoundedInvestigator(provider=build_investigator_provider())
        mode = "LLM_ENABLED"

    resolved_store = store or CaseStore()
    workflow = MerchantShieldWorkflow(
        repository=InMemoryCaseRepository(
            resolved_world.candidates, resolved_world.applications
        ),
        router=resolved_router,
        investigator=resolved_investigator,
        memory=CaseMemory(resolved_store.prior_records()),
    )
    resolved_gateway = gateway or select_gateway()
    service = ReviewService(
        workflow=workflow,
        store=resolved_store,
        candidates=resolved_world.by_id,
        gateway=resolved_gateway,
    )
    return MerchantShieldRuntime(
        world=resolved_world,
        service=service,
        workflow=workflow,
        store=resolved_store,
        investigator_mode=mode,
        gateway_mode=resolved_gateway.mode.value,
    )


_RUNTIME: Optional[MerchantShieldRuntime] = None
_LOCK = asyncio.Lock()


async def get_runtime() -> MerchantShieldRuntime:
    """Process-wide runtime singleton, built on first use."""
    global _RUNTIME
    if _RUNTIME is None:
        async with _LOCK:
            if _RUNTIME is None:
                _RUNTIME = await build_runtime()
    return _RUNTIME


def reset_runtime() -> None:
    """Drop the cached runtime. Used by tests and by the seed/reset scripts."""
    global _RUNTIME
    _RUNTIME = None
