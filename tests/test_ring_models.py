"""Train/test contracts for graph, tabular, and hybrid ring scorers."""
from __future__ import annotations

import pytest

from merchantshield.evaluation import (
    GraphMLBaseline,
    GraphOnlyBaseline,
    HybridRingBaseline,
    TabularOnlyBaseline,
    generate_synthetic_dataset,
    group_train_test_split,
)


@pytest.mark.asyncio
async def test_fitted_models_score_only_the_held_out_partition():
    split = group_train_test_split(generate_synthetic_dataset(seed=404), seed=405)
    expected = {row.example_id for row in split.test}

    graph_ml = GraphMLBaseline(seed=7)
    graph_ml.fit(split.train)
    graph_scores = graph_ml.score(split.test)

    tabular = TabularOnlyBaseline(seed=7)
    await tabular.fit(split.train)
    tabular_scores = await tabular.score(split.test)

    hybrid = HybridRingBaseline(seed=7)
    await hybrid.fit(split.train)
    hybrid_scores = await hybrid.score(split.test)

    deterministic_scores = GraphOnlyBaseline().score(split.test)
    for scores in (
        graph_scores,
        tabular_scores,
        hybrid_scores,
        deterministic_scores,
    ):
        assert set(scores) == expected
        assert all(0.0 <= score <= 1.0 for score in scores.values())


def test_models_refuse_scoring_before_fit():
    examples = generate_synthetic_dataset(seed=505)[:3]
    with pytest.raises(RuntimeError, match="fitted"):
        GraphMLBaseline().score(examples)

