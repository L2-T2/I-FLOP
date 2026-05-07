from __future__ import annotations

import numpy as np

from iflop_final.config import GiesScoreConfig, SearchConfig
from iflop_final.data.simulation import generate_linear_gaussian_dataset
from iflop_final.graph.dag import adjacency_from_parents
from iflop_final.runtime.native import NATIVE_AVAILABLE, evaluate_native_order
from iflop_final.score.gies_bic import GiesBICScorer
from iflop_final.search.grow_shrink import parent_sets_for_order


def test_rust_fixed_order_matches_python_iflop_envwise() -> None:
    assert NATIVE_AVAILABLE
    dataset = generate_linear_gaussian_dataset(num_vars=4, samples_per_env=90, num_interventions=2, seed=131)
    order = tuple(range(dataset.num_vars))
    search_config = SearchConfig(ils_restarts=0, max_sweeps=1, random_seed=0)
    gies_config = GiesScoreConfig(variant="envwise")

    scorer = GiesBICScorer(dataset, gies_config)
    parents = parent_sets_for_order(scorer, order, atol=search_config.atol)
    python_score = scorer.total_score(parents)
    python_adjacency = adjacency_from_parents(parents, dataset.num_vars)

    native = evaluate_native_order(
        dataset,
        score_key="i_flop_envwise",
        order=order,
        search_config=search_config,
        gies_config=gies_config,
    )

    assert native.score_key == "i_flop_envwise"
    assert native.order == list(order)
    assert native.score_metadata["adjacency_type"] == "cpdag"
    assert set(np.unique(native.adjacency)).issubset({0, 1, 2})
    native_dag = np.asarray(native.score_metadata["dag_adjacency"], dtype=int)
    assert np.array_equal(native_dag, python_adjacency)
    assert np.isclose(native.total_score, python_score, rtol=1.0e-8, atol=1.0e-8)
