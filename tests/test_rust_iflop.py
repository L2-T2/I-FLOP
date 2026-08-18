from __future__ import annotations

import _path_setup  # noqa: F401

import numpy as np

from iflop.config import IFlopScoreConfig, SearchConfig
from iflop.data.simulation import generate_linear_gaussian_dataset
from iflop.graph.dag import adjacency_from_parents
from iflop.runtime.rust import RUST_AVAILABLE, evaluate_rust_order
from iflop.score.iflop_bic import IFlopBICScorer
from iflop.search.grow_shrink import parent_sets_for_order


def test_rust_fixed_order_matches_python_iflop() -> None:
    assert RUST_AVAILABLE
    dataset = generate_linear_gaussian_dataset(num_vars=4, samples_per_env=90, num_interventions=2, seed=131)
    order = tuple(range(dataset.num_vars))
    search_config = SearchConfig(ils_restarts=0, max_sweeps=1, random_seed=0)
    score_config = IFlopScoreConfig()

    scorer = IFlopBICScorer(dataset, score_config)
    parents = parent_sets_for_order(scorer, order, atol=search_config.atol)
    python_score = scorer.total_score(parents)
    python_adjacency = adjacency_from_parents(parents, dataset.num_vars)

    rust_result = evaluate_rust_order(
        dataset,
        score_key="iflop",
        order=order,
        search_config=search_config,
        score_config=score_config,
    )

    assert rust_result.score_key == "iflop"
    assert rust_result.order == list(order)
    assert rust_result.score_metadata["adjacency_type"] == "i_cpdag"
    assert set(np.unique(rust_result.adjacency)).issubset({0, 1, 2})
    rust_dag = np.asarray(rust_result.score_metadata["dag_adjacency"], dtype=int)
    assert np.array_equal(rust_dag, python_adjacency)
    assert np.isclose(rust_result.total_score, python_score, rtol=1.0e-8, atol=1.0e-8)
