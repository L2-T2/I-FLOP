from __future__ import annotations

import _path_setup  # noqa: F401

import numpy as np

from iflop.config import SearchConfig
from iflop.data.simulation import generate_linear_gaussian_dataset
from iflop.graph.dag import adjacency_from_parents
from iflop.runtime.rust import RUST_AVAILABLE, evaluate_rust_order
from iflop.score.obs_bic import FlopEnvwiseScorer, ObsBICScorer
from iflop.search.grow_shrink import parent_sets_for_order


def test_rust_fixed_order_matches_python_flop_obs() -> None:
    assert RUST_AVAILABLE
    dataset = generate_linear_gaussian_dataset(num_vars=4, samples_per_env=80, num_interventions=2, seed=171)
    order = tuple(range(dataset.num_vars))
    search_config = SearchConfig(ils_restarts=0, max_sweeps=1, random_seed=0)

    scorer = ObsBICScorer(dataset.observational_only())
    parents = parent_sets_for_order(scorer, order, atol=search_config.atol)
    python_score = scorer.total_score(parents)
    python_adjacency = adjacency_from_parents(parents, dataset.num_vars)

    rust_result = evaluate_rust_order(dataset, score_key="flop_obs", order=order, search_config=search_config)

    assert rust_result.score_key == "flop_obs"
    assert rust_result.score_metadata["adjacency_type"] == "cpdag"
    rust_dag = np.asarray(rust_result.score_metadata["dag_adjacency"], dtype=int)
    assert np.array_equal(rust_dag, python_adjacency)
    assert np.isclose(rust_result.total_score, python_score, rtol=1.0e-8, atol=1.0e-8)


def test_rust_fixed_order_matches_python_flop_envwise() -> None:
    assert RUST_AVAILABLE
    dataset = generate_linear_gaussian_dataset(num_vars=4, samples_per_env=80, num_interventions=2, seed=172)
    order = tuple(range(dataset.num_vars))
    search_config = SearchConfig(ils_restarts=0, max_sweeps=1, random_seed=0)

    scorer = FlopEnvwiseScorer(dataset)
    parents = parent_sets_for_order(scorer, order, atol=search_config.atol)
    python_score = scorer.total_score(parents)
    python_adjacency = adjacency_from_parents(parents, dataset.num_vars)

    rust_result = evaluate_rust_order(dataset, score_key="flop_envwise", order=order, search_config=search_config)

    assert rust_result.score_key == "flop_envwise"
    assert rust_result.score_metadata["adjacency_type"] == "cpdag"
    rust_dag = np.asarray(rust_result.score_metadata["dag_adjacency"], dtype=int)
    assert np.array_equal(rust_dag, python_adjacency)
    assert np.isclose(rust_result.total_score, python_score, rtol=1.0e-8, atol=1.0e-8)
