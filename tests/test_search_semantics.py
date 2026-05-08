from __future__ import annotations

import _path_setup  # noqa: F401

import numpy as np

from iflop_final import run_iflop_envwise
from iflop_final.config import SearchConfig
from iflop_final.data.simulation import generate_linear_gaussian_dataset
from iflop_final.graph.dag import adjacency_from_parents, is_acyclic
from iflop_final.score.gies_bic import GiesBICScorer
from iflop_final.search.grow_shrink import parent_sets_for_order
from iflop_final.search.ils import default_perturbation_size, perturb_order
from iflop_final.search.local_search import local_reinsertion_search
from iflop_final.search.state import _CandidateState


def _legacy_full_recompute_score(dataset, config: SearchConfig) -> _CandidateState:
    scorer = GiesBICScorer(dataset)
    p = dataset.num_vars
    rng = np.random.default_rng(config.random_seed)
    cache: dict[tuple[int, ...], _CandidateState] = {}

    def evaluate(order: tuple[int, ...]) -> _CandidateState:
        if order not in cache:
            parents = parent_sets_for_order(scorer, order, atol=config.atol)
            adjacency = adjacency_from_parents(parents, p)
            total = scorer.total_score(parents)
            cache[order] = _CandidateState(order=order, score=total, parents=parents, adjacency=adjacency)
        return cache[order]

    initial_orders = [tuple(range(p))]
    for _ in range(max(int(config.ils_restarts), 0)):
        initial_orders.append(tuple(int(node) for node in rng.permutation(p)))

    best: _CandidateState | None = None
    k = config.perturbation_size or default_perturbation_size(p, config.dynamic_k_mode)
    for start in initial_orders:
        local_state, _trajectory = local_reinsertion_search(start, evaluate, config)
        current = local_state
        for _ in range(max(int(config.ils_restarts), 1)):
            perturbed = perturb_order(current.order, rng, k)
            candidate, _trajectory = local_reinsertion_search(perturbed, evaluate, config)
            if candidate.score < current.score - config.atol:
                current = candidate
        if best is None or current.score < best.score - config.atol:
            best = current
    assert best is not None
    return best


def test_optimized_search_matches_legacy_full_recompute_on_small_case() -> None:
    dataset = generate_linear_gaussian_dataset(num_vars=4, samples_per_env=35, num_interventions=2, seed=601)
    config = SearchConfig(ils_restarts=2, random_seed=17, max_sweeps=3)

    optimized = run_iflop_envwise(dataset, search_config=config, backend="python")
    legacy = _legacy_full_recompute_score(dataset, config)

    assert np.isclose(optimized.total_score, legacy.score, rtol=1.0e-10, atol=1.0e-10)
    assert optimized.order == list(legacy.order)
    assert optimized.parents == legacy.parents


def test_parent_sets_remain_valid_prefixes_and_output_is_acyclic() -> None:
    dataset = generate_linear_gaussian_dataset(num_vars=5, samples_per_env=30, num_interventions=2, seed=602)
    result = run_iflop_envwise(dataset, search_config=SearchConfig(ils_restarts=2, random_seed=3), backend="python")
    rank = {node: idx for idx, node in enumerate(result.order)}
    for child, parents in result.parents.items():
        assert all(rank[parent] < rank[child] for parent in parents)
    dag = adjacency_from_parents(result.parents, dataset.num_vars)
    assert is_acyclic(dag)
