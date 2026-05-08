"""FLOP-aligned order search using decomposable local scores."""

from __future__ import annotations

from collections.abc import Iterable, Mapping

import numpy as np

from iflop_final.config import SearchConfig
from iflop_final.graph.dag import adjacency_from_parents
from iflop_final.search.grow_shrink import LocalScorer, parent_sets_for_order
from iflop_final.search.ils import default_perturbation_size, perturb_order
from iflop_final.search.local_search import local_reinsertion_search_incremental
from iflop_final.search.state import SearchResult, _CandidateState


def run_decomposable_order_search(
    scorer: LocalScorer,
    *,
    score_key: str,
    config: SearchConfig | None = None,
    metadata: dict[str, object] | None = None,
) -> SearchResult:
    cfg = config or SearchConfig()
    p = int(getattr(scorer, "num_vars"))
    rng = np.random.default_rng(cfg.random_seed)
    cache: dict[tuple[int, ...], _CandidateState] = {}
    search_stats = {
        "full_order_initializations": 0,
        "local_search_calls": 0,
        "local_search_passes": 0,
        "reinsert_calls": 0,
        "affected_node_updates": 0,
    }

    def evaluate(order: tuple[int, ...]) -> _CandidateState:
        if order not in cache:
            search_stats["full_order_initializations"] += 1
            parents = parent_sets_for_order(scorer, order, atol=cfg.atol)
            adjacency = adjacency_from_parents(parents, p)
            local_scores = {
                int(node): float(getattr(scorer, "local_score")(node, parents.get(node, ())))
                for node in range(p)
            }
            total = float(sum(local_scores.values()))
            cache[order] = _CandidateState(
                order=order,
                score=total,
                parents=parents,
                adjacency=adjacency,
                local_scores=local_scores,
            )
        return cache[order]

    initial_orders = [tuple(range(p))]
    for _ in range(max(int(cfg.ils_restarts), 0)):
        initial_orders.append(tuple(int(node) for node in rng.permutation(p)))

    best: _CandidateState | None = None
    full_trajectory: list[float] = []
    k = cfg.perturbation_size or default_perturbation_size(p, cfg.dynamic_k_mode)
    for start in initial_orders:
        start_state = evaluate(start)
        local_state, trajectory, stats = local_reinsertion_search_incremental(start_state, scorer, cfg)
        search_stats["local_search_calls"] += 1
        search_stats["local_search_passes"] += stats["local_search_passes"]
        search_stats["reinsert_calls"] += stats["reinsert_calls"]
        search_stats["affected_node_updates"] += stats["affected_node_updates"]
        full_trajectory.extend(trajectory)
        current = local_state
        for _ in range(max(int(cfg.ils_restarts), 1)):
            perturbed = perturb_order(current.order, rng, k)
            perturbed_state = evaluate(perturbed)
            candidate, trajectory, stats = local_reinsertion_search_incremental(perturbed_state, scorer, cfg)
            search_stats["local_search_calls"] += 1
            search_stats["local_search_passes"] += stats["local_search_passes"]
            search_stats["reinsert_calls"] += stats["reinsert_calls"]
            search_stats["affected_node_updates"] += stats["affected_node_updates"]
            full_trajectory.extend(trajectory)
            if candidate.score < current.score - cfg.atol:
                current = candidate
        if best is None or current.score < best.score - cfg.atol:
            best = current
    assert best is not None

    score_metadata = dict(metadata or {})
    if hasattr(scorer, "local_diagnostics"):
        score_metadata["local_diagnostics"] = {
            int(node): getattr(scorer, "local_diagnostics")(node, best.parents.get(node, ()))
            for node in range(p)
        }
    score_metadata["cache_size"] = len(cache)
    score_metadata["search_stats"] = dict(search_stats)
    if hasattr(getattr(scorer, "cache", None), "stats"):
        score_metadata["score_cache_stats"] = getattr(scorer, "cache").stats()
    if hasattr(scorer, "scatter_construction_count"):
        score_metadata["scatter_construction_count"] = int(getattr(scorer, "scatter_construction_count"))
    best_adjacency = adjacency_from_parents(best.parents, p)
    return SearchResult(
        adjacency=best_adjacency,
        order=list(best.order),
        parents=best.parents,
        total_score=float(best.score),
        score_key=score_key,
        score_metadata=score_metadata,
        trajectory=full_trajectory,
    )


def run_flop_search(
    scorer: LocalScorer,
    *,
    score_key: str,
    config: SearchConfig | None = None,
    metadata: Mapping[str, object] | None = None,
) -> SearchResult:
    return run_decomposable_order_search(scorer, score_key=score_key, config=config, metadata=dict(metadata or {}))
