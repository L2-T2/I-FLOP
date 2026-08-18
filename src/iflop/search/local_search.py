"""Order local-search helpers."""

from __future__ import annotations

import collections.abc as _abc

from iflop.config import SearchConfig
from iflop.graph.dag import adjacency_from_parents
from iflop.search.grow_shrink import LocalScorer, grow_shrink_parent_set_with_score
from iflop.search.state import _CandidateState

_Evaluator = _abc.Callable[[tuple[int, ...]], _CandidateState]


def reinsert_order(order: tuple[int, ...], source: int, dest: int) -> tuple[int, ...]:
    values = list(order)
    node = values.pop(int(source))
    values.insert(int(dest), node)
    return tuple(values)


def local_reinsertion_search(
    start_order: tuple[int, ...],
    evaluator: _Evaluator,
    config: SearchConfig,
) -> tuple[_CandidateState, list[float]]:
    current = evaluator(start_order)
    trajectory = [float(current.score)]
    max_sweeps = config.max_sweeps if config.max_sweeps is not None else max(2, 2 * len(start_order))
    for _sweep in range(int(max_sweeps)):
        best = current
        p = len(current.order)
        for source in range(p):
            for dest in range(p):
                if source == dest:
                    continue
                candidate_order = reinsert_order(current.order, source, dest)
                candidate = evaluator(candidate_order)
                if candidate.score < best.score - config.atol:
                    best = candidate
        if best.score < current.score - config.atol:
            current = best
            trajectory.append(float(current.score))
            continue
        break
    return current, trajectory


def affected_nodes_for_reinsert(order: tuple[int, ...], source: int, dest: int) -> tuple[int, ...]:
    if source == dest:
        return ()
    moved = reinsert_order(order, source, dest)
    lo = min(int(source), int(dest))
    hi = max(int(source), int(dest))
    return moved[lo : hi + 1]


def candidate_after_reinsert(
    current: _CandidateState,
    *,
    source: int,
    dest: int,
    scorer: LocalScorer,
    atol: float,
) -> _CandidateState:
    candidate_order = reinsert_order(current.order, source, dest)
    affected = set(affected_nodes_for_reinsert(current.order, source, dest))
    parents = {int(node): set(parent_set) for node, parent_set in current.parents.items()}
    local_scores = dict(current.local_scores)
    total = float(current.score)
    for position, node in enumerate(candidate_order):
        node_i = int(node)
        if node_i not in affected:
            continue
        old_parents = parents.get(node_i, set())
        old_score = float(local_scores.get(node_i, scorer.local_score(node_i, old_parents)))
        prefix = candidate_order[:position]
        new_parents, new_score = grow_shrink_parent_set_with_score(
            scorer,
            node_i,
            prefix,
            atol=atol,
            initial_parents=old_parents,
        )
        parents[node_i] = new_parents
        local_scores[node_i] = float(new_score)
        total += new_score - old_score
    return _CandidateState(
        order=candidate_order,
        score=float(total),
        parents=parents,
        adjacency=current.adjacency,
        local_scores=local_scores,
    )


def local_reinsertion_search_incremental(
    start_state: _CandidateState,
    scorer: LocalScorer,
    config: SearchConfig,
) -> tuple[_CandidateState, list[float], dict[str, int]]:
    current = start_state
    trajectory = [float(current.score)]
    max_sweeps = config.max_sweeps if config.max_sweeps is not None else max(2, 2 * len(current.order))
    stats = {
        "local_search_passes": 0,
        "reinsert_calls": 0,
        "affected_node_updates": 0,
    }
    for _sweep in range(int(max_sweeps)):
        stats["local_search_passes"] += 1
        best = current
        p = len(current.order)
        for source in range(p):
            for dest in range(p):
                if source == dest:
                    continue
                stats["reinsert_calls"] += 1
                stats["affected_node_updates"] += abs(int(dest) - int(source)) + 1
                candidate = candidate_after_reinsert(
                    current,
                    source=source,
                    dest=dest,
                    scorer=scorer,
                    atol=config.atol,
                )
                if candidate.score < best.score - config.atol:
                    best = candidate
        if best.score < current.score - config.atol:
            current = best
            trajectory.append(float(current.score))
            continue
        break
    return current, trajectory, stats
