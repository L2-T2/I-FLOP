"""Order local-search helpers."""

from __future__ import annotations

from collections.abc import Callable

from iflop_final.config import SearchConfig
from iflop_final.search.state import _CandidateState

Evaluator = Callable[[tuple[int, ...]], _CandidateState]


def reinsert_order(order: tuple[int, ...], source: int, dest: int) -> tuple[int, ...]:
    values = list(order)
    node = values.pop(int(source))
    values.insert(int(dest), node)
    return tuple(values)


def local_reinsertion_search(
    start_order: tuple[int, ...],
    evaluator: Evaluator,
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
