"""Prefix-constrained parent-set grow-shrink routine."""

from __future__ import annotations

import collections.abc as _abc
import typing as _typing

from iflop.score._linear import parent_tuple


class LocalScorer(_typing.Protocol):
    def local_score(self, node: int, parents: _abc.Iterable[int]) -> float: ...


def grow_shrink_parent_set(
    scorer: LocalScorer,
    node: int,
    prefix: _abc.Iterable[int],
    *,
    atol: float = 1.0e-10,
    initial_parents: _abc.Iterable[int] | None = None,
) -> set[int]:
    parents, _score = grow_shrink_parent_set_with_score(
        scorer,
        node,
        prefix,
        atol=atol,
        initial_parents=initial_parents,
    )
    return parents


def grow_shrink_parent_set_with_score(
    scorer: LocalScorer,
    node: int,
    prefix: _abc.Iterable[int],
    *,
    atol: float = 1.0e-10,
    initial_parents: _abc.Iterable[int] | None = None,
) -> tuple[set[int], float]:
    candidates = set(int(item) for item in prefix if int(item) != int(node))
    parents: set[int] = {int(parent) for parent in (initial_parents or ()) if int(parent) in candidates}
    current = float(scorer.local_score(int(node), parents))
    changed = True
    while changed:
        changed = False
        best_add: int | None = None
        best_add_score = current
        for cand in sorted(candidates - parents):
            score = float(scorer.local_score(int(node), parent_tuple((*parents, cand))))
            if score < best_add_score - atol:
                best_add_score = score
                best_add = cand
        if best_add is not None:
            parents.add(best_add)
            current = best_add_score
            changed = True

        while parents:
            best_remove: int | None = None
            best_remove_score = current
            for cand in sorted(parents):
                trial = set(parents)
                trial.remove(cand)
                score = float(scorer.local_score(int(node), trial))
                if score < best_remove_score - atol:
                    best_remove_score = score
                    best_remove = cand
            if best_remove is None:
                break
            parents.remove(best_remove)
            current = best_remove_score
            changed = True
    return parents, float(current)


def parent_sets_for_order(
    scorer: LocalScorer,
    order: _abc.Iterable[int],
    *,
    atol: float = 1.0e-10,
) -> dict[int, set[int]]:
    order_tuple = tuple(int(node) for node in order)
    parents: dict[int, set[int]] = {}
    for position, node in enumerate(order_tuple):
        parents[node] = grow_shrink_parent_set(scorer, node, order_tuple[:position], atol=atol)
    return parents
