"""Prefix-constrained parent-set grow-shrink routine."""

from __future__ import annotations

from collections.abc import Iterable
from typing import Protocol

from iflop_final.score._linear import parent_tuple


class LocalScorer(Protocol):
    def local_score(self, node: int, parents: Iterable[int]) -> float: ...


def grow_shrink_parent_set(
    scorer: LocalScorer,
    node: int,
    prefix: Iterable[int],
    *,
    atol: float = 1.0e-10,
) -> set[int]:
    """Minimize the node-local score over a prefix by greedy grow-shrink."""

    candidates = set(int(item) for item in prefix if int(item) != int(node))
    parents: set[int] = set()
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
    return parents


def parent_sets_for_order(scorer: LocalScorer, order: Iterable[int], *, atol: float = 1.0e-10) -> dict[int, set[int]]:
    order_tuple = tuple(int(node) for node in order)
    parents: dict[int, set[int]] = {}
    for position, node in enumerate(order_tuple):
        parents[node] = grow_shrink_parent_set(scorer, node, order_tuple[:position], atol=atol)
    return parents
