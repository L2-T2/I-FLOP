"""CPDAG and I-CPDAG conversion utilities."""

from __future__ import annotations

import collections.abc as _abc

import numpy as np

from iflop.graph.dag import as_square_adjacency, parents_from_adjacency, topological_order


def dag_skeleton(adjacency: np.ndarray) -> np.ndarray:
    arr = as_square_adjacency(adjacency)
    return ((arr + arr.T) > 0).astype(int)


def dag_to_cpdag(adjacency: np.ndarray) -> np.ndarray:
    """Convert a DAG adjacency matrix to a CPDAG adjacency matrix.

    Encoding follows the upstream FLOP convention: ``1`` denotes a directed
    edge from row to column, and symmetric ``2`` entries denote an undirected
    edge.
    """

    dag = as_square_adjacency(adjacency)
    p = int(dag.shape[0])
    parents = {node: set(parents) for node, parents in parents_from_adjacency(dag).items()}
    ordering = {node: idx for idx, node in enumerate(topological_order(dag))}

    edges = [(parent, child) for parent in range(p) for child in range(p) if dag[parent, child] != 0]
    edges.sort(key=lambda edge: (ordering[edge[1]], -ordering[edge[0]]))

    edge_types = np.zeros((p, p), dtype=int)
    for x, y in edges:
        if edge_types[x, y] != 0:
            continue
        parents_y = sorted(parents[y])
        all_adjacent = True
        for w in range(p):
            if edge_types[w, x] == 1:
                if dag[w, y] == 0:
                    edge_types[x, y] = 1
                    all_adjacent = False
                    break
                edge_types[w, y] = 1
        if not all_adjacent:
            continue
        parents_x_plus_x = sorted((*parents[x], x))
        if parents_y != parents_x_plus_x:
            for z in parents_y:
                if z != x:
                    edge_types[z, y] = 1
            edge_types[x, y] = 1
        else:
            edge_types[x, y] = 2

    cpdag = np.zeros((p, p), dtype=int)
    for x in range(p):
        for y in range(p):
            if edge_types[x, y] == 1:
                cpdag[x, y] = 1
            elif edge_types[x, y] == 2:
                cpdag[x, y] = 2
                cpdag[y, x] = 2
    return cpdag


def dag_to_icpdag(
    adjacency: np.ndarray,
    intervention_targets: _abc.Mapping[int, set[int]] | None = None,
) -> np.ndarray:
    """Convert a DAG to an I-CPDAG under known intervention targets."""

    dag = as_square_adjacency(adjacency)
    icpdag = dag_to_cpdag(dag)
    targets = tuple(frozenset(int(node) for node in nodes) for nodes in (intervention_targets or {}).values())
    p = int(dag.shape[0])

    for target_set in targets:
        if not target_set:
            continue
        for u in range(p):
            for v in range(u + 1, p):
                if _has_undirected(icpdag, u, v) and ((u in target_set) != (v in target_set)):
                    _orient_as_dag(icpdag, dag, u, v)

    _apply_meek_closure(icpdag)
    return icpdag


def _orient_as_dag(partial: np.ndarray, dag: np.ndarray, u: int, v: int) -> bool:
    if dag[u, v] != 0:
        partial[u, v] = 1
        partial[v, u] = 0
        return True
    if dag[v, u] != 0:
        partial[v, u] = 1
        partial[u, v] = 0
        return True
    return False


def _has_any_edge(graph: np.ndarray, u: int, v: int) -> bool:
    return bool(graph[u, v] != 0 or graph[v, u] != 0)


def _has_directed(graph: np.ndarray, u: int, v: int) -> bool:
    return bool(graph[u, v] == 1 and graph[v, u] == 0)


def _has_undirected(graph: np.ndarray, u: int, v: int) -> bool:
    return bool(graph[u, v] == 2 and graph[v, u] == 2)


def _orient(partial: np.ndarray, u: int, v: int) -> bool:
    if not _has_undirected(partial, u, v):
        return False
    partial[u, v] = 1
    partial[v, u] = 0
    return True


def _apply_meek_closure(partial: np.ndarray) -> None:
    p = int(partial.shape[0])
    changed = True
    while changed:
        changed = False
        for a in range(p):
            for b in range(p):
                if not _has_directed(partial, a, b):
                    continue
                for c in range(p):
                    if c in {a, b}:
                        continue
                    if _has_undirected(partial, b, c) and not _has_any_edge(partial, a, c):
                        changed = _orient(partial, b, c) or changed

        for a in range(p):
            for b in range(p):
                if not _has_undirected(partial, a, b):
                    continue
                for c in range(p):
                    if c in {a, b}:
                        continue
                    if _has_directed(partial, a, c) and _has_directed(partial, c, b):
                        changed = _orient(partial, a, b) or changed
                        break

        for a in range(p):
            for b in range(p):
                if not _has_undirected(partial, a, b):
                    continue
                candidates = [
                    c
                    for c in range(p)
                    if c not in {a, b} and _has_undirected(partial, a, c) and _has_directed(partial, c, b)
                ]
                for idx, c in enumerate(candidates):
                    for d in candidates[idx + 1 :]:
                        if not _has_any_edge(partial, c, d):
                            changed = _orient(partial, a, b) or changed
                            break
                    if not _has_undirected(partial, a, b):
                        break
