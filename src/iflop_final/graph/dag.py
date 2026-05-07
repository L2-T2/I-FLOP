"""Adjacency-matrix DAG helpers."""

from __future__ import annotations

from collections import deque
from collections.abc import Iterable, Mapping

import numpy as np


def as_square_adjacency(adjacency: np.ndarray) -> np.ndarray:
    arr = np.asarray(adjacency, dtype=int)
    if arr.ndim != 2 or arr.shape[0] != arr.shape[1]:
        raise ValueError("adjacency must be a square matrix.")
    if np.any(np.diag(arr) != 0):
        raise ValueError("self loops are not allowed.")
    return (arr != 0).astype(int)


def parents_of(adjacency: np.ndarray, node: int) -> set[int]:
    arr = as_square_adjacency(adjacency)
    return {int(i) for i in np.flatnonzero(arr[:, int(node)])}


def children_of(adjacency: np.ndarray, node: int) -> set[int]:
    arr = as_square_adjacency(adjacency)
    return {int(i) for i in np.flatnonzero(arr[int(node), :])}


def parents_from_adjacency(adjacency: np.ndarray) -> dict[int, set[int]]:
    arr = as_square_adjacency(adjacency)
    return {node: parents_of(arr, node) for node in range(arr.shape[0])}


def adjacency_from_parents(parents: Mapping[int, Iterable[int]], num_vars: int) -> np.ndarray:
    p = int(num_vars)
    adjacency = np.zeros((p, p), dtype=int)
    for child, parent_set in parents.items():
        child_i = int(child)
        for parent in parent_set:
            parent_i = int(parent)
            if parent_i == child_i:
                raise ValueError("self loops are not allowed.")
            adjacency[parent_i, child_i] = 1
    if not is_acyclic(adjacency):
        raise ValueError("parent map does not define an acyclic graph.")
    return adjacency


def topological_order(adjacency: np.ndarray) -> list[int]:
    arr = as_square_adjacency(adjacency)
    p = arr.shape[0]
    indegree = arr.sum(axis=0).astype(int).tolist()
    queue: deque[int] = deque([node for node, degree in enumerate(indegree) if degree == 0])
    order: list[int] = []
    while queue:
        node = queue.popleft()
        order.append(int(node))
        for child in np.flatnonzero(arr[node, :]):
            indegree[int(child)] -= 1
            if indegree[int(child)] == 0:
                queue.append(int(child))
    if len(order) != p:
        raise ValueError("graph contains a directed cycle.")
    return order


def is_acyclic(adjacency: np.ndarray) -> bool:
    try:
        topological_order(adjacency)
    except ValueError:
        return False
    return True


def order_index(order: Iterable[int]) -> dict[int, int]:
    return {int(node): rank for rank, node in enumerate(order)}


def prefix_nodes(order: Iterable[int], node: int) -> tuple[int, ...]:
    order_list = [int(item) for item in order]
    try:
        position = order_list.index(int(node))
    except ValueError as exc:
        raise ValueError(f"node {node} is not present in the order.") from exc
    return tuple(order_list[:position])
