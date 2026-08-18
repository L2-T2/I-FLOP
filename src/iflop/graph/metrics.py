"""Small graph metric helpers."""

from __future__ import annotations

import numpy as np

from iflop.graph.dag import as_square_adjacency


def _directed_edges(adjacency: np.ndarray) -> set[tuple[int, int]]:
    arr = as_square_adjacency(adjacency)
    return {
        (int(i), int(j))
        for i in range(arr.shape[0])
        for j in range(arr.shape[1])
        if arr[i, j] != 0
    }


def _skeleton_edges(adjacency: np.ndarray) -> set[frozenset[int]]:
    return {frozenset(edge) for edge in _directed_edges(adjacency)}


def shd(estimated: np.ndarray, truth: np.ndarray, *, oriented: bool = True) -> int:
    """Return directed or skeleton structural Hamming distance."""

    if oriented:
        return len(_directed_edges(estimated) ^ _directed_edges(truth))
    return len(_skeleton_edges(estimated) ^ _skeleton_edges(truth))


def precision_recall_f1(
    estimated: np.ndarray,
    truth: np.ndarray,
    *,
    oriented: bool = True,
) -> dict[str, float]:
    if oriented:
        estimated_edges = _directed_edges(estimated)
        true_edges = _directed_edges(truth)
        tp = len(estimated_edges & true_edges)
        fp = len(estimated_edges - true_edges)
        fn = len(true_edges - estimated_edges)
    else:
        estimated_skeleton = _skeleton_edges(estimated)
        true_skeleton = _skeleton_edges(truth)
        tp = len(estimated_skeleton & true_skeleton)
        fp = len(estimated_skeleton - true_skeleton)
        fn = len(true_skeleton - estimated_skeleton)
    precision = tp / (tp + fp) if tp + fp else 0.0
    recall = tp / (tp + fn) if tp + fn else 0.0
    f1 = 2.0 * precision * recall / (precision + recall) if precision + recall else 0.0
    return {"precision": precision, "recall": recall, "f1": f1, "tp": float(tp), "fp": float(fp), "fn": float(fn)}
