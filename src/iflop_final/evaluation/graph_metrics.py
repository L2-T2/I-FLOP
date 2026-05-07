"""DAG-level evaluation metrics."""

from __future__ import annotations

import numpy as np

from iflop_final.graph.metrics import precision_recall_f1, shd


def graph_shd(estimated: np.ndarray, truth: np.ndarray, *, oriented: bool = True) -> int:
    return shd(estimated, truth, oriented=oriented)


def graph_precision_recall_f1(estimated: np.ndarray, truth: np.ndarray, *, oriented: bool = True) -> dict[str, float]:
    return precision_recall_f1(estimated, truth, oriented=oriented)
