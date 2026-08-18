"""Small preprocessing helpers used by search initialization and scores."""

from __future__ import annotations

import numpy as np


def center_columns(matrix: np.ndarray) -> np.ndarray:
    data = np.asarray(matrix, dtype=float)
    return data - np.mean(data, axis=0, keepdims=True)


def standardize_columns(matrix: np.ndarray, eps: float = 1.0e-12) -> np.ndarray:
    data = center_columns(matrix)
    scale = np.std(data, axis=0, keepdims=True)
    return data / np.maximum(scale, eps)

