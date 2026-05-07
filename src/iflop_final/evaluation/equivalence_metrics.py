"""Skeleton-level metrics used as conservative equivalence-class proxies."""

from __future__ import annotations

import numpy as np

from iflop_final.graph.metrics import shd


def skeleton_shd(estimated: np.ndarray, truth: np.ndarray) -> int:
    return shd(estimated, truth, oriented=False)
