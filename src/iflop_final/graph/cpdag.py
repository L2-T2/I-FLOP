"""Minimal CPDAG-facing utilities.

The final package estimates DAGs. These helpers expose skeleton-level views for
evaluation notes; they do not implement a full Chickering CPDAG operator suite.
"""

from __future__ import annotations

import numpy as np

from iflop_final.graph.dag import as_square_adjacency


def dag_skeleton(adjacency: np.ndarray) -> np.ndarray:
    arr = as_square_adjacency(adjacency)
    return ((arr + arr.T) > 0).astype(int)


def dag_to_cpdag_proxy(adjacency: np.ndarray) -> np.ndarray:
    """Return a conservative skeleton proxy, not an exact CPDAG."""

    proxy = dag_skeleton(adjacency)
    np.fill_diagonal(proxy, 0)
    return proxy
