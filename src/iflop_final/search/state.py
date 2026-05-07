"""Search result and internal state objects."""

from __future__ import annotations

from collections.abc import Iterable, Mapping
from dataclasses import dataclass, field

import numpy as np


@dataclass(slots=True)
class SearchResult:
    """Public result returned by final API functions."""

    adjacency: np.ndarray
    order: list[int]
    parents: dict[int, set[int]]
    total_score: float
    score_key: str
    score_metadata: dict[str, object] = field(default_factory=dict)
    trajectory: list[float] = field(default_factory=list)
    score_vector: tuple[int, int] | None = None

    def to_dict(self) -> dict[str, object]:
        return {
            "adjacency": self.adjacency.copy(),
            "order": list(self.order),
            "parents": {int(node): set(parents) for node, parents in self.parents.items()},
            "total_score": float(self.total_score),
            "score_key": self.score_key,
            "score_metadata": dict(self.score_metadata),
            "trajectory": list(self.trajectory),
            "score_vector": self.score_vector,
        }


@dataclass(slots=True)
class _CandidateState:
    order: tuple[int, ...]
    score: float
    parents: dict[int, set[int]]
    adjacency: np.ndarray
    score_vector: tuple[int, int] | None = None


def normalize_parents(parents: Mapping[int, Iterable[int]], num_vars: int) -> dict[int, set[int]]:
    return {node: {int(parent) for parent in parents.get(node, ())} for node in range(int(num_vars))}
