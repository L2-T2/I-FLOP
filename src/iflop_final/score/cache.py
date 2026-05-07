"""Small local-score cache objects."""

from __future__ import annotations

from collections.abc import Callable, Iterable
from dataclasses import dataclass, field
from typing import TypeVar

from iflop_final.score._linear import parent_tuple

T = TypeVar("T")


@dataclass(slots=True)
class LocalScoreCache:
    """Cache local scores keyed by node, sorted parent set, and optional context."""

    values: dict[tuple[str, int, tuple[int, ...]], float] = field(default_factory=dict)

    def get_or_compute(
        self,
        context: str,
        node: int,
        parents: Iterable[int],
        compute: Callable[[], float],
    ) -> float:
        key = (str(context), int(node), parent_tuple(parents))
        if key not in self.values:
            self.values[key] = float(compute())
        return self.values[key]

    def clear(self) -> None:
        self.values.clear()

    def __len__(self) -> int:
        return len(self.values)
