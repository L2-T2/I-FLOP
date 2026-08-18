"""Small local-score cache objects."""

from __future__ import annotations

import collections.abc as _abc
import dataclasses as _dataclasses

from iflop.score._linear import parent_tuple

@_dataclasses.dataclass(slots=True)
class LocalScoreCache:
    values: dict[tuple[str, int, tuple[int, ...]], float] = _dataclasses.field(default_factory=dict)
    hits: int = 0
    misses: int = 0

    def get_or_compute(
        self,
        context: str,
        node: int,
        parents: _abc.Iterable[int],
        compute: _abc.Callable[[], float],
    ) -> float:
        key = (str(context), int(node), parent_tuple(parents))
        if key in self.values:
            self.hits += 1
        else:
            self.misses += 1
            self.values[key] = float(compute())
        return self.values[key]

    def clear(self) -> None:
        self.values.clear()
        self.hits = 0
        self.misses = 0

    @property
    def calls(self) -> int:
        return int(self.hits + self.misses)

    @property
    def hit_rate(self) -> float:
        return float(self.hits / self.calls) if self.calls else 0.0

    def stats(self) -> dict[str, int | float]:
        return {
            "local_score_cache_size": len(self.values),
            "local_score_cache_hits": int(self.hits),
            "local_score_cache_misses": int(self.misses),
            "local_score_cache_calls": int(self.calls),
            "local_score_cache_hit_rate": float(self.hit_rate),
        }

    def __len__(self) -> int:
        return len(self.values)
