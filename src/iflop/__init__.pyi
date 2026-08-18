from collections.abc import Mapping
from typing import Literal, TypeAlias

import numpy.typing as npt

from iflop.config import IFlopScoreConfig as IFlopScoreConfig
from iflop.config import SearchConfig as SearchConfig
from iflop.data.dataset import MultiEnvDataset as MultiEnvDataset
from iflop.search.state import SearchResult as SearchResult

_Backend: TypeAlias = Literal["rust", "python", "auto"]

__all__ = [
    "MultiEnvDataset",
    "IFlopScoreConfig",
    "SearchConfig",
    "SearchResult",
    "available_scores",
    "result_summary",
    "run_flop_envwise",
    "run_flop_obs",
    "run_iflop",
]


def available_scores() -> tuple[str, ...]: ...


def run_flop_obs(
    data: MultiEnvDataset | npt.ArrayLike,
    *,
    search_config: SearchConfig | None = None,
    backend: _Backend = "rust",
) -> SearchResult: ...


def run_flop_envwise(
    data: MultiEnvDataset | npt.ArrayLike,
    *,
    search_config: SearchConfig | None = None,
    backend: _Backend = "rust",
) -> SearchResult: ...


def run_iflop(
    dataset: MultiEnvDataset,
    *,
    score_config: IFlopScoreConfig | None = None,
    search_config: SearchConfig | None = None,
    backend: _Backend = "rust",
) -> SearchResult: ...


def result_summary(result: SearchResult) -> Mapping[str, object]: ...
