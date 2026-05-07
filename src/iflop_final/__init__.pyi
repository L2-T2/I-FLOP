from collections.abc import Mapping
from typing import Literal

import numpy as np

from iflop_final.config import GiesScoreConfig, SearchConfig
from iflop_final.data.dataset import MultiEnvDataset
from iflop_final.search.state import SearchResult

Backend = Literal["rust", "python", "auto"]

def available_scores() -> tuple[str, ...]: ...
def run_flop_obs(
    data: MultiEnvDataset | np.ndarray,
    *,
    search_config: SearchConfig | None = None,
    backend: Backend = "rust",
) -> SearchResult: ...
def run_flop_envwise(
    data: MultiEnvDataset | np.ndarray,
    *,
    search_config: SearchConfig | None = None,
    backend: Backend = "rust",
) -> SearchResult: ...
def run_iflop_envwise(
    dataset: MultiEnvDataset,
    *,
    gies_config: GiesScoreConfig | None = None,
    search_config: SearchConfig | None = None,
    penalty_sample_mode: str | None = None,
    backend: Backend = "rust",
) -> SearchResult: ...
def run_iflop(
    dataset: MultiEnvDataset | np.ndarray,
    *,
    score_key: str = "i_flop_envwise",
    search_config: SearchConfig | None = None,
    score_config: GiesScoreConfig | None = None,
    backend: Backend = "rust",
) -> SearchResult: ...
def result_summary(result: SearchResult) -> Mapping[str, object]: ...
