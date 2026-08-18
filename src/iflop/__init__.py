"""Self-contained I-FLOP algorithm package."""

from iflop.api import (
    result_summary,
    run_flop_envwise,
    run_flop_obs,
    run_iflop,
)
from iflop.config import IFlopScoreConfig, SearchConfig
from iflop.data.dataset import MultiEnvDataset
from iflop.score.catalog import available_scores
from iflop.search.state import SearchResult

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
