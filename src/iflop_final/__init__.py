"""Self-contained I-FLOP algorithm package."""

from iflop_final.api import (
    available_scores,
    run_flop_envwise,
    run_flop_obs,
    run_iflop,
    run_iflop_envwise,
)
from iflop_final.data.dataset import MultiEnvDataset

__all__ = [
    "MultiEnvDataset",
    "available_scores",
    "run_flop_envwise",
    "run_flop_obs",
    "run_iflop",
    "run_iflop_envwise",
]
