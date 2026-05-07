"""Score catalog for the final public API."""

from __future__ import annotations

from iflop_final.config import GiesScoreConfig
from iflop_final.data.dataset import MultiEnvDataset
from iflop_final.score.gies_bic import GiesBICScorer
from iflop_final.score.obs_bic import FlopEnvwiseScorer, ObsBICScorer

FINAL_SCORE_KEYS = (
    "flop_obs",
    "flop_envwise",
    "i_flop_envwise",
)


def available_scores() -> tuple[str, ...]:
    return FINAL_SCORE_KEYS


def make_scorer(dataset: MultiEnvDataset, score_key: str):
    key = str(score_key)
    if key == "flop_obs":
        return ObsBICScorer(dataset.observational_only())
    if key == "flop_envwise":
        return FlopEnvwiseScorer(dataset)
    if key == "i_flop_envwise":
        return GiesBICScorer(dataset, GiesScoreConfig(variant="envwise"))
    raise ValueError(f"unsupported final score key: {score_key}")
