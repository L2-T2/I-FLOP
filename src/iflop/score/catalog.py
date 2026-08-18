"""Score catalog for the public API."""

from __future__ import annotations

from iflop.config import IFlopScoreConfig
from iflop.data.dataset import MultiEnvDataset
from iflop.score.iflop_bic import IFlopBICScorer
from iflop.score.obs_bic import FlopEnvwiseScorer, ObsBICScorer

SCORE_KEYS = (
    "flop_obs",
    "flop_envwise",
    "iflop",
)


def available_scores() -> tuple[str, ...]:
    return SCORE_KEYS


def make_scorer(
    dataset: MultiEnvDataset,
    score_key: str,
) -> ObsBICScorer | FlopEnvwiseScorer | IFlopBICScorer:
    key = str(score_key)
    if key == "flop_obs":
        return ObsBICScorer(dataset.observational_only())
    if key == "flop_envwise":
        return FlopEnvwiseScorer(dataset)
    if key == "iflop":
        return IFlopBICScorer(dataset, IFlopScoreConfig())
    raise ValueError(f"unsupported score key: {score_key}")
