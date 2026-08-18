"""I-FLOP search with intervention-aware local scores."""

from __future__ import annotations

from iflop.config import SearchConfig
from iflop.score.iflop_bic import IFlopBICScorer
from iflop.search.flop_search import run_decomposable_order_search
from iflop.search.state import SearchResult


def run_iflop_search(scorer: IFlopBICScorer, *, config: SearchConfig | None = None) -> SearchResult:
    metadata: dict[str, object] = {
        "method": "I-FLOP",
        "penalty_sample_mode": scorer.config.penalty_sample_mode,
        "residual_mode": scorer.config.residual_mode,
        "intervention_filtering": "node-wise effective environments",
    }
    return run_decomposable_order_search(scorer, score_key=scorer.score_key, config=config, metadata=metadata)
