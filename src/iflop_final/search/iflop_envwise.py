"""I-FLOP-envwise search with intervention-aware local scores."""

from __future__ import annotations

from iflop_final.config import SearchConfig
from iflop_final.score.gies_bic import GiesBICScorer
from iflop_final.search.flop_search import run_decomposable_order_search
from iflop_final.search.state import SearchResult


def run_iflop_envwise_search(scorer: GiesBICScorer, *, config: SearchConfig | None = None) -> SearchResult:
    metadata = {
        "method": "I-FLOP-envwise",
        "variant": scorer.config.variant,
        "penalty_sample_mode": scorer.config.penalty_sample_mode,
        "intervention_filtering": "node-wise effective environments",
    }
    return run_decomposable_order_search(scorer, score_key=scorer.score_key, config=config, metadata=metadata)
