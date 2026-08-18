"""Release-surface regression tests for the canonical I-FLOP package."""

from __future__ import annotations

import iflop
from iflop.graph import conversions
from iflop.graph.cpdag import dag_to_cpdag


def test_public_api_contains_only_canonical_entry_points() -> None:
    assert iflop.__all__ == [
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
    assert conversions.__all__ == [
        "adjacency_from_parents",
        "dag_skeleton",
        "dag_to_cpdag",
        "parents_from_adjacency",
    ]
    assert conversions.dag_to_cpdag is dag_to_cpdag
