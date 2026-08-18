from __future__ import annotations

import _path_setup  # noqa: F401

import numpy as np

from iflop.config import SearchConfig
from iflop.data.simulation import generate_linear_gaussian_dataset
from iflop.runtime.rust import (
    RUST_AVAILABLE,
    SUPPORTED_RUST_SCORE_KEYS,
    build_rust_backend,
    run_rust,
)


def test_rust_backend_builds_and_reports_supported_keys() -> None:
    assert RUST_AVAILABLE
    binary = build_rust_backend()
    assert binary.exists()
    assert set(SUPPORTED_RUST_SCORE_KEYS) == {
        "flop_obs",
        "flop_envwise",
        "iflop",
    }


def test_rust_search_smoke_for_supported_methods() -> None:
    assert RUST_AVAILABLE
    dataset = generate_linear_gaussian_dataset(num_vars=4, samples_per_env=70, num_interventions=2, seed=151)
    search_config = SearchConfig(ils_restarts=0, max_sweeps=1, random_seed=0)

    for score_key in ("flop_obs", "flop_envwise", "iflop"):
        result = run_rust(dataset, score_key=score_key, search_config=search_config)
        assert result.score_key == score_key
        assert result.adjacency.shape == (dataset.num_vars, dataset.num_vars)
        assert sorted(result.order) == list(range(dataset.num_vars))
        assert np.isfinite(result.total_score)
