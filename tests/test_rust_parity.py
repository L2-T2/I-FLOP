from __future__ import annotations

import numpy as np

from iflop_final.config import SearchConfig
from iflop_final.data.simulation import generate_linear_gaussian_dataset
from iflop_final.runtime.native import (
    NATIVE_AVAILABLE,
    SUPPORTED_NATIVE_SCORE_KEYS,
    build_native_backend,
    run_native_iflop,
)


def test_rust_native_backend_builds_and_reports_supported_keys() -> None:
    assert NATIVE_AVAILABLE
    binary = build_native_backend()
    assert binary.exists()
    assert set(SUPPORTED_NATIVE_SCORE_KEYS) == {
        "flop_obs",
        "flop_envwise",
        "i_flop_envwise",
    }


def test_rust_native_search_smoke_for_final_mainlines() -> None:
    assert NATIVE_AVAILABLE
    dataset = generate_linear_gaussian_dataset(num_vars=4, samples_per_env=70, num_interventions=2, seed=151)
    search_config = SearchConfig(ils_restarts=0, max_sweeps=1, random_seed=0)

    for score_key in ("flop_obs", "flop_envwise", "i_flop_envwise"):
        result = run_native_iflop(dataset, score_key=score_key, search_config=search_config)
        assert result.score_key == score_key
        assert result.adjacency.shape == (dataset.num_vars, dataset.num_vars)
        assert sorted(result.order) == list(range(dataset.num_vars))
        assert np.isfinite(result.total_score)
