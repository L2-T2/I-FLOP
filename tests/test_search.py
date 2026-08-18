from __future__ import annotations

import _path_setup  # noqa: F401

from iflop import available_scores, run_iflop
from iflop.data.simulation import generate_linear_gaussian_dataset


def test_available_scores_are_supported_only() -> None:
    keys = set(available_scores())
    assert keys == {"flop_obs", "flop_envwise", "iflop"}


def test_run_iflop() -> None:
    dataset = generate_linear_gaussian_dataset(num_vars=4, samples_per_env=40, seed=51)
    result = run_iflop(dataset, backend="python")
    assert result.adjacency.shape == (4, 4)
    assert result.score_key == "iflop"
