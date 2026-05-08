from __future__ import annotations

import _path_setup  # noqa: F401

from iflop_final import run_iflop_envwise
from iflop_final.data.simulation import generate_linear_gaussian_dataset


def test_iflop_envwise_runs() -> None:
    dataset = generate_linear_gaussian_dataset(num_vars=4, samples_per_env=50, num_interventions=2, seed=31)
    result = run_iflop_envwise(dataset, backend="python")
    assert result.adjacency.shape == (4, 4)
    assert result.score_key == "i_flop_envwise"
    assert result.score_metadata["adjacency_type"] == "i_cpdag"
    assert result.score_metadata["method"] == "I-FLOP-envwise"
    assert "local_diagnostics" in result.score_metadata
