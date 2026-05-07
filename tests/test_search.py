from __future__ import annotations

from iflop_final import available_scores, run_iflop
from iflop_final.data.simulation import generate_linear_gaussian_dataset


def test_available_scores_are_final_only() -> None:
    keys = set(available_scores())
    assert keys == {"flop_obs", "flop_envwise", "i_flop_envwise"}
    legacy = {"mIC", "final_bic", "gir_bic", "srp_bic", "flop" + "_pooled"}
    assert not legacy & keys


def test_run_iflop_dispatch() -> None:
    dataset = generate_linear_gaussian_dataset(num_vars=4, samples_per_env=40, seed=51)
    result = run_iflop(dataset, score_key="i_flop_envwise", backend="python")
    assert result.adjacency.shape == (4, 4)
    assert result.score_key == "i_flop_envwise"
