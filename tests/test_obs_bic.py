from __future__ import annotations

import numpy as np

from iflop_final import run_flop_envwise, run_flop_obs
from iflop_final.data.simulation import generate_linear_gaussian_dataset
from iflop_final.score.obs_bic import FlopEnvwiseScorer, ObsBICScorer


def test_flop_obs_runs() -> None:
    dataset = generate_linear_gaussian_dataset(num_vars=4, samples_per_env=50, seed=21)
    result = run_flop_obs(dataset, backend="python")
    assert result.adjacency.shape == (4, 4)
    assert sorted(result.order) == [0, 1, 2, 3]
    assert result.score_key == "flop_obs"
    assert isinstance(result.total_score, float)


def test_flop_envwise_runs_under_final_name() -> None:
    dataset = generate_linear_gaussian_dataset(num_vars=4, samples_per_env=40, num_interventions=3, seed=23)
    result = run_flop_envwise(dataset, backend="python")
    assert result.adjacency.shape == (4, 4)
    assert result.score_key == "flop_envwise"
    assert result.score_metadata["target_filtering"] == "none"


def test_flop_envwise_single_env_degenerates_to_obs_bic() -> None:
    dataset = generate_linear_gaussian_dataset(num_vars=4, samples_per_env=50, num_interventions=0, seed=24)
    obs = ObsBICScorer(dataset.observational_only())
    envwise = FlopEnvwiseScorer(dataset)
    for node in range(dataset.num_vars):
        parents = tuple(parent for parent in range(dataset.num_vars) if parent != node)[:2]
        assert np.isclose(
            envwise.local_score(node, parents),
            obs.local_score(node, parents),
            rtol=1.0e-9,
            atol=1.0e-9,
        )


def test_flop_envwise_uses_all_envs_without_target_filtering() -> None:
    dataset = generate_linear_gaussian_dataset(num_vars=4, samples_per_env=30, num_interventions=3, seed=25)
    scorer = FlopEnvwiseScorer(dataset)
    diag = scorer.local_diagnostics(0, parents=(1,))
    assert len(diag["per_env"]) == dataset.num_envs
    assert tuple(item["env"] for item in diag["per_env"]) == dataset.env_ids
    assert diag["target_filtering"] == "none"
    assert diag["coefficient_pooling"] == "pooled_covariance_across_all_environments"
    assert isinstance(diag["sigma2"], float)
