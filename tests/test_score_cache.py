from __future__ import annotations

import _path_setup  # noqa: F401

import numpy as np

from iflop.data.simulation import generate_linear_gaussian_dataset
from iflop.score.iflop_bic import IFlopBICScorer
from iflop.score.obs_bic import FlopEnvwiseScorer


def test_local_score_cache_returns_same_value_as_uncached() -> None:
    dataset = generate_linear_gaussian_dataset(num_vars=4, samples_per_env=40, num_interventions=2, seed=401)
    scorer = IFlopBICScorer(dataset)
    parents = (0, 2)

    cached = scorer.local_score(3, parents)
    assert scorer.cache.misses == 1
    repeated = scorer.local_score(3, reversed(parents))
    assert scorer.cache.hits == 1

    scorer.cache.clear()
    uncached = scorer.local_score(3, parents)
    assert np.isclose(cached, repeated)
    assert np.isclose(cached, uncached)


def test_precomputed_envwise_scatter_matches_recomputed_scatter() -> None:
    dataset = generate_linear_gaussian_dataset(num_vars=4, samples_per_env=35, num_interventions=2, seed=402)
    scorer = FlopEnvwiseScorer(dataset)
    recomputed = sum(
        (dataset.env_data[env] - dataset.env_data[env].mean(axis=0, keepdims=True)).T
        @ (dataset.env_data[env] - dataset.env_data[env].mean(axis=0, keepdims=True))
        for env in dataset.env_ids
    )
    assert np.allclose(scorer.pooled_scatter, recomputed)
    assert scorer.scatter_construction_count == dataset.num_envs
