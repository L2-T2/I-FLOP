from __future__ import annotations

import _path_setup  # noqa: F401

import numpy as np

from iflop_final.data.dataset import MultiEnvDataset


def test_dataset_validation() -> None:
    dataset = MultiEnvDataset(
        env_data={
            0: np.ones((5, 3)),
            1: np.ones((7, 3)) * 2.0,
            2: np.ones((9, 3)) * 3.0,
        },
        intervention_targets={0: set(), 1: {1}, 2: {0, 2}},
        variable_names=["a", "b", "c"],
    )
    assert dataset.sample_sizes == {0: 5, 1: 7, 2: 9}
    assert dataset.intervention_targets[2] == {0, 2}
    assert dataset.effective_envs(1) == (0, 2)
    assert dataset.effective_envs(0) == (0, 1)
    assert dataset.pooled_as_observational().total_samples == 21
