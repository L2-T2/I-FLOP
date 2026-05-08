"""Multi-environment dataset model for I-FLOP algorithms."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np


@dataclass(slots=True)
class MultiEnvDataset:
    """Container for observational and interventional environments.

    `intervention_targets[e] == set()` marks an observational environment.
    Grouped interventions are represented by target sets with more than one node;
    ungrouped interventions are represented by singleton target sets.
    """

    env_data: dict[int, np.ndarray]
    intervention_targets: dict[int, set[int]]
    variable_names: list[str] | None = None
    true_dag: np.ndarray | None = None
    metadata: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        self.env_data = {int(env): np.asarray(data, dtype=float) for env, data in self.env_data.items()}
        self.intervention_targets = {
            int(env): {int(node) for node in targets}
            for env, targets in self.intervention_targets.items()
        }
        self.metadata = dict(self.metadata or {})
        self._validate()
        if self.variable_names is None:
            self.variable_names = [f"X{node}" for node in range(self.num_vars)]
        else:
            self.variable_names = [str(name) for name in self.variable_names]
            if len(self.variable_names) != self.num_vars:
                raise ValueError("variable_names length must match the number of columns.")
        if self.true_dag is not None:
            self.true_dag = np.asarray(self.true_dag, dtype=int)
            if self.true_dag.shape != (self.num_vars, self.num_vars):
                raise ValueError("true_dag must be square with shape (num_vars, num_vars).")

    def _validate(self) -> None:
        if not self.env_data:
            raise ValueError("env_data must contain at least one environment.")
        if set(self.env_data) != set(self.intervention_targets):
            raise ValueError("env_data and intervention_targets must have identical environment ids.")
        widths: set[int] = set()
        for env, matrix in self.env_data.items():
            if matrix.ndim != 2:
                raise ValueError(f"environment {env} must be a two-dimensional matrix.")
            if matrix.shape[0] <= 0:
                raise ValueError(f"environment {env} must contain at least one row.")
            widths.add(int(matrix.shape[1]))
        if len(widths) != 1:
            raise ValueError("all environment matrices must share the same number of columns.")
        p = next(iter(widths))
        for env, targets in self.intervention_targets.items():
            invalid = [node for node in targets if node < 0 or node >= p]
            if invalid:
                raise ValueError(f"environment {env} has invalid intervention targets: {invalid}.")

    @property
    def env_ids(self) -> tuple[int, ...]:
        return tuple(sorted(self.env_data))

    @property
    def num_envs(self) -> int:
        return len(self.env_data)

    @property
    def num_vars(self) -> int:
        first = next(iter(self.env_data.values()))
        return int(first.shape[1])

    @property
    def sample_sizes(self) -> dict[int, int]:
        return {env: int(data.shape[0]) for env, data in self.env_data.items()}

    @property
    def total_samples(self) -> int:
        return int(sum(self.sample_sizes.values()))

    @property
    def observational_envs(self) -> tuple[int, ...]:
        return tuple(env for env in self.env_ids if not self.intervention_targets[env])

    @property
    def interventional_envs(self) -> tuple[int, ...]:
        return tuple(env for env in self.env_ids if self.intervention_targets[env])

    def effective_envs(self, node: int) -> tuple[int, ...]:
        """Return environments where `node` is not directly intervened upon."""

        node = int(node)
        return tuple(env for env in self.env_ids if node not in self.intervention_targets[env])

    def observational_only(self) -> "MultiEnvDataset":
        envs = {env: self.env_data[env].copy() for env in self.observational_envs}
        if not envs:
            raise ValueError("observational_only requires at least one observational environment.")
        return MultiEnvDataset(
            env_data=envs,
            intervention_targets={env: set() for env in envs},
            variable_names=list(self.variable_names or []),
            true_dag=self.true_dag,
            metadata={**self.metadata, "data_view": "observational_only"},
        )

    def pooled_as_observational(self, env_id: int = 0) -> "MultiEnvDataset":
        pooled = np.vstack([self.env_data[env] for env in self.env_ids])
        return MultiEnvDataset(
            env_data={int(env_id): pooled},
            intervention_targets={int(env_id): set()},
            variable_names=list(self.variable_names or []),
            true_dag=self.true_dag,
            metadata={**self.metadata, "data_view": "pooled_as_observational"},
        )

    def copy(self) -> "MultiEnvDataset":
        return MultiEnvDataset(
            env_data={env: data.copy() for env, data in self.env_data.items()},
            intervention_targets={env: set(targets) for env, targets in self.intervention_targets.items()},
            variable_names=list(self.variable_names or []),
            true_dag=None if self.true_dag is None else self.true_dag.copy(),
            metadata=dict(self.metadata),
        )
