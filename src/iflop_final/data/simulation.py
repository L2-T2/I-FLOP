"""Synthetic multi-environment generators for I-FLOP experiments."""

from __future__ import annotations

import numpy as np

from iflop_final.data.dataset import MultiEnvDataset
from iflop_final.graph.dag import topological_order


def _build_random_sem(
    *,
    num_vars: int,
    edge_prob: float,
    rng: np.random.Generator,
) -> tuple[np.ndarray, np.ndarray, list[int]]:
    p = int(num_vars)
    order = list(range(p))
    adjacency = np.zeros((p, p), dtype=int)
    weights = np.zeros((p, p), dtype=float)
    for i, parent in enumerate(order):
        for child in order[(i + 1) :]:
            if rng.random() < float(edge_prob):
                adjacency[parent, child] = 1
                sign = -1.0 if rng.random() < 0.5 else 1.0
                weights[parent, child] = sign * rng.uniform(0.4, 1.2)
    topo = topological_order(adjacency)
    return adjacency, weights, topo


def _singleton_targets(num_vars: int, num_interventions: int) -> dict[int, set[int]]:
    p = int(num_vars)
    targets: dict[int, set[int]] = {0: set()}
    for env in range(1, int(num_interventions) + 1):
        targets[env] = {int((env - 1) % p)}
    return targets


def _grouped_targets(num_vars: int, num_interventions: int, group_size: int) -> dict[int, set[int]]:
    p = int(num_vars)
    g = max(1, min(int(group_size), p))
    targets: dict[int, set[int]] = {0: set()}
    for env in range(1, int(num_interventions) + 1):
        start = int((env - 1) % p)
        targets[env] = {int((start + offset) % p) for offset in range(g)}
    return targets


def _residual_std_by_env(num_interventions: int, variance_ratio: float) -> dict[int, float]:
    ratio = max(float(variance_ratio), 1.0)
    stds: dict[int, float] = {0: 1.0}
    if int(num_interventions) <= 0:
        return stds
    if int(num_interventions) == 1:
        stds[1] = 1.0
        return stds

    # Keep the geometric center at 1.0 while changing the max/min variance ratio.
    exponents = np.linspace(-0.5, 0.5, int(num_interventions))
    variance_multipliers = ratio**exponents
    for env, variance_multiplier in enumerate(variance_multipliers, start=1):
        stds[int(env)] = float(np.sqrt(variance_multiplier))
    return stds


def _shift_strength_by_env(
    num_interventions: int,
    base_strength: float,
    shift_multiplier: float,
) -> dict[int, float]:
    multipliers = np.ones(int(num_interventions) + 1, dtype=float)
    if int(num_interventions) > 0:
        multipliers[1:] = np.linspace(1.0, max(float(shift_multiplier), 1.0), int(num_interventions))
    return {int(env): float(float(base_strength) * mult) for env, mult in enumerate(multipliers)}


def _sample_from_sem(
    *,
    adjacency: np.ndarray,
    weights: np.ndarray,
    topo: list[int],
    samples_per_env: int,
    targets: dict[int, set[int]],
    residual_std_by_env: dict[int, float],
    intervention_shift_by_env: dict[int, float],
    rng: np.random.Generator,
) -> dict[int, np.ndarray]:
    p = int(adjacency.shape[0])
    env_data: dict[int, np.ndarray] = {}
    for env in sorted(targets):
        data = np.zeros((int(samples_per_env), p), dtype=float)
        noise = rng.normal(size=data.shape)
        residual_std = float(residual_std_by_env.get(int(env), 1.0))
        shift = float(intervention_shift_by_env.get(int(env), 0.0))
        intervened = set(targets[int(env)])
        for node in topo:
            if intervened and node in intervened:
                data[:, node] = shift + residual_std * noise[:, node]
                continue
            parents = np.where(adjacency[:, node] != 0)[0]
            if len(parents):
                data[:, node] = data[:, parents] @ weights[parents, node] + residual_std * noise[:, node]
            else:
                data[:, node] = residual_std * noise[:, node]
        env_data[int(env)] = data
    return env_data


def generate_linear_gaussian_dataset(
    *,
    num_vars: int = 5,
    num_interventions: int = 3,
    samples_per_env: int = 200,
    edge_prob: float = 0.25,
    intervention_strength: float = 2.0,
    seed: int = 0,
) -> MultiEnvDataset:
    """Generate one observational environment plus singleton interventions."""

    rng = np.random.default_rng(int(seed))
    adjacency, weights, topo = _build_random_sem(num_vars=int(num_vars), edge_prob=float(edge_prob), rng=rng)
    targets = _singleton_targets(int(num_vars), int(num_interventions))
    env_data = _sample_from_sem(
        adjacency=adjacency,
        weights=weights,
        topo=topo,
        samples_per_env=int(samples_per_env),
        targets=targets,
        residual_std_by_env={env: 1.0 for env in range(int(num_interventions) + 1)},
        intervention_shift_by_env={env: 0.0 if env == 0 else float(intervention_strength) for env in range(int(num_interventions) + 1)},
        rng=rng,
    )
    return MultiEnvDataset(
        env_data=env_data,
        intervention_targets=targets,
        variable_names=[f"X{j}" for j in range(int(num_vars))],
        true_dag=adjacency,
        metadata={"generator": "linear_gaussian", "seed": int(seed)},
    )


def generate_heterogeneity_dataset(
    *,
    regime: str,
    heterogeneity_level: int | float,
    num_vars: int = 20,
    num_interventions: int = 5,
    samples_per_env: int = 1000,
    edge_prob: float = 0.2,
    intervention_strength: float = 2.5,
    seed: int = 0,
) -> MultiEnvDataset:
    """Generate the I-FLOP heterogeneity stress datasets.

    Regimes:
    - H0_homogeneous: baseline singleton interventions with homogeneous noise.
    - H1_env_specific_residual_variance: environment-specific residual variance,
      invariant graph and coefficients.
    - H2_covariate_distribution_shift: stronger target shifts across
      intervention environments, invariant graph and residual variance.
    - H3_grouped_multitarget_encoding: grouped hard interventions with target
      sets of the requested size.
    """

    rng = np.random.default_rng(int(seed))
    adjacency, weights, topo = _build_random_sem(num_vars=int(num_vars), edge_prob=float(edge_prob), rng=rng)
    regime_key = str(regime)
    level = float(heterogeneity_level)

    if regime_key == "H3_grouped_multitarget_encoding":
        targets = _grouped_targets(int(num_vars), int(num_interventions), int(round(level)))
    else:
        targets = _singleton_targets(int(num_vars), int(num_interventions))

    if regime_key == "H0_homogeneous":
        residual_std = {env: 1.0 for env in range(int(num_interventions) + 1)}
        shift_by_env = {env: 0.0 if env == 0 else float(intervention_strength) for env in range(int(num_interventions) + 1)}
    elif regime_key == "H1_env_specific_residual_variance":
        residual_std = _residual_std_by_env(int(num_interventions), level)
        shift_by_env = {env: 0.0 if env == 0 else float(intervention_strength) for env in range(int(num_interventions) + 1)}
    elif regime_key == "H2_covariate_distribution_shift":
        residual_std = {env: 1.0 for env in range(int(num_interventions) + 1)}
        shift_by_env = _shift_strength_by_env(int(num_interventions), float(intervention_strength), level)
        shift_by_env[0] = 0.0
    elif regime_key == "H3_grouped_multitarget_encoding":
        residual_std = {env: 1.0 for env in range(int(num_interventions) + 1)}
        shift_by_env = {env: 0.0 if env == 0 else float(intervention_strength) for env in range(int(num_interventions) + 1)}
    else:
        raise ValueError(f"unknown heterogeneity regime: {regime_key}")

    env_data = _sample_from_sem(
        adjacency=adjacency,
        weights=weights,
        topo=topo,
        samples_per_env=int(samples_per_env),
        targets=targets,
        residual_std_by_env=residual_std,
        intervention_shift_by_env=shift_by_env,
        rng=rng,
    )
    metadata = {
        "generator": "linear_gaussian_heterogeneity",
        "seed": int(seed),
        "heterogeneity_regime": regime_key,
        "heterogeneity_level": level,
        "residual_std_by_env": residual_std,
        "intervention_shift_by_env": shift_by_env,
    }
    return MultiEnvDataset(
        env_data=env_data,
        intervention_targets=targets,
        variable_names=[f"X{j}" for j in range(int(num_vars))],
        true_dag=adjacency,
        metadata=metadata,
    )
