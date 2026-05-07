"""FLOP-aligned Gaussian BIC baseline scores."""

from __future__ import annotations

from collections.abc import Iterable, Mapping

import numpy as np

from iflop_final.data.dataset import MultiEnvDataset
from iflop_final.score._linear import (
    bic_cost_terms,
    centered_scatter,
    parent_tuple,
    residual_variance_ols,
    residual_variance_scatter,
)
from iflop_final.score.cache import LocalScoreCache


class ObsBICScorer:
    """Decomposable BIC score for observational FLOP-aligned baselines."""

    score_key = "obs_bic_only"

    def __init__(self, dataset: MultiEnvDataset, *, eps: float = 1.0e-8) -> None:
        self.dataset = dataset
        self.eps = float(eps)
        self.data = np.vstack([dataset.env_data[env] for env in dataset.env_ids])
        self.cache = LocalScoreCache()

    @property
    def num_vars(self) -> int:
        return int(self.data.shape[1])

    def local_score(self, node: int, parents: Iterable[int]) -> float:
        pset = parent_tuple(parents)

        def compute() -> float:
            sigma2, _rss = residual_variance_ols(self.data, int(node), pset, self.eps)
            total, _fit, _penalty = bic_cost_terms(sigma2, self.data.shape[0], self.data.shape[0], len(pset))
            return total

        return self.cache.get_or_compute(self.score_key, int(node), pset, compute)

    def local_diagnostics(self, node: int, parents: Iterable[int]) -> dict[str, object]:
        pset = parent_tuple(parents)
        sigma2, rss = residual_variance_ols(self.data, int(node), pset, self.eps)
        total, fit, penalty = bic_cost_terms(sigma2, self.data.shape[0], self.data.shape[0], len(pset))
        return {
            "score": total,
            "fit_term": fit,
            "penalty": penalty,
            "sigma2": sigma2,
            "rss": rss,
            "n_fit": int(self.data.shape[0]),
            "n_penalty": int(self.data.shape[0]),
            "parents": pset,
        }

    def total_score(self, parents: Mapping[int, Iterable[int]]) -> float:
        return float(sum(self.local_score(node, parents.get(node, ())) for node in range(self.num_vars)))


class FlopEnvwiseScorer:
    """FLOP-aligned envwise BIC score that ignores intervention targets.

    This baseline keeps the FLOP order / prefix-constrained parent-set search
    interpretation, but scores multi-environment data by pooling centered
    environment scatter matrices before computing one Gaussian residual
    variance. Unlike GIES-envwise, all environments are valid for every node:
    intervention targets are not used for local target filtering.
    """

    score_key = "flop_envwise"

    def __init__(self, dataset: MultiEnvDataset, *, eps: float = 1.0e-8) -> None:
        self.dataset = dataset
        self.eps = float(eps)
        self.cache = LocalScoreCache()

    @property
    def num_vars(self) -> int:
        return int(self.dataset.num_vars)

    @property
    def n_penalty(self) -> int:
        return int(self.dataset.total_samples)

    def _pooled_terms(self, node: int, parents: tuple[int, ...]) -> tuple[float, int, list[dict[str, object]]]:
        pooled_scatter = np.zeros((self.num_vars, self.num_vars), dtype=float)
        n_fit = 0
        per_env: list[dict[str, object]] = []
        for env in self.dataset.env_ids:
            data = self.dataset.env_data[env]
            n_env = int(data.shape[0])
            pooled_scatter += centered_scatter(data)
            n_fit += n_env
            per_env.append(
                {
                    "env": int(env),
                    "n": n_env,
                    "targets": tuple(sorted(self.dataset.intervention_targets[env])),
                    "included": True,
                    "inclusion_rule": "all environments; FLOP baseline ignores intervention targets",
                }
            )
        sigma2 = residual_variance_scatter(pooled_scatter, n_fit, int(node), parents, self.eps)
        return float(sigma2), int(n_fit), per_env

    def local_score(self, node: int, parents: Iterable[int]) -> float:
        pset = parent_tuple(parents)

        def compute() -> float:
            sigma2, n_fit, _per_env = self._pooled_terms(int(node), pset)
            total, _fit, _penalty = bic_cost_terms(sigma2, n_fit, self.n_penalty, len(pset))
            return float(total)

        return self.cache.get_or_compute(self.score_key, int(node), pset, compute)

    def local_diagnostics(self, node: int, parents: Iterable[int]) -> dict[str, object]:
        pset = parent_tuple(parents)
        sigma2, n_fit, per_env = self._pooled_terms(int(node), pset)
        total, fit, penalty = bic_cost_terms(sigma2, n_fit, self.n_penalty, len(pset))
        return {
            "score": float(total),
            "fit_term": float(fit),
            "penalty": float(penalty),
            "sigma2": float(sigma2),
            "n_fit": int(n_fit),
            "n_penalty": self.n_penalty,
            "parents": pset,
            "per_env": per_env,
            "score_family": "flop_envwise",
            "target_filtering": "none",
            "coefficient_pooling": "pooled_covariance_across_all_environments",
        }

    def total_score(self, parents: Mapping[int, Iterable[int]]) -> float:
        return float(sum(self.local_score(node, parents.get(node, ())) for node in range(self.num_vars)))
