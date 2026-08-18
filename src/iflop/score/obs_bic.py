"""FLOP-aligned Gaussian BIC baseline scores."""

from __future__ import annotations

import collections.abc as _abc

import numpy as np

from iflop.data.dataset import MultiEnvDataset
from iflop.score._linear import (
    bic_cost_terms,
    centered_scatter,
    parent_tuple,
    residual_variance_scatter,
)
from iflop.score.cache import LocalScoreCache


class ObsBICScorer:
    score_key = "obs_bic_only"

    def __init__(self, dataset: MultiEnvDataset, *, eps: float = 1.0e-8) -> None:
        self.dataset = dataset
        self.eps = float(eps)
        self.data = np.vstack([dataset.env_data[env] for env in dataset.env_ids])
        self.scatter = centered_scatter(self.data)
        self.n_fit = int(self.data.shape[0])
        self.cache = LocalScoreCache()
        self.scatter_construction_count = 1

    @property
    def num_vars(self) -> int:
        return int(self.data.shape[1])

    def local_score(self, node: int, parents: _abc.Iterable[int]) -> float:
        pset = parent_tuple(parents)

        def compute() -> float:
            sigma2 = residual_variance_scatter(self.scatter, self.n_fit, int(node), pset, self.eps)
            total, _fit, _penalty = bic_cost_terms(sigma2, self.n_fit, self.n_fit, len(pset))
            return total

        return self.cache.get_or_compute(self.score_key, int(node), pset, compute)

    def local_diagnostics(self, node: int, parents: _abc.Iterable[int]) -> dict[str, object]:
        pset = parent_tuple(parents)
        sigma2 = residual_variance_scatter(self.scatter, self.n_fit, int(node), pset, self.eps)
        total, fit, penalty = bic_cost_terms(sigma2, self.n_fit, self.n_fit, len(pset))
        return {
            "score": total,
            "fit_term": fit,
            "penalty": penalty,
            "sigma2": sigma2,
            "rss": float(sigma2 * max(self.n_fit, 1)),
            "n_fit": self.n_fit,
            "n_penalty": self.n_fit,
            "parents": pset,
        }

    def total_score(
        self,
        parents: _abc.Mapping[int, _abc.Iterable[int]],
    ) -> float:
        return float(sum(self.local_score(node, parents.get(node, ())) for node in range(self.num_vars)))


class FlopEnvwiseScorer:
    score_key = "flop_envwise"

    def __init__(self, dataset: MultiEnvDataset, *, eps: float = 1.0e-8) -> None:
        self.dataset = dataset
        self.eps = float(eps)
        self.cache = LocalScoreCache()
        self.pooled_scatter = np.zeros((self.num_vars, self.num_vars), dtype=float)
        self.n_fit = 0
        self.per_env_terms: list[dict[str, object]] = []
        for env in self.dataset.env_ids:
            data = self.dataset.env_data[env]
            n_env = int(data.shape[0])
            self.pooled_scatter += centered_scatter(data)
            self.n_fit += n_env
            self.per_env_terms.append(
                {
                    "env": int(env),
                    "n": n_env,
                    "targets": tuple(sorted(self.dataset.intervention_targets[env])),
                    "included": True,
                    "inclusion_rule": "all environments; FLOP baseline ignores intervention targets",
                }
            )
        self.scatter_construction_count = len(self.dataset.env_ids)

    @property
    def num_vars(self) -> int:
        return int(self.dataset.num_vars)

    @property
    def n_penalty(self) -> int:
        return int(self.dataset.total_samples)

    def _pooled_terms(self, node: int, parents: tuple[int, ...]) -> tuple[float, int, list[dict[str, object]]]:
        sigma2 = residual_variance_scatter(self.pooled_scatter, self.n_fit, int(node), parents, self.eps)
        return float(sigma2), int(self.n_fit), list(self.per_env_terms)

    def local_score(self, node: int, parents: _abc.Iterable[int]) -> float:
        pset = parent_tuple(parents)

        def compute() -> float:
            sigma2, n_fit, _per_env = self._pooled_terms(int(node), pset)
            total, _fit, _penalty = bic_cost_terms(sigma2, n_fit, self.n_penalty, len(pset))
            return float(total)

        return self.cache.get_or_compute(self.score_key, int(node), pset, compute)

    def local_diagnostics(self, node: int, parents: _abc.Iterable[int]) -> dict[str, object]:
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

    def total_score(
        self,
        parents: _abc.Mapping[int, _abc.Iterable[int]],
    ) -> float:
        return float(sum(self.local_score(node, parents.get(node, ())) for node in range(self.num_vars)))
