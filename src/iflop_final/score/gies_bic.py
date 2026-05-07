"""GIES-style interventional Gaussian BIC scores embedded in I-FLOP search."""

from __future__ import annotations

from collections.abc import Iterable, Mapping

import numpy as np

from iflop_final.config import GiesScoreConfig
from iflop_final.data.dataset import MultiEnvDataset
from iflop_final.score._linear import (
    bic_cost_terms,
    centered_scatter,
    parent_tuple,
    residual_variance_scatter,
)
from iflop_final.score.cache import LocalScoreCache

GIES_VARIANTS = {"envwise"}


class GiesBICScorer:
    """Intervention-aware local BIC scorer.

    For node j, only environments where j is not directly intervened upon are
    used. The search object is still the I-FLOP order/parent-set state.
    """

    def __init__(self, dataset: MultiEnvDataset, config: GiesScoreConfig | None = None) -> None:
        self.dataset = dataset
        self.config = config or GiesScoreConfig()
        if self.config.variant not in GIES_VARIANTS:
            raise ValueError(f"unknown GIES-style variant: {self.config.variant}")
        self.variant = "envwise"
        self.score_key = "i_flop_envwise"
        self.cache = LocalScoreCache()

    @property
    def num_vars(self) -> int:
        return self.dataset.num_vars

    def _effective_arrays(self, node: int) -> list[tuple[int, np.ndarray]]:
        envs = self.dataset.effective_envs(int(node))
        if not envs:
            raise ValueError(f"node {node} has no effective environments.")
        return [(env, self.dataset.env_data[env]) for env in envs]

    def _effective_n(self, node: int) -> int:
        return int(sum(data.shape[0] for _env, data in self._effective_arrays(node)))

    def _penalty_n(self, node: int) -> int:
        mode = self.config.penalty_sample_mode
        arrays = self._effective_arrays(node)
        if mode == "total":
            return self.dataset.total_samples
        if mode == "effective":
            return int(sum(data.shape[0] for _env, data in arrays))
        if mode == "max_env":
            return int(max(data.shape[0] for _env, data in arrays))
        raise ValueError(f"unknown penalty_sample_mode: {mode}")

    def local_score(self, node: int, parents: Iterable[int]) -> float:
        pset = parent_tuple(parents)
        return self.cache.get_or_compute(
            self.score_key,
            int(node),
            pset,
            lambda: float(self.local_diagnostics(int(node), pset)["score"]),
        )

    def local_diagnostics(self, node: int, parents: Iterable[int]) -> dict[str, object]:
        pset = parent_tuple(parents)
        return self._envwise_diagnostics(int(node), pset)

    def total_score(self, parents: Mapping[int, Iterable[int]]) -> float:
        return float(sum(self.local_score(node, parents.get(node, ())) for node in range(self.num_vars)))

    def _envwise_diagnostics(self, node: int, parents: tuple[int, ...]) -> dict[str, object]:
        arrays = self._effective_arrays(node)
        pooled_scatter = np.zeros((self.num_vars, self.num_vars), dtype=float)
        n_fit = 0
        for _env, data in arrays:
            pooled_scatter += centered_scatter(data)
            n_fit += int(data.shape[0])
        if self.config.envwise_residual_mode == "pooled_covariance":
            sigma2 = residual_variance_scatter(pooled_scatter, n_fit, node, parents, self.config.eps)
            n_penalty = self._penalty_n(node)
            total, fit, penalty = bic_cost_terms(
                sigma2,
                n_fit,
                n_penalty,
                len(parents),
                fit_weight=self.config.fit_weight,
                penalty_weight=self.config.penalty_weight,
            )
            return {
                "score": total,
                "fit_term": float(fit),
                "penalty": float(penalty),
                "sigma2": float(sigma2),
                "n_fit": n_fit,
                "n_penalty": n_penalty,
                "parents": parents,
                "effective_envs": self.dataset.effective_envs(node),
                "variant": self.config.variant,
                "coefficient_pooling": "pooled_covariance_across_effective_environments",
                "per_env": [{"env": int(env), "n": int(data.shape[0])} for env, data in arrays],
            }
        if self.config.envwise_residual_mode != "env_residuals":
            raise ValueError(f"unknown envwise_residual_mode: {self.config.envwise_residual_mode}")
        if parents:
            s_xx = np.asarray(pooled_scatter[np.ix_(parents, parents)], dtype=float)
            s_xy = np.asarray(pooled_scatter[np.ix_(parents, [node])], dtype=float).reshape(len(parents))
            try:
                beta = np.linalg.solve(s_xx, s_xy)
            except np.linalg.LinAlgError:
                beta = np.linalg.pinv(s_xx) @ s_xy
        else:
            beta = np.asarray([], dtype=float)

        fit = 0.0
        per_env: list[dict[str, float | int]] = []
        for env, data in arrays:
            scatter = centered_scatter(data)
            n_env = int(data.shape[0])
            s_yy = float(scatter[node, node])
            if parents:
                s_xx_env = np.asarray(scatter[np.ix_(parents, parents)], dtype=float)
                s_xy_env = np.asarray(scatter[np.ix_(parents, [node])], dtype=float).reshape(len(parents))
                rss = s_yy - 2.0 * float(beta @ s_xy_env) + float(beta @ s_xx_env @ beta)
            else:
                rss = s_yy
            sigma2 = max(rss / max(n_env, 1), self.config.eps)
            env_fit = self.config.fit_weight * 0.5 * int(data.shape[0]) * (1.0 + np.log(sigma2))
            fit += float(env_fit)
            n_fit += n_env
            per_env.append({"env": int(env), "n": int(data.shape[0]), "sigma2": float(sigma2), "fit_term": float(env_fit)})
        n_penalty = self._penalty_n(node)
        penalty = self.config.penalty_weight * 0.5 * np.log(max(n_penalty, 2)) * (len(parents) + 1)
        total = float(fit + penalty)
        return {
            "score": total,
            "fit_term": float(fit),
            "penalty": float(penalty),
            "sigma2": None,
            "n_fit": n_fit,
            "n_penalty": n_penalty,
            "parents": parents,
            "effective_envs": self.dataset.effective_envs(node),
            "variant": self.config.variant,
            "coefficient_pooling": "shared_across_effective_environments",
            "per_env": per_env,
        }
