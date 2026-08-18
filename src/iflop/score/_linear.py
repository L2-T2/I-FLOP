"""Numerical helpers for local linear-Gaussian scores."""

from __future__ import annotations

import collections.abc as _abc

import numpy as np


def parent_tuple(parents: _abc.Iterable[int]) -> tuple[int, ...]:
    return tuple(sorted({int(parent) for parent in parents}))


def centered(data: np.ndarray) -> np.ndarray:
    arr = np.asarray(data, dtype=float)
    return arr - arr.mean(axis=0, keepdims=True)


def residual_variance_ols(
    data: np.ndarray,
    node: int,
    parents: _abc.Iterable[int],
    eps: float,
) -> tuple[float, float]:
    arr = np.asarray(data, dtype=float)
    y = arr[:, int(node)]
    pset = parent_tuple(parents)
    if pset:
        design = np.column_stack([np.ones(arr.shape[0]), arr[:, pset]])
    else:
        design = np.ones((arr.shape[0], 1), dtype=float)
    beta, *_ = np.linalg.lstsq(design, y, rcond=None)
    residual = y - design @ beta
    rss = float(residual @ residual)
    sigma2 = max(rss / max(arr.shape[0], 1), float(eps))
    return sigma2, rss


def centered_scatter(data: np.ndarray) -> np.ndarray:
    z = centered(data)
    return z.T @ z


def residual_variance_scatter(
    scatter: np.ndarray,
    n_samples: int,
    node: int,
    parents: _abc.Iterable[int],
    eps: float,
) -> float:
    node = int(node)
    pset = parent_tuple(parents)
    n = max(int(n_samples), 1)
    s_yy = float(scatter[node, node])
    if not pset:
        return max(s_yy / n, float(eps))
    s_xx = np.asarray(scatter[np.ix_(pset, pset)], dtype=float)
    s_xy = np.asarray(scatter[np.ix_(pset, [node])], dtype=float).reshape(len(pset))
    try:
        beta = np.linalg.solve(s_xx, s_xy)
    except np.linalg.LinAlgError:
        beta = np.linalg.pinv(s_xx) @ s_xy
    rss = s_yy - float(s_xy @ beta)
    return max(rss / n, float(eps))


def residual_variance_cholesky(
    scatter: np.ndarray,
    n_samples: int,
    node: int,
    parents: _abc.Iterable[int],
    eps: float,
) -> float:
    pset = parent_tuple(parents)
    if not pset:
        return residual_variance_scatter(scatter, n_samples, node, pset, eps)
    cov_pp = np.asarray(scatter[np.ix_(pset, pset)], dtype=float)
    cov_py = np.asarray(scatter[np.ix_(pset, [int(node)])], dtype=float).reshape(len(pset))
    s_yy = float(scatter[int(node), int(node)])
    ridge = float(eps)
    try:
        chol = np.linalg.cholesky(cov_pp + ridge * np.eye(len(pset)))
        tmp = np.linalg.solve(chol, cov_py)
        beta_term = float(tmp @ tmp)
        rss = s_yy - beta_term
        return max(rss / max(int(n_samples), 1), float(eps))
    except np.linalg.LinAlgError:
        return residual_variance_scatter(scatter, n_samples, node, pset, eps)


def bic_cost_terms(
    sigma2: float,
    n_fit: int,
    n_penalty: int,
    num_parents: int,
    *,
    fit_weight: float = 1.0,
    penalty_weight: float = 1.0,
) -> tuple[float, float, float]:
    n_fit_i = max(int(n_fit), 1)
    n_pen_i = max(int(n_penalty), 2)
    fit = float(fit_weight) * 0.5 * n_fit_i * (1.0 + np.log(float(sigma2)))
    penalty = float(penalty_weight) * 0.5 * np.log(n_pen_i) * (int(num_parents) + 1)
    return fit + penalty, fit, penalty
