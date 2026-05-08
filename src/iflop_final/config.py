"""Small configuration objects for I-FLOP runs."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True, slots=True)
class SearchConfig:
    """Search hyperparameters shared by final algorithms."""

    ils_restarts: int = 2
    perturbation_size: int | None = None
    dynamic_k_mode: str = "round_ln_p"
    random_seed: int = 0
    max_sweeps: int | None = None
    atol: float = 1.0e-10


@dataclass(frozen=True, slots=True)
class GiesScoreConfig:
    """I-FLOP-envwise local BIC settings."""

    variant: str = "envwise"
    eps: float = 1.0e-8
    penalty_sample_mode: str = "total"
    fit_weight: float = 1.0
    penalty_weight: float = 1.0
    envwise_residual_mode: str = "pooled_covariance"
