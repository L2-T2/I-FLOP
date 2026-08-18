"""Small configuration objects for I-FLOP runs."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True, slots=True)
class SearchConfig:
    ils_restarts: int = 2
    perturbation_size: int | None = None
    dynamic_k_mode: str = "round_ln_p"
    random_seed: int = 0
    max_sweeps: int | None = None
    atol: float = 1.0e-10


@dataclass(frozen=True, slots=True)
class IFlopScoreConfig:
    eps: float = 1.0e-8
    penalty_sample_mode: str = "total"
    fit_weight: float = 1.0
    penalty_weight: float = 1.0
    residual_mode: str = "pooled_covariance"

    def __post_init__(self) -> None:
        if self.eps <= 0:
            raise ValueError("eps must be positive")
        if self.penalty_sample_mode not in {"total", "effective", "max_env"}:
            raise ValueError("penalty_sample_mode must be 'total', 'effective', or 'max_env'")
        if self.residual_mode not in {"pooled_covariance", "env_residuals"}:
            raise ValueError("residual_mode must be 'pooled_covariance' or 'env_residuals'")
