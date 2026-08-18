"""Perturbation and iterated-local-search helpers."""

from __future__ import annotations

import math

import numpy as np


def default_perturbation_size(num_vars: int, mode: str = "round_ln_p") -> int:
    if mode == "round_ln_p":
        return max(1, int(round(math.log(max(int(num_vars), 2)))))
    if mode == "sqrt_p":
        return max(1, int(round(math.sqrt(max(int(num_vars), 1)))))
    if mode == "one":
        return 1
    raise ValueError(f"unknown dynamic_k_mode: {mode}")


def perturb_order(order: tuple[int, ...], rng: np.random.Generator, k: int) -> tuple[int, ...]:
    values = list(order)
    if len(values) <= 1:
        return tuple(values)
    for _ in range(max(int(k), 0)):
        source = int(rng.integers(0, len(values)))
        node = values.pop(source)
        dest = int(rng.integers(0, len(values) + 1))
        values.insert(dest, node)
    return tuple(values)
