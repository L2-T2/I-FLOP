"""Minimal CLI for examples and quick smoke runs."""

from __future__ import annotations

import argparse
import json

from iflop_final.api import result_summary, run_flop_envwise, run_flop_obs, run_iflop_envwise
from iflop_final.data.simulation import generate_linear_gaussian_dataset


def main() -> None:
    parser = argparse.ArgumentParser(description="Run a final I-FLOP smoke example.")
    parser.add_argument("--method", choices=["flop_obs", "flop_envwise", "i_flop_envwise"], default="i_flop_envwise")
    parser.add_argument("--backend", choices=["rust", "python", "auto"], default="rust")
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()
    dataset = generate_linear_gaussian_dataset(num_vars=5, samples_per_env=80, seed=args.seed)
    if args.method == "flop_obs":
        result = run_flop_obs(dataset, backend=args.backend)
    elif args.method == "flop_envwise":
        result = run_flop_envwise(dataset, backend=args.backend)
    else:
        result = run_iflop_envwise(dataset, backend=args.backend)
    print(json.dumps(result_summary(result), indent=2))


if __name__ == "__main__":
    main()
