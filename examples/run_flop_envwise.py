from __future__ import annotations

from iflop_final import run_flop_envwise
from iflop_final.api import result_summary
from iflop_final.data.simulation import generate_linear_gaussian_dataset


dataset = generate_linear_gaussian_dataset(num_vars=5, samples_per_env=80, seed=12)
result = run_flop_envwise(dataset, backend="auto")
print(result_summary(result))
