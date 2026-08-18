from __future__ import annotations

from iflop import result_summary, run_flop_obs
from iflop.data.simulation import generate_linear_gaussian_dataset


dataset = generate_linear_gaussian_dataset(num_vars=5, samples_per_env=80, seed=11)
result = run_flop_obs(dataset, backend="auto")
print(result_summary(result))
