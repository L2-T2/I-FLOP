from __future__ import annotations

from iflop import result_summary, run_iflop
from iflop.data.simulation import generate_linear_gaussian_dataset


dataset = generate_linear_gaussian_dataset(num_vars=5, samples_per_env=80, seed=13)
result = run_iflop(dataset, backend="auto")
print(result_summary(result))
