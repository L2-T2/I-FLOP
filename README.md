# Final I-FLOP Algorithm Package

This directory is the release package for the final FLOP-aligned and
I-FLOP-envwise methods. It follows the upstream FLOP packaging boundary: the
release-facing Python API calls a Rust native algorithm core by default, while a
small Python reference backend remains available for parity checks.

## Supported Methods

- `flop_obs`: FLOP-aligned order search using observational environments only.
- `flop_envwise`: FLOP-aligned order search using all environments and no
  intervention-target filtering.
- `i_flop_envwise`: the final intervention-aware I-FLOP method using
  node-specific effective environments.

Legacy aliases and exploratory score families are not part of this release.
The package intentionally does not ship an R interface.

## Installation From Source

Requirements:

- Python 3.10 or newer
- NumPy 1.24 or newer
- Rust toolchain with Cargo for the default native backend

```bash
cd final-algorithm
python -m pip install -e ".[test]"
```

## Quick Smoke Tests

```bash
python -m pytest -q
cd rust/iflop_native
cargo test
```

## Minimal Python API

```python
from iflop_final import run_flop_envwise, run_flop_obs, run_iflop_envwise
from iflop_final.data.simulation import generate_linear_gaussian_dataset

dataset = generate_linear_gaussian_dataset(num_vars=5, samples_per_env=100, seed=7)

obs_result = run_flop_obs(dataset)
envwise_result = run_flop_envwise(dataset)
iflop_result = run_iflop_envwise(dataset)
```

Each function accepts `backend="rust"`, `backend="python"`, or
`backend="auto"`. The default is `backend="rust"` so release users exercise the
native core. The Python backend is intended for debugging and score-parity
tests.

## Examples

```bash
python examples/run_flop_obs.py
python examples/run_flop_envwise.py
python examples/run_iflop_envwise.py
python -m iflop_final.cli.run --method i_flop_envwise --backend auto
```

## Output

All public methods return `SearchResult` with:

- `adjacency`: learned adjacency matrix.
- `order`: variable order used by the search.
- `parents`: parent sets by node.
- `total_score`: minimized scalar score.
- `score_key`: one of `flop_obs`, `flop_envwise`, `i_flop_envwise`.
- `score_metadata`: score diagnostics and backend metadata.

The package estimates DAGs directly. CPDAG helpers are only used internally for
native parity metadata.

## Release Notes

- License: MPL-2.0.
- Python type marker and public API stub are included.
- R bindings are out of scope for this release.
- Generated artifacts such as `target/`, `.pytest_cache/`, `__pycache__/`, and
  local reports are not part of the release tree.
