# I-FLOP Algorithm Package

This repository contains the release implementation of I-FLOP and its two FLOP-aligned comparison baselines. The Python API calls a Rust core by default, while a Python reference backend remains available for parity checks.

## Supported Methods

- `flop_obs`: FLOP-aligned order search using observational environments only.
  The public graph output is a CPDAG.
- `flop_envwise`: FLOP-aligned order search using all environments and no intervention-target filtering. The public graph output is a CPDAG.
- `iflop`: I-FLOP, using node-specific effective environments and returning an I-CPDAG.

The package intentionally does not ship an R interface.

## Installation From Source

Requirements:

- Python 3.10 or newer
- NumPy 1.24 or newer
- Rust toolchain with Cargo for the default Rust backend

```bash
python -m pip install -e ".[test]"
```

For a GitHub source zip, unzip it, enter the repository root, and run the same commands. The tests also include a local `src/` path bootstrap so `python tests/test_iflop.py` can import the package before installation, but `python -m pytest -q` is the recommended test entry.

## Quick Smoke Tests

```bash
python -m pytest -q
cd src/iflop/rust
cargo test
```

## Minimal Python API

```python
from iflop import run_flop_envwise, run_flop_obs, run_iflop
from iflop.data.simulation import generate_linear_gaussian_dataset

dataset = generate_linear_gaussian_dataset(num_vars=5, samples_per_env=100, seed=7)

obs_result = run_flop_obs(dataset)
envwise_result = run_flop_envwise(dataset)
iflop_result = run_iflop(dataset)
```

Each function accepts `backend="rust"`, `backend="python"`, or `backend="auto"`. The default is `backend="rust"`. The Python backend is intended for debugging and score-parity tests.

## Examples

```bash
python examples/run_flop_obs.py
python examples/run_flop_envwise.py
python examples/run_iflop.py
python -m iflop.cli.run --method iflop --backend auto
# or, after installation:
iflop --method iflop --backend auto
```

## Output

All public methods return `SearchResult` with:

- `adjacency`: learned adjacency matrix.
- `order`: variable order used by the search.
- `parents`: parent sets by node.
- `total_score`: minimized scalar score.
- `score_key`: `iflop`, `flop_obs`, or `flop_envwise`.
- `score_metadata`: score diagnostics and backend metadata.

Graph output semantics are explicit:

- `flop_obs` and `flop_envwise` return a CPDAG in `adjacency`.
- `iflop` returns an I-CPDAG in `adjacency`, orienting edges that are identifiable from intervention targets and applying conservative Meek closure.
- All methods retain the learned DAG adjacency in `score_metadata["dag_adjacency"]` for debugging and backend parity checks.

## Paper and Citation

This repository is the release implementation accompanying
[I-FLOP: Fast Learning of Order and Parents from Interventional Data](https://openreview.net/forum?id=P0tl7B8p4I),
by Liuting Chen and Alex Markham, published at the 13th International
Conference on Probabilistic Graphical Models (PGM 2026).

If you use I-FLOP in your research, please cite:

```bibtex
@inproceedings{
chen2026iflop,
title={I-{FLOP}: Fast Learning of Order and Parents from Interventional Data},
author={Liuting Chen and Alex Markham},
booktitle={The 13th International Conference on Probabilistic Graphical Models},
year={2026},
url={https://openreview.net/forum?id=P0tl7B8p4I}
}
```

## Release Notes

- License: MPL-2.0.
- Python type marker and public API stub are included.
- R bindings are out of scope for this release.
- Generated artifacts such as `target/`, `.pytest_cache/`, `__pycache__/`, and
  local reports are not part of the release tree.
