# Implementation Notes

## File Structure

- `src/iflop/api.py`: public Python API and backend selection.
- `src/iflop/data/`: dataset model, loaders, preprocessing, simulation.
- `src/iflop/graph/`: DAG and adjacency helpers.
- `src/iflop/score/`: Python reference scorers.
- `src/iflop/search/`: Python reference search shell.
- `src/iflop/runtime/rust.py`: Rust execution bridge.
- `src/iflop/rust/`: the single Rust algorithm project and CLI protocol.

## Public API

The release public surface is intentionally small:

- `run_flop_obs(...)`
- `run_flop_envwise(...)`
- `run_iflop(...)`

Each function accepts `backend="rust"`, `backend="python"`, or `backend="auto"`. The Rust backend is the default.

Output graph semantics:

- `flop_obs` returns CPDAG adjacency.
- `flop_envwise` returns CPDAG adjacency.
- `iflop` returns I-CPDAG adjacency.
- The DAG selected by order search is preserved in
  `SearchResult.score_metadata["dag_adjacency"]`.

## Python And Rust Boundary

The Rust backend is built with Cargo on demand and receives one serialized request per run. The Python reference backend does not call external research trees and is used for deterministic parity tests.

## Excluded From Release

- R bindings.
- Exploratory score families.
- Generated reports and outputs.
- Build and test caches.
