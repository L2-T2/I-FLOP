# Implementation Notes

## File Structure

- `src/iflop_final/api.py`: public Python API and backend dispatch.
- `src/iflop_final/data/`: dataset model, loaders, preprocessing, simulation.
- `src/iflop_final/graph/`: DAG and adjacency helpers.
- `src/iflop_final/score/`: Python reference scorers.
- `src/iflop_final/search/`: Python reference search shell.
- `src/iflop_final/runtime/`: Rust native backend bridge.
- `rust/iflop_native/`: Rust native algorithm core and CLI protocol.

## Public API

The release public surface is intentionally small:

- `run_flop_obs(...)`
- `run_flop_envwise(...)`
- `run_iflop_envwise(...)`
- `run_iflop(..., score_key=...)`

Each function accepts `backend="rust"`, `backend="python"`, or
`backend="auto"`. Release users should use the default Rust backend.

Output graph semantics:

- `flop_obs` returns CPDAG adjacency.
- `flop_envwise` returns CPDAG adjacency.
- `i_flop_envwise` returns I-CPDAG adjacency.
- The DAG selected by order search is preserved in
  `SearchResult.score_metadata["dag_adjacency"]`.

## Python And Rust Boundary

The native backend is built with Cargo on demand and receives one serialized
request per run. The Python reference backend does not call external research
trees and is used for deterministic parity tests.

## Excluded From Release

- R bindings.
- Legacy method aliases.
- Exploratory score families.
- Historical reports and generated outputs.
- Build and test caches.
