# Algorithm Overview

## Data Object

`MultiEnvDataset` stores one numeric data matrix per environment, known intervention target sets, optional variable names, and optional metadata.
Observational environments use an empty target set.

## Search State

The shared search state is an order `pi`, prefix-constrained parent sets `Pa_pi(j) subset Pre_pi(j)`, an adjacency matrix, a scalar score, and score metadata.
The search methods estimate DAGs directly; the public API then converts them to the appropriate equivalence-class graph.

## FLOP-Aligned Search

For `flop_obs` and `flop_envwise`, the search evaluates decomposable local scores:

```text
for node j in order pi:
    choose Pa(j) within Pre_pi(j) by greedy grow-shrink
sum node-local scores
apply reinsertion local search over pi
apply restart / perturbation ILS
return DAG(pi, Pa)
```

`flop_obs` uses observational environments only.
`flop_envwise` uses all environments but ignores intervention target labels.

## I-FLOP

`iflop` keeps the FLOP order-search shell but changes each node-local score to use only effective environments:

```text
E_j = {e : j not in I_e}
```

This is the intervention-aware algorithm exposed by `run_iflop`.

## Backend Boundary

The Python API defaults to the Rust backend. The Python implementation is kept as a reference backend for tests and debugging.
