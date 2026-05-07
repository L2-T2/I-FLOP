# Score Definitions

Let `V = {1,...,p}` be the node set, `E` the environment set, and `X^(e)` the
data matrix for environment `e`. Each environment has a known intervention
target set `I_e`. All scores are minimized.

## FLOP-Obs

`flop_obs` uses observational rows only. For parent set `P`,

```text
s_j(P) = 0.5 n (1 + log sigma_j^2(P)) + 0.5 log(n) (|P| + 1)
S(G) = sum_j s_j(Pa_G(j))
```

`sigma_j^2(P)` is the residual variance from least squares with an intercept.

## FLOP-Envwise

`flop_envwise` uses every environment for every node:

```text
E_j^FLOP = E
```

It ignores intervention target labels and computes one residual variance from
the pooled centered scatter:

```text
s_j(P) = 0.5 N (1 + log sigma_j^2(P; sum_{e in E} S_e))
       + 0.5 log(N) (|P| + 1)
N = sum_{e in E} n_e
```

## I-FLOP-Envwise

`i_flop_envwise` uses effective environments for each node:

```text
E_j = {e in E : j not in I_e}
```

The local score pools centered scatters over `E_j`, computes one conditional
residual variance, and applies one parent-set penalty:

```text
s_j(P) = 0.5 N_j (1 + log sigma_j^2(P; sum_{e in E_j} S_e))
       + 0.5 log(N_penalty(j)) (|P| + 1)
```

`N_penalty(j)` is controlled by `GiesScoreConfig.penalty_sample_mode`:

- `total`: all rows in the dataset.
- `effective`: rows in `E_j`.
- `max_env`: largest effective environment size.
