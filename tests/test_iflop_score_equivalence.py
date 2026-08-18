from __future__ import annotations

import _path_setup  # noqa: F401

import numpy as np

from iflop.data.simulation import generate_linear_gaussian_dataset
from iflop.score._linear import residual_variance_scatter
from iflop.score.iflop_bic import IFlopBICScorer


def test_solve_score_matches_explicit_inverse_formula() -> None:
    dataset = generate_linear_gaussian_dataset(num_vars=4, samples_per_env=50, num_interventions=2, seed=501)
    scorer = IFlopBICScorer(dataset)
    node = 3
    parents = (0, 1)
    terms = scorer._node_terms[node]
    scatter = np.asarray(terms.pooled_scatter, dtype=float)
    n_fit = int(terms.n_fit)

    solve_sigma2 = residual_variance_scatter(scatter, n_fit, node, parents, scorer.config.eps)
    s_xx = scatter[np.ix_(parents, parents)]
    s_xy = scatter[np.ix_(parents, [node])].reshape(len(parents))
    inverse_sigma2 = max((scatter[node, node] - float(s_xy @ np.linalg.inv(s_xx) @ s_xy)) / n_fit, scorer.config.eps)

    assert np.isclose(solve_sigma2, inverse_sigma2, rtol=1.0e-10, atol=1.0e-10)


def test_iflop_target_filtering_excludes_only_targeted_response_envs() -> None:
    dataset = generate_linear_gaussian_dataset(num_vars=5, samples_per_env=30, num_interventions=3, seed=502)
    scorer = IFlopBICScorer(dataset)
    for node in range(dataset.num_vars):
        expected = tuple(env for env in dataset.env_ids if node not in dataset.intervention_targets[env])
        diag = scorer.local_diagnostics(node, parents=())
        assert tuple(diag["effective_envs"]) == expected
        assert tuple(item["env"] for item in diag["per_env"]) == expected
