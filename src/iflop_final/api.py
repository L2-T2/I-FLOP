"""Public Python API for the I-FLOP algorithm package."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Literal

import numpy as np

from iflop_final.config import GiesScoreConfig, SearchConfig
from iflop_final.data.dataset import MultiEnvDataset
from iflop_final.graph.cpdag import dag_to_cpdag, dag_to_icpdag
from iflop_final.graph.dag import adjacency_from_parents
from iflop_final.score.catalog import available_scores
from iflop_final.score.gies_bic import GiesBICScorer
from iflop_final.score.obs_bic import FlopEnvwiseScorer, ObsBICScorer
from iflop_final.search.flop_search import run_flop_search
from iflop_final.search.iflop_envwise import run_iflop_envwise_search
from iflop_final.search.state import SearchResult

Backend = Literal["rust", "python", "auto"]


def _dataset_from_input(data: MultiEnvDataset | np.ndarray) -> MultiEnvDataset:
    if isinstance(data, MultiEnvDataset):
        return data
    matrix = np.asarray(data, dtype=float)
    return MultiEnvDataset(env_data={0: matrix}, intervention_targets={0: set()})


def _resolve_backend(backend: Backend) -> Backend:
    if backend not in {"rust", "python", "auto"}:
        raise ValueError("backend must be one of 'rust', 'python', or 'auto'.")
    if backend == "auto":
        from iflop_final.runtime.native import NATIVE_AVAILABLE

        return "rust" if NATIVE_AVAILABLE else "python"
    return backend


def _run_native(
    dataset: MultiEnvDataset,
    *,
    score_key: str,
    search_config: SearchConfig | None,
) -> SearchResult:
    from iflop_final.runtime.native import run_native_iflop

    return run_native_iflop(dataset, score_key=score_key, search_config=search_config)


def _with_cpdag_output(result: SearchResult, dataset: MultiEnvDataset) -> SearchResult:
    dag_adjacency = np.asarray(
        result.score_metadata.get("dag_adjacency", adjacency_from_parents(result.parents, dataset.num_vars)),
        dtype=int,
    )
    result.adjacency = dag_to_cpdag(dag_adjacency)
    result.score_metadata["adjacency_type"] = "cpdag"
    result.score_metadata["dag_adjacency"] = dag_adjacency.tolist()
    return result


def _with_icpdag_output(result: SearchResult, dataset: MultiEnvDataset) -> SearchResult:
    dag_adjacency = np.asarray(
        result.score_metadata.get("dag_adjacency", adjacency_from_parents(result.parents, dataset.num_vars)),
        dtype=int,
    )
    result.adjacency = dag_to_icpdag(dag_adjacency, dataset.intervention_targets)
    result.score_metadata["adjacency_type"] = "i_cpdag"
    result.score_metadata["dag_adjacency"] = dag_adjacency.tolist()
    result.score_metadata["intervention_targets"] = {
        int(env): tuple(sorted(int(node) for node in targets))
        for env, targets in dataset.intervention_targets.items()
    }
    return result


def run_flop_obs(
    data: MultiEnvDataset | np.ndarray,
    *,
    search_config: SearchConfig | None = None,
    backend: Backend = "rust",
) -> SearchResult:
    """Run FLOP-obs on observational environments.

    The default backend is the Rust native core, matching the upstream FLOP
    package boundary. Use ``backend="python"`` for the reference implementation.
    """

    dataset = _dataset_from_input(data).observational_only()
    if _resolve_backend(backend) == "rust":
        return _run_native(dataset, score_key="flop_obs", search_config=search_config)
    scorer = ObsBICScorer(dataset)
    return _with_cpdag_output(run_flop_search(
        scorer,
        score_key="flop_obs",
        config=search_config,
        metadata={"method": "FLOP-aligned observational baseline", "base_score": "obs_bic_only"},
    ), dataset)


def run_flop_envwise(
    data: MultiEnvDataset | np.ndarray,
    *,
    search_config: SearchConfig | None = None,
    backend: Backend = "rust",
) -> SearchResult:
    """Run FLOP-envwise on all environments without target filtering."""

    dataset = _dataset_from_input(data)
    if _resolve_backend(backend) == "rust":
        return _run_native(dataset, score_key="flop_envwise", search_config=search_config)
    scorer = FlopEnvwiseScorer(dataset)
    return _with_cpdag_output(run_flop_search(
        scorer,
        score_key="flop_envwise",
        config=search_config,
        metadata={
            "method": "FLOP-aligned envwise baseline",
            "canonical_method": "FLOP-envwise",
            "base_score": "flop_envwise",
            "target_filtering": "none",
        },
    ), dataset)


def run_iflop_envwise(
    dataset: MultiEnvDataset,
    *,
    gies_config: GiesScoreConfig | None = None,
    search_config: SearchConfig | None = None,
    penalty_sample_mode: str | None = None,
    backend: Backend = "rust",
) -> SearchResult:
    """Run the I-FLOP-envwise interventional extension.

    This is the release-facing I-FLOP method: the FLOP order-search shell with
    node-wise effective environments and the envwise GIES-style local BIC score.
    """

    if not isinstance(dataset, MultiEnvDataset):
        raise TypeError("run_iflop_envwise requires a MultiEnvDataset.")
    if gies_config is None:
        gies_config = GiesScoreConfig(
            variant="envwise",
            penalty_sample_mode=penalty_sample_mode or GiesScoreConfig().penalty_sample_mode,
        )
    elif penalty_sample_mode is not None:
        gies_config = GiesScoreConfig(
            variant="envwise",
            eps=gies_config.eps,
            penalty_sample_mode=penalty_sample_mode,
            fit_weight=gies_config.fit_weight,
            penalty_weight=gies_config.penalty_weight,
            envwise_residual_mode=gies_config.envwise_residual_mode,
        )
    elif gies_config.variant != "envwise":
        gies_config = GiesScoreConfig(
            variant="envwise",
            eps=gies_config.eps,
            penalty_sample_mode=gies_config.penalty_sample_mode,
            fit_weight=gies_config.fit_weight,
            penalty_weight=gies_config.penalty_weight,
            envwise_residual_mode=gies_config.envwise_residual_mode,
        )

    if _resolve_backend(backend) == "rust":
        return _run_native(dataset, score_key="i_flop_envwise", search_config=search_config)

    scorer = GiesBICScorer(dataset, gies_config)
    result = run_iflop_envwise_search(scorer, config=search_config)
    result.score_key = "i_flop_envwise"
    result.score_metadata["method"] = "I-FLOP-envwise"
    result.score_metadata["base_score"] = "envwise_gies_bic"
    return _with_icpdag_output(result, dataset)


def run_iflop(
    dataset: MultiEnvDataset | np.ndarray,
    *,
    score_key: str = "i_flop_envwise",
    search_config: SearchConfig | None = None,
    score_config: GiesScoreConfig | None = None,
    backend: Backend = "rust",
) -> SearchResult:
    """Dispatch to one of the final supported algorithm entries."""

    key = str(score_key)
    if key == "flop_obs":
        return run_flop_obs(dataset, search_config=search_config, backend=backend)
    if key == "flop_envwise":
        return run_flop_envwise(dataset, search_config=search_config, backend=backend)
    if key == "i_flop_envwise":
        gies_config = score_config if isinstance(score_config, GiesScoreConfig) else None
        return run_iflop_envwise(
            _dataset_from_input(dataset),
            gies_config=gies_config,
            search_config=search_config,
            backend=backend,
        )
    raise ValueError(f"unsupported score_key {score_key!r}; available keys: {available_scores()}")


def result_summary(result: SearchResult) -> Mapping[str, object]:
    return {
        "score_key": result.score_key,
        "total_score": result.total_score,
        "order": result.order,
        "num_edges": int(result.adjacency.sum()),
        "score_vector": result.score_vector,
        "adjacency_type": result.score_metadata.get("adjacency_type"),
    }
