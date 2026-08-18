"""Public Python API for the I-FLOP algorithm package."""

from __future__ import annotations

import collections.abc as _abc
import typing as _typing

import numpy as np
import numpy.typing as npt

from iflop.config import IFlopScoreConfig, SearchConfig
from iflop.data.dataset import MultiEnvDataset
from iflop.graph.cpdag import dag_to_cpdag, dag_to_icpdag
from iflop.graph.dag import adjacency_from_parents
from iflop.score.iflop_bic import IFlopBICScorer
from iflop.score.obs_bic import FlopEnvwiseScorer, ObsBICScorer
from iflop.search.flop_search import run_flop_search
from iflop.search.iflop import run_iflop_search
from iflop.search.state import SearchResult

def _dataset_from_input(data: MultiEnvDataset | npt.ArrayLike) -> MultiEnvDataset:
    if isinstance(data, MultiEnvDataset):
        return data
    matrix = np.asarray(data, dtype=float)
    return MultiEnvDataset(env_data={0: matrix}, intervention_targets={0: set()})


def _resolve_backend(
    backend: _typing.Literal["rust", "python", "auto"],
) -> _typing.Literal["rust", "python", "auto"]:
    if backend not in {"rust", "python", "auto"}:
        raise ValueError("backend must be one of 'rust', 'python', or 'auto'.")
    if backend == "auto":
        from iflop.runtime.rust import RUST_AVAILABLE

        return "rust" if RUST_AVAILABLE else "python"
    return backend


def _run_rust(
    dataset: MultiEnvDataset,
    *,
    score_key: str,
    search_config: SearchConfig | None,
    score_config: IFlopScoreConfig | None = None,
) -> SearchResult:
    from iflop.runtime.rust import run_rust

    return run_rust(
        dataset,
        score_key=score_key,
        search_config=search_config,
        score_config=score_config,
    )


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
    data: MultiEnvDataset | npt.ArrayLike,
    *,
    search_config: SearchConfig | None = None,
    backend: _typing.Literal["rust", "python", "auto"] = "rust",
) -> SearchResult:
    """Run FLOP-obs on observational environments.

    The default backend is the Rust core, matching the upstream FLOP
    package boundary. Use ``backend="python"`` for the reference implementation.
    """

    dataset = _dataset_from_input(data).observational_only()
    if _resolve_backend(backend) == "rust":
        return _run_rust(dataset, score_key="flop_obs", search_config=search_config)
    scorer = ObsBICScorer(dataset)
    return _with_cpdag_output(run_flop_search(
        scorer,
        score_key="flop_obs",
        config=search_config,
        metadata={"method": "FLOP-aligned observational baseline", "base_score": "obs_bic_only"},
    ), dataset)


def run_flop_envwise(
    data: MultiEnvDataset | npt.ArrayLike,
    *,
    search_config: SearchConfig | None = None,
    backend: _typing.Literal["rust", "python", "auto"] = "rust",
) -> SearchResult:
    """Run FLOP-envwise on all environments without target filtering."""

    dataset = _dataset_from_input(data)
    if _resolve_backend(backend) == "rust":
        return _run_rust(dataset, score_key="flop_envwise", search_config=search_config)
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


def run_iflop(
    dataset: MultiEnvDataset,
    *,
    score_config: IFlopScoreConfig | None = None,
    search_config: SearchConfig | None = None,
    backend: _typing.Literal["rust", "python", "auto"] = "rust",
) -> SearchResult:
    """Run I-FLOP on a multi-environment interventional dataset.

    I-FLOP combines the FLOP order-search shell with node-wise effective
    environments and an intervention-aware local BIC score.
    """

    if not isinstance(dataset, MultiEnvDataset):
        raise TypeError("run_iflop requires a MultiEnvDataset.")
    score_config = score_config or IFlopScoreConfig()

    if _resolve_backend(backend) == "rust":
        return _run_rust(
            dataset,
            score_key="iflop",
            search_config=search_config,
            score_config=score_config,
        )

    scorer = IFlopBICScorer(dataset, score_config)
    result = run_iflop_search(scorer, config=search_config)
    result.score_key = "iflop"
    result.score_metadata["method"] = "I-FLOP"
    result.score_metadata["base_score"] = "iflop_bic"
    return _with_icpdag_output(result, dataset)


def result_summary(result: SearchResult) -> _abc.Mapping[str, object]:
    skeleton = (result.adjacency != 0) | (result.adjacency.T != 0)
    return {
        "score_key": result.score_key,
        "total_score": result.total_score,
        "order": result.order,
        "num_edges": int(np.triu(skeleton, k=1).sum()),
        "score_vector": result.score_vector,
        "adjacency_type": result.score_metadata.get("adjacency_type"),
    }
