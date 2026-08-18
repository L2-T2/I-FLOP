"""Rust backend bridge for I-FLOP algorithms."""

from __future__ import annotations

import os
import shutil
import subprocess
import tempfile
from pathlib import Path
import collections.abc as _abc

import numpy as np

from iflop.config import IFlopScoreConfig, SearchConfig
from iflop.data.dataset import MultiEnvDataset
from iflop.graph.cpdag import dag_to_icpdag
from iflop.search.state import SearchResult

_BIN_NAME = "iflop.exe" if os.name == "nt" else "iflop"


_RUST_DIR = Path(__file__).resolve().parents[1] / "rust"

RUST_AVAILABLE = _RUST_DIR.joinpath("Cargo.toml").exists() and shutil.which("cargo") is not None
SUPPORTED_RUST_SCORE_KEYS = ("flop_obs", "flop_envwise", "iflop")


def rust_project_dir() -> Path:
    return _RUST_DIR


def rust_binary_path(*, release: bool = False) -> Path:
    profile = "release" if release else "debug"
    return _rust_target_dir() / profile / _BIN_NAME


def _rust_target_dir() -> Path:
    configured = os.environ.get("IFLOP_TARGET_DIR")
    if configured:
        return Path(configured)
    return Path(tempfile.gettempdir()) / "iflop_target"


def build_rust_backend(*, release: bool = False, quiet: bool = True) -> Path:
    if not _RUST_DIR.joinpath("Cargo.toml").exists():
        raise RuntimeError(f"Rust project not found at {_RUST_DIR}")
    if shutil.which("cargo") is None:
        raise RuntimeError("cargo is required to build the Rust backend")
    path = rust_binary_path(release=release)
    if _binary_is_current(path):
        return path
    cmd = ["cargo", "build"]
    if release:
        cmd.append("--release")
    cmd.extend(["--target-dir", str(_rust_target_dir())])
    completed = subprocess.run(
        cmd,
        cwd=_RUST_DIR,
        check=False,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    )
    if completed.returncode != 0:
        detail = completed.stderr.strip() or completed.stdout.strip()
        raise RuntimeError(f"failed to build Rust backend: {detail}")
    if not path.exists():
        raise RuntimeError(f"Rust backend build succeeded but binary is missing: {path}")
    if not quiet and completed.stdout:
        print(completed.stdout)
    return path


def _binary_is_current(path: Path) -> bool:
    if not path.exists():
        return False
    binary_mtime = path.stat().st_mtime
    sources = [_RUST_DIR / "Cargo.toml", *(_RUST_DIR / "src").glob("*.rs")]
    return all(source.exists() and source.stat().st_mtime <= binary_mtime for source in sources)


def require_rust_backend() -> None:
    build_rust_backend()


def run_rust(
    dataset: MultiEnvDataset,
    *,
    score_key: str = "iflop",
    search_config: SearchConfig | None = None,
    score_config: IFlopScoreConfig | None = None,
    release: bool = False,
) -> SearchResult:
    return _invoke_rust(
        dataset,
        score_key=score_key,
        mode="run",
        order=None,
        search_config=search_config,
        score_config=score_config,
        release=release,
    )


def evaluate_rust_order(
    dataset: MultiEnvDataset,
    *,
    score_key: str,
    order: _abc.Iterable[int],
    search_config: SearchConfig | None = None,
    score_config: IFlopScoreConfig | None = None,
    release: bool = False,
) -> SearchResult:
    return _invoke_rust(
        dataset,
        score_key=score_key,
        mode="eval_order",
        order=tuple(int(node) for node in order),
        search_config=search_config,
        score_config=score_config,
        release=release,
    )


def _invoke_rust(
    dataset: MultiEnvDataset,
    *,
    score_key: str,
    mode: str,
    order: tuple[int, ...] | None,
    search_config: SearchConfig | None,
    score_config: IFlopScoreConfig | None,
    release: bool,
) -> SearchResult:
    if not isinstance(dataset, MultiEnvDataset):
        raise TypeError("Rust backend requires a MultiEnvDataset")
    key = str(score_key)
    if key not in SUPPORTED_RUST_SCORE_KEYS:
        raise ValueError(f"Rust backend supports {SUPPORTED_RUST_SCORE_KEYS}, got {score_key!r}")
    cfg = search_config or SearchConfig()
    score = score_config or IFlopScoreConfig()
    binary = build_rust_backend(release=release)
    request = _serialize_request(
        dataset,
        mode=mode,
        score_key=key,
        order=order,
        search_config=cfg,
        score_config=score,
    )
    with tempfile.TemporaryDirectory(prefix="iflop_") as tmpdir:
        request_path = Path(tmpdir) / "request.txt"
        request_path.write_text(request, encoding="utf-8")
        completed = subprocess.run(
            [str(binary), str(request_path)],
            check=False,
            text=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
        )
    if completed.returncode != 0:
        detail = completed.stdout.strip() or completed.stderr.strip()
        raise RuntimeError(f"Rust backend failed: {detail}")
    result = _parse_response(completed.stdout, score_key=key, binary=binary)
    if key == "iflop":
        dag_adjacency = np.asarray(result.score_metadata["dag_adjacency"], dtype=int)
        result.adjacency = dag_to_icpdag(dag_adjacency, dataset.intervention_targets)
        result.score_metadata["adjacency_type"] = "i_cpdag"
        result.score_metadata["intervention_targets"] = {
            int(env): tuple(sorted(int(node) for node in targets))
            for env, targets in dataset.intervention_targets.items()
        }
    return result


def _serialize_request(
    dataset: MultiEnvDataset,
    *,
    mode: str,
    score_key: str,
    order: tuple[int, ...] | None,
    search_config: SearchConfig,
    score_config: IFlopScoreConfig,
) -> str:
    lines: list[str] = [
        "IFLOP_V1",
        f"MODE {mode}",
        f"SCORE_KEY {score_key}",
        f"P {dataset.num_vars}",
        f"EPS {float(score_config.eps):.17g}",
        f"PENALTY_SAMPLE_MODE {score_config.penalty_sample_mode}",
        f"SCORE_FIT_WEIGHT {float(score_config.fit_weight):.17g}",
        f"SCORE_PENALTY_WEIGHT {float(score_config.penalty_weight):.17g}",
        f"RESIDUAL_MODE {score_config.residual_mode}",
        f"SEARCH_ILS {int(search_config.ils_restarts)}",
        f"RANDOM_SEED {int(search_config.random_seed)}",
        "MAX_SWEEPS none" if search_config.max_sweeps is None else f"MAX_SWEEPS {int(search_config.max_sweeps)}",
        "PERTURBATION_SIZE none"
        if search_config.perturbation_size is None
        else f"PERTURBATION_SIZE {int(search_config.perturbation_size)}",
        f"ATOL {float(search_config.atol):.17g}",
    ]
    if order is not None:
        lines.append(f"ORDER {len(order)} {' '.join(str(int(node)) for node in order)}")
    for env in dataset.env_ids:
        data = np.asarray(dataset.env_data[env], dtype=float)
        targets = sorted(int(node) for node in dataset.intervention_targets[env])
        lines.append(f"ENV {int(env)} {int(data.shape[0])} {len(targets)} {' '.join(str(node) for node in targets)}".rstrip())
        for row in data:
            lines.append(" ".join(f"{float(value):.17g}" for value in row))
    lines.append("END")
    return "\n".join(lines) + "\n"


def _parse_response(text: str, *, score_key: str, binary: Path) -> SearchResult:
    lines = [line.strip() for line in text.splitlines() if line.strip()]
    if not lines or lines[0] != "IFLOP_RESULT_V1":
        raise RuntimeError(f"unexpected Rust response header: {text[:200]}")
    status = _field_value(lines, "status")
    if status != "ok":
        reason = _field_value(lines, "failure_reason", default="unknown Rust failure")
        raise RuntimeError(reason)
    p = int(_field_value(lines, "p"))
    total_score = float(_field_value(lines, "total_score"))
    adjacency_type = _field_value(lines, "adjacency_type", default="dag")
    order_parts = _field_value(lines, "order").split()
    order_len = int(order_parts[0])
    order = [int(item) for item in order_parts[1:]]
    if len(order) != order_len:
        raise RuntimeError("Rust response order length mismatch")

    vector_value = _field_value(lines, "score_vector")
    score_vector: tuple[int, int] | None
    if vector_value == "none":
        score_vector = None
    else:
        vector_parts = vector_value.split()
        score_vector = (int(vector_parts[0]), int(vector_parts[1]))

    parents: dict[int, set[int]] = {}
    start = lines.index("parents_start") + 1
    end = lines.index("parents_end")
    for line in lines[start:end]:
        parts = line.split()
        if parts[0] != "parents":
            raise RuntimeError(f"unexpected parents row: {line}")
        child = int(parts[1])
        count = int(parts[2])
        values = {int(item) for item in parts[3:]}
        if len(values) != count:
            raise RuntimeError(f"Rust response parent count mismatch for node {child}")
        parents[child] = values
    for node in range(p):
        parents.setdefault(node, set())

    adj_start = lines.index("adjacency_start") + 1
    adj_end = lines.index("adjacency_end")
    adjacency_rows = []
    for line in lines[adj_start:adj_end]:
        adjacency_rows.append([int(item) for item in line.split()])
    adjacency = np.asarray(adjacency_rows, dtype=int)
    if adjacency.shape != (p, p):
        raise RuntimeError(f"Rust response adjacency shape mismatch: {adjacency.shape}")

    dag_adjacency = None
    if "dag_adjacency_start" in lines and "dag_adjacency_end" in lines:
        dag_start = lines.index("dag_adjacency_start") + 1
        dag_end = lines.index("dag_adjacency_end")
        dag_rows = []
        for line in lines[dag_start:dag_end]:
            dag_rows.append([int(item) for item in line.split()])
        dag_adjacency = np.asarray(dag_rows, dtype=int)
        if dag_adjacency.shape != (p, p):
            raise RuntimeError(f"Rust response DAG adjacency shape mismatch: {dag_adjacency.shape}")

    method_metadata: dict[str, object]
    if score_key == "iflop":
        method_metadata = {"method": "I-FLOP", "base_score": "iflop_bic"}
    elif score_key == "flop_obs":
        method_metadata = {
            "method": "FLOP-aligned observational baseline",
            "base_score": "obs_bic_only",
        }
    else:
        method_metadata = {
            "method": "FLOP-aligned envwise baseline",
            "base_score": "flop_envwise",
            "target_filtering": "none",
        }

    return SearchResult(
        adjacency=adjacency,
        order=order,
        parents=parents,
        total_score=total_score,
        score_key=score_key,
        score_metadata={
            "backend": "rust",
            "rust_binary": str(binary),
            "adjacency_type": adjacency_type,
            "dag_adjacency": None if dag_adjacency is None else dag_adjacency.tolist(),
            "supported_rust_score_keys": SUPPORTED_RUST_SCORE_KEYS,
            "parity_note": "fixed-order score parity is deterministic; ILS random sequence is Rust-specific",
            **method_metadata,
        },
        trajectory=[],
        score_vector=score_vector,
    )


def _field_value(lines: list[str], field: str, *, default: str | None = None) -> str:
    prefix = f"{field} "
    for line in lines:
        if line.startswith(prefix):
            return line[len(prefix) :]
    if default is not None:
        return default
    raise RuntimeError(f"missing Rust response field: {field}")
