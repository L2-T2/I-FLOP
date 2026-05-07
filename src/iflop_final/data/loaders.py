"""Minimal CSV/JSON loader for final I-FLOP datasets."""

from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import Any

import numpy as np

from iflop_final.data.dataset import MultiEnvDataset


def read_csv_matrix(path: str | Path) -> np.ndarray:
    resolved = Path(path)
    with resolved.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.reader(handle)
        rows = [[float(cell) for cell in row] for row in reader if row]
    return np.asarray(rows, dtype=float)


def load_dataset_manifest(path: str | Path) -> MultiEnvDataset:
    """Load a small manifest with env CSVs and intervention targets.

    Expected keys: `env_files`, `intervention_targets`. Optional keys:
    `variable_names`, `true_dag_csv`, `metadata`.
    """

    manifest_path = Path(path).resolve()
    with manifest_path.open("r", encoding="utf-8") as handle:
        manifest: dict[str, Any] = json.load(handle)
    base = manifest_path.parent
    env_data = {
        int(env): read_csv_matrix(base / rel_path)
        for env, rel_path in dict(manifest["env_files"]).items()
    }
    targets = {
        int(env): {int(node) for node in values}
        for env, values in dict(manifest["intervention_targets"]).items()
    }
    true_dag = None
    if manifest.get("true_dag_csv"):
        true_dag = read_csv_matrix(base / str(manifest["true_dag_csv"])).astype(int)
    metadata = dict(manifest.get("metadata", {}) or {})
    metadata["manifest_path"] = str(manifest_path)
    return MultiEnvDataset(
        env_data=env_data,
        intervention_targets=targets,
        variable_names=list(manifest.get("variable_names", []) or []) or None,
        true_dag=true_dag,
        metadata=metadata,
    )

