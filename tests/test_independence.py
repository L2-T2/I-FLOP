from __future__ import annotations

import _path_setup  # noqa: F401

from pathlib import Path


def _token(*parts: str) -> str:
    return "".join(parts)


FORBIDDEN = (
    _token("GIES", "-", "IGSP", "-", "style"),
    _token("vendor", "/", "compare"),
    _token("flopsearch", "-", "main"),
    _token("external", "/"),
    _token("sys", ".", "path"),
    _token("nat", "ive"),
    _token("iflop", "_", "final"),
    _token("iflop", "_", "envwise"),
    _token("i", "_", "flop", "_", "envwise"),
    _token("dag", "_to_", "cpdag", "_proxy"),
)

IGNORED_GENERATED_DIRS = {
    ".git",
    ".mypy_cache",
    ".pytest_cache",
    "__pycache__",
    "build",
    "dist",
    "target",
}


def test_no_old_local_dependency() -> None:
    root = Path(__file__).resolve().parents[1]
    for path in root.rglob("*"):
        relative_parts = path.relative_to(root).parts
        if (
            path.is_dir()
            or any(part in IGNORED_GENERATED_DIRS for part in relative_parts)
            or any(part.endswith(".egg-info") for part in relative_parts)
            or path.name.endswith(".pyc")
        ):
            continue
        text = path.read_text(errors="ignore")
        for token in FORBIDDEN:
            assert token not in text, f"forbidden retired token {token!r} found in {path}"
