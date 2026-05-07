from __future__ import annotations

from pathlib import Path


def _token(*parts: str) -> str:
    return "".join(parts)


FORBIDDEN = (
    _token("GIES", "-", "IGSP", "-", "style"),
    _token("vendor", "/", "compare"),
    _token("flopsearch", "-", "main"),
    _token("external", "/"),
    _token("sys", ".", "path"),
)


def test_no_old_local_dependency() -> None:
    root = Path(__file__).resolve().parents[1]
    for path in root.rglob("*"):
        if path.is_dir() or "__pycache__" in path.parts or path.name.endswith(".pyc"):
            continue
        text = path.read_text(errors="ignore")
        for token in FORBIDDEN:
            assert token not in text, f"forbidden legacy token {token!r} found in {path}"
