from __future__ import annotations

import sys
from pathlib import Path


SRC_DIR = Path(__file__).resolve().parents[1] / "src"
path_entries = getattr(sys, "path")
if str(SRC_DIR) not in path_entries:
    path_entries.insert(0, str(SRC_DIR))
