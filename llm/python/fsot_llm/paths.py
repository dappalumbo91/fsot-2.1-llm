from __future__ import annotations

import sys
from pathlib import Path


def workspace_root() -> Path:
    # llm/python/fsot_llm/paths.py -> workspace
    return Path(__file__).resolve().parents[3]


def ensure_sys_path() -> None:
    root = workspace_root()
    core_py = root / "fsot_core" / "python"
    llm_py = root / "llm" / "python"
    for p in (str(core_py), str(llm_py)):
        if p not in sys.path:
            sys.path.insert(0, p)
