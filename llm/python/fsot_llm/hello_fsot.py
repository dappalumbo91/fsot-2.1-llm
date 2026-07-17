"""Smoke test: FSOT engine + environment + hardware envelope."""
from __future__ import annotations

import os
import platform
import sys

from fsot_llm.paths import ensure_sys_path, workspace_root

ensure_sys_path()

from fsot_bridge import FSOTEngine  # noqa: E402


def _gpu_line() -> str:
    try:
        import torch

        if torch.cuda.is_available():
            name = torch.cuda.get_device_name(0)
            mem = torch.cuda.get_device_properties(0).total_memory / (1024**3)
            return f"CUDA OK — {name} ({mem:.1f} GB)"
        return "CUDA not available (CPU-only torch)"
    except ImportError:
        return "torch not installed yet (run scripts/bootstrap.ps1)"


def main() -> int:
    print("FSOT 2.1 LLM — hello")
    print(f"  workspace : {workspace_root()}")
    print(f"  python    : {sys.version.split()[0]} ({platform.platform()})")
    print(f"  ARCHIVE   : {os.environ.get('FSOT_ARCHIVE_ROOT', '(unset)')}")
    print(f"  COMPUTE   : {os.environ.get('FSOT_COMPUTE_PATH', '(unset)')}")
    print(f"  PORTABLE  : {os.environ.get('FSOT_PORTABLE', '(unset)')}")
    print(f"  GPU       : {_gpu_line()}")
    print()
    FSOTEngine().module  # force load
    from fsot_bridge import hello

    hello()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
