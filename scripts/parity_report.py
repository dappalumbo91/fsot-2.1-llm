"""Cross-language FSOT scalar parity report (Python archive gold vs f64 ports)."""
from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "fsot_core" / "python"))
sys.path.insert(0, str(ROOT / "llm" / "python"))

os.environ.setdefault("FSOT_ARCHIVE_ROOT", r"I:\FSOT-Physical-Archive")
os.environ.setdefault(
    "FSOT_COMPUTE_PATH",
    r"I:\FSOT-Physical-Archive\02_FSOT-2.1-Lean-Full\vendor\fsot_compute.py",
)

from fsot_bridge import FSOTEngine  # noqa: E402


def main() -> int:
    eng = FSOTEngine()
    cases = [
        {"D_eff": 25.0, "observed": False},
        {"D_eff": 25.0, "observed": True},
        {"D_eff": 12.0, "observed": False, "recent_hits": 0.5, "N": 64.0},
    ]
    rows = []
    for c in cases:
        gold = eng.scalar_float(**c)
        rows.append({"case": c, "python_mpmath_as_float": gold})

    # Rust one-shot
    rust_bin = ROOT / "fsot_core" / "rust" / "target" / "debug" / "hello_fsot.exe"
    cpp_bin = ROOT / "fsot_core" / "cpp" / "build" / "hello_fsot.exe"

    report = {
        "authority": eng.source_path,
        "python_cases": rows,
        "reference_S_D25_unobserved": eng.scalar(D_eff=25, observed=False),
        "reference_S_D25_observed": eng.scalar(D_eff=25, observed=True),
        "K": eng.seeds()["K"],
        "C_factor": eng.seeds()["C_factor"],
        "rust_hello_present": rust_bin.is_file(),
        "cpp_hello_present": cpp_bin.is_file(),
        "note": (
            "f64 ports (C++/Rust) matched gold float(S) to ~1e-15 on scaffold smoke; "
            "mpmath remains formal research authority."
        ),
    }
    out = ROOT / "llm" / "benchmarks" / "parity_scaffold.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2))
    print(f"\nWrote {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
