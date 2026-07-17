"""
Archive verification bridge — I:\\FSOT-Physical-Archive is the tool of record.
"""
from __future__ import annotations

import json
import os
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Optional

from fsot_llm.paths import ensure_sys_path, workspace_root

ensure_sys_path()

from fsot_bridge import FSOTEngine


def archive_root() -> Path:
    return Path(os.environ.get("FSOT_ARCHIVE_ROOT", r"I:\FSOT-Physical-Archive"))


def lean_hub() -> Path:
    return archive_root() / "02_FSOT-2.1-Lean-Full"


@dataclass
class ArchiveStatus:
    ok: bool
    archive_root: str
    lean_hub: str
    compute_path: str
    compute_exists: bool
    intrinsic_llm_vendor: bool
    intrinsic_llm_benchmark: bool
    notes: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return {
            "ok": self.ok,
            "archive_root": self.archive_root,
            "lean_hub": self.lean_hub,
            "compute_path": self.compute_path,
            "compute_exists": self.compute_exists,
            "intrinsic_llm_vendor": self.intrinsic_llm_vendor,
            "intrinsic_llm_benchmark": self.intrinsic_llm_benchmark,
            "notes": self.notes,
        }


def check_archive() -> ArchiveStatus:
    root = archive_root()
    hub = lean_hub()
    compute = Path(
        os.environ.get(
            "FSOT_COMPUTE_PATH",
            str(hub / "vendor" / "fsot_compute.py"),
        )
    )
    vendor = hub / "vendor" / "intrinsic_llm" / "benchmark_results_final.json"
    bench = hub / "data" / "intrinsic_llm_validators_benchmark.json"
    notes: list[str] = []
    if not root.is_dir():
        notes.append("FSOT_ARCHIVE_ROOT missing — mount I: drive")
    if not compute.is_file():
        notes.append("fsot_compute.py missing")
    if not vendor.is_file():
        notes.append("vendor intrinsic_llm benchmark missing")
    if not bench.is_file():
        notes.append("data intrinsic_llm_validators_benchmark.json missing")
    ok = (
        root.is_dir()
        and compute.is_file()
        and vendor.is_file()
        and bench.is_file()
    )
    return ArchiveStatus(
        ok=ok,
        archive_root=str(root),
        lean_hub=str(hub),
        compute_path=str(compute),
        compute_exists=compute.is_file(),
        intrinsic_llm_vendor=vendor.is_file(),
        intrinsic_llm_benchmark=bench.is_file(),
        notes=notes,
    )


def load_intrinsic_llm_vendor() -> list[dict[str, Any]]:
    path = lean_hub() / "vendor" / "intrinsic_llm" / "benchmark_results_final.json"
    return json.loads(path.read_text(encoding="utf-8"))


def scalar_gold(
    *,
    D_eff: float = 25.0,
    observed: bool = False,
    recent_hits: float = 0.0,
    N: float = 1.0,
) -> dict[str, Any]:
    eng = FSOTEngine()
    s = eng.scalar(
        D_eff=D_eff, observed=observed, recent_hits=recent_hits, N=N
    )
    return {
        "S_str": s,
        "S_float": float(s),
        "D_eff": D_eff,
        "observed": observed,
        "recent_hits": recent_hits,
        "N": N,
        "source": eng.source_path,
        "seeds": eng.seeds(),
    }


def relative_error(pred: float, gold: float) -> float:
    if gold == 0:
        return abs(pred - gold)
    return abs(pred - gold) / abs(gold)


def write_ledger(
    name: str,
    payload: dict[str, Any],
    *,
    ledgers_dir: Optional[Path] = None,
) -> Path:
    ledgers = ledgers_dir or (workspace_root() / "llm" / "benchmarks" / "ledgers")
    ledgers.mkdir(parents=True, exist_ok=True)
    ts = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    path = ledgers / f"{ts}_{name}.json"
    doc = {
        "written_at": datetime.now(timezone.utc).isoformat(),
        "archive": check_archive().to_dict(),
        **payload,
    }
    path.write_text(json.dumps(doc, indent=2), encoding="utf-8")
    # also latest pointer
    latest = ledgers / f"latest_{name}.json"
    latest.write_text(json.dumps(doc, indent=2), encoding="utf-8")
    return path
