"""Parity smoke: archive scalar engine loads and is finite."""
from __future__ import annotations

import os as _os
from pathlib import Path as _Path

_REPO_ROOT = _Path(__file__).resolve().parents[2]


def _fsot_local_path(env, default):
    """Local path outside the repo: $env if set, else a path relative to the repo root."""
    _v = _os.environ.get(env, "").strip()
    return _Path(_v) if _v else _REPO_ROOT / default


import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "fsot_core" / "python"))
sys.path.insert(0, str(ROOT / "llm" / "python"))

os.environ.setdefault("FSOT_ARCHIVE_ROOT", _os.fspath(_fsot_local_path('FSOT_ARCHIVE_ROOT', 'data_external/physical_archive')))
os.environ.setdefault(
    "FSOT_COMPUTE_PATH",
    _os.fspath(_fsot_local_path('FSOT_2_1_LEAN_ROOT', '../FSOT-2.1-Lean') / 'vendor' / 'fsot_compute.py'),
)

from fsot_bridge import FSOTEngine  # noqa: E402
from fsot_llm.routing import FSOTRouter, ObserverFold  # noqa: E402


def test_engine_loads_from_archive():
    eng = FSOTEngine()
    assert "fsot_compute" in eng.source_path.replace("\\", "/")
    seeds = eng.seeds()
    assert float(seeds["phi"]) > 1.6


def test_scalar_finite():
    eng = FSOTEngine()
    s = eng.scalar_float(D_eff=25, observed=False)
    assert abs(s) < 1e6
    assert s == s  # not NaN


def test_router_layer_fold():
    r = FSOTRouter()
    fold = ObserverFold(layer_index=0, n_layers=12, token_index=0, seq_len=64)
    assert 4.0 <= fold.D_eff <= 25.0
    b = r.attention_bias(fold)
    assert b == b
