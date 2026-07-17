"""
FSOT 2.1 LLM — bridge to the canonical scalar engine.

Never re-implement seed authority here. Load the archive (or Lean hub) copy of
fsot_compute.py and expose a stable API for the LLM stack.
"""
from __future__ import annotations

import importlib.util
import os
import sys
from pathlib import Path
from types import ModuleType
from typing import Any, Optional

# Workspace root: .../fsot 2.1 llm
_WORKSPACE = Path(__file__).resolve().parents[2]


def default_compute_candidates() -> list[Path]:
    env = os.environ.get("FSOT_COMPUTE_PATH")
    candidates: list[Path] = []
    if env:
        candidates.append(Path(env))
    archive = os.environ.get(
        "FSOT_ARCHIVE_ROOT", r"I:\FSOT-Physical-Archive"
    )
    candidates.extend(
        [
            Path(archive)
            / "02_FSOT-2.1-Lean-Full"
            / "vendor"
            / "fsot_compute.py",
            Path(r"I:\FSOT-Physical-Archive\02_FSOT-2.1-Lean-Full\vendor\fsot_compute.py"),
            _WORKSPACE / "third_party" / "fsot_compute.py",
        ]
    )
    return candidates


def load_fsot_compute(path: Optional[Path] = None) -> ModuleType:
    """Import the canonical FSOT compute module by path (not as installed package)."""
    paths = [path] if path else default_compute_candidates()
    last_err: Optional[Exception] = None
    for p in paths:
        if p is None:
            continue
        p = Path(p)
        if not p.is_file():
            continue
        try:
            spec = importlib.util.spec_from_file_location("fsot_compute_canonical", p)
            if spec is None or spec.loader is None:
                continue
            mod = importlib.util.module_from_spec(spec)
            # Register early so dataclasses / type hints resolve cleanly
            sys.modules["fsot_compute_canonical"] = mod
            spec.loader.exec_module(mod)
            mod.__fsot_source_path__ = str(p.resolve())  # type: ignore[attr-defined]
            return mod
        except Exception as exc:  # pragma: no cover
            last_err = exc
            continue
    raise FileNotFoundError(
        "Could not load canonical fsot_compute.py. "
        "Set FSOT_COMPUTE_PATH or mount I:\\FSOT-Physical-Archive. "
        f"Tried: {[str(x) for x in paths if x]}. Last error: {last_err}"
    )


class FSOTEngine:
    """Thin façade over the archive scalar engine."""

    def __init__(self, compute_path: Optional[Path] = None) -> None:
        self._mod = load_fsot_compute(compute_path)
        self.source_path: str = getattr(
            self._mod, "__fsot_source_path__", "unknown"
        )

    @property
    def module(self) -> ModuleType:
        return self._mod

    def seeds(self) -> dict[str, str]:
        m = self._mod
        return {
            "pi": str(m.PI),
            "e": str(m.E),
            "phi": str(m.PHI),
            "gamma": str(m.GAMMA),
            "G_catalan": str(m.G_CAT),
            "K": str(m.K),
            "C_factor": str(m.C_FACTOR),
            "psi_con": str(m.PSI_CON),
            "source": self.source_path,
        }

    def scalar(
        self,
        *,
        N: float = 1.0,
        P: float = 1.0,
        D_eff: float = 25.0,
        recent_hits: float = 0.0,
        observed: bool = False,
        **kwargs: Any,
    ) -> str:
        """Return raw_S as high-precision string (mpmath)."""
        m = self._mod
        mpf = m.mpf
        inp = m.ScalarInput(
            N=mpf(str(N)),
            P=mpf(str(P)),
            D_eff=mpf(str(D_eff)),
            recent_hits=mpf(str(recent_hits)),
            observed=observed,
            **{
                k: (mpf(str(v)) if not isinstance(v, bool) else v)
                for k, v in kwargs.items()
            },
        )
        return str(m.compute_scalar(inp))

    def scalar_float(
        self,
        *,
        N: float = 1.0,
        P: float = 1.0,
        D_eff: float = 25.0,
        recent_hits: float = 0.0,
        observed: bool = False,
        **kwargs: Any,
    ) -> float:
        return float(
            self.scalar(
                N=N,
                P=P,
                D_eff=D_eff,
                recent_hits=recent_hits,
                observed=observed,
                **kwargs,
            )
        )



def hello() -> None:
    eng = FSOTEngine()
    seeds = eng.seeds()
    print("=== FSOT 2.1 LLM — scalar bridge ===")
    print(f"source: {seeds['source']}")
    for k in ("pi", "e", "phi", "gamma", "G_catalan", "K", "C_factor", "psi_con"):
        print(f"  {k} = {seeds[k]}")
    s0 = eng.scalar(D_eff=25, observed=False)
    s1 = eng.scalar(D_eff=25, observed=True)
    print(f"  S(D_eff=25, observed=False) = {s0}")
    print(f"  S(D_eff=25, observed=True)  = {s1}")
    print("FSOT fluid medium online.")


if __name__ == "__main__":
    hello()
