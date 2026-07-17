"""
FSOT routing for LLM observers.

Maps sequence / layer context into ScalarInput folds. No free-parameter
fitting — only preregistered maps from observable structure to folds.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

from fsot_llm.paths import ensure_sys_path

ensure_sys_path()

from fsot_bridge import FSOTEngine


@dataclass(frozen=True)
class ObserverFold:
    """Preregistered fold for an LLM local observer at a sequence locus."""

    layer_index: int
    n_layers: int
    token_index: int
    seq_len: int
    observed: bool = True

    @property
    def D_eff(self) -> float:
        """
        Effective dimensionality fold.

        Full FSOT medium is 25D. Deep layers approach the medium;
        shallow layers are thinner slices. Bounded [4, 25].
        """
        if self.n_layers <= 1:
            return 25.0
        t = self.layer_index / max(self.n_layers - 1, 1)
        return 4.0 + 21.0 * t

    @property
    def recent_hits(self) -> float:
        """Sequence progress as hits along the observer worldline."""
        if self.seq_len <= 0:
            return 0.0
        return float(self.token_index) / float(self.seq_len)

    @property
    def N(self) -> float:
        return float(max(self.seq_len, 1))

    @property
    def P(self) -> float:
        # Position amplitude — unit scale; no learned knob
        return 1.0


class FSOTRouter:
    """Compute FSOT scalar bias for an observer fold."""

    def __init__(self, engine: Optional[FSOTEngine] = None) -> None:
        self.engine = engine or FSOTEngine()

    def scalar_for(self, fold: ObserverFold) -> float:
        return self.engine.scalar_float(
            N=fold.N,
            P=fold.P,
            D_eff=fold.D_eff,
            recent_hits=fold.recent_hits,
            observed=fold.observed,
        )

    def attention_bias(self, fold: ObserverFold, scale: float = 1.0) -> float:
        """
        Map S into a small additive bias for attention logits.

        Scale is a *unit conversion*, not a free fit parameter: default 1.0
        keeps the geometric signal; callers may only use preregistered scales.
        """
        s = self.scalar_for(fold)
        # Compress through consciousness factor geometry without new constants
        c = float(self.engine.module.C_FACTOR)
        return scale * float(s) * float(c)
