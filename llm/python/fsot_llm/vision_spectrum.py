"""
FSOT visual spectrum — pixels and information states as observation loci.

No free-parameter color fits. Folds are preregistered maps from measurable
image statistics into ScalarInput fields of the canonical engine.
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Optional, Union

import numpy as np

from fsot_llm.paths import ensure_sys_path

ensure_sys_path()

from fsot_bridge import FSOTEngine
from fsot_llm.routing import FSOTRouter, ObserverFold


PathLike = Union[str, Path]


@dataclass(frozen=True)
class SpectrumState:
    """Information state extracted from an image under FSOT folds."""

    width: int
    height: int
    mean_luminance: float  # [0, 1]
    entropy_proxy: float  # [0, 1] approx
    r_frac: float
    g_frac: float
    b_frac: float
    edge_energy: float  # [0, 1]
    # Preregistered soft trinary bins (inspired by archive photonic trinary)
    bin_low: float
    bin_mid: float
    bin_high: float

    def to_dict(self) -> dict[str, Any]:
        return {
            "width": self.width,
            "height": self.height,
            "mean_luminance": self.mean_luminance,
            "entropy_proxy": self.entropy_proxy,
            "r_frac": self.r_frac,
            "g_frac": self.g_frac,
            "b_frac": self.b_frac,
            "edge_energy": self.edge_energy,
            "bin_low": self.bin_low,
            "bin_mid": self.bin_mid,
            "bin_high": self.bin_high,
        }


def _load_rgb(path: PathLike) -> np.ndarray:
    """Load image as float RGB HxWx3 in [0,1]. Requires Pillow."""
    from PIL import Image

    img = Image.open(path).convert("RGB")
    arr = np.asarray(img, dtype=np.float64) / 255.0
    return arr


def analyze_pixels(rgb: np.ndarray) -> SpectrumState:
    """Map pixel array → FSOT spectrum information state."""
    if rgb.ndim != 3 or rgb.shape[2] != 3:
        raise ValueError(f"expected HxWx3 RGB, got {rgb.shape}")
    h, w, _ = rgb.shape
    # Luminance (Rec. 601 weights — fixed, not learned)
    lum = 0.299 * rgb[..., 0] + 0.587 * rgb[..., 1] + 0.114 * rgb[..., 2]
    mean_lum = float(np.clip(lum.mean(), 0.0, 1.0))

    # Shannon entropy proxy on 32-bin luminance histogram
    hist, _ = np.histogram(lum.ravel(), bins=32, range=(0.0, 1.0), density=True)
    hist = hist + 1e-12
    hist = hist / hist.sum()
    entropy = float(-(hist * np.log(hist)).sum() / np.log(32.0))  # normalize ~[0,1]

    channel_sum = rgb.reshape(-1, 3).sum(axis=0) + 1e-12
    fracs = channel_sum / channel_sum.sum()
    r_f, g_f, b_f = float(fracs[0]), float(fracs[1]), float(fracs[2])

    # Edge energy via simple finite differences
    gx = np.abs(np.diff(lum, axis=1)).mean() if w > 1 else 0.0
    gy = np.abs(np.diff(lum, axis=0)).mean() if h > 1 else 0.0
    edge = float(np.clip((gx + gy) * 4.0, 0.0, 1.0))

    # Soft trinary bins on luminance (low / mid / high)
    low = float((lum < 0.33).mean())
    mid = float(((lum >= 0.33) & (lum < 0.66)).mean())
    high = float((lum >= 0.66).mean())

    return SpectrumState(
        width=w,
        height=h,
        mean_luminance=mean_lum,
        entropy_proxy=entropy,
        r_frac=r_f,
        g_frac=g_f,
        b_frac=b_f,
        edge_energy=edge,
        bin_low=low,
        bin_mid=mid,
        bin_high=high,
    )


def analyze_image(path: PathLike) -> SpectrumState:
    return analyze_pixels(_load_rgb(path))


def spectrum_to_scalar_kwargs(state: SpectrumState) -> dict[str, Any]:
    """
    Preregistered map: SpectrumState → compute_scalar kwargs.

    - D_eff: thinner slices for low structure, fuller medium for high edge+entropy
    - rho: mean luminance as density proxy
    - recent_hits: edge energy (structure encounters along the worldline)
    - scale/amplitude: entropy + mid-bin dominance
    """
    structure = 0.5 * state.entropy_proxy + 0.5 * state.edge_energy
    d_eff = 4.0 + 21.0 * float(np.clip(structure, 0.0, 1.0))
    # N is observer capacity fold — use log10(pixels), not raw pixel count
    # (raw W*H would explode the N*P/sqrt(D) term without being a free fit).
    pixels = max(state.width * state.height, 1)
    n_fold = float(max(np.log10(float(pixels)), 1.0))
    return {
        "N": n_fold,
        "P": 1.0,
        "D_eff": d_eff,
        "rho": max(state.mean_luminance, 1e-6),
        "recent_hits": state.edge_energy,
        "scale": 0.5 + 0.5 * state.entropy_proxy,
        "amplitude": 0.5 + 0.5 * state.bin_mid,
        "observed": True,
    }



class VisualSpectrumObserver:
    """FSOT observer over images: spectrum state + scalar coupling."""

    def __init__(self, engine: Optional[FSOTEngine] = None) -> None:
        self.engine = engine or FSOTEngine()
        self.router = FSOTRouter(self.engine)

    def observe_path(self, path: PathLike) -> dict[str, Any]:
        state = analyze_image(path)
        return self.observe_state(state)

    def observe_state(self, state: SpectrumState) -> dict[str, Any]:
        kw = spectrum_to_scalar_kwargs(state)
        s = self.engine.scalar_float(
            N=kw["N"],
            P=kw["P"],
            D_eff=kw["D_eff"],
            recent_hits=kw["recent_hits"],
            observed=True,
            **{
                k: v
                for k, v in kw.items()
                if k
                not in ("N", "P", "D_eff", "recent_hits", "observed")
            },
        )
        # Layer-agnostic fold for routing bias
        fold = ObserverFold(
            layer_index=int(state.edge_energy * 11),
            n_layers=12,
            token_index=int(state.mean_luminance * 63),
            seq_len=64,
            observed=True,
        )
        bias = self.router.attention_bias(fold)
        return {
            "spectrum": state.to_dict(),
            "scalar_kwargs": kw,
            "S": s,
            "attention_bias": bias,
            "trinary_mass": {
                "low": state.bin_low,
                "mid": state.bin_mid,
                "high": state.bin_high,
            },
        }
