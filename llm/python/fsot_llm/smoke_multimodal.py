"""
Smoke: FSOT spectrum on synthetic image + optional dual-observer inference.

Usage:
  python -m fsot_llm.smoke_multimodal              # spectrum only (fast)
  python -m fsot_llm.smoke_multimodal --load-coder # + code companion
  python -m fsot_llm.smoke_multimodal --load-vl    # + primary VL (downloads)
  python -m fsot_llm.smoke_multimodal --all
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from fsot_llm.paths import ensure_sys_path, workspace_root

ensure_sys_path()

from fsot_llm.vision_spectrum import VisualSpectrumObserver


def make_synthetic_spectrum(path: Path) -> Path:
    """Synthetic RGB field: gradient + sine interference (physical-form toy)."""
    from PIL import Image

    h, w = 256, 256
    y = np.linspace(0, 1, h)[:, None]
    x = np.linspace(0, 1, w)[None, :]
    xx = np.broadcast_to(x, (h, w)).astype(np.float64)
    yy = np.broadcast_to(y, (h, w)).astype(np.float64)
    # R: horizontal ramp, G: vertical, B: interference pattern
    r = xx
    g = yy
    b = 0.5 + 0.5 * np.sin(2 * np.pi * (3 * xx + 2 * yy))
    rgb = np.stack([r, g, b], axis=-1)
    rgb = (np.clip(rgb, 0, 1) * 255).astype(np.uint8)
    path.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(rgb, mode="RGB").save(path)
    return path


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--load-vl", action="store_true")
    ap.add_argument("--load-coder", action="store_true")
    ap.add_argument("--all", action="store_true")
    args = ap.parse_args()
    if args.all:
        args.load_vl = True
        args.load_coder = True

    root = workspace_root()
    img = make_synthetic_spectrum(root / "llm" / "data" / "synthetic_spectrum.png")
    print("=== FSOT visual spectrum ===")
    print(f"image: {img}")
    obs = VisualSpectrumObserver()
    report = obs.observe_path(img)
    print(json.dumps(report, indent=2))

    out = root / "llm" / "benchmarks" / "spectrum_smoke.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(f"wrote {out}")

    if args.load_coder:
        print("\n=== Code companion ===")
        from fsot_llm.models import load_coder_observer, vram_report

        coder = load_coder_observer()
        print(f"loaded {coder.hf_id} on {coder.device}")
        print(vram_report())
        code = coder.generate_text(
            "Write a Python function that converts an RGB uint8 image array "
            "to mean luminance using Rec.601 weights. Only code.",
            max_new_tokens=200,
            temperature=0.2,
        )
        print("--- generation ---")
        print(code)

    if args.load_vl:
        print("\n=== Primary multimodal (Qwen2.5-VL) ===")
        from fsot_llm.models import load_vl_observer, vram_report

        vl = load_vl_observer()
        print(f"loaded {vl.hf_id} on {vl.device}")
        print(vram_report())
        ans = vl.generate_vision(
            "Describe the color structure of this image as physical information "
            "states (channels, gradients, interference). Be precise and short.",
            img,
            max_new_tokens=160,
            temperature=0.2,
        )
        print("--- vision generation ---")
        print(ans)
        code_from_vision = vl.generate_vision(
            "Write a short Python snippet using numpy that could regenerate a "
            "similar RGB interference field. Only code.",
            img,
            max_new_tokens=200,
            temperature=0.2,
        )
        print("--- code-from-vision ---")
        print(code_from_vision)

    print("\nFSOT multimodal smoke done.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
