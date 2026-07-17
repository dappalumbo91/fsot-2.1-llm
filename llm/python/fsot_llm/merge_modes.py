"""
Merge K mode LoRA adapters into one effective adapter via FSOT α weights.

W_delta_eff = Σ_k α_k · W_delta_k

This is the parameter-layer superposition: same base weights, modes as
superposed layers, topics only change α (almost free in memory).
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Optional

import torch

from fsot_llm.paths import workspace_root
from fsot_llm.superposition import TopicMemoryBank, build_default_bank


def latest_mode_bank() -> Optional[Path]:
    ptr = workspace_root() / "llm" / "models" / "mode_bank" / "latest.txt"
    if ptr.is_file():
        p = Path(ptr.read_text(encoding="utf-8").strip())
        if p.is_dir():
            return p
    root = workspace_root() / "llm" / "models" / "mode_bank"
    if not root.is_dir():
        return None
    cands = sorted(
        [p for p in root.iterdir() if p.is_dir() and p.name != "merged"],
        reverse=True,
    )
    return cands[0] if cands else None


def _load_adapter_tensors(adapter_dir: Path) -> dict[str, torch.Tensor]:
    # peft saves adapter_model.safetensors or .bin
    st_path = adapter_dir / "adapter_model.safetensors"
    bin_path = adapter_dir / "adapter_model.bin"
    if st_path.is_file():
        from safetensors.torch import load_file

        return load_file(str(st_path))
    if bin_path.is_file():
        return torch.load(bin_path, map_location="cpu")
    raise FileNotFoundError(f"no adapter weights in {adapter_dir}")


def merge_modes_for_topic(
    topic_id: str,
    *,
    bank: Optional[TopicMemoryBank] = None,
    mode_bank_dir: Optional[Path] = None,
    out_dir: Optional[Path] = None,
) -> Path:
    bank = bank or build_default_bank()
    mode_bank_dir = mode_bank_dir or latest_mode_bank()
    if mode_bank_dir is None:
        raise FileNotFoundError("no mode bank trained yet")

    alpha = bank.alpha(topic_id)
    # discover trained modes
    mode_dirs = sorted(mode_bank_dir.glob("mode_*"))
    if not mode_dirs:
        raise FileNotFoundError(f"no mode_* in {mode_bank_dir}")

    merged: dict[str, torch.Tensor] = {}
    used = []
    weight_sum = 0.0
    for md in mode_dirs:
        mid = int(md.name.split("_")[1])
        if mid >= len(alpha):
            continue
        w = float(alpha[mid])
        if w < 1e-6:
            continue
        tensors = _load_adapter_tensors(md)
        used.append({"mode": mid, "alpha": w})
        weight_sum += w
        for k, v in tensors.items():
            v = v.float()
            if k not in merged:
                merged[k] = w * v
            else:
                merged[k] = merged[k] + w * v

    if weight_sum > 0:
        for k in merged:
            merged[k] = merged[k] / weight_sum

    out_dir = out_dir or (
        workspace_root()
        / "llm"
        / "models"
        / "mode_bank"
        / "merged"
        / f"topic_{topic_id}"
    )
    out_dir.mkdir(parents=True, exist_ok=True)

    # copy adapter config from first mode
    cfg_src = mode_dirs[0] / "adapter_config.json"
    if cfg_src.is_file():
        (out_dir / "adapter_config.json").write_text(
            cfg_src.read_text(encoding="utf-8"), encoding="utf-8"
        )

    try:
        from safetensors.torch import save_file

        save_file(merged, str(out_dir / "adapter_model.safetensors"))
    except Exception:
        torch.save(merged, out_dir / "adapter_model.bin")

    (out_dir / "merge_meta.json").write_text(
        json.dumps(
            {
                "topic_id": topic_id,
                "alpha": alpha.tolist(),
                "modes_used": used,
                "source_bank": str(mode_bank_dir),
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    return out_dir
