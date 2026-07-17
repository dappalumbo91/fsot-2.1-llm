"""
Train K shared FSOT modes (LoRA banks). Topics superpose onto modes — no new
full model per topic.

Pipeline:
  1. Build topic bank + curriculum tagged by dominant mode
  2. For each mode k, LoRA-train on examples whose topic couples strongly to k
  3. Save mode adapters under llm/models/mode_bank/mode_{k}/
"""
from __future__ import annotations

import argparse
import json
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import torch
from torch.utils.data import Dataset

from fsot_llm.archive_verify import check_archive, write_ledger
from fsot_llm.curriculum import build_curriculum, write_curriculum
from fsot_llm.paths import ensure_sys_path, workspace_root
from fsot_llm.superposition import (
    DEFAULT_K_MODES,
    TopicMemoryBank,
    build_default_bank,
    effective_depth_metric,
)
from fsot_llm.superposed_generate import resolve_topic

ensure_sys_path()


class JsonlChatDataset(Dataset):
    def __init__(self, rows: list[dict], tokenizer, max_len: int = 768):
        self.rows = rows
        self.tok = tokenizer
        self.max_len = max_len

    def __len__(self) -> int:
        return max(len(self.rows), 1)

    def __getitem__(self, idx: int) -> dict[str, torch.Tensor]:
        row = self.rows[idx % len(self.rows)]
        messages = row["messages"]
        text = self.tok.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=False
        )
        enc = self.tok(
            text,
            truncation=True,
            max_length=self.max_len,
            padding="max_length",
            return_tensors="pt",
        )
        input_ids = enc["input_ids"].squeeze(0)
        attn = enc["attention_mask"].squeeze(0)
        labels = input_ids.clone()
        labels[attn == 0] = -100
        return {
            "input_ids": input_ids,
            "attention_mask": attn,
            "labels": labels,
        }


def _tag_rows_by_mode(
    rows: list[dict[str, Any]], bank: TopicMemoryBank
) -> dict[int, list[dict]]:
    """Assign each curriculum row to its dominant mode via topic resolve."""
    buckets: dict[int, list[dict]] = defaultdict(list)
    for row in rows:
        # user content
        user = ""
        for m in row.get("messages", []):
            if m.get("role") == "user":
                user = m.get("content") or ""
                break
        tid = resolve_topic(bank, user)
        if tid not in bank.entries:
            tid = "scalar"
        alpha = bank.alpha(tid)
        dom = int(alpha.argmax())
        # soft multi-assign: also add to second mode if mass high
        buckets[dom].append(row)
        order = list(reversed(alpha.argsort()))
        if len(order) > 1 and float(alpha[order[1]]) > 0.15:
            buckets[int(order[1])].append(row)
    return buckets


def _train_one_mode(
    *,
    mode_id: int,
    rows: list[dict],
    base_id: str,
    cache: Path,
    out_dir: Path,
    epochs: int,
    lr: float,
    lora_r: int,
) -> dict[str, Any]:
    from transformers import AutoModelForCausalLM, AutoTokenizer
    from peft import LoraConfig, get_peft_model, TaskType

    if not rows:
        return {"mode_id": mode_id, "skipped": True, "reason": "no rows"}

    tok = AutoTokenizer.from_pretrained(base_id, cache_dir=str(cache))
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token

    dtype = torch.bfloat16 if torch.cuda.is_available() else torch.float32
    model = AutoModelForCausalLM.from_pretrained(
        base_id,
        dtype=dtype,
        device_map="auto" if torch.cuda.is_available() else None,
        cache_dir=str(cache),
    )
    model.config.use_cache = False
    lora = LoraConfig(
        task_type=TaskType.CAUSAL_LM,
        r=lora_r,
        lora_alpha=32,
        lora_dropout=0.05,
        target_modules=[
            "q_proj",
            "k_proj",
            "v_proj",
            "o_proj",
            "gate_proj",
            "up_proj",
            "down_proj",
        ],
    )
    model = get_peft_model(model, lora)

    ds = JsonlChatDataset(rows, tok)
    loader = torch.utils.data.DataLoader(ds, batch_size=1, shuffle=True)
    opt = torch.optim.AdamW(
        (p for p in model.parameters() if p.requires_grad), lr=lr
    )

    model.train()
    losses: list[float] = []
    steps = 0
    grad_accum = 2
    for _ in range(epochs):
        for batch in loader:
            batch = {k: v.to(model.device) for k, v in batch.items()}
            out = model(**batch)
            (out.loss / grad_accum).backward()
            if (steps + 1) % grad_accum == 0:
                opt.step()
                opt.zero_grad(set_to_none=True)
            losses.append(float(out.loss.detach().cpu()))
            steps += 1

    mode_dir = out_dir / f"mode_{mode_id}"
    mode_dir.mkdir(parents=True, exist_ok=True)
    model.save_pretrained(str(mode_dir))
    tok.save_pretrained(str(mode_dir))
    meta = {
        "mode_id": mode_id,
        "n_rows": len(rows),
        "steps": steps,
        "mean_loss": sum(losses) / max(len(losses), 1),
        "final_loss": losses[-1] if losses else None,
    }
    (mode_dir / "mode_meta.json").write_text(
        json.dumps(meta, indent=2), encoding="utf-8"
    )
    # free VRAM
    del model
    del opt
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    return meta


def train_mode_bank(
    *,
    k_modes: int = DEFAULT_K_MODES,
    epochs: int = 2,
    lr: float = 2e-4,
    lora_r: int = 8,
    max_modes: int | None = None,
) -> dict[str, Any]:
    st = check_archive()
    if not st.compute_exists:
        raise RuntimeError(f"archive missing: {st.notes}")

    bank = build_default_bank()
    bank_path = bank.save()
    depth = effective_depth_metric(bank)

    # curriculum + linked-memory augmented copies
    base_rows = build_curriculum()
    write_curriculum()
    # augment: prepend linked memory for each user turn
    aug: list[dict] = []
    for row in base_rows:
        msgs = row["messages"]
        user = next((m for m in msgs if m["role"] == "user"), None)
        if not user:
            continue
        tid = resolve_topic(bank, user["content"])
        linked = bank.linked_context(tid)
        new_user = (
            f"FSOT linked memory (superposition depth, fixed modes):\n{linked}\n\n"
            f"{user['content']}"
        )
        new_msgs = []
        for m in msgs:
            if m is user:
                new_msgs.append({**m, "content": new_user})
            else:
                new_msgs.append(m)
        aug.append({"messages": new_msgs, "topic_id": tid})
    all_rows = base_rows + aug

    buckets = _tag_rows_by_mode(all_rows, bank)
    cache = workspace_root() / "llm" / "models"
    out_dir = (
        workspace_root()
        / "llm"
        / "models"
        / "mode_bank"
        / datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    )
    out_dir.mkdir(parents=True, exist_ok=True)

    base_id = "Qwen/Qwen2.5-Coder-0.5B-Instruct"
    mode_reports = []
    mode_ids = sorted(buckets.keys())
    if max_modes is not None:
        mode_ids = mode_ids[:max_modes]

    for mid in mode_ids:
        print(f"=== training mode {mid} rows={len(buckets[mid])} ===", flush=True)
        rep = _train_one_mode(
            mode_id=mid,
            rows=buckets[mid],
            base_id=base_id,
            cache=cache,
            out_dir=out_dir,
            epochs=epochs,
            lr=lr,
            lora_r=lora_r,
        )
        mode_reports.append(rep)
        print(json.dumps(rep), flush=True)

    # modes with no data: skip (superposition still uses available modes)
    summary = {
        "mode_bank_dir": str(out_dir),
        "k_modes_design": k_modes,
        "modes_trained": mode_reports,
        "topic_bank": str(bank_path),
        "depth_metric": depth,
        "base": base_id,
        "note": (
            "New topics add bank rows only; parameter modes stay fixed. "
            "Inference superposes trained modes via alpha(topic)."
        ),
    }
    (out_dir / "bank_summary.json").write_text(
        json.dumps(summary, indent=2), encoding="utf-8"
    )
    write_ledger("train_mode_bank", summary)
    # pointer
    latest = workspace_root() / "llm" / "models" / "mode_bank" / "latest.txt"
    latest.write_text(str(out_dir), encoding="utf-8")
    return summary


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--epochs", type=int, default=2)
    ap.add_argument("--lora-r", type=int, default=8)
    ap.add_argument("--max-modes", type=int, default=None)
    args = ap.parse_args()
    rep = train_mode_bank(
        epochs=args.epochs, lora_r=args.lora_r, max_modes=args.max_modes
    )
    print(json.dumps(rep, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
