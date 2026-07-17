"""
LoRA fine-tune for FSOT Coder companion (small enough for local refine).

Uses PEFT LoRA on Qwen2.5-Coder-0.5B-Instruct with the FSOT curriculum.
Vision LoRA is a later stage once K1/K2 are green.
"""
from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import torch
from torch.utils.data import Dataset

from fsot_llm.archive_verify import check_archive, write_ledger
from fsot_llm.curriculum import write_curriculum
from fsot_llm.paths import ensure_sys_path, workspace_root

ensure_sys_path()


class JsonlChatDataset(Dataset):
    def __init__(self, path: Path, tokenizer, max_len: int = 1024):
        self.rows = [
            json.loads(line)
            for line in path.read_text(encoding="utf-8").splitlines()
            if line.strip()
        ]
        self.tok = tokenizer
        self.max_len = max_len

    def __len__(self) -> int:
        return len(self.rows)

    def __getitem__(self, idx: int) -> dict[str, torch.Tensor]:
        messages = self.rows[idx]["messages"]
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


def train(
    *,
    epochs: int = 3,
    lr: float = 2e-4,
    batch_size: int = 1,
    grad_accum: int = 4,
    lora_r: int = 16,
    max_steps: int | None = None,
) -> dict[str, Any]:
    st = check_archive()
    if not st.compute_exists:
        raise RuntimeError(f"Archive not ready: {st.notes}")

    from transformers import AutoModelForCausalLM, AutoTokenizer
    from peft import LoraConfig, get_peft_model, TaskType

    curr_path = write_curriculum()
    cache = workspace_root() / "llm" / "models"
    hf_id = "Qwen/Qwen2.5-Coder-0.5B-Instruct"
    out_dir = (
        workspace_root()
        / "llm"
        / "models"
        / "adapters"
        / f"coder05_fsot_{datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ')}"
    )
    out_dir.mkdir(parents=True, exist_ok=True)

    tok = AutoTokenizer.from_pretrained(hf_id, cache_dir=str(cache))
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token

    dtype = torch.bfloat16 if torch.cuda.is_available() else torch.float32
    model = AutoModelForCausalLM.from_pretrained(
        hf_id,
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
    model.print_trainable_parameters()

    ds = JsonlChatDataset(curr_path, tok)
    loader = torch.utils.data.DataLoader(
        ds, batch_size=batch_size, shuffle=True
    )

    opt = torch.optim.AdamW(
        (p for p in model.parameters() if p.requires_grad), lr=lr
    )

    model.train()
    step = 0
    losses: list[float] = []
    for epoch in range(epochs):
        for batch in loader:
            batch = {k: v.to(model.device) for k, v in batch.items()}
            out = model(**batch)
            loss = out.loss / grad_accum
            loss.backward()
            if (step + 1) % grad_accum == 0:
                opt.step()
                opt.zero_grad(set_to_none=True)
            losses.append(float(out.loss.detach().cpu()))
            step += 1
            if max_steps is not None and step >= max_steps:
                break
        if max_steps is not None and step >= max_steps:
            break

    model.save_pretrained(str(out_dir))
    tok.save_pretrained(str(out_dir))
    (out_dir / "fsot_train_meta.json").write_text(
        json.dumps(
            {
                "base": hf_id,
                "curriculum": str(curr_path),
                "epochs": epochs,
                "steps": step,
                "mean_loss": sum(losses) / max(len(losses), 1),
                "final_loss": losses[-1] if losses else None,
                "lr": lr,
                "lora_r": lora_r,
                "archive": st.to_dict(),
            },
            indent=2,
        ),
        encoding="utf-8",
    )

    report = {
        "adapter_dir": str(out_dir),
        "steps": step,
        "mean_loss": sum(losses) / max(len(losses), 1),
        "final_loss": losses[-1] if losses else None,
        "curriculum": str(curr_path),
        "base": hf_id,
    }
    write_ledger("train_lora_coder", report)
    return report


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--epochs", type=int, default=4)
    ap.add_argument("--lr", type=float, default=2e-4)
    ap.add_argument("--max-steps", type=int, default=None)
    ap.add_argument("--lora-r", type=int, default=16)
    args = ap.parse_args()
    rep = train(
        epochs=args.epochs,
        lr=args.lr,
        max_steps=args.max_steps,
        lora_r=args.lora_r,
    )
    print(json.dumps(rep, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
