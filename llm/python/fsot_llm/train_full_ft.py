"""
FSOT full fine-tune of Qwen2.5-Coder-0.5B-Instruct (all weights).

Fold-structured training — not industry multi-task soup.
Default: math fold via Math-generator rules + scrubbed GSM8K CoT.

Ontology: Fluid Spacetime Omni-Theory.
  - No N-param growth as a refinement lever
  - Full-FT embeds quantified structure into the observer fluid
  - Version registry enables rollback when anti-poison fails
"""
from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Optional

import torch

from fsot_llm.archive_verify import check_archive, write_ledger
from fsot_llm.paths import ensure_sys_path, workspace_root
from fsot_llm.train_lora import JsonlChatDataset
from fsot_llm.version_registry import (
    full_ft_root,
    get_active,
    promote,
    register_version,
)

ensure_sys_path()

BASE_HF_ID = "Qwen/Qwen2.5-Coder-0.5B-Instruct"
FOLDS = ("math", "mcq", "code", "ontology")


def _build_math_curriculum(stamp: str, math_extra: int) -> Path:
    from fsot_llm.math_rules_bridge import build_math_rules_curriculum, math_generator_root

    if math_generator_root().is_dir():
        return build_math_rules_curriculum(
            gsm8k_n=max(math_extra, 200),
            rule_n=max(60, math_extra // 4),
            stamp=f"fullft_{stamp}",
        )
    # fallback scrubbed GSM8K only
    from fsot_llm.curriculum import FSOT_SYSTEM, _chat
    from fsot_llm.domain_routing import allocation_for_pack, inject_route_context
    from fsot_llm.external_data import extract_gsm8k_gold, load_pack_rows
    import random

    alloc = allocation_for_pack("gsm8k_test")
    rows = []
    pool = load_pack_rows("gsm8k_train", limit=min(math_extra * 3, 2000))
    random.Random(42).shuffle(pool)
    for row in pool:
        if len(rows) >= math_extra:
            break
        q = row.get("question") or ""
        full = (row.get("answer") or "").strip()
        if "####" not in full or len(full.split("####")[0]) < 40:
            continue
        if "read quantities carefully" in full.lower():
            continue
        user = inject_route_context(
            "Solve the grade-school math problem with pure arithmetic. "
            "Show step-by-step reasoning. End with #### <number>.\n\n"
            f"Problem: {q}",
            alloc,
        )
        rows.append(_chat(user, full, FSOT_SYSTEM))
    out = (
        workspace_root()
        / "llm"
        / "data"
        / "curriculum"
        / "pathways"
        / f"math_fullft_{stamp}.jsonl"
    )
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    return out


def train_full_fold(
    fold: str = "math",
    *,
    epochs: int = 2,
    lr: float = 5e-6,
    math_extra: int = 280,
    max_len: int = 1024,
    grad_accum: int = 8,
    continue_from: Optional[Path] = None,
    promote_after: bool = False,
) -> dict[str, Any]:
    if fold not in FOLDS:
        raise ValueError(f"fold must be one of {FOLDS}")

    st = check_archive()
    if not st.compute_exists:
        print("[warn] archive compute missing — continuing full-FT anyway", flush=True)

    from transformers import AutoModelForCausalLM, AutoTokenizer

    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    version_id = f"{stamp}_fullft_{fold}"
    out_dir = full_ft_root() / version_id
    out_dir.mkdir(parents=True, exist_ok=True)

    # Curriculum by fold
    if fold == "math":
        curriculum = _build_math_curriculum(stamp, math_extra)
    else:
        # reuse pathway deepen curricula builders via deepen dry path
        from fsot_llm.pathway_adapters import deepen_pathway

        # build curriculum only by calling build helpers — for mcq/code use deepen's file
        # simpler: invoke math_rules only for math; for others use existing deepen jsonl builders
        raise NotImplementedError(
            f"full-FT fold={fold} — start with math; mcq/code next wave"
        )

    print(f"=== FSOT FULL-FT fold={fold} curriculum={curriculum} ===", flush=True)
    cache = workspace_root() / "llm" / "models"
    base = str(continue_from) if continue_from and Path(continue_from).is_dir() else BASE_HF_ID
    if continue_from:
        print(f"  continue full weights from {continue_from}", flush=True)
    else:
        print(f"  bottom-up from base {BASE_HF_ID}", flush=True)

    tok = AutoTokenizer.from_pretrained(
        base if Path(base).is_dir() else BASE_HF_ID,
        cache_dir=str(cache),
        trust_remote_code=True,
    )
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token

    dtype = torch.bfloat16 if torch.cuda.is_available() else torch.float32
    model = AutoModelForCausalLM.from_pretrained(
        base if Path(base).is_dir() else BASE_HF_ID,
        dtype=dtype,
        device_map="auto" if torch.cuda.is_available() else None,
        cache_dir=str(cache),
        trust_remote_code=True,
    )
    model.config.use_cache = False
    # FULL parameter train — all weights require grad
    for p in model.parameters():
        p.requires_grad = True
    n_train = sum(p.numel() for p in model.parameters() if p.requires_grad)
    n_all = sum(p.numel() for p in model.parameters())
    print(f"  trainable {n_train}/{n_all} ({100*n_train/max(n_all,1):.1f}%)", flush=True)

    ds = JsonlChatDataset(curriculum, tok, max_len=max_len)
    loader = torch.utils.data.DataLoader(ds, batch_size=1, shuffle=True)
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=0.01)
    model.train()
    losses: list[float] = []
    step = 0
    opt.zero_grad(set_to_none=True)
    for ep in range(epochs):
        for batch in loader:
            batch = {k: v.to(model.device) for k, v in batch.items()}
            out = model(**batch)
            loss = out.loss / grad_accum
            loss.backward()
            if (step + 1) % grad_accum == 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                opt.step()
                opt.zero_grad(set_to_none=True)
            losses.append(float(out.loss.detach().cpu()))
            step += 1
            if step % 50 == 0:
                print(
                    f"  step {step} ep {ep+1}/{epochs} loss={losses[-1]:.4f}",
                    flush=True,
                )

    # final step if leftover
    if step % grad_accum != 0:
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()

    print(f"=== save full model → {out_dir} ===", flush=True)
    model.save_pretrained(str(out_dir), safe_serialization=True)
    tok.save_pretrained(str(out_dir))

    train_meta = {
        "fold": fold,
        "base": base,
        "curriculum": str(curriculum),
        "epochs": epochs,
        "lr": lr,
        "steps": step,
        "n_rows": len(ds),
        "mean_loss": sum(losses) / max(len(losses), 1),
        "final_loss": losses[-1] if losses else None,
        "trainable_params": n_train,
        "all_params": n_all,
        "dtype": str(dtype),
    }
    (out_dir / "fsot_train_meta.json").write_text(
        json.dumps(train_meta, indent=2), encoding="utf-8"
    )

    parent = None
    act = get_active()
    if act:
        parent = act.get("version_id")

    entry = register_version(
        version_id=version_id,
        fold=fold,
        path=out_dir,
        parent=parent,
        notes=f"FSOT full-FT fold={fold}; rules+GSM8K; rollback via registry",
        train_meta=train_meta,
    )
    if promote_after:
        promote(version_id)
        entry["status"] = "promoted"

    summary = {
        "version_id": version_id,
        "path": str(out_dir),
        "entry": entry,
        "train": train_meta,
        "ontology": "FSOT 2.1 full-fold fine-tune",
    }
    write_ledger(f"full_ft_{fold}", summary)
    del model
    del opt
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    return summary


def main() -> int:
    ap = argparse.ArgumentParser(description="FSOT full fine-tune by fold")
    ap.add_argument("--fold", default="math", choices=list(FOLDS))
    ap.add_argument("--epochs", type=int, default=2)
    ap.add_argument("--lr", type=float, default=5e-6)
    ap.add_argument("--math-extra", type=int, default=280)
    ap.add_argument(
        "--continue-from",
        default=None,
        help="path to prior full-FT checkpoint (fold chain)",
    )
    ap.add_argument(
        "--promote",
        action="store_true",
        help="mark version active immediately (prefer after eval gate)",
    )
    ap.add_argument("--list", action="store_true", help="list version registry")
    ap.add_argument("--rollback", action="store_true", help="rollback to parent version")
    args = ap.parse_args()

    if args.list:
        from fsot_llm.version_registry import list_versions, load_registry

        print(json.dumps(load_registry(), indent=2))
        return 0
    if args.rollback:
        from fsot_llm.version_registry import rollback

        print(json.dumps(rollback(), indent=2))
        return 0

    cont = Path(args.continue_from) if args.continue_from else None
    rep = train_full_fold(
        args.fold,
        epochs=args.epochs,
        lr=args.lr,
        math_extra=args.math_extra,
        continue_from=cont,
        promote_after=args.promote,
    )
    print(json.dumps(rep, indent=2)[:4000])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
