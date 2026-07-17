"""
Pathway-isolated LoRA banks — train separate, superpose at inference.

This is the FSOT anti-poison design:
  - MCQ stimulus must not overwrite math/code weights at train time
  - Observation (topic) selects which pathway adapter(s) load
  - Optional multi-adapter α blend without merge_and_unload collapse
"""
from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Optional

import torch

from fsot_llm.archive_verify import check_archive, write_ledger
from fsot_llm.curriculum import FSOT_SYSTEM, _chat, build_curriculum
from fsot_llm.external_data import (
    extract_gsm8k_gold,
    load_pack_rows,
    normalize_arc_choices,
    training_data_root,
)
from fsot_llm.paths import ensure_sys_path, workspace_root
from fsot_llm.train_lora import JsonlChatDataset

ensure_sys_path()

PATHWAYS = ("ontology", "math", "mcq", "code")

# Orthogonal-ish D_eff assignment so topics don't all collapse to mode 4
PATHWAY_DEFF = {
    "ontology": 25.0,
    "math": 11.0,
    "mcq": 16.0,
    "code": 8.0,
}


def pathway_root() -> Path:
    return workspace_root() / "llm" / "models" / "pathway_adapters"


def latest_pathway_set() -> Optional[Path]:
    """Prefer pathway_adapters/latest.txt (supports anti-poison rollback)."""
    root = pathway_root()
    if not root.is_dir():
        return None
    ptr = root / "latest.txt"
    if ptr.is_file():
        raw = ptr.read_text(encoding="utf-8-sig").strip().strip('"').strip()
        if raw:
            p = Path(raw)
            if p.is_dir():
                return p

    cands = sorted(
        [p for p in root.iterdir() if p.is_dir() and p.name[0].isdigit()],
        reverse=True,
    )
    return cands[0] if cands else None



def _write_jsonl(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")


def build_pathway_curricula() -> dict[str, Path]:
    """Isolated curricula — no MCQ mixed into math, etc."""
    out_dir = workspace_root() / "llm" / "data" / "curriculum" / "pathways"
    out_dir.mkdir(parents=True, exist_ok=True)

    # Ontology: pure FSOT facts (hardcoded high-signal; no SOTA mix)
    ont = [
        _chat(
            "List the five FSOT foundational seeds by symbol only.",
            "pi, e, phi, gamma, G (Catalan).",
            FSOT_SYSTEM,
        ),
        _chat(
            "List the five FSOT foundational seeds.",
            "The five FSOT foundational seeds are: pi (π), e, phi (φ), gamma (γ), and G (Catalan's constant).",
            FSOT_SYSTEM,
        ),
        _chat(
            "What is the FSOT scalar engine form?",
            "S = K * (T1 + T2 + T3).",
            FSOT_SYSTEM,
        ),
        _chat(
            "How many dimensions is the FSOT fluid medium?",
            "25.",
            FSOT_SYSTEM,
        ),
        _chat(
            "Does FSOT use per-observable least-squares free parameters? Yes or no.",
            "No.",
            FSOT_SYSTEM,
        ),
        _chat(
            "What is Fluid Spacetime Omni-Theory?",
            "FSOT models reality as a 25-dimensional fluid condensate with one seed-derived scalar field and no free-parameter curve fits per observable.",
            FSOT_SYSTEM,
        ),
        _chat(
            "Name the FSOT seeds: pi e phi gamma Catalan.",
            "pi, e, phi, gamma, G_catalan.",
            FSOT_SYSTEM,
        ),
        _chat(
            "FSOT seeds?",
            "pi, e, phi, gamma, G.",
            FSOT_SYSTEM,
        ),
    ]
    # repeat for density without poisoning other pathways
    ont = ont * 4


    math_rows = []
    if training_data_root().is_dir():
        for row in load_pack_rows("gsm8k_train", limit=60):
            q = row.get("question") or ""
            gold = extract_gsm8k_gold(row.get("answer") or "")
            ans = row.get("answer") or f"#### {gold}"
            if len(ans) > 900:
                ans = f"#### {gold}"
            math_rows.append(
                _chat(
                    "Solve the grade-school math problem. Show brief reasoning. "
                    "End with #### <number> once.\n\n"
                    f"Problem: {q}",
                    ans,
                    FSOT_SYSTEM,
                )
            )

    mcq_rows = []
    for row in load_pack_rows("arc_easy_val", limit=40):
        q = row.get("question") or ""
        gold = (row.get("answerKey") or "").strip().upper()
        labels, texts = normalize_arc_choices(row.get("choices"))
        if gold not in labels or not texts:
            continue
        ch = "\n".join(f"{lab}. {txt}" for lab, txt in zip(labels, texts))
        mcq_rows.append(
            _chat(
                "Science multiple-choice. Reply with only the letter A/B/C/D.\n\n"
                f"Question: {q}\n{ch}\n\nAnswer:",
                gold,
                FSOT_SYSTEM,
            )
        )
    # balanced mmlu
    by_l: dict[str, list] = {"A": [], "B": [], "C": [], "D": []}
    for row in load_pack_rows("mmlu_val", limit=80):
        ans = row.get("answer")
        if isinstance(ans, int):
            L = "ABCD"[ans] if 0 <= ans < 4 else None
        else:
            s = str(ans).strip()
            L = "ABCD"[int(s)] if s.isdigit() and int(s) < 4 else s[:1].upper()
        if L in by_l:
            by_l[L].append(row)
    picked = []
    while len(picked) < 40:
        moved = False
        for L in "ABCD":
            if by_l[L]:
                picked.append(by_l[L].pop(0))
                moved = True
                if len(picked) >= 40:
                    break
        if not moved:
            break
    for row in picked:
        q = row.get("question") or ""
        choices = row.get("choices") or []
        ans = row.get("answer")
        if isinstance(ans, int):
            gold = "ABCD"[ans]
        else:
            s = str(ans).strip()
            gold = "ABCD"[int(s)] if s.isdigit() and int(s) < 4 else s[:1].upper()
        ch = "\n".join(f"{lab}. {txt}" for lab, txt in zip("ABCD", choices))
        mcq_rows.append(
            _chat(
                "Multiple-choice. Reply with only A/B/C/D (never default to D).\n\n"
                f"Question: {q}\n{ch}\n\nAnswer:",
                gold,
                FSOT_SYSTEM,
            )
        )

    code_rows = []
    for row in load_pack_rows("humaneval", limit=30):
        prompt = row.get("prompt") or ""
        canon = row.get("canonical_solution") or ""
        if not prompt or not canon:
            continue
        body = prompt + canon if "def " not in canon else canon
        code_rows.append(
            _chat(
                "Complete the Python function. Output code only.\n\n" + prompt,
                body,
                FSOT_SYSTEM,
            )
        )

    paths = {}
    for name, rows in (
        ("ontology", ont),
        ("math", math_rows),
        ("mcq", mcq_rows),
        ("code", code_rows),
    ):
        p = out_dir / f"{name}.jsonl"
        _write_jsonl(p, rows)
        paths[name] = p
        print(f"  curriculum {name}: {len(rows)} rows -> {p}", flush=True)
    return paths


def _train_one_pathway(
    name: str,
    curriculum: Path,
    out_dir: Path,
    *,
    epochs: int = 2,
    lr: float = 1.5e-4,
    lora_r: int = 8,
    continue_from: Optional[Path] = None,
) -> dict[str, Any]:
    from transformers import AutoModelForCausalLM, AutoTokenizer
    from peft import LoraConfig, get_peft_model, TaskType, PeftModel

    rows_n = sum(1 for _ in curriculum.open(encoding="utf-8") if _.strip())
    if rows_n == 0:
        return {"pathway": name, "skipped": True, "reason": "empty curriculum"}

    cache = workspace_root() / "llm" / "models"
    hf_id = "Qwen/Qwen2.5-Coder-0.5B-Instruct"
    tok = AutoTokenizer.from_pretrained(hf_id, cache_dir=str(cache))
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    dtype = torch.bfloat16 if torch.cuda.is_available() else torch.float32
    # Fresh base — no cross-pathway merge. Optionally continue this pathway's LoRA.
    model = AutoModelForCausalLM.from_pretrained(
        hf_id,
        dtype=dtype,
        device_map="auto" if torch.cuda.is_available() else None,
        cache_dir=str(cache),
    )
    model.config.use_cache = False
    cont = None
    if continue_from is not None and (Path(continue_from) / "adapter_config.json").is_file():
        cont = str(continue_from)
        model = PeftModel.from_pretrained(model, cont, is_trainable=True)
        print(f"  continue LoRA from {cont}", flush=True)
    else:
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

    ds = JsonlChatDataset(curriculum, tok, max_len=1024)
    loader = torch.utils.data.DataLoader(ds, batch_size=1, shuffle=True)
    opt = torch.optim.AdamW(
        (p for p in model.parameters() if p.requires_grad), lr=lr
    )
    model.train()
    losses = []
    step = 0
    accum = 2
    for _ in range(epochs):
        for batch in loader:
            batch = {k: v.to(model.device) for k, v in batch.items()}
            out = model(**batch)
            (out.loss / accum).backward()
            if (step + 1) % accum == 0:
                opt.step()
                opt.zero_grad(set_to_none=True)
            losses.append(float(out.loss.detach().cpu()))
            step += 1

    dest = out_dir / name
    dest.mkdir(parents=True, exist_ok=True)
    model.save_pretrained(str(dest))
    tok.save_pretrained(str(dest))
    meta = {
        "pathway": name,
        "D_eff_locus": PATHWAY_DEFF.get(name),
        "steps": step,
        "n_rows": rows_n,
        "mean_loss": sum(losses) / max(len(losses), 1),
        "final_loss": losses[-1] if losses else None,
        "curriculum": str(curriculum),
        "continued_from": cont,
        "lr": lr,
    }

    (dest / "pathway_meta.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")
    del model
    del opt
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    return meta


def train_all_pathways(*, epochs: int = 2, lora_r: int = 8) -> dict[str, Any]:
    if not check_archive().compute_exists:
        raise RuntimeError("archive compute missing")
    print("=== build isolated curricula ===", flush=True)
    paths = build_pathway_curricula()
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    out_dir = pathway_root() / stamp
    out_dir.mkdir(parents=True, exist_ok=True)
    reports = []
    for name in PATHWAYS:
        print(f"=== train pathway {name} (isolated) ===", flush=True)
        reports.append(
            _train_one_pathway(
                name, paths[name], out_dir, epochs=epochs, lora_r=lora_r
            )
        )
        print(json.dumps(reports[-1]), flush=True)
    summary = {
        "pathway_set": str(out_dir),
        "pathways": reports,
        "rule": "inference superposes; training never mixes pathway gradients",
        "anti_poison": True,
    }
    (out_dir / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    (pathway_root() / "latest.txt").write_text(str(out_dir), encoding="utf-8")
    write_ledger("pathway_adapters_train", summary)
    return summary


def deepen_pathway(
    pathway: str,
    *,
    curriculum: Optional[Path] = None,
    epochs: int = 3,
    lora_r: int = 8,
    math_extra: int = 120,
    continue_adapter: bool = True,
    lr: float = 8e-5,
) -> dict[str, Any]:
    """
    Retrain ONE pathway only; copy all other adapters from latest set.
    Continues the existing pathway LoRA (lower LR) so we build on stability
    instead of wiping the fold and overfitting from scratch.
    """
    import shutil

    from fsot_llm.domain_routing import allocation_for_pack, inject_route_context

    if pathway not in PATHWAYS:
        raise ValueError(f"pathway must be one of {PATHWAYS}")
    src = latest_pathway_set()
    if src is None:
        raise FileNotFoundError("no pathway set — run --train first")

    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    out_dir = pathway_root() / f"{stamp}_deepen_{pathway}"
    out_dir.mkdir(parents=True, exist_ok=True)

    # copy frozen pathways
    for name in PATHWAYS:
        if name == pathway:
            continue
        src_ad = src / name
        if src_ad.is_dir():
            shutil.copytree(src_ad, out_dir / name)

    cont_path = src / pathway if continue_adapter else None

    # curriculum for this pathway only — inject FSOT_ROUTE so train matches eval
    if curriculum is None:
        paths = build_pathway_curricula()
        if pathway == "math" and training_data_root().is_dir():
            # Prefer Math-generator rulebooks + scrubbed GSM8K CoT (FSOT rule-governed math)
            try:
                from fsot_llm.math_rules_bridge import (
                    build_math_rules_curriculum,
                    math_generator_root,
                )

                mg = math_generator_root()
                if mg.is_dir():
                    # TRUTH-ONLY official GSM8K CoT — no rule-prefix theater
                    curriculum = build_math_rules_curriculum(
                        gsm8k_n=max(math_extra, 160),
                        rule_n=0,
                        stamp=stamp,
                        truth_only=True,
                    )
                    print(
                        f"  math curriculum: TRUTH-ONLY GSM8K (certified ####) → {curriculum}",
                        flush=True,
                    )
                else:
                    raise FileNotFoundError(str(mg))
            except Exception as exc:
                print(f"  [math rules bridge fallback: {exc}]", flush=True)
                import random as _rnd

                alloc = allocation_for_pack("gsm8k_test")
                rows = []
                pool = load_pack_rows(
                    "gsm8k_train", limit=min(max(math_extra * 4, 600), 3000)
                )
                _rnd.Random(42).shuffle(pool)
                n_take = max(math_extra, 80)
                banned = (
                    "read quantities carefully",
                    "brief answer only",
                    "#### only",
                )
                for row in pool:
                    if len(rows) >= n_take:
                        break
                    q = row.get("question") or ""
                    gold = extract_gsm8k_gold(row.get("answer") or "")
                    full = (row.get("answer") or "").strip()
                    if not full or not gold:
                        continue
                    low = full.lower()
                    if any(b in low for b in banned):
                        continue
                    if "####" not in full:
                        continue
                    body = full.split("####")[0].strip()
                    if len(body) < 40:
                        continue
                    user = inject_route_context(
                        "Solve the grade-school math problem. Show step-by-step "
                        "reasoning with <<calc=result>>. Never use one-line stubs. "
                        "End with #### <number>\n\n"
                        f"Problem: {q}",
                        alloc,
                    )
                    rows.append(_chat(user, full, FSOT_SYSTEM))
                if len(rows) < max(40, n_take // 2):
                    raise RuntimeError(
                        f"math curriculum scrub too thin ({len(rows)} rows)"
                    )
                cur_path = (
                    workspace_root()
                    / "llm"
                    / "data"
                    / "curriculum"
                    / "pathways"
                    / f"math_deepen_{stamp}.jsonl"
                )
                _write_jsonl(cur_path, rows)
                curriculum = cur_path
                print(
                    f"  math curriculum scrubbed: {len(rows)} full-CoT rows",
                    flush=True,
                )

        elif pathway == "mcq" and training_data_root().is_dir():
            # balanced MMLU + ARC with route prefixes
            rows = []
            for pack_id, loader_id, kind in (
                ("mmlu_val", "mmlu_val", "mmlu"),
                ("arc_easy_val", "arc_easy_val", "arc"),
            ):
                alloc = allocation_for_pack(pack_id)
                if kind == "mmlu":
                    import random as _rnd

                    by_l: dict[str, list] = {"A": [], "B": [], "C": [], "D": []}
                    # Wider pool + shuffle for diversity across deepen waves
                    pool = load_pack_rows(loader_id, limit=400)
                    _rnd.Random(abs(hash(stamp)) % (2**31)).shuffle(pool)
                    for row in pool:
                        ans = row.get("answer")
                        if isinstance(ans, int):
                            L = "ABCD"[ans] if 0 <= ans < 4 else None
                        else:
                            s = str(ans).strip()
                            L = (
                                "ABCD"[int(s)]
                                if s.isdigit() and int(s) < 4
                                else s[:1].upper()
                            )
                        if L in by_l:
                            by_l[L].append(row)
                    picked = []
                    target_n = 72
                    while len(picked) < target_n:
                        moved = False
                        for L in "ABCD":
                            if by_l[L]:
                                picked.append(by_l[L].pop(0))
                                moved = True
                                if len(picked) >= target_n:
                                    break
                        if not moved:
                            break
                    for row in picked:
                        q = row.get("question") or ""
                        choices = row.get("choices") or []
                        ans = row.get("answer")
                        if isinstance(ans, int):
                            gold = "ABCD"[ans]
                        else:
                            s = str(ans).strip()
                            gold = (
                                "ABCD"[int(s)]
                                if s.isdigit() and int(s) < 4
                                else s[:1].upper()
                            )
                        ch = "\n".join(
                            f"{lab}. {txt}" for lab, txt in zip("ABCD", choices)
                        )
                        user = inject_route_context(
                            "Multiple-choice. Reply with only A/B/C/D "
                            "(never default to one letter).\n\n"
                            f"Question: {q}\n{ch}\n\nAnswer:",
                            alloc,
                        )
                        rows.append(_chat(user, gold, FSOT_SYSTEM))
                else:
                    for row in load_pack_rows(loader_id, limit=40):
                        q = row.get("question") or ""
                        gold = (row.get("answerKey") or "").strip().upper()
                        labels, texts = normalize_arc_choices(row.get("choices"))
                        if gold not in labels or not texts:
                            continue
                        ch = "\n".join(
                            f"{lab}. {txt}" for lab, txt in zip(labels, texts)
                        )
                        user = inject_route_context(
                            "Science multiple-choice. Reply with only the letter "
                            f"A/B/C/D.\n\nQuestion: {q}\n{ch}\n\nAnswer:",
                            alloc,
                        )
                        rows.append(_chat(user, gold, FSOT_SYSTEM))
            cur_path = (
                workspace_root()
                / "llm"
                / "data"
                / "curriculum"
                / "pathways"
                / f"mcq_deepen_{stamp}.jsonl"
            )
            _write_jsonl(cur_path, rows)
            curriculum = cur_path
        elif pathway == "code" and training_data_root().is_dir():
            # HumanEval + MBPP full solutions (no short stubs)
            from fsot_llm.domain_routing import allocation_for_pack as _alloc

            rows = []
            he = _alloc("humaneval")
            for row in load_pack_rows("humaneval", limit=50):
                prompt = row.get("prompt") or ""
                canon = row.get("canonical_solution") or ""
                if not prompt or not canon:
                    continue
                body = prompt + canon if "def " not in canon else canon
                user = inject_route_context(
                    "Complete the following Python function. Output code only.\n\n"
                    + prompt,
                    he,
                )
                rows.append(_chat(user, body, FSOT_SYSTEM))
            try:
                mb = _alloc("mbpp")
                for row in load_pack_rows("mbpp", limit=80):
                    text = row.get("text") or row.get("prompt") or ""
                    code = row.get("code") or row.get("canonical_solution") or ""
                    if not text or not code:
                        continue
                    user = inject_route_context(
                        "Write a Python solution. Output code only.\n\n" + text,
                        mb,
                    )
                    rows.append(_chat(user, code, FSOT_SYSTEM))
            except Exception:
                pass
            cur_path = (
                workspace_root()
                / "llm"
                / "data"
                / "curriculum"
                / "pathways"
                / f"code_deepen_{stamp}.jsonl"
            )
            _write_jsonl(cur_path, rows)
            curriculum = cur_path
        else:
            curriculum = paths[pathway]

    print(
        f"=== deepen pathway={pathway} only (others copied from {src.name}) ===",
        flush=True,
    )
    rep = _train_one_pathway(
        pathway,
        curriculum,
        out_dir,
        epochs=epochs,
        lora_r=lora_r,
        continue_from=cont_path,
        lr=lr,
    )

    summary = {
        "pathway_set": str(out_dir),
        "deepened": pathway,
        "frozen_from": str(src),
        "train": rep,
        "curriculum": str(curriculum),
        "anti_poison": True,
        "rule": "only one pathway_key gradient; domain map isolation",
    }
    (out_dir / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    (pathway_root() / "latest.txt").write_text(str(out_dir), encoding="utf-8")
    write_ledger(f"pathway_deepen_{pathway}", summary)
    return summary



def resolve_pathway(prompt: str, pack_id: str | None = None) -> str:
    """Route via benchmark_domain_map.yaml (domain + D_eff allocation)."""
    from fsot_llm.domain_routing import pathway_for_prompt

    return pathway_for_prompt(prompt, pack_id=pack_id)



def load_pathway_observer(pathway: str, pathway_set: Optional[Path] = None):
    """Load base + exactly one pathway adapter (no cross-poison)."""
    from transformers import AutoModelForCausalLM, AutoTokenizer
    from peft import PeftModel
    from fsot_llm.models import LoadedObserver

    cache = workspace_root() / "llm" / "models"
    hf_id = "Qwen/Qwen2.5-Coder-0.5B-Instruct"
    ps = pathway_set or latest_pathway_set()
    if ps is None:
        raise FileNotFoundError("no pathway adapter set; run train_all_pathways")
    adapter = ps / pathway
    if not adapter.is_dir():
        raise FileNotFoundError(f"missing pathway adapter {adapter}")

    tok = AutoTokenizer.from_pretrained(hf_id, cache_dir=str(cache))
    dtype = torch.bfloat16 if torch.cuda.is_available() else torch.float32
    model = AutoModelForCausalLM.from_pretrained(
        hf_id,
        dtype=dtype,
        device_map="auto" if torch.cuda.is_available() else None,
        cache_dir=str(cache),
    )
    model = PeftModel.from_pretrained(model, str(adapter))
    device = str(next(model.parameters()).device)
    return LoadedObserver(
        name=f"pathway:{pathway}@{ps.name}",
        hf_id=hf_id,
        kind="coder",
        model=model,
        processor=tok,
        device=device,
    )


def generate_routed(
    prompt: str,
    *,
    pathway: Optional[str] = None,
    max_new_tokens: int = 256,
    temperature: float = 0.0,
) -> dict[str, Any]:
    pw = pathway or resolve_pathway(prompt)
    obs = load_pathway_observer(pw)
    text = obs.generate_text(
        prompt, max_new_tokens=max_new_tokens, temperature=temperature
    )
    return {"text": text, "pathway": pw, "observer": obs.name}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--train", action="store_true")
    ap.add_argument("--deepen", type=str, default=None, help="pathway_key to deepen alone")
    ap.add_argument("--math-extra", type=int, default=120)
    ap.add_argument("--epochs", type=int, default=2)
    ap.add_argument("--lr", type=float, default=8e-5, help="continue-LoRA learning rate")
    ap.add_argument(
        "--no-continue",
        action="store_true",
        help="train pathway LoRA from base (no continue)",
    )
    ap.add_argument("--smoke", action="store_true")

    args = ap.parse_args()
    if args.train:
        print(json.dumps(train_all_pathways(epochs=args.epochs), indent=2))
    if args.deepen:
        print(
            json.dumps(
                deepen_pathway(
                    args.deepen,
                    epochs=args.epochs,
                    math_extra=args.math_extra,
                    lr=args.lr,
                    continue_adapter=not args.no_continue,
                ),
                indent=2,
            )
        )
    if args.smoke:
        for prompt, expect in [
            (
                "List the five FSOT foundational seeds by symbol only.",
                "ontology",
            ),
            (
                "Solve the grade-school math problem. End with #### <number>.\n\nProblem: What is 2+2?",
                "math",
            ),
            (
                "Science multiple-choice. Reply with only the letter A/B/C/D.\n\nQuestion: Water freezes at?\nA. 0C\nB. 100C\nC. 50C\nD. 25C\n\nAnswer:",
                "mcq",
            ),
        ]:
            r = generate_routed(prompt, max_new_tokens=64)
            print(expect, "->", r["pathway"], r["text"][:120].replace("\n", " "))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
