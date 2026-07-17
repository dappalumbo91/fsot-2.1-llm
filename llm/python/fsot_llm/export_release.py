"""
Export a leaderboard-ready single HuggingFace model from pathway adapters.

Open LLM Leaderboard / lm-eval / Kaggle expect:
  AutoTokenizer + AutoModelForCausalLM.from_pretrained(path)

Multi-pathway isolation stays the research system; this module produces the
public "FSOT-Instruct" product checkpoint by:
  1) weighted-merge of pathway LoRA tensors
  2) PEFT load on Qwen2.5-Coder-0.5B-Instruct
  3) merge_and_unload → full dense weights
  4) model card + release meta for hub/Kaggle submit
"""
from __future__ import annotations

import argparse
import json
import shutil
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Optional

import torch

from fsot_llm.archive_verify import check_archive, write_ledger
from fsot_llm.paths import ensure_sys_path, workspace_root
from fsot_llm.pathway_adapters import PATHWAYS, latest_pathway_set, pathway_root

ensure_sys_path()

BASE_HF_ID = "Qwen/Qwen2.5-Coder-0.5B-Instruct"
RELEASE_NAME = "FSOT-2.1-Instruct-0.5B"

# Performance-tilted blend from current best (code/mcq strong; math fragile; ontology light)
DEFAULT_WEIGHTS = {
    "code": 0.32,
    "mcq": 0.35,
    "math": 0.25,
    "ontology": 0.08,
}


def release_root() -> Path:
    return workspace_root() / "llm" / "models" / "release"


def _load_adapter_tensors(adapter_dir: Path) -> dict[str, torch.Tensor]:
    st = adapter_dir / "adapter_model.safetensors"
    bn = adapter_dir / "adapter_model.bin"
    if st.is_file():
        from safetensors.torch import load_file

        return load_file(str(st))
    if bn.is_file():
        return torch.load(bn, map_location="cpu")
    raise FileNotFoundError(f"no adapter weights in {adapter_dir}")


def weighted_merge_pathway_adapters(
    pathway_set: Path,
    *,
    weights: Optional[dict[str, float]] = None,
    out_dir: Path,
) -> dict[str, Any]:
    """Σ_p w_p · ΔW_p  (normalized). Writes a single PEFT adapter dir."""
    weights = dict(weights or DEFAULT_WEIGHTS)
    # only pathways that exist
    present = [p for p in PATHWAYS if (pathway_set / p).is_dir()]
    if not present:
        raise FileNotFoundError(f"no pathway dirs in {pathway_set}")
    wsum = sum(float(weights.get(p, 0.0)) for p in present)
    if wsum <= 0:
        raise ValueError("all pathway weights are zero")
    norm = {p: float(weights.get(p, 0.0)) / wsum for p in present}

    merged: dict[str, torch.Tensor] = {}
    for p in present:
        w = norm[p]
        if w < 1e-9:
            continue
        tensors = _load_adapter_tensors(pathway_set / p)
        for k, v in tensors.items():
            v = v.float() * w
            if k not in merged:
                merged[k] = v
            else:
                merged[k] = merged[k] + v

    out_dir.mkdir(parents=True, exist_ok=True)
    # adapter config from strongest pathway (mcq default)
    cfg_src = None
    for pref in ("mcq", "code", "math", "ontology"):
        c = pathway_set / pref / "adapter_config.json"
        if c.is_file():
            cfg_src = c
            break
    if cfg_src is None:
        raise FileNotFoundError("no adapter_config.json in pathway set")
    cfg = json.loads(cfg_src.read_text(encoding="utf-8"))
    cfg["base_model_name_or_path"] = BASE_HF_ID
    cfg["inference_mode"] = True
    (out_dir / "adapter_config.json").write_text(
        json.dumps(cfg, indent=2), encoding="utf-8"
    )

    try:
        from safetensors.torch import save_file

        save_file(merged, str(out_dir / "adapter_model.safetensors"))
    except Exception:
        torch.save(merged, out_dir / "adapter_model.bin")

    meta = {
        "pathway_set": str(pathway_set),
        "weights_raw": weights,
        "weights_normalized": norm,
        "pathways_merged": present,
        "n_tensors": len(merged),
    }
    (out_dir / "merge_meta.json").write_text(
        json.dumps(meta, indent=2), encoding="utf-8"
    )
    return meta


def _model_card(
    *,
    release_name: str,
    pathway_set: str,
    weights: dict[str, float],
    smoke: dict[str, Any],
    refine_scores: Optional[dict[str, float]],
) -> str:
    scores_block = ""
    if refine_scores:
        scores_block = (
            "\n### Internal refine-loop scores (n≈16, not official leaderboard)\n\n"
            "| Pack | Accuracy |\n|------|----------|\n"
            + "\n".join(
                f"| {k} | {v:.1%} |" for k, v in refine_scores.items()
            )
            + "\n\nThese are **not** lm-eval full-set numbers. Run the official harness "
            "before claiming leaderboard results.\n"
        )
    return f"""---
license: apache-2.0
base_model: {BASE_HF_ID}
tags:
  - fsot
  - fsot-2.1
  - qwen2.5
  - code
  - instruct
  - peft-merged
  - text-generation
language:
  - en
library_name: transformers
pipeline_tag: text-generation
---

# {release_name}

**FSOT 2.1 Instruct (0.5B)** — open instruct model derived from
[`{BASE_HF_ID}`](https://huggingface.co/{BASE_HF_ID}) with
**FSOT-guided pathway post-training** (isolated math / MCQ / code / ontology
LoRAs, domain-map routing during research training, anti-poison gates).

This release is a **single dense checkpoint** (pathway LoRAs weighted-merged
then `merge_and_unload`) so it loads with standard:

```python
from transformers import AutoModelForCausalLM, AutoTokenizer
tok = AutoTokenizer.from_pretrained("PATH_OR_HUB_ID", trust_remote_code=True)
model = AutoModelForCausalLM.from_pretrained("PATH_OR_HUB_ID", trust_remote_code=True)
```

Compatible with **lm-evaluation-harness**, Hugging Face model hub upload,
and Kaggle Models / notebooks.

## Method (short)

- Base: Qwen2.5-Coder-0.5B-Instruct
- Training: pathway-isolated LoRA (r=8) on FSOT-routed curricula
- Research system keeps adapters separate at inference (anti-poison)
- **Leaderboard export**: weighted merge of pathway adapters → merge into base
- FSOT influence: foundational seeds / domain map / D_eff routing during train;
  release is a standard causal LM

### Merge weights used for this file

```json
{json.dumps(weights, indent=2)}
```

Source pathway set: `{pathway_set}`
{scores_block}
## Intended use

- Small-model instruct / code / science QA experiments
- Baseline for FSOT routing research vs merged export
- Community leaderboard submissions (after full lm-eval)

## Limitations

- 0.5B parameter scale — not frontier SOTA overall
- Merged export can lag the multi-adapter research stack on some packs
- Math (GSM8K) remains the weakest pathway in refine loops
- Always report **full harness** numbers, not n=16 refine scores

## License

Apache-2.0 for this merge packaging and adapters contribution, subject to the
base model license (Qwen). Cite both.

## Citation

```
@software{{fsot_21_instruct_05b,
  title  = {{FSOT 2.1 Instruct 0.5B}},
  year   = {{2026}},
  note   = {{Derived from Qwen2.5-Coder-0.5B-Instruct with FSOT pathway post-training}}
}}
```

## Smoke

```json
{json.dumps(smoke, indent=2)}
```
"""


KAGGLE_OWNER = "damianpalumbo"
KAGGLE_MODEL_SLUG = "fsot-21-instruct-05b"
KAGGLE_INSTANCE = f"{KAGGLE_OWNER}/{KAGGLE_MODEL_SLUG}/PyTorch/transformers"


def stage_kaggle_package(release_dir: Path) -> Path:
    """Copy only publishable files into llm/models/release/kaggle_package/."""
    pkg = release_root() / "kaggle_package"
    pkg.mkdir(parents=True, exist_ok=True)
    keep = (
        "model.safetensors",
        "config.json",
        "generation_config.json",
        "tokenizer.json",
        "tokenizer_config.json",
        "chat_template.jinja",
        "README.md",
        "fsot_release_meta.json",
    )
    for name in keep:
        src = release_dir / name
        if src.is_file():
            shutil.copy2(src, pkg / name)
    # preserve instance metadata if present
    meta_src = pkg / "model-instance-metadata.json"
    if not meta_src.is_file():
        meta = {
            "ownerSlug": KAGGLE_OWNER,
            "modelSlug": KAGGLE_MODEL_SLUG,
            "instanceSlug": "transformers",
            "framework": "pyTorch",
            "overview": (
                "Dense FSOT-2.1 instruct merge: load with transformers "
                "AutoModelForCausalLM. Hosted on Kaggle."
            ),
            "usage": (
                "# Load\n```python\nfrom transformers import AutoModelForCausalLM, "
                "AutoTokenizer\npath = "
                "'/kaggle/input/fsot-21-instruct-05b/pytorch/transformers/1'\n```\n"
            ),
            "licenseName": "Apache 2.0",
            "fineTunable": True,
            "trainingData": [],
            "modelInstanceType": "Unspecified",
            "baseModelInstance": "",
            "externalBaseModelUrl": "",
        }
        meta_src.write_text(json.dumps(meta, indent=2), encoding="utf-8")
    return pkg


def push_kaggle_version(
    release_dir: Path,
    *,
    notes: str = "FSOT refine promote — desktop sync",
) -> dict[str, Any]:
    """Stage package and create a new Kaggle model instance version."""
    import subprocess

    pkg = stage_kaggle_package(release_dir)
    cmd = [
        "kaggle",
        "models",
        "instances",
        "versions",
        "create",
        KAGGLE_INSTANCE,
        "-p",
        str(pkg),
        "-n",
        notes,
        "-r",
        "skip",
    ]
    print("=== kaggle push ===", " ".join(cmd), flush=True)
    proc = subprocess.run(cmd, capture_output=True, text=True)
    out = (proc.stdout or "") + (proc.stderr or "")
    print(out[-2000:] if len(out) > 2000 else out, flush=True)
    result = {
        "ok": proc.returncode == 0 or "created" in out.lower() or "Upload successful" in out,
        "returncode": proc.returncode,
        "instance": KAGGLE_INSTANCE,
        "url": f"https://www.kaggle.com/models/{KAGGLE_OWNER}/{KAGGLE_MODEL_SLUG}/PyTorch/transformers",
        "notes": notes,
        "package": str(pkg),
        "log_tail": out[-1500:],
    }
    write_ledger("kaggle_publish", result)
    return result


def export_release(
    *,
    pathway_set: Optional[Path] = None,
    name: str = RELEASE_NAME,
    weights: Optional[dict[str, float]] = None,
    push_hub: Optional[str] = None,
    push_kaggle: bool = False,
    kaggle_notes: str = "FSOT refine promote — desktop sync",
    dtype_name: str = "bfloat16",
) -> dict[str, Any]:
    """Build full HF model under llm/models/release/<name>/."""
    archive = check_archive()
    src = Path(pathway_set) if pathway_set else latest_pathway_set()
    if src is None or not src.is_dir():
        raise FileNotFoundError("no pathway set — train/deepen first")

    weights = dict(weights or DEFAULT_WEIGHTS)
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    out = release_root() / name
    peft_dir = out / "_merged_peft"
    if out.exists():
        # keep previous under stamp backup pointer only if full model present
        pass
    out.mkdir(parents=True, exist_ok=True)

    print(f"=== weighted-merge pathways from {src.name} ===", flush=True)
    merge_meta = weighted_merge_pathway_adapters(
        src, weights=weights, out_dir=peft_dir
    )
    print(json.dumps(merge_meta, indent=2), flush=True)

    from transformers import AutoModelForCausalLM, AutoTokenizer
    from peft import PeftModel

    cache = workspace_root() / "llm" / "models"
    torch_dtype = {
        "bfloat16": torch.bfloat16,
        "float16": torch.float16,
        "float32": torch.float32,
    }.get(dtype_name, torch.bfloat16)
    if not torch.cuda.is_available() and torch_dtype == torch.bfloat16:
        torch_dtype = torch.float32

    print(f"=== load base {BASE_HF_ID} + PEFT merge_and_unload ===", flush=True)
    tok = AutoTokenizer.from_pretrained(BASE_HF_ID, cache_dir=str(cache))
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    model = AutoModelForCausalLM.from_pretrained(
        BASE_HF_ID,
        dtype=torch_dtype,
        device_map="auto" if torch.cuda.is_available() else None,
        cache_dir=str(cache),
    )
    model = PeftModel.from_pretrained(model, str(peft_dir))
    model = model.merge_and_unload()
    model.eval()

    # smoke generation
    smoke_prompt = (
        "Complete the Python function. Output code only.\n\n"
        "def add(a, b):\n"
    )
    messages = [
        {
            "role": "system",
            "content": "You are FSOT-Instruct, a careful coding and reasoning assistant.",
        },
        {"role": "user", "content": smoke_prompt},
    ]
    text = tok.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True
    )
    inputs = tok(text, return_tensors="pt")
    if torch.cuda.is_available():
        inputs = {k: v.to(model.device) for k, v in inputs.items()}
    with torch.no_grad():
        gen = model.generate(
            **inputs,
            max_new_tokens=64,
            do_sample=False,
            pad_token_id=tok.pad_token_id,
        )
    out_ids = gen[0][inputs["input_ids"].shape[-1] :]
    smoke_text = tok.decode(out_ids, skip_special_tokens=True)
    smoke = {
        "prompt": smoke_prompt,
        "completion_preview": smoke_text[:400],
        "dtype": str(torch_dtype),
        "device": str(next(model.parameters()).device),
    }
    print("smoke:", smoke["completion_preview"][:200].replace("\n", " "), flush=True)

    print(f"=== save full model → {out} ===", flush=True)
    model.save_pretrained(str(out), safe_serialization=True)
    tok.save_pretrained(str(out))

    # refine scores from best_checkpoint ledger if present
    refine_scores = None
    best_ptr = (
        workspace_root()
        / "llm"
        / "benchmarks"
        / "ledgers"
        / "latest_best_checkpoint.json"
    )
    if best_ptr.is_file():
        try:
            best = json.loads(best_ptr.read_text(encoding="utf-8"))
            refine_scores = best.get("scores")
        except Exception:
            pass

    card = _model_card(
        release_name=name,
        pathway_set=str(src),
        weights=merge_meta["weights_normalized"],
        smoke=smoke,
        refine_scores=refine_scores,
    )
    (out / "README.md").write_text(card, encoding="utf-8")

    # generation config defaults
    gen_cfg = {
        "max_new_tokens": 512,
        "do_sample": False,
        "temperature": 0.0,
        "transformers_version": None,
    }
    try:
        import transformers

        gen_cfg["transformers_version"] = transformers.__version__
    except Exception:
        pass
    (out / "fsot_release_meta.json").write_text(
        json.dumps(
            {
                "name": name,
                "base_model": BASE_HF_ID,
                "pathway_set": str(src),
                "merge": merge_meta,
                "exported_at": datetime.now(timezone.utc).isoformat(),
                "stamp": stamp,
                "smoke": smoke,
                "refine_scores_n16": refine_scores,
                "leaderboard": {
                    "load": "AutoModelForCausalLM.from_pretrained",
                    "lm_eval": (
                        f"lm_eval --model hf --model_args pretrained={out.as_posix()} "
                        "--tasks mmlu,gsm8k,arc_easy,hellaswag,humaneval "
                        "--batch_size auto --device cuda:0"
                    ),
                    "hub_submit": "huggingface-cli upload USER/FSOT-2.1-Instruct-0.5B "
                    + str(out),
                },
                "archive_ok": bool(getattr(archive, "ok", False)),
            },
            indent=2,
        ),
        encoding="utf-8",
    )

    # pointer + copy peft sidecar optional keep
    release_root().mkdir(parents=True, exist_ok=True)
    (release_root() / "latest.txt").write_text(str(out), encoding="utf-8")

    # hub push optional
    hub_url = None
    if push_hub:
        try:
            from huggingface_hub import HfApi

            api = HfApi()
            api.create_repo(push_hub, exist_ok=True, private=False)
            api.upload_folder(
                folder_path=str(out),
                repo_id=push_hub,
                repo_type="model",
            )
            hub_url = f"https://huggingface.co/{push_hub}"
            print("pushed:", hub_url, flush=True)
        except Exception as exc:
            print(f"[hub push failed: {exc}]", flush=True)

    kaggle_rep = None
    if push_kaggle:
        kaggle_rep = push_kaggle_version(out, notes=kaggle_notes)

    summary = {
        "release_dir": str(out),
        "name": name,
        "base_model": BASE_HF_ID,
        "pathway_set": str(src),
        "merge": merge_meta,
        "smoke": smoke,
        "hub_url": hub_url,
        "kaggle": kaggle_rep,
        "lm_eval_hint": (
            f'lm_eval --model hf --model_args pretrained="{out}" '
            "--tasks leaderboard_mmlu_pro,gsm8k,arc_easy,hellaswag "
            "--batch_size auto"
        ),
    }
    write_ledger("export_release", summary)
    # free VRAM
    del model
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    return summary


def write_lm_eval_script(release_dir: Optional[Path] = None) -> Path:
    """Write a Windows-friendly runner for official harness tasks."""
    root = workspace_root()
    scripts = root / "llm" / "scripts"
    scripts.mkdir(parents=True, exist_ok=True)
    rel = release_dir or Path(
        (release_root() / "latest.txt").read_text(encoding="utf-8-sig").strip()
        if (release_root() / "latest.txt").is_file()
        else release_root() / RELEASE_NAME
    )
    # PowerShell
    ps1 = scripts / "run_lm_eval_release.ps1"
    ps1.write_text(
        f"""# Official-style eval for FSOT release (install once: pip install lm_eval)
$ErrorActionPreference = "Stop"
$Model = if ($args[0]) {{ $args[0] }} else {{ "{rel.as_posix()}" }}
$Tasks = if ($args[1]) {{ $args[1] }} else {{ "mmlu,gsm8k,arc_easy,hellaswag,humaneval_instruct" }}
$Out = Join-Path (Split-Path $Model -Parent) "lm_eval_results"
New-Item -ItemType Directory -Force -Path $Out | Out-Null
Write-Host "Model: $Model"
Write-Host "Tasks: $Tasks"
lm_eval --model hf `
  --model_args "pretrained=$Model,dtype=float16,trust_remote_code=True" `
  --tasks $Tasks `
  --batch_size auto `
  --apply_chat_template `
  --fewshot_as_multiturn `
  --output_path $Out `
  --log_samples `
  --confirm_run_unsafe_code
Write-Host "Results -> $Out"
""",
        encoding="utf-8",
    )
    # bash
    sh = scripts / "run_lm_eval_release.sh"
    sh.write_text(
        f"""#!/usr/bin/env bash
set -euo pipefail
MODEL="${{1:-{rel.as_posix()}}}"
TASKS="${{2:-mmlu,gsm8k,arc_easy,hellaswag,humaneval_instruct}}"
OUT="$(dirname "$MODEL")/lm_eval_results"
mkdir -p "$OUT"
lm_eval --model hf \\
  --model_args "pretrained=$MODEL,dtype=float16,trust_remote_code=True" \\
  --tasks "$TASKS" \\
  --batch_size auto \\
  --apply_chat_template \\
  --fewshot_as_multiturn \\
  --output_path "$OUT" \\
  --log_samples \\
  --confirm_run_unsafe_code
echo "Results -> $OUT"
""",
        encoding="utf-8",
    )
    # hub + kaggle checklist
    checklist = scripts / "LEADERBOARD_SUBMIT.md"
    checklist.write_text(
        f"""# Leaderboard submit path — {RELEASE_NAME}

## 1. Local release (done by `python -m fsot_llm.export_release`)

- Full HF model: `llm/models/release/{RELEASE_NAME}/`
- Must load with `AutoModelForCausalLM` / `AutoTokenizer` only

## 2. Official numbers (required before claiming board ranks)

```powershell
pip install lm_eval
.\\llm\\scripts\\run_lm_eval_release.ps1
```

Or full Open-LLM-Leaderboard-style task groups if available in your lm_eval version:

```powershell
lm_eval --model hf --model_args pretrained="llm/models/release/{RELEASE_NAME}" --tasks leaderboard --batch_size auto
```

## 3. Hugging Face Hub (primary public ID)

```powershell
pip install huggingface_hub
huggingface-cli login
huggingface-cli upload YOUR_USER/{RELEASE_NAME} "llm/models/release/{RELEASE_NAME}"
```

Or during export:

```powershell
python -m fsot_llm.export_release --push-hub YOUR_USER/{RELEASE_NAME}
```

Then open the model page → request / document evals on Open LLM Leaderboard
(when submissions are open) and paste lm-eval tables into the model card.

## 4. Kaggle Models (secondary mirror + notebooks)

```powershell
kaggle models create -p llm/models/release/{RELEASE_NAME} --model-slug fsot-21-instruct-05b
# or dataset + notebook that runs lm_eval on GPU
```

## 5. Naming

- Public name: **{RELEASE_NAME}**
- Base disclosure: `{BASE_HF_ID}`
- Your contribution: FSOT pathway post-training + merge export

## 6. Do not claim n=16 refine scores as leaderboard results
""",
        encoding="utf-8",
    )
    return ps1


def main() -> int:
    ap = argparse.ArgumentParser(description="Export leaderboard-ready FSOT release")
    ap.add_argument(
        "--pathway-set",
        default=None,
        help="path to pathway adapter set (default: latest.txt)",
    )
    ap.add_argument("--name", default=RELEASE_NAME)
    ap.add_argument(
        "--push-hub",
        default=None,
        help="OPTIONAL HF repo id (not used in Kaggle-first workflow)",
    )
    ap.add_argument(
        "--push-kaggle",
        action="store_true",
        help="After export, upload a new Kaggle model version (desktop↔Kaggle sync)",
    )
    ap.add_argument(
        "--kaggle-notes",
        default="FSOT refine promote — desktop sync",
        help="Version notes for Kaggle model version",
    )
    ap.add_argument(
        "--dtype",
        default="bfloat16",
        choices=["bfloat16", "float16", "float32"],
    )
    ap.add_argument(
        "--scripts-only",
        action="store_true",
        help="only write lm-eval / submit scripts",
    )
    ap.add_argument(
        "--kaggle-only",
        action="store_true",
        help="Stage + push existing release dir to Kaggle (no re-merge)",
    )
    args = ap.parse_args()
    if args.scripts_only:
        p = write_lm_eval_script()
        print("wrote", p)
        return 0
    if args.kaggle_only:
        rel = (
            Path(args.pathway_set)
            if args.pathway_set
            else Path(
                (release_root() / "latest.txt").read_text(encoding="utf-8-sig").strip()
            )
        )
        # pathway-set flag reused as release path if --kaggle-only
        if args.name and (release_root() / args.name).is_dir():
            rel = release_root() / args.name
        print(json.dumps(push_kaggle_version(rel, notes=args.kaggle_notes), indent=2))
        return 0
    rep = export_release(
        pathway_set=Path(args.pathway_set) if args.pathway_set else None,
        name=args.name,
        push_hub=args.push_hub,
        push_kaggle=args.push_kaggle,
        kaggle_notes=args.kaggle_notes,
        dtype_name=args.dtype,
    )
    write_lm_eval_script(Path(rep["release_dir"]))
    print(json.dumps({k: rep[k] for k in rep if k != "merge"}, indent=2)[:3000])
    print("release_dir:", rep["release_dir"])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
