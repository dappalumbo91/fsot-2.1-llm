# FSOT 2.1 LLM — Versioning & Rollback

**Ontology:** Fluid Spacetime Omni-Theory  
**Rule:** On failure, **circle back** to the last pure fold state. Never “fix” by growing parameter count.

## Why version

Full fine-tune updates the whole observer fluid. A bad wave can poison emissions (math stubs, code collapse). Git + a **checkpoint registry** give:

1. **Code history** (GitHub) — training scripts, domain map, rule bridge  
2. **Weight history** (local `llm/models/full_ft/`) — full checkpoints, gitignored  
3. **Registry pointer** (`registry.json`) — which version is active, parent for rollback  

## Layout

```text
llm/models/full_ft/
  registry.json          # tracked in git
  latest.txt             # path to active full model
  2026..._fullft_math/   # full HF weights (NOT in git)
    model.safetensors
    fsot_version_manifest.json
    fsot_train_meta.json
```

## Commands

```powershell
cd "C:\Users\damia\Desktop\fsot 2.1 llm"
$env:PYTHONPATH = "llm\python"

# Full-FT math fold (bottom-up from Qwen base)
python -m fsot_llm.train_full_ft --fold math --epochs 2 --lr 5e-6 --math-extra 280

# List versions
python -m fsot_llm.train_full_ft --list

# After eval gate passes
# (promote is also --promote on train, prefer post-eval)
python -c "from fsot_llm.version_registry import promote; promote('VERSION_ID')"

# Anti-poison fail → roll back
python -m fsot_llm.train_full_ft --rollback
```

## Git workflow

```powershell
git add -A
git commit -m "fsot: describe fold change"
git tag -a "fold-math-YYYYMMDD" -m "math full-FT candidate"
git push origin main --tags
```

Weights stay local. Tags mark **code + registry** that pair with a `version_id`.

## Eval gate (required before promote)

Match n, packs, and **routed** observer when comparing:

```powershell
python -m fsot_llm.sota_benchmarks --limit 16 --packs gsm8k,arc_easy,mmlu,humaneval
```

Promote only if no pathway drops more than poison ε (0.05).

## GitHub

https://github.com/dappalumbo91/fsot-2.1-llm
