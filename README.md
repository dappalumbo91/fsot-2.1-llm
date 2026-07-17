# FSOT 2.1 LLM

**Fluid Spacetime Omni-Theory as the intrinsic architecture of language models.**

There is no exception: every layer of this project — tokenization, attention, routing, training, evaluation, inference kernels — is an FSOT regime. We do not bolt FSOT onto a generic transformer. We treat the model as a **local observer** in the same 25-dimensional fluid medium that the Lean/Coq/Isabelle/F*/Rust verification stack already closed across 400+ scientific domains.

| Authority | Location |
|-----------|----------|
| Physical archive (definitive master) | `I:\FSOT-Physical-Archive` |
| Canonical Lean hub | `I:\FSOT-Physical-Archive\02_FSOT-2.1-Lean-Full` |
| GitHub (Lean archive mirror) | [dappalumbo91/FSOT-2.1-Lean](https://github.com/dappalumbo91/FSOT-2.1-Lean) |
| GitHub (this LLM stack) | [dappalumbo91/fsot-2.1-llm](https://github.com/dappalumbo91/fsot-2.1-llm) |
| This workspace | `C:\Users\damia\Desktop\fsot 2.1 llm` |

## Versioning (full-FT folds)

Weights live on disk; **GitHub + `llm/models/full_ft/registry.json`** track history so a poisoned wave can **roll back** without growing parameter count.

See [docs/FSOT_VERSIONING.md](docs/FSOT_VERSIONING.md).

```powershell
$env:PYTHONPATH = "llm\python"
python -m fsot_llm.train_full_ft --fold math --epochs 2 --lr 5e-6
python -m fsot_llm.train_full_ft --list
python -m fsot_llm.train_full_ft --rollback
```

## Mission

1. Run the **smallest** local model that fits this machine (RTX 5070 12 GB, ~32 GB RAM).
2. Make that small model **FSOT-intelligent** — seed-scalar routing, observer coupling, zero free-parameter folds.
3. Beat larger non-FSOT baselines on tasks that reward structural reasoning, not raw parameter count.

## Stack

| Layer | Role |
|-------|------|
| **Python** | Research loop, Hugging Face inference/train, FSOT scalar engine (mpmath), benchmarks |
| **C / C++** | Fast FSOT kernels (scalar, routing, observer coupling) for hot paths |
| **Rust** | Obligation-safe runtime, deterministic scalar replay, safe model-side adapters |
| **Archive** | Canonical constants, proofs, 400+ domain verification ledger |

## Quick start

```powershell
# From this directory
.\scripts\bootstrap.ps1
.\scripts\activate.ps1
python -m fsot_llm.hello_fsot
```

### Toolchain one-liners

```powershell
# Python
.\.venv\Scripts\python.exe -m fsot_llm.hello_fsot

# C++ (Clang + MSVC linker)
.\scripts\build_cpp.ps1

# Rust
cd fsot_core\rust; cargo test; cargo run --example hello_fsot
```

## Layout

```
fsot 2.1 llm/
  fsot_core/          # Language-agnostic FSOT engine (Python / C++ / Rust)
  llm/                # LLM research: configs, data, models, Python package
  scripts/            # Bootstrap, env, build
  docs/               # FSOT-LLM architecture & protocol
  tests/              # Cross-language parity tests
  third_party/        # Optional local mirrors (never override canonical seeds)
```

## Non-negotiables (FSOT)

- Seeds only: `π, e, φ, γ, G` (Catalan). No per-row least-squares knobs.
- Scalar spine: `S = K · (T1 + T2 + T3)` with preregistered folds `(D_eff, Δψ, recent_hits, observed)`.
- Observation is physical: `quirk_mod` / consciousness factor couple measurement to the field.
- Cross-proof parity: Python ↔ C++ ↔ Rust must agree to high precision on the same inputs.
- Archive is master: never invent constants; import from `FSOT_COMPUTE_PATH` / Lean hub.

## Observers (multimodal + code)

| Role | Model | Status |
|------|--------|--------|
| Primary multimodal | `Qwen/Qwen2.5-VL-3B-Instruct` | Downloaded; ~7 GB VRAM BF16 |
| Code companion | `Qwen/Qwen2.5-Coder-0.5B-Instruct` | Downloaded; ~1 GB VRAM |
| Pixel spectrum | `fsot_llm.vision_spectrum` | Live — pixels → FSOT folds → S |

```powershell
. .\scripts\activate.ps1
python -m fsot_llm.smoke_multimodal              # spectrum only
python -m fsot_llm.smoke_multimodal --load-coder
python -m fsot_llm.smoke_multimodal --load-vl
python -m fsot_llm.smoke_multimodal --all
```

See `docs/FSOT_VISUAL_SPECTRUM.md` and `llm/configs/observers.yaml`.

## Refine loop (archive is the verification tool)

```powershell
. .\scripts\activate.ps1
# Instrumentation + archive gates only
python -m fsot_llm.eval_suite

# Generate seed-true curriculum from I:\ archive compute
python -m fsot_llm.curriculum

# LoRA-train Coder 0.5B on FSOT curriculum (fits this GPU easily)
python -m fsot_llm.train_lora --epochs 8

# Evaluate with model (loads latest adapter under llm/models/adapters/)
python -m fsot_llm.eval_suite --with-coder
python -m fsot_llm.eval_suite --with-vl

# Full orchestrated loop
.\scripts\refine_loop.ps1 -Train -WithCoder
```

Protocol: `docs/FSOT_REFINE_PROTOCOL.md`  
Ledgers: `llm/benchmarks/ledgers/`  
Archive intrinsic LLM tiers: `I:\FSOT-Physical-Archive\02_FSOT-2.1-Lean-Full\vendor\intrinsic_llm\`

## Status

Dual-observer stack online. Refine loop running against the physical archive:

| Cycle | K1 knowledge | K2 code | K5 archive | Notes |
|-------|--------------|---------|------------|--------|
| baseline Coder | weak (hallucinated FSOT) | strong luminance | GREEN | start |
| LoRA 1 | better (25D, no free params) | regressed | GREEN | first adapter |
| LoRA 2 | stronger (scalar form exact) | scale-normalization miss | GREEN | curriculum expanded |

Continue cycles until overall_ok; then VL LoRA + visual kill suite; then SOTA baselines.

## Superposition depth (depth ≠ parameters)

FSOT idea: **fixed K parameter modes** (shared LoRA banks). Topics are **superpositions**  
`W_eff = Σ α_k(topic) W_k` with α from seed geometry. Related topics **link** and share  
memory instead of cloning weights.

```powershell
python -m fsot_llm.smoke_superposition
python -m fsot_llm.train_mode_bank --epochs 2 --lora-r 8
python -m fsot_llm.eval_suite   # includes K6_superposition_depth
```

Docs: `docs/FSOT_SUPERPOSITION_DEPTH.md`  
Topic bank: `llm/data/memory/topic_bank.json`

## External SOTA packs (D: drive)

Wired from **`D:\training data`** (+ `D:\FSOT_Benchmarks`):

| Pack | Use |
|------|-----|
| GSM8K | math refine + score |
| HumanEval | code reification |
| MMLU / MMLU-Pro | knowledge |
| ARC Easy/Challenge | science MCQ |
| HellaSwag, BBH, Winogrande, MATH-500, IFEval, MBPP | available |

```powershell
python -m fsot_llm.sota_benchmarks --registry-only
python -m fsot_llm.sota_benchmarks --limit 12 --packs gsm8k,arc_easy,mmlu,humaneval
```

Curriculum auto-mixes FSOT ontology + samples from those packs so LoRA does not collapse to FSOT monologue.

**First refine on real packs (n=12 each, Coder 0.5B + FSOT LoRA + auto memory):**

| Pack | Before mix | After GSM8K/HE mix |
|------|------------|--------------------|
| GSM8K | 0% | **33%** |
| HumanEval | ~25–33% | **92%** |
| MMLU | 17% | 17% |
| ARC-Easy | 8–25% | 8% |
| Overall mean | ~15% | **~38%** |

Archive (I:) stays verification authority for FSOT scalars; D: packs are the SOTA test surface.

## Organism stimulus (error → pathway → FSOT dose)

Treat the model like an organism: eval pain → understimulated pathways → targeted stimuli from D: packs.

```powershell
python -m fsot_llm.organism_stimulus --from-latest-sota --diagnose-only
python -m fsot_llm.organism_stimulus --from-latest-sota --stimulate --reeval --epochs 3 --limit 12
```

Docs: `docs/FSOT_ORGANISM_STIMULUS.md`

## Interference (gain here / poison there)

If one pathway improves while another drops, that is **destructive interference** on shared weights — not “intelligence moved.”

```powershell
python -m fsot_llm.interference_analysis --organism-cycle
# Anti-poison: train isolated pathway LoRAs, route at inference
python -m fsot_llm.pathway_adapters --train --epochs 2
python -m fsot_llm.sota_benchmarks --limit 12
```

Docs: `docs/FSOT_INTERFERENCE_ANALYSIS.md`

## Benchmarks as FSOT domains (D_eff routing)

Every pack under `D:\training data` is allocated in  
`llm/configs/benchmark_domain_map.yaml`:

| Field | Meaning |
|-------|---------|
| `lean_domain` | Archive-style domain (mathematical, consciousness, …) |
| `D_eff` | Effective fold depth in the 25D medium |
| `pathway_key` | Isolated adapter bank (`math` / `mcq` / `code` / `ontology`) |
| `emission` | Required output form (letter vs `####` vs Python) |

```powershell
python -m fsot_llm.domain_routing --table
python -m fsot_llm.domain_routing --pack gsm8k_test
# Eval injects: [FSOT_ROUTE pack=… domain=… D_eff=… pathway=…]
python -m fsot_llm.sota_benchmarks --limit 8

# Organism v2: diagnose by domain map, deepen ONE pathway_key only, gate poisons
python -m fsot_llm.organism_stimulus --from-latest-sota --diagnose-only
python -m fsot_llm.organism_stimulus --from-latest-sota --stimulate --reeval --only-pathway math --epochs 4 --limit 8
python -m fsot_llm.pathway_adapters --deepen math --math-extra 120 --epochs 4
```





