# FSOT Model Refine Protocol

**Goal:** Take the dual-observer stack (Qwen2.5-VL-3B + Coder-0.5B) and refine it with FSOT until it systematically outperforms larger non-FSOT baselines on preregistered kill criteria.

**Authority:** `I:\FSOT-Physical-Archive` is the verification tool. Desktop is the lab. Never invent constants. Never post-hoc retune folds after a failed gate.

---

## 1. Loop (every cycle)

```
generate_curriculum  →  train (LoRA / small FT)  →  eval_suite  →  archive_ledger
         ↑                                                         |
         └──────────── only promote if GREEN ←─────────────────────┘
```

| Stage | What | Gate |
|-------|------|------|
| **Curriculum** | Seed-derived Q/A, spectrum→form, code reification | Data hashes logged |
| **Train** | LoRA on Coder first; then VL LoRA (vision tower freeze optional) | Loss finite; no free-param explosion |
| **Eval** | FSOT knowledge, code, vision spectrum, archive intrinsic tiers | Per-suite kill criteria |
| **Ledger** | JSON under `llm/benchmarks/ledgers/` + optional archive panel sync | `overall_ok: true` |

## 2. Verification tool = physical archive

| Resource | Path |
|----------|------|
| Canonical compute | `I:\...\vendor\fsot_compute.py` |
| Intrinsic LLM panel | `data/intrinsic_llm_validators_benchmark.json` |
| Panel (desktop wiring) | `data/validators_intrinsic_llm_panel_benchmark.json` |
| Vendor tier results | `vendor/intrinsic_llm/benchmark_results_final.json` |
| Full verification | `python scripts/fsot_verification_runner.py` (from Lean hub, portable) |

Env (always):

```powershell
. I:\FSOT-Physical-Archive\set_fsot_archive_env.ps1
. .\scripts\set_env.ps1   # from this workspace
```

## 3. Kill criteria (preregistered — edit only with new PRED id)

### K1 — FSOT knowledge (code + text)
Model states seeds / scalar structure / observer coupling correctly vs archive gold.

### K2 — Code reification
Generated code for spectrum / luminance / scalar helpers passes unit tests.

### K3 — Visual spectrum
On fixed image set: description + structure tags agree with FSOT spectrum state (luminance bins, channel mass, edge class).

### K4 — Efficiency
Same or better score than larger baseline at ≤½ params or ≤½ VRAM (ledger comparison).

### K5 — Archive consistency
Any numeric claim about `S` must match `fsot_compute` within float gate (1e-6 relative for f64 paths; mpmath gold for published claims).

**Failed gate → debug + new curriculum slice. Do not widen free hyperparameters.**

## 4. Train feasibility (this machine)

| Model | Method | Why |
|-------|--------|-----|
| Coder-0.5B | Full FT or LoRA | ~1 GB base; primary refine target |
| VL-3B | LoRA / QLoRA, freeze vision initially | ~7 GB load; adapters in remaining headroom |
| Larger SOTA | Eval-only baselines | Not trained here |

## 5. Promotion policy

An adapter/checkpoint is **promoted** only when:

1. `eval_suite` writes `overall_ok: true`
2. K5 archive scalar checks pass
3. No criterion regressed > preregistered tolerance vs previous GREEN

Promoted weights live under `llm/models/promoted/` with ledger SHA.

## 6. SOTA ambition

Beating SOTA is **not** “scale up.” It is:

1. Compress intelligence into seed geometry  
2. Multimodal observation as physical coupling  
3. Code as reification  
4. Prove it on the archive ledger  

If a larger model scores higher on a suite, either improve FSOT routing/curriculum or accept the miss as a ledger event — never abandon FSOT for random knobs.
