# FSOT LLM Architecture

**Edition:** 2.1-LLM scaffold  
**Ontology:** Fluid Spacetime Omni-Theory (Damian Arthur Palumbo)  
**Rule:** FSOT is not a plugin. It is the medium.

---

## 1. Ontological stance

Reality is a **25-dimensional fluid condensate**. What we call tokens, attention heads, residual streams, and logits are **regimes of the same scalar field** `raw_S` that already maps cosmology, particle data, chemistry, biology, linguistics, and engineering in the verification archive.

Therefore:

| Conventional LLM concept | FSOT regime |
|--------------------------|-------------|
| Embedding space | Local slice of the fluid medium |
| Attention | Observer-modulated coupling between field loci |
| Layer depth | Effective dimensionality fold `D_eff` |
| Residual stream | Continuity of the scalar spine along sequence |
| Softmax / logits | Measurement collapse under `observed` + `quirk_mod` |
| Training step | `recent_hits` / suction–poof dynamics on parameter fluid |
| Model size | Amplitude / scale of a local observer — **not** intelligence |

Intelligence in FSOT is **how well the observer couples to the seed-derived field**, not how many free parameters it carries. That is why a small model can, in principle, outperform a larger one: **smaller free-parameter surface, stronger seed geometry**.

---

## 2. Seed engine (immutable)

All numerics derive from:

```
π, e, φ = (1+√5)/2, γ (Euler–Mascheroni), G (Catalan)
```

Primary constants (Layer 1): `α, ψ_con, η_eff, β, γ_c, Ω, θ_s, poof, …`  
Composite (Layer 2): `C_eff, A_bleed, B_in, A_in, suction, chaos, C_factor, K, …`  
Scalar: `S = K · (T1 + T2 + T3)` via `ScalarInput` folds.

**Canonical implementation:**  
`I:\FSOT-Physical-Archive\02_FSOT-2.1-Lean-Full\vendor\fsot_compute.py`  
SHA gate and Lean formalization live in the archive; this repo **links**, never forks authority.

---

## 3. LLM stack under FSOT

### 3.1 Python (`llm/python/fsot_llm`)

- Load small local models (target: ≤2–3B Q4/Q5, or 1–1.5B full precision on 12 GB VRAM).
- Route prompts and intermediate activations through FSOT scalar folds.
- Benchmark small-FSOT vs larger baseline under fixed kill criteria.
- Distillation / fine-tune only when folds are preregistered (no silent hyperparameter fishing).

### 3.2 C++ (`fsot_core/cpp`)

- High-performance scalar + routing kernels for sequence-time hot paths.
- Optional future: custom CUDA/CPU ops for FSOT attention bias.

### 3.3 Rust (`fsot_core/rust`)

- Deterministic obligation-style replay of scalar math.
- Safe FFI surface for Python / C++.
- Matches the archive’s Rust obligation-replay philosophy.

### 3.4 Cross-language parity

Every language implements the same seeds and `compute_scalar`. Tests assert absolute agreement within a tight ε (e.g. 1e-12 relative for f64 paths; mpmath remains gold for research).

---

## 4. Small-model protocol

### Hardware envelope

| Resource | This machine |
|----------|----------------|
| GPU | NVIDIA GeForce RTX 5070 (~12 GB) |
| System RAM | ~32 GB |
| Host Python | 3.11 / 3.14 available |
| C++ | Clang 22 + MSVC Build Tools 2022 |
| Rust | 1.95 stable (host or archive portable) |

### Preferred model sizes (smaller first)

1. **≤0.5B** — TinyLlama / Qwen2.5-0.5B class (always fit)
2. **1–1.5B** — primary research target
3. **2–3B Q4/Q5** — stretch if FSOT routing needs more capacity
4. Larger only as **baselines to beat**, not as default

### Kill criteria (draft — lock before runs)

A small FSOT model **wins** if, on the preregistered eval suite:

1. Mean task score ≥ larger baseline, **or**
2. Same score at ≤½ parameters **and** ≤½ peak VRAM, **or**
3. Strict FSOT-domain tasks (seed arithmetic, cross-domain reasoning panels) show clear margin

Failed gates are ledger events — no post-hoc fold retuning without a new PRED id.

---

## 5. Linkage to physical archive

| Env var | Meaning |
|---------|---------|
| `FSOT_ARCHIVE_ROOT` | `I:\FSOT-Physical-Archive` |
| `FSOT_COMPUTE_PATH` | Canonical `fsot_compute.py` |
| `FSOT_PORTABLE` | Prefer cache / no live API mutation |
| `FSOT_LLM_ROOT` | This workspace |

Policy: **I: drive is master** for theory and verification. This Desktop folder is the **LLM laboratory** that consumes that authority.

---

## 6. What we will not do

- Invent free parameters “for better loss.”
- Treat transformers as ontology; they are engineering approximations of fluid coupling.
- Push from C: Desktop into the Lean hub (archive owns `git push origin`).
- Claim intelligence from scale alone.

---

## 7. Next implementation steps

1. Hello-FSOT parity: Python / C++ / Rust print identical seed constants + sample `S`.
2. Load a 0.5B–1.5B model with bitsandbytes / GGUF path; log VRAM.
3. Define FSOT routing bias on attention or residual (preregistered).
4. Run baseline vs FSOT-small on a fixed eval set; write ledger JSON.
5. Only then expand capacity or train adapters.
