# FSOT Cross-Pathway Interference Analysis

**Question:** Why did ARC gains come with GSM8K/HumanEval losses?  
**Answer:** Proactive stimulation on one pathway **poisoned shared parameter mass** on others — not FSOT “depth,” but **mode collapse + destructive overwrite**.

---

## 1. Measured deltas (organism cycle)

| Pathway | Before | After stimulus | Δ | Interpretation |
|---------|--------|----------------|---|----------------|
| ARC-Easy | 8.3% | **33.3%** | **+25%** | MCQ letter-collapse fixed |
| MMLU | 16.7% | 16.7% | 0 | Same family, no further gain |
| GSM8K | 33.3% | 25.0% | **−8%** | Math pathway regressed |
| HumanEval | 91.7% | 83.3% | **−8%** | Code pathway regressed |
| Mean | 37.5% | 39.6% | +2% | Net up, **but uneven** |

That pattern is the signature of **shared-capacity interference**, not independent improvement.

---

## 2. Root causes (FSOT language)

### 2.1 Mode collapse — all pathways claimed dominant_mode = 4

Diagnosis assigned ARC, MMLU, GSM8K, HumanEval → **same mode**.  
In the fluid medium, that means they were not orthogonal regimes; training on one **moves the same locus**.

| Physics | Training event |
|---------|----------------|
| One spatial mode | One LoRA bank updated for all tasks |
| Constructive interference (desired) | Shared structure helps |
| Destructive interference (observed) | MCQ head habits overwrite math/code habits |

### 2.2 Stimulus mass imbalance (poison dose)

Of **123 stimulus rows**:

| Stimulus | Rows | Share |
|----------|------|-------|
| ARC MCQ | 53 | 43% |
| MMLU MCQ | 44 | 36% |
| GSM8K math | 23 | 19% |
| HumanEval code | **0** | **0%** (homeostatic — no maintenance) |

~**79% MCQ** vs 19% math vs **0% code maintenance**.  
MCQ trains: *emit a single letter, stop.*  
That is **proactive** for ARC, **poison** for GSM8K (`#### number`) and HumanEval (long code).

### 2.3 Destructive merge (`merge_and_unload`)

Stimulus train did:

1. Load prior FSOT adapter  
2. **`merge_and_unload()` into base weights**  
3. Train one new LoRA on MCQ-heavy mix  

So previous math/code gains were **baked then overwritten** by MCQ gradient mass.  
That is not superposition — that is **collapse of the wavefunction into the last strong measurement**.

### 2.4 Output-head conflict (behavioral)

| Pathway | Required emission |
|---------|-------------------|
| ARC/MMLU | Single letter A–D |
| GSM8K | Multi-step + `#### n` |
| HumanEval | Full function body |

Shared last-layer habits cannot hold all three without **task-conditional routing**.  
Without routing, the strongest recent training distribution wins (MCQ).

---

## 3. What is *not* the fix

- Bigger LR / more epochs on the same mixed LoRA  
- Free-parameter “task weights” fit after the fact  
- Dropping FSOT to chase SOTA  
- Ignoring regressions because mean went up  

Mean-up + pathway-down is **false health**.

---

## 4. FSOT-correct way forward (no regression)

### Principle

> **Train pathway modes in isolation; superpose only at observation time.**  
> Shared medium remains base model + seed geometry.  
> Measurement (prompt/topic) selects α — it must not permanently erase other modes.

### Architecture

```
                    base (frozen or lightly shared)
                              │
        ┌──────────┬──────────┼──────────┬──────────┐
        ▼          ▼          ▼          ▼          ▼
     LoRA_ont   LoRA_math  LoRA_mcq  LoRA_code  (K modes…)
        │          │          │          │
        └──────────┴──── α(topic) ───────┘
                              │
                     W_eff = Σ α_p W_p
                     (inference only)
```

| Rule | Practice |
|------|----------|
| **Orthogonalize topics** | Force pathway D_eff / loci apart so dominant modes differ |
| **Isolated stimulus** | Train `adapter_mcq` only on ARC/MMLU; `adapter_math` only on GSM8K; etc. |
| **No merge_and_unload of specialists** | Keep adapters as separate files |
| **Homeostatic replay** | Even if deficit=0, small maintenance dose for code/math |
| **Gate promotion** | Promote only if **no pathway drops > ε** (e.g. 5% absolute) |

### ε-promotion (preregistered)

A new adapter set is **GREEN** only if:

1. Target pathway improves, **and**  
2. Every other pathway: `Δ ≥ −0.05` (no severe poison), **and**  
3. Archive K5 still OK  

Otherwise: keep prior specialists; reduce dose; re-route.

---

## 5. Immediate next engineering

1. `interference_analysis.py` — ledger poison/synergy report  
2. `pathway_adapters.py` — isolated LoRA per pathway + α route at generate  
3. Organism stimulus v2 — **never** single mixed LoRA for multi-pathway deficit  
4. Re-baseline with routed specialists (expect: keep ARC gains, restore math/code)

---

## 6. One-line diagnosis

**ARC improved because MCQ pathway was stimulated; GSM8K/HumanEval fell because the same shared LoRA was poisoned by MCQ output habits and zero code maintenance — FSOT fix is superpose pathway adapters at inference, not average them at train time.**
