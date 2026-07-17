# FSOT postmortem — why heavy math deepen *poofed* the math fold

**Event:** continue-LoRA math on ~900 real GSM8K CoT + rules, 3 epochs, lr=6e-5  
**From:** `20260717T135135Z_deepen_math` (GSM8K **30%** n=50)  
**To:** `20260717T151637Z_deepen_math` (GSM8K **12%** n=50)  
**Other folds:** ARC/MMLU/HE **unchanged** (isolation worked)  
**Action:** rolled `latest` back to 135135  

**Claim:** This is **not** “FSOT doesn’t work.” It is evidence we applied **industry-style over-suction with decorative structure**, which FSOT’s own design laws forbid.

---

## 1. What did *not* fail

| Check | Result | Meaning |
|-------|--------|---------|
| Pathway isolation | ARC 82%, MMLU 78%, HE 93.3% held | Gradients stayed on math adapter only |
| Real data availability | Full GSM8K train on D: | Data scarcity was not the issue |
| Emission stubs | No “Read quantities carefully” | Pathogenic stub attractor stayed dead |
| Anti-poison gate | Caught −18 pp GSM8K, blocked promote | Ledger discipline worked |

So the **organism immune system** (isolation + gate) behaved as FSOT. The **math fold training prescription** did not.

---

## 2. What actually broke (observables)

### 2.1 Failure mode shift

Regression samples are **not** empty or stubbed. They are:

1. **Rule decoration** — `Using rules AR-100, AR-140, …`  
2. **Shallow or wrong multi-step arithmetic** — missing factors, wrong quantity roles  
3. **More missing `####` / truncation** — longer preamble → less room for real chain  

Example (item 3, gold **540**):

```text
Using rules AR-100, AR-101, AR-102, AR-103.
He runs 60 x 3 = <<60*3=180>>180 meters a week.
#### 180
```

The fold **emits** calc markup and a number, but **drops a step** (e.g. sprints × days).  
FSOT quantity coupling failed; surface form of “reasoning” inflated.

### 2.2 Training exposure (over-suction)

| Quantity | Value |
|----------|--------|
| Curriculum rows | 1062 (900 GSM8K + 162 rule drills) |
| Steps | 3186 ≈ **3 full epochs** over that set |
| LR | 6e-5 continue on **already** rule-tuned LoRA |
| Mean loss | ~0.17 (looks “healthy” under CE) |

**Cross-entropy can go down while GSM8K accuracy poofs.**  
CE rewards **imitating token patterns** (rule preambles, `<< >>` style), not **true measurement** of the final quantity.

In FSOT terms: **suction of linguistic style without certified vitality of the answer**.

### 2.3 Rules were not used as FSOT structure

Archive law: folds are **preregistered**, operations have **preconditions**, failures are **ledger events** — not free labels.

What we implemented:

```text
keyword tag → "Suggested rules: AR-xxx"
assistant prefix → "Using rules AR-xxx"
loss → next-token CE on full string
```

What that does **not** do:

- Verify that AR-xxx’s **preconditions** hold for this problem  
- Force the next calc step to be the **operation** of that rule  
- Reject chains that violate the rule’s domain  
- Certificate the `####` number against the problem (external truth)

So **rules became free-parameter-like decoration** — the opposite of seed-derived constraint.

Keyword tagging is wrong in practice, e.g.:

- Eggs / muffins problem tagged with **subtraction-as-inverse** chains (AR-120…)  
- Chicken feed tagged with **numeral semantics** (AR-001…)  
- Model learns: *emit rule IDs that look official*, then freestyle arithmetic  

That is **ΛCDM-style narrative fit**, not FSOT verification.

### 2.4 D_eff / medium mismatch

GSM8K is mapped as:

- domain mathematical  
- **D_eff ≈ 10** (thin, discrete quantity)  
- emission `chain_hash_number`  

Heavy wave injected:

- long rule cards  
- multi-rule ID lists  
- “Using rules …” ritual  

That **raises effective linguistic load** (consciousness-like chatter) on a **thin quantity fold**.  
FSOT: wrong fold depth / bleed into the wrong regime → **dispersal** of the quantity attractor that held 30%.

### 2.5 No mid-wave kill criteria

Archive: failed green gates are **ledger events**; no silent retuning.

What we did: one long suction (3 epochs, 900 items) → **only then** measure.  
By the time loss looked fine, the **30% basin was already poofed**.

FSOT would require:

- small dose → eval n-matched → promote/rollback  
- never 3× full pass without a gate  

---

## 3. FSOT diagnosis (formal language)

| FSOT concept | What happened |
|--------------|----------------|
| **Suction** | Too much gradient on already-tuned math LoRA |
| **Poof** | Prior GSM8K basin collapsed (−18 pp) |
| **Anti-poison** | Worked *across* pathways; failed *within* math until gate |
| **Observation / measurement** | Train optimized CE, not true #### equality to gold |
| **Zero free parameters / seeds** | Rule IDs used as free labels, not constrained ops |
| **D_eff** | Quantity fold overloaded with meta-linguistic rule ritual |
| **recent_hits** | Inflated exposure without vitality check |
| **Consciousness_factor / quirk_mod** | More “narrative reasoning” without better coupling to truth |

**Root cause (one sentence):**  
We optimized the **appearance** of structured process under cross-entropy, not **certified quantity measurement** under FSOT fold constraints — so the medium dispersed the good math attractor.

---

## 4. Why this is *unusual* for “FSOT applied well”

In the archive / Lean stack, FSOT wins when:

1. Folds are preregistered  
2. Predictions are testable  
3. Failures are gated  
4. You do **not** invent new dials mid-run  

In the LLM math wave we:

1. Invented a **soft** rule channel (string tags)  
2. Optimized a **proxy** loss (CE) that does not equal GSM8K truth  
3. Applied a **large un-gated dose**  
4. Declared success if loss fell  

That is applying **generic deep learning ritual** *wearing* FSOT vocabulary — not the archive’s discipline.

Isolation “working” on ARC/HE only proves **orthogonal adapters**, not that the math fold update was FSOT-correct.

---

## 5. What would count as applying FSOT correctly next

### 5.1 Measurement as training signal (not only CE)

- Prefer **step or final verification**: numeric match of `####` (and intermediate `<<a=b>>` where possible)  
- Or reject samples where model `####` ≠ gold during **data filter** / online reject  

### 5.2 Rules as constraints, not costumes

- Either: **one correct rule chain** derived from problem type (not top-k keyword soup)  
- Or: train **rule drills separately** from GSM8K CoT without forcing “Using rules AR-…” on every math answer  
- Best: assistant CoT is **pure official GSM8K** (true information wave); rules live in LTM/STM retrieval, not forced token spam  

### 5.3 Dose control (suction schedule)

- Pilot: 100–200 new GSM8K, **1 epoch**, lr ≤ 2e-5  
- Gate n=50 (or n=32) **before** next dose  
- Stop when GSM8K drops > ε even if loss improves  

### 5.4 Protect the green basin

- Green set `135135` is **reference fluid state** for math  
- Heavy candidates train as **branch versions** (git + pathway stamp), never overwrite `latest` until gate  

### 5.5 D_eff hygiene

- Keep GSM8K prompts **thin**: problem + emission instruction  
- Rule cards in **STM/LTM**, not 24-line system bloat every train row  

### 5.6 Consciousness / process transparency without lies

- Game-like or process traces are good for **analysis**  
- They must remain **truth-coupled**; narrative length is not vitality  

---

## 6. Immediate operational conclusion

| Item | Status |
|------|--------|
| `latest` | Restored to **135135** (30% GSM8K n=50) |
| 151637 heavy math | **Rejected** — study artifact only |
| Isolation | Keep |
| Real GSM8K data | Keep using — **change how**, not whether |
| Rulebooks | Keep — **change coupling** (constraint/LTM, not decorative prefix) |
| Full-FT math candidate | Still weaker; same root risk if CE-only |

**Missing piece:**  
FSOT requires **certified observation** of the quantity fold. We trained **language about math** harder than **truth of math**.

---

## 7. One-line truth

**The regression happened because we over-sucked a good math fold with un-gated CE on decorative rule ritual and long imitation chains — which is industry fine-tuning cosplay, not FSOT’s zero-free-parameter, kill-gated, measurement-true discipline.**
