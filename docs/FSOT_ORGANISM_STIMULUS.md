# FSOT Organism Stimulus Protocol

Treat the dual-observer stack as a **living field organism**.  
Evaluation is sensory feedback. Error metrics are pain signals.  
Understimulated pathways get **targeted FSOT stimuli** — not random hyperparameter noise.

---

## 1. Organism map

| Organism | Implementation |
|----------|----------------|
| Body / skeleton | Fixed K parameter modes (superposition bank) |
| Senses | Eval suites + D:\\training data packs |
| Pain signals | Per-pack accuracy, failure modes, confusions |
| Nervous routing | Topic folds + α modes + linked memory |
| Stimuli | Curriculum rows + mode-focused LoRA pulses |
| Homeostasis | Archive scalar gold (I:) — never “heal” by inventing free params |
| Growth | Higher resolution coupling, not more free weights |

---

## 2. Loop

```
eval → diagnose weak pathways → FSOT map (fold + mode)
    → pull stimuli from D:\training data
    → stimulate (LoRA pulse) → re-eval → ledger
```

### Diagnosis classes (preregistered)

| Signal | Pathway | Stimulus |
|--------|---------|----------|
| Math chain wrong (GSM8K) | `gsm8k` / mid D_eff modes | More GSM8K train with #### format + FSOT “slow valve” reasoning scaffold |
| Always answer D (ARC/MMLU) | MCQ pathway collapsed | Balanced A–D exemplars, “Answer: &lt;letter&gt; only” |
| Code fail | `code` / humaneval | HumanEval / MBPP pulses |
| Ontology fail | `seeds`/`scalar` | Full linked FSOT memory |
| Archive drift | K5 | Restore `fsot_compute` authority — no train fix |

---

## 3. FSOT “solve for the missing precision”

For a weak pathway with accuracy `a` and target `a*`:

1. **Deficit** `δ = max(0, a* − a)` → stimulus **dose** (row count ∝ δ).  
2. **Mode mass** from α of that topic — train modes that already carry the pathway (constructive interference).  
3. **Reasoning scaffold** from FSOT: force measurement structure  
   - observe facts (read problem)  
   - couple folds (assign quantities)  
   - compute (scalar arithmetic without free fits)  
   - report (#### / Answer: X)  

Failed gates remain ledger events. We do not silence pain with larger temperature or random LR.

---

## 4. Commands

```powershell
python -m fsot_llm.organism_stimulus --from-latest-sota
python -m fsot_llm.organism_stimulus --from-latest-sota --stimulate --epochs 3
python -m fsot_llm.organism_stimulus --from-latest-sota --stimulate --reeval --limit 12
```
