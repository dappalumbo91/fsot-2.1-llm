# FSOT Superposition Depth — depth without parameter growth

**Claim:** Intelligence depth is not proportional to parameter count. In FSOT, one fluid medium already carries all domains; “depth” is **resolution of coupling** in that medium, not extra free weights.

---

## 1. Problem

Standard scaling:

```
more topics  →  more parameters  →  more VRAM
```

FSOT scaling (this lab):

```
more topics  →  same parameter modes  →  denser superpositions  →  deeper effective theory
```

---

## 2. Physics picture

| Physics | LLM engineering |
|---------|-----------------|
| One fluid field `raw_S` | Shared parameter **modes** (basis LoRA / residual banks) |
| Domain regimes | **Topics** as observation folds `(D_eff, ρ, hits, observed)` |
| Superposition | Topic state = Σ α_k · mode_k |
| Interference / coupling | Related topics share α mass (constructive memory) |
| As above, so below | Cross-topic links from seed geometry, not separate matrices |
| Measurement | `observed=True` collapses readout toward task-active modes |

Parameter count stays **fixed** (K modes × mode size).  
**Depth** grows with how well α and topic links resolve structure.

---

## 3. Architecture (implemented)

```
                    ┌─────────────────────┐
  topic / query ──►│ FSOT fold features  │
                    │ D_eff, ρ, S, …      │
                    └──────────┬──────────┘
                               ▼
                    ┌─────────────────────┐
                    │ Coupling to K modes │  α = softmax( β · sim(f, mode_locus) )
                    └──────────┬──────────┘
                               ▼
              W_eff = Σ_k α_k W_k     (superposed adapter)
                               ▼
                    base small model forward
```

**Topic memory bank** stores compact FSOT descriptors + exemplar text, **not** a full weight copy per topic.  
Linked topics borrow each other’s exemplars weighted by coupling — **memory savings**.

---

## 4. Non-negotiables

1. Mode loci and default folds are **seed-derived / preregistered**, not free-fit embeddings for each new topic.
2. Adding a topic does **not** add a new full adapter; it adds a row in the topic bank.
3. Archive `fsot_compute` remains gold for any numeric S claim.
4. Kill criteria: same K1–K5; plus **K6 depth-without-params** — N topics with fixed K modes must beat unlinked baseline on multi-topic retrieval.

---

## 5. Practical knobs (not free parameters)

| Symbol | Meaning | Policy |
|--------|---------|--------|
| K | number of modes | preregistered (default 8) |
| β | coupling temperature | preregistered (default from ψ_con geometry) |
| topic folds | D_eff etc. | from domain map / spectrum / curriculum tags |

Changing these requires a new PRED / ledger note — not silent tuning mid-eval.

---

## 6. Training implication

Curriculum teaches **linked topic bundles** together so superpositions form constructive interference.  
We do not train 48 separate specialists; we train **shared modes** on rotating topic mixtures drawn by FSOT coupling.
