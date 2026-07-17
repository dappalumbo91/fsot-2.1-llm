# FSOT memory hierarchy — LTM / STM / process

**Ontology:** Fluid Spacetime Omni-Theory  
**Rule:** Do not load the entire knowledge fluid into GPU VRAM. Route amplitude by fold; store persistence on disk.

## Human → FSOT → machine map

| Human-scale picture | FSOT | Implementation |
|---------------------|------|----------------|
| Long-term memory (mind archive / “soul” ledger of known info) | Persistent waveform of observations | **SQLite LTM** on disk (`D:\FSOT_LLM_Memory` or `llm/data/memory`) |
| Short-term / working memory | Active coupling set + recent_hits | **RAM STM** (deque of turns + retrieved LTM slices) |
| Immediate process (body / parallel act) | Measurement collapse under `observed` | **GPU/CPU** forward on **one** pathway adapter only |
| Routing / attention | Domain map D_eff, pathway, emission | `resolve_allocation` + inject `FSOT_ROUTE` |

This is the same “depth without parameter growth” idea as topic banks: **topics and memories are rows and retrievals**, not new full weight copies.

## Data flow

```
user text
   │
   ▼
 observe() ──► domain map (pathway, D_eff, emission)
   │
   ├─► LTM.retrieve(query, pathway)   [disk → RAM only]
   │
   ▼
 STM: route + LTM hits + recent dialogue   [RAM only]
   │
   ▼
 process_generate(): load pathway fold once, run on GPU/CPU
   │
   ▼
 optional remember() write-back to LTM   [RAM → disk]
```

**Never:** dump all LTM into the residual stream or load all pathway adapters at once.  
**Always:** one process fold + small STM context block.

## CLI

```powershell
cd "C:\Users\damia\Desktop\fsot 2.1 llm"
$env:PYTHONPATH = "llm\python"

# Status
python -m fsot_llm.memory_hierarchy --status

# Seed LTM from existing topic bank
python -m fsot_llm.memory_hierarchy --seed-bank

# Store / query without GPU
python -m fsot_llm.memory_hierarchy --store "FSOT seeds: pi e phi gamma G Catalan"
python -m fsot_llm.memory_hierarchy --query "seeds"

# Full cycle (uses pathway adapters + GPU if capable)
python -m fsot_llm.memory_hierarchy --chat "What is 2+2? End with #### n" --pack gsm8k_test
```

Override LTM location:

```powershell
$env:FSOT_LTM_PATH = "I:\FSOT-Physical-Archive\08_Verified-Desktop-Projects\fsot_ltm.sqlite"
```

## Relation to training

| Component | Memory layer |
|-----------|----------------|
| Pathway LoRAs / full-FT folds | Process specialists (body) |
| Math-generator rules + GSM8K train | LTM candidates when stored as episodes |
| Anti-poison / version registry | Long-term ledger of which fluid state is pure |
| Chat session | STM |

Training still updates process weights by fold. LTM holds **retrievable information waves** without baking everything into parameters.

## Experiments to run on desktop

1. Seed LTM from topic bank + successful GSM8K CoT exemplars.  
2. Compare process with `use_ltm` vs empty LTM on same GSM8K items.  
3. Decay unused LTM (poof weak strength) vs reinforce on hit (suction).  
4. Keep pathway routing; only STM block size changes (not model size).

## Non-negotiables

1. LTM lives on **disk** (external drive preferred).  
2. STM lives in **RAM** and is bounded.  
3. Process loads **one pathway** (or one full-FT fold) onto accelerator.  
4. Intelligence metric remains **coupling quality**, not parameter count.
