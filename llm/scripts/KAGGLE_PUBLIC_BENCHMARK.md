# Public benchmarks live on Kaggle — not your desktop

## Why local lm-eval was wrong for “public”

| Place | Role |
|-------|------|
| **Desktop** | Train pathways, anti-poison gates, export merge, push model |
| **Kaggle Models** | Public weights (already: v5) |
| **Kaggle GPU notebook/kernel** | **Public benchmark runs** — free GPU, shareable output, no desktop RAM |
| **Kaggle Community Benchmarks** | Optional leaderboard product (tasks you define / join); often API-catalog models; custom open weights usually go through a notebook that loads *your* Model input |

Your desktop is for **building**. Kaggle is for **showing**.

## What’s already public

- Model: https://www.kaggle.com/models/damianpalumbo/fsot-21-instruct-05b  
- Variation: PyTorch / transformers — **version 5** (rule-guided release)

## Public benchmark kernel (this repo)

```text
llm/kaggle_public_benchmark/
  fsot_public_benchmark.py   # runs lm_eval on Kaggle GPU against attached model
  kernel-metadata.json
```

### Push & run (from your PC — only uploads the script, compute is on Kaggle)

```powershell
cd "C:\Users\damia\Desktop\fsot 2.1 llm"
kaggle kernels push -p "llm\kaggle_public_benchmark"
```

Then open the kernel page, confirm **GPU** + model attached, wait for the run.  
Outputs:

- `/kaggle/working/FSOT_21_PUBLIC_BENCHMARK.json`
- `/kaggle/working/FSOT_21_PUBLIC_BENCHMARK.md`
- lm-eval artifacts under `lm_eval_public/`

Pull results:

```powershell
kaggle kernels output damianpalumbo/fsot-21-public-benchmark -p "llm\kaggle_public_benchmark\output"
```

### Full vs subsample

- Default `limit=100` — publishable sample with stderr (fast enough on free GPU).
- Full suite: set env in the script / version notes to `FSOT_PUBLIC_LIMIT=0` (long run).

## Community Benchmarks (optional next)

1. https://www.kaggle.com/benchmarks  
2. Create a **task** (math rules / GSM8K-style / code) with the Benchmarks UI or SDK  
3. Run against catalog models **and** document your FSOT model via the public notebook above  

Community Benchmarks ≠ automatic scoring of every private desktop checkpoint. Your open model is scored by **attaching it in a Kaggle run**.
