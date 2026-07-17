# Leaderboard submit path — FSOT-2.1-Instruct-0.5B

## 1. Local release (done by `python -m fsot_llm.export_release`)

- Full HF model: `llm/models/release/FSOT-2.1-Instruct-0.5B/`
- Must load with `AutoModelForCausalLM` / `AutoTokenizer` only

## 2. Official numbers (required before claiming board ranks)

```powershell
pip install lm_eval
.\llm\scripts\run_lm_eval_release.ps1
```

Or full Open-LLM-Leaderboard-style task groups if available in your lm_eval version:

```powershell
lm_eval --model hf --model_args pretrained="llm/models/release/FSOT-2.1-Instruct-0.5B" --tasks leaderboard --batch_size auto
```

## 3. Hugging Face Hub (primary public ID)

```powershell
pip install huggingface_hub
huggingface-cli login
huggingface-cli upload YOUR_USER/FSOT-2.1-Instruct-0.5B "llm/models/release/FSOT-2.1-Instruct-0.5B"
```

Or during export:

```powershell
python -m fsot_llm.export_release --push-hub YOUR_USER/FSOT-2.1-Instruct-0.5B
```

Then open the model page → request / document evals on Open LLM Leaderboard
(when submissions are open) and paste lm-eval tables into the model card.

## 4. Kaggle Models (secondary mirror + notebooks)

```powershell
kaggle models create -p llm/models/release/FSOT-2.1-Instruct-0.5B --model-slug fsot-21-instruct-05b
# or dataset + notebook that runs lm_eval on GPU
```

## 5. Naming

- Public name: **FSOT-2.1-Instruct-0.5B**
- Base disclosure: `Qwen/Qwen2.5-Coder-0.5B-Instruct`
- Your contribution: FSOT pathway post-training + merge export

## 6. Do not claim n=16 refine scores as leaderboard results
