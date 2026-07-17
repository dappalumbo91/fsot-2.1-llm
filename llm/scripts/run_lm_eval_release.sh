#!/usr/bin/env bash
set -euo pipefail
MODEL="${1:-C:/Users/damia/Desktop/fsot 2.1 llm/llm/models/release/FSOT-2.1-Instruct-0.5B}"
TASKS="${2:-mmlu,gsm8k,arc_easy,hellaswag,humaneval_instruct}"
OUT="$(dirname "$MODEL")/lm_eval_results"
mkdir -p "$OUT"
lm_eval --model hf \
  --model_args "pretrained=$MODEL,dtype=float16,trust_remote_code=True" \
  --tasks "$TASKS" \
  --batch_size auto \
  --apply_chat_template \
  --fewshot_as_multiturn \
  --output_path "$OUT" \
  --log_samples \
  --confirm_run_unsafe_code
echo "Results -> $OUT"
