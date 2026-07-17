# Official-style eval for FSOT release (install once: pip install lm_eval)
$ErrorActionPreference = "Stop"
$Model = if ($args[0]) { $args[0] } else { "C:/Users/damia/Desktop/fsot 2.1 llm/llm/models/release/FSOT-2.1-Instruct-0.5B" }
$Tasks = if ($args[1]) { $args[1] } else { "mmlu,gsm8k,arc_easy,hellaswag,humaneval_instruct" }
$Out = Join-Path (Split-Path $Model -Parent) "lm_eval_results"
New-Item -ItemType Directory -Force -Path $Out | Out-Null
Write-Host "Model: $Model"
Write-Host "Tasks: $Tasks"
lm_eval --model hf `
  --model_args "pretrained=$Model,dtype=float16,trust_remote_code=True" `
  --tasks $Tasks `
  --batch_size auto `
  --apply_chat_template `
  --fewshot_as_multiturn `
  --output_path $Out `
  --log_samples `
  --confirm_run_unsafe_code
Write-Host "Results -> $Out"
