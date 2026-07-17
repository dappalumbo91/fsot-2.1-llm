"""
FSOT 2.1 Instruct 0.5B — PUBLIC benchmark on Kaggle.

Failure mode we hit on v1:
  Tesla P100 (cuda sm_60) + modern Kaggle/PyTorch (sm_70+) →
  "CUDA error: no kernel image is available for execution on the device"

Fix:
  Detect GPU compute capability; if < 7.0 (Pascal P100/P4), force CPU.
  Install lm_eval without upgrading torch (upgrade-strategy only-if-needed).
"""
from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path

INPUT = Path("/kaggle/input")
WORK = Path("/kaggle/working")
WORK.mkdir(parents=True, exist_ok=True)


def find_model_dir() -> Path:
    for c in INPUT.rglob("config.json"):
        parent = c.parent
        if (parent / "model.safetensors").is_file() or any(parent.glob("*.safetensors")):
            return parent
    for ver in ("5", "4", "3", "2", "1"):
        p = (
            INPUT
            / "models"
            / "damianpalumbo"
            / "fsot-21-instruct-05b"
            / "pytorch"
            / "transformers"
            / ver
        )
        if p.is_dir():
            return p
        p2 = INPUT / "fsot-21-instruct-05b" / "pytorch" / "transformers" / ver
        if p2.is_dir():
            return p2
    raise FileNotFoundError(
        f"Model not found under {INPUT}. Attach damianpalumbo/fsot-21-instruct-05b."
    )


def pip_install(args: list[str]) -> None:
    subprocess.check_call(
        [sys.executable, "-m", "pip", "install", "-q", *args],
        stdout=sys.stdout,
        stderr=sys.stderr,
    )


def pick_device() -> str:
    """P100 is sm_60; current Kaggle torch builds often lack Pascal kernels."""
    import torch

    print("torch", torch.__version__, flush=True)
    print("cuda_available", torch.cuda.is_available(), flush=True)
    if not torch.cuda.is_available():
        return "cpu"
    try:
        name = torch.cuda.get_device_name(0)
        major, minor = torch.cuda.get_device_capability(0)
        print(f"GPU: {name} capability sm_{major}{minor}", flush=True)
        # sm_70 = Volta V100; P100 is 6.0 — not in modern wheels
        if major < 7:
            print(
                "WARNING: GPU compute capability < 7.0 (e.g. Tesla P100). "
                "This PyTorch build has no kernels for it. Forcing CPU.",
                flush=True,
            )
            return "cpu"
        # smoke a tiny op
        x = torch.zeros(1, device="cuda")
        _ = x + 1
        del x
        torch.cuda.empty_cache()
        return "cuda:0"
    except Exception as exc:
        print("GPU smoke failed → CPU:", repr(exc), flush=True)
        return "cpu"


print("=== FSOT 2.1 public benchmark (Kaggle) ===", flush=True)

# Install harness WITHOUT forcing a newer torch that breaks P100 further
print("Installing lm_eval (no torch upgrade)...", flush=True)
pip_install(
    [
        "lm_eval==0.4.12",
        "--upgrade-strategy",
        "only-if-needed",
    ]
)

MODEL_DIR = find_model_dir()
print("MODEL_DIR =", MODEL_DIR, flush=True)

DEVICE = pick_device()
print("DEVICE =", DEVICE, flush=True)

# CPU is slower: keep a solid public subsample. GPU can go higher.
DEFAULT_LIMIT = "50" if DEVICE == "cpu" else "100"
LIMIT = int(os.environ.get("FSOT_PUBLIC_LIMIT", DEFAULT_LIMIT))
# Avoid expanding full MMLU into 57 subjects × limit on first public run.
# mmlu alone spawned 23k loglikelihood calls and died on CUDA pad.
TASKS = os.environ.get(
    "FSOT_PUBLIC_TASKS",
    "arc_easy,gsm8k,hellaswag",
)

OUT = WORK / "lm_eval_public"
OUT.mkdir(parents=True, exist_ok=True)

dtype = "float32" if DEVICE == "cpu" else "float16"
# lm_eval 0.4.x: pass device ONLY via --device (not also in model_args)
# or you get: HFLM() got multiple values for keyword argument 'device'
model_args = f"pretrained={MODEL_DIR},dtype={dtype},trust_remote_code=True"

cmd = [
    sys.executable,
    "-m",
    "lm_eval",
    "--model",
    "hf",
    "--model_args",
    model_args,
    "--tasks",
    TASKS,
    "--limit",
    str(LIMIT),
    "--batch_size",
    "1" if DEVICE == "cpu" else "4",
    "--device",
    DEVICE,
    "--apply_chat_template",
    "--fewshot_as_multiturn",
    "--output_path",
    str(OUT),
    "--log_samples",
    "--confirm_run_unsafe_code",
]
print("RUN:", " ".join(cmd), flush=True)
rc = subprocess.call(cmd)
print("lm_eval exit:", rc, flush=True)

# Collect metrics
results_files = list(OUT.rglob("*results*.json"))
metrics: dict = {}
results_file = None
for rf in results_files:
    try:
        data = json.loads(rf.read_text(encoding="utf-8"))
        if "results" in data:
            for task, vals in data["results"].items():
                metrics[task] = vals
            results_file = str(rf)
    except Exception as exc:
        print("parse skip", rf, exc, flush=True)

summary = {
    "model": "damianpalumbo/fsot-21-instruct-05b",
    "model_dir": str(MODEL_DIR),
    "platform": "kaggle",
    "device": DEVICE,
    "harness": "lm_eval",
    "tasks": TASKS,
    "limit": LIMIT,
    "lm_eval_exit": rc,
    "results_file": results_file,
    "metrics": metrics,
    "gpu_note": (
        "If device=cpu: Kaggle assigned Tesla P100 (sm_60) which is incompatible "
        "with current PyTorch wheels (sm_70+). CPU fallback is intentional and valid "
        "for a public 0.5B eval; scores are still harness-standard."
    ),
    "note": (
        "Public Kaggle run. limit=N is a subsample with stderr. "
        "Dense merged release ≠ multi-adapter pathway refine scores."
    ),
}
out_json = WORK / "FSOT_21_PUBLIC_BENCHMARK.json"
out_json.write_text(json.dumps(summary, indent=2), encoding="utf-8")
print(json.dumps(summary, indent=2)[:12000], flush=True)

md = [
    "# FSOT 2.1 Instruct 0.5B — Public Kaggle Benchmark",
    "",
    f"- **Model:** [damianpalumbo/fsot-21-instruct-05b](https://www.kaggle.com/models/damianpalumbo/fsot-21-instruct-05b)",
    f"- **Device:** `{DEVICE}`",
    f"- **Harness:** lm-eval",
    f"- **Tasks:** `{TASKS}`",
    f"- **Limit:** {LIMIT}",
    f"- **Exit:** {rc}",
    "",
    "## Metrics",
    "",
]
for task, vals in metrics.items():
    if isinstance(vals, dict):
        for k, v in vals.items():
            if isinstance(v, (int, float)) and any(
                x in str(k) for x in ("acc", "exact_match", "pass", "em")
            ):
                md.append(f"- **{task}** `{k}`: **{float(v):.4f}**")
    else:
        md.append(f"- **{task}**: {vals}")
if not metrics:
    md.append("_No metrics (run failed before results were written)._")
md.append("")
md.append("## GPU compatibility")
md.append(summary["gpu_note"])
(WORK / "FSOT_21_PUBLIC_BENCHMARK.md").write_text("\n".join(md) + "\n", encoding="utf-8")
print("\n".join(md), flush=True)

if rc != 0:
    sys.exit(rc)
print("DONE public Kaggle benchmark.", flush=True)
