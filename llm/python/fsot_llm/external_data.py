"""
External training / benchmark data registry.

Primary: D:\\training data  (catalog + SOTA packs)
Secondary: D:\\FSOT_Benchmarks
Archive: I:\\FSOT-Physical-Archive (verification authority — not replaced)
"""
from __future__ import annotations

import json
import os
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterator, Optional


def training_data_root() -> Path:
    raw = os.environ.get("FSOT_TRAINING_DATA_ROOT", r"D:\training data")
    return Path(raw)


def fsot_benchmarks_root() -> Path:
    raw = os.environ.get("FSOT_BENCHMARKS_ROOT", r"D:\FSOT_Benchmarks")
    return Path(raw)


@dataclass(frozen=True)
class Pack:
    id: str
    path: Path
    kind: str  # gsm8k | humaneval | mmlu | arc | hellaswag | truthfulqa | bbh | winogrande | mbpp | math
    split: str
    notes: str = ""


def discover_packs() -> list[Pack]:
    root = training_data_root()
    fb = fsot_benchmarks_root()
    packs: list[Pack] = []

    def add(pid: str, rel: str, kind: str, split: str, notes: str = "") -> None:
        p = root / rel
        if p.is_file() or p.is_dir():
            packs.append(Pack(pid, p, kind, split, notes))

    add("gsm8k_test", "gsm8k/test.jsonl", "gsm8k", "test")
    add("gsm8k_train", "gsm8k/train.jsonl", "gsm8k", "train")
    add("humaneval", "humaneval/HumanEval.jsonl", "humaneval", "test")
    add("mmlu_val", "mmlu/validation.jsonl", "mmlu", "validation")
    add("mmlu_test", "mmlu/test.jsonl", "mmlu", "test")
    add("hellaswag_val", "hellaswag/validation.jsonl", "hellaswag", "validation")
    add("truthfulqa_val", "truthfulqa/validation.jsonl", "truthfulqa", "validation")
    add("bbh_mix", "bbh/bbh_mix.jsonl", "bbh", "mix")
    add("winogrande_val", "winogrande/validation.jsonl", "winogrande", "validation")
    add("math_500", "math/math.jsonl", "math", "test")
    add("mmlu_pro", "mmlu_pro/test.jsonl", "mmlu_pro", "test")
    add("ifeval", "ifeval/train.jsonl", "ifeval", "train")
    add("arc_easy_val", "ARC-Easy_validation.csv", "arc", "validation", "ARC-Easy")
    add("arc_challenge_val", "ARC-Challenge_validation.csv", "arc", "validation", "ARC-Challenge")

    if fb.is_dir():
        mbpp = fb / "mbpp.jsonl"
        if mbpp.is_file():
            packs.append(Pack("mbpp", mbpp, "mbpp", "test", "FSOT_Benchmarks"))
        mmlu_csv = fb / "mmlu.csv"
        if mmlu_csv.is_file():
            packs.append(Pack("mmlu_fsot_bench", mmlu_csv, "mmlu_csv", "mixed", "FSOT_Benchmarks"))

    return packs


def registry_report() -> dict[str, Any]:
    packs = discover_packs()
    return {
        "training_data_root": str(training_data_root()),
        "training_data_exists": training_data_root().is_dir(),
        "fsot_benchmarks_root": str(fsot_benchmarks_root()),
        "fsot_benchmarks_exists": fsot_benchmarks_root().is_dir(),
        "packs": [
            {
                "id": p.id,
                "path": str(p.path),
                "kind": p.kind,
                "split": p.split,
                "exists": p.path.exists(),
                "size_mb": round(p.path.stat().st_size / 1e6, 2)
                if p.path.is_file()
                else None,
                "notes": p.notes,
            }
            for p in packs
        ],
        "n_packs": len(packs),
    }


def iter_jsonl(path: Path, limit: Optional[int] = None) -> Iterator[dict[str, Any]]:
    n = 0
    with path.open(encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            yield json.loads(line)
            n += 1
            if limit is not None and n >= limit:
                break


def load_pack_rows(pack_id: str, limit: int = 50) -> list[dict[str, Any]]:
    packs = {p.id: p for p in discover_packs()}
    if pack_id not in packs:
        raise KeyError(f"unknown pack {pack_id}; known={list(packs)}")
    p = packs[pack_id]
    if p.kind == "arc":
        import csv

        rows = []
        with p.path.open(encoding="utf-8", newline="") as f:
            reader = csv.DictReader(f)
            for i, row in enumerate(reader):
                if i >= limit:
                    break
                rows.append(dict(row))
        return rows
    if p.path.suffix == ".jsonl":
        return list(iter_jsonl(p.path, limit=limit))
    if p.path.suffix == ".csv":
        import csv

        rows = []
        with p.path.open(encoding="utf-8", newline="") as f:
            reader = csv.DictReader(f)
            for i, row in enumerate(reader):
                if i >= limit:
                    break
                rows.append(dict(row))
        return rows
    raise ValueError(f"unsupported pack format: {p.path}")


# --- Answer extraction helpers (scoring) ---

_NUM = re.compile(r"[-+]?\d*\.?\d+(?:[eE][-+]?\d+)?")


def extract_gsm8k_gold(answer_field: str) -> str:
    # format: reasoning\n#### 42
    if "####" in answer_field:
        return answer_field.split("####")[-1].strip().replace(",", "")
    nums = _NUM.findall(answer_field.replace(",", ""))
    return nums[-1] if nums else answer_field.strip()


def extract_final_number(text: str) -> Optional[str]:
    # Prefer the first #### answer line (models sometimes ramble after)
    if "####" in text:
        after = text.split("####", 1)[1].strip().splitlines()[0]
        after = after.replace(",", "").replace("$", "")
        nums = _NUM.findall(after)
        if nums:
            return nums[0]
    nums = _NUM.findall(text.replace(",", "").replace("$", ""))
    return nums[-1] if nums else None



def extract_mc_letter(text: str) -> Optional[str]:
    t = text.strip().upper()
    # prefer explicit "Answer: A"
    m = re.search(r"(?:ANSWER|CHOICE|OPTION)\s*[:\-]?\s*([A-D])\b", t)
    if m:
        return m.group(1)
    m = re.search(r"\b([A-D])\b", t)
    return m.group(1) if m else None


def normalize_arc_choices(raw: Any) -> tuple[list[str], list[str]]:
    """Return (labels, texts). Handles HF-export CSV quirks with array([...])."""
    if isinstance(raw, str):
        s = raw.strip()
        # Pattern: {'text': array(['a', 'b', ...], dtype=object), 'label': array(['A',...], dtype=object)}
        texts_m = re.search(
            r"array\(\s*\[(.*?)\]\s*(?:,\s*dtype\s*=\s*object)?\s*\)",
            s,
            flags=re.S,
        )
        # Find both arrays: text first, label second typically
        arrays = re.findall(
            r"array\(\s*\[(.*?)\]\s*(?:,\s*dtype\s*=\s*object)?\s*\)",
            s,
            flags=re.S,
        )
        if len(arrays) >= 2:
            def _split_quoted(blob: str) -> list[str]:
                return re.findall(r"'([^']*)'|\"([^\"]*)\"", blob)

            def _flat(pairs: list[tuple[str, str]]) -> list[str]:
                return [a or b for a, b in pairs]

            texts = _flat(_split_quoted(arrays[0]))
            labels = _flat(_split_quoted(arrays[1]))
            if texts and labels and len(texts) == len(labels):
                return labels, texts
            if texts:
                labs = list("ABCD")[: len(texts)]
                return labs, texts
        try:
            import ast

            raw = ast.literal_eval(s)
        except Exception:
            return (["A", "B", "C", "D"], [s[:200]])
    if isinstance(raw, dict):
        labels = list(raw.get("label") or raw.get("labels") or ["A", "B", "C", "D"])
        texts = list(raw.get("text") or raw.get("texts") or [])
        labels = [str(x) for x in labels]
        texts = [str(x) for x in texts]
        return labels, texts
    return (["A", "B", "C", "D"], [])

