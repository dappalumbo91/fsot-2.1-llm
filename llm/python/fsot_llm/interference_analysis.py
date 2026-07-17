"""
Cross-pathway interference analysis — proactive vs poison.

Compares two SOTA ledgers (or organism cycle delta) and attributes
regression to shared-mode collapse / stimulus imbalance / emission conflict.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Optional

from fsot_llm.archive_verify import write_ledger
from fsot_llm.paths import workspace_root


def _sota_files() -> list[Path]:
    led = workspace_root() / "llm" / "benchmarks" / "ledgers"
    return sorted(
        f
        for f in led.glob("*sota_external_benchmarks.json")
        if not f.name.startswith("latest_")
    )


def load_acc(path: Path) -> dict[str, float]:
    d = json.loads(path.read_text(encoding="utf-8"))
    return {
        k: float(v.get("accuracy") or 0.0)
        for k, v in (d.get("results") or {}).items()
    }


def analyze_pair(
    before: Path,
    after: Path,
    *,
    stimulus_meta: Optional[dict[str, Any]] = None,
    poison_eps: float = 0.05,
) -> dict[str, Any]:
    a0 = load_acc(before)
    a1 = load_acc(after)
    packs = sorted(set(a0) | set(a1))
    rows = []
    for p in packs:
        dlt = a1.get(p, 0.0) - a0.get(p, 0.0)
        if dlt >= poison_eps:
            kind = "proactive_gain"
        elif dlt <= -poison_eps:
            kind = "poison_regression"
        elif abs(dlt) < 1e-9:
            kind = "flat"
        else:
            kind = "mild_change"
        rows.append(
            {
                "pack": p,
                "before": a0.get(p),
                "after": a1.get(p),
                "delta": dlt,
                "kind": kind,
            }
        )

    gains = [r for r in rows if r["kind"] == "proactive_gain"]
    poisons = [r for r in rows if r["kind"] == "poison_regression"]

    # Stimulus imbalance if meta provided
    stim_share = {}
    if stimulus_meta and stimulus_meta.get("plan"):
        total = sum(int(x.get("n_rows") or 0) for x in stimulus_meta["plan"])
        for x in stimulus_meta["plan"]:
            stim_share[x["pack"]] = {
                "n_rows": x.get("n_rows"),
                "share": (x.get("n_rows") or 0) / max(total, 1),
                "failure_modes": x.get("modes"),
                "dominant_mode": x.get("dominant_mode"),
            }

    modes = {
        x.get("pack"): x.get("dominant_mode")
        for x in (stimulus_meta or {}).get("plan") or []
    }
    mode_collapse = len(set(m for m in modes.values() if m is not None)) <= 1 and len(modes) > 1

    emission_conflict = {
        "mcq": "single letter A-D",
        "gsm8k": "chain + #### number",
        "humaneval": "full Python function",
        "note": "Shared output head without routing → last strong train dist wins",
    }

    report = {
        "before_ledger": str(before),
        "after_ledger": str(after),
        "pathways": rows,
        "proactive_gains": gains,
        "poison_regressions": poisons,
        "mean_delta": sum(r["delta"] for r in rows) / max(len(rows), 1),
        "stimulus_share": stim_share,
        "mode_collapse_all_same_dominant": mode_collapse,
        "dominant_modes": modes,
        "emission_conflict": emission_conflict,
        "fsot_diagnosis": (
            "Gains and losses co-occurring with shared dominant mode and "
            "MCQ-heavy stimulus indicate destructive interference on one "
            "parameter locus — not independent multi-domain learning."
            if (gains and poisons and mode_collapse)
            else "See pathway table; check stimulus share and mode assignments."
        ),
        "fsot_prescription": [
            "Isolate pathway LoRAs (math / mcq / code / ontology)",
            "Superpose only at inference via alpha(topic)",
            "Never merge_and_unload specialists into one mixed train",
            "Homeostatic maintenance dose for non-deficit pathways",
            "Promote only if no pathway delta < -poison_eps",
        ],
        "poison_eps": poison_eps,
    }
    return report


def analyze_organism_cycle() -> dict[str, Any]:
    led = workspace_root() / "llm" / "benchmarks" / "ledgers"
    cycles = sorted(
        f for f in led.glob("*organism_cycle.json") if not f.name.startswith("latest_")
    )
    if not cycles:
        raise FileNotFoundError("no organism_cycle ledger")
    cycle = json.loads(cycles[-1].read_text(encoding="utf-8"))
    before = Path(cycle["source_ledger"])
    # after = newest sota after cycle time, or latest sota
    sotas = _sota_files()
    after = sotas[-1]
    stim = cycle.get("stimulus_meta")
    rep = analyze_pair(before, after, stimulus_meta=stim)
    rep["organism_cycle"] = str(cycles[-1])
    rep["cycle_delta_field"] = cycle.get("delta")
    return rep


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--organism-cycle", action="store_true")
    ap.add_argument("--before", type=Path, default=None)
    ap.add_argument("--after", type=Path, default=None)
    args = ap.parse_args()
    if args.organism_cycle or (not args.before):
        rep = analyze_organism_cycle()
    else:
        rep = analyze_pair(args.before, args.after)
    path = write_ledger("interference_analysis", rep)
    print(json.dumps(rep, indent=2))
    print("ledger", path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
