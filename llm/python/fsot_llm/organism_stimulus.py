"""
FSOT organism stimulus v2 — domain-map pathway isolation.

Diagnosis uses benchmark_domain_map.yaml (pack → domain, D_eff, pathway_key).
Stimulus trains ONLY that pathway_key adapter; never mixes emissions.
"""
from __future__ import annotations

import argparse
import json
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Optional

from fsot_llm.archive_verify import check_archive, write_ledger
from fsot_llm.domain_routing import allocation_for_pack, resolve_allocation
from fsot_llm.external_data import load_pack_rows
from fsot_llm.paths import ensure_sys_path, workspace_root

ensure_sys_path()

# Homeostasis targets (preregistered)
PACK_TARGETS = {
    "gsm8k": 0.45,
    "arc_easy": 0.35,
    "mmlu": 0.30,
    "humaneval": 0.85,
}

# ledger pack name → domain map pack_id
PACK_TO_MAP = {
    "gsm8k": "gsm8k_test",
    "arc_easy": "arc_easy_val",
    "mmlu": "mmlu_val",
    "humaneval": "humaneval",
}

POISON_EPS = 0.05  # absolute accuracy — reject if any non-target drops more


def latest_sota_ledger() -> Path:
    root = workspace_root() / "llm" / "benchmarks" / "ledgers"
    cands = sorted(
        f
        for f in root.glob("*sota_external_benchmarks.json")
        if not f.name.startswith("latest_")
    )
    if not cands:
        raise FileNotFoundError("no sota ledger")
    return cands[-1]


def diagnose(ledger: dict[str, Any]) -> dict[str, Any]:
    results = ledger.get("results") or {}
    pathways: list[dict[str, Any]] = []

    for pack, res in results.items():
        acc = float(res.get("accuracy") or 0.0)
        target = PACK_TARGETS.get(pack, 0.4)
        deficit = max(0.0, target - acc)
        details = res.get("details") or []
        fails = [d for d in details if not d.get("ok")]

        map_id = PACK_TO_MAP.get(pack, pack)
        try:
            alloc = allocation_for_pack(map_id)
        except KeyError:
            alloc = resolve_allocation(pack_id=map_id) if False else None
            # soft fallback
            from fsot_llm.domain_routing import DomainAllocation

            alloc = DomainAllocation(
                pack_id=map_id,
                display=pack,
                lean_domain="consciousness",
                D_eff=12.0,
                pathway_key="ontology",
                emission="free",
                tags=(),
                route_cues=(),
                memory_mode="minimal",
            )

        pred_letters = Counter(
            str(d.get("pred")).upper()
            for d in details
            if d.get("pred") and str(d.get("pred")).upper() in "ABCD"
        )
        modes: list[str] = []
        if deficit <= 0.02:
            modes.append("homeostatic_ok")
        if pack in ("arc_easy", "mmlu") and pred_letters:
            tot = sum(pred_letters.values())
            if pred_letters.get("D", 0) / max(tot, 1) >= 0.5:
                modes.append("mcq_letter_collapse_to_D")
            if pred_letters.get("A", 0) / max(tot, 1) >= 0.5:
                modes.append("mcq_letter_collapse_to_A")
        if pack == "gsm8k" and fails:
            stub_n = sum(
                1
                for d in fails
                if "read quantities carefully" in str(d.get("ans") or "").lower()
            )
            if stub_n >= max(1, len(fails) // 3):
                modes.append("pathogenic_stub_emission")  # emission collapse
            modes.append("math_chain_misread_or_arithmetic")
        if pack == "humaneval" and deficit > 0.02:
            modes.append("code_exec_or_structure")

        dose = 0
        if deficit > 0.02:
            dose = int(20 * (1.0 + 5.0 * deficit))
            dose = min(max(dose, 24), 150)

        pathways.append(
            {
                "pack": pack,
                "map_pack_id": alloc.pack_id,
                "lean_domain": alloc.lean_domain,
                "D_eff": alloc.D_eff,
                "pathway_key": alloc.pathway_key,
                "emission": alloc.emission,
                "accuracy": acc,
                "target": target,
                "deficit": deficit,
                "n": res.get("n"),
                "hits": res.get("hits"),
                "failure_modes": modes,
                "pred_letter_hist": dict(pred_letters),
                "stimulus_dose": dose,
                "understimulated": deficit > 0.02,
                "sample_fails": fails[:3],
            }
        )

    pathways.sort(key=lambda p: p["deficit"], reverse=True)

    # group understimulated packs by pathway_key (never mix emissions across keys)
    by_key: dict[str, list[str]] = defaultdict(list)
    for p in pathways:
        if p["understimulated"]:
            by_key[p["pathway_key"]].append(p["pack"])

    return {
        "diagnosed_at": datetime.now(timezone.utc).isoformat(),
        "archive": check_archive().to_dict(),
        "source_mean_accuracy": ledger.get("overall_mean_accuracy"),
        "pathways": pathways,
        "priority_packs": [p["pack"] for p in pathways if p["understimulated"]],
        "priority_pathway_keys": list(by_key.keys()),
        "packs_by_pathway_key": dict(by_key),
        "organism_note": (
            "v2: each pack maps to domain+D_eff+pathway_key. "
            "Stimulus deepens ONE pathway_key at a time — no emission mixing."
        ),
        "poison_eps": POISON_EPS,
    }


def deepen_understimulated(
    diagnosis: dict[str, Any],
    *,
    epochs: int = 3,
    only_pathway: Optional[str] = None,
) -> list[dict[str, Any]]:
    """
    For each understimulated pathway_key (or only_pathway), deepen that adapter only.
    """
    from fsot_llm.pathway_adapters import deepen_pathway, latest_pathway_set

    if latest_pathway_set() is None:
        raise FileNotFoundError(
            "No pathway adapter set. Run: python -m fsot_llm.pathway_adapters --train"
        )

    keys = diagnosis.get("priority_pathway_keys") or []
    if only_pathway:
        keys = [only_pathway]
    # order: largest total deficit first
    deficit_by_key: dict[str, float] = defaultdict(float)
    dose_by_key: dict[str, int] = defaultdict(int)
    for p in diagnosis["pathways"]:
        if p["understimulated"]:
            deficit_by_key[p["pathway_key"]] += p["deficit"]
            dose_by_key[p["pathway_key"]] = max(
                dose_by_key[p["pathway_key"]], int(p["stimulus_dose"])
            )
    keys = sorted(keys, key=lambda k: deficit_by_key[k], reverse=True)

    reports = []
    for key in keys:
        extra = dose_by_key.get(key, 80)
        if key == "math":
            # more GSM8K mass for math deepen
            extra = max(extra, 100)
        print(
            f"=== organism deepen pathway_key={key} dose~{extra} (isolated) ===",
            flush=True,
        )
        rep = deepen_pathway(
            key,
            epochs=epochs,
            math_extra=extra if key == "math" else 0,
        )
        reports.append(rep)
    return reports


def anti_poison_gate(
    before: dict[str, float],
    after: dict[str, float],
    *,
    target_pathway_keys: Optional[list[str]] = None,
) -> dict[str, Any]:
    """
    GREEN only if no pathway drops more than POISON_EPS.
    target_pathway_keys may be allowed to change freely; others must hold.
    """
    deltas = {}
    poisons = []
    for pack, b in before.items():
        a = after.get(pack, b)
        d = a - b
        deltas[pack] = d
        if d < -POISON_EPS:
            poisons.append({"pack": pack, "delta": d, "before": b, "after": a})
    ok = len(poisons) == 0
    return {
        "ok": ok,
        "poison_eps": POISON_EPS,
        "deltas": deltas,
        "poisons": poisons,
        "promote": ok,
        "note": "Promote pathway set only if ok; else keep previous latest",
    }


def run_cycle(
    *,
    stimulate: bool = False,
    reeval: bool = False,
    epochs: int = 3,
    limit: int = 12,
    only_pathway: Optional[str] = None,
    ledger_path: Optional[Path] = None,
) -> dict[str, Any]:
    from fsot_llm.pathway_adapters import latest_pathway_set

    src = ledger_path or latest_sota_ledger()
    ledger = json.loads(src.read_text(encoding="utf-8"))
    before_acc = {
        k: float(v.get("accuracy") or 0)
        for k, v in (ledger.get("results") or {}).items()
    }
    diagnosis = diagnose(ledger)
    dpath = write_ledger("organism_diagnosis_v2", diagnosis)

    print(
        json.dumps(
            {
                "priority_packs": diagnosis["priority_packs"],
                "priority_pathway_keys": diagnosis["priority_pathway_keys"],
                "packs_by_pathway_key": diagnosis["packs_by_pathway_key"],
                "rows": [
                    {
                        k: p[k]
                        for k in (
                            "pack",
                            "lean_domain",
                            "D_eff",
                            "pathway_key",
                            "emission",
                            "accuracy",
                            "target",
                            "deficit",
                            "stimulus_dose",
                            "failure_modes",
                        )
                    }
                    for p in diagnosis["pathways"]
                ],
            },
            indent=2,
        ),
        flush=True,
    )

    train_reps: list[dict[str, Any]] = []
    prev_set = latest_pathway_set()
    if stimulate:
        train_reps = deepen_understimulated(
            diagnosis, epochs=epochs, only_pathway=only_pathway
        )

    eval_rep = None
    gate = None
    if reeval:
        from fsot_llm.sota_benchmarks import run_sota_suite

        # Match prior ledger sample size when possible (avoid false poison from n=8 vs n=12)
        prior_ns = [
            int(v.get("n") or 0)
            for v in (ledger.get("results") or {}).values()
            if v.get("n")
        ]
        if prior_ns:
            limit = max(min(prior_ns), 1) if min(prior_ns) == max(prior_ns) else limit
            # if all packs share same n, use that n for fair delta
            if min(prior_ns) == max(prior_ns):
                limit = prior_ns[0]
                print(f"=== re-eval limit matched to prior n={limit} ===", flush=True)

        print("=== re-eval (domain-routed) ===", flush=True)
        eval_rep = run_sota_suite(
            limit=limit,
            use_superposition=True,
            packs=list((ledger.get("results") or {}).keys())
            or ["gsm8k", "arc_easy", "mmlu", "humaneval"],
        )

        after_acc = {
            k: float(v.get("accuracy") or 0)
            for k, v in (eval_rep.get("results") or {}).items()
        }
        gate = anti_poison_gate(before_acc, after_acc)
        print("anti_poison_gate", json.dumps(gate, indent=2), flush=True)

        # If poisoned, restore previous pathway set pointer
        # (also when reeval-only after an external --deepen changed latest)
        cur_set = latest_pathway_set()
        if not gate["ok"] and prev_set is not None and cur_set != prev_set:
            from fsot_llm.pathway_adapters import pathway_root

            (pathway_root() / "latest.txt").write_text(
                str(prev_set), encoding="utf-8"
            )
            gate["rolled_back_to"] = str(prev_set)
            print(f"ROLLBACK pathway latest → {prev_set}", flush=True)
        elif not gate["ok"] and not stimulate:
            # reeval after failed deepen without stimulate flag: roll to source's best
            from fsot_llm.pathway_adapters import pathway_root

            # Prefer explicit prev if we captured; else leave for manual
            if prev_set is not None:
                (pathway_root() / "latest.txt").write_text(
                    str(prev_set), encoding="utf-8"
                )
                gate["rolled_back_to"] = str(prev_set)
                print(f"ROLLBACK pathway latest → {prev_set}", flush=True)


    cycle = {
        "version": 2,
        "source_ledger": str(src),
        "diagnosis_ledger": str(dpath),
        "domain_map": "llm/configs/benchmark_domain_map.yaml",
        "train": train_reps,
        "reeval_mean": (eval_rep or {}).get("overall_mean_accuracy"),
        "gate": gate,
        "archive_ok": check_archive().ok,
    }
    write_ledger("organism_cycle_v2", cycle)
    return cycle


def main() -> int:
    ap = argparse.ArgumentParser(description="FSOT organism stimulus v2")
    ap.add_argument("--from-latest-sota", action="store_true")
    ap.add_argument("--ledger", type=Path, default=None)
    ap.add_argument("--stimulate", action="store_true")
    ap.add_argument("--reeval", action="store_true")
    ap.add_argument("--epochs", type=int, default=3)
    ap.add_argument("--limit", type=int, default=12)
    ap.add_argument(
        "--only-pathway",
        type=str,
        default=None,
        help="math|mcq|code|ontology — deepen only this pathway_key",
    )
    ap.add_argument("--diagnose-only", action="store_true")
    args = ap.parse_args()

    if args.diagnose_only:
        src = args.ledger or latest_sota_ledger()
        diagnosis = diagnose(json.loads(src.read_text(encoding="utf-8")))
        path = write_ledger("organism_diagnosis_v2", diagnosis)
        print(json.dumps({"diagnosis": str(path), **{
            k: diagnosis[k]
            for k in (
                "priority_packs",
                "priority_pathway_keys",
                "packs_by_pathway_key",
            )
        }}, indent=2))
        return 0

    rep = run_cycle(
        stimulate=args.stimulate,
        reeval=args.reeval,
        epochs=args.epochs,
        limit=args.limit,
        only_pathway=args.only_pathway,
        ledger_path=args.ledger,
    )
    print(json.dumps(rep, indent=2, default=str)[:4000])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
