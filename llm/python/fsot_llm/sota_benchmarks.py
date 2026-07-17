"""
Run FSOT observer against real packs under D:\\training data.

Default limits are small for iterative refine; raise with --limit for full runs.
Always logs to llm/benchmarks/ledgers/ with archive status.
"""
from __future__ import annotations

import argparse
import json
import re
import traceback
from typing import Any, Callable, Optional

from fsot_llm.archive_verify import check_archive, write_ledger
from fsot_llm.external_data import (
    discover_packs,
    extract_final_number,
    extract_gsm8k_gold,
    extract_mc_letter,
    load_pack_rows,
    normalize_arc_choices,
    registry_report,
)
from fsot_llm.paths import ensure_sys_path
from fsot_llm.superposition import TopicFold, build_default_bank
from fsot_llm.superposed_generate import generate_with_depth

ensure_sys_path()


GenerateFn = Callable[[str], str]


def _default_generate(use_superposition: bool = True) -> GenerateFn:
    """
    Route each call through FSOT domain map (pack/domain/D_eff → pathway).
    Prefer pathway-isolated adapters when trained.
    """
    from fsot_llm.domain_routing import (
        inject_route_context,
        register_benchmarks_in_bank,
        resolve_allocation,
    )

    try:
        from fsot_llm.pathway_adapters import (
            latest_pathway_set,
            load_pathway_observer,
        )

        if latest_pathway_set() is not None:
            cache: dict = {}

            # Math CoT needs headroom; short caps truncate #### answers.
            _TOK = {
                "math": 512,
                "code": 384,
                "mcq": 64,
                "ontology": 128,
            }

            def gen_routed(prompt: str, pack_id: str | None = None) -> str:
                alloc = resolve_allocation(prompt, pack_id=pack_id)
                routed = inject_route_context(prompt, alloc)
                pw = alloc.pathway_key
                if pw not in cache:
                    print(
                        f"  [FSOT_ROUTE pack={alloc.pack_id} "
                        f"domain={alloc.lean_domain} D_eff={alloc.D_eff} "
                        f"pathway={pw}]",
                        flush=True,
                    )
                    cache[pw] = load_pathway_observer(pw)
                return cache[pw].generate_text(
                    routed,
                    max_new_tokens=_TOK.get(pw, 256),
                    temperature=0.0,
                )

            # sota evals call gen(prompt) only — wrap with pack-aware helpers
            def gen(prompt: str) -> str:
                return gen_routed(prompt, pack_id=None)

            gen.for_pack = lambda pack_id: (  # type: ignore[attr-defined]
                lambda prompt, _p=pack_id: gen_routed(prompt, pack_id=_p)
            )
            return gen
    except Exception as exc:
        print(f"  [pathway route unavailable: {exc}]", flush=True)

    from fsot_llm.models import load_coder_observer

    obs = load_coder_observer()
    bank = register_benchmarks_in_bank()

    def gen(prompt: str) -> str:
        alloc = resolve_allocation(prompt)
        routed = inject_route_context(prompt, alloc)
        if use_superposition:
            return generate_with_depth(
                routed,
                observer=obs,
                bank=bank,
                topic_id=alloc.pack_id,
                temperature=0.0,
                max_new_tokens=256,
                memory_mode=alloc.memory_mode,
            )["text"]
        return obs.generate_text(routed, max_new_tokens=256, temperature=0.0)

    return gen


def _benchmark_topics() -> list[tuple[str, str, float, tuple[str, ...], str]]:
    return [
        (
            "gsm8k",
            "Grade-school math (GSM8K)",
            11.0,
            ("math", "benchmark"),
            "GSM8K: multi-step arithmetic. Put the final numeric answer after ####.",
        ),
        (
            "humaneval",
            "HumanEval code synthesis",
            12.0,
            ("code", "benchmark"),
            "HumanEval: complete the Python function. Prefer correct, minimal code.",
        ),
        (
            "mmlu",
            "MMLU multi-subject",
            16.0,
            ("knowledge", "benchmark"),
            "MMLU: answer A/B/C/D from world knowledge. Prefer Answer: <letter>.",
        ),
        (
            "arc",
            "ARC science QA",
            15.0,
            ("science", "benchmark"),
            "ARC: grade-school science multiple choice. Answer with A/B/C/D.",
        ),
        (
            "hellaswag",
            "HellaSwag commonsense",
            9.0,
            ("commonsense", "benchmark"),
            "HellaSwag: pick the most plausible continuation ending.",
        ),
        (
            "bbh",
            "BIG-Bench Hard reasoning",
            13.0,
            ("reasoning", "benchmark"),
            "BBH: multi-step reasoning; show steps then final answer.",
        ),
    ]



def _pack_generate(generate: GenerateFn, pack_id: str) -> GenerateFn:
    """Bind pack_id so routing uses domain map D_eff, not prompt keywords alone."""
    for_pack = getattr(generate, "for_pack", None)
    if callable(for_pack):
        return for_pack(pack_id)
    # fallback: prepend will still happen if generate uses resolve on text
    from fsot_llm.domain_routing import inject_route_context, resolve_allocation

    def wrapped(prompt: str) -> str:
        alloc = resolve_allocation(prompt, pack_id=pack_id)
        return generate(inject_route_context(prompt, alloc))

    return wrapped


def eval_gsm8k(generate: GenerateFn, limit: int = 20) -> dict[str, Any]:
    generate = _pack_generate(generate, 'gsm8k_test')
    rows = load_pack_rows("gsm8k_test", limit=limit)
    # Thin prompt — measurement emission only (no rule-theater bloat)
    hits = 0
    details = []
    for i, row in enumerate(rows):
        q = row.get("question") or row.get("prompt") or ""
        gold = extract_gsm8k_gold(row.get("answer") or "")
        prompt = (
            "Solve the grade-school math problem with pure arithmetic "
            "(no code, no Python). Show brief reasoning with <<calc=result>> "
            "when helpful. Never use one-line stubs. "
            "End with #### <number> as the final answer.\n\n"
            f"Problem: {q}"
        )
        try:
            ans = generate(prompt)
            pred = extract_final_number(ans)
            ok = pred is not None and _num_eq(pred, gold)
        except Exception as exc:
            ans, pred, ok = repr(exc), None, False
        hits += int(ok)
        if i < 5 or not ok:
            details.append(
                {"i": i, "ok": ok, "gold": gold, "pred": pred, "ans": (ans or "")[:240]}
            )
    return {
        "pack": "gsm8k_test",
        "n": len(rows),
        "hits": hits,
        "accuracy": hits / max(len(rows), 1),
        "details": details[:12],
    }


def eval_arc(generate: GenerateFn, pack_id: str = "arc_easy_val", limit: int = 25) -> dict[str, Any]:
    generate = _pack_generate(generate, pack_id)
    rows = load_pack_rows(pack_id, limit=limit)
    hits = 0
    details = []
    for i, row in enumerate(rows):
        q = row.get("question") or ""
        gold = (row.get("answerKey") or row.get("answer") or "").strip().upper()
        labels, texts = normalize_arc_choices(row.get("choices"))
        choices_txt = "\n".join(f"{lab}. {txt}" for lab, txt in zip(labels, texts))
        prompt = (
            "Science multiple-choice. Reply with only the letter of the best answer "
            f"(A/B/C/D).\n\nQuestion: {q}\n{choices_txt}\n\nAnswer:"
        )
        try:
            ans = generate(prompt)
            pred = extract_mc_letter(ans)
            ok = pred is not None and pred == gold
        except Exception as exc:
            ans, pred, ok = repr(exc), None, False
        hits += int(ok)
        if i < 5 or not ok:
            details.append(
                {"i": i, "ok": ok, "gold": gold, "pred": pred, "ans": (ans or "")[:200]}
            )
    return {
        "pack": pack_id,
        "n": len(rows),
        "hits": hits,
        "accuracy": hits / max(len(rows), 1),
        "details": details[:12],
    }


def eval_mmlu(generate: GenerateFn, limit: int = 25) -> dict[str, Any]:
    generate = _pack_generate(generate, 'mmlu_val')
    rows = load_pack_rows("mmlu_val", limit=limit)
    hits = 0
    details = []
    for i, row in enumerate(rows):
        # common mmlu jsonl schemas
        q = row.get("question") or row.get("input") or ""
        choices = row.get("choices") or row.get("options")
        gold = row.get("answer")
        if isinstance(gold, int):
            gold_letter = "ABCD"[gold] if 0 <= gold < 4 else str(gold)
        else:
            gold_letter = str(gold).strip().upper()
            if gold_letter.isdigit():
                gold_letter = "ABCD"[int(gold_letter)] if int(gold_letter) < 4 else gold_letter

        if isinstance(choices, list):
            choices_txt = "\n".join(
                f"{lab}. {txt}" for lab, txt in zip("ABCD", choices)
            )
        elif isinstance(choices, dict):
            choices_txt = "\n".join(f"{k}. {v}" for k, v in choices.items())
        else:
            # subject / answer fields variant
            a = row.get("A") or row.get("option_a")
            if a is not None:
                choices_txt = "\n".join(
                    f"{lab}. {row.get(lab)}" for lab in "ABCD" if row.get(lab) is not None
                )
            else:
                choices_txt = str(choices or "")

        prompt = (
            "Multiple-choice knowledge question. Reply with only the letter "
            f"(A/B/C/D).\n\nQuestion: {q}\n{choices_txt}\n\nAnswer:"
        )
        try:
            ans = generate(prompt)
            pred = extract_mc_letter(ans)
            ok = pred is not None and pred == gold_letter[:1]
        except Exception as exc:
            ans, pred, ok = repr(exc), None, False
        hits += int(ok)
        if i < 5 or not ok:
            details.append(
                {
                    "i": i,
                    "ok": ok,
                    "gold": gold_letter,
                    "pred": pred,
                    "ans": (ans or "")[:200],
                }
            )
    return {
        "pack": "mmlu_val",
        "n": len(rows),
        "hits": hits,
        "accuracy": hits / max(len(rows), 1),
        "details": details[:12],
    }


def eval_humaneval_structure(generate: GenerateFn, limit: int = 10) -> dict[str, Any]:
    generate = _pack_generate(generate, 'humaneval')
    """
    Structural + optional lightweight exec check (not full HumanEval harness).
    Full pass@k can be added later with official evaluator.
    """
    rows = load_pack_rows("humaneval", limit=limit)
    hits = 0
    details = []
    for i, row in enumerate(rows):
        prompt_code = row.get("prompt") or ""
        entry = row.get("entry_point") or ""
        test = row.get("test") or ""
        prompt = (
            "Complete the following Python function. Output code only.\n\n"
            f"{prompt_code}"
        )
        try:
            ans = generate(prompt)
            code = ans
            m = re.search(r"```(?:python)?\s*([\s\S]*?)```", ans)
            if m:
                code = m.group(1)
            # If model only returns body, prepend prompt
            if "def " not in code and "def " in prompt_code:
                code = prompt_code + "\n" + code
            ok = False
            detail = "no_exec"
            if entry and test:
                ns: dict[str, Any] = {}
                try:
                    exec(code, ns, ns)  # noqa: S102
                    exec(test, ns, ns)
                    check = ns.get("check")
                    if callable(check):
                        check(ns[entry])
                        ok = True
                        detail = "passed_check"
                    else:
                        detail = "no_check_fn"
                except Exception as exc:
                    detail = f"fail:{type(exc).__name__}:{exc}"
                    ok = False
            else:
                ok = "def " in code
                detail = "structure_only"
        except Exception as exc:
            code, ok, detail = "", False, repr(exc)
            ans = detail
        hits += int(ok)
        details.append(
            {
                "i": i,
                "task": row.get("task_id"),
                "ok": ok,
                "detail": detail,
                "ans": (ans or "")[:200],
            }
        )
    return {
        "pack": "humaneval",
        "n": len(rows),
        "hits": hits,
        "accuracy": hits / max(len(rows), 1),
        "details": details[:12],
        "note": "lightweight check(); not official HumanEval harness",
    }


def _num_eq(a: str, b: str, tol: float = 1e-6) -> bool:
    try:
        return abs(float(a) - float(b)) <= tol * max(1.0, abs(float(b)))
    except Exception:
        return a.strip() == b.strip()


def run_sota_suite(
    *,
    limit: int = 20,
    use_superposition: bool = True,
    packs: Optional[list[str]] = None,
) -> dict[str, Any]:
    arch = check_archive()
    reg = registry_report()
    generate = _default_generate(use_superposition=use_superposition)

    want = packs or ["gsm8k", "arc_easy", "mmlu", "humaneval"]
    results: dict[str, Any] = {}
    if "gsm8k" in want:
        print("... gsm8k", flush=True)
        results["gsm8k"] = eval_gsm8k(generate, limit=limit)
        print(
            f"  gsm8k {results['gsm8k']['hits']}/{results['gsm8k']['n']} "
            f"= {results['gsm8k']['accuracy']:.3f}",
            flush=True,
        )
    if "arc_easy" in want:
        print("... arc_easy", flush=True)
        results["arc_easy"] = eval_arc(generate, "arc_easy_val", limit=limit)
        print(
            f"  arc {results['arc_easy']['hits']}/{results['arc_easy']['n']} "
            f"= {results['arc_easy']['accuracy']:.3f}",
            flush=True,
        )
    if "mmlu" in want:
        print("... mmlu", flush=True)
        results["mmlu"] = eval_mmlu(generate, limit=limit)
        print(
            f"  mmlu {results['mmlu']['hits']}/{results['mmlu']['n']} "
            f"= {results['mmlu']['accuracy']:.3f}",
            flush=True,
        )
    if "humaneval" in want:
        print("... humaneval", flush=True)
        results["humaneval"] = eval_humaneval_structure(
            generate, limit=min(limit, 15)
        )
        print(
            f"  humaneval {results['humaneval']['hits']}/{results['humaneval']['n']} "
            f"= {results['humaneval']['accuracy']:.3f}",
            flush=True,
        )

    accs = [v["accuracy"] for v in results.values() if "accuracy" in v]
    report = {
        "overall_mean_accuracy": sum(accs) / max(len(accs), 1),
        "use_superposition": use_superposition,
        "limit_per_pack": limit,
        "archive": arch.to_dict(),
        "registry": reg,
        "results": results,
        "fsot_note": (
            "Scores are refine baselines on external packs under D:\\training data. "
            "Archive remains verification authority for FSOT scalars. "
            "Superposition depth: linked topic memory + fixed mode bank."
        ),
    }
    path = write_ledger("sota_external_benchmarks", report)
    report["ledger_path"] = str(path)
    return report


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--limit", type=int, default=15)
    ap.add_argument("--no-superposition", action="store_true")
    ap.add_argument(
        "--packs",
        default="gsm8k,arc_easy,mmlu,humaneval",
        help="comma list: gsm8k,arc_easy,mmlu,humaneval",
    )
    ap.add_argument("--registry-only", action="store_true")
    args = ap.parse_args()
    if args.registry_only:
        print(json.dumps(registry_report(), indent=2))
        return 0
    packs = [p.strip() for p in args.packs.split(",") if p.strip()]
    rep = run_sota_suite(
        limit=args.limit,
        use_superposition=not args.no_superposition,
        packs=packs,
    )
    print(json.dumps({k: rep[k] for k in rep if k != "registry"}, indent=2)[:4000])
    print("ledger:", rep.get("ledger_path"))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
