"""Wire domain_routing into pathway_adapters, sota_benchmarks, superposed_generate."""
from __future__ import annotations

import re
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
PY = ROOT / "llm" / "python" / "fsot_llm"


def patch_pathway_adapters() -> None:
    p = PY / "pathway_adapters.py"
    t = p.read_text(encoding="utf-8")
    m = re.search(
        r"def resolve_pathway\(prompt: str\) -> str:.*?(?=\ndef |\Z)",
        t,
        re.S,
    )
    if not m:
        raise SystemExit("resolve_pathway not found")
    new = '''def resolve_pathway(prompt: str, pack_id: str | None = None) -> str:
    """Route via benchmark_domain_map.yaml (domain + D_eff allocation)."""
    from fsot_llm.domain_routing import pathway_for_prompt

    return pathway_for_prompt(prompt, pack_id=pack_id)


'''
    p.write_text(t[: m.start()] + new + t[m.end() :], encoding="utf-8")
    print("OK pathway_adapters.resolve_pathway")


def patch_sota() -> None:
    p = PY / "sota_benchmarks.py"
    t = p.read_text(encoding="utf-8")

    # Ensure gen_routed accepts pack context via closure factory
    if "pack_id=" not in t or "domain_routing" not in t:
        # Replace _default_generate entirely
        start = t.index("def _default_generate")
        end = t.index("def _benchmark_topics")
        new = '''def _default_generate(use_superposition: bool = True) -> GenerateFn:
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
                    routed, max_new_tokens=256, temperature=0.0
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


'''
        t = t[:start] + new + t[end:]

    # Patch eval functions to use pack-aware generate when available
    # eval_gsm8k: generate(prompt) → prefer gen.for_pack
    if "gen_for_pack" not in t:
        t = t.replace(
            "def eval_gsm8k(generate: GenerateFn, limit: int = 20) -> dict[str, Any]:",
            "def eval_gsm8k(generate: GenerateFn, limit: int = 20) -> dict[str, Any]:\n"
            "    generate = _pack_generate(generate, 'gsm8k_test')",
        )
        t = t.replace(
            "def eval_arc(generate: GenerateFn, pack_id: str = \"arc_easy_val\", limit: int = 25) -> dict[str, Any]:",
            "def eval_arc(generate: GenerateFn, pack_id: str = \"arc_easy_val\", limit: int = 25) -> dict[str, Any]:\n"
            "    generate = _pack_generate(generate, pack_id)",
        )
        t = t.replace(
            "def eval_mmlu(generate: GenerateFn, limit: int = 25) -> dict[str, Any]:",
            "def eval_mmlu(generate: GenerateFn, limit: int = 25) -> dict[str, Any]:\n"
            "    generate = _pack_generate(generate, 'mmlu_val')",
        )
        t = t.replace(
            "def eval_humaneval_structure(generate: GenerateFn, limit: int = 10) -> dict[str, Any]:",
            "def eval_humaneval_structure(generate: GenerateFn, limit: int = 10) -> dict[str, Any]:\n"
            "    generate = _pack_generate(generate, 'humaneval')",
        )
        helper = '''
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


'''
        # insert helper before eval_gsm8k
        t = t.replace(
            "def eval_gsm8k",
            helper + "def eval_gsm8k",
            1,
        )

    p.write_text(t, encoding="utf-8")
    print("OK sota_benchmarks domain routing")


def patch_superposed() -> None:
    p = PY / "superposed_generate.py"
    t = p.read_text(encoding="utf-8")
    if "domain_routing" in t:
        print("superposed already wired")
        return
    # replace resolve_topic body to prefer domain map
    m = re.search(
        r"def resolve_topic\(bank: TopicMemoryBank, text: str\) -> str:.*?(?=\ndef )",
        t,
        re.S,
    )
    if not m:
        raise SystemExit("resolve_topic not found")
    new = '''def resolve_topic(bank: TopicMemoryBank, text: str) -> str:
    """Prefer benchmark_domain_map pack_id; fall back to bank topic keywords."""
    from fsot_llm.domain_routing import resolve_allocation

    alloc = resolve_allocation(text)
    if alloc.pack_id in bank.entries:
        return alloc.pack_id
    # map pathway to nearest existing bank topic
    pathway_fallback = {
        "math": "gsm8k" if "gsm8k" in bank.entries else "scalar",
        "mcq": "mmlu" if "mmlu" in bank.entries else "vision",
        "code": "code",
        "ontology": "seeds",
    }
    fb = pathway_fallback.get(alloc.pathway_key, "scalar")
    if fb in bank.entries:
        return fb
    return next(iter(bank.entries))


'''
    t = t[: m.start()] + new + t[m.end() :]
    # ensure generate_with_depth can register benchmarks
    t = t.replace(
        "bank = bank or build_default_bank()",
        "from fsot_llm.domain_routing import register_benchmarks_in_bank\n"
        "    bank = bank or register_benchmarks_in_bank()",
    )
    p.write_text(t, encoding="utf-8")
    print("OK superposed_generate")


def main() -> None:
    patch_pathway_adapters()
    patch_sota()
    patch_superposed()
    print("done")


if __name__ == "__main__":
    main()
