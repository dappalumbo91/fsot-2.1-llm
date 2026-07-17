from pathlib import Path

p = Path(__file__).resolve().parents[1] / "llm" / "python" / "fsot_llm" / "sota_benchmarks.py"
text = p.read_text(encoding="utf-8")
start = text.index("def _default_generate")
end = text.index("def _benchmark_topics")
new = '''def _default_generate(use_superposition: bool = True) -> GenerateFn:
    """Prefer pathway-isolated adapters when present (anti-poison)."""
    try:
        from fsot_llm.pathway_adapters import (
            latest_pathway_set,
            load_pathway_observer,
            resolve_pathway,
        )
        if latest_pathway_set() is not None:
            cache: dict = {}

            def gen_routed(prompt: str) -> str:
                pw = resolve_pathway(prompt)
                if pw not in cache:
                    print(f"  [route->{pw}]", flush=True)
                    cache[pw] = load_pathway_observer(pw)
                return cache[pw].generate_text(
                    prompt, max_new_tokens=256, temperature=0.0
                )

            return gen_routed
    except Exception as exc:
        print(f"  [pathway route unavailable: {exc}]", flush=True)

    from fsot_llm.models import load_coder_observer

    obs = load_coder_observer()
    bank = build_default_bank()
    for tid, name, d_eff, tags, ex in _benchmark_topics():
        bank.add(
            TopicFold(tid, name, d_eff, tags=tags),
            [ex],
        )

    def gen(prompt: str) -> str:
        if use_superposition:
            return generate_with_depth(
                prompt,
                observer=obs,
                bank=bank,
                temperature=0.0,
                max_new_tokens=256,
                memory_mode="auto",
            )["text"]
        return obs.generate_text(prompt, max_new_tokens=256, temperature=0.0)

    return gen


'''
p.write_text(text[:start] + new + text[end:], encoding="utf-8")
assert "gen_routed" in p.read_text(encoding="utf-8")
print("patched", p)
