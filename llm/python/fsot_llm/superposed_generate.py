"""
Generate with FSOT topic memory: linked exemplars + optional superposed LoRA mix.

Saves memory by retrieving coupled topics instead of loading topic-specific models.
"""
from __future__ import annotations

from pathlib import Path
from typing import Optional

from fsot_llm.models import LoadedObserver, load_coder_observer
from fsot_llm.superposition import TopicFold, TopicMemoryBank, build_default_bank


def resolve_topic(bank: TopicMemoryBank, text: str) -> str:
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



def generate_with_depth(
    prompt: str,
    *,
    observer: Optional[LoadedObserver] = None,
    bank: Optional[TopicMemoryBank] = None,
    topic_id: Optional[str] = None,
    max_new_tokens: int = 256,
    temperature: float = 0.2,
    use_linked_memory: bool = True,
    memory_mode: str = "auto",
) -> dict:
    """
    memory_mode:
      auto    — full FSOT links for ontology; minimal for math/code/MCQ benchmarks
      full    — full linked topic bank (FSOT knowledge tasks)
      minimal — one-line topic tag only (SOTA packs; avoid ontology spam)
      off     — raw prompt

    Returns {text, topic_id, alpha, linked_preview, memory_mode}.
    """
    from fsot_llm.domain_routing import register_benchmarks_in_bank
    bank = bank or register_benchmarks_in_bank()
    obs = observer or load_coder_observer()
    tid = topic_id or resolve_topic(bank, prompt)
    if tid not in bank.entries:
        tid = next(iter(bank.entries))

    mode = memory_mode
    if mode == "auto":
        # Benchmark-style prompts should not be drowned by ontology dump
        pl = prompt.lower()
        if any(
            k in pl
            for k in (
                "grade-school math",
                "####",
                "multiple-choice",
                "reply with only the letter",
                "complete the following python",
                "humaneval",
                "problem:",
            )
        ):
            mode = "minimal"
        elif tid in ("seeds", "scalar", "observer", "archive", "memory", "sota"):
            mode = "full"
        else:
            mode = "minimal"

    alpha = bank.alpha(tid).tolist()
    linked = ""
    if use_linked_memory and mode == "full":
        linked = bank.linked_context(tid)
        full = (
            "FSOT superposed memory (shared modes; linked topics — not extra parameters):\n"
            f"{linked}\n\n"
            f"User question:\n{prompt}"
        )
    elif use_linked_memory and mode == "minimal":
        ent = bank.entries[tid]
        # One short exemplar max — coupling without ontology monologue
        ex = (ent.exemplars[0][:160] if ent.exemplars else "")
        tag = f"[FSOT topic={ent.topic.name} D_eff={ent.topic.D_eff:.0f}]"
        full = f"{tag}\n{ex}\n\n{prompt}" if ex else f"{tag}\n\n{prompt}"
        linked = tag
    else:
        full = prompt

    text = obs.generate_text(
        full, max_new_tokens=max_new_tokens, temperature=temperature
    )
    ent = bank.entries[tid]
    return {
        "text": text,
        "topic_id": tid,
        "topic_name": ent.topic.name,
        "D_eff": ent.topic.D_eff,
        "alpha": alpha,
        "dominant_mode": int(max(range(len(alpha)), key=lambda i: alpha[i])),
        "linked_preview": (linked or "")[:400],
        "memory_mode": mode,
        "observer": obs.name,
    }

