"""Smoke: FSOT superposition depth + linked-memory generation."""
from __future__ import annotations

import json

from fsot_llm.archive_verify import write_ledger
from fsot_llm.superposition import (
    build_default_bank,
    effective_depth_metric,
)
from fsot_llm.superposed_generate import generate_with_depth


def main() -> int:
    bank = build_default_bank()
    path = bank.save()
    depth = effective_depth_metric(bank)
    usage = bank.mode_usage_report()
    print("=== depth without parameters ===")
    print(json.dumps(depth, indent=2))
    print("topic bank:", path)
    print("modes K=", usage["k_modes"], "beta=", round(usage["beta"], 4))
    for tid, info in usage["topics"].items():
        print(
            f"  {tid:10} D_eff={info['D_eff']:5.1f} "
            f"dom_mode={info['dominant_mode']} H={info['entropy']:.3f}"
        )

    print("\n=== linked context (vision) ===")
    print(bank.linked_context("vision"))

    print("\n=== generate with superposed memory ===")
    out = generate_with_depth(
        "List the five FSOT foundational seeds by symbol only.",
        bank=bank,
        max_new_tokens=120,
        temperature=0.1,
    )
    print("topic:", out["topic_id"], "dominant_mode:", out["dominant_mode"])
    print("alpha:", [round(a, 3) for a in out["alpha"]])
    print("text:", out["text"][:500])

    write_ledger(
        "superposition_smoke",
        {"depth": depth, "usage": usage, "sample_gen": {
            k: out[k] for k in ("topic_id", "dominant_mode", "alpha", "text")
        }},
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
