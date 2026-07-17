"""
FSOT domain + D_eff routing for benchmarks and prompts.

Benchmarks are not a separate ontology — they are allocated folds of the same
medium. Context routing injects pack→domain→D_eff; pathway adapters load by
pathway_key so emissions don't poison each other.
"""
from __future__ import annotations

import re
from dataclasses import asdict, dataclass
from functools import lru_cache
from pathlib import Path
from typing import Any, Optional

import yaml

from fsot_llm.paths import workspace_root
from fsot_llm.superposition import TopicFold, TopicMemoryBank, superposition_weights, build_mode_loci


@dataclass(frozen=True)
class DomainAllocation:
    pack_id: str
    display: str
    lean_domain: str
    D_eff: float
    pathway_key: str
    emission: str
    tags: tuple[str, ...]
    route_cues: tuple[str, ...]
    memory_mode: str
    notes: str = ""
    archive_cluster: str = ""

    def to_topic_fold(self) -> TopicFold:
        # Pathway/emission → orthogonal fold offsets (preregistered, not free fits)
        # so benchmarks don't all collapse to the same alpha mode.
        pathway_rho = {
            "ontology": 1.0,
            "math": 0.55,
            "mcq": 0.75,
            "code": 0.35,
        }
        pathway_hits = {
            "ontology": 0.05,
            "math": 0.25,
            "mcq": 0.55,
            "code": 0.85,
        }
        return TopicFold(
            topic_id=self.pack_id,
            name=self.display,
            D_eff=self.D_eff,
            rho=pathway_rho.get(self.pathway_key, 0.6),
            recent_hits=pathway_hits.get(self.pathway_key, 0.3),
            observed=True,
            tags=self.tags + (self.lean_domain, self.pathway_key, self.emission),
        )


    def context_prefix(self, template: str) -> str:
        return template.format(
            pack_id=self.pack_id,
            lean_domain=self.lean_domain,
            D_eff=f"{self.D_eff:.1f}",
            pathway_key=self.pathway_key,
            emission=self.emission,
        ).strip()

    def to_dict(self) -> dict[str, Any]:
        d = asdict(self)
        d["tags"] = list(self.tags)
        d["route_cues"] = list(self.route_cues)
        return d


@lru_cache(maxsize=1)
def _load_map() -> dict[str, Any]:
    path = workspace_root() / "llm" / "configs" / "benchmark_domain_map.yaml"
    with path.open(encoding="utf-8") as f:
        return yaml.safe_load(f)


def map_path() -> Path:
    return workspace_root() / "llm" / "configs" / "benchmark_domain_map.yaml"


def all_allocations() -> dict[str, DomainAllocation]:
    raw = _load_map()
    out: dict[str, DomainAllocation] = {}
    for pid, row in (raw.get("benchmarks") or {}).items():
        out[pid] = DomainAllocation(
            pack_id=row.get("pack_id") or pid,
            display=str(row.get("display") or pid),
            lean_domain=str(row.get("lean_domain") or "consciousness"),
            D_eff=float(row.get("D_eff") or 12.0),
            pathway_key=str(row.get("pathway_key") or "ontology"),
            emission=str(row.get("emission") or "free"),
            tags=tuple(row.get("tags") or ()),
            route_cues=tuple(row.get("route_cues") or ()),
            memory_mode=str(row.get("memory_mode") or "minimal"),
            notes=str(row.get("notes") or ""),
            archive_cluster=str(row.get("archive_cluster") or ""),
        )
    return out


def pathway_defaults() -> dict[str, dict[str, Any]]:
    return dict(_load_map().get("pathways") or {})


def prefix_template() -> str:
    return str(
        _load_map().get("context_prefix_template")
        or "[FSOT_ROUTE pack={pack_id} domain={lean_domain} D_eff={D_eff} pathway={pathway_key}]"
    )


def allocation_for_pack(pack_id: str) -> DomainAllocation:
    allocs = all_allocations()
    if pack_id not in allocs:
        # aliases
        aliases = {
            "gsm8k": "gsm8k_test",
            "arc_easy": "arc_easy_val",
            "arc": "arc_easy_val",
            "mmlu": "mmlu_val",
            "hellaswag": "hellaswag_val",
            "winogrande": "winogrande_val",
            "truthfulqa": "truthfulqa_val",
            "bbh": "bbh_mix",
        }
        pack_id = aliases.get(pack_id, pack_id)
    if pack_id not in allocs:
        raise KeyError(
            f"pack {pack_id!r} not in benchmark_domain_map.yaml; "
            f"known={list(allocs)}"
        )
    return allocs[pack_id]


def resolve_allocation(
    text: str = "",
    *,
    pack_id: Optional[str] = None,
) -> DomainAllocation:
    """
    Resolve FSOT domain allocation from explicit pack_id or prompt cues.
    Prefer pack_id when scoring a known benchmark.
    """
    if pack_id:
        return allocation_for_pack(pack_id)

    t = (text or "").lower()
    allocs = all_allocations()
    best: Optional[DomainAllocation] = None
    best_score = 0
    for alloc in allocs.values():
        score = 0
        for cue in alloc.route_cues:
            if cue.lower() in t:
                # longer cues weigh more
                score += max(1, len(cue) // 4)
        if score > best_score:
            best_score = score
            best = alloc

    if best is not None and best_score > 0:
        return best

    # FSOT ontology fallback (full medium)
    return DomainAllocation(
        pack_id="fsot_ontology",
        display="FSOT ontology",
        lean_domain="consciousness",
        D_eff=25.0,
        pathway_key="ontology",
        emission="free",
        tags=("ontology", "fsot"),
        route_cues=(),
        memory_mode="full",
        notes="Default full-medium fold when no benchmark cues match.",
    )


def inject_route_context(prompt: str, alloc: DomainAllocation) -> str:
    """Prepend preregistered FSOT fold line — enables consistent routing/memory."""
    prefix = alloc.context_prefix(prefix_template())
    if prefix in prompt:
        return prompt
    return f"{prefix}\n\n{prompt}"


def pathway_for_prompt(prompt: str, *, pack_id: Optional[str] = None) -> str:
    return resolve_allocation(prompt, pack_id=pack_id).pathway_key


def register_benchmarks_in_bank(bank: Optional[TopicMemoryBank] = None) -> TopicMemoryBank:
    """
    Seed topic memory with every benchmark as a TopicFold at its D_eff.
    This is what was missing: benchmarks enter the same fold geometry as FSOT topics.
    """
    from fsot_llm.superposition import build_default_bank

    bank = bank or build_default_bank()
    for alloc in all_allocations().values():
        fold = alloc.to_topic_fold()
        exemplar = (
            f"Benchmark {alloc.display}: lean_domain={alloc.lean_domain}, "
            f"D_eff={alloc.D_eff}, pathway={alloc.pathway_key}, "
            f"emission={alloc.emission}. {alloc.notes}"
        )
        bank.add(fold, [exemplar], lean_domain=alloc.lean_domain)
    return bank


def routing_report(prompt: str = "", pack_id: Optional[str] = None) -> dict[str, Any]:
    alloc = resolve_allocation(prompt, pack_id=pack_id)
    fold = alloc.to_topic_fold()
    modes = build_mode_loci(8)
    from fsot_bridge import FSOTEngine

    eng = FSOTEngine()
    alpha = superposition_weights(fold, modes, eng)
    return {
        "allocation": alloc.to_dict(),
        "context_prefix": alloc.context_prefix(prefix_template()),
        "alpha": alpha.tolist(),
        "dominant_mode": int(alpha.argmax()),
        "S": eng.scalar_float(D_eff=alloc.D_eff, observed=True),
    }


def registry_table() -> list[dict[str, Any]]:
    rows = []
    for alloc in all_allocations().values():
        rows.append(
            {
                "pack_id": alloc.pack_id,
                "display": alloc.display,
                "lean_domain": alloc.lean_domain,
                "D_eff": alloc.D_eff,
                "pathway_key": alloc.pathway_key,
                "emission": alloc.emission,
            }
        )
    rows.sort(key=lambda r: (-r["D_eff"], r["pack_id"]))
    return rows


def main() -> int:
    import json
    import argparse

    ap = argparse.ArgumentParser(description="FSOT benchmark domain routing")
    ap.add_argument("--table", action="store_true")
    ap.add_argument("--pack", type=str, default=None)
    ap.add_argument("--prompt", type=str, default="")
    args = ap.parse_args()
    if args.table:
        print(json.dumps(registry_table(), indent=2))
        return 0
    print(json.dumps(routing_report(args.prompt, pack_id=args.pack), indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
