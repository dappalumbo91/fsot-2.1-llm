"""
FSOT superposition depth — fixed parameter modes, topic-linked memory.

Depth comes from superposing shared modes with seed-geometry couplings,
not from growing the parameter count per topic.
"""
from __future__ import annotations

import json
import math
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Iterable, Optional

import numpy as np

from fsot_llm.paths import ensure_sys_path, workspace_root

ensure_sys_path()

from fsot_bridge import FSOTEngine


# ---------------------------------------------------------------------------
# Preregistered geometry (not free fits)
# ---------------------------------------------------------------------------
# Mode count K: fixed budget. Topics reuse these modes.
DEFAULT_K_MODES = 8
# Coupling sharpness from FSOT consciousness factor scale (clamped)
# α_k ∝ exp(β * cos_sim); β derived from psi_con * 8  (preregistered map)
def default_beta(engine: Optional[FSOTEngine] = None) -> float:
    eng = engine or FSOTEngine()
    psi = float(eng.module.PSI_CON)
    return float(psi * 8.0)  # ~5.06 — sharp but not a free fit


@dataclass(frozen=True)
class TopicFold:
    """Observation fold for a topic in the FSOT medium."""

    topic_id: str
    name: str
    D_eff: float
    rho: float = 1.0
    recent_hits: float = 0.0
    observed: bool = True
    tags: tuple[str, ...] = ()

    def feature_vector(self, engine: FSOTEngine) -> np.ndarray:
        """
        Fixed-length feature in seed geometry (not a learned embedding table).

        Components are preregistered transforms of folds + scalar S.
        """
        s = engine.scalar_float(
            D_eff=self.D_eff,
            rho=self.rho,
            recent_hits=self.recent_hits,
            observed=self.observed,
            N=max(self.D_eff, 1.0),
        )
        # Normalize folds into [0,1]-ish channels
        d = self.D_eff / 25.0
        r = min(max(self.rho, 0.0), 2.0) / 2.0
        h = min(max(self.recent_hits, 0.0), 1.0)
        o = 1.0 if self.observed else 0.0
        # Include seed constants as absolute anchors (shared by all topics)
        seeds = engine.seeds()
        phi = float(seeds["phi"])
        k = float(seeds["K"])
        cf = float(seeds["C_factor"])
        # Superposition-friendly unit features
        vec = np.array(
            [
                d,
                r,
                h,
                o,
                math.tanh(s),  # compress S
                math.sin(math.pi * d),
                math.cos(math.pi * d * phi % (2 * math.pi)),
                k,
                cf,
                math.sin(2 * math.pi * h),
                math.cos(2 * math.pi * r),
                (d * r * (0.5 + 0.5 * o)),
            ],
            dtype=np.float64,
        )
        n = np.linalg.norm(vec) + 1e-12
        return vec / n


@dataclass
class ModeLocus:
    """One shared parameter mode's locus in FSOT feature space."""

    mode_id: int
    # Locus is seed-derived from mode index — not trained freely
    fold: TopicFold

    def feature(self, engine: FSOTEngine) -> np.ndarray:
        return self.fold.feature_vector(engine)


def build_mode_loci(k: int = DEFAULT_K_MODES) -> list[ModeLocus]:
    """
    Place K modes across D_eff spectrum (4..25) with alternating observed flag.
    Loci are preregistered functions of mode index only.
    """
    loci: list[ModeLocus] = []
    for i in range(k):
        t = (i + 0.5) / k
        d_eff = 4.0 + 21.0 * t
        rho = 0.5 + 0.5 * math.sin(math.pi * t)
        hits = t
        observed = (i % 2 == 0)
        loci.append(
            ModeLocus(
                mode_id=i,
                fold=TopicFold(
                    topic_id=f"mode_{i}",
                    name=f"FSOT mode {i}",
                    D_eff=d_eff,
                    rho=rho,
                    recent_hits=hits,
                    observed=observed,
                    tags=("mode",),
                ),
            )
        )
    return loci


def cosine(a: np.ndarray, b: np.ndarray) -> float:
    return float(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-12))


def superposition_weights(
    topic: TopicFold,
    modes: list[ModeLocus],
    engine: Optional[FSOTEngine] = None,
    beta: Optional[float] = None,
) -> np.ndarray:
    """α_k = softmax(β · cos(f_topic, f_mode_k))."""
    eng = engine or FSOTEngine()
    b = default_beta(eng) if beta is None else beta
    f = topic.feature_vector(eng)
    logits = np.array(
        [b * cosine(f, m.feature(eng)) for m in modes], dtype=np.float64
    )
    logits = logits - logits.max()
    ex = np.exp(logits)
    return ex / (ex.sum() + 1e-12)


def topic_coupling(
    a: TopicFold,
    b: TopicFold,
    engine: Optional[FSOTEngine] = None,
) -> float:
    """Pairwise topic link strength in [0,1] from FSOT feature cosine."""
    eng = engine or FSOTEngine()
    return 0.5 * (1.0 + cosine(a.feature_vector(eng), b.feature_vector(eng)))


# ---------------------------------------------------------------------------
# Topic memory bank — store descriptors + exemplars, not full weight copies
# ---------------------------------------------------------------------------
@dataclass
class MemoryEntry:
    topic: TopicFold
    exemplars: list[str] = field(default_factory=list)
    meta: dict[str, Any] = field(default_factory=dict)


class TopicMemoryBank:
    """
    Compact topic memory. Linking saves memory: related topics share retrieval
    mass instead of duplicating full parameter blocks.
    """

    def __init__(
        self,
        *,
        k_modes: int = DEFAULT_K_MODES,
        engine: Optional[FSOTEngine] = None,
    ) -> None:
        self.engine = engine or FSOTEngine()
        self.k_modes = k_modes
        self.modes = build_mode_loci(k_modes)
        self.beta = default_beta(self.engine)
        self.entries: dict[str, MemoryEntry] = {}

    def add(
        self,
        topic: TopicFold,
        exemplars: Optional[Iterable[str]] = None,
        **meta: Any,
    ) -> None:
        ex = list(exemplars or [])
        if topic.topic_id in self.entries:
            self.entries[topic.topic_id].exemplars.extend(ex)
            self.entries[topic.topic_id].meta.update(meta)
        else:
            self.entries[topic.topic_id] = MemoryEntry(
                topic=topic, exemplars=ex, meta=dict(meta)
            )

    def alpha(self, topic_id: str) -> np.ndarray:
        e = self.entries[topic_id]
        return superposition_weights(
            e.topic, self.modes, self.engine, self.beta
        )

    def coupling_matrix(self) -> dict[str, Any]:
        ids = list(self.entries.keys())
        n = len(ids)
        mat = np.zeros((n, n), dtype=np.float64)
        for i, a in enumerate(ids):
            for j, b in enumerate(ids):
                mat[i, j] = topic_coupling(
                    self.entries[a].topic,
                    self.entries[b].topic,
                    self.engine,
                )
        return {"ids": ids, "matrix": mat.tolist()}

    def linked_context(
        self,
        topic_id: str,
        *,
        top_m: int = 3,
        max_chars: int = 1200,
    ) -> str:
        """
        Build a memory-efficient context: self exemplars + linked topics'
        exemplars weighted by FSOT coupling (not full parameter reload).
        """
        if topic_id not in self.entries:
            return ""
        self_e = self.entries[topic_id]
        scores: list[tuple[float, str]] = []
        for tid, ent in self.entries.items():
            if tid == topic_id:
                continue
            c = topic_coupling(self_e.topic, ent.topic, self.engine)
            scores.append((c, tid))
        scores.sort(reverse=True)
        parts: list[str] = [
            f"[FSOT topic={self_e.topic.name} D_eff={self_e.topic.D_eff:.2f}]"
        ]
        for ex in self_e.exemplars[:2]:
            parts.append(ex)
        for c, tid in scores[:top_m]:
            if c < 0.55:
                break
            ent = self.entries[tid]
            parts.append(
                f"[linked:{ent.topic.name} coupling={c:.3f} D_eff={ent.topic.D_eff:.2f}]"
            )
            for ex in ent.exemplars[:1]:
                parts.append(ex)
        text = "\n".join(parts)
        return text[:max_chars]

    def mode_usage_report(self) -> dict[str, Any]:
        """How topics superpose onto the fixed K modes (depth without params)."""
        report = {
            "k_modes": self.k_modes,
            "beta": self.beta,
            "n_topics": len(self.entries),
            "parameter_policy": "fixed_K_modes_topics_share_superposition",
            "topics": {},
        }
        for tid, ent in self.entries.items():
            a = self.alpha(tid)
            report["topics"][tid] = {
                "name": ent.topic.name,
                "D_eff": ent.topic.D_eff,
                "alpha": a.tolist(),
                "dominant_mode": int(a.argmax()),
                "entropy": float(-(a * np.log(a + 1e-12)).sum()),
            }
        return report

    def save(self, path: Optional[Path] = None) -> Path:
        path = path or (
            workspace_root()
            / "llm"
            / "data"
            / "memory"
            / "topic_bank.json"
        )
        path.parent.mkdir(parents=True, exist_ok=True)
        doc = {
            "k_modes": self.k_modes,
            "beta": self.beta,
            "entries": {
                tid: {
                    "topic": asdict(ent.topic),
                    "exemplars": ent.exemplars,
                    "meta": ent.meta,
                }
                for tid, ent in self.entries.items()
            },
            "coupling": self.coupling_matrix(),
            "mode_usage": self.mode_usage_report(),
        }
        # tuples -> lists for JSON
        for tid in doc["entries"]:
            t = doc["entries"][tid]["topic"]
            t["tags"] = list(t.get("tags") or [])
        path.write_text(json.dumps(doc, indent=2), encoding="utf-8")
        return path

    @classmethod
    def load(cls, path: Optional[Path] = None) -> "TopicMemoryBank":
        path = path or (
            workspace_root()
            / "llm"
            / "data"
            / "memory"
            / "topic_bank.json"
        )
        doc = json.loads(path.read_text(encoding="utf-8"))
        bank = cls(k_modes=int(doc.get("k_modes", DEFAULT_K_MODES)))
        for tid, ent in doc.get("entries", {}).items():
            t = ent["topic"]
            fold = TopicFold(
                topic_id=t["topic_id"],
                name=t["name"],
                D_eff=float(t["D_eff"]),
                rho=float(t.get("rho", 1.0)),
                recent_hits=float(t.get("recent_hits", 0.0)),
                observed=bool(t.get("observed", True)),
                tags=tuple(t.get("tags") or ()),
            )
            bank.add(fold, ent.get("exemplars") or [], **(ent.get("meta") or {}))
        return bank


def default_fsot_topics() -> list[tuple[TopicFold, list[str]]]:
    """Preregistered lab topics spanning the medium — linked by geometry."""
    return [
        (
            TopicFold("seeds", "Foundational seeds", 25.0, 1.0, 0.0, True, ("ontology",)),
            [
                "FSOT seeds: pi, e, phi, gamma, G (Catalan). No free parameters.",
            ],
        ),
        (
            TopicFold("scalar", "Scalar engine S=K(T1+T2+T3)", 22.0, 1.0, 0.1, True, ("ontology", "math")),
            [
                "S = K * (T1 + T2 + T3). T1 observer base, T2 linear, T3 valve-acoustic-phase.",
            ],
        ),
        (
            TopicFold("observer", "Observation coupling", 18.0, 0.9, 0.3, True, ("consciousness",)),
            [
                "Observation is physical: observed=True couples quirk/consciousness into T1.",
            ],
        ),
        (
            TopicFold("vision", "Visual spectrum / pixels", 14.0, 0.7, 0.4, True, ("vision", "spectrum")),
            [
                "Pixels are information states. Luminance, entropy, edges map to folds D_eff, rho, hits.",
            ],
        ),
        (
            TopicFold("code", "Code reification", 12.0, 0.8, 0.5, True, ("code",)),
            [
                "Code reifies physical form. mean_luminance uses Rec.601 0.299/0.587/0.114.",
            ],
        ),
        (
            TopicFold("memory", "Superposition memory", 10.0, 0.6, 0.6, True, ("architecture",)),
            [
                "Depth without parameters: fixed K modes, topics are superpositions; links save memory.",
            ],
        ),
        (
            TopicFold("archive", "Archive verification", 20.0, 1.0, 0.2, True, ("verification",)),
            [
                "I:\\FSOT-Physical-Archive is authority. fsot_compute is gold. Failed gates are ledger events.",
            ],
        ),
        (
            TopicFold("sota", "Small beats large", 8.0, 0.5, 0.8, True, ("mission",)),
            [
                "Intelligence is coupling quality to the seed field, not parameter count.",
            ],
        ),
    ]


def build_default_bank() -> TopicMemoryBank:
    bank = TopicMemoryBank()
    for fold, exemplars in default_fsot_topics():
        bank.add(fold, exemplars)
    return bank


def effective_depth_metric(bank: TopicMemoryBank) -> dict[str, Any]:
    """
    Quantitative 'depth without params':
    - topics / modes ratio
    - mean alpha entropy (higher = more superposed depth)
    - mean off-diagonal coupling (topic linking / memory sharing)
    """
    usage = bank.mode_usage_report()
    coup = bank.coupling_matrix()
    mat = np.array(coup["matrix"], dtype=np.float64)
    n = mat.shape[0]
    off = []
    for i in range(n):
        for j in range(n):
            if i != j:
                off.append(mat[i, j])
    ents = [v["entropy"] for v in usage["topics"].values()]
    return {
        "k_modes": bank.k_modes,
        "n_topics": n,
        "topics_per_mode": n / max(bank.k_modes, 1),
        "mean_alpha_entropy": float(np.mean(ents)) if ents else 0.0,
        "mean_topic_coupling": float(np.mean(off)) if off else 0.0,
        "memory_model": "O(topics * exemplars + K * mode_size) not O(topics * full_weights)",
        "parameter_growth_for_new_topic": 0,
    }
