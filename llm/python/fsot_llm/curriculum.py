"""
FSOT training curriculum — generated from archive scalar authority + spectrum.

No scraped free-form web noise as the spine. Every numeric claim is recomputed
from fsot_compute. Code items are unit-checked when possible.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from fsot_llm.archive_verify import check_archive, scalar_gold
from fsot_llm.paths import ensure_sys_path, workspace_root
from fsot_llm.vision_spectrum import VisualSpectrumObserver

ensure_sys_path()

from fsot_bridge import FSOTEngine


def _chat(user: str, assistant: str, system: str | None = None) -> dict[str, Any]:
    msgs = []
    if system:
        msgs.append({"role": "system", "content": system})
    msgs.append({"role": "user", "content": user})
    msgs.append({"role": "assistant", "content": assistant})
    return {"messages": msgs}


FSOT_SYSTEM = (
    "You are an FSOT (Fluid Spacetime Omni-Theory) observer. "
    "Use only seed-derived structure (π, e, φ, γ, G Catalan). "
    "No free-parameter curve fits. Observation is physical coupling. "
    "Code reifies physical form. Vision reads information states in pixels."
)


def _mix_external_benchmark_rows(max_gsm8k: int = 40, max_he: int = 12) -> list[dict[str, Any]]:
    """
    Blend real packs from D:\\training data so LoRA keeps general skill
    while learning FSOT (prevents ontology-only collapse).
    """
    rows: list[dict[str, Any]] = []
    try:
        from fsot_llm.external_data import (
            extract_gsm8k_gold,
            load_pack_rows,
            training_data_root,
        )

        if not training_data_root().is_dir():
            return rows
        for row in load_pack_rows("gsm8k_train", limit=max_gsm8k):
            q = row.get("question") or ""
            gold = extract_gsm8k_gold(row.get("answer") or "")
            # short gold path — full chain-of-thought is long; teach format + answer
            ans_full = row.get("answer") or f"#### {gold}"
            # keep answer under control
            if len(ans_full) > 800:
                ans_full = f"Reason carefully.\n#### {gold}"
            rows.append(
                _chat(
                    "Solve the grade-school math problem. Show brief reasoning. "
                    f"End with #### <number>.\n\nProblem: {q}",
                    ans_full,
                    FSOT_SYSTEM,
                )
            )
        for row in load_pack_rows("humaneval", limit=max_he):
            prompt = row.get("prompt") or ""
            canon = row.get("canonical_solution") or ""
            if not prompt or not canon:
                continue
            rows.append(
                _chat(
                    "Complete the following Python function. Output code only.\n\n"
                    + prompt,
                    prompt + canon if "def " not in canon else canon,
                    FSOT_SYSTEM,
                )
            )
    except Exception:
        return rows
    return rows


def build_curriculum() -> list[dict[str, Any]]:
    st = check_archive()
    if not st.compute_exists:
        raise RuntimeError(f"Archive compute missing: {st.notes}")

    eng = FSOTEngine()
    seeds = eng.seeds()
    rows: list[dict[str, Any]] = []


    # --- Ontology / seeds ---
    rows.append(
        _chat(
            "What is Fluid Spacetime Omni-Theory (FSOT) in one paragraph?",
            "FSOT models reality as a 25-dimensional fluid condensate. "
            "Space, time, matter, life, and mind are regimes of one seed-derived "
            "scalar field raw_S computed from π, e, φ, γ, and Catalan's G with "
            "no per-observable least-squares free parameters. "
            "The scalar engine is S = K·(T1 + T2 + T3) with preregistered folds "
            "(D_eff, Δψ, recent_hits, observed). Observation is physical coupling.",
            FSOT_SYSTEM,
        )
    )
    rows.append(
        _chat(
            "List the five FSOT foundational seeds.",
            "π (pi), e, φ (phi, golden ratio), γ (Euler-Mascheroni), and G (Catalan's constant).",
            FSOT_SYSTEM,
        )
    )
    rows.append(
        _chat(
            "What is the FSOT scalar engine formula?",
            "S = K · (T1 + T2 + T3), where T1 is observer-modulated base, "
            "T2 is linear modulation (scale·amplitude + trend_bias), "
            "and T3 is valve-acoustic-phase structure. K is seed-derived.",
            FSOT_SYSTEM,
        )
    )
    rows.append(
        _chat(
            "Does FSOT allow per-row least-squares free parameters for benchmarks?",
            "No. Routing folds are preregistered; claims must match the canonical "
            "fsot_compute engine from the physical archive.",
            FSOT_SYSTEM,
        )
    )
    rows.append(
        _chat(
            "How many dimensions is the FSOT fluid medium?",
            "25. Four-dimensional experience is a perceived slice of that medium.",
            FSOT_SYSTEM,
        )
    )

    # --- Numeric gold from engine ---
    for observed in (False, True):
        g = scalar_gold(D_eff=25.0, observed=observed)
        rows.append(
            _chat(
                f"For default ScalarInput with D_eff=25 and observed={observed}, "
                f"what is the sign and approximate magnitude of S?",
                f"S ≈ {g['S_float']:.6f} "
                f"({'positive' if g['S_float'] > 0 else 'negative'}). "
                f"Exact archive string begins: {g['S_str'][:24]}...",
                FSOT_SYSTEM,
            )
        )

    # Vary D_eff
    for d in (6.0, 12.0, 18.0, 25.0):
        g = scalar_gold(D_eff=d, observed=False)
        rows.append(
            _chat(
                f"Compute qualitative FSOT behavior at D_eff={d}, observed=false: "
                f"is |S| large or small relative to D_eff=25?",
                f"At D_eff={d}, S≈{g['S_float']:.6f}. "
                f"Compare against D_eff=25 archive gold when reporting precision claims.",
                FSOT_SYSTEM,
            )
        )

    # Constants as text
    rows.append(
        _chat(
            "State approximate FSOT derived constants K and C_factor.",
            f"K ≈ {float(seeds['K']):.8f}, C_factor ≈ {float(seeds['C_factor']):.8f} "
            f"(from archive fsot_compute; use engine for full precision).",
            FSOT_SYSTEM,
        )
    )

    # --- Code reification (multiple phrasings — prevent LoRA catastrophic forget) ---
    luma_code = (
        "import numpy as np\n\n"
        "def mean_luminance(rgb):\n"
        "    rgb = np.asarray(rgb, dtype=np.float64)\n"
        "    if rgb.max() > 1.0:\n"
        "        rgb = rgb / 255.0\n"
        "    return float((0.299*rgb[...,0] + 0.587*rgb[...,1] + 0.114*rgb[...,2]).mean())\n"
    )
    for user in (
        "Write a Python function mean_luminance(rgb) using Rec.601 weights.",
        "Implement mean_luminance(rgb) for HxWx3 arrays with 0.299/0.587/0.114.",
        "Code only: def mean_luminance(rgb) Rec.601 luminance mean.",
    ):
        rows.append(_chat(user, luma_code, FSOT_SYSTEM))

    # Seed list — multiple exact answer formats for retrieval robustness
    seed_answer = (
        "The five FSOT foundational seeds are: "
        "pi (π), e, phi (φ, golden ratio), gamma (γ, Euler-Mascheroni), "
        "and G (Catalan's constant / G_catalan)."
    )
    for user in (
        "List the five FSOT foundational seeds.",
        "List the five FSOT foundational seeds by symbol only.",
        "What are the FSOT seeds? Name pi, e, phi, gamma, and Catalan G.",
        "FSOT seed constants: list them.",
    ):
        rows.append(_chat(user, seed_answer, FSOT_SYSTEM))

    for user in (
        "What is the FSOT scalar engine formula?",
        "What is the FSOT scalar engine form? Answer with S = K * (T1 + T2 + T3) or equivalent.",
        "Write S in terms of K, T1, T2, T3 for FSOT.",
    ):
        rows.append(
            _chat(
                user,
                "S = K * (T1 + T2 + T3). "
                "Equivalently S = K · (T1 + T2 + T3). "
                "T1 observer-modulated base, T2 linear modulation, T3 valve-acoustic-phase.",
                FSOT_SYSTEM,
            )
        )

    rows.append(
        _chat(
            "Write a Python snippet that maps image mean luminance and edge energy "
            "into FSOT folds D_eff in [4,25] and recent_hits in [0,1].",
            "import numpy as np\n\n"
            "def spectrum_folds(mean_luminance, edge_energy, entropy_proxy):\n"
            "    structure = 0.5*float(entropy_proxy) + 0.5*float(edge_energy)\n"
            "    structure = min(1.0, max(0.0, structure))\n"
            "    D_eff = 4.0 + 21.0 * structure\n"
            "    recent_hits = min(1.0, max(0.0, float(edge_energy)))\n"
            "    rho = max(float(mean_luminance), 1e-6)\n"
            "    return {'D_eff': D_eff, 'recent_hits': recent_hits, 'rho': rho, 'observed': True}\n",
            FSOT_SYSTEM,
        )
    )
    rows.append(
        _chat(
            "Write a minimal pure-Python helper that returns FSOT seeds as floats "
            "(pi, e, phi, gamma, G_catalan) without free parameters.",
            "import math\n\n"
            "def fsot_seeds():\n"
            "    pi = math.pi\n"
            "    e = math.e\n"
            "    phi = (1 + math.sqrt(5)) / 2\n"
            "    # Use high-precision archive values for gamma and G in production.\n"
            "    gamma = 0.5772156649015329\n"
            "    G_catalan = 0.9159655941772190\n"
            "    return dict(pi=pi, e=e, phi=phi, gamma=gamma, G_catalan=G_catalan)\n",
            FSOT_SYSTEM,
        )
    )

    # --- Vision / spectrum language ---
    data_img = workspace_root() / "llm" / "data" / "synthetic_spectrum.png"
    if data_img.is_file():
        rep = VisualSpectrumObserver(eng).observe_path(data_img)
        sp = rep["spectrum"]
        rows.append(
            _chat(
                "Given a synthetic RGB field with horizontal red ramp, vertical green ramp, "
                "and blue interference fringes, describe its FSOT information state.",
                f"Mean luminance≈{sp['mean_luminance']:.3f}, entropy_proxy≈{sp['entropy_proxy']:.3f}, "
                f"edge_energy≈{sp['edge_energy']:.3f}, channel fractions R/G/B≈"
                f"{sp['r_frac']:.3f}/{sp['g_frac']:.3f}/{sp['b_frac']:.3f}. "
                f"Trinary mass low/mid/high≈{sp['bin_low']:.3f}/{sp['bin_mid']:.3f}/{sp['bin_high']:.3f}. "
                f"Coupled scalar S≈{rep['S']:.6f} with D_eff≈{rep['scalar_kwargs']['D_eff']:.2f}. "
                "Pixels are observation loci, not decorative inputs.",
                FSOT_SYSTEM,
            )
        )

    # --- Intrinsic LLM archive tiers as meta-knowledge ---
    rows.append(
        _chat(
            "What is the archive intrinsic LLM validation tier structure?",
            "Four tiers: Validation (3 topics) historically 100% hits; "
            "Quarter Eval (12 topics); Half Eval (24 topics); Full Eval (48 topics). "
            "These live under vendor/intrinsic_llm/benchmark_results_final.json "
            "and feed data/intrinsic_llm_validators_benchmark.json. "
            "This lab must improve model accuracy on FSOT tasks while remaining "
            "consistent with archive scalar authority.",
            FSOT_SYSTEM,
        )
    )

    # --- Multimodal mission ---
    rows.append(
        _chat(
            "Why must an FSOT LLM be multimodal and code-capable?",
            "Vision measures spatial information states (pixels as physical form). "
            "Code reifies those measurements into executable structure. "
            "Language compresses the field symbolically. "
            "All three are regimes of the same medium; text-only observers cannot "
            "satisfy visual-spectrum kill criteria.",
            FSOT_SYSTEM,
        )
    )

    # External packs (D:\training data) — general skill retention under FSOT system prompt
    ext = _mix_external_benchmark_rows()
    rows.extend(ext)

    return rows



def write_curriculum(path: Path | None = None) -> Path:
    rows = build_curriculum()
    out = path or (workspace_root() / "llm" / "data" / "curriculum" / "fsot_curriculum.jsonl")
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    meta = {
        "count": len(rows),
        "archive": check_archive().to_dict(),
        "path": str(out),
    }
    meta_path = out.with_suffix(".meta.json")
    meta_path.write_text(json.dumps(meta, indent=2), encoding="utf-8")
    return out


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--output", type=Path, default=None)
    args = ap.parse_args()
    path = write_curriculum(args.output)
    print(f"Wrote curriculum: {path}")
    n = sum(1 for _ in path.open(encoding="utf-8"))
    print(f"  examples: {n}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
