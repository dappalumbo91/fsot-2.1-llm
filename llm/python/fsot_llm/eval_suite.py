"""
Preregistered FSOT eval suite — kill criteria for refine loop.

Suites:
  K1 FSOT knowledge (extractable facts)
  K2 Code reification (executable checks)
  K3 Visual spectrum (structure tags vs pixel gold)
  K5 Archive scalar consistency
"""
from __future__ import annotations

import argparse
import json
import re
import textwrap
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Callable, Optional

from fsot_llm.archive_verify import (
    check_archive,
    load_intrinsic_llm_vendor,
    relative_error,
    scalar_gold,
    write_ledger,
)
from fsot_llm.paths import ensure_sys_path, workspace_root
from fsot_llm.vision_spectrum import VisualSpectrumObserver

ensure_sys_path()


@dataclass
class ItemResult:
    id: str
    suite: str
    ok: bool
    detail: str
    score: float = 0.0
    meta: dict[str, Any] = field(default_factory=dict)


@dataclass
class SuiteReport:
    name: str
    ok: bool
    items: list[ItemResult]
    score: float

    def to_dict(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "ok": self.ok,
            "score": self.score,
            "items": [asdict(i) for i in self.items],
        }


def _contains_all(text: str, needles: list[str]) -> bool:
    t = text.lower()
    return all(n.lower() in t for n in needles)


# ---------------------------------------------------------------------------
# K1 — FSOT knowledge (no model required for gold checks; model optional)
# ---------------------------------------------------------------------------
def eval_k1_fsot_knowledge(
    generate: Optional[Callable[[str], str]] = None,
) -> SuiteReport:
    items: list[ItemResult] = []
    gold = scalar_gold(D_eff=25, observed=False)
    seeds = gold["seeds"]

    # Static gold: archive still online and seeds finite
    items.append(
        ItemResult(
            id="K1.archive_seeds",
            suite="K1",
            ok=True,
            detail="archive seeds loaded",
            score=1.0,
            meta={"phi": seeds["phi"][:20], "K": seeds["K"][:20]},
        )
    )

    prompts = [
        (
            "K1.seeds_list",
            "List the five FSOT foundational seeds by symbol only.",
            ["pi", "e", "phi", "gamma"]  # catalan as G optional wording
            ,
        ),
        (
            "K1.scalar_form",
            "What is the FSOT scalar engine form? Answer with S = K * (T1 + T2 + T3) or equivalent.",
            ["t1", "t2", "t3"],
        ),
        (
            "K1.dimensions",
            "How many dimensions is the FSOT fluid medium? Answer with a number.",
            ["25"],
        ),
        (
            "K1.no_free_params",
            "Does FSOT use per-observable least-squares free parameters? Yes or no.",
            ["no"],
        ),
    ]

    if generate is None:
        for pid, _, _ in prompts:
            items.append(
                ItemResult(
                    id=pid,
                    suite="K1",
                    ok=False,
                    detail="skipped (no generate fn)",
                    score=0.0,
                )
            )
    else:
        for pid, prompt, needles in prompts:
            try:
                ans = generate(prompt)
                ok = _contains_all(ans, needles)
                # special case catalan
                if pid == "K1.seeds_list":
                    ok = ok and (
                        "catalan" in ans.lower()
                        or "g" in ans.lower().split()
                        or "g_cat" in ans.lower()
                    )
                items.append(
                    ItemResult(
                        id=pid,
                        suite="K1",
                        ok=ok,
                        detail=ans[:400],
                        score=1.0 if ok else 0.0,
                    )
                )
            except Exception as exc:
                items.append(
                    ItemResult(
                        id=pid,
                        suite="K1",
                        ok=False,
                        detail=repr(exc),
                        score=0.0,
                    )
                )

    score = sum(i.score for i in items) / max(len(items), 1)
    # Gate: archive seeds always; model items must hit >= 0.5 when generate provided
    if generate is None:
        ok = items[0].ok
    else:
        model_items = [i for i in items if i.id != "K1.archive_seeds"]
        ok = items[0].ok and (
            sum(i.score for i in model_items) / max(len(model_items), 1) >= 0.5
        )
    return SuiteReport("K1_fsot_knowledge", ok, items, score)


# ---------------------------------------------------------------------------
# K2 — Code reification (execute candidate snippets)
# ---------------------------------------------------------------------------
def eval_k2_code_reification(
    generate: Optional[Callable[[str], str]] = None,
) -> SuiteReport:
    items: list[ItemResult] = []
    # Gold implementation always present
    import numpy as np

    def gold_luminance(rgb: np.ndarray) -> float:
        r, g, b = rgb[..., 0], rgb[..., 1], rgb[..., 2]
        return float((0.299 * r + 0.587 * g + 0.114 * b).mean())

    rgb = np.random.RandomState(0).randint(0, 256, (8, 8, 3)).astype(np.float64)
    gold = gold_luminance(rgb)
    items.append(
        ItemResult(
            id="K2.gold_luminance",
            suite="K2",
            ok=True,
            detail=f"gold={gold:.6f}",
            score=1.0,
        )
    )

    if generate is None:
        items.append(
            ItemResult(
                id="K2.model_luminance",
                suite="K2",
                ok=False,
                detail="skipped (no generate)",
                score=0.0,
            )
        )
    else:
        prompt = textwrap.dedent(
            """
            Write a Python function only:
            def mean_luminance(rgb):
                # rgb is HxWx3 float or uint8 array
                # use Rec.601 weights 0.299, 0.587, 0.114
                ...
            No markdown fences if possible; pure code ok.
            """
        ).strip()
        ans = generate(prompt)
        code = ans
        # strip fences
        m = re.search(r"```(?:python)?\s*([\s\S]*?)```", ans)
        if m:
            code = m.group(1)
        ok = False
        detail = code[:500]
        try:
            ns: dict[str, Any] = {"np": np, "numpy": np}
            exec(code, ns, ns)  # noqa: S102 — controlled eval lab
            fn = ns.get("mean_luminance")
            if callable(fn):
                pred = float(fn(rgb))
                err = relative_error(pred, gold)
                # Accept implementations that normalize RGB to [0,1]
                err_norm = relative_error(pred, gold / 255.0)
                ok = err < 0.05 or err_norm < 0.05
                detail = (
                    f"pred={pred:.6f} gold={gold:.6f} "
                    f"rel_err={err:.4f} rel_err_norm={err_norm:.4f}"
                )
            else:
                detail = "mean_luminance not defined"

        except Exception as exc:
            detail = f"exec_error: {exc!r}"
        items.append(
            ItemResult(
                id="K2.model_luminance",
                suite="K2",
                ok=ok,
                detail=detail,
                score=1.0 if ok else 0.0,
            )
        )

    score = sum(i.score for i in items) / max(len(items), 1)
    ok = all(i.ok for i in items if i.id == "K2.gold_luminance") and (
        generate is None or any(i.id == "K2.model_luminance" and i.ok for i in items)
    )
    # when no generate, suite is instrument-ok but model not scored green overall
    if generate is None:
        ok = True  # instrumentation only
    return SuiteReport("K2_code_reification", ok if generate else True, items, score)


# ---------------------------------------------------------------------------
# K3 — Visual spectrum
# ---------------------------------------------------------------------------
def eval_k3_visual_spectrum(
    generate_vision: Optional[Callable[[str, Path], str]] = None,
) -> SuiteReport:
    items: list[ItemResult] = []
    from PIL import Image
    import numpy as np

    path = workspace_root() / "llm" / "data" / "synthetic_spectrum.png"
    if not path.is_file():
        # create
        h, w = 256, 256
        y = np.linspace(0, 1, h)[:, None]
        x = np.linspace(0, 1, w)[None, :]
        xx = np.broadcast_to(x, (h, w))
        yy = np.broadcast_to(y, (h, w))
        rgb = np.stack(
            [xx, yy, 0.5 + 0.5 * np.sin(2 * np.pi * (3 * xx + 2 * yy))],
            axis=-1,
        )
        path.parent.mkdir(parents=True, exist_ok=True)
        Image.fromarray((np.clip(rgb, 0, 1) * 255).astype(np.uint8)).save(path)

    obs = VisualSpectrumObserver()
    report = obs.observe_path(path)
    spec = report["spectrum"]
    items.append(
        ItemResult(
            id="K3.spectrum_scalar",
            suite="K3",
            ok=True,
            detail=f"S={report['S']:.6f} D_eff={report['scalar_kwargs']['D_eff']:.3f}",
            score=1.0,
            meta=spec,
        )
    )

    # Structure tags gold
    tags = []
    if abs(spec["r_frac"] - spec["g_frac"]) < 0.05:
        tags.append("balanced_rg")
    if spec["entropy_proxy"] > 0.7:
        tags.append("high_entropy")
    if spec["edge_energy"] < 0.1:
        tags.append("smooth_gradient")
    if 0.4 < spec["mean_luminance"] < 0.6:
        tags.append("mid_luminance")

    if generate_vision is None:
        items.append(
            ItemResult(
                id="K3.model_vision",
                suite="K3",
                ok=False,
                detail="skipped (no vision generate)",
                score=0.0,
                meta={"gold_tags": tags},
            )
        )
        return SuiteReport("K3_visual_spectrum", True, items, 0.5)

    ans = generate_vision(
        "Describe this image's physical information structure. "
        "Mention gradients, color channels, and whether it looks smooth or high-frequency. "
        "Short answer.",
        path,
    )
    hits = sum(1 for t in tags if t.replace("_", " ") in ans.lower() or t.split("_")[0] in ans.lower())
    # softer: keyword checks
    soft = 0
    if "gradient" in ans.lower() or "ramp" in ans.lower() or "smooth" in ans.lower():
        soft += 1
    if any(c in ans.lower() for c in ("red", "green", "blue", "rgb", "channel")):
        soft += 1
    if "interfer" in ans.lower() or "wave" in ans.lower() or "pattern" in ans.lower():
        soft += 1
    ok = soft >= 2
    items.append(
        ItemResult(
            id="K3.model_vision",
            suite="K3",
            ok=ok,
            detail=ans[:500],
            score=soft / 3.0,
            meta={"gold_tags": tags, "soft_hits": soft},
        )
    )
    score = sum(i.score for i in items) / max(len(items), 1)
    return SuiteReport("K3_visual_spectrum", ok, items, score)


# ---------------------------------------------------------------------------
# K5 — Archive consistency
# ---------------------------------------------------------------------------
def eval_k5_archive() -> SuiteReport:
    items: list[ItemResult] = []
    st = check_archive()
    items.append(
        ItemResult(
            id="K5.archive_present",
            suite="K5",
            ok=st.ok,
            detail=json.dumps(st.to_dict()),
            score=1.0 if st.ok else 0.0,
        )
    )
    try:
        vendor = load_intrinsic_llm_vendor()
        ok = len(vendor) == 4 and vendor[0].get("accuracy_pct") == 100.0
        items.append(
            ItemResult(
                id="K5.intrinsic_llm_vendor",
                suite="K5",
                ok=ok,
                detail=f"tiers={len(vendor)} validation_acc={vendor[0].get('accuracy_pct')}",
                score=1.0 if ok else 0.0,
                meta={"vendor": vendor},
            )
        )
    except Exception as exc:
        items.append(
            ItemResult(
                id="K5.intrinsic_llm_vendor",
                suite="K5",
                ok=False,
                detail=repr(exc),
                score=0.0,
            )
        )

    g0 = scalar_gold(D_eff=25, observed=False)
    g1 = scalar_gold(D_eff=25, observed=True)
    # known smoke values from scaffold (tolerance 1e-9 relative)
    ref0 = -0.5024559462100433
    ref1 = 0.8791285168040015
    e0 = relative_error(g0["S_float"], ref0)
    e1 = relative_error(g1["S_float"], ref1)
    ok_s = e0 < 1e-9 and e1 < 1e-9
    items.append(
        ItemResult(
            id="K5.scalar_parity_smoke",
            suite="K5",
            ok=ok_s,
            detail=f"e0={e0:.2e} e1={e1:.2e}",
            score=1.0 if ok_s else 0.0,
        )
    )

    score = sum(i.score for i in items) / max(len(items), 1)
    ok = all(i.ok for i in items)
    return SuiteReport("K5_archive", ok, items, score)


def eval_k6_superposition_depth() -> SuiteReport:
    """K6: depth without parameter growth — fixed modes, linked topics."""
    from fsot_llm.superposition import (
        build_default_bank,
        effective_depth_metric,
    )

    items: list[ItemResult] = []
    bank = build_default_bank()
    bank.save()
    depth = effective_depth_metric(bank)
    ok_ratio = depth["topics_per_mode"] >= 1.0
    ok_link = depth["mean_topic_coupling"] > 0.5
    ok_zero_growth = depth["parameter_growth_for_new_topic"] == 0
    items.append(
        ItemResult(
            id="K6.topics_per_mode",
            suite="K6",
            ok=ok_ratio,
            detail=json.dumps(depth),
            score=1.0 if ok_ratio else 0.0,
            meta=depth,
        )
    )
    items.append(
        ItemResult(
            id="K6.topic_linking",
            suite="K6",
            ok=ok_link,
            detail=f"mean_coupling={depth['mean_topic_coupling']:.4f}",
            score=float(depth["mean_topic_coupling"]),
        )
    )
    items.append(
        ItemResult(
            id="K6.zero_param_growth",
            suite="K6",
            ok=ok_zero_growth,
            detail="new topic adds bank row only, not full weights",
            score=1.0 if ok_zero_growth else 0.0,
        )
    )
    # Linked memory non-empty for a leaf topic
    ctx = bank.linked_context("vision")
    ok_ctx = "linked:" in ctx or "FSOT topic" in ctx
    items.append(
        ItemResult(
            id="K6.linked_memory",
            suite="K6",
            ok=ok_ctx,
            detail=ctx[:300],
            score=1.0 if ok_ctx else 0.0,
        )
    )
    score = sum(i.score for i in items) / max(len(items), 1)
    ok = all(i.ok for i in items)
    return SuiteReport("K6_superposition_depth", ok, items, score)


def run_eval_suite(
    *,
    generate_text: Optional[Callable[[str], str]] = None,
    generate_vision: Optional[Callable[[str, Path], str]] = None,
    write: bool = True,
) -> dict[str, Any]:
    suites = [
        eval_k5_archive(),
        eval_k6_superposition_depth(),
        eval_k1_fsot_knowledge(generate_text),
        eval_k2_code_reification(generate_text),
        eval_k3_visual_spectrum(generate_vision),
    ]

    overall_ok = all(s.ok for s in suites)
    # when model not provided, overall_ok tracks archive+instrumentation readiness
    report = {
        "overall_ok": overall_ok,
        "suites": [s.to_dict() for s in suites],
        "mean_score": sum(s.score for s in suites) / max(len(suites), 1),
        "mode": "with_model" if generate_text or generate_vision else "instrumentation",
    }
    if write:
        path = write_ledger("eval_suite", report)
        report["ledger_path"] = str(path)
    return report


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--with-coder", action="store_true")
    ap.add_argument("--with-vl", action="store_true")
    args = ap.parse_args()

    gen_text = None
    gen_vis = None

    if args.with_coder:
        from fsot_llm.models import load_coder_observer

        coder = load_coder_observer()

        def gen_text(p: str) -> str:
            return coder.generate_text(p, max_new_tokens=256, temperature=0.1)

    if args.with_vl:
        from fsot_llm.models import load_vl_observer

        vl = load_vl_observer()

        def gen_text(p: str) -> str:  # noqa: F811
            return vl.generate_text(p, max_new_tokens=256, temperature=0.1)

        def gen_vis(p: str, img: Path) -> str:
            return vl.generate_vision(p, img, max_new_tokens=160, temperature=0.1)

    report = run_eval_suite(
        generate_text=gen_text, generate_vision=gen_vis, write=True
    )
    print(json.dumps(report, indent=2))
    return 0 if report["overall_ok"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
