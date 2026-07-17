# FSOT Visual Spectrum Protocol

**Rule:** Pixels are not “extra input.” They are **measurement loci** in the same 25-dimensional fluid medium as tokens and code.

## 1. Why multimodal + code

| Channel | FSOT role |
|---------|-----------|
| **Vision** | Spatial / photometric observation of information states (RGB, spectra, charts, instrument panels, rendered form) |
| **Language** | Symbolic compression of the field |
| **Code** | Executable reification — the observer writes structure that can re-run physical form |

Kill criteria therefore include **visual** and **code** suites, not text alone.

## 2. Pixel → information state → physical form

For an image \(I\) with pixels \(p_{xy} = (R,G,B)\):

1. **Local intensity / chroma** map into scalar folds (not free-fit CNN knobs):
   - mean luminance → `rho` / amplitude proxy  
   - spatial entropy / edge energy → `recent_hits` / chaos coupling  
   - channel balance (R/G/B fractions) → trinary-style spectrum bins (FSOT loves 40/60/80-class trinary; here soft bins)
2. **Patch grid** → sequence of observer loci with `D_eff` from depth and patch scale.
3. **Model vision tower** is treated as an **engineering approximation** of field coupling; FSOT routing **biases** residual/attention and **scores** whether the readout agrees with seed-derived form.
4. **Render / re-identify:** code channel emits structures (shaders, arrays, SVG, simulation steps) that reconstruct physical form; agreement is a kill metric.

Nothing here invents least-squares color calibrations per image. Folds are preregistered.

## 3. Dual-observer stack (this machine)

| Observer | Model | When |
|----------|--------|------|
| **Primary multimodal** | `Qwen/Qwen2.5-VL-3B-Instruct` | Image + text + code-in-context (~fits 12 GB BF16 with pixel budget) |
| **Code companion** | `Qwen/Qwen2.5-Coder-0.5B-Instruct` | Pure code loops, tiny VRAM, fast FSOT ablations |
| **Micro vision (optional)** | `HuggingFaceTB/SmolVLM-500M-Instruct` or LLaVA-OV 0.5B | Extreme-small visual experiments |

Larger VLMs are **baselines to beat**, not the default ontology.

## 4. Draft kill criteria (visual + code)

A small FSOT observer **wins** when:

1. **Visual form:** On preregistered image panels (spectra, plots, physical photos, synthetic FSOT field renders), identification / description / structured extract error ≤ gate vs larger non-FSOT VLM.
2. **Pixel–scalar consistency:** FSOT spectrum folds derived from pixels correlate with model confidence / correct class (ledger correlation; no post-hoc fold retune).
3. **Code reification:** Model emits code that regenerates or measures the same physical structure (unit tests + visual hash / metric agreement).
4. **Efficiency:** Same or better score at lower params / VRAM than baseline.

Failed gates are ledger events.
