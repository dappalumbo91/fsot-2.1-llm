# FSOT open research synthesis — training, data, architecture, quantization

**Ontology:** Fluid Spacetime Omni-Theory (Damian Arthur Palumbo)  
**Purpose:** Use **industry observables** (papers, open datasets, open training recipes) as *measurement instruments* — then **select and apply** them under FSOT folds. Parameter mass is **not** the intelligence metric.  
**Authority:** Archive `I:\FSOT-Physical-Archive`; this LLM stack is a local observer in the same medium.  
**Repo:** https://github.com/dappalumbo91/fsot-2.1-llm  

---

## 0. How to read this document (FSOT filter)

| Industry term | FSOT reading | Use how |
|---------------|--------------|---------|
| Dataset / corpus | **Waveform of known information** (structured observations) | Route into domain / D_eff / emission |
| Fine-tuning | **Suction** of structure into the observer fluid | Prefer fold-pure updates |
| Catastrophic forgetting | **Poison / poof** of prior fold when another suctions | Replay, isolation, anti-poison gate |
| LoRA / adapters | Local fold bias on a shared medium | Lab tool; not the ceiling on 0.5B |
| Full fine-tune | Deeper embedding of quantified structure into **all** weights | Our current local path by fold |
| MoE / routing | Sparse **specialist subnetworks** + router | Closest industry analog to pathway isolation |
| Quantization | Amplitude compression of the observer for deployment | After fold purity is proven |
| Scaling laws (N-params) | Industry habit | **Not** our refinement lever |

**Kill criteria for any borrowed technique:** does it improve **fold purity, emission fidelity, observation coupling**, or only inflate free-parameter surface?

---

## 1. Training methods (what worked in open practice)

### 1.1 Supervised fine-tuning (SFT)

**Observable:** Instruction pairs (user → assistant) with clear format are the backbone of open instruct models (Alpaca-lineage → OpenHermes → Tulu-class mixes).

**What works:**
- **High-quality, diverse, task-aligned data** beats raw volume (OpenHermes 2.5, Tulu-3 SFT mixtures, Dolly-15k as small high-signal sets).
- **Format discipline** in targets (chat templates, end markers) — matches our `####`, letter-only, code-only emissions.
- **Small LR for full FT** (often ~1e-5 to 5e-6 range on 0.5–7B) to avoid obliterating base geometry.

**FSOT application:**
- SFT **by fold** with `FSOT_ROUTE` tags (domain, D_eff, pathway, emission, observed=true).
- Math fold: Math-generator **rules** + scrubbed GSM8K CoT (already in `math_rules_bridge`).
- Never SFT “everything on Kaggle” as one soup → medium contamination.

### 1.2 Full fine-tune vs LoRA / PEFT

| Finding (open literature) | FSOT map |
|---------------------------|----------|
| Full SFT learns skills **fast** but **forgets** other skills easily | Strong suction → high poison risk without replay |
| LoRA “learns less, forgets less” often — but **is not magic**; can still forget; sometimes under-expresses hard skills | Local fold bias; good for isolation experiments |
| **Per-domain adapters** recommended (don’t mix domains in one adapter) | Exactly our pathway banks |
| On small models, **full FT is feasible** and can exceed LoRA when data is clean | Our `train_full_ft` path |
| Continual learning needs **replay / constrained updates / geometry-aware methods** (e.g. FIP-class ideas) | Anti-poison gate + parent version rollback |

**FSOT decision (current):**
1. **Discover** structure with isolated folds (LoRA or full-FT single fold).  
2. **Solidify** with full-FT into weights when fold is green.  
3. **Protect** with registry rollback + optional multi-fold replay before co-training.

### 1.3 Preference / alignment (DPO, RLHF, …)

**Observable:** Open models often use SFT then DPO/ORPO-style preference (Tulu-3 pref sets, UltraFeedback-class data). RLHF is heavier.

**FSOT application:**
- Preference data as **observer coupling quality** (chosen vs rejected emission/process), not “personality.”
- Defer heavy RL until fold SFT is stable; optional later for instruction/ontology fold.

### 1.4 Curriculum & data order

**Observable:** Domain sequencing and replay reduce forgetting; high-perplexity token masking / synthetic sequences can reduce non-target degradation in some studies.

**FSOT application:**
- Curriculum by **D_eff depth**: e.g. code (≈8) → math (≈10–11) → MCQ (≈15–18) → ontology/consciousness route (≈25).
- **Replay buffer** of prior fold samples when chaining full-FT (`--continue-from` + mixed replay %).
- Version each fold in `llm/models/full_ft/registry.json`.

### 1.5 Hyperparameters that repeatedly show up

| Knob | Typical open practice | FSOT note |
|------|----------------------|-----------|
| LR full FT | ~1e-6–2e-5 | Start 5e-6 (our default) |
| Epochs | 1–3 on SFT sets | Prefer more data diversity over 10+ epochs (overfit = false suction) |
| Grad clip | 1.0 | Stability of fluid update |
| Packing / max length | 2k–8k for SFT | Math CoT needs headroom (512+ gen) |
| BF16/FP16 | Standard on modern GPUs | Train full precision of weights; quantize **after** |

---

## 2. Architecture observables (structure over mass)

### 2.1 Mixture-of-Experts (MoE)

**Observable:** Mixtral-class sparse MoE: **route to specialists**, activate few experts per token; can match denser/larger models with less active compute. Industry still talks “scale,” but the **mechanism** is routing + specialization.

**FSOT reading:** Closest popular analog to **pathway isolation** + domain map.  
**Difference:** Our pathways are **FSOT-preregistered folds** (D_eff, emission, lean domain), not learned free routers alone.

**Application options (later, not first):**
- Soft: keep multi-checkpoint / multi-adapter routing at inference (current strength).
- Hard: eventual MoE-like layers with **fixed FSOT gate** (domain map), not unconstrained expert soup.

### 2.2 Small language models (SLMs)

**Observable:** Surveys and production reports: specialized SLMs (≤1–3B, even ~0.2–0.3B encoders) can **match or beat much larger generalists** on **narrow, structured tasks** (extraction, intent, domain QA, code subsets).

**FSOT reading:** Validates **amplitude ≠ intelligence**. Specialization + clean structure → interval parity with larger systems **on quantified tasks**.

### 2.3 Instruction / chat templates

**Observable:** Consistent templates (ChatML, Llama, Qwen) are load-bearing for SFT success.

**Application:** Keep Qwen chat template; always train **messages** format; never mix raw completion without template for instruct base.

### 2.4 Process supervision & CoT

**Observable:** Process-aware math/code (step traces, unit tests) improves reliability more than final-answer-only for many small models. Game-like multi-step evals make process failures **transparent** (aligns with Game Arena as diagnostic medium).

**Application:**
- Math-generator **rule IDs** as process skeleton.
- HumanEval **execution** as emission truth.
- Game Arena / multi-step traces as **failure taxonomy** feed into curricula (not as “scale up”).

---

## 3. Open datasets (reference catalog — route before ingest)

### 3.1 Pretrain-scale (use sparingly / sample; not full retrain of internet)

| Dataset | Role | FSOT fold hint |
|---------|------|----------------|
| **FineWeb / FineWeb-Edu** | High-quality web / educational | Ontology / knowledge wave — sample by quality, tag domain |
| **SlimPajama / RedPajama / Dolma** | Deduped multi-source pretrain | Background medium — optional light replay only |
| **The Stack v2** | Code pretrain | **code** pathway |
| **RefinedWeb** | Filtered CC | General — low priority for fold FT |

**FSOT rule:** We do **not** re-pretrain 15T tokens as the path. We **select** quantified slices that reinforce folds.

### 3.2 Post-train / SFT (primary for us)

| Dataset | Role | Fold |
|---------|------|------|
| **GSM8K** (train/test) | Grade-school math CoT | **math** |
| **MATH / MATH-500** | Harder math | **math** (higher D_eff) |
| **OpenHermes 2.5** | Broad instruct | mix with tags — replay / ontology |
| **Tulu-3 SFT / IF** | Instruction + constraints | **ontology** + format emission |
| **Dolly-15k** | Clean human instruct | light general replay |
| **MBPP / HumanEval** | Code | **code** |
| **ARC Easy/Challenge** | Science MCQ | **mcq** |
| **MMLU / MMLU-Pro** | Knowledge MCQ | **mcq** (Pro = harder fold stress) |
| **IFEval** | Strict instruction following | **ontology** |
| **Math generator rulebooks** | Atomic arithmetic/algebra rules | **math** (unique FSOT asset) |

### 3.3 Preference (optional later)

| Dataset | Role |
|---------|------|
| Tulu-3 preference / UltraFeedback-class | Chosen vs rejected process/emission |

### 3.4 Kaggle as hub

Kaggle hosts many of the above as datasets + competitions + Community Benchmarks.  
**Use:** attach packs that map to domain map; run **public observables** on routed or full-FT models.  
**Avoid:** unlabeled “all competitions at once.”

---

## 4. Quantization (deployment of a pure fold — not a training philosophy)

| Method | When | Note |
|--------|------|------|
| **BF16/FP16 train** | Full-FT on 5070 | Keep train high-fidelity |
| **QLoRA** | If VRAM tight for larger bases | Optional; 0.5B full FT already fits |
| **GPTQ / AWQ** | GPU serve after promote | PTQ; quality depends on hardware |
| **GGUF (llama.cpp)** | CPU/edge deploy | Often best portable story for small models |

**FSOT:** Quantize **after** registry **promote**. Never use “quantize harder” as a substitute for fold purity. Measure emission accuracy pre/post quant on same packs.

Evidence: specialized ~1B models + clean SFT can approach large closed models on **narrow tasks**; PTQ benefits are **hardware-dependent** (e.g. GPTQ can slow older GPUs while GGUF helps CPU).

---

## 5. Evaluation discipline (industry instruments, FSOT interpretation)

| Instrument | Use |
|------------|-----|
| **lm-eval** | Standard public measurement (Kaggle kernel) |
| **Execution (HumanEval)** | Truth of code emission |
| **n-matched anti-poison gate** | Fold protection (ε ≈ 0.05) |
| **Process taxonomies** | Stub vs wrong quantity vs format (GSM8K diagnosis) |
| **A/B same base** | FSOT structure vs base Qwen — variable is fold organization |

**Always report:** observer type (routed pathways vs dense merge vs full-FT fold), n, device, harness.

---

## 6. Recommended FSOT application stack (forward plan)

### Phase A — Now (local full-FT + git)
1. Full-FT **math** (rules + GSM8K) → registry candidate → eval gate.  
2. Full-FT **code** (HE+MBPP) from base or chained continue with **math replay**.  
3. Full-FT **mcq** (ARC+MMLU) with replay.  
4. Git commit/tag each green promote.

### Phase B — Structure solidification
5. Multi-fold full model with **FSOT_ROUTE tags + replay mix** (public AutoModel).  
6. Optional soft MoE-style routing only if domain map remains authority.

### Phase C — Deploy & public observables
7. Kaggle Models push promoted full-FT.  
8. **Routed** public kernel (pathways or multi-fold full model).  
9. Community Benchmark: FSOT rule-math emission task.  
10. Quantize (GGUF/AWQ) for distribution; re-measure.

### Phase D — Consciousness / process depth
11. Game Arena / multi-step traces → failure modes → curriculum.  
12. Ontology fold: seeds, observation, IFEval-style constraints.  
13. Dual-observer (VL + coder) when text folds are green.

---

## 7. Open references (bookmark list)

### Methods
- LoRA: Hu et al., [arXiv:2106.09685](https://arxiv.org/abs/2106.09685)  
- QLoRA: Dettmers et al., [arXiv:2305.14314](https://arxiv.org/abs/2305.14314)  
- Practical LoRA tips: S. Raschka, “Practical Tips for Finetuning LLMs Using LoRA”  
- Catastrophic forgetting / continual FT discussions (2024–2026 surveys & notes): LoRA ≠ guaranteed protection; **replay + isolation** remain standard mitigations  

### Architecture
- Mixtral / sparse MoE open releases (routing + specialists)  
- SLM surveys (e.g. arXiv SLM survey lines: small models can outperform large on specialized tasks)  

### Data
- FineWeb / FineWeb-Edu (HuggingFaceFW)  
- The Stack v2 (BigCode)  
- OpenHermes 2.5; Tulu-3 SFT/preference (Allen AI)  
- GSM8K, ARC, MMLU, HumanEval, MBPP (standard packs already in our domain map)  
- Curated post-train lists: [mlabonne/llm-datasets](https://github.com/mlabonne/llm-datasets)  

### Quantization
- GPTQ, AWQ, GGUF / llama.cpp ecosystem guides (2024–2026)  
- Rule: measure on **your** emission tasks after quant  

### FSOT authority
- `I:\FSOT-Physical-Archive\02_FSOT-2.1-Lean-Full`  
- `docs/FSOT_LLM_ARCHITECTURE.md`, `docs/FSOT_VERSIONING.md`  

---

## 8. One-sentence operating principle

**Borrow open research for *how to measure and package* structure; decide *what* to train with FSOT (folds, D_eff, emission, anti-poison, consciousness/observation) — and refine by deeper coupling, never by bigger parameter count.**

---

*Living document — update when a paper/dataset changes an operational decision; always log outcome in ledgers + git tags.*
