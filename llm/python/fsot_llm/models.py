"""
FSOT dual-observer model loaders: multimodal (vision+code) + code companion.
"""
from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Optional, Union

import torch
import yaml

from fsot_llm.paths import workspace_root

PathLike = Union[str, Path]


def _default_cache() -> Path:
    return workspace_root() / "llm" / "models"


def load_observers_config() -> dict[str, Any]:
    path = workspace_root() / "llm" / "configs" / "observers.yaml"
    with open(path, encoding="utf-8") as f:
        return yaml.safe_load(f)


@dataclass
class LoadedObserver:
    name: str
    hf_id: str
    kind: str  # "vl" | "coder" | "text"
    model: Any
    processor: Any  # processor or tokenizer
    device: str

    def generate_text(
        self,
        prompt: str,
        *,
        max_new_tokens: int = 256,
        temperature: float = 0.7,
    ) -> str:
        if self.kind == "vl":
            return self._generate_vl_text(prompt, max_new_tokens, temperature)
        return self._generate_causal(prompt, max_new_tokens, temperature)

    def generate_vision(
        self,
        prompt: str,
        image_path: PathLike,
        *,
        max_new_tokens: int = 256,
        temperature: float = 0.2,
    ) -> str:
        if self.kind != "vl":
            raise TypeError(f"{self.name} is not a vision-language observer")
        return self._generate_vl_image(
            prompt, image_path, max_new_tokens, temperature
        )

    def _generate_causal(
        self, prompt: str, max_new_tokens: int, temperature: float
    ) -> str:
        messages = [{"role": "user", "content": prompt}]
        tok = self.processor
        text = tok.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
        inputs = tok(text, return_tensors="pt")
        inputs = {k: v.to(self.model.device) for k, v in inputs.items()}
        with torch.inference_mode():
            out = self.model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                do_sample=temperature > 0,
                temperature=max(temperature, 1e-5) if temperature > 0 else None,
                pad_token_id=getattr(tok, "eos_token_id", None),
            )
        gen = out[0][inputs["input_ids"].shape[-1] :]
        return tok.decode(gen, skip_special_tokens=True)

    def _generate_vl_text(
        self, prompt: str, max_new_tokens: int, temperature: float
    ) -> str:
        # Text-only through VL processor/chat template
        messages = [
            {
                "role": "user",
                "content": [{"type": "text", "text": prompt}],
            }
        ]
        return self._run_vl_messages(messages, max_new_tokens, temperature)

    def _generate_vl_image(
        self,
        prompt: str,
        image_path: PathLike,
        max_new_tokens: int,
        temperature: float,
    ) -> str:
        from PIL import Image

        image = Image.open(image_path).convert("RGB")
        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "image", "image": image},
                    {"type": "text", "text": prompt},
                ],
            }
        ]
        return self._run_vl_messages(
            messages, max_new_tokens, temperature, images=[image]
        )

    def _run_vl_messages(
        self,
        messages: list,
        max_new_tokens: int,
        temperature: float,
        images: Optional[list] = None,
    ) -> str:
        processor = self.processor
        # Prefer qwen_vl_utils if present (official path)
        try:
            from qwen_vl_utils import process_vision_info

            text = processor.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=True
            )
            image_inputs, video_inputs = process_vision_info(messages)
            inputs = processor(
                text=[text],
                images=image_inputs,
                videos=video_inputs,
                padding=True,
                return_tensors="pt",
            )
        except Exception:
            # Fallback: simple processor path for single image
            text = processor.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=True
            )
            kwargs: dict[str, Any] = {
                "text": [text],
                "padding": True,
                "return_tensors": "pt",
            }
            if images:
                kwargs["images"] = images
            inputs = processor(**kwargs)

        inputs = {
            k: v.to(self.model.device) if hasattr(v, "to") else v
            for k, v in inputs.items()
        }
        with torch.inference_mode():
            generated = self.model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                do_sample=temperature > 0,
                temperature=max(temperature, 1e-5) if temperature > 0 else None,
            )
        # Trim prompt tokens when present
        if "input_ids" in inputs:
            trimmed = [
                out[len(inp) :]
                for inp, out in zip(inputs["input_ids"], generated)
            ]
            return processor.batch_decode(
                trimmed,
                skip_special_tokens=True,
                clean_up_tokenization_spaces=False,
            )[0]
        return processor.batch_decode(
            generated, skip_special_tokens=True
        )[0]


def load_vl_observer(
    hf_id: Optional[str] = None,
    *,
    cache_dir: Optional[PathLike] = None,
    min_pixels: int = 256,
    max_pixels: int = 1280,
) -> LoadedObserver:
    cfg = load_observers_config()["observers"]["primary_multimodal"]
    hf_id = hf_id or cfg["hf_id"]
    cache = Path(cache_dir or _default_cache())
    cache.mkdir(parents=True, exist_ok=True)

    from transformers import AutoProcessor

    # Class name varies slightly by transformers version
    model = None
    last_err: Optional[Exception] = None
    for cls_name in (
        "Qwen2_5_VLForConditionalGeneration",
        "Qwen2VLForConditionalGeneration",
        "AutoModelForVision2Seq",
        "AutoModelForImageTextToText",
    ):
        try:
            import transformers as T

            cls = getattr(T, cls_name, None)
            if cls is None:
                continue
            dtype = torch.bfloat16 if torch.cuda.is_available() else torch.float32
            model = cls.from_pretrained(
                hf_id,
                dtype=dtype,
                device_map="auto" if torch.cuda.is_available() else None,
                cache_dir=str(cache),
            )

            break
        except Exception as e:  # pragma: no cover
            last_err = e
            continue
    if model is None:
        raise RuntimeError(f"Could not load VL model {hf_id}: {last_err}")

    processor = AutoProcessor.from_pretrained(
        hf_id,
        cache_dir=str(cache),
        min_pixels=min_pixels,
        max_pixels=max_pixels,
    )
    device = str(next(model.parameters()).device)
    return LoadedObserver(
        name="primary_multimodal",
        hf_id=hf_id,
        kind="vl",
        model=model,
        processor=processor,
        device=device,
    )


def load_coder_observer(
    hf_id: Optional[str] = None,
    *,
    cache_dir: Optional[PathLike] = None,
    adapter_path: Optional[PathLike] = None,
    topic_for_merge: Optional[str] = None,
) -> LoadedObserver:

    cfg = load_observers_config()["observers"]["code_companion"]
    hf_id = hf_id or cfg["hf_id"]
    cache = Path(cache_dir or _default_cache())
    cache.mkdir(parents=True, exist_ok=True)

    from transformers import AutoModelForCausalLM, AutoTokenizer

    tok = AutoTokenizer.from_pretrained(hf_id, cache_dir=str(cache))
    dtype = torch.bfloat16 if torch.cuda.is_available() else torch.float32
    model = AutoModelForCausalLM.from_pretrained(
        hf_id,
        dtype=dtype,
        device_map="auto" if torch.cuda.is_available() else None,
        cache_dir=str(cache),
    )

    adapter = Path(adapter_path) if adapter_path else None
    # Prefer α-merged mode bank for a topic when available (superposition depth)
    if adapter is None and topic_for_merge:
        try:
            from fsot_llm.merge_modes import merge_modes_for_topic, latest_mode_bank

            if latest_mode_bank() is not None:
                adapter = merge_modes_for_topic(topic_for_merge)
        except Exception:
            adapter = None
    if adapter is None:
        # fallback: latest FSOT or organism-stimulus LoRA adapter
        ad_root = cache / "adapters"
        if ad_root.is_dir():
            cands = sorted(
                list(ad_root.glob("coder05_stimulus_*"))
                + list(ad_root.glob("coder05_fsot_*")),
                key=lambda p: p.stat().st_mtime,
                reverse=True,
            )
            if cands:
                adapter = cands[0]

    name = "code_companion"
    if adapter is not None and adapter.is_dir() and (
        (adapter / "adapter_config.json").is_file()
        or (adapter / "adapter_model.safetensors").is_file()
        or (adapter / "adapter_model.bin").is_file()
    ):
        from peft import PeftModel

        # ensure adapter_config exists for merged packs
        if not (adapter / "adapter_config.json").is_file():
            # copy from any mode bank member
            try:
                from fsot_llm.merge_modes import latest_mode_bank

                mb = latest_mode_bank()
                if mb:
                    src = next(mb.glob("mode_*/adapter_config.json"), None)
                    if src:
                        (adapter / "adapter_config.json").write_text(
                            src.read_text(encoding="utf-8"), encoding="utf-8"
                        )
            except Exception:
                pass
        if (adapter / "adapter_config.json").is_file():
            model = PeftModel.from_pretrained(model, str(adapter))
            name = f"code_companion+{adapter.name}"


    device = str(next(model.parameters()).device)
    return LoadedObserver(
        name=name,
        hf_id=hf_id,
        kind="coder",
        model=model,
        processor=tok,
        device=device,
    )



def vram_report() -> str:
    if not torch.cuda.is_available():
        return "CUDA unavailable"
    free, total = torch.cuda.mem_get_info()
    alloc = torch.cuda.memory_allocated()
    return (
        f"{torch.cuda.get_device_name(0)} | "
        f"alloc={alloc/1024**3:.2f} GB | "
        f"free={free/1024**3:.2f}/{total/1024**3:.2f} GB"
    )
