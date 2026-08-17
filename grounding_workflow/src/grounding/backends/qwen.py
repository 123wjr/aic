"""Qwen3-VL backend；只负责 Transformers 加载和批量生成。"""

from __future__ import annotations

import gc
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from ..types import BackendResponse, GroundingBackend, InferenceUnit


@dataclass(frozen=True)
class QwenConfig:
    model: str
    dtype: str = "auto"
    max_new_tokens: int = 128
    min_pixels: int | None = None
    max_pixels: int | None = None
    attn_implementation: str | None = None
    gpu_memory_limit: str | None = None


class QwenBackend(GroundingBackend):
    """官方 Qwen Transformers 接口适配器。"""

    def __init__(self, config: QwenConfig):
        try:
            import torch
            from transformers import AutoModelForImageTextToText, AutoProcessor
        except ImportError as exc:
            raise RuntimeError("缺少 Qwen 依赖，请安装 requirements.txt") from exc
        dtype: Any = config.dtype
        if config.dtype == "float16":
            dtype = torch.float16
        elif config.dtype == "bfloat16":
            dtype = torch.bfloat16
        kwargs: dict[str, Any] = {"dtype": dtype, "device_map": "auto"}
        if config.attn_implementation:
            kwargs["attn_implementation"] = config.attn_implementation
        if config.gpu_memory_limit:
            kwargs["max_memory"] = {0: config.gpu_memory_limit, "cpu": "64GiB"}
        self.torch = torch
        self.model = AutoModelForImageTextToText.from_pretrained(config.model, **kwargs)
        self.processor = AutoProcessor.from_pretrained(config.model)
        self.config = config

    def infer(self, units: list[InferenceUnit]) -> BackendResponse:
        from PIL import Image

        conversations: list[list[dict[str, Any]]] = []
        images = []
        for unit in units:
            for prompt in unit.prompts:
                with Image.open(unit.image_path) as source:
                    image = source.convert("RGB").copy()
                images.append(image)
                image_content: dict[str, Any] = {"type": "image", "image": image}
                if self.config.min_pixels is not None:
                    image_content["min_pixels"] = self.config.min_pixels
                if self.config.max_pixels is not None:
                    image_content["max_pixels"] = self.config.max_pixels
                conversations.append([{"role": "user", "content": [image_content, {"type": "text", "text": prompt}]}])
        inputs = self.processor.apply_chat_template(conversations, tokenize=True, add_generation_prompt=True, return_dict=True, return_tensors="pt", padding=True).to(self.model.device)
        with self.torch.inference_mode():
            generated = self.model.generate(**inputs, max_new_tokens=self.config.max_new_tokens, do_sample=False)
        trimmed = [output[len(source):] for source, output in zip(inputs.input_ids, generated)]
        texts = self.processor.batch_decode(trimmed, skip_special_tokens=False, clean_up_tokenization_spaces=False)
        return BackendResponse(tuple(texts), None)

    @staticmethod
    def is_oom(error: BaseException) -> bool:
        return "out of memory" in str(error).lower()

    def clear_cuda_cache(self) -> None:
        gc.collect()
        if self.torch.cuda.is_available():
            self.torch.cuda.empty_cache()
            self.torch.cuda.ipc_collect()
