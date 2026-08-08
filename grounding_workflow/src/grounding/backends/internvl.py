"""InternVL3.5-HF backend，支持视觉特征 LRU 缓存。"""

from __future__ import annotations

import gc
from collections import OrderedDict
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from ..types import BackendResponse, GroundingBackend, InferenceUnit


@dataclass(frozen=True)
class InternVLConfig:
    """InternVL 推理参数。"""

    model: str
    dtype: str = "bfloat16"
    max_new_tokens: int = 128
    image_cache_size: int = 1
    gpu_memory_limit: str | None = None


def build_prompt(query: str) -> str:
    """InternVL3.5 grounding 提示词。

    坐标使用模型常见的 0-1000 图像坐标系，公共 parser 会再归一化到 0-1。
    """

    return (
        "Locate the visual target described by the query. The target may be one object or a group of objects.\n"
        f"Target description: {query.strip()}\n\n"
        "If the query describes multiple instances, a count, or a collective target, "
        "include all matching instances in one tight enclosing bounding box. "
        "If the query identifies one instance by a relation or attribute, box only that instance.\n"
        "Return exactly one bounding box. Preferred format: "
        '{"bbox_2d": [x1, y1, x2, y2]}\n'
        "Use integer coordinates on the full-image 0-1000 grid. "
        "The box must contain all and only the described target instances, satisfy x1 < x2 and y1 < y2, "
        "and include no explanation, Markdown, or extra objects."
    )


class InternVLBackend(GroundingBackend):
    """InternVL3.5-HF 的 Transformers 适配器。"""

    def __init__(self, config: InternVLConfig):
        try:
            import torch
            from transformers import AutoModelForImageTextToText, AutoProcessor
        except ImportError as exc:
            try:
                from transformers import AutoModel, AutoProcessor
                AutoModelForImageTextToText = AutoModel
            except ImportError:
                raise RuntimeError("缺少 InternVL 依赖，请安装 requirements.txt") from exc

        dtype: Any = torch.bfloat16 if config.dtype == "bfloat16" else torch.float16
        kwargs: dict[str, Any] = {
            "torch_dtype": dtype,
            "device_map": "auto",
            "trust_remote_code": True,
            "low_cpu_mem_usage": True,
        }
        if config.gpu_memory_limit:
            kwargs["max_memory"] = {0: config.gpu_memory_limit, "cpu": "96GiB"}

        self.torch = torch
        self.config = config
        self.model = AutoModelForImageTextToText.from_pretrained(config.model, **kwargs).eval()
        self.processor = AutoProcessor.from_pretrained(config.model, trust_remote_code=True)
        self.tokenizer = getattr(self.processor, "tokenizer", self.processor)
        self._cache: OrderedDict[str, dict[str, Any]] = OrderedDict()
        self._cache_limit = max(0, config.image_cache_size)
        self._feature_api = hasattr(getattr(self.model, "model", None), "get_image_features")
        self._feature_device = self._find_device(getattr(getattr(self.model, "model", None), "vision_tower", None))
        self._embed_device = self.model.get_input_embeddings().weight.device

    @staticmethod
    def _find_device(module: Any):
        try:
            return next(module.parameters()).device
        except (AttributeError, StopIteration):
            return None

    def _load_image(self, path: Path):
        from PIL import Image

        with Image.open(path) as source:
            return source.convert("RGB").copy()

    def _get_cached_image(self, key: str, path: Path) -> dict[str, Any]:
        cached = self._cache.get(key)
        if cached is not None:
            self._cache.move_to_end(key)
            return cached
        image = self._load_image(path)
        item: dict[str, Any] = {"image": image}
        if self._feature_api:
            image_inputs = self.processor.image_processor(images=[image], return_tensors="pt")
            pixel_values = image_inputs["pixel_values"]
            num_patches = image_inputs.get("num_patches")
            if num_patches is None:
                num_patches = [int(pixel_values.shape[0])]
            item["pixel_values"] = pixel_values
            item["num_patches"] = [int(value) for value in num_patches]
            with self.torch.inference_mode():
                vision = pixel_values.to(self._feature_device or self._embed_device)
                features = self.model.model.get_image_features(pixel_values=vision, return_dict=True).pooler_output
            item["features"] = features.detach()
        if self._cache_limit:
            self._cache[key] = item
            self._cache.move_to_end(key)
            while len(self._cache) > self._cache_limit:
                self._cache.popitem(last=False)
        return item

    def _template_text(self, prompt: str, image: Any) -> str:
        messages = [{"role": "user", "content": [{"type": "image", "image": image}, {"type": "text", "text": build_prompt(prompt)}]}]
        return self.processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)

    def _cached_row(self, prompt: str, item: dict[str, Any]):
        """用缓存视觉特征构造一行 inputs_embeds，避免再次跑视觉塔。"""

        if "features" not in item:
            raise RuntimeError("InternVL checkpoint 未提供 get_image_features")
        image_token = getattr(self.processor, "image_token", "<image>")
        if image_token not in (text := self._template_text(prompt, item["image"])):
            raise RuntimeError(f"InternVL chat template 未找到 image token: {image_token!r}")
        seq_len = int(getattr(self.model.config, "image_seq_length", getattr(self.model.model.config, "image_seq_length", 1)))
        start = getattr(self.processor, "start_image_token", "")
        end = getattr(self.processor, "end_image_token", "")
        replacement = start + image_token * (seq_len * sum(item["num_patches"])) + end
        expanded = text.replace(image_token, replacement, 1)
        tokens = self.tokenizer(expanded, return_tensors="pt", add_special_tokens=False)
        input_ids = tokens["input_ids"].to(self._embed_device)
        embeddings = self.model.model.get_input_embeddings()(input_ids)
        features = item["features"].to(device=embeddings.device, dtype=embeddings.dtype)
        mask = self.model.model.get_placeholder_mask(input_ids, embeddings, features)
        embeddings = embeddings.masked_scatter(mask, features.reshape(-1))
        return embeddings[0], tokens["attention_mask"][0].to(self._embed_device)

    def _infer_cached(self, rows: list[tuple[str, dict[str, Any]]]) -> tuple[str, ...]:
        packed = [self._cached_row(prompt, item) for prompt, item in rows]
        max_length = max(int(emb.shape[0]) for emb, _ in packed)
        hidden = int(packed[0][0].shape[-1])
        embeds = self.torch.zeros((len(packed), max_length, hidden), dtype=packed[0][0].dtype, device=self._embed_device)
        mask = self.torch.zeros((len(packed), max_length), dtype=self.torch.long, device=self._embed_device)
        for index, (row_embeds, row_mask) in enumerate(packed):
            length = row_embeds.shape[0]
            embeds[index, :length] = row_embeds
            mask[index, :length] = row_mask
        with self.torch.inference_mode():
            generated = self.model.language_model.generate(inputs_embeds=embeds, attention_mask=mask, max_new_tokens=self.config.max_new_tokens, do_sample=False)
        return tuple(self.tokenizer.batch_decode(generated, skip_special_tokens=False, clean_up_tokenization_spaces=False))

    def _infer_standard(self, rows: list[tuple[str, dict[str, Any]]]) -> tuple[str, ...]:
        messages = []
        for prompt, item in rows:
            messages.append([{"role": "user", "content": [{"type": "image", "image": item["image"]}, {"type": "text", "text": build_prompt(prompt)}]}])
        inputs = self.processor.apply_chat_template(messages, tokenize=True, add_generation_prompt=True, return_dict=True, return_tensors="pt", padding=True).to(self._embed_device)
        with self.torch.inference_mode():
            generated = self.model.generate(**inputs, max_new_tokens=self.config.max_new_tokens, do_sample=False)
        trimmed = [output[len(source) :] for source, output in zip(inputs["input_ids"], generated)]
        return tuple(self.processor.batch_decode(trimmed, skip_special_tokens=False, clean_up_tokenization_spaces=False))

    def infer(self, units: list[InferenceUnit]) -> BackendResponse:
        rows: list[tuple[str, dict[str, Any]]] = []
        for unit in units:
            item = self._get_cached_image(unit.image_key, unit.image_path)
            rows.extend((prompt, item) for prompt in unit.prompts)
        try:
            texts = self._infer_cached(rows) if self._feature_api else self._infer_standard(rows)
        except (RuntimeError, ValueError, KeyError, AttributeError) as exc:
            # 不同 InternVL checkpoint 的 chat template 细节可能不同，保留官方标准路径。
            if self.is_oom(exc):
                raise
            texts = self._infer_standard(rows)
        return BackendResponse(tuple(texts), None)

    @staticmethod
    def is_oom(error: BaseException) -> bool:
        return "out of memory" in str(error).lower()

    def clear_cuda_cache(self) -> None:
        gc.collect()
        if self.torch.cuda.is_available():
            self.torch.cuda.empty_cache()
            self.torch.cuda.ipc_collect()
