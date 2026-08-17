"""LocateAnything 官方 batch runtime 适配器。"""

from __future__ import annotations

import gc
import importlib
import os
import sys
from collections import OrderedDict
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from ..types import BackendResponse, GroundingBackend, InferenceUnit


@dataclass(frozen=True)
class LocateAnythingConfig:
    model: str
    attn: str = "sdpa"
    vision_attn: str = "auto"
    scheduler: str = "eager"
    group_size: int = 0
    max_new_tokens: int = 2048
    temperature: float = 0.0
    top_p: float | None = None
    top_k: int | None = None
    repetition_penalty: float = 1.1
    feature_cache_size: int = 1


class LocateAnythingBackend(GroundingBackend):
    """LocateAnything 的差异仅限于官方加载、视觉特征和生成调用。"""

    def __init__(self, config: LocateAnythingConfig):
        self.config = config
        os.environ.update(
            {
                "LA_FLASH_MODEL": config.model,
                "LA_FLASH_ATTN": config.attn,
                "LA_FLASH_VISION_ATTN": config.vision_attn,
                "LA_FLASH_HYBRID_SCHEDULER": config.scheduler,
                "LA_FLASH_HYBRID_GROUP_SIZE": str(config.group_size),
                "LA_FLASH_HYBRID_PREFILL": "shared",
                "LA_FLASH_CACHE_TOKENIZE": "1",
            }
        )
        model_dir = self._resolve_model_dir(config.model)
        if str(model_dir) not in sys.path:
            sys.path.insert(0, str(model_dir))
        batch_utils = importlib.import_module("batch_utils")
        runtime = importlib.import_module("batch_utils.hybrid_runtime")
        self._generate = batch_utils.generate_batch_grouped_hybrid
        self._stats = batch_utils.get_last_hybrid_stats
        self._load = batch_utils.load
        self._encode_images = runtime._encode_images
        self._load_pil = runtime.load_pil
        runtime._PROMPT = ""
        self._features: OrderedDict[str, tuple[Any, Any]] = OrderedDict()
        self._cache_limit = max(0, config.feature_cache_size)
        self._load()

    @staticmethod
    def _resolve_model_dir(model: str) -> Path:
        path = Path(model).expanduser()
        if path.is_dir():
            return path.resolve()
        try:
            from huggingface_hub import snapshot_download
        except ImportError as exc:
            raise RuntimeError("LocateAnything 需要本地目录或 huggingface_hub") from exc
        return Path(snapshot_download(model)).resolve()

    def _batch_features(self, units: list[InferenceUnit]) -> dict[str, tuple[Any, Any]]:
        found: dict[str, tuple[Any, Any]] = {}
        missing: list[tuple[str, Path]] = []
        for unit in units:
            if unit.image_key in found:
                continue
            if unit.image_key in self._features:
                found[unit.image_key] = self._features[unit.image_key]
                self._features.move_to_end(unit.image_key)
            else:
                missing.append((unit.image_key, unit.image_path))
        if missing:
            images = [self._load_pil(path) for _, path in missing]
            encoded = self._encode_images(images)
            for (key, _), image, feature in zip(missing, images, encoded):
                found[key] = (image, feature)
                if self._cache_limit:
                    self._features[key] = (image, feature)
            while len(self._features) > self._cache_limit:
                self._features.popitem(last=False)
        return found

    def infer(self, units: list[InferenceUnit]) -> BackendResponse:
        cache = self._batch_features(units)
        pairs = [(cache[u.image_key][0], list(u.prompts)) for u in units]
        features = [cache[u.image_key][1] for u in units]
        outputs = self._generate(
            pairs,
            temperature=self.config.temperature,
            top_p=self.config.top_p,
            top_k=self.config.top_k,
            repetition_penalty=self.config.repetition_penalty,
            max_new_tokens=self.config.max_new_tokens,
            scheduler=self.config.scheduler,
            group_size=self.config.group_size,
            vision_features=features,
        )
        return BackendResponse(
            tuple(text for group in outputs for text in group), self._stats()
        )

    @staticmethod
    def is_oom(error: BaseException) -> bool:
        return "out of memory" in str(error).lower()

    @staticmethod
    def clear_cuda_cache() -> None:
        gc.collect()
        try:
            import torch

            if torch.cuda.is_available():
                torch.cuda.empty_cache()
                torch.cuda.ipc_collect()
        except Exception:
            pass
