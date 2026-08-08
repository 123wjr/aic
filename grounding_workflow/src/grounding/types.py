"""工作流内部使用的稳定数据契约。"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Protocol


@dataclass(frozen=True)
class QueryRecord:
    """一条查询及其原始字段。source 永远不被推理流程修改。"""

    query_id: str
    source: dict[str, Any]
    image_key: str
    image_path: Path
    query: str


@dataclass(frozen=True)
class ImageGroup:
    """共享同一张可见光图像的一组查询。"""

    image_key: str
    image_path: Path
    records: tuple[QueryRecord, ...]


@dataclass(frozen=True)
class PromptRequest:
    """提示词提供器可以看到的只读上下文。"""

    record: QueryRecord
    group_query_ids: tuple[str, ...]
    group_queries: tuple[str, ...]


@dataclass(frozen=True)
class PromptResult:
    """提示词结果及其可复现标识。"""

    text: str
    provider: str
    version: str
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class InferenceUnit:
    """一次 backend 调用的最小工作单元。"""

    image_key: str
    image_path: Path
    records: tuple[QueryRecord, ...]
    prompts: tuple[str, ...]


@dataclass(frozen=True)
class BackendResponse:
    """backend 返回的原始文本和运行统计。"""

    texts: tuple[str, ...]
    stats: dict[str, Any] | None = None


class GroundingBackend(Protocol):
    """Qwen、LocateAnything 等模型必须实现的最小接口。"""

    def infer(self, units: list[InferenceUnit]) -> BackendResponse:
        ...

    @staticmethod
    def is_oom(error: BaseException) -> bool:
        ...

    @staticmethod
    def clear_cuda_cache() -> None:
        ...
