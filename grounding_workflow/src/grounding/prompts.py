"""可插拔提示词处理和持久化缓存。"""

from __future__ import annotations

import hashlib
import importlib
import json
from dataclasses import replace
from pathlib import Path
from typing import Any, Callable, Protocol

from .types import ImageGroup, PromptRequest, PromptResult, QueryRecord

# LocateAnything 官方 runtime 的默认前缀允许一个 query 描述多个实例。
DEFAULT_PREFIX = "Locate all the instances that matches the following description: "


class PromptProvider(Protocol):
    name: str
    version: str
    prefix: str

    def build(self, request: PromptRequest) -> PromptResult:
        ...


class DefaultPromptProvider:
    name = "default"
    version = "1"
    prefix = DEFAULT_PREFIX

    def build(self, request: PromptRequest) -> PromptResult:
        return PromptResult(request.record.query.rstrip(" ."), self.name, self.version)


class CallablePromptProvider:
    """把 module:function 适配为提示词提供器。"""

    def __init__(self, function: Callable[[PromptRequest], str | PromptResult], name: str):
        self.function = function
        self.name = name
        self.version = str(getattr(function, "__prompt_version__", "external"))
        self.prefix = str(getattr(function, "__prompt_prefix__", DEFAULT_PREFIX))

    def build(self, request: PromptRequest) -> PromptResult:
        result = self.function(request)
        if isinstance(result, PromptResult):
            return replace(result, provider=self.name, version=self.version)
        if not isinstance(result, str) or not result.strip():
            raise ValueError(f"提示词提供器 {self.name} 返回空文本")
        return PromptResult(result.strip().rstrip(" ."), self.name, self.version)


def load_provider(spec: str | None) -> PromptProvider:
    """加载默认提供器或用户指定的 module:function。"""

    if not spec:
        return DefaultPromptProvider()
    if ":" not in spec:
        raise ValueError("--prompt_provider 必须使用 module:function 格式")
    module_name, attr_name = spec.split(":", 1)
    target = getattr(importlib.import_module(module_name), attr_name)
    if hasattr(target, "build"):
        return target
    if callable(target):
        return CallablePromptProvider(target, spec)
    raise TypeError(f"提示词目标不可调用: {spec}")


def _request(group: ImageGroup, record: QueryRecord) -> PromptRequest:
    return PromptRequest(record, tuple(item.query_id for item in group.records), tuple(item.query for item in group.records))


class PromptCache:
    """按 query、provider 和版本缓存提示词，支持断点续跑。"""

    def __init__(self, path: Path):
        self.path = path
        self.entries: dict[str, dict[str, Any]] = {}
        if path.is_file():
            for number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
                if not line.strip():
                    continue
                try:
                    row = json.loads(line)
                except json.JSONDecodeError as exc:
                    raise ValueError(f"{path}:{number}: JSON 无效") from exc
                if isinstance(row, dict) and isinstance(row.get("query_id"), str):
                    self.entries[row["query_id"]] = row

    def get(self, record: QueryRecord, provider: PromptProvider) -> PromptResult | None:
        row = self.entries.get(record.query_id)
        digest = hashlib.sha256(record.query.encode("utf-8")).hexdigest()
        if not row or row.get("query_sha256") != digest or row.get("provider") != provider.name or row.get("version") != provider.version:
            return None
        text = row.get("text")
        return PromptResult(text, str(row["provider"]), str(row["version"]), dict(row.get("metadata") or {})) if isinstance(text, str) and text.strip() else None

    def put(self, record: QueryRecord, result: PromptResult) -> None:
        row = {
            "query_id": record.query_id,
            "query_sha256": hashlib.sha256(record.query.encode("utf-8")).hexdigest(),
            "provider": result.provider,
            "version": result.version,
            "text": result.text,
            "metadata": result.metadata,
        }
        self.path.parent.mkdir(parents=True, exist_ok=True)
        with self.path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")
        self.entries[record.query_id] = row


def prepare_prompt_results(groups: list[ImageGroup], provider: PromptProvider, cache: PromptCache) -> dict[str, PromptResult]:
    """优先命中缓存，再按 query 调用 provider。"""

    results: dict[str, PromptResult] = {}
    for group in groups:
        for record in group.records:
            result = cache.get(record, provider)
            if result is None:
                result = provider.build(_request(group, record))
                if isinstance(result, str):
                    result = PromptResult(result.strip(), provider.name, provider.version)
                if not isinstance(result, PromptResult) or not result.text.strip():
                    raise ValueError(f"{record.query_id}: 提示词结果无效")
                result = replace(result, provider=provider.name, version=provider.version)
                cache.put(record, result)
            results[record.query_id] = result
    return results
