"""可插拔提示词处理和持久化缓存。"""

from __future__ import annotations

import hashlib
import importlib
import json
from dataclasses import replace
from pathlib import Path
from typing import Any, Callable, Protocol

from .types import ImageGroup, PromptRequest, PromptResult, QueryRecord

# Prompt generation is model-aware only through an explicit profile. Backends
# receive the resulting text and must never add model instructions themselves.
PROMPT_PROFILES = (
    "qwen_json",
    "locateanything",
    "locateanything_multi",
    "locateanything_text",
    "internvl_json_union",
    "internvl_native_box",
    "internvl_minimal",
)
DEFAULT_PROFILE_BY_BACKEND = {
    "qwen": "qwen_json",
    "locateanything": "locateanything",
    "internvl": "internvl_json_union",
}


def normalize_prompt_profile(value: str | None) -> str:
    aliases = {
        "": "qwen_json",
        "qwen": "qwen_json",
        "json": "qwen_json",
        "locate": "locateanything",
        "la": "locateanything",
        "locate_multi": "locateanything_multi",
        "la_multi": "locateanything_multi",
        "locate_text": "locateanything_text",
        "la_text": "locateanything_text",
        "internvl": "internvl_json_union",
        "json_union": "internvl_json_union",
        "native_box": "internvl_native_box",
        "minimal": "internvl_minimal",
    }
    raw = (value or "qwen_json").strip().lower().replace("-", "_")
    profile = aliases.get(raw, raw)
    if profile not in PROMPT_PROFILES:
        raise ValueError(
            f"prompt profile must be one of {', '.join(PROMPT_PROFILES)}; got {value!r}"
        )
    return profile


def default_prompt_profile(backend: str) -> str:
    try:
        return DEFAULT_PROFILE_BY_BACKEND[backend]
    except KeyError as exc:
        raise ValueError(f"unknown backend for default prompt profile: {backend!r}") from exc


def build_prompt(query: str, profile: str = "qwen_json") -> str:
    """Convert one raw query into the complete model-facing prompt."""

    profile = normalize_prompt_profile(profile)
    target = query.strip()
    if not target:
        raise ValueError("query must not be empty")
    if profile == "qwen_json":
        return (
            "Locate the visual target described by the query; it may be one object or multiple objects.\n"
            f"Target: {target}\n\n"
            "For a plural, counted, or collective target, return one tight box enclosing all matching instances. "
            "For a query selecting one instance by relation or attribute, return only that instance.\n"
            'Return only this JSON object: {"bbox_2d": [x1, y1, x2, y2]}\n'
            "Use integer coordinates on a relative 0-1000 grid and include no explanation."
        )
    if profile == "locateanything":
        return (
            "Locate all the instances that match the following description.\n"
            f"{target}\n"
            "Return the box covering all described instances."
        )
    if profile == "locateanything_multi":
        return (
            f"Locate all the instances that match the following description: {target}. "
            "Return one box per instance."
        )
    if profile == "locateanything_text":
        return (
            f"Locate the text that matches the following description: {target}. "
            f"Please locate the text referred as {target}."
        )
    if profile == "internvl_json_union":
        return (
            "Locate the visual target described by the query.\n"
            "The target may be one object or multiple objects.\n"
            f"Target description: {target}\n\n"
            "If the query describes multiple instances, a count, or a collective target, "
            "include all matching instances in one tight enclosing bounding box. "
            "If the query identifies one instance by a relation or attribute, box only that instance.\n"
            "Return exactly one JSON object:\n"
            '{"bbox_2d": [x1, y1, x2, y2]}\n'
            "Use integer coordinates on the full-image 0-1000 grid. "
            "The box must contain all and only the described target instances, satisfy x1 < x2 and y1 < y2, "
            "and include no explanation, Markdown, or extra objects."
        )
    if profile == "internvl_native_box":
        return (
            "Find the visual target described below.\n"
            "The target can be one object or multiple instances.\n"
            f"Target description: {target}\n\n"
            "If the query describes multiple instances, a count, or a collective target, "
            "return one box enclosing all matching instances. "
            "If the query identifies one instance by a relation or attribute, box only that instance.\n"
            "Return exactly one box in this format:\n"
            "<box><x1><y1><x2><y2></box>\n"
            "Coordinates must be integer values on the full-image 0-1000 grid. "
            "Return no explanation or extra boxes."
        )
    return (
        "Locate the target described by the query.\n"
        "If the query describes multiple instances, a count, or a collective target, "
        "enclose all matching instances in one box.\n"
        f"Target description: {target}\n\n"
        'Return one bbox on a 0-1000 image grid: {"bbox_2d": [x1, y1, x2, y2]}\n'
        "Return no explanation or Markdown."
    )


class PromptProvider(Protocol):
    name: str
    version: str

    def build(self, request: PromptRequest) -> PromptResult:
        ...

    def build_all(self, requests: tuple[PromptRequest, ...]) -> dict[str, PromptResult]:
        ...


class DefaultPromptProvider:
    name = "default"

    def __init__(self, profile: str = "qwen_json"):
        self.profile = normalize_prompt_profile(profile)
        self.version = f"2:{self.profile}"

    def build(self, request: PromptRequest) -> PromptResult:
        return PromptResult(build_prompt(request.record.query, self.profile), self.name, self.version)

    def build_all(self, requests: tuple[PromptRequest, ...]) -> dict[str, PromptResult]:
        return {request.record.query_id: self.build(request) for request in requests}


class CallablePromptProvider:
    """把 module:function 适配为提示词提供器。"""

    def __init__(self, function: Callable[[PromptRequest], str | PromptResult], name: str):
        self.function = function
        self.name = name
        self.version = str(getattr(function, "__prompt_version__", "external"))
        self.profile = None
        self.batch = bool(getattr(function, "__prompt_batch__", False))

    def build(self, request: PromptRequest) -> PromptResult:
        result = self.function(request)
        if isinstance(result, PromptResult):
            return replace(result, provider=self.name, version=self.version)
        if not isinstance(result, str) or not result.strip():
            raise ValueError(f"提示词提供器 {self.name} 返回空文本")
        return PromptResult(result.strip().rstrip(" ."), self.name, self.version)

    def build_all(self, requests: tuple[PromptRequest, ...]) -> dict[str, PromptResult]:
        if self.batch:
            result = self.function(requests)
            if isinstance(result, dict):
                return result
            if isinstance(result, (tuple, list)) and len(result) == len(requests):
                return {request.record.query_id: value for request, value in zip(requests, result)}
            raise TypeError(
                "batch prompt provider must return a query_id mapping or one result per request"
            )
        return {request.record.query_id: self.build(request) for request in requests}


def load_provider(spec: str | None, profile: str = "qwen_json") -> PromptProvider:
    """加载默认提供器或用户指定的 module:function。"""

    if not spec:
        return DefaultPromptProvider(profile)
    if ":" not in spec:
        raise ValueError("--prompt_provider 必须使用 module:function 格式")
    module_name, attr_name = spec.split(":", 1)
    target = getattr(importlib.import_module(module_name), attr_name)
    if hasattr(target, "build") or hasattr(target, "build_all"):
        return target
    if callable(target):
        return CallablePromptProvider(target, spec)
    raise TypeError(f"提示词目标不可调用: {spec}")


def _request(group: ImageGroup, record: QueryRecord) -> PromptRequest:
    return PromptRequest(record, tuple(item.query_id for item in group.records), tuple(item.query for item in group.records))


class PromptCache:
    """按 query、provider 和版本缓存提示词，支持断点续跑。"""

    def __init__(self, path: Path, namespace: str = "default"):
        self.path = path
        self.namespace = namespace
        self.entries: dict[tuple[str, str], dict[str, Any]] = {}
        if path.is_file():
            for number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
                if not line.strip():
                    continue
                try:
                    row = json.loads(line)
                except json.JSONDecodeError as exc:
                    raise ValueError(f"{path}:{number}: JSON 无效") from exc
                if isinstance(row, dict) and isinstance(row.get("query_id"), str):
                    self.entries[(str(row.get("namespace", "default")), row["query_id"])] = row

    def get(
        self,
        record: QueryRecord,
        provider: PromptProvider,
        query_set_sha256: str | None = None,
    ) -> PromptResult | None:
        row = self.entries.get((self.namespace, record.query_id))
        digest = hashlib.sha256(record.query.encode("utf-8")).hexdigest()
        if (
            not row
            or row.get("namespace", "default") != self.namespace
            or row.get("query_set_sha256") != query_set_sha256
            or row.get("query_sha256") != digest
            or row.get("provider") != provider.name
            or row.get("version") != provider.version
        ):
            return None
        text = row.get("text")
        return PromptResult(text, str(row["provider"]), str(row["version"]), dict(row.get("metadata") or {})) if isinstance(text, str) and text.strip() else None

    def put(
        self,
        record: QueryRecord,
        result: PromptResult,
        query_set_sha256: str | None = None,
    ) -> None:
        row = {
            "query_id": record.query_id,
            "namespace": self.namespace,
            "query_set_sha256": query_set_sha256,
            "query_sha256": hashlib.sha256(record.query.encode("utf-8")).hexdigest(),
            "provider": result.provider,
            "version": result.version,
            "text": result.text,
            "metadata": result.metadata,
        }
        self.path.parent.mkdir(parents=True, exist_ok=True)
        with self.path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")
        self.entries[(self.namespace, record.query_id)] = row


def prepare_prompt_results(groups: list[ImageGroup], provider: PromptProvider, cache: PromptCache) -> dict[str, PromptResult]:
    """Build prompts once for the complete query set before any model batching."""

    results: dict[str, PromptResult] = {}
    requests = [_request(group, record) for group in groups for record in group.records]
    query_set_payload = [(request.record.query_id, request.record.query) for request in requests]
    query_set_sha256 = hashlib.sha256(
        json.dumps(query_set_payload, ensure_ascii=False, separators=(",", ":")).encode("utf-8")
    ).hexdigest()
    missing_ids: set[str] = set()
    for request in requests:
        record = request.record
        result = cache.get(record, provider, query_set_sha256)
        if result is None:
            missing_ids.add(record.query_id)
        else:
            results[record.query_id] = result
    if missing_ids:
        build_all = getattr(provider, "build_all", None)
        built = build_all(tuple(requests)) if callable(build_all) else {
            request.record.query_id: provider.build(request) for request in requests
        }
        if not isinstance(built, dict):
            raise TypeError("prompt provider build_all must return a query_id -> PromptResult mapping")
        for request in requests:
            if request.record.query_id not in missing_ids:
                continue
            result = built.get(request.record.query_id)
            if isinstance(result, str):
                result = PromptResult(result.strip(), provider.name, provider.version)
            if not isinstance(result, PromptResult) or not result.text.strip():
                raise ValueError(f"{request.record.query_id}: 提示词结果无效")
            result = replace(result, provider=provider.name, version=provider.version)
            cache.put(request.record, result, query_set_sha256)
            results[request.record.query_id] = result
    return results
