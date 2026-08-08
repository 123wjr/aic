"""统一推理流程：分组、批处理、OOM 拆分、断点和输出。"""

from __future__ import annotations

import hashlib
import logging
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from .boxes import parse_bbox
from .checkpoint import DEFAULT_BOX, append_jsonl, atomic_dump, load_json_object
from .data import load_query_groups
from .postprocess import sanitize_bbox
from .prompts import PromptCache, PromptProvider, prepare_prompt_results
from .types import GroundingBackend, ImageGroup, InferenceUnit

LOG = logging.getLogger("grounding")


@dataclass(frozen=True)
class RunConfig:
    data_dir: Path
    query_file: Path | None
    output: Path
    partial: Path
    raw: Path
    prompt_cache: Path
    image_field: str = "visible"
    limit: int = 0
    batch_size: int = 4
    save_every: int = 100
    resume: bool = True
    shard_index: int = 0
    shard_count: int = 1


def _units(groups: list[ImageGroup], prompts: dict[str, Any], batch_size: int):
    """尽量让同图 query 共用一次视觉前缀，超大组再拆分。"""

    pending: list[InferenceUnit] = []
    rows = 0
    for group in groups:
        records = list(group.records)
        chunks = [records[i : i + batch_size] for i in range(0, len(records), batch_size)]
        for chunk in chunks:
            unit = InferenceUnit(group.image_key, group.image_path, tuple(chunk), tuple(prompts[r.query_id].text for r in chunk))
            if pending and rows + len(chunk) > batch_size:
                yield pending
                pending, rows = [], 0
            pending.append(unit)
            rows += len(chunk)
            if len(chunk) == batch_size:
                yield pending
                pending, rows = [], 0
    if pending:
        yield pending


def _split(unit: InferenceUnit) -> tuple[InferenceUnit, ...]:
    middle = max(1, len(unit.records) // 2)
    return tuple(InferenceUnit(unit.image_key, unit.image_path, tuple(unit.records[start:end]), tuple(unit.prompts[start:end])) for start, end in ((0, middle), (middle, len(unit.records))) if start < end)


def _infer_resilient(backend: GroundingBackend, units: list[InferenceUnit]) -> list[tuple[InferenceUnit, tuple[str, ...], dict[str, Any] | None]]:
    try:
        response = backend.infer(units)
        expected = sum(len(unit.records) for unit in units)
        if len(response.texts) != expected:
            raise RuntimeError(f"backend 返回数量异常: 期望 {expected}，实际 {len(response.texts)}")
        rows: list[tuple[InferenceUnit, tuple[str, ...], dict[str, Any] | None]] = []
        offset = 0
        for unit in units:
            count = len(unit.records)
            rows.append((unit, response.texts[offset : offset + count], response.stats))
            offset += count
        return rows
    except Exception as exc:
        if not backend.is_oom(exc):
            raise
        total = sum(len(unit.records) for unit in units)
        backend._recommended_batch_size = max(1, total // 2)
        logging.getLogger("grounding").warning("显存不足：当前工作量 %d，后续 batch 降至不超过 %d", total, backend._recommended_batch_size)
        backend.clear_cuda_cache()
        if len(units) > 1:
            middle = max(1, len(units) // 2)
            return _infer_resilient(backend, units[:middle]) + _infer_resilient(backend, units[middle:])
        if len(units[0].records) <= 1:
            raise
        parts = _split(units[0])
        return _infer_resilient(backend, [parts[0]]) + _infer_resilient(backend, [parts[1]])


def run(config: RunConfig, provider: PromptProvider, backend: GroundingBackend) -> dict[str, Any]:
    """执行一次可恢复的完整推理。"""

    if not config.resume:
        config.partial.unlink(missing_ok=True)
        config.raw.unlink(missing_ok=True)
    groups, queries = load_query_groups(config.data_dir, config.query_file, image_field=config.image_field, limit=config.limit)
    if config.shard_count > 1:
        groups = [group for index, group in enumerate(groups) if index % config.shard_count == config.shard_index]
        queries = {record.query_id: record.source for group in groups for record in group.records}
    # 先按 worker 分片，再调用提示词 provider，避免多 GPU 重复请求外部 LLM。
    prompt_results = prepare_prompt_results(groups, provider, PromptCache(config.prompt_cache))
    resume_source = config.partial if config.partial.is_file() else config.output
    results = load_json_object(resume_source) if config.resume else {}
    results = {key: value for key, value in results.items() if key in queries and isinstance(value, dict)}
    pending_groups = [ImageGroup(group.image_key, group.image_path, tuple(record for record in group.records if record.query_id not in results)) for group in groups]
    pending_groups = [group for group in pending_groups if group.records]
    processed = 0
    effective_batch_size = config.batch_size
    group_cursors = [0] * len(pending_groups)
    group_index = 0
    while group_index < len(pending_groups):
        # 一个全局 micro-batch 可以包含多个图像组，但每个图像组仍保持为独立 unit。
        batch: list[InferenceUnit] = []
        batch_rows = 0
        while group_index < len(pending_groups) and batch_rows < effective_batch_size:
            group = pending_groups[group_index]
            records = group.records
            start = group_cursors[group_index]
            if start >= len(records):
                group_index += 1
                continue
            size = min(effective_batch_size - batch_rows, len(records) - start)
            chunk = tuple(records[start : start + size])
            batch.append(InferenceUnit(group.image_key, group.image_path, chunk, tuple(prompt_results[r.query_id].text for r in chunk)))
            group_cursors[group_index] += size
            batch_rows += size
            if group_cursors[group_index] >= len(records):
                group_index += 1

        batch_started = time.time()
        try:
            rows = _infer_resilient(backend, batch)
        except Exception as exc:
            message = f"{type(exc).__name__}: {exc}"
            for unit in batch:
                for record in unit.records:
                    prompt = prompt_results[record.query_id]
                    append_jsonl(config.raw, {"query_id": record.query_id, "image_key": record.image_key, "status": "error", "raw_output": "", "bbox": None, "parse_status": "error", "error": message, "prompt_provider": prompt.provider, "prompt_version": prompt.version, "prompt": prompt.text})
            atomic_dump(results, config.partial)
            LOG.exception("批次失败，已保存断点: %s", config.partial)
            raise
        for unit, texts, stats in rows:
            for record, raw_text in zip(unit.records, texts):
                parsed = parse_bbox(raw_text)
                bbox = sanitize_bbox(parsed or DEFAULT_BOX)
                results[record.query_id] = {**record.source, "bbox": bbox}
                prompt = prompt_results[record.query_id]
                append_jsonl(config.raw, {"query_id": record.query_id, "image_key": record.image_key, "status": "success" if parsed else "fallback", "raw_output": raw_text, "bbox": bbox, "parse_status": "ok" if parsed else "fallback", "error": None, "prompt_provider": prompt.provider, "prompt_version": prompt.version, "prompt": prompt.text, "prompt_sha256": hashlib.sha256(prompt.text.encode("utf-8")).hexdigest(), "runtime_stats": stats, "elapsed_seconds": round(time.time() - batch_started, 6)})
                processed += 1
        recommended = getattr(backend, "_recommended_batch_size", None)
        if isinstance(recommended, int):
            effective_batch_size = max(1, min(effective_batch_size, recommended))
        atomic_dump(results, config.partial)
        if processed and (processed % config.save_every < batch_rows or len(results) == len(queries)):
            LOG.info("断点保存: %d/%d，当前 batch=%d", len(results), len(queries), effective_batch_size)
    atomic_dump(results, config.output)
    config.partial.unlink(missing_ok=True)
    return {"count": len(results), "new": processed, "skipped": len(results) - processed, "output": str(config.output), "effective_batch_size": effective_batch_size}
