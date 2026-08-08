"""读取查询文件，并按可见光图像构建分组。"""

from __future__ import annotations

import json
from collections import OrderedDict
from pathlib import Path
from typing import Any

from .types import ImageGroup, QueryRecord


def load_query_groups(
    data_dir: Path,
    query_file: Path | None = None,
    *,
    image_field: str = "visible",
    limit: int | None = None,
) -> tuple[list[ImageGroup], dict[str, dict[str, Any]]]:
    """加载查询；只解析 visible 路径，其他模态字段原样保留。"""

    root = data_dir.resolve()
    path = (query_file or root / "queries" / "queries.json").resolve()
    with path.open("r", encoding="utf-8") as handle:
        payload = json.load(handle)
    if not isinstance(payload, dict):
        raise ValueError(f"{path}: 顶层必须是 JSON object")

    grouped: OrderedDict[str, list[QueryRecord]] = OrderedDict()
    selected: dict[str, dict[str, Any]] = {}
    items = list(payload.items())[: limit if limit and limit > 0 else None]
    for query_id, source in items:
        if not isinstance(query_id, str) or not query_id:
            raise ValueError("query ID 必须是非空字符串")
        if not isinstance(source, dict):
            raise ValueError(f"{query_id}: 查询记录必须是 object")
        raw_image = source.get(image_field)
        if not isinstance(raw_image, str) or not raw_image:
            raise ValueError(f"{query_id}: 缺少非空 {image_field} 字段")
        image_path = (root / raw_image).resolve()
        if not image_path.is_relative_to(root):
            raise ValueError(f"{query_id}: 图像路径越出数据目录: {raw_image}")
        if not image_path.is_file():
            raise FileNotFoundError(f"{query_id}: 图像不存在: {image_path}")
        query = source.get("query")
        if not isinstance(query, str) or not query.strip():
            raise ValueError(f"{query_id}: query 必须是非空字符串")
        record = QueryRecord(query_id, dict(source), raw_image.replace("\\", "/"), image_path, query.strip())
        grouped.setdefault(record.image_key, []).append(record)
        selected[query_id] = dict(source)

    groups = [ImageGroup(key, rows[0].image_path, tuple(rows)) for key, rows in grouped.items()]
    return groups, selected
