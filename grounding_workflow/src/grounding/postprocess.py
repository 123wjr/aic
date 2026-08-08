"""提交格式、原字段不变和 bbox 几何校验。"""

from __future__ import annotations

import math
from typing import Any

from .checkpoint import DEFAULT_BOX


def sanitize_bbox(value: Any) -> list[float]:
    """将模型结果变成合法框；无法修复时使用固定兜底框。"""

    if not isinstance(value, (list, tuple)) or len(value) != 4 or any(isinstance(v, bool) or not isinstance(v, (int, float)) for v in value):
        return DEFAULT_BOX.copy()
    coords = [float(v) for v in value]
    if any(not math.isfinite(v) for v in coords):
        return DEFAULT_BOX.copy()
    x1, y1, x2, y2 = coords
    return [round(v, 6) for v in coords] if 0 <= x1 < x2 <= 1 and 0 <= y1 < y2 <= 1 else DEFAULT_BOX.copy()


def validate_submission(predictions: Any, queries: Any, *, allow_subset: bool = False) -> list[str]:
    """严格检查评测要求的字段和归一化 bbox。"""

    if not isinstance(predictions, dict) or not isinstance(queries, dict):
        return ["预测和查询顶层都必须是 JSON object"]
    errors: list[str] = []
    ids = set(predictions)
    expected = set(queries)
    if not allow_subset and expected - ids:
        errors.append(f"缺少 query: {sorted(expected - ids)[:5]}")
    if ids - expected:
        errors.append(f"多余 query: {sorted(ids - expected)[:5]}")
    for query_id in sorted(ids & expected):
        source, prediction = queries[query_id], predictions[query_id]
        if not isinstance(source, dict) or not isinstance(prediction, dict):
            errors.append(f"{query_id}: 记录必须是 object")
            continue
        if set(prediction) != set(source) | {"bbox"}:
            errors.append(f"{query_id}: 只能保留原字段并新增 bbox")
        for field, value in source.items():
            if prediction.get(field) != value:
                errors.append(f"{query_id}: 原字段 {field!r} 被修改")
        bbox = prediction.get("bbox")
        if not isinstance(bbox, list) or len(bbox) != 4 or any(isinstance(v, bool) or not isinstance(v, (int, float)) for v in bbox):
            errors.append(f"{query_id}: bbox 格式非法")
            continue
        coords = [float(v) for v in bbox]
        if any(not math.isfinite(v) for v in coords) or not (0 <= coords[0] < coords[2] <= 1 and 0 <= coords[1] < coords[3] <= 1):
            errors.append(f"{query_id}: bbox 几何非法 {bbox}")
    return errors
