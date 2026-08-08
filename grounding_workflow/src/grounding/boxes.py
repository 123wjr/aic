"""从模型文本中提取并归一化 bbox。"""

from __future__ import annotations

import json
import math
import re
from collections.abc import Iterable, Sequence
from typing import Any

_NUMBER = r"[-+]?(?:\d+(?:\.\d*)?|\.\d+)"
_KEYED = re.compile(rf"[\"']?(?:bbox_2d|bbox)[\"']?\s*:\s*\[\s*({_NUMBER})\s*,\s*({_NUMBER})\s*,\s*({_NUMBER})\s*,\s*({_NUMBER})\s*\]", re.I)
_BOX = re.compile(rf"<box>\s*<?\s*({_NUMBER})\s*>?\s*<?\s*({_NUMBER})\s*>?\s*<?\s*({_NUMBER})\s*>?\s*<?\s*({_NUMBER})\s*>?\s*</box>", re.I)
_LIST = re.compile(rf"\[\s*({_NUMBER})\s*,\s*({_NUMBER})\s*,\s*({_NUMBER})\s*,\s*({_NUMBER})\s*\]")


def normalize_bbox(values: Sequence[Any]) -> list[float] | None:
    """接受 0-1 或 0-1000 坐标，并拒绝反向、越界、空框。"""

    if len(values) != 4 or any(isinstance(v, bool) or not isinstance(v, (int, float)) for v in values):
        return None
    coords = [float(v) for v in values]
    if any(not math.isfinite(v) or v < 0 for v in coords):
        return None
    scale = 1.0 if max(coords) <= 1.0 else 1000.0 if max(coords) <= 1000.0 else 0.0
    if not scale:
        return None
    x1, y1, x2, y2 = [v / scale for v in coords]
    return [round(v, 6) for v in (x1, y1, x2, y2)] if 0 <= x1 < x2 <= 1 and 0 <= y1 < y2 <= 1 else None


def _json_values(text: str) -> Iterable[Any]:
    decoder = json.JSONDecoder()
    for index, char in enumerate(text):
        if char not in "[{":
            continue
        try:
            value, _ = decoder.raw_decode(text[index:])
        except json.JSONDecodeError:
            continue
        yield value


def _candidates(value: Any) -> Iterable[Sequence[Any]]:
    if isinstance(value, dict):
        for key in ("bbox_2d", "bbox"):
            candidate = value.get(key)
            if isinstance(candidate, list):
                yield candidate
        for nested in value.values():
            if isinstance(nested, (dict, list)):
                yield from _candidates(nested)
    elif isinstance(value, list):
        if len(value) == 4:
            yield value
        else:
            for nested in value:
                if isinstance(nested, (dict, list)):
                    yield from _candidates(nested)


def parse_bbox(text: str) -> list[float] | None:
    """只识别显式 bbox，避免把推理过程中的普通数字误当坐标。"""

    if not isinstance(text, str) or not text.strip():
        return None
    for value in _json_values(text):
        for candidate in _candidates(value):
            result = normalize_bbox(candidate)
            if result is not None:
                return result
    for pattern in (_BOX, _KEYED, _LIST):
        for match in pattern.finditer(text):
            result = normalize_bbox([float(item) for item in match.groups()])
            if result is not None:
                return result
    return None
