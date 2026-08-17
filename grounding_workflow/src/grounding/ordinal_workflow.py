"""Ordinal workflow helpers: LA candidates, Qwen selection, one final box."""

from __future__ import annotations

import json
import re
from dataclasses import dataclass
from typing import Any

from .boxes import normalize_bbox, parse_bbox


_ORDINAL_RANK = {
    "first": 0,
    "second": 1,
    "third": 2,
    "fourth": 3,
    "fifth": 4,
    "sixth": 5,
    "seventh": 6,
    "eighth": 7,
    "ninth": 8,
    "tenth": 9,
}
_BOX = re.compile(
    r"<box>\s*<?\s*([-+]?(?:\d+(?:\.\d*)?|\.\d+))\s*>?\s*"
    r"<?\s*([-+]?(?:\d+(?:\.\d*)?|\.\d+))\s*>?\s*"
    r"<?\s*([-+]?(?:\d+(?:\.\d*)?|\.\d+))\s*>?\s*"
    r"<?\s*([-+]?(?:\d+(?:\.\d*)?|\.\d+))\s*>?\s*</box>",
    re.IGNORECASE,
)


@dataclass(frozen=True)
class OrdinalPlan:
    target: str
    order: str
    rank: int


def plan_ordinal_query(query: str) -> OrdinalPlan:
    text = query.strip().rstrip(" .")
    lower = text.lower()
    order = "left_to_right"
    rank = 0
    if "rightmost" in lower:
        order = "right_to_left"
    elif "from right to left" in lower:
        order = "right_to_left"
    elif "closest" in lower or "nearest" in lower:
        order = "near_to_far"
    elif "farthest" in lower or "furthest" in lower:
        order = "far_to_near"

    for word, value in _ORDINAL_RANK.items():
        if re.search(rf"\b{word}\b", lower):
            rank = value
            break
    if re.search(r"\blast\b", lower):
        rank = -1

    target = re.sub(r"\bfrom\s+(?:left|right)\s+to\s+(?:left|right)\b,?\s*", "", text, flags=re.IGNORECASE)
    target = re.sub(
        r"\b(?:first|second|third|fourth|fifth|sixth|seventh|eighth|ninth|tenth|"
        r"last|leftmost|rightmost|closest|nearest|farthest|furthest)\b",
        "",
        target,
        flags=re.IGNORECASE,
    )
    target = re.sub(r"\b(?:the|a|an)\b", " ", target, flags=re.IGNORECASE)
    target = re.sub(r"\s+", " ", target).strip(" ,.")
    return OrdinalPlan(target or text, order, rank)


def parse_candidate_boxes(text: str) -> list[list[float]]:
    boxes: list[list[float]] = []
    for match in _BOX.finditer(text or ""):
        box = normalize_bbox([float(item) for item in match.groups()])
        if box is not None:
            boxes.append(box)
    return boxes


def _grid_box(box: list[float]) -> list[int]:
    return [round(value * 1000) for value in box]


def build_selector_prompt(query: str, candidates: list[list[float]]) -> str:
    rows = "\n".join(f"{index}. {_grid_box(box)}" for index, box in enumerate(candidates, 1))
    return (
        "Select the candidate box that best satisfies the original visual grounding query.\n"
        f"Query: {query.strip()}\n"
        "Candidate boxes are on the 0-1000 image grid:\n"
        f"{rows}\n\n"
        'Return only JSON: {"candidate": N, "bbox_2d": [x1, y1, x2, y2]}'
    )


def choose_candidate_from_text(text: str, candidates: list[list[float]]) -> list[float] | None:
    if not candidates:
        return None
    for value in _json_values(text):
        if isinstance(value, dict):
            candidate = value.get("candidate")
            if isinstance(candidate, int) and 1 <= candidate <= len(candidates):
                return candidates[candidate - 1]
            bbox = value.get("bbox_2d") or value.get("bbox")
            if isinstance(bbox, list):
                normalized = normalize_bbox(bbox)
                if normalized is not None:
                    return normalized
    match = re.search(r"\bcandidate\b\D*(\d+)", text or "", re.IGNORECASE)
    if match:
        index = int(match.group(1))
        if 1 <= index <= len(candidates):
            return candidates[index - 1]
    return parse_bbox(text or "")


def pick_by_plan(plan: OrdinalPlan, candidates: list[list[float]]) -> list[float] | None:
    if not candidates:
        return None
    if plan.order == "right_to_left":
        ordered = sorted(candidates, key=lambda box: (box[0] + box[2]) / 2, reverse=True)
    elif plan.order == "near_to_far":
        ordered = sorted(candidates, key=_area, reverse=True)
    elif plan.order == "far_to_near":
        ordered = sorted(candidates, key=_area)
    else:
        ordered = sorted(candidates, key=lambda box: (box[0] + box[2]) / 2)
    index = plan.rank if plan.rank >= 0 else len(ordered) - 1
    return ordered[index] if 0 <= index < len(ordered) else None


def _area(box: list[float]) -> float:
    return max(0.0, box[2] - box[0]) * max(0.0, box[3] - box[1])


def _json_values(text: str) -> list[Any]:
    decoder = json.JSONDecoder()
    values: list[Any] = []
    for index, char in enumerate(text or ""):
        if char not in "[{":
            continue
        try:
            value, _ = decoder.raw_decode(text[index:])
        except json.JSONDecodeError:
            continue
        values.append(value)
    return values
