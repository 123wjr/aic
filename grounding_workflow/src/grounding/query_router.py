"""高精度 query workflow 路由。

路由只识别会改变执行策略的明显信号；无法确定时交给默认模型路径。
"""

from __future__ import annotations

import re
from dataclasses import dataclass


WORKFLOWS = ("default", "multi_instance", "ordinal", "text_sign", "relation")

_PRIORITY = ("text_sign", "ordinal", "multi_instance", "relation")

_TRIGGERS: dict[str, tuple[tuple[str, re.Pattern[str]], ...]] = {
    "text_sign": tuple(
        (word, re.compile(rf"\b{re.escape(word)}\b", re.IGNORECASE))
        for word in (
            "traffic signs",
            "traffic sign",
            "text",
            "letters",
            "letter",
            "words",
            "word",
            "numbers",
            "number",
            "logos",
            "logo",
            "signs",
            "sign",
            "printed",
            "written",
        )
    ),
    "ordinal": tuple(
        (word, re.compile(rf"\b{re.escape(word)}\b", re.IGNORECASE))
        for word in (
            "first",
            "second",
            "third",
            "fourth",
            "fifth",
            "sixth",
            "seventh",
            "eighth",
            "ninth",
            "tenth",
            "last",
            "leftmost",
            "rightmost",
            "closest",
            "nearest",
            "farthest",
            "furthest",
        )
    )
    + (
        ("from left to right", re.compile(r"\bfrom\s+left\s+to\s+right\b", re.IGNORECASE)),
        ("from right to left", re.compile(r"\bfrom\s+right\s+to\s+left\b", re.IGNORECASE)),
    ),
    "multi_instance": tuple(
        (word, re.compile(rf"\b{re.escape(word)}\b", re.IGNORECASE))
        for word in (
            "all",
            "both",
            "multiple",
            "several",
            "group of",
            "row of",
            "line of",
            "cluster of",
            "pair of",
            "two",
            "three",
            "four",
            "five",
            "six",
            "seven",
            "eight",
            "nine",
            "ten",
        )
    )
    + (
        (
            "plural noun",
            re.compile(
                r"\b(?:people|persons|men|women|boys|girls|cars|vehicles|umbrellas|planters|"
                r"bicycles|scooters|motorcycles|lamps|posts|signs|letters|chairs|tables)\b",
                re.IGNORECASE,
            ),
        ),
    ),
    "relation": tuple(
        (word, re.compile(rf"\b{re.escape(word)}\b", re.IGNORECASE))
        for word in (
            "beside",
            "behind",
            "in front of",
            "above",
            "below",
            "under",
            "over",
            "next to",
            "between",
            "inside",
            "outside",
            "to the left of",
            "to the right of",
            "left of",
            "right of",
        )
    ),
}


@dataclass(frozen=True)
class QueryRoute:
    workflow: str
    tags: tuple[str, ...]
    reason: str


def route_query(query: str) -> QueryRoute:
    """Classify only explicit workflow cues; ambiguous queries use default."""

    if not isinstance(query, str) or not query.strip():
        raise ValueError("query must be a non-empty string")
    matches = {
        workflow: tuple(name for name, pattern in triggers if pattern.search(query))
        for workflow, triggers in _TRIGGERS.items()
    }
    if re.search(r"\bnumber\s+of\b", query, re.IGNORECASE):
        matches["text_sign"] = tuple(tag for tag in matches["text_sign"] if tag != "number")
    if not matches["multi_instance"] and _looks_plural_target(query):
        matches["multi_instance"] = ("plural noun",)
    workflow = next((name for name in _PRIORITY if matches[name]), "default")
    tags = matches.get(workflow, ())
    reason = (
        f"matched {workflow} trigger: {tags[0]}"
        if workflow != "default" and tags
        else "no_explicit_workflow_cue"
    )
    return QueryRoute(workflow, tags, reason)


def classify_query(query: str) -> QueryRoute:
    return route_query(query)


_HEAD_BOUNDARY = re.compile(
    r"\b(?:beside|behind|above|below|under|over|next\s+to|between|inside|outside|"
    r"with|wearing|holding|carrying|near|in|on|at|from)\b",
    re.IGNORECASE,
)
_IRREGULAR_PLURALS = {"people", "persons", "men", "women", "children", "geese", "feet", "teeth"}
_SINGULAR_S_ENDINGS = ("ss", "us", "is", "ous")
_SINGULAR_S_WORDS = {"lens", "news", "series", "species", "means", "headquarters"}


def _looks_plural_target(query: str) -> bool:
    """Recognize a plural head noun while avoiding common singular -s words."""

    head = _HEAD_BOUNDARY.split(query, maxsplit=1)[0]
    words = re.findall(r"[a-z]+", head.lower())
    if not words:
        return False
    noun = words[-1]
    return noun in _IRREGULAR_PLURALS or (
        noun.endswith("s")
        and len(noun) > 3
        and not noun.endswith(_SINGULAR_S_ENDINGS)
        and noun not in _SINGULAR_S_WORDS
    )
