#!/usr/bin/env python3
"""筛选适合独立 workflow 的 query。"""

from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from pathlib import Path
from typing import Any

from grounding.query_router import WORKFLOWS, route_query


def filter_records(
    payload: dict[str, Any],
    workflow: str = "all",
    limit: int = 0,
) -> list[dict[str, Any]]:
    """Return compact route records without changing the source query data."""

    if not isinstance(payload, dict):
        raise ValueError("queries 顶层必须是 JSON object")
    if workflow != "all" and workflow not in WORKFLOWS:
        raise ValueError(f"unknown workflow: {workflow}")
    records = routed_records(payload)
    if workflow != "all":
        records = [record for record in records if record["workflow"] == workflow]
    return records[: limit if limit > 0 else None]


def routed_records(payload: dict[str, Any]) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    for query_id, source in payload.items():
        if not isinstance(source, dict) or not isinstance(source.get("query"), str):
            raise ValueError(f"{query_id}: query record 无效")
        route = route_query(source["query"])
        records.append(
            {
                "query_id": query_id,
                "visible": source.get("visible"),
                "query": source["query"],
                "workflow": route.workflow,
                "tags": list(route.tags),
                "reason": route.reason,
            }
        )
    return records


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="筛选 workflow query")
    parser.add_argument("--queries", type=Path, required=True)
    parser.add_argument("--output", type=Path, default=None, help="默认输出 stdout")
    parser.add_argument("--format", choices=("jsonl", "json"), default="jsonl")
    parser.add_argument("--workflow", choices=("all", *WORKFLOWS), default="all")
    parser.add_argument("--limit", type=int, default=0)
    args = parser.parse_args(argv)
    if args.limit < 0:
        raise SystemExit("--limit 不能为负数")

    payload = json.loads(args.queries.read_text(encoding="utf-8"))
    all_records = routed_records(payload)
    counts = Counter(record["workflow"] for record in all_records)
    print(" ".join(f"{workflow}={counts.get(workflow, 0)}" for workflow in WORKFLOWS), file=sys.stderr)
    records = filter_records(payload, args.workflow, args.limit)
    if args.format == "json":
        text = json.dumps(records, ensure_ascii=False, indent=2) + "\n"
    else:
        text = "".join(json.dumps(record, ensure_ascii=False) + "\n" for record in records)
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(text, encoding="utf-8")
    else:
        print(text, end="")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
