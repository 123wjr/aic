#!/usr/bin/env python3
"""合并多 GPU worker 的分片预测，并检查 query ID 不重复。"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from grounding.checkpoint import atomic_dump, load_json_object
from grounding.postprocess import validate_submission


def main() -> int:
    parser = argparse.ArgumentParser(description="合并多 GPU 分片预测")
    parser.add_argument("--queries", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("inputs", nargs="+", type=Path)
    args = parser.parse_args()
    queries = json.loads(args.queries.read_text(encoding="utf-8"))
    merged: dict[str, dict] = {}
    for path in args.inputs:
        for query_id, row in load_json_object(path).items():
            if query_id in merged:
                raise SystemExit(f"重复 query ID: {query_id}")
            merged[query_id] = row
    errors = validate_submission(merged, queries, allow_subset=True)
    if errors:
        raise SystemExit("分片结果校验失败:\n" + "\n".join(f"- {error}" for error in errors[:20]))
    atomic_dump(merged, args.output)
    print(f"已合并 {len(merged)} 条分片预测: {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
