#!/usr/bin/env python3
"""命令行提交校验器。"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from grounding.postprocess import validate_submission


def main() -> int:
    parser = argparse.ArgumentParser(description="严格校验 predictions.json")
    parser.add_argument("--predictions", type=Path, required=True)
    parser.add_argument("--queries", type=Path, required=True)
    parser.add_argument("--allow_subset", action="store_true")
    args = parser.parse_args()
    with args.predictions.open(encoding="utf-8") as handle:
        predictions = json.load(handle)
    with args.queries.open(encoding="utf-8") as handle:
        queries = json.load(handle)
    errors = validate_submission(predictions, queries, allow_subset=args.allow_subset)
    if errors:
        print(f"校验失败，共 {len(errors)} 项")
        print("\n".join(f"- {error}" for error in errors[:50]))
        return 1
    print("校验通过")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
