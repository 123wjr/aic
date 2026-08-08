#!/usr/bin/env python3
"""校验并生成只包含 predictions.json 的提交包。"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from zipfile import ZIP_DEFLATED, ZipFile

from grounding.postprocess import validate_submission


def main() -> int:
    parser = argparse.ArgumentParser(description="生成 submission.zip")
    parser.add_argument("--predictions", type=Path, required=True)
    parser.add_argument("--queries", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    predictions = json.loads(args.predictions.read_text(encoding="utf-8"))
    queries = json.loads(args.queries.read_text(encoding="utf-8"))
    errors = validate_submission(predictions, queries)
    if errors:
        raise SystemExit("拒绝打包:\n" + "\n".join(f"- {error}" for error in errors[:20]))
    args.output.parent.mkdir(parents=True, exist_ok=True)
    payload = json.dumps(predictions, ensure_ascii=False, separators=(",", ":")).encode("utf-8")
    with ZipFile(args.output, "w", ZIP_DEFLATED) as archive:
        archive.writestr("predictions.json", payload)
    print(f"提交包已生成: {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
