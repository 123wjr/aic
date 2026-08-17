#!/usr/bin/env python3
"""Ordinal workflow: LocateAnything multi-box candidates, Qwen final selection."""

from __future__ import annotations

import argparse
import logging
import time
from pathlib import Path

from grounding.backends import LocateAnythingBackend, LocateAnythingConfig, QwenBackend, QwenConfig
from grounding.checkpoint import DEFAULT_BOX, append_jsonl, atomic_dump
from grounding.data import load_query_groups
from grounding.ordinal_workflow import (
    build_selector_prompt,
    build_plan_prompt,
    choose_candidate_from_text,
    parse_ordinal_plan,
    parse_candidate_boxes,
    pick_by_plan,
    plan_ordinal_query,
)
from grounding.prompts import build_prompt
from grounding.query_router import route_query
from grounding.types import InferenceUnit


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="LA 多框候选 -> Qwen 选择 -> ordinal 单框")
    parser.add_argument("--qwen_model", required=True)
    parser.add_argument("--locateanything_model", required=True)
    parser.add_argument("--data_dir", type=Path, required=True)
    parser.add_argument("--query_file", type=Path, default=None)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--raw_output", type=Path, default=None)
    parser.add_argument("--image_field", default="visible")
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--qwen_dtype", choices=("auto", "bfloat16", "float16"), default="auto")
    parser.add_argument("--qwen_max_new_tokens", type=int, default=128)
    parser.add_argument("--locateanything_max_new_tokens", type=int, default=2048)
    parser.add_argument("--attn", choices=("sdpa", "eager", "magi", "la_flash"), default="sdpa")
    parser.add_argument("--vision_attn", default="auto")
    parser.add_argument("--scheduler", choices=("eager", "hold_ar", "ar_first", "pipeline", "adaptive"), default="eager")
    parser.add_argument("--group_size", type=int, default=0)
    parser.add_argument("--feature_cache_size", type=int, default=1)
    return parser


def main() -> int:
    args = build_parser().parse_args()
    if args.limit < 0:
        raise SystemExit("--limit 不能为负数")
    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")

    output = args.output.resolve()
    raw = (args.raw_output or output.with_suffix(".raw.jsonl")).resolve()
    groups, _ = load_query_groups(args.data_dir.resolve(), args.query_file.resolve() if args.query_file else None, image_field=args.image_field)
    records = [record for group in groups for record in group.records if route_query(record.query).workflow == "ordinal"]
    if args.limit:
        records = records[: args.limit]
    logging.info("ordinal workflow records=%d", len(records))

    la = LocateAnythingBackend(
        LocateAnythingConfig(
            args.locateanything_model,
            args.attn,
            args.vision_attn,
            args.scheduler,
            args.group_size,
            args.locateanything_max_new_tokens,
            feature_cache_size=args.feature_cache_size,
        )
    )
    qwen = QwenBackend(QwenConfig(args.qwen_model, args.qwen_dtype, args.qwen_max_new_tokens))

    results = {}
    for index, record in enumerate(records, 1):
        started = time.time()
        plan_prompt = build_plan_prompt(record.query)
        plan_text = qwen.infer([InferenceUnit(record.image_key, record.image_path, (record,), (plan_prompt,))]).texts[0]
        try:
            plan = parse_ordinal_plan(plan_text)
            plan_source = "qwen_plan"
        except ValueError:
            plan = plan_ordinal_query(record.query)
            plan_source = "regex_fallback"

        la_prompt = ""
        la_text = ""
        candidates = []
        qwen_prompt = ""
        qwen_text = ""
        selected_source = "default_box"
        # LA remains the candidate generator even for complex ordinal relations.
        # Qwen's plan preserves the relation; Qwen then selects among LA candidates.
        la_prompt = build_prompt(plan.target, "locateanything_multi")
        la_text = la.infer([InferenceUnit(record.image_key, record.image_path, (record,), (la_prompt,))]).texts[0]
        candidates = parse_candidate_boxes(la_text)
        qwen_prompt = build_selector_prompt(record.query, candidates) if candidates else build_prompt(record.query, "qwen_json")
        qwen_text = qwen.infer([InferenceUnit(record.image_key, record.image_path, (record,), (qwen_prompt,))]).texts[0]
        selected_by_qwen = choose_candidate_from_text(qwen_text, candidates)
        selected = selected_by_qwen or pick_by_plan(plan, candidates) or choose_candidate_from_text(qwen_text, []) or DEFAULT_BOX.copy()
        selected_source = "qwen_select" if selected_by_qwen else "plan_fallback" if candidates else "qwen_direct" if selected != DEFAULT_BOX else "default_box"
        results[record.query_id] = {**record.source, "bbox": selected}
        append_jsonl(
            raw,
            {
                "query_id": record.query_id,
                "image_key": record.image_key,
                "query": record.query,
                "plan": plan.__dict__,
                "plan_prompt": plan_prompt,
                "plan_raw_output": plan_text,
                "plan_source": plan_source,
                "la_prompt": la_prompt,
                "la_raw_output": la_text,
                "candidates": candidates,
                "qwen_prompt": qwen_prompt,
                "qwen_raw_output": qwen_text,
                "bbox": selected,
                "selected_source": selected_source,
                "elapsed_seconds": round(time.time() - started, 6),
            },
        )
        if index % 10 == 0 or index == len(records):
            logging.info("processed %d/%d", index, len(records))

    atomic_dump(results, output)
    logging.info("written %d predictions to %s", len(results), output)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
