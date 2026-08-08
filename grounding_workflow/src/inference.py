#!/usr/bin/env python3
"""统一推理入口；--backend 只选择模型差异。"""

from __future__ import annotations

import argparse
import logging
from pathlib import Path

from grounding.backends import LocateAnythingBackend, LocateAnythingConfig, QwenBackend, QwenConfig
from grounding.prompts import load_provider
from grounding.runner import RunConfig, run

ROOT = Path(__file__).resolve().parents[1]


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="RGB/红外/深度查询目标定位统一工作流")
    parser.add_argument("--backend", choices=("qwen", "locateanything"), default="qwen")
    parser.add_argument("--model", required=True, help="本地模型目录或 HuggingFace ID")
    parser.add_argument("--data_dir", type=Path, required=True)
    parser.add_argument("--query_file", type=Path, default=None)
    parser.add_argument("--image_field", default="visible")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--partial", type=Path, default=None)
    parser.add_argument("--raw_output", type=Path, default=None)
    parser.add_argument("--prompt_cache", type=Path, default=None)
    parser.add_argument("--prompt_provider", default=None)
    parser.add_argument("--prompt_prefix", default=None)
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--batch_size", type=int, default=4)
    parser.add_argument("--save_every", type=int, default=100)
    parser.add_argument("--shard_index", type=int, default=0, help="多 GPU worker 下的分片编号")
    parser.add_argument("--shard_count", type=int, default=1, help="多 GPU worker 总数")
    parser.add_argument("--no_resume", dest="resume", action="store_false")
    parser.set_defaults(resume=True)
    # Qwen 参数
    parser.add_argument("--dtype", choices=("auto", "bfloat16", "float16"), default="auto")
    parser.add_argument("--max_new_tokens", type=int, default=None)
    parser.add_argument("--min_pixels", type=int, default=None)
    parser.add_argument("--max_pixels", type=int, default=None)
    parser.add_argument("--attn_implementation", choices=("eager", "sdpa", "flash_attention_2"), default=None)
    parser.add_argument("--gpu_memory_limit", default=None)
    # LocateAnything 参数
    parser.add_argument("--attn", choices=("sdpa", "eager", "magi", "la_flash"), default="sdpa")
    parser.add_argument("--vision_attn", default="auto")
    parser.add_argument("--scheduler", choices=("eager", "hold_ar", "ar_first", "pipeline", "adaptive"), default="eager")
    parser.add_argument("--group_size", type=int, default=0)
    parser.add_argument("--feature_cache_size", type=int, default=1)
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--top_p", type=float, default=None)
    parser.add_argument("--top_k", type=int, default=None)
    parser.add_argument("--repetition_penalty", type=float, default=1.1)
    return parser


def main() -> int:
    args = build_parser().parse_args()
    if args.limit < 0 or args.batch_size <= 0 or args.save_every <= 0 or args.shard_count <= 0 or not 0 <= args.shard_index < args.shard_count:
        raise SystemExit("limit 不能为负数，batch_size/save_every/shard_count 必须有效")
    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
    provider = load_provider(args.prompt_provider)
    if args.prompt_prefix is not None:
        provider.prefix = args.prompt_prefix
    if args.backend == "qwen":
        backend = QwenBackend(QwenConfig(args.model, args.dtype, args.max_new_tokens or 128, args.min_pixels, args.max_pixels, args.attn_implementation, args.gpu_memory_limit))
    else:
        backend = LocateAnythingBackend(LocateAnythingConfig(args.model, args.attn, args.vision_attn, args.scheduler, args.group_size, args.max_new_tokens or 2048, args.temperature, args.top_p, args.top_k, args.repetition_penalty, args.feature_cache_size, provider.prefix))
    output = args.output.resolve()
    config = RunConfig(args.data_dir.resolve(), args.query_file.resolve() if args.query_file else None, output, (args.partial or output.with_suffix(".partial.json")).resolve(), (args.raw_output or output.with_suffix(".raw.jsonl")).resolve(), (args.prompt_cache or output.with_suffix(".prompts.jsonl")).resolve(), args.image_field, args.limit, args.batch_size, args.save_every, args.resume, args.shard_index, args.shard_count)
    summary = run(config, provider, backend)
    logging.info("完成: %s", summary)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
