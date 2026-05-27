#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

if __package__ in (None, ''):
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
    from scripts.test_raw_eval_pipeline_opt.config import DEFAULT_DATASET_FILE, DEFAULT_OUTPUT_ROOT, DEFAULT_SWIFT_BIN, config_from_args
    from scripts.test_raw_eval_pipeline_opt.eval import run_eval
    from scripts.test_raw_eval_pipeline_opt.infer import run_infer
else:
    from .config import DEFAULT_DATASET_FILE, DEFAULT_OUTPUT_ROOT, DEFAULT_SWIFT_BIN, config_from_args
    from .eval import run_eval
    from .infer import run_infer


def add_shared_run_args(parser: argparse.ArgumentParser) -> None:
    parser.add_argument('--model', required=True, help='模型名，例如 Qwen/Qwen3-VL-2B-Instruct')
    parser.add_argument('--adapter', default='None', help='adapter 路径，不传时使用 None')
    parser.add_argument('--dataset-file', default=DEFAULT_DATASET_FILE, help='测试集文件路径')
    parser.add_argument('--output-root', default=DEFAULT_OUTPUT_ROOT, help='所有运行输出根目录')
    parser.add_argument('--swift-bin', default=DEFAULT_SWIFT_BIN, help='swift 可执行文件路径')
    parser.add_argument('--max-new-tokens', type=int, default=8192, help='infer 的 max_new_tokens')
    parser.add_argument('--cuda-visible-devices', default=None, help='覆盖 CUDA_VISIBLE_DEVICES')
    parser.add_argument('--max-batch-size', '--max_batch_size', dest='max_batch_size', type=int, default=1,
                        help='swift infer 的 max_batch_size')
    parser.add_argument('--infer-backend', choices=['transformers', 'vllm'], default=None,
                        help='opt 包装层的 infer backend；默认 transformers')
    parser.add_argument('--use-vllm', action='store_true', help='兼容旧参数，等价于 --infer-backend vllm')
    parser.add_argument('--vllm-gpu-memory-utilization', type=float, default=0.9, help='vLLM 的 gpu memory utilization')
    parser.add_argument('--vllm-max-model-len', type=int, default=8192, help='vLLM 的 max model len')


def add_eval_only_args(parser: argparse.ArgumentParser) -> None:
    parser.add_argument('--pred-file', default=None, help='直接指定预测文件，优先级高于自动推导')
    parser.add_argument('--run-dir', default=None, help='直接指定运行目录，内部自动定位 pred/eval 路径')
    parser.add_argument('--use-latest', action='store_true', help='读取 output_root/latest_run.json 对应的最近一次运行')
    parser.add_argument('--fuzzy-threshold', type=float, default=0.8, help='模糊匹配阈值')
    parser.add_argument('--judge-url', default=None, help='Judge API base URL')
    parser.add_argument('--judge-model', default=None, help='Judge 模型名')
    parser.add_argument('--judge-api-key', default=os.environ.get('JUDGE_API_KEY'), help='Judge API key')
    parser.add_argument('--judge-max-workers', type=int, default=8, help='Judge 并发数')
    parser.add_argument('--judge-max-tokens', type=int, default=10240, help='Judge 最大生成 token')
    parser.add_argument('--judge-max-retries', type=int, default=5, help='Judge 最大重试次数')
    parser.add_argument('--resume', default=None, help='复用历史 judge 结果 JSON')


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description='Optimized raw infer/eval pipeline')
    subparsers = parser.add_subparsers(dest='command', required=True)

    infer_parser = subparsers.add_parser('infer', help='只执行推理')
    add_shared_run_args(infer_parser)

    eval_parser = subparsers.add_parser('eval', help='只执行评测')
    add_shared_run_args(eval_parser)
    add_eval_only_args(eval_parser)

    run_parser = subparsers.add_parser('run', help='顺序执行 infer + eval')
    add_shared_run_args(run_parser)
    add_eval_only_args(run_parser)

    return parser


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()

    if args.command == 'infer':
        config = config_from_args(args)
        paths = run_infer(config)
        print(f'Infer completed: {paths.pred_file}')
        return

    if args.command == 'eval':
        run_eval(args)
        return

    if args.command == 'run':
        config = config_from_args(args)
        paths = run_infer(config)
        print(f'Infer completed: {paths.pred_file}')
        args.run_dir = str(paths.run_dir)
        args.pred_file = str(paths.pred_file)
        run_eval(args)
        return

    raise ValueError(f'Unsupported command: {args.command}')


if __name__ == '__main__':
    main()
