#!/usr/bin/env python3
from __future__ import annotations

import argparse
import os
import subprocess
import sys

from benchmark_pipeline import (
    MODEL_SPECS,
    build_common_pipeline_args,
    collect_missing_images,
    command_path,
    format_counts,
    load_jsonl,
    prepare_dataset,
    release_root,
    shell_join_redacted,
)


DATASET_DEFAULTS = {
    'raw': 'eval_data/test-raw-measl-bench.jsonl',
    'rgb': 'eval_data/test-rgb-measl-bench.jsonl',
}
DEFAULT_OUTPUT_ROOT = 'eval/output_benchmark'


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description='Run benchmark-renamed inference followed by offline evaluation.')
    parser.add_argument(
        '--model-size',
        choices=['all', *sorted(MODEL_SPECS)],
        default=os.environ.get('MODEL_SIZE', 'all'))
    parser.add_argument('--dataset', choices=sorted(DATASET_DEFAULTS), default=os.environ.get('DATASET', 'raw'))
    parser.add_argument('--dataset-file', default=os.environ.get('DATASET_FILE'))
    parser.add_argument('--prepared-dataset-file', default=os.environ.get('PREPARED_DATASET_FILE'))
    parser.add_argument('--output-root', default=os.environ.get('OUTPUT_ROOT', DEFAULT_OUTPUT_ROOT))
    parser.add_argument('--image-root', default=os.environ.get('IMAGE_ROOT'),
                        help='Optional root containing benchmark images for an external dataset.')
    parser.add_argument('--skip-image-check', action='store_true')
    parser.add_argument('--swift-bin', default=os.environ.get('SWIFT_BIN', 'swift'))
    parser.add_argument('--python-bin', default=os.environ.get('PYTHON_BIN', sys.executable))
    parser.add_argument('--cuda-visible-devices', default=os.environ.get('CUDA_VISIBLE_DEVICES'))
    parser.add_argument('--max-new-tokens', type=int, default=int(os.environ.get('MAX_NEW_TOKENS', '8192')))
    parser.add_argument('--max-batch-size', type=int, default=None)
    parser.add_argument('--infer-backend', choices=['transformers', 'vllm'],
                        default=os.environ.get('INFER_BACKEND', 'transformers'))
    parser.add_argument('--vllm-gpu-memory-utilization', type=float,
                        default=float(os.environ.get('VLLM_GPU_MEMORY_UTILIZATION', '0.9')))
    parser.add_argument('--vllm-max-model-len', type=int, default=int(os.environ.get('VLLM_MAX_MODEL_LEN', '8192')))
    parser.add_argument('--fuzzy-threshold', type=float, default=float(os.environ.get('FUZZY_THRESHOLD', '0.8')))
    parser.add_argument('--judge-url', default=os.environ.get('JUDGE_URL'))
    parser.add_argument('--judge-model', default=os.environ.get('JUDGE_MODEL'))
    parser.add_argument('--judge-api-key', default=os.environ.get('JUDGE_API_KEY'))
    parser.add_argument('--judge-max-workers', type=int, default=int(os.environ.get('JUDGE_MAX_WORKERS', '32')))
    parser.add_argument('--judge-max-tokens', type=int, default=int(os.environ.get('JUDGE_MAX_TOKENS', '10240')))
    parser.add_argument('--judge-max-retries', type=int, default=int(os.environ.get('JUDGE_MAX_RETRIES', '5')))
    parser.add_argument('--resume', default=os.environ.get('RESUME'))
    parser.add_argument('--dry-run', action='store_true')
    return parser.parse_args()


def model_sizes(requested: str) -> list[str]:
    if requested == 'all':
        return ['2b', '4b', '8b']
    return [requested]


def default_prepared_dataset_file(dataset_file: str, image_root: str | None) -> str | None:
    if not image_root:
        return None
    stem = os.path.splitext(os.path.basename(dataset_file))[0]
    return f'eval/prepared_data/{stem}_with_image_root.jsonl'


def main() -> None:
    args = parse_args()
    root = release_root(__file__)
    dataset_file = args.dataset_file or DATASET_DEFAULTS[args.dataset]
    prepared_file = args.prepared_dataset_file or default_prepared_dataset_file(dataset_file, args.image_root)
    sizes = model_sizes(args.model_size)

    if args.resume and len(sizes) != 1:
        raise ValueError('--resume can only be used with one --model-size, not --model-size all')

    prepared = prepare_dataset(
        root=root,
        dataset_file=dataset_file,
        output_file=prepared_file,
        image_root=args.image_root,
        require_all_categories=True,
    )
    items = load_jsonl(prepared.dataset_file)
    missing_images = collect_missing_images(items, root)
    if missing_images:
        message = (
            'Missing image files. The packaged benchmark expects images under eval_data/image/. '
            'If using an external dataset, pass --image-root /path/to/image/root. Examples: ' + '; '.join(missing_images)
        )
        if args.skip_image_check or args.dry_run:
            print(f'WARNING: {message}', file=sys.stderr)
        else:
            raise FileNotFoundError(message)

    dataset_arg = command_path(root, prepared.dataset_file)
    output_root_arg = command_path(root, args.output_root)

    print(f'Release root: {root}')
    print(f'Dataset: {dataset_arg}')
    print(f'Rows: {prepared.rows}')
    print(f'Benchmark categories: {format_counts(prepared.counts)}')
    print(f'Output root: {output_root_arg}')
    print(f'Model sizes: {", ".join(sizes)}')

    for size in sizes:
        spec = MODEL_SPECS[size]
        batch_size = args.max_batch_size if args.max_batch_size is not None else spec.default_batch_size
        command = [
            args.python_bin,
            'scripts/test_raw_eval_pipeline_opt/pipeline.py',
            'run',
        ]
        command.extend(build_common_pipeline_args(
            swift_bin=args.swift_bin,
            model=spec.model,
            adapter=command_path(root, spec.adapter),
            dataset_file=dataset_arg,
            output_root=output_root_arg,
            max_new_tokens=args.max_new_tokens,
            max_batch_size=batch_size,
            infer_backend=args.infer_backend,
            cuda_visible_devices=args.cuda_visible_devices,
            vllm_gpu_memory_utilization=args.vllm_gpu_memory_utilization,
            vllm_max_model_len=args.vllm_max_model_len,
        ))
        command.extend(['--fuzzy-threshold', str(args.fuzzy_threshold)])
        if args.judge_url:
            command.extend(['--judge-url', args.judge_url])
        if args.judge_model:
            command.extend(['--judge-model', args.judge_model])
        command.extend(['--judge-max-workers', str(args.judge_max_workers)])
        command.extend(['--judge-max-tokens', str(args.judge_max_tokens)])
        command.extend(['--judge-max-retries', str(args.judge_max_retries)])
        if args.resume:
            command.extend(['--resume', args.resume])

        print()
        print(f'[{size}] {spec.model}')
        print(f'Command: {shell_join_redacted(command)}')

        if not args.dry_run:
            child_env = os.environ.copy()
            if args.judge_api_key:
                child_env['JUDGE_API_KEY'] = args.judge_api_key
            subprocess.run(command, cwd=root, check=True, env=child_env)


if __name__ == '__main__':
    main()
