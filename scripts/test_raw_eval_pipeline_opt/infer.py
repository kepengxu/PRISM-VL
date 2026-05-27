from __future__ import annotations

import os
import subprocess

from .config import PipelineConfig
from .paths import RunPaths, save_latest_run


def get_swift_infer_backend(config: PipelineConfig) -> str:
    return 'vllm' if config.infer_backend == 'vllm' else 'pt'


def build_infer_command(config: PipelineConfig, paths: RunPaths) -> list[str]:
    swift_infer_backend = get_swift_infer_backend(config)
    cmd = [
        config.swift_bin,
        'infer',
        '--model',
        config.model,
        '--stream',
        'false',
        '--val_dataset',
        config.dataset_file,
        '--max_new_tokens',
        str(config.max_new_tokens),
        '--remove_unused_columns',
        'false',
        '--infer_backend',
        swift_infer_backend,
        '--result_path',
        str(paths.pred_file),
        '--max_batch_size',
        str(config.max_batch_size),
    ]
    if config.adapter != 'None':
        cmd.extend(['--adapters', config.adapter])
    if config.infer_backend == 'vllm':
        cmd.extend([
            '--vllm_gpu_memory_utilization',
            str(config.vllm_gpu_memory_utilization),
            '--vllm_max_model_len',
            str(config.vllm_max_model_len),
        ])
    return cmd


def run_infer(config: PipelineConfig) -> RunPaths:
    paths = config.save()
    env = os.environ.copy()
    if config.cuda_visible_devices:
        env['CUDA_VISIBLE_DEVICES'] = config.cuda_visible_devices

    cmd = build_infer_command(config, paths)
    swift_infer_backend = get_swift_infer_backend(config)
    print(f'swift_bin: {config.swift_bin}')
    print(f'model: {config.model}')
    print(f'adapter: {config.adapter}')
    print(f'dataset_file: {config.dataset_file}')
    print(f'output_dir: {paths.run_dir}')
    print(f'pred_file: {paths.pred_file}')
    print(f'max_new_tokens: {config.max_new_tokens}')
    print(f'max_batch_size: {config.max_batch_size}')
    print(f'infer_backend: {config.infer_backend}')
    print(f'swift_infer_backend: {swift_infer_backend}')
    if config.infer_backend == 'vllm':
        print(f'vllm_gpu_memory_utilization: {config.vllm_gpu_memory_utilization}')
        print(f'vllm_max_model_len: {config.vllm_max_model_len}')
    if config.cuda_visible_devices:
        print(f'cuda_visible_devices: {config.cuda_visible_devices}')

    subprocess.run(cmd, check=True, env=env)
    save_latest_run(paths)
    return paths
