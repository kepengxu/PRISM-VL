from __future__ import annotations

import json
import os
import sys
from dataclasses import asdict, dataclass
from pathlib import Path

from .paths import RunPaths, build_run_name


DEFAULT_SWIFT_BIN = 'swift'
DEFAULT_DATASET_FILE = 'eval_data/test_raw_full_benchmark.jsonl'
DEFAULT_OUTPUT_ROOT = 'eval/output_benchmark'


@dataclass
class PipelineConfig:
    model: str
    dataset_file: str
    output_root: str
    adapter: str = 'None'
    swift_bin: str = DEFAULT_SWIFT_BIN
    max_new_tokens: int = 8192
    cuda_visible_devices: str | None = None
    max_batch_size: int = 16
    infer_backend: str = 'transformers'
    vllm_gpu_memory_utilization: float = 0.9
    vllm_max_model_len: int = 8192

    def build_paths(self) -> RunPaths:
        output_root = Path(self.output_root)
        run_name = build_run_name(self.model, self.adapter, self.dataset_file)
        return RunPaths(output_root=output_root, run_name=run_name)

    def to_payload(self) -> dict:
        payload = asdict(self)
        paths = self.build_paths()
        payload['use_vllm'] = self.infer_backend == 'vllm'
        payload['swift_infer_backend'] = 'vllm' if self.infer_backend == 'vllm' else 'pt'
        payload['paths'] = {
            'output_root': str(paths.output_root),
            'run_name': paths.run_name,
            'run_dir': str(paths.run_dir),
            'pred_file': str(paths.pred_file),
            'eval_dir': str(paths.eval_dir),
            'run_config_file': str(paths.run_config_file),
            'latest_file': str(paths.latest_file),
        }
        return payload

    def save(self) -> RunPaths:
        paths = self.build_paths()
        paths.run_dir.mkdir(parents=True, exist_ok=True)
        with open(paths.run_config_file, 'w', encoding='utf-8') as f:
            json.dump(self.to_payload(), f, indent=2, ensure_ascii=False)
        return paths


def config_from_args(args) -> PipelineConfig:
    infer_backend = args.infer_backend or ('vllm' if args.use_vllm else 'transformers')
    if infer_backend != 'vllm':
        explicitly_set_vllm_args = []
        if '--vllm-gpu-memory-utilization' in sys.argv:
            explicitly_set_vllm_args.append('--vllm-gpu-memory-utilization')
        if '--vllm-max-model-len' in sys.argv:
            explicitly_set_vllm_args.append('--vllm-max-model-len')
        if explicitly_set_vllm_args:
            raise ValueError(
                f'vLLM-only args {explicitly_set_vllm_args} require --infer-backend vllm (or --use-vllm).')
    return PipelineConfig(
        model=args.model,
        dataset_file=args.dataset_file or os.environ.get('DATASET_FILE', DEFAULT_DATASET_FILE),
        output_root=args.output_root,
        adapter=args.adapter,
        swift_bin=args.swift_bin,
        max_new_tokens=args.max_new_tokens,
        cuda_visible_devices=args.cuda_visible_devices,
        max_batch_size=args.max_batch_size,
        infer_backend=infer_backend,
        vllm_gpu_memory_utilization=args.vllm_gpu_memory_utilization,
        vllm_max_model_len=args.vllm_max_model_len,
    )
