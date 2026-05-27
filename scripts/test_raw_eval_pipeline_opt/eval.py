from __future__ import annotations

import argparse
import importlib.util
from pathlib import Path

from .config import PipelineConfig
from .paths import RunPaths, build_run_name, load_latest_run


def _load_legacy_eval_module():
    legacy_path = Path(__file__).resolve().parent / 'eval_offline.py'
    spec = importlib.util.spec_from_file_location('raw_eval_legacy', legacy_path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f'Unable to load legacy eval module: {legacy_path}')
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def build_eval_args(pred_file: Path, output_dir: Path, cli_args) -> argparse.Namespace:
    return argparse.Namespace(
        pred_file=str(pred_file),
        dataset_file=cli_args.dataset_file,
        output_dir=str(output_dir),
        fuzzy_threshold=cli_args.fuzzy_threshold,
        judge_url=cli_args.judge_url,
        judge_model=cli_args.judge_model,
        judge_api_key=cli_args.judge_api_key,
        judge_max_workers=cli_args.judge_max_workers,
        judge_max_tokens=cli_args.judge_max_tokens,
        judge_max_retries=cli_args.judge_max_retries,
        resume=cli_args.resume,
    )


def resolve_run_paths(cli_args) -> RunPaths:
    if getattr(cli_args, 'run_dir', None):
        run_dir = Path(cli_args.run_dir)
        return RunPaths(output_root=run_dir.parent, run_name=run_dir.name)

    if getattr(cli_args, 'pred_file', None):
        pred_file = Path(cli_args.pred_file)
        return RunPaths(output_root=pred_file.parent.parent, run_name=pred_file.parent.name)

    if getattr(cli_args, 'use_latest', False):
        latest = load_latest_run(Path(cli_args.output_root))
        if not latest:
            raise FileNotFoundError(f'No latest_run.json found under: {cli_args.output_root}')
        run_dir = Path(latest['run_dir'])
        return RunPaths(output_root=run_dir.parent, run_name=run_dir.name)

    run_name = build_run_name(cli_args.model, cli_args.adapter, cli_args.dataset_file)
    return RunPaths(output_root=Path(cli_args.output_root), run_name=run_name)


def run_eval(cli_args) -> dict:
    paths = resolve_run_paths(cli_args)
    pred_file = paths.pred_file
    if not pred_file.exists():
        raise FileNotFoundError(f'Prediction file not found: {pred_file}')

    legacy = _load_legacy_eval_module()
    eval_args = build_eval_args(pred_file=pred_file, output_dir=paths.eval_dir, cli_args=cli_args)

    judge_model = None
    if eval_args.judge_url:
        judge_model = eval_args.judge_model or legacy.detect_judge_model(eval_args.judge_url, eval_args.judge_api_key)
        if judge_model:
            print(f'Judge 模型: {judge_model}')
        else:
            print('Judge API 无法检测模型，跳过 Judge')
            eval_args.judge_url = None

    existing_judge = legacy.load_resume_results(eval_args.resume)
    if existing_judge:
        print(f'加载 Resume judge 结果: {len(existing_judge)} 条')

    dataset_items = legacy.load_json_or_jsonl(Path(eval_args.dataset_file))
    pred_items = legacy.load_json_or_jsonl(pred_file)

    print('=' * 60)
    print('离线评测（优化入口）')
    print('=' * 60)
    print(f'run_dir:   {paths.run_dir}')
    print(f'pred_file: {pred_file}')
    print(f'dataset:   {eval_args.dataset_file}')
    print(f'output:    {paths.eval_dir}')
    print(f'judge_url: {eval_args.judge_url or "未启用"}')
    print()

    report = legacy.evaluate_single_file(
        dataset_name=Path(eval_args.dataset_file).stem,
        dataset_items=dataset_items,
        pred_items=pred_items,
        fuzzy_threshold=eval_args.fuzzy_threshold,
        judge_url=eval_args.judge_url,
        judge_model=judge_model,
        judge_api_key=eval_args.judge_api_key,
        judge_max_workers=eval_args.judge_max_workers,
        judge_max_tokens=eval_args.judge_max_tokens,
        judge_max_retries=eval_args.judge_max_retries,
        existing_judge=existing_judge,
    )
    report['pred_file'] = str(pred_file)
    report['dataset_file'] = str(eval_args.dataset_file)
    print(paths)
    legacy.save_report(report, paths.eval_dir, eval_args)
    return report


def build_config_for_eval(cli_args) -> PipelineConfig:
    return PipelineConfig(
        model=cli_args.model,
        dataset_file=cli_args.dataset_file,
        output_root=cli_args.output_root,
        adapter=cli_args.adapter,
        swift_bin=cli_args.swift_bin,
        max_new_tokens=cli_args.max_new_tokens,
        cuda_visible_devices=cli_args.cuda_visible_devices,
    )
