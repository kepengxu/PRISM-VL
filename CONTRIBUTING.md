# Contributing

This repository is a release artifact for the metadata-aware Qwen3-VL proxy benchmark. Contributions should keep the artifact reproducible and easy to run.

## Good Changes

- Fix broken paths, documentation, or install instructions.
- Improve the benchmark runner without changing default metrics silently.
- Add focused tests for release scripts and data validation.
- Report missing Git LFS assets or corrupted benchmark files.

## Before Opening A PR

```bash
python -m py_compile eval/benchmark_pipeline.py eval/infer_and_eval.py inference/demo_infer.py scripts/test_raw_eval_pipeline_opt/pipeline.py
bash -n eval/run_infer_and_eval.sh
bash -n inference/run_demo_inference.sh
bash eval/run_infer_and_eval.sh --dry-run
bash inference/run_demo_inference.sh --dry-run
```

Do not commit generated evaluation outputs, optimizer states, local logs, or cache directories. The `.gitignore` file already excludes the common cases.
