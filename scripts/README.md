# Release Scripts

This folder contains the shared implementation used by the public inference and evaluation entrypoints. It is intentionally slim: only the code needed to run MeasL-Bench-V1 is kept in the release copy.

## Included Pipeline Files

| File | Purpose |
|---|---|
| `test_raw_eval_pipeline_opt/pipeline.py` | Top-level run command used by `eval/run_infer_and_eval.sh`. |
| `test_raw_eval_pipeline_opt/infer.py` | Model inference utilities. |
| `test_raw_eval_pipeline_opt/eval.py` | Metric computation helpers. |
| `test_raw_eval_pipeline_opt/eval_offline.py` | Offline evaluation entrypoint. |
| `test_raw_eval_pipeline_opt/config.py` | Shared configuration defaults. |
| `test_raw_eval_pipeline_opt/paths.py` | Path and output helpers. |

## Public Entry Point

Most users should not call these files directly. Use the release wrapper instead:

```bash
MODEL_SIZE=2b CUDA_VISIBLE_DEVICES=0 bash eval/run_infer_and_eval.sh
```

The wrapper validates dataset labels, resolves release-local paths, selects the correct adapter, and writes outputs under `eval/output_benchmark/`.

## What Was Pruned

Generated proxy outputs, scratch scripts, dataset conversion utilities, RGB-only experiments, upload helpers, and local environment checks are not required by the packaged benchmark and have been removed from this release copy.

## Related Docs

- Main release README: [`../README.md`](../README.md)
- Evaluation wrapper: [`../eval/README.md`](../eval/README.md)
- Benchmark dataset card: [`../eval_data/README.md`](../eval_data/README.md)
