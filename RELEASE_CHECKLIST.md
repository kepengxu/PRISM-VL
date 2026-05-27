# Release Checklist

This checklist tracks public-release readiness of `release_neurips_version_code`.

## Scope

- Package root only includes release artifacts and project metadata.
- Benchmark/eval/demo commands are aligned with README documentation.
- Data and checkpoint paths resolve inside the release directory.

## Verified Items

- [x] `README.md` and `README_CN.md` present consistent quickstart, demo, and eval commands.
- [x] Single-image VQA demo uses packaged sample by default (`inference/run_demo_inference.sh`).
- [x] Full evaluation entrypoint works in dry-run mode (`eval/run_infer_and_eval.sh --dry-run`).
- [x] Benchmark dataset is deduplicated and renamed (`eval_data/test_raw_full_benchmark.jsonl`, 2,183 rows).
- [x] Local image paths are packaged under `eval_data/images/`.
- [x] Released LoRA checkpoints (2B/4B/8B) are included under `exps/BANALCED_150K_META_VIT_PROXY/`.
- [x] Internal cleanup reports and local-only pre-commit config are removed from release root.
- [x] `__pycache__` directories removed from release tree.

## Repro Commands

```bash
cd release_neurips_version_code
bash install_editable.sh
MODEL_SIZE=2b CUDA_VISIBLE_DEVICES=0 bash inference/run_demo_inference.sh --dry-run
MODEL_SIZE=2b CUDA_VISIBLE_DEVICES=0 bash eval/run_infer_and_eval.sh --dry-run
```
