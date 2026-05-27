# Release Manifest

## Project Metadata

- `README.md`
- `README_CN.md`
- `LICENSE`
- `CODE_OF_CONDUCT.md`
- `CONTRIBUTING.md`
- `CONTRIBUTING_CN.md`
- `.gitignore`
- `.gitattributes`

## Main Package

- `swift/`
- `setup.py`
- `setup.cfg`
- `requirements.txt`
- `requirements/`
- `MANIFEST.in`
- `Makefile`

## Final Training Configs

- `configs/qwen3_vl_150k_llmmeta_vit_proxy/`

## Editable Dependency Snapshots

- `libs/datasets-3.6.0/`
- `libs/qwen-vl-utils/`
- `libs/transformers-4.57.3/`

## Included LoRA Adapters

- `exps/BANALCED_150K_META_VIT_PROXY/output-Qwen3-VL-2B-Instruct/v8-20260421-133546/checkpoint-95000/`
- `exps/BANALCED_150K_META_VIT_PROXY/output-Qwen3-VL-4B-Instruct/v12-20260425-113029/checkpoint-85000/`
- `exps/BANALCED_150K_META_VIT_PROXY/output-Qwen3-VL-8B-Instruct/v2-20260423-205317/checkpoint-95000/`

Only inference-required adapter files are included. Optimizer state, scheduler state, RNG state, trainer state, and training argument binaries are excluded.


## Training Data

- `training_data/README.md`
- `training_data/`
- `training_data/train.jsonl`
- `training_data/images/`

The release-local dataset registry maps both training dataset names to `training_data/train.jsonl`. Training JSONL image paths point to `training_data/images/`, which contains 48,002 copied image files totaling about 94.53 GiB. The merged training JSONL has 152,517 rows after removing 46 strict overlaps with the RAW benchmark.

## Benchmark And Evaluation Assets

- `eval/`
- `inference/`
- `scripts/README.md`
- `scripts/test_raw_eval_pipeline_opt/pipeline.py`
- `scripts/test_raw_eval_pipeline_opt/infer.py`
- `scripts/test_raw_eval_pipeline_opt/eval.py`
- `scripts/test_raw_eval_pipeline_opt/eval_offline.py`
- `scripts/test_raw_eval_pipeline_opt/config.py`
- `scripts/test_raw_eval_pipeline_opt/paths.py`
- `eval_data/README.md`
- `eval_data/test_raw_full_benchmark.jsonl`
- `eval_data/test_rgb_full_benchmark.jsonl`
- `eval_data/images/`

## Release Helper Files

- `install_editable.sh`
- `requirements_editable.txt`
