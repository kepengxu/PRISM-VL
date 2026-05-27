# Benchmark Inference And Evaluation

This folder is the public evaluation entrypoint for MeasL-Bench-V1. It runs model inference through the shared optimized pipeline, then performs offline metric computation with the benchmark-renamed capability labels.

## What This Wrapper Does

| Step | Action |
|---|---|
| 1 | Resolve the selected benchmark split: Meas.-XYZ (`raw`) or matched RGB (`rgb`). |
| 2 | Validate all `question_type` values against the public MeasL-Bench taxonomy. |
| 3 | Check that referenced images exist under the release-local image layout. |
| 4 | Launch `scripts/test_raw_eval_pipeline_opt/pipeline.py run` with the selected PRSIMVL adapter. |
| 5 | Write inference and offline evaluation outputs under `eval/output_benchmark/`. |

Benchmark labels:

```text
CAG, NG, DSG, HER, LER, STR, GVG, CVR, SRU, MSQ, EAQ, DS, AEI, BVV
```

## Data Contract

The packaged benchmark follows the Hugging Face release layout:

| Split | JSONL | Images |
|---|---|---|
| Meas.-XYZ | `eval_data/test-raw-measl-bench.jsonl` | `eval_data/image/` |
| RGB | `eval_data/test-rgb-measl-bench.jsonl` | `eval_data/image/` |

The JSONL records use release-local paths such as `eval_data/image/measl_bench_raw_000000.png`, so `IMAGE_ROOT` is not needed for the packaged benchmark. Use `IMAGE_ROOT` only when overriding `DATASET_FILE` with an external image tree.

## Common Workflows

Dry-run the exact commands for all three released LoRA checkpoints:

```bash
bash eval/run_infer_and_eval.sh --dry-run
```

Run one model on the Meas.-XYZ benchmark:

```bash
MODEL_SIZE=2b \
CUDA_VISIBLE_DEVICES=0 \
bash eval/run_infer_and_eval.sh
```

Run the matched RGB benchmark:

```bash
DATASET=rgb \
MODEL_SIZE=4b \
CUDA_VISIBLE_DEVICES=0 \
bash eval/run_infer_and_eval.sh
```

Evaluate every released adapter in sequence:

```bash
MODEL_SIZE=all \
CUDA_VISIBLE_DEVICES=0 \
bash eval/run_infer_and_eval.sh
```

Use an external dataset file or image folder:

```bash
DATASET_FILE=/path/to/test.jsonl \
IMAGE_ROOT=/path/to/images \
MODEL_SIZE=2b \
CUDA_VISIBLE_DEVICES=0 \
bash eval/run_infer_and_eval.sh
```

## Optional Judge

Judge-based evaluation is optional. Use an OpenAI-compatible endpoint:

```bash
MODEL_SIZE=2b \
JUDGE_URL=https://openrouter.ai/api/v1 \
JUDGE_MODEL=openai/gpt-5 \
JUDGE_API_KEY=... \
bash eval/run_infer_and_eval.sh
```

## Useful Environment Variables

| Variable | Default | Meaning |
|---|---|---|
| `MODEL_SIZE` | `all` | One of `2b`, `4b`, `8b`, or `all`. |
| `DATASET` | `raw` | `raw` for Meas.-XYZ or `rgb` for matched RGB. |
| `DATASET_FILE` | unset | Override the selected benchmark JSONL. |
| `IMAGE_ROOT` | unset | Root for external images when paths need rewriting. |
| `OUTPUT_ROOT` | `eval/output_benchmark` | Where inference and evaluation outputs are written. |
| `INFER_BACKEND` | `transformers` | Inference backend passed to the shared pipeline. |
| `MAX_NEW_TOKENS` | `8192` | Generation length cap. |
| `JUDGE_URL` | unset | OpenAI-compatible judge endpoint. |
| `JUDGE_MODEL` | unset | Judge model name. |
| `JUDGE_API_KEY` | unset | Judge API key. |

## Outputs

Outputs are grouped by model size under `eval/output_benchmark/`. The shared pipeline writes model predictions, metric summaries, and optional judge artifacts. Re-run with `--dry-run` first if you want to inspect the exact underlying command before spending GPU time.

## Troubleshooting

| Symptom | Likely cause | Fix |
|---|---|---|
| Missing image errors | Dataset JSONL points to a non-packaged image tree. | Set `IMAGE_ROOT=/path/to/images` or restore `eval_data/image/`. |
| Unknown `question_type` | Dataset is not benchmark-renamed. | Use the released MeasL-Bench files or normalize labels before evaluation. |
| Judge request failures | Endpoint, model, or key is missing/mismatched. | Check `JUDGE_URL`, `JUDGE_MODEL`, and `JUDGE_API_KEY`; run without judge to isolate inference. |
| GPU OOM | Model size or batch size is too large for the device. | Use a smaller `MODEL_SIZE` or set `MAX_BATCH_SIZE` through the underlying pipeline args if needed. |

## Related Files

- Main release README: [`../README.md`](../README.md)
- Benchmark dataset card: [`../eval_data/README.md`](../eval_data/README.md)
- Shared implementation: [`../scripts/test_raw_eval_pipeline_opt/`](../scripts/test_raw_eval_pipeline_opt/)
