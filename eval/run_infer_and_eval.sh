#!/usr/bin/env bash
set -euo pipefail

RELEASE_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${RELEASE_ROOT}"

PYTHON_BIN="${PYTHON_BIN:-python}"

ARGS=(
  --model-size "${MODEL_SIZE:-all}"
  --dataset "${DATASET:-raw}"
  --output-root "${OUTPUT_ROOT:-eval/output_benchmark}"
)

if [[ -n "${DATASET_FILE:-}" ]]; then
  ARGS+=(--dataset-file "${DATASET_FILE}")
fi
if [[ -n "${PREPARED_DATASET_FILE:-}" ]]; then
  ARGS+=(--prepared-dataset-file "${PREPARED_DATASET_FILE}")
fi
if [[ -n "${IMAGE_ROOT:-}" ]]; then
  ARGS+=(--image-root "${IMAGE_ROOT}")
fi
if [[ -n "${SWIFT_BIN:-}" ]]; then
  ARGS+=(--swift-bin "${SWIFT_BIN}")
fi
if [[ -n "${CUDA_VISIBLE_DEVICES:-}" ]]; then
  ARGS+=(--cuda-visible-devices "${CUDA_VISIBLE_DEVICES}")
fi
if [[ -n "${MAX_BATCH_SIZE:-}" ]]; then
  ARGS+=(--max-batch-size "${MAX_BATCH_SIZE}")
fi
if [[ -n "${MAX_NEW_TOKENS:-}" ]]; then
  ARGS+=(--max-new-tokens "${MAX_NEW_TOKENS}")
fi
if [[ -n "${INFER_BACKEND:-}" ]]; then
  ARGS+=(--infer-backend "${INFER_BACKEND}")
fi
if [[ -n "${FUZZY_THRESHOLD:-}" ]]; then
  ARGS+=(--fuzzy-threshold "${FUZZY_THRESHOLD}")
fi
if [[ -n "${JUDGE_URL:-}" ]]; then
  ARGS+=(--judge-url "${JUDGE_URL}")
fi
if [[ -n "${JUDGE_MODEL:-}" ]]; then
  ARGS+=(--judge-model "${JUDGE_MODEL}")
fi
if [[ -n "${JUDGE_MAX_WORKERS:-}" ]]; then
  ARGS+=(--judge-max-workers "${JUDGE_MAX_WORKERS}")
fi
if [[ -n "${RESUME:-}" ]]; then
  ARGS+=(--resume "${RESUME}")
fi

"${PYTHON_BIN}" eval/infer_and_eval.py "${ARGS[@]}" "$@"
