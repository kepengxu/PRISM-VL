#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
RELEASE_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"
cd "${RELEASE_ROOT}"

if [[ -n "${CONDA_ENV_NAME:-}" ]]; then
  # Optional conda activation for local environments.
  source "$(conda info --base)/etc/profile.d/conda.sh"
  conda activate "${CONDA_ENV_NAME}"
fi

export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"
export IMAGE_MAX_TOKEN_NUM="${IMAGE_MAX_TOKEN_NUM:-1024}"
export VIDEO_MAX_TOKEN_NUM="${VIDEO_MAX_TOKEN_NUM:-32}"
export FPS_MAX_FRAMES="${FPS_MAX_FRAMES:-16}"
export NPROC_PER_NODE="${NPROC_PER_NODE:-2}"
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0,1}"
export MASTER_PORT="${MASTER_PORT:-29500}"
export SWIFT_DEBUG_METADATA="${SWIFT_DEBUG_METADATA:-1}"

SWIFT_BIN="${SWIFT_BIN:-swift}"
"${SWIFT_BIN}" sft --config "${SCRIPT_DIR}/sft_qwen3_vl_2b_prsimvl_v1.yaml"
