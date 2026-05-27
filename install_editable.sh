#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

python -m pip install -e "${ROOT}/libs/datasets-3.6.0"
python -m pip install -e "${ROOT}/libs/qwen-vl-utils"
python -m pip install -e "${ROOT}/libs/transformers-4.57.3"
python -m pip install -e "${ROOT}"

python -m pip list -e
