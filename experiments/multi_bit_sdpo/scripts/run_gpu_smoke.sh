#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)"
cd "$ROOT_DIR"

python3 -m experiments.multi_bit_sdpo.gpu_smoke \
  --model Qwen/Qwen2.5-0.5B-Instruct \
  --instance Easy-2 \
  --alpha 0.9 \
  --beta 0.1 \
  "$@"
