#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)"
cd "$ROOT_DIR"

python -m experiments.multi_bit_sdpo.runner \
  --experiment informativeness_sweep \
  --instance Easy-2 \
  --divergence forward \
  --proposal-kind epsilon_greedy \
  --epsilon 0.1 \
  --alpha 0.9 \
  --beta 0.1 \
  --iterations 300 \
  --seeds 0 1 2 "$@"
