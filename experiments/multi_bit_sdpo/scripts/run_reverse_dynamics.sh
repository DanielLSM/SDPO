#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)"
cd "$ROOT_DIR"

python -m experiments.multi_bit_sdpo.runner \
  --experiment reverse_dynamics \
  --instance Hard-2 \
  --divergence reverse \
  --proposal-kind epsilon_greedy \
  --epsilon 0.05 \
  --alpha 1.0 \
  --beta 0.0 \
  --iterations 500 \
  --seeds 0 1 2 "$@"
