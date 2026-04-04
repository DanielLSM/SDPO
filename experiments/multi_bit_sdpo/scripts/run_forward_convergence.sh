#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)"
cd "$ROOT_DIR"

python3 -m experiments.multi_bit_sdpo.runner \
  --experiment forward_convergence \
  --instance Easy-2 \
  --divergence forward \
  --proposal-kind epsilon_greedy \
  --epsilon 0.1 \
  --alpha 1.0 \
  --beta 0.0 \
  --iterations 300 \
  --seeds 0 1 2 "$@"
