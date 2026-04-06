#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT=$(cd "$(dirname "$0")/../../.." && pwd)
cd "$REPO_ROOT"

python3 -m experiments.multi_bit_sdpo.llm_rollout \
  --model Qwen/Qwen2.5-0.5B-Instruct \
  --instance Easy-2 \
  --proposal-kind epsilon_greedy \
  --epsilon 0.1 \
  --alpha 0.9 \
  --beta 0.1 \
  --iterations 8 \
  --seeds 0
