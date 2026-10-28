#!/usr/bin/env bash
# Phase 1: reply-only vs parent+reply baselines, 5 seeds each.
# Usage: bash run_baselines.sh [model]   (default roberta-base)
set -euo pipefail

MODEL="${1:-roberta-base}"
SEEDS=(13 42 87 123 2024)

if [ ! -d data/counter_context ]; then
  mkdir -p data
  git clone --depth 1 https://github.com/xinchenyu/counter_context data/counter_context
fi

for INPUT in target context_target; do
  for SEED in "${SEEDS[@]}"; do
    python baseline.py --model "$MODEL" --input "$INPUT" --seed "$SEED"
  done
done

python aggregate.py --output_dir outputs
