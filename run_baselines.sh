#!/usr/bin/env bash
# Phase 1: reply-only vs parent+reply baselines, 5 seeds each.
# Usage: bash run_baselines.sh [model]   (default roberta-base)
#
# Optional environment variables:
#   OUTPUT_DIR    where results are written            (default: outputs)
#   PREV_RESULTS  space-separated dirs of earlier results to reuse instead of retraining
#                 (e.g. a Kaggle input such as /kaggle/input/<notebook>/outputs)
set -euo pipefail

MODEL="${1:-roberta-base}"
OUTPUT_DIR="${OUTPUT_DIR:-outputs}"
PREV_RESULTS="${PREV_RESULTS:-}"
SEEDS=(13 42 87 123 2024)

if [ ! -d data/counter_context ]; then
  mkdir -p data
  git clone --depth 1 https://github.com/xinchenyu/counter_context data/counter_context
fi

for INPUT in target context_target; do
  for SEED in "${SEEDS[@]}"; do
    # shellcheck disable=SC2086
    python baseline.py --model "$MODEL" --input "$INPUT" --seed "$SEED" \
      --output_dir "$OUTPUT_DIR" --prev_results $PREV_RESULTS
  done
done

python aggregate.py --output_dir "$OUTPUT_DIR"
