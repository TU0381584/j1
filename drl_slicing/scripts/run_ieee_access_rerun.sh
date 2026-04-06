#!/usr/bin/env bash
set -euo pipefail

BASE_DIR="$(cd "$(dirname "$0")/.." && pwd)"
cd "$BASE_DIR"

OUT_DIR="${1:-results/ieee_access_rerun}"
HORIZON_STEPS="${HORIZON_STEPS:-1000}"

if [[ $# -gt 1 ]]; then
  SEEDS=("${@:2}")
else
  SEEDS=(42 123 456 789 999 1001 1002 1003 1004 1005)
fi

echo "Running IEEE Access short rerun package"
echo "  Output dir : $OUT_DIR"
echo "  Horizon    : $HORIZON_STEPS"
echo "  Seeds      : ${SEEDS[*]}"

PYTHONPATH=. python3 scripts/train_drl.py \
  --config configs/rfsim_dqn_training.yaml \
  --controllers dqn_train a2c_train rule_based static random \
  --seeds "${SEEDS[@]}" \
  --horizon-steps "$HORIZON_STEPS" \
  --step-seconds 0 \
  --output-dir "$OUT_DIR" \
  --continue-on-error \
  --compare

echo "Done. Results are in: $OUT_DIR"
