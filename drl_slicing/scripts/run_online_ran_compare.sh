#!/usr/bin/env bash
set -euo pipefail

BASE_DIR="$(cd "$(dirname "$0")/.." && pwd)"
cd "$BASE_DIR"

CONFIG="${1:-configs/online_ran_random_train.yaml}"
OUT_DIR="${2:-results/online_ran_compare}"

# Use 10 seeds by default for publication-grade significance
SEEDS=(42 123 456 789 999 1001 1002 1003 1004 1005)

echo "[1/5] Running DQN only (online randomized RAN)..."
PYTHONPATH=. python3 scripts/train_drl.py \
  --config "$CONFIG" \
  --algorithms dqn \
  --seeds "${SEEDS[@]}" \
  --output-dir "$OUT_DIR" \
  --train

echo "[2/5] Running A2C only (online randomized RAN)..."
PYTHONPATH=. python3 scripts/train_drl.py \
  --config "$CONFIG" \
  --algorithms a2c \
  --seeds "${SEEDS[@]}" \
  --output-dir "$OUT_DIR" \
  --train

echo "[3/5] Creating aggregate comparison..."
PYTHONPATH=. python3 scripts/train_drl.py \
  --output-dir "$OUT_DIR" \
  --compare

echo "[4/5] Plotting comparison bars..."
PYTHONPATH=. python3 scripts/evaluate_and_plot.py \
  --config "$CONFIG" \
  --output-dir "$OUT_DIR" \
  --seeds "${SEEDS[@]}" \
  --skip-run

echo "[5/5] Plotting convergence over time..."
PYTHONPATH=. python3 scripts/plot_convergence.py \
  --output-dir "$OUT_DIR" \
  --mode train

echo "Done. Results: $OUT_DIR"
