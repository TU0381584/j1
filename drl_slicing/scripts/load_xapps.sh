#!/usr/bin/env bash
set -euo pipefail

BASE_DIR="$(cd "$(dirname "$0")/.." && pwd)"
cd "$BASE_DIR"

CONFIG="${1:-configs/rfsim_dqn_training.yaml}"
REPEAT="${2:-1}"

mkdir -p results/xapps

PYTHONPATH=. nohup python3 scripts/xapp_service.py \
  --config "$CONFIG" \
  --algorithm dqn \
  --train \
  --repeat "$REPEAT" \
  --name xapp-dqn-train \
  > results/xapps/xapp-dqn.log 2>&1 &
DQN_PID=$!

PYTHONPATH=. nohup python3 scripts/xapp_service.py \
  --config "$CONFIG" \
  --algorithm a2c \
  --train \
  --repeat "$REPEAT" \
  --name xapp-a2c-train \
  > results/xapps/xapp-a2c.log 2>&1 &
A2C_PID=$!

echo "$DQN_PID" > results/xapps/xapp-dqn.pid
echo "$A2C_PID" > results/xapps/xapp-a2c.pid

echo "Started xApps:"
echo "  DQN PID: $DQN_PID"
echo "  A2C PID: $A2C_PID"
echo "Logs: results/xapps/xapp-dqn.log, results/xapps/xapp-a2c.log"
