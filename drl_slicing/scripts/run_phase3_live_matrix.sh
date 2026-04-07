#!/usr/bin/env bash
set -euo pipefail

BASE_DIR="$(cd "$(dirname "$0")/.." && pwd)"
cd "$BASE_DIR"

OUT_ROOT="${1:-results/phase3/live_matrix}"
HORIZON_STEPS="${HORIZON_STEPS:-300}"
STEP_SECONDS="${STEP_SECONDS:-1}"
UE_COUNT="${UE_COUNT:-5}"
ACTIVE_ALGO="${ACTIVE_ALGO:-dqn}"
BASELINE="${BASELINE:-rule_based}"
PERMUTATIONS="${PERMUTATIONS:-5000}"

if [[ $# -gt 1 ]]; then
  SEEDS=("${@:2}")
else
  SEEDS=(42 123 456 789 999 1001 1002 1003 1004 1005)
fi

if [[ -n "${CONTROLLERS_OVERRIDE:-}" ]]; then
  read -r -a CONTROLLERS <<<"$CONTROLLERS_OVERRIDE"
else
  CONTROLLERS=(dqn_train a2c_train ppo_train rule_based static random)
fi

SCENARIOS=(
  "eMBB:configs/openran_live_phase3_embb.yaml"
  "mixed_sla:configs/openran_live_phase3_mixed_sla.yaml"
  "burst:configs/openran_live_phase3_burst.yaml"
  "fading:configs/openran_live_phase3_fading.yaml"
  "five_slice:configs/openran_live_phase3_5slice.yaml"
)

echo "Running Phase 3 live matrix"
echo "  Output root : $OUT_ROOT"
echo "  Horizon     : $HORIZON_STEPS"
echo "  Step seconds: $STEP_SECONDS"
echo "  Seeds       : ${SEEDS[*]}"
echo "  Controllers : ${CONTROLLERS[*]}"

if ! docker ps --format '{{.Names}}' | grep -q '^amf$' \
  || ! docker ps --format '{{.Names}}' | grep -q '^smf$' \
  || ! docker ps --format '{{.Names}}' | grep -q '^upf$' \
  || ! docker ps --format '{{.Names}}' | grep -q '^oaignb$' \
  || ! docker ps --format '{{.Names}}' | grep -q '^nr_ue_'; then
  echo "Live OAI stack not detected. Starting it now..."
  bash scripts/run_closed_loop_sim.sh --ue-count "$UE_COUNT" --active-algo "$ACTIVE_ALGO"
fi

for entry in "${SCENARIOS[@]}"; do
  scenario_name="${entry%%:*}"
  scenario_config="${entry#*:}"
  scenario_out="$OUT_ROOT/$scenario_name"

  echo ""
  echo "=== Scenario: $scenario_name ==="
  echo "Config: $scenario_config"
  echo "Output: $scenario_out"

  PYTHONPATH=. python3 scripts/train_drl.py \
    --config "$scenario_config" \
    --controllers "${CONTROLLERS[@]}" \
    --seeds "${SEEDS[@]}" \
    --horizon-steps "$HORIZON_STEPS" \
    --step-seconds "$STEP_SECONDS" \
    --output-dir "$scenario_out" \
    --continue-on-error \
    --compare

done

PYTHONPATH=. python3 scripts/phase3_stats.py \
  --root "$OUT_ROOT" \
  --baseline "$BASELINE" \
  --permutations "$PERMUTATIONS"

echo ""
echo "Phase 3 live matrix completed. Results are in: $OUT_ROOT"
