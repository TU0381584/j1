#!/usr/bin/env bash
# Orchestrates a Stage Zero live SAC-LB run against the real OAI testbed,
# modeled on drl_slicing/scripts/run_oran_live_testbed_xapps.sh's bring-up
# sequence (UE fleet generation, Open5GS core, subscriber provisioning, UE
# fleet + traffic profiles, stabilization wait), reusing that script's own
# generic (non-algorithm-specific) helper scripts directly rather than
# duplicating them.
#
# Two required differences from that script, both because this run goes
# through the real E2 loop instead of the file-write rrmPolicy.json path:
#   1. The gNB MUST be OAI's natively-built nr-softmodem (oai_ran, built
#      via cmake_targets/build_oai) -- NOT the oaignb.yaml Docker service,
#      which clones vanilla upstream OAI without the E2_AGENT/protobuf-c
#      code this live loop depends on (confirmed by source inspection).
#   2. Running nr-softmodem requires root (per this repo's own README:
#      `sudo ./nr-softmodem ...`) and this environment has no passwordless
#      sudo available to automate that step -- so this script starts
#      everything it safely can (Open5GS core, UE fleet) and then PAUSES
#      with the exact command to run manually in another terminal, rather
#      than silently failing or attempting to smuggle a sudo password.
#      xApp startup only proceeds after you confirm the gNB is up.
#
# Usage:
#   ./run_saclb_live_testbed.sh --algorithm dqn \
#     --checkpoint results/offline/dqn/offline_closed_loop/rep_0/checkpoint.pt \
#     --episodes 2 --run-id live-smoke-dqn-001 [--dry-run]

set -euo pipefail

ROOT_DIR="$(cd "$(dirname "$0")/../.." && pwd)"
DRL_DIR="$ROOT_DIR/drl_slicing"
QOE_DIR="$ROOT_DIR/qoe_oran_framework"
DOCKER_DIR="$ROOT_DIR/docker_open5gs"
OAI_RAN_DIR="$ROOT_DIR/oai_ran"
GEN_DIR="$DOCKER_DIR/generated"

UE_COUNT=3
EMBB_RATIO=0.7
RANDOM_SEED=256
PROFILE_MODE="triad"
TRAFFIC_TARGET_IP="12.1.1.1"
CORE_CONTAINER="amf"
DRY_RUN=0

ALGORITHM=""
CHECKPOINT=""
EPISODES=50
RUN_ID=""
GNB_ID="gnb-0"
CONFIG="$QOE_DIR/configs/saclb_live.yaml"
RESULTS_DIR="$QOE_DIR/results/live"

UE_COMPOSE_FILE="$GEN_DIR/nr-ue-fleet.yaml"
UE_PROFILE_CSV="$GEN_DIR/ue_fleet_profiles.csv"
NR_SOFTMODEM="$OAI_RAN_DIR/cmake_targets/ran_build/build/nr-softmodem"
GNB_CONF="$OAI_RAN_DIR/targets/PROJECTS/GENERIC-NR-5GC/CONF/ORANSlice.gnb.sa.band78.fr1.106PRB.usrpx310.conf"

while [[ $# -gt 0 ]]; do
  case "$1" in
    --algorithm) ALGORITHM="$2"; shift 2 ;;
    --checkpoint) CHECKPOINT="$2"; shift 2 ;;
    --episodes) EPISODES="$2"; shift 2 ;;
    --run-id) RUN_ID="$2"; shift 2 ;;
    --gnb-id) GNB_ID="$2"; shift 2 ;;
    --config) CONFIG="$2"; shift 2 ;;
    --results-dir) RESULTS_DIR="$2"; shift 2 ;;
    --ue-count) UE_COUNT="$2"; shift 2 ;;
    --embb-ratio) EMBB_RATIO="$2"; shift 2 ;;
    --random-seed) RANDOM_SEED="$2"; shift 2 ;;
    --core-container) CORE_CONTAINER="$2"; shift 2 ;;
    --dry-run) DRY_RUN=1; shift ;;
    *)
      echo "Unknown argument: $1"
      echo "Usage: $0 --algorithm {dqn|a2c|rainbow|lb_only} [--checkpoint PATH] [--episodes N] --run-id ID [--config PATH] [--results-dir DIR] [--ue-count N] [--dry-run]"
      exit 2
      ;;
  esac
done

if [[ -z "$ALGORITHM" ]]; then
  echo "--algorithm is required (dqn|a2c|rainbow|lb_only)"
  exit 2
fi
if [[ "$ALGORITHM" != "lb_only" && -z "$CHECKPOINT" ]]; then
  echo "--checkpoint is required unless --algorithm lb_only (live runs evaluate frozen weights only)"
  exit 2
fi
if [[ -z "$RUN_ID" ]]; then
  echo "--run-id is required"
  exit 2
fi

run_cmd() {
  local cmd="$1"
  if [[ "$DRY_RUN" -eq 1 ]]; then
    echo "[DRY-RUN] $cmd"
  else
    eval "$cmd"
  fi
}

echo "Stage Zero live SAC-LB run"
echo "  Algorithm    : $ALGORITHM"
echo "  Checkpoint   : ${CHECKPOINT:-<none, lb_only>}"
echo "  Episodes     : $EPISODES"
echo "  Run ID       : $RUN_ID"
echo "  Config       : $CONFIG"
echo "  gNB binary   : $NR_SOFTMODEM"

echo ""
echo "[1/6] Generate UE fleet compose + profiles"
run_cmd "cd '$DRL_DIR' && /usr/bin/python3 scripts/generate_ue_fleet_compose.py --ue-count '$UE_COUNT' --embb-ratio '$EMBB_RATIO' --random-seed '$RANDOM_SEED' --profile-mode '$PROFILE_MODE' --output-compose '$UE_COMPOSE_FILE' --output-profiles '$UE_PROFILE_CSV' --ue-stack ueransim"

echo "[2/6] Start Open5GS core + metrics"
run_cmd "cd '$DOCKER_DIR' && docker compose -f sa-deploy.yaml up -d"

echo "[3/6] Provision UE subscribers"
run_cmd "cd '$DRL_DIR' && bash scripts/provision_open5gs_subscribers.sh --profiles '$UE_PROFILE_CSV' --core-container '$CORE_CONTAINER'"

echo "[4/6] Start UE fleet containers"
run_cmd "cd '$DOCKER_DIR' && docker compose -f '$UE_COMPOSE_FILE' up -d"

echo "[5/6] Start profile-based traffic loops (eMBB/URLLC/mMTC)"
run_cmd "cd '$DRL_DIR' && bash scripts/start_ue_traffic_profiles.sh --profiles '$UE_PROFILE_CSV' --target-ip '$TRAFFIC_TARGET_IP'"

echo ""
echo "=========================================================================="
echo "[6/6] MANUAL STEP REQUIRED: start the native OAI gNB with the E2 agent."
echo ""
echo "This machine has no passwordless sudo, and nr-softmodem needs root for"
echo "RF-simulator raw sockets -- this cannot be automated from here. In"
echo "ANOTHER terminal, run:"
echo ""
echo "  cd '$OAI_RAN_DIR/cmake_targets/ran_build/build'"
echo "  sudo '$NR_SOFTMODEM' -O '$GNB_CONF' --sa --rfsim"
echo ""
echo "Wait until you see the gNB report it's ready (RRC/NGAP connected to the"
echo "core), THEN come back here and press Enter to continue starting the xApp."
echo "=========================================================================="
if [[ "$DRY_RUN" -eq 0 ]]; then
  read -r -p "Press Enter once the gNB is running... " _
else
  echo "[DRY-RUN] would wait for Enter here"
fi

mkdir -p "$RESULTS_DIR/$ALGORITHM"
OMEGA_PATH="$RESULTS_DIR/$ALGORITHM/${RUN_ID}_omega_log.jsonl"

echo ""
echo "Starting live xApp ($ALGORITHM), omega log -> $OMEGA_PATH"
XAPP_CMD="cd '$ROOT_DIR' && PYTHONPATH='$ROOT_DIR' /usr/bin/python3 qoe_oran_framework/xapp/saclb_xapp.py --config '$CONFIG' --algorithm '$ALGORITHM' --gnb-id '$GNB_ID' --episodes '$EPISODES' --seed '$RANDOM_SEED' --run-id '$RUN_ID' --omega-jsonl '$OMEGA_PATH'"
if [[ -n "$CHECKPOINT" ]]; then
  XAPP_CMD+=" --checkpoint '$CHECKPOINT'"
fi
run_cmd "$XAPP_CMD"

echo ""
echo "Live run complete. Omega log: $OMEGA_PATH"
echo "To stop the testbed cleanly:"
echo "  $QOE_DIR/scripts/stop_saclb_live_testbed.sh"
echo "  (and Ctrl+C / stop the manually-started nr-softmodem process)"
