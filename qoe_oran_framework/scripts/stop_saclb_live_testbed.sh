#!/usr/bin/env bash
# Companion teardown for run_saclb_live_testbed.sh. Mirrors
# drl_slicing/scripts/stop_oran_live_testbed_xapps.sh's pattern. Does NOT
# stop the natively-run nr-softmodem gNB -- that was started manually
# (under sudo, in another terminal) and must be stopped the same way
# (Ctrl+C there, or `sudo pkill -f nr-softmodem` if you started it
# detached).

set -euo pipefail

ROOT_DIR="$(cd "$(dirname "$0")/../.." && pwd)"
DRL_DIR="$ROOT_DIR/drl_slicing"
DOCKER_DIR="$ROOT_DIR/docker_open5gs"
GEN_DIR="$DOCKER_DIR/generated"

KEEP_CORE=0
DRY_RUN=0

while [[ $# -gt 0 ]]; do
  case "$1" in
    --keep-core) KEEP_CORE=1; shift ;;
    --dry-run) DRY_RUN=1; shift ;;
    *)
      echo "Unknown argument: $1"
      echo "Usage: $0 [--keep-core] [--dry-run]"
      exit 2
      ;;
  esac
done

run_cmd() {
  local cmd="$1"
  if [[ "$DRY_RUN" -eq 1 ]]; then
    echo "[DRY-RUN] $cmd"
  else
    eval "$cmd"
  fi
}

echo "Stopping Stage Zero live testbed"

echo "[1/3] Stop UE traffic loops"
run_cmd "cd '$DRL_DIR' && bash scripts/stop_ue_traffic_profiles.sh --profiles '$GEN_DIR/ue_fleet_profiles.csv' || true"

echo "[2/3] Stop generated UE fleet"
run_cmd "cd '$DOCKER_DIR' && docker compose -f '$GEN_DIR/nr-ue-fleet.yaml' down >/dev/null 2>&1 || true"
run_cmd "docker ps -a --format '{{.Names}}' | grep '^nr_ue_' | xargs -r docker rm -f >/dev/null 2>&1 || true"

echo "[3/3] Stop Open5GS core"
if [[ "$KEEP_CORE" -eq 1 ]]; then
  echo "Keeping Open5GS core running (--keep-core set)"
else
  run_cmd "cd '$DOCKER_DIR' && docker compose -f sa-deploy.yaml down >/dev/null 2>&1 || true"
fi

echo "Stop sequence complete."
echo "Remember: the natively-run nr-softmodem gNB (started manually under"
echo "sudo) is NOT stopped by this script -- stop it in its own terminal."
