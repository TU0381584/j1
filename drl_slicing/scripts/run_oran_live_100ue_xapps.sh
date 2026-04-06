#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "$0")/../.." && pwd)"
DRL_DIR="$ROOT_DIR/drl_slicing"
DOCKER_DIR="$ROOT_DIR/docker_open5gs"
GEN_DIR="$DOCKER_DIR/generated"

UE_COUNT=100
EMBB_RATIO=0.7
RANDOM_SEED=42
ACTIVE_ALGO="dqn"
TRAFFIC_TARGET_IP="12.1.1.1"
CORE_CONTAINER="amf"
DRY_RUN=0

LIVE_CONFIG="$DRL_DIR/configs/openran_live_prom_true_100ue_randomized.yaml"
UE_COMPOSE_FILE="$GEN_DIR/nr-ue-fleet.yaml"
UE_PROFILE_CSV="$GEN_DIR/ue_fleet_profiles.csv"

while [[ $# -gt 0 ]]; do
  case "$1" in
    --ue-count)
      UE_COUNT="$2"
      shift 2
      ;;
    --embb-ratio)
      EMBB_RATIO="$2"
      shift 2
      ;;
    --random-seed)
      RANDOM_SEED="$2"
      shift 2
      ;;
    --active-algo)
      ACTIVE_ALGO="$2"
      shift 2
      ;;
    --traffic-target-ip)
      TRAFFIC_TARGET_IP="$2"
      shift 2
      ;;
    --core-container)
      CORE_CONTAINER="$2"
      shift 2
      ;;
    --config)
      LIVE_CONFIG="$2"
      shift 2
      ;;
    --dry-run)
      DRY_RUN=1
      shift
      ;;
    *)
      echo "Unknown argument: $1"
      echo "Usage: $0 [--ue-count N] [--embb-ratio R] [--random-seed S] [--active-algo dqn|a2c] [--traffic-target-ip IP] [--core-container NAME] [--config PATH] [--dry-run]"
      exit 2
      ;;
  esac
done

if [[ "$ACTIVE_ALGO" != "dqn" && "$ACTIVE_ALGO" != "a2c" ]]; then
  echo "--active-algo must be dqn or a2c"
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

cleanup_stale_ue_containers() {
  if [[ "$DRY_RUN" -eq 1 ]]; then
    echo "[DRY-RUN] remove stale containers named nr_ue_*"
    return
  fi

  local stale
  stale="$(docker ps -a --format '{{.Names}}' | grep '^nr_ue_' || true)"
  if [[ -n "$stale" ]]; then
    echo "$stale" | xargs -r docker rm -f >/dev/null 2>&1 || true
  fi
}

start_xapp() {
  local algorithm="$1"
  local service_name="$2"
  local policy_path="$3"
  local run_log_csv="$4"

  if [[ "$DRY_RUN" -eq 1 ]]; then
    echo "[DRY-RUN] start xApp $service_name ($algorithm) -> $policy_path"
    return
  fi

  mkdir -p "$DRL_DIR/results/xapps"
  (
    cd "$DRL_DIR"
    PYTHONPATH=. nohup /usr/bin/python3 scripts/xapp_service.py \
      --config "$LIVE_CONFIG" \
      --algorithm "$algorithm" \
      --repeat 0 \
      --name "$service_name" \
      --policy-json-path "$policy_path" \
      --run-log-csv "$run_log_csv" \
      > "results/xapps/${service_name}.log" 2>&1 &
    echo $! > "results/xapps/${service_name}.pid"
  )
}

echo "Starting ORAN live E2E run with DRL xApps"
echo "  UE count         : $UE_COUNT"
echo "  eMBB ratio       : $EMBB_RATIO"
echo "  Random seed      : $RANDOM_SEED"
echo "  Active controller: $ACTIVE_ALGO"
echo "  Config           : $LIVE_CONFIG"

echo "[1/10] Generate 100-UE fleet compose + profiles"
run_cmd "cd '$DRL_DIR' && /usr/bin/python3 scripts/generate_ue_fleet_compose.py --ue-count '$UE_COUNT' --embb-ratio '$EMBB_RATIO' --random-seed '$RANDOM_SEED' --output-compose '$UE_COMPOSE_FILE' --output-profiles '$UE_PROFILE_CSV'"

echo "[2/10] Start Open5GS core + metrics"
run_cmd "cd '$DOCKER_DIR' && docker compose -f sa-deploy.yaml up -d"

echo "[3/10] Start UERANSIM gNB"
run_cmd "cd '$DOCKER_DIR' && docker compose -f nr-gnb.yaml up -d"

echo "[4/10] Provision UE subscribers"
run_cmd "cd '$DRL_DIR' && bash scripts/provision_open5gs_subscribers.sh --profiles '$UE_PROFILE_CSV' --core-container '$CORE_CONTAINER'"

echo "[5/10] Start UE fleet containers"
cleanup_stale_ue_containers
run_cmd "cd '$DOCKER_DIR' && docker compose -f '$UE_COMPOSE_FILE' up -d"

echo "[6/10] Start randomized eMBB/URLLC traffic loops"
run_cmd "cd '$DRL_DIR' && bash scripts/start_ue_traffic_profiles.sh --profiles '$UE_PROFILE_CSV' --target-ip '$TRAFFIC_TARGET_IP'"

echo "[7/10] Wait for UE sessions to stabilize"
if [[ "$DRY_RUN" -eq 0 ]]; then
  sleep 15
fi

echo "[8/10] Start DRL xApps (both loaded, one active)"
ACTIVE_POLICY_PATH="$ROOT_DIR/oai_ran/rrmPolicy.json"
DQN_SHADOW_POLICY="$DRL_DIR/results/xapps/xapp-dqn-live/rrmPolicy.json"
A2C_SHADOW_POLICY="$DRL_DIR/results/xapps/xapp-a2c-live/rrmPolicy.json"

if [[ "$ACTIVE_ALGO" == "dqn" ]]; then
  start_xapp "dqn" "xapp-dqn-live" "$ACTIVE_POLICY_PATH" "$DRL_DIR/results/xapps/xapp-dqn-live/run_log.csv"
  start_xapp "a2c" "xapp-a2c-live" "$A2C_SHADOW_POLICY" "$DRL_DIR/results/xapps/xapp-a2c-live/run_log.csv"
else
  start_xapp "dqn" "xapp-dqn-live" "$DQN_SHADOW_POLICY" "$DRL_DIR/results/xapps/xapp-dqn-live/run_log.csv"
  start_xapp "a2c" "xapp-a2c-live" "$ACTIVE_POLICY_PATH" "$DRL_DIR/results/xapps/xapp-a2c-live/run_log.csv"
fi

echo "[9/10] Check active session metric (Prometheus)"
if [[ "$DRY_RUN" -eq 0 ]]; then
  SESSION_JSON="$(curl -s --get 'http://127.0.0.1:9090/api/v1/query' --data-urlencode 'query=fivegs_upffunction_upf_sessionnbr' || true)"
  SESSION_COUNT="$(printf '%s' "$SESSION_JSON" | /usr/bin/python3 -c 'import json,sys
raw=sys.stdin.read().strip()
if not raw:
    print("n/a")
    raise SystemExit(0)
try:
    obj=json.loads(raw)
    data=obj.get("data",{}).get("result",[])
    if not data:
        print("n/a")
    else:
        print(data[0].get("value",[0,"n/a"])[1])
except Exception:
    print("n/a")
')"
  echo "Prometheus fivegs_upffunction_upf_sessionnbr = $SESSION_COUNT"
else
  echo "[DRY-RUN] Prometheus query skipped"
fi

echo "[10/10] Launch complete"
echo "Monitoring commands:"
echo "  docker ps --format 'table {{.Names}}\t{{.Status}}'"
echo "  tail -f $DRL_DIR/results/xapps/xapp-dqn-live.log"
echo "  tail -f $DRL_DIR/results/xapps/xapp-a2c-live.log"
echo "  curl -s --get 'http://127.0.0.1:9090/api/v1/query' --data-urlencode 'query=fivegs_upffunction_upf_sessionnbr'"
echo ""
echo "To stop everything cleanly:"
echo "  $DRL_DIR/scripts/stop_oran_live_100ue_xapps.sh"
