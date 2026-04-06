#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "$0")/../.." && pwd)"
DEFAULT_PROFILE_CSV="$ROOT_DIR/docker_open5gs/generated/ue_fleet_profiles.csv"

PROFILE_CSV="$DEFAULT_PROFILE_CSV"
TARGET_IP="12.1.1.1"
DRY_RUN=0

while [[ $# -gt 0 ]]; do
  case "$1" in
    --profiles)
      PROFILE_CSV="$2"
      shift 2
      ;;
    --target-ip)
      TARGET_IP="$2"
      shift 2
      ;;
    --dry-run)
      DRY_RUN=1
      shift
      ;;
    *)
      echo "Unknown argument: $1"
      echo "Usage: $0 [--profiles <csv>] [--target-ip <ip>] [--dry-run]"
      exit 2
      ;;
  esac
done

if [[ ! -f "$PROFILE_CSV" ]]; then
  echo "Profile CSV not found: $PROFILE_CSV"
  exit 1
fi

echo "Starting profile-based UE traffic"
echo "  Profiles: $PROFILE_CSV"
echo "  Target : $TARGET_IP"

started=0
skipped=0

while IFS=, read -r _service container _component _imsi _imei _imeisv _ki _op _amf profile; do
  [[ -z "$container" ]] && continue
  profile="${profile//$'\r'/}"
  profile="${profile,,}"

  if ! docker ps --format '{{.Names}}' | grep -q "^${container}$"; then
    skipped=$((skipped + 1))
    continue
  fi

  case "$profile" in
    embb)
      traffic_cmd="while true; do ping -I uesimtun0 -s \$((1200 + RANDOM % 180)) -i 0.03 -c 25 ${TARGET_IP} >/dev/null 2>&1 || true; sleep 0.\$((RANDOM % 3 + 1)); done"
      ;;
    urllc)
      traffic_cmd="while true; do ping -I uesimtun0 -s \$((48 + RANDOM % 64)) -i 0.15 -c 8 ${TARGET_IP} >/dev/null 2>&1 || true; sleep 0.\$((RANDOM % 2 + 1)); done"
      ;;
    *)
      traffic_cmd="while true; do ping -I uesimtun0 -s 256 -i 0.1 -c 10 ${TARGET_IP} >/dev/null 2>&1 || true; sleep 1; done"
      ;;
  esac

  if [[ "$DRY_RUN" -eq 1 ]]; then
    echo "[DRY-RUN] docker exec -d $container bash -lc '<traffic loop for $profile>'"
    started=$((started + 1))
    continue
  fi

  # Avoid self-termination by matching only legacy loop command lines.
  docker exec "$container" bash -lc "pkill -f '^bash -lc while true; do ping -I uesimtun0' >/dev/null 2>&1 || true; pkill -x ping >/dev/null 2>&1 || true"
  docker exec -d "$container" bash -lc "${traffic_cmd}"
  started=$((started + 1))
done < <(tail -n +2 "$PROFILE_CSV")

echo "Traffic launch complete: started=$started skipped_not_running=$skipped"
