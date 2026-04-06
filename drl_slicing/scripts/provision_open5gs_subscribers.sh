#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "$0")/../.." && pwd)"
DEFAULT_PROFILE_CSV="$ROOT_DIR/docker_open5gs/generated/ue_fleet_profiles.csv"

PROFILE_CSV="$DEFAULT_PROFILE_CSV"
CORE_CONTAINER="amf"
DB_URI="mongodb://mongo/open5gs"
DRY_RUN=0

while [[ $# -gt 0 ]]; do
  case "$1" in
    --profiles)
      PROFILE_CSV="$2"
      shift 2
      ;;
    --core-container)
      CORE_CONTAINER="$2"
      shift 2
      ;;
    --db-uri)
      DB_URI="$2"
      shift 2
      ;;
    --dry-run)
      DRY_RUN=1
      shift
      ;;
    *)
      echo "Unknown argument: $1"
      echo "Usage: $0 [--profiles <csv>] [--core-container <name>] [--db-uri <mongodb_uri>] [--dry-run]"
      exit 2
      ;;
  esac
done

if [[ ! -f "$PROFILE_CSV" ]]; then
  echo "Profile CSV not found: $PROFILE_CSV"
  exit 1
fi

if ! docker ps --format '{{.Names}}' | grep -q "^${CORE_CONTAINER}$"; then
  echo "Core container is not running: $CORE_CONTAINER"
  echo "Start Open5GS first (e.g. docker compose -f docker_open5gs/sa-deploy.yaml up -d)."
  exit 1
fi

DBCTL_PATH="$(docker exec "$CORE_CONTAINER" sh -lc '
if [ -x /open5gs/misc/db/open5gs-dbctl ]; then
  echo /open5gs/misc/db/open5gs-dbctl
elif [ -x /open5gs/install/bin/open5gs-dbctl ]; then
  echo /open5gs/install/bin/open5gs-dbctl
elif [ -x misc/db/open5gs-dbctl ]; then
  echo misc/db/open5gs-dbctl
else
  echo ""
fi
')"

if [[ -z "$DBCTL_PATH" ]]; then
  echo "Could not locate open5gs-dbctl inside container: $CORE_CONTAINER"
  exit 1
fi

# Some Open5GS images ship only the legacy mongo client. Provide a mongosh shim for open5gs-dbctl.
if ! docker exec "$CORE_CONTAINER" sh -lc 'command -v mongosh >/dev/null 2>&1'; then
  if docker exec "$CORE_CONTAINER" sh -lc 'command -v mongo >/dev/null 2>&1'; then
    docker exec "$CORE_CONTAINER" sh -lc 'ln -sf "$(command -v mongo)" /usr/local/bin/mongosh'
  else
    echo "Neither mongosh nor mongo is available inside container: $CORE_CONTAINER"
    exit 1
  fi
fi

echo "Using subscriber tool: $DBCTL_PATH"
echo "Using DB URI        : $DB_URI"
echo "Provisioning subscribers from: $PROFILE_CSV"

added=0
skipped=0
failed=0

while IFS=, read -r _service _container _component imsi _imei _imeisv ki op _amf _profile; do
  [[ -z "$imsi" ]] && continue

  if [[ "$DRY_RUN" -eq 1 ]]; then
    echo "[DRY-RUN] $DBCTL_PATH --db_uri=$DB_URI add $imsi $ki $op"
    continue
  fi

  if docker exec "$CORE_CONTAINER" sh -lc "$DBCTL_PATH --db_uri='$DB_URI' add '$imsi' '$ki' '$op'" >/dev/null 2>&1; then
    added=$((added + 1))
  else
    if docker exec "$CORE_CONTAINER" sh -lc "$DBCTL_PATH --db_uri='$DB_URI' showfiltered | grep -q '$imsi'"; then
      skipped=$((skipped + 1))
      echo "Skipped IMSI (already exists): $imsi"
    else
      failed=$((failed + 1))
      echo "Failed to add IMSI: $imsi"
    fi
  fi
done < <(tail -n +2 "$PROFILE_CSV")

if [[ "$DRY_RUN" -eq 1 ]]; then
  echo "Dry-run complete."
else
  echo "Provisioning complete: added=$added skipped=$skipped failed=$failed"
fi
