#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "$0")/../.." && pwd)"
DRL_DIR="$ROOT_DIR/drl_slicing"

DRY_RUN=0
ACTIVE_ALGO="dqn"

while [[ $# -gt 0 ]]; do
  case "$1" in
    --dry-run)
      DRY_RUN=1
      shift
      ;;
    --active-algo)
      ACTIVE_ALGO="$2"
      shift 2
      ;;
    *)
      echo "Unknown argument: $1"
      echo "Usage: $0 [--active-algo dqn|a2c] [--dry-run]"
      exit 2
      ;;
  esac
done

if [[ "$ACTIVE_ALGO" != "dqn" && "$ACTIVE_ALGO" != "a2c" ]]; then
  echo "--active-algo must be dqn or a2c"
  exit 2
fi

CMD=(
  bash "$DRL_DIR/scripts/run_oran_live_100ue_xapps.sh"
  --gnb-mode oai
  --ue-stack external
  --active-algo "$ACTIVE_ALGO"
)

if [[ "$DRY_RUN" -eq 1 ]]; then
  CMD+=(--dry-run)
fi

echo "Launching closed-loop simulation preset"
echo "  gNB mode         : oai"
echo "  UE stack         : external (no UERANSIM)"
echo "  Active controller: $ACTIVE_ALGO"
if [[ "$DRY_RUN" -eq 1 ]]; then
  echo "  Dry-run          : enabled"
fi

"${CMD[@]}"