# DRL Slicing (Live OAI E2E Only)

This module runs SLA-aware slice allocation experiments on the live Open5GS + OAI stack.

Replay and synthetic collectors are disabled by policy and by runtime enforcement.

## What is implemented

- Config-driven experiment runner
- Live Prometheus metrics ingestion (required)
- Controllers:
  - `dqn`, `dqn_train`
  - `a2c`, `a2c_train`
  - `ppo`, `ppo_train`
  - `rule_based`, `static`, `random`
- ORANSlice policy emitter to `rrmPolicy.json`
- Per-step run logging and multi-seed comparison outputs

## Folder structure

- `configs/`: live experiment YAMLs
- `oranslice_drl/`: implementation modules
- `scripts/`: executable entrypoints
- `results/`: generated outputs

## Quick start

From workspace root:

```bash
cd drl_slicing
python3 -m pip install -r requirements.txt
```

Launch the live OAI E2E stack (example: 5 UEs):

```bash
bash scripts/run_closed_loop_sim.sh --ue-count 5 --active-algo dqn
```

Run one experiment against live Prometheus metrics:

```bash
PYTHONPATH=. python3 scripts/run_experiment.py \
  --config configs/openran_live_prom_true_100ue_randomized.yaml
```

Run the IEEE comparison package:

```bash
./scripts/run_ieee_access_rerun.sh
```

## Optional Offline Warm-Start

Phase 2 includes offline imitation warm-start for DRL policies.

Build a state-action dataset from prior baseline/DRL run logs:

```bash
PYTHONPATH=. python3 scripts/build_offline_warmstart_dataset.py \
  --input results/ieee_access_rerun/rule_based \
  --output data/offline_warmstart_dataset.csv
```

Launch training with warm-start enabled (example: PPO):

```bash
PYTHONPATH=. python3 scripts/train_drl.py \
  --config configs/openran_live_prom_true_100ue_randomized.yaml \
  --controllers ppo_train \
  --seeds 42 123 456 \
  --output-dir results/phase2_ppo_warmstart \
  --offline-warm-start \
  --warm-start-source data/offline_warmstart_dataset.csv \
  --warm-start-epochs 3 \
  --warm-start-batch-size 64
```

## Enforced live-only policy

- `collector.mode` must be `prometheus`.
- Runner preflight requires active `amf`, `smf`, `upf`, `oaignb`, and at least one `nr_ue_*` container.
- Any replay or synthetic collector mode is rejected with an explicit error.

## Output artifacts

- Updated policy file: `/home/w5/ors/ORANSlice/oai_ran/rrmPolicy.json`
- Run logs and summaries under `drl_slicing/results/...`
