# ORANSlice DRL: Live OAI E2E Getting Started

This guide is for running DRL slicing experiments only on the live Open5GS + OAI stack.

Replay/synthetic runs are intentionally disabled.

## 1. Prerequisites

```bash
cd /home/w5/ors/ORANSlice/drl_slicing
python3 -m pip install -r requirements.txt
```

## 2. Start the live OAI stack

Start a closed-loop deployment with native OAI gNB/UE:

```bash
bash scripts/run_closed_loop_sim.sh --ue-count 5 --active-algo dqn
```

Notes:
- Increase `--ue-count` as needed once host resources are confirmed.
- The active algorithm can be `dqn` or `a2c`.

## 3. Run a single live experiment

```bash
PYTHONPATH=. python3 scripts/run_experiment.py \
  --config configs/openran_live_prom_true_100ue_randomized.yaml
```

## 4. Run multi-controller IEEE package

```bash
./scripts/run_ieee_access_rerun.sh
```

By default this runs:
- `dqn_train`
- `a2c_train`
- `ppo_train`
- `rule_based`
- `static`
- `random`

Outputs are written under `results/ieee_access_rerun/`.

## 5. Run online controller comparison

```bash
./scripts/run_online_ran_compare.sh
```

Outputs are written under `results/online_ran_compare/`.

## 6. Optional: launch train-mode xApps

```bash
./scripts/load_xapps.sh configs/openran_live_prom_true_100ue_randomized.yaml 1
```

## 7. Optional: offline imitation warm-start

Create a compact warm-start dataset from existing run logs:

```bash
PYTHONPATH=. python3 scripts/build_offline_warmstart_dataset.py \
  --input results/ieee_access_rerun/rule_based \
  --output data/offline_warmstart_dataset.csv
```

Run DRL training with warm-start enabled:

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

## 8. Stop the stack

```bash
bash scripts/stop_oran_live_100ue_xapps.sh
```

## 9. Phase 3 live matrix

Run the full Phase 3 scenario matrix (eMBB, mixed SLA, burst, fading, 5-slice):

```bash
./scripts/run_phase3_live_matrix.sh
```

Optional overrides:

```bash
HORIZON_STEPS=300 STEP_SECONDS=1 ./scripts/run_phase3_live_matrix.sh results/phase3/live_matrix 42 123 456 789 999 1001 1002 1003 1004 1005
```

Outputs:
- Scenario-level summaries: `results/phase3/live_matrix/<scenario>/comparison_summary.csv`
- Aggregate stats: `results/phase3/live_matrix/_stats/phase3_summary.csv`
- Significance tests: `results/phase3/live_matrix/_stats/phase3_significance.csv`

## Live-only enforcement

The runner enforces all of the following:
- `collector.mode` must be `prometheus`.
- Docker containers `amf`, `smf`, `upf`, `oaignb` must be running.
- At least one `nr_ue_*` container must be running.

If any condition fails, execution stops with a clear error.

## Key artifacts

- Policy output: `/home/w5/ors/ORANSlice/oai_ran/rrmPolicy.json`
- Per-run logs: `results/*/seed_*/run_log.csv`
- Aggregates: `comparison_summary.csv` and `comparison.json`
