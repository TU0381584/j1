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

## 7. Stop the stack

```bash
bash scripts/stop_oran_live_100ue_xapps.sh
```

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
