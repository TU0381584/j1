# DRL Slicing Implementation (Phase 1)

This module adds an initial closed-loop experimentation scaffold for SLA-aware slice allocation in ORANSlice.

## What is implemented

- Config-driven experiment runner
- Telemetry collectors:
  - `csv_replay` for deterministic dry runs
  - `prometheus` for live metrics ingestion
- Controllers:
  - `static`
  - `rule_based` (SLA violation driven baseline)
  - `random`
  - `drl_stub` (placeholder interface for upcoming DRL policy)
- ORANSlice policy emitter to `rrmPolicy.json`
- Run logger for reward/violation/action traces

## Folder structure

- `configs/`: experiment YAMLs
- `data/`: sample replay traces
- `oranslice_drl/`: implementation modules
- `scripts/`: executable entrypoints
- `results/`: generated outputs

## Quick start

From workspace root:

```bash
cd drl_slicing
python3 -m pip install -r requirements.txt
PYTHONPATH=. python3 scripts/run_experiment.py --config configs/rfsim_sla_rule_based.yaml
```

Output artifacts:

- Updated policy file: `/home/w5/ors/ORANSlice/oai_ran/rrmPolicy.json`
- Run log CSV: `drl_slicing/results/rfsim_sla_rule_based_log.csv`

## Live mode (Prometheus)

Set collector mode to `prometheus` in YAML and define queries that return per-slice vectors labeled by `slice_id`, `snssai`, or `slice`.

Expected query keys:

- `throughput_kbps`
- `latency_ms`
- `loss_pct`
- `offered_load_kbps` (optional)

## Next implementation milestone

- Replace `drl_stub` with an online DRL policy
- Add offline dataset builder and replay buffer format
- Add multi-seed runner and significance-testing pipeline
