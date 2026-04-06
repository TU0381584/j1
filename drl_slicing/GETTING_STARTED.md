# ORANSLICE DRL Integration Implementation Progress

**Completed Implementation (Phase 1A)**

This is the initial closed-loop scaffold for SLA-driven slice allocation on OAI RAN/CN.

## What Is Working

1. **Config-driven runners**
   - Single-run experiment via `python scripts/run_experiment.py --config configs/<name>.yaml`
   - Multi-seed batch runner via `python scripts/batch_runner.py --config <name>.yaml --seeds 42 123 456 789 999`
   - Unified multi-controller sweep via `python scripts/train_drl.py --config <name>.yaml --controllers dqn_train a2c_train rule_based static random`

2. **Baseline controllers**
   - `static`: Fixed min/max ratios from config
   - `rule_based`: Adaptive controller that increases resource when SLA violated
   - `random`: Probabilistic baseline for comparison

3. **Telemetry**
   - `csv_replay`: Deterministic trace replay (current)
   - `prometheus`: Live metrics query interface (ready for testbed)

4. **Output artifacts**
   - **rrmPolicy.json** → OAI gNB slice control file (ORANSlice-compatible format)
   - **run_log.csv** → Timestamped reward, SLA violations, and actions per step
   - **summary.csv** → Aggregate statistics per seed (from batch runner)

5. **Verification**
   - Dry-run test ✓ (2-step execution on sample trace)
   - Policy format compatible with ORANSlice ✓
   - Log schema correct ✓

## File Structure

```
drl_slicing/
  configs/
    rfsim_sla_static.yaml          # Static allocation baseline
    rfsim_sla_rule_based.yaml      # Rule-based SLA controller
    rfsim_sla_random.yaml          # Random policy for comparison
  data/
    sample_trace.csv               # 10-row test trace
  notebooks/
    analysis.ipynb                 # Jupyter for visualization/stats
  oranslice_drl/
    __init__.py
    config.py                      # Config loading and dataclasses
    collectors.py                  # CSV and Prometheus ingestion
    controllers.py                 # Static, rule-based, random, DRL stub
    policy_io.py                   # JSON policy writer
    reward.py                      # SLA-focused reward function
    runner.py                      # Main experiment loop
    types.py                       # State, action, metric records
  scripts/
    run_experiment.py              # Single-run entrypoint
    batch_runner.py                # Multi-seed orchestrator
    train_drl.py                   # Unified DRL + baseline orchestrator
    run_ieee_access_rerun.sh       # One-command IEEE rerun package
  results/                         # Generated outputs
  README.md                        # Original phase-1 docs
  requirements.txt
  GETTING_STARTED.md               # This file
```

## Quick Start

### Prerequisites
```bash
cd drl_slicing
python3 -m pip install -r requirements.txt
```

### Run a single experiment (rule-based SLA controller)
```bash
cd drl_slicing
PYTHONPATH=. python3 scripts/run_experiment.py \
  --config configs/rfsim_sla_rule_based.yaml
```

After execution:
- Policy file: `/home/w5/ors/ORANSlice/oai_ran/rrmPolicy.json`
- Run log: `results/rfsim_sla_rule_based_log.csv`

### Run multiple baselines for comparison
```bash
# Static baseline (no adaptation)
python3 scripts/run_experiment.py --config configs/rfsim_sla_static.yaml

# Random policy
python3 scripts/run_experiment.py --config configs/rfsim_sla_random.yaml

# All three in a batch with 5 seeds each
python3 scripts/batch_runner.py \
  --config configs/rfsim_sla_rule_based.yaml \
  --seeds 42 123 456 789 999 \
  --output-dir batch_results/rule_based

python3 scripts/batch_runner.py \
  --config configs/rfsim_sla_static.yaml \
  --seeds 42 123 456 789 999 \
  --output-dir batch_results/static

python3 scripts/batch_runner.py \
  --config configs/rfsim_sla_random.yaml \
  --seeds 42 123 456 789 999 \
  --output-dir batch_results/random
```

### Analyze results
```bash
# Open Jupyter in the drl_slicing folder
jupyter notebook notebooks/analysis.ipynb
```

Load and combine CSVs from `batch_results/*/summary.csv` to compute confidence intervals and statistical tests.

### Run IEEE Access short rerun package (10 seeds, 1000 steps)
```bash
cd drl_slicing
./scripts/run_ieee_access_rerun.sh
```

This generates per-controller outputs under `results/ieee_access_rerun/`:
- `dqn_train/seed_*/run_log.csv`
- `a2c_train/seed_*/run_log.csv`
- `rule_based/seed_*/run_log.csv`
- `static/seed_*/run_log.csv`
- `random/seed_*/run_log.csv`
- `comparison_summary.csv` and `comparison.json` at the root of the run folder

## Integration with ORANSlice Testbed

### For Live Metrics (RFC)

1. In your YAML config, switch collector mode:
   ```yaml
   collector:
     mode: prometheus
     prometheus_url: "http://your-prometheus-ip:9090"
     metric_queries:
       throughput_kbps: "rate(upf_pdu_session_packet_count[1m])*8/1000"  # example
       latency_ms: "histogram_quantile(0.95, gnb_pusch_latency_ms)"
       loss_pct: "gnb_pdu_loss_rate*100"
   ```

2. Ensure your CN metrics (SMF/UPF/AMF) export Prometheus endpoints on `:9091`.

3. Export RAN KPIs via E2/KPM:
   - Currently documented in [XAPP_SETUP.md](../XAPP_SETUP.md)
   - Integration with xApp to populate Prometheus pending

### Deploy Policy to gNB

Two options:

**Option A: File-based (current, simple)**
- gNB monitors `SliceConf = "<path>/rrmPolicy.json"`
- Updated JSON is read every ~100ms
- See [doc/rrmPolicyJson.patch](../doc/rrmPolicyJson.patch)

**Option B: E2/xApp control (phase 2)**
- xApp BS Connector polls rrmPolicy.json
- Sends E2 control messages to gNB
- More robust, O-RAN aligned

## Next Phase (DRL Training)

1. **Online DRL Policy**
   - Replace `DrlControllerStub` with actual policy learning loop
   - Use PPO or A3C for continuous action space (min/max ratio bounds)
   - Gradient updates per step or per episode

2. **Offline Warm-Start**
   - Collect traces from static/rule-based baselines
   - Train imitation learning policy (BC) to bootstrap
   - Then fine-tune with online RL

3. **Multi-Seed Experiment Matrix**
   - Sweep offered load, slice mix, channel conditions
   - 10+ seeds per scenario for statistical significance
   - Confidence intervals and ablation studies

4. **Reproducibility Package**
   - Pin all dependencies (versions)
   - Manifest of scenario definitions
   - One-command re-generation of all figures
   - Docker container for artifact evaluation

5. **Publication Deliverables**
   - IEEE paper with results, ablations, significance tests
   - v1.0 code release with artifact tag
   - Reproducibility artifact bundle

## Known Limitations & Assumptions

- **Sample trace**: 10 rows, hardcoded; extend with real traffic captures or synthetic generators
- **Prometheus mode**: Not yet validated with live testbed metrics
- **Action space**: Currently discrete (min/max ratio bins); continuous space possible with TF/PyTorch
- **State representation**: Per-slice throughput, latency, loss; extended with queue depth, mobility, etc. in future
- **Reward**: Simple SLA penalty + efficiency; multi-objective pareto frontiers as phase-2 enhancement
- **Platform**: Verified on Python 3.8+; TensorFlow/PyTorch versions TBD for DRL

## Support & Troubleshooting

- **Config load fails**: Check YAML syntax, paths are relative to working directory
- **Policy JSON empty**: Check collector mode and trace existence
- **Seed reproducibility**: Set `random_seed` in config; numpy and Python random seeded
- **Live Prometheus**: Validate query format in Prometheus UI first

## Architecture Diagram (ASCII)

```
┌─────────────────────────────────────────────────────────────┐
│                      ORANSlice RAN                          │
│    (gNB monitors rrmPolicy.json every ~100ms)              │
└──────────────────────────┬──────────────────────────────────┘
                           │
                    (policy JSON updates)
                           ↓
         ┌─────────────────────────────────────────┐
         │   DRL Control Loop                      │
         │  (This Implementation)                  │
         ├─────────────────────────────────────────┤
         │ [1] Collector: CSV/Prometheus ingestion │
         │ [2] State: per-slice metrics normalize │
         │ [3] Reward: SLA violation penalty       │
         │ [4] Controller: Static/RB/Random/DRL   │
         │ [5] Policy Writer: JSON to disk         │
         └─────────────────────────────────────────┘
                           │
         ┌─────────────────┴──────────────────┐
         │                                    │
      [CSV trace]                      [Prometheus]
      (off-line testing)              (live testbed)
```

## Citation

If you use this implementation, please cite:

- ORANSlice paper: "H. Cheng, S. D'Oro, ..., ORANSlice: An Open-Source 5G Network Slicing Platform for O-RAN," MobiCom 2024
- Your DRL slice allocation work (TBD)

## Running Live 100-UE O-RAN Simulation

To spin up the entire testbed including the 5G Core, O-RAN gNBs, 100 User Equipments (UEs) categorized into eMBB and URLLC classes with active traffic, and run the DRL agents (A2C/DQN) as xApps, execute the following orchestrator scripts.

**1. Start the Live 100-UE O-RAN Orchestrator:**
```bash
cd drl_slicing/scripts
./run_oran_live_100ue_xapps.sh --active-algo dqn
```

This master script will:
- Generate independent identities and profiles for 100 UEs dynamically.
- Spin up Open5GS Core.
- Spin up UERANSIM gNB.
- Provision all generated UE IMSIs directly in the Core DB.
- Spin up the 100 UE UERANSIM instances matching the generated configs.
- Inject randomized multi-profile traffic directly via Pings across the UE network namespces.
- Start your specified RL algo as the actively loaded xApp (`rrmPolicy.json` source) and start the opposite model in passive mode.

**2. Tear Down the Environment:**
Once finished, to cleanly stop all containers and traffic load tools:
```bash
cd drl_slicing/scripts
./stop_oran_live_100ue_xapps.sh
```
