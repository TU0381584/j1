# DRL Slice Allocation for 5G: IEEE Publication Roadmap

**Objective**: Demonstrate that deep reinforcement learning can optimize 5G resource allocation based on user bandwidth behaviour, with SLA-aware fairness and efficiency.

## Implementation Update (2026-04-07)

- Added unified orchestrator: `scripts/train_drl.py`
  - Supports: `dqn_train`, `a2c_train`, `rule_based`, `static`, `random`
  - Adds per-controller multi-seed summaries and top-level `comparison_summary.csv`
  - Supports live polling/runtime overrides: `--horizon-steps` and `--step-seconds`
- Added one-command rerun launcher: `scripts/run_ieee_access_rerun.sh`
  - Default package: 10 seeds x 1000 steps across all five controllers
  - Writes outputs to `results/ieee_access_rerun/`
- Added PPO controller mode wiring for Phase 2 progression
  - New modes: `ppo_train`, `ppo`
  - Integrated in orchestrator and IEEE rerun sweep package
- Added offline imitation warm-start path
  - Run logs now include `state_vector` + `action_index` for dataset generation
  - Controllers support configurable behavior-cloning warm-start before online training
  - Added dataset utility: `scripts/build_offline_warmstart_dataset.py`

### Quick start command

```bash
cd drl_slicing
./scripts/run_ieee_access_rerun.sh
```

## Phase 1: Foundation (✓ Complete)

### What was delivered

- **Experiment scaffold** with config-driven runner
- **Three baseline policies**: static, rule-based SLA controller, random
- **Telemetry pipeline**: CSV replay + Prometheus interface ready
- **Policy emission**: ORANSlice-compatible rrmPolicy.json writer
- **Run logging**: Per-step reward, SLA violations, action traces
- **Multi-seed orchestration**: Batch runner for statistical rigor

### Deliverables for paper

- [ ] Generate baseline runs (static, rule-based, random) across 10+ seeds
- [ ] Compute confidence intervals and statistical tests
- [ ] Create figures: reward convergence, SLA violation trends, per-slice throughput
- [ ] Report baseline performance metrics (acceptance criteria)

### Success metrics

- Baseline controllers runnable end-to-end
- Policy JSON compatible with OAI gNB
- Multi-seed results statistically sound (10+ seeds, CI bands)
- Example figures ready for paper

---

## Phase 2: DRL Policy Implementation (Weeks 1-3)

### Architecture: PPO-based slice allocation

**State representation** (per-slice, normalized):
```python
s_t = [
  throughput_achieved / offered_load,      # efficiency ratio
  latency_ms / sla_budget_ms,              # SLA margin
  loss_pct / loss_budget_pct,
  queue_depth / max_queue,                 # congestion signal
  trend_slope(throughput, window=5),       # velocity
  slice_type (one-hot: eMBB, URLLC, mMTC)
]
```

**Action representation** (per-slice):
```python
a_t = [
  min_ratio_idx,    # 0-11 (maps to 5-60 in steps of 5)
  max_ratio_idx     # 0-11 (maps to 40-100 in steps of 5)
]
# Clipped so min <= max
```

**Reward** (cumulative per step):
```
r_t = efficiency_bonus 
      - 2*latency_violation_penalty 
      - 3*loss_violation_penalty 
      - 0.1*action_smoothness_penalty
```

**Algorithm**:
- PPO with clipped surrogate loss
- Value network for baseline (variance reduction)
- Entropy regularization for exploration
- Batch size: 32 trajectories, 10 epoch updates

### Deliverables

- [x] Policy network (actor-critic) in PyTorch or TensorFlow
- [x] State normalizer and action encoder
- [x] Offline imitation warm-start from baseline traces
- [x] Online training loop with replay buffer
- [x] Training stability metrics (loss, KL divergence, advantage)
- [x] Sample training log and convergence plots

### Success metrics

- Policy learns to reduce SLA violations vs. rule-based baseline
- No catastrophic failures (SLA violations spike < 2× baseline)
- Training converges in <2000 episodes
- Inference latency <1ms per decision

---

## Phase 3: Comprehensive Evaluation (Weeks 4-6)

### Experiment matrix (publication grade)

**Scenarios**:
1. **eMBB-focused** (80% throughput-seeking, 20% URLLC)
2. **Mixed SLA** (50% eMBB, 50% URLLC with strict latency/loss limits)
3. **Bursty traffic** (Poisson arrivals with load ramps)
4. **Mobility & fading** (doppler shift, Rayleigh fading profiles)
5. **3+ slices** (baseline is 3, extend to 5)

**Per scenario**:
- 10+ random seeds (deterministic replay, different seed → different trace sampling)
- 3 controllers: static, rule-based, DRL
- Horizon: 300 steps (5 min at 1-Step/sec)
- Repeat 10 times to compute CI

**KPI set**:
- Per-slice: throughput, latency (p50/p95/p99), loss, PRB utilization
- Aggregate: Jain fairness, total violations, efficiency, stability
- Comparisons: vs. static, vs. rule-based (as primary), vs. PF scheduler

### Ablation studies

1. **Reward components**
   - Full SLA + efficiency + smoothness
   - SLA only
   - Efficiency only
   - → Show SLA component critical

2. **State representation**
   - Full (8-dim)
   - Without trend
   - Without queue depth
   - → Show which features matter

3. **Training strategy**
   - Offline-only (BC from baseline)
   - Online-only (no warm-start)
   - Hybrid (current)
   - → Show hybrid best for convergence

### Deliverables

- [ ] Experiment log files (seed_*/run_log.csv, seed_*/summary.csv)
- [ ] Aggregate CSV: all_results_eMBB.csv, all_results_mixed_sla.csv, etc.
- [ ] Statistical testing: t-tests, CI bands, significance threshold τ=0.05
- [ ] Figures (publication-ready):
  - **Fig 1**: Reward curves (mean ± CI) over 300 steps, 3 policies
  - **Fig 2**: SLA violation rate vs. time for mixed scenario
  - **Fig 3**: Per-slice throughput distribution (box plots, 10 seeds)
  - **Fig 4**: Fairness (Jain index) comparison
  - **Fig 5**: Ablation heatmap (ablation vs. KPI) → Show DRL dominates
- [ ] Supplementary results table (all scenarios × KPIs)

### Success metrics

- DRL outperforms rule-based on ≥3 KPIs (e.g., latency tail, fairness, efficiency)
- Statistical significance: p < 0.05 in ≥1 key scenario
- All 5 scenario types tested, ≥10 seeds each
- No failures: ≤1% seed where DRL crashes or violates hard bounds

---

## Phase 4: Live Testbed Integration (Optional, Weeks 6-8)

### If targeting high-impact venue, add OTA validation

**Setup**:
- gNB + 3 UEs in RF-sim or ZMQ fading
- Real traffic (iperf, VOIP, video) not synthetic traces
- Live Prometheus to capture actual CN metrics
- Policy deployed via rrmPolicy.json file updates

**Experiment**:
- 1 baseline run (rule-based) on live testbed
- 1 DRL policy run on live testbed
- Measure actual throughput, latency, loss vs. simulation
- Report discrepancies (reality gap analysis)

### Deliverables

- [ ] Testbed deployment docs (quick-start for reviewer)
- [ ] Live run comparisons: sim vs. testbed (scatter plots)
- [ ] Conclusion: "DRL policy generalizes with <X% reality gap"

### Success metrics

- Live testbed runs complete without crashes
- DRL policy still preferred in live setting (vs. baseline)
- Reality gap <15% on primary KPI

---

## Publication Submission Checklist

### Main manuscript (~8 pages, IEEE Trans. format)

- [ ] Abstract: 150 words, highlight SLA + fairness innovation
- [ ] Introduction: Problem (5G slicing challenges), prior work, contributions
- [ ] Problem formulation: MDP, state/action/reward
- [ ] Method: DRL architecture (PPO), training approach, offline warm-start
- [ ] Experiments: Scenario setup, baselines, KPI definitions
- [ ] Results: Figures 1-5, statistical tests, ablations, interpretation
- [ ] Conclusion: Impact, limitations, future work

### Author contributions

- [ ] Specify which author did Phase 1 (scaffold), Phase 2 (DRL), Phase 3 (eval), Phase 4 (testbed)

### Supplementary material

- [ ] All run logs, CSVs, seeds (anonymized if needed)
- [ ] Jupyter notebook for result regeneration
- [ ] Docker image or source code on GitHub (with reproducibility tag)
- [ ] Hyperparameter sweeps and sensitivity analysis

### Reproducibility checklist (IEEE standard)

- [ ] All code on GitHub with version tag (e.g., v1.0-ieee-submission)
- [ ] README with 3-command setup (pip install, data download, python run_all.py)
- [ ] Dockerfile for container (optional but recommended)
- [ ] Pre-computed results (all figures regenerable in 1h)

---

## Timeline

| Phase | Duration | Owner | Blockers | Milestone |
|-------|----------|-------|----------|-----------|
| **1. Foundation** | 0.5 weeks | ✓ Complete | None | Baseline runners, multi-seed infra |
| **2. DRL + Training** | 3 weeks | You | Policy debug, hyperparameter tuning | Convergence plots, offline BC |
| **3. Evaluation** | 2 weeks | You | Compute time for 5 scenarios×3 policies×10 seeds | All figures, stats, ablations |
| **4. Testbed (optional)** | 2 weeks | You | gNB/UE RF-sim stability | Live validation, reality-gap report |
| **5. Paper writing** | 2 weeks | You | Results interpretation, related work | Submission-ready draft |
| **6. Rebuttal/revision** | 2-4 weeks | You | Reviewer feedback | Final acceptance |

**Estimated total**: 11-15 weeks to acceptance (8-10 weeks if no testbed phase).

---

## Risk Mitigation

| Risk | Probability | Mitigation |
|------|-------------|-----------|
| DRL policy diverges | Medium | Use imitation warm-start, entropy reg, gradient clipping |
| Simulation-reality gap in metrics | Medium | Collect baseline traces on actual testbed early |
| Insufficient statistical power | Low | Pre-compute 10+ seeds, use CI + t-tests |
| Prometheus metrics unavailable | Med-High | Plan Prometheus integration in parallel, dummy data path ready |
| Multi-slice scaling (>5) | Low | Controllers generically handle any # slices; test 5 as upper limit |

---

## Success Criteria for IEEE Acceptance

1. **Technical novelty**: DRL for SLA-aware allocation is novel (not just DRL for throughput)
2. **Rigor**: 10+ seeds, CI bands, ablations, significance tests
3. **Reproducibility**: Code + data on GitHub, Docker, 1-hour rerun
4. **Evaluation scope**: 5 scenarios, 3 baselines, live testbed (if shot for top-tier)
5. **Impact**: ≥2 KPI improvements vs. rule-based baseline with p<0.05
6. **Writing**: Clear problem, method, results, and actionable insights for practitioners

---

## Current Status

✅ Phase 1 complete: Foundation infrastructure ready
✅ Phase 2 complete: DRL policies + warm-start + stability metrics/logging integrated
🟡 Phase 3 in progress: Live 3-UE (eMBB/URLLC/mMTC) multi-seed sweep launched
🔲 Phase 4 optional: Live testbed validation

**Current run**: `results/phase3/live_3ue_triad_ieee_access_20260407` (10 seeds, 6 controllers, horizon=300)

**Next immediate action**: Let current live sweep finish, aggregate comparison + figures, then execute remaining scenario matrix variants and statistical tests.
