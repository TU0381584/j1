#!/usr/bin/env python3
"""
Quick validation of DQN implementation.
"""
import sys
import os
from pathlib import Path

os.chdir(Path(__file__).parent.parent)
sys.path.insert(0, str(Path(__file__).parent.parent))

from oranslice_drl.config import load_experiment_config
from oranslice_drl.runner import run_experiment

print("Testing DQN implementation...")

# Load DQN training config
cfg = load_experiment_config('configs/rfsim_dqn_training.yaml')
cfg.horizon_steps = 5  # Short run for validation
cfg.collector.step_seconds = 0
cfg.collector.allow_non_live_for_training = True
cfg.output.policy_json_path = 'results/test_dqn_policy.json'
cfg.output.run_log_csv = 'results/test_dqn_log.csv'

policy_file = Path(cfg.output.policy_json_path)
log_file = Path(cfg.output.run_log_csv)
if policy_file.exists():
    policy_file.unlink()
if log_file.exists():
    log_file.unlink()

print(f"Config: {cfg.name}")
print(f"Controller mode: {cfg.controller.mode}")
print(f"Horizon: {cfg.horizon_steps} steps")

try:
    run_experiment(cfg)
    print("✓ DQN training completed successfully")
    
    # Check outputs
    import json
    import csv
    
    if policy_file.exists():
        print(f"✓ Policy JSON written ({policy_file.stat().st_size} bytes)")
    else:
        print(f"✗ Policy JSON not found")
    
    if log_file.exists():
        with log_file.open() as f:
            rows = list(csv.DictReader(f))
        print(f"✓ Run log written ({len(rows)} steps)")
        if rows:
            print(f"  First step reward: {rows[0]['reward']}")
            if len(rows) > 1:
                print(f"  Last step reward: {rows[-1]['reward']}")
    else:
        print(f"✗ Run log not found")
        
except Exception as e:
    print(f"✗ Error: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)

print("\n✓ All tests passed!")
