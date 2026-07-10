#!/usr/bin/env python3
"""
A2C End-to-End Integration Test
Validates A2C training with experiment runner (no testbed).
"""
import sys
import os
from pathlib import Path

os.chdir(Path(__file__).parent.parent)
sys.path.insert(0, str(Path(__file__).parent.parent))

from oranslice_drl.runner import run_experiment
from oranslice_drl.config import load_experiment_config

def test_a2c_integration():
    """Run A2C training via experiment runner."""
    print("Testing A2C integration with experiment runner...")
    
    # Load config
    config = load_experiment_config('configs/rfsim_a2c_training.yaml')
    
    # Validate config
    assert config.controller.mode in ["a2c", "a2c_train"], f"Expected a2c mode, got {config.controller.mode}"
    print("✓ Config loaded: rfsim_a2c_training.yaml")
    print(f"  Controller mode: {config.controller.mode}")
    print(f"  Horizon: {config.horizon_steps} steps")
    
    # Run experiment (short 5-step validation)
    config.horizon_steps = 5
    config.collector.step_seconds = 0
    config.collector.allow_non_live_for_training = True
    config.output.policy_json_path = 'results/test_a2c_policy.json'
    config.output.run_log_csv = 'results/test_a2c_log.csv'
    
    policy_path = Path(config.output.policy_json_path)
    log_path = Path(config.output.run_log_csv)
    if policy_path.exists():
        policy_path.unlink()
    if log_path.exists():
        log_path.unlink()
    
    print("Running 5-step A2C training...")
    run_experiment(config)
    print("✓ A2C training completed successfully")
    
    # Check outputs
    assert policy_path.exists(), "Policy file not generated"
    assert log_path.exists(), "Run log file not generated"
    
    import json
    import csv
    
    with open(policy_path) as f:
        policy = json.load(f)
    assert "rrmPolicyRatio" in policy
    print(f"✓ Policy JSON written ({policy_path.stat().st_size} bytes)")
    
    with open(log_path) as f:
        rows = list(csv.DictReader(f))
    assert len(rows) == 5, f"Expected 5 rows in log, got {len(rows)}"
    print(f"✓ Run log written ({len(rows)} steps)")
    
    rewards = [float(r["reward"]) for r in rows]
    print(f"  First step reward: {rewards[0]}")
    print(f"  Last step reward: {rewards[-1]}")
    
    print("\n✓ All A2C integration tests passed!")

if __name__ == "__main__":
    test_a2c_integration()
