#!/usr/bin/env python3
"""
Sanity check: verify all components are correctly integrated.
Run this after installation to confirm the scaffold is ready.
"""
import sys
import json
import os
import csv
from pathlib import Path

# Color codes for output
GREEN = '\033[92m'
RED = '\033[91m'
YELLOW = '\033[93m'
RESET = '\033[0m'

def check(condition, name):
    if condition:
        print(f"{GREEN}✓{RESET} {name}")
        return True
    else:
        print(f"{RED}✗{RESET} {name}")
        return False

def main():
    all_ok = True
    os.chdir(Path(__file__).parent)
    
    print(f"\n{YELLOW}=== DRL Slicing Sanity Check ==={RESET}\n")
    
    # 1. Check imports
    print("1. Module imports...")
    try:
        from oranslice_drl.config import load_experiment_config
        from oranslice_drl.runner import run_experiment
        all_ok &= check(True, "Core modules importable")
    except Exception as e:
        all_ok &= check(False, f"Core modules: {e}")
        return all_ok
    
    # 2. Check configs exist
    print("\n2. Configuration files...")
    configs = [
        "configs/rfsim_sla_static.yaml",
        "configs/rfsim_sla_rule_based.yaml",
        "configs/rfsim_sla_random.yaml",
    ]
    for cfg in configs:
        all_ok &= check(Path(cfg).exists(), f"{cfg}")
    
    # 3. Check data exists
    print("\n3. Sample data...")
    all_ok &= check(Path("data/sample_trace.csv").exists(), "Sample trace CSV")
    
    # 4. Load a config
    print("\n4. Configuration loading...")
    try:
        cfg = load_experiment_config("configs/rfsim_sla_rule_based.yaml")
        all_ok &= check(cfg.name == "rfsim_sla_rule_based", "Config parsed correctly")
        all_ok &= check(len(cfg.slices) == 3, "All 3 slices loaded")
        all_ok &= check(cfg.controller.mode == "rule_based", "Controller mode set")
    except Exception as e:
        all_ok &= check(False, f"Config loading: {e}")
    
    # 5. Run a mini experiment
    print("\n5. Mini experiment (2 steps)...")
    try:
        test_cfg = load_experiment_config("configs/rfsim_sla_rule_based.yaml")
        test_cfg.horizon_steps = 2
        test_cfg.collector.step_seconds = 0
        Path("results").mkdir(exist_ok=True)
        test_log = Path("results/test_sanity_check_log.csv")
        test_policy = Path("results/test_sanity_check_policy.json")
        test_cfg.output.run_log_csv = str(test_log)
        test_cfg.output.policy_json_path = str(test_policy)
        
        run_experiment(test_cfg)
        
        all_ok &= check(test_policy.exists(), "Policy JSON written")
        all_ok &= check(test_log.exists(), "Run log CSV written")
        
        # Verify policy format
        with open(test_policy) as f:
            policy = json.load(f)
        all_ok &= check("rrmPolicyRatio" in policy, "Policy has rrmPolicyRatio key")
        all_ok &= check(len(policy.get("rrmPolicyRatio", [])) == 3, "Policy has 3 slice entries")
        
        # Verify log format
        with open(test_log) as f:
            reader = csv.DictReader(f)
            rows = list(reader)
        all_ok &= check(len(rows) == 2, "Log has 2 step rows")
        all_ok &= check("reward" in rows[0], "Log has reward column")
        all_ok &= check("sla_violations" in rows[0], "Log has sla_violations column")
        
        # Cleanup
        test_log.unlink()
        test_policy.unlink()
    except Exception as e:
        all_ok &= check(False, f"Mini experiment: {e}")
    
    # 6. Check documentation
    print("\n6. Documentation...")
    all_ok &= check(Path("README.md").exists(), "README.md")
    all_ok &= check(Path("drl_slicing/GETTING_STARTED.md").exists() or Path("GETTING_STARTED.md").exists(), "GETTING_STARTED.md")
    all_ok &= check(Path("IEEE_ROADMAP.md").exists(), "IEEE_ROADMAP.md")
    
    # 7. Check scripts
    print("\n7. Executable scripts...")
    all_ok &= check(Path("scripts/run_experiment.py").exists(), "run_experiment.py")
    all_ok &= check(Path("scripts/batch_runner.py").exists(), "batch_runner.py")
    
    # Summary
    print(f"\n{('='*50)}")
    if all_ok:
        print(f"{GREEN}✓ All checks passed! Ready to run experiments.{RESET}")
        print(f"\nNext steps:")
        print(f"  1. cd drl_slicing")
        print(f"  2. PYTHONPATH=. python3 scripts/run_experiment.py --config configs/rfsim_sla_rule_based.yaml")
        print(f"  3. Check results: results/rfsim_sla_rule_based_log.csv")
        return 0
    else:
        print(f"{RED}✗ Some checks failed. Review errors above.{RESET}")
        return 1

if __name__ == "__main__":
    sys.exit(main())
