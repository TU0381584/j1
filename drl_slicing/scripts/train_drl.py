#!/usr/bin/env python3
"""
DRL Training Orchestrator: Train DQN and A2C policies, evaluate, and compare.
"""
import argparse
import os
import sys
import json
from pathlib import Path
from typing import Dict, List

sys.path.insert(0, str(Path(__file__).parent.parent))

from oranslice_drl.config import load_experiment_config
from oranslice_drl.runner import run_experiment


def run_algorithm(
    algorithm: str,
    config_path: str,
    output_dir: str,
    seeds: List[int],
    train: bool = True,
) -> Dict:
    """
    Train or evaluate an algorithm across multiple seeds.
    
    Args:
        algorithm: "dqn" or "a2c"
        config_path: path to base YAML config
        output_dir: directory for results
        seeds: list of random seeds
        train: if True, use training mode; else inference
        
    Returns:
        summary: aggregated results across seeds
    """
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    algo_dir = output_path / f"{algorithm}_{'train' if train else 'eval'}"
    algo_dir.mkdir(exist_ok=True)

    config_base = load_experiment_config(config_path)
    summaries = []

    print(f"\n{'='*60}")
    print(f"Running {algorithm.upper()} {'Training' if train else 'Evaluation'}")
    print(f"{'='*60}")

    for seed in seeds:
        run_dir = algo_dir / f"seed_{seed:03d}"
        run_dir.mkdir(exist_ok=True)

        config = load_experiment_config(config_path)
        config.random_seed = seed

        # Set controller mode based on algorithm and train flag
        if algorithm == "dqn":
            config.controller.mode = "dqn_train" if train else "dqn"
        elif algorithm == "a2c":
            config.controller.mode = "a2c_train" if train else "a2c"
        else:
            raise ValueError(f"Unknown algorithm: {algorithm}")

        config.output.policy_json_path = str(run_dir / "rrmPolicy.json")
        config.output.run_log_csv = str(run_dir / "run_log.csv")

        print(f"[Seed {seed:03d}] Running {config.name}...")
        try:
            run_experiment(config)
            print(f"[Seed {seed:03d}] ✓ Complete")
        except Exception as e:
            print(f"[Seed {seed:03d}] ✗ Error: {e}")
            continue

        # Aggregate results
        import csv
        run_log = run_dir / "run_log.csv"
        if run_log.exists():
            with run_log.open("r") as handle:
                reader = csv.DictReader(handle)
                rows = list(reader)
                if rows:
                    avg_reward = sum(float(r["reward"]) for r in rows) / len(rows)
                    total_violations = sum(int(r["sla_violations"]) for r in rows)
                    summaries.append({
                        "seed": seed,
                        "avg_reward": avg_reward,
                        "total_sla_violations": total_violations,
                        "num_steps": len(rows),
                    })

    # Save summary
    if summaries:
        summary_path = algo_dir / "summary.csv"
        import csv
        with summary_path.open("w", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=summaries[0].keys())
            writer.writeheader()
            writer.writerows(summaries)
        print(f"\n✓ Summary saved: {summary_path}")

        # Compute aggregate stats
        rewards = [s["avg_reward"] for s in summaries]
        violations = [s["total_sla_violations"] for s in summaries]
        import numpy as np
        stats = {
            "algorithm": algorithm,
            "train_mode": train,
            "num_seeds": len(summaries),
            "reward_mean": float(np.mean(rewards)),
            "reward_std": float(np.std(rewards)),
            "violations_mean": float(np.mean(violations)),
            "violations_std": float(np.std(violations)),
        }
        return stats

    return {}


def compare_algorithms(base_output_dir: str) -> None:
    """
    Compare DQN and A2C results from a base directory.
    """
    output_path = Path(base_output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    comparison_file = output_path / "comparison.json"

    results = {}

    for algo in ["dqn", "a2c"]:
        for mode in ["train", "eval"]:
            algo_dir = output_path / f"{algo}_{mode}"
            summary_file = algo_dir / "summary.csv"

            if not summary_file.exists():
                print(f"⚠ No summary found: {summary_file}")
                continue

            import csv
            import numpy as np
            with summary_file.open("r") as handle:
                reader = csv.DictReader(handle)
                rows = list(reader)

            rewards = [float(r["avg_reward"]) for r in rows]
            violations = [int(r["total_sla_violations"]) for r in rows]

            key = f"{algo}_{mode}"
            results[key] = {
                "num_seeds": len(rows),
                "reward_mean": float(np.mean(rewards)),
                "reward_std": float(np.std(rewards)),
                "reward_min": float(np.min(rewards)),
                "reward_max": float(np.max(rewards)),
                "violations_mean": float(np.mean(violations)),
                "violations_std": float(np.std(violations)),
            }

    if not results:
        print("⚠ No results found to compare.")
        return

    # Save comparison
    with comparison_file.open("w") as handle:
        json.dump(results, handle, indent=2)
    print(f"\n✓ Comparison results: {comparison_file}")
    print("\nSummary:")
    for key, stats in results.items():
        print(f"\n{key}:")
        print(f"  Reward: {stats['reward_mean']:.3f} ± {stats['reward_std']:.3f}")
        print(f"  Violations: {stats['violations_mean']:.1f} ± {stats['violations_std']:.1f}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train and compare DRL algorithms")
    parser.add_argument("--config", required=False, help="Base YAML config")
    parser.add_argument(
        "--algorithms",
        nargs="+",
        default=["dqn"],
        choices=["dqn", "a2c"],
        help="Algorithms to train",
    )
    parser.add_argument("--seeds", nargs="+", type=int, default=[42, 123, 456],
                        help="Random seeds")
    parser.add_argument("--output-dir", default="drl_results",
                        help="Base output directory")
    parser.add_argument("--train", action="store_true",
                        help="Train mode (else inference)")
    parser.add_argument("--compare", action="store_true",
                        help="Compare all results in output-dir")
    return parser.parse_args()


def main() -> int:
    args = parse_args()

    if args.train:
        if not args.config:
            print("Error: --config is required when --train is used")
            return 2
        for algo in args.algorithms:
            run_algorithm(algo, args.config, args.output_dir, args.seeds, train=True)

    elif args.config:
        for algo in args.algorithms:
            run_algorithm(algo, args.config, args.output_dir, args.seeds, train=False)

    if args.compare:
        compare_algorithms(args.output_dir)

    return 0


if __name__ == "__main__":
    sys.exit(main())
