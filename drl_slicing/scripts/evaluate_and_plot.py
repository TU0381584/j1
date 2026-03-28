#!/usr/bin/env python3
"""
Evaluate DQN vs A2C and generate comparison plots.
"""
import argparse
import json
import subprocess
from pathlib import Path

import matplotlib.pyplot as plt


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run evaluation and plot comparison")
    parser.add_argument("--config", required=True, help="Training config")
    parser.add_argument("--output-dir", default="results/phaseB_eval", help="Evaluation output folder")
    parser.add_argument("--seeds", nargs="+", type=int, default=[42, 123, 456], help="Seed list")
    parser.add_argument("--skip-run", action="store_true", help="Skip evaluation and only plot existing results")
    return parser.parse_args()


def run_eval(config: str, output_dir: str, seeds):
    cmd = [
        "python3",
        "scripts/train_drl.py",
        "--config", config,
        "--algorithms", "dqn", "a2c",
        "--seeds", *[str(seed) for seed in seeds],
        "--output-dir", output_dir,
        "--train",
        "--compare",
    ]
    subprocess.run(cmd, check=True)


def plot_comparison(output_dir: str):
    comparison_path = Path(output_dir) / "comparison.json"
    if not comparison_path.exists():
        raise FileNotFoundError(f"Missing comparison file: {comparison_path}")

    with comparison_path.open("r", encoding="utf-8") as handle:
        data = json.load(handle)

    labels = []
    reward_means = []
    reward_stds = []
    violation_means = []
    violation_stds = []

    for key in ["dqn_train", "a2c_train"]:
        if key not in data:
            continue
        labels.append(key)
        reward_means.append(data[key]["reward_mean"])
        reward_stds.append(data[key]["reward_std"])
        violation_means.append(data[key]["violations_mean"])
        violation_stds.append(data[key]["violations_std"])

    plots_dir = Path(output_dir) / "plots"
    plots_dir.mkdir(parents=True, exist_ok=True)

    plt.figure(figsize=(8, 5))
    plt.bar(labels, reward_means, yerr=reward_stds, capsize=6)
    plt.title("Reward Comparison (DQN vs A2C)")
    plt.ylabel("Average Reward")
    plt.tight_layout()
    plt.savefig(plots_dir / "reward_comparison.png", dpi=160)
    plt.close()

    plt.figure(figsize=(8, 5))
    plt.bar(labels, violation_means, yerr=violation_stds, capsize=6)
    plt.title("SLA Violation Comparison (DQN vs A2C)")
    plt.ylabel("Average SLA Violations")
    plt.tight_layout()
    plt.savefig(plots_dir / "violation_comparison.png", dpi=160)
    plt.close()

    print(f"✓ Plots saved in: {plots_dir}")


def main() -> int:
    args = parse_args()
    if not args.skip_run:
        run_eval(args.config, args.output_dir, args.seeds)
    plot_comparison(args.output_dir)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
