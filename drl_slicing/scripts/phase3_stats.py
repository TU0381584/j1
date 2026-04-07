#!/usr/bin/env python3
"""Compute per-scenario Phase 3 stats, confidence intervals, and significance tests."""

import argparse
import csv
import json
import math
from pathlib import Path
from statistics import NormalDist
from typing import Dict, List

import numpy as np


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Phase 3 statistical summary generator")
    parser.add_argument(
        "--root",
        required=True,
        help="Phase 3 root output directory (contains scenario subfolders)",
    )
    parser.add_argument(
        "--baseline",
        default="rule_based",
        help="Baseline controller used for pairwise significance checks",
    )
    parser.add_argument(
        "--permutations",
        type=int,
        default=5000,
        help="Permutation iterations for p-value estimation",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for permutation testing",
    )
    return parser.parse_args()


def _read_summary_values(summary_path: Path) -> Dict[str, np.ndarray]:
    rewards = []
    violations = []
    with summary_path.open("r", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        for row in reader:
            rewards.append(float(row.get("avg_reward", 0.0)))
            if "avg_sla_violations" in row:
                violations.append(float(row.get("avg_sla_violations", 0.0)))
            else:
                num_steps = float(row.get("num_steps", 1.0) or 1.0)
                total_v = float(row.get("total_sla_violations", row.get("violations", 0.0)))
                violations.append(total_v / max(num_steps, 1.0))

    return {
        "reward": np.array(rewards, dtype=np.float64),
        "violations": np.array(violations, dtype=np.float64),
    }


def _mean_std_ci(values: np.ndarray) -> Dict[str, float]:
    if values.size == 0:
        return {
            "mean": 0.0,
            "std": 0.0,
            "ci95_low": 0.0,
            "ci95_high": 0.0,
            "n": 0,
        }

    mean = float(np.mean(values))
    std = float(np.std(values, ddof=1)) if values.size > 1 else 0.0
    sem = std / math.sqrt(values.size) if values.size > 1 else 0.0
    margin = 1.96 * sem
    return {
        "mean": mean,
        "std": std,
        "ci95_low": mean - margin,
        "ci95_high": mean + margin,
        "n": int(values.size),
    }


def _welch_t_and_p(a: np.ndarray, b: np.ndarray) -> Dict[str, float]:
    if a.size < 2 or b.size < 2:
        return {"t_stat": 0.0, "p_value": 1.0}

    mean_a = float(np.mean(a))
    mean_b = float(np.mean(b))
    var_a = float(np.var(a, ddof=1))
    var_b = float(np.var(b, ddof=1))

    denom = math.sqrt((var_a / a.size) + (var_b / b.size))
    if denom == 0.0:
        return {"t_stat": 0.0, "p_value": 1.0}

    t_stat = (mean_a - mean_b) / denom

    # Use normal approximation for two-tailed p-value.
    p_value = 2.0 * (1.0 - NormalDist().cdf(abs(t_stat)))
    return {"t_stat": float(t_stat), "p_value": float(p_value)}


def _perm_pvalue(a: np.ndarray, b: np.ndarray, rng: np.random.Generator, iterations: int = 5000) -> float:
    if a.size == 0 or b.size == 0:
        return 1.0

    observed = abs(float(np.mean(a) - np.mean(b)))
    pooled = np.concatenate([a, b])
    n_a = a.size
    count = 0

    for _ in range(max(1, iterations)):
        rng.shuffle(pooled)
        diff = abs(float(np.mean(pooled[:n_a]) - np.mean(pooled[n_a:])))
        if diff >= observed:
            count += 1

    return float((count + 1) / (max(1, iterations) + 1))


def _scenario_dirs(root: Path) -> List[Path]:
    return sorted([path for path in root.iterdir() if path.is_dir()])


def main() -> int:
    args = parse_args()
    root = Path(args.root)
    root.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(args.seed)

    summary_rows = []
    significance_rows = []

    for scenario_dir in _scenario_dirs(root):
        controller_data: Dict[str, Dict[str, np.ndarray]] = {}

        for controller_dir in sorted([path for path in scenario_dir.iterdir() if path.is_dir()]):
            summary_path = controller_dir / "summary.csv"
            if summary_path.exists():
                controller_data[controller_dir.name] = _read_summary_values(summary_path)

        if not controller_data:
            continue

        for controller, values in controller_data.items():
            reward_stats = _mean_std_ci(values["reward"])
            viol_stats = _mean_std_ci(values["violations"])
            summary_rows.append(
                {
                    "scenario": scenario_dir.name,
                    "controller": controller,
                    "num_seeds": reward_stats["n"],
                    "reward_mean": reward_stats["mean"],
                    "reward_std": reward_stats["std"],
                    "reward_ci95_low": reward_stats["ci95_low"],
                    "reward_ci95_high": reward_stats["ci95_high"],
                    "violations_mean": viol_stats["mean"],
                    "violations_std": viol_stats["std"],
                    "violations_ci95_low": viol_stats["ci95_low"],
                    "violations_ci95_high": viol_stats["ci95_high"],
                }
            )

        baseline_values = controller_data.get(args.baseline)
        if baseline_values is None:
            continue

        for controller, values in controller_data.items():
            if controller == args.baseline:
                continue

            reward_t = _welch_t_and_p(values["reward"], baseline_values["reward"])
            viol_t = _welch_t_and_p(values["violations"], baseline_values["violations"])
            reward_perm = _perm_pvalue(values["reward"], baseline_values["reward"], rng, args.permutations)
            viol_perm = _perm_pvalue(values["violations"], baseline_values["violations"], rng, args.permutations)

            significance_rows.append(
                {
                    "scenario": scenario_dir.name,
                    "controller": controller,
                    "baseline": args.baseline,
                    "reward_t_stat": reward_t["t_stat"],
                    "reward_p_value_norm": reward_t["p_value"],
                    "reward_p_value_perm": reward_perm,
                    "viol_t_stat": viol_t["t_stat"],
                    "viol_p_value_norm": viol_t["p_value"],
                    "viol_p_value_perm": viol_perm,
                }
            )

    stats_dir = root / "_stats"
    stats_dir.mkdir(parents=True, exist_ok=True)

    summary_csv = stats_dir / "phase3_summary.csv"
    if summary_rows:
        with summary_csv.open("w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=list(summary_rows[0].keys()))
            writer.writeheader()
            writer.writerows(summary_rows)

    significance_csv = stats_dir / "phase3_significance.csv"
    if significance_rows:
        with significance_csv.open("w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=list(significance_rows[0].keys()))
            writer.writeheader()
            writer.writerows(significance_rows)

    report_json = stats_dir / "phase3_stats_report.json"
    with report_json.open("w", encoding="utf-8") as handle:
        json.dump(
            {
                "root": str(root),
                "baseline": args.baseline,
                "summary_rows": summary_rows,
                "significance_rows": significance_rows,
            },
            handle,
            indent=2,
        )

    print(f"Summary CSV      : {summary_csv}")
    print(f"Significance CSV : {significance_csv}")
    print(f"JSON report      : {report_json}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
