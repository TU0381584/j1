from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List

import yaml


@dataclass
class SliceTarget:
    slice_id: str
    sst: int
    sd: int
    latency_budget_ms: float
    loss_budget_pct: float
    min_ratio_bounds: List[int]
    max_ratio_bounds: List[int]
    dedicated_ratio: int


@dataclass
class CollectorConfig:
    mode: str
    step_seconds: int
    csv_path: str
    prometheus_url: str
    metric_queries: Dict[str, str]


@dataclass
class ControllerConfig:
    mode: str
    alpha: float
    min_step: int
    max_step: int


@dataclass
class OutputConfig:
    policy_json_path: str
    run_log_csv: str


@dataclass
class ExperimentConfig:
    name: str
    horizon_steps: int
    random_seed: int
    slices: List[SliceTarget]
    collector: CollectorConfig
    controller: ControllerConfig
    output: OutputConfig


def _load_yaml(path: Path) -> dict:
    with path.open("r", encoding="utf-8") as handle:
        return yaml.safe_load(handle)


def load_experiment_config(config_path: str) -> ExperimentConfig:
    raw = _load_yaml(Path(config_path))
    slices = [
        SliceTarget(
            slice_id=item["slice_id"],
            sst=int(item["sst"]),
            sd=int(item["sd"]),
            latency_budget_ms=float(item["latency_budget_ms"]),
            loss_budget_pct=float(item["loss_budget_pct"]),
            min_ratio_bounds=[int(item["min_ratio_bounds"][0]), int(item["min_ratio_bounds"][1])],
            max_ratio_bounds=[int(item["max_ratio_bounds"][0]), int(item["max_ratio_bounds"][1])],
            dedicated_ratio=int(item["dedicated_ratio"]),
        )
        for item in raw["slices"]
    ]
    collector_cfg = raw["collector"]
    controller_cfg = raw["controller"]
    output_cfg = raw["output"]

    return ExperimentConfig(
        name=raw["name"],
        horizon_steps=int(raw["horizon_steps"]),
        random_seed=int(raw.get("random_seed", 42)),
        slices=slices,
        collector=CollectorConfig(
            mode=collector_cfg["mode"],
            step_seconds=int(collector_cfg.get("step_seconds", 1)),
            csv_path=collector_cfg.get("csv_path", ""),
            prometheus_url=collector_cfg.get("prometheus_url", ""),
            metric_queries=collector_cfg.get("metric_queries", {}),
        ),
        controller=ControllerConfig(
            mode=controller_cfg["mode"],
            alpha=float(controller_cfg.get("alpha", 0.2)),
            min_step=int(controller_cfg.get("min_step", 2)),
            max_step=int(controller_cfg.get("max_step", 5)),
        ),
        output=OutputConfig(
            policy_json_path=output_cfg["policy_json_path"],
            run_log_csv=output_cfg["run_log_csv"],
        ),
    )
