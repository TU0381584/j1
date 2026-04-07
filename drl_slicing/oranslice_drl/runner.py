import csv
import os
import subprocess
import time
from typing import Dict, List

from .collectors import MetricsCollector, PrometheusCollector
from .config import ExperimentConfig, SliceTarget
from .controllers import (
    Controller,
    DrlControllerStub,
    RandomController,
    RuleBasedSLAController,
    StaticController,
    DQNController,
    A2CController,
    PPOController,
)
from .policy_io import write_rrm_policy_json
from .reward import compute_sla_reward


def _assert_live_oai_stack_running() -> None:
    """Fail fast unless the live OAI E2E stack is currently running."""
    try:
        result = subprocess.run(
            ["docker", "ps", "--format", "{{.Names}}"],
            check=True,
            capture_output=True,
            text=True,
        )
    except FileNotFoundError as error:
        raise RuntimeError("Docker CLI not found. Live OAI E2E mode requires Docker.") from error
    except subprocess.CalledProcessError as error:
        raise RuntimeError("Failed to query Docker containers for live OAI preflight.") from error

    names = {line.strip() for line in result.stdout.splitlines() if line.strip()}
    required = ["amf", "smf", "upf", "oaignb"]
    missing = [name for name in required if name not in names]

    if missing:
        missing_txt = ", ".join(missing)
        raise RuntimeError(
            "Live OAI E2E stack is not up. Missing containers: "
            f"{missing_txt}. Start it first with: "
            "bash drl_slicing/scripts/run_closed_loop_sim.sh --ue-count 5 --active-algo dqn"
        )

    if not any(name.startswith("nr_ue_") for name in names):
        raise RuntimeError(
            "Live OAI E2E stack check failed: no OAI UE containers (nr_ue_*) are running. "
            "Start the stack with run_closed_loop_sim.sh before training."
        )


def _build_targets(config: ExperimentConfig) -> Dict[str, SliceTarget]:
    return {item.slice_id: item for item in config.slices}


def _build_collector(config: ExperimentConfig, slice_ids: List[str]) -> MetricsCollector:
    mode = config.collector.mode

    if mode != "prometheus":
        raise ValueError(
            "Replay/synthetic collectors are disabled. "
            "Use collector.mode=prometheus with the live OAI E2E stack."
        )

    _assert_live_oai_stack_running()

    if mode == "prometheus":
        return PrometheusCollector(
            config.collector.prometheus_url,
            config.collector.metric_queries,
            slice_ids,
            config.collector.step_seconds,
        )
    raise ValueError(f"Unknown collector mode: {mode}")


def _build_controller(config: ExperimentConfig, targets: Dict[str, SliceTarget]) -> Controller:
    mode = config.controller.mode
    if mode == "static":
        return StaticController(targets)
    if mode == "rule_based":
        return RuleBasedSLAController(targets, config.controller)
    if mode == "random":
        return RandomController(targets, config.random_seed)
    if mode == "drl_stub":
        return DrlControllerStub(targets, config.controller)
    if mode == "dqn":
        return DQNController(targets, config.controller, train=False)
    if mode == "dqn_train":
        return DQNController(targets, config.controller, train=True)
    if mode == "a2c":
        return A2CController(targets, config.controller, train=False)
    if mode == "a2c_train":
        return A2CController(targets, config.controller, train=True)
    if mode == "ppo":
        return PPOController(targets, config.controller, train=False)
    if mode == "ppo_train":
        return PPOController(targets, config.controller, train=True)
    raise ValueError(f"Unknown controller mode: {mode}")


def _append_run_log(path: str, record: dict) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    exists = os.path.exists(path)
    with open(path, "a", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(record.keys()))
        if not exists:
            writer.writeheader()
        writer.writerow(record)


def run_experiment(config: ExperimentConfig) -> None:
    targets = _build_targets(config)
    slice_ids = list(targets.keys())
    collector = _build_collector(config, slice_ids)
    controller = _build_controller(config, targets)

    for step in range(config.horizon_steps):
        state = collector.get_state(step)
        actions = controller.act(state)
        write_rrm_policy_json(config.output.policy_json_path, actions, targets)

        collector.on_action(actions, targets, step)

        done = step >= (config.horizon_steps - 1)
        next_step = step if done else step + 1
        next_state = collector.get_state(next_step)
        reward, violations = compute_sla_reward(next_state, targets)
        train_metrics = controller.train_step(reward, next_state, done)

        summary = ";".join(f"{a.slice_id}:min={a.min_ratio},max={a.max_ratio}" for a in actions)
        row = {
            "experiment": config.name,
            "step": step,
            "timestamp_s": state.timestamp_s,
            "reward": reward,
            "sla_violations": violations,
            "action_summary": summary,
        }
        if train_metrics:
            for key, value in train_metrics.items():
                row[f"train_{key}"] = value
        _append_run_log(config.output.run_log_csv, row)
        time.sleep(config.collector.step_seconds)
