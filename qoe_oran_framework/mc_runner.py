"""Monte-Carlo orchestration: runs a policy through RANEnv for N episodes,
optionally training it (offline) or evaluating frozen weights (live),
fixed seeds throughout, one Omega-tuple per step plus an episode-level
rollup, and a structured drift flag for live runs.
"""

import random
import time
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional, Tuple

import numpy as np

try:
    import torch
except ImportError:  # pragma: no cover
    torch = None

from ._oranslice_path import ensure_oranslice_drl_importable
from .comparators.lb_only_baseline import LB_ONLY_ROUTING_LIMITATION, LbOnlyHeuristic
from .config import SacLbExperimentConfig
from .env import RANEnv, encode_full_request_state, request_state_dim
from .omega_logger import OmegaLogger, OmegaTuple
from .policies.a2c_admission import A2CAdmissionPolicy
from .policies.dqn_admission import DQNAdmissionPolicy
from .policies.rainbow_admission import PrioritizedReplayBuffer, RainbowAdmissionPolicy
from .replay_kpm_source import KpmSource
from .types import AdmissionRequest

ensure_oranslice_drl_importable()
from oranslice_drl.drl_training import ReplayBuffer  # noqa: E402

REPLAY_BASED_ALGORITHMS = {"dqn", "rainbow"}
ON_POLICY_ALGORITHMS = {"a2c"}
NO_LEARNING_ALGORITHMS = {"lb_only"}

EPISODE_HORIZON_LIMITATION = (
    "episode horizon and step cadence follow Stage Zero's protocol (60 "
    "steps/episode, 5s/step live; offline training targets the full "
    "300-episode Table I schedule against a synthetic/replayed KPM feed), "
    "not a literal replay of the papers' original simulated training run."
)

REQUEST_BATCHING_LIMITATION = (
    "each pending request in a step is decided independently by "
    "concatenating a one-hot slice/gNB identity onto the shared step "
    "observation (see env.encode_request_context) and reusing the same "
    "policy instance per request, rather than the papers' one-request-per-step "
    "formulation -- see Stage Zero plan."
)


def set_seeds(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    if torch is not None:
        torch.manual_seed(seed)


@dataclass
class RunSummary:
    run_id: str
    algorithm: str
    mode: str
    seed: int
    n_episodes: int
    mean_reward_per_step: float
    mean_blocks_per_episode: float
    mean_urllc_blocks_per_episode: Optional[float]
    blocks_per_episode: List[int]
    blocks_per_episode_by_slice: List[Dict[str, int]]
    mean_rho: Optional[float]
    omega_path: str
    checkpoint_path: str = ""
    drift_flag: bool = False
    drift_reason: str = ""


def build_policy(algorithm: str, cfg: SacLbExperimentConfig, **overrides):
    dim = request_state_dim(cfg)
    if algorithm == "dqn":
        return DQNAdmissionPolicy(state_dim=dim, **overrides)
    if algorithm == "a2c":
        return A2CAdmissionPolicy(state_dim=dim, **overrides)
    if algorithm == "rainbow":
        return RainbowAdmissionPolicy(state_dim=dim, **overrides)
    if algorithm == "lb_only":
        return LbOnlyHeuristic(cfg, **overrides)
    raise ValueError(
        f"unknown algorithm {algorithm!r} (paper #1's SAC-only comparator is "
        "algorithm='dqn' run against a paper_variant: paper1 config -- see "
        "comparators/sac_only.py, not a separate algorithm string here)"
    )


def _select_actions(
    policy: Any,
    algorithm: str,
    pending: List[AdmissionRequest],
    obs: np.ndarray,
    cluster_state,
    cfg: SacLbExperimentConfig,
    training: bool,
) -> Tuple[List[int], List[Optional[np.ndarray]]]:
    if algorithm == "lb_only":
        actions = policy.decide(pending, cluster_state)
        return list(actions), [None] * len(pending)

    actions: List[int] = []
    request_states: List[Optional[np.ndarray]] = []
    for request in pending:
        req_state = encode_full_request_state(obs, request, cfg)
        action, _info = policy.select_action(req_state, training=training)
        actions.append(int(action))
        request_states.append(req_state)
    return actions, request_states


def _store_and_train(
    algorithm: str,
    policy: Any,
    replay_buffer: Any,
    pending: List[AdmissionRequest],
    obs: np.ndarray,
    next_obs: np.ndarray,
    actions: List[int],
    request_states: List[Optional[np.ndarray]],
    reward: float,
    done: bool,
    cfg: SacLbExperimentConfig,
    batch_size: int,
    warmup_transitions: int,
) -> None:
    if not pending:
        return

    if algorithm == "a2c":
        next_states = [encode_full_request_state(next_obs, req, cfg) for req in pending]
        batch = {
            "states": np.stack(request_states),
            "actions": np.array(actions, dtype=np.int64),
            "rewards": np.full(len(pending), reward, dtype=np.float32),
            "next_states": np.stack(next_states),
            "dones": np.full(len(pending), float(done), dtype=np.float32),
        }
        policy.train_step(batch)
        return

    if algorithm not in REPLAY_BASED_ALGORITHMS or replay_buffer is None:
        return

    for request, action, req_state in zip(pending, actions, request_states):
        next_state = encode_full_request_state(next_obs, request, cfg)
        if algorithm == "dqn":
            replay_buffer.add(req_state, action, reward, next_state, done, action_info=None)
        else:  # rainbow
            replay_buffer.add(req_state, action, reward, next_state, done)

    if len(replay_buffer) < max(batch_size, warmup_transitions):
        return

    if algorithm == "dqn":
        batch = replay_buffer.sample(batch_size)
        policy.train_step(batch)
    else:  # rainbow
        batch = replay_buffer.sample(batch_size, beta=policy.per_beta)
        metrics = policy.train_step(batch)
        replay_buffer.update_priorities(batch["indices"], metrics["td_errors"])


def _make_omega_tuple(
    algorithm: str,
    cfg: SacLbExperimentConfig,
    evidence: Dict[str, Any],
    extra_limitations: List[str],
    run_id: str,
    episode: int,
    step: int,
    timestamp_s: float,
    mode: str,
) -> OmegaTuple:
    objective = "minimize URLLC block rate subject to eMBB/mMTC SLA and PRB budget"
    constraint = f"gNB capacity B={cfg.B}, slices={list(cfg.slice_by_id)}"
    if len(cfg.gnbs) > 1:
        objective += ", and minimize cluster load imbalance (paper #2 LB term)"
        constraint += ", cluster load-balance ratio rho target [0.55, 0.60]"
    limitations = list(extra_limitations) + [EPISODE_HORIZON_LIMITATION, REQUEST_BATCHING_LIMITATION]
    if algorithm == "lb_only":
        limitations.append(LB_ONLY_ROUTING_LIMITATION)
    return OmegaTuple(
        role="admission-controller",
        method=algorithm,
        objective=objective,
        constraint=constraint,
        evidence=evidence,
        limitation="; ".join(limitations),
        run_id=run_id,
        episode=episode,
        step=step,
        timestamp_s=timestamp_s,
        mode=mode,
    )


def run_single(
    env: RANEnv,
    policy: Any,
    algorithm: str,
    omega: OmegaLogger,
    n_episodes: int,
    seed: int,
    run_id: str,
    mode: str,
    training: bool,
    cfg: SacLbExperimentConfig,
    batch_size: int = 16,
    replay_capacity: int = 10000,
    warmup_transitions: int = 32,
    extra_limitations: Optional[List[str]] = None,
) -> RunSummary:
    set_seeds(seed)
    extra_limitations = extra_limitations or []

    replay_buffer = None
    if training and algorithm == "dqn":
        replay_buffer = ReplayBuffer(capacity=replay_capacity)
    elif training and algorithm == "rainbow":
        replay_buffer = PrioritizedReplayBuffer(capacity=replay_capacity, alpha=policy.per_alpha)

    episode_mean_rewards: List[float] = []
    episode_block_totals: List[int] = []
    episode_block_by_slice: List[Dict[str, int]] = []
    episode_rho_means: List[Optional[float]] = []

    obs = env.reset()
    for episode_idx in range(1, n_episodes + 1):
        step_rewards: List[float] = []
        block_by_slice: Dict[str, int] = {}
        rho_values: List[float] = []
        step_idx = 0

        while True:
            step_idx += 1
            step_start = time.monotonic() if mode == "live_testbed" else None
            pending = env.pending_requests()
            cluster_state = env.last_cluster_state
            actions, request_states = _select_actions(
                policy, algorithm, pending, obs, cluster_state, cfg, training
            )

            result = env.step(actions)
            next_obs = result.obs

            if step_start is not None:
                # Respect the configured live step cadence -- without this,
                # LiveKpmSource's poll()/send_control() round-trip completes
                # in milliseconds and the loop would hammer the gNB's E2
                # agent far faster than the papers' Monte-Carlo protocol
                # (and this run's own episode.step_seconds config) intends.
                elapsed = time.monotonic() - step_start
                remaining = cfg.episode.step_seconds - elapsed
                if remaining > 0:
                    time.sleep(remaining)

            if training and algorithm not in NO_LEARNING_ALGORITHMS:
                _store_and_train(
                    algorithm, policy, replay_buffer, pending, obs, next_obs, actions,
                    request_states, result.reward, result.done, cfg, batch_size, warmup_transitions,
                )

            step_rewards.append(result.reward)
            for block in result.info["primary_blocks"]:
                block_by_slice[block["slice_id"]] = block_by_slice.get(block["slice_id"], 0) + 1
            rho_values.append(result.info["fairness_ratio"])

            evidence = {
                "seed": seed,
                "reward": result.reward,
                "primary_block_count": result.info["primary_block_count"],
                "secondary_block_count": result.info["secondary_block_count"],
                "accepted_counts": result.info["accepted_counts"],
                "fairness_ratio": result.info["fairness_ratio"],
                "n_pending": len(pending),
            }
            omega.log(
                _make_omega_tuple(
                    algorithm, cfg, evidence, list(result.info["limitations"]) + extra_limitations,
                    run_id, episode_idx, step_idx, float(result.info["global_step"]), mode,
                )
            )

            obs = next_obs
            if result.done:
                break

        if training and algorithm in ("dqn", "rainbow") and hasattr(policy, "on_episode_end"):
            policy.on_episode_end()
        if training and algorithm == "rainbow":
            policy.anneal_beta(episode_idx / n_episodes)

        episode_block_total = sum(block_by_slice.values())
        episode_mean_rewards.append(float(np.mean(step_rewards)) if step_rewards else 0.0)
        episode_block_totals.append(episode_block_total)
        episode_block_by_slice.append(block_by_slice)
        episode_rho_mean = float(np.mean(rho_values)) if len(cfg.gnbs) > 1 and rho_values else None
        episode_rho_means.append(episode_rho_mean)

        omega.log(
            _make_omega_tuple(
                algorithm, cfg,
                {
                    "seed": seed,
                    "episode_mean_reward": episode_mean_rewards[-1],
                    "episode_block_total": episode_block_total,
                    "episode_block_by_slice": block_by_slice,
                    "episode_mean_rho": episode_rho_mean,
                    "rollup": True,
                },
                extra_limitations,
                run_id, episode_idx, -1, float(episode_idx), mode,
            )
        )

    rho_vals = [r for r in episode_rho_means if r is not None]
    urllc_blocks = [d.get("urllc", 0) for d in episode_block_by_slice]

    summary = RunSummary(
        run_id=run_id,
        algorithm=algorithm,
        mode=mode,
        seed=seed,
        n_episodes=n_episodes,
        mean_reward_per_step=float(np.mean(episode_mean_rewards)) if episode_mean_rewards else 0.0,
        mean_blocks_per_episode=float(np.mean(episode_block_totals)) if episode_block_totals else 0.0,
        mean_urllc_blocks_per_episode=float(np.mean(urllc_blocks)) if urllc_blocks else None,
        blocks_per_episode=episode_block_totals,
        blocks_per_episode_by_slice=episode_block_by_slice,
        mean_rho=float(np.mean(rho_vals)) if rho_vals else None,
        omega_path=str(omega.path),
    )
    if mode == "live_testbed":
        summary.drift_flag, summary.drift_reason = flag_drift(summary)
    return summary


def flag_drift(
    summary: RunSummary,
    urllc_block_rate_threshold: float = 1.0,
    rho_band: Tuple[float, float] = (0.45, 0.70),
) -> Tuple[bool, str]:
    """Structured drift flag for live runs: block rate >2x the acceptance
    bar, or rho outside a wide margin around the [0.55,0.60] target."""
    reasons: List[str] = []
    if summary.mean_urllc_blocks_per_episode is not None and (
        summary.mean_urllc_blocks_per_episode > 2 * urllc_block_rate_threshold
    ):
        reasons.append(
            f"mean URLLC block rate {summary.mean_urllc_blocks_per_episode:.2f}/episode "
            f"exceeds 2x the {urllc_block_rate_threshold}/episode acceptance bar"
        )
    if summary.mean_rho is not None and not (rho_band[0] <= summary.mean_rho <= rho_band[1]):
        reasons.append(f"mean rho {summary.mean_rho:.3f} outside drift band {rho_band}")
    return (len(reasons) > 0, "; ".join(reasons))


def run_mc(
    cfg: SacLbExperimentConfig,
    algorithm: str,
    kpm_source_factory: Callable[[int], KpmSource],
    n_reps: int,
    episodes_per_rep: int,
    base_seed: int,
    mode: str,
    training: bool,
    results_dir: str,
    batch_size: int = 16,
    policy_factory: Optional[Callable[[int], Any]] = None,
) -> List[RunSummary]:
    import os

    summaries: List[RunSummary] = []
    for rep in range(n_reps):
        seed = base_seed + rep
        run_id = f"{algorithm}_{mode}_seed{seed}_rep{rep}"
        kpm_source = kpm_source_factory(seed)
        env = RANEnv(cfg, kpm_source, seed=seed)
        policy = policy_factory(seed) if policy_factory is not None else build_policy(algorithm, cfg)

        rep_dir = os.path.join(results_dir, algorithm, mode, f"rep_{rep}")
        omega_path = os.path.join(rep_dir, "omega_log.jsonl")
        with OmegaLogger(omega_path) as omega:
            summary = run_single(
                env, policy, algorithm, omega, episodes_per_rep, seed, run_id, mode, training, cfg,
                batch_size=batch_size,
            )

        if training and hasattr(policy, "save_checkpoint"):
            ckpt_path = os.path.join(rep_dir, "checkpoint.pt")
            policy.save_checkpoint(ckpt_path)
            summary.checkpoint_path = ckpt_path

        summaries.append(summary)
    return summaries
