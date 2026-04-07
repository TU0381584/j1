import random
from abc import ABC, abstractmethod
from typing import Dict, List
import numpy as np

from .config import ControllerConfig, SliceTarget
from .types import EnvState, SliceAction
from .drl_policy import DQNPolicy, A2CPolicy, PPOPolicy, RLPolicy
from .drl_training import StateEncoder, ActionDecoder, DRLTrainer


class Controller(ABC):
    @abstractmethod
    def act(self, state: EnvState) -> List[SliceAction]:
        raise NotImplementedError

    def train_step(self, reward: float, next_state: EnvState, done: bool = False) -> Dict:
        """Optional training callback; no-op for non-learning controllers."""
        return {}
    
    def get_training_info(self) -> Dict:
        """Return training metrics/info if applicable."""
        return {}


class StaticController(Controller):
    def __init__(self, targets: Dict[str, SliceTarget]) -> None:
        self.targets = targets

    def act(self, state: EnvState) -> List[SliceAction]:
        actions: List[SliceAction] = []
        for target in self.targets.values():
            actions.append(
                SliceAction(
                    slice_id=target.slice_id,
                    min_ratio=target.min_ratio_bounds[0],
                    max_ratio=target.max_ratio_bounds[1],
                    dedicated_ratio=target.dedicated_ratio,
                )
            )
        return actions


class RuleBasedSLAController(Controller):
    def __init__(self, targets: Dict[str, SliceTarget], cfg: ControllerConfig) -> None:
        self.targets = targets
        self.cfg = cfg
        self.current_min: Dict[str, int] = {k: v.min_ratio_bounds[0] for k, v in targets.items()}
        self.current_max: Dict[str, int] = {k: v.max_ratio_bounds[1] for k, v in targets.items()}

    def _compute_adjustment(self, violation_strength: float) -> int:
        if violation_strength <= 0:
            return -self.cfg.min_step
        return min(self.cfg.max_step, max(self.cfg.min_step, int(violation_strength * self.cfg.max_step)))

    def act(self, state: EnvState) -> List[SliceAction]:
        actions: List[SliceAction] = []
        for slice_id, metrics in state.slices.items():
            target = self.targets[slice_id]
            latency_over = max(0.0, (metrics.latency_ms - target.latency_budget_ms) / max(target.latency_budget_ms, 1e-6))
            loss_over = max(0.0, (metrics.loss_pct - target.loss_budget_pct) / max(target.loss_budget_pct, 1e-6))
            violation_strength = 0.6 * latency_over + 0.4 * loss_over
            delta = self._compute_adjustment(violation_strength)

            next_min = self.current_min[slice_id] + delta
            next_max = self.current_max[slice_id] + delta

            next_min = max(target.min_ratio_bounds[0], min(target.min_ratio_bounds[1], next_min))
            next_max = max(target.max_ratio_bounds[0], min(target.max_ratio_bounds[1], next_max))
            if next_min > next_max:
                next_min = next_max

            self.current_min[slice_id] = next_min
            self.current_max[slice_id] = next_max

            actions.append(
                SliceAction(
                    slice_id=slice_id,
                    min_ratio=next_min,
                    max_ratio=next_max,
                    dedicated_ratio=target.dedicated_ratio,
                )
            )
        return actions


class RandomController(Controller):
    def __init__(self, targets: Dict[str, SliceTarget], seed: int) -> None:
        self.targets = targets
        self.rng = random.Random(seed)

    def act(self, state: EnvState) -> List[SliceAction]:
        actions: List[SliceAction] = []
        for target in self.targets.values():
            min_ratio = self.rng.randint(target.min_ratio_bounds[0], target.min_ratio_bounds[1])
            max_ratio = self.rng.randint(max(min_ratio, target.max_ratio_bounds[0]), target.max_ratio_bounds[1])
            actions.append(
                SliceAction(
                    slice_id=target.slice_id,
                    min_ratio=min_ratio,
                    max_ratio=max_ratio,
                    dedicated_ratio=target.dedicated_ratio,
                )
            )
        return actions


class DrlControllerStub(RuleBasedSLAController):
    def __init__(self, targets: Dict[str, SliceTarget], cfg: ControllerConfig) -> None:
        super().__init__(targets, cfg)


def _apply_offline_warm_start(
    policy: RLPolicy,
    cfg: ControllerConfig,
    state_dim: int,
    action_dim: int,
) -> Dict:
    """Apply offline behavior-cloning warm-start if enabled in config."""
    if not cfg.offline_warm_start:
        return {}

    source_path = cfg.warm_start_dataset_path.strip()
    if not source_path:
        raise ValueError(
            "offline_warm_start is enabled, but controller.warm_start_dataset_path is empty."
        )

    from .offline_warmstart import load_state_action_dataset

    states, actions = load_state_action_dataset(
        source_path=source_path,
        expected_state_dim=state_dim,
        max_samples=max(0, int(cfg.warm_start_max_samples)),
    )

    metrics = policy.behavior_clone_train(
        states=states,
        actions=actions,
        epochs=max(1, int(cfg.warm_start_epochs)),
        batch_size=max(1, int(cfg.warm_start_batch_size)),
    )
    metrics.update(
        {
            "source": source_path,
            "samples_loaded": int(states.shape[0]),
            "epochs": int(max(1, int(cfg.warm_start_epochs))),
            "batch_size": int(max(1, int(cfg.warm_start_batch_size))),
        }
    )
    return metrics


class DQNController(Controller):
    """
    DRL controller using Deep Q-Network policy.
    Optionally supports training when replay buffer is updated.
    """

    def __init__(
        self,
        targets: Dict[str, SliceTarget],
        cfg: ControllerConfig,
        learning_rate: float = 1e-3,
        train: bool = False,
        device: str = "cpu",
    ):
        self.targets = targets
        self.cfg = cfg
        self.train_mode = train

        self.state_encoder = StateEncoder(targets)
        self.action_decoder = ActionDecoder(targets)

        # Initialize DQN policy
        self.policy = DQNPolicy(
            state_dim=self.state_encoder.state_dim,
            action_dim=self.action_decoder.action_dim,
            hidden_dim=128,
            learning_rate=learning_rate,
            gamma=0.99,
            epsilon_start=1.0 if train else 0.05,
            epsilon_end=0.05,
            epsilon_decay=0.995,
            device=device,
        )

        # Trainer for online learning
        if train:
            self.trainer = DRLTrainer(
                self.policy,
                self.state_encoder,
                self.action_decoder,
                batch_size=32,
                update_freq=4,
                warmup_steps=100,
            )
        else:
            self.trainer = None

        self._warm_start_metrics: Dict = {}
        self._warm_start_emitted = False
        if train and cfg.offline_warm_start:
            self._warm_start_metrics = _apply_offline_warm_start(
                policy=self.policy,
                cfg=cfg,
                state_dim=self.state_encoder.state_dim,
                action_dim=self.action_decoder.action_dim,
            )

        self._last_state = None
        self._last_action_idx = None

    def act(self, state: EnvState) -> List[SliceAction]:
        """
        Select actions using DQN policy.
        """
        state_vec = self.state_encoder.encode(state)
        action_idx, q_value = self.policy.select_action(state_vec, training=self.train_mode)
        self._last_state = state
        self._last_action_idx = int(action_idx)
        allocation = self.action_decoder.decode(action_idx)

        actions: List[SliceAction] = []
        for slice_id, alloc in allocation.items():
            target = self.targets[slice_id]
            actions.append(
                SliceAction(
                    slice_id=slice_id,
                    min_ratio=alloc["min_ratio"],
                    max_ratio=alloc["max_ratio"],
                    dedicated_ratio=target.dedicated_ratio,
                )
            )
        return actions

    def train_step(self, reward: float, next_state: EnvState, done: bool = False) -> Dict:
        """
        Update policy with experience (requires previous state/action).
        Call this after each environment step.
        """
        if not self.train_mode or self.trainer is None:
            return {}
        if self._last_state is None or self._last_action_idx is None:
            return {}

        self.trainer.step(
            self._last_state,
            self._last_action_idx,
            reward,
            next_state,
            done,
        )
        metrics = self.trainer.get_metrics()
        if self._warm_start_metrics and not self._warm_start_emitted:
            metrics.update({f"warm_start_{k}": v for k, v in self._warm_start_metrics.items()})
            self._warm_start_emitted = True
        return metrics

    def record_transition(self, state: EnvState, action_list: List[SliceAction]) -> None:
        """Record current state/action for next train_step call."""
        self._last_state = state
        # Convert action list back to action index (for replay buffer)
        # This is a simplification; ideally store the index from act()
        self._last_action = 0

    def get_training_info(self) -> Dict:
        """Return current training metrics."""
        if self.trainer is None:
            return {}
        info = {
            "epsilon": self.policy.epsilon,
            "step_count": self.policy.train_step_count,
            **self.trainer.get_metrics(),
        }
        if self._warm_start_metrics:
            info.update({f"warm_start_{k}": v for k, v in self._warm_start_metrics.items()})
        return info

    def save_checkpoint(self, path: str) -> None:
        self.policy.save_checkpoint(path)

    def load_checkpoint(self, path: str) -> None:
        self.policy.load_checkpoint(path)


class A2CController(Controller):
    """
    DRL controller using Advantage Actor-Critic policy.
    Implementation to be completed in Phase 2b.
    """

    def __init__(
        self,
        targets: Dict[str, SliceTarget],
        cfg: ControllerConfig,
        learning_rate: float = 1e-3,
        train: bool = False,
        device: str = "cpu",
    ):
        self.targets = targets
        self.cfg = cfg
        self.train_mode = train

        self.state_encoder = StateEncoder(targets)
        self.action_decoder = ActionDecoder(targets)

        # Initialize A2C policy
        self.policy = A2CPolicy(
            state_dim=self.state_encoder.state_dim,
            action_dim=self.action_decoder.action_dim,
            hidden_dim=128,
            learning_rate=learning_rate,
            gamma=0.99,
            device=device,
        )

        self.trainer = None
        if train:
            from .drl_training import DRLTrainer
            self.trainer = DRLTrainer(
                self.policy,
                self.state_encoder,
                self.action_decoder,
                batch_size=32,
                update_freq=4,
                warmup_steps=100,
            )

        self._warm_start_metrics: Dict = {}
        self._warm_start_emitted = False
        if train and cfg.offline_warm_start:
            self._warm_start_metrics = _apply_offline_warm_start(
                policy=self.policy,
                cfg=cfg,
                state_dim=self.state_encoder.state_dim,
                action_dim=self.action_decoder.action_dim,
            )

        self._last_state = None
        self._last_action_idx = None

    def act(self, state: EnvState) -> List[SliceAction]:
        """
        Select actions using A2C policy.
        """
        state_vec = self.state_encoder.encode(state)
        action_idx, value = self.policy.select_action(state_vec, training=self.train_mode)
        self._last_state = state
        self._last_action_idx = int(action_idx)
        allocation = self.action_decoder.decode(action_idx)

        actions: List[SliceAction] = []
        for slice_id, alloc in allocation.items():
            target = self.targets[slice_id]
            actions.append(
                SliceAction(
                    slice_id=slice_id,
                    min_ratio=alloc["min_ratio"],
                    max_ratio=alloc["max_ratio"],
                    dedicated_ratio=target.dedicated_ratio,
                )
            )
        return actions

    def train_step(self, reward: float, next_state: EnvState, done: bool = False) -> Dict:
        """Update A2C policy with latest transition if training mode is enabled."""
        if not self.train_mode or self.trainer is None:
            return {}
        if self._last_state is None or self._last_action_idx is None:
            return {}

        self.trainer.step(
            self._last_state,
            self._last_action_idx,
            reward,
            next_state,
            done,
        )
        metrics = self.trainer.get_metrics()
        if self._warm_start_metrics and not self._warm_start_emitted:
            metrics.update({f"warm_start_{k}": v for k, v in self._warm_start_metrics.items()})
            self._warm_start_emitted = True
        return metrics

    def get_training_info(self) -> Dict:
        """Return current training metrics."""
        if self.trainer is None:
            return {}
        info = {
            "step_count": self.policy.train_step_count,
            **self.trainer.get_metrics(),
        }
        if self._warm_start_metrics:
            info.update({f"warm_start_{k}": v for k, v in self._warm_start_metrics.items()})
        return info

    def save_checkpoint(self, path: str) -> None:
        self.policy.save_checkpoint(path)

    def load_checkpoint(self, path: str) -> None:
        self.policy.load_checkpoint(path)


class PPOController(Controller):
    """DRL controller using PPO policy."""

    def __init__(
        self,
        targets: Dict[str, SliceTarget],
        cfg: ControllerConfig,
        learning_rate: float = 3e-4,
        train: bool = False,
        device: str = "cpu",
    ):
        self.targets = targets
        self.cfg = cfg
        self.train_mode = train

        self.state_encoder = StateEncoder(targets)
        self.action_decoder = ActionDecoder(targets)

        self.policy = PPOPolicy(
            state_dim=self.state_encoder.state_dim,
            action_dim=self.action_decoder.action_dim,
            hidden_dim=128,
            learning_rate=learning_rate,
            gamma=0.99,
            clip_epsilon=0.2,
            value_coef=0.5,
            entropy_coef=0.01,
            device=device,
        )

        self.trainer = None
        if train:
            self.trainer = DRLTrainer(
                self.policy,
                self.state_encoder,
                self.action_decoder,
                batch_size=32,
                update_freq=4,
                warmup_steps=100,
            )

        self._warm_start_metrics: Dict = {}
        self._warm_start_emitted = False
        if train and cfg.offline_warm_start:
            self._warm_start_metrics = _apply_offline_warm_start(
                policy=self.policy,
                cfg=cfg,
                state_dim=self.state_encoder.state_dim,
                action_dim=self.action_decoder.action_dim,
            )

        self._last_state = None
        self._last_action_idx = None
        self._last_log_prob = None

    def act(self, state: EnvState) -> List[SliceAction]:
        state_vec = self.state_encoder.encode(state)
        action_idx, log_prob = self.policy.select_action(state_vec, training=self.train_mode)
        self._last_state = state
        self._last_action_idx = int(action_idx)
        self._last_log_prob = float(log_prob)

        allocation = self.action_decoder.decode(action_idx)
        actions: List[SliceAction] = []
        for slice_id, alloc in allocation.items():
            target = self.targets[slice_id]
            actions.append(
                SliceAction(
                    slice_id=slice_id,
                    min_ratio=alloc["min_ratio"],
                    max_ratio=alloc["max_ratio"],
                    dedicated_ratio=target.dedicated_ratio,
                )
            )
        return actions

    def train_step(self, reward: float, next_state: EnvState, done: bool = False) -> Dict:
        if not self.train_mode or self.trainer is None:
            return {}
        if self._last_state is None or self._last_action_idx is None:
            return {}

        self.trainer.step(
            self._last_state,
            self._last_action_idx,
            reward,
            next_state,
            done,
            action_info=self._last_log_prob,
        )
        metrics = self.trainer.get_metrics()
        if self._warm_start_metrics and not self._warm_start_emitted:
            metrics.update({f"warm_start_{k}": v for k, v in self._warm_start_metrics.items()})
            self._warm_start_emitted = True
        return metrics

    def get_training_info(self) -> Dict:
        if self.trainer is None:
            return {}
        info = {
            "step_count": self.policy.train_step_count,
            **self.trainer.get_metrics(),
        }
        if self._warm_start_metrics:
            info.update({f"warm_start_{k}": v for k, v in self._warm_start_metrics.items()})
        return info

    def save_checkpoint(self, path: str) -> None:
        self.policy.save_checkpoint(path)

    def load_checkpoint(self, path: str) -> None:
        self.policy.load_checkpoint(path)
