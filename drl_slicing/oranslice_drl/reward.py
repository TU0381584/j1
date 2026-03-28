from typing import Dict, Tuple

from .config import SliceTarget
from .types import EnvState


def compute_sla_reward(state: EnvState, targets: Dict[str, SliceTarget]) -> Tuple[float, int]:
    total_penalty = 0.0
    efficiency_bonus = 0.0
    violations = 0

    for slice_id, metrics in state.slices.items():
        target = targets[slice_id]
        latency_over = max(0.0, metrics.latency_ms - target.latency_budget_ms)
        loss_over = max(0.0, metrics.loss_pct - target.loss_budget_pct)
        if latency_over > 0.0 or loss_over > 0.0:
            violations += 1
        penalty = 2.0 * (latency_over / max(target.latency_budget_ms, 1e-6)) + 3.0 * (
            loss_over / max(target.loss_budget_pct, 1e-6)
        )
        total_penalty += penalty

        offered = max(metrics.offered_load_kbps, 1.0)
        efficiency_bonus += min(metrics.throughput_kbps / offered, 1.0)

    reward = efficiency_bonus - total_penalty
    return reward, violations
