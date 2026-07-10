"""Reward implementing eq. (2) from paper #1 (MECON) and its LB-extended form
from paper #2 (ICRAIE):

  paper #1: r(s_t,a_t) = sum_k(omega_k * R_k - lambda_k) - mu * C_t * ||a_t||_1
  paper #2: r(s_t,a_t) = sum_k(omega_k * R_k - lambda_k) - mu * C_t * ||a_t||_1 - beta * imbalance_t

Interpretation notes (documented, not silent -- also mirrored into the
Omega-tuple `limitation` field of every run):

1. R_k is credited once per accepted request of slice type k this step
   (weighted by omega_k); lambda_k is charged once per step a slice is in
   SLA violation. The papers describe these at the level of "reward for
   serving" and "penalty for violating SLA" without specifying finer
   per-request attribution, so step-level, per-slice-type accounting is the
   most direct reading of the published formula.

2. ||a_t||_1 = number of accepted (admitted) requests this step -- this
   matches the papers' own gloss ("a_t||1 corresponds to the number of
   accepted requests").

3. The papers state F_t as "a fairness indicator ... the min/max PRB
   utilisation ratio", then write the reward term as "- beta * F_t".
   Subtracting the ratio directly would penalize GOOD balance (ratio near
   1.0), which contradicts the papers' own reported result that DRL agents
   achieve a HIGHER ratio than the LB-only baseline. We therefore implement
   the load-imbalance penalty as beta * (1 - fairness_ratio): this is the
   quantity whose minimization is consistent with the papers' narrative and
   results, and is our resolution of an ambiguity in the source text.
"""

from dataclasses import dataclass
from typing import Any, Dict

from .config import RewardWeights, SliceSpec
from .types import ClusterState

REWARD_INTERPRETATION_LIMITATION = (
    "reward eq.2's '-beta*F_t' term is implemented as beta*(1-fairness_ratio) "
    "(load imbalance), not a literal subtraction of the fairness ratio itself, "
    "because the latter would penalize good load balance -- see reward.py docstring."
)


@dataclass
class ViolationCheck:
    """Per-slice SLA violation flags for one step (cluster-wide, OR'd across gNBs)."""

    violated: Dict[str, bool]


def check_violations(cluster_state: ClusterState, slice_specs: Dict[str, SliceSpec]) -> ViolationCheck:
    violated: Dict[str, bool] = {slice_id: False for slice_id in slice_specs}
    for slice_states in cluster_state.per_gnb.values():
        for slice_id, agg in slice_states.items():
            if slice_id not in slice_specs:
                continue
            spec = slice_specs[slice_id]
            queue_violation = agg.queue_len_norm > 1.0
            loss_violation = agg.loss_proxy > spec.loss_budget_pct
            if queue_violation or loss_violation:
                violated[slice_id] = True
    return ViolationCheck(violated=violated)


def _cluster_mean_congestion(cluster_state: ClusterState) -> float:
    values = [
        agg.congestion_level
        for slice_states in cluster_state.per_gnb.values()
        for agg in slice_states.values()
    ]
    return sum(values) / len(values) if values else 0.0


def compute_step_reward(
    cluster_state: ClusterState,
    slice_specs: Dict[str, SliceSpec],
    accepted_counts: Dict[str, int],
    weights: RewardWeights,
    include_lb_term: bool,
) -> "tuple[float, Dict[str, Any]]":
    violations = check_violations(cluster_state, slice_specs)

    service_term = 0.0
    violation_term = 0.0
    for slice_id, spec in slice_specs.items():
        n_accepted = accepted_counts.get(slice_id, 0)
        service_term += spec.priority_weight * spec.accept_reward * n_accepted
        if violations.violated.get(slice_id, False):
            violation_term += spec.violation_penalty

    total_accepted = sum(accepted_counts.values())
    mean_congestion = _cluster_mean_congestion(cluster_state)
    congestion_term = weights.congestion_coeff * mean_congestion * total_accepted

    reward = service_term - violation_term - congestion_term

    imbalance = None
    lb_term = 0.0
    if include_lb_term:
        imbalance = 1.0 - cluster_state.fairness_ratio
        lb_term = weights.lb_coeff * imbalance
        reward -= lb_term

    info: Dict[str, Any] = {
        "service_term": service_term,
        "violation_term": violation_term,
        "congestion_term": congestion_term,
        "lb_term": lb_term,
        "imbalance": imbalance,
        "mean_congestion": mean_congestion,
        "total_accepted": total_accepted,
        "violated_slices": {k: v for k, v in violations.violated.items() if v},
        "reward": reward,
    }
    if include_lb_term:
        info["limitation"] = REWARD_INTERPRETATION_LIMITATION
    return reward, info
