"""Contextless baseline: matches the papers' own stated baseline -- "a
classical LB heuristic ... allocates incoming slice requests to the base
station with the lowest current PRB utilisation ... does not account for
SLA priorities or learning-based optimisation" -- used because Stage
Zero's acceptance numbers are measured against *that* baseline in the
source papers, not an arbitrary random policy.

Simplification, documented: the papers' baseline *routes* a request to
whichever gNB currently has the lowest utilisation. Stage Zero's action
space is a fixed accept/reject decision on a request that already carries
a specific (slice, gNB) assignment (see env.py's _synthesize_requests) --
there is no "choose a different gNB" action. The closest faithful analogue
on this fixed action space is: admit unless the request's assigned gNB is
already at/above a saturation threshold, with zero SLA/priority awareness
-- which preserves the paper's core property (no learning, no SLA
weighting, purely utilisation-driven) even though it cannot literally
re-route.
"""

from typing import List

from ..config import SacLbExperimentConfig
from ..types import AdmissionRequest, ClusterState

LB_ONLY_ROUTING_LIMITATION = (
    "LbOnlyHeuristic approximates the papers' route-to-lowest-utilisation-gNB "
    "baseline as admit-unless-assigned-gNB-is-saturated, since Stage Zero's "
    "action space is accept/reject on a pre-assigned (slice, gNB) request, "
    "not gNB routing -- see lb_only_baseline.py module docstring."
)


class LbOnlyHeuristic:
    """Not an RLPolicy (no learning, no checkpoints) -- exposes decide(),
    called directly by mc_runner instead of select_action()."""

    def __init__(self, cfg: SacLbExperimentConfig, utilization_threshold: float = 0.97):
        self.cfg = cfg
        self.utilization_threshold = utilization_threshold

    def decide(self, pending: List[AdmissionRequest], cluster_state: ClusterState) -> List[int]:
        actions = []
        for request in pending:
            slice_states = cluster_state.per_gnb.get(request.gnb_id, {})
            agg = slice_states.get(request.slice_id)
            utilisation = agg.congestion_level if agg is not None else 0.0
            actions.append(0 if utilisation >= self.utilization_threshold else 1)
        return actions
