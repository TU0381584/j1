from pathlib import Path

import pytest

from qoe_oran_framework.comparators.lb_only_baseline import LbOnlyHeuristic
from qoe_oran_framework.comparators.sac_only import build_sac_only_comparator, load_sac_only_config
from qoe_oran_framework.config import load_saclb_config
from qoe_oran_framework.env import request_state_dim
from qoe_oran_framework.types import AdmissionRequest, ClusterState, SliceAggState

CONFIGS_DIR = Path(__file__).resolve().parent.parent / "configs"


def test_load_sac_only_config_accepts_paper1_config():
    cfg = load_sac_only_config(str(CONFIGS_DIR / "saclb_paper1_sac_only.yaml"))
    assert cfg.paper_variant == "paper1"


def test_load_sac_only_config_rejects_paper2_config():
    with pytest.raises(ValueError):
        load_sac_only_config(str(CONFIGS_DIR / "saclb_offline_dqn.yaml"))


def test_build_sac_only_comparator_is_dqn_admission_policy():
    cfg = load_sac_only_config(str(CONFIGS_DIR / "saclb_paper1_sac_only.yaml"))
    policy = build_sac_only_comparator(str(CONFIGS_DIR / "saclb_paper1_sac_only.yaml"))
    assert policy.action_dim == 2
    assert policy.state_dim == request_state_dim(cfg)


def _cluster_state(utilisation, gnb_id="g0", slice_id="urllc"):
    return ClusterState(
        timestamp_s=0.0,
        per_gnb={gnb_id: {slice_id: SliceAggState(slice_id, gnb_id, prb_used_ratio=utilisation,
                                                    congestion_level=utilisation, queue_len_norm=0.0)}},
    )


def test_lb_only_admits_when_below_threshold():
    cfg = load_saclb_config(str(CONFIGS_DIR / "saclb_paper1_sac_only.yaml"))
    heuristic = LbOnlyHeuristic(cfg, utilization_threshold=0.97)
    req = AdmissionRequest("r1", "urllc", "g0", arrival_step=1)
    actions = heuristic.decide([req], _cluster_state(0.5))
    assert actions == [1]


def test_lb_only_rejects_when_at_or_above_threshold():
    cfg = load_saclb_config(str(CONFIGS_DIR / "saclb_paper1_sac_only.yaml"))
    heuristic = LbOnlyHeuristic(cfg, utilization_threshold=0.97)
    req = AdmissionRequest("r1", "urllc", "g0", arrival_step=1)
    actions = heuristic.decide([req], _cluster_state(0.98))
    assert actions == [0]


def test_lb_only_handles_unknown_slice_gnb_gracefully():
    cfg = load_saclb_config(str(CONFIGS_DIR / "saclb_paper1_sac_only.yaml"))
    heuristic = LbOnlyHeuristic(cfg)
    req = AdmissionRequest("r1", "urllc", "unknown-gnb", arrival_step=1)
    actions = heuristic.decide([req], ClusterState(timestamp_s=0.0))
    assert actions == [1]  # zero utilisation observed -> admit, no SLA awareness either way
