from qoe_oran_framework.config import RewardWeights, SliceSpec
from qoe_oran_framework.reward import check_violations, compute_step_reward
from qoe_oran_framework.types import ClusterState, SliceAggState

SPECS = {
    "urllc": SliceSpec("urllc", 1, 1, 30, 5, 60, 15, 0.8, 4.0, 1.0, 2.0),
    "embb": SliceSpec("embb", 1, 0, 60, 10, 90, 45, 2.0, 2.0, 1.0, 1.0),
}


def _state(urllc_queue=0.0, urllc_loss=0.0, fairness=1.0):
    return ClusterState(
        timestamp_s=0.0,
        per_gnb={
            "g0": {
                "urllc": SliceAggState("urllc", "g0", prb_used_ratio=0.3, congestion_level=0.3,
                                        queue_len_norm=urllc_queue, loss_proxy=urllc_loss),
                "embb": SliceAggState("embb", "g0", prb_used_ratio=0.2, congestion_level=0.3,
                                       queue_len_norm=0.0, loss_proxy=0.0),
            }
        },
        fairness_ratio=fairness,
    )


def test_no_violation_when_within_budget():
    v = check_violations(_state(urllc_queue=0.5, urllc_loss=0.1), SPECS)
    assert v.violated["urllc"] is False


def test_queue_violation_flags_slice():
    v = check_violations(_state(urllc_queue=1.5), SPECS)
    assert v.violated["urllc"] is True


def test_loss_violation_flags_slice():
    v = check_violations(_state(urllc_loss=5.0), SPECS)  # > loss_budget_pct=0.8
    assert v.violated["urllc"] is True


def test_paper1_reward_has_no_lb_term():
    state = _state(fairness=0.2)  # badly imbalanced, must not matter for paper1
    weights = RewardWeights(congestion_coeff=1.0, lb_coeff=5.0)
    reward, info = compute_step_reward(
        state, SPECS, accepted_counts={"urllc": 1, "embb": 0}, weights=weights, include_lb_term=False
    )
    assert info["lb_term"] == 0.0
    assert info["imbalance"] is None
    assert "limitation" not in info


def test_paper2_reward_penalizes_imbalance_not_fairness_directly():
    weights = RewardWeights(congestion_coeff=0.0, lb_coeff=1.0)
    balanced = _state(fairness=1.0)
    imbalanced = _state(fairness=0.2)
    r_balanced, info_balanced = compute_step_reward(
        balanced, SPECS, accepted_counts={}, weights=weights, include_lb_term=True
    )
    r_imbalanced, info_imbalanced = compute_step_reward(
        imbalanced, SPECS, accepted_counts={}, weights=weights, include_lb_term=True
    )
    # perfectly balanced (fairness=1.0) must score >= imbalanced (fairness=0.2):
    # this is the guard against the eq.2 ambiguity noted in reward.py.
    assert r_balanced > r_imbalanced
    assert info_balanced["imbalance"] == 0.0
    assert info_imbalanced["imbalance"] == 0.8
    assert "limitation" in info_imbalanced


def test_service_term_scales_with_priority_weight_and_accepted_count():
    reward, info = compute_step_reward(
        _state(), SPECS, accepted_counts={"urllc": 2, "embb": 0},
        weights=RewardWeights(congestion_coeff=0.0, lb_coeff=0.0), include_lb_term=False,
    )
    # urllc: omega=4.0, R_k=1.0, n=2 -> service_term=8.0
    assert info["service_term"] == 8.0


def test_congestion_term_scales_with_accepted_count_and_mean_congestion():
    reward, info = compute_step_reward(
        _state(), SPECS, accepted_counts={"urllc": 3, "embb": 0},
        weights=RewardWeights(congestion_coeff=2.0, lb_coeff=0.0), include_lb_term=False,
    )
    # mean_congestion over the 2 slice entries = 0.3, total_accepted = 3
    assert info["congestion_term"] == 2.0 * 0.3 * 3
