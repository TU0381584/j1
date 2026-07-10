import numpy as np

from qoe_oran_framework.replay_kpm_source import ClosedLoopKpmSource


def _served_and_backlog(source, gnb_id, slice_id, n_steps):
    served, backlog = [], []
    for _ in range(n_steps):
        samples = source.poll()
        group = [s for s in samples if s.gnb_id == gnb_id and s.nssai_sd == {"embb": 0, "urllc": 1, "mmtc": 2}[slice_id]]
        served.append(sum(s.avg_prbs_dl for s in group))
        backlog.append(sum(s.dl_mac_buffer_occupation for s in group))
    return served, backlog


def test_served_never_exceeds_ceiling():
    source = ClosedLoopKpmSource(
        seed=1, gnb_ids=["g0"], slice_ids=["urllc"],
        mean_offered_ratio={"urllc": 0.9}, initial_ceiling_ratio=100.0,
    )
    source.send_control("g0", 1, 1, min_ratio=5, max_ratio=20)  # sd=1 -> urllc
    served, _ = _served_and_backlog(source, "g0", "urllc", 20)
    assert max(served) <= 20.0 + 1e-6  # ceiling_prb = 20% of B=100


def test_low_ceiling_causes_backlog_growth():
    source = ClosedLoopKpmSource(
        seed=2, gnb_ids=["g0"], slice_ids=["urllc"],
        mean_offered_ratio={"urllc": 0.8}, offered_volatility=0.01,
    )
    source.send_control("g0", 1, 1, min_ratio=5, max_ratio=10)  # far below offered demand (~80)
    _, backlog = _served_and_backlog(source, "g0", "urllc", 30)
    assert backlog[-1] > backlog[0]
    assert backlog[-1] > 10  # genuinely accumulating, not noise


def test_raising_ceiling_relieves_backlog():
    source = ClosedLoopKpmSource(
        seed=3, gnb_ids=["g0"], slice_ids=["urllc"],
        mean_offered_ratio={"urllc": 0.3}, offered_volatility=0.01,
    )
    source.send_control("g0", 1, 1, min_ratio=5, max_ratio=10)  # below demand -> backlog builds
    _, backlog_low = _served_and_backlog(source, "g0", "urllc", 20)
    assert backlog_low[-1] > 0

    source.send_control("g0", 1, 1, min_ratio=5, max_ratio=100)  # relieve constraint
    _, backlog_relieved = _served_and_backlog(source, "g0", "urllc", 30)
    assert backlog_relieved[-1] < backlog_low[-1]


def test_bler_rises_with_backlog():
    source = ClosedLoopKpmSource(
        seed=4, gnb_ids=["g0"], slice_ids=["urllc"],
        mean_offered_ratio={"urllc": 0.9}, offered_volatility=0.01,
    )
    samples_before = source.poll()
    bler_before = np.mean([s.dl_bler for s in samples_before if s.nssai_sd == 1])

    source.send_control("g0", 1, 1, min_ratio=5, max_ratio=5)  # heavily constrain
    for _ in range(20):
        samples = source.poll()
    bler_after = np.mean([s.dl_bler for s in samples if s.nssai_sd == 1])
    assert bler_after > bler_before


def test_send_control_only_affects_matching_slice():
    source = ClosedLoopKpmSource(
        seed=5, gnb_ids=["g0"], slice_ids=["urllc", "embb"],
        mean_offered_ratio={"urllc": 0.9, "embb": 0.2},
    )
    source.send_control("g0", 1, 1, min_ratio=5, max_ratio=15)  # urllc only
    served, _ = _served_and_backlog(source, "g0", "embb", 10)
    # embb ceiling untouched (still 100), so served should track its low offered demand, not be clamped to 15
    assert max(served) > 15


def test_reproducible_with_fixed_seed():
    def run():
        source = ClosedLoopKpmSource(seed=42, gnb_ids=["g0"], slice_ids=["urllc"])
        source.send_control("g0", 1, 1, min_ratio=5, max_ratio=30)
        served, backlog = _served_and_backlog(source, "g0", "urllc", 15)
        return served, backlog

    served_a, backlog_a = run()
    served_b, backlog_b = run()
    assert served_a == served_b
    assert backlog_a == backlog_b


def test_unknown_gnb_or_slice_in_send_control_is_ignored():
    source = ClosedLoopKpmSource(seed=6, gnb_ids=["g0"], slice_ids=["urllc"])
    source.send_control("unknown-gnb", 1, 1, min_ratio=5, max_ratio=10)  # should not raise
    source.poll()
