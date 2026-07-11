import time
from pathlib import Path

from qoe_oran_framework.config import load_saclb_config
from qoe_oran_framework.mc_runner import build_policy, flag_drift, run_mc, run_single, RunSummary
from qoe_oran_framework.omega_logger import OmegaLogger, read_omega_jsonl
from qoe_oran_framework.env import RANEnv
from qoe_oran_framework.replay_kpm_source import SyntheticKpmSource

CONFIGS_DIR = Path(__file__).resolve().parent.parent / "configs"


def _cfg(paper2=True):
    name = "saclb_offline_dqn.yaml" if paper2 else "saclb_paper1_sac_only.yaml"
    return load_saclb_config(str(CONFIGS_DIR / name))


def test_build_policy_dispatch():
    cfg = _cfg()
    assert build_policy("dqn", cfg).action_dim == 2
    assert build_policy("a2c", cfg).action_dim == 2
    assert build_policy("rainbow", cfg).action_dim == 2
    assert hasattr(build_policy("lb_only", cfg), "decide")


def test_run_single_dqn_training_two_episodes(tmp_path):
    cfg = _cfg()
    cfg.episode.steps_per_episode = 5
    source = SyntheticKpmSource(seed=1, gnb_ids=cfg.gnb_ids, slice_ids=list(cfg.slice_by_id))
    env = RANEnv(cfg, source, seed=1)
    policy = build_policy("dqn", cfg)
    omega_path = str(tmp_path / "omega.jsonl")
    with OmegaLogger(omega_path) as omega:
        summary = run_single(
            env, policy, "dqn", omega, n_episodes=2, seed=1, run_id="test-run",
            mode="offline_synthetic", training=True, cfg=cfg, batch_size=4, warmup_transitions=4,
        )
    assert summary.n_episodes == 2
    assert len(summary.blocks_per_episode) == 2
    assert summary.mean_rho is not None  # multi-gNB config

    rows = read_omega_jsonl(omega_path)
    assert len(rows) > 0
    assert all(row["limitation"] for row in rows)
    assert all(row["mode"] == "offline_synthetic" for row in rows)


def test_run_single_lb_only_requires_no_training(tmp_path):
    cfg = _cfg()
    cfg.episode.steps_per_episode = 5
    source = SyntheticKpmSource(seed=2, gnb_ids=cfg.gnb_ids, slice_ids=list(cfg.slice_by_id))
    env = RANEnv(cfg, source, seed=2)
    policy = build_policy("lb_only", cfg)
    omega_path = str(tmp_path / "omega.jsonl")
    with OmegaLogger(omega_path) as omega:
        summary = run_single(
            env, policy, "lb_only", omega, n_episodes=1, seed=2, run_id="test-lb",
            mode="offline_synthetic", training=False, cfg=cfg,
        )
    assert summary.algorithm == "lb_only"
    assert summary.checkpoint_path == ""


def test_run_single_a2c_on_policy(tmp_path):
    cfg = _cfg()
    cfg.episode.steps_per_episode = 5
    source = SyntheticKpmSource(seed=3, gnb_ids=cfg.gnb_ids, slice_ids=list(cfg.slice_by_id))
    env = RANEnv(cfg, source, seed=3)
    policy = build_policy("a2c", cfg)
    omega_path = str(tmp_path / "omega.jsonl")
    with OmegaLogger(omega_path) as omega:
        summary = run_single(
            env, policy, "a2c", omega, n_episodes=2, seed=3, run_id="test-a2c",
            mode="offline_synthetic", training=True, cfg=cfg,
        )
    assert summary.n_episodes == 2


def test_run_single_rainbow_per_updates(tmp_path):
    cfg = _cfg()
    cfg.episode.steps_per_episode = 5
    source = SyntheticKpmSource(seed=4, gnb_ids=cfg.gnb_ids, slice_ids=list(cfg.slice_by_id))
    env = RANEnv(cfg, source, seed=4)
    policy = build_policy("rainbow", cfg)
    beta_before = policy.per_beta
    omega_path = str(tmp_path / "omega.jsonl")
    with OmegaLogger(omega_path) as omega:
        run_single(
            env, policy, "rainbow", omega, n_episodes=3, seed=4, run_id="test-rainbow",
            mode="offline_synthetic", training=True, cfg=cfg, batch_size=4, warmup_transitions=4,
        )
    assert policy.per_beta > beta_before  # anneal_beta called each episode


def test_run_single_propagates_extra_limitations(tmp_path):
    cfg = _cfg(paper2=False)
    cfg.episode.steps_per_episode = 3
    source = SyntheticKpmSource(seed=11, gnb_ids=cfg.gnb_ids, slice_ids=list(cfg.slice_by_id))
    env = RANEnv(cfg, source, seed=11)
    policy = build_policy("dqn", cfg)
    omega_path = str(tmp_path / "omega.jsonl")
    with OmegaLogger(omega_path) as omega:
        run_single(
            env, policy, "dqn", omega, n_episodes=1, seed=11, run_id="test-extra",
            mode="live_testbed", training=False, cfg=cfg,
            extra_limitations=["single physical gNB only"],
        )
    rows = read_omega_jsonl(omega_path)
    assert all("single physical gNB only" in row["limitation"] for row in rows)


def test_live_mode_respects_step_cadence(tmp_path):
    cfg = _cfg(paper2=False)
    cfg.episode.steps_per_episode = 3
    cfg.episode.step_seconds = 0.1
    source = SyntheticKpmSource(seed=12, gnb_ids=cfg.gnb_ids, slice_ids=list(cfg.slice_by_id))
    env = RANEnv(cfg, source, seed=12)
    policy = build_policy("lb_only", cfg)
    omega_path = str(tmp_path / "omega.jsonl")
    start = time.monotonic()
    with OmegaLogger(omega_path) as omega:
        run_single(
            env, policy, "lb_only", omega, n_episodes=1, seed=12, run_id="test-cadence",
            mode="live_testbed", training=False, cfg=cfg,
        )
    elapsed = time.monotonic() - start
    assert elapsed >= 0.3 * 0.9  # 3 steps * 0.1s, small tolerance for scheduling jitter


def test_offline_mode_does_not_sleep(tmp_path):
    cfg = _cfg(paper2=False)
    cfg.episode.steps_per_episode = 3
    cfg.episode.step_seconds = 5.0  # would be 15s if cadence applied -- must not apply offline
    source = SyntheticKpmSource(seed=13, gnb_ids=cfg.gnb_ids, slice_ids=list(cfg.slice_by_id))
    env = RANEnv(cfg, source, seed=13)
    policy = build_policy("lb_only", cfg)
    omega_path = str(tmp_path / "omega.jsonl")
    start = time.monotonic()
    with OmegaLogger(omega_path) as omega:
        run_single(
            env, policy, "lb_only", omega, n_episodes=1, seed=13, run_id="test-no-cadence",
            mode="offline_synthetic", training=False, cfg=cfg,
        )
    elapsed = time.monotonic() - start
    assert elapsed < 2.0


def test_run_mc_produces_one_summary_per_rep(tmp_path):
    cfg = _cfg()
    cfg.episode.steps_per_episode = 5

    def factory(seed):
        return SyntheticKpmSource(seed=seed, gnb_ids=cfg.gnb_ids, slice_ids=list(cfg.slice_by_id))

    summaries = run_mc(
        cfg, "dqn", factory, n_reps=2, episodes_per_rep=1, base_seed=10,
        mode="offline_synthetic", training=True, results_dir=str(tmp_path), batch_size=4,
    )
    assert len(summaries) == 2
    assert summaries[0].seed == 10
    assert summaries[1].seed == 11
    assert summaries[0].checkpoint_path != ""
    assert Path(summaries[0].checkpoint_path).exists()


class _TrackingKpmSource(SyntheticKpmSource):
    close_count = 0  # class-level: shared across instances created by the factory

    def close(self):
        _TrackingKpmSource.close_count += 1
        super().close()


def test_run_mc_closes_env_between_reps(tmp_path):
    cfg = _cfg()
    cfg.episode.steps_per_episode = 3
    _TrackingKpmSource.close_count = 0

    def factory(seed):
        return _TrackingKpmSource(seed=seed, gnb_ids=cfg.gnb_ids, slice_ids=list(cfg.slice_by_id))

    run_mc(
        cfg, "lb_only", factory, n_reps=3, episodes_per_rep=1, base_seed=20,
        mode="offline_synthetic", training=False, results_dir=str(tmp_path),
    )
    assert _TrackingKpmSource.close_count == 3


def test_flag_drift_flags_high_block_rate():
    summary = RunSummary(
        run_id="r", algorithm="dqn", mode="live_testbed", seed=1, n_episodes=5,
        mean_reward_per_step=0.0, mean_blocks_per_episode=5.0,
        mean_urllc_blocks_per_episode=3.0, blocks_per_episode=[5] * 5,
        blocks_per_episode_by_slice=[{}] * 5, mean_rho=0.58, omega_path="x",
    )
    flagged, reason = flag_drift(summary)
    assert flagged is True
    assert "block rate" in reason


def test_flag_drift_flags_rho_out_of_band():
    summary = RunSummary(
        run_id="r", algorithm="dqn", mode="live_testbed", seed=1, n_episodes=5,
        mean_reward_per_step=0.0, mean_blocks_per_episode=0.5,
        mean_urllc_blocks_per_episode=0.5, blocks_per_episode=[0] * 5,
        blocks_per_episode_by_slice=[{}] * 5, mean_rho=0.9, omega_path="x",
    )
    flagged, reason = flag_drift(summary)
    assert flagged is True
    assert "rho" in reason


def test_flag_drift_clean_run_not_flagged():
    summary = RunSummary(
        run_id="r", algorithm="dqn", mode="live_testbed", seed=1, n_episodes=5,
        mean_reward_per_step=0.0, mean_blocks_per_episode=0.5,
        mean_urllc_blocks_per_episode=0.5, blocks_per_episode=[0] * 5,
        blocks_per_episode_by_slice=[{}] * 5, mean_rho=0.58, omega_path="x",
    )
    flagged, reason = flag_drift(summary)
    assert flagged is False
    assert reason == ""
