"""DQN admission policy: binary accept/reject over a per-request state.

Reuses oranslice_drl.drl_policy.DQNPolicy/QNetwork directly (not a copy) --
that class already operates generically on (state_dim, action_dim,
n_branches) with no ratio-bin logic baked into it; the ratio-specific
behaviour lives in oranslice_drl.drl_training.ActionDecoder, which this
module never imports. Fixing action_dim=2, n_branches=1 turns it into a
binary admission-control policy for free.

Table I (papers #1/#2, DQN column): gamma=0.95, alpha=0.001, batch_size=16,
target update every 10 episodes (not the base class's every-100-train-steps
schedule -- see on_episode_end()), uniform replay.
"""

from .._oranslice_path import ensure_oranslice_drl_importable

ensure_oranslice_drl_importable()

from oranslice_drl.drl_policy import DQNPolicy  # noqa: E402

PAPER_TABLE_I_DQN_DEFAULTS = dict(
    gamma=0.95,
    learning_rate=0.001,
    epsilon_start=1.0,
    epsilon_end=0.05,
    epsilon_decay=0.995,
)


class DQNAdmissionPolicy(DQNPolicy):
    def __init__(
        self,
        state_dim: int,
        hidden_dim: int = 128,
        device: str = "cpu",
        **overrides,
    ):
        params = {**PAPER_TABLE_I_DQN_DEFAULTS, **overrides}
        super().__init__(
            state_dim=state_dim,
            action_dim=2,
            n_branches=1,
            hidden_dim=hidden_dim,
            device=device,
            **params,
        )

    def on_episode_end(self) -> None:
        """Table I: target update every 10 episodes. Called explicitly by
        mc_runner at episode boundaries, on top of (not instead of) the
        base class's every-100-train-step safety-net sync -- redundant
        syncing is harmless, it just means the target network can only be
        *more* current than the paper's schedule, never staler."""
        self.target_network.load_state_dict(self.q_network.state_dict())
