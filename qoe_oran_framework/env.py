"""RANEnv: a gym-like reset()/step() environment implementing the paper
#1/#2 admission-control MDP against any KpmSource (replay, synthetic, or
live). No such step()/reset() abstraction exists in drl_slicing's runner.py
(confirmed during Stage Zero design survey) -- this is new.
"""

from typing import Any, Dict, List, Optional, Tuple

import numpy as np

from .action_mapping import BLOCK_MAPPING_LIMITATION, AdmissionGate
from .config import SacLbExperimentConfig
from .kpm_adapter import build_cluster_state
from .replay_kpm_source import KpmSource
from .reward import compute_step_reward
from .types import AdmissionRequest, ClusterState, StepResult, UeSample


def encode_state(cluster_state: ClusterState, cfg: SacLbExperimentConfig) -> np.ndarray:
    """eq. 1: per-gNB, per-slice [U_k(t)/B, C_t, L_k(t)/Lmax], concatenated
    across gNBs in cfg.gnb_ids order and slices in cfg.slices order, with
    the cluster-wide fairness ratio F_t appended once as a global scalar
    (paper #2 only; omitted when only one gNB is configured, matching
    paper #1's single-gNB formulation which has no F_t term)."""
    features: List[float] = []
    for gnb_id in cfg.gnb_ids:
        slice_states = cluster_state.per_gnb.get(gnb_id, {})
        for spec in cfg.slices:
            agg = slice_states.get(spec.slice_id)
            if agg is None:
                features.extend([0.0, 0.0, 0.0])
            else:
                features.extend([agg.prb_used_ratio, agg.congestion_level, agg.queue_len_norm])
    if len(cfg.gnbs) > 1:
        features.append(cluster_state.fairness_ratio)
    return np.array(features, dtype=np.float32)


def state_dim(cfg: SacLbExperimentConfig) -> int:
    dim = len(cfg.gnbs) * len(cfg.slices) * 3
    if len(cfg.gnbs) > 1:
        dim += 1
    return dim


def encode_request_context(request: AdmissionRequest, cfg: SacLbExperimentConfig) -> np.ndarray:
    """One-hot [slice_onehot..., gnb_onehot...] identifying which pending
    request a decision is for.

    The papers' MDP considers one request per step; Stage Zero's RANEnv
    instead batches every request pending at a step (see env.py module
    docstring / build_order notes) so a single shared cluster observation
    can cover several simultaneous admission decisions. Since
    oranslice_drl.RLPolicy.select_action takes one state vector and returns
    one action, each request in the batch is decided by concatenating this
    per-request identity onto the shared cluster observation and calling
    select_action once per request -- this is how a single reused policy
    instance makes slice/gNB-differentiated decisions within one step,
    rather than needing a bespoke multi-request policy interface.
    """
    slice_oh = [1.0 if request.slice_id == s.slice_id else 0.0 for s in cfg.slices]
    gnb_oh = [1.0 if request.gnb_id == gid else 0.0 for gid in cfg.gnb_ids]
    return np.array(slice_oh + gnb_oh, dtype=np.float32)


def request_state_dim(cfg: SacLbExperimentConfig) -> int:
    return state_dim(cfg) + len(cfg.slices) + len(cfg.gnb_ids)


def encode_full_request_state(
    obs: np.ndarray, request: AdmissionRequest, cfg: SacLbExperimentConfig
) -> np.ndarray:
    return np.concatenate([obs, encode_request_context(request, cfg)]).astype(np.float32)


class RANEnv:
    def __init__(self, cfg: SacLbExperimentConfig, kpm_source: KpmSource, seed: Optional[int] = None):
        self.cfg = cfg
        self.kpm_source = kpm_source
        self.gate = AdmissionGate(
            cfg.slice_by_id, cfg.gnb_ids, step_ratio=cfg.arrivals.ceiling_step_ratio
        )
        self._rng = np.random.RandomState(seed if seed is not None else cfg.random_seed)
        self._step_in_episode = 0
        self._episode_idx = 0
        self._global_step = 0
        self._seen_rntis: set = set()
        self._pending: List[AdmissionRequest] = []
        self._last_cluster_state: Optional[ClusterState] = None
        self.include_lb_term = cfg.paper_variant == "paper2"

    def reset(self) -> np.ndarray:
        self._step_in_episode = 0
        self._episode_idx += 1
        self.gate.reset_ceilings()
        ue_samples = self.kpm_source.poll()
        self._last_cluster_state = self._build_cluster_state(ue_samples)
        self._pending = self._synthesize_requests(ue_samples)
        return encode_state(self._last_cluster_state, self.cfg)

    def pending_requests(self) -> List[AdmissionRequest]:
        return list(self._pending)

    @property
    def last_cluster_state(self) -> Optional[ClusterState]:
        return self._last_cluster_state

    def step(self, actions: List[int]) -> StepResult:
        if self._last_cluster_state is None:
            raise RuntimeError("step() called before reset()")

        self._global_step += 1
        apply_result = self.gate.apply(
            self._pending, actions, self._last_cluster_state, self._global_step
        )

        for (gnb_id, slice_id), ceiling in apply_result.ceilings.items():
            spec = self.cfg.slice_by_id[slice_id]
            self.kpm_source.send_control(
                gnb_id, spec.sst, spec.sd, ceiling.min_ratio, ceiling.max_ratio
            )

        reward, reward_info = compute_step_reward(
            self._last_cluster_state,
            self.cfg.slice_by_id,
            apply_result.accepted_counts,
            self.cfg.reward,
            include_lb_term=self.include_lb_term,
        )

        ue_samples = self.kpm_source.poll()
        next_cluster_state = self._build_cluster_state(ue_samples)
        self._last_cluster_state = next_cluster_state
        self._pending = self._synthesize_requests(ue_samples)

        self._step_in_episode += 1
        done = self._step_in_episode >= self.cfg.episode.steps_per_episode

        limitations = list(next_cluster_state.limitations) + [BLOCK_MAPPING_LIMITATION]
        if "limitation" in reward_info:
            limitations.append(reward_info["limitation"])

        info: Dict[str, Any] = {
            "episode": self._episode_idx,
            "step": self._step_in_episode,
            "global_step": self._global_step,
            "primary_blocks": [vars(b) for b in apply_result.primary_blocks],
            "secondary_blocks": [vars(b) for b in apply_result.secondary_blocks],
            "primary_block_count": len(apply_result.primary_blocks),
            "secondary_block_count": len(apply_result.secondary_blocks),
            "accepted_counts": apply_result.accepted_counts,
            "fairness_ratio": next_cluster_state.fairness_ratio,
            "reward_breakdown": reward_info,
            "limitations": limitations,
        }
        return StepResult(
            obs=encode_state(next_cluster_state, self.cfg), reward=reward, done=done, info=info
        )

    def close(self) -> None:
        self.kpm_source.close()

    def _build_cluster_state(self, ue_samples: List[UeSample]) -> ClusterState:
        return build_cluster_state(
            ue_samples, self.cfg.gnb_ids, self.cfg.slice_by_id, self.cfg.B, self.cfg.Lmax,
            timestamp_s=float(self._global_step),
        )

    def _synthesize_requests(self, ue_samples: List[UeSample]) -> List[AdmissionRequest]:
        """One request per newly-observed RNTI (real UE attach), plus a
        config-driven number of synthetic background arrivals so the RL
        problem stays well-posed even when live UE churn is low. Seeded via
        self._rng for reproducibility."""
        requests: List[AdmissionRequest] = []
        default_gnb = self.cfg.gnb_ids[0] if self.cfg.gnb_ids else ""
        for ue in ue_samples:
            if ue.rnti not in self._seen_rntis:
                self._seen_rntis.add(ue.rnti)
                slice_id = _resolve_slice_id(ue.nssai_sd, self.cfg)
                if slice_id is None:
                    continue
                requests.append(
                    AdmissionRequest(
                        request_id=f"ue:{ue.rnti}:{self._global_step}",
                        slice_id=slice_id,
                        gnb_id=ue.gnb_id or default_gnb,
                        arrival_step=self._global_step,
                        synthetic=False,
                    )
                )

        n_synthetic = self.cfg.arrivals.synthetic_arrivals_per_step
        slice_ids = [s.slice_id for s in self.cfg.slices]
        gnb_ids = self.cfg.gnb_ids
        for i in range(n_synthetic):
            if not slice_ids or not gnb_ids:
                break
            slice_id = slice_ids[self._rng.randint(0, len(slice_ids))]
            gnb_id = gnb_ids[self._rng.randint(0, len(gnb_ids))]
            requests.append(
                AdmissionRequest(
                    request_id=f"synthetic:{self._global_step}:{i}",
                    slice_id=slice_id,
                    gnb_id=gnb_id,
                    arrival_step=self._global_step,
                    synthetic=True,
                )
            )

        max_pending = self.cfg.arrivals.max_pending_per_step
        if max_pending and len(requests) > max_pending:
            requests = requests[:max_pending]
        return requests


def _resolve_slice_id(nssai_sd: int, cfg: SacLbExperimentConfig) -> Optional[str]:
    from .config import SD_TO_SLICE_ID

    slice_id = SD_TO_SLICE_ID.get(int(nssai_sd))
    if slice_id in cfg.slice_by_id:
        return slice_id
    return None
