from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List

import yaml

# sd -> slice_id convention, matching drl_slicing/oranslice_drl/reward.py::_resolve_slice_profile
SD_TO_SLICE_ID = {0: "embb", 1: "urllc", 2: "mmtc"}


@dataclass
class SliceSpec:
    slice_id: str            # "urllc" | "embb" | "mmtc"
    sst: int
    sd: int
    nominal_ratio: int       # nominal PRB quota, as a % of gNB capacity B (paper #1: urllc=30,embb=70; paper #2: urllc=30,embb=60,mmtc=10)
    min_ratio_floor: int     # admission ceiling may never be nudged below this
    max_ratio_cap: int       # admission ceiling may never be nudged above this
    latency_budget_ms: float
    loss_budget_pct: float
    priority_weight: float   # omega_k
    accept_reward: float     # R_k
    violation_penalty: float # lambda_k


@dataclass
class GnbSpec:
    gnb_id: str
    prb_capacity: int = 100  # B


@dataclass
class RewardWeights:
    congestion_coeff: float = 1.0   # mu
    lb_coeff: float = 0.0           # beta (0 disables the LB term -> paper #1 mode)


@dataclass
class ArrivalConfig:
    synthetic_arrivals_per_step: int = 2
    max_pending_per_step: int = 8
    ceiling_step_ratio: int = 5     # PRB-ratio nudge size per accept/reject


@dataclass
class EpisodeConfig:
    step_seconds: float = 5.0
    steps_per_episode: int = 60


@dataclass
class SacLbExperimentConfig:
    name: str
    random_seed: int
    paper_variant: str              # "paper1" (SAC only) | "paper2" (SAC-LB)
    B: float
    Lmax: float                     # queue-length normalisation constant
    gnbs: List[GnbSpec]
    slices: List[SliceSpec]
    reward: RewardWeights
    arrivals: ArrivalConfig
    episode: EpisodeConfig

    @property
    def slice_by_id(self) -> Dict[str, SliceSpec]:
        return {s.slice_id: s for s in self.slices}

    @property
    def gnb_ids(self) -> List[str]:
        return [g.gnb_id for g in self.gnbs]


def _load_yaml(path: Path) -> dict:
    with path.open("r", encoding="utf-8") as handle:
        return yaml.safe_load(handle)


def load_saclb_config(config_path: str) -> SacLbExperimentConfig:
    raw = _load_yaml(Path(config_path))

    slices = [
        SliceSpec(
            slice_id=item["slice_id"],
            sst=int(item.get("sst", 1)),
            sd=int(item["sd"]),
            nominal_ratio=int(item["nominal_ratio"]),
            min_ratio_floor=int(item.get("min_ratio_floor", 5)),
            max_ratio_cap=int(item.get("max_ratio_cap", 100)),
            latency_budget_ms=float(item["latency_budget_ms"]),
            loss_budget_pct=float(item["loss_budget_pct"]),
            priority_weight=float(item["priority_weight"]),
            accept_reward=float(item.get("accept_reward", 1.0)),
            violation_penalty=float(item.get("violation_penalty", 1.0)),
        )
        for item in raw["slices"]
    ]
    gnbs = [
        GnbSpec(gnb_id=item["gnb_id"], prb_capacity=int(item.get("prb_capacity", 100)))
        for item in raw["gnbs"]
    ]
    reward_cfg = raw.get("reward", {})
    arrivals_cfg = raw.get("arrivals", {})
    episode_cfg = raw.get("episode", {})

    return SacLbExperimentConfig(
        name=raw["name"],
        random_seed=int(raw.get("random_seed", 256)),
        paper_variant=raw.get("paper_variant", "paper2"),
        B=float(raw.get("B", 100)),
        Lmax=float(raw.get("Lmax", 100)),
        gnbs=gnbs,
        slices=slices,
        reward=RewardWeights(
            congestion_coeff=float(reward_cfg.get("congestion_coeff", 1.0)),
            lb_coeff=float(reward_cfg.get("lb_coeff", 0.0)),
        ),
        arrivals=ArrivalConfig(
            synthetic_arrivals_per_step=int(arrivals_cfg.get("synthetic_arrivals_per_step", 2)),
            max_pending_per_step=int(arrivals_cfg.get("max_pending_per_step", 8)),
            ceiling_step_ratio=int(arrivals_cfg.get("ceiling_step_ratio", 5)),
        ),
        episode=EpisodeConfig(
            step_seconds=float(episode_cfg.get("step_seconds", 5.0)),
            steps_per_episode=int(episode_cfg.get("steps_per_episode", 60)),
        ),
    )
