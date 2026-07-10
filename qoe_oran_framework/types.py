from dataclasses import dataclass, field
from typing import Any, Dict, List

import numpy as np


@dataclass
class UeSample:
    """One parsed ue_info_m record from a RAN_indication_response.

    Field names mirror xapp-oai/base-xapp/oai-oran-protolib/ran_messages.proto
    (ue_info_m) so parsing code can construct these with **kwargs from the
    protobuf message directly.
    """

    rnti: int
    timestamp_s: float
    nssai_sst: int
    nssai_sd: int
    avg_prbs_dl: float
    gnb_id: str = ""
    dl_total_bytes: float = 0.0
    dl_errors: float = 0.0
    dl_bler: float = 0.0
    ul_total_bytes: float = 0.0
    ul_errors: float = 0.0
    dl_mac_buffer_occupation: float = 0.0


@dataclass
class SliceAggState:
    """Per-slice, per-gNB aggregate state for one step (eq. 1 fields)."""

    slice_id: str
    gnb_id: str
    prb_used_ratio: float          # U_k(t) / B
    congestion_level: float        # C_t (duplicated per slice for encoding convenience)
    queue_len_norm: float          # L_k(t) / Lmax
    queue_used_fallback: bool = False
    loss_proxy: float = 0.0        # derived from dl_bler, compared against loss_budget_pct
    n_ues: int = 0


@dataclass
class ClusterState:
    """RANEnv observation container (pre-encode)."""

    timestamp_s: float
    per_gnb: Dict[str, Dict[str, SliceAggState]] = field(default_factory=dict)
    fairness_ratio: float = 1.0    # F_t: min/max PRB utilisation ratio across the gNB cluster
    limitations: List[str] = field(default_factory=list)


@dataclass
class AdmissionRequest:
    request_id: str
    slice_id: str
    gnb_id: str
    arrival_step: int
    synthetic: bool = False


@dataclass
class BlockEvent:
    request_id: str
    slice_id: str
    gnb_id: str
    step: int
    kind: str   # "primary_reject" | "secondary_over_ceiling"


@dataclass
class StepResult:
    obs: np.ndarray
    reward: float
    done: bool
    info: Dict[str, Any] = field(default_factory=dict)
