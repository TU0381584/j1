"""Offline KpmSource implementations: a JSONL trace player and a synthetic
generator. Both implement the same protocol `live_kpm_source.LiveKpmSource`
will implement, so RANEnv is fully testable with zero network/testbed
dependency (Stage Zero build-order step 4's "critical gate").
"""

import json
from pathlib import Path
from typing import Any, Dict, List, Optional, Protocol

import numpy as np

from .types import UeSample


class KpmSource(Protocol):
    def poll(self) -> List[UeSample]: ...
    def send_control(self, gnb_id: str, sst: int, sd: int, min_ratio: int, max_ratio: int) -> None: ...
    def close(self) -> None: ...


class ReplayKpmSource:
    """Plays back a JSONL trace of UE-sample snapshots, one row per poll()."""

    def __init__(self, trace_path: str, loop: bool = True):
        self._rows = self._load_trace(trace_path)
        if not self._rows:
            raise ValueError(f"trace file {trace_path} contains no rows")
        self._idx = 0
        self._loop = loop
        self.sent_controls: List[Dict[str, Any]] = []

    @staticmethod
    def _load_trace(trace_path: str) -> List[dict]:
        rows = []
        with Path(trace_path).open("r", encoding="utf-8") as handle:
            for line in handle:
                line = line.strip()
                if line:
                    rows.append(json.loads(line))
        return rows

    def poll(self) -> List[UeSample]:
        if self._idx >= len(self._rows):
            if self._loop:
                self._idx = 0
            else:
                return []
        row = self._rows[self._idx]
        self._idx += 1
        return [UeSample(**ue) for ue in row.get("ue_samples", [])]

    def send_control(self, gnb_id: str, sst: int, sd: int, min_ratio: int, max_ratio: int) -> None:
        self.sent_controls.append(
            {"gnb_id": gnb_id, "sst": sst, "sd": sd, "min_ratio": min_ratio, "max_ratio": max_ratio}
        )

    def close(self) -> None:
        pass


class SyntheticKpmSource:
    """Deterministic (seeded) procedural KPM feed for offline dev/testing
    and offline training, requiring no hand-authored fixture file.

    Traffic per slice is generated as a Poisson-ish UE-count process with a
    configurable mean utilisation per gNB; PRB demand per UE is sampled
    around a per-slice nominal value. This is a modelling choice for
    Stage Zero's offline-training leg only -- never used for the live
    acceptance-bar numbers, which come exclusively from LiveKpmSource.
    """

    def __init__(
        self,
        seed: int,
        gnb_ids: List[str],
        slice_ids: List[str],
        mean_prb_per_ue: Optional[Dict[str, float]] = None,
        mean_ues_per_slice: Optional[Dict[str, float]] = None,
    ):
        self._rng = np.random.RandomState(seed)
        self._gnb_ids = gnb_ids
        self._slice_ids = slice_ids
        self._mean_prb_per_ue = mean_prb_per_ue or {s: 3.0 for s in slice_ids}
        self._mean_ues_per_slice = mean_ues_per_slice or {s: 4.0 for s in slice_ids}
        self._t = 0.0
        self._rnti_counter = 0
        self.sent_controls: List[Dict[str, Any]] = []

    def poll(self) -> List[UeSample]:
        self._t += 1.0
        samples: List[UeSample] = []
        for gnb_id in self._gnb_ids:
            for slice_id_idx, slice_id in enumerate(self._slice_ids):
                sd = {"embb": 0, "urllc": 1, "mmtc": 2}[slice_id]
                n_ues = self._rng.poisson(self._mean_ues_per_slice[slice_id])
                for _ in range(n_ues):
                    self._rnti_counter += 1
                    prb = max(0.0, self._rng.normal(self._mean_prb_per_ue[slice_id], 1.0))
                    samples.append(
                        UeSample(
                            rnti=self._rnti_counter,
                            timestamp_s=self._t,
                            nssai_sst=1,
                            nssai_sd=sd,
                            avg_prbs_dl=prb,
                            gnb_id=gnb_id,
                            dl_total_bytes=prb * 1000.0,
                            dl_errors=max(0.0, self._rng.normal(0.1, 0.05)),
                            dl_bler=max(0.0, self._rng.normal(0.5, 0.3)),
                            dl_mac_buffer_occupation=max(0.0, self._rng.normal(5.0, 2.0)),
                        )
                    )
        return samples

    def send_control(self, gnb_id: str, sst: int, sd: int, min_ratio: int, max_ratio: int) -> None:
        self.sent_controls.append(
            {"gnb_id": gnb_id, "sst": sst, "sd": sd, "min_ratio": min_ratio, "max_ratio": max_ratio}
        )

    def close(self) -> None:
        pass
