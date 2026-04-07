import csv
import json
import random
import urllib.parse
import urllib.request
from abc import ABC, abstractmethod
from typing import Dict, List

from .types import EnvState, SliceMetrics
from .config import SliceTarget
from .types import SliceAction


class MetricsCollector(ABC):
    @abstractmethod
    def get_state(self, step: int) -> EnvState:
        raise NotImplementedError

    def on_action(self, actions: List[SliceAction], targets: Dict[str, SliceTarget], step: int) -> None:
        return


class CsvReplayCollector(MetricsCollector):
    def __init__(self, csv_path: str, known_slice_ids: List[str], step_seconds: int) -> None:
        self.rows = self._read_rows(csv_path)
        self.known_slice_ids = known_slice_ids
        self.step_seconds = step_seconds

    def _read_rows(self, csv_path: str) -> List[dict]:
        with open(csv_path, "r", encoding="utf-8") as handle:
            reader = csv.DictReader(handle)
            return [row for row in reader]

    def get_state(self, step: int) -> EnvState:
        if not self.rows:
            raise RuntimeError("CSV replay collector has no rows")
        index = min(step, len(self.rows) - 1)
        row = self.rows[index]

        state = EnvState(timestamp_s=float(row.get("timestamp_s", step * self.step_seconds)))
        default_throughput = float(row.get("throughput_kbps", 0.0))
        default_latency = float(row.get("latency_ms", 0.0))
        default_loss = float(row.get("loss_pct", 0.0))
        default_offered = float(row.get("offered_load_kbps", default_throughput))

        for slice_id in self.known_slice_ids:
            prefix = f"{slice_id}_"
            throughput = float(row.get(prefix + "throughput_kbps", default_throughput))
            latency = float(row.get(prefix + "latency_ms", default_latency))
            loss = float(row.get(prefix + "loss_pct", default_loss))
            offered = float(row.get(prefix + "offered_load_kbps", default_offered))
            state.slices[slice_id] = SliceMetrics(
                slice_id=slice_id,
                throughput_kbps=throughput,
                latency_ms=latency,
                loss_pct=loss,
                offered_load_kbps=offered,
            )
        return state


class PrometheusCollector(MetricsCollector):
    def __init__(
        self,
        base_url: str,
        metric_queries: Dict[str, str],
        known_slice_ids: List[str],
        step_seconds: int,
        enable_action_feedback: bool = False,
        action_feedback_gain: float = 0.35,
    ) -> None:
        self.base_url = base_url.rstrip("/")
        self.metric_queries = metric_queries
        self.known_slice_ids = known_slice_ids
        self.step_seconds = step_seconds
        self.enable_action_feedback = enable_action_feedback
        self.action_feedback_gain = max(0.0, float(action_feedback_gain))
        self.last_actions: Dict[str, SliceAction] = {}

    def on_action(self, actions: List[SliceAction], targets: Dict[str, SliceTarget], step: int) -> None:
        self.last_actions = {action.slice_id: action for action in actions}

    def _query(self, query: str) -> Dict[str, float]:
        encoded = urllib.parse.urlencode({"query": query})
        url = f"{self.base_url}/api/v1/query?{encoded}"
        with urllib.request.urlopen(url, timeout=8) as response:
            payload = json.loads(response.read().decode("utf-8"))

        data = payload.get("data", {}).get("result", [])
        out: Dict[str, float] = {}
        for item in data:
            metric = item.get("metric", {})
            slice_id = metric.get("slice_id") or metric.get("snssai") or metric.get("slice")
            if not slice_id:
                continue
            value = float(item.get("value", [0, 0])[1])
            out[slice_id] = value
        return out

    @staticmethod
    def _render_query(query: str, step: int) -> str:
        # Allow configs to inject step-indexed behavior (for deterministic scenarios)
        # without requiring Prometheus wall-clock alignment.
        return query.replace("__STEP__", str(step))

    def _normalized_shares(self) -> Dict[str, float]:
        if not self.last_actions:
            equal = 1.0 / max(len(self.known_slice_ids), 1)
            return {slice_id: equal for slice_id in self.known_slice_ids}

        raw_shares: Dict[str, float] = {}
        for slice_id in self.known_slice_ids:
            action = self.last_actions.get(slice_id)
            if action is None:
                raw_shares[slice_id] = 1.0
                continue
            raw_shares[slice_id] = max(0.01, (action.min_ratio + action.max_ratio) / 200.0)

        total = sum(raw_shares.values())
        if total <= 0.0:
            equal = 1.0 / max(len(self.known_slice_ids), 1)
            return {slice_id: equal for slice_id in self.known_slice_ids}

        return {slice_id: raw_shares[slice_id] / total for slice_id in self.known_slice_ids}

    def _apply_action_feedback(self, state: EnvState) -> None:
        if not self.enable_action_feedback:
            return
        if not self.last_actions:
            return

        shares = self._normalized_shares()
        baseline_share = 1.0 / max(len(self.known_slice_ids), 1)

        for slice_id in self.known_slice_ids:
            metrics = state.slices[slice_id]
            relative_share = shares[slice_id] / max(baseline_share, 1e-6)
            bias = 1.0 + self.action_feedback_gain * (relative_share - 1.0)
            bias = max(0.35, min(2.2, bias))

            throughput = max(0.0, metrics.throughput_kbps * bias)
            metrics.throughput_kbps = float(min(metrics.offered_load_kbps, throughput))

            latency_scale = max(0.25, min(3.0, 1.0 / max(bias, 1e-6)))
            loss_scale = max(0.2, min(4.0, 1.0 / max(0.8 * bias + 0.2, 1e-6)))

            metrics.latency_ms = float(max(0.1, metrics.latency_ms * latency_scale))
            metrics.loss_pct = float(max(0.0, metrics.loss_pct * loss_scale))

    def get_state(self, step: int) -> EnvState:
        throughput_query = self._render_query(self.metric_queries["throughput_kbps"], step)
        latency_query = self._render_query(self.metric_queries["latency_ms"], step)
        loss_query = self._render_query(self.metric_queries["loss_pct"], step)
        offered_query = self._render_query(
            self.metric_queries.get("offered_load_kbps", self.metric_queries["throughput_kbps"]),
            step,
        )

        throughput_map = self._query(throughput_query)
        latency_map = self._query(latency_query)
        loss_map = self._query(loss_query)
        offered_map = self._query(offered_query)

        state = EnvState(timestamp_s=step * self.step_seconds)
        for slice_id in self.known_slice_ids:
            state.slices[slice_id] = SliceMetrics(
                slice_id=slice_id,
                throughput_kbps=float(throughput_map.get(slice_id, 0.0)),
                latency_ms=float(latency_map.get(slice_id, 0.0)),
                loss_pct=float(loss_map.get(slice_id, 0.0)),
                offered_load_kbps=float(offered_map.get(slice_id, throughput_map.get(slice_id, 0.0))),
            )

        self._apply_action_feedback(state)
        return state


class OnlineRanRandomCollector(MetricsCollector):
    """
    Online randomized RAN traffic collector with action-feedback dynamics.
    Simulates eMBB/URLLC load variation and KPI response to allocation actions.
    """

    def __init__(
        self,
        known_slice_ids: List[str],
        targets: Dict[str, SliceTarget],
        step_seconds: int,
        random_seed: int,
        total_capacity_kbps: float = 30000.0,
    ) -> None:
        self.known_slice_ids = known_slice_ids
        self.targets = targets
        self.step_seconds = step_seconds
        self.total_capacity_kbps = total_capacity_kbps
        self.rng = random.Random(random_seed)

        self.last_actions: Dict[str, SliceAction] = {}
        self.channel_quality: Dict[str, float] = {sid: 1.0 for sid in known_slice_ids}
        self.prev_offered: Dict[str, float] = {sid: 0.0 for sid in known_slice_ids}

    def on_action(self, actions: List[SliceAction], targets: Dict[str, SliceTarget], step: int) -> None:
        self.last_actions = {action.slice_id: action for action in actions}

    def _slice_profile(self, slice_id: str) -> str:
        target = self.targets[slice_id]
        return "urllc" if target.latency_budget_ms <= 25.0 else "embb"

    def _random_offered_load(self, slice_id: str) -> float:
        profile = self._slice_profile(slice_id)
        if profile == "urllc":
            base = self.rng.uniform(800.0, 4500.0)
            burst = self.rng.uniform(1500.0, 3500.0) if self.rng.random() < 0.22 else 0.0
        else:
            base = self.rng.uniform(7000.0, 19000.0)
            burst = self.rng.uniform(2500.0, 7000.0) if self.rng.random() < 0.28 else 0.0
        offered = 0.65 * self.prev_offered[slice_id] + 0.35 * (base + burst)
        self.prev_offered[slice_id] = offered
        return max(offered, 100.0)

    def _allocation_share(self, slice_id: str, actions_by_slice: Dict[str, SliceAction]) -> float:
        if slice_id not in actions_by_slice:
            return 1.0 / max(len(self.known_slice_ids), 1)
        action = actions_by_slice[slice_id]
        return max(0.01, (action.min_ratio + action.max_ratio) / 200.0)

    def get_state(self, step: int) -> EnvState:
        state = EnvState(timestamp_s=step * self.step_seconds)
        actions_by_slice = self.last_actions

        shares = {
            sid: self._allocation_share(sid, actions_by_slice)
            for sid in self.known_slice_ids
        }
        total_share = sum(shares.values())
        if total_share <= 0:
            total_share = float(len(self.known_slice_ids))

        for slice_id in self.known_slice_ids:
            profile = self._slice_profile(slice_id)

            offered_load = self._random_offered_load(slice_id)
            share = shares[slice_id] / total_share
            allocated_capacity = max(200.0, self.total_capacity_kbps * share)

            q_prev = self.channel_quality[slice_id]
            q_new = 0.82 * q_prev + 0.18 * self.rng.uniform(0.72, 1.18)
            q_new = max(0.55, min(1.25, q_new))
            self.channel_quality[slice_id] = q_new

            raw_capacity = allocated_capacity * q_new
            throughput = min(offered_load, raw_capacity)

            queue_pressure = max(0.0, offered_load - throughput) / max(allocated_capacity, 1.0)
            if profile == "urllc":
                latency = 4.0 + 18.0 * queue_pressure + self.rng.uniform(0.2, 2.0)
                loss = 0.05 + 2.2 * queue_pressure + self.rng.uniform(0.02, 0.35)
            else:
                latency = 14.0 + 26.0 * queue_pressure + self.rng.uniform(0.5, 3.5)
                loss = 0.15 + 1.3 * queue_pressure + self.rng.uniform(0.03, 0.45)

            state.slices[slice_id] = SliceMetrics(
                slice_id=slice_id,
                throughput_kbps=float(max(0.0, throughput)),
                latency_ms=float(max(0.5, latency)),
                loss_pct=float(max(0.0, min(loss, 15.0))),
                offered_load_kbps=float(offered_load),
            )

        return state
