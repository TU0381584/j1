import csv
import json
import urllib.parse
import urllib.request
from abc import ABC, abstractmethod
from typing import Dict, List

from .types import EnvState, SliceMetrics


class MetricsCollector(ABC):
    @abstractmethod
    def get_state(self, step: int) -> EnvState:
        raise NotImplementedError


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
    def __init__(self, base_url: str, metric_queries: Dict[str, str], known_slice_ids: List[str], step_seconds: int) -> None:
        self.base_url = base_url.rstrip("/")
        self.metric_queries = metric_queries
        self.known_slice_ids = known_slice_ids
        self.step_seconds = step_seconds

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

    def get_state(self, step: int) -> EnvState:
        throughput_map = self._query(self.metric_queries["throughput_kbps"])
        latency_map = self._query(self.metric_queries["latency_ms"])
        loss_map = self._query(self.metric_queries["loss_pct"])
        offered_map = self._query(self.metric_queries.get("offered_load_kbps", self.metric_queries["throughput_kbps"]))

        state = EnvState(timestamp_s=step * self.step_seconds)
        for slice_id in self.known_slice_ids:
            state.slices[slice_id] = SliceMetrics(
                slice_id=slice_id,
                throughput_kbps=float(throughput_map.get(slice_id, 0.0)),
                latency_ms=float(latency_map.get(slice_id, 0.0)),
                loss_pct=float(loss_map.get(slice_id, 0.0)),
                offered_load_kbps=float(offered_map.get(slice_id, throughput_map.get(slice_id, 0.0))),
            )
        return state
