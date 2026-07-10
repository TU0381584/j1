"""Live KpmSource: the real E2-loop transport, talking to a running OAI
gNB via the o-ran-e2sim OAI-protobuf bridge (kpm_sim, binds UDP 6600/6655)
and xapp-oai/base-xapp's confirmed-working RIC-bypass wire protocol (UDP
subscribe on 7001, streamed indications + control over a TCP socket on
4200) -- see Stage Zero plan's Path A decision. The encode/parse logic is
reimplemented here from slicing_ctrl_influxdb_xapp_kpm.py (a demo script,
not an importable module) rather than copied verbatim, and its InfluxDB
side-logging is dropped (not needed for Stage Zero).

Environment note, discovered empirically: the pre-generated
ran_messages_pb2.py in xapp-oai was built against an older protoc and
raises "Descriptors cannot be created directly" under this environment's
protobuf 4.25.6 (C++-backed) implementation. Forcing the pure-Python
implementation via PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION fixes this
without touching the xapp-oai repo or downgrading protobuf, at some
(here, irrelevant) parsing-speed cost.
"""

import os
import socket
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Optional

os.environ.setdefault("PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION", "python")

DEFAULT_PROTO_BUILD_DIR = "/home/w5/ors/xapp-oai/base-xapp/oai-oran-protolib/builds"
_proto_dir = os.environ.get("XAPP_OAI_PROTO_DIR", DEFAULT_PROTO_BUILD_DIR)
if not Path(_proto_dir).is_dir():
    raise ImportError(
        f"xapp-oai protobuf build dir not found at {_proto_dir!r}. Set XAPP_OAI_PROTO_DIR "
        "to the directory containing ran_messages_pb2.py (normally "
        "<xapp-oai>/base-xapp/oai-oran-protolib/builds)."
    )
if _proto_dir not in sys.path:
    sys.path.insert(0, _proto_dir)

import ran_messages_pb2  # noqa: E402

from .types import UeSample  # noqa: E402

DEFAULT_HOST = "127.0.0.1"
DEFAULT_SUB_PORT = 7001
DEFAULT_CONTROL_PORT = 4200
MAX_RECV_BYTES = 4096

_SD_TO_SLICE_ID = {0: "embb", 1: "urllc", 2: "mmtc"}


def build_indication_request() -> bytes:
    message = ran_messages_pb2.RAN_message()
    message.msg_type = ran_messages_pb2.RAN_message_type.INDICATION_REQUEST
    inner = ran_messages_pb2.RAN_indication_request()
    inner.target_params.extend(
        [ran_messages_pb2.RAN_parameter.GNB_ID, ran_messages_pb2.RAN_parameter.UE_LIST]
    )
    message.ran_indication_request.CopyFrom(inner)
    return message.SerializeToString()


def build_control_request(sst: int, sd: int, min_ratio: int, max_ratio: int) -> bytes:
    slicing = ran_messages_pb2.slicing_control_m()
    slicing.sst = int(sst)
    slicing.sd = int(sd)
    slicing.min_ratio = int(min_ratio)
    slicing.max_ratio = int(max_ratio)

    entry = ran_messages_pb2.RAN_param_map_entry()
    entry.key = ran_messages_pb2.RAN_parameter.SLICING_CONTROL
    entry.slicing_ctrl.CopyFrom(slicing)

    inner = ran_messages_pb2.RAN_control_request()
    inner.target_param_map.append(entry)

    message = ran_messages_pb2.RAN_message()
    message.msg_type = ran_messages_pb2.RAN_message_type.CONTROL
    message.ran_control_request.CopyFrom(inner)
    return message.SerializeToString()


def parse_indication_response(data: bytes, gnb_id: str, timestamp_s: float) -> List[UeSample]:
    response = ran_messages_pb2.RAN_indication_response()
    response.ParseFromString(data)
    samples: List[UeSample] = []
    for entry in response.param_map:
        if entry.key != ran_messages_pb2.RAN_parameter.UE_LIST:
            continue
        for ue in entry.ue_list.ue_info:
            samples.append(
                UeSample(
                    rnti=ue.rnti,
                    timestamp_s=timestamp_s,
                    nssai_sst=ue.nssai_sST,
                    nssai_sd=ue.nssai_sD,
                    avg_prbs_dl=ue.avg_prbs_dl,
                    gnb_id=gnb_id,
                    dl_total_bytes=ue.dl_total_bytes,
                    dl_errors=ue.dl_errors,
                    dl_bler=ue.dl_bler,
                    dl_mac_buffer_occupation=ue.dl_mac_buffer_occupation,
                )
            )
    return samples


def open_control_socket(port: int, host: str = "", timeout_s: Optional[float] = None) -> socket.socket:
    """Blocks until the bridge connects as a TCP client -- same handshake
    as xapp-oai/base-xapp/xapp_control.py::open_control_socket,
    reimplemented (that module is a demo script, not an importable
    library) rather than imported."""
    server = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    server.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    server.bind((host, port))
    server.listen(1)
    if timeout_s is not None:
        server.settimeout(timeout_s)
    try:
        conn, _addr = server.accept()
    except socket.timeout:
        server.close()
        raise TimeoutError(
            f"no E2/bridge connection on control port {port} within {timeout_s}s -- "
            "is the o-ran-e2sim bridge (kpm_sim) running and pointed at this port?"
        )
    server.close()
    return conn


class LiveKpmSource:
    """Real E2-loop transport implementing the KpmSource protocol.

    poll() blocks on the TCP socket until the bridge sends the next
    indication -- there is no separate step_seconds sleep here, the live
    cadence *is* the report cadence set on the E2-node/bridge side.
    """

    def __init__(
        self,
        gnb_id: str,
        host: str = DEFAULT_HOST,
        sub_port: int = DEFAULT_SUB_PORT,
        control_port: int = DEFAULT_CONTROL_PORT,
        connect_timeout_s: float = 60.0,
    ):
        self._gnb_id = gnb_id
        self._host = host
        self._sub_port = sub_port
        self._control_port = control_port
        self._connect_timeout_s = connect_timeout_s
        self._control_sck: Optional[socket.socket] = None
        self.sent_controls: List[Dict[str, Any]] = []

    def _ensure_connected(self) -> None:
        if self._control_sck is not None:
            return
        self._send_subscription()
        self._control_sck = open_control_socket(self._control_port, timeout_s=self._connect_timeout_s)

    def _send_subscription(self) -> None:
        buf = build_indication_request()
        sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        try:
            sock.sendto(buf, (self._host, self._sub_port))
        finally:
            sock.close()

    def poll(self) -> List[UeSample]:
        self._ensure_connected()
        data = self._control_sck.recv(MAX_RECV_BYTES)
        if not data:
            return []
        return parse_indication_response(data, self._gnb_id, time.time())

    def send_control(self, gnb_id: str, sst: int, sd: int, min_ratio: int, max_ratio: int) -> None:
        self._ensure_connected()
        buf = build_control_request(sst, sd, min_ratio, max_ratio)
        self._control_sck.send(buf)
        self.sent_controls.append(
            {"gnb_id": gnb_id, "sst": sst, "sd": sd, "min_ratio": min_ratio, "max_ratio": max_ratio}
        )

    def close(self) -> None:
        if self._control_sck is not None:
            self._control_sck.close()
            self._control_sck = None
