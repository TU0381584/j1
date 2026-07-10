"""End-to-end transport tests against a real loopback TCP/UDP pair (a
fake E2/bridge peer run in a background thread) -- no live gNB needed to
verify the wire protocol itself is correct."""

import os
import socket
import sys
import threading
from pathlib import Path

import pytest

os.environ.setdefault("PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION", "python")

_proto_dir = "/home/w5/ors/xapp-oai/base-xapp/oai-oran-protolib/builds"
if not Path(_proto_dir).is_dir():
    pytest.skip(f"xapp-oai protobuf build dir not present at {_proto_dir}", allow_module_level=True)

from qoe_oran_framework.live_kpm_source import (  # noqa: E402
    LiveKpmSource,
    build_control_request,
    build_indication_request,
    parse_indication_response,
)

if _proto_dir not in sys.path:
    sys.path.insert(0, _proto_dir)
import ran_messages_pb2  # noqa: E402

SUB_PORT = 27001
CONTROL_PORT = 24200


def test_build_indication_request_shape():
    buf = build_indication_request()
    msg = ran_messages_pb2.RAN_message()
    msg.ParseFromString(buf)
    assert msg.msg_type == ran_messages_pb2.RAN_message_type.INDICATION_REQUEST
    assert list(msg.ran_indication_request.target_params) == [
        ran_messages_pb2.RAN_parameter.GNB_ID, ran_messages_pb2.RAN_parameter.UE_LIST,
    ]


def test_build_control_request_round_trips_fields():
    buf = build_control_request(sst=1, sd=1, min_ratio=5, max_ratio=30)
    msg = ran_messages_pb2.RAN_message()
    msg.ParseFromString(buf)
    slicing = msg.ran_control_request.target_param_map[0].slicing_ctrl
    assert (slicing.sst, slicing.sd, slicing.min_ratio, slicing.max_ratio) == (1, 1, 5, 30)


def test_parse_indication_response_extracts_ue_samples():
    resp = ran_messages_pb2.RAN_indication_response()
    entry = resp.param_map.add()
    entry.key = ran_messages_pb2.RAN_parameter.UE_LIST
    entry.ue_list.connected_ues = 1
    ue = entry.ue_list.ue_info.add()
    ue.rnti = 99
    ue.avg_prbs_dl = 12.5
    ue.nssai_sST = 1
    ue.nssai_sD = 1
    ue.dl_mac_buffer_occupation = 3.0
    ue.dl_bler = 0.1
    ue.dl_errors = 0.2

    samples = parse_indication_response(resp.SerializeToString(), gnb_id="gnb-0", timestamp_s=1.0)
    assert len(samples) == 1
    assert samples[0].rnti == 99
    assert samples[0].avg_prbs_dl == pytest.approx(12.5)
    assert samples[0].gnb_id == "gnb-0"
    assert samples[0].nssai_sd == 1


def _fake_bridge(sub_socket_ready, received_subscription, control_connected, sent_response, received_control):
    """Simulates the E2/bridge peer: receives the UDP subscription, then
    connects as a TCP client to the xApp's control port, sends one
    fabricated indication, and reads back one control message."""
    udp = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    udp.bind(("127.0.0.1", SUB_PORT))
    sub_socket_ready.set()
    data, _addr = udp.recvfrom(4096)
    udp.close()
    received_subscription["data"] = data

    tcp = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    tcp.connect(("127.0.0.1", CONTROL_PORT))
    control_connected.set()

    resp = ran_messages_pb2.RAN_indication_response()
    entry = resp.param_map.add()
    entry.key = ran_messages_pb2.RAN_parameter.UE_LIST
    entry.ue_list.connected_ues = 1
    ue = entry.ue_list.ue_info.add()
    ue.rnti = 7
    ue.avg_prbs_dl = 4.0
    ue.nssai_sST = 1
    ue.nssai_sD = 1
    tcp.send(resp.SerializeToString())
    sent_response.set()

    control_data = tcp.recv(4096)
    received_control["data"] = control_data
    tcp.close()


def test_live_kpm_source_end_to_end_over_loopback():
    sub_socket_ready = threading.Event()
    control_connected = threading.Event()
    sent_response = threading.Event()
    received_subscription = {}
    received_control = {}

    bridge = threading.Thread(
        target=_fake_bridge,
        args=(sub_socket_ready, received_subscription, control_connected, sent_response, received_control),
        daemon=True,
    )
    bridge.start()
    assert sub_socket_ready.wait(timeout=5)

    source = LiveKpmSource(
        gnb_id="gnb-0", sub_port=SUB_PORT, control_port=CONTROL_PORT, connect_timeout_s=5.0,
    )
    samples = source.poll()  # triggers subscription + blocks for the bridge's TCP connect + first message

    assert control_connected.wait(timeout=5)
    assert sent_response.wait(timeout=5)
    assert len(samples) == 1
    assert samples[0].rnti == 7
    assert samples[0].avg_prbs_dl == pytest.approx(4.0)
    assert samples[0].gnb_id == "gnb-0"

    source.send_control("gnb-0", sst=1, sd=1, min_ratio=5, max_ratio=25)
    bridge.join(timeout=5)
    assert not bridge.is_alive()

    ctrl_msg = ran_messages_pb2.RAN_message()
    ctrl_msg.ParseFromString(received_control["data"])
    slicing = ctrl_msg.ran_control_request.target_param_map[0].slicing_ctrl
    assert (slicing.min_ratio, slicing.max_ratio) == (5, 25)

    source.close()


def test_connect_timeout_raises_when_no_bridge_connects():
    source = LiveKpmSource(
        gnb_id="gnb-0", sub_port=SUB_PORT + 1, control_port=CONTROL_PORT + 1, connect_timeout_s=0.3,
    )
    with pytest.raises(TimeoutError):
        source.poll()
