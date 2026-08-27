from __future__ import annotations

import asyncio
import json
import socket
from unittest.mock import patch

import pytest

from wireless_comm.benchmark import (
    BenchmarkSettings,
    run_initiator,
    run_responder,
    summarize_samples,
)
from wireless_comm.config import RuntimeConfig
from wireless_comm.diagnostics import (
    counter_delta,
    kernel_counter_delta,
    parse_context_switches,
    parse_interrupt,
    parse_iw_link,
    parse_softirqs,
    parse_softnet_stat,
)
from wireless_comm.load import LoadServer, run_load
from wireless_comm.types import CommConfig, Peer


def unused_local_port() -> int:
    with socket.socket() as listener:
        listener.bind(("127.0.0.1", 0))
        return listener.getsockname()[1]


def unused_adjacent_ports() -> int:
    while True:
        port = unused_local_port()
        if port < 65535:
            with socket.socket() as listener:
                try:
                    listener.bind(("127.0.0.1", port + 1))
                except OSError:
                    continue
            return port


def runtime_pair() -> tuple[RuntimeConfig, RuntimeConfig]:
    node_a = Peer("node-a", "127.0.0.1", unused_adjacent_ports())
    node_b = Peer("node-b", "127.0.0.1", unused_adjacent_ports())
    return (
        RuntimeConfig(node_a, (node_b,), node_a.host, CommConfig()),
        RuntimeConfig(node_b, (node_a,), node_b.host, CommConfig()),
    )


def fake_environment(interface: str | None, peer_host: str) -> dict[str, object]:
    return {
        "interface": None,
        "kernel_network_counters": None,
        "peer_host": peer_host,
        "selected_interface": interface,
    }


def fake_ping(peer_host: str, interface: str | None, count: int) -> dict[str, object]:
    return {"peer_host": peer_host, "interface": interface, "received": count}


def test_settings_clamp_message_count() -> None:
    settings = BenchmarkSettings(
        throughput_target_bytes=1024,
        min_throughput_messages=2,
        max_throughput_messages=8,
    )
    assert settings.message_count(1) == 8
    assert settings.message_count(512) == 2
    assert settings.message_count(4096) == 2


def test_settings_validate_latency_under_load() -> None:
    with pytest.raises(ValueError, match="matching throughput directions"):
        BenchmarkSettings(
            directions=("upload",),
            load_fractions=(0.5,),
            load_directions=("download",),
        )


@pytest.mark.asyncio
@pytest.mark.parametrize("direction", ["upload", "download", "bidirectional"])
async def test_independent_background_load(direction: str) -> None:
    port = unused_local_port()
    server = LoadServer("127.0.0.1", port)
    await server.start()
    try:
        result = await run_load(
            "127.0.0.1",
            port,
            direction=direction,
            streams=2,
            duration=0.1,
            target_mbps=10,
        )
    finally:
        await server.close()
    assert result.direction == direction
    assert result.streams == (4 if direction == "bidirectional" else 2)
    assert result.transferred_bytes > 0
    assert result.goodput_mbps > 0


def test_latency_summary_uses_interpolated_percentiles() -> None:
    summary = summarize_samples([1.0, 2.0, 3.0, 4.0])
    assert summary["mean"] == 2.5
    assert summary["p50"] == 2.5
    assert summary["p95"] == pytest.approx(3.85)
    assert summary["max"] == 4.0
    assert summary["jitter_mean_abs_delta"] == 1.0
    assert summary["outlier_count"] == 0


def test_parse_iw_link() -> None:
    fields = parse_iw_link(
        """Connected to aa:bb:cc:dd:ee:ff (on wlan0)
        SSID: test-wifi
        freq: 5180
        signal: -47.00 dBm
        rx bitrate: 866.7 MBit/s VHT-MCS 9
        tx bitrate: 780.0 MBit/s VHT-MCS 8
        """
    )
    assert fields == {
        "access_point": "aa:bb:cc:dd:ee:ff",
        "ssid": "test-wifi",
        "frequency_mhz": 5180,
        "signal_dbm": -47.0,
        "rx_bitrate": "866.7 MBit/s VHT-MCS 9",
        "tx_bitrate": "780.0 MBit/s VHT-MCS 8",
    }


def test_counter_delta() -> None:
    names = (
        "rx_bytes",
        "rx_packets",
        "rx_errors",
        "rx_dropped",
        "tx_bytes",
        "tx_packets",
        "tx_errors",
        "tx_dropped",
    )
    before = {"statistics": {name: 10 for name in names}}
    after = {"statistics": {name: 14 for name in names}}
    assert counter_delta(before, after) == {name: 4 for name in names}


def test_parse_kernel_network_counters() -> None:
    assert parse_context_switches("cpu 1 2 3\nctxt 1234\n") == 1234
    assert parse_softirqs("NET_TX: 1 2\nNET_RX: 3 4\n") == {
        "NET_TX": {"total": 3, "per_cpu": [1, 2]},
        "NET_RX": {"total": 7, "per_cpu": [3, 4]},
    }
    softnet = parse_softnet_stat(
        "00000001 00000002 00000003 0 0 0 0 0 00000004 00000005 00000006\n"
        "0000000a 0000000b 0000000c 0 0 0 0 0 0000000d 0000000e 0000000f\n"
    )
    assert softnet["processed"] == {"total": 11, "per_cpu": [1, 10]}
    assert softnet["dropped"] == {"total": 13, "per_cpu": [2, 11]}
    assert softnet["time_squeeze"] == {"total": 15, "per_cpu": [3, 12]}
    assert softnet["received_rps"] == {"total": 19, "per_cpu": [5, 14]}


def test_parse_interface_interrupt_and_delta() -> None:
    output = "           CPU0       CPU1\n 42:         10         20  PCI-MSI  wifi\n"
    before_irq = parse_interrupt(output, 42)
    after_irq = parse_interrupt(output.replace("10", "13").replace("20", "25"), 42)
    assert before_irq == {
        "irq": 42,
        "total": 30,
        "per_cpu": [10, 20],
        "description": "PCI-MSI wifi",
    }
    before = {
        "context_switches": 100,
        "softirqs": {"NET_RX": {"total": 5, "per_cpu": [2, 3]}},
        "softnet": {"dropped": {"total": 0, "per_cpu": [0, 0]}},
        "interface_irq": before_irq,
    }
    after = {
        "context_switches": 125,
        "softirqs": {"NET_RX": {"total": 12, "per_cpu": [6, 6]}},
        "softnet": {"dropped": {"total": 1, "per_cpu": [0, 1]}},
        "interface_irq": after_irq,
    }
    assert kernel_counter_delta(before, after) == {
        "context_switches": 25,
        "softirqs": {"NET_RX": {"total": 7, "per_cpu": [4, 3]}},
        "softnet": {"dropped": {"total": 1, "per_cpu": [0, 1]}},
        "interface_irq": {
            "irq": 42,
            "description": "PCI-MSI wifi",
            "total": 8,
            "per_cpu": [3, 5],
        },
    }


@pytest.mark.asyncio
async def test_complete_benchmark_session_over_tcp() -> None:
    initiator_config, responder_config = runtime_pair()
    settings = BenchmarkSettings(
        latency_sizes=(0, 8),
        latency_rounds=3,
        warmup_rounds=1,
        throughput_sizes=(128,),
        throughput_target_bytes=256,
        min_throughput_messages=2,
        max_throughput_messages=2,
        throughput_trials=2,
        directions=("upload", "download", "bidirectional"),
        ping_count=1,
        operation_timeout=5,
        sample_interval=1,
        load_fractions=(0.5,),
        load_directions=("upload",),
        load_concurrencies=(1,),
        load_duration=0.2,
        load_probe_interval=0.03,
    )
    patches = (
        patch("wireless_comm.benchmark.resolve_interface", return_value=None),
        patch(
            "wireless_comm.benchmark.collect_environment",
            side_effect=fake_environment,
        ),
        patch("wireless_comm.benchmark.run_ping", side_effect=fake_ping),
    )
    for active_patch in patches:
        active_patch.start()
    try:
        responder_peer = responder_config.peers[0]
        responder = asyncio.create_task(
            run_responder(
                responder_config,
                responder_peer,
                interface=None,
                operation_timeout=5,
                sample_interval=1,
            )
        )
        await asyncio.sleep(0.05)
        report = await run_initiator(
            initiator_config,
            initiator_config.peers[0],
            settings,
            interface=None,
        )
        responder_report = await responder
    finally:
        for active_patch in reversed(patches):
            active_patch.stop()

    assert report["schema_version"] == 4
    assert report["initiator"]["process_observation_duration_s"] > 0
    assert report["responder"]["process_observation_duration_s"] > 0
    assert len(report["latency_under_load"]) == 1
    load = report["latency_under_load"][0]
    assert load["direction"] == "upload"
    assert load["probe_summary_ms"]["sample_count"] > 0
    assert load["background_load"]["goodput_mbps"] > 0
    assert len(report["latency"]) == 2
    assert report["responder"] == responder_report
    for direction in ("upload", "download", "bidirectional"):
        result = report["throughput"][direction][0]
        assert result["payload_bytes_per_message"] == 128
        assert result["messages_per_direction"] == 2
        assert result["trial_count"] == 2
        assert result["aggregate_goodput_mbps"]["p50"] > 0
        assert result["per_direction_goodput_mbps"]["p50"] > 0
        assert len(result["trials"]) == 2
    assert report["throughput"]["bidirectional"][0]["streams"] == 2
    json.dumps(report)
