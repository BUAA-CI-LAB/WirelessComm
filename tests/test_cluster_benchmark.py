from __future__ import annotations

import asyncio
import socket
from pathlib import Path
from unittest.mock import patch

import pytest

from wireless_comm.cluster_benchmark import launch_cluster, run_cluster_node
from wireless_comm.cluster_config import (
    ClusterBenchmarkConfig,
    ClusterNode,
    ClusterTopology,
    load_cluster_config,
)
from wireless_comm.multirank import MultiRankSettings
from wireless_comm.types import CommConfig


class FakeMonitor:
    def __init__(self, interface: str | None, interval: float) -> None:
        self.interface = interface
        self.interval = interval

    def start(self) -> None:
        pass

    async def stop(self) -> list[object]:
        return []


def free_port_block(count: int) -> int:
    while True:
        with socket.socket() as probe:
            probe.bind(("127.0.0.1", 0))
            base = probe.getsockname()[1]
        if base + count > 65535:
            continue
        sockets = []
        try:
            for port in range(base, base + count):
                listener = socket.socket()
                listener.bind(("127.0.0.1", port))
                sockets.append(listener)
        except OSError:
            continue
        finally:
            for listener in sockets:
                listener.close()
        return base


def fake_environment(interface: str | None, peer_host: str) -> dict[str, object]:
    return {
        "interface": None,
        "kernel_network_counters": None,
        "peer_host": peer_host,
        "selected_interface": interface,
    }


def four_node_config() -> ClusterBenchmarkConfig:
    nodes = tuple(
        ClusterNode(
            f"node-{rank}",
            "127.0.0.1",
            (rank,),
            f"ssh-{rank}",
            "lo",
            bind_host="127.0.0.1",
        )
        for rank in range(4)
    )
    topology = ClusterTopology("node-0", free_port_block(4), nodes, (0, 2, 1, 3))
    topology.validate()
    settings = MultiRankSettings(
        payload_sizes=(64,),
        rounds=3,
        warmup_rounds=1,
        timeout=5,
    )
    return ClusterBenchmarkConfig(topology, settings)


def four_node_advanced_config() -> ClusterBenchmarkConfig:
    config = four_node_config()
    settings = MultiRankSettings(
        payload_sizes=(64,),
        rounds=2,
        warmup_rounds=1,
        collectives=(
            "broadcast",
            "ring_exchange",
            "all_to_all_exchange",
            "ring_allreduce",
        ),
        timeout=5,
        broadcast_root=2,
        ring_exchange_concurrency=1,
        allreduce_block_bytes=4,
    )
    return ClusterBenchmarkConfig(
        config.topology,
        settings,
        CommConfig(egress_quantum_bytes=16),
    )


def test_load_cluster_config() -> None:
    config = load_cluster_config("configs/cluster-four-node.example.yaml")
    assert config.topology.controller == "node-a"
    assert config.topology.world_size == 4
    assert config.topology.coordinator_rank("node-c") == 2
    assert config.topology.peers()[3].host == "192.0.2.103"
    assert config.benchmark.payload_sizes[-1] == 1048576


def test_topology_rejects_missing_rank() -> None:
    nodes = (
        ClusterNode("a", "a", (0,), "a", "wlan0"),
        ClusterNode("b", "b", (2,), "b", "wlan0"),
    )
    with pytest.raises(ValueError, match="contiguous"):
        ClusterTopology("a", 9400, nodes, (0, 2)).validate()


@pytest.mark.asyncio
async def test_four_physical_node_collectives() -> None:
    config = four_node_config()
    with (
        patch("wireless_comm.cluster_benchmark.LinkMonitor", FakeMonitor),
        patch(
            "wireless_comm.cluster_benchmark.collect_environment",
            side_effect=fake_environment,
        ),
    ):
        reports = await asyncio.gather(
            *(
                run_cluster_node(config, node.node_id, sample_interval=1)
                for node in config.topology.nodes
            )
        )
    controller = reports[0]
    assert controller["physical_node_count"] == 4
    assert controller["world_size"] == 4
    assert controller["ring"] == [0, 2, 1, 3]
    assert sorted(controller["nodes"]) == ["node-0", "node-1", "node-2", "node-3"]
    assert [result["rank"] for result in controller["rank_results"]] == [0, 1, 2, 3]
    assert (
        controller["collective_summary"]["ring_allreduce"]["64"][
            "collective_completion_ms"
        ]["sample_count"]
        == 3
    )


@pytest.mark.asyncio
async def test_configurable_collective_variants() -> None:
    config = four_node_advanced_config()
    with (
        patch("wireless_comm.cluster_benchmark.LinkMonitor", FakeMonitor),
        patch(
            "wireless_comm.cluster_benchmark.collect_environment",
            side_effect=fake_environment,
        ),
    ):
        reports = await asyncio.gather(
            *(
                run_cluster_node(config, node.node_id, sample_interval=1)
                for node in config.topology.nodes
            )
        )

    summary = reports[0]["collective_summary"]
    assert set(summary) == {
        "broadcast",
        "ring_exchange",
        "all_to_all_exchange",
        "ring_allreduce",
    }
    assert all(
        result["64"]["collective_completion_ms"]["sample_count"] == 2
        for result in summary.values()
    )
    exchange = summary["ring_exchange"]["64"]
    assert exchange["jain_fairness"]["sample_count"] == 2
    assert exchange["effective_round_mbps"]["p50"] > 0
    assert set(exchange["send_service_ms"]) == {"0", "1", "2", "3"}
    all_to_all = summary["all_to_all_exchange"]["64"]
    assert all_to_all["jain_fairness"]["sample_count"] == 2
    assert all_to_all["effective_round_mbps"]["p50"] > 0


@pytest.mark.asyncio
async def test_launcher_dry_run(tmp_path: Path) -> None:
    report = await launch_cluster(
        Path("configs/cluster-four-node.example.yaml"),
        output=tmp_path / "unused.json",
        timeout=10,
        dry_run=True,
    )
    assert len(report["copies"]) == 4
    assert set(report["workers"]) == {"node-a", "node-b", "node-c", "node-d"}
    assert "wireless_comm.cluster_benchmark worker" in report["workers"]["node-a"]
