from __future__ import annotations

import asyncio
import socket
from unittest.mock import patch

import pytest

from wireless_comm import Comm
from wireless_comm.multirank import MultiRankSettings, Topology, run_side


def adjacent_ports() -> int:
    while True:
        with socket.socket() as first:
            first.bind(("127.0.0.1", 0))
            port = first.getsockname()[1]
        if port == 65535:
            continue
        with socket.socket() as second:
            try:
                second.bind(("127.0.0.1", port + 1))
            except OSError:
                continue
        return port


def fake_environment(interface: str | None, peer_host: str) -> dict[str, object]:
    return {
        "interface": None,
        "kernel_network_counters": None,
        "peer_host": peer_host,
        "selected_interface": interface,
    }


def test_alternating_rank_topology() -> None:
    topology = Topology("host-a", "host-b", 9400, 3, True)
    assert topology.local_ranks == (0, 2, 4)
    assert [(peer.host, peer.port) for peer in topology.peers()] == [
        ("host-a", 9400),
        ("host-b", 9401),
        ("host-a", 9402),
        ("host-b", 9403),
        ("host-a", 9404),
        ("host-b", 9405),
    ]


@pytest.mark.asyncio
async def test_four_local_comm_directories_are_distinct() -> None:
    port = adjacent_ports()
    topology = Topology("127.0.0.1", "127.0.0.1", port, 2, True)
    directory = topology.peers()
    comms = await asyncio.gather(
        *(
            Comm.create(
                local=directory[rank],
                peers=tuple(
                    peer for peer in directory if peer.node_id != f"rank-{rank}"
                ),
                bind_host="127.0.0.1",
            )
            for rank in topology.local_ranks
        )
    )
    try:
        assert [comm.local.node_id for comm in comms] == ["rank-0", "rank-2"]
        assert all(len(comm.peers()) == 3 for comm in comms)
    finally:
        await asyncio.gather(*(comm.close() for comm in comms))


@pytest.mark.asyncio
async def test_two_side_collectives_over_tcp() -> None:
    port = adjacent_ports()
    initiator_topology = Topology("127.0.0.1", "127.0.0.1", port, 1, True)
    responder_topology = Topology("127.0.0.1", "127.0.0.1", port, 1, False)
    settings = MultiRankSettings(
        payload_sizes=(64,),
        rounds=3,
        warmup_rounds=1,
        collectives=("barrier", "broadcast", "allgather", "ring_allreduce"),
        timeout=5,
    )
    patches = (
        patch("wireless_comm.multirank.resolve_interface", return_value=None),
        patch(
            "wireless_comm.multirank.collect_environment",
            side_effect=fake_environment,
        ),
    )
    for active_patch in patches:
        active_patch.start()
    try:
        responder = asyncio.create_task(
            run_side(
                responder_topology,
                settings=None,
                interface=None,
                sample_interval=1,
            )
        )
        report = await run_side(
            initiator_topology,
            settings=settings,
            interface=None,
            sample_interval=1,
        )
        responder_report = await responder
    finally:
        for active_patch in reversed(patches):
            active_patch.stop()

    assert report["world_size"] == 2
    assert report["remote_side"]["local_ranks"] == [1]
    assert [result["rank"] for result in report["rank_results"]] == [0, 1]
    assert (
        report["collective_summary"]["barrier"]["0"]["collective_completion_ms"][
            "sample_count"
        ]
        == 3
    )
    assert (
        report["collective_summary"]["allgather"]["64"]["collective_completion_ms"][
            "p50"
        ]
        > 0
    )
    assert (
        report["collective_summary"]["ring_allreduce"]["64"][
            "collective_completion_ms"
        ]["p50"]
        > 0
    )
    assert responder_report["side"]["local_ranks"] == [1]
