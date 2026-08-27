from __future__ import annotations

import asyncio
import socket

import pytest

from wireless_comm import Comm, CommConfig, CommOptions, Peer
from wireless_comm.scheduler import ByteScheduler


def unused_local_port() -> int:
    with socket.socket() as listener:
        listener.bind(("127.0.0.1", 0))
        return listener.getsockname()[1]


@pytest.mark.asyncio
async def test_byte_scheduler_rotates_active_peers() -> None:
    scheduler = ByteScheduler(quantum_bytes=4, rate_bytes_per_second=None, burst_bytes=4)
    start = asyncio.Event()
    order = []

    async def request_twice(peer_id: str) -> None:
        await start.wait()
        for _ in range(2):
            assert await scheduler.acquire(peer_id, 10) == 4
            order.append(peer_id)

    tasks = [asyncio.create_task(request_twice(peer)) for peer in ("a", "b", "c")]
    start.set()
    await asyncio.gather(*tasks)
    await scheduler.close()

    assert order[:3] == ["a", "b", "c"]
    assert order[3:] == ["a", "b", "c"]


@pytest.mark.asyncio
async def test_scheduled_comm_sends_complete_frames_to_multiple_peers() -> None:
    peer_a = Peer("a", "127.0.0.1", unused_local_port())
    peer_b = Peer("b", "127.0.0.1", unused_local_port())
    peer_c = Peer("c", "127.0.0.1", unused_local_port())
    config = CommConfig(
        egress_quantum_bytes=4096,
        egress_rate_bytes_per_second=100 * 1024 * 1024,
    )
    a = await Comm.create(local=peer_a, peers=(peer_b, peer_c), config=config)
    b = await Comm.create(local=peer_b, peers=(peer_a,))
    c = await Comm.create(local=peer_c, peers=(peer_a,))
    payload_b = b"b" * (256 * 1024)
    payload_c = b"c" * (256 * 1024)
    try:
        await asyncio.gather(
            a.send(payload_b, peer_b),
            a.send(payload_c, peer_c),
        )
        received_b, received_c = await asyncio.gather(
            b.recv(peer_a, CommOptions(timeout=2)),
            c.recv(peer_a, CommOptions(timeout=2)),
        )
        assert received_b == (payload_b, None)
        assert received_c == (payload_c, None)
    finally:
        await asyncio.gather(a.close(), b.close(), c.close())
