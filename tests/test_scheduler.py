from __future__ import annotations

import asyncio
import socket

import pytest

from wireless_comm import Comm, CommConfig, CommOptions, Peer
from wireless_comm.errors import ConnectionClosedError, OperationTimeoutError
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
async def test_close_interrupts_an_active_pacing_wait() -> None:
    scheduler = ByteScheduler(
        quantum_bytes=100,
        rate_bytes_per_second=1,
        burst_bytes=100,
    )
    assert await scheduler.acquire("a", 100) == 100
    waiting = asyncio.create_task(scheduler.acquire("a", 100))
    await asyncio.sleep(0.01)

    await asyncio.wait_for(scheduler.close(), timeout=0.1)

    with pytest.raises(ConnectionClosedError, match="closed before grant"):
        await waiting


@pytest.mark.asyncio
async def test_cancelled_queued_grant_releases_its_peer() -> None:
    scheduler = ByteScheduler(
        quantum_bytes=100,
        rate_bytes_per_second=1000,
        burst_bytes=100,
    )
    assert await scheduler.acquire("a", 100) == 100
    pacing = asyncio.create_task(scheduler.acquire("a", 100))
    await asyncio.sleep(0)
    queued = asyncio.create_task(scheduler.acquire("b", 100))
    await asyncio.sleep(0)

    queued.cancel()
    with pytest.raises(asyncio.CancelledError):
        await queued
    pacing.cancel()
    with pytest.raises(asyncio.CancelledError):
        await pacing

    assert await asyncio.wait_for(scheduler.acquire("b", 1), timeout=0.05) == 1
    await scheduler.close()


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


@pytest.mark.asyncio
async def test_cancelled_partial_frame_is_not_reused() -> None:
    peer_a = Peer("a", "127.0.0.1", unused_local_port())
    peer_b = Peer("b", "127.0.0.1", unused_local_port())
    config = CommConfig(
        egress_quantum_bytes=4096,
        egress_rate_bytes_per_second=4096,
    )
    a = await Comm.create(local=peer_a, peers=(peer_b,), config=config)
    b = await Comm.create(local=peer_b, peers=(peer_a,))
    try:
        send = asyncio.create_task(
            a.send(
                b"x" * (64 * 1024),
                peer_b,
                options=CommOptions(tag=1, timeout=0.05),
            )
        )
        for _ in range(100):
            connections = a._states[peer_b.node_id].connections
            if connections:
                old_connection_id = next(iter(connections))
                break
            await asyncio.sleep(0)
        else:
            pytest.fail("sender did not establish its first connection")

        with pytest.raises(OperationTimeoutError):
            await send
        assert old_connection_id not in a._states[peer_b.node_id].connections

        await a.send(
            b"next-frame",
            peer_b,
            options=CommOptions(tag=2, timeout=1),
        )
        assert await b.recv(peer_a, CommOptions(tag=2, timeout=1)) == (
            b"next-frame",
            None,
        )
    finally:
        await asyncio.gather(a.close(), b.close())
