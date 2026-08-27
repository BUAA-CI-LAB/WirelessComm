"""Byte-granularity round-robin scheduling and optional egress pacing."""

from __future__ import annotations

import asyncio
import time
from collections import deque
from dataclasses import dataclass

from .errors import ConnectionClosedError


@dataclass(slots=True)
class _GrantRequest:
    peer_id: str
    requested_bytes: int
    completion: asyncio.Future[int]


class ByteScheduler:
    """Grant equal byte quanta to active peers in round-robin order."""

    def __init__(
        self,
        quantum_bytes: int,
        rate_bytes_per_second: int | None,
        burst_bytes: int,
    ) -> None:
        self.quantum_bytes = quantum_bytes
        self._rate = rate_bytes_per_second
        self._burst = burst_bytes
        self._tokens = float(burst_bytes)
        self._updated_at = time.monotonic()
        self._requests: deque[_GrantRequest] = deque()
        self._waiting_peers: set[str] = set()
        self._wakeup = asyncio.Event()
        self._runner: asyncio.Task[None] | None = None
        self._closed = False

    async def acquire(self, peer_id: str, requested_bytes: int) -> int:
        if self._closed:
            raise ConnectionClosedError("byte scheduler is closed")
        if peer_id in self._waiting_peers:
            raise RuntimeError(f"peer {peer_id!r} already has a pending byte grant")
        completion = asyncio.get_running_loop().create_future()
        self._requests.append(_GrantRequest(peer_id, requested_bytes, completion))
        self._waiting_peers.add(peer_id)
        self._wakeup.set()
        if self._runner is None or self._runner.done():
            self._runner = asyncio.create_task(
                self._run(), name="wireless-comm-byte-scheduler"
            )
        return await completion

    async def close(self) -> None:
        if self._closed:
            return
        self._closed = True
        while self._requests:
            request = self._requests.popleft()
            self._waiting_peers.discard(request.peer_id)
            if not request.completion.done():
                request.completion.set_exception(
                    ConnectionClosedError("byte scheduler closed before grant")
                )
        self._wakeup.set()
        if self._runner is not None:
            await self._runner

    async def _run(self) -> None:
        while not self._closed:
            await self._wakeup.wait()
            await asyncio.sleep(0)
            while self._requests:
                request = self._requests.popleft()
                self._waiting_peers.remove(request.peer_id)
                if request.completion.cancelled():
                    continue
                granted = min(request.requested_bytes, self.quantum_bytes)
                await self._pace(granted)
                if not request.completion.done():
                    request.completion.set_result(granted)
                await asyncio.sleep(0)
            self._wakeup.clear()

    async def _pace(self, byte_count: int) -> None:
        if self._rate is None:
            return
        while True:
            now = time.monotonic()
            elapsed = now - self._updated_at
            self._tokens = min(self._burst, self._tokens + elapsed * self._rate)
            self._updated_at = now
            if self._tokens >= byte_count:
                self._tokens -= byte_count
                return
            await asyncio.sleep((byte_count - self._tokens) / self._rate)
