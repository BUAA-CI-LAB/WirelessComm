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


@dataclass(slots=True)
class _TokenBucket:
    rate: int
    capacity: int
    tokens: float
    updated_at: float

    @classmethod
    def full(cls, rate: int, capacity: int) -> _TokenBucket:
        return cls(rate, capacity, float(capacity), time.monotonic())

    def delay(self, byte_count: int, now: float) -> float:
        elapsed = now - self.updated_at
        self.tokens = min(self.capacity, self.tokens + elapsed * self.rate)
        self.updated_at = now
        if self.tokens >= byte_count:
            return 0.0
        return (byte_count - self.tokens) / self.rate

    def consume(self, byte_count: int) -> None:
        self.tokens -= byte_count


class ByteScheduler:
    """Grant one byte quantum to each active peer in round-robin order."""

    def __init__(
        self,
        quantum_bytes: int,
        rate_bytes_per_second: int | None,
        burst_bytes: int,
    ) -> None:
        if type(quantum_bytes) is not int or quantum_bytes <= 0:
            raise ValueError("quantum_bytes must be positive")
        if rate_bytes_per_second is not None and (
            type(rate_bytes_per_second) is not int
            or rate_bytes_per_second <= 0
        ):
            raise ValueError("rate_bytes_per_second must be positive when configured")
        if type(burst_bytes) is not int or burst_bytes < quantum_bytes:
            raise ValueError("burst_bytes must be at least one quantum")
        self.quantum_bytes = quantum_bytes
        self._bucket = (
            _TokenBucket.full(rate_bytes_per_second, burst_bytes)
            if rate_bytes_per_second is not None
            else None
        )
        self._requests: deque[_GrantRequest] = deque()
        self._pending_by_peer: dict[str, _GrantRequest] = {}
        self._wakeup = asyncio.Event()
        self._pace_wakeup = asyncio.Event()
        self._runner: asyncio.Task[None] | None = None
        self._closed = False

    async def acquire(self, peer_id: str, requested_bytes: int) -> int:
        if not isinstance(peer_id, str) or not peer_id:
            raise ValueError("peer_id must be non-empty")
        if type(requested_bytes) is not int or requested_bytes <= 0:
            raise ValueError("requested_bytes must be positive")
        if self._closed:
            raise ConnectionClosedError("byte scheduler is closed")
        if peer_id in self._pending_by_peer:
            raise RuntimeError(f"peer {peer_id!r} already has a pending byte grant")
        if self._runner is not None and self._runner.done():
            self._runner.result()
            raise RuntimeError("byte scheduler runner stopped unexpectedly")
        completion = asyncio.get_running_loop().create_future()
        request = _GrantRequest(peer_id, requested_bytes, completion)
        self._requests.append(request)
        self._pending_by_peer[peer_id] = request
        self._wakeup.set()
        self._pace_wakeup.set()
        if self._runner is None:
            self._runner = asyncio.create_task(
                self._run(), name="wireless-comm-byte-scheduler"
            )
        try:
            return await completion
        except asyncio.CancelledError:
            try:
                self._requests.remove(request)
            except ValueError:
                pass
            else:
                self._release_peer(request)
            self._pace_wakeup.set()
            raise

    async def close(self) -> None:
        if self._closed:
            return
        self._closed = True
        if self._runner is not None:
            self._runner.cancel()
            try:
                await self._runner
            except asyncio.CancelledError:
                pass
            self._runner = None
        self._fail_queued_requests()

    async def _run(self) -> None:
        try:
            while True:
                while not self._requests:
                    self._wakeup.clear()
                    await self._wakeup.wait()

                # Let requests made in the same event-loop turn join the round.
                await asyncio.sleep(0)
                if not self._requests:
                    continue
                request = self._requests.popleft()
                if request.completion.cancelled():
                    self._release_peer(request)
                    continue
                granted = min(request.requested_bytes, self.quantum_bytes)
                delay = self._grant_delay(granted)
                if delay > 0:
                    self._requests.append(request)
                    await self._wait_for_pacing(delay)
                    continue
                if self._bucket is not None:
                    self._bucket.consume(granted)
                request.completion.set_result(granted)
                self._release_peer(request)
        finally:
            self._fail_queued_requests()

    def _grant_delay(self, byte_count: int) -> float:
        if self._bucket is None:
            return 0.0
        return self._bucket.delay(byte_count, time.monotonic())

    async def _wait_for_pacing(self, delay: float) -> None:
        self._pace_wakeup.clear()
        timer = asyncio.get_running_loop().call_later(delay, self._pace_wakeup.set)
        try:
            await self._pace_wakeup.wait()
        finally:
            timer.cancel()

    def _fail_queued_requests(self) -> None:
        while self._requests:
            request = self._requests.popleft()
            self._release_peer(request)
            if not request.completion.done():
                request.completion.set_exception(
                    ConnectionClosedError("byte scheduler closed before grant")
                )

    def _release_peer(self, request: _GrantRequest) -> None:
        if self._pending_by_peer.get(request.peer_id) is request:
            del self._pending_by_peer[request.peer_id]
