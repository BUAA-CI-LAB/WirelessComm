"""Independent TCP background load for latency-under-load measurements."""

from __future__ import annotations

import asyncio
import json
import time
from dataclasses import dataclass
from typing import Any

_CHUNK = bytes(64 * 1024)


@dataclass(frozen=True, slots=True)
class LoadResult:
    direction: str
    streams: int
    target_mbps: float
    elapsed_s: float
    transferred_bytes: int

    @property
    def goodput_mbps(self) -> float:
        return self.transferred_bytes * 8 / self.elapsed_s / 1_000_000


class LoadServer:
    """Serve short-lived upload and download streams on a separate TCP port."""

    def __init__(self, host: str, port: int) -> None:
        self.host = host
        self.port = port
        self._server: asyncio.AbstractServer | None = None
        self._handlers: set[asyncio.Task[None]] = set()

    async def start(self) -> None:
        self._server = await asyncio.start_server(self._accept, self.host, self.port)

    async def close(self) -> None:
        if self._server is not None:
            self._server.close()
            await self._server.wait_closed()
        if self._handlers:
            await asyncio.gather(*self._handlers, return_exceptions=True)

    async def _accept(
        self, reader: asyncio.StreamReader, writer: asyncio.StreamWriter
    ) -> None:
        task = asyncio.current_task()
        if task is not None:
            self._handlers.add(task)
        try:
            request = json.loads(await reader.readline())
            direction = request["direction"]
            duration = float(request["duration_s"])
            target_mbps = float(request["target_mbps"])
            if direction not in {"upload", "download"}:
                raise ValueError(f"unknown load direction {direction!r}")
            if duration <= 0 or target_mbps <= 0:
                raise ValueError("load duration and target must be positive")
            writer.write(b"READY\n")
            await writer.drain()
            if await reader.readline() != b"GO\n":
                raise RuntimeError("load stream did not receive GO")
            if direction == "upload":
                await _receive_until_eof(reader)
            else:
                await _send_for_duration(writer, duration, target_mbps)
        finally:
            writer.close()
            await writer.wait_closed()
            if task is not None:
                self._handlers.discard(task)


async def run_load(
    host: str,
    port: int,
    *,
    direction: str,
    streams: int,
    duration: float,
    target_mbps: float,
) -> LoadResult:
    """Run aggregate rate-limited load and return initiator-observed bytes."""

    if direction not in {"upload", "download", "bidirectional"}:
        raise ValueError(f"unknown load direction {direction!r}")
    if streams <= 0 or duration <= 0 or target_mbps <= 0:
        raise ValueError("streams, duration, and target must be positive")

    directions = (
        [direction] * streams
        if direction != "bidirectional"
        else ["upload"] * streams + ["download"] * streams
    )
    target_per_stream = target_mbps / len(directions)
    connections = await asyncio.gather(
        *(
            _open_stream(host, port, stream_direction, duration, target_per_stream)
            for stream_direction in directions
        )
    )
    started = time.monotonic()
    for reader, writer, _ in connections:
        writer.write(b"GO\n")
    await asyncio.gather(*(writer.drain() for _, writer, _ in connections))
    transferred = await asyncio.gather(
        *(
            _run_stream(reader, writer, stream_direction, duration, target_per_stream)
            for reader, writer, stream_direction in connections
        )
    )
    elapsed = time.monotonic() - started
    return LoadResult(
        direction, len(directions), target_mbps, elapsed, sum(transferred)
    )


async def _open_stream(
    host: str,
    port: int,
    direction: str,
    duration: float,
    target_mbps: float,
) -> tuple[asyncio.StreamReader, asyncio.StreamWriter, str]:
    reader, writer = await asyncio.open_connection(host, port)
    request = json.dumps(
        {
            "direction": direction,
            "duration_s": duration,
            "target_mbps": target_mbps,
        },
        separators=(",", ":"),
    ).encode()
    writer.write(request + b"\n")
    await writer.drain()
    if await reader.readline() != b"READY\n":
        raise RuntimeError("load stream did not become ready")
    return reader, writer, direction


async def _run_stream(
    reader: asyncio.StreamReader,
    writer: asyncio.StreamWriter,
    direction: str,
    duration: float,
    target_mbps: float,
) -> int:
    try:
        if direction == "upload":
            return await _send_for_duration(writer, duration, target_mbps)
        return await _receive_until_eof(reader)
    finally:
        writer.close()
        await writer.wait_closed()


async def _send_for_duration(
    writer: asyncio.StreamWriter, duration: float, target_mbps: float
) -> int:
    bytes_per_second = target_mbps * 1_000_000 / 8
    started = time.monotonic()
    deadline = started + duration
    sent = 0
    while time.monotonic() < deadline:
        writer.write(_CHUNK)
        sent += len(_CHUNK)
        await writer.drain()
        expected_elapsed = sent / bytes_per_second
        delay = started + expected_elapsed - time.monotonic()
        if delay > 0:
            await asyncio.sleep(delay)
    return sent


async def _receive_until_eof(reader: asyncio.StreamReader) -> int:
    received = 0
    while chunk := await reader.read(256 * 1024):
        received += len(chunk)
    return received


def load_result_dict(result: LoadResult) -> dict[str, Any]:
    return {
        "direction": result.direction,
        "streams": result.streams,
        "target_mbps": result.target_mbps,
        "elapsed_s": result.elapsed_s,
        "transferred_bytes": result.transferred_bytes,
        "goodput_mbps": result.goodput_mbps,
    }
