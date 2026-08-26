"""Versioned framing for TCP peer connections."""

from __future__ import annotations

import asyncio
import struct
from dataclasses import dataclass
from enum import IntEnum

import msgpack

from .codec import BufferSegment, EncodedPayload
from .errors import MessageTooLargeError, ProtocolError
from .types import CommConfig

MAGIC = b"COMM"
VERSION = 1
HEADER = struct.Struct("!4sHBBIQIIQ")


class MessageKind(IntEnum):
    HELLO = 1
    HELLO_ACK = 2
    CLOSE = 3
    DATA = 16


@dataclass(frozen=True, slots=True)
class Frame:
    kind: MessageKind
    tag: int
    message_id: int
    payload: EncodedPayload


def encode_frame(frame: Frame, config: CommConfig) -> tuple[bytes | memoryview, ...]:
    manifest_size = len(frame.payload.manifest)
    payload_size = sum(segment.nbytes for segment in frame.payload.segments)
    segment_count = len(frame.payload.segments)
    wire_size = HEADER.size + manifest_size + payload_size
    if manifest_size > config.max_manifest_bytes:
        raise MessageTooLargeError("manifest exceeds configured limit")
    if segment_count > config.max_segments:
        raise MessageTooLargeError("segment count exceeds configured limit")
    if wire_size > config.max_message_bytes:
        raise MessageTooLargeError("message exceeds configured limit")
    header = HEADER.pack(
        MAGIC,
        VERSION,
        int(frame.kind),
        0,
        frame.tag,
        frame.message_id,
        manifest_size,
        segment_count,
        payload_size,
    )
    return (header, frame.payload.manifest, *(s.view for s in frame.payload.segments))


async def read_frame(reader: asyncio.StreamReader, config: CommConfig) -> Frame:
    raw_header = await reader.readexactly(HEADER.size)
    magic, version, raw_kind, flags, tag, message_id, manifest_size, count, size = (
        HEADER.unpack(raw_header)
    )
    if magic != MAGIC or version != VERSION or flags != 0:
        raise ProtocolError("invalid magic, protocol version, or flags")
    try:
        kind = MessageKind(raw_kind)
    except ValueError as exc:
        raise ProtocolError(f"unknown message kind {raw_kind}") from exc
    if manifest_size > config.max_manifest_bytes or count > config.max_segments:
        raise MessageTooLargeError("incoming manifest or segment count exceeds limit")
    if HEADER.size + manifest_size + size > config.max_message_bytes:
        raise MessageTooLargeError("incoming message exceeds configured limit")

    manifest = await reader.readexactly(manifest_size)
    try:
        metadata = msgpack.unpackb(manifest, raw=False, strict_map_key=False)
        lengths = metadata["segments"]
    except Exception as exc:
        raise ProtocolError(f"invalid segment manifest: {exc}") from exc
    if not isinstance(lengths, list) or len(lengths) != count:
        raise ProtocolError("segment count does not match manifest")
    if any(type(length) is not int or length < 0 for length in lengths):
        raise ProtocolError("manifest contains an invalid segment length")
    if sum(lengths) != size:
        raise ProtocolError("payload size does not match segment lengths")

    segments: list[BufferSegment] = []
    for length in lengths:
        value = await reader.readexactly(length)
        segments.append(BufferSegment(memoryview(value), value))
    return Frame(kind, tag, message_id, EncodedPayload(manifest, tuple(segments)))


def control_payload(fields: dict[str, object]) -> EncodedPayload:
    manifest = msgpack.packb(
        {"control": fields, "segments": []},
        use_bin_type=True,
    )
    return EncodedPayload(manifest, ())


def decode_control(payload: EncodedPayload) -> dict[str, object]:
    if payload.segments:
        raise ProtocolError("control frame must not contain segments")
    try:
        manifest = msgpack.unpackb(payload.manifest, raw=False, strict_map_key=False)
        fields = manifest["control"]
    except Exception as exc:
        raise ProtocolError(f"invalid control frame: {exc}") from exc
    if not isinstance(fields, dict):
        raise ProtocolError("control fields must be a map")
    return fields
