"""Structured P2P communication for Wi-Fi-connected nodes."""

from .comm import Comm
from .errors import (
    CommError,
    ConnectionClosedError,
    ConnectionFailedError,
    MessageTooLargeError,
    OperationTimeoutError,
    ProtocolError,
    SerializationError,
    UnknownCodecError,
    UnsupportedPayloadError,
)
from .types import CommConfig, CommOptions, Metadata, Object, Peer, SendResult

__all__ = [
    "Comm",
    "CommConfig",
    "CommError",
    "CommOptions",
    "ConnectionClosedError",
    "ConnectionFailedError",
    "MessageTooLargeError",
    "Metadata",
    "Object",
    "OperationTimeoutError",
    "Peer",
    "ProtocolError",
    "SendResult",
    "SerializationError",
    "UnknownCodecError",
    "UnsupportedPayloadError",
]
