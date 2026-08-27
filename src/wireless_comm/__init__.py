"""Structured P2P communication for Wi-Fi-connected nodes."""

from .comm import Comm
from .config import RuntimeConfig, load_runtime_config
from .errors import (
    BackpressureError,
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
    "BackpressureError",
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
    "RuntimeConfig",
    "SendResult",
    "SerializationError",
    "UnknownCodecError",
    "UnsupportedPayloadError",
    "load_runtime_config",
]
