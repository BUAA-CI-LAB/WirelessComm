"""Public value types for the P2P runtime."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, TypeAlias

Object: TypeAlias = Any
Metadata: TypeAlias = dict[str, Any]


@dataclass(frozen=True, slots=True)
class Peer:
    """The stable identity and advertised TCP endpoint of one node."""

    node_id: str
    host: str
    port: int

    def __post_init__(self) -> None:
        if not self.node_id:
            raise ValueError("Peer.node_id must be non-empty")
        if not self.host:
            raise ValueError("Peer.host must be non-empty")
        if not 1 <= self.port <= 65535:
            raise ValueError("Peer.port must be between 1 and 65535")


@dataclass(frozen=True, slots=True)
class CommOptions:
    """Options shared by one send or receive operation."""

    tag: int = 0
    timeout: float | None = None

    def __post_init__(self) -> None:
        if not 0 <= self.tag <= 0xFFFFFFFF:
            raise ValueError("tag must fit in an unsigned 32-bit integer")
        if self.timeout is not None and self.timeout <= 0:
            raise ValueError("timeout must be positive")


@dataclass(frozen=True, slots=True)
class CommConfig:
    """Runtime-wide protocol and safety settings."""

    max_message_bytes: int = 256 * 1024 * 1024
    max_manifest_bytes: int = 4 * 1024 * 1024
    max_metadata_bytes: int = 64 * 1024
    max_segments: int = 4096
    max_container_depth: int = 64
    max_container_nodes: int = 1_000_000
    allow_unsafe_pickle: bool = False
    tcp_nodelay: bool = True


@dataclass(frozen=True, slots=True)
class SendResult:
    """Information returned after a complete frame is locally written."""

    message_id: int
    wire_bytes: int
