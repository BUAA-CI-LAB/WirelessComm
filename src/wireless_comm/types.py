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
        if not isinstance(self.node_id, str) or not self.node_id:
            raise ValueError("Peer.node_id must be non-empty")
        if not isinstance(self.host, str) or not self.host:
            raise ValueError("Peer.host must be non-empty")
        if type(self.port) is not int or not 1 <= self.port <= 65535:
            raise ValueError("Peer.port must be between 1 and 65535")


@dataclass(frozen=True, slots=True)
class CommOptions:
    """Options shared by one send or receive operation."""

    tag: int = 0
    timeout: float | None = None
    wait_for_capacity: bool = True

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
    max_peer_queued_messages: int = 1024
    max_peer_queued_bytes: int = 128 * 1024 * 1024
    allow_unsafe_pickle: bool = False
    tcp_nodelay: bool = True
    egress_quantum_bytes: int | None = None
    egress_rate_bytes_per_second: int | None = None
    egress_burst_bytes: int | None = None

    def __post_init__(self) -> None:
        positive_limits = {
            "max_message_bytes": self.max_message_bytes,
            "max_manifest_bytes": self.max_manifest_bytes,
            "max_metadata_bytes": self.max_metadata_bytes,
            "max_segments": self.max_segments,
            "max_container_depth": self.max_container_depth,
            "max_container_nodes": self.max_container_nodes,
            "max_peer_queued_messages": self.max_peer_queued_messages,
            "max_peer_queued_bytes": self.max_peer_queued_bytes,
        }
        for name, value in positive_limits.items():
            if type(value) is not int or value <= 0:
                raise ValueError(f"{name} must be positive")
        optional_limits = {
            "egress_quantum_bytes": self.egress_quantum_bytes,
            "egress_rate_bytes_per_second": self.egress_rate_bytes_per_second,
            "egress_burst_bytes": self.egress_burst_bytes,
        }
        for name, value in optional_limits.items():
            if value is not None and (type(value) is not int or value <= 0):
                raise ValueError(f"{name} must be positive when configured")
        if self.egress_rate_bytes_per_second is not None and self.egress_quantum_bytes is None:
            raise ValueError("egress pacing requires egress_quantum_bytes")
        if self.egress_burst_bytes is not None and self.egress_quantum_bytes is None:
            raise ValueError("egress burst requires egress_quantum_bytes")
        if (
            self.egress_burst_bytes is not None
            and self.egress_quantum_bytes is not None
            and self.egress_burst_bytes < self.egress_quantum_bytes
        ):
            raise ValueError("egress burst must be at least one quantum")


@dataclass(frozen=True, slots=True)
class SendResult:
    """Information returned after a complete frame is locally written."""

    message_id: int
    wire_bytes: int
