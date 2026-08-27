"""Strict configuration for physical multi-node Wi-Fi benchmarks."""

from __future__ import annotations

from dataclasses import dataclass, field, fields
from pathlib import Path
from typing import Any

import yaml

from .multirank import MultiRankSettings
from .types import CommConfig, Peer


@dataclass(frozen=True, slots=True)
class ClusterNode:
    node_id: str
    host: str
    ranks: tuple[int, ...]
    ssh_alias: str
    interface: str
    bind_host: str = "0.0.0.0"
    python: str = "python3"
    workdir: str = "."


@dataclass(frozen=True, slots=True)
class ClusterTopology:
    controller: str
    base_port: int
    nodes: tuple[ClusterNode, ...]
    ring: tuple[int, ...]

    @property
    def world_size(self) -> int:
        return len(self.ring)

    def node(self, node_id: str) -> ClusterNode:
        for node in self.nodes:
            if node.node_id == node_id:
                return node
        raise KeyError(f"unknown cluster node {node_id!r}")

    def owner(self, rank: int) -> ClusterNode:
        for node in self.nodes:
            if rank in node.ranks:
                return node
        raise KeyError(f"rank {rank} has no physical owner")

    def peers(self) -> tuple[Peer, ...]:
        return tuple(
            Peer(f"rank-{rank}", self.owner(rank).host, self.base_port + rank)
            for rank in range(self.world_size)
        )

    def coordinator_rank(self, node_id: str) -> int:
        return min(self.node(node_id).ranks)

    def validate(self) -> None:
        if len(self.nodes) < 2:
            raise ValueError("cluster benchmark requires at least two physical nodes")
        if not isinstance(self.controller, str) or not self.controller:
            raise ValueError("controller must be a non-empty string")
        if type(self.base_port) is not int:
            raise TypeError("base_port must be an integer")
        for node in self.nodes:
            for name, value in {
                "node_id": node.node_id,
                "host": node.host,
                "ssh_alias": node.ssh_alias,
                "interface": node.interface,
                "bind_host": node.bind_host,
                "python": node.python,
                "workdir": node.workdir,
            }.items():
                if not isinstance(value, str) or not value:
                    raise ValueError(f"{node.node_id}.{name} must be non-empty")
        node_ids = [node.node_id for node in self.nodes]
        if len(set(node_ids)) != len(node_ids):
            raise ValueError("cluster node_id values must be unique")
        if self.controller not in node_ids:
            raise ValueError("controller must name a configured node")
        ranks = [rank for node in self.nodes for rank in node.ranks]
        if any(not node.ranks for node in self.nodes):
            raise ValueError("every physical node must own at least one rank")
        if sorted(ranks) != list(range(len(ranks))):
            raise ValueError("ranks must be unique and contiguous from zero")
        if set(self.ring) != set(ranks) or len(self.ring) != len(ranks):
            raise ValueError("ring must contain every rank exactly once")
        if 0 not in self.node(self.controller).ranks:
            raise ValueError("controller node must own rank 0")
        if self.base_port <= 0 or self.base_port + len(ranks) - 1 > 65535:
            raise ValueError("rank port range is invalid")


@dataclass(frozen=True, slots=True)
class ClusterBenchmarkConfig:
    topology: ClusterTopology
    benchmark: MultiRankSettings
    comm: CommConfig = field(default_factory=CommConfig)


def load_cluster_config(path: str | Path) -> ClusterBenchmarkConfig:
    with Path(path).open(encoding="utf-8") as stream:
        document = yaml.safe_load(stream)
    if not isinstance(document, dict):
        raise TypeError("cluster configuration root must be a mapping")
    _reject_unknown(
        document,
        {"controller", "base_port", "nodes", "ring", "benchmark", "comm"},
        "configuration",
    )

    raw_nodes = document.get("nodes")
    if not isinstance(raw_nodes, list):
        raise TypeError("nodes must be a list")
    nodes = tuple(_parse_node(raw, index) for index, raw in enumerate(raw_nodes))
    raw_ring = document.get("ring")
    if not isinstance(raw_ring, list) or any(
        type(rank) is not int for rank in raw_ring
    ):
        raise TypeError("ring must be a list of integer ranks")
    topology = ClusterTopology(
        controller=document.get("controller"),
        base_port=document.get("base_port"),
        nodes=nodes,
        ring=tuple(raw_ring),
    )
    topology.validate()

    raw_benchmark = document.get("benchmark")
    if not isinstance(raw_benchmark, dict):
        raise TypeError("benchmark must be a mapping")
    _reject_unknown(
        raw_benchmark,
        {
            "payload_sizes",
            "rounds",
            "warmup_rounds",
            "collectives",
            "timeout",
            "broadcast_root",
            "ring_exchange_concurrency",
            "allreduce_block_bytes",
        },
        "benchmark",
    )
    benchmark = MultiRankSettings(
        payload_sizes=_integer_tuple(raw_benchmark, "payload_sizes"),
        rounds=raw_benchmark.get("rounds", 20),
        warmup_rounds=raw_benchmark.get("warmup_rounds", 3),
        collectives=tuple(
            raw_benchmark.get(
                "collectives",
                ("barrier", "broadcast", "allgather", "ring_allreduce"),
            )
        ),
        timeout=raw_benchmark.get("timeout", 120.0),
        broadcast_root=raw_benchmark.get("broadcast_root", 0),
        ring_exchange_concurrency=raw_benchmark.get("ring_exchange_concurrency"),
        allreduce_block_bytes=raw_benchmark.get("allreduce_block_bytes"),
    )
    raw_comm = document.get("comm", {})
    if not isinstance(raw_comm, dict):
        raise TypeError("comm must be a mapping")
    _reject_unknown(
        raw_comm,
        {field.name for field in fields(CommConfig)},
        "comm",
    )
    return ClusterBenchmarkConfig(topology, benchmark, CommConfig(**raw_comm))


def _parse_node(raw: Any, index: int) -> ClusterNode:
    if not isinstance(raw, dict):
        raise TypeError(f"nodes[{index}] must be a mapping")
    allowed = {
        "node_id",
        "host",
        "ranks",
        "ssh_alias",
        "interface",
        "bind_host",
        "python",
        "workdir",
    }
    _reject_unknown(raw, allowed, f"nodes[{index}]")
    required = {"node_id", "host", "ranks", "ssh_alias", "interface"}
    missing = required - raw.keys()
    if missing:
        raise ValueError(
            f"nodes[{index}] is missing required fields: {', '.join(sorted(missing))}"
        )
    ranks = raw["ranks"]
    if not isinstance(ranks, list) or any(type(rank) is not int for rank in ranks):
        raise TypeError(f"nodes[{index}].ranks must be a list of integers")
    return ClusterNode(
        node_id=raw["node_id"],
        host=raw["host"],
        ranks=tuple(ranks),
        ssh_alias=raw["ssh_alias"],
        interface=raw["interface"],
        bind_host=raw.get("bind_host", "0.0.0.0"),
        python=raw.get("python", "python3"),
        workdir=raw.get("workdir", "."),
    )


def _integer_tuple(mapping: dict[str, Any], name: str) -> tuple[int, ...]:
    value = mapping.get(name, (1024, 65536, 1048576))
    if not isinstance(value, (list, tuple)) or any(
        type(item) is not int for item in value
    ):
        raise TypeError(f"benchmark.{name} must be a list of integers")
    return tuple(value)


def _reject_unknown(value: dict[str, Any], allowed: set[str], location: str) -> None:
    unknown = value.keys() - allowed
    if unknown:
        raise ValueError(f"unknown fields in {location}: {', '.join(sorted(unknown))}")
