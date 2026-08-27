"""Strict YAML configuration for statically known Wi-Fi peers."""

from __future__ import annotations

from dataclasses import dataclass, fields
from pathlib import Path
from typing import Any

import yaml

from .types import CommConfig, Peer


@dataclass(frozen=True, slots=True)
class RuntimeConfig:
    """Everything required to initialize one Comm runtime."""

    local: Peer
    peers: tuple[Peer, ...]
    bind_host: str
    comm: CommConfig


def load_runtime_config(path: str | Path) -> RuntimeConfig:
    """Load and validate a node configuration from YAML."""

    config_path = Path(path)
    with config_path.open("r", encoding="utf-8") as stream:
        document = yaml.safe_load(stream)
    if not isinstance(document, dict):
        raise TypeError("configuration root must be a mapping")
    _reject_unknown_keys(document, {"local", "peers", "comm"}, "configuration")

    local_fields = _require_mapping(document, "local")
    _reject_unknown_keys(
        local_fields,
        {"node_id", "host", "port", "bind_host"},
        "local",
    )
    local = _parse_peer(local_fields, "local")
    bind_host = local_fields.get("bind_host", local.host)
    if not isinstance(bind_host, str) or not bind_host:
        raise ValueError("local.bind_host must be a non-empty string")

    raw_peers = document.get("peers")
    if not isinstance(raw_peers, list):
        raise TypeError("peers must be a list")
    peers: list[Peer] = []
    seen = {local.node_id}
    for index, raw_peer in enumerate(raw_peers):
        if not isinstance(raw_peer, dict):
            raise TypeError(f"peers[{index}] must be a mapping")
        _reject_unknown_keys(raw_peer, {"node_id", "host", "port"}, f"peers[{index}]")
        peer = _parse_peer(raw_peer, f"peers[{index}]")
        if peer.node_id in seen:
            raise ValueError(f"duplicate node_id {peer.node_id!r}")
        seen.add(peer.node_id)
        peers.append(peer)

    raw_comm = document.get("comm", {})
    if not isinstance(raw_comm, dict):
        raise TypeError("comm must be a mapping")
    comm_fields = {field.name for field in fields(CommConfig)}
    _reject_unknown_keys(raw_comm, comm_fields, "comm")
    try:
        comm = CommConfig(**raw_comm)
    except TypeError as exc:
        raise ValueError(f"invalid comm configuration: {exc}") from exc
    return RuntimeConfig(local, tuple(peers), bind_host, comm)


def _parse_peer(raw: dict[str, Any], location: str) -> Peer:
    missing = {"node_id", "host", "port"} - raw.keys()
    if missing:
        names = ", ".join(sorted(missing))
        raise ValueError(f"{location} is missing required fields: {names}")
    try:
        return Peer(raw["node_id"], raw["host"], raw["port"])
    except (TypeError, ValueError) as exc:
        raise ValueError(f"invalid {location}: {exc}") from exc


def _require_mapping(document: dict[str, Any], key: str) -> dict[str, Any]:
    value = document.get(key)
    if not isinstance(value, dict):
        raise TypeError(f"{key} must be a mapping")
    return value


def _reject_unknown_keys(
    value: dict[str, Any], allowed: set[str], location: str
) -> None:
    unknown = value.keys() - allowed
    if unknown:
        names = ", ".join(sorted(unknown))
        raise ValueError(f"unknown fields in {location}: {names}")
