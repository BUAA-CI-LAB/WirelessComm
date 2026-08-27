"""Command-line entry point for testing Comm across Wi-Fi nodes."""

from __future__ import annotations

import argparse
import asyncio
import json
from pathlib import Path
from typing import Any

from .comm import Comm
from .config import RuntimeConfig, load_runtime_config
from .types import CommOptions


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True, help="node YAML configuration")
    subparsers = parser.add_subparsers(dest="command", required=True)

    send = subparsers.add_parser("send", help="send one object to a peer")
    send.add_argument("--dst", required=True, help="destination node_id")
    payload = send.add_mutually_exclusive_group(required=True)
    payload.add_argument("--json", help="JSON object or value")
    payload.add_argument("--text", help="UTF-8 string payload")
    payload.add_argument("--file", type=Path, help="file to send as bytes")
    payload.add_argument(
        "--tensor-shape",
        help="generate a contiguous Tensor, for example 3,224,224",
    )
    send.add_argument("--tensor-dtype", default="float32")
    send.add_argument("--metadata", help="JSON metadata mapping")
    send.add_argument("--tag", type=int, default=0)
    send.add_argument("--timeout", type=float)
    send.add_argument(
        "--fail-fast",
        action="store_true",
        help="raise immediately instead of waiting for queue capacity",
    )
    send.add_argument(
        "--expect-reply",
        action="store_true",
        help="wait for an echo reply with the same tag",
    )

    receive = subparsers.add_parser("recv", help="receive objects from one peer")
    _add_receive_arguments(receive)

    echo = subparsers.add_parser("echo", help="receive and echo objects to one peer")
    _add_receive_arguments(echo)
    return parser


def _add_receive_arguments(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--src", required=True, help="source node_id")
    parser.add_argument("--tag", type=int, default=0)
    parser.add_argument("--timeout", type=float)
    parser.add_argument(
        "--count",
        type=int,
        default=1,
        help="number of messages; 0 means run forever",
    )


async def run(args: argparse.Namespace) -> None:
    runtime_config = load_runtime_config(args.config)
    async with await _create_comm(runtime_config) as comm:
        if args.command == "send":
            await _run_send(comm, args)
        elif args.command == "recv":
            await _run_receive(comm, args, echo=False)
        else:
            await _run_receive(comm, args, echo=True)


async def _create_comm(config: RuntimeConfig) -> Comm:
    return await Comm.create(
        local=config.local,
        peers=config.peers,
        bind_host=config.bind_host,
        config=config.comm,
    )


async def _run_send(comm: Comm, args: argparse.Namespace) -> None:
    payload = _parse_payload(args)
    metadata = _parse_metadata(args.metadata)
    options = CommOptions(
        tag=args.tag,
        timeout=args.timeout,
        wait_for_capacity=not args.fail_fast,
    )
    result = await comm.send(
        payload,
        comm.peer(args.dst),
        piggypayload=metadata,
        options=options,
    )
    print(f"sent message_id={result.message_id} wire_bytes={result.wire_bytes}")
    if args.expect_reply:
        reply, reply_metadata = await comm.recv(comm.peer(args.dst), options)
        _print_received(reply, reply_metadata)


async def _run_receive(
    comm: Comm,
    args: argparse.Namespace,
    *,
    echo: bool,
) -> None:
    if args.count < 0:
        raise ValueError("count must be non-negative")
    peer = comm.peer(args.src)
    options = CommOptions(tag=args.tag, timeout=args.timeout)
    received = 0
    while args.count == 0 or received < args.count:
        object, metadata = await comm.recv(peer, options)
        _print_received(object, metadata)
        if echo:
            await comm.send(
                object,
                peer,
                piggypayload=metadata,
                options=options,
            )
        received += 1


def _parse_payload(args: argparse.Namespace) -> Any:
    if args.json is not None:
        return json.loads(args.json)
    if args.text is not None:
        return args.text
    if args.file is not None:
        return args.file.read_bytes()
    return _make_tensor(args.tensor_shape, args.tensor_dtype)


def _parse_metadata(raw: str | None) -> dict[str, Any] | None:
    if raw is None:
        return None
    value = json.loads(raw)
    if not isinstance(value, dict):
        raise TypeError("metadata JSON must be an object")
    return value


def _make_tensor(shape_text: str, dtype_name: str) -> Any:
    try:
        import torch
    except ImportError as exc:
        raise RuntimeError("Tensor CLI payloads require wireless-comm[tensor]") from exc
    try:
        shape = tuple(int(dimension) for dimension in shape_text.split(","))
    except ValueError as exc:
        raise ValueError("tensor shape must be comma-separated integers") from exc
    if not shape or any(dimension < 0 for dimension in shape):
        raise ValueError("tensor dimensions must be non-negative")
    dtype = getattr(torch, dtype_name, None)
    if dtype is None or not isinstance(dtype, torch.dtype):
        raise ValueError(f"unknown Tensor dtype {dtype_name!r}")
    return torch.arange(_product(shape), dtype=torch.int64).to(dtype).reshape(shape)


def _product(shape: tuple[int, ...]) -> int:
    result = 1
    for dimension in shape:
        result *= dimension
    return result


def _print_received(object: Any, metadata: dict[str, Any] | None) -> None:
    print(f"received object={_summarize(object)}")
    print(f"piggypayload={metadata!r}")


def _summarize(value: Any) -> str:
    try:
        import torch
    except ImportError:
        torch = None
    if torch is not None and type(value) is torch.Tensor:
        return (
            f"Tensor(shape={tuple(value.shape)}, dtype={value.dtype}, "
            f"contiguous={value.is_contiguous()})"
        )
    if isinstance(value, bytes):
        return f"bytes(length={len(value)})"
    return repr(value)


def main() -> None:
    args = build_parser().parse_args()
    try:
        asyncio.run(run(args))
    except KeyboardInterrupt:
        pass


if __name__ == "__main__":
    main()
