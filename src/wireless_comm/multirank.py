"""Multi-rank collective benchmark distributed across two physical hosts."""

from __future__ import annotations

import argparse
import asyncio
import json
import time
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from .benchmark import summarize_samples
from .comm import Comm
from .diagnostics import (
    LinkMonitor,
    collect_environment,
    counter_delta,
    kernel_counter_delta,
    resolve_interface,
)
from .types import CommConfig, CommOptions, Peer

CONTROL_TAG = 0xFFF0_0000
_COLLECTIVE_TAGS = {
    "barrier": 0x1000_0000,
    "broadcast": 0x2000_0000,
    "allgather": 0x3000_0000,
    "ring_allreduce": 0x4000_0000,
    "ring_exchange": 0x5000_0000,
    "all_to_all_exchange": 0x6000_0000,
}


@dataclass(frozen=True, slots=True)
class MultiRankSettings:
    payload_sizes: tuple[int, ...] = (1024, 65536, 1048576)
    rounds: int = 20
    warmup_rounds: int = 3
    collectives: tuple[str, ...] = (
        "barrier",
        "broadcast",
        "allgather",
        "ring_allreduce",
    )
    timeout: float = 120.0
    broadcast_root: int = 0
    ring_exchange_concurrency: int | None = None
    allreduce_block_bytes: int | None = None

    def __post_init__(self) -> None:
        allowed = set(_COLLECTIVE_TAGS)
        if not self.payload_sizes or any(size <= 0 for size in self.payload_sizes):
            raise ValueError("payload sizes must be positive")
        if self.rounds <= 0 or self.warmup_rounds < 0 or self.timeout <= 0:
            raise ValueError("rounds and timeout must be positive")
        if not self.collectives or set(self.collectives) - allowed:
            raise ValueError("unknown collective")
        if type(self.broadcast_root) is not int or self.broadcast_root < 0:
            raise ValueError("broadcast root must be a non-negative rank")
        if (
            self.ring_exchange_concurrency is not None
            and self.ring_exchange_concurrency <= 0
        ):
            raise ValueError("ring exchange concurrency must be positive")
        if self.allreduce_block_bytes is not None and self.allreduce_block_bytes <= 0:
            raise ValueError("allreduce block size must be positive")


@dataclass(frozen=True, slots=True)
class Topology:
    local_host: str
    remote_host: str
    base_port: int
    ranks_per_host: int
    initiator_side: bool
    bind_host: str = "0.0.0.0"

    @property
    def world_size(self) -> int:
        return self.ranks_per_host * 2

    @property
    def local_ranks(self) -> tuple[int, ...]:
        parity = 0 if self.initiator_side else 1
        return tuple(range(parity, self.world_size, 2))

    def peers(self) -> tuple[Peer, ...]:
        return tuple(
            Peer(
                f"rank-{rank}",
                self.local_host if rank in self.local_ranks else self.remote_host,
                self.base_port + rank,
            )
            for rank in range(self.world_size)
        )

    def validate(self) -> None:
        if self.ranks_per_host <= 0:
            raise ValueError("ranks per host must be positive")
        if self.base_port <= 0 or self.base_port + self.world_size - 1 > 65535:
            raise ValueError("rank port range is invalid")


async def run_side(
    topology: Topology,
    *,
    settings: MultiRankSettings | None,
    interface: str | None,
    sample_interval: float,
) -> dict[str, Any]:
    """Run the initiator side or wait for settings on the responder side."""

    topology.validate()
    directory = topology.peers()
    comms = await asyncio.gather(
        *(
            Comm.create(
                local=directory[rank],
                peers=tuple(
                    peer for peer in directory if peer.node_id != f"rank-{rank}"
                ),
                bind_host=topology.bind_host,
                config=CommConfig(),
            )
            for rank in topology.local_ranks
        )
    )
    comm_by_rank = dict(zip(topology.local_ranks, comms))
    selected_interface = interface or resolve_interface(topology.remote_host)
    before = await asyncio.to_thread(
        collect_environment, selected_interface, topology.remote_host
    )
    monitor = LinkMonitor(selected_interface, sample_interval)
    monitor.start()
    try:
        if topology.initiator_side:
            if settings is None:
                raise ValueError("initiator requires benchmark settings")
            await _wait_for_port(
                topology.remote_host,
                topology.base_port + 1,
                settings.timeout,
            )
            control = comm_by_rank[0]
            rank_one = directory[1]
            await control.send(
                {"op": "start", "settings": asdict(settings)},
                rank_one,
                options=CommOptions(tag=CONTROL_TAG, timeout=settings.timeout),
            )
            response, _ = await control.recv(
                rank_one, CommOptions(tag=CONTROL_TAG, timeout=settings.timeout)
            )
            if response != {"op": "ready"}:
                raise RuntimeError("remote ranks did not become ready")
        else:
            control = comm_by_rank[1]
            rank_zero = directory[0]
            request, _ = await control.recv(
                rank_zero, CommOptions(tag=CONTROL_TAG, timeout=3600)
            )
            if not isinstance(request, dict) or request.get("op") != "start":
                raise RuntimeError("invalid multi-rank start message")
            settings = _settings_from_mapping(request["settings"])
            await control.send(
                {"op": "ready"},
                rank_zero,
                options=CommOptions(tag=CONTROL_TAG, timeout=settings.timeout),
            )

        if settings is None:
            raise RuntimeError("multi-rank settings were not established")
        _validate_settings_for_world(settings, topology.world_size)
        started_at = datetime.now(timezone.utc).isoformat()
        rank_results = await asyncio.gather(
            *(
                _run_rank(
                    rank,
                    comm_by_rank[rank],
                    directory,
                    settings,
                    tuple(range(topology.world_size)),
                )
                for rank in topology.local_ranks
            )
        )

        after = await asyncio.to_thread(
            collect_environment, selected_interface, topology.remote_host
        )
        side_report = {
            "local_ranks": list(topology.local_ranks),
            "environment_before": before,
            "environment_after": after,
            "interface_counter_delta": counter_delta(
                before["interface"], after["interface"]
            ),
            "kernel_counter_delta": kernel_counter_delta(
                before["kernel_network_counters"],
                after["kernel_network_counters"],
            ),
            "link_samples": await monitor.stop(),
        }

        if topology.initiator_side:
            remote, _ = await comm_by_rank[0].recv(
                directory[1], CommOptions(tag=CONTROL_TAG, timeout=settings.timeout)
            )
            if not isinstance(remote, dict) or remote.get("op") != "results":
                raise RuntimeError("invalid remote multi-rank results")
            all_rank_results = [*rank_results, *remote["rank_results"]]
            remote_side = remote["side"]
        else:
            await comm_by_rank[1].send(
                {"op": "results", "rank_results": rank_results, "side": side_report},
                directory[0],
                options=CommOptions(tag=CONTROL_TAG, timeout=settings.timeout),
            )
            all_rank_results = rank_results
            remote_side = None
        report = {
            "schema_version": 1,
            "started_at": started_at,
            "finished_at": datetime.now(timezone.utc).isoformat(),
            "mode": "two_physical_stations_with_alternating_logical_ranks",
            "limitations": (
                "Models multi-connection contention and collective synchronization; "
                "does not model independent radios, hidden terminals, or AP airtime "
                "fairness among more than two stations."
            ),
            "world_size": topology.world_size,
            "settings": asdict(settings),
            "side": side_report,
            "rank_results": sorted(all_rank_results, key=lambda item: item["rank"]),
        }
        if topology.initiator_side:
            report["collective_summary"] = _summarize_collectives(all_rank_results)
            report["remote_side"] = remote_side
        return report
    finally:
        await monitor.stop()
        await asyncio.gather(*(comm.close() for comm in comms))


async def _run_rank(
    rank: int,
    comm: Comm,
    directory: tuple[Peer, ...],
    settings: MultiRankSettings,
    ring: tuple[int, ...],
) -> dict[str, Any]:
    options = CommOptions(timeout=settings.timeout)
    results: dict[str, dict[str, list[float]]] = {}
    send_service_results: dict[str, dict[str, list[list[float]]]] = {}
    for collective in settings.collectives:
        sizes = (0,) if collective == "barrier" else settings.payload_sizes
        results[collective] = {}
        for size in sizes:
            samples = []
            total_rounds = settings.warmup_rounds + settings.rounds
            for iteration in range(total_rounds):
                started = time.perf_counter_ns()
                flow_service_ms = await _run_collective(
                    collective,
                    rank,
                    comm,
                    directory,
                    size,
                    iteration,
                    options,
                    ring,
                    settings,
                )
                elapsed_ms = (time.perf_counter_ns() - started) / 1_000_000
                if iteration >= settings.warmup_rounds:
                    samples.append(elapsed_ms)
                    if flow_service_ms is not None:
                        send_service_results.setdefault(collective, {}).setdefault(
                            str(size), []
                        ).append(flow_service_ms)
            results[collective][str(size)] = samples
    rank_result = {"rank": rank, "samples_ms": results}
    if send_service_results:
        rank_result["send_service_ms"] = send_service_results
    return rank_result


async def _run_collective(
    collective: str,
    rank: int,
    comm: Comm,
    directory: tuple[Peer, ...],
    size: int,
    iteration: int,
    options: CommOptions,
    ring: tuple[int, ...],
    settings: MultiRankSettings,
) -> list[float] | None:
    if collective == "barrier":
        await _barrier(rank, comm, directory, iteration, options)
    elif collective == "broadcast":
        await _broadcast(
            rank,
            comm,
            directory,
            bytes(size),
            iteration,
            options,
            settings.broadcast_root,
        )
    elif collective == "allgather":
        await _allgather(rank, comm, directory, bytes(size), iteration, options, ring)
    elif collective == "ring_allreduce":
        await _ring_allreduce(
            rank,
            comm,
            directory,
            size,
            iteration,
            options,
            ring,
            settings.allreduce_block_bytes,
        )
    else:
        if collective == "ring_exchange":
            service_ms = await _ring_exchange(
                rank,
                comm,
                directory,
                bytes(size),
                iteration,
                options,
                ring,
                settings.ring_exchange_concurrency,
            )
            return [service_ms]
        return await _all_to_all_exchange(
            rank,
            comm,
            directory,
            bytes(size),
            iteration,
            options,
        )
    return None


async def _barrier(
    rank: int,
    comm: Comm,
    directory: tuple[Peer, ...],
    iteration: int,
    options: CommOptions,
) -> None:
    operation_options = _tagged(options, "barrier", iteration)
    await _barrier_with_options(rank, comm, directory, operation_options)


async def _barrier_with_options(
    rank: int,
    comm: Comm,
    directory: tuple[Peer, ...],
    operation_options: CommOptions,
) -> None:
    if rank == 0:
        await asyncio.gather(
            *(comm.recv(peer, operation_options) for peer in directory[1:])
        )
        await asyncio.gather(
            *(
                comm.send(b"release", peer, options=operation_options)
                for peer in directory[1:]
            )
        )
    else:
        await comm.send(b"arrive", directory[0], options=operation_options)
        await comm.recv(directory[0], operation_options)


async def _broadcast(
    rank: int,
    comm: Comm,
    directory: tuple[Peer, ...],
    payload: bytes,
    iteration: int,
    options: CommOptions,
    root: int,
) -> None:
    operation_options = _tagged(options, "broadcast", iteration)
    if rank == root:
        await asyncio.gather(
            *(
                comm.send(payload, peer, options=operation_options)
                for peer_rank, peer in enumerate(directory)
                if peer_rank != root
            )
        )
    else:
        received, _ = await comm.recv(directory[root], operation_options)
        if not isinstance(received, bytes) or len(received) != len(payload):
            raise RuntimeError("broadcast payload is invalid")


async def _ring_exchange(
    rank: int,
    comm: Comm,
    directory: tuple[Peer, ...],
    payload: bytes,
    iteration: int,
    options: CommOptions,
    ring: tuple[int, ...],
    concurrency: int | None,
) -> float:
    world_size = len(directory)
    active_count = concurrency if concurrency is not None else world_size
    ring_position = ring.index(rank)
    previous_rank = ring[(ring_position - 1) % world_size]
    following_rank = ring[(ring_position + 1) % world_size]
    send_service_ms: float | None = None

    for phase, start in enumerate(range(0, world_size, active_count)):
        active_senders = set(ring[start : start + active_count])
        operation_options = _tagged(options, "ring_exchange", iteration, phase * 2)
        operations = []
        send_result_index = None
        if rank in active_senders:
            send_result_index = len(operations)
            operations.append(
                _measure_send_service(
                    comm,
                    payload,
                    directory[following_rank],
                    operation_options,
                )
            )
        if previous_rank in active_senders:
            operations.append(comm.recv(directory[previous_rank], operation_options))
        results = await asyncio.gather(*operations)
        if send_result_index is not None:
            send_service_ms = results[send_result_index]
        if previous_rank in active_senders:
            received, metadata = results[-1]
            if (
                not isinstance(received, bytes)
                or len(received) != len(payload)
                or metadata is not None
            ):
                raise RuntimeError("ring exchange received an invalid payload")
        await _barrier_with_options(
            rank,
            comm,
            directory,
            _tagged(options, "ring_exchange", iteration, phase * 2 + 1),
        )
    if send_service_ms is None:
        raise RuntimeError("ring exchange did not schedule the local sender")
    return send_service_ms


async def _measure_send_service(
    comm: Comm,
    payload: bytes,
    peer: Peer,
    options: CommOptions,
) -> float:
    started = time.perf_counter_ns()
    await comm.send(payload, peer, options=options)
    return (time.perf_counter_ns() - started) / 1_000_000


async def _all_to_all_exchange(
    rank: int,
    comm: Comm,
    directory: tuple[Peer, ...],
    payload: bytes,
    iteration: int,
    options: CommOptions,
) -> list[float]:
    operation_options = _tagged(options, "all_to_all_exchange", iteration)
    remote_ranks = [peer_rank for peer_rank in range(len(directory)) if peer_rank != rank]
    send_count = len(remote_ranks)
    results = await asyncio.gather(
        *(
            _measure_send_service(
                comm,
                payload,
                directory[peer_rank],
                operation_options,
            )
            for peer_rank in remote_ranks
        ),
        *(
            comm.recv(directory[peer_rank], operation_options)
            for peer_rank in remote_ranks
        ),
    )
    for received, metadata in results[send_count:]:
        if (
            not isinstance(received, bytes)
            or len(received) != len(payload)
            or metadata is not None
        ):
            raise RuntimeError("all-to-all exchange received an invalid payload")
    return results[:send_count]


async def _allgather(
    rank: int,
    comm: Comm,
    directory: tuple[Peer, ...],
    payload: Any,
    iteration: int,
    options: CommOptions,
    ring: tuple[int, ...],
) -> list[Any]:
    world_size = len(directory)
    ring_position = ring.index(rank)
    previous = directory[ring[(ring_position - 1) % world_size]]
    following = directory[ring[(ring_position + 1) % world_size]]
    gathered = {rank: payload}
    current = {"origin": rank, "payload": payload}
    for step in range(world_size - 1):
        operation_options = _tagged(options, "allgather", iteration, step)
        _, received = await asyncio.gather(
            comm.send(current, following, options=operation_options),
            comm.recv(previous, operation_options),
        )
        item, metadata = received
        if not isinstance(item, dict) or metadata is not None:
            raise RuntimeError("allgather received an invalid item")
        origin = item["origin"]
        gathered[origin] = item["payload"]
        current = item
    return [gathered[origin] for origin in ring]


async def _ring_allreduce(
    rank: int,
    comm: Comm,
    directory: tuple[Peer, ...],
    size: int,
    iteration: int,
    options: CommOptions,
    ring: tuple[int, ...],
    block_bytes: int | None,
) -> None:
    world_size = len(directory)
    shard_size = size // world_size
    selected_block_bytes = min(block_bytes or shard_size, shard_size)
    expected_value = 0
    for source_rank in range(world_size):
        expected_value ^= source_rank
    tag_step = 0
    for offset in range(0, shard_size, selected_block_bytes):
        current_block_bytes = min(selected_block_bytes, shard_size - offset)
        chunks = [bytes([rank]) * current_block_bytes for _ in range(world_size)]
        tag_step = await _allreduce_block(
            rank,
            comm,
            directory,
            chunks,
            iteration,
            options,
            ring,
            tag_step,
        )
        expected = bytes([expected_value]) * current_block_bytes
        if any(chunk != expected for chunk in chunks):
            raise RuntimeError("ring allreduce produced an incorrect result")


async def _allreduce_block(
    rank: int,
    comm: Comm,
    directory: tuple[Peer, ...],
    chunks: list[bytes],
    iteration: int,
    options: CommOptions,
    ring: tuple[int, ...],
    tag_step: int,
) -> int:
    world_size = len(directory)
    ring_position = ring.index(rank)
    previous = directory[ring[(ring_position - 1) % world_size]]
    following = directory[ring[(ring_position + 1) % world_size]]
    for step in range(world_size - 1):
        send_index = (ring_position - step) % world_size
        receive_index = (ring_position - step - 1) % world_size
        operation_options = _tagged(
            options, "ring_allreduce", iteration, tag_step
        )
        _, received = await asyncio.gather(
            comm.send(chunks[send_index], following, options=operation_options),
            comm.recv(previous, operation_options),
        )
        received_chunk, metadata = received
        if not isinstance(received_chunk, bytes) or metadata is not None:
            raise RuntimeError("allreduce received an invalid chunk")
        chunks[receive_index] = _xor_bytes(chunks[receive_index], received_chunk)
        tag_step += 1

    for step in range(world_size - 1):
        send_index = (ring_position - step + 1) % world_size
        receive_index = (ring_position - step) % world_size
        operation_options = _tagged(
            options, "ring_allreduce", iteration, tag_step
        )
        _, received = await asyncio.gather(
            comm.send(chunks[send_index], following, options=operation_options),
            comm.recv(previous, operation_options),
        )
        received_chunk, metadata = received
        if not isinstance(received_chunk, bytes) or metadata is not None:
            raise RuntimeError("allreduce received an invalid chunk")
        chunks[receive_index] = received_chunk
        tag_step += 1
    return tag_step


def _xor_bytes(left: bytes, right: bytes) -> bytes:
    if len(left) != len(right):
        raise RuntimeError("allreduce chunks have different sizes")
    return (int.from_bytes(left, "little") ^ int.from_bytes(right, "little")).to_bytes(
        len(left), "little"
    )


def _tagged(
    options: CommOptions,
    collective: str,
    iteration: int,
    step: int = 0,
) -> CommOptions:
    tag = _COLLECTIVE_TAGS[collective] + iteration * 256 + step
    return CommOptions(tag=tag, timeout=options.timeout)


def _summarize_collectives(rank_results: list[dict[str, Any]]) -> dict[str, Any]:
    summary: dict[str, Any] = {}
    example = rank_results[0]["samples_ms"]
    for collective, sizes in example.items():
        summary[collective] = {}
        for size, first_samples in sizes.items():
            per_rank = {
                str(result["rank"]): summarize_samples(
                    result["samples_ms"][collective][size]
                )
                for result in rank_results
            }
            completion = [
                max(
                    result["samples_ms"][collective][size][index]
                    for result in rank_results
                )
                for index in range(len(first_samples))
            ]
            summary[collective][size] = {
                "collective_completion_ms": summarize_samples(completion),
                "completion_samples_ms": completion,
                "per_rank_ms": per_rank,
            }
            if collective in {"ring_exchange", "all_to_all_exchange"}:
                service_samples = {
                    str(result["rank"]): result["send_service_ms"][collective][
                        size
                    ]
                    for result in rank_results
                }
                fairness = []
                payload_bytes = int(size)
                for index in range(len(first_samples)):
                    flow_mbps = [
                        payload_bytes * 8 / (duration * 1000)
                        for rank_samples in service_samples.values()
                        for duration in rank_samples[index]
                    ]
                    fairness.append(_jain_fairness(flow_mbps))
                flow_count = sum(
                    len(rank_samples[0]) for rank_samples in service_samples.values()
                )
                effective_round_mbps = [
                    payload_bytes * flow_count * 8 / (sample * 1000)
                    for sample in completion
                ]
                summary[collective][size].update(
                    {
                        "send_service_ms": {
                            rank: summarize_samples(
                                [
                                    duration
                                    for iteration_samples in samples
                                    for duration in iteration_samples
                                ]
                            )
                            for rank, samples in service_samples.items()
                        },
                        "effective_round_mbps": summarize_samples(
                            effective_round_mbps
                        ),
                        "jain_fairness": summarize_samples(fairness),
                    }
                )
    return summary


def _jain_fairness(values: list[float]) -> float:
    total = sum(values)
    return total * total / (len(values) * sum(value * value for value in values))


def _settings_from_mapping(value: Any) -> MultiRankSettings:
    if not isinstance(value, dict):
        raise TypeError("multi-rank settings must be a mapping")
    return MultiRankSettings(
        payload_sizes=tuple(value["payload_sizes"]),
        rounds=value["rounds"],
        warmup_rounds=value["warmup_rounds"],
        collectives=tuple(value["collectives"]),
        timeout=value["timeout"],
        broadcast_root=value.get("broadcast_root", 0),
        ring_exchange_concurrency=value.get("ring_exchange_concurrency"),
        allreduce_block_bytes=value.get("allreduce_block_bytes"),
    )


def _validate_settings_for_world(
    settings: MultiRankSettings, world_size: int
) -> None:
    if settings.broadcast_root >= world_size:
        raise ValueError("broadcast root must be smaller than world size")
    if (
        settings.ring_exchange_concurrency is not None
        and settings.ring_exchange_concurrency > world_size
    ):
        raise ValueError("ring exchange concurrency cannot exceed world size")
    if "ring_allreduce" not in settings.collectives:
        return
    if any(size % world_size for size in settings.payload_sizes):
        raise ValueError(
            f"ring-allreduce payload sizes must be multiples of {world_size} bytes"
        )
    if settings.allreduce_block_bytes is None:
        return
    for size in settings.payload_sizes:
        shard_size = size // world_size
        blocks = (shard_size + settings.allreduce_block_bytes - 1) // (
            settings.allreduce_block_bytes
        )
        tag_steps = blocks * 2 * (world_size - 1)
        if tag_steps > 256:
            raise ValueError(
                "allreduce block size creates more than 256 protocol steps"
            )


async def _wait_for_port(host: str, port: int, timeout: float) -> None:
    deadline = time.monotonic() + timeout
    while True:
        try:
            _, writer = await asyncio.open_connection(host, port)
        except OSError:
            if time.monotonic() >= deadline:
                raise TimeoutError(f"timed out waiting for {host}:{port}")
            await asyncio.sleep(0.1)
            continue
        writer.close()
        await writer.wait_closed()
        return


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--local-host", required=True)
    parser.add_argument("--remote-host", required=True)
    parser.add_argument("--base-port", type=int, default=9400)
    parser.add_argument("--ranks-per-host", type=int, default=2)
    parser.add_argument("--bind-host", default="0.0.0.0")
    parser.add_argument("--interface")
    parser.add_argument("--sample-interval", type=float, default=1.0)
    parser.add_argument("--output", type=Path)
    subparsers = parser.add_subparsers(dest="command", required=True)
    subparsers.add_parser("responder")
    initiator = subparsers.add_parser("initiator")
    initiator.add_argument("--payload-sizes", default="1024,65536,1048576")
    initiator.add_argument("--rounds", type=int, default=20)
    initiator.add_argument("--warmup-rounds", type=int, default=3)
    initiator.add_argument(
        "--collectives", default="barrier,broadcast,allgather,ring_allreduce"
    )
    initiator.add_argument("--timeout", type=float, default=120.0)
    initiator.add_argument("--broadcast-root", type=int, default=0)
    initiator.add_argument("--ring-exchange-concurrency", type=int)
    initiator.add_argument("--allreduce-block-bytes", type=int)
    return parser


async def _main(args: argparse.Namespace) -> dict[str, Any]:
    initiator_side = args.command == "initiator"
    topology = Topology(
        args.local_host,
        args.remote_host,
        args.base_port,
        args.ranks_per_host,
        initiator_side,
        args.bind_host,
    )
    settings = None
    if initiator_side:
        settings = MultiRankSettings(
            payload_sizes=tuple(int(size) for size in args.payload_sizes.split(",")),
            rounds=args.rounds,
            warmup_rounds=args.warmup_rounds,
            collectives=tuple(args.collectives.split(",")),
            timeout=args.timeout,
            broadcast_root=args.broadcast_root,
            ring_exchange_concurrency=args.ring_exchange_concurrency,
            allreduce_block_bytes=args.allreduce_block_bytes,
        )
    return await run_side(
        topology,
        settings=settings,
        interface=args.interface,
        sample_interval=args.sample_interval,
    )


def main() -> None:
    args = build_parser().parse_args()
    report = asyncio.run(_main(args))
    serialized = json.dumps(report, indent=2, sort_keys=True)
    if args.output is None:
        print(serialized)
    else:
        args.output.write_text(serialized + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
