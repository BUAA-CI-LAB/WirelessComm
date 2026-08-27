"""End-to-end Wi-Fi benchmark for the WirelessComm P2P transport."""

from __future__ import annotations

import argparse
import asyncio
import itertools
import json
import math
import random
import resource
import statistics
import sys
import time
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from .comm import Comm
from .config import RuntimeConfig, load_runtime_config
from .diagnostics import (
    LinkMonitor,
    collect_environment,
    collect_tcp_sockets,
    counter_delta,
    kernel_counter_delta,
    resolve_interface,
    run_ping,
)
from .load import LoadServer, load_result_dict, run_load
from .types import CommOptions, Peer

SCHEMA_VERSION = 4
PROTOCOL_VERSION = 1
CONTROL_TAG = 0xFFFF_FF00
DATA_TAG = 0xFFFF_FF01


@dataclass(frozen=True, slots=True)
class BenchmarkSettings:
    latency_sizes: tuple[int, ...] = (0, 64, 1024, 65536)
    latency_rounds: int = 100
    warmup_rounds: int = 10
    throughput_sizes: tuple[int, ...] = (1024, 65536, 1048576, 8388608)
    throughput_target_bytes: int = 16 * 1024 * 1024
    min_throughput_messages: int = 4
    max_throughput_messages: int = 4096
    throughput_trials: int = 5
    directions: tuple[str, ...] = ("upload", "download", "bidirectional")
    tensor_sizes: tuple[int, ...] = ()
    tensor_rounds: int = 10
    ping_count: int = 10
    operation_timeout: float = 120.0
    sample_interval: float = 1.0
    load_fractions: tuple[float, ...] = ()
    load_directions: tuple[str, ...] = ("upload", "download", "bidirectional")
    load_concurrencies: tuple[int, ...] = (1, 4)
    load_duration: float = 10.0
    load_probe_interval: float = 0.05
    load_probe_size: int = 64
    random_seed: int = 0

    def __post_init__(self) -> None:
        if not self.latency_sizes or any(size < 0 for size in self.latency_sizes):
            raise ValueError("latency sizes must be non-negative")
        if not self.throughput_sizes or any(
            size <= 0 for size in self.throughput_sizes
        ):
            raise ValueError("throughput sizes must be positive")
        for name in (
            "latency_rounds",
            "throughput_target_bytes",
            "min_throughput_messages",
            "max_throughput_messages",
            "throughput_trials",
            "ping_count",
            "tensor_rounds",
        ):
            if getattr(self, name) <= 0:
                raise ValueError(f"{name} must be positive")
        if self.warmup_rounds < 0:
            raise ValueError("warmup_rounds must be non-negative")
        if self.min_throughput_messages > self.max_throughput_messages:
            raise ValueError("minimum message count cannot exceed maximum")
        if any(size <= 0 or size % 4 for size in self.tensor_sizes):
            raise ValueError("Tensor sizes must be positive multiples of 4 bytes")
        allowed = {"upload", "download", "bidirectional"}
        if not self.directions or set(self.directions) - allowed:
            raise ValueError("unknown benchmark direction")
        if self.operation_timeout <= 0 or self.sample_interval <= 0:
            raise ValueError("timeouts and sample interval must be positive")
        if any(fraction <= 0 for fraction in self.load_fractions):
            raise ValueError("load fractions must be positive")
        if set(self.load_directions) - allowed:
            raise ValueError("unknown load direction")
        if self.load_fractions and not set(self.load_directions) <= set(
            self.directions
        ):
            raise ValueError("load directions require matching throughput directions")
        if not self.load_concurrencies or any(
            concurrency <= 0 for concurrency in self.load_concurrencies
        ):
            raise ValueError("load concurrencies must be positive")
        if self.load_duration <= 0 or self.load_probe_interval <= 0:
            raise ValueError("load duration and probe interval must be positive")
        if self.load_probe_size < 0:
            raise ValueError("load probe size must be non-negative")

    def message_count(self, size: int) -> int:
        target = math.ceil(self.throughput_target_bytes / size)
        return min(
            self.max_throughput_messages,
            max(self.min_throughput_messages, target),
        )


async def run_responder(
    config: RuntimeConfig,
    peer: Peer,
    *,
    interface: str | None,
    operation_timeout: float,
    sample_interval: float,
) -> dict[str, Any]:
    """Serve one benchmark session and return the responder-side report."""

    selected_interface = interface or resolve_interface(peer.host)
    before = await asyncio.to_thread(collect_environment, selected_interface, peer.host)
    usage_before = _process_usage()
    usage_started = time.monotonic()
    monitor = LinkMonitor(selected_interface, sample_interval)
    monitor.start()
    load_server: LoadServer | None = None
    report: dict[str, Any] | None = None
    options = CommOptions(tag=CONTROL_TAG, timeout=operation_timeout)
    data_options = CommOptions(tag=DATA_TAG, timeout=operation_timeout)

    async with await _create_comm(config) as comm:
        while True:
            request = await _recv_mapping(comm, peer, options)
            operation = request.get("op")
            if operation == "hello":
                if request.get("protocol_version") != PROTOCOL_VERSION:
                    raise RuntimeError("benchmark protocol version mismatch")
                if request.get("latency_under_load"):
                    if config.local.port == 65535:
                        raise ValueError("load server requires local.port below 65535")
                    load_server = LoadServer(config.bind_host, config.local.port + 1)
                    await load_server.start()
                await _send_control(
                    comm,
                    peer,
                    {"op": "hello", "protocol_version": PROTOCOL_VERSION},
                    options,
                )
            elif operation == "diagnostics":
                ping_count = _positive_int(request, "ping_count")
                reverse_ping = await asyncio.to_thread(
                    run_ping, peer.host, selected_interface, ping_count
                )
                await _send_control(
                    comm,
                    peer,
                    {
                        "op": "diagnostics",
                        "environment_before": before,
                        "ping_to_initiator": reverse_ping,
                    },
                    options,
                )
            elif operation == "ping":
                payload = request.get("payload")
                if not isinstance(payload, bytes):
                    raise TypeError("ping payload must be bytes")
                await _send_control(
                    comm,
                    peer,
                    {
                        "op": "pong",
                        "sequence": request.get("sequence"),
                        "payload": payload,
                    },
                    options,
                )
            elif operation == "tensor_ping":
                tensor = _validate_tensor(request.get("payload"))
                expected_bytes = _positive_int(request, "size")
                if tensor.numel() * tensor.element_size() != expected_bytes:
                    raise RuntimeError("Tensor payload has an unexpected size")
                await _send_control(
                    comm,
                    peer,
                    {
                        "op": "tensor_pong",
                        "sequence": request.get("sequence"),
                        "payload": tensor,
                    },
                    options,
                )
            elif operation == "tensor_structures":
                payload = request.get("payload")
                _validate_tensor_structures(payload)
                await _send_control(
                    comm,
                    peer,
                    {"op": "tensor_structures", "payload": payload},
                    options,
                )
            elif operation == "upload":
                size = _positive_int(request, "size")
                count = _positive_int(request, "count")
                await _send_control(comm, peer, {"op": "ready"}, options)
                started = time.monotonic()
                await _receive_payloads(comm, peer, data_options, size, count)
                elapsed = time.monotonic() - started
                await _send_control(
                    comm,
                    peer,
                    _completion("upload", size, count, elapsed),
                    options,
                )
            elif operation == "download":
                size = _positive_int(request, "size")
                count = _positive_int(request, "count")
                await _send_control(comm, peer, {"op": "ready"}, options)
                payload = bytes(size)
                started = time.monotonic()
                await _send_payloads(comm, peer, data_options, payload, count)
                elapsed = time.monotonic() - started
                await _send_control(
                    comm,
                    peer,
                    _completion("download", size, count, elapsed),
                    options,
                )
            elif operation == "bidirectional":
                size = _positive_int(request, "size")
                count = _positive_int(request, "count")
                await _send_control(comm, peer, {"op": "ready"}, options)
                payload = bytes(size)
                started = time.monotonic()
                await asyncio.gather(
                    _send_payloads(comm, peer, data_options, payload, count),
                    _receive_payloads(comm, peer, data_options, size, count),
                )
                elapsed = time.monotonic() - started
                await _send_control(
                    comm,
                    peer,
                    _completion("bidirectional", size, count, elapsed),
                    options,
                )
            elif operation == "stop":
                after = await asyncio.to_thread(
                    collect_environment, selected_interface, peer.host
                )
                usage_duration = time.monotonic() - usage_started
                usage_delta = _usage_delta(usage_before, _process_usage())
                report = {
                    "node_id": config.local.node_id,
                    "interface": selected_interface,
                    "environment_before": before,
                    "environment_after": after,
                    "interface_counter_delta": counter_delta(
                        before["interface"], after["interface"]
                    ),
                    "kernel_counter_delta": kernel_counter_delta(
                        before["kernel_network_counters"],
                        after["kernel_network_counters"],
                    ),
                    "process_observation_duration_s": usage_duration,
                    "process_usage_delta": usage_delta,
                    "process_usage_rates_per_s": _usage_rates(
                        usage_delta, usage_duration
                    ),
                    "link_samples": await monitor.stop(),
                }
                await _send_control(
                    comm,
                    peer,
                    {"op": "stopped", "report": report},
                    options,
                )
                break
            else:
                raise ValueError(f"unknown benchmark operation {operation!r}")

    if load_server is not None:
        await load_server.close()
    if report is None:
        raise RuntimeError("benchmark responder stopped without a report")
    return report


async def run_initiator(
    config: RuntimeConfig,
    peer: Peer,
    settings: BenchmarkSettings,
    *,
    interface: str | None,
) -> dict[str, Any]:
    """Run the benchmark suite and combine both nodes' observations."""

    selected_interface = interface or resolve_interface(peer.host)
    before = await asyncio.to_thread(collect_environment, selected_interface, peer.host)
    ping = await asyncio.to_thread(
        run_ping, peer.host, selected_interface, settings.ping_count
    )
    usage_before = _process_usage()
    usage_started = time.monotonic()
    monitor = LinkMonitor(selected_interface, settings.sample_interval)
    monitor.start()
    options = CommOptions(tag=CONTROL_TAG, timeout=settings.operation_timeout)
    data_options = CommOptions(tag=DATA_TAG, timeout=settings.operation_timeout)

    async with await _create_comm(config) as comm:
        started_at = datetime.now(timezone.utc).isoformat()
        connection_started = time.monotonic()
        await _send_control(
            comm,
            peer,
            {
                "op": "hello",
                "protocol_version": PROTOCOL_VERSION,
                "ping_count": settings.ping_count,
                "latency_under_load": bool(settings.load_fractions),
            },
            options,
        )
        hello = await _recv_mapping(comm, peer, options)
        _expect_operation(hello, "hello")
        connection_setup_ms = (time.monotonic() - connection_started) * 1000

        await _send_control(
            comm,
            peer,
            {"op": "diagnostics", "ping_count": settings.ping_count},
            options,
        )
        diagnostics = await _recv_mapping(comm, peer, options)
        _expect_operation(diagnostics, "diagnostics")

        for sequence in range(settings.warmup_rounds):
            await _ping(comm, peer, options, sequence, b"")

        latency = []
        sequence = settings.warmup_rounds
        for size in settings.latency_sizes:
            payload = bytes(size)
            samples = []
            for _ in range(settings.latency_rounds):
                started = time.perf_counter_ns()
                await _ping(comm, peer, options, sequence, payload)
                samples.append((time.perf_counter_ns() - started) / 1_000_000)
                sequence += 1
            latency.append(
                {
                    "payload_bytes": size,
                    "rounds": settings.latency_rounds,
                    "summary_ms": summarize_samples(samples),
                    "samples_ms": samples,
                }
            )

        throughput: dict[str, list[dict[str, Any]]] = {
            direction: [] for direction in settings.directions
        }
        for size in settings.throughput_sizes:
            count = settings.message_count(size)
            if "upload" in throughput:
                throughput["upload"].append(
                    summarize_throughput_trials(
                        [
                            await _measure_upload(
                                comm, peer, options, data_options, size, count
                            )
                            for _ in range(settings.throughput_trials)
                        ]
                    )
                )
            if "download" in throughput:
                throughput["download"].append(
                    summarize_throughput_trials(
                        [
                            await _measure_download(
                                comm, peer, options, data_options, size, count
                            )
                            for _ in range(settings.throughput_trials)
                        ]
                    )
                )
            if "bidirectional" in throughput:
                throughput["bidirectional"].append(
                    summarize_throughput_trials(
                        [
                            await _measure_bidirectional(
                                comm, peer, options, data_options, size, count
                            )
                            for _ in range(settings.throughput_trials)
                        ]
                    )
                )

        tensor = None
        if settings.tensor_sizes:
            tensor = await _measure_tensors(comm, peer, options, settings)

        latency_under_load = []
        if settings.load_fractions:
            latency_under_load = await _measure_latency_under_load(
                comm, peer, options, settings, throughput
            )

        tcp_socket_before_stop = await asyncio.to_thread(collect_tcp_sockets, peer.host)
        await _send_control(comm, peer, {"op": "stop"}, options)
        stopped = await _recv_mapping(comm, peer, options)
        _expect_operation(stopped, "stopped")

    after = await asyncio.to_thread(collect_environment, selected_interface, peer.host)
    usage_duration = time.monotonic() - usage_started
    usage_delta = _usage_delta(usage_before, _process_usage())
    return {
        "schema_version": SCHEMA_VERSION,
        "started_at": started_at,
        "finished_at": datetime.now(timezone.utc).isoformat(),
        "transport": "tcp",
        "semantics": {
            "latency": "application-level payload echo round-trip time",
            "goodput": "application payload bytes divided by end-to-end phase time",
            "jitter_mean_abs_delta": (
                "mean absolute difference between consecutive samples"
            ),
            "jitter_p95_abs_delta": (
                "p95 absolute difference between consecutive samples"
            ),
            "outlier": "sample above p75 + 1.5 * interquartile range",
            "bidirectional_goodput": (
                "aggregate counts both directions; per-direction counts one stream"
            ),
            "packet_loss": (
                "TCP masks link-layer loss; use ping, interface counters, and radio "
                "diagnostics for loss and retransmission evidence"
            ),
        },
        "settings": asdict(settings),
        "connection_setup_ms": connection_setup_ms,
        "ping_to_responder": ping,
        "latency": latency,
        "throughput": throughput,
        "latency_under_load": latency_under_load,
        "tensor": tensor,
        "initiator": {
            "node_id": config.local.node_id,
            "interface": selected_interface,
            "environment_before": before,
            "environment_after": after,
            "interface_counter_delta": counter_delta(
                before["interface"], after["interface"]
            ),
            "kernel_counter_delta": kernel_counter_delta(
                before["kernel_network_counters"],
                after["kernel_network_counters"],
            ),
            "process_observation_duration_s": usage_duration,
            "process_usage_delta": usage_delta,
            "process_usage_rates_per_s": _usage_rates(usage_delta, usage_duration),
            "link_samples": await monitor.stop(),
            "tcp_socket_before_stop": tcp_socket_before_stop,
        },
        "responder": stopped["report"],
        "responder_environment_at_handshake": diagnostics["environment_before"],
        "ping_to_initiator": diagnostics["ping_to_initiator"],
    }


def summarize_samples(samples: list[float]) -> dict[str, float | int]:
    if not samples:
        raise ValueError("cannot summarize an empty sample set")
    ordered = sorted(samples)
    mean = statistics.fmean(ordered)
    median = _percentile(ordered, 0.50)
    p25 = _percentile(ordered, 0.25)
    p75 = _percentile(ordered, 0.75)
    stddev = statistics.pstdev(ordered)
    deltas = [
        abs(current - previous) for previous, current in itertools.pairwise(samples)
    ]
    outlier_threshold = p75 + 1.5 * (p75 - p25)
    outlier_count = sum(sample > outlier_threshold for sample in samples)
    return {
        "sample_count": len(samples),
        "percentile_resolution": 1 / len(samples),
        "min": ordered[0],
        "mean": mean,
        "p01": _percentile(ordered, 0.01),
        "p05": _percentile(ordered, 0.05),
        "p25": p25,
        "p50": median,
        "p75": p75,
        "p90": _percentile(ordered, 0.90),
        "p95": _percentile(ordered, 0.95),
        "p99": _percentile(ordered, 0.99),
        "p99_9": _percentile(ordered, 0.999),
        "max": ordered[-1],
        "range": ordered[-1] - ordered[0],
        "stddev": stddev,
        "coefficient_of_variation": stddev / mean if mean else 0.0,
        "median_absolute_deviation": statistics.median(
            abs(sample - median) for sample in samples
        ),
        "jitter_mean_abs_delta": statistics.fmean(deltas) if deltas else 0.0,
        "jitter_p95_abs_delta": (_percentile(sorted(deltas), 0.95) if deltas else 0.0),
        "jitter_rms_delta": (
            math.sqrt(statistics.fmean(delta * delta for delta in deltas))
            if deltas
            else 0.0
        ),
        "outlier_threshold": outlier_threshold,
        "outlier_count": outlier_count,
        "outlier_fraction": outlier_count / len(samples),
    }


def summarize_throughput_trials(trials: list[dict[str, Any]]) -> dict[str, Any]:
    if not trials:
        raise ValueError("cannot summarize empty throughput trials")
    first = trials[0]
    identity = (
        "payload_bytes_per_message",
        "messages_per_direction",
        "streams",
        "total_payload_bytes",
    )
    if any(any(trial[name] != first[name] for name in identity) for trial in trials):
        raise ValueError("throughput trials do not describe the same workload")
    return {
        **{name: first[name] for name in identity},
        "trial_count": len(trials),
        "total_payload_bytes_all_trials": first["total_payload_bytes"] * len(trials),
        "aggregate_goodput_mbps": summarize_samples(
            [trial["aggregate_goodput_mbps"] for trial in trials]
        ),
        "per_direction_goodput_mbps": summarize_samples(
            [trial["per_direction_goodput_mbps"] for trial in trials]
        ),
        "elapsed_s": summarize_samples([trial["elapsed_s"] for trial in trials]),
        "messages_per_s": summarize_samples(
            [trial["messages_per_s"] for trial in trials]
        ),
        "trials": trials,
    }


async def _measure_upload(
    comm: Comm,
    peer: Peer,
    options: CommOptions,
    data_options: CommOptions,
    size: int,
    count: int,
) -> dict[str, Any]:
    await _send_control(
        comm, peer, {"op": "upload", "size": size, "count": count}, options
    )
    _expect_operation(await _recv_mapping(comm, peer, options), "ready")
    payload = bytes(size)
    started = time.monotonic()
    await _send_payloads(comm, peer, data_options, payload, count)
    completion = await _recv_mapping(comm, peer, options)
    elapsed = time.monotonic() - started
    _expect_operation(completion, "complete")
    return _throughput_result("upload", size, count, elapsed, completion)


async def _measure_download(
    comm: Comm,
    peer: Peer,
    options: CommOptions,
    data_options: CommOptions,
    size: int,
    count: int,
) -> dict[str, Any]:
    await _send_control(
        comm, peer, {"op": "download", "size": size, "count": count}, options
    )
    _expect_operation(await _recv_mapping(comm, peer, options), "ready")
    started = time.monotonic()
    await _receive_payloads(comm, peer, data_options, size, count)
    completion = await _recv_mapping(comm, peer, options)
    elapsed = time.monotonic() - started
    _expect_operation(completion, "complete")
    return _throughput_result("download", size, count, elapsed, completion)


async def _measure_bidirectional(
    comm: Comm,
    peer: Peer,
    options: CommOptions,
    data_options: CommOptions,
    size: int,
    count: int,
) -> dict[str, Any]:
    await _send_control(
        comm,
        peer,
        {"op": "bidirectional", "size": size, "count": count},
        options,
    )
    _expect_operation(await _recv_mapping(comm, peer, options), "ready")
    payload = bytes(size)
    started = time.monotonic()
    await asyncio.gather(
        _send_payloads(comm, peer, data_options, payload, count),
        _receive_payloads(comm, peer, data_options, size, count),
    )
    completion = await _recv_mapping(comm, peer, options)
    elapsed = time.monotonic() - started
    _expect_operation(completion, "complete")
    return _throughput_result(
        "bidirectional", size, count, elapsed, completion, streams=2
    )


async def _ping(
    comm: Comm,
    peer: Peer,
    options: CommOptions,
    sequence: int,
    payload: bytes,
) -> None:
    await _send_control(
        comm,
        peer,
        {"op": "ping", "sequence": sequence, "payload": payload},
        options,
    )
    response = await _recv_mapping(comm, peer, options)
    _expect_operation(response, "pong")
    if response.get("sequence") != sequence or response.get("payload") != payload:
        raise RuntimeError("ping response did not match its request")


async def _measure_tensors(
    comm: Comm,
    peer: Peer,
    options: CommOptions,
    settings: BenchmarkSettings,
) -> dict[str, Any]:
    torch = _require_torch()
    results = []
    sequence = 0
    for size in settings.tensor_sizes:
        tensor = torch.arange(size // 4, dtype=torch.float32)
        rtt_samples = []
        goodput_samples = []
        for _ in range(settings.tensor_rounds):
            started = time.perf_counter_ns()
            await _send_control(
                comm,
                peer,
                {
                    "op": "tensor_ping",
                    "sequence": sequence,
                    "size": size,
                    "payload": tensor,
                },
                options,
            )
            response = await _recv_mapping(comm, peer, options)
            elapsed_s = (time.perf_counter_ns() - started) / 1_000_000_000
            _expect_operation(response, "tensor_pong")
            if response.get("sequence") != sequence or not torch.equal(
                _validate_tensor(response.get("payload")), tensor
            ):
                raise RuntimeError("Tensor response did not match its request")
            rtt_samples.append(elapsed_s * 1000)
            goodput_samples.append(size * 2 * 8 / elapsed_s / 1_000_000)
            sequence += 1
        results.append(
            {
                "payload_bytes": size,
                "elements": tensor.numel(),
                "dtype": str(tensor.dtype),
                "rounds": settings.tensor_rounds,
                "round_trip_ms": summarize_samples(rtt_samples),
                "round_trip_goodput_mbps": summarize_samples(goodput_samples),
                "samples_ms": rtt_samples,
            }
        )

    structures = {
        "single": torch.arange(16, dtype=torch.float32),
        "tensor_list": [
            torch.arange(8, dtype=torch.int64),
            torch.arange(12, dtype=torch.float16),
        ],
        "tensor_dict": {
            "float": torch.arange(10, dtype=torch.float32),
            "bool": torch.tensor([True, False, True]),
        },
    }
    await _send_control(
        comm,
        peer,
        {"op": "tensor_structures", "payload": structures},
        options,
    )
    response = await _recv_mapping(comm, peer, options)
    _expect_operation(response, "tensor_structures")
    if not _tensor_structures_equal(response.get("payload"), structures):
        raise RuntimeError("Tensor structure response did not match its request")
    return {
        "torch_version": torch.__version__,
        "device": "cpu",
        "contiguous": True,
        "round_trip": results,
        "structure_round_trip": {
            "single_tensor": True,
            "tensor_list": True,
            "tensor_dict": True,
        },
    }


async def _measure_latency_under_load(
    comm: Comm,
    peer: Peer,
    options: CommOptions,
    settings: BenchmarkSettings,
    throughput: dict[str, list[dict[str, Any]]],
) -> list[dict[str, Any]]:
    """Measure foreground Comm RTT while independent TCP streams carry load."""

    if peer.port == 65535:
        raise ValueError("latency-under-load requires peer.port below 65535")
    random_source = random.Random(settings.random_seed)
    results = []
    sequence = 1_000_000
    for direction in settings.load_directions:
        capacity_mbps = max(
            result["aggregate_goodput_mbps"]["p50"] for result in throughput[direction]
        )
        for concurrency in settings.load_concurrencies:
            for fraction in settings.load_fractions:
                target_mbps = capacity_mbps * fraction
                load_task = asyncio.create_task(
                    run_load(
                        peer.host,
                        peer.port + 1,
                        direction=direction,
                        streams=concurrency,
                        duration=settings.load_duration,
                        target_mbps=target_mbps,
                    )
                )
                await asyncio.sleep(min(0.2, settings.load_duration / 10))
                deadline = time.monotonic() + settings.load_duration * 0.8
                samples = []
                payload = bytes(settings.load_probe_size)
                while time.monotonic() < deadline and not load_task.done():
                    started = time.perf_counter_ns()
                    await _ping(comm, peer, options, sequence, payload)
                    samples.append((time.perf_counter_ns() - started) / 1_000_000)
                    sequence += 1
                    interval = settings.load_probe_interval * random_source.uniform(
                        0.5, 1.5
                    )
                    await asyncio.sleep(interval)
                load_result = await load_task
                results.append(
                    {
                        "direction": direction,
                        "concurrency_per_direction": concurrency,
                        "capacity_reference_mbps": capacity_mbps,
                        "load_fraction": fraction,
                        "probe_payload_bytes": settings.load_probe_size,
                        "probe_interval_s": settings.load_probe_interval,
                        "probe_summary_ms": summarize_samples(samples),
                        "probe_samples_ms": samples,
                        "background_load": load_result_dict(load_result),
                        "delivered_load_fraction": (
                            load_result.goodput_mbps / capacity_mbps
                        ),
                    }
                )
    return results


def _validate_tensor(value: Any) -> Any:
    torch = _require_torch()
    if type(value) is not torch.Tensor:
        raise TypeError("benchmark Tensor payload must be a torch.Tensor")
    if value.device.type != "cpu" or not value.is_contiguous():
        raise ValueError("benchmark Tensor payload must be contiguous and on CPU")
    return value


def _validate_tensor_structures(value: Any) -> None:
    if not isinstance(value, dict) or set(value) != {
        "single",
        "tensor_list",
        "tensor_dict",
    }:
        raise TypeError("invalid Tensor structure benchmark payload")
    _validate_tensor(value["single"])
    tensor_list = value["tensor_list"]
    tensor_dict = value["tensor_dict"]
    if not isinstance(tensor_list, list) or len(tensor_list) != 2:
        raise TypeError("invalid Tensor list benchmark payload")
    if not isinstance(tensor_dict, dict) or set(tensor_dict) != {"float", "bool"}:
        raise TypeError("invalid Tensor dict benchmark payload")
    for tensor in [*tensor_list, *tensor_dict.values()]:
        _validate_tensor(tensor)


def _tensor_structures_equal(received: Any, expected: Any) -> bool:
    try:
        _validate_tensor_structures(received)
    except (TypeError, ValueError):
        return False
    torch = _require_torch()
    return (
        torch.equal(received["single"], expected["single"])
        and all(
            torch.equal(left, right)
            for left, right in zip(received["tensor_list"], expected["tensor_list"])
        )
        and all(
            torch.equal(received["tensor_dict"][key], expected["tensor_dict"][key])
            for key in expected["tensor_dict"]
        )
    )


def _require_torch() -> Any:
    try:
        import torch
    except ImportError as exc:
        raise RuntimeError(
            "Tensor benchmark requested but PyTorch is not installed"
        ) from exc
    return torch


async def _send_payloads(
    comm: Comm,
    peer: Peer,
    options: CommOptions,
    payload: bytes,
    count: int,
) -> None:
    for _ in range(count):
        await comm.send(payload, peer, options=options)


async def _receive_payloads(
    comm: Comm,
    peer: Peer,
    options: CommOptions,
    size: int,
    count: int,
) -> None:
    for _ in range(count):
        payload, metadata = await comm.recv(peer, options)
        if not isinstance(payload, bytes) or len(payload) != size:
            raise RuntimeError("benchmark data payload has an unexpected size or type")
        if metadata is not None:
            raise RuntimeError("benchmark data payload unexpectedly contains metadata")


async def _send_control(
    comm: Comm,
    peer: Peer,
    message: dict[str, Any],
    options: CommOptions,
) -> None:
    await comm.send(message, peer, options=options)


async def _recv_mapping(comm: Comm, peer: Peer, options: CommOptions) -> dict[str, Any]:
    message, metadata = await comm.recv(peer, options)
    if not isinstance(message, dict) or metadata is not None:
        raise RuntimeError("invalid benchmark control message")
    return message


async def _create_comm(config: RuntimeConfig) -> Comm:
    return await Comm.create(
        local=config.local,
        peers=config.peers,
        bind_host=config.bind_host,
        config=config.comm,
    )


def _completion(
    operation: str, size: int, count: int, elapsed: float
) -> dict[str, Any]:
    return {
        "op": "complete",
        "phase": operation,
        "size": size,
        "count": count,
        "responder_elapsed_s": elapsed,
    }


def _throughput_result(
    direction: str,
    size: int,
    count: int,
    elapsed: float,
    completion: dict[str, Any],
    *,
    streams: int = 1,
) -> dict[str, Any]:
    if (
        completion.get("phase") != direction
        or completion.get("size") != size
        or completion.get("count") != count
    ):
        raise RuntimeError("throughput completion does not match its phase")
    payload_bytes = size * count * streams
    per_direction_payload_bytes = size * count
    return {
        "payload_bytes_per_message": size,
        "messages_per_direction": count,
        "streams": streams,
        "total_payload_bytes": payload_bytes,
        "elapsed_s": elapsed,
        "aggregate_goodput_mbps": payload_bytes * 8 / elapsed / 1_000_000,
        "per_direction_goodput_mbps": (
            per_direction_payload_bytes * 8 / elapsed / 1_000_000
        ),
        "messages_per_s": count * streams / elapsed,
        "responder_elapsed_s": completion["responder_elapsed_s"],
    }


def _positive_int(mapping: dict[str, Any], name: str) -> int:
    value = mapping.get(name)
    if type(value) is not int or value <= 0:
        raise ValueError(f"benchmark field {name!r} must be a positive integer")
    return value


def _expect_operation(message: dict[str, Any], expected: str) -> None:
    if message.get("op") != expected:
        raise RuntimeError(
            f"expected benchmark operation {expected!r}, got {message.get('op')!r}"
        )


def _percentile(ordered: list[float], fraction: float) -> float:
    position = (len(ordered) - 1) * fraction
    lower = math.floor(position)
    upper = math.ceil(position)
    if lower == upper:
        return ordered[lower]
    weight = position - lower
    return ordered[lower] * (1 - weight) + ordered[upper] * weight


def _process_usage() -> dict[str, float | int]:
    usage = resource.getrusage(resource.RUSAGE_SELF)
    return {
        "user_cpu_s": usage.ru_utime,
        "system_cpu_s": usage.ru_stime,
        "max_rss_kib": usage.ru_maxrss,
        "voluntary_context_switches": usage.ru_nvcsw,
        "involuntary_context_switches": usage.ru_nivcsw,
    }


def _usage_delta(
    before: dict[str, float | int], after: dict[str, float | int]
) -> dict[str, float | int]:
    return {name: after[name] - before[name] for name in before}


def _usage_rates(delta: dict[str, float | int], duration: float) -> dict[str, float]:
    return {
        "user_cpu_s": delta["user_cpu_s"] / duration,
        "system_cpu_s": delta["system_cpu_s"] / duration,
        "voluntary_context_switches": (delta["voluntary_context_switches"] / duration),
        "involuntary_context_switches": (
            delta["involuntary_context_switches"] / duration
        ),
    }


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True, help="node YAML configuration")
    subparsers = parser.add_subparsers(dest="command", required=True)

    responder = subparsers.add_parser("responder")
    _add_common_arguments(responder)

    initiator = subparsers.add_parser("initiator")
    _add_common_arguments(initiator)
    initiator.add_argument("--latency-sizes", default="0,64,1024,65536")
    initiator.add_argument("--latency-rounds", type=int, default=100)
    initiator.add_argument("--warmup-rounds", type=int, default=10)
    initiator.add_argument("--throughput-sizes", default="1024,65536,1048576,8388608")
    initiator.add_argument(
        "--throughput-target-bytes", type=int, default=16 * 1024 * 1024
    )
    initiator.add_argument("--min-throughput-messages", type=int, default=4)
    initiator.add_argument("--max-throughput-messages", type=int, default=4096)
    initiator.add_argument("--throughput-trials", type=int, default=5)
    initiator.add_argument("--directions", default="upload,download,bidirectional")
    initiator.add_argument("--ping-count", type=int, default=10)
    initiator.add_argument(
        "--tensor-sizes",
        default="",
        help="optional comma-separated contiguous CPU Tensor sizes in bytes",
    )
    initiator.add_argument("--tensor-rounds", type=int, default=10)
    initiator.add_argument(
        "--load-fractions",
        default="",
        help="capacity fractions for latency-under-load, for example 0.25,0.5,0.9",
    )
    initiator.add_argument("--load-directions", default="upload,download,bidirectional")
    initiator.add_argument("--load-concurrencies", default="1,4")
    initiator.add_argument("--load-duration", type=float, default=10.0)
    initiator.add_argument("--load-probe-interval", type=float, default=0.05)
    initiator.add_argument("--load-probe-size", type=int, default=64)
    initiator.add_argument("--random-seed", type=int, default=0)
    return parser


def _add_common_arguments(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--peer", required=True, help="configured remote node_id")
    parser.add_argument(
        "--interface", help="Wi-Fi interface; defaults to the route-selected interface"
    )
    parser.add_argument(
        "--output", type=Path, help="write the JSON report to this file"
    )
    parser.add_argument("--timeout", type=float, default=120.0)
    parser.add_argument("--sample-interval", type=float, default=1.0)


async def _main(args: argparse.Namespace) -> dict[str, Any]:
    config = load_runtime_config(args.config)
    peers = {peer.node_id: peer for peer in config.peers}
    try:
        peer = peers[args.peer]
    except KeyError as exc:
        raise ValueError(f"unknown configured peer {args.peer!r}") from exc

    if args.command == "responder":
        return await run_responder(
            config,
            peer,
            interface=args.interface,
            operation_timeout=args.timeout,
            sample_interval=args.sample_interval,
        )
    settings = BenchmarkSettings(
        latency_sizes=_parse_sizes(args.latency_sizes),
        latency_rounds=args.latency_rounds,
        warmup_rounds=args.warmup_rounds,
        throughput_sizes=_parse_sizes(args.throughput_sizes),
        throughput_target_bytes=args.throughput_target_bytes,
        min_throughput_messages=args.min_throughput_messages,
        max_throughput_messages=args.max_throughput_messages,
        throughput_trials=args.throughput_trials,
        directions=tuple(args.directions.split(",")),
        tensor_sizes=_parse_sizes(args.tensor_sizes, allow_empty=True),
        tensor_rounds=args.tensor_rounds,
        ping_count=args.ping_count,
        operation_timeout=args.timeout,
        sample_interval=args.sample_interval,
        load_fractions=_parse_floats(args.load_fractions, allow_empty=True),
        load_directions=tuple(args.load_directions.split(",")),
        load_concurrencies=_parse_sizes(args.load_concurrencies),
        load_duration=args.load_duration,
        load_probe_interval=args.load_probe_interval,
        load_probe_size=args.load_probe_size,
        random_seed=args.random_seed,
    )
    return await run_initiator(
        config,
        peer,
        settings,
        interface=args.interface,
    )


def _parse_sizes(raw: str, *, allow_empty: bool = False) -> tuple[int, ...]:
    if allow_empty and not raw:
        return ()
    try:
        return tuple(int(value) for value in raw.split(","))
    except ValueError as exc:
        raise ValueError("message sizes must be comma-separated integers") from exc


def _parse_floats(raw: str, *, allow_empty: bool = False) -> tuple[float, ...]:
    if allow_empty and not raw:
        return ()
    try:
        return tuple(float(value) for value in raw.split(","))
    except ValueError as exc:
        raise ValueError("values must be comma-separated numbers") from exc


def main() -> None:
    args = build_parser().parse_args()
    report = asyncio.run(_main(args))
    serialized = json.dumps(report, indent=2, sort_keys=True)
    if args.output is None:
        print(serialized)
    else:
        args.output.write_text(serialized + "\n", encoding="utf-8")
        print(f"wrote benchmark report to {args.output}", file=sys.stderr)


if __name__ == "__main__":
    main()
