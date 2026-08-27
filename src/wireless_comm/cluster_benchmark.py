"""Controller/worker runner for physical multi-node Wi-Fi collectives."""

from __future__ import annotations

import argparse
import asyncio
import json
import shlex
import sys
import time
from dataclasses import asdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from .cluster_config import (
    ClusterBenchmarkConfig,
    ClusterNode,
    ClusterTopology,
    load_cluster_config,
)
from .comm import Comm
from .diagnostics import (
    LinkMonitor,
    collect_environment,
    counter_delta,
    kernel_counter_delta,
    resolve_interface,
)
from .errors import CommError
from .multirank import (
    CONTROL_TAG,
    MultiRankSettings,
    _run_rank,
    _summarize_collectives,
    _validate_settings_for_world,
)
from .types import CommOptions, Peer


async def run_cluster_node(
    config: ClusterBenchmarkConfig,
    node_id: str,
    *,
    sample_interval: float = 1.0,
) -> dict[str, Any]:
    """Run all ranks owned by one physical node."""

    topology = config.topology
    settings = config.benchmark
    node = topology.node(node_id)
    directory = topology.peers()
    _validate_settings_for_world(settings, topology.world_size)
    comms = await asyncio.gather(
        *(
            Comm.create(
                local=directory[rank],
                peers=tuple(
                    peer for index, peer in enumerate(directory) if index != rank
                ),
                bind_host=node.bind_host,
                config=config.comm,
            )
            for rank in node.ranks
        )
    )
    comm_by_rank = dict(zip(node.ranks, comms))
    remote_host = next(
        cluster_node.host for cluster_node in topology.nodes if cluster_node != node
    )
    interface = node.interface or resolve_interface(remote_host)
    before = await asyncio.to_thread(collect_environment, interface, remote_host)
    monitor = LinkMonitor(interface, sample_interval)
    monitor.start()
    control_options = CommOptions(tag=CONTROL_TAG, timeout=settings.timeout)
    try:
        if node_id == topology.controller:
            await _controller_barrier(topology, comm_by_rank[0], directory, settings)
        else:
            await _worker_barrier(topology, node_id, comm_by_rank, directory, settings)

        started_at = datetime.now(timezone.utc).isoformat()
        rank_results = await asyncio.gather(
            *(
                _run_rank(rank, comm_by_rank[rank], directory, settings, topology.ring)
                for rank in node.ranks
            )
        )
        after = await asyncio.to_thread(collect_environment, interface, remote_host)
        side_report = {
            "node_id": node_id,
            "local_ranks": list(node.ranks),
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

        if node_id == topology.controller:
            remote_reports = await asyncio.gather(
                *(
                    _receive_worker_results(
                        comm_by_rank[0],
                        directory[topology.coordinator_rank(remote.node_id)],
                        control_options,
                        remote.node_id,
                    )
                    for remote in topology.nodes
                    if remote.node_id != topology.controller
                )
            )
            all_rank_results = [*rank_results]
            node_reports = {node_id: side_report}
            for remote in remote_reports:
                all_rank_results.extend(remote["rank_results"])
                node_reports[remote["node_id"]] = remote["side"]
            all_rank_results.sort(key=lambda result: result["rank"])
            return {
                "schema_version": 1,
                "started_at": started_at,
                "finished_at": datetime.now(timezone.utc).isoformat(),
                "mode": "physical_multi_node_wifi",
                "physical_node_count": len(topology.nodes),
                "world_size": topology.world_size,
                "ring": list(topology.ring),
                "settings": asdict(settings),
                "nodes": node_reports,
                "rank_results": all_rank_results,
                "collective_summary": _summarize_collectives(all_rank_results),
            }

        coordinator = comm_by_rank[topology.coordinator_rank(node_id)]
        await coordinator.send(
            {
                "op": "results",
                "node_id": node_id,
                "rank_results": rank_results,
                "side": side_report,
            },
            directory[0],
            options=control_options,
        )
        return {
            "schema_version": 1,
            "node_id": node_id,
            "side": side_report,
            "rank_results": rank_results,
        }
    finally:
        await monitor.stop()
        await asyncio.gather(*(comm.close() for comm in comms))


async def _controller_barrier(
    topology: ClusterTopology,
    control: Comm,
    directory: tuple[Peer, ...],
    settings: MultiRankSettings,
) -> None:
    options = CommOptions(tag=CONTROL_TAG, timeout=settings.timeout)
    workers = [node for node in topology.nodes if node.node_id != topology.controller]
    ready_messages = await asyncio.gather(
        *(
            control.recv(
                directory[topology.coordinator_rank(node.node_id)],
                options,
            )
            for node in workers
        )
    )
    for node, (message, metadata) in zip(workers, ready_messages):
        if message != {"op": "ready", "node_id": node.node_id} or metadata is not None:
            raise RuntimeError(f"node {node.node_id!r} sent an invalid ready message")
    await asyncio.gather(
        *(
            control.send(
                {"op": "go"},
                directory[topology.coordinator_rank(node.node_id)],
                options=options,
            )
            for node in workers
        )
    )


async def _worker_barrier(
    topology: ClusterTopology,
    node_id: str,
    comm_by_rank: dict[int, Comm],
    directory: tuple[Peer, ...],
    settings: MultiRankSettings,
) -> None:
    coordinator = comm_by_rank[topology.coordinator_rank(node_id)]
    deadline = time.monotonic() + settings.timeout
    await _send_with_retry(
        coordinator,
        {"op": "ready", "node_id": node_id},
        directory[0],
        deadline,
    )
    message, metadata = await coordinator.recv(
        directory[0], CommOptions(tag=CONTROL_TAG, timeout=settings.timeout)
    )
    if message != {"op": "go"} or metadata is not None:
        raise RuntimeError("controller sent an invalid go message")


async def _send_with_retry(
    comm: Comm,
    message: dict[str, Any],
    peer: Peer,
    deadline: float,
) -> None:
    while True:
        remaining = deadline - time.monotonic()
        if remaining <= 0:
            raise TimeoutError("timed out connecting to cluster controller")
        try:
            await comm.send(
                message,
                peer,
                options=CommOptions(tag=CONTROL_TAG, timeout=min(5.0, remaining)),
            )
            return
        except CommError:
            await asyncio.sleep(min(0.2, remaining))


async def _receive_worker_results(
    control: Comm,
    peer: Peer,
    options: CommOptions,
    expected_node_id: str,
) -> dict[str, Any]:
    message, metadata = await control.recv(peer, options)
    if (
        not isinstance(message, dict)
        or message.get("op") != "results"
        or message.get("node_id") != expected_node_id
        or metadata is not None
    ):
        raise RuntimeError(f"node {expected_node_id!r} sent invalid results")
    return message


async def launch_cluster(
    config_path: Path,
    *,
    output: Path | None,
    timeout: float,
    dry_run: bool,
) -> dict[str, Any]:
    """Launch every physical-node worker through its configured SSH alias."""

    config = load_cluster_config(config_path)
    commands = {
        node.node_id: _remote_worker_command(node, ".wireless_comm_cluster.yaml")
        for node in config.topology.nodes
    }
    if dry_run:
        return {"copies": _copy_commands(config, config_path), "workers": commands}

    copy_processes = [
        await asyncio.create_subprocess_exec(*command)
        for command in _copy_commands(config, config_path)
    ]
    copy_codes = await asyncio.gather(*(process.wait() for process in copy_processes))
    if any(code != 0 for code in copy_codes):
        raise RuntimeError("failed to copy cluster configuration to every node")

    workers = [
        node
        for node in config.topology.nodes
        if node.node_id != config.topology.controller
    ]
    workers.append(config.topology.node(config.topology.controller))
    processes = {
        node.node_id: await asyncio.create_subprocess_exec(
            "ssh",
            node.ssh_alias,
            "bash",
            "-lc",
            shlex.quote(commands[node.node_id]),
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
        )
        for node in workers
    }
    try:
        results = await asyncio.wait_for(
            asyncio.gather(*(process.communicate() for process in processes.values())),
            timeout,
        )
    except asyncio.TimeoutError:
        for process in processes.values():
            process.terminate()
        await asyncio.gather(*(process.wait() for process in processes.values()))
        raise TimeoutError(f"cluster benchmark exceeded {timeout:g}s") from None

    outputs = dict(zip(processes, results))
    failures = {
        node_id: stderr.decode(errors="replace")
        for node_id, (_, stderr) in outputs.items()
        if processes[node_id].returncode != 0
    }
    if failures:
        raise RuntimeError(f"cluster workers failed: {failures}")
    controller_stdout = outputs[config.topology.controller][0]
    report = json.loads(controller_stdout)
    if output is not None:
        output.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    return report


def _copy_commands(
    config: ClusterBenchmarkConfig, config_path: Path
) -> list[list[str]]:
    return [
        [
            "scp",
            str(config_path),
            f"{node.ssh_alias}:{node.workdir}/.wireless_comm_cluster.yaml",
        ]
        for node in config.topology.nodes
    ]


def _remote_worker_command(node: ClusterNode, config_name: str) -> str:
    command = shlex.join(
        [
            node.python,
            "-m",
            "wireless_comm.cluster_benchmark",
            "worker",
            "--config",
            config_name,
            "--node-id",
            node.node_id,
        ]
    )
    return f"cd {shlex.quote(node.workdir)} && PYTHONPATH=src {command}"


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)
    worker = subparsers.add_parser("worker")
    worker.add_argument("--config", type=Path, required=True)
    worker.add_argument("--node-id", required=True)
    worker.add_argument("--sample-interval", type=float, default=1.0)
    launcher = subparsers.add_parser("launch")
    launcher.add_argument("--config", type=Path, required=True)
    launcher.add_argument("--output", type=Path)
    launcher.add_argument("--timeout", type=float, default=3600.0)
    launcher.add_argument("--dry-run", action="store_true")
    return parser


async def _main(args: argparse.Namespace) -> dict[str, Any]:
    if args.command == "worker":
        return await run_cluster_node(
            load_cluster_config(args.config),
            args.node_id,
            sample_interval=args.sample_interval,
        )
    return await launch_cluster(
        args.config,
        output=args.output,
        timeout=args.timeout,
        dry_run=args.dry_run,
    )


def main() -> None:
    args = build_parser().parse_args()
    report = asyncio.run(_main(args))
    if args.command == "launch" and args.output is not None and not args.dry_run:
        print(f"wrote cluster report to {args.output}", file=sys.stderr)
    else:
        print(json.dumps(report, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
