"""Portable, read-only diagnostics used by the Wi-Fi benchmark."""

from __future__ import annotations

import asyncio
import os
import platform
import re
import shutil
import socket
import subprocess
import sys
import time
from pathlib import Path
from typing import Any

_INTERFACE_COUNTERS = (
    "rx_bytes",
    "rx_packets",
    "rx_errors",
    "rx_dropped",
    "tx_bytes",
    "tx_packets",
    "tx_errors",
    "tx_dropped",
)

_SOFTNET_FIELDS = {
    "processed": 0,
    "dropped": 1,
    "time_squeeze": 2,
    "cpu_collision": 8,
    "received_rps": 9,
    "flow_limit_count": 10,
}


def resolve_interface(peer_host: str) -> str | None:
    """Return the interface selected by the kernel route to ``peer_host``."""

    result = run_command(["ip", "-o", "route", "get", peer_host])
    if result["returncode"] != 0:
        return None
    fields = result["stdout"].split()
    try:
        return fields[fields.index("dev") + 1]
    except (ValueError, IndexError):
        return None


def collect_environment(interface: str | None, peer_host: str) -> dict[str, Any]:
    """Capture system, route, radio, and network state without requiring root."""

    return {
        "captured_at_unix_s": time.time(),
        "hostname": socket.gethostname(),
        "platform": platform.platform(),
        "machine": platform.machine(),
        "python": sys.version,
        "cpu_count": os.cpu_count(),
        "cpu_details": run_command(["lscpu", "-J"]),
        "load_average": _load_average(),
        "memory": _memory_info(),
        "power_mode": run_command(["nvpmodel", "-q"]),
        "clock_sync": run_command(
            ["timedatectl", "show", "-p", "NTPSynchronized", "-p", "TimeUSec"]
        ),
        "interface": collect_interface(interface),
        "route": run_command(["ip", "route", "get", peer_host]),
        "qdisc": (
            run_command(["tc", "-s", "qdisc", "show", "dev", interface])
            if interface is not None
            else None
        ),
        "radio": collect_radio(interface),
        "kernel_network_counters": collect_kernel_network_counters(interface),
        "tcp_sockets": collect_tcp_sockets(peer_host),
        "tcp": {
            "congestion_control": _read_text(
                "/proc/sys/net/ipv4/tcp_congestion_control"
            ),
            "available_congestion_control": _read_text(
                "/proc/sys/net/ipv4/tcp_available_congestion_control"
            ),
            "tcp_mtu_probing": _read_text("/proc/sys/net/ipv4/tcp_mtu_probing"),
            "tcp_rmem": _read_text("/proc/sys/net/ipv4/tcp_rmem"),
            "tcp_wmem": _read_text("/proc/sys/net/ipv4/tcp_wmem"),
            "default_qdisc": _read_text("/proc/sys/net/core/default_qdisc"),
        },
    }


def collect_kernel_network_counters(interface: str | None) -> dict[str, Any]:
    """Read scheduler, softirq, softnet, and interface IRQ counters."""

    proc_stat = _read_text("/proc/stat")
    softirqs = _read_text("/proc/softirqs")
    softnet = _read_text("/proc/net/softnet_stat")
    return {
        "context_switches": (
            parse_context_switches(proc_stat) if proc_stat is not None else None
        ),
        "softirqs": parse_softirqs(softirqs) if softirqs is not None else None,
        "softnet": parse_softnet_stat(softnet) if softnet is not None else None,
        "interface_irq": collect_interface_irq(interface),
    }


def collect_interface_irq(interface: str | None) -> dict[str, Any] | None:
    if interface is None:
        return None
    irq = _read_int_optional(Path("/sys/class/net") / interface / "device" / "irq")
    interrupts = _read_text("/proc/interrupts")
    if irq is None or interrupts is None:
        return None
    return parse_interrupt(interrupts, irq)


def parse_context_switches(output: str) -> int:
    for line in output.splitlines():
        fields = line.split()
        if fields and fields[0] == "ctxt":
            return int(fields[1])
    raise ValueError("/proc/stat does not contain a ctxt counter")


def parse_softirqs(output: str) -> dict[str, dict[str, Any]]:
    counters: dict[str, dict[str, Any]] = {}
    for line in output.splitlines():
        name, separator, raw_values = line.partition(":")
        name = name.strip()
        if separator and name in {"NET_RX", "NET_TX"}:
            per_cpu = [int(value) for value in raw_values.split()]
            counters[name] = {"total": sum(per_cpu), "per_cpu": per_cpu}
    return counters


def parse_softnet_stat(output: str) -> dict[str, dict[str, Any]]:
    rows = [[int(value, 16) for value in line.split()] for line in output.splitlines()]
    return {
        name: {
            "total": sum(row[index] for row in rows),
            "per_cpu": [row[index] for row in rows],
        }
        for name, index in _SOFTNET_FIELDS.items()
    }


def parse_interrupt(output: str, irq: int) -> dict[str, Any] | None:
    lines = output.splitlines()
    if not lines:
        return None
    cpu_count = len(lines[0].split())
    prefix = f"{irq}:"
    for line in lines[1:]:
        fields = line.split()
        if not fields or fields[0] != prefix:
            continue
        per_cpu = [int(value) for value in fields[1 : cpu_count + 1]]
        return {
            "irq": irq,
            "total": sum(per_cpu),
            "per_cpu": per_cpu,
            "description": " ".join(fields[cpu_count + 1 :]),
        }
    return None


def kernel_counter_delta(
    before: dict[str, Any] | None, after: dict[str, Any] | None
) -> dict[str, Any] | None:
    if before is None or after is None:
        return None
    return {
        "context_switches": _optional_delta(
            before["context_switches"], after["context_switches"]
        ),
        "softirqs": _counter_groups_delta(before["softirqs"], after["softirqs"]),
        "softnet": _counter_groups_delta(before["softnet"], after["softnet"]),
        "interface_irq": _irq_delta(before["interface_irq"], after["interface_irq"]),
    }


def collect_interface(interface: str | None) -> dict[str, Any] | None:
    if interface is None:
        return None
    root = Path("/sys/class/net") / interface
    if not root.exists():
        raise ValueError(f"network interface {interface!r} does not exist")
    return {
        "name": interface,
        "mtu": _read_int(root / "mtu"),
        "operstate": _read_text(root / "operstate"),
        "address": _read_text(root / "address"),
        "statistics": collect_interface_statistics(interface),
        "addresses": run_command(["ip", "-j", "address", "show", "dev", interface]),
    }


def collect_radio(interface: str | None) -> dict[str, Any] | None:
    if interface is None:
        return None
    link = collect_link(interface)
    return {
        "link": link["command"],
        "link_fields": link["fields"],
        "station_dump": run_command(["iw", "dev", interface, "station", "dump"]),
        "survey_dump": run_command(["iw", "dev", interface, "survey", "dump"]),
        "power_save": run_command(["iw", "dev", interface, "get", "power_save"]),
        "device": run_command(["iw", "dev", interface, "info"]),
        "regulatory_domain": run_command(["iw", "reg", "get"]),
        "driver": run_command(["ethtool", "-i", interface]),
        "driver_statistics": run_command(["ethtool", "-S", interface]),
        "offload_features": run_command(["ethtool", "-k", interface]),
    }


def collect_link(interface: str) -> dict[str, Any]:
    command = run_command(["iw", "dev", interface, "link"])
    return {"command": command, "fields": parse_iw_link(command["stdout"])}


def collect_interface_statistics(interface: str) -> dict[str, int]:
    root = Path("/sys/class/net") / interface / "statistics"
    return {name: _read_int(root / name) for name in _INTERFACE_COUNTERS}


def parse_iw_link(output: str) -> dict[str, Any]:
    """Extract stable fields while retaining the complete raw ``iw`` output."""

    fields: dict[str, Any] = {}
    patterns = {
        "ssid": r"^\s*SSID:\s*(.+)$",
        "frequency_mhz": r"^\s*freq:\s*([\d.]+)$",
        "signal_dbm": r"^\s*signal:\s*(-?[\d.]+)\s+dBm$",
        "rx_bitrate": r"^\s*rx bitrate:\s*(.+)$",
        "tx_bitrate": r"^\s*tx bitrate:\s*(.+)$",
    }
    for name, pattern in patterns.items():
        match = re.search(pattern, output, re.MULTILINE)
        if match:
            fields[name] = match.group(1)
    connected = re.search(r"^Connected to\s+(\S+)", output, re.MULTILINE)
    if connected:
        fields["access_point"] = connected.group(1)
    if "frequency_mhz" in fields:
        frequency = float(fields["frequency_mhz"])
        fields["frequency_mhz"] = (
            int(frequency) if frequency.is_integer() else frequency
        )
    if "signal_dbm" in fields:
        fields["signal_dbm"] = float(fields["signal_dbm"])
    return fields


def run_ping(peer_host: str, interface: str | None, count: int) -> dict[str, Any]:
    command = ["ping", "-n", "-c", str(count), "-W", "2"]
    if interface is not None:
        command.extend(["-I", interface])
    command.append(peer_host)
    result = run_command(command, timeout=max(5.0, count * 2.5))
    parsed: dict[str, Any] = {"command": result}
    packets = re.search(
        r"(\d+) packets transmitted, (\d+) received,.*?([\d.]+)% packet loss",
        result["stdout"],
    )
    if packets:
        parsed.update(
            {
                "transmitted": int(packets.group(1)),
                "received": int(packets.group(2)),
                "loss_percent": float(packets.group(3)),
            }
        )
    timing = re.search(
        r"(?:rtt|round-trip) min/avg/max/(?:mdev|stddev) = "
        r"([\d.]+)/([\d.]+)/([\d.]+)/([\d.]+) ms",
        result["stdout"],
    )
    if timing:
        parsed.update(
            {
                "min_ms": float(timing.group(1)),
                "avg_ms": float(timing.group(2)),
                "max_ms": float(timing.group(3)),
                "jitter_ms": float(timing.group(4)),
            }
        )
    return parsed


def collect_tcp_sockets(peer_host: str) -> dict[str, Any]:
    """Capture Linux TCP_INFO-style diagnostics exposed by ``ss``."""

    return run_command(["ss", "-tinm", "dst", peer_host])


def counter_delta(
    before: dict[str, Any] | None, after: dict[str, Any] | None
) -> dict[str, int] | None:
    if before is None or after is None:
        return None
    old = before["statistics"]
    new = after["statistics"]
    return {name: new[name] - old[name] for name in _INTERFACE_COUNTERS}


def run_command(command: list[str], *, timeout: float = 5.0) -> dict[str, Any]:
    executable = shutil.which(command[0])
    if executable is None:
        return {
            "command": command,
            "available": False,
            "returncode": None,
            "stdout": "",
            "stderr": f"{command[0]} is not installed",
        }
    try:
        completed = subprocess.run(
            [executable, *command[1:]],
            capture_output=True,
            text=True,
            timeout=timeout,
            check=False,
        )
    except subprocess.TimeoutExpired as exc:
        return {
            "command": command,
            "available": True,
            "returncode": None,
            "stdout": exc.stdout or "",
            "stderr": f"command timed out after {timeout:g}s",
        }
    return {
        "command": command,
        "available": True,
        "returncode": completed.returncode,
        "stdout": completed.stdout.strip(),
        "stderr": completed.stderr.strip(),
    }


class LinkMonitor:
    """Periodically sample radio and interface counters during a benchmark."""

    def __init__(self, interface: str | None, interval: float) -> None:
        self.interface = interface
        self.interval = interval
        self.samples: list[dict[str, Any]] = []
        self._task: asyncio.Task[None] | None = None

    def start(self) -> None:
        if self.interface is not None:
            self._task = asyncio.create_task(self._run())

    async def stop(self) -> list[dict[str, Any]]:
        if self._task is not None:
            self._task.cancel()
            await asyncio.gather(self._task, return_exceptions=True)
        return self.samples

    async def _run(self) -> None:
        while True:
            link, statistics = await asyncio.gather(
                asyncio.to_thread(collect_link, self.interface),
                asyncio.to_thread(collect_interface_statistics, self.interface),
            )
            self.samples.append(
                {
                    "captured_at_unix_s": time.time(),
                    "link_fields": link["fields"],
                    "interface_statistics": statistics,
                    "kernel_network_counters": collect_kernel_network_counters(
                        self.interface
                    ),
                }
            )
            await asyncio.sleep(self.interval)


def _load_average() -> list[float] | None:
    try:
        return list(os.getloadavg())
    except OSError:
        return None


def _memory_info() -> dict[str, int]:
    values: dict[str, int] = {}
    with Path("/proc/meminfo").open(encoding="utf-8") as stream:
        for line in stream:
            name, raw = line.split(":", 1)
            if name in {"MemTotal", "MemAvailable", "SwapTotal", "SwapFree"}:
                values[f"{name}_kib"] = int(raw.split()[0])
    return values


def _read_text(path: str | Path) -> str | None:
    try:
        return Path(path).read_text(encoding="utf-8").strip()
    except FileNotFoundError:
        return None


def _read_int(path: str | Path) -> int:
    value = _read_text(path)
    if value is None:
        raise FileNotFoundError(path)
    return int(value)


def _read_int_optional(path: str | Path) -> int | None:
    value = _read_text(path)
    return int(value) if value is not None else None


def _optional_delta(before: int | None, after: int | None) -> int | None:
    if before is None or after is None:
        return None
    return after - before


def _counter_groups_delta(
    before: dict[str, dict[str, Any]] | None,
    after: dict[str, dict[str, Any]] | None,
) -> dict[str, dict[str, Any]] | None:
    if before is None or after is None:
        return None
    return {
        name: {
            "total": after[name]["total"] - old["total"],
            "per_cpu": [
                new - previous
                for previous, new in zip(old["per_cpu"], after[name]["per_cpu"])
            ],
        }
        for name, old in before.items()
        if name in after
    }


def _irq_delta(
    before: dict[str, Any] | None, after: dict[str, Any] | None
) -> dict[str, Any] | None:
    if before is None or after is None or before["irq"] != after["irq"]:
        return None
    return {
        "irq": after["irq"],
        "description": after["description"],
        "total": after["total"] - before["total"],
        "per_cpu": [new - old for old, new in zip(before["per_cpu"], after["per_cpu"])],
    }
