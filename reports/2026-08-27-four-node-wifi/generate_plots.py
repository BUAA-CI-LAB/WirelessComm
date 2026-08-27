"""Generate the four-node WiFi benchmark figures from raw result files."""

from __future__ import annotations

import argparse
import json
import re
from collections import defaultdict
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from matplotlib import font_manager
from matplotlib.patches import Patch

NODE_NAMES = {
    "agx-orin": "Orin",
    "agx-thor": "Thor",
    "agx-orin-2": "Orin 2",
    "orin-nx": "Orin NX",
}
NODE_ORDER = ["agx-orin", "agx-thor", "agx-orin-2", "orin-nx"]
SCENARIO_NAMES = {
    "disjoint": "两组独立流",
    "incast": "三对一 Incast",
    "fanout": "一对三 Fanout",
    "ring": "四节点 Ring",
}
COLLECTIVE_NAMES = {
    "broadcast": "Broadcast",
    "allgather": "Allgather",
    "ring_allreduce": "Ring Allreduce",
}


def load_json(path: Path) -> dict:
    with path.open(encoding="utf-8") as file:
        return json.load(file)


def received_mbps(path: Path) -> float:
    result = load_json(path)
    return result["end"]["sum_received"]["bits_per_second"] / 1e6


def configure_style() -> None:
    regular_font = "/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc"
    bold_font = "/usr/share/fonts/opentype/noto/NotoSansCJK-Bold.ttc"
    font_manager.fontManager.addfont(regular_font)
    font_manager.fontManager.addfont(bold_font)
    plt.rcParams.update(
        {
            "font.family": "sans-serif",
            # Matplotlib exposes this TTC's first face as JP; it still contains
            # the Chinese glyphs used by these figures.
            "font.sans-serif": ["Noto Sans CJK JP", "DejaVu Sans"],
            "axes.spines.top": False,
            "axes.spines.right": False,
            "axes.titleweight": "bold",
            "axes.grid": True,
            "axes.axisbelow": True,
            "grid.alpha": 0.22,
            "figure.dpi": 140,
            "savefig.dpi": 180,
            "savefig.bbox": "tight",
        }
    )


def save_figure(figure: plt.Figure, output_dir: Path, name: str) -> None:
    figure.savefig(output_dir / f"{name}.svg")
    figure.savefig(output_dir / f"{name}.png")
    plt.close(figure)


def plot_pairwise(pairwise_dir: Path, output_dir: Path) -> None:
    samples: dict[tuple[str, str], list[float]] = defaultdict(list)
    pattern = re.compile(r"(.+)_to_(.+)_trial\d+\.json$")
    for path in pairwise_dir.glob("*.json"):
        match = pattern.fullmatch(path.name)
        if match is None:
            raise ValueError(f"Unexpected pairwise filename: {path.name}")
        samples[match.group(1), match.group(2)].append(received_mbps(path))

    matrix = np.full((len(NODE_ORDER), len(NODE_ORDER)), np.nan)
    for row, source in enumerate(NODE_ORDER):
        for column, destination in enumerate(NODE_ORDER):
            if source != destination:
                matrix[row, column] = np.median(samples[source, destination])

    figure, axis = plt.subplots(figsize=(8.2, 6.4))
    image = axis.imshow(matrix, cmap="YlGnBu", vmin=120, vmax=180)
    labels = [NODE_NAMES[node] for node in NODE_ORDER]
    axis.set_xticks(range(4), labels)
    axis.set_yticks(range(4), labels)
    axis.set_xlabel("接收端")
    axis.set_ylabel("发送端")
    axis.set_title("单 TCP 流方向吞吐矩阵（3 次试验中位数）")
    axis.grid(False)
    for row in range(4):
        for column in range(4):
            if row == column:
                axis.text(column, row, "—", ha="center", va="center", color="#666")
            else:
                value = matrix[row, column]
                color = "white" if value > 162 else "#17202a"
                axis.text(column, row, f"{value:.1f}", ha="center", va="center", color=color)
    colorbar = figure.colorbar(image, ax=axis, shrink=0.82)
    colorbar.set_label("接收吞吐 (Mbit/s)")
    figure.text(
        0.5,
        0.015,
        "同一对设备的两个方向并不等价：Orin 2 → Orin NX 为 175.9，反向仅 130.8 Mbit/s（1.34×）。",
        ha="center",
        color="#8b2e2e",
    )
    figure.tight_layout(rect=(0, 0.04, 1, 1))
    save_figure(figure, output_dir, "01_pairwise_throughput")


def jain_fairness(values: list[float]) -> float:
    array = np.asarray(values)
    return float(array.sum() ** 2 / (len(array) * np.square(array).sum()))


def plot_contention(contention_dir: Path, output_dir: Path) -> None:
    trials: dict[str, dict[int, dict[int, float]]] = defaultdict(lambda: defaultdict(dict))
    pattern = re.compile(r"(.+)_flow(\d+)_trial(\d+)\.json$")
    for path in contention_dir.glob("*.json"):
        match = pattern.fullmatch(path.name)
        if match is None:
            raise ValueError(f"Unexpected contention filename: {path.name}")
        scenario, flow, trial = match.group(1), int(match.group(2)), int(match.group(3))
        trials[scenario][trial][flow] = received_mbps(path)

    figure, axes = plt.subplots(2, 2, figsize=(12, 8), sharey=True)
    colors = ["#1769aa", "#00a6a6", "#ef8354", "#7d5fff"]
    for axis, scenario in zip(axes.flat, SCENARIO_NAMES, strict=True):
        scenario_trials = trials[scenario]
        trial_numbers = sorted(scenario_trials)
        bottoms = np.zeros(len(trial_numbers))
        max_flows = max(len(flows) for flows in scenario_trials.values())
        for flow in range(max_flows):
            values = [scenario_trials[trial][flow] for trial in trial_numbers]
            axis.bar(trial_numbers, values, bottom=bottoms, color=colors[flow], label=f"流 {flow + 1}")
            bottoms += values
        for x, total, trial in zip(trial_numbers, bottoms, trial_numbers, strict=True):
            values = list(scenario_trials[trial].values())
            axis.text(x, total + 5, f"Σ {total:.0f}\nJ={jain_fairness(values):.2f}", ha="center", fontsize=9)
        axis.set_title(SCENARIO_NAMES[scenario])
        axis.set_xticks(trial_numbers, [f"试验 {trial}" for trial in trial_numbers])
        axis.set_ylim(0, 280)
    axes[0, 0].set_ylabel("接收吞吐 (Mbit/s)")
    axes[1, 0].set_ylabel("接收吞吐 (Mbit/s)")
    figure.suptitle("多流竞争：总吞吐不能代表单流公平性", fontsize=15, fontweight="bold")
    figure.legend(
        handles=[Patch(color=color, label=f"流 {index + 1}") for index, color in enumerate(colors)],
        frameon=False,
        ncol=4,
        loc="upper center",
        bbox_to_anchor=(0.5, 0.95),
    )
    figure.text(
        0.5,
        0.01,
        "柱高是总吞吐，色块是各流份额；J 为 Jain 公平性（1.0 最公平）。Fanout 最不稳定，最低 J=0.58。",
        ha="center",
        color="#8b2e2e",
    )
    figure.tight_layout(rect=(0, 0.04, 1, 0.91))
    save_figure(figure, output_dir, "02_contention_fairness")


def parse_ping_samples(path: Path) -> list[float]:
    pattern = re.compile(r"time=([0-9.]+) ms")
    return [float(match.group(1)) for match in pattern.finditer(path.read_text(encoding="utf-8"))]


def plot_latency_under_load(latency_dir: Path, output_dir: Path) -> None:
    load_levels = [25, 75, 100]
    percentiles = {"p50": [], "p99": [], "max": []}
    aggregate_throughput = []
    flow_throughputs: list[list[float]] = []
    for load in load_levels:
        ping_samples = []
        for path in latency_dir.glob(f"load{load}_ping_*.txt"):
            ping_samples.extend(parse_ping_samples(path))
        percentiles["p50"].append(float(np.percentile(ping_samples, 50)))
        percentiles["p99"].append(float(np.percentile(ping_samples, 99)))
        percentiles["max"].append(max(ping_samples))
        flows = [received_mbps(path) for path in sorted(latency_dir.glob(f"load{load}_flow*.json"))]
        flow_throughputs.append(flows)
        aggregate_throughput.append(sum(flows))

    figure, (latency_axis, throughput_axis) = plt.subplots(1, 2, figsize=(12, 4.8))
    for label, color, marker in [("p50", "#1769aa", "o"), ("p99", "#ef8354", "s"), ("max", "#922b21", "^")]:
        latency_axis.plot(load_levels, percentiles[label], marker=marker, linewidth=2.3, label=label, color=color)
        for x, value in zip(load_levels, percentiles[label], strict=True):
            latency_axis.annotate(f"{value:.0f}", (x, value), xytext=(0, 7), textcoords="offset points", ha="center")
    latency_axis.set_yscale("log")
    latency_axis.set_xticks(load_levels, [f"{load}%" for load in load_levels])
    latency_axis.set_xlabel("每条 Ring 流的设定负载")
    latency_axis.set_ylabel("ICMP RTT (ms, 对数轴)")
    latency_axis.set_title("负载越高，长尾呈非线性放大")
    latency_axis.legend(frameon=False)

    x = np.arange(len(load_levels))
    bottoms = np.zeros(len(load_levels))
    colors = ["#1769aa", "#00a6a6", "#ef8354", "#7d5fff"]
    for flow in range(4):
        values = [flows[flow] for flows in flow_throughputs]
        throughput_axis.bar(x, values, bottom=bottoms, color=colors[flow], label=f"流 {flow + 1}")
        bottoms += values
    throughput_axis.set_xticks(x, [f"{load}%" for load in load_levels])
    throughput_axis.set_xlabel("每条 Ring 流的设定负载")
    throughput_axis.set_ylabel("聚合接收吞吐 (Mbit/s)")
    throughput_axis.set_title("吞吐增加仍在继续，但延迟代价陡增")
    throughput_axis.legend(frameon=False, ncol=2)
    for index, total in enumerate(aggregate_throughput):
        throughput_axis.text(index, total + 4, f"Σ {total:.0f}", ha="center")

    figure.suptitle("四节点 WiFi Ring：吞吐—延迟权衡", fontsize=15, fontweight="bold")
    figure.tight_layout(rect=(0, 0, 1, 0.94))
    save_figure(figure, output_dir, "03_latency_under_load")


def plot_collectives(collective_path: Path, output_dir: Path) -> None:
    summary = load_json(collective_path)["collective_summary"]
    payloads = [1024, 65536, 1048576]
    payload_labels = ["1 KiB", "64 KiB", "1 MiB"]
    barrier = summary["barrier"]["0"]["collective_completion_ms"]
    figure, axes = plt.subplots(1, 3, figsize=(13, 4.8), sharey=True)
    colors = ["#1769aa", "#ef8354"]
    for axis, payload, payload_label in zip(axes, payloads, payload_labels, strict=True):
        x = np.arange(3)
        width = 0.34
        for offset, percentile in enumerate(("p50", "p99")):
            values = [summary[name][str(payload)]["collective_completion_ms"][percentile] for name in COLLECTIVE_NAMES]
            axis.bar(x + (offset - 0.5) * width, values, width, label=percentile, color=colors[offset])
            for x_value, value in zip(x + (offset - 0.5) * width, values, strict=True):
                axis.text(x_value, value * 1.10, f"{value:.0f}", ha="center", fontsize=8, rotation=90)
        axis.axhline(barrier["p50"], color="#555", linestyle="--", linewidth=1, label="Barrier p50" if payload == 1024 else None)
        axis.axhline(barrier["p99"], color="#922b21", linestyle=":", linewidth=1.3, label="Barrier p99" if payload == 1024 else None)
        axis.set_yscale("log")
        axis.set_xticks(x, COLLECTIVE_NAMES.values(), rotation=18)
        axis.set_title(payload_label)
    axes[0].set_ylabel("完成延迟 (ms, 对数轴)")
    axes[0].legend(frameon=False, fontsize=8)
    figure.suptitle("四个物理节点上的 Collective 完成延迟（每项 n=20）", fontsize=15, fontweight="bold")
    figure.text(
        0.5,
        0.005,
        "Barrier p50=8.5 ms、p99=193.4 ms。n=20 时 p99 近似最大值，只能用于发现风险，不能作为稳定 SLA。",
        ha="center",
        color="#8b2e2e",
    )
    figure.tight_layout(rect=(0, 0.06, 1, 0.94))
    save_figure(figure, output_dir, "04_collective_latency")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--pairwise-dir", type=Path, required=True)
    parser.add_argument("--contention-dir", type=Path, required=True)
    parser.add_argument("--latency-dir", type=Path, required=True)
    parser.add_argument("--collective-report", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, default=Path(__file__).parent)
    args = parser.parse_args()

    args.output_dir.mkdir(parents=True, exist_ok=True)
    configure_style()
    plot_pairwise(args.pairwise_dir, args.output_dir)
    plot_contention(args.contention_dir, args.output_dir)
    plot_latency_under_load(args.latency_dir, args.output_dir)
    plot_collectives(args.collective_report, args.output_dir)


if __name__ == "__main__":
    main()
