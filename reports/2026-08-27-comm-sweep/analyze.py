"""Analyze repeated Comm scheduler and collective sweep reports."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from matplotlib import font_manager

PAYLOAD = 4 * 1024 * 1024
RINGS = (
    "ring-0-1-2-3",
    "ring-0-1-3-2",
    "ring-0-2-1-3",
    "ring-0-2-3-1",
    "ring-0-3-1-2",
    "ring-0-3-2-1",
)
BROADCASTS = tuple(f"broadcast-root{root}" for root in range(4))
BLOCKS = tuple(f"allreduce-block{size}k" for size in (64, 128, 256))


def load_json(path: Path) -> dict:
    with path.open(encoding="utf-8") as file:
        return json.load(file)


def configure_style() -> None:
    for path in (
        "/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc",
        "/usr/share/fonts/opentype/noto/NotoSansCJK-Bold.ttc",
    ):
        font_manager.fontManager.addfont(path)
    plt.rcParams.update(
        {
            "font.family": "sans-serif",
            "font.sans-serif": ["Noto Sans CJK JP", "DejaVu Sans"],
            "axes.spines.top": False,
            "axes.spines.right": False,
            "axes.grid": True,
            "axes.axisbelow": True,
            "grid.alpha": 0.2,
            "figure.dpi": 140,
            "savefig.dpi": 180,
            "savefig.bbox": "tight",
        }
    )


def percentiles(values: list[float]) -> dict[str, float | int]:
    return {
        "n": len(values),
        "p05": float(np.percentile(values, 5)),
        "p50": float(np.percentile(values, 50)),
        "p95": float(np.percentile(values, 95)),
        "p99": float(np.percentile(values, 99)),
        "max": max(values),
    }


def scheduler_samples(run_dirs: list[Path], concurrency: int) -> dict[str, list[float]]:
    completion = []
    fairness = []
    for run_dir in run_dirs:
        report = load_json(run_dir / "reports" / f"scheduler-c{concurrency}.json")
        metrics = report["collective_summary"]["ring_exchange"][str(PAYLOAD)]
        completion.extend(metrics["completion_samples_ms"])
        per_rank = [
            rank["ring_exchange_send_service_ms"][str(PAYLOAD)]
            for rank in report["rank_results"]
        ]
        for service_times in zip(*per_rank, strict=True):
            rates = [PAYLOAD * 8 / (duration * 1000) for duration in service_times]
            fairness.append(sum(rates) ** 2 / (len(rates) * sum(x * x for x in rates)))
    effective_mbps = [PAYLOAD * 4 * 8 / (duration * 1000) for duration in completion]
    return {
        "completion_ms": completion,
        "effective_mbps": effective_mbps,
        "jain_fairness": fairness,
    }


def topology_samples(run_dirs: list[Path], name: str, collective: str) -> list[float]:
    values = []
    for run_dir in run_dirs:
        report = load_json(run_dir / "reports" / f"{name}.json")
        values.extend(
            report["collective_summary"][collective]["1048576"][
                "completion_samples_ms"
            ]
        )
    return values


def save_figure(figure: plt.Figure, output_dir: Path, name: str) -> None:
    figure.savefig(output_dir / f"{name}.svg")
    figure.savefig(output_dir / f"{name}.png")
    plt.close(figure)


def plot_scheduler(data: dict[int, dict[str, list[float]]], output_dir: Path) -> None:
    figure, axes = plt.subplots(1, 3, figsize=(13, 4.8))
    colors = ["#1769aa", "#00a6a6", "#ef8354"]
    specs = (
        ("completion_ms", "整轮完成时间 (ms)", "越低越好"),
        ("effective_mbps", "有效聚合吞吐 (Mbit/s)", "越高越好"),
        ("jain_fairness", "Jain 公平性", "越接近 1 越公平"),
    )
    labels = ["并发 4\n无限制", "并发 2", "并发 1\n串行"]
    for axis, (metric, title, direction) in zip(axes, specs, strict=True):
        values = [data[concurrency][metric] for concurrency in (4, 2, 1)]
        boxes = axis.boxplot(
            values,
            tick_labels=labels,
            patch_artist=True,
            showfliers=False,
            widths=0.58,
        )
        for box, color in zip(boxes["boxes"], colors, strict=True):
            box.set_facecolor(color)
            box.set_alpha(0.78)
        axis.set_title(f"{title}\n{direction}")
        for index, samples in enumerate(values, start=1):
            axis.text(index, np.percentile(samples, 75), f"p50 {np.median(samples):.2f}", ha="center", fontsize=9)
    figure.suptitle("Comm 全局 bulk 并发限制 A/B（3 个时间块，每种共 n=130）", fontsize=15, fontweight="bold")
    figure.tight_layout(rect=(0, 0, 1, 0.92))
    save_figure(figure, output_dir, "01_scheduler_ab")


def plot_topology(
    data: dict[str, dict[str, list[float]]], output_dir: Path
) -> None:
    figure, axes = plt.subplots(1, 3, figsize=(15, 5.2))
    groups = (
        ("ring", RINGS, "Ring 顺序", lambda name: name.removeprefix("ring-")),
        ("broadcast", BROADCASTS, "Broadcast root", lambda name: name[-1]),
        ("chunk", BLOCKS, "Allreduce block", lambda name: name.split("block")[1]),
    )
    for axis, (group, names, title, labeler) in zip(axes, groups, strict=True):
        x = np.arange(len(names))
        p50 = [np.percentile(data[group][name], 50) for name in names]
        p99 = [np.percentile(data[group][name], 99) for name in names]
        axis.bar(x - 0.18, p50, 0.36, label="p50", color="#1769aa")
        axis.bar(x + 0.18, p99, 0.36, label="p99", color="#ef8354")
        axis.set_xticks(x, [labeler(name) for name in names], rotation=28)
        axis.set_title(title)
        axis.set_ylabel("1 MiB 完成时间 (ms)")
        axis.legend(frameon=False)
    figure.suptitle("Collective 参数 sweep（2 个时间块，每候选 n=80）", fontsize=15, fontweight="bold")
    figure.tight_layout(rect=(0, 0, 1, 0.93))
    save_figure(figure, output_dir, "02_collective_sweep")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--scheduler-run", action="append", type=Path, required=True)
    parser.add_argument("--topology-run", action="append", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, default=Path(__file__).parent)
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    configure_style()

    scheduler = {
        concurrency: scheduler_samples(args.scheduler_run, concurrency)
        for concurrency in (4, 2, 1)
    }
    topology = {
        "ring": {
            name: topology_samples(args.topology_run, name, "ring_allreduce")
            for name in RINGS
        },
        "broadcast": {
            name: topology_samples(args.topology_run, name, "broadcast")
            for name in BROADCASTS
        },
        "chunk": {
            name: topology_samples(args.topology_run, name, "ring_allreduce")
            for name in BLOCKS
        },
    }
    plot_scheduler(scheduler, args.output_dir)
    plot_topology(topology, args.output_dir)
    summary = {
        "scheduler": {
            str(concurrency): {
                metric: percentiles(values) for metric, values in metrics.items()
            }
            for concurrency, metrics in scheduler.items()
        },
        "collective_sweep": {
            group: {name: percentiles(values) for name, values in candidates.items()}
            for group, candidates in topology.items()
        },
    }
    (args.output_dir / "summary.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )


if __name__ == "__main__":
    main()
