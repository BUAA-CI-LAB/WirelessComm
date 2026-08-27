"""Run controlled four-node scheduler and collective topology experiments."""

from __future__ import annotations

import argparse
import asyncio
import copy
import itertools
import json
import random
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml

from .cluster_benchmark import launch_cluster


@dataclass(frozen=True, slots=True)
class ExperimentVariant:
    name: str
    suite: str
    collective: str
    payload_bytes: int
    settings: dict[str, int]
    ring: tuple[int, ...] = (0, 1, 2, 3)
    comm: dict[str, int] | None = None


def build_variants(
    suites: tuple[str, ...],
    *,
    rounds: int,
    warmup_rounds: int,
) -> list[ExperimentVariant]:
    variants = []
    if "scheduler" in suites:
        for concurrency in (4, 2, 1):
            variants.append(
                ExperimentVariant(
                    f"scheduler-c{concurrency}",
                    "scheduler",
                    "ring_exchange",
                    4 * 1024 * 1024,
                    {
                        "rounds": rounds,
                        "warmup_rounds": warmup_rounds,
                        "ring_exchange_concurrency": concurrency,
                    },
                )
            )
    if "ring" in suites:
        for tail in itertools.permutations((1, 2, 3)):
            ring = (0, *tail)
            variants.append(
                ExperimentVariant(
                    "ring-" + "-".join(str(rank) for rank in ring),
                    "ring",
                    "ring_allreduce",
                    1024 * 1024,
                    {"rounds": rounds, "warmup_rounds": warmup_rounds},
                    ring,
                )
            )
    if "broadcast" in suites:
        for root in range(4):
            variants.append(
                ExperimentVariant(
                    f"broadcast-root{root}",
                    "broadcast",
                    "broadcast",
                    1024 * 1024,
                    {
                        "rounds": rounds,
                        "warmup_rounds": warmup_rounds,
                        "broadcast_root": root,
                    },
                )
            )
    if "chunk" in suites:
        for block_bytes in (64 * 1024, 128 * 1024, 256 * 1024):
            variants.append(
                ExperimentVariant(
                    f"allreduce-block{block_bytes // 1024}k",
                    "chunk",
                    "ring_allreduce",
                    1024 * 1024,
                    {
                        "rounds": rounds,
                        "warmup_rounds": warmup_rounds,
                        "allreduce_block_bytes": block_bytes,
                    },
                )
            )
    if "p2p" in suites:
        variants.append(
            ExperimentVariant(
                "p2p-uncontrolled",
                "p2p",
                "all_to_all_exchange",
                1024 * 1024,
                {"rounds": rounds, "warmup_rounds": warmup_rounds},
                comm={},
            )
        )
        for quantum_kib in (16, 64, 256):
            variants.append(
                ExperimentVariant(
                    f"p2p-byte-rr{quantum_kib}k",
                    "p2p",
                    "all_to_all_exchange",
                    1024 * 1024,
                    {"rounds": rounds, "warmup_rounds": warmup_rounds},
                    comm={"egress_quantum_bytes": quantum_kib * 1024},
                )
            )
        for rate_mbps in (15, 20, 25, 30, 35):
            variants.append(
                ExperimentVariant(
                    f"p2p-byte-rr64k-rate{rate_mbps}",
                    "p2p",
                    "all_to_all_exchange",
                    1024 * 1024,
                    {"rounds": rounds, "warmup_rounds": warmup_rounds},
                    comm={
                        "egress_quantum_bytes": 64 * 1024,
                        "egress_rate_bytes_per_second": int(rate_mbps * 1e6 / 8),
                    },
                )
            )
        for quantum_kib in (16, 256):
            variants.append(
                ExperimentVariant(
                    f"p2p-byte-rr{quantum_kib}k-rate30",
                    "p2p",
                    "all_to_all_exchange",
                    1024 * 1024,
                    {"rounds": rounds, "warmup_rounds": warmup_rounds},
                    comm={
                        "egress_quantum_bytes": quantum_kib * 1024,
                        "egress_rate_bytes_per_second": 3_750_000,
                    },
                )
            )
    if "p2p-knee" in suites:
        for rate_mbps in (40, 45, 50):
            variants.append(
                ExperimentVariant(
                    f"p2p-byte-rr64k-rate{rate_mbps}",
                    "p2p-knee",
                    "all_to_all_exchange",
                    1024 * 1024,
                    {"rounds": rounds, "warmup_rounds": warmup_rounds},
                    comm={
                        "egress_quantum_bytes": 64 * 1024,
                        "egress_rate_bytes_per_second": int(rate_mbps * 1e6 / 8),
                    },
                )
            )
        for quantum_kib in (16, 256):
            variants.append(
                ExperimentVariant(
                    f"p2p-byte-rr{quantum_kib}k-rate35",
                    "p2p-knee",
                    "all_to_all_exchange",
                    1024 * 1024,
                    {"rounds": rounds, "warmup_rounds": warmup_rounds},
                    comm={
                        "egress_quantum_bytes": quantum_kib * 1024,
                        "egress_rate_bytes_per_second": 4_375_000,
                    },
                )
            )
    if "p2p-confirm" in suites:
        variants.append(
            ExperimentVariant(
                "p2p-uncontrolled",
                "p2p-confirm",
                "all_to_all_exchange",
                1024 * 1024,
                {"rounds": rounds, "warmup_rounds": warmup_rounds},
                comm={},
            )
        )
        for rate_mbps in (35, 40, 45, 50):
            variants.append(
                ExperimentVariant(
                    f"p2p-byte-rr64k-rate{rate_mbps}",
                    "p2p-confirm",
                    "all_to_all_exchange",
                    1024 * 1024,
                    {"rounds": rounds, "warmup_rounds": warmup_rounds},
                    comm={
                        "egress_quantum_bytes": 64 * 1024,
                        "egress_rate_bytes_per_second": int(rate_mbps * 1e6 / 8),
                    },
                )
            )
        variants.append(
            ExperimentVariant(
                "p2p-byte-rr256k-rate40",
                "p2p-confirm",
                "all_to_all_exchange",
                1024 * 1024,
                {"rounds": rounds, "warmup_rounds": warmup_rounds},
                comm={
                    "egress_quantum_bytes": 256 * 1024,
                    "egress_rate_bytes_per_second": 5_000_000,
                },
            )
        )
    known_suites = {
        "scheduler",
        "ring",
        "broadcast",
        "chunk",
        "p2p",
        "p2p-knee",
        "p2p-confirm",
    }
    unknown = set(suites) - known_suites
    if unknown:
        raise ValueError(f"unknown experiment suites: {', '.join(sorted(unknown))}")
    return variants


def variant_document(
    base: dict[str, Any],
    variant: ExperimentVariant,
    *,
    base_port: int,
) -> dict[str, Any]:
    document = copy.deepcopy(base)
    document["base_port"] = base_port
    document["ring"] = list(variant.ring)
    document["comm"] = variant.comm if variant.comm is not None else {}
    document["benchmark"] = {
        "payload_sizes": [variant.payload_bytes],
        "rounds": variant.settings["rounds"],
        "warmup_rounds": variant.settings["warmup_rounds"],
        "collectives": [variant.collective],
        "timeout": 180,
        **{
            key: value
            for key, value in variant.settings.items()
            if key not in {"rounds", "warmup_rounds"}
        },
    }
    return document


def summarize_variant(
    variant: ExperimentVariant,
    report: dict[str, Any],
    report_path: Path,
) -> dict[str, Any]:
    metrics = report["collective_summary"][variant.collective][
        str(variant.payload_bytes)
    ]
    summary = {
        "name": variant.name,
        "suite": variant.suite,
        "collective": variant.collective,
        "payload_bytes": variant.payload_bytes,
        "ring": list(variant.ring),
        "settings": variant.settings,
        "report": str(report_path),
        "completion_ms": metrics["collective_completion_ms"],
    }
    if variant.collective in {"ring_exchange", "all_to_all_exchange"}:
        summary["jain_fairness"] = metrics["jain_fairness"]
        summary["effective_round_mbps"] = metrics["effective_round_mbps"]
    return summary


async def run_sweep(
    base_config: Path,
    output_dir: Path,
    *,
    suites: tuple[str, ...],
    rounds: int,
    warmup_rounds: int,
    seed: int,
    first_port: int,
    launch_timeout: float,
    resume: bool,
) -> dict[str, Any]:
    with base_config.open(encoding="utf-8") as stream:
        base = yaml.safe_load(stream)
    if not isinstance(base, dict):
        raise TypeError("base cluster configuration must be a mapping")

    output_dir.mkdir(parents=True, exist_ok=True)
    config_dir = output_dir / "configs"
    report_dir = output_dir / "reports"
    config_dir.mkdir(exist_ok=True)
    report_dir.mkdir(exist_ok=True)
    variants = build_variants(
        suites,
        rounds=rounds,
        warmup_rounds=warmup_rounds,
    )
    random.Random(seed).shuffle(variants)
    summaries = []
    for index, variant in enumerate(variants):
        config_path = config_dir / f"{variant.name}.yaml"
        report_path = report_dir / f"{variant.name}.json"
        document = variant_document(
            base,
            variant,
            base_port=first_port + index * 8,
        )
        config_path.write_text(
            yaml.safe_dump(document, sort_keys=False), encoding="utf-8"
        )
        if resume and report_path.exists():
            report = json.loads(report_path.read_text(encoding="utf-8"))
        else:
            report = await launch_cluster(
                config_path,
                output=report_path,
                timeout=launch_timeout,
                dry_run=False,
            )
        summaries.append(summarize_variant(variant, report, report_path))
        manifest = {
            "seed": seed,
            "rounds": rounds,
            "warmup_rounds": warmup_rounds,
            "completed": summaries,
            "remaining": [item.name for item in variants[index + 1 :]],
        }
        (output_dir / "summary.json").write_text(
            json.dumps(manifest, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
    return manifest


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument(
        "--suites", default="scheduler,ring,broadcast,chunk"
    )
    parser.add_argument("--rounds", type=int, default=30)
    parser.add_argument("--warmup-rounds", type=int, default=3)
    parser.add_argument("--seed", type=int, default=20260827)
    parser.add_argument("--first-port", type=int, default=9800)
    parser.add_argument("--launch-timeout", type=float, default=1800)
    parser.add_argument("--resume", action="store_true")
    return parser


def main() -> None:
    args = build_parser().parse_args()
    report = asyncio.run(
        run_sweep(
            args.config,
            args.output_dir,
            suites=tuple(args.suites.split(",")),
            rounds=args.rounds,
            warmup_rounds=args.warmup_rounds,
            seed=args.seed,
            first_port=args.first_port,
            launch_timeout=args.launch_timeout,
            resume=args.resume,
        )
    )
    print(json.dumps(report, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
