from __future__ import annotations

from wireless_comm.experiment_sweep import build_variants, variant_document


def test_builds_every_four_node_variant() -> None:
    variants = build_variants(
        ("scheduler", "ring", "broadcast", "chunk"),
        rounds=30,
        warmup_rounds=3,
    )
    assert len(variants) == 16
    assert len({variant.name for variant in variants}) == 16
    assert sum(variant.suite == "ring" for variant in variants) == 6
    assert sum(variant.suite == "broadcast" for variant in variants) == 4


def test_variant_document_replaces_benchmark_settings() -> None:
    variant = build_variants(
        ("scheduler",), rounds=10, warmup_rounds=2
    )[0]
    document = variant_document(
        {"base_port": 9000, "ring": [0, 3, 2, 1], "benchmark": {"old": True}},
        variant,
        base_port=9800,
    )
    assert document["base_port"] == 9800
    assert document["ring"] == [0, 1, 2, 3]
    assert document["benchmark"] == {
        "payload_sizes": [4 * 1024 * 1024],
        "rounds": 10,
        "warmup_rounds": 2,
        "collectives": ["ring_exchange"],
        "timeout": 180,
        "ring_exchange_concurrency": 4,
    }


def test_builds_p2p_scheduler_and_pacing_variants() -> None:
    variants = build_variants(("p2p",), rounds=10, warmup_rounds=2)
    assert [variant.name for variant in variants[:4]] == [
        "p2p-uncontrolled",
        "p2p-byte-rr16k",
        "p2p-byte-rr64k",
        "p2p-byte-rr256k",
    ]
    assert len(variants) == 11
    assert variants[-1].comm == {
        "egress_quantum_bytes": 256 * 1024,
        "egress_rate_bytes_per_second": 3_750_000,
    }


def test_builds_p2p_knee_variants() -> None:
    variants = {
        variant.name: variant
        for variant in build_variants(
            ("p2p-knee",), rounds=5, warmup_rounds=1
        )
    }
    assert set(variants) == {
        "p2p-byte-rr64k-rate40",
        "p2p-byte-rr64k-rate45",
        "p2p-byte-rr64k-rate50",
        "p2p-byte-rr16k-rate35",
        "p2p-byte-rr256k-rate35",
    }
    assert variants["p2p-byte-rr256k-rate35"].comm == {
        "egress_quantum_bytes": 256 * 1024,
        "egress_rate_bytes_per_second": 4_375_000,
    }


def test_builds_p2p_confirmation_block() -> None:
    variants = build_variants(
        ("p2p-confirm",), rounds=30, warmup_rounds=3
    )

    assert [variant.name for variant in variants] == [
        "p2p-uncontrolled",
        "p2p-byte-rr64k-rate35",
        "p2p-byte-rr64k-rate40",
        "p2p-byte-rr64k-rate45",
        "p2p-byte-rr64k-rate50",
        "p2p-byte-rr256k-rate40",
    ]
    assert variants[-1].comm == {
        "egress_quantum_bytes": 256 * 1024,
        "egress_rate_bytes_per_second": 5_000_000,
    }
