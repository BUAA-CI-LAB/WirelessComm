from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from wireless_comm import load_runtime_config


class ConfigTests(unittest.TestCase):
    def test_load_complete_peer_directory(self) -> None:
        config = self._load(
            """
local:
  node_id: node-a
  host: 192.0.2.10
  bind_host: 0.0.0.0
  port: 9000
peers:
  - node_id: node-b
    host: 192.0.2.11
    port: 9001
comm:
  max_peer_queued_messages: 8
  max_peer_queued_bytes: 4096
  egress_quantum_bytes: 1024
  egress_rate_bytes_per_second: 1048576
"""
        )
        self.assertEqual(config.local.node_id, "node-a")
        self.assertEqual(config.bind_host, "0.0.0.0")
        self.assertEqual(config.peers[0].node_id, "node-b")
        self.assertEqual(config.comm.max_peer_queued_messages, 8)
        self.assertEqual(config.comm.egress_quantum_bytes, 1024)
        self.assertEqual(config.comm.egress_rate_bytes_per_second, 1048576)

    def test_pacing_requires_byte_scheduler(self) -> None:
        with self.assertRaisesRegex(ValueError, "requires egress_quantum_bytes"):
            self._load(
                """
local: {node_id: node-a, host: 127.0.0.1, port: 9000}
peers: []
comm:
  egress_rate_bytes_per_second: 1048576
"""
            )

    def test_duplicate_peer_is_rejected(self) -> None:
        with self.assertRaisesRegex(ValueError, "duplicate node_id"):
            self._load(
                """
local: {node_id: node-a, host: 127.0.0.1, port: 9000}
peers:
  - {node_id: node-a, host: 127.0.0.1, port: 9001}
"""
            )

    def test_unknown_config_field_is_rejected(self) -> None:
        with self.assertRaisesRegex(ValueError, "unknown fields"):
            self._load(
                """
local: {node_id: node-a, host: 127.0.0.1, port: 9000}
peers: []
comm:
  imaginary_option: true
"""
            )

    def _load(self, contents: str):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "node.yaml"
            path.write_text(contents, encoding="utf-8")
            return load_runtime_config(path)
