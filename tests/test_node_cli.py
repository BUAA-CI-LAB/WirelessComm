from __future__ import annotations

import os
import socket
import subprocess
import sys
import tempfile
import time
import unittest
from pathlib import Path


def unused_local_port() -> int:
    with socket.socket() as listener:
        listener.bind(("127.0.0.1", 0))
        return listener.getsockname()[1]


class NodeCliIntegrationTests(unittest.TestCase):
    def test_json_echo_across_cli_processes(self) -> None:
        port_a = unused_local_port()
        port_b = unused_local_port()
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            config_a = root / "node-a.yaml"
            config_b = root / "node-b.yaml"
            config_a.write_text(
                self._config("node-a", port_a, "node-b", port_b),
                encoding="utf-8",
            )
            config_b.write_text(
                self._config("node-b", port_b, "node-a", port_a),
                encoding="utf-8",
            )
            environment = os.environ.copy()
            echo = subprocess.Popen(
                [
                    sys.executable,
                    "-m",
                    "wireless_comm.node",
                    "--config",
                    str(config_b),
                    "echo",
                    "--src",
                    "node-a",
                    "--tag",
                    "7",
                    "--count",
                    "1",
                ],
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                text=True,
                env=environment,
            )
            try:
                time.sleep(0.3)
                sender = subprocess.run(
                    [
                        sys.executable,
                        "-m",
                        "wireless_comm.node",
                        "--config",
                        str(config_a),
                        "send",
                        "--dst",
                        "node-b",
                        "--tag",
                        "7",
                        "--json",
                        '{"message":"hello"}',
                        "--metadata",
                        '{"trace_id":"cli-test"}',
                        "--expect-reply",
                    ],
                    check=True,
                    capture_output=True,
                    text=True,
                    timeout=5,
                    env=environment,
                )
                echo_output, _ = echo.communicate(timeout=5)
            finally:
                if echo.poll() is None:
                    echo.terminate()
                    echo.wait(timeout=5)
            self.assertIn("sent message_id=", sender.stdout)
            self.assertIn("'message': 'hello'", sender.stdout)
            self.assertIn("'trace_id': 'cli-test'", echo_output)
            self.assertEqual(echo.returncode, 0)

    @staticmethod
    def _config(local_id: str, local_port: int, peer_id: str, peer_port: int) -> str:
        return f"""
local:
  node_id: {local_id}
  host: 127.0.0.1
  port: {local_port}
peers:
  - node_id: {peer_id}
    host: 127.0.0.1
    port: {peer_port}
"""
