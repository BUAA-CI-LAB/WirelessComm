from __future__ import annotations

import asyncio
import dataclasses
import socket
import unittest

from wireless_comm import (
    BackpressureError,
    Comm,
    CommConfig,
    CommOptions,
    OperationTimeoutError,
    Peer,
)

try:
    import torch
except ImportError:
    torch = None


def unused_local_port() -> int:
    with socket.socket() as listener:
        listener.bind(("127.0.0.1", 0))
        return listener.getsockname()[1]


@dataclasses.dataclass
class Record:
    sequence: int
    label: str


class CommIntegrationTests(unittest.IsolatedAsyncioTestCase):
    async def asyncSetUp(self) -> None:
        self.peer_a = Peer("node-a", "127.0.0.1", unused_local_port())
        self.peer_b = Peer("node-b", "127.0.0.1", unused_local_port())
        self.a = await Comm.create(local=self.peer_a, peers=[self.peer_b])
        self.b = await Comm.create(local=self.peer_b, peers=[self.peer_a])

    async def asyncTearDown(self) -> None:
        await self.a.close()
        await self.b.close()

    async def test_peer_directory_is_initialized_once(self) -> None:
        self.assertIs(self.a.peer("node-b"), self.peer_b)
        self.assertEqual(self.a.peers(), (self.peer_b,))
        with self.assertRaises(KeyError):
            self.a.peer("unknown")

    async def test_send_recv_with_optional_piggypayload(self) -> None:
        result = await self.a.send(
            {"values": [1, 2, 3]},
            self.a.peer("node-b"),
            piggypayload={"trace_id": "run-1"},
            options=CommOptions(tag=4),
        )
        object, metadata = await self.b.recv(
            self.b.peer("node-a"),
            CommOptions(tag=4, timeout=1),
        )
        self.assertGreater(result.wire_bytes, 0)
        self.assertEqual(object, {"values": [1, 2, 3]})
        self.assertEqual(metadata, {"trace_id": "run-1"})

    async def test_connection_is_reused_in_both_directions(self) -> None:
        await self.a.send("a-to-b", self.a.peer("node-b"))
        self.assertEqual(
            await self.b.recv(self.b.peer("node-a"), CommOptions(timeout=1)),
            ("a-to-b", None),
        )
        await self.b.send("b-to-a", self.b.peer("node-a"))
        self.assertEqual(
            await self.a.recv(self.a.peer("node-b"), CommOptions(timeout=1)),
            ("b-to-a", None),
        )
        self.assertEqual(len(self.a._states["node-b"].connections), 1)
        self.assertEqual(len(self.b._states["node-a"].connections), 1)

    async def test_tag_filter_preserves_other_messages(self) -> None:
        await self.a.send("first", self.a.peer("node-b"), options=CommOptions(tag=1))
        await self.a.send("second", self.a.peer("node-b"), options=CommOptions(tag=2))
        second = await self.b.recv(self.b.peer("node-a"), CommOptions(tag=2, timeout=1))
        first = await self.b.recv(self.b.peer("node-a"), CommOptions(tag=1, timeout=1))
        self.assertEqual(second[0], "second")
        self.assertEqual(first[0], "first")

    async def test_registered_dataclass(self) -> None:
        for runtime in (self.a, self.b):
            runtime.register_dataclass(Record, type_id="tests.Record")
        value = Record(9, "message")
        await self.a.send(value, self.a.peer("node-b"))
        received, _ = await self.b.recv(self.b.peer("node-a"), CommOptions(timeout=1))
        self.assertEqual(received, value)

    async def test_simultaneous_first_send(self) -> None:
        await asyncio.gather(
            self.a.send("from-a", self.a.peer("node-b")),
            self.b.send("from-b", self.b.peer("node-a")),
        )
        from_a, from_b = await asyncio.gather(
            self.b.recv(self.b.peer("node-a"), CommOptions(timeout=1)),
            self.a.recv(self.a.peer("node-b"), CommOptions(timeout=1)),
        )
        self.assertEqual(from_a, ("from-a", None))
        self.assertEqual(from_b, ("from-b", None))

    async def test_recv_timeout(self) -> None:
        with self.assertRaises(OperationTimeoutError):
            await self.b.recv(
                self.b.peer("node-a"),
                CommOptions(timeout=0.01),
            )

    @unittest.skipIf(torch is None, "PyTorch is not installed")
    async def test_tensor_dict_over_tcp(self) -> None:
        payload = {
            "matrix": torch.arange(12, dtype=torch.float32).reshape(3, 4),
            "indices": [torch.tensor([2, 4, 6], dtype=torch.int64)],
        }
        await self.a.send(
            payload,
            self.a.peer("node-b"),
            piggypayload={"kind": "tensor-dict"},
        )
        received, metadata = await self.b.recv(
            self.b.peer("node-a"),
            CommOptions(timeout=1),
        )
        self.assertTrue(torch.equal(received["matrix"], payload["matrix"]))
        self.assertTrue(torch.equal(received["indices"][0], payload["indices"][0]))
        self.assertEqual(metadata, {"kind": "tensor-dict"})


class BackpressureIntegrationTests(unittest.IsolatedAsyncioTestCase):
    async def asyncSetUp(self) -> None:
        self.peer_a = Peer("queue-a", "127.0.0.1", unused_local_port())
        self.peer_b = Peer("queue-b", "127.0.0.1", unused_local_port())
        config = CommConfig(
            max_peer_queued_messages=1,
            max_peer_queued_bytes=1024 * 1024,
        )
        self.a = await Comm.create(
            local=self.peer_a,
            peers=[self.peer_b],
            config=config,
        )
        self.b = await Comm.create(
            local=self.peer_b,
            peers=[self.peer_a],
            config=config,
        )

    async def asyncTearDown(self) -> None:
        await self.a.close()
        await self.b.close()

    async def test_fail_fast_and_wait_for_capacity(self) -> None:
        gate = asyncio.Event()
        original_connection_for = self.a._connection_for

        async def blocked_connection_for(peer: Peer):
            await gate.wait()
            return await original_connection_for(peer)

        self.a._connection_for = blocked_connection_for
        first = asyncio.create_task(self.a.send("first", self.a.peer("queue-b")))
        await self._wait_for_pending_messages(1)

        with self.assertRaises(BackpressureError):
            await self.a.send(
                "rejected",
                self.a.peer("queue-b"),
                options=CommOptions(wait_for_capacity=False),
            )

        waiting = asyncio.create_task(self.a.send("second", self.a.peer("queue-b")))
        await asyncio.sleep(0)
        self.assertFalse(waiting.done())
        gate.set()
        await asyncio.gather(first, waiting)
        self.assertEqual(
            await self.b.recv(self.b.peer("queue-a"), CommOptions(timeout=1)),
            ("first", None),
        )
        self.assertEqual(
            await self.b.recv(self.b.peer("queue-a"), CommOptions(timeout=1)),
            ("second", None),
        )

    async def test_message_larger_than_queue_limit_is_rejected(self) -> None:
        with self.assertRaises(BackpressureError):
            await self.a.send(b"x" * (1024 * 1024), self.a.peer("queue-b"))

    async def test_timeout_cancels_active_send_and_releases_capacity(self) -> None:
        gate = asyncio.Event()

        async def blocked_connection_for(peer: Peer):
            await gate.wait()
            raise AssertionError(f"unexpected connection for {peer.node_id}")

        self.a._connection_for = blocked_connection_for
        with self.assertRaises(OperationTimeoutError):
            await self.a.send(
                "times-out",
                self.a.peer("queue-b"),
                options=CommOptions(timeout=0.01),
            )
        state = self.a._states["queue-b"]
        self.assertEqual(state.pending_messages, 0)
        self.assertEqual(state.pending_bytes, 0)

    async def _wait_for_pending_messages(self, expected: int) -> None:
        for _ in range(100):
            if self.a._states["queue-b"].pending_messages == expected:
                return
            await asyncio.sleep(0)
        self.fail(f"send queue did not reach {expected} pending messages")
