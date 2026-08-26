from __future__ import annotations

import asyncio
import multiprocessing
import socket
import unittest
from queue import Empty

from wireless_comm import Comm, CommOptions, Peer


def unused_local_port() -> int:
    with socket.socket() as listener:
        listener.bind(("127.0.0.1", 0))
        return listener.getsockname()[1]


def receiver_process(local: Peer, sender: Peer, results: object) -> None:
    async def receive() -> None:
        comm = await Comm.create(local=local, peers=[sender])
        results.put(("ready", None))
        try:
            object, metadata = await comm.recv(
                comm.peer(sender.node_id),
                CommOptions(tag=12, timeout=5),
            )
            results.put((object, metadata))
        finally:
            await comm.close()

    asyncio.run(receive())


class MultiprocessIntegrationTests(unittest.TestCase):
    def test_payload_crosses_process_boundary(self) -> None:
        sender = Peer("sender", "127.0.0.1", unused_local_port())
        receiver = Peer("receiver", "127.0.0.1", unused_local_port())
        context = multiprocessing.get_context("spawn")
        results = context.Queue()
        process = context.Process(
            target=receiver_process,
            args=(receiver, sender, results),
        )
        process.start()
        try:
            self.assertEqual(results.get(timeout=5), ("ready", None))

            async def send() -> None:
                comm = await Comm.create(local=sender, peers=[receiver])
                try:
                    await comm.send(
                        {"values": [1, 2, 3], "raw": b"payload"},
                        comm.peer(receiver.node_id),
                        piggypayload={"source": "multiprocess"},
                        options=CommOptions(tag=12, timeout=5),
                    )
                finally:
                    await comm.close()

            asyncio.run(send())
            self.assertEqual(
                results.get(timeout=5),
                (
                    {"values": [1, 2, 3], "raw": b"payload"},
                    {"source": "multiprocess"},
                ),
            )
        except Empty as exc:
            self.fail(f"child process did not respond: {exc}")
        finally:
            process.join(timeout=5)
            if process.is_alive():
                process.terminate()
                process.join()
        self.assertEqual(process.exitcode, 0)
