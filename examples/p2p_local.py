"""Run a two-node P2P exchange on localhost."""

import asyncio

from wireless_comm import Comm, CommOptions, Peer


async def main() -> None:
    peer_a = Peer("node-a", "127.0.0.1", 9101)
    peer_b = Peer("node-b", "127.0.0.1", 9102)

    async with (
        await Comm.create(local=peer_a, peers=[peer_b]) as node_a,
        await Comm.create(local=peer_b, peers=[peer_a]) as node_b,
    ):
        await node_a.send(
            {"samples": [1.0, 2.0, 3.0]},
            node_a.peer("node-b"),
            piggypayload={"trace_id": "demo"},
            options=CommOptions(tag=1),
        )
        object, metadata = await node_b.recv(
            node_b.peer("node-a"),
            CommOptions(tag=1),
        )
        print(object, metadata)


if __name__ == "__main__":
    asyncio.run(main())
