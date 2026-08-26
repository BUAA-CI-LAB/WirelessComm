# Wireless Comm

An asyncio-based P2P communication baseline for nodes sharing a Wi-Fi network.
The V0.1 implementation provides reusable full-duplex TCP connections and a
structured payload codec with native contiguous CPU Tensor segments.

## Install

```bash
pip install -e '.[tensor,test]'
```

PyTorch is optional at package import time. Install the `tensor` extra to send
or receive `torch.Tensor` values.

## P2P API

Every node receives the complete static peer directory during initialization:

```python
from wireless_comm import Comm, CommOptions, Peer

local = Peer("node-a", "192.0.2.10", 9000)
remote = Peer("node-b", "192.0.2.11", 9000)

comm = await Comm.create(
    local=local,
    peers=[remote],
    bind_host="0.0.0.0",
)

await comm.send(
    {"tensor": contiguous_cpu_tensor},
    comm.peer("node-b"),
    piggypayload={"trace_id": "request-42"},
    options=CommOptions(tag=1),
)

object, piggypayload = await comm.recv(
    comm.peer("node-b"),
    CommOptions(tag=1),
)
```

`piggypayload` is optional metadata. It accepts nested MessagePack primitives
with string dictionary keys and defaults to a 64 KiB limit. Large data belongs
in `object`.

Supported main payload values include primitives, bytes-like values, lists,
tuples, dictionaries, contiguous dense CPU Tensors, registered dataclasses,
and registered custom objects. Unregistered arbitrary Python objects require
`CommConfig(allow_unsafe_pickle=True)` and must only be used with trusted peers.

Tensor storage is read directly through `Tensor.data_ptr()`. Do not mutate or
resize a submitted Tensor until `await comm.send(...)` completes.

## Run

```bash
python examples/p2p_local.py
python -m unittest discover -s tests -v
```
