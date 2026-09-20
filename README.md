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

## Two-node Wi-Fi CLI

Copy and edit `configs/node-a.yaml` and `configs/node-b.yaml`. `local.host` is
the address advertised to peers, while `local.bind_host` may be `0.0.0.0`.
Every node must list all allowed remote peers during initialization.

Start an echo receiver on node B:

```bash
python -m wireless_comm.node \
  --config configs/node-b.yaml \
  echo --src node-a --tag 1 --count 1
```

Send a JSON payload from node A and wait for the echo:

```bash
python -m wireless_comm.node \
  --config configs/node-a.yaml \
  send --dst node-b --tag 1 \
  --json '{"message":"hello"}' \
  --metadata '{"trace_id":"wifi-test-1"}' \
  --expect-reply
```

Generate and send a contiguous Tensor:

```bash
python -m wireless_comm.node \
  --config configs/node-a.yaml \
  send --dst node-b --tensor-shape 3,224,224 --tensor-dtype float32
```

For a localhost two-process check, replace the configs with
`configs/local-node-a.yaml` and `configs/local-node-b.yaml`.

## Wi-Fi benchmark

The benchmark combines application-level Comm measurements with system, route,
radio, interface, and TCP diagnostics. Two recorded studies live under
[`reports/`](reports/): a scheduler A/B sweep and a four-node 5 GHz Wi-Fi
profile. Start the responder first:

```bash
python -m wireless_comm.benchmark \
  --config configs/agx-thor-wifi.yaml \
  responder --peer agx-orin --interface wlP1p1s0 \
  --output thor-benchmark.json
```

Then run the initiator:

```bash
python -m wireless_comm.benchmark \
  --config configs/agx-orin-wifi.yaml \
  initiator --peer agx-thor --interface wlP1p1s0 \
  --output orin-to-thor-benchmark.json
```

The initiator report contains both nodes' observations:

- route, MTU, addresses, memory, CPU load, and TCP kernel settings;
- SSID, access point, frequency, signal, PHY bitrates, station statistics, and
  regulatory-domain output when `iw` is installed;
- bidirectional ICMP loss, latency, and jitter;
- interface byte, packet, error, and drop deltas;
- host context-switch, NET_RX/NET_TX softirq, softnet drop/time-squeeze, and
  Wi-Fi device IRQ deltas, including their per-CPU distribution;
- per-process CPU and context-switch deltas, observation durations, and rates;
- periodic signal, PHY bitrate, interface-counter, and kernel-network samples;
- first-connection cost and application RTT p50/p95/p99/stddev by payload size;
- latency p01/p05/p25/p50/p75/p90/p95/p99/p99.9, consecutive-sample jitter,
  median absolute deviation, coefficient of variation, and outlier rate;
- repeated upload, download, and simultaneous full-duplex payload goodput trials,
  including their percentile distributions and raw samples;
- optional contiguous CPU Tensor round trips plus Tensor list/dict correctness.

The default throughput sweep transfers enough data for useful measurements but
can take several minutes on a slow link. A shorter diagnostic run can override
the matrix:

```bash
python -m wireless_comm.benchmark \
  --config configs/agx-orin-wifi.yaml \
  initiator --peer agx-thor --interface wlP1p1s0 \
  --latency-sizes 0,64,1024 --latency-rounds 20 \
  --throughput-sizes 1024,65536,1048576 \
  --throughput-target-bytes 4194304 \
  --throughput-trials 3 \
  --output quick-benchmark.json
```

When both processes run in Python environments with PyTorch, enable Tensor
round trips with byte sizes divisible by four (the benchmark uses float32):

```bash
python -m wireless_comm.benchmark \
  --config configs/agx-orin-wifi.yaml \
  initiator --peer agx-thor --interface wlP1p1s0 \
  --tensor-sizes 4096,1048576,16777216 --tensor-rounds 10 \
  --output tensor-benchmark.json
```

### Latency under load

Background traffic uses separate TCP connections from the foreground Comm
probe. Fractions are relative to the measured median capacity for each
direction; the report also records the actually delivered fraction.

```bash
python -m wireless_comm.benchmark \
  --config configs/agx-orin-wifi.yaml \
  initiator --peer agx-thor --interface wlP1p1s0 \
  --throughput-sizes 1048576 --throughput-trials 3 \
  --load-fractions 0.25,0.5,0.75,0.9,1.0 \
  --load-directions upload,download,bidirectional \
  --load-concurrencies 1,4 \
  --load-duration 10 --load-probe-interval 0.05 \
  --output latency-under-load.json
```

Each scenario reports foreground RTT percentiles and raw samples, background
target and delivered Mbps, direction, and TCP stream count. Randomized probe
spacing reduces synchronization with Wi-Fi scheduling cycles.

### Two-host multi-rank collectives

Start the odd ranks on the second physical host:

```bash
python -m wireless_comm.multirank \
  --local-host 192.0.2.100 --remote-host 192.0.2.101 \
  --base-port 9400 --ranks-per-host 2 --interface wlP1p1s0 \
  --output thor-multirank.json responder
```

Then start the even ranks on the first host:

```bash
python -m wireless_comm.multirank \
  --local-host 192.0.2.101 --remote-host 192.0.2.100 \
  --base-port 9400 --ranks-per-host 2 --interface wlP1p1s0 \
  --output orin-multirank.json initiator \
  --payload-sizes 1024,65536,1048576 --rounds 20 \
  --collectives barrier,broadcast,allgather,ring_allreduce
```

Ranks alternate between hosts, so every ring edge crosses Wi-Fi. Ring
allreduce uses bitwise XOR on byte chunks to keep the scenario portable and
is validated on every iteration. This mode represents multiple connections,
rank synchronization, and slow-rank amplification across two stations. It
does not represent independent radios, hidden terminals, or access-point
airtime fairness among three or more physical stations.

### Physical N-node cluster

For three or more physical Wi-Fi stations, copy
`configs/cluster-four-node.example.yaml` and fill in each node's Wi-Fi address,
SSH alias, interface, Python executable, working directory, and owned ranks.
Ranks must be unique and contiguous, the controller must own rank 0, and
`ring` may specify any permutation of the ranks.

Preview every copy and remote worker command without changing remote nodes:

```bash
python -m wireless_comm.cluster_benchmark launch \
  --config configs/cluster-four-node.yaml --dry-run
```

Once the same project and Python dependencies exist in every configured
working directory, launch the cluster from the wired control machine:

```bash
python -m wireless_comm.cluster_benchmark launch \
  --config configs/cluster-four-node.yaml \
  --output four-node-collectives.json
```

The launcher copies only the topology file, starts non-controller workers
before the controller, and returns one controller report containing every
physical node's radio diagnostics, every rank's raw samples, and aggregate
collective percentiles. Comm payloads use the configured Wi-Fi `host` values;
SSH aliases are used only by the control plane.

Goodput is measured from application payload bytes, not Ethernet or TCP bytes.
TCP retransmits link-layer losses, so the report does not claim a TCP message
loss rate; use the ping, radio, and interface diagnostics to identify loss.

## Backpressure

Each peer has one FIFO writer and a queue bounded by both messages and bytes:

```yaml
comm:
  max_peer_queued_messages: 1024
  max_peer_queued_bytes: 134217728
```

`send()` waits for capacity by default. Use fail-fast behavior when blocking is
undesirable:

```python
await comm.send(
    payload,
    peer,
    options=CommOptions(wait_for_capacity=False),
)
```

A full queue raises `BackpressureError`. A timeout or cancellation removes a
queued message. If a message is already active, its peer writer is cancelled
before the borrowed Tensor or buffer is released.
