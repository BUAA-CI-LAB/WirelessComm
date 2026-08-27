"""Node-level P2P communication over reusable full-duplex TCP connections."""

from __future__ import annotations

import asyncio
import contextlib
import socket
import uuid
from collections import defaultdict, deque
from collections.abc import Callable, Iterable, Mapping
from dataclasses import dataclass, field
from typing import Any

from typing_extensions import Self

from .codec import CodecRegistry, EncodedPayload, PayloadCodec
from .errors import (
    BackpressureError,
    CommError,
    ConnectionClosedError,
    ConnectionFailedError,
    OperationTimeoutError,
    ProtocolError,
)
from .protocol import (
    Frame,
    MessageKind,
    control_payload,
    decode_control,
    encode_frame,
    read_frame,
)
from .scheduler import ByteScheduler
from .types import CommConfig, CommOptions, Metadata, Object, Peer, SendResult


@dataclass(slots=True)
class _InboundMessage:
    tag: int
    message_id: int
    payload: EncodedPayload


@dataclass(slots=True)
class _Connection:
    connection_id: str
    peer_id: str
    reader: asyncio.StreamReader
    writer: asyncio.StreamWriter
    write_lock: asyncio.Lock = field(default_factory=asyncio.Lock)
    reader_task: asyncio.Task[None] | None = None


@dataclass(slots=True)
class _OutboundMessage:
    frame: Frame
    chunks: tuple[bytes | memoryview, ...]
    wire_bytes: int
    completion: asyncio.Future[SendResult]
    active: bool = False


@dataclass(slots=True)
class _PeerState:
    peer: Peer
    connections: dict[str, _Connection] = field(default_factory=dict)
    connect_lock: asyncio.Lock = field(default_factory=asyncio.Lock)
    queue: deque[_OutboundMessage] = field(default_factory=deque)
    queue_condition: asyncio.Condition = field(default_factory=asyncio.Condition)
    pending_messages: int = 0
    pending_bytes: int = 0
    writer_task: asyncio.Task[None] | None = None

    def preferred(self) -> _Connection | None:
        if not self.connections:
            return None
        return self.connections[min(self.connections)]


class Comm:
    """P2P runtime for one node.

    All peers are supplied at creation time. Connections are established lazily,
    then used in both directions and reused. The optional ``piggypayload`` is
    small MessagePack metadata delivered atomically with the main object.
    """

    def __init__(
        self,
        *,
        local: Peer,
        peers: Iterable[Peer],
        bind_host: str,
        config: CommConfig,
    ) -> None:
        directory: dict[str, Peer] = {local.node_id: local}
        for peer in peers:
            if peer.node_id in directory:
                raise ValueError(f"duplicate peer node_id {peer.node_id!r}")
            directory[peer.node_id] = peer

        self.local = local
        self.config = config
        self._bind_host = bind_host
        self._directory = directory
        self._states = {
            node_id: _PeerState(peer)
            for node_id, peer in directory.items()
            if node_id != local.node_id
        }
        self.registry = CodecRegistry()
        self._codec = PayloadCodec(self.registry, config)
        self._server: asyncio.AbstractServer | None = None
        self._next_message_id = 1
        self._inbound: dict[str, deque[_InboundMessage]] = defaultdict(deque)
        self._peer_errors: dict[str, CommError] = {}
        self._inbound_condition = asyncio.Condition()
        self._connection_tasks: set[asyncio.Task[None]] = set()
        self._closed = False
        self._byte_scheduler = (
            ByteScheduler(
                config.egress_quantum_bytes,
                config.egress_rate_bytes_per_second,
                config.egress_burst_bytes or config.egress_quantum_bytes,
            )
            if config.egress_quantum_bytes is not None
            else None
        )

    @classmethod
    async def create(
        cls,
        *,
        local: Peer,
        peers: Iterable[Peer],
        bind_host: str | None = None,
        config: CommConfig | None = None,
    ) -> Self:
        """Create a Comm with a complete static peer directory."""

        runtime = cls(
            local=local,
            peers=peers,
            bind_host=bind_host if bind_host is not None else local.host,
            config=config if config is not None else CommConfig(),
        )
        runtime._server = await asyncio.start_server(
            runtime._accept_connection,
            runtime._bind_host,
            local.port,
        )
        return runtime

    def peer(self, node_id: str) -> Peer:
        """Resolve a peer from the directory supplied at initialization."""

        try:
            peer = self._directory[node_id]
        except KeyError as exc:
            raise KeyError(f"unknown peer {node_id!r}") from exc
        if peer.node_id == self.local.node_id:
            raise ValueError("local node is not a remote peer")
        return peer

    def peers(self) -> tuple[Peer, ...]:
        """Return every configured remote peer in initialization order."""

        return tuple(
            peer
            for node_id, peer in self._directory.items()
            if node_id != self.local.node_id
        )

    def register_dataclass(
        self, python_type: type, *, type_id: str, version: int = 1
    ) -> None:
        self.registry.register_dataclass(
            python_type,
            type_id=type_id,
            version=version,
        )

    def register_codec(
        self,
        python_type: type,
        *,
        type_id: str,
        encoder: Callable[[Any], Any],
        decoder: Callable[[Any], Any],
        version: int = 1,
    ) -> None:
        self.registry.register_codec(
            python_type,
            type_id=type_id,
            version=version,
            encoder=encoder,
            decoder=decoder,
        )

    async def send(
        self,
        object: Object,
        dst: Peer,
        *,
        piggypayload: Mapping[str, Any] | None = None,
        options: CommOptions | None = None,
    ) -> SendResult:
        """Send a main object and optional small metadata to a configured peer.

        Completion means the frame has been passed to the local TCP transport.
        It does not mean the remote application has consumed the message. Tensor
        storage must not be modified until this coroutine completes.
        """

        operation_options = options if options is not None else CommOptions()
        self._validate_peer(dst)
        try:
            if operation_options.timeout is None:
                return await self._send(
                    object,
                    piggypayload,
                    dst,
                    operation_options.tag,
                    operation_options.wait_for_capacity,
                )
            return await asyncio.wait_for(
                self._send(
                    object,
                    piggypayload,
                    dst,
                    operation_options.tag,
                    operation_options.wait_for_capacity,
                ),
                operation_options.timeout,
            )
        except asyncio.TimeoutError as exc:
            raise OperationTimeoutError(f"send to {dst.node_id!r} timed out") from exc

    async def recv(
        self,
        src: Peer,
        options: CommOptions | None = None,
    ) -> tuple[Object, Metadata | None]:
        """Receive the next matching object and piggyback metadata from ``src``."""

        operation_options = options if options is not None else CommOptions()
        self._validate_peer(src)
        try:
            if operation_options.timeout is None:
                message = await self._receive_matching(
                    src.node_id, operation_options.tag
                )
            else:
                message = await asyncio.wait_for(
                    self._receive_matching(src.node_id, operation_options.tag),
                    operation_options.timeout,
                )
        except asyncio.TimeoutError as exc:
            raise OperationTimeoutError(
                f"receive from {src.node_id!r} timed out"
            ) from exc
        decoded = self._codec.decode(message.payload)
        return decoded.object, decoded.piggypayload

    async def close(self) -> None:
        """Close the listener and every active peer connection."""

        if self._closed:
            return
        self._closed = True
        if self._server is not None:
            self._server.close()

        writer_tasks: list[asyncio.Task[None]] = []
        for state in self._states.values():
            async with state.queue_condition:
                while state.queue:
                    message = state.queue.popleft()
                    state.pending_messages -= 1
                    state.pending_bytes -= message.wire_bytes
                    if not message.completion.done():
                        message.completion.set_exception(
                            ConnectionClosedError("Comm closed before message was sent")
                        )
                state.queue_condition.notify_all()
            if state.writer_task is not None:
                state.writer_task.cancel()
                writer_tasks.append(state.writer_task)
        if writer_tasks:
            await asyncio.gather(*writer_tasks, return_exceptions=True)
        if self._byte_scheduler is not None:
            await self._byte_scheduler.close()

        connections = [
            connection
            for state in self._states.values()
            for connection in state.connections.values()
        ]
        for connection in connections:
            connection.writer.close()
        for connection in connections:
            with contextlib.suppress(Exception):
                await connection.writer.wait_closed()
        if self._server is not None:
            await self._server.wait_closed()

        tasks = list(self._connection_tasks)
        for task in tasks:
            task.cancel()
        if tasks:
            await asyncio.gather(*tasks, return_exceptions=True)
        async with self._inbound_condition:
            self._inbound_condition.notify_all()

    async def __aenter__(self) -> Self:
        return self

    async def __aexit__(self, *exc_info: object) -> None:
        await self.close()

    async def _send(
        self,
        object: Object,
        piggypayload: Mapping[str, Any] | None,
        dst: Peer,
        tag: int,
        wait_for_capacity: bool,
    ) -> SendResult:
        if self._closed:
            raise ConnectionClosedError("Comm is closed")
        payload = self._codec.encode(object, piggypayload)
        message_id = self._allocate_message_id()
        frame = Frame(MessageKind.DATA, tag, message_id, payload)
        chunks = encode_frame(frame, self.config)
        wire_bytes = sum(len(chunk) for chunk in chunks)
        completion = asyncio.get_running_loop().create_future()
        message = _OutboundMessage(frame, chunks, wire_bytes, completion)
        await self._enqueue(dst, message, wait_for_capacity)
        try:
            return await asyncio.shield(completion)
        except asyncio.CancelledError:
            removed = await self._remove_queued(dst.node_id, message)
            if not removed and message.active:
                await self._cancel_active_send(dst.node_id, message)
            raise

    async def _enqueue(
        self,
        peer: Peer,
        message: _OutboundMessage,
        wait_for_capacity: bool,
    ) -> None:
        if message.wire_bytes > self.config.max_peer_queued_bytes:
            raise BackpressureError(
                "message is larger than the per-peer queued-byte limit"
            )
        state = self._states[peer.node_id]
        async with state.queue_condition:
            while self._queue_is_full(state, message.wire_bytes):
                if not wait_for_capacity:
                    raise BackpressureError(f"send queue for {peer.node_id!r} is full")
                await state.queue_condition.wait()
                if self._closed:
                    raise ConnectionClosedError("Comm is closed")
            state.queue.append(message)
            state.pending_messages += 1
            state.pending_bytes += message.wire_bytes
            if state.writer_task is None or state.writer_task.done():
                state.writer_task = asyncio.create_task(
                    self._writer_loop(state),
                    name=f"wireless-comm-tx-{peer.node_id}",
                )
            state.queue_condition.notify_all()

    async def _remove_queued(
        self,
        peer_id: str,
        message: _OutboundMessage,
    ) -> bool:
        state = self._states[peer_id]
        async with state.queue_condition:
            try:
                state.queue.remove(message)
            except ValueError:
                return False
            state.pending_messages -= 1
            state.pending_bytes -= message.wire_bytes
            message.completion.cancel()
            state.queue_condition.notify_all()
            return True

    async def _cancel_active_send(
        self,
        peer_id: str,
        message: _OutboundMessage,
    ) -> None:
        state = self._states[peer_id]
        writer_task = state.writer_task
        if writer_task is None or not message.active:
            return
        writer_task.cancel()
        await asyncio.gather(writer_task, return_exceptions=True)
        with contextlib.suppress(Exception):
            await asyncio.shield(message.completion)
        async with state.queue_condition:
            if state.queue and not self._closed:
                state.writer_task = asyncio.create_task(
                    self._writer_loop(state),
                    name=f"wireless-comm-tx-{peer_id}",
                )

    async def _writer_loop(self, state: _PeerState) -> None:
        while True:
            async with state.queue_condition:
                while not state.queue:
                    if self._closed:
                        return
                    await state.queue_condition.wait()
                message = state.queue.popleft()
                message.active = True
            connection: _Connection | None = None
            try:
                connection = await self._connection_for(state.peer)
                async with connection.write_lock:
                    await self._write_chunks(
                        state.peer.node_id,
                        connection.writer,
                        message.chunks,
                    )
                if not message.completion.done():
                    message.completion.set_result(
                        SendResult(message.frame.message_id, message.wire_bytes)
                    )
            except asyncio.CancelledError:
                if not message.completion.done():
                    message.completion.set_exception(
                        ConnectionClosedError("Comm closed while sending message")
                    )
                raise
            except (CommError, ConnectionError, OSError) as exc:
                if connection is not None and isinstance(
                    exc, (ConnectionError, OSError)
                ):
                    await self._drop_connection(connection)
                if not message.completion.done():
                    if isinstance(exc, CommError):
                        message.completion.set_exception(exc)
                    else:
                        message.completion.set_exception(
                            ConnectionClosedError(
                                f"connection to {state.peer.node_id!r} failed during send"
                            )
                        )
            finally:
                message.active = False
                async with state.queue_condition:
                    state.pending_messages -= 1
                    state.pending_bytes -= message.wire_bytes
                    state.queue_condition.notify_all()

    async def _write_chunks(
        self,
        peer_id: str,
        writer: asyncio.StreamWriter,
        chunks: tuple[bytes | memoryview, ...],
    ) -> None:
        if self._byte_scheduler is None:
            for chunk in chunks:
                writer.write(chunk)
            await writer.drain()
            return

        views = tuple(memoryview(chunk) for chunk in chunks if len(chunk))
        remaining = sum(view.nbytes for view in views)
        chunk_index = 0
        chunk_offset = 0
        while remaining:
            granted = await self._byte_scheduler.acquire(peer_id, remaining)
            grant_remaining = granted
            while grant_remaining:
                view = views[chunk_index]
                portion = min(grant_remaining, view.nbytes - chunk_offset)
                writer.write(view[chunk_offset : chunk_offset + portion])
                chunk_offset += portion
                grant_remaining -= portion
                remaining -= portion
                if chunk_offset == view.nbytes:
                    chunk_index += 1
                    chunk_offset = 0
            await writer.drain()

    def _queue_is_full(self, state: _PeerState, wire_bytes: int) -> bool:
        return (
            state.pending_messages >= self.config.max_peer_queued_messages
            or state.pending_bytes + wire_bytes > self.config.max_peer_queued_bytes
        )

    async def _connection_for(self, peer: Peer) -> _Connection:
        state = self._states[peer.node_id]
        connection = state.preferred()
        if connection is not None:
            return connection
        async with state.connect_lock:
            connection = state.preferred()
            if connection is not None:
                return connection
            return await self._dial(peer)

    async def _dial(self, peer: Peer) -> _Connection:
        connection_id = uuid.uuid4().hex
        writer: asyncio.StreamWriter | None = None
        try:
            reader, writer = await asyncio.open_connection(peer.host, peer.port)
            self._configure_socket(writer)
            hello = Frame(
                MessageKind.HELLO,
                0,
                0,
                control_payload(
                    {
                        "node_id": self.local.node_id,
                        "connection_id": connection_id,
                    }
                ),
            )
            await self._write_handshake(writer, hello)
            ack = await read_frame(reader, self.config)
            if ack.kind is not MessageKind.HELLO_ACK:
                raise ProtocolError("peer did not acknowledge HELLO")
            fields = decode_control(ack.payload)
            if fields.get("node_id") != peer.node_id:
                raise ProtocolError("connected endpoint returned an unexpected node_id")
            if fields.get("connection_id") != connection_id:
                raise ProtocolError("HELLO_ACK returned an unexpected connection_id")
            connection = _Connection(connection_id, peer.node_id, reader, writer)
            self._register_connection(connection)
            return connection
        except (OSError, asyncio.IncompleteReadError) as exc:
            if writer is not None:
                writer.close()
            raise ConnectionFailedError(
                f"could not connect to {peer.node_id!r} at {peer.host}:{peer.port}"
            ) from exc
        except CommError:
            if writer is not None:
                writer.close()
            raise
        except asyncio.CancelledError:
            if writer is not None:
                writer.close()
                with contextlib.suppress(Exception):
                    await writer.wait_closed()
            raise

    async def _accept_connection(
        self,
        reader: asyncio.StreamReader,
        writer: asyncio.StreamWriter,
    ) -> None:
        self._configure_socket(writer)
        try:
            hello = await read_frame(reader, self.config)
            if hello.kind is not MessageKind.HELLO:
                raise ProtocolError("first frame on a connection must be HELLO")
            fields = decode_control(hello.payload)
            peer_id = fields.get("node_id")
            connection_id = fields.get("connection_id")
            if not isinstance(peer_id, str) or peer_id not in self._states:
                raise ProtocolError(f"HELLO identifies unknown peer {peer_id!r}")
            if not isinstance(connection_id, str) or not connection_id:
                raise ProtocolError("HELLO contains an invalid connection_id")
            ack = Frame(
                MessageKind.HELLO_ACK,
                0,
                0,
                control_payload(
                    {
                        "node_id": self.local.node_id,
                        "connection_id": connection_id,
                    }
                ),
            )
            await self._write_handshake(writer, ack)
            self._register_connection(
                _Connection(connection_id, peer_id, reader, writer)
            )
        except (CommError, OSError, asyncio.IncompleteReadError):
            writer.close()
            with contextlib.suppress(Exception):
                await writer.wait_closed()

    def _register_connection(self, connection: _Connection) -> None:
        state = self._states[connection.peer_id]
        previous = state.connections.get(connection.connection_id)
        if previous is not None:
            connection.writer.close()
            return
        state.connections[connection.connection_id] = connection
        self._peer_errors.pop(connection.peer_id, None)
        task = asyncio.create_task(
            self._reader_loop(connection),
            name=f"wireless-comm-rx-{connection.peer_id}",
        )
        connection.reader_task = task
        self._connection_tasks.add(task)
        task.add_done_callback(self._connection_tasks.discard)

    async def _reader_loop(self, connection: _Connection) -> None:
        error: CommError = ConnectionClosedError(
            f"peer {connection.peer_id!r} closed its connection"
        )
        try:
            while not self._closed:
                frame = await read_frame(connection.reader, self.config)
                if frame.kind is MessageKind.DATA:
                    async with self._inbound_condition:
                        self._inbound[connection.peer_id].append(
                            _InboundMessage(frame.tag, frame.message_id, frame.payload)
                        )
                        self._inbound_condition.notify_all()
                elif frame.kind is MessageKind.CLOSE:
                    break
                else:
                    raise ProtocolError(f"unexpected frame kind {frame.kind.name}")
        except asyncio.CancelledError:
            return
        except asyncio.IncompleteReadError:
            pass
        except CommError as exc:
            error = exc
        except (ConnectionError, OSError) as exc:
            error = ConnectionClosedError(str(exc))
        finally:
            await self._drop_connection(connection, error)

    async def _drop_connection(
        self,
        connection: _Connection,
        error: CommError | None = None,
    ) -> None:
        state = self._states[connection.peer_id]
        if state.connections.get(connection.connection_id) is connection:
            del state.connections[connection.connection_id]
        connection.writer.close()
        if connection.reader_task is not asyncio.current_task():
            with contextlib.suppress(Exception):
                await connection.writer.wait_closed()
        if error is not None and not state.connections:
            async with self._inbound_condition:
                self._peer_errors[connection.peer_id] = error
                self._inbound_condition.notify_all()

    async def _receive_matching(
        self,
        peer_id: str,
        tag: int,
    ) -> _InboundMessage:
        async with self._inbound_condition:
            while True:
                queue = self._inbound[peer_id]
                for index, message in enumerate(queue):
                    if message.tag == tag:
                        del queue[index]
                        return message
                error = self._peer_errors.pop(peer_id, None)
                if error is not None:
                    raise error
                if self._closed:
                    raise ConnectionClosedError("Comm is closed")
                await self._inbound_condition.wait()

    async def _write_handshake(
        self,
        writer: asyncio.StreamWriter,
        frame: Frame,
    ) -> None:
        for chunk in encode_frame(frame, self.config):
            writer.write(chunk)
        await writer.drain()

    def _validate_peer(self, peer: Peer) -> None:
        configured = self._directory.get(peer.node_id)
        if configured != peer or peer.node_id == self.local.node_id:
            raise ValueError(f"peer {peer.node_id!r} is not configured in this Comm")

    def _allocate_message_id(self) -> int:
        message_id = self._next_message_id
        self._next_message_id += 1
        if self._next_message_id > 0xFFFFFFFFFFFFFFFF:
            raise RuntimeError("message ID space exhausted; recreate Comm")
        return message_id

    def _configure_socket(self, writer: asyncio.StreamWriter) -> None:
        tcp_socket = writer.get_extra_info("socket")
        if tcp_socket is not None and self.config.tcp_nodelay:
            tcp_socket.setsockopt(socket.IPPROTO_TCP, socket.TCP_NODELAY, 1)
