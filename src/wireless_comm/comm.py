"""Node-level P2P communication over reusable full-duplex TCP connections."""

from __future__ import annotations

import asyncio
import contextlib
import socket
import uuid
from collections import defaultdict, deque
from collections.abc import Callable, Iterable, Mapping
from dataclasses import dataclass, field
from typing import Any, Self

from .codec import CodecRegistry, EncodedPayload, PayloadCodec
from .errors import (
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
class _PeerState:
    connections: dict[str, _Connection] = field(default_factory=dict)
    connect_lock: asyncio.Lock = field(default_factory=asyncio.Lock)

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
            node_id: _PeerState() for node_id in directory if node_id != local.node_id
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
                )
            async with asyncio.timeout(operation_options.timeout):
                return await self._send(
                    object,
                    piggypayload,
                    dst,
                    operation_options.tag,
                )
        except TimeoutError as exc:
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
                async with asyncio.timeout(operation_options.timeout):
                    message = await self._receive_matching(
                        src.node_id,
                        operation_options.tag,
                    )
        except TimeoutError as exc:
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
    ) -> SendResult:
        if self._closed:
            raise ConnectionClosedError("Comm is closed")
        payload = self._codec.encode(object, piggypayload)
        message_id = self._allocate_message_id()
        frame = Frame(MessageKind.DATA, tag, message_id, payload)
        chunks = encode_frame(frame, self.config)
        connection = await self._connection_for(dst)

        async with connection.write_lock:
            try:
                for chunk in chunks:
                    connection.writer.write(chunk)
                await connection.writer.drain()
            except (ConnectionError, OSError) as exc:
                await self._drop_connection(connection)
                raise ConnectionClosedError(
                    f"connection to {dst.node_id!r} failed during send"
                ) from exc
        return SendResult(message_id, sum(len(chunk) for chunk in chunks))

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
