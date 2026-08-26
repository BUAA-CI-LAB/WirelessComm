"""Structured payload encoding with native contiguous Tensor segments."""

from __future__ import annotations

import ctypes
import dataclasses
import math
import pickle
from collections.abc import Callable, Mapping
from dataclasses import dataclass
from typing import Any

import msgpack

from .errors import SerializationError, UnknownCodecError, UnsupportedPayloadError
from .types import CommConfig, Metadata

try:
    import torch
except ImportError:
    torch = None  # type: ignore[assignment]


@dataclass(frozen=True, slots=True)
class BufferSegment:
    """A wire buffer and the object that keeps its storage alive."""

    view: memoryview
    owner: object

    @property
    def nbytes(self) -> int:
        return self.view.nbytes


@dataclass(frozen=True, slots=True)
class EncodedPayload:
    manifest: bytes
    segments: tuple[BufferSegment, ...]


@dataclass(frozen=True, slots=True)
class DecodedPayload:
    object: Any
    piggypayload: Metadata | None


@dataclass(frozen=True, slots=True)
class _CustomCodec:
    python_type: type
    type_id: str
    version: int
    encoder: Callable[[Any], Any]
    decoder: Callable[[Any], Any]


class CodecRegistry:
    """Registry for dataclasses and application-defined object codecs."""

    def __init__(self) -> None:
        self.dataclasses_by_type: dict[type, tuple[str, int]] = {}
        self.dataclasses_by_id: dict[tuple[str, int], type] = {}
        self.custom_by_type: dict[type, _CustomCodec] = {}
        self.custom_by_id: dict[tuple[str, int], _CustomCodec] = {}

    def register_dataclass(
        self, python_type: type, *, type_id: str, version: int = 1
    ) -> None:
        if not dataclasses.is_dataclass(python_type):
            raise TypeError(f"{python_type!r} is not a dataclass type")
        self._ensure_available(type_id, version)
        self.dataclasses_by_type[python_type] = (type_id, version)
        self.dataclasses_by_id[(type_id, version)] = python_type

    def register_codec(
        self,
        python_type: type,
        *,
        type_id: str,
        version: int,
        encoder: Callable[[Any], Any],
        decoder: Callable[[Any], Any],
    ) -> None:
        self._ensure_available(type_id, version)
        codec = _CustomCodec(python_type, type_id, version, encoder, decoder)
        self.custom_by_type[python_type] = codec
        self.custom_by_id[(type_id, version)] = codec

    def _ensure_available(self, type_id: str, version: int) -> None:
        if not type_id:
            raise ValueError("type_id must be non-empty")
        if version < 1:
            raise ValueError("codec version must be positive")
        key = (type_id, version)
        if key in self.dataclasses_by_id or key in self.custom_by_id:
            raise ValueError(f"codec {type_id!r} version {version} is registered")


class PayloadCodec:
    """Flatten supported Python values into a manifest and buffer segments."""

    def __init__(self, registry: CodecRegistry, config: CommConfig) -> None:
        self.registry = registry
        self.config = config

    def encode(
        self, object: Any, piggypayload: Mapping[str, Any] | None
    ) -> EncodedPayload:
        metadata = self._encode_metadata(piggypayload)
        segments: list[BufferSegment] = []
        active_containers: set[int] = set()
        node_count = 0

        def walk(value: Any, depth: int) -> Any:
            nonlocal node_count
            node_count += 1
            if node_count > self.config.max_container_nodes:
                raise SerializationError("payload contains too many nodes")
            if depth > self.config.max_container_depth:
                raise SerializationError("payload exceeds maximum container depth")

            if value is None:
                return ["none"]
            if type(value) is bool:
                return ["bool", value]
            if type(value) is int:
                if not -(1 << 63) <= value < (1 << 64):
                    raise SerializationError(
                        "integer is outside MessagePack 64-bit range"
                    )
                return ["int", value]
            if type(value) is float:
                return ["float", value]
            if type(value) is str:
                return ["str", value]

            if torch is not None and type(value) is torch.Tensor:
                return self._encode_tensor(value, segments)
            if isinstance(value, (bytes, bytearray, memoryview)):
                return self._encode_bytes(value, segments)

            value_id = id(value)
            if type(value) in (list, tuple, dict) or dataclasses.is_dataclass(value):
                if value_id in active_containers:
                    raise SerializationError(
                        "cyclic payload structures are unsupported"
                    )
                active_containers.add(value_id)
                try:
                    if type(value) is list:
                        return ["list", [walk(item, depth + 1) for item in value]]
                    if type(value) is tuple:
                        return ["tuple", [walk(item, depth + 1) for item in value]]
                    if type(value) is dict:
                        return [
                            "dict",
                            [
                                [walk(key, depth + 1), walk(item, depth + 1)]
                                for key, item in value.items()
                            ],
                        ]

                    registration = self.registry.dataclasses_by_type.get(type(value))
                    if registration is None:
                        raise UnsupportedPayloadError(
                            f"dataclass {type(value).__qualname__} is not registered"
                        )
                    type_id, version = registration
                    fields = [
                        [field.name, walk(getattr(value, field.name), depth + 1)]
                        for field in dataclasses.fields(value)
                    ]
                    return ["dataclass", type_id, version, fields]
                finally:
                    active_containers.remove(value_id)

            custom = self.registry.custom_by_type.get(type(value))
            if custom is not None:
                encoded = custom.encoder(value)
                return [
                    "custom",
                    custom.type_id,
                    custom.version,
                    walk(encoded, depth + 1),
                ]

            if self.config.allow_unsafe_pickle:
                try:
                    blob = pickle.dumps(value, protocol=pickle.HIGHEST_PROTOCOL)
                except Exception as exc:
                    raise SerializationError(f"pickle encoding failed: {exc}") from exc
                return self._append_owned_bytes("pickle", blob, segments)

            raise UnsupportedPayloadError(
                f"unsupported payload type {type(value).__module__}."
                f"{type(value).__qualname__}; register a codec or explicitly enable "
                "trusted pickle"
            )

        root = walk(object, 0)
        manifest = msgpack.packb(
            {
                "root": root,
                "piggypayload": metadata,
                "segments": [segment.nbytes for segment in segments],
            },
            use_bin_type=True,
        )
        if len(manifest) > self.config.max_manifest_bytes:
            raise SerializationError("payload manifest exceeds configured limit")
        if len(segments) > self.config.max_segments:
            raise SerializationError("payload has too many buffer segments")
        return EncodedPayload(manifest, tuple(segments))

    def decode(self, payload: EncodedPayload) -> DecodedPayload:
        try:
            manifest = msgpack.unpackb(
                payload.manifest, raw=False, strict_map_key=False
            )
            root = manifest["root"]
            piggypayload = manifest["piggypayload"]
            lengths = manifest["segments"]
        except Exception as exc:
            raise SerializationError(f"invalid payload manifest: {exc}") from exc
        if lengths != [segment.nbytes for segment in payload.segments]:
            raise SerializationError("payload segment lengths do not match manifest")
        self._validate_decoded_metadata(piggypayload)

        def walk(node: Any, depth: int) -> Any:
            if depth > self.config.max_container_depth:
                raise SerializationError("decoded payload exceeds maximum depth")
            if not isinstance(node, list) or not node:
                raise SerializationError("invalid payload tree node")
            opcode = node[0]
            if opcode == "none":
                return None
            if opcode in {"bool", "int", "float", "str"}:
                return node[1]
            if opcode == "bytes":
                return bytes(self._segment(node, payload.segments))
            if opcode == "pickle":
                if not self.config.allow_unsafe_pickle:
                    raise UnsupportedPayloadError(
                        "received pickle while trusted pickle is disabled"
                    )
                try:
                    return pickle.loads(self._segment(node, payload.segments))
                except Exception as exc:
                    raise SerializationError(f"pickle decoding failed: {exc}") from exc
            if opcode == "tensor":
                return self._decode_tensor(node, payload.segments)
            if opcode == "list":
                return [walk(item, depth + 1) for item in node[1]]
            if opcode == "tuple":
                return tuple(walk(item, depth + 1) for item in node[1])
            if opcode == "dict":
                try:
                    return {
                        walk(pair[0], depth + 1): walk(pair[1], depth + 1)
                        for pair in node[1]
                    }
                except TypeError as exc:
                    raise SerializationError(
                        "decoded dict contains an unhashable key"
                    ) from exc
            if opcode == "dataclass":
                python_type = self.registry.dataclasses_by_id.get((node[1], node[2]))
                if python_type is None:
                    raise UnknownCodecError(
                        f"unknown dataclass codec {node[1]!r} version {node[2]}"
                    )
                values = {name: walk(value, depth + 1) for name, value in node[3]}
                try:
                    return python_type(**values)
                except Exception as exc:
                    raise SerializationError(
                        f"dataclass construction failed: {exc}"
                    ) from exc
            if opcode == "custom":
                codec = self.registry.custom_by_id.get((node[1], node[2]))
                if codec is None:
                    raise UnknownCodecError(
                        f"unknown object codec {node[1]!r} version {node[2]}"
                    )
                try:
                    return codec.decoder(walk(node[3], depth + 1))
                except Exception as exc:
                    raise SerializationError(
                        f"custom object decoding failed: {exc}"
                    ) from exc
            raise SerializationError(f"unknown payload opcode {opcode!r}")

        return DecodedPayload(walk(root, 0), piggypayload)

    def _encode_metadata(
        self, piggypayload: Mapping[str, Any] | None
    ) -> Metadata | None:
        if piggypayload is None:
            return None
        if not isinstance(piggypayload, Mapping):
            raise SerializationError("piggypayload must be a mapping or None")
        metadata = dict(piggypayload)
        self._validate_decoded_metadata(metadata)
        encoded = msgpack.packb(metadata, use_bin_type=True)
        if len(encoded) > self.config.max_metadata_bytes:
            raise SerializationError("piggypayload exceeds configured metadata limit")
        return metadata

    def _validate_decoded_metadata(self, value: Any, depth: int = 0) -> None:
        if depth == 0:
            try:
                encoded_size = len(msgpack.packb(value, use_bin_type=True))
            except (OverflowError, TypeError, ValueError) as exc:
                raise SerializationError(
                    f"piggypayload is not MessagePack-safe: {exc}"
                ) from exc
            if encoded_size > self.config.max_metadata_bytes:
                raise SerializationError(
                    "piggypayload exceeds configured metadata limit"
                )
        if depth > 16:
            raise SerializationError("piggypayload exceeds maximum depth")
        if value is None or type(value) in (bool, int, float, str, bytes):
            return
        if type(value) is list:
            for item in value:
                self._validate_decoded_metadata(item, depth + 1)
            return
        if type(value) is dict:
            for key, item in value.items():
                if type(key) is not str:
                    raise SerializationError("piggypayload keys must be strings")
                self._validate_decoded_metadata(item, depth + 1)
            return
        raise SerializationError(
            f"unsupported piggypayload value type {type(value).__qualname__}"
        )

    def _encode_bytes(
        self, value: bytes | bytearray | memoryview, segments: list[BufferSegment]
    ) -> list[Any]:
        view = memoryview(value)
        if not view.c_contiguous:
            return self._append_owned_bytes("bytes", bytes(view), segments)
        if view.format != "B" or view.ndim != 1:
            view = view.cast("B")
        index = len(segments)
        segments.append(BufferSegment(view, value))
        return ["bytes", index, view.nbytes]

    @staticmethod
    def _append_owned_bytes(
        opcode: str, value: bytes, segments: list[BufferSegment]
    ) -> list[Any]:
        index = len(segments)
        segments.append(BufferSegment(memoryview(value), value))
        return [opcode, index, len(value)]

    def _encode_tensor(self, value: Any, segments: list[BufferSegment]) -> list[Any]:
        assert torch is not None
        if value.device.type != "cpu":
            raise UnsupportedPayloadError("only CPU tensors are supported")
        if value.layout != torch.strided or value.is_quantized:
            raise UnsupportedPayloadError(
                "only dense, non-quantized tensors are supported"
            )
        if not value.is_contiguous():
            raise UnsupportedPayloadError("tensor must be contiguous")
        supported_dtypes = {
            torch.uint8,
            torch.int8,
            torch.int16,
            torch.int32,
            torch.int64,
            torch.float16,
            torch.bfloat16,
            torch.float32,
            torch.float64,
            torch.bool,
        }
        if value.dtype not in supported_dtypes:
            raise UnsupportedPayloadError(f"unsupported tensor dtype {value.dtype}")

        nbytes = value.numel() * value.element_size()
        if nbytes == 0:
            view = memoryview(b"")
        else:
            raw_buffer = (ctypes.c_ubyte * nbytes).from_address(value.data_ptr())
            view = memoryview(raw_buffer).cast("B")
        index = len(segments)
        segments.append(BufferSegment(view, value))
        dtype = str(value.dtype).removeprefix("torch.")
        return ["tensor", index, nbytes, dtype, list(value.shape)]

    def _decode_tensor(
        self, node: list[Any], segments: tuple[BufferSegment, ...]
    ) -> Any:
        if torch is None:
            raise UnsupportedPayloadError(
                "received a Tensor but PyTorch is not installed; install wireless-comm[tensor]"
            )
        raw = self._segment(node, segments)
        dtype = getattr(torch, node[3], None)
        if dtype is None or not isinstance(dtype, torch.dtype):
            raise SerializationError(f"unknown tensor dtype {node[3]!r}")
        shape = tuple(node[4])
        if len(shape) > 64 or any(type(dim) is not int or dim < 0 for dim in shape):
            raise SerializationError("invalid tensor shape")
        expected = math.prod(shape) * torch.empty((), dtype=dtype).element_size()
        if expected != raw.nbytes:
            raise SerializationError(
                "tensor byte length does not match dtype and shape"
            )
        if expected == 0:
            return torch.empty(shape, dtype=dtype)
        try:
            return torch.frombuffer(bytearray(raw), dtype=dtype).reshape(shape)
        except Exception as exc:
            raise SerializationError(f"tensor reconstruction failed: {exc}") from exc

    @staticmethod
    def _segment(node: list[Any], segments: tuple[BufferSegment, ...]) -> memoryview:
        try:
            segment = segments[node[1]]
        except (IndexError, TypeError) as exc:
            raise SerializationError("payload segment index is out of range") from exc
        if segment.nbytes != node[2]:
            raise SerializationError("payload segment length is invalid")
        return segment.view
