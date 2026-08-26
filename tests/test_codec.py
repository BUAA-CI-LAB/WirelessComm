from __future__ import annotations

import ctypes
import dataclasses
import unittest

from wireless_comm.codec import CodecRegistry, PayloadCodec
from wireless_comm.errors import SerializationError, UnsupportedPayloadError
from wireless_comm.types import CommConfig


@dataclasses.dataclass
class Observation:
    name: str
    values: list[int]


class UnregisteredObject:
    def __init__(self, value: int) -> None:
        self.value = value

    def __eq__(self, other: object) -> bool:
        return isinstance(other, UnregisteredObject) and other.value == self.value


class RegisteredObject:
    def __init__(self, value: int) -> None:
        self.value = value

    def __eq__(self, other: object) -> bool:
        return isinstance(other, RegisteredObject) and other.value == self.value


class CodecTests(unittest.TestCase):
    def setUp(self) -> None:
        self.registry = CodecRegistry()
        self.codec = PayloadCodec(self.registry, CommConfig())

    def test_nested_payload_and_metadata_round_trip(self) -> None:
        payload = {
            "tuple": (1, True, None),
            "list": [b"bytes", {"value": 3.5}],
        }
        encoded = self.codec.encode(payload, {"trace_id": "abc", "attempt": 2})
        decoded = self.codec.decode(encoded)
        self.assertEqual(decoded.object, payload)
        self.assertEqual(decoded.piggypayload, {"trace_id": "abc", "attempt": 2})

    def test_registered_dataclass_round_trip(self) -> None:
        self.registry.register_dataclass(
            Observation,
            type_id="tests.Observation",
        )
        value = Observation("sample", [1, 2, 3])
        self.assertEqual(
            self.codec.decode(self.codec.encode(value, None)).object, value
        )

    def test_unregistered_object_is_rejected(self) -> None:
        with self.assertRaises(UnsupportedPayloadError):
            self.codec.encode(UnregisteredObject(3), None)

    def test_registered_object_round_trip(self) -> None:
        self.registry.register_codec(
            RegisteredObject,
            type_id="tests.RegisteredObject",
            version=1,
            encoder=lambda value: {"value": value.value},
            decoder=lambda value: RegisteredObject(value["value"]),
        )
        value = RegisteredObject(5)
        self.assertEqual(
            self.codec.decode(self.codec.encode(value, None)).object, value
        )

    def test_explicit_trusted_pickle_round_trip(self) -> None:
        codec = PayloadCodec(
            CodecRegistry(),
            CommConfig(allow_unsafe_pickle=True),
        )
        value = UnregisteredObject(7)
        self.assertEqual(codec.decode(codec.encode(value, None)).object, value)

    def test_piggypayload_is_small_metadata_only(self) -> None:
        codec = PayloadCodec(
            CodecRegistry(),
            CommConfig(max_metadata_bytes=16),
        )
        with self.assertRaises(SerializationError):
            codec.encode("payload", {"large": "x" * 64})
        with self.assertRaises(SerializationError):
            codec.encode("payload", {"tensor-like": object()})

    def test_cycles_are_rejected(self) -> None:
        value: list[object] = []
        value.append(value)
        with self.assertRaises(SerializationError):
            self.codec.encode(value, None)


try:
    import torch
except ImportError:
    torch = None


@unittest.skipIf(torch is None, "PyTorch is not installed")
class TensorCodecTests(unittest.TestCase):
    def setUp(self) -> None:
        self.codec = PayloadCodec(CodecRegistry(), CommConfig())

    def test_tensor_list_and_dict_round_trip(self) -> None:
        first = torch.arange(12, dtype=torch.float32).reshape(3, 4)
        second = torch.tensor([1, 2, 3], dtype=torch.int64)
        encoded = self.codec.encode({"items": [first, second]}, None)

        self.assertIs(encoded.segments[0].owner, first)
        decoded = self.codec.decode(encoded).object
        self.assertTrue(torch.equal(decoded["items"][0], first))
        self.assertTrue(torch.equal(decoded["items"][1], second))

    def test_tensor_segment_points_at_data_ptr(self) -> None:
        tensor = torch.arange(8, dtype=torch.float32)
        encoded = self.codec.encode(tensor, None)
        segment_address = ctypes.addressof(
            ctypes.c_ubyte.from_buffer(encoded.segments[0].view)
        )
        self.assertEqual(segment_address, tensor.data_ptr())

    def test_supported_tensor_shapes_and_dtypes(self) -> None:
        dtypes = (
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
        )
        for dtype in dtypes:
            with self.subTest(dtype=dtype):
                tensor = torch.tensor([0, 1, 1, 0], dtype=dtype)
                decoded = self.codec.decode(self.codec.encode(tensor, None)).object
                self.assertEqual(decoded.dtype, dtype)
                self.assertTrue(torch.equal(decoded, tensor))

        scalar = torch.tensor(3.25, dtype=torch.float32)
        empty = torch.empty((2, 0, 3), dtype=torch.float32)
        self.assertTrue(
            torch.equal(
                self.codec.decode(self.codec.encode(scalar, None)).object, scalar
            )
        )
        self.assertEqual(
            self.codec.decode(self.codec.encode(empty, None)).object.shape,
            empty.shape,
        )

    def test_non_contiguous_tensor_is_rejected(self) -> None:
        tensor = torch.arange(12).reshape(3, 4).transpose(0, 1)
        self.assertFalse(tensor.is_contiguous())
        with self.assertRaises(UnsupportedPayloadError):
            self.codec.encode(tensor, None)
