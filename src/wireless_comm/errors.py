"""Public errors raised by :mod:`wireless_comm`."""


class CommError(Exception):
    """Base class for communication errors."""


class ConnectionClosedError(CommError):
    """A peer connection closed while an operation was in progress."""


class ConnectionFailedError(CommError):
    """A peer connection could not be established."""


class ProtocolError(CommError):
    """A peer sent an invalid or incompatible frame."""


class SerializationError(CommError):
    """A payload could not be encoded or decoded."""


class UnsupportedPayloadError(SerializationError):
    """The configured codecs cannot represent a payload value."""


class UnknownCodecError(SerializationError):
    """The receiver has not registered a required application codec."""


class MessageTooLargeError(CommError):
    """A message exceeds a configured resource limit."""


class OperationTimeoutError(CommError):
    """A send or receive operation timed out."""
