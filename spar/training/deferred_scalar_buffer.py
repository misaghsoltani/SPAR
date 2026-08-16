"""Device-side buffering for scalar training metrics."""

from collections.abc import Iterable
from typing import Protocol

import torch
from torch import Tensor


class FloatMetricSink(Protocol):
    """Destination that can extend itself with floating-point metrics."""

    def extend(self, values: Iterable[float], /) -> None:
        """Append floating-point metrics.

        Args:
            values: Scalar values to append.
        """
        ...


class DeferredScalarBuffer:
    """Buffer scalar tensors until an observable metrics boundary."""

    def __init__(self, capacity: int) -> None:
        """Initialize an empty fixed-capacity buffer.

        Args:
            capacity: Maximum number of scalar values between flushes.

        Raises:
            ValueError: If capacity is less than one.
        """
        if capacity < 1:
            raise ValueError("capacity must be at least one")

        self._capacity = capacity
        self._buffer: Tensor | None = None
        self._size = 0

    def append(self, value: Tensor) -> None:
        """Copy one detached scalar into the device buffer.

        Args:
            value: A tensor containing exactly one scalar value.

        Raises:
            ValueError: If value does not contain exactly one element.
            RuntimeError: If the buffer is full or its device or dtype changes
                before a flush.
        """
        if value.numel() != 1:
            raise ValueError("value must contain exactly one element")

        buffer = self._buffer
        if buffer is None or (self._size == 0 and (buffer.device != value.device or buffer.dtype != value.dtype)):
            buffer = torch.empty(self._capacity, device=value.device, dtype=value.dtype)
            self._buffer = buffer

        if buffer.device != value.device or buffer.dtype != value.dtype:
            raise RuntimeError("scalar device and dtype must remain stable between flushes")
        if self._size >= self._capacity:
            raise RuntimeError("scalar buffer capacity exceeded before flush")

        buffer[self._size].copy_(value.detach().reshape(()))
        self._size += 1

    def flush_into(self, target: FloatMetricSink) -> None:
        """Transfer buffered values to a Python list in insertion order.

        Args:
            target: Destination list for the scalar values.
        """
        if self._size == 0:
            return

        buffer = self._buffer
        if buffer is None:
            raise RuntimeError("nonempty scalar buffer has no storage")

        host_values = buffer[: self._size].to(device="cpu")
        target.extend(float(value) for value in host_values)
        self._size = 0
