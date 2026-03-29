"""Zero-copy CPURetrievalFuture (ADR-010).

On CPU, all computation is synchronous — by the time render() returns,
results are already in host memory. No D2H transfer is needed.
"""
from __future__ import annotations

import numpy as np
from numpy.typing import NDArray


class CPURetrievalFuture:
    """Zero-copy RetrievalFuture for the CPU backend.

    .wait() is a no-op (CPU dispatch is blocking).
    .result() returns an unpadded view of the output buffer.
    """

    def __init__(
        self,
        node_id: str,
        padded_buffer: np.ndarray,
        logical_shape: tuple[int, ...],
    ) -> None:
        self._node_id = node_id
        self._padded_buffer = padded_buffer
        self._logical_shape = logical_shape
        self._released = False

    @property
    def node_id(self) -> str:
        return self._node_id

    def wait(self) -> None:
        """No-op — CPU computation is synchronous."""

    def result(self) -> NDArray[np.floating]:
        """Return unpadded numpy array with logical shape."""
        if self._released or self._padded_buffer is None:
            raise RuntimeError(
                f"RetrievalFuture for '{self._node_id}' has been released"
            )
        # Reshape to logical shape by slicing off padding
        total_logical = 1
        for d in self._logical_shape:
            total_logical *= d
        flat = self._padded_buffer.ravel()[:total_logical]
        return flat.reshape(self._logical_shape)

    def release(self) -> None:
        """Release the reference to the backing buffer."""
        self._padded_buffer = None  # type: ignore[assignment]
        self._released = True
