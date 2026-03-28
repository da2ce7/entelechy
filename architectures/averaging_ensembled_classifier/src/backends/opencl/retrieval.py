# src/backends/opencl/retrieval.py
"""OpenCL implementation of the RetrievalFuture protocol (ADR-010)."""
from __future__ import annotations

from typing import Any

import numpy as np
import pyopencl as cl
from numpy.typing import NDArray


class OpenCLRetrievalFuture:
    """OpenCL implementation of RetrievalFuture (ADR-010).

    Wraps a cl.Event from an async D2H enqueue_read. The renderer
    constructs this when processing a RetrievalNode.
    """

    def __init__(
        self,
        node_id: str,
        event: cl.Event,
        host_buffer: NDArray[np.floating[Any]],
        logical_shape: tuple[int, ...],
        padded_shape: tuple[int, ...],
    ) -> None:
        self._node_id = node_id
        self._event: cl.Event | None = event
        self._host_buffer: NDArray[np.floating[Any]] | None = host_buffer
        self._logical_shape = logical_shape
        self._padded_shape = padded_shape
        self._released = False
        self._result_cache: NDArray[np.floating[Any]] | None = None

    @property
    def node_id(self) -> str:
        return self._node_id

    @property
    def event(self) -> cl.Event:
        """The underlying OpenCL event for dependency tracking."""
        assert self._event is not None, "event accessed after release()"
        return self._event

    def wait(self) -> None:
        if not self._released and self._event is not None:
            self._event.wait()

    def result(self) -> NDArray[np.floating[Any]]:
        if self._released:
            raise RuntimeError("RetrievalFuture has been released")
        if self._result_cache is None:
            self.wait()
            # Strip padding: extract logical shape from padded host buffer
            assert self._host_buffer is not None
            reshaped = self._host_buffer.reshape(self._padded_shape)
            slices = tuple(slice(0, s) for s in self._logical_shape)
            self._result_cache = np.array(reshaped[slices], copy=True)
        return self._result_cache

    def release(self) -> None:
        self._released = True
        self._host_buffer = None
        self._result_cache = None
        self._event = None
