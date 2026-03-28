# src/backends/opencl/kernel_bindings/base.py
"""KernelBinding base class for OpenCL dispatch adapters (ADR-007)."""
from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any, Callable

import numpy as np
import pyopencl as cl

from ....shared.buffer_lifecycle import BufferHandle


class KernelBinding(ABC):
    """Base class for OpenCL kernel dispatch adapters (ADR-007).

    Translates abstract plan-level parameters (buffer handles, scalar
    values, tile index) into concrete OpenCL kernel arguments and
    dispatch dimensions.
    """

    @abstractmethod
    def get_kernel_name(self) -> str:
        """Return the kernel function name as it appears in kernels.cl.h."""
        ...

    @abstractmethod
    def compute_grid(
        self,
        tile_index: int,
        scalar_params: dict[str, int | float],
        hardware_simd_width: int,
    ) -> tuple[tuple[int, ...], tuple[int, ...] | None]:
        """Compute (global_size, local_size) for one tile dispatch."""
        ...

    @abstractmethod
    def marshal_args(
        self,
        get_buffer: Callable[[BufferHandle], cl.Buffer],
        buffer_bindings: dict[str, BufferHandle],
        scalar_params: dict[str, int | float],
        tile_index: int,
    ) -> list[Any]:
        """Marshal the ordered argument list for clEnqueueNDRange."""
        ...
