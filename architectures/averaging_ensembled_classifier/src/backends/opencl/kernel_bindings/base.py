# src/backends/opencl/kernel_bindings/base.py
"""KernelBinding base class for OpenCL dispatch adapters (ADR-007).

A KernelBinding translates abstract plan-level parameters (buffer handles,
scalar values, tile index) into the concrete argument list required by
``clEnqueueNDRangeKernel`` and the corresponding dispatch geometry.

The three-method protocol mirrors the three questions the OpenCL runtime
needs answered for every dispatch:

    1. **Which kernel?**        → ``get_kernel_name()``
    2. **What grid shape?**     → ``compute_grid()``
    3. **What arguments?**      → ``marshal_args()``

Subclasses MAY additionally expose ``marshal_args_direct`` /
``compute_grid_direct`` convenience methods for the Orchestration tier's
reduction-tree renderer, which composes kernels programmatically rather
than through plan nodes.  These are not part of the base protocol because
their signatures are kernel-specific.

Reference: CONTRACT.md §7 (Kernel Contract Model — KernelBinding).
"""
from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any, Callable

import pyopencl as cl

from ....shared.buffer_lifecycle import BufferHandle


class KernelBinding(ABC):
    """Abstract base for OpenCL kernel dispatch adapters.

    Stateless by default — subclasses that carry configuration (e.g.
    ``workgroup_size``) should declare ``__slots__`` accordingly.
    """

    __slots__ = ()

    @abstractmethod
    def get_kernel_name(self) -> str:
        """Return the kernel function name as declared in ``kernels.cl.h``.

        This is the lookup key against the compiled ``cl.Program``.
        """
        ...

    @abstractmethod
    def compute_grid(
        self,
        tile_index: int,
        scalar_params: dict[str, Any],
        hardware_simd_width: int,
    ) -> tuple[tuple[int, ...], tuple[int, ...] | None]:
        """Compute ``(global_size, local_size)`` for a single tile dispatch.

        Parameters
        ----------
        tile_index:
            The tile index for this dispatch (Placement Contract key).
            Ignored by kernels whose grid is tile-independent.
        scalar_params:
            Kernel scalars keyed by their CONTRACT name, plus
            underscore-prefixed orchestration metadata
            (``_compute_type_size_bytes``, ``_compute_dtype``, etc.).
        hardware_simd_width:
            ``HardwareProfile.simd_width`` — the SIMD lane count that
            governs work-group sizing for kernels with explicit local
            dimensions.

        Returns
        -------
        (global_size, local_size):
            ``local_size`` is ``None`` when the backend should select
            the work-group size autonomously.
        """
        ...

    @abstractmethod
    def marshal_args(
        self,
        get_buffer: Callable[[BufferHandle], cl.Buffer],
        buffer_bindings: dict[str, BufferHandle],
        scalar_params: dict[str, Any],
        tile_index: int,
    ) -> list[Any]:
        """Marshal the ordered argument list for ``clEnqueueNDRangeKernel``.

        Arguments are returned in **kernel signature order** as declared
        in ``kernels.cl.h``.  Local-memory allocations appear as
        ``cl.LocalMemory(nbytes)`` at the corresponding position.

        Parameters
        ----------
        get_buffer:
            Resolves a ``BufferHandle`` to the underlying ``cl.Buffer``.
        buffer_bindings:
            Maps CONTRACT buffer parameter names to their ``BufferHandle``.
        scalar_params:
            Kernel scalars keyed by their CONTRACT name, plus
            underscore-prefixed orchestration metadata.
        tile_index:
            The tile index for this dispatch (Placement Contract key).
            Delivered as ``src_scalar_NATURAL_flat_tile_index`` to
            Partial Renderer kernels; ignored by others.
        """
        ...

    def __repr__(self) -> str:
        return f"{type(self).__name__}(kernel={self.get_kernel_name()!r})"
