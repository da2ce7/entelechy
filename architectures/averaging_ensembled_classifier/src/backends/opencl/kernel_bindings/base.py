# src/backends/opencl/kernel_bindings/base.py
"""KernelBinding base class for OpenCL dispatch adapters (ADR-007).

A KernelBinding translates abstract plan-level parameters (buffer handles,
scalar values, tile index) into the concrete argument list required by
``clEnqueueNDRangeKernel`` and the corresponding dispatch geometry.

The four-method lifecycle mirrors the sequence the OpenCL runtime needs
for every dispatch:

    1. **Prepare?**            → ``prepare_dispatch()`` (optional)
    2. **Which kernel?**       → ``get_kernel_name()``
    3. **What grid shape?**    → ``compute_grid()``
    4. **What arguments?**     → ``marshal_args()``

The ``prepare_dispatch`` phase exists for Orchestration-tier resource
preparation that depends on binding-layer configuration (e.g., workgroup
size) and cannot be resolved at plan-construction time.  The base
implementation is a no-op; subclasses override when device-side setup
(schedule uploads, etc.) is required before argument marshaling.

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

    Lifecycle per dispatch:

        1. ``prepare_dispatch()`` — Orchestration-tier resource preparation
        2. ``compute_grid()``     — dispatch geometry
        3. ``marshal_args()``     — pure argument translation

    ``prepare_dispatch()`` is the designated location for device-side
    resource preparation that depends on binding-tier configuration
    (e.g., workgroup_size) and cannot be computed at plan-construction
    time.  ``marshal_args()`` SHALL be purely functional — no device
    writes, no queue operations, no mutation of shared state.

    Stateless by default — subclasses that carry configuration (e.g.
    ``workgroup_size``) should declare ``__slots__`` accordingly.
    """

    __slots__ = ()

    def prepare_dispatch(
        self,
        queue: cl.CommandQueue,
        get_buffer: Callable[[BufferHandle], cl.Buffer],
        buffer_bindings: dict[str, BufferHandle],
        scalar_params: dict[str, Any],
        tile_index: int,
    ) -> dict[str, Any]:
        """Orchestration-tier resource preparation (optional).

        Called exactly once before ``marshal_args`` for each tile dispatch.
        Returns a (potentially augmented) ``scalar_params`` dict whose
        additional keys are available to the subsequent ``marshal_args``
        and ``compute_grid`` calls.  The base implementation returns
        ``scalar_params`` unchanged.

        Implementations that perform device writes (e.g., uploading a
        computed schedule) MUST document the side effect.  The plan
        renderer guarantees:

          - The queue is valid and synchronized to the dispatch's
            dependency point.
          - All buffer handles in ``buffer_bindings`` are allocated.
          - ``prepare_dispatch()`` is called at most once per tile.

        Keys prefixed with ``_prepared_`` are reserved for binding-
        computed values that flow into ``marshal_args``.

        Parameters
        ----------
        queue:
            The active ``cl.CommandQueue`` for any required H2D copies.
        get_buffer:
            Resolves a ``BufferHandle`` to the underlying ``cl.Buffer``.
        buffer_bindings:
            Maps CONTRACT buffer parameter names to their ``BufferHandle``.
        scalar_params:
            Kernel scalars keyed by their CONTRACT name, plus underscore-
            prefixed orchestration metadata.
        tile_index:
            The tile index for this dispatch (Placement Contract key).

        Returns
        -------
        dict[str, Any]:
            The original dict, possibly augmented with binding-computed
            values (keyed with ``_prepared_`` prefix).
        """
        return scalar_params

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
