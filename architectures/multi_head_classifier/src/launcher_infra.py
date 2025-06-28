# launcher_infra.py

"""
The Definitive Infrastructure for the Contractual Kernel Launch Layer.

This module provides the foundational, non-kernel-specific classes required
to orchestrate device computation. Its design is a physical manifestation of
the principles of architectural elegance, contractual obligation, and strict
separation of concerns.

The `BufferManager` in this file has been explicitly refactored to consume
the `MemoryLayout` abstraction, removing all internal, implicit padding logic
and instead relying on explicit instructions from the orchestrator.

Core Components:
- BufferHandle & Enums: The canonical, type-safe lexicon for the host.
- MemoryLayout Abstractions: Imported tools for explicit memory planning.
- BufferManager: The sole authority on device memory allocation and lifecycle.
- KernelSignature (ABC): The abstract contract for all kernel launch objects.
- KernelExecutor: The pure, stateless dispatcher that executes KernelSignatures.
- PingPongManager: A stateful helper managing transient buffers for reductions.
"""

import abc
from dataclasses import dataclass
import enum
from typing import Dict, List, Optional, Tuple, Union

import numpy as np
import pyopencl as cl

# --- Local Imports ---
# This module relies on the MemoryLayout abstraction being in a separate, co-located file.
from memory_layout import MemoryLayout, PaddingType, PaddingStrategy


# --- Global Type Definitions ---
# This establishes the abstract numerical type used throughout the system.
# Changing this one line is sufficient to re-target the system's precision.
SCALAR_NP_TYPE = np.float32


# --- Canonical Lexicon & Data Structures (The Host-Side Contract) ---


@dataclass(frozen=True, eq=True)
class BufferHandle:
    """An opaque, immutable handle to a memory buffer managed by the BufferManager."""

    id: int


class BufferRole(enum.Enum):
    """Programmatic enforcement of the Canonical Lexicon (System Contract: Art. 8)."""

    # Core Data
    INPUT = enum.auto()
    HIDDEN_ACTIVATION = enum.auto()
    LOGITS = enum.auto()
    PROBS = enum.auto()
    LOSS = enum.auto()
    TARGETS = enum.auto()
    SAMPLE_MASK = enum.auto()
    HIDDEN_MASK = enum.auto()
    # Learnable Parameters
    SHARED_WEIGHTS = enum.auto()
    SHARED_BIAS = enum.auto()
    MODULE_WEIGHTS = enum.auto()
    MODULE_BIAS = enum.auto()
    TEMPERATURES = enum.auto()
    # Gradients & Optimizer State
    PARTIAL_GRADIENT = enum.auto()
    CLIPPED_GRADIENT = enum.auto()
    SUMMED_GRADIENT = enum.auto()
    FINAL_GRADIENT = enum.auto()
    ADAM_M1 = enum.auto()
    ADAM_M2 = enum.auto()
    # Specialized Intermediates
    PERMUTED_GRAD_H = enum.auto()


@dataclass(frozen=True)
class WorkTile:
    """Defines a single, independent unit of work for tiled kernels.
    This object encapsulates the scalar parameters that define a chunk.
    """

    flat_tile_index: int
    module_chunk_index: int
    class_chunk_index: int
    num_class_chunks: int
    modules_per_chunk: int
    classes_per_chunk: int


# --- Memory Management Layer ---


class BufferManager:
    """Manages the lifecycle of ALL OpenCL buffers, enforcing memory contracts.

    This class is the sole authority for creating, accessing, and releasing
    device memory. Its design has been refactored to be a "dumb worker" that
    executes explicit `MemoryLayout` plans provided by the orchestrator,
    abolishing all internal, implicit padding logic.
    """

    def __init__(self, context: cl.Context):
        self._context = context
        self._next_handle_id = 0
        self._handle_to_buffer: Dict[BufferHandle, cl.Buffer] = {}
        self._handle_to_spec: Dict[BufferHandle, Tuple[Tuple[int, ...], np.dtype]] = {}
        self._name_to_handle: Dict[str, BufferHandle] = {}

    def _get_new_handle(self) -> BufferHandle:
        handle = BufferHandle(id=self._next_handle_id)
        self._next_handle_id += 1
        return handle

    def create_named_buffer(self, name: str, layout: MemoryLayout, dtype: np.dtype) -> BufferHandle:
        """Creates a named, long-lived buffer based on an EXPLICIT layout plan."""
        if name in self._name_to_handle:
            raise ValueError(f"Buffer with name '{name}' already exists.")
        if not isinstance(layout, MemoryLayout):
            raise TypeError("`layout` argument must be an instance of MemoryLayout.")

        # The BufferManager is now a simple worker. It executes the plan it is given.
        padded_shape = layout.get_padded_shape(dtype)
        byte_size = int(np.prod(padded_shape) * dtype().itemsize) if padded_shape else 4

        handle = self._get_new_handle()
        self._name_to_handle[name] = handle
        self._handle_to_buffer[handle] = cl.Buffer(self._context, cl.mem_flags.READ_WRITE, size=max(4, byte_size))
        self._handle_to_spec[handle] = (padded_shape, dtype)
        return handle

    def acquire_transient_buffer(self, size_bytes: int) -> BufferHandle:
        """Acquires an unnamed, temporary buffer. Used for intermediates like reduction trees."""
        handle = self._get_new_handle()
        self._handle_to_buffer[handle] = cl.Buffer(self._context, cl.mem_flags.READ_WRITE, size=max(4, size_bytes))
        return handle

    def release_transient_buffer(self, handle: BufferHandle):
        """Returns a transient buffer, allowing its memory to be reclaimed by the driver."""
        if handle in self._handle_to_buffer:
            self._handle_to_buffer[handle].release()
            del self._handle_to_buffer[handle]
        if handle in self._handle_to_spec:
            del self._handle_to_spec[handle]

    def get_cl_buffer(self, ref: Union[str, BufferHandle]) -> cl.Buffer:
        """The single gateway to resolve a reference to a cl.Buffer object."""
        handle = self._name_to_handle.get(ref) if isinstance(ref, str) else ref
        if handle is None or handle not in self._handle_to_buffer:
            raise KeyError(f"No buffer found for reference: {ref}")
        return self._handle_to_buffer[handle]

    def get_spec(self, ref: Union[str, BufferHandle]) -> Tuple[Tuple[int, ...], np.dtype]:
        """Returns the (padded_shape, dtype) specification for a named buffer."""
        handle = self._name_to_handle.get(ref) if isinstance(ref, str) else ref
        if handle is None or handle not in self._handle_to_spec:
            raise KeyError(f"No spec found for named reference: {ref}")
        return self._handle_to_spec[handle]


class PingPongManager:
    """Manages a pair of recyclable 'ping-pong' buffers for a reduction."""

    def __init__(self, buffer_mgr: BufferManager, max_bytes: int):
        self._buffer_mgr = buffer_mgr
        self.ping: BufferHandle = buffer_mgr.acquire_transient_buffer(max_bytes)
        self.pong: BufferHandle = buffer_mgr.acquire_transient_buffer(max_bytes)
        self._is_ping_current_input = True

    def get_io(self) -> Tuple[BufferHandle, BufferHandle]:
        """Returns the current (input, output) buffer handles."""
        return (self.ping, self.pong) if self._is_ping_current_input else (self.pong, self.ping)

    def swap(self):
        """Swaps the input/output roles of the buffers for the next reduction stage."""
        self._is_ping_current_input = not self._is_ping_current_input

    def release(self):
        """Releases the managed buffers back to the BufferManager."""
        self._buffer_mgr.release_transient_buffer(self.ping)
        self._buffer_mgr.release_transient_buffer(self.pong)


# --- Kernel Launch Layer (The Abstract Contract and Pure Dispatcher) ---


class KernelSignature(abc.ABC):
    """Abstract base class for a kernel launch specification.

    This object is a pure, self-contained, and contractually valid representation
    of a single kernel launch. It holds all arguments and is responsible for
    calculating its own launch grid. Its existence enforces correctness by design.
    """

    def __init__(self, buffer_mgr: BufferManager):
        # A read-only reference to the buffer manager is required for signatures
        # to derive their own parameters from buffer specifications.
        if not isinstance(buffer_mgr, BufferManager):
            raise TypeError("KernelSignature requires a valid BufferManager instance.")
        self._buffer_mgr = buffer_mgr

    @property
    @abc.abstractmethod
    def kernel_name(self) -> str:
        """The exact name of the kernel in the OpenCL program."""
        pass

    @abc.abstractmethod
    def get_grid(self) -> Tuple[Tuple[int, ...], Optional[Tuple[int, ...]]]:
        """Returns the (global_size, local_size) execution grid."""
        pass

    @abc.abstractmethod
    def get_args(self) -> List:
        """Returns the list of arguments in the exact order required by the kernel."""
        pass


class KernelExecutor:
    """A pure, stateless dispatcher for KernelSignature objects.

    This class has no knowledge of problem dimensions, buffer names, or
    orchestration logic. Its sole purpose is to accept a fully-formed
    KernelSignature and dispatch it to the device.
    """

    def __init__(self, program: cl.Program):
        if not isinstance(program, cl.Program):
            raise TypeError("KernelExecutor requires a valid pyopencl.Program instance.")
        self.program = program

    def launch(
        self, queue: cl.CommandQueue, signature: KernelSignature, wait_for: Optional[List[cl.Event]] = None
    ) -> cl.Event:
        """Executes a single, fully-defined kernel launch."""
        if not isinstance(signature, KernelSignature):
            raise TypeError("The 'signature' argument must be an instance of KernelSignature.")

        kernel = getattr(self.program, signature.kernel_name)
        global_size, local_size = signature.get_grid()
        kernel_args = signature.get_args()

        return kernel(queue, global_size, local_size, *kernel_args, wait_for=wait_for)
