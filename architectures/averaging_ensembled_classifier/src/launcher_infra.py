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
- Workload Primitives (WorkTile, ExecutionGrid): Utilities to define work.
- HostView: A utility for safe, padding-aware data reads from device to host.
- BufferManager: The sole authority on device memory allocation and lifecycle.
- PingPongManager: A stateful helper managing transient buffers for reductions.
- KernelSignature (ABC): The abstract contract for all kernel launch objects.
- KernelExecutor: The pure, stateless dispatcher that executes KernelSignatures.
"""

import abc
from dataclasses import dataclass
import enum
from typing import Dict, List, Optional, Tuple, Union

import numpy as np
import pyopencl as cl

# --- Local Imports ---
# This module relies on the MemoryLayout abstraction being in a separate, co-located file.
from .memory_layout import MemoryLayout


# --- Global Type Definitions ---
SCALAR_NP_TYPE = np.float32


# --- Canonical Lexicon & Data Structures (The Host-Side Contract) ---


# --- Standardized Dependency Bundle ---
@dataclass(frozen=True)
class Services:
    """A simple container for passing core system components for dependency injection."""

    q: cl.CommandQueue
    ex: KernelExecutor
    bm: BufferManager
    model_spec: ModelSpec
    arch_consts: Dict[str, int]


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
    SOFTMAX_PARAMS = enum.auto()


class HostView:
    """A helper class for safe, padding-aware reads from device to host."""

    def __init__(self, padded_shape: Tuple, dtype: np.dtype, real_shape: Tuple):
        self.padded_shape = padded_shape
        self.dtype = dtype
        self.real_shape = real_shape
        # Allocate a host-side buffer with the full padded shape
        self.host_data = np.empty(self.padded_shape, dtype=self.dtype)

    def enqueue_read(self, queue: cl.CommandQueue, cl_buffer: cl.Buffer, wait_for=None) -> cl.Event:
        """Enqueues a non-blocking copy from the device buffer to this host view."""
        return cl.enqueue_copy(queue, self.host_data, cl_buffer, wait_for=wait_for or [])

    def get(self) -> np.ndarray:
        """Returns a numpy array sliced to the real shape, effectively removing padding."""
        slicing = tuple(slice(0, dim) for dim in self.real_shape)
        return self.host_data[slicing] if slicing else self.host_data


# --- Memory Management Layer ---


class BufferManager:
    """Manages the lifecycle of ALL OpenCL buffers, enforcing memory contracts."""

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
        if name in self._name_to_handle:
            raise ValueError(f"Buffer with name '{name}' already exists.")
        padded_shape = layout.get_padded_shape(dtype)
        byte_size = int(np.prod(padded_shape) * dtype().itemsize) if padded_shape else 4
        handle = self._get_new_handle()
        self._name_to_handle[name] = handle
        self._handle_to_buffer[handle] = cl.Buffer(self._context, cl.mem_flags.READ_WRITE, size=max(4, byte_size))
        self._handle_to_spec[handle] = (padded_shape, dtype)
        return handle

    def acquire_transient_buffer(self, size_bytes: int) -> BufferHandle:
        handle = self._get_new_handle()
        self._handle_to_buffer[handle] = cl.Buffer(self._context, cl.mem_flags.READ_WRITE, size=max(4, size_bytes))
        return handle

    def release_transient_buffer(self, handle: BufferHandle):
        if handle in self._handle_to_buffer:
            self._handle_to_buffer[handle].release()
            del self._handle_to_buffer[handle]

    def get_cl_buffer(self, ref: Union[str, BufferHandle]) -> cl.Buffer:
        handle = self._name_to_handle.get(ref) if isinstance(ref, str) else ref
        if handle is None or handle not in self._handle_to_buffer:
            raise KeyError(f"No buffer found for reference: {ref}")
        return self._handle_to_buffer[handle]

    def get_handle_by_name(self, name: str) -> BufferHandle:
        """Resolves a string name to its unique, opaque BufferHandle."""
        if name not in self._name_to_handle:
            raise KeyError(f"No buffer found with name: {name}")
        return self._name_to_handle[name]

    def get_spec(self, ref: Union[str, BufferHandle]) -> Tuple[Tuple[int, ...], np.dtype]:
        handle = self._name_to_handle.get(ref) if isinstance(ref, str) else ref
        if handle is None or handle not in self._handle_to_spec:
            raise KeyError(f"No spec found for reference: {ref}")
        return self._handle_to_spec[handle]


class PingPongManager:
    """Manages a pair of recyclable 'ping-pong' buffers for a reduction."""

    def __init__(self):
        self._buffer_mgr: Optional[BufferManager] = None
        self.ping: Optional[BufferHandle] = None
        self.pong: Optional[BufferHandle] = None
        self._is_ping_current_input = True

    def initialize(self, buffer_mgr: BufferManager, max_bytes: int):
        """Lazily initializes the manager and acquires transient buffers."""
        if self.ping is not None or self.pong is not None:
            raise RuntimeError("PingPongManager is already initialized.")
        self._buffer_mgr = buffer_mgr
        self.ping = buffer_mgr.acquire_transient_buffer(max_bytes)
        self.pong = buffer_mgr.acquire_transient_buffer(max_bytes)

    def get_io(self) -> Tuple[BufferHandle, BufferHandle]:
        if self.ping is None or self.pong is None:
            raise RuntimeError("PingPongManager must be initialized before use.")
        return (self.ping, self.pong) if self._is_ping_current_input else (self.pong, self.ping)

    def swap(self):
        self._is_ping_current_input = not self._is_ping_current_input

    def release(self):
        if self._buffer_mgr and self.ping and self.pong:
            self._buffer_mgr.release_transient_buffer(self.ping)
            self._buffer_mgr.release_transient_buffer(self.pong)
        self.__init__()  # Reset to uninitialized state


# --- Kernel Launch Layer (The Abstract Contract and Pure Dispatcher) ---


class KernelSignature(abc.ABC):
    """Abstract base class for a kernel launch specification."""

    def __init__(self, buffer_mgr: BufferManager):
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
    """A pure, stateless dispatcher for KernelSignature objects."""

    def __init__(self, program: cl.Program):
        if not isinstance(program, cl.Program):
            raise TypeError("KernelExecutor requires a valid pyopencl.Program instance.")
        self.program = program

    def launch(
        self, queue: cl.CommandQueue, signature: KernelSignature, wait_for: Optional[List[cl.Event]] = None
    ) -> cl.Event:
        if not isinstance(signature, KernelSignature):
            raise TypeError("The 'signature' argument must be an instance of KernelSignature.")
        kernel = getattr(self.program, signature.kernel_name)
        global_size, local_size = signature.get_grid()
        kernel_args = signature.get_args()
        return kernel(queue, global_size, local_size, *kernel_args, wait_for=wait_for)
