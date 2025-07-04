# launcher_infra.py

"""
The Definitive Infrastructure for the Contractual Kernel Launch Layer.

Jurisdictional Mandate:
This module provides the foundational, non-kernel-specific classes required to
orchestrate device computation. It is the system's "arsenal," containing the
tools and authorities that physicalize a computational plan. Its jurisdiction
is the management of device memory (`BufferManager`), the execution of commands
(`KernelExecutor`), and the definition of the abstract contracts (`KernelSignature`)
that govern their interaction.

Architectural Role:
This module upholds the principle of strict separation of concerns. It provides
the "physics" of the system, upon which the "chemistry" of the graph recipes can operate.
- The `BufferManager` is the sole authority on memory allocation, ensuring that
  all memory layouts are a direct, verifiable reflection of kernel contracts.
- The `KernelExecutor` is a pure, stateless dispatcher, guaranteeing that the
  act of launching a kernel is decoupled from the strategy of *what* to launch.
- The `KernelSignature` serves as the abstract blueprint, a sacred contract
  that ensures every kernel launch is verifiable and correct by design.
"""

import abc
from dataclasses import dataclass
import enum
from typing import Dict, List, Optional, Tuple, Union, Type

import numpy as np
import pyopencl as cl

# --- Foundational Imports from Sibling Modules ---
from .memory_layout import MemoryLayout
from .model_spec import ModelSpec
from .cl_context_manager import DiscoveredArchConstants


# =========================================================================
# === Canonical Lexicon & Data Structures (The Host-Side Contract)      ===
# =========================================================================


@dataclass(frozen=True)
class Services:
    """
    A humble vessel for passing core system components via dependency injection.
    It ensures that functions and objects receive all necessary context without
    relying on global state, making the system's data flows explicit and traceable.
    """
    q: cl.CommandQueue
    # Forward references are used here as a form of architectural courtesy,
    # preventing circular import dependencies at runtime.
    ex: "KernelExecutor"
    bm: "BufferManager"
    model_spec: ModelSpec
    arch_consts: DiscoveredArchConstants


@dataclass(frozen=True, eq=True)
class BufferHandle:
    """An opaque, immutable handle to a device memory buffer. It serves as a
    token of authority, granted by the BufferManager, abstracting away the
    raw details of memory addresses."""
    id: int


class HostView:
    """
    A utility for safe, padding-aware data retrieval from device to host.

    Architectural Mandate:
    This class solves a classic problem in high-performance computing: the
    discrepancy between a buffer's logical shape and its physical, padded
    in-memory representation. It performs the final, crucial act of "un-padding"
    on the host, presenting the user with a clean numpy array that reflects
    only the true, logical data.
    """

    def __init__(self, padded_shape: Tuple, dtype: Union[np.dtype, Type[np.floating]], real_shape: Tuple):
        self.padded_shape = padded_shape
        # WHY: This normalization step makes the constructor robust. It can
        # accept either a raw type (e.g., np.float32) or a dtype object,
        # ensuring consistent internal state.
        self.dtype = np.dtype(dtype)
        self.real_shape = real_shape
        # The host-side buffer is allocated with the full padded shape to
        # perfectly match the device-side source.
        self.host_data = np.empty(self.padded_shape, dtype=self.dtype)

    def enqueue_read(self, queue: cl.CommandQueue, cl_buffer: cl.Buffer, wait_for=None) -> cl.Event:
        """Enqueues a non-blocking copy from device buffer to this host view's memory."""
        return cl.enqueue_copy(queue, self.host_data, cl_buffer, wait_for=wait_for or [])

    def get(self) -> np.ndarray:
        """
        Returns a numpy array sliced to the real, logical shape. This is the
        final translation from physical representation to logical meaning.
        """
        slicing = tuple(slice(0, dim) for dim in self.real_shape)
        return self.host_data[slicing] if slicing else self.host_data


# =========================================================================
# === Memory Management Layer                                           ===
# =========================================================================


class BufferManager:
    """
    The sole and sovereign authority on device memory. This manager translates
    `MemoryLayout` contracts into physical buffers, tracks their lifecycle, and
    provides access via opaque `BufferHandle` tokens.
    """

    def __init__(self, context: cl.Context):
        self._context = context
        self._next_handle_id = 0
        self._handle_to_buffer: Dict[BufferHandle, cl.Buffer] = {}
        self._handle_to_spec: Dict[BufferHandle, Tuple[Tuple[int, ...], Type[np.floating]]] = {}
        # REFINEMENT: The lexicon is now internally consistent. This dictionary
        # contractually maps a string name to its unique BufferHandle object.
        self._name_to_handle: Dict[str, BufferHandle] = {}

    def _get_new_handle(self) -> BufferHandle:
        handle = BufferHandle(id=self._next_handle_id)
        self._next_handle_id += 1
        return handle

    def create_named_buffer(self, name: str, layout: MemoryLayout, dtype: Type[np.floating]) -> BufferHandle:
        """Creates a permanent, named buffer according to a memory layout plan."""
        if name in self._name_to_handle:
            raise ValueError(f"Buffer with name '{name}' already exists.")
        # The manager honors the layout contract, computing the padded shape.
        padded_shape = layout.get_padded_shape(np.dtype(dtype))
        byte_size = int(np.prod(padded_shape) * np.dtype(dtype).itemsize) if padded_shape else 4
        handle = self._get_new_handle()
        self._name_to_handle[name] = handle
        self._handle_to_buffer[handle] = cl.Buffer(self._context, cl.mem_flags.READ_WRITE, size=max(4, byte_size))
        # Only named buffers have a "spec" that is tracked by the manager.
        self._handle_to_spec[handle] = (padded_shape, dtype)
        return handle

    def acquire_transient_buffer(self, size_bytes: int) -> BufferHandle:
        """Acquires a temporary, unnamed buffer of a specified byte size."""
        handle = self._get_new_handle()
        self._handle_to_buffer[handle] = cl.Buffer(self._context, cl.mem_flags.READ_WRITE, size=max(4, size_bytes))
        return handle

    def release_transient_buffer(self, handle: BufferHandle):
        """Releases a temporary buffer. A contractually obligated cleanup step."""
        if handle in self._handle_to_buffer:
            self._handle_to_buffer[handle].release()
            del self._handle_to_buffer[handle]
        # REFINEMENT: Removed deletion from `_handle_to_spec`. Transient buffers
        # are not recorded there, making the check unnecessary and bringing the
        # logic into perfect alignment with its architectural purpose.

    def get_cl_buffer(self, ref: Union[str, BufferHandle]) -> cl.Buffer:
        """Retrieves the raw PyOpenCL buffer object for a given reference."""
        # This logic is now certifiably correct due to the refined type hint.
        handle = self._name_to_handle[ref] if isinstance(ref, str) else ref
        if handle not in self._handle_to_buffer:
            raise KeyError(f"No buffer found for reference: {ref}")
        return self._handle_to_buffer[handle]

    def get_handle_by_name(self, name: str) -> BufferHandle:
        """Retrieves the opaque BufferHandle for a given canonical name."""
        if name not in self._name_to_handle:
            raise KeyError(f"No buffer found with name: {name}")
        return self._name_to_handle[name]

    def get_spec(self, ref: Union[str, BufferHandle]) -> Tuple[Tuple[int, ...], Type[np.floating]]:
        """Retrieves the (padded_shape, dtype) spec for a named buffer."""
        handle = self._name_to_handle[ref] if isinstance(ref, str) else ref
        if handle not in self._handle_to_spec:
            # It is architecturally correct to raise an error for transient
            # buffers, as they have no predefined shape or type spec.
            raise KeyError(f"No spec found for reference: {ref}. (Is it a transient buffer?)")
        return self._handle_to_spec[handle]


class PingPongManager:
    """Manages a pair of recyclable 'ping-pong' buffers, typically for reductions."""

    def __init__(self):
        self._reset()

    def _reset(self):
        """A private helper to restore the manager to its initial, uninitialized state."""
        self._buffer_mgr: Optional[BufferManager] = None
        self.ping: Optional[BufferHandle] = None
        self.pong: Optional[BufferHandle] = None
        self._is_ping_current_input = True

    def initialize(self, buffer_mgr: BufferManager, max_bytes: int):
        """Acquires two transient buffers of a given size to begin operations."""
        if self.ping is not None or self.pong is not None:
            raise RuntimeError("PingPongManager is already initialized.")
        self._buffer_mgr = buffer_mgr
        self.ping = buffer_mgr.acquire_transient_buffer(max_bytes)
        self.pong = buffer_mgr.acquire_transient_buffer(max_bytes)

    def get_io(self) -> Tuple[BufferHandle, BufferHandle]:
        """Returns the current (input, output) pair of buffer handles."""
        if self.ping is None or self.pong is None:
            raise RuntimeError("PingPongManager must be initialized before use.")
        return (self.ping, self.pong) if self._is_ping_current_input else (self.pong, self.ping)

    def swap(self):
        """Swaps the input and output roles of the internal buffers."""
        self._is_ping_current_input = not self._is_ping_current_input

    def release(self):
        """Releases the managed transient buffers and resets the manager."""
        if self._buffer_mgr:
            if self.ping: self._buffer_mgr.release_transient_buffer(self.ping)
            if self.pong: self._buffer_mgr.release_transient_buffer(self.pong)
        self._reset()


# =========================================================================
# === Kernel Launch Layer (The Abstract Contract and Pure Dispatcher)   ===
# =========================================================================


class KernelSignature(abc.ABC):
    """
    The abstract contract for a kernel launch. An instance of a concrete
    signature is a complete, self-contained, and verifiable specification for a
    single kernel dispatch.
    """
    def __post_init__(self):
        # WHY: This exists to provide a no-op `__post_init__` for subclasses,
        # so they do not need to call `super()` if the base class behavior
        # is ever extended. It is a gesture of forward-looking design.
        pass

    @property
    @abc.abstractmethod
    def kernel_name(self) -> str:
        """The exact name of the kernel function in the OpenCL program."""
        pass

    @abc.abstractmethod
    def get_grid(self) -> Tuple[Tuple[int, ...], Optional[Tuple[int, ...]]]:
        """Returns the (global_size, local_size) execution grid tuple."""
        pass

    @abc.abstractmethod
    def get_args(self) -> List:
        """Returns the list of arguments in the exact order required by the kernel contract."""
        pass


class KernelExecutor:
    """A pure, stateless dispatcher. Its sole function is to execute a `KernelSignature`."""

    def __init__(self, program: cl.Program):
        if not isinstance(program, cl.Program):
            raise TypeError("KernelExecutor requires a valid pyopencl.Program instance.")
        self.program = program

    def launch(
        self, queue: cl.CommandQueue, signature: KernelSignature, wait_for: Optional[List[cl.Event]] = None
    ) -> cl.Event:
        """Executes a single kernel launch based on a signature object."""
        if not isinstance(signature, KernelSignature):
            raise TypeError("The 'signature' argument must be an instance of a KernelSignature subclass.")
        kernel = getattr(self.program, signature.kernel_name)
        global_size, local_size = signature.get_grid()
        kernel_args = signature.get_args()
        # The robust `wait_for or []` idiom handles the optional event list.
        return kernel(queue, global_size, local_size, *kernel_args, wait_for=wait_for or [])
