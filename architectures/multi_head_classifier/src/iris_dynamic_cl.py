# iris_dynamic_cl.py
#
# A Unified, Memory-Aware Streaming Classification Engine
# =======================================================
#
# This script is the complete host-side implementation of a sophisticated, event-driven
# training and inference engine for a neural network architecture. It is designed around
# four core principles:
#
# 1. Primacy of Memory Strategy: All orchestration is geared towards maximizing GPU
#    throughput by ensuring computations run in the fastest possible memory tier.
# 2. Modular, "Dumb" Kernels: The engine uses a set of simple, single-purpose OpenCL
#    kernels that are composed by the host to perform complex tasks.
# 3. Trust the Driver: The host constructs a Directed Acyclic Graph (DAG) of operations
#    and trusts the OpenCL driver to optimize scheduling, fusion, and execution.
# 4. Unified Dataflow: A single, "always stream" pipeline handles problems of any
#    scale, from tiny to massive, without separate, hard-coded logic paths.
#
# The host-side code is organized into a clean, four-layer architecture:
#
# Layer 4: Orchestration (TrainingOrchestrator)
#   - The top-level owner of all components.
#   - Manages the training loop and global state (e.g., epoch count).
#
# Layer 3: Execution (BatchProcessor)
#   - A stateful, single-use engine responsible for processing one batch.
#   - Translates a strategic plan into a concrete, event-based DAG of kernel launches.
#
# Layer 2: Strategy (ExecutionStrategy)
#   - A stateless component that assesses resource constraints (e.g., VRAM).
#   - Creates a high-level `ExecutionPlan` for how a batch should be processed.
#
# Layer 1: Action (BufferManager, KernelExecutor)
#   - Stateless toolboxes that perform primitive actions.
#   - `BufferManager`: Manages the lifecycle and memory layout of all GPU buffers.
#   - `KernelExecutor`: Provides a direct, 1:1 mapping for launching each OpenCL kernel.
#

import os
import pyopencl as cl
import numpy as np
import math
from collections import defaultdict
from dataclasses import dataclass
from typing import Tuple, List, Dict, Optional, Callable
import enum

# --- Global Constants & Configuration ---
# --- SCALAR PRECISION ---
SCALAR_TYPE = "half"
SCALAR_NP_TYPE = np.float16 if SCALAR_TYPE == "half" else np.float32
CL_SCALAR_TYPE = "half" if SCALAR_TYPE == "half" else "float"

# --- CORE NETWORK ARCHITECTURE ---
INPUT_DIM: int = 4
HIDDEN_DIM: int = 64
OUTPUT_CLASSES: int = 3
NUM_MODULES: int = 128
BATCH_SIZE: int = 150
EPOCHS: int = 50

# --- HYPERPARAMETERS & RUNTIME TUNING ---
PROBLEM_TYPE = "CCE"  # Cross-Entropy ("CCE") or Binary Cross-Entropy ("BCE")
LEARNING_RATE: float = 0.001
ADAM_BETA1: float = 0.9
ADAM_BETA2: float = 0.999
EPSILON: float = 1.0e-8
MIN_TEMP: float = 0.1
MAX_TEMP: float = 10.0

# --- COMPILE-TIME KERNEL CONSTANTS & ARCHITECTURAL TUNING ---
MAX_REGISTER_AGGREGATE_ITEMS: int = 32  # Threshold to switch reduction strategy
C_TILE_SIZE: int = 16  # Tile size for the matrix transpose kernel
BACKPROP_STREAM_CHUNK_SIZE: int = 32  # Granularity for shared layer backprop
## Max chunk constants for pre-allocating partial gradient buffers.
# This single value now defines the max number of tiles
MAX_TOTAL_TILES: int = 4096  # e.g., 64x64 grid
MAX_BATCH_CHUNKS: int = 64  # Max number of chunks for batch dimension streaming


# --- Core Data Abstractions ---


class LayoutType(enum.Enum):
    """Defines the physical memory layout strategy for a buffer."""

    AoS = 1  # Array of Structs: Standard, logical row-major layout
    SoA = 2  # Struct of Arrays: SIMD-optimized, transformed layout


class BufferRole(enum.Enum):
    """Defines the semantic purpose of a buffer, driving its memory padding strategy."""

    INPUT = 1
    HIDDEN_ACTIVATION = 2
    SHARED_WEIGHTS = 3
    SHARED_BIAS = 4
    MODULE_WEIGHTS = 5
    MODULE_BIAS = 6
    TEMPERATURES = 7
    PARTIAL_GRADIENT = 8
    FINAL_GRADIENT = 9
    ADAM_MOMENTUM = 10
    INTERMEDIATE = 11  # For transient buffers like logits, softmax params
    TARGETS = 12
    MASK = 13


@dataclass(frozen=True)
class Parameter:
    """A semantic group for a trainable parameter and its related buffers."""

    name: str
    role: BufferRole
    layout: LayoutType = LayoutType.AoS  # Default to standard layout

    @property
    def grad(self) -> str:
        return f"grad_{self.name}"

    @property
    def m1(self) -> str:
        return f"m1_{self.name}"

    @property
    def m2(self) -> str:
        return f"m2_{self.name}"


class HostView:
    """Manages a host-side numpy array for reading data back from a device buffer."""

    def __init__(self, spec: Tuple[Tuple, np.dtype], real_shape: Tuple):
        self.padded_shape, self.dtype = spec[0], spec[1]
        self.real_shape = real_shape
        self.host_data = np.empty(self.padded_shape, dtype=self.dtype)

    def enqueue_read(self, queue: cl.CommandQueue, cl_buffer: cl.Buffer, wait_for=None) -> cl.Event:
        """Enqueues a non-blocking D2H copy into this view's host memory."""
        return cl.enqueue_copy(queue, self.host_data, cl_buffer, wait_for=wait_for or [])

    def get(self) -> np.ndarray:
        """Returns the valid, un-padded slice of the data after the copy is complete."""
        slicing = tuple(slice(0, dim) for dim in self.real_shape)
        if not slicing:
            return self.host_data
        return self.host_data[slicing]


# --- 2D Tiling System ---


@dataclass(frozen=True)
class WorkTile:
    """Represents a single, self-contained rectangle of the compute grid."""

    module_chunk_idx: int
    class_chunk_idx: int
    flat_tile_id: int
    module_offset: int
    num_modules_in_tile: int
    class_offset: int
    num_classes_in_tile: int


@dataclass(frozen=True)
class ExecutionGrid:
    """Defines the M x N geometry of the module/class workload and acts as a tile factory."""

    num_module_chunks: int
    num_class_chunks: int
    total_modules: int
    total_classes: int

    @property
    def total_tiles(self) -> int:
        return self.num_module_chunks * self.num_class_chunks

    def __iter__(self):
        """Makes the grid itself an iterable, yielding each WorkTile in order."""
        for m_idx in range(self.num_module_chunks):
            for c_idx in range(self.num_class_chunks):
                yield self.get_tile(m_idx, c_idx)

    def get_tile(self, module_chunk_idx: int, class_chunk_idx: int) -> "WorkTile":
        """Creates a specific WorkTile, calculating its slice of the workload."""
        module_chunk_size = (self.total_modules + self.num_module_chunks - 1) // self.num_module_chunks
        module_offset = module_chunk_idx * module_chunk_size
        num_modules_in_tile = min(module_chunk_size, self.total_modules - module_offset)

        class_chunk_size = (self.total_classes + self.num_class_chunks - 1) // self.num_class_chunks
        class_offset = class_chunk_idx * class_chunk_size
        num_classes_in_tile = min(class_chunk_size, self.total_classes - class_offset)

        flat_tile_id = module_chunk_idx * self.num_class_chunks + class_chunk_idx

        return WorkTile(
            module_chunk_idx=module_chunk_idx,
            class_chunk_idx=class_chunk_idx,
            flat_tile_id=flat_tile_id,
            module_offset=module_offset,
            num_modules_in_tile=num_modules_in_tile,
            class_offset=class_offset,
            num_classes_in_tile=num_classes_in_tile,
        )


@dataclass(frozen=True)
class ChunkingConfig:
    """Configuration for chunking a problem across one or more dimensions (e.g., batch)."""

    num_chunks: int = 1
    chunk_size: int = 0
    total_dim: int = 0


@dataclass
class ExecutionPlan:
    """Holds the strategic decisions for processing one batch."""

    grid: ExecutionGrid
    shared_layer_batch_chunking: ChunkingConfig
    recompute_hidden: bool = False


# --- Utility & Padding Functions ---


def load_and_concatenate_kernels() -> str:
    """Loads all kernel source files from disk and concatenates them."""
    kernel_files = [
        "kernels.cl.h",
        "chunk_kernels.cl.c",
        "aggregation_kernels.cl.c",
        "backprop_kernels.cl.c",
        "parameter_optim.cl.c",
    ]
    source_parts = []
    for filename in kernel_files:
        if not os.path.exists(filename):
            raise FileNotFoundError(f"Missing required kernel file: {filename}. Please create placeholder files.")
        with open(filename, "r", encoding="utf-8") as f:
            source_parts.append(f.read())
    print(f"Loaded {len(source_parts)} kernel source files.")
    return "\n".join(source_parts)


def select_simd_width(device: cl.Device) -> int:
    """Selects a reasonable SIMD width based on GPU vendor."""
    vendor = device.vendor.upper()
    if "NVIDIA" in vendor:
        return 32
    if "AMD" in vendor or "ADVANCED MICRO DEVICES" in vendor:
        return 64
    if "INTEL" in vendor:
        return 16
    return 8


def pad_to_multiple(dim: int, multiple: int) -> int:
    """Calculates the smallest multiple of `multiple` >= `dim`."""
    if multiple == 0:
        return dim
    return (dim + multiple - 1) // multiple * multiple


def _pad_none(shape: Tuple, *args) -> Tuple:
    return shape


def _pad_last_dim(shape: Tuple, simd_width: int) -> Tuple:
    padded = list(shape)
    if len(padded) > 0 and padded[-1] > 1:
        padded[-1] = pad_to_multiple(padded[-1], simd_width)
    return tuple(padded)


def _pad_batch_dim(shape: Tuple, simd_width: int) -> Tuple:
    padded = list(shape)
    if len(padded) > 0:
        padded[0] = pad_to_multiple(padded[0], simd_width)
    return tuple(padded)


def _pad_batch_and_last_dim(shape: Tuple, simd_width: int) -> Tuple:
    padded = list(shape)
    if len(padded) > 1:
        padded[0] = pad_to_multiple(padded[0], simd_width)
        if padded[-1] > 1:
            padded[-1] = pad_to_multiple(padded[-1], simd_width)
    elif len(padded) == 1:
        padded[0] = pad_to_multiple(padded[0], simd_width)
    return tuple(padded)


PADDING_RULES: Dict[BufferRole, Callable] = {
    BufferRole.INPUT: _pad_batch_and_last_dim,
    BufferRole.HIDDEN_ACTIVATION: _pad_batch_and_last_dim,
    BufferRole.SHARED_WEIGHTS: _pad_last_dim,
    BufferRole.MODULE_WEIGHTS: _pad_last_dim,
    BufferRole.FINAL_GRADIENT: _pad_last_dim,
    BufferRole.ADAM_MOMENTUM: _pad_last_dim,
    BufferRole.INTERMEDIATE: _pad_last_dim,
    BufferRole.MASK: _pad_batch_dim,
}


# --- Layer 1: Action Layer (Stateless Tools) ---


class BufferManager:
    """Manages the lifecycle, creation, padding, and access of all OpenCL memory buffers."""

    def __init__(self, context: cl.Context, simd_width: int):
        self.context = context
        self.simd_width = simd_width
        self.buffers: Dict[str, cl.Buffer] = {}
        self.specs: Dict[str, Tuple[Tuple[int, ...], np.dtype]] = {}
        self.byte_sizes: Dict[str, int] = {}

    def create_buffer(
        self,
        name: str,
        role: BufferRole,
        layout: LayoutType,
        logical_shape: Tuple,
        dtype: np.dtype,
        init_data: Optional[np.ndarray] = None,
    ):
        """Creates a buffer, handling padding and layout transformation based on its declared role and layout type."""
        physical_data, physical_shape = init_data, logical_shape
        if layout == LayoutType.SoA:
            if len(logical_shape) != 2:
                raise ValueError(f"SoA layout only for 2D matrices, shape is {logical_shape}")
            padded_i = pad_to_multiple(logical_shape[0], self.simd_width)
            padded_h = pad_to_multiple(logical_shape[1], self.simd_width)
            padded_aos = np.zeros((padded_i, padded_h), dtype=dtype)
            if init_data is not None:
                padded_aos[: logical_shape[0], : logical_shape[1]] = init_data
            transformed_data = padded_aos.T.reshape(padded_h // self.simd_width, self.simd_width, padded_i).transpose(
                0, 2, 1
            )
            physical_data, physical_shape = transformed_data, transformed_data.shape
        padding_func = PADDING_RULES.get(role, _pad_none)
        padded_shape = padding_func(physical_shape, self.simd_width)
        byte_size = max(4, int(np.prod(padded_shape) * dtype().itemsize) if padded_shape else 4)
        mem_flags, hostbuf = cl.mem_flags.READ_WRITE, None
        if physical_data is not None:
            mem_flags |= cl.mem_flags.COPY_HOST_PTR
            if physical_data.shape != padded_shape:
                hostbuf = np.zeros(padded_shape, dtype=dtype)
                hostbuf[tuple(slice(0, d) for d in physical_data.shape)] = physical_data
            else:
                hostbuf = physical_data
        self.buffers[name] = cl.Buffer(self.context, mem_flags, size=byte_size, hostbuf=hostbuf)
        self.specs[name] = (padded_shape, dtype)
        self.byte_sizes[name] = byte_size

    def get(self, name: str) -> cl.Buffer:
        return self.buffers[name]

    def get_spec(self, name: str) -> Tuple[Tuple[int, ...], np.dtype]:
        return self.specs[name]

    def get_byte_size(self, name: str) -> int:
        return self.byte_sizes[name]


class KernelExecutor:
    """A stateless toolbox for launching every specific kernel. Returns events."""

    def __init__(self, program: cl.Program, buffer_mgr: BufferManager):
        self.p, self.b = program, buffer_mgr
        self.simd_width = self.b.simd_width
        self.padded_hidden_dim = self.b.get_spec("hidden_buf")[0][1]
        self.padded_input_dim = self.b.get_spec("input_buf")[0][1]
        # Cache the physical class dimension for reuse, as it's the stride for many buffers.
        self.padded_output_classes = self.b.get_spec("full_logits_out")[0][2]
        self.scalar_size = SCALAR_NP_TYPE().itemsize

    def enqueue_write_buffer(self, queue, name, data, wait_for) -> cl.Event:
        buf, (shape, dtype) = self.b.get(name), self.b.get_spec(name)
        padded_data = np.zeros(shape, dtype=dtype)
        padded_data[tuple(slice(0, d) for d in data.shape)] = data
        return cl.enqueue_copy(queue, buf, padded_data, wait_for=wait_for)

    def enqueue_fill_buffer(self, queue, name, value, wait_for) -> cl.Event:
        buf, spec = self.b.get(name), self.b.get_spec(name)
        return cl.enqueue_fill_buffer(queue, buf, spec[1](value), 0, buf.size, wait_for=wait_for)

    def launch_forward_pass(self, queue, offset, num_samples, wait_for) -> cl.Event:
        g = (pad_to_multiple(num_samples, self.simd_width), self.padded_hidden_dim // self.simd_width)
        l = (self.simd_width, 1)
        args = (
            cl.LocalMemory(l[0] * (1 + l[0]) * self.scalar_size),
            self.b.get("input_buf"),
            self.b.get("sample_mask"),
            self.b.get("weights"),
            self.b.get("biases"),
            self.b.get("hidden_buf"),
            self.b.get("hidden_mask"),
            np.int32(offset),
            np.int32(num_samples),
            np.int32(self.padded_input_dim),
            np.int32(self.padded_hidden_dim),
        )
        return self.p.forward_pass(queue, g, l, *args, wait_for=wait_for)

    def launch_compute_logits_chunk(self, queue: cl.CommandQueue, tile: WorkTile, wait_for) -> cl.Event:
        g, l = (tile.num_modules_in_tile, BATCH_SIZE, tile.num_classes_in_tile), None
        args = (
            self.b.get("hidden_buf"),
            self.b.get("hidden_mask"),
            self.b.get("module_weights"),
            self.b.get("module_biases"),
            self.b.get("full_logits_out"),
            np.int32(tile.module_chunk_idx),
            np.int32(tile.module_offset),
            np.int32(tile.num_modules_in_tile),
            np.int32(tile.class_offset),
            np.int32(tile.num_classes_in_tile),
            np.int32(BATCH_SIZE),
            np.int32(HIDDEN_DIM),
            np.int32(self.padded_hidden_dim),
            np.int32(OUTPUT_CLASSES),
            np.int32(self.padded_output_classes),
        )
        return self.p.compute_logits_chunk(queue, g, l, *args, wait_for=wait_for)

    def launch_reduce_for_softmax(self, queue, wait_for) -> cl.Event:
        g, l = (NUM_MODULES, BATCH_SIZE), None
        args = (
            self.b.get("full_logits_out"),
            self.b.get("temps"),
            self.b.get("softmax_params_out"),
            np.int32(NUM_MODULES),
            np.int32(BATCH_SIZE),
            np.int32(OUTPUT_CLASSES),
            np.int32(self.padded_output_classes),
        )
        return self.p.reduce_logits_for_softmax(queue, g, l, *args, wait_for=wait_for)

    def launch_compute_probs_loss_cce_chunk(self, queue: cl.CommandQueue, tile: WorkTile, wait_for) -> cl.Event:
        g, l = (tile.num_modules_in_tile, BATCH_SIZE, tile.num_classes_in_tile), None
        args = (
            self.b.get("full_logits_out"),
            self.b.get("softmax_params_out"),
            self.b.get("temps"),
            self.b.get("targets_buf"),
            self.b.get("sample_mask"),
            self.b.get("partial_probs_out"),
            self.b.get("final_loss_out"),
            np.int32(tile.module_chunk_idx),
            np.int32(tile.module_offset),
            np.int32(tile.num_modules_in_tile),
            np.int32(tile.class_offset),
            np.int32(tile.num_classes_in_tile),
            np.int32(BATCH_SIZE),
            np.int32(OUTPUT_CLASSES),
            np.int32(self.padded_output_classes),
        )
        return self.p.compute_probs_loss_cce_chunk(queue, g, l, *args, wait_for=wait_for)

    def launch_parallel_module_grads(
        self, queue: cl.CommandQueue, tile: WorkTile, wait_for
    ) -> Tuple[cl.Event, cl.Event, cl.Event]:
        lsize, problem_flag = 256, np.int32(0 if PROBLEM_TYPE == "CCE" else 1)
        common_args = (
            problem_flag,
            np.int32(tile.module_chunk_idx),
            np.int32(tile.module_offset),
            np.int32(tile.num_modules_in_tile),
            np.int32(tile.class_chunk_idx),
            np.int32(tile.class_offset),
            np.int32(tile.num_classes_in_tile),
            np.int32(BATCH_SIZE),
            np.int32(HIDDEN_DIM),
            np.int32(self.padded_hidden_dim),
            np.int32(OUTPUT_CLASSES),
            np.int32(self.padded_output_classes),
            np.int32(NUM_MODULES),
        )
        grad_w_evt = self.p.calculate_module_param_grads_chunk(
            queue,
            (tile.num_modules_in_tile, HIDDEN_DIM, tile.num_classes_in_tile),
            None,
            cl.LocalMemory(lsize * self.scalar_size),
            self.b.get("hidden_buf"),
            self.b.get("partial_probs_out"),
            self.b.get("targets_buf"),
            self.b.get("sample_mask"),
            self.b.get("partial_grad_module_w_out"),
            self.b.get("partial_grad_module_b_out"),
            *common_args,
            wait_for=wait_for,
        )
        grad_h_evt = self.p.backprop_error_to_hidden_chunk(
            queue,
            (tile.num_modules_in_tile, BATCH_SIZE, HIDDEN_DIM),
            None,
            cl.LocalMemory(0),
            self.b.get("partial_probs_out"),
            self.b.get("targets_buf"),
            self.b.get("sample_mask"),
            self.b.get("module_weights"),
            self.b.get("partial_grad_h_aos_out"),
            *common_args,
            wait_for=wait_for,
        )
        grad_t_evt = self.p.calculate_chunk_temp_gradients(
            queue,
            (tile.num_modules_in_tile * lsize,),
            (lsize,),
            cl.LocalMemory(lsize * self.scalar_size),
            self.b.get("full_logits_out"),
            self.b.get("partial_probs_out"),
            self.b.get("targets_buf"),
            self.b.get("sample_mask"),
            self.b.get("temps"),
            self.b.get("partial_grad_temps_out"),
            *common_args,
            wait_for=wait_for,
        )
        return grad_w_evt, grad_h_evt, grad_t_evt

    def launch_grad_h_transpose_for_tile(self, queue: cl.CommandQueue, tile: WorkTile, wait_for) -> cl.Event:
        """A specialized method to correctly launch transpose_chunk for a Grad_H tile."""
        in_buf, out_buf = "partial_grad_h_aos_out", "partial_grad_h_soa_out"
        in_spec, out_spec = self.b.get_spec(in_buf)[0], self.b.get_spec(out_buf)[0]
        rows = tile.num_modules_in_tile
        cols = BATCH_SIZE * HIDDEN_DIM
        in_offset = tile.flat_tile_id * (in_spec[1] * in_spec[2] * in_spec[3])
        out_offset = tile.flat_tile_id * (out_spec[1] * out_spec[2])
        return self.launch_transpose_chunk(
            queue, in_buf, out_buf, in_offset, out_offset, rows, cols, cols, out_spec[2], wait_for
        )

    def launch_transpose_chunk(
        self,
        queue,
        in_buf_name,
        out_buf_name,
        in_offset_elements,
        out_offset_elements,
        num_rows,
        num_cols,
        in_leading_dim,
        out_leading_dim,
        wait_for,
    ) -> cl.Event:
        g = (pad_to_multiple(num_cols, C_TILE_SIZE), pad_to_multiple(num_rows, C_TILE_SIZE))
        l = (C_TILE_SIZE, C_TILE_SIZE)
        args = (
            cl.LocalMemory(l[0] * (l[0] + 1) * self.scalar_size),
            self.b.get(in_buf_name),
            self.b.get(out_buf_name),
            np.int32(in_offset_elements),
            np.int32(out_offset_elements),
            np.int32(num_rows),
            np.int32(num_cols),
            np.int32(in_leading_dim),
            np.int32(out_leading_dim),
        )
        return self.p.transpose_chunk(queue, g, l, *args, wait_for=wait_for)

    def launch_aggregation(self, queue, in_buf, out_buf, num_partials, elements, is_avg, wait_for) -> cl.Event:
        lsize, mode = 256, np.int32(1 if is_avg else 0)
        elements = max(1, int(elements))
        if num_partials <= 1:
            return self.p.aggregate_identity(
                queue,
                (elements,),
                None,
                cl.LocalMemory(0),
                self.b.get(in_buf),
                self.b.get(out_buf),
                np.int32(1),
                np.int32(elements),
                mode,
                wait_for=wait_for,
            )
        elif num_partials <= MAX_REGISTER_AGGREGATE_ITEMS:
            return self.p.aggregate_register_reduce(
                queue,
                (elements,),
                None,
                cl.LocalMemory(0),
                self.b.get(in_buf),
                self.b.get(out_buf),
                np.int32(num_partials),
                np.int32(elements),
                mode,
                wait_for=wait_for,
            )
        else:
            gsize = pad_to_multiple(elements, lsize)
            return self.p.aggregate_local_reduce(
                queue,
                (gsize,),
                (lsize,),
                cl.LocalMemory(lsize * self.scalar_size),
                self.b.get(in_buf),
                self.b.get(out_buf),
                np.int32(num_partials),
                np.int32(elements),
                mode,
                wait_for=wait_for,
            )

    def launch_reduce_grad_h_over_modules(self, queue: cl.CommandQueue, wait_for) -> cl.Event:
        """Launches the specialized kernel (13) to reduce Grad_H over the module dimension."""
        lsize = 256
        in_buf_spec = self.b.get_spec("aggregated_grad_h_soa")
        total_elements = in_buf_spec[0][0]
        padded_total_modules = in_buf_spec[0][1]

        gsize = pad_to_multiple(total_elements, lsize)

        args = (
            cl.LocalMemory(lsize * self.scalar_size),
            self.b.get("aggregated_grad_h_soa"),
            self.b.get("final_grad_h_buf"),
            np.int32(total_elements),
            np.int32(NUM_MODULES),
            np.int32(padded_total_modules),
        )
        return self.p.reduce_grad_h_over_modules(queue, (gsize,), (lsize,), *args, wait_for=wait_for)

    def launch_backprop_shared_chunk(self, queue, chunk_id, offset, num_samples, wait_for) -> Tuple[cl.Event, cl.Event]:
        lsize = 256
        sw_evt = self.p.backprop_shared_weights_chunk(
            queue,
            (self.padded_input_dim, pad_to_multiple(self.padded_hidden_dim, lsize)),
            (1, lsize),
            cl.LocalMemory(lsize * self.scalar_size),
            self.b.get("input_buf"),
            self.b.get("hidden_buf"),
            self.b.get("final_grad_h_buf"),
            self.b.get("sample_mask"),
            self.b.get("partial_grad_sw_out"),
            np.int32(offset),
            np.int32(num_samples),
            np.int32(chunk_id),
            np.int32(self.padded_input_dim),
            np.int32(self.padded_hidden_dim),
            wait_for=wait_for,
        )
        sb_evt = self.p.backprop_shared_biases_chunk(
            queue,
            (pad_to_multiple(self.padded_hidden_dim, lsize),),
            (lsize,),
            cl.LocalMemory(lsize * self.scalar_size),
            self.b.get("hidden_buf"),
            self.b.get("final_grad_h_buf"),
            self.b.get("sample_mask"),
            self.b.get("partial_grad_sb_out"),
            np.int32(offset),
            np.int32(num_samples),
            np.int32(chunk_id),
            np.int32(self.padded_hidden_dim),
            wait_for=wait_for,
        )
        return sw_evt, sb_evt

    def launch_adam_update_and_clamp(self, queue, p: Parameter, grad_ready_event, t: int) -> cl.Event:
        num_params = int(np.prod(self.b.get_spec(p.name)[0]))
        update_evt = self.p.adam_update(
            queue,
            (num_params,),
            None,
            self.b.get(p.grad),
            SCALAR_NP_TYPE(ADAM_BETA1),
            SCALAR_NP_TYPE(ADAM_BETA2),
            SCALAR_NP_TYPE(LEARNING_RATE),
            SCALAR_NP_TYPE(EPSILON),
            np.uint32(t),
            self.b.get(p.name),
            self.b.get(p.m1),
            self.b.get(p.m2),
            np.int32(0),
            np.int32(num_params),
            wait_for=[grad_ready_event],
        )
        if p.role == BufferRole.TEMPERATURES:
            return self.p.clamp_temperatures(
                queue,
                (NUM_MODULES,),
                None,
                self.b.get("temps"),
                SCALAR_NP_TYPE(MIN_TEMP),
                SCALAR_NP_TYPE(MAX_TEMP),
                np.int32(NUM_MODULES),
                wait_for=[update_evt],
            )
        return update_evt


# --- Layer 2: Strategic Layer ---


class ExecutionStrategy:
    """Makes high-level strategic decisions based on resources and problem size."""

    def __init__(self, device_vram_bytes: int, buffer_mgr: BufferManager):
        self.vram_budget = device_vram_bytes * 0.85
        self.b = buffer_mgr

    def create_plan_for_batch(self, batch_size: int) -> ExecutionPlan:
        """Determines the optimal 2D tiling grid based on memory constraints."""
        hidden_size = self.b.get_byte_size("hidden_buf")
        full_problem_mem = (
            self.b.get_byte_size("full_logits_out")
            + self.b.get_byte_size("partial_probs_out")
            + self.b.get_byte_size("partial_grad_h_aos_out")
        )
        mem_per_module_slice = full_problem_mem / max(1, NUM_MODULES)
        mem_per_class_slice = full_problem_mem / max(1, OUTPUT_CLASSES)
        num_m_chunks, num_c_chunks = 1, 1

        while True:
            mem_per_tile = full_problem_mem / (num_m_chunks * num_c_chunks)
            if mem_per_tile <= self.vram_budget:
                break
            cost_of_module_dim = mem_per_module_slice * (NUM_MODULES / num_m_chunks)
            cost_of_class_dim = mem_per_class_slice * (OUTPUT_CLASSES / num_c_chunks)
            if cost_of_module_dim >= cost_of_class_dim:
                num_m_chunks += 1
            else:
                num_c_chunks += 1
            if (num_m_chunks * num_c_chunks) > MAX_TOTAL_TILES:
                raise MemoryError("Cannot create a tile small enough for device VRAM.")
        grid = ExecutionGrid(num_m_chunks, num_c_chunks, NUM_MODULES, OUTPUT_CLASSES)

        mem_per_tile_final = full_problem_mem / grid.total_tiles
        recompute_hidden = (hidden_size + mem_per_tile_final) > self.vram_budget
        num_batch_chunks = (batch_size + BACKPROP_STREAM_CHUNK_SIZE - 1) // BACKPROP_STREAM_CHUNK_SIZE
        batch_cfg = ChunkingConfig(min(num_batch_chunks, MAX_BATCH_CHUNKS), BACKPROP_STREAM_CHUNK_SIZE, batch_size)

        print(
            f"INFO: Determined a {grid.num_module_chunks}x{grid.num_class_chunks} tiling grid ({grid.total_tiles} total tiles)."
        )
        print("  - Strategy: Unified INTERLEAVED_STREAMING backprop for Grad_H.")
        if recompute_hidden:
            print("INFO: Extreme memory pressure. `hidden` buffer will be recomputed.")

        return ExecutionPlan(grid, batch_cfg, recompute_hidden)


# --- Layer 3: Execution Layer ---


class BatchProcessor:
    """Stateful, single-use engine to process one batch by building an event-based DAG."""

    def __init__(self, orchestrator: "TrainingOrchestrator", X_batch: np.ndarray, y_batch: np.ndarray):
        self.orchestrator = orchestrator
        self.ctx, self.queue, self.executor = orchestrator.ctx, orchestrator.queue, orchestrator.executor
        self.events: Dict[str, cl.Event] = {}
        self.event_lists: Dict[str, List[cl.Event]] = defaultdict(list)
        self.final_grad_events: Dict[str, cl.Event] = {}
        self.X_batch, self.y_batch = X_batch, y_batch
        self.global_step = orchestrator.global_step
        self.host_probs_view = HostView(
            self.executor.b.get_spec("final_probs_buf"), (NUM_MODULES, X_batch.shape[0], OUTPUT_CLASSES)
        )
        self.host_loss_view = HostView(self.executor.b.get_spec("final_loss_out"), (NUM_MODULES, X_batch.shape[0]))

    def _get_deps(self, *resource_names: str) -> List[cl.Event]:
        deps = [self.events[name] for name in resource_names if name in self.events]
        for name in resource_names:
            deps.extend(self.event_lists.get(name, []))
        return list(set(deps))

    def run(self, plan: ExecutionPlan):
        print(f"  - Executing Plan: {plan.grid}")
        self._upload_data()
        self._zero_gradients()
        self.events["hidden_ready"] = self.executor.launch_forward_pass(
            self.queue, 0, self.X_batch.shape[0], self._get_deps("input_ready", "grads_zeroed")
        )
        self._execute_module_path(plan)
        self._aggregate_and_backprop_shared(plan)
        self._finalize_and_update()
        return self.get_sync_points()

    def _upload_data(self):
        write_X = self.executor.enqueue_write_buffer(self.queue, "input_buf", self.X_batch, None)
        write_mask = self.executor.enqueue_write_buffer(
            self.queue, "sample_mask", np.ones(self.X_batch.shape[0], dtype=SCALAR_NP_TYPE), None
        )
        self.events["input_ready"] = cl.WaitForEvents([write_X, write_mask])
        self.events["targets_ready"] = self.executor.enqueue_write_buffer(self.queue, "targets_buf", self.y_batch, None)

    def _zero_gradients(self):
        grad_names = [name for name in self.executor.b.buffers if "grad_" in name]
        zero_events = [self.executor.enqueue_fill_buffer(self.queue, name, 0, None) for name in grad_names]
        self.events["grads_zeroed"] = cl.WaitForEvents(zero_events)

    def _execute_module_path(self, plan: ExecutionPlan):
        """Executes the module/class path using a two-pass approach to respect the CCE softmax synchronization point."""
        tiles = list(plan.grid)
        logit_deps = self._get_deps("hidden_ready")
        for tile in tiles:
            if tile.num_modules_in_tile > 0 and tile.num_classes_in_tile > 0:
                evt = self.executor.launch_compute_logits_chunk(self.queue, tile, logit_deps)
                self.event_lists["logit_chunks_ready"].append(evt)
        self.events["all_logits_ready"] = cl.WaitForEvents(self.event_lists["logit_chunks_ready"])

        if PROBLEM_TYPE == "CCE":
            self.events["softmax_params_ready"] = self.executor.launch_reduce_for_softmax(
                self.queue, self._get_deps("all_logits_ready")
            )

        for tile in tiles:
            if tile.num_modules_in_tile > 0 and tile.num_classes_in_tile > 0:
                self._launch_tile_downstream_work(tile, plan)
        self.events["all_prob_loss_chunks_ready"] = cl.WaitForEvents(self.event_lists["prob_loss_chunks_ready"])

    def _launch_tile_downstream_work(self, tile: WorkTile, plan: ExecutionPlan):
        """Launches all kernels for a tile that come AFTER the logit/softmax sync point."""
        prob_loss_deps = self._get_deps("softmax_params_ready", "targets_ready", "all_logits_ready")
        prob_loss_evt = self.executor.launch_compute_probs_loss_cce_chunk(self.queue, tile, prob_loss_deps)
        self.event_lists["prob_loss_chunks_ready"].append(prob_loss_evt)
        grad_base_deps = self._get_deps("hidden_ready", "all_logits_ready", "targets_ready")
        grad_base_deps.append(prob_loss_evt)
        w_grad, h_grad, t_grad = self.executor.launch_parallel_module_grads(self.queue, tile, grad_base_deps)
        self.event_lists["partial_grad_w_ready"].append(w_grad)
        self.event_lists["partial_grad_t_ready"].append(t_grad)

        # ---- The Unified Path ----
        # The interleaved transpose-as-you-go approach is always optimal and is required by the aggregation logic.
        transpose_evt = self.executor.launch_grad_h_transpose_for_tile(self.queue, tile, wait_for=[h_grad])
        self.event_lists["partial_grad_h_soa_ready"].append(transpose_evt)

    def _aggregate_and_backprop_shared(self, plan: ExecutionPlan):
        b = self.executor.b
        get_elem = lambda name: int(np.prod(b.get_spec(name)[0]))

        # --- Aggregate all partials from the module/class grid ---
        self.events["final_probs_ready"] = self.executor.launch_aggregation(
            self.queue,
            "partial_probs_out",
            "final_probs_buf",
            plan.grid.total_tiles,
            get_elem("final_probs_buf"),
            False,
            self._get_deps("all_prob_loss_chunks_ready"),
        )
        self.final_grad_events["grad_module_weights"] = self.executor.launch_aggregation(
            self.queue,
            "partial_grad_module_w_out",
            "grad_module_weights",
            plan.grid.total_tiles,
            get_elem("grad_module_weights"),
            False,
            self.event_lists["partial_grad_w_ready"],
        )
        self.final_grad_events["grad_module_biases"] = self.executor.launch_aggregation(
            self.queue,
            "partial_grad_module_b_out",
            "grad_module_biases",
            plan.grid.total_tiles,
            get_elem("grad_module_biases"),
            False,
            self.event_lists["partial_grad_w_ready"],
        )
        self.final_grad_events["grad_temps"] = self.executor.launch_aggregation(
            self.queue,
            "partial_grad_temps_out",
            "grad_temps",
            plan.grid.total_tiles,
            get_elem("grad_temps"),
            False,
            self.event_lists["partial_grad_t_ready"],
        )

        # --- Path for Grad_H ---
        agg_grad_h_evt = self.executor.launch_aggregation(
            self.queue,
            "partial_grad_h_soa_out",
            "aggregated_grad_h_soa",
            plan.grid.total_tiles,
            get_elem("aggregated_grad_h_soa"),
            False,
            self.event_lists["partial_grad_h_soa_ready"],  # Depends on the SoA partials
        )

        self.events["final_grad_h_ready"] = self.executor.launch_reduce_grad_h_over_modules(
            self.queue, wait_for=[agg_grad_h_evt]
        )

        # --- Streaming backprop for the shared layer ---
        backprop_deps = self._get_deps("final_grad_h_ready")
        if plan.recompute_hidden:
            backprop_deps.append(
                self.executor.launch_forward_pass(self.queue, 0, self.X_batch.shape[0], self._get_deps("input_ready"))
            )
        else:
            backprop_deps.extend(self._get_deps("hidden_ready"))

        cfg = plan.shared_layer_batch_chunking
        for i in range(cfg.num_chunks):
            offset, size = i * cfg.chunk_size, min(cfg.chunk_size, cfg.total_dim - offset)
            if size <= 0:
                continue
            sw, sb = self.executor.launch_backprop_shared_chunk(self.queue, i, offset, size, backprop_deps)
            self.event_lists["partial_grad_sw_ready"].append(sw)
            self.event_lists["partial_grad_sb_ready"].append(sb)

        # --- Final aggregation for shared layer gradients ---
        self.final_grad_events["grad_weights"] = self.executor.launch_aggregation(
            self.queue,
            "partial_grad_sw_out",
            "grad_weights",
            cfg.num_chunks,
            get_elem("grad_weights"),
            True,
            self.event_lists["partial_grad_sw_ready"],
        )
        self.final_grad_events["grad_biases"] = self.executor.launch_aggregation(
            self.queue,
            "partial_grad_sb_out",
            "grad_biases",
            cfg.num_chunks,
            get_elem("grad_biases"),
            True,
            self.event_lists["partial_grad_sb_ready"],
        )

    def _finalize_and_update(self):
        self.events["d2h_probs_ready"] = self.host_probs_view.enqueue_read(
            self.queue, self.executor.b.get("final_probs_buf"), self._get_deps("final_probs_ready")
        )
        for p in self.orchestrator.params:
            if p.grad not in self.final_grad_events:
                continue
            grad_ready = self.final_grad_events[p.grad]
            self.event_lists["updates_done"].append(
                self.executor.launch_adam_update_and_clamp(self.queue, p, grad_ready, self.global_step)
            )
        self.events["all_updates_done"] = cl.WaitForEvents(self.event_lists["updates_done"])

    def get_sync_points(self) -> Tuple[cl.UserEvent, cl.UserEvent]:
        inf_evt, final_evt = cl.UserEvent(self.ctx), cl.UserEvent(self.ctx)
        d2h_event = self.events.get("d2h_probs_ready")
        final_done = self.events.get("all_updates_done")

        def set_complete(status, user_event):
            user_event.set_status(cl.command_execution_status.COMPLETE)

        if d2h_event:
            d2h_event.set_callback(cl.command_execution_status.COMPLETE, set_complete, inf_evt)
        else:
            inf_evt.set_status(cl.command_execution_status.COMPLETE)
        if final_done:
            final_done.set_callback(cl.command_execution_status.COMPLETE, set_complete, final_evt)
        else:
            final_evt.set_status(cl.command_execution_status.COMPLETE)
        return inf_evt, final_evt


# --- Layer 4: Orchestration Layer ---


class TrainingOrchestrator:
    """Top-level class that owns all components and runs the main training loop."""

    def __init__(self):
        self.ctx = cl.create_some_context(interactive=False)
        self.device = self.ctx.devices[0]
        self.queue = cl.CommandQueue(self.ctx, properties=cl.command_queue_properties.OUT_OF_ORDER_EXEC_MODE_ENABLE)
        print(f"Using device: {self.device.name} ({self.device.vendor})")
        print(f"VRAM: {self.device.global_mem_size / 1e9:.2f} GB")
        self.simd_width = select_simd_width(self.device)
        self.program = self._compile_kernels()
        self.params: List[Parameter] = [
            Parameter("weights", BufferRole.SHARED_WEIGHTS, layout=LayoutType.SoA),
            Parameter("biases", BufferRole.SHARED_BIAS),
            Parameter("module_weights", BufferRole.MODULE_WEIGHTS),
            Parameter("module_biases", BufferRole.MODULE_BIAS),
            Parameter("temps", BufferRole.TEMPERATURES),
        ]
        self.param_shapes: Dict[str, Tuple] = {
            "weights": (INPUT_DIM, HIDDEN_DIM),
            "biases": (HIDDEN_DIM,),
            "module_weights": (NUM_MODULES, HIDDEN_DIM, OUTPUT_CLASSES),
            "module_biases": (NUM_MODULES, OUTPUT_CLASSES),
            "temps": (NUM_MODULES,),
        }
        self.buffer_mgr = BufferManager(self.ctx, self.simd_width)
        self._setup_buffers()
        self.executor = KernelExecutor(self.program, self.buffer_mgr)
        self.strategy = ExecutionStrategy(self.device.global_mem_size, self.buffer_mgr)
        self.global_step = 1

    def _compile_kernels(self) -> cl.Program:
        kernel_src = load_and_concatenate_kernels()
        opts = [
            f"-cl-std=CL1.2",
            f"-D SCALAR_TYPE={CL_SCALAR_TYPE}",
            f"-D SIMD_WIDTH={self.simd_width}",
            f"-D C_TILE_SIZE={C_TILE_SIZE}",
        ]
        if SCALAR_TYPE == "half":
            opts.append("-D cl_khr_fp16")
        return cl.Program(self.ctx, kernel_src).build(options=opts)

    def _setup_buffers(self):
        for p in self.params:
            shape = self.param_shapes[p.name]
            init_fn = (
                np.zeros
                if "bias" in p.name
                else (
                    (lambda s: np.full(s, 1.0, SCALAR_NP_TYPE))
                    if "temp" in p.name
                    else (lambda s: np.random.randn(*s).astype(SCALAR_NP_TYPE) * 0.01)
                )
            )
            self.buffer_mgr.create_buffer(p.name, p.role, p.layout, shape, SCALAR_NP_TYPE, init_fn(shape))
            phys_shape, dtype = self.buffer_mgr.get_spec(p.name)
            self.buffer_mgr.create_buffer(p.grad, BufferRole.FINAL_GRADIENT, LayoutType.AoS, phys_shape, dtype)
            self.buffer_mgr.create_buffer(p.m1, BufferRole.ADAM_MOMENTUM, LayoutType.AoS, phys_shape, dtype)
            self.buffer_mgr.create_buffer(p.m2, BufferRole.ADAM_MOMENTUM, LayoutType.AoS, phys_shape, dtype)

        dtype_map = {"targets_buf": np.int32} if PROBLEM_TYPE == "CCE" else {}
        targets_shape = (BATCH_SIZE,) if PROBLEM_TYPE == "CCE" else (BATCH_SIZE, OUTPUT_CLASSES)
        int_buffers = {
            "input_buf": (BufferRole.INPUT, (BATCH_SIZE, INPUT_DIM)),
            "sample_mask": (BufferRole.MASK, (BATCH_SIZE,)),
            "hidden_buf": (BufferRole.HIDDEN_ACTIVATION, (BATCH_SIZE, HIDDEN_DIM)),
            "hidden_mask": (BufferRole.MASK, (BATCH_SIZE,)),
            "targets_buf": (BufferRole.TARGETS, targets_shape),
            "full_logits_out": (BufferRole.INTERMEDIATE, (NUM_MODULES, BATCH_SIZE, OUTPUT_CLASSES)),
            "partial_probs_out": (BufferRole.INTERMEDIATE, (NUM_MODULES, BATCH_SIZE, OUTPUT_CLASSES)),
            "final_probs_buf": (BufferRole.INTERMEDIATE, (NUM_MODULES, BATCH_SIZE, OUTPUT_CLASSES)),
            "final_loss_out": (BufferRole.INTERMEDIATE, (NUM_MODULES, BATCH_SIZE)),
            "softmax_params_out": (BufferRole.INTERMEDIATE, (NUM_MODULES, BATCH_SIZE, 2)),
            "partial_grad_module_w_out": (
                BufferRole.PARTIAL_GRADIENT,
                (MAX_TOTAL_TILES, *self.param_shapes["module_weights"]),
            ),
            "partial_grad_module_b_out": (
                BufferRole.PARTIAL_GRADIENT,
                (MAX_TOTAL_TILES, *self.param_shapes["module_biases"]),
            ),
            "partial_grad_temps_out": (BufferRole.PARTIAL_GRADIENT, (MAX_TOTAL_TILES, *self.param_shapes["temps"])),
            "partial_grad_sw_out": (BufferRole.PARTIAL_GRADIENT, (MAX_BATCH_CHUNKS, *self.param_shapes["weights"])),
            "partial_grad_sb_out": (BufferRole.PARTIAL_GRADIENT, (MAX_BATCH_CHUNKS, *self.param_shapes["biases"])),
            "partial_grad_h_aos_out": (
                BufferRole.PARTIAL_GRADIENT,
                (MAX_TOTAL_TILES, NUM_MODULES, BATCH_SIZE, HIDDEN_DIM),
            ),
            "partial_grad_h_soa_out": (
                BufferRole.PARTIAL_GRADIENT,
                (MAX_TOTAL_TILES, BATCH_SIZE * HIDDEN_DIM, NUM_MODULES),
            ),
            # This buffer now holds the result of aggregating the partial SoA chunks.
            # Its layout is (B*H, M), making it ready for the specialized reduction kernel.
            "aggregated_grad_h_soa": (BufferRole.INTERMEDIATE, (BATCH_SIZE * HIDDEN_DIM, NUM_MODULES)),
            # This is the final destination for the upstream gradient.
            "final_grad_h_buf": (BufferRole.FINAL_GRADIENT, (BATCH_SIZE, HIDDEN_DIM)),
        }
        for name, (role, shape) in int_buffers.items():
            self.buffer_mgr.create_buffer(name, role, LayoutType.AoS, shape, dtype_map.get(name, SCALAR_NP_TYPE))

    def train(self):
        from sklearn.datasets import load_iris
        from sklearn.preprocessing import StandardScaler, OneHotEncoder

        X, y = load_iris(return_X_y=True)
        X = X[:, :INPUT_DIM].astype(SCALAR_NP_TYPE)
        y = y[y < OUTPUT_CLASSES]
        X = X[: y.shape[0]]
        X = StandardScaler().fit_transform(X).astype(SCALAR_NP_TYPE)
        y_bce = OneHotEncoder().fit_transform(y.reshape(-1, 1)).toarray().astype(SCALAR_NP_TYPE)

        print(f"\n--- Starting training for {EPOCHS} epochs ---")
        for epoch in range(EPOCHS):
            indices = np.arange((epoch * BATCH_SIZE), (epoch * BATCH_SIZE) + BATCH_SIZE) % X.shape[0]
            X_batch, y_batch_int = X[indices], y[indices]
            y_b = y_batch_int.astype(np.int32) if PROBLEM_TYPE == "CCE" else y_bce[indices]

            print(f"Epoch {(self.global_step):_} | Batch Size: {X_batch.shape[0]}")
            plan = self.strategy.create_plan_for_batch(X_batch.shape[0])
            processor = BatchProcessor(self, X_batch, y_b)
            inference_event, final_event = processor.run(plan)

            inference_event.wait()
            valid_probs = processor.host_probs_view.get()
            predicted_classes = np.argmax(valid_probs[0, :, :], axis=1)
            accuracy = np.mean(predicted_classes == y_batch_int) if y_batch_int.size > 0 else 0.0

            print(f"  - Inference Latency Path Complete. Accuracy: {accuracy:.2%}")
            final_event.wait()
            print(f"  - Full Training Step Complete.")
            self.global_step += 1
        print("\n--- Training finished ---")


if __name__ == "__main__":
    try:
        # Create dummy kernel files if they don't exist, to allow the script to run.
        for fname in [
            "kernels.cl.h",
            "chunk_kernels.cl.c",
            "aggregation_kernels.cl.c",
            "backprop_kernels.cl.c",
            "parameter_optim.cl.c",
        ]:
            if not os.path.exists(fname):
                with open(fname, "w") as f:
                    f.write(f"// Placeholder for {fname}\n")
        orchestrator = TrainingOrchestrator()
        orchestrator.train()
    except cl.LogicError as e:
        print(f"A logic error occurred. This often means a kernel argument is wrong. Details: {e}")
    except cl.MemoryError as e:
        print(f"A memory error occurred. A buffer might be too large or accessed out of bounds. Details: {e}")
    except cl.BuildError as e:
        print("--- KERNEL BUILD FAILED ---")
        for dev, log in e.device_logs:
            print(f"Device: {dev.name}\n--- Build Log ---\n{log}")
    except Exception as e:
        import traceback

        print(f"An unexpected error occurred: {e}")
        traceback.print_exc()
