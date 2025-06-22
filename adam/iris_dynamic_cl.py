#
# A Unified, Memory-Aware Streaming Engine
# ========================================
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
NUM_EXITS: int = 128
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
MAX_EXIT_CHUNKS: int = 64  # Max number of chunks for exit/class dimension streaming
MAX_BATCH_CHUNKS: int = 64  # Max number of chunks for batch dimension streaming


# --- Core Data Abstractions ---


class BufferRole(enum.Enum):
    """Defines the semantic purpose of a buffer, driving its memory padding strategy."""

    INPUT = 1
    HIDDEN_ACTIVATION = 2
    SHARED_WEIGHTS = 3
    SHARED_BIAS = 4
    EXIT_WEIGHTS = 5
    EXIT_BIAS = 6
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


# --- Utility & Padding Functions ---


def load_and_concatenate_kernels() -> str:
    """Loads all kernel source files from disk and concatenates them."""
    # This list must be present in the same directory.
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
    BufferRole.EXIT_WEIGHTS: _pad_last_dim,
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
        self, name: str, role: BufferRole, real_shape: Tuple, dtype: np.dtype, init_data: Optional[np.ndarray] = None
    ) -> None:
        """Creates a buffer with memory padding determined by its semantic role."""
        padding_func = PADDING_RULES.get(role, _pad_none)
        padded_shape = padding_func(real_shape, self.simd_width)
        byte_size = int(np.prod(padded_shape) * dtype().itemsize) if padded_shape else 4
        byte_size = max(byte_size, 4)  # Ensure buffer is never zero-sized

        mem_flags = cl.mem_flags.READ_WRITE
        hostbuf = None
        if init_data is not None:
            mem_flags |= cl.mem_flags.COPY_HOST_PTR
            padded_data = np.zeros(padded_shape, dtype=dtype)
            slicing = tuple(slice(0, d) for d in real_shape)
            if slicing:
                padded_data[slicing] = init_data
            else:  # Scalar case
                padded_data = init_data
            hostbuf = padded_data

        buf = cl.Buffer(self.context, mem_flags, size=byte_size, hostbuf=hostbuf)
        self.buffers[name] = buf
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
        self.p = program
        self.b = buffer_mgr
        self.simd_width = self.b.simd_width
        self.padded_hidden_dim = self.b.get_spec("hidden_buf")[0][1]
        self.padded_input_dim = self.b.get_spec("input_buf")[0][1]
        self.scalar_size = SCALAR_NP_TYPE().itemsize

    def enqueue_write_buffer(self, queue, name, data, wait_for) -> cl.Event:
        buf = self.b.get(name)
        shape, dtype = self.b.get_spec(name)
        padded_data = np.zeros(shape, dtype=dtype)
        padded_data[tuple(slice(0, d) for d in data.shape)] = data
        return cl.enqueue_copy(queue, buf, padded_data, wait_for=wait_for)

    def enqueue_fill_buffer(self, queue, name, value, wait_for) -> cl.Event:
        buf, spec = self.b.get(name), self.b.get_spec(name)
        return cl.enqueue_fill_buffer(queue, buf, spec[1](value), 0, buf.size, wait_for=wait_for)

    def launch_forward_pass(self, queue, offset, num_samples, wait_for) -> cl.Event:
        g, l = (pad_to_multiple(num_samples, self.simd_width), self.padded_hidden_dim // self.simd_width), (
            self.simd_width,
            1,
        )
        args = (
            cl.LocalMemory(l[0] * (1 + l[0]) * self.scalar_size),
            self.b.get("input_buf"),
            self.b.get("input_mask"),
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

    def launch_compute_logits_chunk(
        self, queue, exit_chunk_id, exit_offset, num_exits, class_offset, num_classes, wait_for
    ) -> cl.Event:
        g, l = (num_exits, BATCH_SIZE, num_classes), None
        args = (
            self.b.get("hidden_buf"),
            self.b.get("hidden_mask"),
            self.b.get("exit_weights"),
            self.b.get("exit_biases"),
            self.b.get("full_logits_out"),
            np.int32(exit_chunk_id),
            np.int32(exit_offset),
            np.int32(num_exits),
            np.int32(class_offset),
            np.int32(num_classes),
            np.int32(BATCH_SIZE),
            np.int32(HIDDEN_DIM),
            np.int32(self.padded_hidden_dim),
            np.int32(OUTPUT_CLASSES),
        )
        return self.p.compute_logits_chunk(queue, g, l, *args, wait_for=wait_for)

    def launch_reduce_for_softmax(self, queue, wait_for) -> cl.Event:
        g, l = (NUM_EXITS, BATCH_SIZE), None
        args = (
            self.b.get("full_logits_out"),
            self.b.get("temps"),
            self.b.get("softmax_params_out"),
            np.int32(NUM_EXITS),
            np.int32(BATCH_SIZE),
            np.int32(OUTPUT_CLASSES),
        )
        return self.p.reduce_logits_for_softmax(queue, g, l, *args, wait_for=wait_for)

    def launch_compute_probs_loss_cce_chunk(
        self, queue, exit_chunk_id, exit_offset, num_exits, class_offset, num_classes, wait_for
    ) -> cl.Event:
        g, l = (num_exits, BATCH_SIZE, num_classes), None
        args = (
            self.b.get("full_logits_out"),
            self.b.get("softmax_params_out"),
            self.b.get("temps"),
            self.b.get("targets_buf"),
            self.b.get("input_mask"),
            self.b.get("partial_probs_out"),
            self.b.get("final_loss_out"),
            np.int32(exit_chunk_id),
            np.int32(exit_offset),
            np.int32(num_exits),
            np.int32(class_offset),
            np.int32(num_classes),
            np.int32(BATCH_SIZE),
            np.int32(OUTPUT_CLASSES),
        )
        return self.p.compute_probs_loss_cce_chunk(queue, g, l, *args, wait_for=wait_for)

    def launch_compute_probs_loss_bce_chunk(
        self, queue, exit_chunk_id, exit_offset, num_exits, class_chunk_id, class_offset, num_classes, wait_for
    ) -> cl.Event:
        """Placeholder: Launch implementation for BCE would be symmetric to CCE."""
        # This would be fully implemented for a production system.
        # For now, we return a pre-completed event if BCE is selected.
        user_event = cl.UserEvent(queue.context)
        user_event.set_status(cl.command_execution_status.COMPLETE)
        return user_event

    def launch_parallel_exit_grads(
        self, queue, exit_chunk_id, exit_offset, num_exits, class_chunk_id, class_offset, num_classes, wait_for
    ) -> Tuple[cl.Event, cl.Event, cl.Event]:
        lsize = 256
        # Note: A problem_type_flag of 1 indicates CCE, 0 for BCE in problem code, but C header has 0=CCE.
        # Assuming header is correct. Python code `PROBLEM_TYPE == "CCE"` is True (1).
        problem_flag = np.int32(0 if PROBLEM_TYPE == "CCE" else 1)

        grad_w_evt = self.p.calculate_exit_param_grads_chunk(
            queue,
            (num_exits, HIDDEN_DIM, num_classes),
            None,  # Let driver choose local size
            cl.LocalMemory(lsize * self.scalar_size),
            self.b.get("hidden_buf"),
            self.b.get("partial_probs_out"),
            self.b.get("targets_buf"),
            self.b.get("partial_grad_exit_w_out"),
            self.b.get("partial_grad_exit_b_out"),
            problem_flag,
            np.int32(exit_chunk_id),
            np.int32(exit_offset),
            np.int32(num_exits),
            np.int32(class_chunk_id),
            np.int32(class_offset),
            np.int32(num_classes),
            np.int32(BATCH_SIZE),
            np.int32(HIDDEN_DIM),
            np.int32(self.padded_hidden_dim),
            np.int32(OUTPUT_CLASSES),
            np.int32(NUM_EXITS),
            wait_for=wait_for,
        )
        grad_h_evt = self.p.backprop_error_to_hidden_chunk(
            queue,
            (num_exits, BATCH_SIZE, HIDDEN_DIM),
            None,  # Let driver choose local size
            cl.LocalMemory(0),  # Unused in kernel
            self.b.get("partial_probs_out"),
            self.b.get("targets_buf"),
            self.b.get("exit_weights"),
            self.b.get("partial_grad_h_aos_out"),
            problem_flag,
            np.int32(exit_chunk_id),
            np.int32(exit_offset),
            np.int32(num_exits),
            np.int32(class_chunk_id),
            np.int32(class_offset),
            np.int32(num_classes),
            np.int32(BATCH_SIZE),
            np.int32(HIDDEN_DIM),
            np.int32(OUTPUT_CLASSES),
            np.int32(NUM_EXITS),
            wait_for=wait_for,
        )
        grad_t_evt = self.p.calculate_chunk_temp_gradients(
            queue,
            (num_exits * lsize,),
            (lsize,),
            cl.LocalMemory(lsize * self.scalar_size),
            self.b.get("full_logits_out"),
            self.b.get("partial_probs_out"),
            self.b.get("targets_buf"),
            self.b.get("input_mask"),
            self.b.get("temps"),
            self.b.get("partial_grad_temps_out"),
            problem_flag,
            np.int32(exit_chunk_id),
            np.int32(exit_offset),
            np.int32(num_exits),
            np.int32(class_chunk_id),
            np.int32(class_offset),
            np.int32(num_classes),
            np.int32(BATCH_SIZE),
            np.int32(OUTPUT_CLASSES),
            np.int32(NUM_EXITS),
            wait_for=wait_for,
        )
        return grad_w_evt, grad_h_evt, grad_t_evt

    def launch_transpose_grad_h(self, queue, in_buf_name, out_buf_name, num_rows, num_cols, wait_for) -> cl.Event:
        g = (pad_to_multiple(num_cols, C_TILE_SIZE), pad_to_multiple(num_rows, C_TILE_SIZE))
        l = (C_TILE_SIZE, C_TILE_SIZE)
        args = (
            cl.LocalMemory(C_TILE_SIZE * (C_TILE_SIZE + 1) * self.scalar_size),
            self.b.get(in_buf_name),
            self.b.get(out_buf_name),
            np.int32(num_rows),
            np.int32(num_cols),
        )
        return self.p.transpose_grad_h(queue, g, l, *args, wait_for=wait_for)

    def launch_aggregation(
        self, queue, in_buf_name, out_buf_name, num_partials, elements_per_partial, is_avg, wait_for
    ) -> cl.Event:
        lsize = 256
        mode = np.int32(1 if is_avg else 0)
        in_buf, out_buf = self.b.get(in_buf_name), self.b.get(out_buf_name)

        # Ensure elements_per_partial is a non-zero integer
        elements_per_partial = max(1, int(elements_per_partial))

        if num_partials <= 1:
            return self.p.aggregate_identity(
                queue,
                (elements_per_partial,),
                None,
                cl.LocalMemory(0),
                in_buf,
                out_buf,
                np.int32(1),
                np.int32(elements_per_partial),
                mode,
                wait_for=wait_for,
            )
        elif num_partials <= MAX_REGISTER_AGGREGATE_ITEMS:
            return self.p.aggregate_register_reduce(
                queue,
                (elements_per_partial,),
                None,
                cl.LocalMemory(0),
                in_buf,
                out_buf,
                np.int32(num_partials),
                np.int32(elements_per_partial),
                mode,
                wait_for=wait_for,
            )
        else:
            gsize = pad_to_multiple(elements_per_partial, lsize)
            return self.p.aggregate_local_reduce(
                queue,
                (gsize,),
                (lsize,),
                cl.LocalMemory(lsize * self.scalar_size),
                in_buf,
                out_buf,
                np.int32(num_partials),
                np.int32(elements_per_partial),
                mode,
                wait_for=wait_for,
            )

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
            self.b.get("input_mask"),
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
            self.b.get("input_mask"),
            self.b.get("partial_grad_sb_out"),
            np.int32(offset),
            np.int32(num_samples),
            np.int32(chunk_id),
            np.int32(self.padded_hidden_dim),
            wait_for=wait_for,
        )
        return sw_evt, sb_evt

    def launch_adam_update_and_clamp(self, queue, p: Parameter, grad_ready_event, b1_t, b2_t) -> cl.Event:
        num_params = int(np.prod(self.b.get_spec(p.name)[0]))
        update_evt = self.p.adam_update(
            queue,
            (num_params,),
            None,
            self.b.get(p.grad),
            SCALAR_NP_TYPE(ADAM_BETA1),
            SCALAR_NP_TYPE(ADAM_BETA2),
            SCALAR_NP_TYPE(b1_t),
            SCALAR_NP_TYPE(b2_t),
            SCALAR_NP_TYPE(LEARNING_RATE),
            SCALAR_NP_TYPE(EPSILON),
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
                (NUM_EXITS,),
                None,
                self.b.get("temps"),
                SCALAR_NP_TYPE(MIN_TEMP),
                SCALAR_NP_TYPE(MAX_TEMP),
                np.int32(NUM_EXITS),
                wait_for=[update_evt],
            )
        return update_evt


# --- Layer 2: Strategic Layer ---


@dataclass
class ExecutionPlan:
    """Holds the strategic decisions for processing one batch."""

    num_exit_chunks: int = 1
    num_batch_chunks: int = 1
    recompute_hidden: bool = False


class ExecutionStrategy:
    """Makes high-level strategic decisions based on resources and problem size."""

    def __init__(self, device_vram_bytes: int, buffer_mgr: BufferManager):
        # Use 85% of VRAM as a safe budget
        self.vram_budget = device_vram_bytes * 0.85
        self.b = buffer_mgr

    def create_plan_for_batch(self, batch_size: int) -> ExecutionPlan:
        # --- Estimate memory for persistent intermediate buffers ---
        # These are buffers that must live for a large portion of the batch.
        hidden_size = self.b.get_byte_size("hidden_buf")
        logits_size = self.b.get_byte_size("full_logits_out")
        probs_size = self.b.get_byte_size("partial_probs_out")

        ## Use the correct (large) size for the chunked partial grad_h.
        grad_h_aos_size = self.b.get_byte_size("partial_grad_h_aos_out")

        # Estimate memory for one full, unchunked pass
        required_fwd_mem = logits_size + probs_size
        required_bwd_mem = grad_h_aos_size  # grad_h is the largest partial grad

        # --- Make Strategic Decisions ---
        plan = ExecutionPlan()

        # Decision 1: Do we need to stream the exit/class dimension?
        if required_fwd_mem + required_bwd_mem > self.vram_budget:
            # Not enough memory to hold all intermediate buffers. Must stream.
            num_chunks = math.ceil((required_fwd_mem + required_bwd_mem) / self.vram_budget)
            plan.num_exit_chunks = min(num_chunks, MAX_EXIT_CHUNKS)
            print(f"INFO: High memory pressure. Streaming exits in {plan.num_exit_chunks} chunks.")

        # Decision 2: Do we need to recompute hidden activations?
        # This is for the "Behemoth" case: network is so fat that just the
        # hidden activations are too large to keep during backprop.
        if hidden_size + max(required_fwd_mem, required_bwd_mem) > self.vram_budget:
            plan.recompute_hidden = True
            print(f"INFO: Extreme memory pressure. `hidden` buffer will be recomputed.")

        # Decision 3: Determine batch streaming chunks for shared layer backprop.
        # This is based on a fixed chunk size for now, but could also be memory-based.
        plan.num_batch_chunks = (batch_size + BACKPROP_STREAM_CHUNK_SIZE - 1) // BACKPROP_STREAM_CHUNK_SIZE
        plan.num_batch_chunks = min(plan.num_batch_chunks, MAX_BATCH_CHUNKS)

        return plan


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
            self.executor.b.get_spec("final_probs_buf"), (NUM_EXITS, X_batch.shape[0], OUTPUT_CLASSES)
        )
        self.host_loss_view = HostView(self.executor.b.get_spec("final_loss_out"), (NUM_EXITS, X_batch.shape[0]))

    def _get_deps(self, *resource_names: str) -> List[cl.Event]:
        deps = [self.events[name] for name in resource_names if name in self.events]
        for name in resource_names:
            deps.extend(self.event_lists.get(name, []))
        return list(set(deps))

    def run(self, plan: ExecutionPlan):
        self._upload_data()
        self._zero_gradients()
        fwd_pass_deps = self._get_deps("input_ready", "grads_zeroed")
        self.events["hidden_ready"] = self.executor.launch_forward_pass(
            self.queue, 0, self.X_batch.shape[0], wait_for=fwd_pass_deps
        )
        self._execute_exit_path(plan)
        self._aggregate_and_backprop_shared(plan)
        self._finalize_and_update()
        return self.get_sync_points()

    def _upload_data(self):
        write_X = self.executor.enqueue_write_buffer(self.queue, "input_buf", self.X_batch, None)
        write_mask = self.executor.enqueue_write_buffer(
            self.queue, "input_mask", np.ones(self.X_batch.shape[0], dtype=SCALAR_NP_TYPE), None
        )
        self.events["input_ready"] = cl.WaitForEvents([write_X, write_mask])
        self.events["targets_ready"] = self.executor.enqueue_write_buffer(self.queue, "targets_buf", self.y_batch, None)

    def _zero_gradients(self):
        # Get all buffers that start with 'grad_' or 'partial_grad_'
        grad_names = [name for name in self.executor.b.buffers if "grad_" in name]
        zero_events = [self.executor.enqueue_fill_buffer(self.queue, name, 0, None) for name in grad_names]
        self.events["grads_zeroed"] = cl.WaitForEvents(zero_events)

    def _execute_exit_path(self, plan: ExecutionPlan):
        exit_chunk_size = (NUM_EXITS + plan.num_exit_chunks - 1) // plan.num_exit_chunks
        class_chunk_size = (OUTPUT_CLASSES + plan.num_exit_chunks - 1) // plan.num_exit_chunks

        hidden_deps = self._get_deps("hidden_ready")
        for i in range(plan.num_exit_chunks):
            exit_offset, exit_size = i * exit_chunk_size, min(exit_chunk_size, NUM_EXITS - i * exit_chunk_size)
            class_offset, class_size = i * class_chunk_size, min(
                class_chunk_size, OUTPUT_CLASSES - i * class_chunk_size
            )
            if exit_size <= 0 or class_size <= 0:
                continue

            self.event_lists["logit_chunks_ready"].append(
                self.executor.launch_compute_logits_chunk(
                    self.queue, i, exit_offset, exit_size, class_offset, class_size, hidden_deps
                )
            )
        self.events["all_logits_ready"] = cl.WaitForEvents(self.event_lists["logit_chunks_ready"])

        ## Create a per-chunk event list to manage dependencies correctly.
        prob_loss_events = []

        if PROBLEM_TYPE == "CCE":
            self.events["softmax_params_ready"] = self.executor.launch_reduce_for_softmax(
                self.queue, self._get_deps("all_logits_ready")
            )

        prob_loss_deps = self._get_deps("softmax_params_ready", "targets_ready")

        for i in range(plan.num_exit_chunks):
            exit_offset, exit_size = i * exit_chunk_size, min(exit_chunk_size, NUM_EXITS - i * exit_chunk_size)
            class_offset, class_size = i * class_chunk_size, min(
                class_chunk_size, OUTPUT_CLASSES - i * class_chunk_size
            )
            if exit_size <= 0 or class_size <= 0:
                continue

            if PROBLEM_TYPE == "CCE":
                evt = self.executor.launch_compute_probs_loss_cce_chunk(
                    self.queue, i, exit_offset, exit_size, class_offset, class_size, prob_loss_deps
                )
                prob_loss_events.append(evt)
            else:  # BCE Path
                evt = self.executor.launch_compute_probs_loss_bce_chunk(
                    self.queue, i, exit_offset, exit_size, i, class_offset, class_size, prob_loss_deps
                )
                prob_loss_events.append(evt)
        self.events["all_prob_loss_chunks_ready"] = cl.WaitForEvents(prob_loss_events)

        grad_base_deps = self._get_deps("hidden_ready", "all_logits_ready", "targets_ready")
        for i in range(plan.num_exit_chunks):
            exit_offset, exit_size = i * exit_chunk_size, min(exit_chunk_size, NUM_EXITS - i * exit_chunk_size)
            class_offset, class_size = i * class_chunk_size, min(
                class_chunk_size, OUTPUT_CLASSES - i * class_chunk_size
            )
            if exit_size <= 0 or class_size <= 0:
                continue

            ## Depend only on the current chunk's prob/loss event.
            grad_deps = grad_base_deps + [prob_loss_events[i]]

            w, h, t = self.executor.launch_parallel_exit_grads(
                self.queue, i, exit_offset, exit_size, i, class_offset, class_size, grad_deps
            )
            self.event_lists["partial_grad_w_ready"].append(w)
            self.event_lists["partial_grad_h_ready"].append(h)
            self.event_lists["partial_grad_t_ready"].append(t)

        self.events["all_partials_ready"] = cl.WaitForEvents(
            self.event_lists["partial_grad_w_ready"]
            + self.event_lists["partial_grad_h_ready"]
            + self.event_lists["partial_grad_t_ready"]
        )

    def _aggregate_and_backprop_shared(self, plan: ExecutionPlan):
        ## Correctly aggregate all partial results and implement recompute strategy.

        # --- Aggregate Phase 1 (Exit Layer Gradients) ---
        num_exit_chunks = plan.num_exit_chunks

        # Aggregate partial probabilities
        self.events["final_probs_ready"] = self.executor.launch_aggregation(
            self.queue,
            "partial_probs_out",
            "final_probs_buf",
            num_partials=num_exit_chunks,
            elements_per_partial=self.executor.b.get("final_probs_buf").size // num_exit_chunks,
            is_avg=False,
            wait_for=self._get_deps("all_prob_loss_chunks_ready"),
        )

        # Aggregate Exit Weights and Biases Gradients
        self.final_grad_events["grad_exit_weights"] = self.executor.launch_aggregation(
            self.queue,
            "partial_grad_exit_w_out",
            "grad_exit_weights",
            num_partials=num_exit_chunks,
            elements_per_partial=self.executor.b.get("grad_exit_weights").size,
            is_avg=False,
            wait_for=self.event_lists["partial_grad_w_ready"],
        )
        self.final_grad_events["grad_exit_biases"] = self.executor.launch_aggregation(
            self.queue,
            "partial_grad_exit_b_out",
            "grad_exit_biases",
            num_partials=num_exit_chunks,
            elements_per_partial=self.executor.b.get("grad_exit_biases").size,
            is_avg=False,
            wait_for=self.event_lists["partial_grad_w_ready"],
        )

        # Aggregate Temperature Gradients
        self.final_grad_events["grad_temps"] = self.executor.launch_aggregation(
            self.queue,
            "partial_grad_temps_out",
            "grad_temps",
            num_partials=num_exit_chunks,
            elements_per_partial=self.executor.b.get("grad_temps").size,
            is_avg=False,
            wait_for=self.event_lists["partial_grad_t_ready"],
        )

        # Aggregate and Transform Hidden Gradients (Grad_H)
        agg_h_evt = self.executor.launch_aggregation(
            self.queue,
            "partial_grad_h_aos_out",
            "aggregated_grad_h_aos_buf",
            num_partials=num_exit_chunks,
            elements_per_partial=self.executor.b.get("aggregated_grad_h_aos_buf").size,
            is_avg=False,
            wait_for=self.event_lists["partial_grad_h_ready"],
        )

        transpose_evt = self.executor.launch_transpose_grad_h(
            self.queue,
            "aggregated_grad_h_aos_buf",
            "grad_h_soa_buf",
            num_rows=NUM_EXITS,
            num_cols=(BATCH_SIZE * HIDDEN_DIM),
            wait_for=[agg_h_evt],
        )

        self.events["final_grad_h_ready"] = self.executor.launch_aggregation(
            self.queue,
            "grad_h_soa_buf",
            "final_grad_h_buf",
            num_partials=NUM_EXITS,
            elements_per_partial=(BATCH_SIZE * HIDDEN_DIM),
            is_avg=False,
            wait_for=[transpose_evt],
        )

        # --- Backprop Shared Layer ---
        backprop_deps = self._get_deps("final_grad_h_ready")

        ## Implement Recompute vs. Cache strategy for `hidden_buf`.
        if plan.recompute_hidden:
            print("  - Recomputing hidden activations for backprop.")
            # Simple recompute: re-run full forward pass. More advanced would be chunked recompute.
            recompute_event = self.executor.launch_forward_pass(
                self.queue, 0, self.X_batch.shape[0], wait_for=self._get_deps("input_ready")
            )
            backprop_deps.append(recompute_event)
        else:
            backprop_deps.extend(self._get_deps("hidden_ready"))

        for i in range(plan.num_batch_chunks):
            offset = i * BACKPROP_STREAM_CHUNK_SIZE
            size = min(BACKPROP_STREAM_CHUNK_SIZE, BATCH_SIZE - offset)
            if size <= 0:
                continue
            sw, sb = self.executor.launch_backprop_shared_chunk(self.queue, i, offset, size, backprop_deps)
            self.event_lists["partial_grad_sw_ready"].append(sw)
            self.event_lists["partial_grad_sb_ready"].append(sb)

        # Aggregate Shared Layer Gradients
        self.final_grad_events["grad_weights"] = self.executor.launch_aggregation(
            self.queue,
            "partial_grad_sw_out",
            "grad_weights",
            num_partials=plan.num_batch_chunks,
            elements_per_partial=self.executor.b.get("grad_weights").size,
            is_avg=True,
            wait_for=self.event_lists["partial_grad_sw_ready"],
        )

        self.final_grad_events["grad_biases"] = self.executor.launch_aggregation(
            self.queue,
            "partial_grad_sb_out",
            "grad_biases",
            num_partials=plan.num_batch_chunks,
            elements_per_partial=self.executor.b.get("grad_biases").size,
            is_avg=True,
            wait_for=self.event_lists["partial_grad_sb_ready"],
        )

    def _finalize_and_update(self):
        self.events["d2h_probs_ready"] = self.host_probs_view.enqueue_read(
            self.queue, self.executor.b.get("final_probs_buf"), self._get_deps("final_probs_ready")
        )
        b1_t, b2_t = (ADAM_BETA1**self.global_step), (ADAM_BETA2**self.global_step)
        for p in self.orchestrator.params:
            if p.grad not in self.final_grad_events:
                # This can happen if a parameter group has no gradients in this step
                continue
            grad_ready_event = self.final_grad_events[p.grad]
            self.event_lists["updates_done"].append(
                self.executor.launch_adam_update_and_clamp(self.queue, p, grad_ready_event, b1_t, b2_t)
            )
        self.events["all_updates_done"] = cl.WaitForEvents(self.event_lists["updates_done"])

    def get_sync_points(self) -> Tuple[cl.UserEvent, cl.UserEvent]:
        inference_event, final_batch_event = cl.UserEvent(self.ctx), cl.UserEvent(self.ctx)
        d2h_event = self.events.get("d2h_probs_ready")
        if d2h_event:
            cl.enqueue_barrier_with_wait_list([d2h_event]).set_callback(
                cl.command_execution_status.COMPLETE,
                lambda s, u: inference_event.set_status(cl.command_execution_status.COMPLETE),
            )
        else:  # Always complete inference event if no copy back happened
            inference_event.set_status(cl.command_execution_status.COMPLETE)

        final_event = self.events.get("all_updates_done")
        if final_event:
            cl.enqueue_barrier_with_wait_list([final_event]).set_callback(
                cl.command_execution_status.COMPLETE,
                lambda s, u: final_batch_event.set_status(cl.command_execution_status.COMPLETE),
            )
        else:  # Always complete final event
            final_batch_event.set_status(cl.command_execution_status.COMPLETE)

        return inference_event, final_batch_event


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
            Parameter("weights", BufferRole.SHARED_WEIGHTS),
            Parameter("biases", BufferRole.SHARED_BIAS),
            Parameter("exit_weights", BufferRole.EXIT_WEIGHTS),
            Parameter("exit_biases", BufferRole.EXIT_BIAS),
            Parameter("temps", BufferRole.TEMPERATURES),
        ]
        self.param_shapes: Dict[str, Tuple] = {
            "weights": (INPUT_DIM, HIDDEN_DIM),
            "biases": (HIDDEN_DIM,),
            "exit_weights": (NUM_EXITS, HIDDEN_DIM, OUTPUT_CLASSES),
            "exit_biases": (NUM_EXITS, OUTPUT_CLASSES),
            "temps": (NUM_EXITS,),
        }

        self.buffer_mgr = BufferManager(self.ctx, self.simd_width)
        self._setup_buffers()

        self.executor = KernelExecutor(self.program, self.buffer_mgr)
        self.strategy = ExecutionStrategy(self.device.global_mem_size, self.buffer_mgr)
        self.global_step = 1

    def _compile_kernels(self) -> cl.Program:
        kernel_src = load_and_concatenate_kernels()
        build_opts = [
            f"-cl-std=CL1.2",
            f"-D SCALAR_TYPE={CL_SCALAR_TYPE}",
            f"-D SIMD_WIDTH={self.simd_width}",
            f"-D C_TILE_SIZE={C_TILE_SIZE}",
        ]
        if SCALAR_TYPE == "half":
            build_opts.append("-D cl_khr_fp16")
        return cl.Program(self.ctx, kernel_src).build(options=build_opts)

    def _setup_buffers(self):
        # Create trainable parameters and their Adam optimizer states
        for p in self.params:
            shape = self.param_shapes[p.name]
            init_fn = (
                np.zeros
                if "bias" in p.name
                else (
                    (lambda s: np.full(s, 1.0, dtype=SCALAR_NP_TYPE))
                    if "temp" in p.name
                    else (lambda s: np.random.randn(*s).astype(SCALAR_NP_TYPE) * 0.01)
                )
            )
            self.buffer_mgr.create_buffer(p.name, p.role, shape, SCALAR_NP_TYPE, init_fn(shape))
            padded_shape, dtype = self.buffer_mgr.get_spec(p.name)
            self.buffer_mgr.create_buffer(p.grad, BufferRole.FINAL_GRADIENT, padded_shape, dtype)
            self.buffer_mgr.create_buffer(p.m1, BufferRole.ADAM_MOMENTUM, padded_shape, dtype)
            self.buffer_mgr.create_buffer(p.m2, BufferRole.ADAM_MOMENTUM, padded_shape, dtype)

        # Create all intermediate, mask, and target buffers
        dtype_map = {"targets_buf": np.int32} if PROBLEM_TYPE == "CCE" else {}

        ## Define buffer shapes correctly, especially for partial gradients.
        ## These buffers now have a leading dimension for storing chunked results.
        int_buffers = {
            "input_buf": (BufferRole.INPUT, (BATCH_SIZE, INPUT_DIM)),
            "input_mask": (BufferRole.MASK, (BATCH_SIZE,)),
            "hidden_buf": (BufferRole.HIDDEN_ACTIVATION, (BATCH_SIZE, HIDDEN_DIM)),
            "hidden_mask": (BufferRole.MASK, (BATCH_SIZE,)),
            "targets_buf": (
                BufferRole.TARGETS,
                (BATCH_SIZE,) if PROBLEM_TYPE == "CCE" else (BATCH_SIZE, OUTPUT_CLASSES),
            ),
            "full_logits_out": (BufferRole.INTERMEDIATE, (NUM_EXITS, BATCH_SIZE, OUTPUT_CLASSES)),
            "partial_probs_out": (BufferRole.INTERMEDIATE, (NUM_EXITS, BATCH_SIZE, OUTPUT_CLASSES)),
            "final_probs_buf": (BufferRole.INTERMEDIATE, (NUM_EXITS, BATCH_SIZE, OUTPUT_CLASSES)),
            "final_loss_out": (BufferRole.INTERMEDIATE, (NUM_EXITS, BATCH_SIZE)),
            "softmax_params_out": (BufferRole.INTERMEDIATE, (NUM_EXITS, BATCH_SIZE, 2)),
            # Partial gradient buffers now sized for worst-case chunking
            "partial_grad_exit_w_out": (
                BufferRole.PARTIAL_GRADIENT,
                (MAX_EXIT_CHUNKS, *self.param_shapes["exit_weights"]),
            ),
            "partial_grad_exit_b_out": (
                BufferRole.PARTIAL_GRADIENT,
                (MAX_EXIT_CHUNKS, *self.param_shapes["exit_biases"]),
            ),
            "partial_grad_temps_out": (BufferRole.PARTIAL_GRADIENT, (MAX_EXIT_CHUNKS, *self.param_shapes["temps"])),
            "partial_grad_h_aos_out": (
                BufferRole.PARTIAL_GRADIENT,
                (MAX_EXIT_CHUNKS, NUM_EXITS, BATCH_SIZE, HIDDEN_DIM),
            ),
            "partial_grad_sw_out": (BufferRole.PARTIAL_GRADIENT, (MAX_BATCH_CHUNKS, *self.param_shapes["weights"])),
            "partial_grad_sb_out": (BufferRole.PARTIAL_GRADIENT, (MAX_BATCH_CHUNKS, *self.param_shapes["biases"])),
            # Buffers for multi-step gradient calculations (e.g., aggregate -> transpose -> aggregate)
            "aggregated_grad_h_aos_buf": (BufferRole.PARTIAL_GRADIENT, (NUM_EXITS, BATCH_SIZE, HIDDEN_DIM)),
            "grad_h_soa_buf": (BufferRole.PARTIAL_GRADIENT, (BATCH_SIZE * HIDDEN_DIM, NUM_EXITS)),
            "final_grad_h_buf": (BufferRole.FINAL_GRADIENT, (BATCH_SIZE, HIDDEN_DIM)),
        }
        for name, (role, shape) in int_buffers.items():
            self.buffer_mgr.create_buffer(name, role, shape, dtype_map.get(name, SCALAR_NP_TYPE))

    def train(self):
        from sklearn.datasets import load_iris
        from sklearn.preprocessing import StandardScaler, OneHotEncoder

        X, y = load_iris(return_X_y=True)
        # Ensure data matches our network dimensions
        X = X[:, :INPUT_DIM].astype(SCALAR_NP_TYPE)
        y = y[y < OUTPUT_CLASSES]
        X = X[: y.shape[0]]

        X = StandardScaler().fit_transform(X).astype(SCALAR_NP_TYPE)

        print(f"\n--- Starting training for {EPOCHS} epochs ---")
        for epoch in range(EPOCHS):
            full_batch_size = X.shape[0]
            start_idx = (epoch * BATCH_SIZE) % full_batch_size
            end_idx = start_idx + BATCH_SIZE
            indices = np.arange(start_idx, end_idx) % full_batch_size

            X_batch, y_batch_int = X[indices], y[indices]

            if PROBLEM_TYPE == "CCE":
                y_b = y_batch_int.astype(np.int32)
            else:  # BCE
                y_b = OneHotEncoder().fit_transform(y_batch_int.reshape(-1, 1)).toarray().astype(SCALAR_NP_TYPE)

            print(f"Epoch {self.global_step:4d} | Batch Size: {X_batch.shape[0]}")
            plan = self.strategy.create_plan_for_batch(X_batch.shape[0])
            processor = BatchProcessor(self, X_batch, y_b)
            inference_event, final_event = processor.run(plan)

            inference_event.wait()
            valid_probs = processor.host_probs_view.get()

            # Note: For logging simplicity, accuracy is based on the first exit's predictions.
            predicted_classes = np.argmax(valid_probs[0, :, :], axis=1)
            accuracy = np.mean(predicted_classes == y_batch_int)

            print(f"  - Inference Latency Path Complete. Accuracy: {accuracy:.2%}")
            final_event.wait()
            print(f"  - Full Training Step Complete.")
            self.global_step += 1
        print("\n--- Training finished ---")


if __name__ == "__main__":
    try:
        # Create dummy kernel files if they don't exist, to allow the script to run.
        # In a real scenario, these files would contain the actual OpenCL kernel code.
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
