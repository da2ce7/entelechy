# iris_dynamic_cl.py
#
# A Unified, Memory-Aware Streaming Classification Engine (Host Implementation)
# ===========================================================================
#
# This system implements the host-side contract defined in the Architectural
# Hierarchy:
#
# 1. Design Document: The canonical specification (Layer 1 authority)
# 2. kernels.cl.h: Device contract (Layer 2 authority)
# 3. This Implementation: Host logic (Layer 3 authority)
#
# Core Architectural Compliance:
# - Strict adherence to kernel interface contracts (no direct buffer access)
# - Pure expression of Primacy of Memory Strategy through chunking/recomputation
# - Event-driven DAG construction with driver-level optimization trust
# - Unified streaming paradigm scaled via recursive halving renderer

# Key Host Responsibilities
# -------------------------
# Orchestration:
#   - Whole-training lifecycle management
#   - Strategic VRAM budgeting and chunking
#
# Execution Planning:
#   - Adaptive selection of cache/recompute strategies
#   - Recursive reduction tree configuration (K param)
#
# Kernel Contract Enforcement:
#   - Physical buffer layout maintenance (SoA/AoS)
#   - Implicit padding management for all kernel arguments
#
# Memory Sovereignty:
#   - Opaque handle-based memory management
#   - Strict separation of transient vs. persistent allocations
#
# Reference: See DESIGN.md §3 for architectural diagrams and host/device contracts.

import os
import pyopencl as cl
import numpy as np
import math
from collections import defaultdict
from dataclasses import dataclass
from typing import Tuple, List, Dict, Optional, Callable, Union
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
MAX_REGISTER_AGGREGATE_ITEMS: int = 32
C_TILE_SIZE: int = 16
BACKPROP_STREAM_CHUNK_SIZE: int = 32
REDUCTION_BATCH_SIZE_K: int = 16  # Defines the width of the recursive reduction front
MAX_BATCH_CHUNKS: int = 64


# --- Core Data Abstractions ---


@dataclass(frozen=True, eq=True)
class BufferHandle:
    """An opaque handle to a memory buffer managed by the BufferManager."""

    id: int


class LayoutType(enum.Enum):
    AoS = 1
    SoA = 2


class BufferRole(enum.Enum):
    (
        INPUT,
        HIDDEN_ACTIVATION,
        SHARED_WEIGHTS,
        SHARED_BIAS,
        MODULE_WEIGHTS,
        MODULE_BIAS,
        TEMPERATURES,
        PARTIAL_GRADIENT,
        FINAL_GRADIENT,
        ADAM_MOMENTUM,
        INTERMEDIATE,
        TARGETS,
        MASK,
    ) = range(13)


@dataclass(frozen=True)
class Parameter:
    name: str
    role: BufferRole
    layout: LayoutType = LayoutType.AoS

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
    def __init__(self, spec: Tuple[Tuple, np.dtype], real_shape: Tuple):
        self.padded_shape, self.dtype = spec[0], spec[1]
        self.real_shape = real_shape
        self.host_data = np.empty(self.padded_shape, dtype=self.dtype)

    def enqueue_read(self, queue: cl.CommandQueue, cl_buffer: cl.Buffer, wait_for=None) -> cl.Event:
        return cl.enqueue_copy(queue, self.host_data, cl_buffer, wait_for=wait_for or [])

    def get(self) -> np.ndarray:
        slicing = tuple(slice(0, dim) for dim in self.real_shape)
        return self.host_data[slicing] if slicing else self.host_data


# --- 2D Tiling & Strategic Planning ---
@dataclass(frozen=True)
class WorkTile:
    module_chunk_idx: int
    class_chunk_idx: int
    flat_tile_id: int
    module_offset: int
    num_modules_in_tile: int
    class_offset: int
    num_classes_in_tile: int


@dataclass(frozen=True)
class ExecutionGrid:
    num_module_chunks: int
    num_class_chunks: int
    total_modules: int
    total_classes: int

    @property
    def total_tiles(self) -> int:
        return self.num_module_chunks * self.num_class_chunks

    def __iter__(self):
        for m_idx in range(self.num_module_chunks):
            for c_idx in range(self.num_class_chunks):
                yield self.get_tile(m_idx, c_idx)

    def get_tile(self, m_idx: int, c_idx: int) -> "WorkTile":
        m_chunk_size = (self.total_modules + self.num_module_chunks - 1) // self.num_module_chunks
        c_chunk_size = (self.total_classes + self.num_class_chunks - 1) // self.num_class_chunks
        m_offset, num_m = m_idx * m_chunk_size, min(m_chunk_size, self.total_modules - (m_idx * m_chunk_size))
        c_offset, num_c = c_idx * c_chunk_size, min(c_chunk_size, self.total_classes - (c_idx * c_chunk_size))
        return WorkTile(m_idx, c_idx, m_idx * self.num_class_chunks + c_idx, m_offset, num_m, c_offset, num_c)


@dataclass(frozen=True)
class ChunkingConfig:
    num_chunks: int = 1
    chunk_size: int = 0
    total_dim: int = 0


@dataclass(frozen=True)
class ReductionPlan:
    k: int


@dataclass
class ExecutionPlan:
    grid: ExecutionGrid
    shared_layer_batch_chunking: ChunkingConfig
    reduction_plan: ReductionPlan
    recompute_hidden: bool = False


# --- Utility & Padding Functions ---
def load_and_concatenate_kernels() -> str:
    kernel_files = [
        "kernels.cl.h",
        "chunk_kernels.cl.c",
        "aggregation_kernels.cl.c",
        "backprop_kernels.cl.c",
        "parameter_optim.cl.c",
    ]
    source_parts = [open(f, "r", encoding="utf-8").read() for f in kernel_files if os.path.exists(f)]
    if len(source_parts) != len(kernel_files):
        raise FileNotFoundError("Missing required kernel files.")
    print(f"Loaded {len(source_parts)} kernel source files.")
    return "\n".join(source_parts)


def select_simd_width(device: cl.Device) -> int:
    vendor = device.vendor.upper()
    if "NVIDIA" in vendor:
        return 32
    if any(v in vendor for v in ["AMD", "ADVANCED MICRO DEVICES"]):
        return 64
    if "INTEL" in vendor:
        return 16
    return 8


def pad_to_multiple(dim: int, multiple: int) -> int:
    return (dim + multiple - 1) // multiple * multiple if multiple != 0 else dim


_pad_none = lambda s, *a: s


def _pad_last_dim(s, w):
    p = list(s)
    p[-1] = pad_to_multiple(p[-1], w) if len(p) > 0 and p[-1] > 1 else p[-1]
    return tuple(p)


def _pad_batch_dim(s, w):
    p = list(s)
    p[0] = pad_to_multiple(p[0], w) if len(p) > 0 else p[0]
    return tuple(p)


def _pad_batch_and_last_dim(s, w):
    p = list(s)
    if len(p) > 1:
        p[0] = pad_to_multiple(p[0], w)
        p[-1] = pad_to_multiple(p[-1], w) if p[-1] > 1 else p[-1]
    elif len(p) == 1:
        p[0] = pad_to_multiple(p[0], w)
    return tuple(p)


PADDING_RULES: Dict[BufferRole, Callable] = {
    BufferRole.INPUT: _pad_batch_and_last_dim,
    B: _pad_batch_and_last_dim,
    BufferRole.SHARED_WEIGHTS: _pad_last_dim,
    BufferRole.MODULE_WEIGHTS: _pad_last_dim,
    BufferRole.FINAL_GRADIENT: _pad_last_dim,
    BufferRole.ADAM_MOMENTUM: _pad_last_dim,
    BufferRole.INTERMEDIATE: _pad_last_dim,
    BufferRole.MASK: _pad_batch_dim,
}

# --- Layer 1: Action Layer ---


class BufferManager:
    """Manages the lifecycle of ALL OpenCL buffers, issuing opaque handles."""

    def __init__(self, context: cl.Context, simd_width: int):
        self.context = context
        self.simd_width = simd_width
        self._next_handle_id = 0
        self._handle_to_buffer: Dict[BufferHandle, cl.Buffer] = {}
        self._handle_to_spec: Dict[BufferHandle, Tuple[Tuple[int, ...], np.dtype]] = {}
        self._name_to_handle: Dict[str, BufferHandle] = {}

    def _get_new_handle(self) -> BufferHandle:
        handle = BufferHandle(id=self._next_handle_id)
        self._next_handle_id += 1
        return handle

    def create_buffer(
        self,
        name: str,
        role: BufferRole,
        layout: LayoutType,
        logical_shape: Tuple,
        dtype: np.dtype,
        init_data: Optional[np.ndarray] = None,
    ) -> None:
        """Creates a named, long-lived buffer, managed internally."""
        physical_data, physical_shape = init_data, logical_shape
        if layout == LayoutType.SoA:
            if len(logical_shape) != 2:
                raise ValueError(f"SoA layout only for 2D matrices, shape is {logical_shape}")
            padded_i, padded_h = pad_to_multiple(logical_shape[0], self.simd_width), pad_to_multiple(
                logical_shape[1], self.simd_width
            )
            padded_aos = np.zeros((padded_i, padded_h), dtype=dtype)
            padded_aos[: logical_shape[0], : logical_shape[1]] = init_data if init_data is not None else 0
            transformed = padded_aos.T.reshape(padded_h // self.simd_width, self.simd_width, padded_i).transpose(
                0, 2, 1
            )
            physical_data, physical_shape = transformed, transformed.shape
        padded_shape = PADDING_RULES.get(role, _pad_none)(physical_shape, self.simd_width)
        byte_size = max(4, int(np.prod(padded_shape) * dtype().itemsize) if padded_shape else 4)
        mem_flags, hostbuf = cl.mem_flags.READ_WRITE, None
        if physical_data is not None:
            mem_flags |= cl.mem_flags.COPY_HOST_PTR
            if physical_data.shape != padded_shape:
                hostbuf = np.zeros(padded_shape, dtype=dtype)
                hostbuf[tuple(slice(0, d) for d in physical_data.shape)] = physical_data
            else:
                hostbuf = physical_data
        handle = self._get_new_handle()
        self._name_to_handle[name] = handle
        self._handle_to_buffer[handle] = cl.Buffer(self.context, mem_flags, size=byte_size, hostbuf=hostbuf)
        self._handle_to_spec[handle] = (padded_shape, dtype)

    def acquire_transient_buffer(self, size_bytes: int) -> BufferHandle:
        """Leases an unnamed, temporary buffer from the manager."""
        handle = self._get_new_handle()
        self._handle_to_buffer[handle] = cl.Buffer(self.context, cl.mem_flags.READ_WRITE, size=max(4, size_bytes))
        return handle

    def release_transient_buffer(self, handle: BufferHandle):
        """Returns a transient buffer to the manager, allowing it to be recycled."""
        if handle in self._handle_to_buffer:
            del self._handle_to_buffer[handle]

    def get_cl_buffer(self, ref: Union[str, BufferHandle]) -> cl.Buffer:
        """The single gateway to resolve a reference (name or handle) to a cl.Buffer."""
        handle = self._name_to_handle.get(ref) if isinstance(ref, str) else ref
        if handle is None or handle not in self._handle_to_buffer:
            raise KeyError(f"No buffer found for reference: {ref}")
        return self._handle_to_buffer[handle]

    def get_spec(self, ref: Union[str, BufferHandle]) -> Tuple[Tuple[int, ...], np.dtype]:
        handle = self._name_to_handle.get(ref) if isinstance(ref, str) else ref
        if handle is None or handle not in self._handle_to_spec:
            raise KeyError(f"No spec found for reference: {ref}")
        return self._handle_to_spec[handle]


class KernelExecutor:
    """A pure, stateless toolbox for launching every specific kernel. Accepts handles."""

    def __init__(self, program: cl.Program, buffer_mgr: BufferManager):
        self.p, self.b = program, buffer_mgr
        self.simd_width = self.b.simd_width
        self.padded_hidden_dim = self.b.get_spec("hidden_buf")[0][1]
        self.padded_input_dim = self.b.get_spec("input_buf")[0][1]
        self.padded_output_classes = self.b.get_spec("full_logits_out")[0][2]
        self.scalar_size = SCALAR_NP_TYPE().itemsize

    def enqueue_write_buffer(self, queue, ref, data, wait_for) -> cl.Event:
        buf, (shape, dtype) = self.b.get_cl_buffer(ref), self.b.get_spec(ref)
        padded_data = np.zeros(shape, dtype=dtype)
        padded_data[tuple(slice(0, d) for d in data.shape)] = data
        return cl.enqueue_copy(queue, buf, padded_data, wait_for=wait_for)

    def enqueue_fill_buffer(self, queue, ref, value, wait_for) -> cl.Event:
        buf, spec = self.b.get_cl_buffer(ref), self.b.get_spec(ref)
        return cl.enqueue_fill_buffer(queue, buf, spec[1](value), 0, buf.size, wait_for=wait_for)

    def launch_forward_pass(
        self, queue, offset, num_samples, wait_for, in_ref, mask_ref, w_ref, b_ref, h_out_ref, h_mask_ref
    ) -> cl.Event:
        g, l = (pad_to_multiple(num_samples, self.simd_width), self.padded_hidden_dim // self.simd_width), (
            self.simd_width,
            1,
        )
        args = (
            cl.LocalMemory(l[0] * (1 + l[0]) * self.scalar_size),
            self.b.get_cl_buffer(in_ref),
            self.b.get_cl_buffer(mask_ref),
            self.b.get_cl_buffer(w_ref),
            self.b.get_cl_buffer(b_ref),
            self.b.get_cl_buffer(h_out_ref),
            self.b.get_cl_buffer(h_mask_ref),
            np.int32(offset),
            np.int32(num_samples),
            np.int32(self.padded_input_dim),
            np.int32(self.padded_hidden_dim),
        )
        return self.p.forward_pass(queue, g, l, *args, wait_for=wait_for)

    def launch_compute_logits_chunk(
        self, queue: cl.CommandQueue, tile: WorkTile, wait_for, h_ref, h_mask_ref, w_ref, b_ref, logit_out_ref
    ) -> cl.Event:
        g, l = (tile.num_modules_in_tile, BATCH_SIZE, tile.num_classes_in_tile), None
        args = (
            self.b.get_cl_buffer(h_ref),
            self.b.get_cl_buffer(h_mask_ref),
            self.b.get_cl_buffer(w_ref),
            self.b.get_cl_buffer(b_ref),
            self.b.get_cl_buffer(logit_out_ref),
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

    def launch_reduce_for_softmax(
        self, queue: cl.CommandQueue, wait_for, logit_ref, temp_ref, sm_param_ref
    ) -> cl.Event:
        g, l = (NUM_MODULES, BATCH_SIZE), None
        args = (
            self.b.get_cl_buffer(logit_ref),
            self.b.get_cl_buffer(temp_ref),
            self.b.get_cl_buffer(sm_param_ref),
            np.int32(NUM_MODULES),
            np.int32(BATCH_SIZE),
            np.int32(OUTPUT_CLASSES),
            np.int32(self.padded_output_classes),
        )
        return self.p.reduce_logits_for_softmax(queue, g, l, *args, wait_for=wait_for)

    def launch_compute_probs_loss_cce_chunk(
        self,
        queue,
        tile: WorkTile,
        wait_for,
        logit_ref,
        sm_param_ref,
        temp_ref,
        target_ref,
        mask_ref,
        prob_out_ref,
        loss_out_ref,
    ) -> cl.Event:
        g, l = (tile.num_modules_in_tile, BATCH_SIZE, tile.num_classes_in_tile), None
        args = (
            self.b.get_cl_buffer(logit_ref),
            self.b.get_cl_buffer(sm_param_ref),
            self.b.get_cl_buffer(temp_ref),
            self.b.get_cl_buffer(target_ref),
            self.b.get_cl_buffer(mask_ref),
            self.b.get_cl_buffer(prob_out_ref),
            self.b.get_cl_buffer(loss_out_ref),
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
        self,
        q: cl.CommandQueue,
        tile: WorkTile,
        wait_for,
        h_ref,
        prob_ref,
        target_ref,
        mask_ref,
        w_ref,
        gw_out_ref,
        gb_out_ref,
        gh_out_ref,
        logit_ref,
        temp_ref,
        gt_out_ref,
    ) -> Tuple[cl.Event, cl.Event, cl.Event]:
        l, p, c = (
            256,
            np.int32(0 if PROBLEM_TYPE == "CCE" else 1),
            (
                p,
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
            ),
        )
        gwe = self.p.calculate_module_param_grads_chunk(
            q,
            (tile.num_modules_in_tile, HIDDEN_DIM, tile.num_classes_in_tile),
            None,
            cl.LocalMemory(l * self.scalar_size),
            self.b.get_cl_buffer(h_ref),
            self.b.get_cl_buffer(prob_ref),
            self.b.get_cl_buffer(target_ref),
            self.b.get_cl_buffer(mask_ref),
            self.b.get_cl_buffer(gw_out_ref),
            self.b.get_cl_buffer(gb_out_ref),
            *c,
            wait_for=wait_for,
        )
        ghe = self.p.backprop_error_to_hidden_chunk(
            q,
            (tile.num_modules_in_tile, BATCH_SIZE, HIDDEN_DIM),
            None,
            cl.LocalMemory(0),
            self.b.get_cl_buffer(prob_ref),
            self.b.get_cl_buffer(target_ref),
            self.b.get_cl_buffer(mask_ref),
            self.b.get_cl_buffer(w_ref),
            self.b.get_cl_buffer(gh_out_ref),
            *c,
            wait_for=wait_for,
        )
        gte = self.p.calculate_chunk_temp_gradients(
            q,
            (tile.num_modules_in_tile * l,),
            (l,),
            cl.LocalMemory(l * self.scalar_size),
            self.b.get_cl_buffer(logit_ref),
            self.b.get_cl_buffer(prob_ref),
            self.b.get_cl_buffer(target_ref),
            self.b.get_cl_buffer(mask_ref),
            self.b.get_cl_buffer(temp_ref),
            self.b.get_cl_buffer(gt_out_ref),
            *c,
            wait_for=wait_for,
        )
        return gwe, ghe, gte

    def launch_transpose_chunk(
        self, q, wait_for, in_ref, out_ref, in_offset_e, out_offset_e, num_rows, num_cols, in_ld, out_ld
    ) -> cl.Event:
        g, l = (pad_to_multiple(num_cols, C_TILE_SIZE), pad_to_multiple(num_rows, C_TILE_SIZE)), (
            C_TILE_SIZE,
            C_TILE_SIZE,
        )
        args = (
            cl.LocalMemory(l[0] * (l[0] + 1) * self.scalar_size),
            self.b.get_cl_buffer(in_ref),
            self.b.get_cl_buffer(out_ref),
            np.int32(in_offset_e),
            np.int32(out_offset_e),
            np.int32(num_rows),
            np.int32(num_cols),
            np.int32(in_ld),
            np.int32(out_ld),
        )
        return self.p.transpose_chunk(q, g, l, *args, wait_for=wait_for)

    def launch_aggregation(
        self, q, wait_for, in_ref, out_ref, n_partials, elems, is_avg, in_offset_e=0, out_offset_e=0
    ) -> cl.Event:
        lsize, mode, elems = (256, np.int32(1 if is_avg else 0), max(1, int(elems)))
        args = (
            self.b.get_cl_buffer(in_ref),
            self.b.get_cl_buffer(out_ref),
            np.int32(n_partials),
            np.int32(elems),
            mode,
        )
        if n_partials <= 1:
            return self.p.aggregate_identity(q, (elems,), None, cl.LocalMemory(0), *args, wait_for=wait_for)
        if n_partials <= MAX_REGISTER_AGGREGATE_ITEMS:
            return self.p.aggregate_register_reduce(q, (elems,), None, cl.LocalMemory(0), *args, wait_for=wait_for)
        return self.p.aggregate_local_reduce(
            q,
            (pad_to_multiple(elems, lsize),),
            (lsize,),
            cl.LocalMemory(lsize * self.scalar_size),
            *args,
            wait_for=wait_for,
        )

    def launch_reduce_grad_h_over_modules(self, q, wait_for, in_ref, out_ref) -> cl.Event:
        lsize = 256
        spec = self.b.get_spec(in_ref)[0]
        total_elems, padded_m = spec[0], spec[1]
        args = (
            cl.LocalMemory(lsize * self.scalar_size),
            self.b.get_cl_buffer(in_ref),
            self.b.get_cl_buffer(out_ref),
            np.int32(total_elems),
            np.int32(NUM_MODULES),
            np.int32(padded_m),
        )
        return self.p.reduce_grad_h_over_modules(
            q, (pad_to_multiple(total_elems, lsize),), (lsize,), *args, wait_for=wait_for
        )

    def launch_backprop_shared_chunk(
        self, q, wait_for, chunk_id, offset, num_samples, in_ref, h_ref, gh_ref, mask_ref, gsw_out_ref, gsb_out_ref
    ) -> Tuple[cl.Event, cl.Event]:
        l = 256
        sw_evt = self.p.backprop_shared_weights_chunk(
            q,
            (self.padded_input_dim, pad_to_multiple(self.padded_hidden_dim, l)),
            (1, l),
            cl.LocalMemory(l * self.scalar_size),
            self.b.get_cl_buffer(in_ref),
            self.b.get_cl_buffer(h_ref),
            self.b.get_cl_buffer(gh_ref),
            self.b.get_cl_buffer(mask_ref),
            self.b.get_cl_buffer(gsw_out_ref),
            np.int32(offset),
            np.int32(num_samples),
            np.int32(chunk_id),
            np.int32(self.padded_input_dim),
            np.int32(self.padded_hidden_dim),
            wait_for=wait_for,
        )
        sb_evt = self.p.backprop_shared_biases_chunk(
            q,
            (pad_to_multiple(self.padded_hidden_dim, l),),
            (l,),
            cl.LocalMemory(l * self.scalar_size),
            self.b.get_cl_buffer(h_ref),
            self.b.get_cl_buffer(gh_ref),
            self.b.get_cl_buffer(mask_ref),
            self.b.get_cl_buffer(gsb_out_ref),
            np.int32(offset),
            np.int32(num_samples),
            np.int32(chunk_id),
            np.int32(self.padded_hidden_dim),
            wait_for=wait_for,
        )
        return sw_evt, sb_evt

    def launch_adam_update_and_clamp(self, q, wait_for, t: int, p: Parameter) -> cl.Event:
        n_params = int(np.prod(self.b.get_spec(p.name)[0]))
        up_evt = self.p.adam_update(
            q,
            (n_params,),
            None,
            self.b.get_cl_buffer(p.grad),
            SCALAR_NP_TYPE(ADAM_BETA1),
            SCALAR_NP_TYPE(ADAM_BETA2),
            SCALAR_NP_TYPE(LEARNING_RATE),
            SCALAR_NP_TYPE(EPSILON),
            np.uint32(t),
            self.b.get_cl_buffer(p.name),
            self.b.get_cl_buffer(p.m1),
            self.b.get_cl_buffer(p.m2),
            np.int32(0),
            np.int32(n_params),
            wait_for=wait_for,
        )
        if p.role == BufferRole.TEMPERATURES:
            return self.p.clamp_temperatures(
                q,
                (NUM_MODULES,),
                None,
                self.b.get_cl_buffer("temps"),
                SCALAR_NP_TYPE(MIN_TEMP),
                SCALAR_NP_TYPE(MAX_TEMP),
                np.int32(NUM_MODULES),
                wait_for=[up_evt],
            )
        return up_evt


class PingPongManager:
    """Manages a pair of recyclable 'ping-pong' buffers for a reduction tree."""

    def __init__(self):
        self.ping_handle: Optional[BufferHandle] = None
        self.pong_handle: Optional[BufferHandle] = None
        self._is_ping_input = True

    def initialize(self, buffer_mgr: BufferManager, max_bytes: int):
        self.ping_handle = buffer_mgr.acquire_transient_buffer(max_bytes)
        self.pong_handle = buffer_mgr.acquire_transient_buffer(max_bytes)

    def get_io_handles(self) -> Tuple[BufferHandle, BufferHandle]:
        return (self.ping_handle, self.pong_handle) if self._is_ping_input else (self.pong_handle, self.ping_handle)

    def swap(self):
        self._is_ping_input = not self._is_ping_input

    def get_final_buffer_handle(self) -> BufferHandle:
        return self.get_io_handles()[1]

    def release(self, buffer_mgr: BufferManager):
        if self.ping_handle:
            buffer_mgr.release_transient_buffer(self.ping_handle)
        if self.pong_handle:
            buffer_mgr.release_transient_buffer(self.pong_handle)


# --- Layer 2 / Layer 3 ---


class ExecutionStrategy:
    """Makes high-level strategic decisions based on resources and problem size."""

    def __init__(self, device_vram_bytes: int, buffer_mgr: BufferManager):
        self.vram_budget = device_vram_bytes * 0.85
        self.b = buffer_mgr

    def create_plan_for_batch(self, batch_size: int) -> ExecutionPlan:
        hidden_size = self.b.get_cl_buffer("hidden_buf").size
        full_problem_mem = sum(
            self.b.get_cl_buffer(n).size for n in ["full_logits_out", "partial_probs_out", "partial_grad_h_aos_out"]
        )
        nm_chunks, nc_chunks = 1, 1
        while True:
            mem_per_tile = full_problem_mem / (nm_chunks * nc_chunks)
            if mem_per_tile <= self.vram_budget:
                break
            cost_m, cost_c = full_problem_mem / nm_chunks, full_problem_mem / nc_chunks
            if cost_m >= cost_c:
                nm_chunks += 1
            else:
                nc_chunks += 1
            if (nm_chunks * nc_chunks) > (REDUCTION_BATCH_SIZE_K * 1024):
                raise MemoryError("Cannot create a tile small enough for VRAM.")
        grid = ExecutionGrid(nm_chunks, nc_chunks, NUM_MODULES, OUTPUT_CLASSES)
        recompute_hidden, num_b_chunks = (hidden_size + mem_per_tile) > self.vram_budget, (
            batch_size + BACKPROP_STREAM_CHUNK_SIZE - 1
        ) // BACKPROP_STREAM_CHUNK_SIZE
        batch_cfg, reduction_plan = ChunkingConfig(
            min(num_b_chunks, MAX_BATCH_CHUNKS), BACKPROP_STREAM_CHUNK_SIZE, batch_size
        ), ReductionPlan(k=REDUCTION_BATCH_SIZE_K)
        print(
            f"INFO: Determined a {grid.num_module_chunks}x{grid.num_class_chunks} grid ({grid.total_tiles} tiles). Recompute Hidden: {recompute_hidden}."
        )
        return ExecutionPlan(grid, batch_cfg, reduction_plan, recompute_hidden)


class RecursiveAggregator:
    """Stateful engine executing a dynamic, multi-stage reduction tree."""

    def __init__(self, queue: cl.CommandQueue, executor: KernelExecutor):
        self.queue, self.executor, self.ppm = queue, executor, PingPongManager()

    def execute(
        self,
        plan: ReductionPlan,
        in_ref: str,
        out_ref: str,
        n_partials: int,
        elems_per_partial: int,
        is_avg: bool,
        wait_for: List[cl.Event],
    ) -> cl.Event:
        if n_partials <= 1:
            return self.executor.launch_aggregation(self.queue, wait_for, in_ref, out_ref, 1, elems_per_partial, is_avg)
        k = plan.k
        scalar_size = self.executor.scalar_size
        l1_n_partials = (n_partials + k - 1) // k
        max_bytes = int(l1_n_partials * elems_per_partial * scalar_size)
        self.ppm.initialize(self.executor.b, max_bytes)
        _, first_out_handle = self.ppm.get_io_handles()
        stage_evts = []
        for i in range(0, n_partials, k):
            n_now = min(k, n_partials - i)
            evt = self.executor.launch_aggregation(
                self.queue,
                wait_for,
                in_ref,
                first_out_handle,
                n_now,
                elems_per_partial,
                False,
                in_offset_e=i * elems_per_partial,
                out_offset_e=(i // k) * elems_per_partial,
            )
            stage_evts.append(evt)
        self.ppm.swap()
        current_n_partials, dependencies = l1_n_partials, stage_evts
        while current_n_partials > 1:
            n_outs = (current_n_partials + k - 1) // k
            in_handle, out_handle = self.ppm.get_io_handles()
            stage_evts = []
            for i in range(0, current_n_partials, k):
                n_now = min(k, current_n_partials - i)
                evt = self.executor.launch_aggregation(
                    self.queue,
                    dependencies,
                    in_handle,
                    out_handle,
                    n_now,
                    elems_per_partial,
                    False,
                    in_offset_e=i * elems_per_partial,
                    out_offset_e=(i // k) * elems_per_partial,
                )
                stage_evts.append(evt)
            self.ppm.swap()
            current_n_partials, dependencies = n_outs, stage_evts
        final_handle = self.ppm.get_final_buffer_handle()
        final_evt = self.executor.launch_aggregation(
            self.queue, dependencies, final_handle, out_ref, 1, elems_per_partial, is_avg
        )
        self.ppm.release(self.executor.b)
        return final_evt


class BatchProcessor:
    """Stateful, single-use engine to process one batch by building an event-based DAG."""

    def __init__(self, orchestrator: "TrainingOrchestrator", X_batch: np.ndarray, y_batch: np.ndarray):
        self.orchestrator = orchestrator
        self.ctx, self.q, self.ex = orchestrator.ctx, orchestrator.queue, orchestrator.executor
        self.events, self.event_lists = ({}, defaultdict(list))
        self.final_grad_events = {}
        self.X_batch, self.y_batch, self.step = X_batch, y_batch, orchestrator.global_step
        self.host_probs = HostView(
            self.ex.b.get_spec("final_probs_buf"), (NUM_MODULES, X_batch.shape[0], OUTPUT_CLASSES)
        )
        self.host_loss = HostView(self.ex.b.get_spec("final_loss_out"), (NUM_MODULES, X_batch.shape[0]))
        self.aggregator = RecursiveAggregator(self.q, self.ex)

    def _deps(self, *names: str) -> List[cl.Event]:
        return list(
            set(
                [self.events[n] for n in names if n in self.events]
                + [e for n in names for e in self.event_lists.get(n, [])]
            )
        )

    def run(self, plan: ExecutionPlan):
        self._upload_data()
        self._zero_gradients()
        self.events["hidden_ready"] = self.ex.launch_forward_pass(
            self.q,
            0,
            self.X_batch.shape[0],
            self._deps("input_ready", "grads_zeroed"),
            "input_buf",
            "sample_mask",
            "weights",
            "biases",
            "hidden_buf",
            "hidden_mask",
        )
        self._execute_module_path(plan)
        self._aggregate_and_backprop(plan)
        self._finalize_and_update()
        return self.get_sync_points()

    def _upload_data(self):
        wX, wM = self.ex.enqueue_write_buffer(self.q, "input_buf", self.X_batch, None), self.ex.enqueue_write_buffer(
            self.q, "sample_mask", np.ones(self.X_batch.shape[0], SCALAR_NP_TYPE), None
        )
        self.events["input_ready"] = cl.WaitForEvents([wX, wM])
        self.events["targets_ready"] = self.ex.enqueue_write_buffer(self.q, "targets_buf", self.y_batch, None)

    def _zero_gradients(self):
        self.events["grads_zeroed"] = cl.WaitForEvents(
            [self.ex.enqueue_fill_buffer(self.q, n, 0, None) for n in self.ex.b._name_to_handle if "grad_" in n]
        )

    def _execute_module_path(self, plan: ExecutionPlan):
        tiles = list(plan.grid)
        for tile in tiles:
            if tile.num_modules_in_tile > 0 and tile.num_classes_in_tile > 0:
                evt = self.ex.launch_compute_logits_chunk(
                    self.q,
                    tile,
                    self._deps("hidden_ready"),
                    "hidden_buf",
                    "hidden_mask",
                    "module_weights",
                    "module_biases",
                    "full_logits_out",
                )
                self.event_lists["logit_chunks_ready"].append(evt)
        self.events["all_logits_ready"] = cl.WaitForEvents(self.event_lists["logit_chunks_ready"])
        if PROBLEM_TYPE == "CCE":
            self.events["softmax_params_ready"] = self.ex.launch_reduce_for_softmax(
                self.q, self._deps("all_logits_ready"), "full_logits_out", "temps", "softmax_params_out"
            )
        for tile in tiles:
            if tile.num_modules_in_tile > 0 and tile.num_classes_in_tile > 0:
                self._launch_tile_downstream_work(tile)
        self.events["all_prob_loss_chunks_ready"] = cl.WaitForEvents(self.event_lists["prob_loss_chunks_ready"])

    def _launch_tile_downstream_work(self, tile: WorkTile):
        deps = self._deps("softmax_params_ready", "targets_ready", "all_logits_ready")
        evt = self.ex.launch_compute_probs_loss_cce_chunk(
            self.q,
            tile,
            deps,
            "full_logits_out",
            "softmax_params_out",
            "temps",
            "targets_buf",
            "sample_mask",
            "partial_probs_out",
            "final_loss_out",
        )
        self.event_lists["prob_loss_chunks_ready"].append(evt)
        grad_deps = self._deps("hidden_ready", "all_logits_ready", "targets_ready") + [evt]
        w, h, t = self.ex.launch_parallel_module_grads(
            self.q,
            tile,
            grad_deps,
            "hidden_buf",
            "partial_probs_out",
            "targets_buf",
            "sample_mask",
            "module_weights",
            "partial_grad_module_w_out",
            "partial_grad_module_b_out",
            "partial_grad_h_aos_out",
            "full_logits_out",
            "temps",
            "partial_grad_temps_out",
        )
        self.event_lists["partial_grad_w_ready"].append(w)
        self.event_lists["partial_grad_t_ready"].append(t)
        in_sp, out_sp = self.ex.b.get_spec("partial_grad_h_aos_out")[0], self.ex.b.get_spec("partial_grad_h_soa_out")[0]
        in_off, out_off = tile.flat_tile_id * (in_sp[1] * in_sp[2] * in_sp[3]), tile.flat_tile_id * (
            out_sp[1] * out_sp[2]
        )
        rows, cols = tile.num_modules_in_tile, BATCH_SIZE * HIDDEN_DIM
        tp_evt = self.ex.launch_transpose_chunk(
            self.q,
            [h],
            "partial_grad_h_aos_out",
            "partial_grad_h_soa_out",
            in_off,
            out_off,
            rows,
            cols,
            cols,
            out_sp[2],
        )
        self.event_lists["partial_grad_h_soa_ready"].append(tp_evt)

    def _aggregate_and_backprop(self, plan: ExecutionPlan):
        b, rp, g = self.ex.b, plan.reduction_plan, plan.grid
        get_spec, get_elem = b.get_spec, lambda r: int(np.prod(b.get_spec(r)[0]))
        self.events["final_probs_ready"] = self.aggregator.execute(
            rp,
            "partial_probs_out",
            "final_probs_buf",
            g.total_tiles,
            get_elem("final_probs_buf"),
            False,
            self._deps("all_prob_loss_chunks_ready"),
        )
        self.final_grad_events["grad_module_weights"] = self.aggregator.execute(
            rp,
            "partial_grad_module_w_out",
            "grad_module_weights",
            g.total_tiles,
            get_elem("grad_module_weights"),
            False,
            self.event_lists["partial_grad_w_ready"],
        )
        self.final_grad_events["grad_module_biases"] = self.aggregator.execute(
            rp,
            "partial_grad_module_b_out",
            "grad_module_biases",
            g.total_tiles,
            get_elem("grad_module_biases"),
            False,
            self.event_lists["partial_grad_w_ready"],
        )
        self.final_grad_events["grad_temps"] = self.aggregator.execute(
            rp,
            "partial_grad_temps_out",
            "grad_temps",
            g.total_tiles,
            get_elem("grad_temps"),
            False,
            self.event_lists["partial_grad_t_ready"],
        )
        agg_h_evt = self.aggregator.execute(
            rp,
            "partial_grad_h_soa_out",
            "aggregated_grad_h_soa",
            g.total_tiles,
            get_elem("aggregated_grad_h_soa"),
            False,
            self.event_lists["partial_grad_h_soa_ready"],
        )
        self.events["final_grad_h_ready"] = self.ex.launch_reduce_grad_h_over_modules(
            self.q, [agg_h_evt], "aggregated_grad_h_soa", "final_grad_h_buf"
        )
        deps = self._deps("final_grad_h_ready")
        (
            deps.append(
                self.ex.launch_forward_pass(
                    self.q,
                    0,
                    self.X_batch.shape[0],
                    self._deps("input_ready"),
                    "input_buf",
                    "sample_mask",
                    "weights",
                    "biases",
                    "hidden_buf",
                    "hidden_mask",
                )
            )
            if plan.recompute_hidden
            else deps.extend(self._deps("hidden_ready"))
        )
        cfg = plan.shared_layer_batch_chunking
        for i in range(cfg.num_chunks):
            off, size = i * cfg.chunk_size, min(cfg.chunk_size, cfg.total_dim - off)
            if size > 0:
                sw, sb = self.ex.launch_backprop_shared_chunk(
                    self.q,
                    deps,
                    i,
                    off,
                    size,
                    "input_buf",
                    "hidden_buf",
                    "final_grad_h_buf",
                    "sample_mask",
                    "partial_grad_sw_out",
                    "partial_grad_sb_out",
                )
                self.event_lists["partial_grad_sw_ready"].append(sw)
                self.event_lists["partial_grad_sb_ready"].append(sb)
        self.final_grad_events["grad_weights"] = self.aggregator.execute(
            rp,
            "partial_grad_sw_out",
            "grad_weights",
            cfg.num_chunks,
            get_elem("grad_weights"),
            True,
            self.event_lists["partial_grad_sw_ready"],
        )
        self.final_grad_events["grad_biases"] = self.aggregator.execute(
            rp,
            "partial_grad_sb_out",
            "grad_biases",
            cfg.num_chunks,
            get_elem("grad_biases"),
            True,
            self.event_lists["partial_grad_sb_ready"],
        )

    def _finalize_and_update(self):
        d2h_deps = self._deps("final_probs_ready")
        d2h_buf = self.ex.b.get_cl_buffer("final_probs_buf")
        self.events["d2h_probs_ready"] = self.host_probs.enqueue_read(self.q, d2h_buf, d2h_deps)
        for p in self.orchestrator.params:
            if p.grad in self.final_grad_events:
                self.event_lists["updates_done"].append(
                    self.ex.launch_adam_update_and_clamp(self.q, [self.final_grad_events[p.grad]], self.step, p)
                )
        self.events["all_updates_done"] = cl.WaitForEvents(self.event_lists["updates_done"])

    def get_sync_points(self) -> Tuple[cl.UserEvent, cl.UserEvent]:
        inf, final = cl.UserEvent(self.ctx), cl.UserEvent(self.ctx)
        cb = lambda s, e: e.set_status(cl.command_execution_status.COMPLETE)
        self.events.get("d2h_probs_ready", inf).set_callback(cl.command_execution_status.COMPLETE, cb, inf)
        self.events.get("all_updates_done", final).set_callback(cl.command_execution_status.COMPLETE, cb, final)
        return inf, final


# --- Layer 4: Orchestration Layer ---


class TrainingOrchestrator:
    """Top-level class that owns all components and runs the main training loop."""

    def __init__(self):
        self.ctx = cl.create_some_context(interactive=False)
        self.device = self.ctx.devices[0]
        self.queue = cl.CommandQueue(self.ctx, properties=cl.command_queue_properties.OUT_OF_ORDER_EXEC_MODE_ENABLE)
        print(
            f"Using device: {self.device.name} ({self.device.vendor}) | VRAM: {self.device.global_mem_size/1e9:.2f} GB"
        )
        self.simd_width = select_simd_width(self.device)
        self.program = self._compile_kernels()
        self.params: List[Parameter] = [
            Parameter("weights", BufferRole.SHARED_WEIGHTS, LayoutType.SoA),
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
        opts = [
            f"-cl-std=CL1.2",
            f"-D SCALAR_TYPE={CL_SCALAR_TYPE}",
            f"-D SIMD_WIDTH={self.simd_width}",
            f"-D C_TILE_SIZE={C_TILE_SIZE}",
            "-D cl_khr_fp16" if SCALAR_TYPE == "half" else "",
        ]
        return cl.Program(self.ctx, load_and_concatenate_kernels()).build(options=opts)

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
                (REDUCTION_BATCH_SIZE_K, *self.param_shapes["module_weights"]),
            ),
            "partial_grad_module_b_out": (
                BufferRole.PARTIAL_GRADIENT,
                (REDUCTION_BATCH_SIZE_K, *self.param_shapes["module_biases"]),
            ),
            "partial_grad_temps_out": (
                BufferRole.PARTIAL_GRADIENT,
                (REDUCTION_BATCH_SIZE_K, *self.param_shapes["temps"]),
            ),
            "partial_grad_sw_out": (BufferRole.PARTIAL_GRADIENT, (MAX_BATCH_CHUNKS, *self.param_shapes["weights"])),
            "partial_grad_sb_out": (BufferRole.PARTIAL_GRADIENT, (MAX_BATCH_CHUNKS, *self.param_shapes["biases"])),
            "partial_grad_h_aos_out": (
                BufferRole.PARTIAL_GRADIENT,
                (REDUCTION_BATCH_SIZE_K, NUM_MODULES, BATCH_SIZE, HIDDEN_DIM),
            ),
            "partial_grad_h_soa_out": (
                BufferRole.PARTIAL_GRADIENT,
                (REDUCTION_BATCH_SIZE_K, BATCH_SIZE * HIDDEN_DIM, NUM_MODULES),
            ),
            "aggregated_grad_h_soa": (BufferRole.INTERMEDIATE, (BATCH_SIZE * HIDDEN_DIM, NUM_MODULES)),
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
            valid_probs = processor.host_probs.get()
            predicted_classes = np.argmax(valid_probs[0, :, :], axis=1)
            accuracy = np.mean(predicted_classes == y_batch_int) if y_batch_int.size > 0 else 0.0
            print(f"  - Inference Latency Path Complete. Accuracy: {accuracy:.2%}")
            final_event.wait()
            print(f"  - Full Training Step Complete.")
            self.global_step += 1
        print("\n--- Training finished ---")


if __name__ == "__main__":
    try:
        for fname in [
            "kernels.cl.h",
            "chunk_kernels.cl.c",
            "aggregation_kernels.cl.c",
            "backprop_kernels.cl.c",
            "parameter_optim.cl.c",
        ]:
            if not os.path.exists(fname):
                open(fname, "w").write(f"// Placeholder for {fname}\n")
        TrainingOrchestrator().train()
    except (cl.LogicError, cl.MemoryError) as e:
        print(f"A CL error occurred: {e}")
    except cl.BuildError as e:
        print(
            "--- KERNEL BUILD FAILED ---\n"
            + "\n".join([f"Device: {d.name}\n--- Log ---\n{l}" for d, l in e.device_logs])
        )
    except Exception as e:
        import traceback

        print(f"An unexpected error occurred: {e}")
        traceback.print_exc()
