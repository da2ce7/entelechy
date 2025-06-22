import os
import pyopencl as cl
import numpy as np
import math
from collections import defaultdict
from dataclasses import dataclass
from typing import Tuple, List, Dict, Optional, Callable
from abc import ABC, abstractmethod
import enum


# --- Configuration ---
SCALAR_TYPE = "half"
SCALAR_NP_TYPE = np.float16 if SCALAR_TYPE == "half" else np.float32
CL_SCALAR_TYPE = "half" if SCALAR_TYPE == "half" else "float"

# --- Core Network Architecture ---
INPUT_DIM: int = 4
HIDDEN_DIM: int = 64
OUTPUT_CLASSES: int = 3
NUM_EXITS: int = 128
BATCH_SIZE: int = 150
EPOCHS: int = 50

# --- Hyperparameters & Runtime Tuning Constants ---
PROBLEM_TYPE = "CCE"  # "CCE" or "BCE"
LEARNING_RATE: float = 0.001
ADAM_BETA1: float = 0.9
ADAM_BETA2: float = 0.999
EPSILON: float = 1.0e-8
MIN_TEMP: float = 0.1
MAX_TEMP: float = 10.0

# --- Compile-time Kernel Constants & Architectural Tuning ---
MAX_REGISTER_AGGREGATE_ITEMS: int = 32
C_TILE_SIZE: int = 16
BACKPROP_STREAM_CHUNK_SIZE: int = 32


# --- Core Abstractions---


class BufferRole(enum.Enum):
    """Defines the semantic purpose of a buffer, driving its padding strategy."""

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
    INTERMEDIATE = 11
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
        return self.host_data[tuple(slice(0, dim) for dim in self.real_shape)]


# --- Utility Functions ---


def load_and_concatenate_kernels() -> str:
    """Loads kernel sources from disk."""
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
            raise FileNotFoundError(f"Missing kernel file: {filename}.")
        with open(filename, "r", encoding="utf-8") as f:
            source_parts.append(f.read())
    print(f"Loaded {len(source_parts)} kernel source files.")
    return "\n".join(source_parts)


def select_simd_width(device: cl.Device) -> int:
    """Selects a reasonable SIMD width based on vendor."""
    vendor = device.vendor.upper()
    if "NVIDIA" in vendor:
        return 32
    if "AMD" in vendor or "ADVANCED MICRO DEVICES" in vendor:
        return 64
    if "INTEL" in vendor:
        return 16
    return 8


def pad_to_multiple(dim: int, multiple: int) -> int:
    """Calculates the smallest multiple of `multiple` greater than or equal to `dim`."""
    return (dim + multiple - 1) // multiple * multiple


# --- Padding Logic (Driven by BufferRole) ---


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
    """Manages creation, padding, and access to all OpenCL memory buffers."""

    def __init__(self, context: cl.Context, simd_width: int):
        self.context = context
        self.simd_width = simd_width
        self.buffers: Dict[str, cl.Buffer] = {}
        self.specs: Dict[str, Tuple[Tuple[int, ...], np.dtype]] = {}

    def create_buffer(
        self, name: str, role: BufferRole, real_shape: Tuple, dtype: np.dtype, init_data: Optional[np.ndarray] = None
    ) -> None:
        """Creates a buffer with padding determined by its role."""
        padding_func = PADDING_RULES.get(role, _pad_none)
        padded_shape = padding_func(real_shape, self.simd_width)
        byte_size = int(np.prod(padded_shape) * dtype().itemsize) if padded_shape else 4

        mem_flags = cl.mem_flags.READ_WRITE
        hostbuf = None
        if init_data is not None:
            mem_flags |= cl.mem_flags.COPY_HOST_PTR
            padded_data = np.zeros(padded_shape, dtype=dtype)
            # This allows initializing padded buffers with unpadded data
            padded_data[tuple(slice(0, d) for d in real_shape)] = init_data
            hostbuf = padded_data

        buf = cl.Buffer(self.context, mem_flags, size=max(byte_size, 4), hostbuf=hostbuf)
        self.buffers[name] = buf
        self.specs[name] = (padded_shape, dtype)

    def get(self, name: str) -> cl.Buffer:
        return self.buffers[name]

    def get_spec(self, name: str) -> Tuple[Tuple[int, ...], np.dtype]:
        return self.specs[name]


class KernelExecutor:
    """A stateless toolbox for launching every specific kernel. Returns events."""

    def __init__(self, program: cl.Program, buffer_mgr: BufferManager):
        self.p = program
        self.b = buffer_mgr
        self.simd_width = self.b.simd_width
        self.padded_hidden_dim = self.b.get_spec("hidden_buf")[0][1]
        self.padded_input_dim = self.b.get_spec("input_buf")[0][1]
        self.scalar_size = SCALAR_NP_TYPE().itemsize

    def enqueue_write_buffer(self, queue, role, name, data, wait_for) -> cl.Event:
        buf = self.b.get(name)
        shape, dtype = self.b.get_spec(name)
        padded_data = np.zeros(shape, dtype=dtype)
        padded_data[tuple(slice(0, d) for d in data.shape)] = data
        return cl.enqueue_copy(queue, buf, padded_data, wait_for=wait_for)

    def enqueue_fill_buffer(self, queue, name, value, wait_for) -> cl.Event:
        return cl.enqueue_fill_buffer(
            queue, self.b.get(name), SCALAR_NP_TYPE(value), 0, self.b.get(name).size, wait_for=wait_for
        )

    def launch_forward_pass(self, queue, offset, num_samples, wait_for) -> cl.Event:
        padded_samples = pad_to_multiple(num_samples, self.simd_width)
        g, l = (padded_samples, self.padded_hidden_dim // self.simd_width), (self.simd_width, 1)
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
            self.padded_input_dim,
            self.padded_hidden_dim,
        )
        return self.p.forward_pass(queue, g, l, *args, wait_for=wait_for)

    def launch_compute_logits_chunk(
        self, queue, exit_chunk_id, exit_offset, num_exits_in_chunk, class_offset, num_classes_in_chunk, wait_for
    ) -> cl.Event:
        g, l = (num_exits_in_chunk, BATCH_SIZE, num_classes_in_chunk), None
        args = (
            self.b.get("hidden_buf"),
            self.b.get("hidden_mask"),
            self.b.get("exit_weights"),
            self.b.get("exit_biases"),
            self.b.get("full_logits_out"),
            np.int32(exit_chunk_id),
            np.int32(exit_offset),
            np.int32(num_exits_in_chunk),
            np.int32(class_offset),
            np.int32(num_classes_in_chunk),
            np.int32(BATCH_SIZE),
            np.int32(HIDDEN_DIM),
            self.padded_hidden_dim,
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

    def launch_parallel_exit_grads(
        self, queue, exit_chunk_id, exit_offset, num_exits, class_chunk_id, class_offset, num_classes, wait_for
    ) -> Tuple[cl.Event, cl.Event, cl.Event]:
        lsize = 256
        grad_w_evt = self.p.calculate_exit_param_grads_chunk(
            queue,
            (num_exits, HIDDEN_DIM, num_classes),
            (1, 1, 1),
            cl.LocalMemory(lsize * self.scalar_size),
            self.b.get("hidden_buf"),
            self.b.get("partial_probs_out"),
            self.b.get("targets_buf"),
            self.b.get("partial_grad_exit_w_out"),
            self.b.get("partial_grad_exit_b_out"),
            np.int32(PROBLEM_TYPE == "CCE"),
            np.int32(exit_chunk_id),
            np.int32(exit_offset),
            np.int32(num_exits),
            np.int32(class_chunk_id),
            np.int32(class_offset),
            np.int32(num_classes),
            np.int32(BATCH_SIZE),
            np.int32(HIDDEN_DIM),
            self.padded_hidden_dim,
            np.int32(OUTPUT_CLASSES),
            np.int32(NUM_EXITS),
            wait_for=wait_for,
        )
        grad_h_evt = self.p.backprop_error_to_hidden_chunk(
            queue,
            (num_exits, BATCH_SIZE, HIDDEN_DIM),
            None,
            cl.LocalMemory(0),
            self.b.get("partial_probs_out"),
            self.b.get("targets_buf"),
            self.b.get("exit_weights"),
            self.b.get("partial_grad_h_aos_out"),
            np.int32(PROBLEM_TYPE == "CCE"),
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
            np.int32(PROBLEM_TYPE == "CCE"),
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

    def launch_transpose_grad_h(self, queue, num_rows, num_cols, wait_for) -> cl.Event:
        g = (pad_to_multiple(num_cols, C_TILE_SIZE), pad_to_multiple(num_rows, C_TILE_SIZE))
        l = (C_TILE_SIZE, C_TILE_SIZE)
        args = (
            cl.LocalMemory(C_TILE_SIZE * (C_TILE_SIZE + 1) * self.scalar_size),
            self.b.get("partial_grad_h_aos_out"),
            self.b.get("grad_h_soa_buf"),
            np.int32(num_rows),
            np.int32(num_cols),
        )
        return self.p.transpose_grad_h(queue, g, l, *args, wait_for=wait_for)

    def launch_aggregation(self, queue, in_buf, out_buf, num_partials, elements_per_partial, wait_for) -> cl.Event:
        lsize = 256
        if num_partials <= 1:
            g, l, kernel, args = (
                (elements_per_partial,),
                None,
                self.p.aggregate_identity,
                (
                    cl.LocalMemory(0),
                    self.b.get(in_buf),
                    self.b.get(out_buf),
                    np.int32(1),
                    np.int32(elements_per_partial),
                    np.int32(0),
                ),
            )
        elif num_partials <= MAX_REGISTER_AGGREGATE_ITEMS:
            g, l, kernel, args = (
                (elements_per_partial,),
                None,
                self.p.aggregate_register_reduce,
                (
                    cl.LocalMemory(0),
                    self.b.get(in_buf),
                    self.b.get(out_buf),
                    np.int32(num_partials),
                    np.int32(elements_per_partial),
                    np.int32(0),
                ),
            )
        else:
            g, l, kernel, args = (
                (elements_per_partial * lsize,),
                (lsize,),
                self.p.aggregate_local_reduce,
                (
                    cl.LocalMemory(lsize * self.scalar_size),
                    self.b.get(in_buf),
                    self.b.get(out_buf),
                    np.int32(num_partials),
                    np.int32(elements_per_partial),
                    np.int32(0),
                ),
            )
        return kernel(queue, g, l, *args, wait_for=wait_for)

    def launch_backprop_shared_chunk(self, queue, chunk_id, offset, num_samples, wait_for) -> Tuple[cl.Event, cl.Event]:
        lsize = 256
        sw_evt = self.p.backprop_shared_weights_chunk(
            queue,
            (self.padded_input_dim * lsize, self.padded_hidden_dim),
            (lsize, 1),
            cl.LocalMemory(lsize * self.scalar_size),
            self.b.get("input_buf"),
            self.b.get("hidden_buf"),
            self.b.get("final_grad_h_buf"),
            self.b.get("input_mask"),
            self.b.get("partial_grad_sw_out"),
            np.int32(offset),
            np.int32(num_samples),
            np.int32(chunk_id),
            self.padded_input_dim,
            self.padded_hidden_dim,
            wait_for=wait_for,
        )
        sb_evt = self.p.backprop_shared_biases_chunk(
            queue,
            (self.padded_hidden_dim * lsize,),
            (lsize,),
            cl.LocalMemory(lsize * self.scalar_size),
            self.b.get("hidden_buf"),
            self.b.get("final_grad_h_buf"),
            self.b.get("input_mask"),
            self.b.get("partial_grad_sb_out"),
            np.int32(offset),
            np.int32(num_samples),
            np.int32(chunk_id),
            self.padded_hidden_dim,
            wait_for=wait_for,
        )
        return sw_evt, sb_evt

    def launch_adam_update_and_clamp(self, queue, p: Parameter, grad_ready_event, b1_t, b2_t) -> cl.Event:
        """Accepts a Parameter object to find related buffers and apply Adam update."""
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
    # Note: recompute_hidden is part of the arch design but not implemented here for brevity
    recompute_hidden: bool = False


class ExecutionStrategy:
    """Makes high-level strategic decisions based on resources and problem size."""

    def __init__(self, device_vram_bytes: int, buffer_mgr: BufferManager):
        self.vram_budget = device_vram_bytes * 0.90
        self.b = buffer_mgr

    def create_plan_for_batch(self, batch_size: int) -> ExecutionPlan:
        # Simple strategy implementation: chunk if major buffers exceed VRAM budget
        # A more advanced strategy would analyze each buffer and chunk dimension separately.
        logits_size = self.b.get("full_logits_out").size
        probs_size = self.b.get("partial_probs_out").size
        grad_h_size = self.b.get("partial_grad_h_aos_out").size
        required_mem = logits_size + probs_size + grad_h_size

        if required_mem > self.vram_budget:
            num_chunks = math.ceil(required_mem / self.vram_budget)
            print(f"VRAM pressure detected. Streaming with {num_chunks} exit chunks.")
            return ExecutionPlan(num_exit_chunks=num_chunks)
        return ExecutionPlan(num_exit_chunks=1)


# --- Layer 3: Execution Layer ---


class BatchProcessor:
    """Stateful engine to process one batch by building the event-based DAG."""

    def __init__(self, orchestrator: "TrainingOrchestrator", X_batch: np.ndarray, y_batch: np.ndarray):
        """Receives the orchestrator to access its context, executor, params, etc."""
        self.orchestrator = orchestrator
        self.ctx, self.queue, self.executor = orchestrator.ctx, orchestrator.queue, orchestrator.executor
        self.events: Dict[str, cl.Event] = {}
        self.event_lists: Dict[str, List[cl.Event]] = defaultdict(list)
        self.final_grad_events: Dict[str, cl.Event] = {}
        self.X_batch, self.y_batch = X_batch, y_batch
        self.global_step = orchestrator.global_step

        # Instantiate HostViews for all buffers that require D2H copies for inspection/results.
        self.host_probs_view = HostView(
            self.executor.b.get_spec("final_probs_buf"), real_shape=(NUM_EXITS, X_batch.shape[0], OUTPUT_CLASSES)
        )
        self.host_loss_view = HostView(
            self.executor.b.get_spec("final_loss_out"), real_shape=(NUM_EXITS, X_batch.shape[0])
        )

    def _get_deps(self, *resource_names: str) -> List[cl.Event]:
        deps = [self.events[name] for name in resource_names if name in self.events]
        for name in resource_names:
            deps.extend(self.event_lists.get(name, []))
        return list(set(deps))

    def run(self, plan: ExecutionPlan):
        self._upload_data()
        self._zero_gradients()
        self.events["hidden_ready"] = self.executor.launch_forward_pass(
            self.queue, 0, self.X_batch.shape[0], wait_for=self._get_deps("input_ready")
        )
        self._execute_exit_path(plan)
        self._aggregate_and_backprop_shared(plan)
        self._finalize_and_update()
        return self.get_sync_points()

    def _upload_data(self):
        write_X = self.executor.enqueue_write_buffer(
            self.queue, BufferRole.INPUT, "input_buf", self.X_batch, wait_for=None
        )
        write_mask = self.executor.enqueue_write_buffer(
            self.queue,
            BufferRole.MASK,
            "input_mask",
            np.ones(self.X_batch.shape[0], dtype=SCALAR_NP_TYPE),
            wait_for=None,
        )
        self.events["input_ready"] = cl.WaitForEvents([write_X, write_mask])
        self.events["targets_ready"] = self.executor.enqueue_write_buffer(
            self.queue, BufferRole.TARGETS, "targets_buf", self.y_batch, wait_for=None
        )

    def _zero_gradients(self):
        grad_names = [p.grad for p in self.orchestrator.params] + ["partial_grad_sw_out", "partial_grad_sb_out"]
        self.events["grads_zeroed"] = cl.WaitForEvents(
            [self.executor.enqueue_fill_buffer(self.queue, g, 0, wait_for=None) for g in grad_names]
        )

    def _execute_exit_path(self, plan: ExecutionPlan):
        exit_chunk_size = (NUM_EXITS + plan.num_exit_chunks - 1) // plan.num_exit_chunks
        hidden_deps = self._get_deps("hidden_ready")
        for i in range(plan.num_exit_chunks):
            offset, size = i * exit_chunk_size, min(exit_chunk_size, NUM_EXITS - i * exit_chunk_size)
            evt = self.executor.launch_compute_logits_chunk(
                self.queue, i, offset, size, 0, OUTPUT_CLASSES, wait_for=hidden_deps
            )
            self.event_lists["logit_chunks_ready"].append(evt)
        self.events["all_logits_ready"] = cl.WaitForEvents(self.event_lists["logit_chunks_ready"])

        if PROBLEM_TYPE == "CCE":
            self.events["softmax_params_ready"] = self.executor.launch_reduce_for_softmax(
                self.queue, wait_for=[self.events["all_logits_ready"]]
            )

        for i in range(plan.num_exit_chunks):
            offset, size = i * exit_chunk_size, min(exit_chunk_size, NUM_EXITS - i * exit_chunk_size)
            if PROBLEM_TYPE == "CCE":
                prob_loss_deps = self._get_deps("softmax_params_ready", "targets_ready")
                evt = self.executor.launch_compute_probs_loss_cce_chunk(
                    self.queue, i, offset, size, 0, OUTPUT_CLASSES, wait_for=prob_loss_deps
                )
                self.event_lists["prob_loss_chunks_ready"].append(evt)

            grad_deps = self._get_deps("hidden_ready", f"prob_loss_chunks_ready") + [self.events["all_logits_ready"]]
            w_evt, h_evt, t_evt = self.executor.launch_parallel_exit_grads(
                self.queue, i, offset, size, i, 0, OUTPUT_CLASSES, wait_for=grad_deps
            )
            self.event_lists["partial_grad_w_ready"].append(w_evt)
            self.event_lists["partial_grad_h_ready"].append(h_evt)
            self.event_lists["partial_grad_t_ready"].append(t_evt)
        self.events["all_prob_loss_chunks_ready"] = cl.WaitForEvents(self.event_lists["prob_loss_chunks_ready"])

    def _aggregate_and_backprop_shared(self, plan: ExecutionPlan):
        transpose_evt = self.executor.launch_transpose_grad_h(
            self.queue, NUM_EXITS, BATCH_SIZE * HIDDEN_DIM, wait_for=self.event_lists["partial_grad_h_ready"]
        )
        self.events["final_grad_h_ready"] = self.executor.launch_aggregation(
            self.queue,
            "grad_h_soa_buf",
            "final_grad_h_buf",
            NUM_EXITS,
            BATCH_SIZE * HIDDEN_DIM,
            wait_for=[transpose_evt],
        )

        # Aggregate all partial results into final buffers and store the completion event
        self.final_grad_events["grad_exit_weights"] = self.executor.launch_aggregation(
            self.queue,
            "partial_grad_exit_w_out",
            "grad_exit_weights",
            plan.num_exit_chunks,
            self.executor.b.get("grad_exit_weights").size // plan.num_exit_chunks,
            wait_for=self.event_lists["partial_grad_w_ready"],
        )
        self.final_grad_events["grad_exit_biases"] = self.executor.launch_aggregation(
            self.queue,
            "partial_grad_exit_b_out",
            "grad_exit_biases",
            plan.num_exit_chunks,
            self.executor.b.get("grad_exit_biases").size // plan.num_exit_chunks,
            wait_for=self.event_lists["partial_grad_w_ready"],
        )
        self.final_grad_events["grad_temps"] = self.executor.launch_aggregation(
            self.queue,
            "partial_grad_temps_out",
            "grad_temps",
            plan.num_exit_chunks,
            NUM_EXITS,
            wait_for=self.event_lists["partial_grad_t_ready"],
        )
        self.events["final_probs_ready"] = self.executor.launch_aggregation(
            self.queue,
            "partial_probs_out",
            "final_probs_buf",
            plan.num_exit_chunks,
            self.executor.b.get("final_probs_buf").size,
            wait_for=self.event_lists["prob_loss_chunks_ready"],
        )

        backprop_deps = self._get_deps("hidden_ready", "final_grad_h_ready")
        for i in range(plan.num_batch_chunks):  # Note: num_batch_chunks is always 1 in this simplified version
            offset, size = i * BACKPROP_STREAM_CHUNK_SIZE, min(
                BACKPROP_STREAM_CHUNK_SIZE, BATCH_SIZE - i * BACKPROP_STREAM_CHUNK_SIZE
            )
            if size <= 0:
                continue
            sw_evt, sb_evt = self.executor.launch_backprop_shared_chunk(
                self.queue, i, offset, size, wait_for=backprop_deps
            )
            self.event_lists["partial_grad_sw_ready"].append(sw_evt)
            self.event_lists["partial_grad_sb_ready"].append(sb_evt)

        self.final_grad_events["grad_weights"] = self.executor.launch_aggregation(
            self.queue,
            "partial_grad_sw_out",
            "grad_weights",
            plan.num_batch_chunks,
            self.executor.b.get("grad_weights").size,
            wait_for=self.event_lists["partial_grad_sw_ready"],
        )
        self.final_grad_events["grad_biases"] = self.executor.launch_aggregation(
            self.queue,
            "partial_grad_sb_out",
            "grad_biases",
            plan.num_batch_chunks,
            self.executor.b.get("grad_biases").size,
            wait_for=self.event_lists["partial_grad_sb_ready"],
        )

    def _finalize_and_update(self):
        # Enqueue D2H copy using the HostView utility
        self.events["d2h_probs_ready"] = self.host_probs_view.enqueue_read(
            self.queue, self.executor.b.get("final_probs_buf"), wait_for=self._get_deps("final_probs_ready")
        )

        # Loop over semantic Parameter objects to launch updates
        b1_t, b2_t = (ADAM_BETA1**self.global_step), (ADAM_BETA2**self.global_step)
        for p in self.orchestrator.params:
            grad_ready_event = self.final_grad_events[p.grad]
            self.event_lists["updates_done"].append(
                self.executor.launch_adam_update_and_clamp(self.queue, p, grad_ready_event, b1_t, b2_t)
            )
        self.events["all_updates_done"] = cl.WaitForEvents(self.event_lists["updates_done"])

    def get_sync_points(self) -> Tuple[cl.UserEvent, cl.UserEvent]:
        inference_event, final_batch_event = cl.UserEvent(self.ctx), cl.UserEvent(self.ctx)
        cl.enqueue_marker(self.queue, wait_for=[self.events["d2h_probs_ready"]]).set_callback(
            cl.command_execution_status.COMPLETE,
            lambda s, u: inference_event.set_status(cl.command_execution_status.COMPLETE),
        )
        cl.enqueue_marker(self.queue, wait_for=[self.events["all_updates_done"]]).set_callback(
            cl.command_execution_status.COMPLETE,
            lambda s, u: final_batch_event.set_status(cl.command_execution_status.COMPLETE),
        )
        return inference_event, final_batch_event


# --- Layer 4: Orchestration Layer ---


class TrainingOrchestrator:
    """Top-level class that owns all components and runs the training loop."""

    def __init__(self):
        self.ctx = cl.create_some_context(interactive=False)
        self.device = self.ctx.devices[0]
        self.queue = cl.CommandQueue(self.ctx, properties=cl.command_queue_properties.OUT_OF_ORDER_EXEC_MODE_ENABLE)
        print(f"Using device: {self.device.name} with out-of-order queue.")

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
        # Create parameter buffers and their associated optimizer state buffers
        for p in self.params:
            shape = self.param_shapes[p.name]
            init_fn = (
                np.zeros
                if "biases" in p.name
                else (
                    (lambda s: np.full(s, 1.0, dtype=SCALAR_NP_TYPE))
                    if "temps" in p.name
                    else (lambda s: np.random.randn(*s).astype(SCALAR_NP_TYPE) * 0.1)
                )
            )
            self.buffer_mgr.create_buffer(p.name, p.role, shape, SCALAR_NP_TYPE, init_fn(shape))

            padded_shape, dtype = self.buffer_mgr.get_spec(p.name)
            self.buffer_mgr.create_buffer(p.grad, BufferRole.FINAL_GRADIENT, padded_shape, dtype)
            self.buffer_mgr.create_buffer(p.m1, BufferRole.ADAM_MOMENTUM, padded_shape, dtype)
            self.buffer_mgr.create_buffer(p.m2, BufferRole.ADAM_MOMENTUM, padded_shape, dtype)

        # Create intermediate and I/O buffers using BufferRole to drive padding
        dtype_map = {"targets_buf": np.int32} if PROBLEM_TYPE == "CCE" else {}
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
            "partial_grad_exit_w_out": (BufferRole.PARTIAL_GRADIENT, self.param_shapes["exit_weights"]),
            "partial_grad_exit_b_out": (BufferRole.PARTIAL_GRADIENT, self.param_shapes["exit_biases"]),
            "partial_grad_temps_out": (BufferRole.PARTIAL_GRADIENT, (NUM_EXITS * 20,)),  # Sized for many chunks
            "partial_grad_h_aos_out": (BufferRole.PARTIAL_GRADIENT, (NUM_EXITS, BATCH_SIZE * HIDDEN_DIM)),
            "grad_h_soa_buf": (BufferRole.PARTIAL_GRADIENT, (BATCH_SIZE * HIDDEN_DIM, NUM_EXITS)),
            "final_grad_h_buf": (BufferRole.FINAL_GRADIENT, (BATCH_SIZE, HIDDEN_DIM)),
            "partial_grad_sw_out": (
                BufferRole.PARTIAL_GRADIENT,
                (BACKPROP_STREAM_CHUNK_SIZE * 2, INPUT_DIM, HIDDEN_DIM),
            ),
            "partial_grad_sb_out": (BufferRole.PARTIAL_GRADIENT, (BACKPROP_STREAM_CHUNK_SIZE * 2, HIDDEN_DIM)),
        }
        for name, (role, shape) in int_buffers.items():
            self.buffer_mgr.create_buffer(name, role, shape, dtype_map.get(name, SCALAR_NP_TYPE))

    def train(self):
        from sklearn.datasets import load_iris
        from sklearn.preprocessing import StandardScaler

        X, y = load_iris(return_X_y=True)
        X = StandardScaler().fit_transform(X).astype(SCALAR_NP_TYPE)

        print(f"Starting training for {EPOCHS} epochs...")
        for epoch in range(EPOCHS):
            X_batch, y_batch_int = X[:BATCH_SIZE], y[:BATCH_SIZE]
            y_b = (
                y_batch_int.astype(np.int32)
                if PROBLEM_TYPE == "CCE"
                else np.eye(OUTPUT_CLASSES)[y_batch_int].astype(SCALAR_NP_TYPE)
            )

            plan = self.strategy.create_plan_for_batch(X_batch.shape[0])
            processor = BatchProcessor(self, X_batch, y_b)
            inference_event, final_event = processor.run(plan)

            inference_event.wait()
            # Use the HostView to get the valid, un-padded result data
            valid_probs = processor.host_probs_view.get()
            predicted_classes = np.argmax(valid_probs[0, :, :], axis=1)  # Using first exit's probs for accuracy
            accuracy = np.mean(predicted_classes == y_batch_int)

            final_event.wait()

            print(f"Epoch {self.global_step:4d} | Accuracy: {accuracy:.2%}")
            self.global_step += 1

        print("\nTraining finished.")


if __name__ == "__main__":
    orchestrator = TrainingOrchestrator()
    orchestrator.train()
