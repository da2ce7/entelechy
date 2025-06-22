import os
import pyopencl as cl
import numpy as np
import math
from collections import defaultdict
from dataclasses import dataclass
from typing import Tuple, List, Dict, Optional, Callable
from abc import ABC, abstractmethod

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
# Max partials to reduce in registers; above this, local memory is used.
MAX_REGISTER_AGGREGATE_ITEMS: int = 32
# Tile size for the `transpose_grad_h` kernel.
C_TILE_SIZE: int = 16
# Defines the granularity of the streaming backpropagation for the shared layer
BACKPROP_STREAM_CHUNK_SIZE: int = 32


# --- Utility Functions ---
def load_and_concatenate_kernels() -> str:
    """Loads kernel sources from disk. Assumes they are in the same directory."""
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
            raise FileNotFoundError(
                f"Missing required kernel file: {filename}. Please ensure it's in the current directory."
            )
        with open(filename, "r", encoding="utf-8") as f:
            source_parts.append(f.read())
    print(f"Loaded {len(source_parts)} kernel source files.")
    return "\n".join(source_parts)


def select_simd_width(device: cl.Device) -> int:
    """Selects a reasonable SIMD width based on vendor."""
    if "NVIDIA" in device.vendor:
        return 32
    if "AMD" in device.vendor:
        return 64
    if "Intel" in device.vendor:
        return 16
    return 8


def pad_to_multiple(dim: int, multiple: int) -> int:
    """Calculates the smallest multiple of `multiple` greater than or equal to `dim`."""
    return (dim + multiple - 1) // multiple * multiple


# --- Layer 1: Action Layer (Stateless Tools) ---


class BufferManager:
    """Manages creation, padding, and access to all OpenCL memory buffers."""

    def __init__(self, context: cl.Context, simd_width: int):
        self.context = context
        self.simd_width = simd_width
        self.buffers: Dict[str, cl.Buffer] = {}
        self.specs: Dict[str, Tuple[Tuple[int, ...], np.dtype]] = {}

    def create_buffer(
        self, name: str, real_shape: Tuple[int, ...], dtype: np.dtype, init_data: Optional[np.ndarray] = None
    ) -> None:
        """Creates a buffer with appropriate padding and optional initial data."""
        padded_shape = list(real_shape)
        if name.endswith("_buf") or name.endswith("_out"):  # Simplified padding rule
            if len(padded_shape) > 0 and padded_shape[-1] > 1:
                padded_shape[-1] = pad_to_multiple(padded_shape[-1], self.simd_width)
            if len(padded_shape) > 1 and (name.startswith("input") or name.startswith("hidden")):
                padded_shape[0] = pad_to_multiple(padded_shape[0], self.simd_width)

        padded_shape = tuple(padded_shape)
        byte_size = int(np.prod(padded_shape) * dtype().itemsize) if padded_shape else 4

        mem_flags = cl.mem_flags.READ_WRITE
        if init_data is not None:
            mem_flags |= cl.mem_flags.COPY_HOST_PTR
            padded_data = np.zeros(padded_shape, dtype=dtype)
            padded_data[tuple(slice(0, d) for d in real_shape)] = init_data
            hostbuf = padded_data
        else:
            hostbuf = None

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

    def enqueue_write_buffer(self, queue, name, data, wait_for) -> cl.Event:
        """Enqueues a non-blocking write to a buffer."""
        buf = self.b.get(name)
        shape, dtype = self.b.get_spec(name)
        padded_data = np.zeros(shape, dtype=dtype)
        padded_data[tuple(slice(0, d) for d in data.shape)] = data
        return cl.enqueue_copy(queue, buf, padded_data, wait_for=wait_for)

    def enqueue_read_buffer(self, queue, name, host_dest, wait_for) -> cl.Event:
        """Enqueues a non-blocking read from a buffer."""
        return cl.enqueue_copy(queue, host_dest, self.b.get(name), wait_for=wait_for)

    def enqueue_fill_buffer(self, queue, name, value, wait_for) -> cl.Event:
        """Enqueues a fill operation for a buffer."""
        return cl.enqueue_fill_buffer(
            queue, self.b.get(name), SCALAR_NP_TYPE(value), 0, self.b.get(name).size, wait_for=wait_for
        )

    # --- Node 4 ---
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
            np.int32(self.padded_input_dim),
            np.int32(self.padded_hidden_dim),
        )
        return self.p.forward_pass(queue, g, l, *args, wait_for=wait_for)

    # --- Node 5 ---
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

    # --- Node 6 ---
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

    # --- Node 7a ---
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

    # --- Node 8, 9, 10 ---
    def launch_parallel_exit_grads(
        self, queue, exit_chunk_id, exit_offset, num_exits, class_chunk_id, class_offset, num_classes, wait_for
    ) -> Tuple[cl.Event, cl.Event, cl.Event]:
        # Launching all three kernels for a chunk back-to-back.
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
            None,
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

    # --- Node 11 ---
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

    # --- Node 12 & 15 ---
    def launch_aggregation(self, queue, in_buf, out_buf, num_partials, elements_per_partial, wait_for) -> cl.Event:
        lsize = 256
        if num_partials <= 1:
            g, l, kernel, args = (
                (elements_per_partial,),
                None,
                self.p.aggregate_identity,
                (
                    None,
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
                    None,
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

    # --- Node 13 & 14 ---
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

    # --- Node 17 & 18 ---
    def launch_adam_update_and_clamp(self, queue, p_name, grad_ready_event, b1_t, b2_t) -> cl.Event:
        num_params = int(np.prod(self.b.get_spec(p_name)[0]))
        update_evt = self.p.adam_update(
            queue,
            (num_params,),
            None,
            self.b.get(f"grad_{p_name}"),
            SCALAR_NP_TYPE(ADAM_BETA1),
            SCALAR_NP_TYPE(ADAM_BETA2),
            SCALAR_NP_TYPE(b1_t),
            SCALAR_NP_TYPE(b2_t),
            SCALAR_NP_TYPE(LEARNING_RATE),
            SCALAR_NP_TYPE(EPSILON),
            self.b.get(p_name),
            self.b.get(f"m1_{p_name}"),
            self.b.get(f"m2_{p_name}"),
            np.int32(0),
            np.int32(num_params),
            wait_for=[grad_ready_event],
        )
        if p_name == "temps":
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
    recompute_hidden: bool = False  # Placeholder for future implementation


class ExecutionStrategy:
    """Makes high-level strategic decisions based on resources and problem size."""

    def __init__(self, device_vram_bytes: int, buffer_mgr: BufferManager):
        self.vram_budget = device_vram_bytes * 0.90  # Use 90% of VRAM
        self.b = buffer_mgr

    def create_plan_for_batch(self, batch_size: int) -> ExecutionPlan:
        # Calculate memory for the largest transient buffers
        logits_size = self.b.get("full_logits_out").size
        probs_size = self.b.get("partial_probs_out").size
        grad_h_size = self.b.get("partial_grad_h_aos_out").size

        required_mem = logits_size + probs_size + grad_h_size

        if required_mem > self.vram_budget:
            # Simple strategy: chunk by the exit dimension to reduce memory pressure
            num_chunks = math.ceil(required_mem / self.vram_budget)
            print(f"VRAM pressure detected. Streaming with {num_chunks} exit chunks.")
            return ExecutionPlan(num_exit_chunks=num_chunks)

        return ExecutionPlan(num_exit_chunks=1)


# --- Layer 3: Execution Layer ---


class BatchProcessor:
    """Stateful engine to process one batch by building the event-based DAG."""

    def __init__(self, context, queue, executor, global_step):
        self.ctx, self.queue, self.executor = context, queue, executor
        self.events: Dict[str, cl.Event] = {}
        self.event_lists: Dict[str, List[cl.Event]] = defaultdict(list)
        self.global_step = global_step

    def _get_deps(self, *resource_names: str) -> List[cl.Event]:
        deps = []
        for name in resource_names:
            if name in self.events:
                deps.append(self.events[name])
            if name in self.event_lists:
                deps.extend(self.event_lists[name])
        return list(set(deps))

    def run(self, X_batch, y_batch, plan: ExecutionPlan):
        # Phase 0-3: Setup
        self._upload_data(X_batch, y_batch)
        self._zero_gradients()

        # Phase 4: Shared Forward
        self.events["hidden_ready"] = self.executor.launch_forward_pass(
            self.queue, 0, X_batch.shape[0], wait_for=self._get_deps("input_ready")
        )

        # Phase 5-10: Exit Path (streamed)
        self._execute_exit_path(plan)

        # Phase 11-15: Aggregation & Shared Backprop
        self._aggregate_and_backprop_shared(plan)

        # Phase 16-18: Finalization
        self._finalize_and_update()

        return self.get_sync_points()

    def _upload_data(self, X, y):
        # NOTE: CCE needs int32 targets, BCE might need floats. We handle inside main loop.
        write_X = self.executor.enqueue_write_buffer(self.queue, "input_buf", X, wait_for=None)
        write_mask = self.executor.enqueue_write_buffer(
            self.queue, "input_mask", np.ones(X.shape[0], dtype=SCALAR_NP_TYPE), wait_for=None
        )
        self.events["input_ready"] = cl.WaitForEvents([write_X, write_mask])
        self.events["targets_ready"] = self.executor.enqueue_write_buffer(self.queue, "targets_buf", y, wait_for=None)

    def _zero_gradients(self):
        grad_names = [
            "grad_weights",
            "grad_biases",
            "grad_exit_weights",
            "grad_exit_biases",
            "grad_temps",
            "partial_grad_sw_out",
            "partial_grad_sb_out",
        ]
        self.events["grads_zeroed"] = cl.WaitForEvents(
            [self.executor.enqueue_fill_buffer(self.queue, g, 0, wait_for=None) for g in grad_names]
        )

    def _execute_exit_path(self, plan: ExecutionPlan):
        exit_chunk_size = (NUM_EXITS + plan.num_exit_chunks - 1) // plan.num_exit_chunks

        # Stream Node 5
        hidden_deps = self._get_deps("hidden_ready")
        for i in range(plan.num_exit_chunks):
            offset, size = i * exit_chunk_size, min(exit_chunk_size, NUM_EXITS - i * exit_chunk_size)
            evt = self.executor.launch_compute_logits_chunk(
                self.queue, i, offset, size, 0, OUTPUT_CLASSES, wait_for=hidden_deps
            )
            self.event_lists["logit_chunks_ready"].append(evt)
        self.events["all_logits_ready"] = cl.WaitForEvents(self.event_lists["logit_chunks_ready"])

        softmax_params_deps = [self.events["all_logits_ready"]]
        if PROBLEM_TYPE == "CCE":
            self.events["softmax_params_ready"] = self.executor.launch_reduce_for_softmax(
                self.queue, wait_for=softmax_params_deps
            )

        # Stream Node 7 & 8,9,10
        for i in range(plan.num_exit_chunks):
            offset, size = i * exit_chunk_size, min(exit_chunk_size, NUM_EXITS - i * exit_chunk_size)
            if PROBLEM_TYPE == "CCE":
                prob_loss_deps = [self.events["softmax_params_ready"]]
                evt = self.executor.launch_compute_probs_loss_cce_chunk(
                    self.queue, i, offset, size, 0, OUTPUT_CLASSES, wait_for=prob_loss_deps
                )
                self.event_lists["prob_loss_chunks_ready"].append(evt)
            else:  # BCE not implemented in detail for brevity
                pass

            grad_deps = self._get_deps("hidden_ready", f"prob_loss_chunks_ready") + [self.events["all_logits_ready"]]
            w_evt, h_evt, t_evt = self.executor.launch_parallel_exit_grads(
                self.queue, i, offset, size, i, 0, OUTPUT_CLASSES, wait_for=grad_deps
            )
            self.event_lists["partial_grad_w_ready"].append(w_evt)
            self.event_lists["partial_grad_h_ready"].append(h_evt)
            self.event_lists["partial_grad_t_ready"].append(t_evt)
        self.events["all_prob_loss_chunks_ready"] = cl.WaitForEvents(self.event_lists["prob_loss_chunks_ready"])

    def _aggregate_and_backprop_shared(self, plan: ExecutionPlan):
        # Node 11 & 12
        num_h_items = BATCH_SIZE * HIDDEN_DIM
        transpose_deps = self.event_lists["partial_grad_h_ready"]
        transpose_evt = self.executor.launch_transpose_grad_h(
            self.queue, NUM_EXITS, num_h_items, wait_for=transpose_deps
        )

        self.events["final_grad_h_ready"] = self.executor.launch_aggregation(
            self.queue, "grad_h_soa_buf", "final_grad_h_buf", NUM_EXITS, num_h_items, wait_for=[transpose_evt]
        )
        self.events["final_grad_exit_w_ready"] = self.executor.launch_aggregation(
            self.queue,
            "partial_grad_exit_w_out",
            "grad_exit_weights",
            plan.num_exit_chunks,
            self.b.get("grad_exit_weights").size // plan.num_exit_chunks,
            wait_for=self.event_lists["partial_grad_w_ready"],
        )
        self.events["final_grad_exit_b_ready"] = self.executor.launch_aggregation(
            self.queue,
            "partial_grad_exit_b_out",
            "grad_exit_biases",
            plan.num_exit_chunks,
            self.b.get("grad_exit_biases").size // plan.num_exit_chunks,
            wait_for=self.event_lists["partial_grad_w_ready"],
        )
        self.events["final_grad_temps_ready"] = self.executor.launch_aggregation(
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
            self.b.get("final_probs_buf").size,
            wait_for=self.event_lists["prob_loss_chunks_ready"],
        )

        # Node 13, 14, 15
        backprop_deps = self._get_deps("hidden_ready", "final_grad_h_ready")
        for i in range(plan.num_batch_chunks):
            offset, size = i * BACKPROP_STREAM_CHUNK_SIZE, min(
                BACKPROP_STREAM_CHUNK_SIZE, BATCH_SIZE - i * BACKPROP_STREAM_CHUNK_SIZE
            )
            sw_evt, sb_evt = self.executor.launch_backprop_shared_chunk(
                self.queue, i, offset, size, wait_for=backprop_deps
            )
            self.event_lists["partial_grad_sw_ready"].append(sw_evt)
            self.event_lists["partial_grad_sb_ready"].append(sb_evt)

        self.events["final_grad_sw_ready"] = self.executor.launch_aggregation(
            self.queue,
            "partial_grad_sw_out",
            "grad_weights",
            plan.num_batch_chunks,
            self.b.get("grad_weights").size,
            wait_for=self.event_lists["partial_grad_sw_ready"],
        )
        self.events["final_grad_sb_ready"] = self.executor.launch_aggregation(
            self.queue,
            "partial_grad_sb_out",
            "grad_biases",
            plan.num_batch_chunks,
            self.b.get("grad_biases").size,
            wait_for=self.event_lists["partial_grad_sb_ready"],
        )

    def _finalize_and_update(self):
        # Node 16
        self.events["d2h_probs_ready"] = self.executor.enqueue_read_buffer(
            self.queue, "final_probs_buf", self.host_probs, wait_for=self._get_deps("final_probs_ready")
        )

        # Node 17 & 18
        b1_t, b2_t = (1.0 - ADAM_BETA1**self.global_step), (1.0 - ADAM_BETA2**self.global_step)
        adam_deps = {
            "weights": "final_grad_sw_ready",
            "biases": "final_grad_sb_ready",
            "exit_weights": "final_grad_exit_w_ready",
            "exit_biases": "final_grad_exit_b_ready",
            "temps": "final_grad_temps_ready",
        }
        for name, dep_key in adam_deps.items():
            self.event_lists["updates_done"].append(
                self.executor.launch_adam_update_and_clamp(self.queue, name, self.events[dep_key], b1_t, b2_t)
            )
        self.events["all_updates_done"] = cl.WaitForEvents(self.event_lists["updates_done"])

    def get_sync_points(self) -> Tuple[cl.UserEvent, cl.UserEvent]:
        # Pre-allocate host-side buffers before returning
        self.host_probs = np.empty(self.executor.b.get_spec("final_probs_buf")[0], dtype=SCALAR_NP_TYPE)

        inference_event, final_batch_event = cl.UserEvent(self.ctx), cl.UserEvent(self.ctx)
        cl.enqueue_marker(self.queue, wait_for=[self.events["d2h_probs_ready"]]).set_callback(
            cl.command_execution_status.COMPLETE, lambda s, u: inference_event.set_status(s)
        )
        cl.enqueue_marker(self.queue, wait_for=[self.events["all_updates_done"]]).set_callback(
            cl.command_execution_status.COMPLETE, lambda s, u: final_batch_event.set_status(s)
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
        params = [
            ("weights", (INPUT_DIM, HIDDEN_DIM)),
            ("biases", (HIDDEN_DIM,)),
            ("exit_weights", (NUM_EXITS, HIDDEN_DIM, OUTPUT_CLASSES)),
            ("exit_biases", (NUM_EXITS, OUTPUT_CLASSES)),
            ("temps", (NUM_EXITS,)),
        ]
        for name, shape in params:
            init_fn = (
                np.zeros
                if "biases" in name
                else (
                    (lambda s: np.full(s, 1.0, dtype=SCALAR_NP_TYPE))
                    if "temps" in name
                    else (lambda s: np.random.randn(*s).astype(SCALAR_NP_TYPE) * 0.1)
                )
            )
            self.buffer_mgr.create_buffer(name, shape, SCALAR_NP_TYPE, init_fn(shape))
            for prefix in ["grad_", "m1_", "m2_"]:
                self.buffer_mgr.create_buffer(f"{prefix}{name}", self.buffer_mgr.get_spec(name)[0], SCALAR_NP_TYPE)

        # Intermediate buffers
        int_buffers = {
            "input_buf": (BATCH_SIZE, INPUT_DIM),
            "input_mask": (BATCH_SIZE,),
            "hidden_buf": (BATCH_SIZE, HIDDEN_DIM),
            "hidden_mask": (BATCH_SIZE,),
            "targets_buf": (BATCH_SIZE,) if PROBLEM_TYPE == "CCE" else (BATCH_SIZE, OUTPUT_CLASSES),
            "full_logits_out": (NUM_EXITS, BATCH_SIZE, OUTPUT_CLASSES),
            "partial_probs_out": (NUM_EXITS, BATCH_SIZE, OUTPUT_CLASSES),
            "final_probs_buf": (NUM_EXITS, BATCH_SIZE, OUTPUT_CLASSES),
            "final_loss_out": (NUM_EXITS, BATCH_SIZE),
            "softmax_params_out": (NUM_EXITS, BATCH_SIZE, 2),
            "partial_grad_exit_w_out": self.buffer_mgr.get_spec("grad_exit_weights")[0],
            "partial_grad_exit_b_out": self.buffer_mgr.get_spec("grad_exit_biases")[0],
            "partial_grad_temps_out": (NUM_EXITS * 20,),  # Large enough for many chunks
            "partial_grad_h_aos_out": (NUM_EXITS, BATCH_SIZE * HIDDEN_DIM),
            "grad_h_soa_buf": (BATCH_SIZE * HIDDEN_DIM, NUM_EXITS),
            "final_grad_h_buf": (BATCH_SIZE, HIDDEN_DIM),
            "partial_grad_sw_out": (BACKPROP_STREAM_CHUNK_SIZE * 2, INPUT_DIM, HIDDEN_DIM),
            "partial_grad_sb_out": (BACKPROP_STREAM_CHUNK_SIZE * 2, HIDDEN_DIM),
        }

        for name, shape in int_buffers.items():
            dtype = np.int32 if name == "targets_buf" and PROBLEM_TYPE == "CCE" else SCALAR_NP_TYPE
            self.buffer_mgr.create_buffer(name, shape, dtype)

    def train(self):
        from sklearn.datasets import load_iris
        from sklearn.preprocessing import StandardScaler

        X, y = load_iris(return_X_y=True)
        X = StandardScaler().fit_transform(X).astype(SCALAR_NP_TYPE)

        print(f"Starting training for {EPOCHS} epochs...")
        for epoch in range(EPOCHS):
            # A real implementation would loop over minibatches here.
            X_batch, y_batch_int = X[:BATCH_SIZE], y[:BATCH_SIZE]

            # Prepare targets based on problem type
            y_b = (
                y_batch_int.astype(np.int32)
                if PROBLEM_TYPE == "CCE"
                else np.eye(OUTPUT_CLASSES)[y_batch_int].astype(SCALAR_NP_TYPE)
            )

            # 1. Create the strategic plan for this batch
            plan = self.strategy.create_plan_for_batch(X_batch.shape[0])

            # 2. Instantiate a fresh processor and run
            processor = BatchProcessor(self.ctx, self.queue, self.executor, self.global_step)
            inference_event, final_event = processor.run(X_batch, y_b, plan)

            # 3. Wait and process results
            inference_event.wait()
            valid_probs = processor.host_probs[0, : X_batch.shape[0], :]  # Probs are per-exit
            predicted_classes = np.argmax(valid_probs, axis=1)
            accuracy = np.mean(predicted_classes == y_batch_int)

            final_event.wait()

            print(f"Epoch {self.global_step:4d} | Accuracy: {accuracy:.2%}")
            self.global_step += 1

        print("\nTraining finished.")


if __name__ == "__main__":
    orchestrator = TrainingOrchestrator()
    orchestrator.train()
