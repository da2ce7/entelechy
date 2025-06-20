import os
import pyopencl as cl
import numpy as np
import math
from math import (
    gcd,
)
import networkx as nx
from collections import deque, defaultdict
from enum import Enum, auto
from dataclasses import dataclass, field
from typing import Tuple, List, Set, Dict, Callable, Optional, Union
from abc import ABC, abstractmethod
from sklearn.datasets import load_iris
from sklearn.preprocessing import StandardScaler

# --- Configuration ---
SCALAR_TYPE = "half"
SCALAR_NP_TYPE = np.float16 if SCALAR_TYPE == "half" else np.float32
CL_SCALAR_TYPE = "half" if SCALAR_TYPE == "half" else "float"
SCALAR_SIZE: int = SCALAR_NP_TYPE().itemsize
INT_SIZE: int = np.int32().itemsize

# --- Core Network Architecture ---
INPUT_DIM: int = 4
HIDDEN_DIM: int = 65
OUTPUT_CLASSES: int = 3
NUM_EXITS: int = 128  # Can be set to 1, 20, or 5000 to test different aggregation tiers.
BATCH_SIZE: int = 130
EPOCHS: int = 100

# --- Hyperparameters & Runtime Tuning Constants ---
PROBLEM_TYPE = "CCE"
LEARNING_RATE: float = 0.001
ADAM_BETA1: float = 0.9
ADAM_BETA2: float = 0.999
EPSILON: float = 1.0e-8
MIN_TEMP: float = 1.0e-3
MAX_TEMP: float = 10.0

# --- Compile-time Kernel Constants ---
# Architectural Insight: These constants define the thresholds for the tiered memory strategy.
# They are compile-time constants because they can affect kernel logic (e.g., loop bounds).
MAX_REGISTER_AGGREGATE_ITEMS: int = 64  # Tier 1 threshold: Max items to reduce in registers.
C_TILE_SIZE: int = 32

# --- File Configuration ---
KERNEL_DIR: str = "kernels"
CL_HEADERS: List[str] = ["kernels.cl.h"]
CL_SOURCES: List[str] = [
    "chunk_kernels.cl.c",
    "aggregation_kernels.cl.c",
    "global_kernels.cl.c",
    "parameter_optim.cl.c",
]


### UTILITY FUNCTIONS (Unchanged)
def validate_problem_config():
    if PROBLEM_TYPE == "CCE" and OUTPUT_CLASSES <= 1:
        raise ValueError(f"CCE requires at least two output classes, but OUTPUT_CLASSES is {OUTPUT_CLASSES}")


def load_and_concatenate_kernels(kernel_dir: str, headers: List[str], sources: List[str]) -> str:
    full_source_parts = []
    all_files_in_order = headers + sources
    for filename in all_files_in_order:
        full_path = os.path.join(kernel_dir, filename)
        try:
            with open(full_path, "r", encoding="utf-8") as f:
                full_source_parts.append(f.read())
        except FileNotFoundError:
            raise IOError(f"Fatal: Required kernel source file not found at '{full_path}'.")
    print(f"Successfully loaded {len(all_files_in_order)} kernel files from '{kernel_dir}'.")
    return "\n".join(full_source_parts)


def pad_to_multiple(dim: int, multiple: int) -> int:
    return (dim + multiple - 1) // multiple * multiple


def he_init(shape: Tuple[int, ...]) -> np.ndarray:
    fan_in = shape[0] if len(shape) == 2 else shape[1]
    scale = np.sqrt(2.0 / fan_in)
    return np.random.normal(0, scale, shape).astype(SCALAR_NP_TYPE)


def he_init_simd_major(
    initial_weights: np.ndarray, real_shape: Tuple[int, int], padded_shape: Tuple[int, int, int], simd_width: int
) -> np.ndarray:
    input_dim, hidden_dim = real_shape
    padded_hidden_dim = pad_to_multiple(hidden_dim, simd_width)
    padded_weights = np.zeros((input_dim, padded_hidden_dim), dtype=SCALAR_NP_TYPE)
    padded_weights[:, :hidden_dim] = initial_weights
    return padded_weights.reshape(input_dim, padded_hidden_dim // simd_width, simd_width).transpose(1, 0, 2)


def select_simd_width(device: cl.Device) -> int:
    if "Intel" in device.vendor:
        return 16
    if "NVIDIA" in device.vendor:
        return 32
    if "AMD" in device.vendor:
        return 64
    return 8


def validate_targets(y: np.ndarray):
    if PROBLEM_TYPE == "CCE" and not np.issubdtype(y.dtype, np.integer):
        raise ValueError("For CCE, target labels must be integers.")


def pad_tensor(data: np.ndarray, padded_shape: Tuple[int, ...]) -> np.ndarray:
    if data.shape == padded_shape:
        return data
    padded = np.zeros(padded_shape, dtype=data.dtype)
    padded[tuple(slice(0, d) for d in data.shape)] = data
    return padded


### CORE ABSTRACTIONS (Unchanged)
class BufferRole(Enum):
    WEIGHTS, BIAS, EXIT_WEIGHTS, EXIT_BIAS, TEMPERATURES, GRADIENT, ADAM_M1, ADAM_M2 = [auto() for _ in range(8)]
    INPUT_DATA, TARGETS, HIDDEN_ACTIVATION, INTERMEDIATE, FINAL_LOSS = [auto() for _ in range(5)]


@dataclass(frozen=True)
class Parameter:
    name: str

    @property
    def value(self) -> str:
        return self.name

    @property
    def grad(self) -> str:
        return f"grad_{self.name}"

    @property
    def m1(self) -> str:
        return f"m1_{self.name}"

    @property
    def m2(self) -> str:
        return f"m2_{self.name}"


@dataclass
class PaddingContext:
    simd_width: int

    @classmethod
    def from_device(cls, device: cl.Device) -> "PaddingContext":
        return cls(simd_width=select_simd_width(device))


class PaddingStrategy(ABC):
    @abstractmethod
    def get_padded_shape(self, real_shape: Tuple[int, ...], ctx: PaddingContext, batch_size: int) -> Tuple[int, ...]:
        pass


@dataclass
class NoPaddingStrategy(PaddingStrategy):
    def get_padded_shape(self, real_shape: Tuple[int, ...], ctx: PaddingContext, batch_size: int) -> Tuple[int, ...]:
        return real_shape


@dataclass
class PadLastDimStrategy(PaddingStrategy):
    def get_padded_shape(self, real_shape: Tuple[int, ...], ctx: PaddingContext, batch_size: int) -> Tuple[int, ...]:
        if not real_shape:
            return ()
            return (*real_shape[:-1], pad_to_multiple(real_shape[-1], ctx.simd_width))


@dataclass
class PadBatchAndLastDimStrategy(PaddingStrategy):
    def get_padded_shape(self, real_shape: Tuple[int, ...], ctx: PaddingContext, batch_size: int) -> Tuple[int, ...]:
        if len(real_shape) < 2:
            return (pad_to_multiple(batch_size, ctx.simd_width),)
        return (
            pad_to_multiple(batch_size, ctx.simd_width),
            *real_shape[1:-1],
            pad_to_multiple(real_shape[-1], ctx.simd_width),
        )


@dataclass
class PadBatchDimOnlyStrategy(PaddingStrategy):
    def get_padded_shape(self, real_shape: Tuple[int, ...], ctx: PaddingContext, batch_size: int) -> Tuple[int, ...]:
        if not real_shape:
            return ()
            return (pad_to_multiple(batch_size, ctx.simd_width), *real_shape[1:])


@dataclass
class SimdMajorWeightPadding(PaddingStrategy):
    def get_padded_shape(self, real_shape: Tuple[int, ...], ctx: PaddingContext, batch_size: int) -> Tuple[int, ...]:
        input_dim, hidden_dim = real_shape
        return (pad_to_multiple(hidden_dim, ctx.simd_width) // ctx.simd_width, input_dim, ctx.simd_width)


@dataclass
class StandardMatrixPadding(PaddingStrategy):
    def get_padded_shape(self, real_shape: Tuple[int, ...], ctx: PaddingContext, batch_size: int) -> Tuple[int, ...]:
        return (pad_to_multiple(real_shape[0], ctx.simd_width), pad_to_multiple(real_shape[1], ctx.simd_width))


@dataclass
class PadLastTwoDimsStrategy(PaddingStrategy):
    def get_padded_shape(self, real_shape: Tuple[int, ...], ctx: PaddingContext, batch_size: int) -> Tuple[int, ...]:
        return (
            *real_shape[:-2],
            pad_to_multiple(real_shape[-2], ctx.simd_width),
            pad_to_multiple(real_shape[-1], ctx.simd_width),
        )


PADDING_RULE_REGISTRY: Dict[BufferRole, Dict[Optional[int], PaddingStrategy]] = {
    BufferRole.WEIGHTS: {2: SimdMajorWeightPadding()},
    BufferRole.BIAS: {None: PadLastDimStrategy()},
    BufferRole.EXIT_WEIGHTS: {None: PadLastTwoDimsStrategy()},
    BufferRole.EXIT_BIAS: {None: PadLastTwoDimsStrategy()},
    BufferRole.TEMPERATURES: {None: PadLastDimStrategy()},
    BufferRole.GRADIENT: {1: PadLastDimStrategy(), 2: StandardMatrixPadding(), 3: PadLastTwoDimsStrategy()},
    BufferRole.ADAM_M1: {1: PadLastDimStrategy(), 2: StandardMatrixPadding(), 3: PadLastTwoDimsStrategy()},
    BufferRole.ADAM_M2: {1: PadLastDimStrategy(), 2: StandardMatrixPadding(), 3: PadLastTwoDimsStrategy()},
    BufferRole.INPUT_DATA: {None: PadBatchDimOnlyStrategy()},
    BufferRole.TARGETS: {1: PadBatchDimOnlyStrategy(), 2: PadBatchAndLastDimStrategy()},
    BufferRole.HIDDEN_ACTIVATION: {None: PadBatchAndLastDimStrategy()},
    BufferRole.INTERMEDIATE: {1: NoPaddingStrategy(), 2: PadBatchAndLastDimStrategy(), 3: PadBatchAndLastDimStrategy()},
    BufferRole.FINAL_LOSS: {None: NoPaddingStrategy()},
}


@dataclass(frozen=True)
class BufferSpec:
    name: str
    role: BufferRole
    real_shape: Tuple[int, ...]
    padded_shape: Tuple[int, ...]
    dtype: np.dtype


@dataclass
class Buffer:
    spec: BufferSpec
    cl_buffer: cl.Buffer
    mask_buffer: Optional[cl.Buffer] = None


class BufferManager:
    def __init__(self, context: cl.Context, device: cl.Device, batch_size: int):
        self.context, self.padding_ctx, self.buffers, self.batch_size, self.padding_registry = (
            context,
            PaddingContext.from_device(device),
            {},
            batch_size,
            PADDING_RULE_REGISTRY,
        )

    def _get_padded_shape(self, role: BufferRole, real_shape: Tuple[int, ...]) -> Tuple[int, ...]:
        rank = len(real_shape)
        strategy = self.padding_registry[role].get(rank, self.padding_registry[role].get(None))
        return strategy.get_padded_shape(real_shape, self.padding_ctx, self.batch_size)

    def create_buffer(
        self,
        name: str,
        role: BufferRole,
        real_shape: Tuple[int, ...],
        dtype: np.dtype,
        init_data: Optional[np.ndarray] = None,
    ) -> Buffer:
        if name in self.buffers:
            return self.buffers[name]
        padded_shape = self._get_padded_shape(role, real_shape)
        size = int(np.prod(padded_shape) * np.dtype(dtype).itemsize) if np.prod(padded_shape) > 0 else 0
        cl_buf = cl.Buffer(self.context, cl.mem_flags.READ_WRITE, size=max(size, 4))
        mask_buf = None
        if role in [BufferRole.INPUT_DATA, BufferRole.TARGETS, BufferRole.HIDDEN_ACTIVATION]:
            mask_buf = cl.Buffer(
                self.context,
                cl.mem_flags.READ_WRITE,
                size=max(int(pad_to_multiple(self.batch_size, self.padding_ctx.simd_width) * SCALAR_SIZE), 4),
            )
        buffer_obj = Buffer(BufferSpec(name, role, real_shape, padded_shape, dtype), cl_buf, mask_buf)
        self.buffers[name] = buffer_obj
        if init_data is not None:
            cl.enqueue_copy(cl.CommandQueue(self.context), cl_buf, pad_tensor(init_data, padded_shape)).wait()
        return buffer_obj

    def get(self, name: str) -> Buffer:
        return self.buffers[name]


class BatchPadder:
    def __init__(self, input_spec: BufferSpec, target_spec: BufferSpec):
        self.input_spec, self.target_spec = input_spec, target_spec

    def pad_batch(self, X: np.ndarray, y_true: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        X_padded, y_padded = pad_tensor(X, self.input_spec.padded_shape), pad_tensor(
            y_true, self.target_spec.padded_shape
        )
        mask_padded = pad_tensor(np.ones(X.shape[0], dtype=SCALAR_NP_TYPE), (self.input_spec.padded_shape[0],))
        return X_padded, y_padded, mask_padded


class HostView:
    def __init__(self, buffer: Buffer):
        self.spec, self.host_data = buffer.spec, np.empty(buffer.spec.padded_shape, dtype=buffer.spec.dtype)

    def update_from_device(self, queue: cl.CommandQueue, cl_buffer: cl.Buffer, wait_for=None):
        return cl.enqueue_copy(queue, self.host_data, cl_buffer, wait_for=wait_for or [])

    @property
    def valid_slice(self):
        return self.host_data[tuple(slice(0, dim) for dim in self.spec.real_shape)]


class NodeType(Enum):
    COMPUTE, TRANSFER, SYNC = auto(), auto(), auto()


class AccessMode(Enum):
    SHARED_READ, EXCLUSIVE_WRITE, EXCLUSIVE_UPDATE = auto(), auto(), auto()


ACCESS_WRITE_MODES: Set[AccessMode] = {AccessMode.EXCLUSIVE_WRITE, AccessMode.EXCLUSIVE_UPDATE}


@dataclass
class ExecutionNode:
    uid: int
    name: str
    node_type: NodeType
    queue_name: str
    access_map: Dict[str, AccessMode]
    operation_fn: Callable
    event: Optional[cl.Event] = None


class WorkManager:
    def __init__(self, context: cl.Context):
        self.context, self.properties, self.graph = (
            context,
            cl.command_queue_properties.OUT_OF_ORDER_EXEC_MODE_ENABLE,
            nx.DiGraph(),
        )
        self.queues: Dict[str, cl.CommandQueue] = {
            "Q_TRAINING": cl.CommandQueue(context, properties=self.properties),
            "Q_INFERENCE": cl.CommandQueue(context, properties=self.properties),
        }
        self.node_id_counter, self.last_writer = 0, {}

    def _get_uid(self) -> int:
        uid = self.node_id_counter
        self.node_id_counter += 1
        return uid

    def create_node(
        self, name: str, node_type: NodeType, queue_name: str, access_map: Dict[str, AccessMode], op_fn: Callable
    ) -> int:
        uid = self._get_uid()
        node = ExecutionNode(uid, name, node_type, queue_name, access_map, op_fn)
        self.graph.add_node(uid, node=node)
        for res, mode in access_map.items():
            if res in self.last_writer:
                self.graph.add_edge(self.last_writer[res], uid)
            if mode in ACCESS_WRITE_MODES:
                self.last_writer[res] = uid
        return uid

    def create_host_notification_point(self, name: str, dependencies: List[int]) -> cl.UserEvent:
        uid, user_event = self._get_uid(), cl.UserEvent(self.context)

        def _cb(q, wf):
            m = cl.enqueue_marker(q, wait_for=wf)
            m.set_callback(
                cl.command_execution_status.COMPLETE,
                lambda e, s: (
                    user_event.set_status(cl.command_execution_status.COMPLETE)
                    if s == cl.command_execution_status.COMPLETE
                    else None
                ),
            )
            return m

        q_name = self.graph.nodes[dependencies[0]]["node"].queue_name if dependencies else "Q_TRAINING"
        node = ExecutionNode(uid, name, NodeType.SYNC, q_name, {}, _cb)
        self.graph.add_node(uid, node=node)
        [self.graph.add_edge(dep, uid) for dep in dependencies]
        return user_event

    def commit(self):
        ordered_nodes, completion_events = list(nx.topological_sort(self.graph)), {}
        for uid in ordered_nodes:
            node, q = self.graph.nodes[uid]["node"], self.queues[self.graph.nodes[uid]["node"].queue_name]
            wait_for = [
                completion_events[p_uid] for p_uid in self.graph.predecessors(uid) if p_uid in completion_events
            ]
            event = node.operation_fn(q, wait_for)
            if event:
                completion_events[uid] = event

    def reset(self):
        self.graph.clear()
        self.node_id_counter, self.last_writer = 0, {}


### KERNEL WRAPPER (Refactored for the new modular architecture)
class KernelWrapper:
    def __init__(self, program: cl.Program, manager: WorkManager, buffer_manager: BufferManager, global_step: int):
        self.p, self.m, self.b = program, manager, buffer_manager
        self.padded_input_dim = self.b.get("input_buf").spec.padded_shape[1]
        _hp = self.b.get("hidden_buf").spec.padded_shape
        self.padded_hidden_dim, _hpb = _hp[1], _hp[0]
        self.padded_batch_size = _hpb
        self.padded_output_classes = self.b.get("exit_weights").spec.padded_shape[2]
        self.beta1_t, self.beta2_t = SCALAR_NP_TYPE(1.0 - (ADAM_BETA1**global_step)), SCALAR_NP_TYPE(
            1.0 - (ADAM_BETA2**global_step)
        )

    def _create_kernel_node(self, name: str, g: Tuple, l: Tuple, q: str, acc: Dict, *args) -> int:
        op = lambda q, wf: getattr(self.p, name)(q, g, l, *args, wait_for=wf)
        return self.m.create_node(name, NodeType.COMPUTE, q, acc, op)

    def forward_pass(self, q: str, off: int, n: int) -> int:
        g, l = (self.padded_batch_size, self.padded_hidden_dim // self.b.padding_ctx.simd_width), (
            self.b.padding_ctx.simd_width,
        )
        acc = {
            "input_buf": AccessMode.SHARED_READ,
            "weights": AccessMode.SHARED_READ,
            "biases": AccessMode.SHARED_READ,
            "hidden_buf": AccessMode.EXCLUSIVE_WRITE,
        }
        args = (
            cl.LocalMemory(2 * l[0] * SCALAR_SIZE),
            self.b.get("input_buf").cl_buffer,
            self.b.get("input_buf").mask_buffer,
            self.b.get("weights").cl_buffer,
            self.b.get("biases").cl_buffer,
            self.b.get("hidden_buf").cl_buffer,
            self.b.get("hidden_buf").mask_buffer,
            np.int32(off),
            np.int32(n),
            np.int32(self.padded_input_dim),
            np.int32(self.padded_hidden_dim),
        )
        return self._create_kernel_node("forward_pass", g, l, q, acc, *args)

    def compute_chunk_outputs(self, q: str, cid: int, cs: int, po: int) -> int:
        g, l = (cs, self.padded_batch_size), None
        acc = {
            "hidden_buf": AccessMode.SHARED_READ,
            "targets_buf": AccessMode.SHARED_READ,
            "exit_weights": AccessMode.SHARED_READ,
            "exit_biases": AccessMode.SHARED_READ,
            "temps": AccessMode.SHARED_READ,
            "partial_logits_buf": AccessMode.EXCLUSIVE_WRITE,
            "partial_probs_buf": AccessMode.EXCLUSIVE_WRITE,
            "partial_loss_buf": AccessMode.EXCLUSIVE_WRITE,
        }
        args = (
            self.b.get("hidden_buf").cl_buffer,
            self.b.get("targets_buf").cl_buffer,
            self.b.get("hidden_buf").mask_buffer,
            self.b.get("targets_buf").mask_buffer,
            self.b.get("exit_weights").cl_buffer,
            self.b.get("exit_biases").cl_buffer,
            self.b.get("temps").cl_buffer,
            self.b.get("partial_logits_buf").cl_buffer,
            self.b.get("partial_probs_buf").cl_buffer,
            self.b.get("partial_loss_buf").cl_buffer,
            np.int32(PROBLEM_TYPE == "CCE"),
            np.int32(cid),
            np.int32(po),
            np.int32(self.padded_batch_size),
            np.int32(self.padded_hidden_dim),
            np.int32(self.padded_output_classes),
            np.int32(cs),
        )
        return self._create_kernel_node("compute_chunk_outputs", g, l, q, acc, *args)

    def calculate_chunk_gradients(self, q: str, cid: int, cs: int, po: int) -> int:
        ls = 256
        g, l = (cs, self.padded_hidden_dim, ls), (1, 1, ls)
        acc = {
            "hidden_buf": AccessMode.SHARED_READ,
            "partial_probs_buf": AccessMode.SHARED_READ,
            "targets_buf": AccessMode.SHARED_READ,
            "exit_weights": AccessMode.SHARED_READ,
            "partial_grad_h_buf": AccessMode.EXCLUSIVE_WRITE,
            "grad_exit_weights": AccessMode.EXCLUSIVE_UPDATE,
            "grad_exit_biases": AccessMode.EXCLUSIVE_UPDATE,
        }
        args = (
            cl.LocalMemory(ls * SCALAR_SIZE),
            self.b.get("hidden_buf").cl_buffer,
            self.b.get("partial_probs_buf").cl_buffer,
            self.b.get("targets_buf").cl_buffer,
            self.b.get("exit_weights").cl_buffer,
            self.b.get("partial_grad_h_buf").cl_buffer,
            self.b.get("grad_exit_weights").cl_buffer,
            self.b.get("grad_exit_biases").cl_buffer,
            np.int32(PROBLEM_TYPE == "CCE"),
            np.int32(cid),
            np.int32(po),
            np.int32(self.padded_batch_size),
            np.int32(self.padded_hidden_dim),
            np.int32(self.padded_output_classes),
            np.int32(cs),
        )
        return self._create_kernel_node("calculate_chunk_gradients", g, l, q, acc, *args)

    def calculate_chunk_temp_gradients(self, q: str, cid: int, cs: int, po: int) -> int:
        ls = 256
        g, l = (cs * ls,), (ls,)
        acc = {
            "partial_logits_buf": AccessMode.SHARED_READ,
            "partial_probs_buf": AccessMode.SHARED_READ,
            "targets_buf": AccessMode.SHARED_READ,
            "temps": AccessMode.SHARED_READ,
            "partial_grad_temps_buf": AccessMode.EXCLUSIVE_WRITE,
        }
        args = (
            cl.LocalMemory(ls * SCALAR_SIZE),
            self.b.get("partial_logits_buf").cl_buffer,
            self.b.get("partial_probs_buf").cl_buffer,
            self.b.get("targets_buf").cl_buffer,
            self.b.get("targets_buf").mask_buffer,
            self.b.get("temps").cl_buffer,
            self.b.get("partial_grad_temps_buf").cl_buffer,
            np.int32(PROBLEM_TYPE == "CCE"),
            np.int32(cid),
            np.int32(po),
            np.int32(self.padded_batch_size),
            np.int32(self.padded_output_classes),
            np.int32(cs),
        )
        return self._create_kernel_node("calculate_chunk_temp_gradients", g, l, q, acc, *args)

    def aggregate(
        self,
        q: str,
        in_buf: str,
        out_buf: str,
        num_items: int,
        item_stride: int,
        mode: int = 0,
        avg_div: Optional[int] = None,
    ) -> int:
        acc = {in_buf: AccessMode.SHARED_READ, out_buf: AccessMode.EXCLUSIVE_WRITE}
        divisor = np.int32(avg_div if avg_div is not None else num_items)
        if num_items == 1:
            g, l = (item_stride,), None
            args = (self.b.get(in_buf).cl_buffer, self.b.get(out_buf).cl_buffer, np.int32(item_stride))
            return self._create_kernel_node("aggregate_identity", g, l, q, acc, *args)
        elif 1 < num_items <= MAX_REGISTER_AGGREGATE_ITEMS:
            g, l = (item_stride,), None
            args = (
                self.b.get(in_buf).cl_buffer,
                self.b.get(out_buf).cl_buffer,
                np.int32(num_items),
                np.int32(item_stride),
                np.int32(mode),
                divisor,
            )
            return self._create_kernel_node("aggregate_register_reduce", g, l, q, acc, *args)
        else:
            ls = 256
            g, l = (item_stride * ls,), (ls,)
            args = (
                cl.LocalMemory(ls * SCALAR_SIZE),
                self.b.get(in_buf).cl_buffer,
                self.b.get(out_buf).cl_buffer,
                np.int32(num_items),
                np.int32(item_stride),
                np.int32(mode),
                divisor,
            )
            return self._create_kernel_node("aggregate_local_reduce", g, l, q, acc, *args)

    def finalize_backprop_activation(self, q: str) -> int:
        g, l = (self.padded_batch_size, self.padded_hidden_dim), None
        acc = {
            "final_grad_h_buf": AccessMode.SHARED_READ,
            "hidden_buf": AccessMode.SHARED_READ,
            "grad_pre_activation_buf": AccessMode.EXCLUSIVE_WRITE,
        }
        args = (
            self.b.get("final_grad_h_buf").cl_buffer,
            self.b.get("hidden_buf").cl_buffer,
            self.b.get("hidden_buf").mask_buffer,
            self.b.get("grad_pre_activation_buf").cl_buffer,
            np.int32(self.padded_batch_size),
            np.int32(self.padded_hidden_dim),
        )
        return self._create_kernel_node("finalize_backprop_activation", g, l, q, acc, *args)

    def calculate_dense_layer_gradients(self, q: str) -> int:
        ls = 256
        g, l = (self.padded_input_dim, self.padded_hidden_dim * ls), (1, ls)
        acc = {
            "input_buf": AccessMode.SHARED_READ,
            "grad_pre_activation_buf": AccessMode.SHARED_READ,
            "grad_weights": AccessMode.EXCLUSIVE_UPDATE,
            "grad_biases": AccessMode.EXCLUSIVE_UPDATE,
        }
        args = (
            cl.LocalMemory(ls * SCALAR_SIZE),
            cl.LocalMemory(ls * SCALAR_SIZE),
            self.b.get("input_buf").cl_buffer,
            self.b.get("input_buf").mask_buffer,
            self.b.get("grad_pre_activation_buf").cl_buffer,
            self.b.get("grad_weights").cl_buffer,
            self.b.get("grad_biases").cl_buffer,
            np.int32(self.padded_batch_size),
            np.int32(self.padded_input_dim),
            np.int32(self.padded_hidden_dim),
        )
        return self._create_kernel_node("calculate_dense_layer_gradients", g, l, q, acc, *args)

    def adam_update(
        self, p: Parameter, q: str, offset_in_elements: int = 0, count_in_elements: Optional[int] = None
    ) -> int:
        full_param_count = int(np.prod(self.b.get(p.value).spec.padded_shape))
        num_elements = count_in_elements if count_in_elements is not None else full_param_count - offset_in_elements
        g, l = (num_elements,), None
        acc = {
            p.grad: AccessMode.SHARED_READ,
            p.value: AccessMode.EXCLUSIVE_UPDATE,
            p.m1: AccessMode.EXCLUSIVE_UPDATE,
            p.m2: AccessMode.EXCLUSIVE_UPDATE,
        }
        args = (
            self.b.get(p.grad).cl_buffer,
            SCALAR_NP_TYPE(ADAM_BETA1),
            SCALAR_NP_TYPE(ADAM_BETA2),
            self.beta1_t,
            self.beta2_t,
            SCALAR_NP_TYPE(LEARNING_RATE),
            SCALAR_NP_TYPE(EPSILON),
            self.b.get(p.value).cl_buffer,
            self.b.get(p.m1).cl_buffer,
            self.b.get(p.m2).cl_buffer,
            np.int32(offset_in_elements),
            np.int32(num_elements),
        )
        return self._create_kernel_node("adam_update", g, l, q, acc, *args)

    def clamp_temperatures(self, q: str) -> int:
        g, l = (NUM_EXITS,), None
        acc = {"temps": AccessMode.EXCLUSIVE_UPDATE}
        args = (self.b.get("temps").cl_buffer, SCALAR_NP_TYPE(MIN_TEMP), SCALAR_NP_TYPE(MAX_TEMP), np.int32(NUM_EXITS))
        return self._create_kernel_node("clamp_temperatures", g, l, q, acc, *args)

    def zero_gradients(self, grad_buf: str, q: str) -> int:
        buf = self.b.get(grad_buf)
        op = lambda q, wf: cl.enqueue_fill_buffer(
            q, buf.cl_buffer, SCALAR_NP_TYPE(0), 0, buf.cl_buffer.size, wait_for=wf
        )
        return self.m.create_node(f"zero_{grad_buf}", NodeType.COMPUTE, q, {grad_buf: AccessMode.EXCLUSIVE_WRITE}, op)


### MAIN EXECUTION ###
validate_problem_config()
ctx = cl.create_some_context(interactive=False)
device = ctx.devices[0]
print(f"Using device: {device.name} from vendor: {device.vendor}")
manager = WorkManager(ctx)
buffer_mgr = BufferManager(ctx, device, BATCH_SIZE)
if SCALAR_TYPE == "half" and "cl_khr_fp16" not in device.extensions:
    raise RuntimeError("FP16 not supported")
params = [
    Parameter("weights"),
    Parameter("biases"),
    Parameter("exit_weights"),
    Parameter("exit_biases"),
    Parameter("temps"),
]
param_specs = {
    "weights": (BufferRole.WEIGHTS, (INPUT_DIM, HIDDEN_DIM), he_init),
    "biases": (BufferRole.BIAS, (HIDDEN_DIM,), lambda s: np.zeros(s, dtype=SCALAR_NP_TYPE)),
    "exit_weights": (BufferRole.EXIT_WEIGHTS, (NUM_EXITS, HIDDEN_DIM, OUTPUT_CLASSES), he_init),
    "exit_biases": (BufferRole.EXIT_BIAS, (NUM_EXITS, OUTPUT_CLASSES), lambda s: np.zeros(s, dtype=SCALAR_NP_TYPE)),
    "temps": (BufferRole.TEMPERATURES, (NUM_EXITS,), lambda s: np.full(s, 1.0, dtype=SCALAR_NP_TYPE)),
}
for p in params:
    role, shape, init_fn = param_specs[p.name]
    if p.name == "weights":
        std_data = init_fn(shape)
        buffer_mgr.create_buffer("weights_standard", BufferRole.GRADIENT, shape, SCALAR_NP_TYPE, init_data=std_data)
        sw, ps = buffer_mgr.padding_ctx.simd_width, buffer_mgr._get_padded_shape(role, shape)
        simd_data = he_init_simd_major(std_data, shape, ps, sw)
        buffer_mgr.buffers[p.value] = Buffer(
            BufferSpec(p.value, role, shape, ps, SCALAR_NP_TYPE),
            cl.Buffer(ctx, cl.mem_flags.READ_WRITE, size=simd_data.nbytes),
        )
        cl.enqueue_copy(cl.CommandQueue(ctx), buffer_mgr.get(p.value).cl_buffer, simd_data).wait()
        for suffix in ["grad", "m1", "m2"]:
            buffer_mgr.create_buffer(getattr(p, suffix), BufferRole.GRADIENT, shape, SCALAR_NP_TYPE)
    else:
        buffer_mgr.create_buffer(p.value, role, shape, SCALAR_NP_TYPE, init_data=init_fn(shape))
        [
            buffer_mgr.create_buffer(getattr(p, s), BufferRole.GRADIENT, shape, SCALAR_NP_TYPE)
            for s in ["grad", "m1", "m2"]
        ]
data_buffer_specs = {
    "input_buf": (BufferRole.INPUT_DATA, (BATCH_SIZE, INPUT_DIM), SCALAR_NP_TYPE),
    "targets_buf": (BufferRole.TARGETS, (BATCH_SIZE,), np.int32),
    "hidden_buf": (BufferRole.HIDDEN_ACTIVATION, (BATCH_SIZE, HIDDEN_DIM), SCALAR_NP_TYPE),
    "partial_logits_buf": (BufferRole.INTERMEDIATE, (NUM_EXITS, BATCH_SIZE, OUTPUT_CLASSES), SCALAR_NP_TYPE),
    "partial_probs_buf": (BufferRole.INTERMEDIATE, (NUM_EXITS, BATCH_SIZE, OUTPUT_CLASSES), SCALAR_NP_TYPE),
    "partial_loss_buf": (BufferRole.INTERMEDIATE, (NUM_EXITS, BATCH_SIZE), np.float32),
    "partial_grad_h_buf": (BufferRole.INTERMEDIATE, (NUM_EXITS, BATCH_SIZE, HIDDEN_DIM), SCALAR_NP_TYPE),
    "partial_grad_temps_buf": (BufferRole.GRADIENT, (NUM_EXITS,), SCALAR_NP_TYPE),
    "final_probs_buf": (BufferRole.INTERMEDIATE, (BATCH_SIZE, OUTPUT_CLASSES), SCALAR_NP_TYPE),
    "final_loss_buf": (BufferRole.INTERMEDIATE, (BATCH_SIZE,), np.float32),
    "final_grad_h_buf": (BufferRole.INTERMEDIATE, (BATCH_SIZE, HIDDEN_DIM), SCALAR_NP_TYPE),
    "grad_pre_activation_buf": (BufferRole.INTERMEDIATE, (BATCH_SIZE, HIDDEN_DIM), SCALAR_NP_TYPE),
    "grad_input_buf": (BufferRole.INTERMEDIATE, (BATCH_SIZE, INPUT_DIM), SCALAR_NP_TYPE),
    "final_scalar_loss_buf": (BufferRole.FINAL_LOSS, (1,), np.float32),
}
for name, (role, shape, dtype) in data_buffer_specs.items():
    buffer_mgr.create_buffer(name, role, shape, dtype)
X, y_int = load_iris(return_X_y=True)
X_normalized = StandardScaler().fit_transform(X).astype(SCALAR_NP_TYPE)
y_true = y_int.astype(np.int32)
validate_targets(y_true)
batch_padder = BatchPadder(buffer_mgr.get("input_buf").spec, buffer_mgr.get("targets_buf").spec)
kernel_src = load_and_concatenate_kernels(KERNEL_DIR, CL_HEADERS, CL_SOURCES)
build_opts = [
    f"-cl-std=CL1.2",
    f"-D SCALAR_TYPE={CL_SCALAR_TYPE}",
    f"-D SIMD_WIDTH={buffer_mgr.padding_ctx.simd_width}",
    f"-D C_TILE_SIZE={C_TILE_SIZE}",
    f"-D PROBLEM_TYPE_CCE=0",
    f"-D AGG_MODE_SUM=0",
    f"-D AGG_MODE_AVERAGE=1",
] + ([f"-D cl_khr_fp16"] if SCALAR_TYPE == "half" else [])
program = cl.Program(ctx, kernel_src).build(options=build_opts)
loss_view, probs_view = HostView(buffer_mgr.get("final_scalar_loss_buf")), HostView(buffer_mgr.get("final_probs_buf"))
print(f"Starting training for {EPOCHS} epochs...")
global_step = 1
for epoch in range(EPOCHS):
    shuffled_indices = np.random.permutation(len(X))
    epoch_loss, correct_predictions, total_samples = 0.0, 0, 0
    for i in range(0, len(X), BATCH_SIZE):
        manager.reset()
        batch_indices = shuffled_indices[i : i + BATCH_SIZE]
        actual_batch_size = len(batch_indices)
        X_batch, y_batch = X_normalized[batch_indices], y_true[batch_indices]
        X_padded, y_padded, mask_padded = batch_padder.pad_batch(X_batch, y_batch)
        k = KernelWrapper(program, manager, buffer_mgr, global_step)

        # === Host Orchestrator: Dynamic DAG Construction ===

        # --- Phase 0: PLAN ---
        def _calculate_transient_mem() -> int:
            # A simplified VRAM budgeter. A real-world version would be more precise.
            # It estimates the size of the largest temporary buffers needed for a full-batch run.
            mem = 0
            mem += buffer_mgr.get("hidden_buf").cl_buffer.size  # Hidden activations
            mem += buffer_mgr.get("partial_probs_buf").cl_buffer.size  # Probs for all exits
            mem += buffer_mgr.get("partial_grad_h_buf").cl_buffer.size  # Grad_H for all exits
            return mem

        required_mem = _calculate_transient_mem()
        # Use 90% of VRAM for safety margin
        if required_mem > device.global_mem_size * 0.9:
            num_chunks = math.ceil(required_mem / (device.global_mem_size * 0.5))  # Heuristic for chunk count
            num_chunks = min(num_chunks, NUM_EXITS)
        else:
            num_chunks = 1
        chunk_size = (NUM_EXITS + num_chunks - 1) // num_chunks
        print(
            f"Epoch {epoch+1}, Batch {i//BATCH_SIZE+1}: Strategy={num_chunks} chunk(s) of size {chunk_size} based on VRAM budget."
        )

        # --- Initial State Setup ---
        h2d_ops = {
            "input_buf": X_padded,
            "input_buf:mask": mask_padded,
            "targets_buf": y_padded,
            "targets_buf:mask": mask_padded,
        }
        for name, data in h2d_ops.items():
            base_name, is_mask = name.split(":")[:2]
            target_buf = buffer_mgr.get(base_name).mask_buffer if is_mask else buffer_mgr.get(base_name).cl_buffer
            op_fn = lambda d=data, b=target_buf: (lambda q, wf: cl.enqueue_copy(q, b, d, wait_for=wf))
            manager.create_node(
                f"h2d_{name}", NodeType.TRANSFER, "Q_TRAINING", {base_name: AccessMode.EXCLUSIVE_WRITE}, op_fn()
            )
        for param in params:
            k.zero_gradients(param.grad, "Q_TRAINING")
        fp_node = k.forward_pass("Q_TRAINING", 0, actual_batch_size)

        # --- Phase 3: STREAMING ---
        partial_nodes = defaultdict(list)
        exit_update_nodes = []
        for c in range(num_chunks):
            param_offset = c * chunk_size
            current_chunk_size = min(chunk_size, NUM_EXITS - param_offset)
            outputs_uid = k.compute_chunk_outputs("Q_TRAINING", c, current_chunk_size, param_offset)
            grads_uid = k.calculate_chunk_gradients("Q_TRAINING", c, current_chunk_size, param_offset)
            temps_uid = k.calculate_chunk_temp_gradients("Q_TRAINING", c, current_chunk_size, param_offset)

            # Architectural Insight: Streaming Adam update for this chunk's exit parameters. This is a key
            # memory optimization, as we don't need to store all exit gradients simultaneously.
            # This assumes the `adam_update` kernel now supports an `offset_in_elements` argument.
            ew_spec = buffer_mgr.get("exit_weights").spec
            ew_offset = param_offset * ew_spec.real_shape[1] * ew_spec.real_shape[2]
            ew_count = current_chunk_size * ew_spec.real_shape[1] * ew_spec.real_shape[2]
            eb_spec = buffer_mgr.get("exit_biases").spec
            eb_offset = param_offset * eb_spec.real_shape[1]
            eb_count = current_chunk_size * eb_spec.real_shape[1]
            ew_update = k.adam_update(Parameter("exit_weights"), "Q_TRAINING", ew_offset, ew_count)
            manager.graph.add_edge(grads_uid, ew_update)
            eb_update = k.adam_update(Parameter("exit_biases"), "Q_TRAINING", eb_offset, eb_count)
            manager.graph.add_edge(grads_uid, eb_update)
            exit_update_nodes.extend([ew_update, eb_update])

            partial_nodes["Probs"].append(outputs_uid)
            partial_nodes["Loss"].append(outputs_uid)
            partial_nodes["Grad_H"].append(grads_uid)
            partial_nodes["Grad_Temps"].append(temps_uid)

        # --- Phase 4: AGGREGATION ---
        aggregated_nodes = {}
        buffers_to_agg = {
            "Probs": "partial_probs_buf",
            "Loss": "partial_loss_buf",
            "Grad_H": "partial_grad_h_buf",
            "Grad_Temps": "partial_grad_temps_buf",
        }
        for name, partial_buf in buffers_to_agg.items():
            final_buf = f"final_{name.lower()}_buf" if name != "Grad_Temps" else "grad_temps"
            item_stride = int(np.prod(buffer_mgr.get(partial_buf).spec.padded_shape[1:]))
            agg_node = k.aggregate("Q_TRAINING", partial_buf, final_buf, num_chunks, item_stride)
            aggregated_nodes[name] = agg_node

        # --- Phase 5: FINALIZATION & DISPATCH ---
        final_training_nodes = list(exit_update_nodes)
        d2h_op = lambda v, b: (lambda q, wf: v.update_from_device(q, buffer_mgr.get(b).cl_buffer, wf))
        probs_d2h = manager.create_node(
            "d2h_probs",
            NodeType.TRANSFER,
            "Q_INFERENCE",
            {"final_probs_buf": AccessMode.SHARED_READ},
            d2h_op(probs_view, "final_probs_buf"),
        )
        manager.graph.add_edge(aggregated_nodes["Probs"], probs_d2h)
        backprop = k.finalize_backprop_activation("Q_TRAINING")
        manager.graph.add_edge(aggregated_nodes["Grad_H"], backprop)
        dense_grads = k.calculate_dense_layer_gradients("Q_TRAINING")
        manager.graph.add_edge(backprop, dense_grads)
        for p_name in ["weights", "biases"]:
            update = k.adam_update(Parameter(p_name), "Q_TRAINING")
            manager.graph.add_edge(dense_grads, update)
            final_training_nodes.append(update)
        temps_update = k.adam_update(Parameter("temps"), "Q_TRAINING")
        manager.graph.add_edge(aggregated_nodes["Grad_Temps"], temps_update)
        clamp = k.clamp_temperatures("Q_TRAINING")
        manager.graph.add_edge(temps_update, clamp)
        final_training_nodes.append(clamp)

        # Architectural Insight: Final loss calculation is now a generic aggregation. We reduce the
        # per-sample final_loss_buf to a single scalar, but crucially, we use AVERAGE mode and
        # provide the *actual* batch size as the divisor to get a correct, unpadded average.
        item_stride = 1
        num_loss_items = buffer_mgr.get("final_loss_buf").spec.padded_shape[0]
        loss_scalar_node = k.aggregate(
            "Q_TRAINING",
            "final_loss_buf",
            "final_scalar_loss_buf",
            num_loss_items,
            item_stride,
            mode=1,
            avg_div=actual_batch_size,
        )
        manager.graph.add_edge(aggregated_nodes["Loss"], loss_scalar_node)
        loss_d2h = manager.create_node(
            "d2h_loss",
            NodeType.TRANSFER,
            "Q_TRAINING",
            {"final_scalar_loss_buf": AccessMode.SHARED_READ},
            d2h_op(loss_view, "final_scalar_loss_buf"),
        )
        manager.graph.add_edge(loss_scalar_node, loss_d2h)
        final_training_nodes.append(loss_d2h)

        # --- Host Notification & Synchronization ---
        inference_event = manager.create_host_notification_point("inference_sync", dependencies=[probs_d2h])
        final_batch_event = manager.create_host_notification_point("training_sync", dependencies=final_training_nodes)

        # === COMMIT AND STAGGERED WAIT ===
        manager.commit()
        inference_event.wait()
        valid_probs = probs_view.valid_slice[:actual_batch_size]
        predicted_classes = np.argmax(valid_probs, axis=1)
        correct_predictions += np.sum(predicted_classes == y_batch)
        total_samples += actual_batch_size
        final_batch_event.wait()
        total_batch_loss = loss_view.valid_slice[0]
        epoch_loss += total_batch_loss * actual_batch_size
        global_step += 1

    avg_epoch_loss = epoch_loss / total_samples if total_samples > 0 else 0
    train_acc = correct_predictions / total_samples if total_samples > 0 else 0
    print(f"Epoch {epoch+1:3d}/{EPOCHS} | Loss: {avg_epoch_loss:.4f} | Accuracy: {train_acc:.2%}")

print("\nTraining finished.")
