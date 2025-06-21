import os
import pyopencl as cl
import numpy as np
import math
from collections import defaultdict
from enum import Enum, auto
from dataclasses import dataclass
from typing import Tuple, List, Set, Dict, Callable, Optional
from abc import ABC, abstractmethod
import networkx as nx
from sklearn.datasets import load_iris
from sklearn.preprocessing import StandardScaler

# --- Configuration ---
SCALAR_TYPE = "half"
SCALAR_NP_TYPE = np.float16 if SCALAR_TYPE == "half" else np.float32
CL_SCALAR_TYPE = "half" if SCALAR_TYPE == "half" else "float"
SCALAR_SIZE: int = SCALAR_NP_TYPE().itemsize

# --- Core Network Architecture ---
INPUT_DIM: int = 4
HIDDEN_DIM: int = 65
OUTPUT_CLASSES: int = 3
NUM_EXITS: int = 128
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

# --- Compile-time Kernel Constants & Architectural Tuning ---
MAX_REGISTER_AGGREGATE_ITEMS: int = 64
C_TILE_SIZE: int = 32
# Defines the granularity of the streaming backpropagation for the shared layer
BACKPROP_STREAM_CHUNK_SIZE: int = 32

# --- File Configuration ---
KERNEL_DIR: str = "."
CL_HEADERS: List[str] = ["kernels.cl.h"]
CL_SOURCES: List[str] = [
    "chunk_kernels.cl.c",
    "aggregation_kernels.cl.c",
    "backprop_kernels.cl.c",
    "parameter_optim.cl.c",
]


### UTILITY FUNCTIONS
def validate_problem_config():
    if PROBLEM_TYPE == "CCE" and OUTPUT_CLASSES <= 1:
        raise ValueError("CCE requires at least two output classes.")


def load_and_concatenate_kernels(kernel_dir: str, headers: List[str], sources: List[str]) -> str:
    full_source_parts = []
    for filename in headers + sources:
        full_path = os.path.join(kernel_dir, filename)
        try:
            with open(full_path, "r", encoding="utf-8") as f:
                full_source_parts.append(f.read())
        except FileNotFoundError:
            raise IOError(f"Required kernel source file not found at '{full_path}'.")
    print(f"Loaded {len(headers) + len(sources)} kernel files.")
    return "\n".join(full_source_parts)


def pad_to_multiple(dim: int, multiple: int) -> int:
    return (dim + multiple - 1) // multiple * multiple


def he_init(shape: Tuple[int, ...]) -> np.ndarray:
    fan_in = shape[0] if len(shape) == 2 else shape[1]
    scale = np.sqrt(2.0 / fan_in)
    return np.random.normal(0, scale, shape).astype(SCALAR_NP_TYPE)


def select_simd_width(device: cl.Device) -> int:
    if "Intel" in device.vendor:
        return 16
    if "NVIDIA" in device.vendor:
        return 32
    if "AMD" in device.vendor:
        return 64
    return 8


def pad_tensor(data: np.ndarray, padded_shape: Tuple[int, ...]) -> np.ndarray:
    if data.shape == padded_shape:
        return data
    padded = np.zeros(padded_shape, dtype=data.dtype)
    padded[tuple(slice(0, d) for d in data.shape)] = data
    return padded


### CORE ABSTRACTIONS
class BufferRole(Enum):
    (
        WEIGHTS,
        BIAS,
        EXIT_WEIGHTS,
        EXIT_BIAS,
        TEMPERATURES,
        GRADIENT,
        ADAM_M1,
        ADAM_M2,
        INPUT_DATA,
        TARGETS,
        HIDDEN_ACTIVATION,
        INTERMEDIATE,
        FINAL_LOSS,
    ) = [auto() for _ in range(13)]


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
            if mode in {AccessMode.EXCLUSIVE_WRITE, AccessMode.EXCLUSIVE_UPDATE}:
                self.last_writer[res] = uid
        return uid

    def create_host_notification_point(self, name: str, dependencies: List[int]) -> cl.UserEvent:
        uid, user_event = self._get_uid(), cl.UserEvent(self.context)

        def _cb(q, wf):
            m = cl.enqueue_marker(q, wait_for=wf)
            m.set_callback(
                cl.command_execution_status.COMPLETE,
                lambda e, s: user_event.set_status(cl.command_execution_status.COMPLETE),
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
            node, q = self.graph.nodes[uid]["node"], self.queues[node.queue_name]
            wait_for = [
                completion_events[p_uid] for p_uid in self.graph.predecessors(uid) if p_uid in completion_events
            ]
            event = node.operation_fn(q, wait_for)
            if event:
                completion_events[uid] = event

    def reset(self):
        self.graph.clear()
        self.node_id_counter, self.last_writer = 0, {}


@dataclass
class PaddingContext:
    simd_width: int


class PaddingStrategy(ABC):
    @abstractmethod
    def get_padded_shape(self, real_shape: Tuple[int, ...], ctx: PaddingContext, batch_size: int) -> Tuple[int, ...]:
        pass


class NoPaddingStrategy(PaddingStrategy):
    def get_padded_shape(self, real_shape: Tuple[int, ...], ctx: PaddingContext, batch_size: int) -> Tuple[int, ...]:
        return real_shape


class PadToMultipleStrategy(PaddingStrategy):
    def __init__(self, multiple_fn: Callable[[PaddingContext], int]):
        self.multiple_fn = multiple_fn

    def get_padded_shape(self, real_shape: Tuple[int, ...], ctx: PaddingContext, batch_size: int) -> Tuple[int, ...]:
        multiple = self.multiple_fn(ctx)
        return tuple(pad_to_multiple(d, multiple) for d in real_shape)


class PadLastDimStrategy(PaddingStrategy):
    def get_padded_shape(self, real_shape: Tuple[int, ...], ctx: PaddingContext, batch_size: int) -> Tuple[int, ...]:
        if not real_shape:
            return ()
        return (*real_shape[:-1], pad_to_multiple(real_shape[-1], ctx.simd_width))


class PadBatchAndLastDimStrategy(PaddingStrategy):
    def get_padded_shape(self, real_shape: Tuple[int, ...], ctx: PaddingContext, batch_size: int) -> Tuple[int, ...]:
        if len(real_shape) < 2:
            return (pad_to_multiple(batch_size, ctx.simd_width),)
        return (
            pad_to_multiple(batch_size, ctx.simd_width),
            *real_shape[1:-1],
            pad_to_multiple(real_shape[-1], ctx.simd_width),
        )


class PadBatchDimOnlyStrategy(PaddingStrategy):
    def get_padded_shape(self, real_shape: Tuple[int, ...], ctx: PaddingContext, batch_size: int) -> Tuple[int, ...]:
        if not real_shape:
            return (pad_to_multiple(batch_size, ctx.simd_width),)
        return (pad_to_multiple(batch_size, ctx.simd_width), *real_shape[1:])


class SimdMajorWeightPadding(PaddingStrategy):
    def get_padded_shape(self, real_shape: Tuple[int, ...], ctx: PaddingContext, batch_size: int) -> Tuple[int, ...]:
        input_dim, hidden_dim = real_shape
        padded_hidden_dim = pad_to_multiple(hidden_dim, ctx.simd_width)
        return (padded_hidden_dim // ctx.simd_width, input_dim, ctx.simd_width)


PADDING_RULE_REGISTRY: Dict[BufferRole, PaddingStrategy] = {
    BufferRole.WEIGHTS: SimdMajorWeightPadding(),
    BufferRole.BIAS: PadLastDimStrategy(),
    BufferRole.EXIT_WEIGHTS: PadLastDimStrategy(),
    BufferRole.EXIT_BIAS: PadLastDimStrategy(),
    BufferRole.TEMPERATURES: PadLastDimStrategy(),
    BufferRole.GRADIENT: PadToMultipleStrategy(lambda ctx: ctx.simd_width),
    BufferRole.ADAM_M1: PadToMultipleStrategy(lambda ctx: ctx.simd_width),
    BufferRole.ADAM_M2: PadToMultipleStrategy(lambda ctx: ctx.simd_width),
    BufferRole.INPUT_DATA: PadBatchAndLastDimStrategy(),
    BufferRole.TARGETS: PadBatchDimOnlyStrategy(),
    BufferRole.HIDDEN_ACTIVATION: PadBatchAndLastDimStrategy(),
    BufferRole.INTERMEDIATE: NoPaddingStrategy(),
    BufferRole.FINAL_LOSS: NoPaddingStrategy(),
}


class BufferManager:
    def __init__(self, context: cl.Context, device: cl.Device):
        self.context = context
        self.padding_ctx = PaddingContext(simd_width=select_simd_width(device))
        self.buffers: Dict[str, Buffer] = {}
        self.padding_registry = PADDING_RULE_REGISTRY

    def _get_padded_shape(self, role: BufferRole, real_shape: Tuple[int, ...], batch_size: int) -> Tuple[int, ...]:
        strategy = self.padding_registry.get(role, NoPaddingStrategy())
        return strategy.get_padded_shape(real_shape, self.padding_ctx, batch_size)

    def create_buffer(
        self,
        name: str,
        role: BufferRole,
        dtype: np.dtype,
        real_shape: Tuple[int, ...],
        batch_size: int,
        init_data: Optional[np.ndarray] = None,
    ) -> Buffer:
        padded_shape = self._get_padded_shape(role, real_shape, batch_size)
        size = int(np.prod(padded_shape) * np.dtype(dtype).itemsize) if padded_shape else 0
        mask_buf = None
        if role in {BufferRole.INPUT_DATA, BufferRole.HIDDEN_ACTIVATION, BufferRole.TARGETS}:
            mask_buf = cl.Buffer(
                self.context,
                cl.mem_flags.READ_WRITE,
                size=max(int(pad_to_multiple(batch_size, self.padding_ctx.simd_width) * SCALAR_SIZE), 4),
            )
        cl_buf = cl.Buffer(self.context, cl.mem_flags.READ_WRITE, size=max(size, 4))
        self.buffers[name] = Buffer(BufferSpec(name, role, real_shape, padded_shape, dtype), cl_buf, mask_buf)
        if init_data is not None:
            cl.enqueue_copy(cl.CommandQueue(self.context), cl_buf, pad_tensor(init_data, padded_shape)).wait()
        return self.buffers[name]

    def get(self, name: str) -> "Buffer":
        return self.buffers[name]


class KernelWrapper:
    def __init__(self, program: cl.Program, manager: WorkManager, buffer_manager: BufferManager, global_step: int):
        self.p, self.m, self.b = program, manager, buffer_manager
        self.simd_width = select_simd_width(manager.context.devices[0])
        self.padded_input_dim = self.b.get("input_buf").spec.padded_shape[1]
        self.padded_hidden_dim = self.b.get("hidden_buf").spec.padded_shape[1]
        self.hidden_dim = self.b.get("hidden_buf").spec.real_shape[1]
        self.output_classes = self.b.get("exit_weights").spec.real_shape[2]
        self.beta1_t = SCALAR_NP_TYPE(1.0 - (ADAM_BETA1**global_step))
        self.beta2_t = SCALAR_NP_TYPE(1.0 - (ADAM_BETA2**global_step))

    def _create_kernel_node(self, name: str, g: Tuple, l: Tuple, q: str, acc: Dict, *args):
        op = lambda q, wf: getattr(self.p, name)(q, g, l, *args, wait_for=wf)
        return self.m.create_node(name, NodeType.COMPUTE, q, acc, op)

    def forward_pass(self, q: str, off: int, n: int) -> int:
        g = (pad_to_multiple(n, self.simd_width), pad_to_multiple(self.hidden_dim, self.simd_width) // self.simd_width)
        l = (self.simd_width, 1)
        acc = {
            "input_buf": AccessMode.SHARED_READ,
            "weights": AccessMode.SHARED_READ,
            "biases": AccessMode.SHARED_READ,
            "hidden_buf": AccessMode.EXCLUSIVE_WRITE,
        }
        local_mem_size = l[0] * (1 + l[0]) * SCALAR_SIZE
        args = (
            cl.LocalMemory(local_mem_size),
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

    def compute_chunk_outputs(self, q: str, cid: int, cs: int, po: int, batch_size: int) -> int:
        g, l = (cs, batch_size), None
        acc = {
            "hidden_buf": AccessMode.SHARED_READ,
            "targets_buf": AccessMode.SHARED_READ,
            "exit_weights": AccessMode.SHARED_READ,
            "exit_biases": AccessMode.SHARED_READ,
            "temps": AccessMode.SHARED_READ,
            "partial_logits_out": AccessMode.EXCLUSIVE_WRITE,
            "partial_probs_out": AccessMode.EXCLUSIVE_WRITE,
            "partial_loss_out": AccessMode.EXCLUSIVE_WRITE,
        }
        args = (
            self.b.get("hidden_buf").cl_buffer,
            self.b.get("targets_buf").cl_buffer,
            self.b.get("hidden_buf").mask_buffer,
            self.b.get("targets_buf").mask_buffer,
            self.b.get("exit_weights").cl_buffer,
            self.b.get("exit_biases").cl_buffer,
            self.b.get("temps").cl_buffer,
            self.b.get("partial_logits_out").cl_buffer,
            self.b.get("partial_probs_out").cl_buffer,
            self.b.get("partial_loss_out").cl_buffer,
            np.int32(PROBLEM_TYPE == "CCE"),
            np.int32(cid),
            np.int32(po),
            np.int32(batch_size),
            np.int32(self.hidden_dim),
            np.int32(self.output_classes),
            np.int32(cs),
            np.int32(self.padded_hidden_dim),
        )
        return self._create_kernel_node("compute_chunk_outputs", g, l, q, acc, *args)

    def calculate_chunk_gradients(self, q: str, cid: int, cs: int, po: int, batch_size: int) -> int:
        ls = 256
        g, l = (cs * ls, self.hidden_dim), (ls, 1)
        acc = {
            "hidden_buf": AccessMode.SHARED_READ,
            "partial_probs_out": AccessMode.SHARED_READ,
            "targets_buf": AccessMode.SHARED_READ,
            "exit_weights": AccessMode.SHARED_READ,
            "partial_grad_h_aos_out": AccessMode.EXCLUSIVE_WRITE,
            "partial_grad_exit_w_out": AccessMode.EXCLUSIVE_WRITE,
            "partial_grad_exit_b_out": AccessMode.EXCLUSIVE_WRITE,
        }
        args = (
            cl.LocalMemory(ls * SCALAR_SIZE),
            self.b.get("hidden_buf").cl_buffer,
            self.b.get("partial_probs_out").cl_buffer,
            self.b.get("targets_buf").cl_buffer,
            self.b.get("exit_weights").cl_buffer,
            self.b.get("partial_grad_h_aos_out").cl_buffer,
            self.b.get("partial_grad_exit_w_out").cl_buffer,
            self.b.get("partial_grad_exit_b_out").cl_buffer,
            np.int32(PROBLEM_TYPE == "CCE"),
            np.int32(cid),
            np.int32(po),
            np.int32(batch_size),
            np.int32(self.hidden_dim),
            np.int32(self.output_classes),
            np.int32(cs),
            np.int32(self.padded_hidden_dim),
        )
        return self._create_kernel_node("calculate_chunk_gradients", g, l, q, acc, *args)

    def calculate_chunk_temp_gradients(self, q: str, cid: int, cs: int, po: int, batch_size: int) -> int:
        ls = 256
        g, l = (cs * ls,), (ls,)
        acc = {
            "partial_logits_out": AccessMode.SHARED_READ,
            "partial_probs_out": AccessMode.SHARED_READ,
            "targets_buf": AccessMode.SHARED_READ,
            "temps": AccessMode.SHARED_READ,
            "partial_grad_temps_out": AccessMode.EXCLUSIVE_WRITE,
        }
        args = (
            cl.LocalMemory(ls * SCALAR_SIZE),
            self.b.get("partial_logits_out").cl_buffer,
            self.b.get("partial_probs_out").cl_buffer,
            self.b.get("targets_buf").cl_buffer,
            self.b.get("targets_buf").mask_buffer,
            self.b.get("temps").cl_buffer,
            self.b.get("partial_grad_temps_out").cl_buffer,
            np.int32(PROBLEM_TYPE == "CCE"),
            np.int32(cid),
            np.int32(po),
            np.int32(batch_size),
            np.int32(self.output_classes),
            np.int32(cs),
        )
        return self._create_kernel_node("calculate_chunk_temp_gradients", g, l, q, acc, *args)

    def transpose_grad_h(self, q: str, in_buf: str, out_buf: str, num_chunks: int, num_items: int) -> int:
        g = (pad_to_multiple(num_items, C_TILE_SIZE), pad_to_multiple(num_chunks, C_TILE_SIZE))
        l = (C_TILE_SIZE, C_TILE_SIZE)
        acc = {in_buf: AccessMode.SHARED_READ, out_buf: AccessMode.EXCLUSIVE_WRITE}
        local_mem_size = C_TILE_SIZE * (C_TILE_SIZE + 1) * SCALAR_SIZE
        args = (
            cl.LocalMemory(local_mem_size),
            self.b.get(in_buf).cl_buffer,
            self.b.get(out_buf).cl_buffer,
            np.int32(num_chunks),
            np.int32(num_items),
        )
        return self._create_kernel_node("transpose_grad_h", g, l, q, acc, *args)

    def aggregate(
        self, q: str, in_buf_name: str, out_buf_name: str, num_items_to_reduce: int, item_stride: int, mode: int = 0
    ) -> int:
        acc = {in_buf_name: AccessMode.SHARED_READ, out_buf_name: AccessMode.EXCLUSIVE_WRITE}
        in_buf, out_buf = self.b.get(in_buf_name).cl_buffer, self.b.get(out_buf_name).cl_buffer
        if num_items_to_reduce <= 1:
            op_name, op_func, g, l, args = (
                "aggregate_identity",
                self.p.aggregate_identity,
                (item_stride,),
                None,
                (
                    cl.LocalMemory(0),
                    in_buf,
                    out_buf,
                    np.int32(num_items_to_reduce),
                    np.int32(item_stride),
                    np.int32(mode),
                ),
            )
        elif 1 < num_items_to_reduce <= MAX_REGISTER_AGGREGATE_ITEMS:
            op_name, op_func, g, l, args = (
                "aggregate_register_reduce",
                self.p.aggregate_register_reduce,
                (item_stride,),
                None,
                (
                    cl.LocalMemory(0),
                    in_buf,
                    out_buf,
                    np.int32(num_items_to_reduce),
                    np.int32(item_stride),
                    np.int32(mode),
                ),
            )
        else:
            ls = 256
            op_name, op_func, g, l, args = (
                "aggregate_local_reduce",
                self.p.aggregate_local_reduce,
                (item_stride * ls,),
                (ls,),
                (
                    cl.LocalMemory(ls * SCALAR_SIZE),
                    in_buf,
                    out_buf,
                    np.int32(num_items_to_reduce),
                    np.int32(item_stride),
                    np.int32(mode),
                ),
            )
        return self.m.create_node(op_name, NodeType.COMPUTE, q, acc, lambda q, wf: op_func(q, g, l, *args, wait_for=wf))

    def backprop_shared_weights_chunk(self, q: str, chunk_id: int, batch_offset: int, num_batch_samples: int) -> int:
        ls = 256
        g, l = (self.padded_input_dim * ls, self.padded_hidden_dim), (ls, 1)
        acc = {
            "input_buf": AccessMode.SHARED_READ,
            "hidden_buf": AccessMode.SHARED_READ,
            "final_grad_h_buf": AccessMode.SHARED_READ,
            "partial_grad_sw_out": AccessMode.EXCLUSIVE_WRITE,
        }
        args = (
            cl.LocalMemory(ls * SCALAR_SIZE),
            self.b.get("input_buf").cl_buffer,
            self.b.get("hidden_buf").cl_buffer,
            self.b.get("final_grad_h_buf").cl_buffer,
            self.b.get("input_buf").mask_buffer,
            self.b.get("partial_grad_sw_out").cl_buffer,
            np.int32(batch_offset),
            np.int32(num_batch_samples),
            np.int32(chunk_id),
            np.int32(self.padded_input_dim),
            np.int32(self.padded_hidden_dim),
        )
        return self._create_kernel_node("backprop_shared_weights_chunk", g, l, q, acc, *args)

    def backprop_shared_biases_chunk(self, q: str, chunk_id: int, batch_offset: int, num_batch_samples: int) -> int:
        ls = 256
        g, l = (self.padded_hidden_dim * ls,), (ls,)
        acc = {
            "hidden_buf": AccessMode.SHARED_READ,
            "final_grad_h_buf": AccessMode.SHARED_READ,
            "partial_grad_sb_out": AccessMode.EXCLUSIVE_WRITE,
        }
        args = (
            cl.LocalMemory(ls * SCALAR_SIZE),
            self.b.get("hidden_buf").cl_buffer,
            self.b.get("final_grad_h_buf").cl_buffer,
            self.b.get("input_buf").mask_buffer,
            self.b.get("partial_grad_sb_out").cl_buffer,
            np.int32(batch_offset),
            np.int32(num_batch_samples),
            np.int32(chunk_id),
            np.int32(self.padded_hidden_dim),
        )
        return self._create_kernel_node("backprop_shared_biases_chunk", g, l, q, acc, *args)

    def adam_update(self, p: Parameter, q: str) -> int:
        num_elements = int(np.prod(self.b.get(p.value).spec.padded_shape))
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
            np.int32(0),
            np.int32(num_elements),
        )
        return self._create_kernel_node("adam_update", g, l, q, acc, *args)

    def clamp_temperatures(self, q: str) -> int:
        g, l = (NUM_EXITS,), None
        acc = {"temps": AccessMode.EXCLUSIVE_UPDATE}
        args = (self.b.get("temps").cl_buffer, SCALAR_NP_TYPE(MIN_TEMP), SCALAR_NP_TYPE(MAX_TEMP), np.int32(NUM_EXITS))
        return self._create_kernel_node("clamp_temperatures", g, l, q, acc, *args)

    def zero_gradients(self, grad_buf_name: str, q: str) -> int:
        buf = self.b.get(grad_buf_name)
        op = lambda q, wf: cl.enqueue_fill_buffer(
            q, buf.cl_buffer, SCALAR_NP_TYPE(0), 0, buf.cl_buffer.size, wait_for=wf
        )
        return self.m.create_node(
            f"zero_{grad_buf_name}", NodeType.COMPUTE, q, {grad_buf_name: AccessMode.EXCLUSIVE_WRITE}, op
        )


### MAIN EXECUTION ###
def main():
    validate_problem_config()
    ctx = cl.create_some_context(interactive=False)
    device = ctx.devices[0]
    print(f"Using device: {device.name} from vendor: {device.vendor} with {device.global_mem_size / 1e9:.2f} GB VRAM")
    if SCALAR_TYPE == "half" and "cl_khr_fp16" not in device.extensions:
        raise RuntimeError("FP16 not supported")
    simd_width = select_simd_width(device)
    manager = WorkManager(ctx)
    buffer_mgr = BufferManager(ctx, device)

    params = [
        Parameter("weights"),
        Parameter("biases"),
        Parameter("exit_weights"),
        Parameter("exit_biases"),
        Parameter("temps"),
    ]
    param_shapes = {
        "weights": (INPUT_DIM, HIDDEN_DIM),
        "biases": (HIDDEN_DIM,),
        "exit_weights": (NUM_EXITS, HIDDEN_DIM, OUTPUT_CLASSES),
        "exit_biases": (NUM_EXITS, OUTPUT_CLASSES),
        "temps": (NUM_EXITS,),
    }
    for p in params:
        role = BufferRole[p.name.upper()]
        shape = param_shapes[p.name]
        init_fn = {
            "weights": he_init,
            "biases": np.zeros,
            "exit_weights": he_init,
            "exit_biases": np.zeros,
            "temps": lambda s: np.full(s, 1.0, dtype=SCALAR_NP_TYPE),
        }[p.name]
        init_data = init_fn(shape).astype(SCALAR_NP_TYPE)
        buffer_mgr.create_buffer(p.value, role, SCALAR_NP_TYPE, shape, BATCH_SIZE, init_data)
        for suffix in ["grad", "m1", "m2"]:
            buffer_mgr.create_buffer(
                getattr(p, suffix),
                BufferRole.GRADIENT,
                SCALAR_NP_TYPE,
                buffer_mgr.get(p.value).spec.padded_shape,
                BATCH_SIZE,
            )

    num_batch_chunks = (BATCH_SIZE + BACKPROP_STREAM_CHUNK_SIZE - 1) // BACKPROP_STREAM_CHUNK_SIZE
    data_buffer_specs = {
        "input_buf": (BufferRole.INPUT_DATA, (BATCH_SIZE, INPUT_DIM)),
        "targets_buf": (BufferRole.TARGETS, (BATCH_SIZE,)),
        "hidden_buf": (BufferRole.HIDDEN_ACTIVATION, (BATCH_SIZE, HIDDEN_DIM)),
        "partial_logits_out": (BufferRole.INTERMEDIATE, (NUM_EXITS, BATCH_SIZE, OUTPUT_CLASSES)),
        "partial_probs_out": (BufferRole.INTERMEDIATE, (NUM_EXITS, BATCH_SIZE, OUTPUT_CLASSES)),
        "partial_loss_out": (BufferRole.INTERMEDIATE, (NUM_EXITS, BATCH_SIZE)),
        "partial_grad_h_aos_out": (BufferRole.INTERMEDIATE, (NUM_EXITS, BATCH_SIZE * HIDDEN_DIM)),
        "grad_h_soa_buf": (BufferRole.INTERMEDIATE, (BATCH_SIZE * HIDDEN_DIM, NUM_EXITS)),
        "partial_grad_exit_w_out": (BufferRole.GRADIENT, (NUM_EXITS, HIDDEN_DIM, OUTPUT_CLASSES)),
        "partial_grad_exit_b_out": (BufferRole.GRADIENT, (NUM_EXITS, OUTPUT_CLASSES)),
        "partial_grad_temps_out": (BufferRole.GRADIENT, (NUM_EXITS,)),
        "partial_grad_sw_out": (BufferRole.GRADIENT, (num_batch_chunks, INPUT_DIM, HIDDEN_DIM)),
        "partial_grad_sb_out": (BufferRole.GRADIENT, (num_batch_chunks, HIDDEN_DIM)),
        "final_probs_buf": (BufferRole.INTERMEDIATE, (BATCH_SIZE, OUTPUT_CLASSES)),
        "final_loss_buf": (BufferRole.INTERMEDIATE, (BATCH_SIZE,)),
        "final_grad_h_buf": (BufferRole.INTERMEDIATE, (BATCH_SIZE, HIDDEN_DIM)),
        "final_scalar_loss_buf": (BufferRole.FINAL_LOSS, (1,)),
    }
    dtypes = {BufferRole.TARGETS: np.int32, BufferRole.FINAL_LOSS: np.float32, BufferRole.INTERMEDIATE: np.float32}
    for name, (role, shape) in data_buffer_specs.items():
        buffer_mgr.create_buffer(name, role, dtypes.get(role, SCALAR_NP_TYPE), shape, BATCH_SIZE)

    X, y_int = load_iris(return_X_y=True)
    X_normalized = StandardScaler().fit_transform(X).astype(SCALAR_NP_TYPE)
    kernel_src = load_and_concatenate_kernels(KERNEL_DIR, CL_HEADERS, CL_SOURCES)
    build_opts = [
        f"-cl-std=CL1.2",
        f"-D SCALAR_TYPE={CL_SCALAR_TYPE}",
        f"-D SIMD_WIDTH={simd_width}",
        f"-D C_TILE_SIZE={C_TILE_SIZE}",
    ] + ([f"-D cl_khr_fp16"] if SCALAR_TYPE == "half" else [])
    program = cl.Program(ctx, kernel_src).build(options=build_opts)

    loss_view = HostView(buffer_mgr.get("final_scalar_loss_buf"))
    probs_view = HostView(buffer_mgr.get("final_probs_buf"))
    print(f"Starting training for {EPOCHS} epochs...")
    global_step = 1

    for epoch in range(EPOCHS):
        shuffled_indices = np.random.permutation(len(X))
        epoch_loss, correct_predictions, total_samples = 0.0, 0, 0
        for i in range(0, len(X), BATCH_SIZE):
            manager.reset()
            batch_indices = shuffled_indices[i : i + BATCH_SIZE]
            actual_batch_size = len(batch_indices)
            X_batch, y_batch = X_normalized[batch_indices], y_int[batch_indices].astype(np.int32)
            k = KernelWrapper(program, manager, buffer_mgr, global_step)

            vram_budget = device.global_mem_size * 0.9
            transient_mem_cost = sum(
                buffer_mgr.get(b).cl_buffer.size for b in ["hidden_buf", "partial_probs_out", "partial_grad_h_aos_out"]
            )
            if transient_mem_cost > vram_budget:
                num_chunks = math.ceil(transient_mem_cost / (device.global_mem_size * 0.5))
                num_chunks = min(num_chunks, NUM_EXITS)
                print(f"Epoch {epoch+1}, Batch {i//BATCH_SIZE+1}: VRAM pressure detected. Using {num_chunks} chunks.")
            else:
                num_chunks = 1
            chunk_size = (NUM_EXITS + num_chunks - 1) // num_chunks

            padded_batch_size = buffer_mgr.get("input_buf").spec.padded_shape[0]
            for name, data in [("input_buf", X_batch), ("targets_buf", y_batch)]:
                buf, mask_buf = buffer_mgr.get(name).cl_buffer, buffer_mgr.get(name).mask_buffer
                op_fn = lambda d=data, b=buf: (
                    lambda q, wf: cl.enqueue_copy(
                        q, b, pad_tensor(d, buffer_mgr.get(name).spec.padded_shape), wait_for=wf
                    )
                )
                manager.create_node(
                    f"h2d_{name}", NodeType.TRANSFER, "Q_TRAINING", {name: AccessMode.EXCLUSIVE_WRITE}, op_fn()
                )
                if mask_buf:
                    mask_op = lambda d=np.ones(data.shape[0], dtype=SCALAR_NP_TYPE), b=mask_buf: (
                        lambda q, wf: cl.enqueue_copy(q, b, pad_tensor(d, (padded_batch_size,)), wait_for=wf)
                    )
                    manager.create_node(
                        f"h2d_{name}_mask",
                        NodeType.TRANSFER,
                        "Q_TRAINING",
                        {f"{name}:mask": AccessMode.EXCLUSIVE_WRITE},
                        mask_op(),
                    )
            for p in params:
                k.zero_gradients(p.grad, "Q_TRAINING")

            partial_nodes = defaultdict(list)
            fp_node = k.forward_pass("Q_TRAINING", 0, actual_batch_size)
            for c in range(num_chunks):
                param_offset = c * chunk_size
                current_chunk_size = min(chunk_size, NUM_EXITS - param_offset)
                outputs_uid = k.compute_chunk_outputs(
                    "Q_TRAINING", c, current_chunk_size, param_offset, padded_batch_size
                )
                manager.graph.add_edge(fp_node, outputs_uid)
                grads_uid = k.calculate_chunk_gradients(
                    "Q_TRAINING", c, current_chunk_size, param_offset, padded_batch_size
                )
                manager.graph.add_edge(fp_node, grads_uid)
                manager.graph.add_edge(outputs_uid, grads_uid)
                temps_uid = k.calculate_chunk_temp_gradients(
                    "Q_TRAINING", c, current_chunk_size, param_offset, padded_batch_size
                )
                manager.graph.add_edge(outputs_uid, temps_uid)
                for name, uid in [
                    ("Probs", outputs_uid),
                    ("Loss", outputs_uid),
                    ("Grad_H", grads_uid),
                    ("Grad_Temps", temps_uid),
                    ("Grad_Exit_W", grads_uid),
                    ("Grad_Exit_B", grads_uid),
                ]:
                    partial_nodes[name].append(uid)

            num_grad_h_items = padded_batch_size * HIDDEN_DIM
            transpose_uid = k.transpose_grad_h(
                "Q_TRAINING", "partial_grad_h_aos_out", "grad_h_soa_buf", NUM_EXITS, num_grad_h_items
            )
            for node_id in partial_nodes["Grad_H"]:
                manager.graph.add_edge(node_id, transpose_uid)

            aggregated_nodes = {}
            agg_grad_h = k.aggregate("Q_TRAINING", "grad_h_soa_buf", "final_grad_h_buf", num_grad_h_items, NUM_EXITS)
            manager.graph.add_edge(transpose_uid, agg_grad_h)
            aggregated_nodes["Grad_H"] = agg_grad_h
            buffers_to_agg = {
                "Probs": ("partial_probs_out", "final_probs_buf"),
                "Loss": ("partial_loss_out", "final_loss_buf"),
                "Grad_Temps": ("partial_grad_temps_out", "grad_temps"),
                "Grad_Exit_W": ("partial_grad_exit_w_out", "grad_exit_weights"),
                "Grad_Exit_B": ("partial_grad_exit_b_out", "grad_exit_biases"),
            }
            for name, (p_buf, f_buf) in buffers_to_agg.items():
                item_stride = int(np.prod(buffer_mgr.get(p_buf).spec.real_shape[1:]))
                agg_node = k.aggregate("Q_TRAINING", p_buf, f_buf, num_chunks, item_stride)
                for dep_node in partial_nodes[name]:
                    manager.graph.add_edge(dep_node, agg_node)
                aggregated_nodes[name] = agg_node

            partial_shared_grad_nodes = defaultdict(list)
            num_batch_stream_chunks = (actual_batch_size + BACKPROP_STREAM_CHUNK_SIZE - 1) // BACKPROP_STREAM_CHUNK_SIZE
            for bc in range(num_batch_stream_chunks):
                batch_offset = bc * BACKPROP_STREAM_CHUNK_SIZE
                current_batch_chunk_size = min(BACKPROP_STREAM_CHUNK_SIZE, actual_batch_size - batch_offset)
                sw_grad_uid = k.backprop_shared_weights_chunk("Q_TRAINING", bc, batch_offset, current_batch_chunk_size)
                manager.graph.add_edge(aggregated_nodes["Grad_H"], sw_grad_uid)
                partial_shared_grad_nodes["SW"].append(sw_grad_uid)
                sb_grad_uid = k.backprop_shared_biases_chunk("Q_TRAINING", bc, batch_offset, current_batch_chunk_size)
                manager.graph.add_edge(aggregated_nodes["Grad_H"], sb_grad_uid)
                partial_shared_grad_nodes["SB"].append(sb_grad_uid)

            item_stride_sw = int(np.prod(buffer_mgr.get("partial_grad_sw_out").spec.real_shape[1:]))
            agg_sw = k.aggregate(
                "Q_TRAINING", "partial_grad_sw_out", "grad_weights", num_batch_stream_chunks, item_stride_sw
            )
            for dep_node in partial_shared_grad_nodes["SW"]:
                manager.graph.add_edge(dep_node, agg_sw)
            item_stride_sb = int(np.prod(buffer_mgr.get("partial_grad_sb_out").spec.real_shape[1:]))
            agg_sb = k.aggregate(
                "Q_TRAINING", "partial_grad_sb_out", "grad_biases", num_batch_stream_chunks, item_stride_sb
            )
            for dep_node in partial_shared_grad_nodes["SB"]:
                manager.graph.add_edge(dep_node, agg_sb)

            final_training_nodes = []
            d2h_op = lambda v, b: (lambda q, wf: v.update_from_device(q, buffer_mgr.get(b).cl_buffer, wf))
            probs_d2h = manager.create_node(
                "d2h_probs",
                NodeType.TRANSFER,
                "Q_INFERENCE",
                {"final_probs_buf": AccessMode.SHARED_READ},
                d2h_op(probs_view, "final_probs_buf"),
            )
            manager.graph.add_edge(aggregated_nodes["Probs"], probs_d2h)
            for p, dep_node in [
                (Parameter("weights"), agg_sw),
                (Parameter("biases"), agg_sb),
                (Parameter("exit_weights"), aggregated_nodes["Grad_Exit_W"]),
                (Parameter("exit_biases"), aggregated_nodes["Grad_Exit_B"]),
                (Parameter("temps"), aggregated_nodes["Grad_Temps"]),
            ]:
                update_node = k.adam_update(p, "Q_TRAINING")
                manager.graph.add_edge(dep_node, update_node)
                if p.name == "temps":
                    clamp_node = k.clamp_temperatures("Q_TRAINING")
                    manager.graph.add_edge(update_node, clamp_node)
                    final_training_nodes.append(clamp_node)
                else:
                    final_training_nodes.append(update_node)

            loss_scalar_node = k.aggregate(
                "Q_TRAINING", "final_loss_buf", "final_scalar_loss_buf", padded_batch_size, 1, mode=0
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

            inference_event = manager.create_host_notification_point("inference_sync", dependencies=[probs_d2h])
            final_batch_event = manager.create_host_notification_point(
                "training_sync", dependencies=final_training_nodes
            )

            manager.commit()
            inference_event.wait()
            valid_probs = probs_view.valid_slice[:actual_batch_size]
            predicted_classes = np.argmax(valid_probs, axis=1)
            correct_predictions += np.sum(predicted_classes == y_batch)
            total_samples += actual_batch_size
            final_batch_event.wait()
            total_batch_loss = loss_view.valid_slice[0]
            epoch_loss += total_batch_loss
            global_step += 1

        avg_epoch_loss = epoch_loss / total_samples if total_samples > 0 else 0
        train_acc = correct_predictions / total_samples if total_samples > 0 else 0
        print(f"Epoch {epoch+1:3d}/{EPOCHS} | Loss: {avg_epoch_loss:.4f} | Accuracy: {train_acc:.2%}")

    print("\nTraining finished.")


if __name__ == "__main__":
    main()
