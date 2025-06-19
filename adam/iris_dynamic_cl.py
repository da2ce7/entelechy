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
NUM_EXITS: int = 40
BATCH_SIZE: int = 130
EPOCHS: int = 100

# --- Hyperparameters & Runtime Tuning Constants ---
LEARNING_RATE: float = 0.001
ADAM_BETA1: float = 0.9
ADAM_BETA2: float = 0.999
EPSILON: float = 1.0e-8
MIN_TEMP: float = 1.0e-3
MAX_TEMP: float = 10.0
TIER3_REDUCE_ITEMS_PER_GROUP: int = 1024  # Work-group granularity for Tier 3 ensemble reduction.

# --- Compile-time Kernel Constants ---
# Passed via -D flags to the OpenCL compiler.
MAX_EXITS_ENSEMBLE: int = 64  # Threshold for using register-based (Tier 1) ensemble reduction.
C_TILE_SIZE: int = 32  # Tile size used in some compute kernels.

# --- File Configuration ---
KERNEL_DIR: str = "kernels"
CL_HEADERS: List[str] = ["kernels.cl.h"]
CL_SOURCES: List[str] = [
    "network_operations.cl.c",
    "autograd.cl.c",
    "parameter_optim.cl.c",
]


### UTILITY FUNCTIONS
def load_and_concatenate_kernels(kernel_dir: str, headers: List[str], sources: List[str]) -> str:
    """
    Loads and concatenates OpenCL kernel source files in order.
    This preprocesses includes by prepending header content, ensuring a correct
    build order without relying on the OpenCL compiler's `#include` for these files.
    """
    full_source_parts = []
    all_files_in_order = headers + sources
    if not os.path.isdir(kernel_dir):
        raise IOError(f"Kernel directory '{kernel_dir}' not found.")
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
    """He initialization, assumes shape[0] or shape[1] is fan_in."""
    fan_in = shape[0] if len(shape) == 2 else shape[1]
    scale = np.sqrt(2.0 / fan_in)
    return np.random.normal(0, scale, shape).astype(SCALAR_NP_TYPE)


def he_init_simd_major(
    initial_weights: np.ndarray, real_shape: Tuple[int, int], padded_shape: Tuple[int, int, int], simd_width: int
) -> np.ndarray:
    """
    Transforms standard layout weights (input_dim, hidden_dim) to SIMD-major
    (hidden_blocks, input_dim, simd_width) for the `forward_pass` kernel.
    """
    input_dim, hidden_dim = real_shape
    padded_hidden_dim = pad_to_multiple(hidden_dim, simd_width)

    padded_weights = np.zeros((input_dim, padded_hidden_dim), dtype=SCALAR_NP_TYPE)
    padded_weights[:, :hidden_dim] = initial_weights

    simd_major_weights = padded_weights.reshape(input_dim, padded_hidden_dim // simd_width, simd_width).transpose(
        1, 0, 2
    )
    assert (
        simd_major_weights.shape == padded_shape
    ), f"SIMD-major weight shape mismatch: {simd_major_weights.shape}, expected {padded_shape}"
    return simd_major_weights


def select_simd_width(device: cl.Device) -> int:
    """Heuristic SIMD width based on device vendor (e.g., warp/wavefront size)."""
    if "Intel" in device.vendor:
        return 16
    if "NVIDIA" in device.vendor:
        return 32
    if "AMD" in device.vendor:
        return 64
    return 8


def validate_targets(y: np.ndarray, num_classes: int):
    if not np.all(np.logical_and(y >= 0, y < num_classes)):
        raise ValueError(f"Target labels must be integers between 0 and {num_classes - 1}")
    if not np.issubdtype(y.dtype, np.integer):
        raise ValueError("Target labels must be integers")


def pad_tensor(data: np.ndarray, padded_shape: Tuple[int, ...]) -> np.ndarray:
    if data.shape == padded_shape:
        return data
    padded = np.zeros(padded_shape, dtype=data.dtype)
    original_slices = tuple(slice(0, d) for d in data.shape)
    padded[original_slices] = data
    return padded


### CORE ABSTRACTIONS
class BufferRole(Enum):
    # Parameters & optimizer states
    WEIGHTS, BIAS, EXIT_WEIGHTS, EXIT_BIAS, TEMPERATURES, GRADIENT, ADAM_M1, ADAM_M2 = [auto() for _ in range(8)]
    # Data & activations
    (
        INPUT_DATA,
        TARGETS,
        HIDDEN_ACTIVATION,
        UNSCALED_LOGITS,
        EXIT_PROBABILITIES,
        ENSEMBLE_WEIGHTS,
        ENSEMBLE_PROBABILITIES,
    ) = [auto() for _ in range(7)]
    # Intermediate/scratch buffers
    GRAD_CONTRIBUTIONS, PARTIAL_LOSS_REDUCTION, FINAL_LOSS, INTERMEDIATE = [auto() for _ in range(4)]


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
    mask_buffer: Optional[cl.Buffer] = None  # For batch items, marks valid (non-padded) entries.


class BufferManager:
    def __init__(self, context: cl.Context, device: cl.Device):
        self.context = context
        self.padding_ctx = PaddingContext.from_device(device)
        self.buffers: Dict[str, Buffer] = {}

    def _get_padded_shape(self, role: BufferRole, real_shape: Tuple[int, ...]) -> Tuple[int, ...]:
        """Determines padded shape based on buffer role and device SIMD width."""
        sw = self.padding_ctx.simd_width
        padded_batch = pad_to_multiple(BATCH_SIZE, sw)

        # `weights` for forward_pass: SIMD-major (hidden_blocks, input_dim, simd_width)
        if role == BufferRole.WEIGHTS and len(real_shape) == 2:
            return (pad_to_multiple(real_shape[1], sw) // sw, real_shape[0], sw)
        # Standard 2D matrices (e.g.,`weights_standard`, gradients): pad both dims to SIMD multiple.
        if (
            role in [BufferRole.WEIGHTS, BufferRole.GRADIENT, BufferRole.ADAM_M1, BufferRole.ADAM_M2]
            and len(real_shape) == 2
        ):
            return (pad_to_multiple(real_shape[0], sw), pad_to_multiple(real_shape[1], sw))
        # 3D matrices (e.g., exit weights/grads): pad last two dims.
        if (
            role in [BufferRole.EXIT_WEIGHTS, BufferRole.GRADIENT, BufferRole.ADAM_M1, BufferRole.ADAM_M2]
            and len(real_shape) == 3
        ):
            return (real_shape[0], pad_to_multiple(real_shape[1], sw), pad_to_multiple(real_shape[2], sw))
        # 1D/2D vectors (biases, temps): pad feature dimension.
        if role in [
            BufferRole.BIAS,
            BufferRole.EXIT_BIAS,
            BufferRole.TEMPERATURES,
            BufferRole.GRADIENT,
            BufferRole.ADAM_M1,
            BufferRole.ADAM_M2,
        ]:
            return (
                (pad_to_multiple(real_shape[0], sw),)
                if len(real_shape) == 1
                else (real_shape[0], pad_to_multiple(real_shape[1], sw))
            )
        # Activations: pad batch and feature dim.
        if role == BufferRole.HIDDEN_ACTIVATION:
            return (padded_batch, pad_to_multiple(real_shape[1], sw))
        # Multi-dimensional intermediate/output tensors: pad batch and last feature dim.
        if (
            role
            in [
                BufferRole.UNSCALED_LOGITS,
                BufferRole.EXIT_PROBABILITIES,
                BufferRole.GRAD_CONTRIBUTIONS,
                BufferRole.INTERMEDIATE,
            ]
            and len(real_shape) > 1
        ):
            last_dim_padded = pad_to_multiple(real_shape[-1], sw) if len(real_shape) > 2 else real_shape[-1]
            return (real_shape[0], padded_batch, last_dim_padded)
        # Input/target/ensemble type buffers: pad batch dim.
        if role in [BufferRole.INPUT_DATA, BufferRole.TARGETS, BufferRole.ENSEMBLE_WEIGHTS] or (
            role == BufferRole.INTERMEDIATE and len(real_shape) == 2
        ):
            return (padded_batch, *real_shape[1:])
        if role in [BufferRole.ENSEMBLE_PROBABILITIES]:
            return (padded_batch, pad_to_multiple(real_shape[1], sw))
        return real_shape  # Default: no special padding.

    def create_buffer(
        self,
        name: str,
        role: BufferRole,
        real_shape: Tuple[int, ...],
        dtype: np.dtype,
        init_data: Optional[np.ndarray] = None,
    ) -> Buffer:
        """Creates or retrieves an OpenCL buffer, handling padding and optional initialization."""
        if name in self.buffers:  # important for temporary buffers in multi-stage kernels
            return self.buffers[name]

        padded_shape = self._get_padded_shape(role, real_shape)
        size = int(np.prod(padded_shape) * np.dtype(dtype).itemsize) if np.prod(padded_shape) > 0 else 0
        cl_buf = cl.Buffer(self.context, cl.mem_flags.READ_WRITE, size=max(size, 4))  # Min size for safety
        mask_buf = None

        if role in [BufferRole.INPUT_DATA, BufferRole.TARGETS, BufferRole.HIDDEN_ACTIVATION]:
            mask_padded_shape = (padded_shape[0],)
            mask_size = int(mask_padded_shape[0] * SCALAR_SIZE)
            mask_buf = cl.Buffer(self.context, cl.mem_flags.READ_WRITE, size=max(mask_size, 4))

        buffer_obj = Buffer(BufferSpec(name, role, real_shape, padded_shape, dtype), cl_buf, mask_buf)
        self.buffers[name] = buffer_obj

        if init_data is not None:
            padded_data = pad_tensor(init_data, padded_shape)  # Pad host data before upload
            cl.enqueue_copy(cl.CommandQueue(self.context), cl_buf, padded_data).wait()
        return buffer_obj

    def get(self, name: str) -> Buffer:
        return self.buffers[name]


class BatchPadder:
    """Handles padding of input X and targets y for a batch."""

    def __init__(self, input_spec: BufferSpec, target_spec: BufferSpec):
        self.input_spec, self.target_spec = input_spec, target_spec

    def pad_batch(self, X: np.ndarray, y_true: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        batch_size = X.shape[0]
        X_padded = pad_tensor(X, self.input_spec.padded_shape)
        y_padded = pad_tensor(y_true, self.target_spec.padded_shape)
        mask_data = np.ones(batch_size, dtype=SCALAR_NP_TYPE)  # 1s for valid samples
        mask_padded = pad_tensor(mask_data, (self.input_spec.padded_shape[0],))
        return X_padded, y_padded, mask_padded


class HostView:
    """NumPy array view for host access to OpenCL buffer data."""

    def __init__(self, buffer: Buffer):
        self.spec = buffer.spec
        self.host_data = np.empty(buffer.spec.padded_shape, dtype=buffer.spec.dtype)  # Pre-allocate host memory

    def update_from_device(self, queue: cl.CommandQueue, cl_buffer: cl.Buffer, wait_for=None):
        return cl.enqueue_copy(queue, self.host_data, cl_buffer, wait_for=wait_for or [])

    @property
    def valid_slice(self):
        """Returns the slice corresponding to 'real' (unpadded) dimensions."""
        slicing = tuple(slice(0, dim) for dim in self.spec.real_shape)
        return self.host_data[slicing]


### WORK MANAGER
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
    params: Dict = field(default_factory=dict)
    event: Optional[cl.Event] = None


class WorkManager:
    """Manages a DAG of operations for sequenced execution on OpenCL queues."""

    def __init__(self, context: cl.Context):
        self.context = context
        self.properties = cl.command_queue_properties.OUT_OF_ORDER_EXEC_MODE_ENABLE
        self.graph = nx.DiGraph()
        self.queues: Dict[str, cl.CommandQueue] = {  # Potentially for higher priority/concurrent tasks
            "Q_TRAINING": cl.CommandQueue(context, self.properties),
            "Q_INFERENCE": cl.CommandQueue(context, self.properties),
        }
        self.node_id_counter: int = 0
        self.last_writer: Dict[str, int] = {}  # Tracks last node UID writing to a buffer for dependency building
        self.notification_events: Dict[int, cl.UserEvent] = {}

    def _get_uid(self) -> int:
        uid = self.node_id_counter
        self.node_id_counter += 1
        return uid

    def create_node(
        self,
        name: str,
        node_type: NodeType,
        queue_name: str,
        access_map: Dict[str, AccessMode],
        op_fn: Callable,
        params: Dict = {},
    ) -> int:
        """Creates a graph node, adding dependencies based on buffer access (RAW, WAW, WAR implicitly handled by last_writer)."""
        uid = self._get_uid()
        node = ExecutionNode(uid, name, node_type, queue_name, access_map, op_fn, params)
        self.graph.add_node(uid, node=node)
        for res, mode in access_map.items():
            if res in self.last_writer:  # Depend on the last node that wrote to this resource
                self.graph.add_edge(self.last_writer[res], uid)
            if mode in ACCESS_WRITE_MODES:  # This node is now the last writer
                self.last_writer[res] = uid
        return uid

    def create_host_notification_point(self, name: str, dependencies: List[int]) -> cl.UserEvent:
        """Creates a sync node that signals the host (via cl.UserEvent) upon completion of its dependencies."""
        uid = self._get_uid()
        user_event = cl.UserEvent(self.context)
        self.notification_events[uid] = user_event

        def _enqueue_marker_with_callback(queue, wait_for):  # Enqueue marker that triggers callback on completion
            marker = cl.enqueue_marker(queue, wait_for=wait_for)

            def _set_event_status_complete(evt, status):
                if status == cl.command_execution_status.COMPLETE:
                    user_event.set_status(cl.command_execution_status.COMPLETE)

            marker.set_callback(cl.command_execution_status.COMPLETE, _set_event_status_complete)
            return marker

        # Place sync node on the queue of its first dependency, or default.
        dep_queue_name = self.graph.nodes[dependencies[0]]["node"].queue_name if dependencies else "Q_TRAINING"
        node = ExecutionNode(uid, name, NodeType.SYNC, dep_queue_name, {}, _enqueue_marker_with_callback)
        self.graph.add_node(uid, node=node)
        for dep_uid in dependencies:
            self.graph.add_edge(dep_uid, uid)
        return user_event

    def commit(self):
        """Enqueues all operations in topological order, respecting dependencies."""
        ordered_nodes = list(nx.topological_sort(self.graph))
        completion_events: Dict[int, cl.Event] = {}
        for uid in ordered_nodes:
            node = self.graph.nodes[uid]["node"]
            queue = self.queues[node.queue_name]
            wait_for = [
                completion_events[p_uid] for p_uid in self.graph.predecessors(uid) if p_uid in completion_events
            ]
            event = node.operation_fn(queue, wait_for)
            if event:
                completion_events[uid] = event

    def reset(self):
        self.graph.clear()
        self.node_id_counter = 0
        self.last_writer.clear()
        self.notification_events.clear()


### KERNEL WRAPPER
class KernelWrapper:
    """Encapsulates OpenCL kernel calls, managing parameters and graph node creation."""

    def __init__(self, program: cl.Program, manager: WorkManager, buffer_manager: BufferManager, global_step: int):
        self.p, self.m, self.b = program, manager, buffer_manager
        # Pre-fetch/calculate frequently used values
        self.padded_input_dim = self.b.get("input_buf").spec.padded_shape[1]
        _hp = self.b.get("hidden_buf").spec.padded_shape
        self.padded_hidden_dim, self.padded_batch_size = _hp[1], _hp[0]
        self.padded_output_classes = self.b.get("exit_weights").spec.padded_shape[2]
        self.beta1_t, self.beta2_t = SCALAR_NP_TYPE(ADAM_BETA1**global_step), SCALAR_NP_TYPE(
            ADAM_BETA2**global_step
        )  # For Adam bias correction

    def _create_kernel_node(self, name, global_size, local_size, queue_name, access_map, *args):
        op_fn = lambda q, wf: getattr(self.p, name)(q, global_size, local_size, *args, wait_for=wf)
        return self.m.create_node(name, NodeType.COMPUTE, queue_name, access_map, op_fn)

    def zero_gradients(self, grad_buffer_name: str, queue_name: str) -> int:
        """Fills a gradient buffer with zeros."""
        buffer = self.b.get(grad_buffer_name)
        op_fn = lambda q, wf: cl.enqueue_fill_buffer(
            q, buffer.cl_buffer, SCALAR_NP_TYPE(0), 0, buffer.cl_buffer.size, wait_for=wf
        )
        return self.m.create_node(
            f"zero_{grad_buffer_name}",
            NodeType.COMPUTE,
            queue_name,
            {grad_buffer_name: AccessMode.EXCLUSIVE_WRITE},
            op_fn,
        )

    def forward_pass(self, queue_name: str) -> int:
        """Hidden layer: hidden = relu(input @ weights_simd_major + biases). Uses tiled matrix multiplication."""
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
            cl.LocalMemory(2 * l[0] * SCALAR_SIZE),  # Local memory for input tile and weights tile
            self.b.get("input_buf").cl_buffer,
            self.b.get("input_buf").mask_buffer,
            self.b.get("weights").cl_buffer,
            self.b.get("biases").cl_buffer,
            self.b.get("hidden_buf").cl_buffer,
            self.b.get("hidden_buf").mask_buffer,
            np.int32(self.padded_input_dim),
            np.int32(self.padded_hidden_dim),
        )
        return self._create_kernel_node("forward_pass", g, l, queue_name, acc, *args)

    def compute_all_exits(self, queue_name: str) -> int:
        """Computes logits, probabilities, and per-exit XE losses for all early exits."""
        g, l = (NUM_EXITS, self.padded_batch_size), None
        acc = {
            "hidden_buf": AccessMode.SHARED_READ,
            "exit_weights": AccessMode.SHARED_READ,
            "exit_biases": AccessMode.SHARED_READ,
            "temps": AccessMode.SHARED_READ,
            "targets_buf": AccessMode.SHARED_READ,
            "unscaled_logits_buf": AccessMode.EXCLUSIVE_WRITE,
            "exit_probs_buf": AccessMode.EXCLUSIVE_WRITE,
            "per_exit_losses_buf": AccessMode.EXCLUSIVE_WRITE,
        }
        args = (
            self.b.get("hidden_buf").cl_buffer,
            self.b.get("hidden_buf").mask_buffer,
            self.b.get("exit_weights").cl_buffer,
            self.b.get("exit_biases").cl_buffer,
            self.b.get("temps").cl_buffer,
            self.b.get("targets_buf").cl_buffer,
            self.b.get("targets_buf").mask_buffer,
            self.b.get("unscaled_logits_buf").cl_buffer,
            self.b.get("exit_probs_buf").cl_buffer,
            self.b.get("per_exit_losses_buf").cl_buffer,
            np.int32(self.padded_batch_size),
            np.int32(self.padded_hidden_dim),
            np.int32(self.padded_output_classes),
            np.int32(NUM_EXITS),
        )
        return self._create_kernel_node("compute_all_exits", g, l, queue_name, acc, *args)

    def ensemble_weights_tier1_reg_reduce(self, queue_name: str) -> int:
        """(Tier 1) Ensemble weights via register reduction (for NUM_EXITS <= MAX_EXITS_ENSEMBLE)."""
        g, l = (self.padded_batch_size,), None
        acc = {
            "exit_probs_buf": AccessMode.SHARED_READ,
            "targets_buf": AccessMode.SHARED_READ,
            "ensemble_weights_buf": AccessMode.EXCLUSIVE_WRITE,
        }
        args = (
            self.b.get("exit_probs_buf").cl_buffer,
            self.b.get("targets_buf").mask_buffer,
            self.b.get("ensemble_weights_buf").cl_buffer,
            np.int32(self.padded_batch_size),
            np.int32(self.padded_output_classes),
            np.int32(NUM_EXITS),
        )
        return self._create_kernel_node("ensemble_weights_reg_reduce", g, l, queue_name, acc, *args)

    def ensemble_weights_tier2_local_reduce(self, queue_name: str) -> int:
        """(Tier 2) Ensemble weights via local memory reduction."""
        ls = 256
        g, l = (self.padded_batch_size * ls,), (ls,)
        acc = {"exit_probs_buf": AccessMode.SHARED_READ, "ensemble_weights_buf": AccessMode.EXCLUSIVE_WRITE}
        args = (
            cl.LocalMemory(ls * SCALAR_SIZE),
            self.b.get("exit_probs_buf").cl_buffer,
            self.b.get("ensemble_weights_buf").cl_buffer,
            np.int32(NUM_EXITS),
            np.int32(self.padded_output_classes),
            np.int32(self.padded_batch_size),
        )
        return self._create_kernel_node("ensemble_weights_local_reduce", g, l, queue_name, acc, *args)

    def ensemble_weights_tier3_map_reduce(self, queue_name: str) -> int:
        """(Tier 3) Ensemble weights: Map (exp_conf) -> Reduce (partial_sums) -> Finalize (normalize)."""
        num_chunks = (NUM_EXITS + TIER3_REDUCE_ITEMS_PER_GROUP - 1) // TIER3_REDUCE_ITEMS_PER_GROUP
        self.b.create_buffer("temp_exp_conf_buf", BufferRole.INTERMEDIATE, (BATCH_SIZE, NUM_EXITS), SCALAR_NP_TYPE)
        self.b.create_buffer("temp_partial_sums_buf", BufferRole.INTERMEDIATE, (BATCH_SIZE, num_chunks), SCALAR_NP_TYPE)
        # Map stage
        g_map, l_map = (self.padded_batch_size, NUM_EXITS), None
        acc_map = {
            "exit_probs_buf": AccessMode.SHARED_READ,
            "targets_buf": AccessMode.SHARED_READ,
            "temp_exp_conf_buf": AccessMode.EXCLUSIVE_WRITE,
        }
        args_map = (
            self.b.get("exit_probs_buf").cl_buffer,
            self.b.get("targets_buf").mask_buffer,
            self.b.get("temp_exp_conf_buf").cl_buffer,
            np.int32(self.padded_batch_size),
            np.int32(NUM_EXITS),
            np.int32(self.padded_output_classes),
        )
        map_node = self._create_kernel_node(
            "ensemble_weights_map_exp_conf", g_map, l_map, queue_name, acc_map, *args_map
        )
        # Reduce stage
        rls = 256
        g_reduce, l_reduce = (self.padded_batch_size * rls, num_chunks), (rls, 1)
        acc_reduce = {"temp_exp_conf_buf": AccessMode.SHARED_READ, "temp_partial_sums_buf": AccessMode.EXCLUSIVE_WRITE}
        args_reduce = (
            cl.LocalMemory(rls * SCALAR_SIZE),
            self.b.get("temp_exp_conf_buf").cl_buffer,
            self.b.get("temp_partial_sums_buf").cl_buffer,
            np.int32(NUM_EXITS),
        )
        reduce_node = self._create_kernel_node(
            "reduce_partial_sums", g_reduce, l_reduce, queue_name, acc_reduce, *args_reduce
        )
        # Finalize stage
        g_fin, l_fin = (self.padded_batch_size,), None
        acc_fin = {
            "temp_exp_conf_buf": AccessMode.SHARED_READ,
            "temp_partial_sums_buf": AccessMode.SHARED_READ,
            "targets_buf": AccessMode.SHARED_READ,
            "ensemble_weights_buf": AccessMode.EXCLUSIVE_WRITE,
        }
        args_fin = (
            self.b.get("temp_exp_conf_buf").cl_buffer,
            self.b.get("temp_partial_sums_buf").cl_buffer,
            self.b.get("targets_buf").mask_buffer,
            self.b.get("ensemble_weights_buf").cl_buffer,
            np.int32(self.padded_batch_size),
            np.int32(NUM_EXITS),
            np.int32(num_chunks),
        )
        finalize_node = self._create_kernel_node(
            "ensemble_weights_finalize", g_fin, l_fin, queue_name, acc_fin, *args_fin
        )
        self.m.graph.add_edge(map_node, reduce_node)
        self.m.graph.add_edge(reduce_node, finalize_node)
        return finalize_node

    def blend_ensemble_probabilities(self, queue_name: str) -> int:
        """Blends exit probabilities using calculated ensemble_weights."""
        g, l = (self.padded_batch_size, self.padded_output_classes), None
        acc = {
            "exit_probs_buf": AccessMode.SHARED_READ,
            "ensemble_weights_buf": AccessMode.SHARED_READ,
            "targets_buf": AccessMode.SHARED_READ,
            "ensemble_probs_buf": AccessMode.EXCLUSIVE_WRITE,
        }
        args = (
            self.b.get("exit_probs_buf").cl_buffer,
            self.b.get("ensemble_weights_buf").cl_buffer,
            self.b.get("targets_buf").mask_buffer,
            self.b.get("ensemble_probs_buf").cl_buffer,
            np.int32(self.padded_batch_size),
            np.int32(self.padded_output_classes),
            np.int32(NUM_EXITS),
        )
        return self._create_kernel_node("blend_ensemble_probabilities", g, l, queue_name, acc, *args)

    def calculate_losses(self, queue_name: str) -> int:
        """Two-stage reduction for batch loss: Partial sums -> Final aggregation."""
        # Stage 1: Partial XE losses, summed into groups.
        g1 = (self.padded_batch_size,)
        l1_val = (1 << (g1[0] - 1).bit_length() >> 1) if g1[0] > 1 else 1
        l1 = (min(256, l1_val if l1_val > 0 else 1),)
        num_groups = math.ceil(g1[0] / l1[0]) if l1[0] > 0 else 0
        self.b.create_buffer(
            "partial_loss_buf", BufferRole.PARTIAL_LOSS_REDUCTION, (num_groups,), np.float32
        )  # Loss sum target
        lm1 = cl.LocalMemory(l1[0] * np.dtype(np.float32).itemsize)
        acc1 = {
            "ensemble_probs_buf": AccessMode.SHARED_READ,
            "targets_buf": AccessMode.SHARED_READ,
            "partial_loss_buf": AccessMode.EXCLUSIVE_WRITE,
        }
        args1 = (
            lm1,
            self.b.get("ensemble_probs_buf").cl_buffer,
            self.b.get("targets_buf").mask_buffer,
            self.b.get("targets_buf").cl_buffer,
            self.b.get("partial_loss_buf").cl_buffer,
            np.int32(self.padded_batch_size),
            np.int32(self.padded_output_classes),
        )
        partial_node = self._create_kernel_node("calculate_partial_losses", g1, l1, queue_name, acc1, *args1)
        # Stage 2: Aggregate partial sums to final loss value.
        l2v_val = (1 << (num_groups - 1).bit_length() >> 1) if num_groups > 1 else 1
        l2v = min(256, l2v_val if l2v_val > 0 else 1)
        g2, l2 = (l2v,), (l2v,)
        lm2 = cl.LocalMemory(l2[0] * np.dtype(np.float32).itemsize)
        acc2 = {"partial_loss_buf": AccessMode.SHARED_READ, "final_loss_buf": AccessMode.EXCLUSIVE_WRITE}
        args2 = (
            lm2,
            self.b.get("partial_loss_buf").cl_buffer,
            self.b.get("final_loss_buf").cl_buffer,
            np.int32(num_groups),
        )
        final_node = self._create_kernel_node("aggregate_partial_losses", g2, l2, queue_name, acc2, *args2)
        self.m.graph.add_edge(partial_node, final_node)
        return final_node

    def calculate_exit_gradients(self, queue_name: str) -> int:
        """Exit gradients: dL/dW_exit, dL/dB_exit, and accumulates dL/dH_contributions from each exit."""
        ls = 256
        g, l = (NUM_EXITS * ls, self.padded_hidden_dim), (ls, 1)  # Reduction over batch dim (ls)
        acc = {
            "hidden_buf": AccessMode.SHARED_READ,
            "exit_probs_buf": AccessMode.SHARED_READ,
            "ensemble_weights_buf": AccessMode.SHARED_READ,
            "targets_buf": AccessMode.SHARED_READ,
            "exit_weights": AccessMode.SHARED_READ,  # Read for dL/dH
            "grad_exit_weights": AccessMode.EXCLUSIVE_UPDATE,
            "grad_exit_biases": AccessMode.EXCLUSIVE_UPDATE,
            "grad_hidden_contributions_buf": AccessMode.EXCLUSIVE_WRITE,
        }
        args = (
            cl.LocalMemory(ls * SCALAR_SIZE),
            cl.LocalMemory(ls * SCALAR_SIZE),
            self.b.get("hidden_buf").cl_buffer,
            self.b.get("exit_probs_buf").cl_buffer,
            self.b.get("ensemble_weights_buf").cl_buffer,
            self.b.get("targets_buf").cl_buffer,
            self.b.get("targets_buf").mask_buffer,
            self.b.get("exit_weights").cl_buffer,
            self.b.get("grad_exit_weights").cl_buffer,
            self.b.get("grad_exit_biases").cl_buffer,
            self.b.get("grad_hidden_contributions_buf").cl_buffer,
            np.int32(self.padded_batch_size),
            np.int32(self.padded_hidden_dim),
            np.int32(self.padded_output_classes),
            np.int32(NUM_EXITS),
        )
        return self._create_kernel_node("calculate_exit_gradients", g, l, queue_name, acc, *args)

    def aggregate_and_backprop_activation(self, queue_name: str) -> int:
        """Aggregates dL/dH_contributions from all exits and backprops through hidden layer's ReLU activation."""
        g, l = (self.padded_batch_size, self.padded_hidden_dim), None
        acc = {
            "hidden_buf": AccessMode.SHARED_READ,
            "grad_hidden_contributions_buf": AccessMode.SHARED_READ,
            "grad_pre_activation_buf": AccessMode.EXCLUSIVE_WRITE,
        }
        args = (
            self.b.get("hidden_buf").cl_buffer,
            self.b.get("hidden_buf").mask_buffer,
            self.b.get("grad_hidden_contributions_buf").cl_buffer,
            self.b.get("grad_pre_activation_buf").cl_buffer,
            np.int32(self.padded_batch_size),
            np.int32(self.padded_hidden_dim),
            np.int32(NUM_EXITS),
        )
        return self._create_kernel_node("aggregate_and_backprop_activation", g, l, queue_name, acc, *args)

    def calculate_dense_layer_gradients(self, queue_name: str) -> int:
        """Calculates dL/dW and dL/dB for the main dense layer."""
        ls = 256
        g, l = (self.padded_input_dim * ls, self.padded_hidden_dim), (ls, 1)  # Reduction over batch dim (ls)
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
            self.b.get("grad_weights").cl_buffer,  # Uses grad_weights (standard layout)
            self.b.get("grad_biases").cl_buffer,
            np.int32(self.padded_batch_size),
            np.int32(self.padded_input_dim),
            np.int32(self.padded_hidden_dim),
        )
        return self._create_kernel_node("calculate_dense_layer_gradients", g, l, queue_name, acc, *args)

    def backprop_input_gradient(self, queue_name: str) -> int:
        """Calculates dL/dX (gradient w.r.t. layer input) using `weights_standard`."""
        g, l = (self.padded_batch_size, self.padded_input_dim), None
        acc = {
            "grad_pre_activation_buf": AccessMode.SHARED_READ,
            "weights_standard": AccessMode.SHARED_READ,
            "grad_input_buf": AccessMode.EXCLUSIVE_WRITE,
        }
        args = (
            self.b.get("grad_pre_activation_buf").cl_buffer,
            self.b.get("weights_standard").cl_buffer,
            self.b.get("grad_input_buf").cl_buffer,
            np.int32(self.padded_batch_size),
            np.int32(self.padded_input_dim),
            np.int32(self.padded_hidden_dim),
        )
        return self._create_kernel_node("backprop_input_gradient", g, l, queue_name, acc, *args)

    def calculate_temp_gradients(self, queue_name: str) -> int:
        """Calculates dL/dT for temperature parameters."""
        ls = 256
        g, l = (NUM_EXITS * ls,), (ls,)  # Reduction over batch dim (ls)
        acc = {
            "unscaled_logits_buf": AccessMode.SHARED_READ,
            "exit_probs_buf": AccessMode.SHARED_READ,
            "ensemble_weights_buf": AccessMode.SHARED_READ,
            "temps": AccessMode.SHARED_READ,
            "targets_buf": AccessMode.SHARED_READ,
            "grad_temps": AccessMode.EXCLUSIVE_UPDATE,
        }
        args = (
            cl.LocalMemory(ls * SCALAR_SIZE),
            self.b.get("unscaled_logits_buf").cl_buffer,
            self.b.get("exit_probs_buf").cl_buffer,
            self.b.get("ensemble_weights_buf").cl_buffer,
            self.b.get("temps").cl_buffer,
            self.b.get("targets_buf").cl_buffer,
            self.b.get("targets_buf").mask_buffer,
            self.b.get("grad_temps").cl_buffer,
            np.int32(self.padded_batch_size),
            np.int32(self.padded_output_classes),
            np.int32(NUM_EXITS),
        )
        return self._create_kernel_node("calculate_temp_gradients", g, l, queue_name, acc, *args)

    def adam_update(self, param: Parameter, queue_name: str) -> int:
        """Adam optimizer update for a given parameter (element-wise)."""
        tp = int(np.prod(self.b.get(param.value).spec.padded_shape))
        g, l = (tp,), None
        acc = {
            param.grad: AccessMode.SHARED_READ,
            param.value: AccessMode.EXCLUSIVE_UPDATE,
            param.m1: AccessMode.EXCLUSIVE_UPDATE,
            param.m2: AccessMode.EXCLUSIVE_UPDATE,
        }
        args = (
            self.b.get(param.grad).cl_buffer,
            SCALAR_NP_TYPE(ADAM_BETA1),
            SCALAR_NP_TYPE(ADAM_BETA2),
            self.beta1_t,
            self.beta2_t,
            SCALAR_NP_TYPE(LEARNING_RATE),
            SCALAR_NP_TYPE(EPSILON),
            self.b.get(param.value).cl_buffer,
            self.b.get(param.m1).cl_buffer,
            self.b.get(param.m2).cl_buffer,
            np.int32(tp),
        )
        return self._create_kernel_node("adam_update", g, l, queue_name, acc, *args)

    def clamp_temperatures(self, queue_name: str) -> int:
        """Clamps temperatures to [MIN_TEMP, MAX_TEMP]."""
        g, l = (NUM_EXITS,), None
        acc = {"temps": AccessMode.EXCLUSIVE_UPDATE}
        args = (self.b.get("temps").cl_buffer, SCALAR_NP_TYPE(MIN_TEMP), SCALAR_NP_TYPE(MAX_TEMP), np.int32(NUM_EXITS))
        return self._create_kernel_node("clamp_temperatures", g, l, queue_name, acc, *args)


### MAIN EXECUTION ###
# 1. Setup
ctx = cl.create_some_context(interactive=False)
device = ctx.devices[0]
print(f"Using device: {device.name} from vendor: {device.vendor}")
manager = WorkManager(ctx)
buffer_mgr = BufferManager(ctx, device)
if SCALAR_TYPE == "half" and "cl_khr_fp16" not in device.extensions:
    raise RuntimeError("Device does not support FP16 (cl_khr_fp16 extension not found)")

# 2. Buffer Creation & Parameter Registration
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
    if p.name == "weights":  # Special handling for main 'weights'
        # `weights_standard`: standard layout (I,H) for some gradient calculations.
        init_data_std = init_fn(shape)
        buffer_mgr.create_buffer("weights_standard", role, shape, SCALAR_NP_TYPE, init_data=init_data_std)
        # `weights` (p.value): SIMD-major layout for optimized forward_pass.
        sw = buffer_mgr.padding_ctx.simd_width
        ps_simd = buffer_mgr._get_padded_shape(role, shape)
        init_data_simd = he_init_simd_major(init_data_std, shape, ps_simd, sw)
        cl_buf_simd = cl.Buffer(buffer_mgr.context, cl.mem_flags.READ_WRITE, size=init_data_simd.nbytes)
        spec_simd = BufferSpec(p.value, role, shape, ps_simd, SCALAR_NP_TYPE)
        buffer_mgr.buffers[p.value] = Buffer(spec_simd, cl_buf_simd)
        cl.enqueue_copy(cl.CommandQueue(buffer_mgr.context), cl_buf_simd, init_data_simd).wait()
        # Gradients for 'weights' use the standard layout matching 'weights_standard'.
        buffer_mgr.create_buffer(p.grad, BufferRole.GRADIENT, shape, SCALAR_NP_TYPE)
        buffer_mgr.create_buffer(p.m1, BufferRole.ADAM_M1, shape, SCALAR_NP_TYPE)
        buffer_mgr.create_buffer(p.m2, BufferRole.ADAM_M2, shape, SCALAR_NP_TYPE)
    else:  # Other parameters
        buffer_mgr.create_buffer(p.value, role, shape, SCALAR_NP_TYPE, init_data=init_fn(shape))
        buffer_mgr.create_buffer(p.grad, BufferRole.GRADIENT, shape, SCALAR_NP_TYPE)
        buffer_mgr.create_buffer(p.m1, BufferRole.ADAM_M1, shape, SCALAR_NP_TYPE)
        buffer_mgr.create_buffer(p.m2, BufferRole.ADAM_M2, shape, SCALAR_NP_TYPE)

data_buffer_specs = {
    "input_buf": (BufferRole.INPUT_DATA, (BATCH_SIZE, INPUT_DIM), SCALAR_NP_TYPE),
    "targets_buf": (BufferRole.TARGETS, (BATCH_SIZE,), np.int32),
    "hidden_buf": (BufferRole.HIDDEN_ACTIVATION, (BATCH_SIZE, HIDDEN_DIM), SCALAR_NP_TYPE),
    "unscaled_logits_buf": (BufferRole.UNSCALED_LOGITS, (NUM_EXITS, BATCH_SIZE, OUTPUT_CLASSES), SCALAR_NP_TYPE),
    "exit_probs_buf": (BufferRole.EXIT_PROBABILITIES, (NUM_EXITS, BATCH_SIZE, OUTPUT_CLASSES), SCALAR_NP_TYPE),
    "per_exit_losses_buf": (BufferRole.INTERMEDIATE, (NUM_EXITS, BATCH_SIZE, 1), SCALAR_NP_TYPE),
    "ensemble_weights_buf": (BufferRole.ENSEMBLE_WEIGHTS, (BATCH_SIZE, NUM_EXITS), SCALAR_NP_TYPE),
    "ensemble_probs_buf": (BufferRole.ENSEMBLE_PROBABILITIES, (BATCH_SIZE, OUTPUT_CLASSES), SCALAR_NP_TYPE),
    "grad_hidden_contributions_buf": (
        BufferRole.GRAD_CONTRIBUTIONS,
        (NUM_EXITS, BATCH_SIZE, HIDDEN_DIM),
        SCALAR_NP_TYPE,
    ),
    "grad_pre_activation_buf": (BufferRole.INTERMEDIATE, (BATCH_SIZE, HIDDEN_DIM), SCALAR_NP_TYPE),
    "grad_input_buf": (BufferRole.INTERMEDIATE, (BATCH_SIZE, INPUT_DIM), SCALAR_NP_TYPE),
    "final_loss_buf": (BufferRole.FINAL_LOSS, (1,), np.float32),
}  # Batch loss often FP32 for precision
for name, (role, shape, dtype) in data_buffer_specs.items():
    buffer_mgr.create_buffer(name, role, shape, dtype)

# 3. Data & Kernel Prep
X, y = load_iris(return_X_y=True)
X_normalized = StandardScaler().fit_transform(X).astype(SCALAR_NP_TYPE)
y_true = y.astype(np.int32)
validate_targets(y_true, OUTPUT_CLASSES)
batch_padder = BatchPadder(buffer_mgr.get("input_buf").spec, buffer_mgr.get("targets_buf").spec)
kernel_src = load_and_concatenate_kernels(KERNEL_DIR, CL_HEADERS, CL_SOURCES)
# Define compile-time macros for kernels
build_opts = [
    f"-cl-std=CL1.2",
    f"-D SCALAR_TYPE={CL_SCALAR_TYPE}",
    f"-D SIMD_WIDTH={buffer_mgr.padding_ctx.simd_width}",
    f"-D C_TILE_SIZE={C_TILE_SIZE}",
    f"-D MAX_EXITS_ENSEMBLE={MAX_EXITS_ENSEMBLE}",
] + (["-cl-khr-fp16"] if SCALAR_TYPE == "half" else [])
program = cl.Program(ctx, kernel_src).build(options=build_opts)
loss_view, probs_view, grad_input_view = (
    HostView(buffer_mgr.get("final_loss_buf")),
    HostView(buffer_mgr.get("ensemble_probs_buf")),
    HostView(buffer_mgr.get("grad_input_buf")),
)

# 4. Training Loop
print(f"Starting training for {EPOCHS} epochs...")
global_step = 1  # For Adam optimizer bias correction
for epoch in range(EPOCHS):
    shuffled_indices = np.random.permutation(len(X))
    epoch_loss, correct_predictions = 0.0, 0
    for i in range(0, len(X), BATCH_SIZE):
        manager.reset()  # Clear graph for the new batch/iteration
        batch_indices = shuffled_indices[i : i + BATCH_SIZE]
        actual_batch_size = len(batch_indices)
        X_batch, y_batch = X_normalized[batch_indices], y_true[batch_indices]
        X_padded, y_padded, mask_padded = batch_padder.pad_batch(X_batch, y_batch)

        k = KernelWrapper(program, manager, buffer_mgr, global_step)

        # --- Define Common Execution Trunk (Q_TRAINING) ---
        h2d_ops = {
            "input_buf": X_padded,
            "input_buf:mask": mask_padded,
            "targets_buf": y_padded,
            "targets_buf:mask": mask_padded,
        }
        for name, data_to_copy in h2d_ops.items():  # H2D transfers
            base_name, is_mask = name.split(":")[0], ":mask" in name
            cl_buf_target = buffer_mgr.get(base_name).mask_buffer if is_mask else buffer_mgr.get(base_name).cl_buffer
            op_fn = lambda d=data_to_copy, b=cl_buf_target: (
                lambda q, wf: cl.enqueue_copy(q, b, d, wait_for=wf)
            )  # Lambda captures current data
            manager.create_node(
                f"h2d_{name}", NodeType.TRANSFER, "Q_TRAINING", {base_name: AccessMode.EXCLUSIVE_WRITE}, op_fn()
            )
        for param in params:
            k.zero_gradients(param.grad, queue_name="Q_TRAINING")  # Zero gradients
        k.forward_pass(queue_name="Q_TRAINING")
        k.compute_all_exits(queue_name="Q_TRAINING")
        # Tiered strategy for ensemble weight calculation
        TIER2_MAX_EXITS = 512
        if NUM_EXITS <= MAX_EXITS_ENSEMBLE:
            k.ensemble_weights_tier1_reg_reduce(queue_name="Q_TRAINING")
        elif NUM_EXITS <= TIER2_MAX_EXITS:
            k.ensemble_weights_tier2_local_reduce(queue_name="Q_TRAINING")
        else:
            k.ensemble_weights_tier3_map_reduce(queue_name="Q_TRAINING")

        # --- FORK POINT: `blend_ensemble_probabilities` is the last shared step before inference/training path split ---
        blend_node_uid = k.blend_ensemble_probabilities(queue_name="Q_TRAINING")

        # --- Path A: High-priority INFERENCE path (Q_INFERENCE) ---
        # Quickly gets blended probabilities for immediate processing.
        d2h_op_factory = lambda view, buf_name: (
            lambda q, wf: view.update_from_device(q, buffer_mgr.get(buf_name).cl_buffer, wf)
        )
        probs_d2h_uid = manager.create_node(
            "d2h_probs",
            NodeType.TRANSFER,
            "Q_INFERENCE",
            {"ensemble_probs_buf": AccessMode.SHARED_READ},
            d2h_op_factory(probs_view, "ensemble_probs_buf"),
        )
        manager.graph.add_edge(blend_node_uid, probs_d2h_uid)

        # --- Path B: Background TRAINING path (Q_TRAINING) ---
        # Includes loss, full backpropagation, and parameter updates.
        final_training_nodes = []  # Leaf nodes of training path for host synchronization
        loss_uid = k.calculate_losses(queue_name="Q_TRAINING")
        manager.graph.add_edge(blend_node_uid, loss_uid)
        final_training_nodes.append(loss_uid)
        exit_grads_uid = k.calculate_exit_gradients(queue_name="Q_TRAINING")
        temp_grads_uid = k.calculate_temp_gradients(
            queue_name="Q_TRAINING"
        )  # Can run somewhat in parallel with exit_grads
        agg_backprop_uid = k.aggregate_and_backprop_activation(queue_name="Q_TRAINING")
        manager.graph.add_edge(exit_grads_uid, agg_backprop_uid)
        dense_grads_uid = k.calculate_dense_layer_gradients(queue_name="Q_TRAINING")
        manager.graph.add_edge(agg_backprop_uid, dense_grads_uid)
        input_grad_uid = k.backprop_input_gradient(queue_name="Q_TRAINING")
        manager.graph.add_edge(agg_backprop_uid, input_grad_uid)
        final_training_nodes.append(input_grad_uid)
        for param in params:  # Adam updates
            update_uid = k.adam_update(param, queue_name="Q_TRAINING")
            if param.name == "temps":  # Temperatures need clamping
                clamp_uid = k.clamp_temperatures(queue_name="Q_TRAINING")
                manager.graph.add_edge(update_uid, clamp_uid)
                final_training_nodes.append(clamp_uid)
            else:
                final_training_nodes.append(update_uid)
        # D2H for training diagnostics
        loss_d2h_uid = manager.create_node(
            "d2h_loss",
            NodeType.TRANSFER,
            "Q_TRAINING",
            {"final_loss_buf": AccessMode.SHARED_READ},
            d2h_op_factory(loss_view, "final_loss_buf"),
        )
        final_training_nodes.append(loss_d2h_uid)
        grad_d2h_uid = manager.create_node(
            "d2h_grad_input",
            NodeType.TRANSFER,
            "Q_TRAINING",
            {"grad_input_buf": AccessMode.SHARED_READ},
            d2h_op_factory(grad_input_view, "grad_input_buf"),
        )
        final_training_nodes.append(grad_d2h_uid)

        # --- Host Notification Points ---
        inference_complete_event = manager.create_host_notification_point(
            "inference_sync", dependencies=[probs_d2h_uid]
        )
        training_complete_event = manager.create_host_notification_point(
            "training_sync", dependencies=final_training_nodes
        )

        # === COMMIT AND STAGGERED WAIT ===
        manager.commit()  # Enqueue all operations
        inference_complete_event.wait()  # Wait for inference path (quick)

        # Process immediate inference result
        valid_probs = probs_view.valid_slice[:actual_batch_size]  # Use actual_batch_size here
        predicted_classes = np.argmax(valid_probs, axis=1)
        correct_predictions += np.sum(predicted_classes == y_batch)

        training_complete_event.wait()  # Sync with background training path before next iteration

        # Process late diagnostic results
        total_batch_loss = loss_view.valid_slice[0]
        epoch_loss += total_batch_loss
        global_step += 1  # For Adam's beta decay

    avg_epoch_loss = epoch_loss / len(X)
    train_acc = correct_predictions / len(X)
    print(f"Epoch {epoch+1:3d}/{EPOCHS} | Loss: {avg_epoch_loss:.4f} | Accuracy: {train_acc:.2%}")

print("\nTraining finished.")
