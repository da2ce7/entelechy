import pyopencl as cl
import numpy as np
import math
from math import gcd
import networkx as nx
import re
from collections import deque, defaultdict
from enum import Enum, auto
from dataclasses import dataclass, field
from typing import Tuple, List, Set, Dict, Callable, Optional, Union
from sklearn.datasets import load_iris
from sklearn.preprocessing import StandardScaler

# Configuration for scalar type
SCALAR_TYPE = "half"  # or "float"
SCALAR_NP_TYPE = np.float16 if SCALAR_TYPE == "half" else np.float32
CL_SCALAR_TYPE = "half" if SCALAR_TYPE == "half" else "float"
SCALAR_SIZE: int = SCALAR_NP_TYPE().itemsize

# Define data type sizes
INT_SIZE: int = np.int32().itemsize

# Network Configuration (modifiable without kernel recompilation)
INPUT_DIM: int = 4
HIDDEN_DIM: int = 65
OUTPUT_CLASSES: int = 3
NUM_EXITS: int = 40
EPOCHS: int = 100
BATCH_SIZE: int = 130
ADAM_BETA1: float = 0.9
ADAM_BETA2: float = 0.999
LEARNING_RATE: float = 0.001
MIN_TEMP: float = 1.0e-3
MAX_TEMP: float = 10.0
EPSILON: float = 1.0e-8

# OpenCL Kernel Configuration
KERNEL_DIR: str = "kernels"  # Directory where .cl.h and .cl.c files are stored
CL_HEADERS: List[str] = [
    "kernels.cl.h",
]
CL_SOURCES: List[str] = [
    "network_operations.cl.c",
    "autograd.cl.c",
    "parameter_optim.cl.c",
]


### UTILITY FUNCTIONS
def load_and_concatenate_kernels(kernel_dir: str, headers: List[str], sources: List[str]) -> str:
    """
    Loads and concatenates OpenCL kernel source files in a specified order, with robust error handling.

    This function programmatically 'includes' headers by prepending their content,
    ensuring a correct build order without relying on the OpenCL compiler's #include directives.

    Args:
        kernel_dir: The directory where kernel files are located.
        headers: An ordered list of header file names (.cl.h) to be included first.
        sources: An ordered list of source file names (.cl.c) to be included after headers.

    Returns:
        A single string containing all concatenated source code.

    Raises:
        IOError: If the kernel directory or any of the specified files cannot be found.
    """
    full_source_parts = []
    all_files_in_order = headers + sources

    if not os.path.isdir(kernel_dir):
        raise IOError(
            f"Kernel directory '{kernel_dir}' not found. "
            f"Please create it and place the required kernel files inside."
        )

    for filename in all_files_in_order:
        full_path = os.path.join(kernel_dir, filename)
        try:
            with open(full_path, "r", encoding="utf-8") as f:
                full_source_parts.append(f.read())
        except FileNotFoundError:
            raise IOError(
                f"Fatal: Required kernel source file not found at '{full_path}'. "
                f"Please ensure all kernel files ({', '.join(all_files_in_order)}) are present in the '{kernel_dir}' directory."
            )

    print(f"Successfully loaded {len(all_files_in_order)} kernel files from '{kernel_dir}'.")
    return "\n".join(full_source_parts)


### UTILITY FUNCTIONS
def lcm(a: int, b: int) -> int:
    return abs(a * b) // gcd(a, b) if a and b else 0


def next_pow2(n: int) -> int:
    return 1 if n == 0 else 1 << (n - 1).bit_length()


def divisors(n: int) -> List[int]:
    divs: Set[int] = set()
    for i in range(1, int(math.sqrt(n)) + 1):
        if n % i == 0:
            divs.add(i)
            divs.add(n // i)
    return sorted(list(divs), reverse=True)


def he_init(shape: Tuple[int, ...]) -> np.ndarray:
    if len(shape) == 2:  # (fan_in, fan_out)
        fan_in = shape[0]
    elif len(shape) == 3:  # (num_items, fan_in, fan_out)
        fan_in = shape[1]
    else:
        raise ValueError(f"Unsupported shape for He initialization: {shape}")
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


def optimal_local_sizes(global_sizes: Tuple[int, ...], device: cl.Device) -> Tuple[int, ...]:
    max_wgs = device.max_work_group_size
    dims = len(global_sizes)
    if not all(global_sizes):
        return tuple([1] * dims)  # Avoid division by zero

    if dims == 1:
        g = global_sizes[0]
        candidates = [d for d in divisors(g) if d <= max_wgs and (d & (d - 1) == 0)]  # Power of 2 preferred
        if candidates:
            return (candidates[0],)
        candidates = [d for d in divisors(g) if d <= max_wgs]
        return (candidates[0] if candidates else 1,)

    local_sizes = [1] * dims
    size_product = 1
    for i in range(dims):
        dim_limit = device.max_work_item_sizes[i]
        candidates = [d for d in divisors(global_sizes[i]) if d * size_product <= max_wgs and d <= dim_limit]
        local_sizes[i] = candidates[0] if candidates else 1
        size_product *= local_sizes[i]

    return tuple(local_sizes)


def validate_targets(y: np.ndarray, num_classes: int):
    if not np.all(np.logical_and(y >= 0, y < num_classes)):
        raise ValueError(f"Target labels must be integers between 0 and {num_classes - 1}")
    if not np.issubdtype(y.dtype, np.integer):
        raise ValueError("Target labels must be integers")


### CORE ABSTRACTIONS


class BufferRole(Enum):
    # --- Parameters and their derivatives ---
    WEIGHTS = auto()
    BIAS = auto()
    EXIT_WEIGHTS = auto()
    EXIT_BIAS = auto()
    TEMPERATURES = auto()
    GRADIENT = auto()
    ADAM_M1 = auto()
    ADAM_M2 = auto()

    # --- Data and Activations ---
    INPUT_DATA = auto()
    TARGETS = auto()
    HIDDEN_ACTIVATION = auto()
    UNSCALED_LOGITS = auto()
    EXIT_PROBABILITIES = auto()
    ENSEMBLE_WEIGHTS = auto()
    ENSEMBLE_PROBABILITIES = auto()

    # --- Intermediate / Scratch Buffers ---
    GRAD_CONTRIBUTIONS = auto()
    PARTIAL_LOSS_REDUCTION = auto()
    FINAL_LOSS = auto()


@dataclass(frozen=True)
class Parameter:
    name: str  # The base name, e.g., "weights"

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


def pad_to_multiple(dim: int, multiple: int) -> int:
    return (dim + multiple - 1) // multiple * multiple


def pad_tensor(data: np.ndarray, padded_shape: Tuple[int, ...]) -> np.ndarray:
    if data.shape == padded_shape:
        return data
    padded = np.zeros(padded_shape, dtype=data.dtype)
    slices = tuple(slice(0, d) for d in data.shape)
    padded[slices] = data
    return padded


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
    def __init__(self, context: cl.Context, device: cl.Device):
        self.context = context
        self.padding_ctx = PaddingContext.from_device(device)
        self.buffers: Dict[str, Buffer] = {}

    def _get_padded_shape(self, role: BufferRole, real_shape: Tuple[int, ...]) -> Tuple[int, ...]:
        sw = self.padding_ctx.simd_width
        padded_batch = pad_to_multiple(BATCH_SIZE, sw)

        if role in [BufferRole.WEIGHTS, BufferRole.GRADIENT, BufferRole.ADAM_M1, BufferRole.ADAM_M2]:
            # This rule applies to any parameter buffer with this role, regardless of name
            return (pad_to_multiple(real_shape[0], sw), pad_to_multiple(real_shape[1], sw))

        if role in [BufferRole.EXIT_WEIGHTS]:
            return (real_shape[0], pad_to_multiple(real_shape[1], sw), pad_to_multiple(real_shape[2], sw))

        if role in [BufferRole.BIAS, BufferRole.EXIT_BIAS, BufferRole.TEMPERATURES]:
            return (
                (pad_to_multiple(real_shape[0], sw),)
                if len(real_shape) == 1
                else (real_shape[0], pad_to_multiple(real_shape[1], sw))
            )

        if role == BufferRole.HIDDEN_ACTIVATION:
            return (padded_batch, pad_to_multiple(real_shape[1], sw))

        if role in [BufferRole.UNSCALED_LOGITS, BufferRole.EXIT_PROBABILITIES, BufferRole.GRAD_CONTRIBUTIONS]:
            return (real_shape[0], padded_batch, pad_to_multiple(real_shape[2], sw))

        if role in [
            BufferRole.INPUT_DATA,
            BufferRole.TARGETS,
            BufferRole.ENSEMBLE_PROBABILITIES,
            BufferRole.ENSEMBLE_WEIGHTS,
        ]:
            return (padded_batch, *real_shape[1:])

        # Default for buffers that don't need special padding (e.g., FINAL_LOSS)
        return real_shape

    def create_buffer(
        self,
        name: str,
        role: BufferRole,
        real_shape: Tuple[int, ...],
        dtype: np.dtype,
        init_data: Optional[np.ndarray] = None,
    ) -> Buffer:
        needs_mask = role in [
            BufferRole.INPUT_DATA,
            BufferRole.TARGETS,
            BufferRole.ENSEMBLE_PROBABILITIES,
            BufferRole.HIDDEN_ACTIVATION,
        ]
        padded_shape = self._get_padded_shape(role, real_shape)
        size = int(np.prod(padded_shape) * np.dtype(dtype).itemsize) if np.prod(padded_shape) > 0 else 0

        cl_buf = cl.Buffer(self.context, cl.mem_flags.READ_WRITE, size=max(size, 4))
        mask_buf = None

        if needs_mask:
            mask_padded_shape = (padded_shape[0],)
            mask_size = int(mask_padded_shape[0] * SCALAR_SIZE)
            mask_buf = cl.Buffer(self.context, cl.mem_flags.READ_WRITE, size=max(mask_size, 4))

        buffer_spec = BufferSpec(name, role, real_shape, padded_shape, dtype)
        buffer_obj = Buffer(buffer_spec, cl_buf, mask_buf)
        self.buffers[name] = buffer_obj

        if init_data is not None:
            padded_data = pad_tensor(init_data, padded_shape)
            init_queue = cl.CommandQueue(self.context)
            cl.enqueue_copy(init_queue, cl_buf, padded_data).wait()

        return buffer_obj

    def get(self, name: str) -> Buffer:
        return self.buffers[name]


class BatchPadder:
    def __init__(self, input_spec: BufferSpec, target_spec: BufferSpec):
        self.input_spec = input_spec
        self.target_spec = target_spec

    def pad_batch(self, X: np.ndarray, y_true: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        batch_size = X.shape[0]
        X_padded = pad_tensor(X, self.input_spec.padded_shape)
        y_padded = pad_tensor(y_true, self.target_spec.padded_shape)
        mask_padded = pad_tensor(np.ones(batch_size, dtype=SCALAR_NP_TYPE), (self.input_spec.padded_shape[0],))
        return X_padded, y_padded, mask_padded


class HostView:
    def __init__(self, buffer: Buffer):
        self.spec = buffer.spec
        self.host_data = np.empty(self.spec.padded_shape, dtype=self.spec.dtype)

    def update_from_device(self, queue: cl.CommandQueue, cl_buffer: cl.Buffer, wait_for=None):
        return cl.enqueue_copy(queue, self.host_data, cl_buffer, wait_for=wait_for or [])

    @property
    def valid_slice(self):
        slicing = tuple(slice(0, dim) for dim in self.spec.real_shape)
        return self.host_data[slicing]


### WORK MANAGER
class NodeType(Enum):
    COMPUTE = auto()
    TRANSFER = auto()
    SYNC = auto()


class SyncType(Enum):
    HOST_SIGNAL = auto()


class AccessMode(Enum):
    SHARED_READ = auto()
    EXCLUSIVE_WRITE = auto()
    EXCLUSIVE_UPDATE = auto()


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
    def __init__(self, context: cl.Context):
        self.context = context
        self.graph = nx.DiGraph()
        self.queues: Dict[str, cl.CommandQueue] = {
            "compute": cl.CommandQueue(context),
            "transfer": cl.CommandQueue(context),
        }
        self.node_id_counter: int = 0
        self.last_writer: Dict[str, int] = {}
        self.user_events: Dict[int, cl.UserEvent] = {}

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
        uid = self._get_uid()
        node = ExecutionNode(uid, name, node_type, queue_name, access_map, op_fn, params)
        self.graph.add_node(uid, node=node)

        for res, mode in access_map.items():
            if mode in ACCESS_WRITE_MODES:
                if res in self.last_writer:
                    self.graph.add_edge(self.last_writer[res], uid)
                self.last_writer[res] = uid
            elif mode == AccessMode.SHARED_READ:
                if res in self.last_writer:
                    self.graph.add_edge(self.last_writer[res], uid)
        return uid

    def create_sync_node(self, name: str, dependencies: List[int]) -> int:
        uid = self._get_uid()
        user_event = cl.UserEvent(self.context)
        self.user_events[uid] = user_event

        def _signal_host(queue, wait_for):
            marker = cl.enqueue_marker(queue, wait_for=wait_for)

            def callback(evt, status):
                if status == cl.command_execution_status.COMPLETE:
                    user_event.set_status(cl.command_execution_status.COMPLETE)

            marker.set_callback(cl.command_execution_status.COMPLETE, callback)
            return marker

        node = ExecutionNode(uid, name, NodeType.SYNC, "compute", {}, _signal_host, {"sync_type": SyncType.HOST_SIGNAL})
        self.graph.add_node(uid, node=node)
        for dep_uid in dependencies:
            self.graph.add_edge(dep_uid, uid)
        return uid

    def commit(self):
        ordered_nodes = list(nx.topological_sort(self.graph))
        completion_events: Dict[int, cl.Event] = {}

        for uid in ordered_nodes:
            node = self.graph.nodes[uid]["node"]
            queue = self.queues[node.queue_name]
            wait_for = [
                completion_events[p_uid] for p_uid in self.graph.predecessors(uid) if p_uid in completion_events
            ]
            node.event = node.operation_fn(queue, wait_for)
            if node.event:
                completion_events[uid] = node.event

    def wait_for_sync(self, sync_uid: int):
        if sync_uid in self.user_events:
            self.user_events.pop(sync_uid).wait()
        else:
            raise KeyError(f"No UserEvent found for sync node UID {sync_uid}")

    def reset(self):
        self.graph.clear()
        self.node_id_counter = 0
        self.last_writer.clear()
        self.user_events.clear()


### KERNEL WRAPPER


class KernelWrapper:
    def __init__(self, program: cl.Program, manager: WorkManager, buffer_manager: BufferManager, global_step: int):
        self.p, self.m, self.b = program, manager, buffer_manager
        self.padded_input_dim = self.b.get("weights").spec.padded_shape[0]
        self.padded_hidden_dim = self.b.get("weights").spec.padded_shape[1]
        self.padded_output_classes = self.b.get("exit_weights").spec.padded_shape[2]
        self.padded_batch_size = self.b.get("input_buf").spec.padded_shape[0]
        self.beta1_t = SCALAR_NP_TYPE(ADAM_BETA1**global_step)
        self.beta2_t = SCALAR_NP_TYPE(ADAM_BETA2**global_step)

    def _create_kernel_node(self, name, global_size, local_size, access_map, *args):
        kernel = getattr(self.p, name)
        op_fn = lambda q, wf: kernel(q, global_size, local_size, *args, wait_for=wf)
        return self.m.create_node(name, NodeType.COMPUTE, "compute", access_map, op_fn)

    def zero_gradients(self, grad_buffer_name: str) -> int:
        buffer = self.b.get(grad_buffer_name)
        op_fn = lambda q, wf: cl.enqueue_fill_buffer(
            q, buffer.cl_buffer, SCALAR_NP_TYPE(0), 0, buffer.cl_buffer.size, wait_for=wf
        )
        return self.m.create_node(
            f"zero_{grad_buffer_name}",
            NodeType.COMPUTE,
            "compute",
            {grad_buffer_name: AccessMode.EXCLUSIVE_WRITE},
            op_fn,
        )

    def forward_pass(self) -> int:
        g, l = (self.padded_batch_size, self.padded_hidden_dim // self.b.padding_ctx.simd_width), None
        access = {
            "input_buf": AccessMode.SHARED_READ,
            "weights": AccessMode.SHARED_READ,
            "biases": AccessMode.SHARED_READ,
            "hidden_buf": AccessMode.EXCLUSIVE_WRITE,
        }
        args = (
            cl.LocalMemory(2 * (l[0] if l else 16) * SCALAR_SIZE),
            self.b.get("input_buf").cl_buffer,
            self.b.get("input_buf").mask_buffer,
            self.b.get("weights").cl_buffer,
            self.b.get("biases").cl_buffer,
            self.b.get("hidden_buf").cl_buffer,
            self.b.get("hidden_buf").mask_buffer,
            np.int32(self.padded_input_dim),
            np.int32(self.padded_hidden_dim),
        )
        return self._create_kernel_node("forward_pass", g, l, access, *args)

    def compute_all_exits(self) -> int:
        g, l = (NUM_EXITS, self.padded_batch_size), None
        access = {
            "hidden_buf": AccessMode.SHARED_READ,
            "exit_weights": AccessMode.SHARED_READ,
            "exit_biases": AccessMode.SHARED_READ,
            "temps": AccessMode.SHARED_READ,
            "targets_buf": AccessMode.SHARED_READ,
            "unscaled_logits_buf": AccessMode.EXCLUSIVE_WRITE,
            "exit_probs_buf": AccessMode.EXCLUSIVE_WRITE,
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
            self.b.get("final_loss_buf").cl_buffer,
            np.int32(self.padded_batch_size),
            np.int32(HIDDEN_DIM),
            np.int32(OUTPUT_CLASSES),
            np.int32(NUM_EXITS),
        )
        return self._create_kernel_node("compute_all_exits", g, l, access, *args)

    def compute_ensemble(self) -> int:
        g, l = (self.padded_batch_size,), None
        access = {
            "exit_probs_buf": AccessMode.SHARED_READ,
            "temps": AccessMode.SHARED_READ,
            "ensemble_weights_buf": AccessMode.EXCLUSIVE_WRITE,
            "ensemble_probs_buf": AccessMode.EXCLUSIVE_WRITE,
        }
        args = (
            self.b.get("exit_probs_buf").cl_buffer,
            self.b.get("targets_buf").mask_buffer,
            self.b.get("temps").cl_buffer,
            self.b.get("ensemble_weights_buf").cl_buffer,
            self.b.get("ensemble_probs_buf").cl_buffer,
            np.int32(self.padded_batch_size),
            np.int32(OUTPUT_CLASSES),
            np.int32(NUM_EXITS),
        )
        return self._create_kernel_node("compute_ensemble", g, l, access, *args)

    def calculate_losses(self) -> int:
        g, l = (self.padded_batch_size,), (256,)
        num_groups = g[0] // l[0]
        # Dynamically create partial loss buffer
        self.b.create_buffer("partial_loss_buf", BufferRole.PARTIAL_LOSS_REDUCTION, (num_groups,), np.float32)

        access1 = {
            "ensemble_probs_buf": AccessMode.SHARED_READ,
            "targets_buf": AccessMode.SHARED_READ,
            "partial_loss_buf": AccessMode.EXCLUSIVE_WRITE,
        }
        args1 = (
            self.b.get("ensemble_probs_buf").cl_buffer,
            self.b.get("targets_buf").mask_buffer,
            self.b.get("targets_buf").cl_buffer,
            self.b.get("partial_loss_buf").cl_buffer,
            np.int32(self.padded_batch_size),
            np.int32(OUTPUT_CLASSES),
        )
        partial_node = self._create_kernel_node("calculate_partial_losses", g, l, access1, *args1)

        g2, l2 = (256,), (256,)
        access2 = {"partial_loss_buf": AccessMode.SHARED_READ, "final_loss_buf": AccessMode.EXCLUSIVE_WRITE}
        args2 = (self.b.get("partial_loss_buf").cl_buffer, self.b.get("final_loss_buf").cl_buffer, np.int32(num_groups))

        final_node = self._create_kernel_node("aggregate_partial_losses", g2, l2, access2, *args2)
        self.m.graph.add_edge(partial_node, final_node)
        return final_node

    def calculate_exit_gradients(self) -> int:
        g, l = (NUM_EXITS, HIDDEN_DIM), None
        access = {
            "hidden_buf": AccessMode.SHARED_READ,
            "exit_probs_buf": AccessMode.SHARED_READ,
            "ensemble_weights_buf": AccessMode.SHARED_READ,
            "targets_buf": AccessMode.SHARED_READ,
            "exit_weights": AccessMode.SHARED_READ,
            "grad_exit_weights": AccessMode.EXCLUSIVE_WRITE,
            "grad_exit_biases": AccessMode.EXCLUSIVE_WRITE,
            "grad_hidden_contributions_buf": AccessMode.EXCLUSIVE_WRITE,
        }
        args = (
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
            np.int32(HIDDEN_DIM),
            np.int32(OUTPUT_CLASSES),
            np.int32(NUM_EXITS),
        )
        return self._create_kernel_node("calculate_exit_gradients", g, l, access, *args)

    def calculate_shared_gradients(self) -> int:
        g, l = (INPUT_DIM, HIDDEN_DIM), None
        access = {
            "input_buf": AccessMode.SHARED_READ,
            "hidden_buf": AccessMode.SHARED_READ,
            "grad_hidden_contributions_buf": AccessMode.SHARED_READ,
            "grad_weights": AccessMode.EXCLUSIVE_WRITE,
            "grad_biases": AccessMode.EXCLUSIVE_WRITE,
        }
        args = (
            self.b.get("input_buf").cl_buffer,
            self.b.get("input_buf").mask_buffer,
            self.b.get("hidden_buf").cl_buffer,
            self.b.get("grad_hidden_contributions_buf").cl_buffer,
            self.b.get("grad_weights").cl_buffer,
            self.b.get("grad_biases").cl_buffer,
            np.int32(self.padded_batch_size),
            np.int32(INPUT_DIM),
            np.int32(HIDDEN_DIM),
            np.int32(NUM_EXITS),
        )
        return self._create_kernel_node("calculate_shared_gradients", g, l, access, *args)

    def calculate_temp_gradients(self) -> int:
        g, l = (NUM_EXITS,), None
        access = {
            "unscaled_logits_buf": AccessMode.SHARED_READ,
            "exit_probs_buf": AccessMode.SHARED_READ,
            "ensemble_weights_buf": AccessMode.SHARED_READ,
            "temps": AccessMode.SHARED_READ,
            "targets_buf": AccessMode.SHARED_READ,
            "grad_temps": AccessMode.EXCLUSIVE_WRITE,
        }
        args = (
            self.b.get("unscaled_logits_buf").cl_buffer,
            self.b.get("exit_probs_buf").cl_buffer,
            self.b.get("ensemble_weights_buf").cl_buffer,
            self.b.get("temps").cl_buffer,
            self.b.get("targets_buf").cl_buffer,
            self.b.get("targets_buf").mask_buffer,
            self.b.get("grad_temps").cl_buffer,
            np.int32(self.padded_batch_size),
            np.int32(OUTPUT_CLASSES),
            np.int32(NUM_EXITS),
        )
        return self._create_kernel_node("calculate_temp_gradients", g, l, access, *args)

    def adam_update(self, param: Parameter) -> int:
        total_params = int(np.prod(self.b.get(param.value).spec.padded_shape))
        g, l = (total_params,), None
        access = {
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
            np.int32(total_params),
        )
        return self._create_kernel_node("adam_update", g, l, access, *args)

    def clamp_temperatures(self) -> int:
        g, l = (NUM_EXITS,), None
        access = {"temps": AccessMode.EXCLUSIVE_UPDATE}
        args = (self.b.get("temps").cl_buffer, SCALAR_NP_TYPE(MIN_TEMP), SCALAR_NP_TYPE(MAX_TEMP), np.int32(NUM_EXITS))
        return self._create_kernel_node("clamp_temperatures", g, l, access, *args)


### MAIN EXECUTION ###

# 1. Setup
ctx = cl.create_some_context(interactive=False)
device = ctx.devices[0]
manager = WorkManager(ctx)
buffer_mgr = BufferManager(ctx, device)

if SCALAR_TYPE == "half" and "cl_khr_fp16" not in device.extensions:
    raise RuntimeError("Device does not support FP16 (cl_khr_fp16 extension)")

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

# Create parameter, gradient, and optimizer state buffers declaratively
for p in params:
    role, shape, init_fn = param_specs[p.name]
    buffer_mgr.create_buffer(p.value, role, shape, SCALAR_NP_TYPE, init_data=init_fn(shape))
    buffer_mgr.create_buffer(p.grad, BufferRole.GRADIENT, shape, SCALAR_NP_TYPE)
    buffer_mgr.create_buffer(p.m1, BufferRole.ADAM_M1, shape, SCALAR_NP_TYPE)
    buffer_mgr.create_buffer(p.m2, BufferRole.ADAM_M2, shape, SCALAR_NP_TYPE)

# Create data buffers with explicit roles
data_buffer_specs = {
    "input_buf": (BufferRole.INPUT_DATA, (BATCH_SIZE, INPUT_DIM), SCALAR_NP_TYPE),
    "targets_buf": (BufferRole.TARGETS, (BATCH_SIZE,), np.int32),
    "hidden_buf": (BufferRole.HIDDEN_ACTIVATION, (BATCH_SIZE, HIDDEN_DIM), SCALAR_NP_TYPE),
    "unscaled_logits_buf": (BufferRole.UNSCALED_LOGITS, (NUM_EXITS, BATCH_SIZE, OUTPUT_CLASSES), SCALAR_NP_TYPE),
    "exit_probs_buf": (BufferRole.EXIT_PROBABILITIES, (NUM_EXITS, BATCH_SIZE, OUTPUT_CLASSES), SCALAR_NP_TYPE),
    "ensemble_weights_buf": (BufferRole.ENSEMBLE_WEIGHTS, (BATCH_SIZE, NUM_EXITS), SCALAR_NP_TYPE),
    "ensemble_probs_buf": (BufferRole.ENSEMBLE_PROBABILITIES, (BATCH_SIZE, OUTPUT_CLASSES), SCALAR_NP_TYPE),
    "grad_hidden_contributions_buf": (
        BufferRole.GRAD_CONTRIBUTIONS,
        (NUM_EXITS, BATCH_SIZE, HIDDEN_DIM),
        SCALAR_NP_TYPE,
    ),
    "final_loss_buf": (BufferRole.FINAL_LOSS, (1,), np.float32),
}
for name, (role, shape, dtype) in data_buffer_specs.items():
    buffer_mgr.create_buffer(name, role, shape, dtype)

# 3. Data and Kernel Prep
X, y = load_iris(return_X_y=True)
X_normalized, y_true = StandardScaler().fit_transform(X).astype(SCALAR_NP_TYPE), y.astype(np.int32)
validate_targets(y_true, OUTPUT_CLASSES)
batch_padder = BatchPadder(buffer_mgr.get("input_buf").spec, buffer_mgr.get("targets_buf").spec)

# Load and build kernels from multiple files
kernel_src = load_and_concatenate_kernels(KERNEL_DIR, CL_HEADERS, CL_SOURCES)

build_opts = [
    f"-cl-std=CL1.2",
    f"-D SCALAR_TYPE={CL_SCALAR_TYPE}",
    f"-D SIMD_WIDTH={buffer_mgr.padding_ctx.simd_width}",
]
if SCALAR_TYPE == "half":
    build_opts.append("-D cl_khr_fp16")
program = cl.Program(ctx, kernel_src).build(options=build_opts)

# HostViews for results
loss_view = HostView(buffer_mgr.get("final_loss_buf"))
probs_view = HostView(buffer_mgr.get("ensemble_probs_buf"))

# 4. Training Loop
print("Starting training...")
global_step = 1
for epoch in range(EPOCHS):
    shuffled_indices = np.random.permutation(len(X))
    epoch_loss, correct_predictions = 0.0, 0

    for i in range(0, len(X), BATCH_SIZE):
        manager.reset()

        batch_indices = shuffled_indices[i : i + BATCH_SIZE]
        actual_batch_size = len(batch_indices)
        X_batch, y_batch = X_normalized[batch_indices], y_true[batch_indices]
        X_padded, y_padded, mask_padded = batch_padder.pad_batch(X_batch, y_batch)

        # H2D Transfers
        h2d_ops = {
            "input_buf": X_padded,
            "input_buf:mask": mask_padded,
            "targets_buf": y_padded,
            "targets_buf:mask": mask_padded,
        }
        for name, data in h2d_ops.items():
            is_mask = ":mask" in name
            base_name = name.split(":")[0]
            buf_obj = buffer_mgr.get(base_name)
            cl_buf = buf_obj.mask_buffer if is_mask else buf_obj.cl_buffer
            op_fn = lambda d=data, b=cl_buf: (lambda q, wf: cl.enqueue_copy(q, b, d, wait_for=wf))
            manager.create_node(
                f"h2d_{name}", NodeType.TRANSFER, "transfer", {base_name: AccessMode.EXCLUSIVE_WRITE}, op_fn()
            )

        k = KernelWrapper(program, manager, buffer_mgr, global_step)

        # Phase 1: Zero Gradients
        for param in params:
            if param.name != "temps":  # In this model, temps grad is totally overwritten
                k.zero_gradients(param.grad)

        # Phase 2: Forward Pass & Loss
        k.forward_pass()
        k.compute_all_exits()
        k.compute_ensemble()
        final_loss_node = k.calculate_losses()

        # Phase 3: Backward Pass
        k.calculate_exit_gradients()
        k.calculate_shared_gradients()
        k.calculate_temp_gradients()

        # Phase 4: Parameter Updates
        adam_nodes, clamp_node = [], None
        for param in params:
            update_node = k.adam_update(param)
            adam_nodes.append(update_node)
            if param.name == "temps":
                clamp_node = k.clamp_temperatures()
                manager.graph.add_edge(update_node, clamp_node)

        # Phase 5: D2H Transfers
        def d2h_op(view, buffer_name):
            buf = buffer_mgr.get(buffer_name).cl_buffer
            return lambda q, wf: view.update_from_device(q, buf, wait_for=wf)

        loss_d2h = manager.create_node(
            "d2h_loss",
            NodeType.TRANSFER,
            "transfer",
            {"final_loss_buf": AccessMode.SHARED_READ},
            d2h_op(loss_view, "final_loss_buf"),
        )
        probs_d2h = manager.create_node(
            "d2h_probs",
            NodeType.TRANSFER,
            "transfer",
            {"ensemble_probs_buf": AccessMode.SHARED_READ},
            d2h_op(probs_view, "ensemble_probs_buf"),
        )

        # Final Sync Node
        all_leaves = adam_nodes + ([clamp_node] if clamp_node else []) + [loss_d2h, probs_d2h]
        sync_node = manager.create_sync_node("final_sync", [uid for uid in all_leaves if uid in manager.graph])

        # Commit and execute
        manager.commit()
        manager.wait_for_sync(sync_node)

        # Process results on host
        batch_loss = loss_view.valid_slice[0]
        epoch_loss += batch_loss * actual_batch_size
        valid_probs = probs_view.valid_slice[:actual_batch_size]
        predicted_classes = np.argmax(valid_probs, axis=1)
        correct_predictions += np.sum(predicted_classes == y_batch)
        global_step += 1

    avg_epoch_loss = epoch_loss / len(X)
    train_acc = correct_predictions / len(X)
    print(f"Epoch {epoch+1:3d}/{EPOCHS} | Loss: {avg_epoch_loss:.4f} | Accuracy: {train_acc:.2%}")

print("\nTraining finished.")
