import pyopencl as cl
import numpy as np
import math
from math import gcd
from sklearn.datasets import load_iris
from sklearn.preprocessing import StandardScaler
import networkx as nx
import re
from collections import deque, defaultdict
from enum import Enum, auto
from dataclasses import dataclass, field
from typing import Tuple, List, Set, Dict, Callable, Optional, Union

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

# OpenCL kernel files (assumed to exist)
CL_HEADER_FILES: List[str] = [
    'kernels.cl.h',       # Common macros/defs
    'math_utils.cl.h'     # Math functions
]

CL_KERNEL_FILES: List[str] = [
    'network_operations.cl.c',    # Forward + exits
    'autograd.cl.c',              # Gradients
    'parameter_optim.cl.c'        # Adam + constraints
]

def lcm(a: int, b: int) -> int:
    """Calculate the least common multiple of two integers."""
    return a * b // gcd(a, b)

def next_pow2(n: int) -> int:
    """Find the next power of 2 greater than or equal to n."""
    return 1 if n == 0 else 1 << (n - 1).bit_length()

def divisors(n: int) -> List[int]:
    """Return a sorted list of divisors of n in descending order."""
    divs: Set[int] = set()
    for i in range(1, int(math.sqrt(n)) + 1):
        if n % i == 0:
            divs.add(i)
            divs.add(n // i)
    return sorted(divs, reverse=True)

def he_init(shape: Tuple[int, ...]) -> np.ndarray:
    """Initialize weights using He initialization for neural networks."""
    if len(shape) == 2:
        fan_in = shape[0]
    elif len(shape) == 3:
        fan_in = shape[1]
    else:
        raise ValueError("Unsupported shape for He initialization")
    scale = np.sqrt(2.0 / fan_in)
    return np.random.normal(0, scale, shape).astype(SCALAR_NP_TYPE)

def select_simd_width(device: cl.Device) -> int:
    """Determine the SIMD width based on the device vendor and properties."""
    if 'Intel' in device.vendor and 'cl_intel_subgroups' in device.extensions:
        return 16
    elif 'AMD' in device.vendor:
        if re.search(r'(?i)CDNA|RDNA', device.name):
            return 8
        else:
            return 4
    elif 'NVIDIA' in device.vendor:
        return 4
    return 1

def optimal_local_sizes(global_sizes: Tuple[int, ...], device: cl.Device) -> Tuple[int, ...]:
    """Calculate optimal local work group sizes for OpenCL kernels."""
    max_work_group_size = device.max_work_group_size
    max_work_item_sizes = device.max_work_item_sizes
    local_sizes = []
    for g, m in zip(global_sizes, max_work_item_sizes[:len(global_sizes)]):
        candidates = [d for d in divisors(g) if d <= min(m, max_work_group_size)]
        local_sizes.append(candidates[0] if candidates else 1)
    total_local = np.prod(local_sizes)
    if total_local > max_work_group_size:
        local_sizes = [min(l, max_work_group_size // (total_local // l)) for l in local_sizes]
    return tuple(local_sizes)

@dataclass
class PaddingContext:
    """Context for padding tensors based on device properties."""
    simd_width: int
    min_alignment: int
    device: cl.Device
    padding_mode: str = 'simd_aware'

    @classmethod
    def from_device(cls, device: cl.Device) -> 'PaddingContext':
        """Create a PaddingContext instance from a device."""
        return cls(
            simd_width=select_simd_width(device),
            min_alignment=max(device.min_data_type_align_size, 16),
            device=device
        )

class PaddingStrategy:
    """Manages different padding strategies for tensors."""
    _strategies: Dict[str, Callable] = {}

    @classmethod
    def register(cls, name: str) -> Callable:
        """Register a padding strategy function."""
        def decorator(func: Callable) -> Callable:
            cls._strategies[name] = func
            return func
        return decorator

    @classmethod
    def apply(cls, context: PaddingContext, shape: Tuple[int, ...], dtype: np.dtype = SCALAR_NP_TYPE) -> Tuple[int, ...]:
        """Apply the specified padding strategy."""
        return cls._strategies[context.padding_mode](context, shape, dtype)

@PaddingStrategy.register('simd_aware')
def _simd_strategy(ctx: PaddingContext, shape: Tuple[int, ...], dtype: np.dtype) -> Tuple[int, ...]:
    """SIMD-aware padding strategy."""
    item_size = np.dtype(dtype).itemsize
    alignment = lcm(ctx.min_alignment, ctx.simd_width * item_size)
    return tuple((dim + alignment - 1) // alignment * alignment for dim in shape)

def pad_tensor(data: np.ndarray, context: PaddingContext, strategy: str = 'simd_aware') -> np.ndarray:
    """Pad a tensor according to the specified strategy."""
    orig_shape = data.shape
    padded_shape = PaddingStrategy.apply(context, orig_shape, data.dtype)
    padded = np.zeros(padded_shape, dtype=data.dtype)
    slices = tuple(slice(0, d) for d in orig_shape)
    padded[slices] = data
    return padded

@dataclass(frozen=True)
class LogicalShape:
    """Describes the original data dimensions"""
    batch: int
    features: tuple  # Other dimensions

@dataclass(frozen=True)
class PaddedShape:
    """Describes actual memory layout after padding"""
    values: tuple

    @property
    def batch_dim(self) -> int:
        return self.values[0]

class BufferPurpose(Enum):
    LOSSES = auto()
    INPUT_DATA = auto()
    HIDDEN_ACT = auto()
    TARGETS = auto()
    MASK = auto()
    EXIT_PROBS = auto()

class BufferType(Enum):
    DATA = auto()
    PARAMETER = auto()
    MASK = auto()

@dataclass
class BufferSpec:
    """Specification for an OpenCL buffer."""
    name: str
    real_shape: Tuple[int, ...]
    dtype: np.dtype
    type: BufferType
    padded_shape: Optional[Tuple[int, ...]] = None
    simd_alignment: Optional[int] = None

# Specialized BufferSpecs
@dataclass
class LossBufferSpec(BufferSpec):
    purpose: BufferPurpose = BufferPurpose.LOSSES
    logical: LogicalShape = field(init=False)
    padded: PaddedShape = field(init=False)

    def __post_init__(self):
        if len(self.real_shape) != 2:
            raise ValueError("Loss buffers must be 2D (batch, exits)")
        self.logical = LogicalShape(
            batch=self.real_shape[1],
            features=(self.real_shape[0],)
        )
        self.padded = PaddedShape(self.padded_shape)

@dataclass
class InputBufferSpec(BufferSpec):
    purpose: BufferPurpose = BufferPurpose.INPUT_DATA
    logical: LogicalShape = field(init=False)
    padded: PaddedShape = field(init=False)

    def __post_init__(self):
        self.logical = LogicalShape(
            batch=self.real_shape[0],
            features=self.real_shape[1:]
        )
        self.padded = PaddedShape(self.padded_shape)

@dataclass
class HiddenActBufferSpec(BufferSpec):
    purpose: BufferPurpose = BufferPurpose.HIDDEN_ACT
    logical: LogicalShape = field(init=False)
    padded: PaddedShape = field(init=False)

    def __post_init__(self):
        self.logical = LogicalShape(
            batch=self.real_shape[0],
            features=self.real_shape[1:]
        )
        self.padded = PaddedShape(self.padded_shape)

@dataclass
class TargetsBufferSpec(BufferSpec):
    purpose: BufferPurpose = BufferPurpose.TARGETS
    logical: LogicalShape = field(init=False)
    padded: PaddedShape = field(init=False)

    def __post_init__(self):
        self.logical = LogicalShape(
            batch=self.real_shape[0],
            features=()
        )
        self.padded = PaddedShape(self.padded_shape)

@dataclass
class MaskBufferSpec(BufferSpec):
    purpose: BufferPurpose = BufferPurpose.MASK
    logical: LogicalShape = field(init=False)
    padded: PaddedShape = field(init=False)

    def __post_init__(self):
        self.logical = LogicalShape(
            batch=self.real_shape[0],
            features=()
        )
        self.padded = PaddedShape(self.padded_shape)

@dataclass
class ExitProbsBufferSpec(BufferSpec):
    purpose: BufferPurpose = BufferPurpose.EXIT_PROBS
    logical: LogicalShape = field(init=False)
    padded: PaddedShape = field(init=False)

    def __post_init__(self):
        if len(self.real_shape) != 3:
            raise ValueError("Exit probabilities buffers must be 3D (batch, exits, classes)")
        self.logical = LogicalShape(
            batch=self.real_shape[0],
            features=(self.real_shape[1], self.real_shape[2])
        )
        self.padded = PaddedShape(self.padded_shape)

class Buffer:
    """Base class for OpenCL buffers."""
    def __init__(self, spec: BufferSpec, cl_buffer: cl.Buffer):
        self.spec = spec
        self.cl_buffer = cl_buffer

class DataBuffer(Buffer):
    """Buffer for data with masking."""
    def __init__(self, spec: BufferSpec, cl_buffer: cl.Buffer, mask_buffer: Optional[cl.Buffer]):
        super().__init__(spec, cl_buffer)
        self.mask_buffer = mask_buffer

class ParameterBuffer(Buffer):
    """Buffer for parameters."""
    def __init__(self, spec: BufferSpec, cl_buffer: cl.Buffer):
        super().__init__(spec, cl_buffer)

class DataBufferManager:
    """Manages data buffers with masking and purpose-specific padding."""
    def __init__(self, context: cl.Context, device: cl.Device, work_manager: 'WorkManager'):
        self.context = context
        self.device = device
        self.work_manager = work_manager
        self.padding_ctx = PaddingContext.from_device(device)

    def create_data_buffer(self, name: str, real_shape: Tuple[int, ...], dtype: np.dtype, purpose: BufferPurpose) -> DataBuffer:
        """Create a data buffer with associated mask buffer based on purpose."""
        if purpose == BufferPurpose.LOSSES:
            spec = LossBufferSpec(name, real_shape, dtype, BufferType.DATA)
        elif purpose == BufferPurpose.INPUT_DATA:
            spec = InputBufferSpec(name, real_shape, dtype, BufferType.DATA)
        elif purpose == BufferPurpose.HIDDEN_ACT:
            spec = HiddenActBufferSpec(name, real_shape, dtype, BufferType.DATA)
        elif purpose == BufferPurpose.TARGETS:
            spec = TargetsBufferSpec(name, real_shape, dtype, BufferType.DATA)
        elif purpose == BufferPurpose.MASK:
            spec = MaskBufferSpec(name, real_shape, dtype, BufferType.MASK)
        elif purpose == BufferPurpose.EXIT_PROBS:
            spec = ExitProbsBufferSpec(name, real_shape, dtype, BufferType.DATA)
        else:
            raise ValueError(f"Unsupported buffer purpose: {purpose}")

        padded_shape = self._pad_data_shape(real_shape, purpose)
        spec.padded_shape = padded_shape
        data_buffer = self._allocate_buffer(name, padded_shape, dtype)

        if purpose != BufferPurpose.MASK:  # Masks don't need their own mask
            mask_name = f"mask_{name}"
            mask_real_shape = (real_shape[0],)
            mask_padded_shape = (padded_shape[0],)
            mask_spec = MaskBufferSpec(mask_name, mask_real_shape, dtype, BufferType.MASK)
            mask_spec.padded_shape = mask_padded_shape
            mask_buffer = self._allocate_buffer(mask_name, mask_padded_shape, dtype)
        else:
            mask_buffer = None

        return DataBuffer(spec, data_buffer, mask_buffer)

    def _pad_data_shape(self, shape: Tuple[int, ...], purpose: BufferPurpose) -> Tuple[int, ...]:
        """Pad the shape based on the buffer purpose."""
        if purpose in [BufferPurpose.TARGETS, BufferPurpose.MASK]:
            padded_batch = next_pow2(shape[0])
            return (padded_batch,) + shape[1:]
        elif purpose in [BufferPurpose.LOSSES]: 
            padded_exits = next_pow2(shape[0])
            padded_batch = next_pow2(shape[1])
            return (padded_exits, padded_batch)
        elif purpose in [BufferPurpose.INPUT_DATA, BufferPurpose.HIDDEN_ACT, BufferPurpose.EXIT_PROBS]:
            padded_batch = next_pow2(shape[0])
            padded_features = tuple(next_pow2(dim) for dim in shape[1:])
            return (padded_batch,) + padded_features
        raise ValueError(f"Unsupported buffer purpose for padding: {purpose}")

    def _allocate_buffer(self, name: str, shape: Tuple[int, ...], dtype: np.dtype) -> cl.Buffer:
        """Allocate an OpenCL buffer."""
        size = np.prod(shape) * dtype().itemsize
        buffer = cl.Buffer(self.context, cl.mem_flags.READ_WRITE, size=size)
        self.work_manager.logical_resources[name] = (buffer, 0)
        return buffer

class HostView:
    """Host-side data access helper."""
    def __init__(self, buffer: DataBuffer):
        self.spec = buffer.spec
        self.host_data = None

    def update_from_device(self, queue: cl.CommandQueue, cl_buffer: cl.Buffer):
        """Copy from device buffer to host, maintaining padding."""
        self.host_data = np.empty(self.spec.padded.values, dtype=self.spec.dtype)
        cl.enqueue_copy(queue, self.host_data, cl_buffer)

    @property
    def valid_slice(self):
        """Returns a view into just the non-padded data."""
        if isinstance(self.spec, LossBufferSpec):
            return self.host_data[:self.spec.logical.features[0], :self.spec.logical.batch]
        elif isinstance(self.spec, (InputBufferSpec, HiddenActBufferSpec, ExitProbsBufferSpec)):
            slices = (slice(0, self.spec.logical.batch),) + tuple(slice(0, f) for f in self.spec.logical.features)
            return self.host_data[slices]
        elif isinstance(self.spec, (TargetsBufferSpec, MaskBufferSpec)):
            return self.host_data[:self.spec.logical.batch]
        raise ValueError(f"Unsupported buffer spec type: {type(self.spec)}")

class BatchPadder:
    """Handles padding for batches of data using buffer specs."""
    def __init__(self, input_buffer_spec: InputBufferSpec, target_buffer_spec: TargetsBufferSpec):
        self.input_spec = input_buffer_spec
        self.target_spec = target_buffer_spec

    def pad_batch(self, X: np.ndarray, y_true: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Pad input and target tensors to fixed buffer sizes."""
        batch_size = X.shape[0]
        X_padded = np.zeros(self.input_spec.padded.values, dtype=SCALAR_NP_TYPE)
        X_padded[:batch_size, :X.shape[1]] = X
        y_padded = np.zeros(self.target_spec.padded.values, dtype=np.int32)
        y_padded[:batch_size] = y_true
        mask = np.zeros(self.input_spec.padded.values[0], dtype=SCALAR_NP_TYPE)
        mask[:batch_size] = SCALAR_NP_TYPE(1.0)
        return X_padded, y_padded, mask

class ParameterBufferManager:
    """Manages parameter buffers."""
    def __init__(self, context: cl.Context, device: cl.Device, work_manager: 'WorkManager'):
        self.context = context
        self.device = device
        self.work_manager = work_manager
        self.padding_ctx = PaddingContext.from_device(device)

    def create_parameter(self, name: str, real_shape: Tuple[int, ...], dtype: np.dtype) -> ParameterBuffer:
        """Create a parameter buffer."""
        spec = BufferSpec(name, real_shape, dtype, BufferType.PARAMETER)
        padded_shape = self._pad_parameter_shape(real_shape)
        spec.padded_shape = padded_shape
        param_buffer = self._allocate_buffer(name, padded_shape, dtype)
        return ParameterBuffer(spec, param_buffer)

    def _pad_parameter_shape(self, shape: Tuple[int, ...]) -> Tuple[int, ...]:
        """Pad dimensions to SIMD width."""
        simd_width = self.padding_ctx.simd_width
        return tuple((dim + simd_width - 1) // simd_width * simd_width for dim in shape)

    def _allocate_buffer(self, name: str, shape: Tuple[int, ...], dtype: np.dtype) -> cl.Buffer:
        """Allocate an OpenCL buffer."""
        size = np.prod(shape) * dtype().itemsize
        buffer = cl.Buffer(self.context, cl.mem_flags.READ_WRITE, size=size)
        self.work_manager.logical_resources[name] = (buffer, 0)
        return buffer

class TransferDirection(Enum):
    H2D = 1
    D2H = 2

class NodeType(Enum):
    MEMORY = 1
    COMPUTE = 2
    TRANSFER = 3
    SYNC = 4

class MemOpType(Enum):
    ALLOC = 1
    FREE = 2

class SyncType(Enum):
    HOST_SIGNAL = 1
    DEVICE_WAIT = 2

class AccessMode(Enum):
    LOCAL = 1
    SHARED_READ = 2
    EXCLUSIVE_WRITE_OVER = 3
    EXCLUSIVE_ZERO = 4
    EXCLUSIVE_UPDATE = 5

ACCESS_WRITE_MODES: Set[AccessMode] = {AccessMode.EXCLUSIVE_WRITE_OVER, AccessMode.EXCLUSIVE_ZERO, AccessMode.EXCLUSIVE_UPDATE}

class WorkManager:
    """Manages the execution graph and OpenCL resources."""
    def __init__(self, context: cl.Context):
        self.context = context
        self.execution_graph = nx.MultiDiGraph()
        self.logical_resources: Dict[str, Tuple[cl.Buffer, int]] = {}
        self.hardware_queues: Dict[str, cl.CommandQueue] = {}
        self.node_id_counter: int = 0
        self.buffer_tracker = defaultdict(lambda: {'version': 0, 'alloc_node': None, 'free_node': None, 'metadata': None})
        self.user_events: Dict[int, cl.UserEvent] = {}
        self.last_writer: Dict[str, int] = {}
        self.pending_readers = defaultdict(set)

    @dataclass
    class ExecutionNode:
        """Represents a node in the execution graph."""
        __slots__ = ['uid', 'node_type', 'queue_type', 'access_map', 'dependencies', 'expected_versions', 'params', 'event']
        uid: int
        node_type: NodeType
        queue_type: str
        access_map: Dict[str, Set[AccessMode]]
        dependencies: Set[int]
        expected_versions: Dict[str, int]
        params: Dict
        event: Optional[cl.Event] = None

    def validate_access_modes(self, modes: Set[AccessMode]) -> None:
        """Validate access modes for a resource."""
        write_count = sum(1 for m in modes if m in ACCESS_WRITE_MODES)
        if write_count > 1:
            raise ValueError(f"Conflicting write modes: {modes}")
        if AccessMode.SHARED_READ in modes and modes & ACCESS_WRITE_MODES:
            raise ValueError("Cannot combine SHARED_READ with write modes")

    def create_node(self, node_type: NodeType, queue_type: str, access_map: Dict[str, Set[AccessMode]], operation_fn: Callable, params: Dict) -> int:
        """Create a new execution node."""
        node = self.ExecutionNode(self.node_id_counter, node_type, queue_type, access_map, set(), {}, params)
        node.execute = operation_fn

        for res, modes in access_map.items():
            self.validate_access_modes(modes)
            write_modes = modes & ACCESS_WRITE_MODES
            read_modes = modes - write_modes

            if write_modes:
                if res in self.last_writer:
                    last_writer_modes = self.execution_graph.nodes[self.last_writer[res]]['node'].access_map[res]
                    if not (AccessMode.EXCLUSIVE_WRITE_OVER in write_modes and 
                            AccessMode.EXCLUSIVE_WRITE_OVER in last_writer_modes):
                        node.dependencies.add(self.last_writer[res])
                for reader_uid in self.pending_readers[res]:
                    node.dependencies.add(reader_uid)
                self.pending_readers[res].clear()
                self.last_writer[res] = node.uid

            if AccessMode.SHARED_READ in read_modes or AccessMode.EXCLUSIVE_UPDATE in read_modes:
                if res in self.last_writer:
                    node.dependencies.add(self.last_writer[res])
                self.pending_readers[res].add(node.uid)

            if AccessMode.SHARED_READ in read_modes or AccessMode.EXCLUSIVE_UPDATE in read_modes:
                node.expected_versions[res] = self.logical_resources.get(res, (None, 0))[1]

        if node_type == NodeType.TRANSFER:
            buffer_name = list(access_map.keys())[0]
            alloc_node = self.buffer_tracker[buffer_name]['alloc_node']
            node.dependencies.add(alloc_node)

        self.execution_graph.add_node(node.uid, node=node)
        self.node_id_counter += 1
        return node.uid

    def create_mem_node(self, name: str, op_type: MemOpType, size: Optional[int] = None, buffer: Optional[cl.Buffer] = None, metadata: Optional[Dict] = None) -> int:
        """Create a memory operation node."""
        if op_type == MemOpType.ALLOC:
            def _alloc_fn(queue: cl.CommandQueue, wait_for: List[cl.Event]) -> None:
                buffer = cl.Buffer(self.context, cl.mem_flags.READ_WRITE, size=size)
                self.logical_resources[name] = (buffer, 0)
                self.buffer_tracker[name]['alloc_node'] = self.node_id_counter
                self.buffer_tracker[name]['metadata'] = metadata
                return None
            node = self.ExecutionNode(self.node_id_counter, NodeType.MEMORY, 'compute', {name: {AccessMode.EXCLUSIVE_WRITE_OVER}}, set(), {'op_type': op_type, 'size': size})
            node.execute = _alloc_fn
        elif op_type == MemOpType.FREE:
            def _free_fn(queue: cl.CommandQueue, wait_for: List[cl.Event]) -> None:
                buffer = self.logical_resources[name][0]
                buffer.release()
                del self.logical_resources[name]
                self.buffer_tracker[name]['free_node'] = self.node_id_counter
                self.buffer_tracker[name]['metadata'] = None
                return None
            node = self.ExecutionNode(self.node_id_counter, NodeType.MEMORY, 'compute', {name: {AccessMode.SHARED_READ}}, set(), {'op_type': op_type})
            node.execute = _free_fn
            for existing_uid in self.execution_graph.nodes:
                existing_node = self.execution_graph.nodes[existing_uid]['node']
                if name in existing_node.access_map:
                    node.dependencies.add(existing_uid)
        self.execution_graph.add_node(node.uid, node=node)
        self.node_id_counter += 1
        return node.uid

    def create_sync_node(self, sync_type: SyncType, dependencies: List[int] = []) -> int:
        """Create a synchronization node."""
        user_event = cl.UserEvent(self.context) if sync_type == SyncType.HOST_SIGNAL else None
        if sync_type == SyncType.DEVICE_WAIT:
            raise ValueError("DEVICE_WAIT requires a user_event, which should be handled differently.")
        node = self.ExecutionNode(self.node_id_counter, NodeType.SYNC, 'compute', {}, set(dependencies), {'sync_type': sync_type, 'user_event': user_event})
        self.execution_graph.add_node(node.uid, node=node)
        if user_event:
            self.user_events[node.uid] = user_event
        self.node_id_counter += 1
        return node.uid

    def get_sync_event(self, node_uid: int) -> Optional[cl.UserEvent]:
        """Get the sync event for a node."""
        return self.user_events.get(node_uid, None)

    def wait_for_sync(self, node_uid: int) -> None:
        """Wait for a synchronization event."""
        if node_uid in self.user_events:
            event = self.user_events.pop(node_uid)
            cl.wait_for_events([event])
        else:
            raise KeyError(f"No UserEvent found for node UID {node_uid}")

    def commit_workload(self) -> None:
        """Commit and execute the workload."""
        ordered_nodes = list(nx.topological_sort(self.execution_graph))
        completion_events: Dict[int, cl.Event] = {}
        for uid in ordered_nodes:
            node = self.execution_graph.nodes[uid]['node']
            queue = self.hardware_queues[node.queue_type]

            for res, modes in node.access_map.items():
                if res in self.logical_resources and (AccessMode.SHARED_READ in modes or AccessMode.EXCLUSIVE_UPDATE in modes):
                    current_version = self.logical_resources[res][1]
                    expected_version = node.expected_versions.get(res, 0)
                    if expected_version != current_version:
                        raise RuntimeError(f"Resource {res} version mismatch for node {uid}")

            wait_for = [completion_events[d] for d in node.dependencies if d in completion_events]
            if node.node_type == NodeType.COMPUTE:
                self._dispatch_compute(node, wait_for)
            elif node.node_type == NodeType.MEMORY:
                self._dispatch_memory(node, wait_for)
            elif node.node_type == NodeType.TRANSFER:
                self._dispatch_transfer(node, wait_for)
            elif node.node_type == NodeType.SYNC:
                self._dispatch_sync(node, wait_for)
            completion_events[uid] = node.event

            for res, modes in node.access_map.items():
                if res in self.logical_resources and modes & ACCESS_WRITE_MODES:
                    buffer, version = self.logical_resources[res]
                    self.logical_resources[res] = (buffer, version + 1)

    def _dispatch_compute(self, node: 'WorkManager.ExecutionNode', wait_for: List[cl.Event]) -> None:
        """Dispatch a compute node."""
        queue = self.hardware_queues[node.queue_type]
        node.event = node.execute(queue, wait_for=wait_for)

    def _dispatch_memory(self, node: 'WorkManager.ExecutionNode', wait_for: List[cl.Event]) -> None:
        """Dispatch a memory node."""
        node.execute(self.hardware_queues[node.queue_type], wait_for=wait_for)

    def _dispatch_transfer(self, node: 'WorkManager.ExecutionNode', wait_for: List[cl.Event]) -> None:
        """Dispatch a transfer node."""
        node.event = node.execute(self.hardware_queues[node.queue_type], wait_for=wait_for, node_params=node.params)

    def _dispatch_sync(self, node: 'WorkManager.ExecutionNode', wait_for: List[cl.Event]) -> None:
        """Dispatch a sync node."""
        queue = self.hardware_queues[node.queue_type]
        if node.params['sync_type'] == SyncType.DEVICE_WAIT:
            all_wait_for = wait_for + [node.params['user_event']]
            node.event = cl.enqueue_marker(queue, wait_for=all_wait_for)
        elif node.params['sync_type'] == SyncType.HOST_SIGNAL:
            node.event = cl.enqueue_marker(queue, wait_for=wait_for)
            def callback(event: cl.Event, status: int) -> None:
                if status == cl.command_execution_status.COMPLETE:
                    node.params['user_event'].set_status(cl.command_execution_status.COMPLETE)
            node.event.set_callback(cl.command_execution_status.COMPLETE, callback)

    def reset_graph(self) -> None:
        """Reset the execution graph."""
        self.execution_graph.clear()
        self.node_id_counter = 0
        self.user_events.clear()

class ParamManager:
    """Manages neural network parameters."""
    PARAM_TYPES: Dict[str, Dict[str, Union[Callable, Tuple[float, float]]]] = {
        'dense_weight': {
            'init': he_init,
            'post_pad_fn': lambda padded, sw: padded.T.reshape(padded.shape[1] // sw, padded.shape[0], sw)
        },
        'exit_weight': {
            'post_pad_fn': lambda padded, sw: padded.reshape(padded.shape[0], sw, padded.shape[1]//sw, padded.shape[2])
        },
        'temperature': {
            'init': lambda s: np.full(s, (MAX_TEMP + MIN_TEMP) / 2, SCALAR_NP_TYPE),
            'constraint': (MIN_TEMP, MAX_TEMP),
            'post_pad_fn': lambda padded, sw: padded
        },
        'bias_vector': {
            'init': lambda s: np.zeros(s, dtype=SCALAR_NP_TYPE),
            'post_pad_fn': lambda padded, sw: padded
        }
    }

    def __init__(self, context: cl.Context, parameter_manager: ParameterBufferManager):
        self.context = context
        self.parameter_manager = parameter_manager
        self.params: Dict[str, Dict] = {}
        self.buffers: Dict[str, ParameterBuffer] = {}
        self.grad_buffers: Dict[str, ParameterBuffer] = {}
        self.m1_buffers: Dict[str, ParameterBuffer] = {}
        self.m2_buffers: Dict[str, ParameterBuffer] = {}

    def register_parameter(self, name: str, shape: Tuple[int, ...], ptype: str, requires_grad: bool = True) -> None:
        """Register a parameter with its properties."""
        if name in self.buffers:
            raise ValueError(f"Parameter {name} already registered")
        if ptype == 'temperature' and (shape[0] != NUM_EXITS):
            raise ValueError("Temperature parameter shape must match NUM_EXITS")
        self.params[name] = {
            'shape': shape,
            'ptype': ptype,
            'requires_grad': requires_grad
        }

    def _create_buffers(self) -> None:
        """Create buffers for all registered parameters."""
        for name, spec in self.params.items():
            ptype = spec['ptype']
            handler = self.PARAM_TYPES[ptype]
            init_values = handler['init'](spec['shape'])
            param_buf = self.parameter_manager.create_parameter(name, spec['shape'], SCALAR_NP_TYPE)
            self.buffers[name] = param_buf
            padded = pad_tensor(init_values, self.parameter_manager.padding_ctx)
            post_pad_fn = handler.get('post_pad_fn', lambda x, sw: x)
            processed = post_pad_fn(padded, self.parameter_manager.padding_ctx.simd_width)
            event = cl.enqueue_copy(self.context.queue, param_buf.cl_buffer, processed)
            event.wait()
            if spec['requires_grad']:
                grad_buf = self.parameter_manager.create_parameter(f"grad_{name}", spec['shape'], SCALAR_NP_TYPE)
                m1_buf = self.parameter_manager.create_parameter(f"m1_{name}", spec['shape'], SCALAR_NP_TYPE)
                m2_buf = self.parameter_manager.create_parameter(f"m2_{name}", spec['shape'], SCALAR_NP_TYPE)
                self.grad_buffers[name] = grad_buf
                self.m1_buffers[name] = m1_buf
                self.m2_buffers[name] = m2_buf
                for buf in [grad_buf.cl_buffer, m1_buf.cl_buffer, m2_buf.cl_buffer]:
                    event = cl.enqueue_fill_buffer(self.context.queue, buf, SCALAR_NP_TYPE(0), 0, buf.size)
                    event.wait()

    def zero_gradients(self, work_manager: 'WorkManager') -> None:
        """Zero out all gradient buffers."""
        for name, spec in self.params.items():
            if spec['requires_grad']:
                grad_buffer = self.grad_buffers[name].cl_buffer
                def _zero_fn(queue: cl.CommandQueue, wait_for: List[cl.Event]) -> cl.Event:
                    return cl.enqueue_fill_buffer(queue, grad_buffer, SCALAR_NP_TYPE(0), 0, grad_buffer.size, wait_for=wait_for)
                work_manager.create_node(
                    node_type=NodeType.COMPUTE,
                    queue_type='compute',
                    access_map={f'grad_{name}': {AccessMode.EXCLUSIVE_WRITE_OVER}},
                    operation_fn=_zero_fn,
                    params={}
                )

class KernelExecutionRequest:
    """Represents a request to execute an OpenCL kernel."""
    def __init__(self, program: cl.Program, kernel_name: str, global_sizes: Tuple[int, ...], local_sizes: Tuple[int, ...]):
        self.program = program
        self.kernel = getattr(program, kernel_name)
        self.global_sizes = global_sizes
        self.local_sizes = local_sizes
        self.local_mem_reqs: Dict[int, Tuple[int, AccessMode]] = {}
        self.argument_bindings: Dict[int, Union[cl.Buffer, np.ndarray, np.int32, SCALAR_NP_TYPE]] = {}

    def set_local_mem_argument(self, arg_index: int, size_bytes: int, access: AccessMode = AccessMode.LOCAL) -> None:
        """Set a local memory argument."""
        self.local_mem_reqs[arg_index] = (size_bytes, access)

    def bind_argument(self, arg_index: int, arg: Union[cl.Buffer, np.ndarray, np.int32, SCALAR_NP_TYPE]) -> None:
        """Bind an argument to the kernel."""
        self.argument_bindings[arg_index] = arg

class KernelWrapper:
    """High-level interface for kernel execution."""
    def __init__(self, program: cl.Program, manager: WorkManager, global_step: int, compute_queue: cl.CommandQueue, data_manager: DataBufferManager, parameter_manager: ParameterBufferManager):
        self.program = program
        self.manager = manager
        self.queue = compute_queue
        self.data_manager = data_manager
        self.parameter_manager = parameter_manager
        self.mask: Optional[cl.Buffer] = None
        self.temperatures: Optional[cl.Buffer] = None
        self.padded_input_dim = self.data_manager.create_data_buffer('input_batch', (BATCH_SIZE, INPUT_DIM), SCALAR_NP_TYPE, BufferPurpose.INPUT_DATA).spec.padded.values[1]
        self.padded_hidden_dim = self.data_manager.create_data_buffer('hidden', (BATCH_SIZE, HIDDEN_DIM), SCALAR_NP_TYPE, BufferPurpose.HIDDEN_ACT).spec.padded.values[1]
        self.padded_output_classes = ((OUTPUT_CLASSES + self.data_manager.padding_ctx.simd_width - 1) // self.data_manager.padding_ctx.simd_width) * self.data_manager.padding_ctx.simd_width
        self.padded_batch_size = self.data_manager.create_data_buffer('input_batch', (BATCH_SIZE, INPUT_DIM), SCALAR_NP_TYPE, BufferPurpose.INPUT_DATA).spec.padded.values[0]
        self.input_dim = INPUT_DIM
        self.hidden_dim = HIDDEN_DIM
        self.output_classes = OUTPUT_CLASSES
        self.num_exits = NUM_EXITS
        self.adam_beta1 = SCALAR_NP_TYPE(ADAM_BETA1)
        self.adam_beta2 = SCALAR_NP_TYPE(ADAM_BETA2)
        self.learning_rate = SCALAR_NP_TYPE(LEARNING_RATE)
        self.min_temp = SCALAR_NP_TYPE(MIN_TEMP)
        self.max_temp = SCALAR_NP_TYPE(MAX_TEMP)
        self.epsilon = SCALAR_NP_TYPE(EPSILON)
        self.beta1_t = SCALAR_NP_TYPE(ADAM_BETA1 ** global_step)
        self.beta2_t = SCALAR_NP_TYPE(ADAM_BETA2 ** global_step)

    def set_mask(self, mask: cl.Buffer) -> 'KernelWrapper':
        """Set the mask buffer."""
        self.mask = mask
        return self

    def set_temps(self, temps: cl.Buffer) -> 'KernelWrapper':
        """Set the temperatures buffer."""
        self.temperatures = temps
        return self

    def forward_pass(self, global_sizes: Tuple[int, ...], local_sizes: Tuple[int, ...], input_buf: DataBuffer, weights_buf: ParameterBuffer, biases_buf: ParameterBuffer, hidden_buf: DataBuffer) -> int:
        """Enqueue the forward pass kernel."""
        req = KernelExecutionRequest(self.program, 'forward_pass', global_sizes, local_sizes)
        req.set_local_mem_argument(0, local_sizes[0] * SCALAR_SIZE, AccessMode.LOCAL)
        req.bind_argument(1, input_buf.cl_buffer)
        req.bind_argument(2, input_buf.mask_buffer)
        req.bind_argument(3, weights_buf.cl_buffer)
        req.bind_argument(4, biases_buf.cl_buffer)
        req.bind_argument(5, hidden_buf.cl_buffer)
        req.bind_argument(6, hidden_buf.mask_buffer)

        access_map = {
            input_buf.spec.name: {AccessMode.SHARED_READ},
            f"mask_{input_buf.spec.name}": {AccessMode.SHARED_READ},
            weights_buf.spec.name: {AccessMode.SHARED_READ},
            biases_buf.spec.name: {AccessMode.SHARED_READ},
            hidden_buf.spec.name: {AccessMode.EXCLUSIVE_WRITE_OVER},
            f"mask_{hidden_buf.spec.name}": {AccessMode.EXCLUSIVE_WRITE_OVER},
        }
        return self.manager.create_node(
            node_type=NodeType.COMPUTE,
            queue_type='compute',
            access_map=access_map,
            operation_fn=self._enqueue_kernel,
            params={'exec_req': req}
        )

    def compute_exit_probabilities(self, global_sizes: Tuple[int, ...], local_sizes: Tuple[int, ...], hidden_buf: DataBuffer, exit_weights_buf: ParameterBuffer, exit_biases_buf: ParameterBuffer, exit_probs_buf: DataBuffer, losses_buf: DataBuffer, targets_buf: DataBuffer, exit_idx: int) -> int:
        """Enqueue the exit probabilities kernel."""
        req = KernelExecutionRequest(self.program, 'compute_exit_probabilities', global_sizes, local_sizes)
        req.set_local_mem_argument(0, local_size * SCALAR_SIZE)
        req.bind_argument(1, hidden_buf.cl_buffer)
        req.bind_argument(2, hidden_buf.mask_buffer)
        req.bind_argument(3, exit_weights_buf.cl_buffer)
        req.bind_argument(4, exit_biases_buf.cl_buffer)
        req.bind_argument(5, exit_probs_buf.cl_buffer)
        req.bind_argument(6, exit_probs_buf.mask_buffer)
        req.bind_argument(7, losses_buf.cl_buffer)
        req.bind_argument(8, losses_buf.mask_buffer)
        req.bind_argument(9, targets_buf.cl_buffer)
        req.bind_argument(10, targets_buf.mask_buffer)
        req.bind_argument(11, np.int32(exit_idx))
        req.bind_argument(12, self.temperatures)
        req.bind_argument(13, np.int32(self.padded_batch_size))
        req.bind_argument(14, np.int32(self.hidden_dim))
        req.bind_argument(15, np.int32(self.output_classes))
        req.bind_argument(16, np.int32(self.padded_hidden_dim))
        req.bind_argument(17, np.int32(self.padded_output_classes))
        req.bind_argument(18, np.int32(self.num_exits))

        access_map = {
            hidden_buf.spec.name: {AccessMode.SHARED_READ},
            f"mask_{hidden_buf.spec.name}": {AccessMode.SHARED_READ},
            exit_weights_buf.spec.name: {AccessMode.SHARED_READ},
            exit_biases_buf.spec.name: {AccessMode.SHARED_READ},
            targets_buf.spec.name: {AccessMode.SHARED_READ},
            f"mask_{targets_buf.spec.name}": {AccessMode.SHARED_READ},
            'temps': {AccessMode.SHARED_READ},
            exit_probs_buf.spec.name: {AccessMode.EXCLUSIVE_WRITE_OVER},
            f"mask_{exit_probs_buf.spec.name}": {AccessMode.EXCLUSIVE_WRITE_OVER},
            losses_buf.spec.name: {AccessMode.EXCLUSIVE_WRITE_OVER},
            f"mask_{losses_buf.spec.name}": {AccessMode.EXCLUSIVE_WRITE_OVER}
        }

        return self.manager.create_node(
            node_type=NodeType.COMPUTE,
            queue_type='compute',
            access_map=access_map,
            operation_fn=self._enqueue_kernel,
            params={'exec_req': req}
        )

    def compute_gradients(self, global_sizes: Tuple[int, ...], local_sizes: Tuple[int, ...], input_buf: DataBuffer, hidden_buf: DataBuffer, exit_probs_buf: DataBuffer, exit_weights_buf: ParameterBuffer, grad_weights_buf: ParameterBuffer, grad_biases_buf: ParameterBuffer, grad_exit_weights_buf: ParameterBuffer, grad_exit_biases_buf: ParameterBuffer, targets_buf: DataBuffer) -> int:
        """Enqueue the gradients computation kernel."""
        req = KernelExecutionRequest(self.program, 'compute_gradients', global_sizes, local_sizes)
        req.set_local_mem_argument(0, HIDDEN_DIM * SCALAR_SIZE)
        req.bind_argument(1, input_buf.cl_buffer)
        req.bind_argument(2, input_buf.mask_buffer)
        req.bind_argument(3, hidden_buf.cl_buffer)
        req.bind_argument(4, hidden_buf.mask_buffer)
        req.bind_argument(5, exit_probs_buf.cl_buffer)
        req.bind_argument(6, exit_probs_buf.mask_buffer)
        req.bind_argument(7, exit_weights_buf.cl_buffer)
        req.bind_argument(8, grad_weights_buf.cl_buffer)
        req.bind_argument(9, grad_biases_buf.cl_buffer)
        req.bind_argument(10, grad_exit_weights_buf.cl_buffer)
        req.bind_argument(11, grad_exit_biases_buf.cl_buffer)
        req.bind_argument(12, targets_buf.cl_buffer)
        req.bind_argument(13, targets_buf.mask_buffer)
        req.bind_argument(14, self.temperatures)
        req.bind_argument(15, np.int32(self.input_dim))
        req.bind_argument(16, np.int32(self.hidden_dim))
        req.bind_argument(17, np.int32(self.output_classes))
        req.bind_argument(18, np.int32(self.padded_input_dim))
        req.bind_argument(19, np.int32(self.padded_hidden_dim))
        req.bind_argument(20, np.int32(self.padded_output_classes))
        req.bind_argument(21, np.int32(self.padded_batch_size))

        access_map = {
            input_buf.spec.name: {AccessMode.SHARED_READ},
            f"mask_{input_buf.spec.name}": {AccessMode.SHARED_READ},
            hidden_buf.spec.name: {AccessMode.SHARED_READ},
            f"mask_{hidden_buf.spec.name}": {AccessMode.SHARED_READ},
            exit_probs_buf.spec.name: {AccessMode.SHARED_READ},
            f"mask_{exit_probs_buf.spec.name}": {AccessMode.SHARED_READ},
            exit_weights_buf.spec.name: {AccessMode.SHARED_READ},
            targets_buf.spec.name: {AccessMode.SHARED_READ},
            f"mask_{targets_buf.spec.name}": {AccessMode.SHARED_READ},
            'temps': {AccessMode.SHARED_READ},
            grad_weights_buf.spec.name: {AccessMode.EXCLUSIVE_WRITE_OVER},
            grad_biases_buf.spec.name: {AccessMode.EXCLUSIVE_WRITE_OVER},
            grad_exit_weights_buf.spec.name: {AccessMode.EXCLUSIVE_WRITE_OVER},
            grad_exit_biases_buf.spec.name: {AccessMode.EXCLUSIVE_WRITE_OVER}
        }

        return self.manager.create_node(
            node_type=NodeType.COMPUTE,
            queue_type='compute',
            access_map=access_map,
            operation_fn=self._enqueue_kernel,
            params={'exec_req': req}
        )

    def compute_temp_gradients(self, global_sizes: Tuple[int, ...], local_sizes: Tuple[int, ...], exit_probs_buf: DataBuffer, targets_buf: DataBuffer, grad_temps_buf: ParameterBuffer) -> int:
        """Enqueue the temperature gradients computation kernel."""
        req = KernelExecutionRequest(self.program, 'compute_temp_gradients', global_sizes, local_sizes)
        req.bind_argument(0, exit_probs_buf.cl_buffer)
        req.bind_argument(1, exit_probs_buf.mask_buffer)
        req.bind_argument(2, targets_buf.cl_buffer)
        req.bind_argument(3, targets_buf.mask_buffer)
        req.bind_argument(4, grad_temps_buf.cl_buffer)
        req.bind_argument(5, self.temperatures)
        req.bind_argument(6, np.int32(self.output_classes))
        req.bind_argument(7, np.int32(self.padded_output_classes))
        req.bind_argument(8, np.int32(self.padded_batch_size))
        req.bind_argument(9, np.int32(self.num_exits))

        access_map = {
            exit_probs_buf.spec.name: {AccessMode.SHARED_READ},
            f"mask_{exit_probs_buf.spec.name}": {AccessMode.SHARED_READ},
            targets_buf.spec.name: {AccessMode.SHARED_READ},
            f"mask_{targets_buf.spec.name}": {AccessMode.SHARED_READ},
            'temps': {AccessMode.SHARED_READ},
            grad_temps_buf.spec.name: {AccessMode.EXCLUSIVE_WRITE_OVER}
        }

        return self.manager.create_node(
            node_type=NodeType.COMPUTE,
            queue_type='compute',
            access_map=access_map,
            operation_fn=self._enqueue_kernel,
            params={'exec_req': req}
        )

    def adam_update(self, global_sizes: Tuple[int, ...], local_sizes: Tuple[int, ...], grad_buf: ParameterBuffer, param_buf: ParameterBuffer, m1_buf: ParameterBuffer, m2_buf: ParameterBuffer, grad_name: str, param_name: str, m1_name: str, m2_name: str, total_params: int) -> int:
        """Enqueue the ADAM update kernel."""
        req = KernelExecutionRequest(self.program, 'adam_update', global_sizes, local_sizes)
        req.bind_argument(0, grad_buf.cl_buffer)
        req.bind_argument(1, param_buf.cl_buffer)
        req.bind_argument(2, m1_buf.cl_buffer)
        req.bind_argument(3, m2_buf.cl_buffer)
        req.bind_argument(4, self.adam_beta1)
        req.bind_argument(5, self.adam_beta2)
        req.bind_argument(6, self.beta1_t)
        req.bind_argument(7, self.beta2_t)
        req.bind_argument(8, self.learning_rate)
        req.bind_argument(9, self.epsilon)
        req.bind_argument(10, np.int32(total_params))

        access_map = {
            grad_buf.spec.name: {AccessMode.SHARED_READ},
            param_buf.spec.name: {AccessMode.EXCLUSIVE_UPDATE},
            m1_buf.spec.name: {AccessMode.EXCLUSIVE_UPDATE},
            m2_buf.spec.name: {AccessMode.EXCLUSIVE_UPDATE}
        }

        return self.manager.create_node(
            node_type=NodeType.COMPUTE,
            queue_type='compute',
            access_map=access_map,
            operation_fn=self._enqueue_kernel,
            params={'exec_req': req}
        )

    def clamp_temperatures(self, global_sizes: Tuple[int, ...], local_sizes: Tuple[int, ...], temps_buf: ParameterBuffer) -> int:
        """Enqueue the temperature clamping kernel."""
        req = KernelExecutionRequest(self.program, 'clamp_temperatures', global_sizes, local_sizes)
        req.bind_argument(0, temps_buf.cl_buffer)
        req.bind_argument(1, self.min_temp)
        req.bind_argument(2, self.max_temp)
        req.bind_argument(3, np.int32(self.num_exits))

        access_map = {
            temps_buf.spec.name: {AccessMode.EXCLUSIVE_UPDATE}
        }

        return self.manager.create_node(
            node_type=NodeType.COMPUTE,
            queue_type='compute',
            access_map=access_map,
            operation_fn=self._enqueue_kernel,
            params={'exec_req': req}
        )

    def _enqueue_kernel(self, queue: cl.CommandQueue, wait_for: List[cl.Event], node_params: Dict) -> cl.Event:
        """Enqueue a kernel execution."""
        req = node_params['exec_req']
        local_mem_args = {idx: cl.LocalMemory(size) for idx, (size, mode) in req.local_mem_reqs.items()}
        args = []
        for i in range(max(req.argument_bindings.keys() | local_mem_args.keys()) + 1):
            args.append(local_mem_args.get(i, req.argument_bindings.get(i)))
        return cl.enqueue_nd_range_kernel(queue, req.kernel, req.global_sizes, req.local_sizes, *args, wait_for=wait_for)

# OpenCL Setup
ctx: cl.Context = cl.create_some_context()
transfer_queue: cl.CommandQueue = cl.CommandQueue(ctx)
compute_queue: cl.CommandQueue = cl.CommandQueue(ctx, properties=cl.command_queue_properties.PROFILING_ENABLE)
device: cl.Device = ctx.devices[0]
device_limits: cl.Device = device

# Check for FP16 support if using half precision
if SCALAR_TYPE == "half" and "cl_khr_fp16" not in device.extensions:
    raise RuntimeError("Device does not support FP16")

# WorkManager Initialization
manager = WorkManager(ctx)
manager.hardware_queues = {'xfer': transfer_queue, 'compute': compute_queue}

# Buffer Managers Initialization
data_mgr = DataBufferManager(ctx, device, manager)
param_mgr = ParameterBufferManager(ctx, device, manager)

# ParamManager Initialization
pm = ParamManager(ctx, param_mgr)
pm.register_parameter('weights', (INPUT_DIM, HIDDEN_DIM), 'dense_weight')
pm.register_parameter('biases', (HIDDEN_DIM,), 'bias_vector')
pm.register_parameter('exit_weights', (NUM_EXITS, HIDDEN_DIM, OUTPUT_CLASSES), 'exit_weight')
pm.register_parameter('exit_biases', (NUM_EXITS, OUTPUT_CLASSES), 'bias_vector')
pm.register_parameter('temps', (NUM_EXITS,), 'temperature', requires_grad=True)

# Create parameter buffers
pm._create_buffers()

# Create data buffers with specific purposes
input_buf = data_mgr.create_data_buffer('input_batch', (BATCH_SIZE, INPUT_DIM), SCALAR_NP_TYPE, BufferPurpose.INPUT_DATA)
hidden_buf = data_mgr.create_data_buffer('hidden', (BATCH_SIZE, HIDDEN_DIM), SCALAR_NP_TYPE, BufferPurpose.HIDDEN_ACT)
exit_probs_buf = data_mgr.create_data_buffer('exit_probs', (NUM_EXITS, BATCH_SIZE, OUTPUT_CLASSES), SCALAR_NP_TYPE, BufferPurpose.EXIT_PROBS)
losses_buf = data_mgr.create_data_buffer('losses', (NUM_EXITS, BATCH_SIZE), SCALAR_NP_TYPE, BufferPurpose.LOSSES)
targets_buf = data_mgr.create_data_buffer('targets_batch', (BATCH_SIZE,), np.int32, BufferPurpose.TARGETS)
mask_buf = data_mgr.create_data_buffer('mask_batch', (BATCH_SIZE,), SCALAR_NP_TYPE, BufferPurpose.MASK)

# Commit initialization workload
manager.commit_workload()

# Data Preparation
iris = load_iris()
X: np.ndarray = iris.data.astype(SCALAR_NP_TYPE)
y_true: np.ndarray = iris.target.astype(np.int32)
scaler = StandardScaler()
X_normalized: np.ndarray = scaler.fit_transform(X).astype(SCALAR_NP_TYPE)

# Compile Kernels
kernel_src: List[str] = []
for fname in CL_HEADER_FILES + CL_KERNEL_FILES:
    try:
        with open(fname, 'r') as f:
            kernel_src.append(f.read())
    except FileNotFoundError:
        print(f"Warning: Kernel file {fname} not found. Please ensure all kernel files are present.")
        kernel_src.append("")
simd_width: int = data_mgr.padding_ctx.simd_width
build_opts: List[str] = [
    f"-D SCALAR_TYPE={CL_SCALAR_TYPE}",
    f"-D SIMD_WIDTH={simd_width}",
    f"-D USE_FAST_MATH=1"
]
program: cl.Program = cl.Program(ctx, "\n".join(kernel_src)).build(options=" ".join(build_opts))

# Initialize BatchPadder
batch_padder = BatchPadder(input_buf.spec, targets_buf.spec)

# HostViews for device-to-host transfers
losses_view = HostView(losses_buf)
exit_probs_view = HostView(exit_probs_buf)
temps_view = HostView(pm.buffers['temps'])

# Training Loop
global_step: int = 1
for epoch in range(EPOCHS):
    shuffled_indices: np.ndarray = np.random.permutation(len(X))
    num_batches: int = (len(X) + BATCH_SIZE - 1) // BATCH_SIZE
    epoch_loss: float = 0.0
    correct_predictions: int = 0
    for batch_idx in range(num_batches):
        manager.reset_graph()
        batch_start = batch_idx * BATCH_SIZE
        batch_end = min(batch_start + BATCH_SIZE, len(X))
        actual_batch_size = batch_end - batch_start
        batch_indices = shuffled_indices[batch_start:batch_end]
        X_batch = X_normalized[batch_indices]
        y_batch = y_true[batch_indices]
        X_padded, y_padded, mask = batch_padder.pad_batch(X_batch, y_batch)

        # Transfer nodes for input data
        def transfer_input_data(queue: cl.CommandQueue, wait_for: List[cl.Event]) -> cl.Event:
            return cl.enqueue_copy(queue, input_buf.cl_buffer, X_padded, wait_for=wait_for)
        def transfer_input_mask(queue: cl.CommandQueue, wait_for: List[cl.Event]) -> cl.Event:
            return cl.enqueue_copy(queue, input_buf.mask_buffer, mask, wait_for=wait_for)
        def transfer_targets_data(queue: cl.CommandQueue, wait_for: List[cl.Event]) -> cl.Event:
            return cl.enqueue_copy(queue, targets_buf.cl_buffer, y_padded, wait_for=wait_for)
        def transfer_targets_mask(queue: cl.CommandQueue, wait_for: List[cl.Event]) -> cl.Event:
            return cl.enqueue_copy(queue, targets_buf.mask_buffer, mask, wait_for=wait_for)
        input_transfer_data = manager.create_node(
            NodeType.TRANSFER, 'xfer',
            {input_buf.spec.name: {AccessMode.EXCLUSIVE_WRITE_OVER}},
            transfer_input_data, {}
        )
        input_transfer_mask = manager.create_node(
            NodeType.TRANSFER, 'xfer',
            {f"mask_{input_buf.spec.name}": {AccessMode.EXCLUSIVE_WRITE_OVER}},
            transfer_input_mask, {}
        )
        targets_transfer_data = manager.create_node(
            NodeType.TRANSFER, 'xfer',
            {targets_buf.spec.name: {AccessMode.EXCLUSIVE_WRITE_OVER}},
            transfer_targets_data, {}
        )
        targets_transfer_mask = manager.create_node(
            NodeType.TRANSFER, 'xfer',
            {f"mask_{targets_buf.spec.name}": {AccessMode.EXCLUSIVE_WRITE_OVER}},
            transfer_targets_mask, {}
        )

        kernel_wrapper = KernelWrapper(program, manager, global_step, compute_queue, data_mgr, param_mgr)
        kernel_wrapper.set_mask(mask_buf.cl_buffer).set_temps(pm.buffers['temps'].cl_buffer)
        pm.zero_gradients(manager)

        # Forward pass
        global_forward = (input_buf.spec.padded.values[0], HIDDEN_DIM)
        local_forward = optimal_local_sizes(global_forward, device_limits)
        forward_node = kernel_wrapper.forward_pass(
            global_forward, local_forward,
            input_buf, pm.buffers['weights'], pm.buffers['biases'], hidden_buf
        )
        manager.execution_graph.add_edge(input_transfer_data, forward_node)
        manager.execution_graph.add_edge(input_transfer_mask, forward_node)

        # Compute exit probabilities
        for exit_idx in range(NUM_EXITS):
            global_exit = (input_buf.spec.padded.values[0],)
            local_exit = optimal_local_sizes(global_exit, device_limits)
            exit_node = kernel_wrapper.compute_exit_probabilities(
                global_exit, local_exit,
                hidden_buf, pm.buffers['exit_weights'], pm.buffers['exit_biases'],
                exit_probs_buf, losses_buf, targets_buf, exit_idx
            )
            manager.execution_graph.add_edge(forward_node, exit_node)
            manager.execution_graph.add_edge(targets_transfer_data, exit_node)
            manager.execution_graph.add_edge(targets_transfer_mask, exit_node)

        # Compute gradients
        global_grad = (INPUT_DIM, HIDDEN_DIM)
        local_grad = optimal_local_sizes(global_grad, device_limits)
        grad_node = kernel_wrapper.compute_gradients(
            global_grad, local_grad,
            input_buf, hidden_buf, exit_probs_buf, pm.buffers['exit_weights'],
            pm.buffers['grad_weights'], pm.buffers['grad_biases'],
            pm.buffers['grad_exit_weights'], pm.buffers['grad_exit_biases'],
            targets_buf
        )
        manager.execution_graph.add_edge(input_transfer_data, grad_node)
        manager.execution_graph.add_edge(input_transfer_mask, grad_node)
        manager.execution_graph.add_edge(forward_node, grad_node)
        manager.execution_graph.add_edge(targets_transfer_data, grad_node)
        manager.execution_graph.add_edge(targets_transfer_mask, grad_node)

        # Compute temperature gradients
        global_temp_grad = (NUM_EXITS,)
        local_temp_grad = optimal_local_sizes(global_temp_grad, device_limits)
        temp_grad_node = kernel_wrapper.compute_temp_gradients(
            global_temp_grad, local_temp_grad,
            exit_probs_buf, targets_buf, pm.buffers['grad_temps']
        )
        manager.execution_graph.add_edge(targets_transfer_data, temp_grad_node)
        manager.execution_graph.add_edge(targets_transfer_mask, temp_grad_node)

        # ADAM update
        for param in ['weights', 'biases', 'exit_weights', 'exit_biases', 'temps']:
            if param in pm.params and pm.params[param]['requires_grad']:
                grad_buf = pm.buffers[f'grad_{param}']
                param_buf = pm.buffers[param]
                m1_buf = pm.buffers[f'm1_{param}']
                m2_buf = pm.buffers[f'm2_{param}']
                total_params = int(np.prod(param_buf.spec.padded_shape))
                global_adam = (total_params,)
                local_adam = optimal_local_sizes(global_adam, device_limits)
                adam_node = kernel_wrapper.adam_update(
                    global_adam, local_adam,
                    grad_buf, param_buf, m1_buf, m2_buf,
                    f'grad_{param}', param, f'm1_{param}', f'm2_{param}', total_params
                )
                manager.execution_graph.add_edge(grad_node if param != 'temps' else temp_grad_node, adam_node)
                if param == 'temps':
                    global_clamp = (NUM_EXITS,)
                    local_clamp = optimal_local_sizes(global_clamp, device_limits)
                    clamp_node = kernel_wrapper.clamp_temperatures(global_clamp, local_clamp, param_buf)
                    manager.execution_graph.add_edge(adam_node, clamp_node)

        # Device-to-host transfers using HostView
        def transfer_losses(queue: cl.CommandQueue, wait_for: List[cl.Event]) -> cl.Event:
            losses_view.update_from_device(queue, losses_buf.cl_buffer)
            return cl.enqueue_marker(queue, wait_for=wait_for)
        def transfer_exit_probs(queue: cl.CommandQueue, wait_for: List[cl.Event]) -> cl.Event:
            exit_probs_view.update_from_device(queue, exit_probs_buf.cl_buffer)
            return cl.enqueue_marker(queue, wait_for=wait_for)
        def transfer_temps(queue: cl.CommandQueue, wait_for: List[cl.Event]) -> cl.Event:
            temps_view.update_from_device(queue, pm.buffers['temps'].cl_buffer)
            return cl.enqueue_marker(queue, wait_for=wait_for)
        losses_transfer = manager.create_node(
            NodeType.TRANSFER, 'xfer',
            {losses_buf.spec.name: {AccessMode.SHARED_READ}},
            transfer_losses, {}
        )
        exit_probs_transfer = manager.create_node(
            NodeType.TRANSFER, 'xfer',
            {exit_probs_buf.spec.name: {AccessMode.SHARED_READ}},
            transfer_exit_probs, {}
        )
        temps_transfer = manager.create_node(
            NodeType.TRANSFER, 'xfer',
            {'temps': {AccessMode.SHARED_READ}},
            transfer_temps, {}
        )

        # Sync node
        sync_uid = manager.create_sync_node(SyncType.HOST_SIGNAL, dependencies=[losses_transfer, exit_probs_transfer, temps_transfer])
        manager.commit_workload()
        manager.wait_for_sync(sync_uid)

        # Process results using HostView valid slices with casting to float32
        valid_losses = losses_view.valid_slice.T.astype(np.float32)  # Shape (actual_batch, NUM_EXITS)
        exit_losses = []
        for exit_idx in range(NUM_EXITS):
            masked_losses = valid_losses[:, exit_idx] * mask[:actual_batch_size].astype(np.float32)
            total_loss = np.sum(masked_losses)
            num_valid = np.sum(mask[:actual_batch_size].astype(np.float32))
            if num_valid > 0:
                exit_loss = total_loss / num_valid
                exit_losses.append(float(exit_loss))
            else:
                exit_losses.append(0.0)
        print(f"Batch {batch_idx}: Per-exit Losses: {exit_losses}")

        valid_exit_probs = exit_probs_view.valid_slice.transpose(1, 0, 2).astype(np.float32)  # Shape (actual_batch, NUM_EXITS, OUTPUT_CLASSES)
        valid_temps = temps_view.valid_slice.astype(np.float32)  # Shape (NUM_EXITS,)

        confidences = np.array([valid_exit_probs[:, i, :].max(axis=1) ** (1 / (valid_temps[i] + 1e-8)) for i in range(NUM_EXITS)])
        weights = np.exp(confidences) / np.sum(np.exp(confidences), axis=0)
        ensemble_probs = np.einsum('ijk,j->ik', valid_exit_probs, weights)
        ensemble_probs /= np.sum(ensemble_probs, axis=1, keepdims=True) + 1e-8
        log_probs = np.log(ensemble_probs + 1e-8)
        batch_loss = -np.mean(log_probs[np.arange(actual_batch_size), y_batch])
        epoch_loss += batch_loss * actual_batch_size
        predicted_classes = np.argmax(ensemble_probs, axis=1)
        correct_predictions += np.sum(predicted_classes == y_batch)
        global_step += 1
    avg_loss = epoch_loss / len(X)
    train_acc = correct_predictions / len(X)
    print(f"Epoch {epoch:3d} | Loss: {avg_loss:.4f} | Acc: {train_acc:.1%}")
    print(f"Temperatures: {valid_temps}")
