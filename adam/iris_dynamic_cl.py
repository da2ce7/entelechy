import pyopencl as cl
import numpy as np
import math
from math import gcd
from sklearn.datasets import load_iris
from sklearn.preprocessing import StandardScaler
import networkx as nx
import re
from collections import deque, defaultdict
from enum import Enum
from dataclasses import dataclass

# Define data type sizes
FLOAT_SIZE = np.float32().itemsize
INT_SIZE = np.int32().itemsize

# Network Configuration
INPUT_DIM = 4
HIDDEN_DIM = 64
OUTPUT_CLASSES = 3
NUM_EXITS = 3
EPOCHS = 100
BATCH_SIZE = 128
ADAM_BETA1 = 0.9
ADAM_BETA2 = 0.999
LEARNING_RATE = 0.001
MIN_TEMP = 1.0e-3
MAX_TEMP = 10.0
EPSILON = 1.0e-8

# OpenCL kernel files (assumed to exist)
CL_KERNEL_FILES = [
    'kernel_forward_pass.cl',
    'kernel_backpropagation.cl',
    'kernel_adam_update.cl',
    'kernel_multi_exit.cl'
]

def lcm(a, b):
    return a * b // gcd(a, b)

def next_pow2(n):
    return 1 if n == 0 else 1 << (n - 1).bit_length()

def divisors(n):
    divs = set()
    for i in range(1, int(math.sqrt(n)) + 1):
        if n % i == 0:
            divs.add(i)
            divs.add(n//i)
    return sorted(divs, reverse=True)

def he_init(shape):
    if len(shape) == 2:
        fan_in = shape[0]
    elif len(shape) == 3:
        fan_in = shape[1]
    else:
        raise ValueError("Unsupported shape for He initialization")
    scale = np.sqrt(2.0 / fan_in)
    return np.random.normal(0, scale, shape).astype(np.float32)

def select_simd_width(device):
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

def optimal_local_sizes(global_sizes, device):
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
    simd_width: int
    min_alignment: int
    device: cl.Device
    padding_mode: str = 'simd_aware'

    @classmethod
    def from_device(cls, device):
        return cls(
            simd_width=select_simd_width(device),
            min_alignment=max(device.min_data_type_align_size, 16),
            device=device
        )

class PaddingStrategy:
    _strategies = {}

    @classmethod
    def register(cls, name):
        def decorator(func):
            cls._strategies[name] = func
            return func
        return decorator

    @classmethod
    def apply(cls, context, shape, dtype=np.float32):
        return cls._strategies[context.padding_mode](context, shape, dtype)

@PaddingStrategy.register('simd_aware')
def _simd_strategy(ctx, shape, dtype):
    item_size = np.dtype(dtype).itemsize
    alignment = lcm(ctx.min_alignment, ctx.simd_width * item_size)
    return tuple((dim + alignment - 1) // alignment * alignment for dim in shape)

def pad_tensor(data, context: PaddingContext, strategy='simd_aware'):
    orig_shape = data.shape
    padded_shape = PaddingStrategy.apply(context, orig_shape, data.dtype)
    padded = np.zeros(padded_shape, dtype=data.dtype)
    slices = tuple(slice(0, d) for d in orig_shape)
    padded[slices] = data
    return padded  # Only padding, no reshaping or permutation

class BatchPadder:
    def __init__(self, context):
        self.ctx = context

    def pad_batch(self, X, y):
        X_padded = pad_tensor(X, self.ctx)
        y_padded = pad_tensor(y, self.ctx, strategy='simd_aware')
        mask = pad_tensor(np.ones(len(X), dtype=np.float32), self.ctx)
        return X_padded, y_padded, mask

class TransferDirection(Enum):
    H2D = 1  # Host to Device
    D2H = 2  # Device to Host

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
    LOCAL = 1          # Temporary, undefined content
    SHARED_READ = 2    # Read-only, concurrent access OK
    EXCLUSIVE_WRITE_OVER = 3  # Overwrite, no dependency on prev writes
    EXCLUSIVE_ZERO = 4 # Must be zero-initialized
    EXCLUSIVE_UPDATE = 5 # Read-modify-write

ACCESS_WRITE_MODES = {AccessMode.EXCLUSIVE_WRITE_OVER, AccessMode.EXCLUSIVE_ZERO, AccessMode.EXCLUSIVE_UPDATE}

class WorkManager:
    def __init__(self, context):
        self.context = context
        self.execution_graph = nx.MultiDiGraph()
        self.logical_resources = {}
        self.hardware_queues = {}
        self.node_id_counter = 0
        self.buffer_tracker = defaultdict(lambda: {'version': 0, 'alloc_node': None, 'free_node': None, 'metadata': None})
        self.user_events = {}  # {node_uid: cl.UserEvent}
        self.last_writer = {}
        self.pending_readers = defaultdict(set)

    @dataclass
    class ExecutionNode:
        __slots__ = ['uid', 'node_type', 'queue_type', 'access_map', 'dependencies', 'expected_versions', 'params', 'event']
        uid: int
        node_type: NodeType
        queue_type: str
        access_map: dict  # {resource: set[AccessMode]}
        dependencies: set
        expected_versions: dict
        params: dict
        event: cl.Event = None

    def validate_access_modes(self, modes):
        write_count = sum(1 for m in modes if m in ACCESS_WRITE_MODES)
        if write_count > 1:
            raise ValueError(f"Conflicting write modes: {modes}")
        if AccessMode.SHARED_READ in modes and modes & ACCESS_WRITE_MODES:
            raise ValueError("Cannot combine SHARED_READ with write modes")

    def create_node(self, node_type, queue_type, access_map, operation_fn, params):
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
            buffer_name = list(access_map.keys())[0]  # Assuming single resource per transfer
            alloc_node = self.buffer_tracker[buffer_name]['alloc_node']
            node.dependencies.add(alloc_node)

        self.execution_graph.add_node(node.uid, node=node)
        self.node_id_counter += 1
        return node.uid

    def create_mem_node(self, name, op_type, size=None, buffer=None, metadata=None):
        if op_type == MemOpType.ALLOC:
            def _alloc_fn(queue, wait_for):
                buffer = cl.Buffer(self.context, cl.mem_flags.READ_WRITE, size=size)
                self.logical_resources[name] = (buffer, 0)
                self.buffer_tracker[name]['alloc_node'] = self.node_id_counter
                self.buffer_tracker[name]['metadata'] = metadata
                return None
            node = self.ExecutionNode(self.node_id_counter, NodeType.MEMORY, 'compute', {name: {AccessMode.EXCLUSIVE_WRITE_OVER}}, set(), {'op_type': op_type, 'size': size})
            node.execute = _alloc_fn
        elif op_type == MemOpType.FREE:
            def _free_fn(queue, wait_for):
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

    def create_sync_node(self, sync_type, dependencies=[]):
        user_event = None
        if sync_type == SyncType.HOST_SIGNAL:
            user_event = cl.UserEvent(self.context)
        elif sync_type == SyncType.DEVICE_WAIT:
            raise ValueError("DEVICE_WAIT requires a user_event, which should be handled differently.")
        node = self.ExecutionNode(self.node_id_counter, NodeType.SYNC, 'compute', {}, set(dependencies), {'sync_type': sync_type, 'user_event': user_event})
        self.execution_graph.add_node(node.uid, node=node)
        if user_event:
            self.user_events[node.uid] = user_event
        self.node_id_counter += 1
        return node.uid

    def get_sync_event(self, node_uid):
        return self.user_events.get(node_uid, None)

    def wait_for_sync(self, node_uid):
        if node_uid in self.user_events:
            event = self.user_events.pop(node_uid)
            cl.wait_for_events([event])
        else:
            raise KeyError(f"No UserEvent found for node UID {node_uid}")

    def commit_workload(self):
        ordered_nodes = list(nx.topological_sort(self.execution_graph))
        completion_events = {}
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

    def _dispatch_compute(self, node, wait_for):
        queue = self.hardware_queues[node.queue_type]
        node.event = node.execute(queue, wait_for=wait_for)

    def _dispatch_memory(self, node, wait_for):
        node.execute(self.hardware_queues[node.queue_type], wait_for=wait_for)

    def _dispatch_transfer(self, node, wait_for):
        node.event = node.execute(self.hardware_queues[node.queue_type], wait_for=wait_for, node_params=node.params)

    def _dispatch_sync(self, node, wait_for):
        queue = self.hardware_queues[node.queue_type]
        if node.params['sync_type'] == SyncType.DEVICE_WAIT:
            all_wait_for = wait_for + [node.params['user_event']]
            node.event = cl.enqueue_marker(queue, wait_for=all_wait_for)
        elif node.params['sync_type'] == SyncType.HOST_SIGNAL:
            node.event = cl.enqueue_marker(queue, wait_for=wait_for)
            def callback(event, status):
                if status == cl.command_execution_status.COMPLETE:
                    node.params['user_event'].set_status(cl.command_execution_status.COMPLETE)
            node.event.set_callback(cl.command_execution_status.COMPLETE, callback)

    def reset_graph(self):
        self.execution_graph.clear()
        self.node_id_counter = 0
        self.user_events.clear()

class BufferManager:
    def __init__(self, context, device, work_manager):
        self.ctx = context
        self.device = device
        self.work_manager = work_manager
        self.padding_ctx = PaddingContext.from_device(device)
        self.buffer_metadata = {}
        self.staging_pool = deque(maxlen=8)
        self.buffers = {}
        self.work_manager.logical_resources.update(self.buffers)
        self.staging_buffer_ownership = {}

    def acquire_buffer(self, name, real_shape, dtype, pad_strategy=None, access_mode=AccessMode.EXCLUSIVE_WRITE_OVER):
        strategy = pad_strategy or self.padding_ctx.padding_mode
        padded_shape = PaddingStrategy.apply(self.padding_ctx, real_shape, dtype)
        size = np.prod(padded_shape) * dtype().itemsize
        metadata = {
            'real_shape': real_shape,
            'padded_shape': padded_shape,
            'dtype': dtype,
            'strategy': strategy
        }
        alloc_node = self.work_manager.create_mem_node(name, MemOpType.ALLOC, size=size, metadata=metadata)
        self.buffer_metadata[name] = metadata

        if access_mode == AccessMode.EXCLUSIVE_ZERO:
            zero_node = self._enqueue_zero_init(name)
            self.work_manager.execution_graph.add_edge(alloc_node, zero_node)

        return alloc_node

    def _enqueue_zero_init(self, name):
        def zero_op(queue, wait_for):
            buffer = self.work_manager.logical_resources[name][0]
            return cl.enqueue_fill_buffer(queue, buffer, np.float32(0), 0, buffer.size, wait_for=wait_for)
        zero_node = self.work_manager.create_node(
            NodeType.COMPUTE, 'compute',
            {name: {AccessMode.EXCLUSIVE_WRITE_OVER}},
            zero_op, {}
        )
        return zero_node

    def release_buffer(self, name):
        if name in self.buffer_metadata:
            self.work_manager.create_mem_node(name, MemOpType.FREE)

    def get_dimensions(self, name):
        return (self.buffer_metadata[name]['real_shape'], self.buffer_metadata[name]['padded_shape'])

    def staged_transfer(self, host_data, buffer_name, is_device_to_host=False):
        metadata = self.buffer_metadata[buffer_name]
        padded_shape = metadata['padded_shape']
        real_shape = metadata['real_shape']
        dtype = metadata['dtype']
        size = np.prod(padded_shape) * dtype().itemsize
        staging_buf = self._get_staging_buffer(size)

        if not is_device_to_host:
            params = {
                'direction': TransferDirection.H2D,
                'source': host_data,
                'destination': buffer_name,
                'size': size,
                'staging_buffer': staging_buf
            }
            node_id = self.work_manager.create_node(
                node_type=NodeType.TRANSFER,
                queue_type='xfer',
                access_map={buffer_name: {AccessMode.EXCLUSIVE_WRITE_OVER}},
                operation_fn=self._execute_transfer,
                params=params
            )
            self.staging_buffer_ownership[staging_buf] = node_id
            return node_id
        else:
            host_array = np.empty(padded_shape, dtype=dtype)
            params = {
                'direction': TransferDirection.D2H,
                'source': buffer_name,
                'destination': host_array,
                'size': size,
                'staging_buffer': staging_buf
            }
            node_id = self.work_manager.create_node(
                node_type=NodeType.TRANSFER,
                queue_type='xfer',
                access_map={buffer_name: {AccessMode.SHARED_READ}},
                operation_fn=self._execute_transfer,
                params=params
            )
            self.staging_buffer_ownership[staging_buf] = node_id
            return node_id, host_array

    def _execute_transfer(self, queue, wait_for, node_params):
        params = node_params
        staging_buf = params['staging_buffer']
        if staging_buf in self.staging_buffer_ownership and self.staging_buffer_ownership[staging_buf] != params.get('node_id', self.work_manager.node_id_counter - 1):
            raise RuntimeError("Staging buffer is already in use by another transfer")
        if params['direction'] == TransferDirection.H2D:
            buffer = self.work_manager.logical_resources[params['destination']][0]
            event1 = cl.enqueue_copy(queue, staging_buf, params['source'], wait_for=wait_for)
            event2 = cl.enqueue_copy(queue, buffer, staging_buf, wait_for=[event1])
            return event2
        elif params['direction'] == TransferDirection.D2H:
            buffer = self.work_manager.logical_resources[params['source']][0]
            event1 = cl.enqueue_copy(queue, staging_buf, buffer, wait_for=wait_for)
            event2 = cl.enqueue_copy(queue, params['destination'], staging_buf, wait_for=[event1])
            return event2

    def _get_staging_buffer(self, size):
        for buf in self.staging_pool:
            if buf.size >= size and buf not in self.staging_buffer_ownership:
                return buf
        buf = cl.Buffer(self.ctx, cl.mem_flags.READ_WRITE | cl.mem_flags.ALLOC_HOST_PTR, size=size)
        self.staging_pool.append(buf)
        return buf

    def validate_memory(self):
        padded_sizes = [
            np.prod(meta['padded_shape']) * np.dtype(meta['dtype']).itemsize
            for meta in self.buffer_metadata.values()
        ]
        total_buffer_alloc = sum(padded_sizes)
        staging_pool_alloc = sum(buf.size for buf in self.staging_pool)
        total_alloc = total_buffer_alloc + staging_pool_alloc
        device_max = self.device.global_mem_size
        if total_alloc / device_max > 0.8:
            raise MemoryError(
                f"Used {total_alloc / 1024**2:.2f}MB of {device_max / 1024**2:.2f}MB device memory"
            )

class ParamManager:
    PARAM_TYPES = {
        'dense_weight': {
            'init': he_init,
            'post_pad_fn': lambda padded, sw: padded.T.reshape(padded.shape[1] // sw, padded.shape[0], sw)
        },
        'exit_weight': {
            'init': he_init,
            'post_pad_fn': lambda padded, sw: padded.transpose(0, 2, 1).reshape(padded.shape[0], padded.shape[2] // sw, padded.shape[1], sw)
        },
        'temperature': {
            'init': lambda s: np.ones(s, dtype=np.float32) * (MAX_TEMP + MIN_TEMP) / 2,
            'constraint': (MIN_TEMP, MAX_TEMP),
            'post_pad_fn': lambda padded, sw: padded
        },
        'bias_vector': {
            'init': lambda s: np.zeros(s, dtype=np.float32),
            'post_pad_fn': lambda padded, sw: padded
        }
    }

    def __init__(self, context, buffer_manager):
        self.ctx = context
        self.buffer_manager = buffer_manager
        self.padding_ctx = PaddingContext.from_device(buffer_manager.device)
        self.params = {}
        self.buffers = {}
        self.grad_buffers = {}
        self.m1_buffers = {}
        self.m2_buffers = {}

    def register_parameter(self, name, shape, ptype, requires_grad=True):
        if name in self.buffers:
            raise ValueError(f"Parameter {name} already registered")
        if ptype == 'temperature' and (shape[0] != NUM_EXITS):
            raise ValueError("Temperature parameter shape must match NUM_EXITS")
        self.params[name] = {
            'shape': shape,
            'ptype': ptype,
            'requires_grad': requires_grad
        }

    def _create_buffers(self):
        for name, spec in self.params.items():
            ptype = spec['ptype']
            handler = self.PARAM_TYPES[ptype]
            init_values = handler['init'](spec['shape'])
            padded = pad_tensor(init_values, self.padding_ctx)
            post_pad_fn = handler.get('post_pad_fn', lambda x, sw: x)
            processed = post_pad_fn(padded, self.padding_ctx.simd_width)
            self.buffer_manager.acquire_buffer(name, spec['shape'], np.float32)
            self.buffer_manager.staged_transfer(processed, name, is_device_to_host=False)
            if spec['requires_grad']:
                self.buffer_manager.acquire_buffer(f"grad_{name}", spec['shape'], np.float32, access_mode=AccessMode.EXCLUSIVE_ZERO)
                self.buffer_manager.acquire_buffer(f"m1_{name}", spec['shape'], np.float32, access_mode=AccessMode.EXCLUSIVE_ZERO)
                self.buffer_manager.acquire_buffer(f"m2_{name}", spec['shape'], np.float32, access_mode=AccessMode.EXCLUSIVE_ZERO)

    def set_buffers(self, logical_resources):
        for name in self.params:
            self.buffers[name] = logical_resources[name][0]
            if self.params[name]['requires_grad']:
                self.grad_buffers[name] = logical_resources[f"grad_{name}"][0]
                self.m1_buffers[name] = logical_resources[f"m1_{name}"][0]
                self.m2_buffers[name] = logical_resources[f"m2_{name}"][0]

    def zero_gradients(self, work_manager):
        for name, spec in self.params.items():
            if spec['requires_grad']:
                grad_buffer = self.grad_buffers[name]
                def _zero_fn(queue, wait_for):
                    return cl.enqueue_fill_buffer(queue, grad_buffer, np.float32(0), 0, grad_buffer.size, wait_for=wait_for)
                work_manager.create_node(
                    node_type=NodeType.COMPUTE,
                    queue_type='compute',
                    access_map={f'grad_{name}': {AccessMode.EXCLUSIVE_WRITE_OVER}},
                    operation_fn=_zero_fn,
                    params={}
                )

class KernelExecutionRequest:
    def __init__(self, program, kernel_name, global_sizes, local_sizes):
        self.program = program
        self.kernel = getattr(program, kernel_name)
        self.global_sizes = global_sizes
        self.local_sizes = local_sizes
        self.local_mem_reqs = {}  # {arg_index: (size_bytes, access_mode)}
        self.argument_bindings = {}

    def set_local_mem_argument(self, arg_index, size_bytes, access=AccessMode.LOCAL):
        self.local_mem_reqs[arg_index] = (size_bytes, access)

    def bind_argument(self, arg_index, arg):
        self.argument_bindings[arg_index] = arg

class KernelWrapper:
    def __init__(self, program, manager, global_step, compute_queue, buffer_manager):
        self.program = program
        self.manager = manager
        self.queue = compute_queue
        self.buffer_manager = buffer_manager
        self.mask = None
        self.temperatures = None
        self.padded_input_dim = self.buffer_manager.buffer_metadata['input_batch']['padded_shape'][1]
        self.padded_hidden_dim = self.buffer_manager.buffer_metadata['hidden']['padded_shape'][0]
        self.padded_output_classes = ((OUTPUT_CLASSES + self.buffer_manager.padding_ctx.simd_width - 1) // self.buffer_manager.padding_ctx.simd_width) * self.buffer_manager.padding_ctx.simd_width
        self.padded_batch_size = self.buffer_manager.buffer_metadata['input_batch']['padded_shape'][0]
        self.input_dim = INPUT_DIM
        self.hidden_dim = HIDDEN_DIM
        self.output_classes = OUTPUT_CLASSES
        self.num_exits = NUM_EXITS
        self.adam_beta1 = np.float32(ADAM_BETA1)
        self.adam_beta2 = np.float32(ADAM_BETA2)
        self.learning_rate = np.float32(LEARNING_RATE)
        self.min_temp = np.float32(MIN_TEMP)
        self.max_temp = np.float32(MAX_TEMP)
        self.epsilon = np.float32(EPSILON)
        self.beta1_t = np.float32(ADAM_BETA1 ** global_step)
        self.beta2_t = np.float32(ADAM_BETA2 ** global_step)

    def set_mask(self, mask):
        self.mask = mask
        return self

    def set_temps(self, temps):
        self.temperatures = temps
        return self

    def forward_pass(self, global_sizes, local_sizes, input_buf, weights_buf, biases_buf, hidden_buf, actual_batch_size):
        req = KernelExecutionRequest(self.program, 'forward_pass', global_sizes, local_sizes)
        req.set_local_mem_argument(0, local_sizes[0] * FLOAT_SIZE, AccessMode.LOCAL)
        req.bind_argument(1, input_buf)
        req.bind_argument(2, weights_buf)
        req.bind_argument(3, biases_buf)
        req.bind_argument(4, hidden_buf)
        req.bind_argument(5, np.int32(actual_batch_size))
        return self.manager.create_node(
            node_type=NodeType.COMPUTE,
            queue_type='compute',
            access_map={
                'input_batch': {AccessMode.SHARED_READ},
                'weights': {AccessMode.SHARED_READ},
                'biases': {AccessMode.SHARED_READ},
                'mask_batch': {AccessMode.SHARED_READ},
                'hidden': {AccessMode.EXCLUSIVE_WRITE_OVER}
            },
            operation_fn=self._enqueue_kernel,
            params={'exec_req': req}
        )

    def compute_exit_probabilities(self, global_sizes, local_sizes, hidden_buf, exit_weights_buf, exit_biases_buf, exit_probs_buf, losses_buf, targets_buf, exit_idx, actual_batch_size):
        req = KernelExecutionRequest(self.program, 'compute_exit_probabilities', global_sizes, local_sizes)
        req.bind_argument(0, hidden_buf)
        req.bind_argument(1, exit_weights_buf)
        req.bind_argument(2, exit_biases_buf)
        req.bind_argument(3, exit_probs_buf)
        req.bind_argument(4, losses_buf)
        req.bind_argument(5, targets_buf)
        req.bind_argument(6, np.int32(exit_idx))
        req.bind_argument(7, self.temperatures)
        req.bind_argument(8, np.int32(self.hidden_dim))
        req.bind_argument(9, np.int32(self.output_classes))
        req.bind_argument(10, np.int32(self.padded_hidden_dim))
        req.bind_argument(11, np.int32(self.padded_output_classes))
        req.bind_argument(12, np.int32(self.padded_batch_size))
        req.bind_argument(13, np.int32(actual_batch_size))
        return self.manager.create_node(
            node_type=NodeType.COMPUTE,
            queue_type='compute',
            access_map={
                'hidden': {AccessMode.SHARED_READ},
                'exit_weights': {AccessMode.SHARED_READ},
                'exit_biases': {AccessMode.SHARED_READ},
                'targets_batch': {AccessMode.SHARED_READ},
                'temps': {AccessMode.SHARED_READ},
                'exit_probs': {AccessMode.EXCLUSIVE_WRITE_OVER},
                'losses': {AccessMode.EXCLUSIVE_WRITE_OVER}
            },
            operation_fn=self._enqueue_kernel,
            params={'exec_req': req}
        )

    def compute_gradients(self, global_sizes, local_sizes, input_buf, hidden_buf, exit_probs_buf, exit_weights_buf, grad_weights_buf, grad_biases_buf, grad_exit_weights_buf, grad_exit_biases_buf, targets_buf, actual_batch_size):
        req = KernelExecutionRequest(self.program, 'compute_gradients', global_sizes, local_sizes)
        req.bind_argument(0, input_buf)
        req.bind_argument(1, hidden_buf)
        req.bind_argument(2, exit_probs_buf)
        req.bind_argument(3, exit_weights_buf)
        req.bind_argument(4, grad_weights_buf)
        req.bind_argument(5, grad_biases_buf)
        req.bind_argument(6, grad_exit_weights_buf)
        req.bind_argument(7, grad_exit_biases_buf)
        req.bind_argument(8, targets_buf)
        req.bind_argument(9, self.temperatures)
        req.bind_argument(10, np.int32(self.input_dim))
        req.bind_argument(11, np.int32(self.hidden_dim))
        req.bind_argument(12, np.int32(self.output_classes))
        req.bind_argument(13, np.int32(self.padded_input_dim))
        req.bind_argument(14, np.int32(self.padded_hidden_dim))
        req.bind_argument(15, np.int32(self.padded_output_classes))
        req.bind_argument(16, np.int32(self.padded_batch_size))
        req.bind_argument(17, np.int32(actual_batch_size))
        req.bind_argument(18, np.int32(self.num_exits))
        return self.manager.create_node(
            node_type=NodeType.COMPUTE,
            queue_type='compute',
            access_map={
                'input_batch': {AccessMode.SHARED_READ},
                'hidden': {AccessMode.SHARED_READ},
                'exit_probs': {AccessMode.SHARED_READ},
                'exit_weights': {AccessMode.SHARED_READ},
                'targets_batch': {AccessMode.SHARED_READ},
                'temps': {AccessMode.SHARED_READ},
                'grad_weights': {AccessMode.EXCLUSIVE_WRITE_OVER},
                'grad_biases': {AccessMode.EXCLUSIVE_WRITE_OVER},
                'grad_exit_weights': {AccessMode.EXCLUSIVE_WRITE_OVER},
                'grad_exit_biases': {AccessMode.EXCLUSIVE_WRITE_OVER}
            },
            operation_fn=self._enqueue_kernel,
            params={'exec_req': req}
        )

    def compute_temp_gradients(self, global_sizes, local_sizes, exit_probs_buf, targets_buf, grad_temps_buf, actual_batch_size):
        req = KernelExecutionRequest(self.program, 'compute_temp_gradients', global_sizes, local_sizes)
        req.bind_argument(0, exit_probs_buf)
        req.bind_argument(1, targets_buf)
        req.bind_argument(2, grad_temps_buf)
        req.bind_argument(3, self.temperatures)
        req.bind_argument(4, np.int32(self.output_classes))
        req.bind_argument(5, np.int32(self.padded_output_classes))
        req.bind_argument(6, np.int32(self.padded_batch_size))
        req.bind_argument(7, np.int32(actual_batch_size))
        req.bind_argument(8, np.int32(self.num_exits))
        return self.manager.create_node(
            node_type=NodeType.COMPUTE,
            queue_type='compute',
            access_map={
                'exit_probs': {AccessMode.SHARED_READ},
                'targets_batch': {AccessMode.SHARED_READ},
                'temps': {AccessMode.SHARED_READ},
                'grad_temps': {AccessMode.EXCLUSIVE_WRITE_OVER}
            },
            operation_fn=self._enqueue_kernel,
            params={'exec_req': req}
        )

    def adam_update(self, global_sizes, local_sizes, grad_buf, param_buf, m1_buf, m2_buf, grad_name, param_name, m1_name, m2_name, total_params):
        req = KernelExecutionRequest(self.program, 'adam_update', global_sizes, local_sizes)
        req.bind_argument(0, grad_buf)
        req.bind_argument(1, param_buf)
        req.bind_argument(2, m1_buf)
        req.bind_argument(3, m2_buf)
        req.bind_argument(4, self.adam_beta1)
        req.bind_argument(5, self.adam_beta2)
        req.bind_argument(6, self.beta1_t)
        req.bind_argument(7, self.beta2_t)
        req.bind_argument(8, self.learning_rate)
        req.bind_argument(9, self.epsilon)
        req.bind_argument(10, np.int32(total_params))
        return self.manager.create_node(
            node_type=NodeType.COMPUTE,
            queue_type='compute',
            access_map={
                grad_name: {AccessMode.SHARED_READ},
                param_name: {AccessMode.EXCLUSIVE_UPDATE},
                m1_name: {AccessMode.EXCLUSIVE_UPDATE},
                m2_name: {AccessMode.EXCLUSIVE_UPDATE}
            },
            operation_fn=self._enqueue_kernel,
            params={'exec_req': req}
        )

    def clamp_temperatures(self, global_sizes, local_sizes, temps_buf):
        req = KernelExecutionRequest(self.program, 'clamp_temperatures', global_sizes, local_sizes)
        req.bind_argument(0, temps_buf)
        req.bind_argument(1, self.min_temp)
        req.bind_argument(2, self.max_temp)
        req.bind_argument(3, np.int32(self.num_exits))
        return self.manager.create_node(
            node_type=NodeType.COMPUTE,
            queue_type='compute',
            access_map={'temps': {AccessMode.EXCLUSIVE_UPDATE}},
            operation_fn=self._enqueue_kernel,
            params={'exec_req': req}
        )

    def _enqueue_kernel(self, queue, wait_for, node_params):
        req = node_params['exec_req']
        local_mem_args = {idx: cl.LocalMemory(size) for idx, (size, mode) in req.local_mem_reqs.items()}
        args = []
        for i in range(max(req.argument_bindings.keys() | local_mem_args.keys()) + 1):
            args.append(local_mem_args.get(i, req.argument_bindings.get(i)))
        return cl.enqueue_nd_range_kernel(queue, req.kernel, req.global_sizes, req.local_sizes, *args, wait_for=wait_for)

# OpenCL Setup
ctx = cl.create_some_context()
transfer_queue = cl.CommandQueue(ctx)
compute_queue = cl.CommandQueue(ctx, properties=cl.command_queue_properties.PROFILING_ENABLE)
device = ctx.devices[0]
device_limits = device

# WorkManager Initialization
manager = WorkManager(ctx)
manager.hardware_queues = {'xfer': transfer_queue, 'compute': compute_queue}

# BufferManager Initialization
buffer_mgr = BufferManager(ctx, device, manager)

# ParamManager Initialization
pm = ParamManager(ctx, buffer_mgr)
pm.register_parameter('weights', (INPUT_DIM, HIDDEN_DIM), 'dense_weight')
pm.register_parameter('biases', (HIDDEN_DIM,), 'bias_vector')
pm.register_parameter('exit_weights', (NUM_EXITS, HIDDEN_DIM, OUTPUT_CLASSES), 'exit_weight')
pm.register_parameter('exit_biases', (NUM_EXITS, OUTPUT_CLASSES), 'bias_vector')
pm.register_parameter('temps', (NUM_EXITS,), 'temperature', requires_grad=True)

# Additional Buffers
buffer_mgr.acquire_buffer('input_batch', (BATCH_SIZE, INPUT_DIM), np.float32)
buffer_mgr.acquire_buffer('hidden', (BATCH_SIZE, HIDDEN_DIM), np.float32)
buffer_mgr.acquire_buffer('exit_probs', (BATCH_SIZE, NUM_EXITS, OUTPUT_CLASSES), np.float32)
buffer_mgr.acquire_buffer('losses', (NUM_EXITS * BATCH_SIZE,), np.float32)
buffer_mgr.acquire_buffer('targets_batch', (BATCH_SIZE,), np.int32)
buffer_mgr.acquire_buffer('mask_batch', (BATCH_SIZE,), np.float32)

pm._create_buffers()

# Commit initialization workload
manager.commit_workload()

# Set buffer references
pm.set_buffers(manager.logical_resources)
for name in buffer_mgr.buffer_metadata:
    buffer_mgr.buffers[name] = manager.logical_resources[name][0]

# Data Preparation
iris = load_iris()
X = iris.data.astype(np.float32)
y_true = iris.target.astype(np.int32)
scaler = StandardScaler()
X_normalized = scaler.fit_transform(X)

# Compile Kernels
kernel_src = []
for fname in CL_KERNEL_FILES:
    try:
        with open(fname, 'r') as f:
            kernel_src.append(f.read())
    except FileNotFoundError:
        print(f"Warning: Kernel file {fname} not found. Please ensure all kernel files are present.")
        kernel_src.append("")
simd_width = buffer_mgr.padding_ctx.simd_width
build_opts = [
    f"-D VECTOR_TYPE={'float' + str(simd_width) if simd_width > 1 else 'float'}",
    f"-D SIMD_WIDTH={simd_width}",
    f"-D LOSS_STRIDE={buffer_mgr.buffer_metadata['losses']['padded_shape'][0] // NUM_EXITS}",
    f"-D USE_FAST_MATH=1"
]
program = cl.Program(ctx, "\n".join(kernel_src)).build(options=" ".join(build_opts))

# Training Loop
global_step = 1
batch_padder = BatchPadder(buffer_mgr.padding_ctx)
for epoch in range(EPOCHS):
    shuffled_indices = np.random.permutation(len(X))
    num_batches = (len(X) + BATCH_SIZE - 1) // BATCH_SIZE
    epoch_loss = 0.0
    correct_predictions = 0
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
        input_transfer = buffer_mgr.staged_transfer(X_padded, 'input_batch')
        targets_transfer = buffer_mgr.staged_transfer(y_padded, 'targets_batch')
        mask_transfer = buffer_mgr.staged_transfer(mask, 'mask_batch')

        kernel_wrapper = KernelWrapper(program, manager, global_step, compute_queue, buffer_mgr)
        kernel_wrapper.set_mask(buffer_mgr.buffers['mask_batch']).set_temps(pm.buffers['temps'])
        pm.zero_gradients(manager)

        # Forward pass node with dependencies
        global_forward = (buffer_mgr.buffer_metadata['input_batch']['padded_shape'][0], HIDDEN_DIM)
        local_forward = optimal_local_sizes(global_forward, device_limits)
        forward_node = kernel_wrapper.forward_pass(
            global_forward, local_forward,
            buffer_mgr.buffers['input_batch'], pm.buffers['weights'],
            pm.buffers['biases'], buffer_mgr.buffers['hidden'], actual_batch_size
        )
        manager.execution_graph.add_edge(input_transfer, forward_node)
        manager.execution_graph.add_edge(mask_transfer, forward_node)

        for exit_idx in range(NUM_EXITS):
            global_exit = (buffer_mgr.buffer_metadata['input_batch']['padded_shape'][0],)
            local_exit = optimal_local_sizes(global_exit, device_limits)
            exit_node = kernel_wrapper.compute_exit_probabilities(
                global_exit, local_exit,
                buffer_mgr.buffers['hidden'], pm.buffers['exit_weights'],
                pm.buffers['exit_biases'], buffer_mgr.buffers['exit_probs'],
                buffer_mgr.buffers['losses'], buffer_mgr.buffers['targets_batch'],
                exit_idx, actual_batch_size
            )
            manager.execution_graph.add_edge(forward_node, exit_node)
            manager.execution_graph.add_edge(targets_transfer, exit_node)

        global_grad = (INPUT_DIM, HIDDEN_DIM)
        local_grad = optimal_local_sizes(global_grad, device_limits)
        grad_node = kernel_wrapper.compute_gradients(
            global_grad, local_grad,
            buffer_mgr.buffers['input_batch'], buffer_mgr.buffers['hidden'],
            buffer_mgr.buffers['exit_probs'], pm.buffers['exit_weights'],
            pm.grad_buffers['weights'], pm.grad_buffers['biases'],
            pm.grad_buffers['exit_weights'], pm.grad_buffers['exit_biases'],
            buffer_mgr.buffers['targets_batch'], actual_batch_size
        )
        manager.execution_graph.add_edge(input_transfer, grad_node)
        manager.execution_graph.add_edge(forward_node, grad_node)
        manager.execution_graph.add_edge(targets_transfer, grad_node)

        global_temp_grad = (NUM_EXITS,)
        local_temp_grad = optimal_local_sizes(global_temp_grad, device_limits)
        temp_grad_node = kernel_wrapper.compute_temp_gradients(
            global_temp_grad, local_temp_grad,
            buffer_mgr.buffers['exit_probs'], buffer_mgr.buffers['targets_batch'],
            pm.grad_buffers['temps'], actual_batch_size
        )
        manager.execution_graph.add_edge(targets_transfer, temp_grad_node)

        for param in ['weights', 'biases', 'exit_weights', 'exit_biases', 'temps']:
            if param in pm.params and pm.params[param]['requires_grad']:
                grad_buf = pm.grad_buffers[param]
                param_buf = pm.buffers[param]
                m1_buf = pm.m1_buffers[param]
                m2_buf = pm.m2_buffers[param]
                grad_name = f'grad_{param}'
                param_name = param
                m1_name = f'm1_{param}'
                m2_name = f'm2_{param}'
                total_params = int(np.prod(buffer_mgr.buffer_metadata[param]['padded_shape']))
                global_adam = (total_params,)
                local_adam = optimal_local_sizes(global_adam, device_limits)
                adam_node = kernel_wrapper.adam_update(
                    global_adam, local_adam,
                    grad_buf, param_buf, m1_buf, m2_buf,
                    grad_name, param_name, m1_name, m2_name, total_params
                )
                manager.execution_graph.add_edge(grad_node if param != 'temps' else temp_grad_node, adam_node)
                if param == 'temps':
                    global_clamp = (NUM_EXITS,)
                    local_clamp = optimal_local_sizes(global_clamp, device_limits)
                    clamp_node = kernel_wrapper.clamp_temperatures(global_clamp, local_clamp, param_buf)
                    manager.execution_graph.add_edge(adam_node, clamp_node)

        # Device-to-host transfers for results
        losses_transfer, losses_host = buffer_mgr.staged_transfer(None, 'losses', is_device_to_host=True)
        exit_probs_transfer, exit_probs_host = buffer_mgr.staged_transfer(None, 'exit_probs', is_device_to_host=True)
        temps_transfer, temps_host = buffer_mgr.staged_transfer(None, 'temps', is_device_to_host=True)

        # Create SYNC node for HOST_SIGNAL
        sync_uid = manager.create_sync_node(SyncType.HOST_SIGNAL, dependencies=[losses_transfer, exit_probs_transfer, temps_transfer])
        manager.commit_workload()

        # Wait for sync
        manager.wait_for_sync(sync_uid)

        # Access results
        exit_losses = []
        for exit_idx in range(NUM_EXITS):
            start = exit_idx * (buffer_mgr.buffer_metadata['losses']['padded_shape'][0] // NUM_EXITS)
            end = min(start + actual_batch_size, len(losses_host))
            if end > start:
                exit_loss = losses_host[start:end].mean()
                exit_losses.append(float(exit_loss))
        print(f"Batch {batch_idx}: Per-exit Losses: {exit_losses}")
        buffer_mgr.validate_memory()
        valid_probs = exit_probs_host[:actual_batch_size].reshape(actual_batch_size, NUM_EXITS, OUTPUT_CLASSES)
        confidences = np.array([valid_probs[:, i, :].max(axis=1) ** (1 / (temps_host[i] + 1e-8)) for i in range(NUM_EXITS)])
        weights = np.exp(confidences) / np.sum(np.exp(confidences), axis=0)
        ensemble_probs = np.einsum('ijk,j->ik', valid_probs, weights)
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
    print(f"Temperatures: {temps_host}")

# Cleanup
for name in list(buffer_mgr.buffer_metadata.keys()):
    buffer_mgr.release_buffer(name)
manager.commit_workload()

if __name__ == "__main__":
    pass
