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
HIDDEN_DIM = 65
OUTPUT_CLASSES = 3
NUM_EXITS = 40
EPOCHS = 100
BATCH_SIZE = 130
ADAM_BETA1 = 0.9
ADAM_BETA2 = 0.999
LEARNING_RATE = 0.001
MIN_TEMP = 1.0e-3
MAX_TEMP = 10.0
EPSILON = 1.0e-8

# OpenCL kernel files (assumed to exist and updated to use masks)
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
    return padded

class BatchPadder:
    def __init__(self, context):
        self.ctx = context

    def pad_batch(self, X, y):
        X_padded = pad_tensor(X, self.ctx)
        y_padded = pad_tensor(y, self.ctx, strategy='simd_aware')
        mask = pad_tensor(np.ones(len(X), dtype=np.float32), self.ctx)
        return X_padded, y_padded, mask

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

ACCESS_WRITE_MODES = {AccessMode.EXCLUSIVE_WRITE_OVER, AccessMode.EXCLUSIVE_ZERO, AccessMode.EXCLUSIVE_UPDATE}

@dataclass
class MaskedBuffer:
    data: str
    mask: str
    real_shape: tuple
    padded_shape: tuple
    mask_real_shape: tuple
    mask_padded_shape: tuple
    dtype: np.dtype
    const_mask: bool = False

    def __post_init__(self):
        if not self.const_mask:
            if self.padded_shape[0] != self.mask_padded_shape[0]:
                raise ValueError("Batch dimension mismatch between data and mask")

class WorkManager:
    def __init__(self, context):
        self.context = context
        self.execution_graph = nx.MultiDiGraph()
        self.logical_resources = {}
        self.hardware_queues = {}
        self.node_id_counter = 0
        self.buffer_tracker = defaultdict(lambda: {'version': 0, 'alloc_node': None, 'free_node': None, 'metadata': None})
        self.user_events = {}
        self.last_writer = {}
        self.pending_readers = defaultdict(set)

    @dataclass
    class ExecutionNode:
        __slots__ = ['uid', 'node_type', 'queue_type', 'access_map', 'dependencies', 'expected_versions', 'params', 'event']
        uid: int
        node_type: NodeType
        queue_type: str
        access_map: dict
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
            buffer_name = list(access_map.keys())[0]
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
        self.masked_buffers = {}
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

        # Create mask buffer
        if len(real_shape) > 0 and real_shape[0] == BATCH_SIZE:
            mask_real_shape = (BATCH_SIZE,)
            mask_padded_shape = (padded_shape[0],)
            const_mask = False
        else:
            mask_real_shape = (1,)
            mask_padded_shape = (1,)
            const_mask = True

        mask_name = f"mask_{name}"
        mask_size = np.prod(mask_padded_shape) * np.float32().itemsize
        mask_metadata = {'real_shape': mask_real_shape, 'padded_shape': mask_padded_shape, 'dtype': np.float32, 'strategy': 'none'}
        mask_alloc_node = self.work_manager.create_mem_node(mask_name, MemOpType.ALLOC, size=mask_size, metadata=mask_metadata)
        self.buffer_metadata[mask_name] = mask_metadata

        if const_mask:
            def init_mask_fn(queue, wait_for):
                buffer = self.work_manager.logical_resources[mask_name][0]
                return cl.enqueue_fill_buffer(queue, buffer, np.float32(1.0), 0, mask_size, wait_for=wait_for)
            init_mask_node = self.work_manager.create_node(
                NodeType.COMPUTE, 'compute',
                {mask_name: {AccessMode.EXCLUSIVE_WRITE_OVER}},
                init_mask_fn, {}
            )
            self.work_manager.execution_graph.add_edge(mask_alloc_node, init_mask_node)

        mbuf = MaskedBuffer(
            data=name,
            mask=mask_name,
            real_shape=real_shape,
            padded_shape=padded_shape,
            mask_real_shape=mask_real_shape,
            mask_padded_shape=mask_padded_shape,
            dtype=dtype,
            const_mask=const_mask
        )
        self.masked_buffers[name] = mbuf
        return mbuf

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
        if name in self.masked_buffers:
            mbuf = self.masked_buffers[name]
            self.work_manager.create_mem_node(mbuf.data, MemOpType.FREE)
            self.work_manager.create_mem_node(mbuf.mask, MemOpType.FREE)
            del self.masked_buffers[name]
            del self.buffer_metadata[mbuf.data]
            del self.buffer_metadata[mbuf.mask]

    def get_dimensions(self, name):
        return (self.buffer_metadata[name]['real_shape'], self.buffer_metadata[name]['padded_shape'])

    def staged_transfer(self, host_data, mbuf: MaskedBuffer, is_device_to_host=False):
        if is_device_to_host:
            data_host = np.empty(mbuf.padded_shape, dtype=mbuf.dtype)
            mask_host = np.empty(mbuf.mask_padded_shape, dtype=np.float32)
            def transfer_data_fn(queue, wait_for):
                buffer = self.work_manager.logical_resources[mbuf.data][0]
                return cl.enqueue_copy(queue, data_host, buffer, wait_for=wait_for)
            def transfer_mask_fn(queue, wait_for):
                buffer = self.work_manager.logical_resources[mbuf.mask][0]
                return cl.enqueue_copy(queue, mask_host, buffer, wait_for=wait_for)
            data_node = self.work_manager.create_node(
                NodeType.TRANSFER, 'xfer',
                {mbuf.data: {AccessMode.SHARED_READ}},
                transfer_data_fn, {}
            )
            mask_node = self.work_manager.create_node(
                NodeType.TRANSFER, 'xfer',
                {mbuf.mask: {AccessMode.SHARED_READ}},
                transfer_mask_fn, {}
            )
            return data_node, data_host, mask_node, mask_host
        else:
            def transfer_data_fn(queue, wait_for):
                buffer = self.work_manager.logical_resources[mbuf.data][0]
                return cl.enqueue_copy(queue, buffer, host_data['data'], wait_for=wait_for)
            def transfer_mask_fn(queue, wait_for):
                buffer = self.work_manager.logical_resources[mbuf.mask][0]
                return cl.enqueue_copy(queue, buffer, host_data['mask'], wait_for=wait_for)
            data_node = self.work_manager.create_node(
                NodeType.TRANSFER, 'xfer',
                {mbuf.data: {AccessMode.EXCLUSIVE_WRITE_OVER}},
                transfer_data_fn, {}
            )
            mask_node = self.work_manager.create_node(
                NodeType.TRANSFER, 'xfer',
                {mbuf.mask: {AccessMode.EXCLUSIVE_WRITE_OVER}},
                transfer_mask_fn, {}
            )
            return data_node, mask_node

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
            mbuf = self.buffer_manager.acquire_buffer(name, spec['shape'], np.float32)
            self.buffer_manager.staged_transfer({'data': processed, 'mask': np.ones((1,), dtype=np.float32)}, mbuf)
            if spec['requires_grad']:
                grad_mbuf = self.buffer_manager.acquire_buffer(f"grad_{name}", spec['shape'], np.float32, access_mode=AccessMode.EXCLUSIVE_ZERO)
                m1_mbuf = self.buffer_manager.acquire_buffer(f"m1_{name}", spec['shape'], np.float32, access_mode=AccessMode.EXCLUSIVE_ZERO)
                m2_mbuf = self.buffer_manager.acquire_buffer(f"m2_{name}", spec['shape'], np.float32, access_mode=AccessMode.EXCLUSIVE_ZERO)
                self.params[name]['grad_mbuf'] = grad_mbuf
                self.params[name]['m1_mbuf'] = m1_mbuf
                self.params[name]['m2_mbuf'] = m2_mbuf

    def set_buffers(self, logical_resources):
        for name in self.params:
            mbuf = self.buffer_manager.masked_buffers[name]
            self.buffers[name] = logical_resources[mbuf.data][0]
            if self.params[name]['requires_grad']:
                grad_mbuf = self.params[name]['grad_mbuf']
                m1_mbuf = self.params[name]['m1_mbuf']
                m2_mbuf = self.params[name]['m2_mbuf']
                self.grad_buffers[name] = logical_resources[grad_mbuf.data][0]
                self.m1_buffers[name] = logical_resources[m1_mbuf.data][0]
                self.m2_buffers[name] = logical_resources[m2_mbuf.data][0]

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
        self.local_mem_reqs = {}
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

    def forward_pass(self, global_sizes, local_sizes, input_mbuf, weights_mbuf, biases_mbuf, hidden_mbuf):
        req = KernelExecutionRequest(self.program, 'forward_pass', global_sizes, local_sizes)
        req.set_local_mem_argument(0, local_sizes[0] * FLOAT_SIZE, AccessMode.LOCAL)
        req.bind_argument(1, self.buffer_manager.buffers[input_mbuf.data])
        req.bind_argument(2, self.buffer_manager.buffers[input_mbuf.mask])
        req.bind_argument(3, self.buffer_manager.buffers[weights_mbuf.data])
        req.bind_argument(4, self.buffer_manager.buffers[weights_mbuf.mask])
        req.bind_argument(5, self.buffer_manager.buffers[biases_mbuf.data])
        req.bind_argument(6, self.buffer_manager.buffers[biases_mbuf.mask])
        req.bind_argument(7, self.buffer_manager.buffers[hidden_mbuf.data])
        req.bind_argument(8, self.buffer_manager.buffers[hidden_mbuf.mask])
        access_map = {
            input_mbuf.data: {AccessMode.SHARED_READ},
            input_mbuf.mask: {AccessMode.SHARED_READ},
            weights_mbuf.data: {AccessMode.SHARED_READ},
            weights_mbuf.mask: {AccessMode.SHARED_READ},
            biases_mbuf.data: {AccessMode.SHARED_READ},
            biases_mbuf.mask: {AccessMode.SHARED_READ},
            hidden_mbuf.data: {AccessMode.EXCLUSIVE_WRITE_OVER},
            hidden_mbuf.mask: {AccessMode.EXCLUSIVE_WRITE_OVER},
        }
        return self.manager.create_node(
            node_type=NodeType.COMPUTE,
            queue_type='compute',
            access_map=access_map,
            operation_fn=self._enqueue_kernel,
            params={'exec_req': req}
        )

    def compute_exit_probabilities(self, global_sizes, local_sizes, hidden_mbuf, exit_weights_mbuf, exit_biases_mbuf, exit_probs_mbuf, losses_mbuf, targets_mbuf, exit_idx):
        req = KernelExecutionRequest(self.program, 'compute_exit_probabilities', global_sizes, local_sizes)
        req.bind_argument(0, self.buffer_manager.buffers[hidden_mbuf.data])
        req.bind_argument(1, self.buffer_manager.buffers[hidden_mbuf.mask])
        req.bind_argument(2, self.buffer_manager.buffers[exit_weights_mbuf.data])
        req.bind_argument(3, self.buffer_manager.buffers[exit_weights_mbuf.mask])
        req.bind_argument(4, self.buffer_manager.buffers[exit_biases_mbuf.data])
        req.bind_argument(5, self.buffer_manager.buffers[exit_biases_mbuf.mask])
        req.bind_argument(6, self.buffer_manager.buffers[exit_probs_mbuf.data])
        req.bind_argument(7, self.buffer_manager.buffers[exit_probs_mbuf.mask])
        req.bind_argument(8, self.buffer_manager.buffers[losses_mbuf.data])
        req.bind_argument(9, self.buffer_manager.buffers[losses_mbuf.mask])
        req.bind_argument(10, self.buffer_manager.buffers[targets_mbuf.data])
        req.bind_argument(11, self.buffer_manager.buffers[targets_mbuf.mask])
        req.bind_argument(12, np.int32(exit_idx))
        req.bind_argument(13, self.temperatures)
        req.bind_argument(14, np.int32(self.hidden_dim))
        req.bind_argument(15, np.int32(self.output_classes))
        req.bind_argument(16, np.int32(self.padded_hidden_dim))
        req.bind_argument(17, np.int32(self.padded_output_classes))
        req.bind_argument(18, np.int32(self.padded_batch_size))
        access_map = {
            hidden_mbuf.data: {AccessMode.SHARED_READ},
            hidden_mbuf.mask: {AccessMode.SHARED_READ},
            exit_weights_mbuf.data: {AccessMode.SHARED_READ},
            exit_weights_mbuf.mask: {AccessMode.SHARED_READ},
            exit_biases_mbuf.data: {AccessMode.SHARED_READ},
            exit_biases_mbuf.mask: {AccessMode.SHARED_READ},
            targets_mbuf.data: {AccessMode.SHARED_READ},
            targets_mbuf.mask: {AccessMode.SHARED_READ},
            'temps': {AccessMode.SHARED_READ},
            exit_probs_mbuf.data: {AccessMode.EXCLUSIVE_WRITE_OVER},
            exit_probs_mbuf.mask: {AccessMode.EXCLUSIVE_WRITE_OVER},
            losses_mbuf.data: {AccessMode.EXCLUSIVE_WRITE_OVER},
            losses_mbuf.mask: {AccessMode.EXCLUSIVE_WRITE_OVER}
        }
        return self.manager.create_node(
            node_type=NodeType.COMPUTE,
            queue_type='compute',
            access_map=access_map,
            operation_fn=self._enqueue_kernel,
            params={'exec_req': req}
        )

    def compute_gradients(self, global_sizes, local_sizes, input_mbuf, hidden_mbuf, exit_probs_mbuf, exit_weights_mbuf, grad_weights_mbuf, grad_biases_mbuf, grad_exit_weights_mbuf, grad_exit_biases_mbuf, targets_mbuf):
        req = KernelExecutionRequest(self.program, 'compute_gradients', global_sizes, local_sizes)
        req.bind_argument(0, self.buffer_manager.buffers[input_mbuf.data])
        req.bind_argument(1, self.buffer_manager.buffers[input_mbuf.mask])
        req.bind_argument(2, self.buffer_manager.buffers[hidden_mbuf.data])
        req.bind_argument(3, self.buffer_manager.buffers[hidden_mbuf.mask])
        req.bind_argument(4, self.buffer_manager.buffers[exit_probs_mbuf.data])
        req.bind_argument(5, self.buffer_manager.buffers[exit_probs_mbuf.mask])
        req.bind_argument(6, self.buffer_manager.buffers[exit_weights_mbuf.data])
        req.bind_argument(7, self.buffer_manager.buffers[exit_weights_mbuf.mask])
        req.bind_argument(8, self.buffer_manager.buffers[grad_weights_mbuf.data])
        req.bind_argument(9, self.buffer_manager.buffers[grad_weights_mbuf.mask])
        req.bind_argument(10, self.buffer_manager.buffers[grad_biases_mbuf.data])
        req.bind_argument(11, self.buffer_manager.buffers[grad_biases_mbuf.mask])
        req.bind_argument(12, self.buffer_manager.buffers[grad_exit_weights_mbuf.data])
        req.bind_argument(13, self.buffer_manager.buffers[grad_exit_weights_mbuf.mask])
        req.bind_argument(14, self.buffer_manager.buffers[grad_exit_biases_mbuf.data])
        req.bind_argument(15, self.buffer_manager.buffers[grad_exit_biases_mbuf.mask])
        req.bind_argument(16, self.buffer_manager.buffers[targets_mbuf.data])
        req.bind_argument(17, self.buffer_manager.buffers[targets_mbuf.mask])
        req.bind_argument(18, self.temperatures)
        req.bind_argument(19, np.int32(self.input_dim))
        req.bind_argument(20, np.int32(self.hidden_dim))
        req.bind_argument(21, np.int32(self.output_classes))
        req.bind_argument(22, np.int32(self.padded_input_dim))
        req.bind_argument(23, np.int32(self.padded_hidden_dim))
        req.bind_argument(24, np.int32(self.padded_output_classes))
        req.bind_argument(25, np.int32(self.padded_batch_size))
        req.bind_argument(26, np.int32(self.num_exits))
        access_map = {
            input_mbuf.data: {AccessMode.SHARED_READ},
            input_mbuf.mask: {AccessMode.SHARED_READ},
            hidden_mbuf.data: {AccessMode.SHARED_READ},
            hidden_mbuf.mask: {AccessMode.SHARED_READ},
            exit_probs_mbuf.data: {AccessMode.SHARED_READ},
            exit_probs_mbuf.mask: {AccessMode.SHARED_READ},
            exit_weights_mbuf.data: {AccessMode.SHARED_READ},
            exit_weights_mbuf.mask: {AccessMode.SHARED_READ},
            targets_mbuf.data: {AccessMode.SHARED_READ},
            targets_mbuf.mask: {AccessMode.SHARED_READ},
            'temps': {AccessMode.SHARED_READ},
            grad_weights_mbuf.data: {AccessMode.EXCLUSIVE_WRITE_OVER},
            grad_weights_mbuf.mask: {AccessMode.EXCLUSIVE_WRITE_OVER},
            grad_biases_mbuf.data: {AccessMode.EXCLUSIVE_WRITE_OVER},
            grad_biases_mbuf.mask: {AccessMode.EXCLUSIVE_WRITE_OVER},
            grad_exit_weights_mbuf.data: {AccessMode.EXCLUSIVE_WRITE_OVER},
            grad_exit_weights_mbuf.mask: {AccessMode.EXCLUSIVE_WRITE_OVER},
            grad_exit_biases_mbuf.data: {AccessMode.EXCLUSIVE_WRITE_OVER},
            grad_exit_biases_mbuf.mask: {AccessMode.EXCLUSIVE_WRITE_OVER}
        }
        return self.manager.create_node(
            node_type=NodeType.COMPUTE,
            queue_type='compute',
            access_map=access_map,
            operation_fn=self._enqueue_kernel,
            params={'exec_req': req}
        )

    def compute_temp_gradients(self, global_sizes, local_sizes, exit_probs_mbuf, targets_mbuf, grad_temps_mbuf):
        req = KernelExecutionRequest(self.program, 'compute_temp_gradients', global_sizes, local_sizes)
        req.bind_argument(0, self.buffer_manager.buffers[exit_probs_mbuf.data])
        req.bind_argument(1, self.buffer_manager.buffers[exit_probs_mbuf.mask])
        req.bind_argument(2, self.buffer_manager.buffers[targets_mbuf.data])
        req.bind_argument(3, self.buffer_manager.buffers[targets_mbuf.mask])
        req.bind_argument(4, self.buffer_manager.buffers[grad_temps_mbuf.data])
        req.bind_argument(5, self.buffer_manager.buffers[grad_temps_mbuf.mask])
        req.bind_argument(6, self.temperatures)
        req.bind_argument(7, np.int32(self.output_classes))
        req.bind_argument(8, np.int32(self.padded_output_classes))
        req.bind_argument(9, np.int32(self.padded_batch_size))
        req.bind_argument(10, np.int32(self.num_exits))
        access_map = {
            exit_probs_mbuf.data: {AccessMode.SHARED_READ},
            exit_probs_mbuf.mask: {AccessMode.SHARED_READ},
            targets_mbuf.data: {AccessMode.SHARED_READ},
            targets_mbuf.mask: {AccessMode.SHARED_READ},
            'temps': {AccessMode.SHARED_READ},
            grad_temps_mbuf.data: {AccessMode.EXCLUSIVE_WRITE_OVER},
            grad_temps_mbuf.mask: {AccessMode.EXCLUSIVE_WRITE_OVER}
        }
        return self.manager.create_node(
            node_type=NodeType.COMPUTE,
            queue_type='compute',
            access_map=access_map,
            operation_fn=self._enqueue_kernel,
            params={'exec_req': req}
        )

    def adam_update(self, global_sizes, local_sizes, grad_mbuf, param_mbuf, m1_mbuf, m2_mbuf, grad_name, param_name, m1_name, m2_name, total_params):
        req = KernelExecutionRequest(self.program, 'adam_update', global_sizes, local_sizes)
        req.bind_argument(0, self.buffer_manager.buffers[grad_mbuf.data])
        req.bind_argument(1, self.buffer_manager.buffers[grad_mbuf.mask])
        req.bind_argument(2, self.buffer_manager.buffers[param_mbuf.data])
        req.bind_argument(3, self.buffer_manager.buffers[param_mbuf.mask])
        req.bind_argument(4, self.buffer_manager.buffers[m1_mbuf.data])
        req.bind_argument(5, self.buffer_manager.buffers[m1_mbuf.mask])
        req.bind_argument(6, self.buffer_manager.buffers[m2_mbuf.data])
        req.bind_argument(7, self.buffer_manager.buffers[m2_mbuf.mask])
        req.bind_argument(8, self.adam_beta1)
        req.bind_argument(9, self.adam_beta2)
        req.bind_argument(10, self.beta1_t)
        req.bind_argument(11, self.beta2_t)
        req.bind_argument(12, self.learning_rate)
        req.bind_argument(13, self.epsilon)
        req.bind_argument(14, np.int32(total_params))
        access_map = {
            grad_mbuf.data: {AccessMode.SHARED_READ},
            grad_mbuf.mask: {AccessMode.SHARED_READ},
            param_mbuf.data: {AccessMode.EXCLUSIVE_UPDATE},
            param_mbuf.mask: {AccessMode.EXCLUSIVE_UPDATE},
            m1_mbuf.data: {AccessMode.EXCLUSIVE_UPDATE},
            m1_mbuf.mask: {AccessMode.EXCLUSIVE_UPDATE},
            m2_mbuf.data: {AccessMode.EXCLUSIVE_UPDATE},
            m2_mbuf.mask: {AccessMode.EXCLUSIVE_UPDATE}
        }
        return self.manager.create_node(
            node_type=NodeType.COMPUTE,
            queue_type='compute',
            access_map=access_map,
            operation_fn=self._enqueue_kernel,
            params={'exec_req': req}
        )

    def clamp_temperatures(self, global_sizes, local_sizes, temps_mbuf):
        req = KernelExecutionRequest(self.program, 'clamp_temperatures', global_sizes, local_sizes)
        req.bind_argument(0, self.buffer_manager.buffers[temps_mbuf.data])
        req.bind_argument(1, self.buffer_manager.buffers[temps_mbuf.mask])
        req.bind_argument(2, self.min_temp)
        req.bind_argument(3, self.max_temp)
        req.bind_argument(4, np.int32(self.num_exits))
        access_map = {
            temps_mbuf.data: {AccessMode.EXCLUSIVE_UPDATE},
            temps_mbuf.mask: {AccessMode.EXCLUSIVE_UPDATE}
        }
        return self.manager.create_node(
            node_type=NodeType.COMPUTE,
            queue_type='compute',
            access_map=access_map,
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
        input_mbuf = buffer_mgr.masked_buffers['input_batch']
        targets_mbuf = buffer_mgr.masked_buffers['targets_batch']
        mask_mbuf = buffer_mgr.masked_buffers['mask_batch']
        input_transfer_data, input_transfer_mask = buffer_mgr.staged_transfer({'data': X_padded, 'mask': mask}, input_mbuf)
        targets_transfer_data, targets_transfer_mask = buffer_mgr.staged_transfer({'data': y_padded, 'mask': mask}, targets_mbuf)
        mask_transfer_data, mask_transfer_mask = buffer_mgr.staged_transfer({'data': mask, 'mask': np.ones_like(mask)}, mask_mbuf)

        kernel_wrapper = KernelWrapper(program, manager, global_step, compute_queue, buffer_mgr)
        kernel_wrapper.set_mask(buffer_mgr.buffers[mask_mbuf.data]).set_temps(buffer_mgr.buffers[buffer_mgr.masked_buffers['temps'].data])
        pm.zero_gradients(manager)

        # Forward pass node with dependencies
        global_forward = (buffer_mgr.buffer_metadata['input_batch']['padded_shape'][0], HIDDEN_DIM)
        local_forward = optimal_local_sizes(global_forward, device_limits)
        forward_node = kernel_wrapper.forward_pass(
            global_forward, local_forward,
            input_mbuf, buffer_mgr.masked_buffers['weights'],
            buffer_mgr.masked_buffers['biases'], buffer_mgr.masked_buffers['hidden']
        )
        manager.execution_graph.add_edge(input_transfer_data, forward_node)
        manager.execution_graph.add_edge(input_transfer_mask, forward_node)
        manager.execution_graph.add_edge(mask_transfer_data, forward_node)
        manager.execution_graph.add_edge(mask_transfer_mask, forward_node)

        for exit_idx in range(NUM_EXITS):
            global_exit = (buffer_mgr.buffer_metadata['input_batch']['padded_shape'][0],)
            local_exit = optimal_local_sizes(global_exit, device_limits)
            exit_node = kernel_wrapper.compute_exit_probabilities(
                global_exit, local_exit,
                buffer_mgr.masked_buffers['hidden'], buffer_mgr.masked_buffers['exit_weights'],
                buffer_mgr.masked_buffers['exit_biases'], buffer_mgr.masked_buffers['exit_probs'],
                buffer_mgr.masked_buffers['losses'], buffer_mgr.masked_buffers['targets_batch'],
                exit_idx
            )
            manager.execution_graph.add_edge(forward_node, exit_node)
            manager.execution_graph.add_edge(targets_transfer_data, exit_node)
            manager.execution_graph.add_edge(targets_transfer_mask, exit_node)

        global_grad = (INPUT_DIM, HIDDEN_DIM)
        local_grad = optimal_local_sizes(global_grad, device_limits)
        grad_node = kernel_wrapper.compute_gradients(
            global_grad, local_grad,
            input_mbuf, buffer_mgr.masked_buffers['hidden'],
            buffer_mgr.masked_buffers['exit_probs'], buffer_mgr.masked_buffers['exit_weights'],
            buffer_mgr.masked_buffers['grad_weights'], buffer_mgr.masked_buffers['grad_biases'],
            buffer_mgr.masked_buffers['grad_exit_weights'], buffer_mgr.masked_buffers['grad_exit_biases'],
            targets_mbuf
        )
        manager.execution_graph.add_edge(input_transfer_data, grad_node)
        manager.execution_graph.add_edge(input_transfer_mask, grad_node)
        manager.execution_graph.add_edge(forward_node, grad_node)
        manager.execution_graph.add_edge(targets_transfer_data, grad_node)
        manager.execution_graph.add_edge(targets_transfer_mask, grad_node)

        global_temp_grad = (NUM_EXITS,)
        local_temp_grad = optimal_local_sizes(global_temp_grad, device_limits)
        temp_grad_node = kernel_wrapper.compute_temp_gradients(
            global_temp_grad, local_temp_grad,
            buffer_mgr.masked_buffers['exit_probs'], targets_mbuf,
            buffer_mgr.masked_buffers['grad_temps']
        )
        manager.execution_graph.add_edge(targets_transfer_data, temp_grad_node)
        manager.execution_graph.add_edge(targets_transfer_mask, temp_grad_node)

        for param in ['weights', 'biases', 'exit_weights', 'exit_biases', 'temps']:
            if param in pm.params and pm.params[param]['requires_grad']:
                grad_mbuf = pm.params[param]['grad_mbuf']
                param_mbuf = buffer_mgr.masked_buffers[param]
                m1_mbuf = pm.params[param]['m1_mbuf']
                m2_mbuf = pm.params[param]['m2_mbuf']
                grad_name = f'grad_{param}'
                param_name = param
                m1_name = f'm1_{param}'
                m2_name = f'm2_{param}'
                total_params = int(np.prod(buffer_mgr.buffer_metadata[param]['padded_shape']))
                global_adam = (total_params,)
                local_adam = optimal_local_sizes(global_adam, device_limits)
                adam_node = kernel_wrapper.adam_update(
                    global_adam, local_adam,
                    grad_mbuf, param_mbuf, m1_mbuf, m2_mbuf,
                    grad_name, param_name, m1_name, m2_name, total_params
                )
                manager.execution_graph.add_edge(grad_node if param != 'temps' else temp_grad_node, adam_node)
                if param == 'temps':
                    global_clamp = (NUM_EXITS,)
                    local_clamp = optimal_local_sizes(global_clamp, device_limits)
                    clamp_node = kernel_wrapper.clamp_temperatures(global_clamp, local_clamp, param_mbuf)
                    manager.execution_graph.add_edge(adam_node, clamp_node)

        # Device-to-host transfers for results
        losses_mbuf = buffer_mgr.masked_buffers['losses']
        exit_probs_mbuf = buffer_mgr.masked_buffers['exit_probs']
        temps_mbuf = buffer_mgr.masked_buffers['temps']
        losses_transfer_data, losses_host, losses_transfer_mask, losses_mask_host = buffer_mgr.staged_transfer(None, losses_mbuf, is_device_to_host=True)
        exit_probs_transfer_data, exit_probs_host, exit_probs_transfer_mask, exit_probs_mask_host = buffer_mgr.staged_transfer(None, exit_probs_mbuf, is_device_to_host=True)
        temps_transfer_data, temps_host, temps_transfer_mask, temps_mask_host = buffer_mgr.staged_transfer(None, temps_mbuf, is_device_to_host=True)

        # Create SYNC node for HOST_SIGNAL
        sync_uid = manager.create_sync_node(SyncType.HOST_SIGNAL, dependencies=[losses_transfer_data, exit_probs_transfer_data, temps_transfer_data])
        manager.commit_workload()

        # Wait for sync
        manager.wait_for_sync(sync_uid)

        # Access results with masking
        padded_batch_size = buffer_mgr.buffer_metadata['losses']['padded_shape'][0] // NUM_EXITS
        losses_reshaped = losses_host.reshape(NUM_EXITS, padded_batch_size)
        exit_losses = []
        for exit_idx in range(NUM_EXITS):
            masked_losses = losses_reshaped[exit_idx] * mask
            total_loss = np.sum(masked_losses)
            num_valid = np.sum(mask)
            if num_valid > 0:
                exit_loss = total_loss / num_valid
                exit_losses.append(float(exit_loss))
            else:
                exit_losses.append(0.0)
        print(f"Batch {batch_idx}: Per-exit Losses: {exit_losses}")
        buffer_mgr.validate_memory()
        valid_mask = mask > 0
        valid_probs = exit_probs_host[valid_mask, :, :OUTPUT_CLASSES]
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
for name in list(buffer_mgr.masked_buffers.keys()):
    buffer_mgr.release_buffer(name)
manager.commit_workload()

if __name__ == "__main__":
    pass