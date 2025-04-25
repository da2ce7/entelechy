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

def preprocess_weights(w, simd_width):
    if w.ndim == 2:
        input_dim, hidden_dim = w.shape
        hidden_padded = ((hidden_dim + simd_width - 1) // simd_width) * simd_width
        input_padded = ((input_dim + simd_width - 1) // simd_width) * simd_width
        w_padded = np.zeros((input_padded, hidden_padded), dtype=np.float32)
        w_padded[:input_dim, :hidden_dim] = w
        return w_padded.T.reshape(hidden_padded // simd_width, input_padded, simd_width)
    elif w.ndim == 3:
        exits, hidden, classes = w.shape
        classes_padded = ((classes + simd_width - 1) // simd_width) * simd_width
        w_padded = np.zeros((exits, hidden, classes_padded), dtype=np.float32)
        w_padded[:, :, :classes] = w
        return w_padded.transpose(0, 2, 1).reshape(exits, classes_padded // simd_width, hidden, simd_width)

def pad_1d(arr, simd_width):
    padded_size = ((arr.shape[0] + simd_width - 1) // simd_width) * simd_width
    arr_padded = np.zeros(padded_size, dtype=arr.dtype)
    arr_padded[:arr.shape[0]] = arr
    return arr_padded

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

def pad_batch(X_batch, y_batch, buffer_mgr):
    actual_batch_size = X_batch.shape[0]
    padded_batch_size = buffer_mgr.buffer_metadata['input_batch']['padded_shape'][0]
    X_padded = np.zeros((padded_batch_size, INPUT_DIM), dtype=np.float32)
    y_padded = np.zeros(padded_batch_size, dtype=np.int32)
    mask = np.zeros(padded_batch_size, dtype=np.float32)
    X_padded[:actual_batch_size] = X_batch
    y_padded[:actual_batch_size] = y_batch
    mask[:actual_batch_size] = 1.0
    if not np.all(mask[:actual_batch_size] == 1.0) or not np.all(mask[actual_batch_size:] == 0.0):
        raise ValueError("Invalid mask generated during batch padding")
    return X_padded, y_padded, mask

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

class WorkManager:
    def __init__(self, context):
        self.context = context
        self.execution_graph = nx.MultiDiGraph()
        self.logical_resources = {}
        self.hardware_queues = {}
        self.node_id_counter = 0
        self.buffer_tracker = defaultdict(lambda: {'version': 0, 'alloc_node': None, 'free_node': None})

    class ExecutionNode:
        __slots__ = ['uid', 'node_type', 'queue_type', 'resources', 'dependencies', 'expected_versions', 'params', 'event']
        def __init__(self, uid, node_type, queue_type, resources, dependencies, params):
            self.uid = uid
            self.node_type = node_type
            self.queue_type = queue_type
            self.resources = resources
            self.dependencies = dependencies
            self.expected_versions = {}
            self.params = params
            self.event = None

    def create_node(self, node_type, queue_type, resource_access, operation_fn, params):
        node = self.ExecutionNode(self.node_id_counter, node_type, queue_type, resource_access, set(), params)
        node.execute = operation_fn
        if node_type == NodeType.TRANSFER:
            buffer_name = list(resource_access['W'])[0] if params['direction'] == TransferDirection.H2D else list(resource_access['R'])[0]
            alloc_node = self.buffer_tracker[buffer_name]['alloc_node']
            node.dependencies.add(alloc_node)
            if params['direction'] == TransferDirection.D2H:
                for uid in self.execution_graph.nodes:
                    existing_node = self.execution_graph.nodes[uid]['node']
                    if buffer_name in existing_node.resources['W']:
                        node.dependencies.add(uid)
        for res in resource_access['R']:
            if res in self.logical_resources:
                node.expected_versions[res] = self.logical_resources[res][1]
            else:
                node.expected_versions[res] = 0
        for res in resource_access['R'] | resource_access['W']:
            if res in self.logical_resources:
                for existing_uid in self.execution_graph.nodes:
                    existing_node = self.execution_graph.nodes[existing_uid]['node']
                    if (res in existing_node.resources['W'] or
                        (res in existing_node.resources['R'] and res in resource_access['W'])):
                        node.dependencies.add(existing_uid)
        self.execution_graph.add_node(node.uid, node=node)
        self.node_id_counter += 1
        return node.uid

    def create_mem_node(self, name, op_type, size=None, buffer=None):
        if op_type == MemOpType.ALLOC:
            def _alloc_fn(queue, wait_for):
                buffer = cl.Buffer(self.context, cl.mem_flags.READ_WRITE, size=size)
                self.logical_resources[name] = (buffer, 0)
                self.buffer_tracker[name]['alloc_node'] = self.node_id_counter
                return None  # No event for memory operations
            node = self.ExecutionNode(self.node_id_counter, NodeType.MEMORY, 'compute', {'W': [name]}, set(), {'op_type': op_type, 'size': size})
            node.execute = _alloc_fn
        elif op_type == MemOpType.FREE:
            def _free_fn(queue, wait_for):
                buffer = self.logical_resources[name][0]
                buffer.release()
                del self.logical_resources[name]
                self.buffer_tracker[name]['free_node'] = self.node_id_counter
                return None
            node = self.ExecutionNode(self.node_id_counter, NodeType.MEMORY, 'compute', {'R': [name]}, set(), {'op_type': op_type})
            node.execute = _free_fn
            for existing_uid in self.execution_graph.nodes:
                existing_node = self.execution_graph.nodes[existing_uid]['node']
                if name in existing_node.resources['W']:
                    node.dependencies.add(existing_uid)
        self.execution_graph.add_node(node.uid, node=node)
        self.node_id_counter += 1
        return node.uid

    def commit_workload(self):
        ordered_nodes = list(nx.topological_sort(self.execution_graph))
        completion_events = {}
        for uid in ordered_nodes:
            node = self.execution_graph.nodes[uid]['node']
            queue = self.hardware_queues[node.queue_type]
            for res in node.resources['R']:
                if res in self.logical_resources:
                    current_version = self.logical_resources[res][1]
                    expected_version = node.expected_versions.get(res, 0)
                    if expected_version != current_version:
                        raise ConcurrentModificationError(f"Resource {res} has been modified unexpectedly for node {uid}")
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
            for res in node.resources['W']:
                if res in self.logical_resources:
                    buffer, version = self.logical_resources[res]
                    self.logical_resources[res] = (buffer, version + 1)

    def _dispatch_compute(self, node, wait_for):
        queue = self.hardware_queues[node.queue_type]
        node.event = node.execute(queue, wait_for=wait_for)

    def _dispatch_memory(self, node, wait_for):
        node.execute(self.hardware_queues[node.queue_type], wait_for=wait_for)

    def _dispatch_transfer(self, node, wait_for):
        node.execute(self.hardware_queues[node.queue_type], wait_for=wait_for)

    def _dispatch_sync(self, node, wait_for):
        pass  # Placeholder for sync operations

    def reset_graph(self):
        self.execution_graph.clear()
        self.node_id_counter = 0

class BufferManager:
    def __init__(self, context, device, work_manager):
        self.ctx = context
        self.device = device
        self.work_manager = work_manager
        self.simd_width = select_simd_width(device)
        self.wavefront_size = 64 if "AMD" in device.vendor else 32
        self.min_alignment = max(device.min_data_type_align_size, self.simd_width * np.float32().itemsize)
        self.buffer_metadata = {}
        self.pad_strategies = {'opencl_optimal': self._default_pad_strategy}
        self.default_pad_strategy = 'opencl_optimal'
        self.staging_pool = deque(maxlen=8)
        self.buffer_versions = {}
        self.buffers = {}
        self.active_allocations = set()
        self.current_events = []
        work_manager.logical_resources.update(self.buffer_versions)

    def register_pad_strategy(self, name, callback):
        self.pad_strategies[name] = callback

    def acquire_buffer(self, name, real_shape, dtype, pad_strategy=None, mode='device'):
        strategy = self.pad_strategies[pad_strategy or self.default_pad_strategy]
        padded_shape = strategy(real_shape, dtype)
        size = np.prod(padded_shape) * dtype().itemsize
        alloc_node = self.work_manager.create_mem_node(name, MemOpType.ALLOC, size=size)
        self.buffer_metadata[name] = {
            'real_shape': real_shape,
            'padded_shape': padded_shape,
            'dtype': dtype,
            'strategy': pad_strategy or self.default_pad_strategy,
            'alloc_node': alloc_node
        }
        return None  # Buffer is not yet allocated

    def release_buffer(self, name):
        if name in self.buffer_metadata:
            free_node = self.work_manager.create_mem_node(name, MemOpType.FREE)
            del self.buffer_metadata[name]

    def _default_pad_strategy(self, real_shape, dtype):
        item_size = np.dtype(dtype).itemsize
        alignment = lcm(self.min_alignment, item_size)
        return tuple((dim + alignment - 1) // alignment * alignment for dim in real_shape)

    def get_dimensions(self, name):
        return (self.buffer_metadata[name]['real_shape'], self.buffer_metadata[name]['padded_shape'])

    def staged_transfer(self, host_data, buffer_name, is_device_to_host=False, callback=None):
        metadata = self.buffer_metadata[buffer_name]
        buffer = self.work_manager.logical_resources[buffer_name][0]
        padded_shape = metadata['padded_shape']
        real_shape = metadata['real_shape']
        dtype = metadata['dtype']
        size = np.prod(padded_shape) * dtype().itemsize
        staging_buf = self._get_staging_buffer(size)
        
        if not is_device_to_host:
            # Host-to-Device (H2D)
            padded_data = np.zeros(padded_shape, dtype=dtype)
            slices = tuple(slice(0, r) for r in real_shape)
            padded_data[slices] = host_data
            params = {
                'direction': TransferDirection.H2D,
                'source': padded_data,
                'destination': buffer,
                'size': size,
                'staging_buffer': staging_buf
            }
            node_id = self.work_manager.create_node(
                node_type=NodeType.TRANSFER,
                queue_type='xfer',
                resource_access={'W': [buffer_name]},
                operation_fn=self._execute_transfer,
                params=params
            )
            return node_id
        else:
            # Device-to-Host (D2H)
            host_array = np.empty(padded_shape, dtype=dtype)
            params = {
                'direction': TransferDirection.D2H,
                'source': buffer,
                'destination': host_array,
                'size': size,
                'staging_buffer': staging_buf
            }
            node_id = self.work_manager.create_node(
                node_type=NodeType.TRANSFER,
                queue_type='xfer',
                resource_access={'R': [buffer_name]},
                operation_fn=self._execute_transfer,
                params=params
            )
            if callback:
                self.work_manager.execution_graph.nodes[node_id]['node'].params['callback'] = callback
            return node_id

    def _execute_transfer(self, queue, wait_for):
        params = self.params
        staging_buf = params['staging_buffer']
        
        if params['direction'] == TransferDirection.H2D:
            event1 = cl.enqueue_copy(queue, staging_buf, params['source'], wait_for=wait_for)
            event2 = cl.enqueue_copy(queue, params['destination'], staging_buf, wait_for=[event1])
            self.event = event2
        elif params['direction'] == TransferDirection.D2H:
            event1 = cl.enqueue_copy(queue, staging_buf, params['source'], wait_for=wait_for)
            event2 = cl.enqueue_copy(queue, params['destination'], staging_buf, wait_for=[event1])
            self.event = event2
            if 'callback' in params:
                self.event.set_callback(cl.command_execution_status.COMPLETE, params['callback'])

    def _get_staging_buffer(self, size):
        for buf in self.staging_pool:
            if buf.size >= size:
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
            'preprocess': lambda w, sw: preprocess_weights(w, sw),
            'shape_transform': lambda s, sw: (((s[1] + sw - 1) // sw) * sw, ((s[0] + sw - 1) // sw) * sw)
        },
        'exit_weight': {
            'init': he_init,
            'preprocess': lambda w, sw: preprocess_weights(w, sw),
            'shape_transform': lambda s, sw: (s[0], ((s[2] + sw - 1) // sw) * sw, s[1])
        },
        'temperature': {
            'init': lambda s: np.ones(s, dtype=np.float32) * (MAX_TEMP + MIN_TEMP) / 2,
            'constraint': (MIN_TEMP, MAX_TEMP)
        },
        'bias_vector': {
            'init': lambda s: np.zeros(s, dtype=np.float32),
            'preprocess': lambda b, sw: pad_1d(b, sw),
            'shape_transform': lambda s, sw: ((s[0] + sw - 1) // sw) * sw
        }
    }

    def __init__(self, context, buffer_manager):
        self.ctx = context
        self.buffer_manager = buffer_manager
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

    def __getitem__(self, name):
        return self.buffers[name]

    def _create_buffers(self):
        for name, spec in self.params.items():
            ptype = spec['ptype']
            handler = self.PARAM_TYPES[ptype]
            init_values = handler['init'](spec['shape'])
            processed = handler['preprocess'](init_values, self.buffer_manager.simd_width)
            self.buffer_manager.acquire_buffer(name, spec['shape'], np.float32)
            self.buffers[name] = self.buffer_manager.logical_resources[name][0]
            if spec['requires_grad']:
                self.grad_buffers[name] = self.buffer_manager.acquire_buffer(f"grad_{name}", spec['shape'], np.float32)
                self.m1_buffers[name] = self.buffer_manager.acquire_buffer(f"m1_{name}", spec['shape'], np.float32)
                self.m2_buffers[name] =-    def zero_gradients(self, work_manager):
        for name, spec in self.params.items():
            if spec['requires_grad']:
                grad_buffer = self.grad_buffers[name]
                def _zero_fn(queue, wait_for):
                    return cl.enqueue_fill_buffer(queue, grad_buffer, np.float32(0), 0, grad_buffer.size, wait_for=wait_for)
                work_manager.create_node(
                    node_type=NodeType.COMPUTE,
                    queue_type='compute',
                    resource_access={'W': [f'grad_{name}']},
                    operation_fn=_zero_fn,
                    params={}
                )

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
        self.padded_output_classes = ((OUTPUT_CLASSES + self.buffer_manager.simd_width - 1) // self.buffer_manager.simd_width) * self.buffer_manager.simd_width
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
        if min(global_sizes) < 1 or any(g > self.buffer_manager.device.max_work_item_sizes for g in global_sizes):
            raise ValueError(f"Invalid global sizes: {global_sizes}")
        local_mem = cl.LocalMemory(local_sizes[0] * FLOAT_SIZE)
        def _enqueue_forward(queue, wait_for):
            return self.program.forward_pass(
                queue, global_sizes, local_sizes, local_mem, input_buf, weights_buf, biases_buf, hidden_buf,
                self.mask, np.int32(self.input_dim), np.int32(self.hidden_dim),
                np.int32(self.padded_input_dim), np.int32(self.padded_hidden_dim),
                np.int32(self.padded_batch_size), np.int32(actual_batch_size),
                wait_for=wait_for
            )
        return self.manager.create_node(
            node_type=NodeType.COMPUTE,
            queue_type='compute',
            resource_access={'R': ['input_batch', 'weights', 'biases', 'mask_batch'], 'W': ['hidden']},
            operation_fn=_enqueue_forward,
            params={}
        )

    def compute_exit_probabilities(self, global_sizes, local_sizes, hidden_buf, exit_weights_buf, exit_biases_buf, exit_probs_buf, losses_buf, targets_buf, exit_idx, actual_batch_size):
        if min(global_sizes) < 1 or any(g > self.buffer_manager.device.max_work_item_sizes for g in global_sizes):
            raise ValueError(f"Invalid global sizes: {global_sizes}")
        def _enqueue_exit(queue, wait_for):
            return self.program.compute_exit_probabilities(
                queue, global_sizes, local_sizes, hidden_buf, exit_weights_buf, exit_biases_buf, exit_probs_buf,
                losses_buf, targets_buf, np.int32(exit_idx), self.temperatures, np.int32(self.hidden_dim),
                np.int32(self.output_classes), np.int32(self.padded_hidden_dim), np.int32(self.padded_output_classes),
                np.int32(self.padded_batch_size), np.int32(actual_batch_size), wait_for=wait_for
            )
        return self.manager.create_node(
            node_type=NodeType.COMPUTE,
            queue_type='compute',
            resource_access={'R': ['hidden', 'exit_weights', 'exit_biases', 'targets_batch', 'temps'], 'W': ['exit_probs', 'losses']},
            operation_fn=_enqueue_exit,
            params={}
        )

    def compute_gradients(self, global_sizes, local_sizes, input_buf, hidden_buf, exit_probs_buf, exit_weights_buf, grad_weights_buf, grad_biases_buf, grad_exit_weights_buf, grad_exit_biases_buf, targets_buf, actual_batch_size):
        if min(global_sizes) < 1 or any(g > self.buffer_manager.device.max_work_item_sizes for g in global_sizes):
            raise ValueError(f"Invalid global sizes: {global_sizes}")
        def _enqueue_grad(queue, wait_for):
            return self.program.compute_gradients(
                queue, global_sizes, local_sizes, input_buf, hidden_buf, exit_probs_buf, exit_weights_buf,
                grad_weights_buf, grad_biases_buf, grad_exit_weights_buf, grad_exit_biases_buf, targets_buf,
                self.temperatures, np.int32(self.input_dim), np.int32(self.hidden_dim), np.int32(self.output_classes),
                np.int32(self.padded_input_dim), np.int32(self.padded_hidden_dim), np.int32(self.padded_output_classes),
                np.int32(self.padded_batch_size), np.int32(actual_batch_size), np.int32(self.num_exits), wait_for=wait_for
            )
        return self.manager.create_node(
            node_type=NodeType.COMPUTE,
            queue_type='compute',
            resource_access={'R': ['input_batch', 'hidden', 'exit_probs', 'exit_weights', 'targets_batch', 'temps'],
                            'W': ['grad_weights', 'grad_biases', 'grad_exit_weights', 'grad_exit_biases']},
            operation_fn=_enqueue_grad,
            params={}
        )

    def compute_temp_gradients(self, global_sizes, local_sizes, exit_probs_buf, targets_buf, grad_temps_buf, actual_batch_size):
        if min(global_sizes) < 1 or any(g > self.buffer_manager.device.max_work_item_sizes for g in global_sizes):
            raise ValueError(f"Invalid global sizes: {global_sizes}")
        def _enqueue_temp_grad(queue, wait_for):
            return self.program.compute_temp_gradients(
                queue, global_sizes, local_sizes, exit_probs_buf, targets_buf, grad_temps_buf, self.temperatures,
                np.int32(self.output_classes), np.int32(self.padded_output_classes), np.int32(self.padded_batch_size),
                np.int32(actual_batch_size), np.int32(self.num_exits), wait_for=wait_for
            )
        return self.manager.create_node(
            node_type=NodeType.COMPUTE,
            queue_type='compute',
            resource_access={'R': ['exit_probs', 'targets_batch', 'temps'], 'W': ['grad_temps']},
            operation_fn=_enqueue_temp_grad,
            params={}
        )

    def adam_update(self, global_sizes, local_sizes, grad_buf, param_buf, m1_buf, m2_buf, grad_name, param_name, m1_name, m2_name, total_params):
        if min(global_sizes) < 1 or any(g > self.buffer_manager.device.max_work_item_sizes for g in global_sizes):
            raise ValueError(f"Invalid global sizes: {global_sizes}")
        def _enqueue_adam(queue, wait_for):
            return self.program.adam_update(
                queue, global_sizes, local_sizes, grad_buf, param_buf, m1_buf, m2_buf, self.adam_beta1,
                self.adam_beta2, self.beta1_t, self.beta2_t, self.learning_rate, self.epsilon, np.int32(total_params),
                wait_for=wait_for
            )
        return self.manager.create_node(
            node_type=NodeType.COMPUTE,
            queue_type='compute',
            resource_access={'R': [grad_name, param_name, m1_name, m2_name], 'W': [param_name, m1_name, m2_name]},
            operation_fn=_enqueue_adam,
            params={}
        )

    def clamp_temperatures(self, global_sizes, local_sizes, temps_buf):
        if min(global_sizes) < 1 or any(g > self.buffer_manager.device.max_work_item_sizes for g in global_sizes):
            raise ValueError(f"Invalid global sizes: {global_sizes}")
        def _enqueue_clamp(queue, wait_for):
            return self.program.clamp_temperatures(
                queue, global_sizes, local_sizes,
                temps_buf, self.min_temp, self.max_temp,
                np.int32(self.num_exits),
                wait_for=wait_for
            )
        return self.manager.create_node(
            node_type=NodeType.COMPUTE,
            queue_type='compute',
            resource_access={'R': ['temps'], 'W': ['temps']},
            operation_fn=_enqueue_clamp,
            params={}
        )

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
loss_elements_per_exit = buffer_mgr.buffer_metadata['losses']['padded_shape'][0] // NUM_EXITS
pm._create_buffers()

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
simd_width = buffer_mgr.simd_width
build_opts = [
    f"-D VECTOR_TYPE={'float' + str(simd_width) if simd_width > 1 else 'float'}",
    f"-D SIMD_WIDTH={simd_width}",
    f"-D LOSS_STRIDE={loss_elements_per_exit}",
    f"-D USE_FAST_MATH=1"
]
program = cl.Program(ctx, "\n".join(kernel_src)).build(options=" ".join(build_opts))

# Training Loop
global_step = 1
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
        X_padded, y_padded, mask = pad_batch(X_batch, y_batch, buffer_mgr)

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
        host_processing_event = cl.UserEvent(ctx)
        def set_event_complete(_):
            host_processing_event.set_status(cl.command_execution_status.COMPLETE)
        losses_transfer = buffer_mgr.staged_transfer(None, 'losses', is_device_to_host=True, callback=set_event_complete)
        exit_probs_transfer = buffer_mgr.staged_transfer(None, 'exit_probs', is_device_to_host=True)
        temps_transfer = buffer_mgr.staged_transfer(None, 'temps', is_device_to_host=True)

        manager.commit_workload()

        # Wait for transfers to complete
        cl.wait_for_events([
            manager.execution_graph.nodes[losses_transfer]['node'].event,
            manager.execution_graph.nodes[exit_probs_transfer]['node'].event,
            manager.execution_graph.nodes[temps_transfer]['node'].event
        ])

        # Access results
        losses_host = manager.execution_graph.nodes[losses_transfer]['node'].params['destination']
        exit_probs_host = manager.execution_graph.nodes[exit_probs_transfer]['node'].params['destination']
        temps_host = manager.execution_graph.nodes[temps_transfer]['node'].params['destination']

        exit_losses = []
        for exit_idx in range(NUM_EXITS):
            start = exit_idx * loss_elements_per_exit
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

if __name__ == "__main__":
    pass