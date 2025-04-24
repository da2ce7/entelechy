import pyopencl as cl
import numpy as np
import math
from math import gcd
from sklearn.datasets import load_iris
from sklearn.preprocessing import StandardScaler
import networkx as nx
import re
from collections import deque, defaultdict

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

# Utility Functions
def lcm(a, b):
    """Calculate the least common multiple of two numbers."""
    return a * b // gcd(a, b)

def next_pow2(n):
    """Return the next power of 2 greater than or equal to n."""
    return 1 if n == 0 else 1 << (n - 1).bit_length()

def divisors(n):
    """Return a sorted list of divisors of n."""
    divs = set()
    for i in range(1, int(math.sqrt(n)) + 1):
        if n % i == 0:
            divs.add(i)
            divs.add(n//i)
    return sorted(divs, reverse=True)

def he_init(shape):
    """He initialization for weights."""
    if len(shape) == 2:
        fan_in = shape[0]
    elif len(shape) == 3:
        fan_in = shape[1]
    else:
        raise ValueError("Unsupported shape for He initialization")
    scale = np.sqrt(2.0 / fan_in)
    return np.random.normal(0, scale, shape).astype(np.float32)

def preprocess_weights(w, simd_width):
    """Preprocess weights for SIMD alignment."""
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
    """Pad a 1D array to a multiple of simd_width."""
    padded_size = ((arr.shape[0] + simd_width - 1) // simd_width) * simd_width
    arr_padded = np.zeros(padded_size, dtype=arr.dtype)
    arr_padded[:arr.shape[0]] = arr
    return arr_padded

def select_simd_width(device):
    """Select SIMD width based on device vendor and capabilities."""
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

# WorkManager Definition
class WorkManager:
    """Manages execution graph and hardware queues for OpenCL operations."""
    def __init__(self, context):
        self.context = context
        self.execution_graph = nx.MultiDiGraph()
        self.logical_resources = {}
        self.hardware_queues = {}
        self.node_id_counter = 0

    class ExecutionNode:
        """Represents a node in the execution graph."""
        __slots__ = ['uid', 'queue_type', 'execute', 'resources', 'dependencies']
        def __init__(self, uid):
            self.uid = uid
            self.queue_type = None
            self.execute = None
            self.resources = {'R': set(), 'W': set()}
            self.dependencies = set()

    def create_node(self, queue_type, resource_access, operation_fn):
        """Create a new execution node with dependencies."""
        node = self.ExecutionNode(self.node_id_counter)
        node.queue_type = queue_type
        node.execute = operation_fn
        node.resources = resource_access

        for res in resource_access['R'] | resource_access['W']:
            for existing in self.execution_graph.nodes.values():
                existing_node = existing['node']
                if (res in existing_node.resources['W'] or
                    (res in existing_node.resources['R'] and res in resource_access['W'])):
                    node.dependencies.add(existing_node.uid)

        self.execution_graph.add_node(node.uid, node=node)
        self.node_id_counter += 1
        return node.uid

    def commit_workload(self):
        """Schedule and execute the workload."""
        ordered_nodes = list(nx.lexicographical_topological_sort(
            self.execution_graph,
            key=lambda x: self.execution_graph.nodes[x]['node'].queue_type
        ))

        schedule = {qt: [] for qt in self.hardware_queues}
        completion_events = {}

        for uid in ordered_nodes:
            node = self.execution_graph.nodes[uid]['node']
            queue = self.hardware_queues[node.queue_type]

            wait_for = [
                completion_events[d] for d in node.dependencies
                if d in completion_events and self.execution_graph.nodes[d]['node'].queue_type != node.queue_type
            ]

            event = node.execute(queue, wait_for=wait_for)
            completion_events[uid] = event
            schedule[node.queue_type].append(event)

            for res in node.resources['W']:
                if res in self.logical_resources:
                    buffer, version = self.logical_resources[res]
                    self.logical_resources[res] = (buffer, version + 1)
                else:
                    self.logical_resources[res] = (None, 0)

        return schedule

    def reset_graph(self):
        """Reset the execution graph for a new iteration."""
        self.execution_graph.clear()
        self.node_id_counter = 0

# BufferManager Definition
class BufferManager:
    """Manages OpenCL buffers and data transfers with padding strategies and memory safety checks."""
    DEVICE_MEMORY_WARNING = 0.8  # 80% threshold

    def __init__(self, context, device, work_manager, default_pad_strategy='opencl_optimal'):
        self.ctx = context
        self.device = device
        self.work_manager = work_manager
        self.simd_width = select_simd_width(device)
        self.wavefront_size = 64 if "AMD" in device.vendor else 32
        self.min_alignment = max(device.min_data_type_align_size, self.simd_width * np.float32().itemsize)
        self.buffer_metadata = {}
        self.pad_strategies = {'opencl_optimal': self._default_pad_strategy}
        self.default_pad_strategy = default_pad_strategy
        self.staging_pool = deque(maxlen=8)
        self.buffer_versions = {}
        self.device_pools = defaultdict(list)
        self.pinned_pools = defaultdict(list)
        work_manager.logical_resources.update(self.buffer_versions)
        self.active_buffers = set()

    def register_pad_strategy(self, name, callback):
        """Add a custom padding strategy."""
        self.pad_strategies[name] = callback

    def acquire_buffer(self, name, real_shape, dtype, pad_strategy=None, mode='device'):
        """Acquire a buffer with the specified padding strategy, reusing if possible."""
        if name in self.buffer_versions:
            existing_meta = self.buffer_metadata[name]
            pad_ok = (np.array(real_shape) <= np.array(existing_meta['real_shape'])).all() and pad_strategy == existing_meta['strategy']
            if pad_ok and existing_meta['buffer'].size >= np.prod(real_shape) * dtype().itemsize:
                return existing_meta['buffer']
            else:
                self.release_buffer(name)

        strategy = self.pad_strategies[pad_strategy or self.default_pad_strategy]
        padded_shape = strategy(real_shape, dtype)
        buffer_key = (tuple(padded_shape), dtype.__name__, mode)
        buffer = self._try_reuse(buffer_key, mode)
        if not buffer:
            buffer = self._create_buffer(padded_shape, dtype, mode)
        self.buffer_metadata[name] = {
            'real_shape': tuple(real_shape),
            'padded_shape': padded_shape,
            'dtype': dtype,
            'strategy': pad_strategy or self.default_pad_strategy,
            'buffer': buffer
        }
        self.buffer_versions[name] = (buffer, 0)
        self.active_buffers.add(buffer)
        return buffer

    def _default_pad_strategy(self, real_shape, dtype):
        """Default padding strategy for SIMD alignment."""
        item_size = np.dtype(dtype).itemsize
        alignment = lcm(self.min_alignment, item_size)
        return tuple((dim + alignment - 1) // alignment * alignment for dim in real_shape)

    def get_dimensions(self, name):
        """Get (real_shape, padded_shape) for a buffer."""
        if name not in self.buffer_metadata:
            raise KeyError(f"Buffer '{name}' not found")
        return (self.buffer_metadata[name]['real_shape'], self.buffer_metadata[name]['padded_shape'])

    def staged_transfer(self, host_data, buffer_name, is_device_to_host=False, callback=None):
        """Perform a staged host-device or device-host transfer with padding."""
        metadata = self.buffer_metadata[buffer_name]
        buffer = metadata['buffer']
        padded_shape = metadata['padded_shape']
        real_shape = metadata['real_shape']
        dtype = metadata['dtype']

        if not is_device_to_host:  # Host to device
            padded_data = np.zeros(padded_shape, dtype=dtype)
            slices = tuple(slice(0, r) for r in real_shape)
            padded_data[slices] = host_data
            staging_buf = self._get_staging_buffer(padded_data.nbytes)
            map_evt = cl.enqueue_map_buffer(
                self.work_manager.hardware_queues['xfer'],
                staging_buf,
                cl.map_flags.WRITE,
                0,
                padded_data.shape,
                dtype,
                wait_for=[],
                is_blocking=False
            )
            with map_evt[0] as mapped:
                np.copyto(mapped, padded_data)
            unmap_evt = cl.enqueue_unmap_mem_object(
                self.work_manager.hardware_queues['xfer'],
                staging_buf,
                map_evt[0],
                wait_for=[map_evt[1]]
            )
            copy_evt = cl.enqueue_copy_buffer(
                self.work_manager.hardware_queues['xfer'],
                staging_buf,
                buffer,
                src_offset=0,
                dst_offset=0,
                size=buffer.size,
                wait_for=[unmap_evt]
            )
            self.buffer_versions[buffer_name] = (buffer, self.buffer_versions[buffer_name][1] + 1)
            if callback:
                copy_evt.set_callback(cl.command_execution_status.COMPLETE, callback)
            return copy_evt
        else:  # Device to host
            host_array = np.empty(padded_shape, dtype=dtype)
            staging_buf = self._get_staging_buffer(host_array.nbytes)
            copy_evt = cl.enqueue_copy_buffer(
                self.work_manager.hardware_queues['xfer'],
                buffer,
                staging_buf,
                src_offset=0,
                dst_offset=0,
                size=buffer.size,
                wait_for=[]
            )
            map_evt = cl.enqueue_map_buffer(
                self.work_manager.hardware_queues['xfer'],
                staging_buf,
                cl.map_flags.READ,
                0,
                host_array.shape,
                dtype,
                wait_for=[copy_evt],
                is_blocking=False
            )
            with map_evt[0] as mapped:
                np.copyto(host_array, mapped)
            unmap_evt = cl.enqueue_unmap_mem_object(
                self.work_manager.hardware_queues['xfer'],
                staging_buf,
                map_evt[0],
                wait_for=[map_evt[1]]
            )
            self.buffer_versions[buffer_name] = (buffer, self.buffer_versions[buffer_name][1] + 1)
            if callback:
                unmap_evt.set_callback(cl.command_execution_status.COMPLETE, callback)
            slices = tuple(slice(0, r) for r in real_shape)
            return host_array[slices]

    def _create_buffer(self, shape, dtype, mode):
        """Create a new OpenCL buffer."""
        buffer_size = np.prod(shape) * np.dtype(dtype).itemsize
        flags = cl.mem_flags.READ_WRITE
        if mode == 'pinned':
            flags |= cl.mem_flags.ALLOC_HOST_PTR
        return cl.Buffer(self.ctx, flags, size=buffer_size)

    def _get_staging_buffer(self, nbytes):
        """Get or create a staging buffer."""
        aligned_size = ((nbytes + self.wavefront_size - 1) // self.wavefront_size) * self.wavefront_size
        for buf in self.staging_pool:
            if buf.size >= aligned_size:
                self.staging_pool.remove(buf)
                return buf
        buffer = cl.Buffer(self.ctx, cl.mem_flags.READ_WRITE | cl.mem_flags.ALLOC_HOST_PTR, size=aligned_size)
        self.staging_pool.append(buffer)
        return buffer

    def _try_reuse(self, buffer_key, mode):
        """Attempt to reuse an existing buffer."""
        pool = self.pinned_pools if mode == 'pinned' else self.device_pools
        if buffer_key in pool and pool[buffer_key]:
            return pool[buffer_key].pop()
        return None

    def release_buffer(self, buffer_name):
        """Release a buffer back to the pool."""
        if buffer_name in self.buffer_versions:
            buffer, _ = self.buffer_versions[buffer_name]
            mode = 'padded' if (buffer.flags & cl.mem_flags.ALLOC_HOST_PTR) else 'device'
            buffer_key = (tuple(self.buffer_metadata[buffer_name]['padded_shape']), self.buffer_metadata[buffer_name]['dtype'].__name__, mode)
            pool = self.pinned_pools if mode == 'padded' else self.device_pools
            pool[buffer_key].append(buffer)
            del self.buffer_metadata[buffer_name]
            del self.buffer_versions[buffer_name]
            self.active_buffers.remove(buffer)

    def validate_memory(self):
        """Check if memory usage exceeds the warning threshold."""
        total_alloc = sum(b.size for b in self.active_buffers)
        device_max = self.device.global_mem_size
        if total_alloc / device_max > self.DEVICE_MEMORY_WARNING:
            raise MemoryError(f"Used {total_alloc/1024**2:.2f}MB of {device_max/1024**2:.2f}MB device memory")

    def __enter__(self):
        self.validate_memory()
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        self.validate_memory()

# ParamManager Definition
class ParamManager:
    """Manages neural network parameters and their buffers with padding strategies."""
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
            'init': lambda s: np.ones(s, dtype=np.float32) * 1.0,
            'constraint': (MIN_TEMP, MAX_TEMP)
        },
        'bias_vector': {
            'init': lambda s: np.zeros(s, dtype=np.float32),
            'preprocess': lambda b, sw: pad_1d(b, sw),
            'shape_transform': lambda s, sw: (((s[0] + sw - 1) // sw) * sw,)
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

    def register_parameter(self, name, real_shape, ptype, pad_strategy='parameter_aware', requires_grad=True):
        """Register a parameter with a specific padding strategy."""
        if pad_strategy not in self.buffer_manager.pad_strategies:
            self.buffer_manager.register_pad_strategy(pad_strategy, self._param_pad_strategy)
        self.params[name] = {
            'real_shape': real_shape,
            'ptype': ptype,
            'pad_strategy': pad_strategy,
            'requires_grad': requires_grad
        }

    def _param_pad_strategy(self, real_shape, dtype):
        """Parameter-specific padding strategy."""
        if len(real_shape) == 2:  # Dense weights
            return (
                ((real_shape[0] + 63) // 64) * 64,  # Input dimension
                ((real_shape[1] + 15) // 16) * 16   # Hidden dimension
            )
        elif len(real_shape) == 3:  # Exit weights
            return (
                real_shape[0],
                real_shape[1],
                ((real_shape[2] + self.buffer_manager.simd_width - 1) // self.buffer_manager.simd_width) * self.buffer_manager.simd_width
            )
        elif len(real_shape) == 1:  # Bias or temperature
            return (((real_shape[0] + self.buffer_manager.simd_width - 1) // self.buffer_manager.simd_width) * self.buffer_manager.simd_width,)
        return self.buffer_manager._default_pad_strategy(real_shape, dtype)

    def get_parameter_spec(self, name):
        """Get real and padded shapes for a parameter."""
        real = self.params[name]['real_shape']
        padded = self.buffer_manager.buffer_metadata[name]['padded_shape']
        return {'real': real, 'padded': padded}

    def _create_buffers(self):
        """Create buffers for all registered parameters using BufferManager."""
        for name, spec in self.params.items():
            ptype = spec['ptype']
            handler = self.PARAM_TYPES[ptype]
            init_values = handler['init'](spec['real_shape'])
            if 'preprocess' in handler:
                init_values = handler['preprocess'](init_values, self.buffer_manager.simd_width)
            buffer = self.buffer_manager.acquire_buffer(name, spec['real_shape'], np.float32, pad_strategy=spec['pad_strategy'])
            self.buffer_manager.staged_transfer(init_values, name)
            self.buffers[name] = buffer
            if spec['requires_grad']:
                self.grad_buffers[name] = self.buffer_manager.acquire_buffer(f"grad_{name}", spec['real_shape'], np.float32, pad_strategy=spec['pad_strategy'])
                self.m1_buffers[name] = self.buffer_manager.acquire_buffer(f"m1_{name}", spec['real_shape'], np.float32, pad_strategy=spec['pad_strategy'])
                self.m2_buffers[name] = self.buffer_manager.acquire_buffer(f"m2_{name}", spec['real_shape'], np.float32, pad_strategy=spec['pad_strategy'])

    def zero_gradients(self, work_manager):
        """Zero out gradient buffers."""
        for name, spec in self.params.items():
            if spec['requires_grad']:
                grad_buffer = self.grad_buffers[name]
                def _zero_fn(queue, wait_for):
                    return cl.enqueue_fill_buffer(queue, grad_buffer, np.float32(0), 0, grad_buffer.size, wait_for=wait_for)
                work_manager.create_node(
                    queue_type='compute',
                    resource_access={'W': {f"grad_{name}"}},
                    operation_fn=_zero_fn
                )

    def __getitem__(self, name):
        """Access a parameter buffer by name."""
        if name not in self.params:
            raise KeyError(f"Unknown parameter {name}")
        return self.buffers[name]

# KernelWrapper Definition
class KernelWrapper:
    """Wraps OpenCL kernels for neural network operations with padding awareness and explicit batch size passing."""
    def __init__(self, program, manager, global_step, compute_queue, buffer_manager):
        self.program = program
        self.manager = manager
        self.queue = compute_queue
        self.buffer_manager = buffer_manager
        self.mask = None
        self.temperatures = None

        self.input_dim, self.padded_input_dim = self.buffer_manager.get_dimensions('input_batch')
        self.hidden_dim, self.padded_hidden_dim = self.buffer_manager.get_dimensions('hidden')
        self.output_dim, self.padded_output_dim = (OUTPUT_CLASSES,), (((OUTPUT_CLASSES + buffer_manager.simd_width - 1) // buffer_manager.simd_width) * buffer_manager.simd_width,)
        self.padded_batch_size = self.padded_input_dim[0]

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
        """Enqueue forward pass kernel with padding awareness and actual batch size."""
        local_mem_size = local_sizes[0] * local_sizes[1] * FLOAT_SIZE if len(local_sizes) > 1 else local_sizes[0] * FLOAT_SIZE
        local_mem = cl.LocalMemory(local_mem_size)
        def _enqueue_forward(queue, wait_for):
            return self.program.forward_pass(
                queue, global_sizes, local_sizes,
                local_mem, input_buf, weights_buf, biases_buf, hidden_buf,
                self.mask,
                np.int32(self.input_dim[1]), np.int32(self.hidden_dim[0]),
                np.int32(self.padded_input_dim[1]), np.int32(self.padded_hidden_dim[0]),
                np.int32(self.padded_batch_size), np.int32(actual_batch_size),
                wait_for=wait_for
            )
        return self.manager.create_node(
            queue_type='compute',
            resource_access={'R': {'input_batch', 'weights', 'biases', 'mask_batch'}, 'W': {'hidden'}},
            operation_fn=_enqueue_forward
        )

    def compute_exit_probabilities(self, global_sizes, local_sizes, hidden_buf, exit_weights_buf, 
                                   exit_biases_buf, exit_probs_buf, losses_buf, targets_buf, exit_idx, actual_batch_size):
        """Enqueue exit probabilities computation kernel with padding awareness and actual batch size."""
        local_mem = cl.LocalMemory(local_sizes[0] * FLOAT_SIZE)
        def _enqueue_exit(queue, wait_for):
            return self.program.compute_exit_probabilities(
                queue, global_sizes, local_sizes,
                hidden_buf, exit_weights_buf, exit_biases_buf, exit_probs_buf, losses_buf, targets_buf,
                self.mask, self.temperatures,
                np.int32(self.hidden_dim[0]), np.int32(self.padded_hidden_dim[0]),
                np.int32(self.output_dim[0]), np.int32(self.padded_output_dim[0]),
                np.int32(NUM_EXITS), np.int32(self.padded_batch_size), np.int32(exit_idx), np.int32(actual_batch_size),
                wait_for=wait_for
            )
        return self.manager.create_node(
            queue_type='compute',
            resource_access={'R': {'hidden', 'exit_weights', 'exit_biases', 'targets_batch', 'mask_batch', 'temps'},
                            'W': {'exit_probs', 'losses'}},
            operation_fn=_enqueue_exit
        )

    def compute_gradients(self, global_sizes, local_sizes, input_buf, hidden_buf, exit_probs_buf, 
                          exit_weights_buf, grad_weights_buf, grad_biases_buf, grad_exit_weights_buf, 
                          grad_exit_biases_buf, targets_buf, actual_batch_size):
        """Enqueue gradient computation kernel with padding awareness and actual batch size."""
        local_mem_size = local_sizes[0] * local_sizes[1] * FLOAT_SIZE if len(local_sizes) > 1 else local_sizes[0] * FLOAT_SIZE
        local_mem = cl.LocalMemory(local_mem_size)
        def _enqueue_grad(queue, wait_for):
            return self.program.compute_gradients(
                queue, global_sizes, local_sizes,
                local_mem, input_buf, hidden_buf, exit_probs_buf, exit_weights_buf,
                grad_weights_buf, grad_biases_buf, grad_exit_weights_buf, grad_exit_biases_buf, targets_buf,
                self.mask, self.temperatures,
                np.int32(self.input_dim[1]), np.int32(self.hidden_dim[0]), np.int32(self.padded_hidden_dim[0]),
                np.int32(self.output_dim[0]), np.int32(self.padded_output_dim[0]), np.int32(NUM_EXITS),
                np.int32(self.padded_batch_size), np.int32(actual_batch_size),
                wait_for=wait_for
            )
        return self.manager.create_node(
            queue_type='compute',
            resource_access={'R': {'input_batch', 'hidden', 'exit_probs', 'exit_weights', 'targets_batch', 'mask_batch', 'temps'},
                             'W': {'grad_weights', 'grad_biases', 'grad_exit_weights', 'grad_exit_biases'}},
            operation_fn=_enqueue_grad
        )

    def compute_temp_gradients(self, global_sizes, local_sizes, exit_probs_buf, targets_buf, grad_temps_buf, actual_batch_size):
        """Enqueue temperature gradient computation kernel with padding awareness and actual batch size."""
        def _enqueue_temp_grad(queue, wait_for):
            return self.program.compute_temp_gradients(
                queue, global_sizes, local_sizes,
                exit_probs_buf, targets_buf, grad_temps_buf,
                self.mask, self.temperatures,
                np.int32(NUM_EXITS), np.int32(self.padded_batch_size), np.int32(actual_batch_size),
                wait_for=wait_for
            )
        return self.manager.create_node(
            queue_type='compute',
            resource_access={'R': {'exit_probs', 'targets_batch', 'mask_batch', 'temps'}, 'W': {'grad_temps'}},
            operation_fn=_enqueue_temp_grad
        )

    def adam_update(self, global_sizes, local_sizes, grad_param, param_buffer, m1_param, m2_param, total_params):
        """Enqueue Adam optimization update kernel with padding awareness."""
        param_name = next(k for k, v in self.manager.logical_resources.items() if v[0] == param_buffer and k in pm.buffers)
        def _enqueue_update(queue, wait_for):
            return self.program.adam_update(
                queue, global_sizes, local_sizes,
                grad_param, param_buffer, m1_param, m2_param,
                self.learning_rate, self.adam_beta1, self.adam_beta2, self.beta1_t, self.beta2_t,
                self.min_temp, self.max_temp, self.epsilon, np.int32(total_params),
                wait_for=wait_for
            )
        return self.manager.create_node(
            queue_type='compute',
            resource_access={'R': {f"grad_{param_name}"}, 'W': {param_name, f"m1_{param_name}", f"m2_{param_name}"}},
            operation_fn=_enqueue_update
        )

# Helper function for optimal local sizes
def optimal_local_sizes(global_sizes, device_limits):
    """Calculate optimal local workgroup sizes."""
    max_wg_size = device_limits['max_work_group_size']
    max_wi_sizes = device_limits['max_work_item_sizes']
    local_sizes = []
    for i, gsize in enumerate(global_sizes):
        divs = divisors(gsize)
        for d in divs:
            if d <= max_wi_sizes[i] and (not local_sizes or np.prod(local_sizes) * d <= max_wg_size):
                local_sizes.append(d)
                break
        else:
            local_sizes.append(1)
    while len(local_sizes) < len(global_sizes):
        local_sizes.append(1)
    return tuple(local_sizes)

# Data Preparation
iris = load_iris()
X = iris.data.astype(np.float32)
y_true = iris.target.astype(np.int32)
scaler = StandardScaler()
X_normalized = scaler.fit_transform(X)

# OpenCL Setup
ctx = cl.create_some_context()
transfer_queue = cl.CommandQueue(ctx)
compute_queue = cl.CommandQueue(ctx, properties=cl.command_queue_properties.PROFILING_ENABLE)
device = ctx.devices[0]

device_limits = {
    'max_work_group_size': device.max_work_group_size,
    'max_work_item_sizes': device.max_work_item_sizes,
}
simd_width = select_simd_width(device)
print(f"Training on {device.name} with SIMD-{simd_width}")

# WorkManager Initialization
manager = WorkManager(ctx)
manager.hardware_queues = {'xfer': transfer_queue, 'compute': compute_queue}

# BufferManager Initialization
buffer_mgr = BufferManager(ctx, device, manager)

# Preallocate main buffers
max_batch_dim = ((BATCH_SIZE + simd_width - 1) // simd_width) * simd_width
buffer_mgr.register_pad_strategy('batch_aware', lambda s, d: (
    max_batch_dim,
    ((s[1] + simd_width - 1) // simd_width) * simd_width
))
buffer_mgr.register_pad_strategy('hidden_aware', lambda s, d: (
    ((s[0] + simd_width - 1) // simd_width) * simd_width,
    max_batch_dim
))
buffers = {
    'input_batch': buffer_mgr.acquire_buffer('input_batch', (BATCH_SIZE, INPUT_DIM), np.float32, pad_strategy='batch_aware'),
    'hidden': buffer_mgr.acquire_buffer('hidden', (HIDDEN_DIM, BATCH_SIZE), np.float32, pad_strategy='hidden_aware'),
    'targets_batch': buffer_mgr.acquire_buffer('targets_batch', (BATCH_SIZE,), np.int32, pad_strategy='batch_aware'),
    'mask_batch': buffer_mgr.acquire_buffer('mask_batch', (BATCH_SIZE,), np.float32, pad_strategy='batch_aware'),
    'exit_probs': buffer_mgr.acquire_buffer('exit_probs', (BATCH_SIZE, NUM_EXITS, OUTPUT_CLASSES), np.float32, pad_strategy='batch_aware'),
    'losses': buffer_mgr.acquire_buffer('losses', (NUM_EXITS * BATCH_SIZE,), np.float32, pad_strategy='batch_aware'),
}
loss_elements_per_exit = buffers['losses'].size // (FLOAT_SIZE * NUM_EXITS)

# ParamManager Initialization
pm = ParamManager(ctx, buffer_mgr)
pm.register_parameter('weights', (INPUT_DIM, HIDDEN_DIM), 'dense_weight')
pm.register_parameter('biases', (HIDDEN_DIM,), 'bias_vector')
pm.register_parameter('exit_weights', (NUM_EXITS, HIDDEN_DIM, OUTPUT_CLASSES), 'exit_weight')
pm.register_parameter('exit_biases', (NUM_EXITS, OUTPUT_CLASSES), 'bias_vector')
pm.register_parameter('temps', (NUM_EXITS,), 'temperature', requires_grad=True)
pm._create_buffers()

# Kernel Compilation
kernel_src = []
for fname in CL_KERNEL_FILES:
    try:
        with open(fname, 'r') as f:
            kernel_src.append(f.read())
    except FileNotFoundError:
        print(f"Warning: Kernel file {fname} not found. Assuming it exists for compilation.")
        kernel_src.append(f"// Placeholder for {fname}\n")
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

        # Prepare batch
        batch_start = batch_idx * BATCH_SIZE
        batch_end = min(batch_start + BATCH_SIZE, len(X))
        actual_batch_size = batch_end - batch_start
        batch_indices = shuffled_indices[batch_start:batch_end]
        X_batch = X_normalized[batch_indices]
        y_batch = y_true[batch_indices]

        # Host-to-device transfers
        buffer_mgr.staged_transfer(X_batch, 'input_batch')
        buffer_mgr.staged_transfer(y_batch, 'targets_batch')
        mask = np.ones(actual_batch_size, dtype=np.float32)
        buffer_mgr.staged_transfer(mask, 'mask_batch')

        # Kernel wrapper
        kernel_wrapper = KernelWrapper(program, manager, global_step, compute_queue, buffer_mgr)
        kernel_wrapper.set_mask(buffers['mask_batch']).set_temps(pm['temps'])

        # Zero gradients
        pm.zero_gradients(manager)

        # Forward pass
        global_forward = (buffer_mgr.buffer_metadata['input_batch']['padded_shape'][0], buffer_mgr.buffer_metadata['hidden']['padded_shape'][0] // simd_width)
        local_forward = optimal_local_sizes(global_forward, device_limits)
        kernel_wrapper.forward_pass(
            global_forward, local_forward,
            buffers['input_batch'], pm['weights'], pm['biases'], buffers['hidden'],
            actual_batch_size
        )

        # Early exits
        for exit_idx in range(NUM_EXITS):
            global_exit = (buffer_mgr.buffer_metadata['exit_probs']['padded_shape'][0],)
            local_exit = optimal_local_sizes(global_exit, device_limits)
            kernel_wrapper.compute_exit_probabilities(
                global_exit, local_exit,
                buffers['hidden'], pm['exit_weights'], pm['exit_biases'],
                buffers['exit_probs'], buffers['losses'], buffers['targets_batch'],
                exit_idx, actual_batch_size
            )

        # Backpropagation
        global_grad = (buffer_mgr.buffer_metadata['input_batch']['padded_shape'][0], HIDDEN_DIM)
        local_grad = optimal_local_sizes(global_grad, device_limits)
        kernel_wrapper.compute_gradients(
            global_grad, local_grad,
            buffers['input_batch'], buffers['hidden'], buffers['exit_probs'], pm['exit_weights'],
            pm.grad_buffers['weights'], pm.grad_buffers['biases'],
            pm.grad_buffers['exit_weights'], pm.grad_buffers['exit_biases'],
            buffers['targets_batch'], actual_batch_size
        )

        # Compute temperature gradients
        global_temp_grad = (NUM_EXITS,)
        local_temp_grad = optimal_local_sizes(global_temp_grad, device_limits)
        kernel_wrapper.compute_temp_gradients(
            global_temp_grad, local_temp_grad,
            buffers['exit_probs'], buffers['targets_batch'], pm.grad_buffers['temps'],
            actual_batch_size
        )

        # Adam update for each parameter
        for param in ['weights', 'biases', 'exit_weights', 'exit_biases', 'temps']:
            if param in pm.params and pm.params[param]['requires_grad']:
                grad_param = pm.grad_buffers[param]
                param_buffer = pm.buffers[param]
                m1_param = pm.m1_buffers[param]
                m2_param = pm.m2_buffers[param]
                total_params = np.prod(pm.get_parameter_spec(param)['padded'])  # Number of elements
                global_adam = (total_params,)
                local_adam = optimal_local_sizes(global_adam, device_limits)
                kernel_wrapper.adam_update(
                    global_adam, local_adam,
                    grad_param, param_buffer, m1_param, m2_param,
                    total_params
                )

        # Device-to-host transfers
        host_processing_event = cl.UserEvent(ctx)
        def set_host_event(_):
            host_processing_event.set_status(cl.command_execution_status.COMPLETE)
        losses_host = buffer_mgr.staged_transfer(None, 'losses', is_device_to_host=True, callback=set_host_event)
        exit_probs_host = buffer_mgr.staged_transfer(None, 'exit_probs', is_device_to_host=True)
        temps_host = buffer_mgr.staged_transfer(None, 'temps', is_device_to_host=True)

        # Execute workload
        schedule = manager.commit_workload()

        # Wait for transfers
        cl.wait_for_events([host_processing_event])

        # Process results
        exit_losses = []
        for exit_idx in range(NUM_EXITS):
            start = exit_idx * loss_elements_per_exit
            end = min(start + actual_batch_size, len(losses_host))
            if end > start:
                exit_loss = np.mean(losses_host[start:end])
                exit_losses.append(float(exit_loss))
            else:
                exit_losses.append(0.0)
        print(f"Batch {batch_idx}: Per-exit Losses: {exit_losses}")

        # Post-process exit probabilities
        exit_probs_padded_shape = buffer_mgr.buffer_metadata['exit_probs']['padded_shape']
        exit_probs_reshaped = exit_probs_host.reshape(exit_probs_padded_shape)
        valid_probs = exit_probs_reshaped[:actual_batch_size, :NUM_EXITS, :OUTPUT_CLASSES]
        valid_mask = mask.astype(bool)
        confidences = np.array([
            valid_probs[valid_mask, i, :].max(axis=1) ** (1 / (temps_host[i] + 1e-8))
            for i in range(NUM_EXITS)
        ])
        weights = np.exp(confidences) / (np.sum(np.exp(confidences), axis=0) + 1e-8)
        ensemble_probs = np.einsum('ijk,j->ik', valid_probs, weights)
        ensemble_probs /= np.sum(ensemble_probs, axis=1, keepdims=True) + 1e-8

        log_probs = np.log(ensemble_probs + 1e-8)
        batch_loss = -np.mean(log_probs[np.arange(actual_batch_size), y_batch])
        epoch_loss += batch_loss * actual_batch_size
        predicted_classes = np.argmax(ensemble_probs, axis=1)
        correct_predictions += np.sum(predicted_classes == y_batch)

        global_step += 1

    # Epoch metrics
    avg_loss = epoch_loss / len(X)
    train_acc = correct_predictions / len(X)
    print(f"Epoch {epoch:3d} | Loss: {avg_loss:.4f} | Acc: {train_acc:.1%}")
    print(f"Temperatures: {temps_host.tolist()}")

# Cleanup
for name in buffers:
    buffer_mgr.release_buffer(name)
for name in pm.buffers:
    buffer_mgr.release_buffer(name)
for name in pm.grad_buffers:
    buffer_mgr.release_buffer(f"grad_{name}")
for name in pm.m1_buffers:
    buffer_mgr.release_buffer(f"m1_{name}")
for name in pm.m2_buffers:
    buffer_mgr.release_buffer(f"m2_{name}")

if __name__ == "__main__":
    print("Training completed.")
