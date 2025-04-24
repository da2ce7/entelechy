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
                if (res in existing.resources['W'] or
                    (res in existing.resources['R'] and res in resource_access['W'])):
                    node.dependencies.add(existing.uid)

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
                if self.execution_graph.nodes[d]['node'].queue_type != node.queue_type
            ]

            event = node.execute(queue, wait_for=wait_for)
            completion_events[uid] = event
            schedule[node.queue_type].append(event)

            for res in node.resources['W']:
                self.logical_resources[res] = (
                    self.logical_resources[res][0],
                    self.logical_resources[res][1] + 1
                )

        return schedule

    def reset_graph(self):
        """Reset the execution graph for a new iteration."""
        self.execution_graph.clear()
        self.node_id_counter = 0

# BufferManager Definition
class BufferManager:
    """Manages OpenCL buffers and data transfers."""
    def __init__(self, context, device, work_manager):
        self.ctx = context
        self.device = device
        self.work_manager = work_manager

        # Memory configuration
        self.simd_width = select_simd_width(device)
        self.wavefront_size = 64 if "AMD" in device.vendor else 32
        self.min_alignment = max(
            device.min_data_type_align_size,
            self.simd_width * np.float32().itemsize
        )

        # Memory pools
        self.device_pools = defaultdict(deque)
        self.pinned_pools = defaultdict(deque)
        self.staging_pool = deque(maxlen=8)
        self.access_tracker = defaultdict(lambda: {'reads': 0, 'writes': 0})

        # Version tracking integrated with WorkManager
        self.buffer_versions = {}
        work_manager.logical_resources.update(self.buffer_versions)

    def acquire_buffer(self, name, shape, dtype, layout='auto', mode='device'):
        """Acquire a buffer with specified properties."""
        padded_shape = self._pad_shape(shape, dtype, layout)
        buffer_key = self._make_key(padded_shape, dtype, mode)

        buffer = self._try_reuse(buffer_key, mode)
        if not buffer:
            buffer = self._create_new_buffer(padded_shape, dtype, mode)

        if name not in self.buffer_versions:
            self.buffer_versions[name] = (buffer, 0)
        else:
            self.buffer_versions[name] = (buffer, self.buffer_versions[name][1] + 1)

        return buffer

    def staged_transfer(self, host_array, buffer_name, callback=None):
        """Perform a staged host-device or device-host transfer."""
        buffer, version = self.buffer_versions.get(buffer_name, (None, 0))
        if not buffer:
            buffer = self.acquire_buffer(buffer_name, host_array.shape, host_array.dtype)

        staging_buf = self._get_staging_buffer(host_array.nbytes)

        if host_array is not None:  # Host to device
            map_evt = cl.enqueue_map_buffer(
                self.work_manager.hardware_queues['xfer'],
                staging_buf,
                host_array.shape,
                host_array.dtype,
                cl.map_flags.WRITE
            )
            np.copyto(map_evt[0], host_array)
            unmap_evt = cl.enqueue_unmap_mem_object(
                self.work_manager.hardware_queues['xfer'],
                staging_buf,
                map_evt[0]
            )
            copy_evt = cl.enqueue_copy_buffer(
                self.work_manager.hardware_queues['xfer'],
                staging_buf, buffer,
                wait_for=[unmap_evt]
            )
            self._update_buffer_access(buffer_name, 'write', copy_evt)
            if callback:
                copy_evt.set_callback(cl.command_execution_status.COMPLETE, callback)
            return copy_evt
        else:  # Device to host
            host_array = np.empty(buffer.shape, dtype=buffer.dtype)
            map_evt = cl.enqueue_map_buffer(
                self.work_manager.hardware_queues['xfer'],
                staging_buf,
                host_array.shape,
                host_array.dtype,
                cl.map_flags.WRITE
            )
            copy_evt = cl.enqueue_copy_buffer(
                self.work_manager.hardware_queues['xfer'],
                buffer, staging_buf
            )
            np.copyto(host_array, map_evt[0])
            unmap_evt = cl.enqueue_unmap_mem_object(
                self.work_manager.hardware_queues['xfer'],
                staging_buf,
                map_evt[0]
            )
            self._update_buffer_access(buffer_name, 'read', unmap_evt)
            if callback:
                unmap_evt.set_callback(cl.command_execution_status.COMPLETE, callback)
            return (host_array, unmap_evt)

    def _pad_shape(self, shape, dtype, layout_mode):
        """Pad shape to align with device requirements."""
        item_size = np.dtype(dtype).itemsize
        alignment = lcm(self.min_alignment, item_size)
        if layout_mode == 'dense' or (layout_mode == 'auto' and self.simd_width > 1):
            return tuple((dim + alignment - 1) // alignment * alignment for dim in shape)
        return shape

    def _update_buffer_access(self, buffer_name, access_type, event):
        """Update buffer access tracking and dependencies."""
        self.access_tracker[buffer_name][access_type + 's'] += 1
        node_id = self.work_manager.create_node(
            queue_type='xfer',
            resource_access={access_type.upper(): {buffer_name}},
            operation_fn=lambda *_: event
        )
        self.work_manager.execution_graph.nodes[node_id]['buffer'] = buffer_name

    def register_layer_buffers(self, layer_config):
        """Register multiple buffers for a layer."""
        for name, spec in layer_config.items():
            layout = 'dense' if 'weight' in name else 'auto'
            self.acquire_buffer(
                name=name,
                shape=spec['shape'],
                dtype=spec['dtype'],
                layout=layout,
                mode='pinned' if spec.get('pinned', False) else 'device'
            )

    def _create_new_buffer(self, shape, dtype, mode):
        """Create a new OpenCL buffer."""
        buffer_size = np.prod(shape) * np.dtype(dtype).itemsize
        flags = cl.mem_flags.READ_WRITE | (cl.mem_flags.ALLOC_HOST_PTR if mode == 'pinned' else 0)
        return cl.Buffer(self.ctx, flags, buffer_size)

    def _get_staging_buffer(self, nbytes):
        """Get or create a staging buffer."""
        aligned_size = ((nbytes + self.wavefront_size - 1) // self.wavefront_size) * self.wavefront_size
        for buf in self.staging_pool:
            if buf.size >= aligned_size:
                self.staging_pool.remove(buf)
                return buf
        return cl.Buffer(self.ctx, cl.mem_flags.READ_WRITE | cl.mem_flags.ALLOC_HOST_PTR, aligned_size)

    def _try_reuse(self, buffer_key, mode):
        """Attempt to reuse an existing buffer."""
        pool = self.pinned_pools if mode == 'pinned' else self.device_pools
        if pool[buffer_key]:
            return pool[buffer_key].pop()
        return None

    def release_buffer(self, buffer_name):
        """Release a buffer back to the pool."""
        if buffer_name in self.buffer_versions:
            buffer, version = self.buffer_versions[buffer_name]
            buffer_key = self._make_key_from_buffer(buffer)
            pool = self.pinned_pools if buffer.host_accessible else self.device_pools
            pool[buffer_key].append(buffer)
            del self.buffer_versions[buffer_name]

    def _make_key(self, shape, dtype, mode):
        return (tuple(shape), np.dtype(dtype).str, mode)

    def _make_key_from_buffer(self, buffer):
        return (buffer.size, buffer.dtype.str, buffer.host_accessible)

# ParamManager Definition
class ParamManager:
    """Manages neural network parameters and their buffers."""
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
            'init': lambda s: np.ones(s, dtype=np.float32),
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
        """Register a parameter with its properties."""
        if ptype not in self.PARAM_TYPES:
            raise ValueError(f"Unknown parameter type: {ptype}")
        self.params[name] = {
            'shape': shape,
            'ptype': ptype,
            'requires_grad': requires_grad
        }

    def _create_buffers(self):
        """Create buffers for all registered parameters."""
        for name, spec in self.params.items():
            ptype = spec['ptype']
            handler = self.PARAM_TYPES[ptype]
            init_values = handler['init'](spec['shape'])
            processed = handler['preprocess'](init_values, self.buffer_manager.simd_width)
            buffer = self.buffer_manager.acquire_buffer(name, processed.shape, np.float32)
            self.buffers[name] = buffer
            if spec['requires_grad']:
                self.grad_buffers[name] = self.buffer_manager.acquire_buffer(f"grad_{name}", processed.shape, np.float32)
                self.m1_buffers[name] = self.buffer_manager.acquire_buffer(f"m1_{name}", processed.shape, np.float32)
                self.m2_buffers[name] = self.buffer_manager.acquire_buffer(f"m2_{name}", processed.shape, np.float32)

    def zero_gradients(self, work_manager):
        """Zero out gradient buffers."""
        for name, spec in self.params.items():
            if spec['requires_grad']:
                grad_buffer = self.grad_buffers[name]
                def _zero_fn(queue, wait_for):
                    return cl.enqueue_fill_buffer(queue, grad_buffer, np.float32(0), 0, grad_buffer.size, wait_for=wait_for)
                work_manager.create_node(
                    queue_type='compute',
                    resource_access={'W': {f'grad_{name}'}},
                    operation_fn=_zero_fn
                )

    def __getitem__(self, name):
        """Access a parameter buffer by name."""
        assert name in self.params, f"Unknown parameter {name}"
        return self.buffers[name]

# KernelWrapper Definition
class KernelWrapper:
    """Wraps OpenCL kernels for neural network operations."""
    def __init__(self, program, manager, global_step, compute_queue, padded_input_dim, padded_hidden_dim, padded_output_classes, padded_batch_size):
        self.program = program
        self.manager = manager
        self.queue = compute_queue
        self.mask = None
        self.temperatures = None

        self.padded_input_dim = np.int32(padded_input_dim)
        self.padded_hidden_dim = np.int32(padded_hidden_dim)
        self.padded_output_classes = np.int32(padded_output_classes)
        self.padded_batch_size = np.int32(padded_batch_size)

        self.input_dim = np.int32(INPUT_DIM)
        self.hidden_dim = np.int32(HIDDEN_DIM)
        self.output_classes = np.int32(OUTPUT_CLASSES)
        self.num_exits = np.int32(NUM_EXITS)
        self.adam_beta1 = np.float32(ADAM_BETA1)
        self.adam_beta2 = np.float32(ADAM_BETA2)
        self.learning_rate = np.float32(LEARNING_RATE)
        self.min_temp = np.float32(MIN_TEMP)
        self.max_temp = np.float32(MAX_TEMP)
        self.epsilon = np.float32(EPSILON)

        self.beta1_t = np.float32(1 / (1 - ADAM_BETA1 ** global_step))
        self.beta2_t = np.float32(1 / (1 - ADAM_BETA2 ** global_step))

    def set_mask(self, mask):
        self.mask = mask
        return self

    def set_temps(self, temps):
        self.temperatures = temps
        return self

    def forward_pass(self, global_sizes, local_sizes, input_buf, weights_buf, biases_buf, hidden_buf):
        """Enqueue forward pass kernel."""
        local_mem = cl.LocalMemory(local_sizes[0] * FLOAT_SIZE)
        def _enqueue_forward(queue, wait_for):
            return self.program.forward_pass(
                queue, global_sizes, local_sizes, local_mem, input_buf, weights_buf, biases_buf, hidden_buf,
                self.mask, self.input_dim, self.hidden_dim, self.padded_hidden_dim, self.padded_batch_size,
                wait_for=wait_for
            )
        return self.manager.create_node(
            queue_type='compute',
            resource_access={'R': {'input_batch', 'weights', 'biases', 'mask_batch'}, 'W': {'hidden'}},
            operation_fn=_enqueue_forward
        )

    def compute_exit_probabilities(self, global_sizes, local_sizes, hidden_buf, exit_weights_buf, 
                                   exit_biases_buf, exit_probs_buf, losses_buf, targets_buf, exit_idx):
        """Enqueue exit probabilities computation kernel."""
        local_mem = cl.LocalMemory(local_sizes[0] * FLOAT_SIZE)
        def _enqueue_exit(queue, wait_for):
            return self.program.compute_exit_probabilities(
                queue, global_sizes, local_sizes, hidden_buf, exit_weights_buf, exit_biases_buf,
                exit_probs_buf, losses_buf, targets_buf, self.mask, self.temperatures,
                self.hidden_dim, self.padded_hidden_dim, self.output_classes, self.padded_output_classes,
                self.num_exits, self.padded_batch_size, exit_idx, wait_for=wait_for
            )
        return self.manager.create_node(
            queue_type='compute',
            resource_access={'R': {'hidden', 'exit_weights', 'exit_biases', 'targets_batch', 'mask_batch', 'temps'},
                            'W': {'exit_probs', 'losses'}},
            operation_fn=_enqueue_exit
        )

    def compute_gradients(self, global_sizes, local_sizes, input_buf, hidden_buf, exit_probs_buf, 
                          exit_weights_buf, grad_weights_buf, grad_biases_buf, grad_exit_weights_buf, 
                          grad_exit_biases_buf, targets_buf):
        """Enqueue gradient computation kernel."""
        local_mem = cl.LocalMemory(local_sizes[0] * FLOAT_SIZE)
        def _enqueue_grad(queue, wait_for):
            return self.program.compute_gradients(
                queue, global_sizes, local_sizes, local_mem, input_buf, hidden_buf, exit_probs_buf, 
                exit_weights_buf, grad_weights_buf, grad_biases_buf, grad_exit_weights_buf, 
                grad_exit_biases_buf, targets_buf, self.mask, self.temperatures, self.input_dim, 
                self.hidden_dim, self.padded_hidden_dim, self.output_classes, self.padded_output_classes, 
                self.num_exits, self.padded_batch_size, wait_for=wait_for
            )
        return self.manager.create_node(
            queue_type='compute',
            resource_access={'R': {'input_batch', 'hidden', 'exit_probs', 'exit_weights', 'targets_batch', 'mask_batch', 'temps'},
                             'W': {'grad_weights', 'grad_biases', 'grad_exit_weights', 'grad_exit_biases'}},
            operation_fn=_enqueue_grad
        )

    def compute_temp_gradients(self, global_sizes, local_sizes, exit_probs_buf, targets_buf, grad_temps_buf):
        """Enqueue temperature gradient computation kernel."""
        def _enqueue_temp_grad(queue, wait_for):
            return self.program.compute_temp_gradients(
                queue, global_sizes, local_sizes, exit_probs_buf, targets_buf, grad_temps_buf,
                self.mask, self.temperatures, self.num_exits, self.padded_batch_size, wait_for=wait_for
            )
        return self.manager.create_node(
            queue_type='compute',
            resource_access={'R': {'exit_probs', 'targets_batch', 'mask_batch', 'temps'}, 'W': {'grad_temps'}},
            operation_fn=_enqueue_temp_grad
        )

    def adam_update(self, global_sizes, local_sizes, grad_param, param_buffer, m1_param, m2_param, total_params):
        """Enqueue Adam optimization update kernel."""
        param_name = [k for k, v in pm.buffers.items() if v == param_buffer][0]
        def _enqueue_update(queue, wait_for):
            return self.program.adam_update(
                queue, global_sizes, local_sizes, grad_param, param_buffer, m1_param, m2_param,
                self.learning_rate, self.adam_beta1, self.adam_beta2, self.beta1_t, self.beta2_t,
                self.min_temp, self.max_temp, self.epsilon, total_params, wait_for=wait_for
            )
        return self.manager.create_node(
            queue_type='compute',
            resource_access={'R': {f'grad_{param_name}'}, 'W': {param_name, f'm1_{param_name}', f'm2_{param_name}'}},
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
            if d <= max_wi_sizes[i] and (len(local_sizes) == 0 or np.prod(local_sizes) * d <= max_wg_size):
                local_sizes.append(d)
                break
        else:
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

# ParamManager Initialization
pm = ParamManager(ctx, buffer_mgr)
pm.register_parameter('weights', (INPUT_DIM, HIDDEN_DIM), 'dense_weight')
pm.register_parameter('biases', (HIDDEN_DIM,), 'bias_vector')
pm.register_parameter('exit_weights', (NUM_EXITS, HIDDEN_DIM, OUTPUT_CLASSES), 'exit_weight')
pm.register_parameter('exit_biases', (NUM_EXITS, OUTPUT_CLASSES), 'bias_vector')
pm.register_parameter('temps', (NUM_EXITS,), 'temperature', requires_grad=True)
pm._create_buffers()

# Compute padded dimensions
padded_input_dim = ((INPUT_DIM + simd_width - 1) // simd_width) * simd_width
padded_hidden_dim = ((HIDDEN_DIM + simd_width - 1) // simd_width) * simd_width
padded_output_classes = ((OUTPUT_CLASSES + simd_width - 1) // simd_width) * simd_width
max_padded_batch = ((BATCH_SIZE + simd_width - 1) // simd_width) * simd_width
loss_elements_per_exit = max_padded_batch

# Kernel Compilation
kernel_src = []
for fname in CL_KERNEL_FILES:
    with open(fname) as f:
        kernel_src.append(f.read())
build_opts = [
    f"-D VECTOR_TYPE={'float' + str(simd_width) if simd_width > 1 else 'float'}",
    f"-D SIMD_WIDTH={simd_width}",
    f"-D LOSS_STRIDE={loss_elements_per_exit}",
    f"-D USE_FAST_MATH=1"
]
program = cl.Program(ctx, "\n".join(kernel_src)).build(options=" ".join(build_opts))

# Buffer Creation for Non-Parameters
buffers = {
    'hidden': buffer_mgr.acquire_buffer('hidden', (max_padded_batch, padded_hidden_dim), np.float32),
    'exit_probs': buffer_mgr.acquire_buffer('exit_probs', (max_padded_batch, NUM_EXITS, padded_output_classes), np.float32),
    'losses': buffer_mgr.acquire_buffer('losses', (NUM_EXITS, loss_elements_per_exit), np.float32),
}

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
        kernel_wrapper = KernelWrapper(
            program, manager, global_step, compute_queue, padded_input_dim, padded_hidden_dim,
            padded_output_classes, max_padded_batch
        ).set_mask(buffer_mgr.buffer_versions['mask_batch'][0]).set_temps(pm['temps'])

        # Zero gradients
        pm.zero_gradients(manager)

        # Forward pass
        global_forward = (max_padded_batch, padded_hidden_dim // simd_width)
        local_forward = optimal_local_sizes(global_forward, device_limits)
        kernel_wrapper.forward_pass(
            global_forward, local_forward, buffer_mgr.buffer_versions['input_batch'][0], pm['weights'],
            pm['biases'], buffers['hidden']
        )

        # Early exits
        for exit_idx in range(NUM_EXITS):
            global_exit = (max_padded_batch,)
            local_exit = optimal_local_sizes(global_exit, device_limits)
            kernel_wrapper.compute_exit_probabilities(
                global_exit, local_exit, buffers['hidden'], pm['exit_weights'], pm['exit_biases'],
                buffers['exit_probs'], buffers['losses'], buffer_mgr.buffer_versions['targets_batch'][0], np.int32(exit_idx)
            )

        # Backpropagation
        global_grad = (padded_input_dim // simd_width, HIDDEN_DIM)
        local_grad = optimal_local_sizes(global_grad, device_limits)
        kernel_wrapper.compute_gradients(
            global_grad, local_grad, buffer_mgr.buffer_versions['input_batch'][0], buffers['hidden'], buffers['exit_probs'],
            pm['exit_weights'], pm.grad_buffers['weights'], pm.grad_buffers['biases'], 
            pm.grad_buffers['exit_weights'], pm.grad_buffers['exit_biases'], buffer_mgr.buffer_versions['targets_batch'][0]
        )

        # Compute temperature gradients
        global_temp_grad = (NUM_EXITS,)
        local_temp_grad = optimal_local_sizes(global_temp_grad, device_limits)
        kernel_wrapper.compute_temp_gradients(
            global_temp_grad, local_temp_grad, buffers['exit_probs'], buffer_mgr.buffer_versions['targets_batch'][0], 
            pm.grad_buffers['temps']
        )

        # Adam update for each parameter
        for param in ['weights', 'biases', 'exit_weights', 'exit_biases', 'temps']:
            if param in pm.params and pm.params[param]['requires_grad']:
                grad_param = pm.grad_buffers[param]
                param_buffer = pm.buffers[param]
                m1_param = pm.m1_buffers[param]
                m2_param = pm.m2_buffers[param]
                total_params = param_buffer.size // FLOAT_SIZE
                global_adam = (total_params,)
                local_adam = optimal_local_sizes(global_adam, device_limits)
                kernel_wrapper.adam_update(
                    global_adam, local_adam, grad_param, param_buffer, m1_param, m2_param, np.int32(total_params)
                )

        # Device-to-host transfers
        host_processing_event = cl.UserEvent(ctx)
        losses_host, losses_evt = buffer_mgr.staged_transfer(None, 'losses', callback=lambda _: host_processing_event.set_status(cl.command_execution_status.COMPLETE))
        exit_probs_host, exit_probs_evt = buffer_mgr.staged_transfer(None, 'exit_probs', callback=lambda _: None)
        temps_host, temps_evt = buffer_mgr.staged_transfer(None, 'temps', callback=lambda _: None)

        # Execute workload
        schedule = manager.commit_workload()

        # Wait for transfers
        cl.wait_for_events([losses_evt, exit_probs_evt, temps_evt, host_processing_event])

        # Process results
        exit_losses = []
        for exit_idx in range(NUM_EXITS):
            start = exit_idx * loss_elements_per_exit
            end = start + actual_batch_size
            exit_loss = losses_host[start:end].mean()
            exit_losses.append(float(exit_loss))
        print(f"Batch {batch_idx}: Per-exit Losses: {exit_losses}")

        valid_probs = exit_probs_host[:actual_batch_size].reshape(actual_batch_size, NUM_EXITS, padded_output_classes)[:, :, :OUTPUT_CLASSES]
        valid_mask = mask[:actual_batch_size].astype(bool)
        confidences = np.array([
            valid_probs[valid_mask, i, :].max(axis=1) ** (1 / (temps_host[i] + 1e-8))
            for i in range(NUM_EXITS)
        ])
        weights = np.exp(confidences) / np.sum(np.exp(confidences), axis=0)
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
    print(f"Temperatures: {temps_host}")

# Cleanup
buffer_mgr.release_buffer('input_batch')
buffer_mgr.release_buffer('targets_batch')
buffer_mgr.release_buffer('mask_batch')
buffer_mgr.release_buffer('hidden')
buffer_mgr.release_buffer('exit_probs')
buffer_mgr.release_buffer('losses')

if __name__ == "__main__":
    pass