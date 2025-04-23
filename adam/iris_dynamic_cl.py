import pyopencl as cl
import numpy as np
import math
from math import gcd
from sklearn.datasets import load_iris
from sklearn.preprocessing import StandardScaler
import networkx as nx
import re

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

def pad_to_multiple(arr, multiple, axis):
    pad_size = (-arr.shape[axis]) % multiple
    padding = [(0, pad_size) if i == axis else (0, 0) for i in range(arr.ndim)]
    return np.pad(arr, padding, mode='constant', constant_values=0) if pad_size != 0 else arr

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

def pad_batch(X_batch, y_batch, batch_multiple, simd_width):
    orig_size = X_batch.shape[0]
    pad_size = (-orig_size % batch_multiple) if batch_multiple > 1 else 0
    padded_size = orig_size + pad_size
    X_pad = np.zeros((padded_size, X_batch.shape[1]), dtype=np.float32)
    X_pad[:orig_size] = X_batch
    y_pad = np.zeros(padded_size, dtype=np.int32)
    y_pad[:orig_size] = y_batch
    mask = np.zeros(padded_size, dtype=np.float32)
    mask[:orig_size] = 1.0
    return X_pad, y_pad, mask, padded_size

def optimal_local_sizes(global_sizes, device_limits):
    max_wg = device_limits['max_work_group_size']
    max_dims = device_limits['max_work_item_sizes']
    if isinstance(global_sizes, int) or len(global_sizes) == 1:
        global_sizes = global_sizes[0] if isinstance(global_sizes, tuple) else global_sizes
        valid = [s for s in range(1, min(max_wg, max_dims[0])+1) if global_sizes % s == 0]
        return (max(valid, key=lambda x: x, default=1),)
    else:
        gx, gy = global_sizes
        valid_x = [s for s in range(1, min(gx, max_dims[0])+1) if gx % s == 0]
        valid_y = [s for s in range(1, min(gy, max_dims[1])+1) if gy % s == 0]
        candidates = [(lx, ly) for lx in valid_x for ly in valid_y if lx * ly <= max_wg]
        return max(candidates, key=lambda x: x[0]*x[1], default=(1,1))

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

# WorkManager Definition with reset_graph method
class WorkManager:
    """Coordinates all work across queues through logical resource tracking"""
    def __init__(self, context):
        self.context = context
        self.execution_graph = nx.MultiDiGraph()
        self.logical_resources = {}  # {"input": (current_phys_buf, version)}
        self.hardware_queues = {}  # {'xfer': queue, 'compute': queue}
        self.node_id_counter = 0

    class ExecutionNode:
        __slots__ = ['uid', 'queue_type', 'execute', 'resources', 'dependencies']
        def __init__(self, uid):
            self.uid = uid
            self.queue_type = None
            self.execute = None  # (queue, deps) → cl.Event
            self.resources = {'R': set(), 'W': set()}
            self.dependencies = set()

    def create_node(self, queue_type, resource_access, operation_fn):
        """Register operation with cross-queue dependency resolution"""
        node = self.ExecutionNode(self.node_id_counter)
        node.queue_type = queue_type
        node.execute = operation_fn
        node.resources = resource_access

        # Cross-queue dependency detection
        for res in resource_access['R'] | resource_access['W']:
            for existing in self.execution_graph.nodes.values():
                if (res in existing.resources['W'] or
                   (res in existing.resources['R'] and res in resource_access['W'])):
                    node.dependencies.add(existing.uid)

        self.execution_graph.add_node(node.uid, node=node)
        self.node_id_counter += 1
        return node.uid

    def commit_workload(self):
        """Generate execution chains with automatic inter-queue sync"""
        ordered_nodes = list(nx.lexicographical_topological_sort(
            self.execution_graph,
            key=lambda x: self.execution_graph.nodes[x]['node'].queue_type
        ))

        schedule = {qt: [] for qt in self.hardware_queues}
        completion_events = {}

        for uid in ordered_nodes:
            node = self.execution_graph.nodes[uid]['node']
            queue = self.hardware_queues[node.queue_type]

            # Collect cross-queue dependencies
            wait_for = [
                completion_events[d] for d in node.dependencies
                if self.execution_graph.nodes[d]['node'].queue_type != node.queue_type
            ]

            # Execute with proper fencing
            event = node.execute(queue, wait_for=wait_for)
            completion_events[uid] = event
            schedule[node.queue_type].append(event)

            # Maintain resource versions
            for res in node.resources['W']:
                self.logical_resources[res] = (
                    self.logical_resources[res][0],
                    self.logical_resources[res][1] + 1
                )

        return schedule

    def reset_graph(self):
        """Reset the execution graph while preserving logical resources"""
        self.execution_graph.clear()
        self.node_id_counter = 0

class TransferWrapper:
    def __init__(self, context, buffer_name, buffer_size, dtype=np.float32, device_buf=None):
        self.ctx = context
        self.logical_name = buffer_name
        self.dtype = dtype
        self.element_size = np.dtype(dtype).itemsize
        self.buffer_size = buffer_size
        self.staging_bufs = [
            self._create_staging_buffer(buffer_size),
            self._create_staging_buffer(buffer_size)
        ]
        self.current_staging = 0
        self.device_buf = device_buf if device_buf is not None else cl.Buffer(
            self.ctx, cl.mem_flags.READ_WRITE, size=buffer_size
        )
        self.staging_resources = [
            f"{buffer_name}_staging0",
            f"{buffer_name}_staging1"
        ]
        self.active_ops = {0: set(), 1: set()}  # Track operations per buffer

    def _create_staging_buffer(self, size):
        return cl.Buffer(
            self.ctx,
            cl.mem_flags.READ_WRITE | cl.mem_flags.ALLOC_HOST_PTR,
            size=size
        )

    def host_to_device(self, manager, host_data, transfer_queue):
        if host_data.nbytes > self.staging_bufs[0].size:
            raise ValueError(f"Data size {host_data.nbytes} exceeds buffer capacity {self.staging_bufs[0].size}")

        staging_idx = self.current_staging
        staging_res = self.staging_resources[staging_idx]
        staging_buf = self.staging_bufs[staging_idx]

        def enqueue_transfer(queue, wait_for):
            # Map and copy to staging buffer
            host_ptr, map_event = cl.enqueue_map_buffer(
                queue, staging_buf, cl.map_flags.WRITE,
                0, host_data.shape, self.dtype,
                wait_for=wait_for
            )
            np.copyto(host_ptr, host_data)

            # Unmap and copy to device buffer
            unmap_event = cl.enqueue_unmap_mem_object(
                queue, staging_buf, host_ptr,
                wait_for=[map_event]
            )
            copy_event = cl.enqueue_copy_buffer(
                queue, staging_buf, self.device_buf,
                wait_for=[unmap_event]
            )

            # Register completion callback
            def completion_callback(event, status):
                if status == cl.command_execution_status.COMPLETE:
                    self.active_ops[staging_idx].discard(event)

            copy_event.set_callback(
                cl.command_execution_status.COMPLETE,
                completion_callback
            )
            self.active_ops[staging_idx].add(copy_event)
            return copy_event

        # Update staging index only after operation is registered
        transfer_node = manager.create_node(
            queue_type='xfer',
            resource_access={
                'W': {self.logical_name, staging_res}
            },
            operation_fn=enqueue_transfer
        )
        self.current_staging = 1 - staging_idx
        return transfer_node

    def device_to_host(self, manager, transfer_queue, host_processing_event):
        staging_idx = self.current_staging
        staging_res = self.staging_resources[staging_idx]
        staging_buf = self.staging_bufs[staging_idx]
        transfer_data = {}

        def enqueue_transfer(queue, wait_for):
            # Copy from device to staging
            copy_event = cl.enqueue_copy_buffer(
                queue, self.device_buf, staging_buf,
                wait_for=wait_for
            )

            # Map buffer for host access
            host_ptr, map_event = cl.enqueue_map_buffer(
                queue, staging_buf, cl.map_flags.READ,
                0, (self.buffer_size // self.element_size,), self.dtype,
                wait_for=[copy_event]
            )

            # Register data ready event
            data_ready = cl.UserEvent(self.ctx)
            map_event.set_callback(
                cl.command_execution_status.COMPLETE,
                lambda e, s: data_ready.set_status(s)
            )
            transfer_data[self.logical_name] = (host_ptr, data_ready)

            # Track operation completion
            self.active_ops[staging_idx].add(map_event)
            def map_completion(e, s):
                self.active_ops[staging_idx].discard(e)
            map_event.set_callback(
                cl.command_execution_status.COMPLETE,
                map_completion
            )
            return map_event

        transfer_node = manager.create_node(
            queue_type='xfer',
            resource_access={'R': {self.logical_name, staging_res}},
            operation_fn=enqueue_transfer
        )

        def enqueue_unmap(queue, wait_for):
            cl.wait_for_events([host_processing_event])
            unmap_event = cl.enqueue_unmap_mem_object(
                queue, staging_buf,
                transfer_data[self.logical_name][0],
                wait_for=wait_for
            )
            return unmap_event

        unmap_node = manager.create_node(
            queue_type='xfer',
            resource_access={'W': {staging_res}},
            operation_fn=enqueue_unmap
        )
        manager.execution_graph.add_edge(transfer_node, unmap_node)

        return transfer_node, transfer_data

    @property
    def device_buffer(self):
        return self.device_buf

    def release(self):
        for buf in self.staging_bufs:
            buf.release()
        self.device_buf.release()

class KernelWrapper:
    def __init__(self, program, manager, compute_queue, padded_input_dim, padded_hidden_dim, padded_output_classes, padded_batch_size):
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

    def set_mask(self, mask):
        self.mask = mask
        return self

    def set_temps(self, temps):
        self.temperatures = temps
        return self

    def forward_pass(self, global_sizes, local_sizes, input_buf, weights_buf, biases_buf, hidden_buf):
        local_mem = cl.LocalMemory(local_sizes[0] * FLOAT_SIZE)

        def _enqueue_forward(queue, wait_for):
            return self.program.forward_pass(
                queue, global_sizes, local_sizes, local_mem, input_buf, weights_buf, biases_buf, hidden_buf,
                self.mask, self.input_dim, self.hidden_dim, self.padded_hidden_dim, self.padded_batch_size,
                wait_for=wait_for
            )

        return self.manager.create_node(
            queue_type='compute',
            resource_access={'R': {'input', 'weights', 'biases', 'mask'}, 'W': {'hidden'}},
            operation_fn=_enqueue_forward
        )

    def compute_exit_probabilities(self, global_sizes, local_sizes, hidden_buf, exit_weights_buf, 
                                   exit_biases_buf, exit_probs_buf, losses_buf, targets_buf, exit_idx):
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
            resource_access={'R': {'hidden', 'exit_weights', 'exit_biases', 'targets', 'mask', 'temps'},
                            'W': {'exit_probs', 'losses'}},
            operation_fn=_enqueue_exit
        )

    def compute_gradients(self, global_sizes, local_sizes, input_buf, hidden_buf, exit_probs_buf, 
                          exit_weights_buf, grad_weights_buf, grad_biases_buf, targets_buf):
        local_mem = cl.LocalMemory(local_sizes[0] * FLOAT_SIZE)

        def _enqueue_grad(queue, wait_for):
            return self.program.compute_gradients(
                queue, global_sizes, local_sizes, local_mem, input_buf, hidden_buf, exit_probs_buf, exit_weights_buf,
                grad_weights_buf, grad_biases_buf, targets_buf, self.mask, self.temperatures,
                self.input_dim, self.hidden_dim, self.padded_hidden_dim, self.output_classes,
                self.padded_output_classes, self.num_exits, self.padded_batch_size, wait_for=wait_for
            )

        return self.manager.create_node(
            queue_type='compute',
            resource_access={'R': {'input', 'hidden', 'exit_probs', 'exit_weights', 'targets', 'mask', 'temps'},
                            'W': {'grad_weights', 'grad_biases'}},
            operation_fn=_enqueue_grad
        )

    def adam_update(self, global_sizes, local_sizes, grad_param, param_buffer, m1_param, m2_param, 
                    beta1_t, beta2_t, total_params):
        def _enqueue_update(queue, wait_for):
            return self.program.adam_update(
                queue, global_sizes, local_sizes, grad_param, param_buffer, m1_param, m2_param,
                beta1_t, beta2_t, self.learning_rate, self.adam_beta1, self.adam_beta2,
                self.min_temp, self.max_temp, self.epsilon, total_params, wait_for=wait_for
            )

        param_name = {buffers['weights']: 'weights', buffers['biases']: 'biases',
                      buffers['exit_weights']: 'exit_weights', buffers['exit_biases']: 'exit_biases',
                      buffers['temps']: 'temps'}[param_buffer]
        return self.manager.create_node(
            queue_type='compute',
            resource_access={'R': {f'grad_{param_name}'}, 'W': {param_name, f'm1_{param_name}', f'm2_{param_name}'}},
            operation_fn=_enqueue_update
        )

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

X_padded = pad_to_multiple(X_normalized, simd_width, axis=1)
padded_input_dim = X_padded.shape[1]
batch_multiple = lcm(simd_width, min(device_limits['max_work_group_size'], next_pow2(HIDDEN_DIM // simd_width) * simd_width))
max_padded_batch = ((BATCH_SIZE + batch_multiple - 1) // batch_multiple) * batch_multiple

# Weight Initialization
weights = he_init((INPUT_DIM, HIDDEN_DIM))
biases = np.zeros(HIDDEN_DIM, dtype=np.float32)
exit_weights = he_init((NUM_EXITS, HIDDEN_DIM, OUTPUT_CLASSES))
exit_biases = np.zeros((NUM_EXITS, OUTPUT_CLASSES), dtype=np.float32)
weights_padded = preprocess_weights(weights, simd_width)
exit_weights_padded = preprocess_weights(exit_weights, simd_width)
biases_padded = pad_1d(biases, simd_width)
exit_biases_padded = np.array([pad_1d(exit_biases[i], simd_width) for i in range(NUM_EXITS)])
exit_temperatures = np.ones(NUM_EXITS, dtype=np.float32)

# Calculate loss_elements_per_exit
wavefront_size = 64 if "AMD" in device.vendor else 32
loss_alignment = wavefront_size * FLOAT_SIZE
loss_elements_per_exit = ((max_padded_batch + loss_alignment - 1) // loss_alignment) * loss_alignment

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

# Buffer Creation
padded_hidden_dim = weights_padded.shape[0] * simd_width
padded_output_classes = ((OUTPUT_CLASSES + simd_width - 1) // simd_width) * simd_width
buffers = {
    'weights': cl.Buffer(ctx, cl.mem_flags.READ_WRITE | cl.mem_flags.COPY_HOST_PTR, hostbuf=weights_padded),
    'biases': cl.Buffer(ctx, cl.mem_flags.READ_WRITE | cl.mem_flags.COPY_HOST_PTR, hostbuf=biases_padded),
    'exit_weights': cl.Buffer(ctx, cl.mem_flags.READ_WRITE | cl.mem_flags.COPY_HOST_PTR, hostbuf=exit_weights_padded),
    'exit_biases': cl.Buffer(ctx, cl.mem_flags.READ_WRITE | cl.mem_flags.COPY_HOST_PTR, hostbuf=exit_biases_padded),
    'hidden': cl.Buffer(ctx, cl.mem_flags.READ_WRITE, max_padded_batch * padded_hidden_dim * FLOAT_SIZE),
    'exit_probs': cl.Buffer(ctx, cl.mem_flags.READ_WRITE, max_padded_batch * NUM_EXITS * padded_output_classes * FLOAT_SIZE),
    'losses': cl.Buffer(ctx, cl.mem_flags.READ_WRITE, NUM_EXITS * loss_elements_per_exit * FLOAT_SIZE),
    'temps': cl.Buffer(ctx, cl.mem_flags.READ_WRITE | cl.mem_flags.COPY_HOST_PTR, hostbuf=exit_temperatures),
    'grad_temps': cl.Buffer(ctx, cl.mem_flags.READ_WRITE, NUM_EXITS * FLOAT_SIZE),
    'm1_temps': cl.Buffer(ctx, cl.mem_flags.READ_WRITE, NUM_EXITS * FLOAT_SIZE),
    'm2_temps': cl.Buffer(ctx, cl.mem_flags.READ_WRITE, NUM_EXITS * FLOAT_SIZE)
}
for param in ['weights', 'biases', 'exit_weights', 'exit_biases']:
    buffers[f'grad_{param}'] = cl.Buffer(ctx, cl.mem_flags.READ_WRITE, buffers[param].size)
    buffers[f'm1_{param}'] = cl.Buffer(ctx, cl.mem_flags.READ_WRITE, buffers[param].size)
    buffers[f'm2_{param}'] = cl.Buffer(ctx, cl.mem_flags.READ_WRITE, buffers[param].size)
    cl.enqueue_fill_buffer(compute_queue, buffers[f'm1_{param}'], np.float32(0), 0, buffers[param].size)
    cl.enqueue_fill_buffer(compute_queue, buffers[f'm2_{param}'], np.float32(0), 0, buffers[param].size)

# TransferWrapper Setup
input_buf_size = max_padded_batch * padded_input_dim * FLOAT_SIZE
targets_buf_size = max_padded_batch * INT_SIZE
mask_buf_size = max_padded_batch * FLOAT_SIZE
losses_buf_size = NUM_EXITS * loss_elements_per_exit * FLOAT_SIZE
exit_probs_buf_size = max_padded_batch * NUM_EXITS * padded_output_classes * FLOAT_SIZE
temps_buf_size = NUM_EXITS * FLOAT_SIZE

input_wrapper = TransferWrapper(ctx, "input", input_buf_size, dtype=np.float32)
targets_wrapper = TransferWrapper(ctx, "targets", targets_buf_size, dtype=np.int32)
mask_wrapper = TransferWrapper(ctx, "mask", mask_buf_size, dtype=np.float32)
losses_wrapper = TransferWrapper(ctx, "losses", losses_buf_size, dtype=np.float32, device_buf=buffers['losses'])
exit_probs_wrapper = TransferWrapper(ctx, "exit_probs", exit_probs_buf_size, dtype=np.float32, device_buf=buffers['exit_probs'])
temps_wrapper = TransferWrapper(ctx, "temps", temps_buf_size, dtype=np.float32, device_buf=buffers['temps'])

# Initialize WorkManager once before training loop
manager = WorkManager(ctx)
manager.hardware_queues = {'xfer': transfer_queue, 'compute': compute_queue}

# Initialize logical resources once with version 0
manager.logical_resources = {
    "input": (input_wrapper.device_buffer, 0),
    "targets": (targets_wrapper.device_buffer, 0),
    "mask": (mask_wrapper.device_buffer, 0),
    "losses": (losses_wrapper.device_buffer, 0),
    "exit_probs": (exit_probs_wrapper.device_buffer, 0),
    "temps": (temps_wrapper.device_buffer, 0),
    "weights": (buffers['weights'], 0),
    "biases": (buffers['biases'], 0),
    "exit_weights": (buffers['exit_weights'], 0),
    "exit_biases": (buffers['exit_biases'], 0),
    "hidden": (buffers['hidden'], 0),
    "grad_weights": (buffers['grad_weights'], 0),
    "grad_biases": (buffers['grad_biases'], 0),
    "grad_exit_weights": (buffers['grad_exit_weights'], 0),
    "grad_exit_biases": (buffers['grad_exit_biases'], 0),
    "grad_temps": (buffers['grad_temps'], 0),
    "m1_weights": (buffers['m1_weights'], 0),
    "m1_biases": (buffers['m1_biases'], 0),
    "m1_exit_weights": (buffers['m1_exit_weights'], 0),
    "m1_exit_biases": (buffers['m1_exit_biases'], 0),
    "m1_temps": (buffers['m1_temps'], 0),
    "m2_weights": (buffers['m2_weights'], 0),
    "m2_biases": (buffers['m2_biases'], 0),
    "m2_exit_weights": (buffers['m2_exit_weights'], 0),
    "m2_exit_biases": (buffers['m2_exit_biases'], 0),
    "m2_temps": (buffers['m2_temps'], 0)
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
        X_batch = X_padded[batch_indices]
        y_batch = y_true[batch_indices]
        X_pad, y_pad, mask_pad, padded_batch_size = pad_batch(X_batch, y_batch, batch_multiple, simd_width)

        # Host-to-device transfers
        input_wrapper.host_to_device(manager, X_pad, transfer_queue)
        targets_wrapper.host_to_device(manager, y_pad, transfer_queue)
        mask_wrapper.host_to_device(manager, mask_pad, transfer_queue)

        # Zero gradients
        for param in ['weights', 'biases', 'exit_weights', 'exit_biases', 'temps']:
            grad_buffer = buffers[f'grad_{param}']
            def enqueue_zero(queue, wait_for):
                return cl.enqueue_fill_buffer(queue, grad_buffer, np.float32(0), 0, grad_buffer.size, wait_for=wait_for)
            manager.create_node(
                queue_type='compute',
                resource_access={'W': {f'grad_{param}'}},
                operation_fn=enqueue_zero
            )

        # Kernel wrapper
        kernel_wrapper = KernelWrapper(
            program, manager, compute_queue, padded_input_dim, padded_hidden_dim,
            padded_output_classes, padded_batch_size
        ).set_mask(mask_wrapper.device_buffer).set_temps(buffers['temps'])

        # Forward pass
        global_forward = (padded_batch_size, padded_hidden_dim // simd_width)
        local_forward = optimal_local_sizes(global_forward, device_limits)
        kernel_wrapper.forward_pass(
            global_forward, local_forward, input_wrapper.device_buffer, buffers['weights'],
            buffers['biases'], buffers['hidden']
        )

        # Early exits
        for exit_idx in range(NUM_EXITS):
            global_exit = (padded_batch_size,)
            local_exit = optimal_local_sizes(global_exit, device_limits)
            kernel_wrapper.compute_exit_probabilities(
                global_exit, local_exit, buffers['hidden'], buffers['exit_weights'], buffers['exit_biases'],
                buffers['exit_probs'], buffers['losses'], targets_wrapper.device_buffer, np.int32(exit_idx)
            )

        # Backpropagation
        global_grad = (padded_input_dim // simd_width, HIDDEN_DIM)
        local_grad = optimal_local_sizes(global_grad, device_limits)
        kernel_wrapper.compute_gradients(
            global_grad, local_grad, input_wrapper.device_buffer, buffers['hidden'], buffers['exit_probs'],
            buffers['exit_weights'], buffers['grad_weights'], buffers['grad_biases'], targets_wrapper.device_buffer
        )

        # Adam update
        beta1_t = np.float32(1 / (1 - ADAM_BETA1 ** global_step))
        beta2_t = np.float32(1 / (1 - ADAM_BETA2 ** global_step))
        for param in ['weights', 'biases', 'exit_weights', 'exit_biases', 'temps']:
            grad_param = buffers[f'grad_{param}']
            param_buffer = buffers[param]
            m1_param = buffers[f'm1_{param}']
            m2_param = buffers[f'm2_{param}']
            total_params = param_buffer.size // FLOAT_SIZE
            global_adam = (total_params,)
            local_adam = optimal_local_sizes(global_adam, device_limits)
            kernel_wrapper.adam_update(
                global_adam, local_adam, grad_param, param_buffer, m1_param, m2_param, beta1_t, beta2_t, np.int32(total_params)
            )

        # Device-to-host transfers
        host_processing_event = cl.UserEvent(ctx)
        transfer_data = {}
        losses_node, losses_data = losses_wrapper.device_to_host(manager, transfer_queue, host_processing_event)
        transfer_data.update(losses_data)
        exit_probs_node, exit_probs_data = exit_probs_wrapper.device_to_host(manager, transfer_queue, host_processing_event)
        transfer_data.update(exit_probs_data)
        temps_node, temps_data = temps_wrapper.device_to_host(manager, transfer_queue, host_processing_event)
        transfer_data.update(temps_data)

        # Execute workload
        schedule = manager.commit_workload()

        # Wait for transfers
        data_ready_events = [transfer_data[key][1] for key in transfer_data]
        cl.wait_for_events(data_ready_events)

        # Process results
        losses_host = transfer_data['losses'][0]
        exit_probs_host = transfer_data['exit_probs'][0]
        temps_host = transfer_data['temps'][0]

        exit_losses = []
        for exit_idx in range(NUM_EXITS):
            start = exit_idx * loss_elements_per_exit
            end = start + actual_batch_size
            exit_loss = losses_host[start:end].mean()
            exit_losses.append(float(exit_loss))
        print(f"Batch {batch_idx}: Per-exit Losses: {exit_losses}")

        valid_probs = exit_probs_host[:actual_batch_size].reshape(actual_batch_size, NUM_EXITS, padded_output_classes)[:, :, :OUTPUT_CLASSES]
        valid_mask = mask_pad[:actual_batch_size].astype(bool)
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

        # Signal host processing complete
        host_processing_event.set_status(cl.command_execution_status.COMPLETE)
        global_step += 1

    # Epoch metrics
    avg_loss = epoch_loss / len(X)
    train_acc = correct_predictions / len(X)
    print(f"Epoch {epoch:3d} | Loss: {avg_loss:.4f} | Acc: {train_acc:.1%}")
    print(f"Temperatures: {temps_host}")

# Cleanup
input_wrapper.release()
targets_wrapper.release()
mask_wrapper.release()
losses_wrapper.release()
exit_probs_wrapper.release()
temps_wrapper.release()

if __name__ == "__main__":
    pass
