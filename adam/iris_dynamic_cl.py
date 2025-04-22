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
MIN_TEMP = 1e-3
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

def optimal_local_size(global_size, device_limits):
    max_wg = device_limits['max_work_group_size']
    max_dims = device_limits['max_work_item_sizes']
    if isinstance(global_size, int) or len(global_size) == 1:
        global_size = global_size[0] if isinstance(global_size, tuple) else global_size
        valid = [s for s in range(1, min(max_wg, max_dims[0])+1) if global_size % s == 0]
        return (max(valid, key=lambda x: x, default=1),)
    else:
        gx, gy = global_size
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

# KernelDAGBuilder Class
class KernelDAGBuilder:
    def __init__(self, queue, device_limits):
        self.dependency_graph = nx.DiGraph()
        self.buffer_states = {}  # {buffer: (version, last_writer_node)}
        self.node_events = {}
        self.queue = queue
        self.device_limits = device_limits

    def add_operation(self, enqueue_fn, buffers_accessed):
        node_id = len(self.dependency_graph.nodes)
        self.dependency_graph.add_node(node_id, enqueue_fn=enqueue_fn, buffers=buffers_accessed.copy())

        for buffer, mode in buffers_accessed.items():
            if mode == 'W':
                readers = [n for n, data in self.dependency_graph.nodes(data=True) 
                           if buffer in data.get('buffers', {}) and data['buffers'][buffer] == 'R']
                for reader in readers:
                    self.dependency_graph.add_edge(reader, node_id)
                last_writer = self.buffer_states.get(buffer, (0, None))[1]
                if last_writer is not None:
                    self.dependency_graph.add_edge(last_writer, node_id)
                self.buffer_states[buffer] = (self.buffer_states.get(buffer, (0, None))[0] + 1, node_id)
            elif mode == 'R':
                last_writer = self.buffer_states.get(buffer, (0, None))[1]
                if last_writer is not None:
                    self.dependency_graph.add_edge(last_writer, node_id)

        return node_id

    def enqueue_with_dependencies(self, node_id):
        parents = list(self.dependency_graph.predecessors(node_id))
        wait_for = [self.node_events[p] for p in parents if p in self.node_events]

        enqueue_fn = self.dependency_graph.nodes[node_id]['enqueue_fn']
        event = enqueue_fn(wait_for=wait_for)
        self.node_events[node_id] = event
        return event

class KernelWrapper:
    def __init__(self, program, dag_builder):
        self.program = program
        self.dag = dag_builder
        self.mask_buf = None
        self.temps_buf = None

    def set_mask(self, mask):
        self.mask_buf = mask
        return self

    def set_temps(self, temps):
        self.temps_buf = temps
        return self

    def forward_pass(self, global_size, local_size, input_dev, weights, biases, hidden, *args):
        buffer_modes = {
            input_dev: 'R',
            weights: 'R',
            biases: 'R',
            hidden: 'W'
        }
        enqueue_fn = lambda wait_for: self.program.forward_pass(
            self.dag.queue, global_size, local_size,
            input_dev, weights, biases, hidden, *args, self.mask_buf, self.temps_buf, wait_for=wait_for
        )
        node_id = self.dag.add_operation(enqueue_fn, buffer_modes)
        return self.dag.enqueue_with_dependencies(node_id)

    def compute_exit_probabilities(self, global_size, local_size, hidden, exit_weights, exit_biases, exit_probs, losses, targets_dev, *args):
        buffer_modes = {
            hidden: 'R',
            exit_weights: 'R',
            exit_biases: 'R',
            exit_probs: 'W',
            losses: 'W',
            targets_dev: 'R'
        }
        enqueue_fn = lambda wait_for: self.program.compute_exit_probabilities(
            self.dag.queue, global_size, local_size,
            hidden, exit_weights, exit_biases, exit_probs, losses, targets_dev, *args, self.mask_buf, self.temps_buf, wait_for=wait_for
        )
        node_id = self.dag.add_operation(enqueue_fn, buffer_modes)
        return self.dag.enqueue_with_dependencies(node_id)

    def compute_gradients(self, global_size, local_size, input_dev, hidden, exit_probs, exit_weights, grad_weights, grad_biases, targets_dev, *args):
        buffer_modes = {
            input_dev: 'R',
            hidden: 'R',
            exit_probs: 'R',
            exit_weights: 'R',
            grad_weights: 'W',
            grad_biases: 'W',
            targets_dev: 'R'
        }
        enqueue_fn = lambda wait_for: self.program.compute_gradients(
            self.dag.queue, global_size, local_size,
            input_dev, hidden, exit_probs, exit_weights, grad_weights, grad_biases, targets_dev, *args, self.mask_buf, self.temps_buf, wait_for=wait_for
        )
        node_id = self.dag.add_operation(enqueue_fn, buffer_modes)
        return self.dag.enqueue_with_dependencies(node_id)

    def adam_update(self, global_size, local_size, grad_param, param, m1_param, m2_param, *args):
        buffer_modes = {
            grad_param: 'R',
            param: 'W',
            m1_param: 'W',
            m2_param: 'W'
        }
        enqueue_fn = lambda wait_for: self.program.adam_update(
            self.dag.queue, global_size, local_size,
            grad_param, param, m1_param, m2_param, *args, wait_for=wait_for
        )
        node_id = self.dag.add_operation(enqueue_fn, buffer_modes)
        return self.dag.enqueue_with_dependencies(node_id)

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

cache_line_size = device.get_info(cl.device_info.GLOBAL_MEM_CACHELINE_SIZE)
loss_alignment = cache_line_size // FLOAT_SIZE

simd_width = select_simd_width(device)
print(f"Training on {device.name} with SIMD-{simd_width}")

X_padded = pad_to_multiple(X_normalized, simd_width, axis=1)
input_dim_padded = X_padded.shape[1]

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

# Temperature Initialization
exit_temperatures = np.ones(NUM_EXITS, dtype=np.float32)

# Calculate loss_elements_per_exit
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
hidden_dim_padded = weights_padded.shape[0] * simd_width
output_classes_padded = ((OUTPUT_CLASSES + simd_width - 1) // simd_width) * simd_width

buffers = {
    'weights': cl.Buffer(ctx, cl.mem_flags.READ_WRITE | cl.mem_flags.COPY_HOST_PTR, hostbuf=weights_padded),
    'biases': cl.Buffer(ctx, cl.mem_flags.READ_WRITE | cl.mem_flags.COPY_HOST_PTR, hostbuf=biases_padded),
    'exit_weights': cl.Buffer(ctx, cl.mem_flags.READ_WRITE | cl.mem_flags.COPY_HOST_PTR, hostbuf=exit_weights_padded),
    'exit_biases': cl.Buffer(ctx, cl.mem_flags.READ_WRITE | cl.mem_flags.COPY_HOST_PTR, hostbuf=exit_biases_padded),
    'hidden': cl.Buffer(ctx, cl.mem_flags.READ_WRITE, max_padded_batch * hidden_dim_padded * FLOAT_SIZE),
    'exit_probs': cl.Buffer(ctx, cl.mem_flags.READ_WRITE, max_padded_batch * NUM_EXITS * output_classes_padded * FLOAT_SIZE),
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

# Double Buffering Setup
input_buf_size = max_padded_batch * input_dim_padded * FLOAT_SIZE
targets_buf_size = max_padded_batch * INT_SIZE
mask_buf_size = max_padded_batch * FLOAT_SIZE

staging_input_A = cl.Buffer(ctx, cl.mem_flags.READ_WRITE | cl.mem_flags.ALLOC_HOST_PTR, size=input_buf_size)
staging_targets_A = cl.Buffer(ctx, cl.mem_flags.READ_WRITE | cl.mem_flags.ALLOC_HOST_PTR, size=targets_buf_size)
staging_mask_A = cl.Buffer(ctx, cl.mem_flags.READ_WRITE | cl.mem_flags.ALLOC_HOST_PTR, size=mask_buf_size)
input_dev_A = cl.Buffer(ctx, cl.mem_flags.READ_WRITE, size=input_buf_size)
targets_dev_A = cl.Buffer(ctx, cl.mem_flags.READ_WRITE, size=targets_buf_size)
mask_dev_A = cl.Buffer(ctx, cl.mem_flags.READ_WRITE, size=mask_buf_size)

staging_input_B = cl.Buffer(ctx, cl.mem_flags.READ_WRITE | cl.mem_flags.ALLOC_HOST_PTR, size=input_buf_size)
staging_targets_B = cl.Buffer(ctx, cl.mem_flags.READ_WRITE | cl.mem_flags.ALLOC_HOST_PTR, size=targets_buf_size)
staging_mask_B = cl.Buffer(ctx, cl.mem_flags.READ_WRITE | cl.mem_flags.ALLOC_HOST_PTR, size=mask_buf_size)
input_dev_B = cl.Buffer(ctx, cl.mem_flags.READ_WRITE, size=input_buf_size)
targets_dev_B = cl.Buffer(ctx, cl.mem_flags.READ_WRITE, size=targets_buf_size)
mask_dev_B = cl.Buffer(ctx, cl.mem_flags.READ_WRITE, size=mask_buf_size)

sets = [
    {'staging_input': staging_input_A, 'staging_targets': staging_targets_A, 'staging_mask': staging_mask_A,
     'input_dev': input_dev_A, 'targets_dev': targets_dev_A, 'mask_dev': mask_dev_A},
    {'staging_input': staging_input_B, 'staging_targets': staging_targets_B, 'staging_mask': staging_mask_B,
     'input_dev': input_dev_B, 'targets_dev': targets_dev_B, 'mask_dev': mask_dev_B}
]

# Training Loop
dag_builder = KernelDAGBuilder(compute_queue, device_limits)
global_step = 1
for epoch in range(EPOCHS):
    shuffled_indices = np.random.permutation(len(X))
    num_batches = (len(X) + BATCH_SIZE - 1) // BATCH_SIZE
    epoch_loss = 0.0
    correct_predictions = 0

    for batch_idx in range(num_batches):
        current_set = sets[batch_idx % 2]
        next_set = sets[(batch_idx + 1) % 2] if batch_idx < num_batches - 1 else None

        # Prepare batch
        batch_start = batch_idx * BATCH_SIZE
        batch_end = min(batch_start + BATCH_SIZE, len(X))
        actual_batch_size = batch_end - batch_start
        batch_indices = shuffled_indices[batch_start:batch_end]
        X_batch = X_padded[batch_indices]
        y_batch = y_true[batch_indices]
        X_pad, y_pad, mask_pad, padded_batch_size = pad_batch(X_batch, y_batch, batch_multiple, simd_width)

        assert loss_elements_per_exit >= padded_batch_size, \
        f"Loss stride {loss_elements_per_exit} < padded batch {padded_batch_size}"

        # Map staging buffers
        host_input, map_event_input = cl.enqueue_map_buffer(
            transfer_queue,
            current_set['staging_input'],
            cl.map_flags.WRITE | cl.map_flags.INVALIDATE_REGION,
            0,
            (padded_batch_size, input_dim_padded),
            np.float32,
            is_blocking=False
        )
        host_targets, map_event_targets = cl.enqueue_map_buffer(
            transfer_queue,
            current_set['staging_targets'],
            cl.map_flags.WRITE | cl.map_flags.INVALIDATE_REGION,
            0,
            (padded_batch_size,),
            np.int32,
            is_blocking=False
        )
        host_mask, map_event_mask = cl.enqueue_map_buffer(
            transfer_queue,
            current_set['staging_mask'],
            cl.map_flags.WRITE | cl.map_flags.INVALIDATE_REGION,
            0,
            (padded_batch_size,),
            np.float32,
            is_blocking=False
        )
        cl.wait_for_events([map_event_input, map_event_targets, map_event_mask])

        # Write to buffers
        host_input[:padded_batch_size, :input_dim_padded] = X_pad
        host_targets[:padded_batch_size] = y_pad
        host_mask[:padded_batch_size] = mask_pad

        # Unmap
        unmap_event_input = cl.enqueue_unmap_mem_object(transfer_queue, current_set['staging_input'], host_input)
        unmap_event_targets = cl.enqueue_unmap_mem_object(transfer_queue, current_set['staging_targets'], host_targets)
        unmap_event_mask = cl.enqueue_unmap_mem_object(transfer_queue, current_set['staging_mask'], host_mask)

        # Transfer to device with DAG
        enqueue_fn_input = lambda wait_for: cl.enqueue_copy_buffer(
            transfer_queue, current_set['staging_input'], current_set['input_dev'],
            0, 0, input_buf_size, wait_for=wait_for
        )
        transfer_node_input = dag_builder.add_operation(enqueue_fn_input, {current_set['input_dev']: 'W'})
        transfer_event_input = dag_builder.enqueue_with_dependencies(transfer_node_input)

        enqueue_fn_targets = lambda wait_for: cl.enqueue_copy_buffer(
            transfer_queue, current_set['staging_targets'], current_set['targets_dev'],
            0, 0, targets_buf_size, wait_for=wait_for
        )
        transfer_node_targets = dag_builder.add_operation(enqueue_fn_targets, {current_set['targets_dev']: 'W'})
        transfer_event_targets = dag_builder.enqueue_with_dependencies(transfer_node_targets)

        enqueue_fn_mask = lambda wait_for: cl.enqueue_copy_buffer(
            transfer_queue, current_set['staging_mask'], current_set['mask_dev'],
            0, 0, mask_buf_size, wait_for=wait_for
        )
        transfer_node_mask = dag_builder.add_operation(enqueue_fn_mask, {current_set['mask_dev']: 'W'})
        transfer_event_mask = dag_builder.enqueue_with_dependencies(transfer_node_mask)

        # Zero gradients
        for param in ['weights', 'biases', 'exit_weights', 'exit_biases', 'temps']:
            grad_buffer = buffers[f'grad_{param}']
            enqueue_fn_zero = (lambda gb: lambda wait_for: cl.enqueue_fill_buffer(
                compute_queue, gb, np.float32(0), 0, gb.size, wait_for=wait_for
            ))(grad_buffer)  # <-- Capture grad_buffer now
            zero_node = dag_builder.add_operation(enqueue_fn_zero, {grad_buffer: 'W'})
            zero_event = dag_builder.enqueue_with_dependencies(zero_node)

        # Kernel execution with DAG
        kernel_wrapper = KernelWrapper(program, dag_builder).set_mask(current_set['mask_dev']).set_temps(buffers['temps'])

        # Forward pass
        global_forward = (padded_batch_size, hidden_dim_padded // simd_width)
        local_forward = optimal_local_size(global_forward, device_limits)
        forward_event = kernel_wrapper.forward_pass(
            global_forward,
            local_forward,
            current_set['input_dev'],
            buffers['weights'],
            buffers['biases'],
            buffers['hidden'],
            np.int32(padded_batch_size),
            np.int32(input_dim_padded),
            np.int32(HIDDEN_DIM),
            np.int32(hidden_dim_padded)
        )

        # Early exits
        exit_events = []
        for exit_idx in range(NUM_EXITS):
            global_exit = (padded_batch_size,)
            local_exit = optimal_local_size(global_exit, device_limits)
            exit_event = kernel_wrapper.compute_exit_probabilities(
                global_exit,
                local_exit,
                buffers['hidden'],
                buffers['exit_weights'],
                buffers['exit_biases'],
                buffers['exit_probs'],
                buffers['losses'],
                current_set['targets_dev'],
                np.int32(padded_batch_size),
                np.int32(HIDDEN_DIM),
                np.int32(hidden_dim_padded),
                np.int32(OUTPUT_CLASSES),
                np.int32(output_classes_padded),
                np.int32(NUM_EXITS),
                np.int32(exit_idx)
            )
            exit_events.append(exit_event)

        losses_host = np.empty(NUM_EXITS * loss_elements_per_exit, dtype=np.float32)
        cl.enqueue_copy(compute_queue, losses_host, buffers['losses'], wait_for=batch_events)

        # Process per-exit losses for logging/monitoring
        exit_losses = []
        for exit_idx in range(NUM_EXITS):
            start = exit_idx * loss_elements_per_exit
            end = start + actual_batch_size
            exit_loss = losses_host[start:end].mean()
            exit_losses.append(exit_loss)
        print(f"Per-exit Losses: {batch_idx}:{exit_losses}")

        # Backpropagation
        global_grad = (input_dim_padded // simd_width, HIDDEN_DIM)
        local_grad = optimal_local_size(global_grad, device_limits)
        grad_event = kernel_wrapper.compute_gradients(
            global_grad,
            local_grad,
            current_set['input_dev'],
            buffers['hidden'],
            buffers['exit_probs'],
            buffers['exit_weights'],
            buffers['grad_weights'],
            buffers['grad_biases'],
            current_set['targets_dev'],
            np.int32(padded_batch_size),
            np.int32(input_dim_padded),
            np.int32(HIDDEN_DIM),
            np.int32(hidden_dim_padded),
            np.int32(OUTPUT_CLASSES),
            np.int32(output_classes_padded),
            np.int32(NUM_EXITS)
        )

        # Adam update
        beta1_t = 1 / (1 - ADAM_BETA1 ** global_step)
        beta2_t = 1 / (1 - ADAM_BETA2 ** global_step)
        update_events = []
        for param in ['weights', 'biases', 'exit_weights', 'exit_biases', 'temps']:
            grad_param = buffers[f'grad_{param}']
            param_buffer = buffers[param]
            m1_param = buffers[f'm1_{param}']
            m2_param = buffers[f'm2_{param}']
            total_params = param_buffer.size // FLOAT_SIZE
            global_adam = (total_params,)
            local_adam = optimal_local_size(global_adam, device_limits)
            update_event = kernel_wrapper.adam_update(
                global_adam,
                local_adam,
                grad_param,
                param_buffer,
                m1_param,
                m2_param,
                np.float32(LEARNING_RATE),
                np.float32(ADAM_BETA1),
                np.float32(ADAM_BETA2),
                np.float32(beta1_t),
                np.float32(beta2_t),
                np.float32(MIN_TEMP),
                np.float32(MAX_TEMP),
                np.float32(EPSILON),
                np.int32(total_params)
            )
            update_events.append(update_event)

        global_step += 1

        # Collect all events for this batch
        batch_events = [transfer_event_input, transfer_event_targets, transfer_event_mask, forward_event] + exit_events + [grad_event] + update_events

        # Read losses and exit probabilities
        exit_probs_host = np.empty((padded_batch_size, NUM_EXITS, output_classes_padded), dtype=np.float32)
        cl.enqueue_copy(compute_queue, exit_probs_host, buffers['exit_probs'], wait_for=batch_events)

        # Read temperatures
        temps_host = np.empty(NUM_EXITS, dtype=np.float32)
        cl.enqueue_copy(compute_queue, temps_host, buffers['temps'], wait_for=batch_events)
        if not (np.all(temps_host >= MIN_TEMP) and np.all(temps_host <= MAX_TEMP)):
            raise RuntimeError(f"Temp violation: {temps_host}")

        # Extract valid probabilities
        valid_probs = exit_probs_host[:actual_batch_size].reshape(actual_batch_size, NUM_EXITS, OUTPUT_CLASSES)

        # Compute temperature-weighted ensemble
        valid_mask = mask_pad[:actual_batch_size].astype(bool)
        confidences = np.array([
            valid_probs[valid_mask, i, :].max(axis=1) ** (1 / (temps_host[i] + 1e-8))
            for i in range(NUM_EXITS)
        ])
        weights = np.exp(confidences) / np.sum(np.exp(confidences), axis=0)
        ensemble_probs = np.einsum('ijk,j->ik', valid_probs, weights)
        ensemble_probs /= np.sum(ensemble_probs, axis=1, keepdims=True) + 1e-8

        # Compute loss and accuracy
        log_probs = np.log(ensemble_probs + 1e-8)
        batch_loss = -np.mean(log_probs[np.arange(actual_batch_size), y_batch])
        epoch_loss += batch_loss * actual_batch_size
        predicted_classes = np.argmax(ensemble_probs, axis=1)
        batch_correct = np.sum(predicted_classes == y_batch)
        correct_predictions += batch_correct

        # Prefetch next batch if needed
        if next_set:
            next_batch_start = (batch_idx + 1) * BATCH_SIZE
            next_batch_end = min(next_batch_start + BATCH_SIZE, len(X))
            next_actual_size = next_batch_end - next_batch_start
            next_indices = shuffled_indices[next_batch_start:next_batch_end]
            X_next = X_padded[next_indices]
            y_next = y_true[next_indices]
            X_next_pad, y_next_pad, mask_next_pad, next_padded_size = pad_batch(X_next, y_next, batch_multiple, simd_width)

            # Map next staging buffers
            next_input, next_map_input = cl.enqueue_map_buffer(
                transfer_queue, next_set['staging_input'], cl.map_flags.WRITE | cl.map_flags.INVALIDATE_REGION, 0,
                (next_padded_size, input_dim_padded), np.float32, is_blocking=False
            )
            next_targets, next_map_targets = cl.enqueue_map_buffer(
                transfer_queue, next_set['staging_targets'], cl.map_flags.WRITE | cl.map_flags.INVALIDATE_REGION, 0,
                (next_padded_size,), np.int32, is_blocking=False
            )
            next_mask, next_map_mask = cl.enqueue_map_buffer(
                transfer_queue, next_set['staging_mask'], cl.map_flags.WRITE | cl.map_flags.INVALIDATE_REGION, 0,
                (next_padded_size,), np.float32, is_blocking=False
            )

            with next_input:
                next_input[:next_padded_size, :input_dim_padded] = X_next_pad
            with next_targets:
                next_targets[:next_padded_size] = y_next_pad
            with next_mask:
                next_mask[:next_padded_size] = mask_next_pad

            # Transfer next batch to device
            cl.enqueue_copy_buffer(
                transfer_queue, next_set['staging_input'], next_set['input_dev'],
                src_offset=0, dst_offset=0, size=next_padded_size * input_dim_padded * FLOAT_SIZE, is_blocking=False
            )
            cl.enqueue_copy_buffer(
                transfer_queue, next_set['staging_targets'], next_set['targets_dev'],
                src_offset=0, dst_offset=0, size=next_padded_size * INT_SIZE, is_blocking=False
            )
            cl.enqueue_copy_buffer(
                transfer_queue, next_set['staging_mask'], next_set['mask_dev'],
                src_offset=0, dst_offset=0, size=next_padded_size * FLOAT_SIZE, is_blocking=False
            )

    # Compute and print epoch metrics
    avg_loss = epoch_loss / len(X)
    train_acc = correct_predictions / len(X)
    print(f"Epoch {epoch:3d} | Loss: {avg_loss:.4f} | Acc: {train_acc:.1%}")
    print(f"Temperatures: {temps_host}")

if __name__ == "__main__":
    pass
