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


class KernelDAGBuilder:
    def __init__(self):
        self.dag = nx.DiGraph()
        self.node_counter = 0
        self.constant_buffers = {}
        self.buffer_versions = {}
        self.buffer_writers = {}
        self.node_data = {}
        self.buffer_last_access = {}

    class OperationNode:
        __slots__ = ['node_id', 'enqueue_fn', 'buffers', 'queue', 'event', 'callbacks']

        def __init__(self, node_id):
            self.node_id = node_id
            self.enqueue_fn = None
            self.buffers = {}
            self.queue = None
            self.event = None
            self.callbacks = []

    def add_operation(self, enqueue_fn, buffers_accessed, queue):
        """Add operation to DAG with automatic dependency resolution"""
        node = self.OperationNode(self.node_counter)
        node.enqueue_fn = enqueue_fn
        node.buffers = buffers_accessed.copy()
        node.queue = queue

        dependencies = self._calculate_dependencies(buffers_accessed)
        self.dag.add_node(node.node_id)
        self.node_data[node.node_id] = node

        for dep in dependencies:
            self.dag.add_edge(dep, node.node_id)

        self._update_buffer_state(node.node_id, buffers_accessed)
        self.node_counter += 1
        return node.node_id

    def register_constant_buffer(self, name, cl_buffer):
        """Mark buffers that persist across DAG executions"""
        self.constant_buffers[name] = cl_buffer

    def register_node_callback(self, node_id, callback_fn):
        """Associate buffer callback with specific operation node"""
        if node_id in self.node_data:
            self.node_data[node_id].callbacks.append(callback_fn)

    def _calculate_dependencies(self, buffers_accessed):
        """Find all necessary dependencies based on buffer access modes"""
        dependencies = set()
        for buffer, mode in buffers_accessed.items():
            current_version = self.buffer_versions.get(buffer, 0)

            if mode == 'W':
                # RAW: Find all readers since last write
                readers = [n for n, v in self.buffer_last_access.get(buffer, [])
                           if v == 'R' and n < self.node_counter]
                dependencies.update(readers)

                # WAW: Find last writer
                if buffer in self.buffer_writers:
                    dependencies.add(self.buffer_writers[buffer])
            elif mode == 'R':
                # WAR: Find the last writer
                if buffer in self.buffer_writers:
                    dependencies.add(self.buffer_writers[buffer])

        return dependencies

    def _update_buffer_state(self, node_id, buffers_accessed):
        """Update global buffer state tracking"""
        for buffer, mode in buffers_accessed.items():
            # Track buffer access pattern
            if buffer not in self.buffer_last_access:
                self.buffer_last_access[buffer] = []
            self.buffer_last_access[buffer].append((node_id, mode))

            if mode == 'W':
                self.buffer_writers[buffer] = node_id
                self.buffer_versions[buffer] = self.buffer_versions.get(buffer, 0) + 1

    def build_execution_order(self):
        """Validate and return topologically sorted execution order"""
        if not nx.is_directed_acyclic_graph(self.dag):
            raise ValueError("DAG contains cycles")
        return list(nx.topological_sort(self.dag))

    def execute(self):
        """Execute all operations in dependency order with proper synchronization"""
        ordered_nodes = self.build_execution_order()
        event_graph = {}

        for node_id in ordered_nodes:
            node = self.node_data[node_id]
            predecessors = list(self.dag.predecessors(node_id))
            wait_events = [event_graph[p] for p in predecessors if p in event_graph]

            # Execute operation with OpenCL event chaining
            node.event = node.enqueue_fn(
                queue=node.queue,
                wait_for=wait_events
            )

            # Add callback callbacks
            if node.callbacks:
                # Need to closure-capture individual callbacks
                def make_callback(fn):
                    def callback(event, status):
                        if status == cl.command_execution_status.COMPLETE:
                            fn()
                    return callback

                for callback_fn in node.callbacks:
                    cb = make_callback(callback_fn)
                    node.event.set_callback(cl.command_execution_status.COMPLETE, cb)

            event_graph[node_id] = node.event

        return event_graph

    def reset(self):
        """Reset DAG state for new computation graph"""
        self.dag.clear()
        self.node_counter = 0
        self.buffer_versions.clear()
        self.buffer_writers.clear()
        self.node_data.clear()
        self.buffer_last_access.clear()

class TransferWrapper:
    def __init__(self, context, buffer_name, buffer_size, dtype=np.float32):
        """
        Initialize the TransferWrapper for managing host-device data transfers.
        
        Args:
            context: OpenCL context
            buffer_name: Logical name for the buffer
            buffer_size: Size of the buffer in bytes
            dtype: NumPy data type for the buffer elements
        """
        self.ctx = context
        self.logical_name = buffer_name
        self.dtype = dtype
        self.element_size = np.dtype(dtype).itemsize
        self.buffer_size = buffer_size

        # Create staging buffers for double buffering
        self.staging_bufs = [
            self._create_staging_buffer(buffer_size),
            self._create_staging_buffer(buffer_size)
        ]
        self.current_staging = 0

        # Device buffer storage
        self.device_buf = cl.Buffer(
            self.ctx,
            cl.mem_flags.READ_WRITE,
            size=buffer_size
        )

        # Track transfer operations
        self.last_transfer_node = None

    def _create_staging_buffer(self, size):
        """Create host-mapped staging buffer"""
        return cl.Buffer(
            self.ctx,
            cl.mem_flags.READ_WRITE | cl.mem_flags.ALLOC_HOST_PTR,
            size=size
        )

    def host_to_device(self, dag, host_data, transfer_queue):
        """
        Enqueue host-to-device transfer with DAG integration.
        
        Args:
            dag: DAG builder for scheduling operations
            host_data: NumPy array of data to transfer to device
            transfer_queue: OpenCL queue for transfer operations
        
        Returns:
            Tuple of (transfer_node, transfer_complete)
            - transfer_node: Final DAG node for the transfer
            - transfer_complete: User event signaling transfer completion
        """
        if host_data.nbytes > self.staging_bufs[0].size:
            raise ValueError(f"Data size {host_data.nbytes} exceeds buffer capacity {self.staging_bufs[0].size}")

        # Select staging buffer using double buffering
        staging_buf = self.staging_bufs[self.current_staging]
        self.current_staging = 1 - self.current_staging  # Toggle buffer index

        transfer_complete = cl.UserEvent(self.ctx)

        # 1. Map and copy to staging
        def map_and_copy(queue, wait_for):
            host_ptr, map_event = cl.enqueue_map_buffer(
                queue, staging_buf, cl.map_flags.WRITE,
                0, host_data.shape, self.dtype,
                wait_for=wait_for
            )
            np.copyto(host_ptr, host_data)
            return map_event

        map_node = dag.add_operation(
            enqueue_fn=map_and_copy,
            buffers_accessed={self.logical_name: 'W'},
            queue=transfer_queue
        )

        # 2. Unmap staging buffer
        def unmap(queue, wait_for):
            unmap_event = cl.enqueue_unmap_mem_object(
                queue, staging_buf, staging_buf.get_host_array(),
                wait_for=wait_for
            )
            return unmap_event

        unmap_node = dag.add_operation(
            enqueue_fn=unmap,
            buffers_accessed={self.logical_name: 'W'},
            queue=transfer_queue
        )
        dag.dag.add_edge(map_node, unmap_node)

        # 3. Copy to device buffer with callback
        def transfer_to_device(queue, wait_for):
            copy_event = cl.enqueue_copy_buffer(
                queue, staging_buf, self.device_buf,
                wait_for=wait_for
            )
            # Set transfer_complete when copy is done
            copy_event.set_callback(
                cl.command_execution_status.COMPLETE,
                lambda event, status: transfer_complete.set_status(status)
            )
            return copy_event

        transfer_node = dag.add_operation(
            enqueue_fn=transfer_to_device,
            buffers_accessed={self.logical_name: 'W'},
            queue=transfer_queue
        )
        dag.dag.add_edge(unmap_node, transfer_node)

        self.last_transfer_node = transfer_node
        return transfer_node, transfer_complete

    def device_to_host(self, dag, transfer_queue, compute_queue):
        """
        Enqueue device-to-host transfer with DAG integration.
        
        Args:
            dag: DAG builder for scheduling operations
            transfer_queue: OpenCL queue for transfer operations
            compute_queue: OpenCL queue for compute operations
        
        Returns:
            Tuple (retrieve_node, staging_buf, data_ready)
            - retrieve_node: Final DAG node
            - staging_buf: Buffer containing the transferred data
            - data_ready: User event signaling when data is ready on the host
        """
        # Select the current staging buffer
        staging_buf = self.staging_bufs[self.current_staging]
        data_ready = cl.UserEvent(self.ctx)

        # Step 1: Copy from device buffer to staging buffer
        def device_to_staging(queue, wait_for):
            return cl.enqueue_copy_buffer(
                queue, self.device_buf, staging_buf,
                wait_for=wait_for
            )

        copy_node = dag.add_operation(
            enqueue_fn=device_to_staging,
            buffers_accessed={self.logical_name: 'R'},
            queue=transfer_queue
        )

        # Step 2: Map the staging buffer to host memory with a callback
        def map_staging(queue, wait_for):
            host_ptr, map_event = cl.enqueue_map_buffer(
                queue, staging_buf, cl.map_flags.READ,
                0, (self.buffer_size // self.element_size,), self.dtype,
                wait_for=wait_for
            )
            # Set data_ready when the map operation completes
            map_event.set_callback(
                cl.command_execution_status.COMPLETE,
                lambda event, status: data_ready.set_status(status)
            )
            return map_event

        map_node = dag.add_operation(
            enqueue_fn=map_staging,
            buffers_accessed={self.logical_name: 'R'},
            queue=transfer_queue
        )
        dag.dag.add_edge(copy_node, map_node)

        # Step 3: Placeholder for finalization (unmapping handled separately)
        def finalize(queue, wait_for):
            return cl.UserEvent(self.ctx)  # Placeholder event

        final_node = dag.add_operation(
            enqueue_fn=finalize,
            buffers_accessed={},
            queue=compute_queue
        )
        dag.dag.add_edge(map_node, final_node)

        return final_node, staging_buf, data_ready

    @property
    def device_buffer(self):
        """Get the device buffer reference"""
        return self.device_buf

    def release(self):
        """Release OpenCL resources"""
        for buf in self.staging_bufs:
            buf.release()
        self.device_buf.release()

class KernelWrapper:
    def __init__(self, program, dag_builder, padded_input_dim, padded_hidden_dim, padded_output_classes, padded_batch_size):
        """
        Initialize the KernelWrapper with an OpenCL program and DAG builder.
        
        Args:
            program: Compiled OpenCL program containing kernel functions.
            dag_builder: Instance of KernelDAGBuilder for operation registration.
            padded_: Dictionary or object containing padded dimensions (assumed to provide padded values).
        """
        self.program = program
        self.dag = dag_builder
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
        """Set the mask buffer and return self for chaining."""
        self.mask = mask
        return self

    def set_temps(self, temps):
        """Set the temperatures buffer and return self for chaining."""
        self.temperatures = temps
        return self

    def forward_pass(self, global_sizes, local_sizes, input_buf, weights_buf, biases_buf, hidden_buf):
        """
        Register a forward propagation operation in the DAG.
        
        Args:
            global_sizes: Tuple of global work sizes.
            local_sizes: Tuple of local work sizes.
            input_buf: Input data buffer.
            weights_buf: Weights buffer.
            biases_buf: Biases buffer.
            hidden_buf: Hidden layer output buffer.
            padded_batch_size: Number of samples in the batch (int32).
        
        Returns:
            Node ID of the registered operation in the DAG.
        """
        def _enqueue_forward(queue, wait_for):
            return self.program.forward_pass(
                queue,
                global_sizes,
                local_sizes,
                input_buf,
                weights_buf,
                biases_buf,
                hidden_buf,
                self.mask,
                self.input_dim,
                self.hidden_dim,
                self.padded_hidden_dim,
                self.padded_batch_size,
                wait_for=wait_for
            )

        buffers_accessed = {
            input_buf: 'R',
            weights_buf: 'R',
            biases_buf: 'R',
            hidden_buf: 'W',
            self.mask: 'R'
        }
        return self.dag.add_operation(
            enqueue_fn=_enqueue_forward,
            buffers_accessed=buffers_accessed,
            queue=self.dag.compute_queue
        )

    def compute_exit_probabilities(self, global_sizes, local_sizes, hidden_buf, exit_weights_buf, 
                                    exit_biases_buf, exit_probs_buf, losses_buf, targets_buf, 
                                    exit_idx):
        """
        Register an exit probability calculation operation for a specific exit index.
        
        Args:
            global_sizes: Tuple of global work sizes.
            local_sizes: Tuple of local work sizes.
            hidden_buf: Hidden layer activations buffer.
            exit_weights_buf: Exit classifier weights buffer.
            exit_biases_buf: Exit classifier biases buffer.
            exit_probs_buf: Exit probabilities output buffer.
            losses_buf: Per-exit loss output buffer.
            targets_buf: Target labels buffer.
            padded_batch_size: Number of samples in the batch (int32).
            exit_idx: Index of the current exit (int32).
        
        Returns:
            Node ID of the registered operation in the DAG.
        """
        def _enqueue_exit(queue, wait_for):
            return self.program.compute_exit_probabilities(
                queue,
                global_sizes,
                local_sizes,
                hidden_buf,
                exit_weights_buf,
                exit_biases_buf,
                exit_probs_buf,
                losses_buf,
                targets_buf,
                self.mask,
                self.temperatures,
                self.hidden_dim,
                self.padded_hidden_dim,
                self.output_classes,
                self.padded_output_classes,
                self.num_exits,
                self.padded_batch_size,
                exit_idx,
                wait_for=wait_for
            )

        buffers_accessed = {
            hidden_buf: 'R',
            exit_weights_buf: 'R',
            exit_biases_buf: 'R',
            targets_buf: 'R',
            exit_probs_buf: 'W',
            losses_buf: 'W',
            self.mask: 'R',
            self.temperatures: 'R'
        }
        return self.dag.add_operation(
            enqueue_fn=_enqueue_exit,
            buffers_accessed=buffers_accessed,
            queue=self.dag.compute_queue
        )

    def compute_gradients(self, global_sizes, local_sizes, input_buf, hidden_buf, exit_probs_buf, 
                          exit_weights_buf, grad_weights_buf, grad_biases_buf, targets_buf):
        """
        Register a gradient computation operation in the DAG.
        
        Args:
            global_sizes: Tuple of global work sizes.
            local_sizes: Tuple of local work sizes.
            input_buf: Input data buffer.
            hidden_buf: Hidden layer activations buffer.
            exit_probs_buf: Exit probabilities buffer.
            exit_weights_buf: Exit classifier weights buffer.
            grad_weights_buf: Gradient buffer for weights.
            grad_biases_buf: Gradient buffer for biases.
            targets_buf: Target labels buffer.
            padded_batch_size: Number of samples in the batch (int32).
        
        Returns:
            Node ID of the registered operation in the DAG.
        """
        def _enqueue_grad(queue, wait_for):
            return self.program.compute_gradients(
                queue,
                global_sizes,
                local_sizes,
                input_buf,
                hidden_buf,
                exit_probs_buf,
                exit_weights_buf,
                grad_weights_buf,
                grad_biases_buf,
                targets_buf,
                self.mask,
                self.temperatures,
                self.input_dim,
                self.hidden_dim,
                self.padded_hidden_dim,
                self.output_classes,
                self.padded_output_classes,
                self.num_exits,
                self.padded_batch_size,
                wait_for=wait_for
            )

        buffers_accessed = {
            input_buf: 'R',
            hidden_buf: 'R',
            exit_probs_buf: 'R',
            exit_weights_buf: 'R',
            targets_buf: 'R',
            grad_weights_buf: 'W',
            grad_biases_buf: 'W',
            self.mask: 'R',
            self.temperatures: 'R'
        }
        return self.dag.add_operation(
            enqueue_fn=_enqueue_grad,
            buffers_accessed=buffers_accessed,
            queue=self.dag.compute_queue
        )

    def adam_update(self, global_sizes, local_sizes, grad_param, param_buffer, m1_param, m2_param, 
                    beta1_t, beta2_t, total_params):
        """
        Register an Adam optimization update operation for a given parameter.
        
        Args:
            global_sizes: Tuple of global work sizes.
            local_sizes: Tuple of local work sizes.
            grad_param: Gradient buffer for the parameter.
            param_buffer: Parameter buffer to update.
            m1_param: First moment estimate buffer.
            m2_param: Second moment estimate buffer.
            beta1_t: Bias-corrected beta1 (float32).
            beta2_t: Bias-corrected beta2 (float32).
            total_params: Total number of parameters (int32).
        
        Returns:
            Node ID of the registered operation in the DAG.
        """
        def _enqueue_update(queue, wait_for):
            return self.program.adam_update(
                queue,
                global_sizes,
                local_sizes,
                grad_param,
                param_buffer,
                m1_param,
                m2_param,
                beta1_t,
                beta2_t,
                self.learning_rate,
                self.adam_beta1,
                self.adam_beta2,
                self.min_temp,
                self.max_temp,
                self.epsilon,
                total_params,
                wait_for=wait_for
            )

        buffers_accessed = {
            grad_param: 'R',
            param_buffer: 'W',
            m1_param: 'W',
            m2_param: 'W'
        }
        return self.dag.add_operation(
            enqueue_fn=_enqueue_update,
            buffers_accessed=buffers_accessed,
            queue=self.dag.compute_queue
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

cache_line_size = device.get_info(cl.device_info.GLOBAL_MEM_CACHELINE_SIZE)
loss_alignment = cache_line_size // FLOAT_SIZE

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


input_wrapper = TransferWrapper(ctx, "input", input_buf_size, dtype=np.float32)
targets_wrapper = TransferWrapper(ctx, "targets", targets_buf_size, dtype=np.int32)
mask_wrapper = TransferWrapper(ctx, "mask", mask_buf_size, dtype=np.float32)
losses_wrapper = TransferWrapper(ctx, "losses", losses_buf_size, dtype=np.float32)

dag_builder = KernelDAGBuilder()

# Training Loop
global_step = 1
for epoch in range(EPOCHS):
    shuffled_indices = np.random.permutation(len(X))
    num_batches = (len(X) + BATCH_SIZE - 1) // BATCH_SIZE
    epoch_loss = 0.0
    correct_predictions = 0
    buffer_set = 0  # For double buffering

    for batch_idx in range(num_batches):
        # Prepare batch
        batch_start = batch_idx * BATCH_SIZE
        batch_end = min(batch_start + BATCH_SIZE, len(X))
        actual_batch_size = batch_end - batch_start
        batch_indices = shuffled_indices[batch_start:batch_end]
        X_batch = X_padded[batch_indices]
        y_batch = y_true[batch_indices]
        X_pad, y_pad, mask_pad, padded_batch_size = pad_batch(X_batch, y_batch, batch_multiple, simd_width)

        # Set active buffer for double buffering
        input_wrapper.set_active_buffer(buffer_set)
        targets_wrapper.set_active_buffer(buffer_set)
        mask_wrapper.set_active_buffer(buffer_set)

        # Enqueue host-to-device transfers
        input_transfer_node = input_wrapper.host_to_device(dag_builder, X_pad, transfer_queue)
        targets_transfer_node = targets_wrapper.host_to_device(dag_builder, y_pad, transfer_queue)
        mask_transfer_node = mask_wrapper.host_to_device(dag_builder, mask_pad, transfer_queue)

        # Zero gradients
        for param in ['weights', 'biases', 'exit_weights', 'exit_biases', 'temps']:
            grad_buffer = buffers[f'grad_{param}']
            enqueue_fn_zero = lambda wait_for: cl.enqueue_fill_buffer(
                compute_queue, grad_buffer, np.float32(0), 0, grad_buffer.size, wait_for=wait_for
            )
            zero_node = dag_builder.add_operation(enqueue_fn_zero, {grad_buffer: 'W'})
            dag_builder.enqueue_with_dependencies(zero_node)

        # Kernel execution with DAG
        kernel_wrapper = KernelWrapper(
            program, dag_builder,
            padded_input_dim, padded_hidden_dim,
            padded_output_classes, padded_batch_size
        ).set_mask(mask_wrapper.device_buffer).set_temps(buffers['temps'])

        # Forward pass
        global_forward = (padded_batch_size, padded_hidden_dim // simd_width)
        local_forward = optimal_local_sizes(global_forward, device_limits)
        forward_event = kernel_wrapper.forward_pass(
            global_forward,
            local_forward,
            input_wrapper.device_buffer,
            buffers['weights'],
            buffers['biases'],
            buffers['hidden'],
        )

        # Early exits
        exit_events = []
        for exit_idx in range(NUM_EXITS):
            global_exit = (padded_batch_size,)
            local_exit = optimal_local_sizes(global_exit, device_limits)
            exit_event = kernel_wrapper.compute_exit_probabilities(
                global_exit,
                local_exit,
                buffers['hidden'],
                buffers['exit_weights'],
                buffers['exit_biases'],
                buffers['exit_probs'],
                losses_wrapper.device_buffer,
                targets_wrapper.device_buffer,
                np.int32(exit_idx),
            )
            exit_events.append(exit_event)

        # Retrieve losses from device to host
        retrieve_node, losses_future = losses_wrapper.device_to_host(dag_builder, transfer_queue, compute_queue)
        losses_host = losses_future.get()  # Wait for transfer to complete

        # Process per-exit losses
        exit_losses = []
        for exit_idx in range(NUM_EXITS):
            start = exit_idx * loss_elements_per_exit
            end = start + actual_batch_size
            exit_loss = losses_host[start:end].mean()
            exit_losses.append(exit_loss)
        print(f"Per-exit Losses: {batch_idx}:{exit_losses}")

        # Backpropagation
        global_grad = (padded_input_dim // simd_width, HIDDEN_DIM)
        local_grad = optimal_local_sizes(global_grad, device_limits)
        grad_event = kernel_wrapper.compute_gradients(
            global_grad,
            local_grad,
            input_wrapper.device_buffer,
            buffers['hidden'],
            buffers['exit_probs'],
            buffers['exit_weights'],
            buffers['grad_weights'],
            buffers['grad_biases'],
            targets_wrapper.device_buffer,
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
            local_adam = optimal_local_sizes(global_adam, device_limits)
            update_event = kernel_wrapper.adam_update(
                global_adam,
                local_adam,
                grad_param,
                param_buffer,
                m1_param,
                m2_param,
                np.float32(beta1_t),
                np.float32(beta2_t),
                np.int32(total_params)
            )
            update_events.append(update_event)

        global_step += 1

        # Collect all events for this batch
        batch_events = [forward_event] + exit_events + [grad_event] + update_events

        # Read exit probabilities and temperatures
        exit_probs_host = np.empty((padded_batch_size, NUM_EXITS, padded_output_classes), dtype=np.float32)
        cl.enqueue_copy(compute_queue, exit_probs_host, buffers['exit_probs'], wait_for=batch_events)

        temps_host = np.empty(NUM_EXITS, dtype=np.float32)
        cl.enqueue_copy(compute_queue, temps_host, buffers['temps'], wait_for=batch_events)

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

        # Toggle buffer set for double buffering
        buffer_set = 1 - buffer_set

    # Compute and print epoch metrics
    avg_loss = epoch_loss / len(X)
    train_acc = correct_predictions / len(X)
    print(f"Epoch {epoch:3d} | Loss: {avg_loss:.4f} | Acc: {train_acc:.1%}")
    print(f"Temperatures: {temps_host}")

# Cleanup
input_wrapper.release()
targets_wrapper.release()
mask_wrapper.release()
losses_wrapper.release()

if __name__ == "__main__":
    pass
