import pyopencl as cl
import numpy as np
import math
from math import gcd
from sklearn.datasets import load_iris

# Network Configuration
INPUT_DIM = 4
HIDDEN_DIM = 64
OUTPUT_CLASSES = 3
NUM_EXITS = 3
EPOCHS = 100
BATCH_SIZE = 128
ADAM_BETA1 = 0.9
ADAM_BETA2 = 0.999
learning_rate = 0.001 
EPSILON = 1.0e-8
CL_KERNEL_FILES = [
    'kernel_forward_pass.cl',
    'kernel_backpropagation.cl',
    'kernel_adam_update.cl',
    'kernel_multi_exit.cl'
]

def lcm(a, b):
    """Calculate Least Common Multiple"""
    return a * b // gcd(a, b)

def next_pow2(n):
    return 1 if n == 0 else 1 << (n - 1).bit_length()

def pad_to_multiple(arr, multiple, axis):
    """Pad array to ensure alignment along specified axis"""
    pad_size = (-arr.shape[axis]) % multiple
    padded = np.pad(arr, [(0, pad_size) if i == axis else (0,0) for i in range(arr.ndim)],
                    mode='constant', constant_values=0) if pad_size !=0 else arr
    assert padded.shape[axis] % multiple == 0, \
        f"Padded axis {axis} to {padded.shape[axis]} not {multiple} aligned"
    return padded

def select_simd_width(device):
    """Intelligent SIMD width selection with architectural awareness"""
    candidates = []
    if 'Intel' in device.vendor:
        subgroup_ext = 'cl_intel_subgroups' in device.extensions
        candidates = [16, 8, 4] if subgroup_ext else [4, 1]
    elif 'AMD' in device.vendor:
        cdna_features = any(s in device.name for s in ['CDNA', 'RDNA'])
        candidates = [8, 4] if cdna_features else [4, 2, 1]
    elif 'NVIDIA' in device.vendor:
        candidates = [4, 1]  # Native float4 support
    else:  #ARM/other
        candidates = [4, 2, 1]
    for sw in sorted(candidates, reverse=True):
        return sw
    return 1  # Scalar fallback

def preprocess_weights(w, simd_width):
    """Transform weights with guaranteed safe zero-padding"""
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
    raise ValueError(f"Unsupported weight dimension: {w.ndim}")

def validate_alignment(arr, simd_width, axis):
    """Ensure proper vector alignment"""
    if arr.shape[axis] % simd_width != 0:
        new_shape = list(arr.shape)
        new_shape[axis] += (-arr.shape[axis]) % simd_width
        padded = np.zeros(new_shape, dtype=arr.dtype)
        slices = tuple(slice(0, s) for s in arr.shape)
        padded[slices] = arr
        return padded
    return arr

def create_aligned_buffer(ctx, host_data, simd_width, mode):
    """Create aligned OpenCL buffer"""
    aligned_data = validate_alignment(host_data, simd_width, axis=-1)
    flags = cl.mem_flags.READ_ONLY if mode == 'r' else cl.mem_flags.READ_WRITE
    return cl.Buffer(ctx, flags | cl.mem_flags.COPY_HOST_PTR, hostbuf=aligned_data)

def validate_workgroup(global_size, local_size, device, local_mem_per_item):
    """Validate workgroup configuration"""
    global_size = tuple(global_size)
    local_size = tuple(local_size)
    total_work_items = np.prod(local_size)
    required_local = total_work_items * local_mem_per_item
    if required_local > device.local_mem_size:
        raise ValueError(f"Local memory overflow: {required_local/1024:.1f}KB > {device.local_mem_size/1024:.1f}KB")
    if total_work_items > device.max_work_group_size:
        raise ValueError(f"Workgroup size {total_work_items} exceeds device limit {device.max_work_group_size}")
    for g_dim, l_dim in zip(global_size, local_size):
        if l_dim == 0 or g_dim % l_dim != 0:
            raise ValueError(f"Global size {g_dim} not divisible by local {l_dim}")
    if len(global_size) != len(local_size):
        raise ValueError(f"Dimension mismatch: Global {len(global_size)}D vs Local {len(local_size)}D")
    return True

def pad_batch(X_batch, y_batch, batch_multiple):
    """Optimized padding with validated alignment controls"""

    # Validate input integrity
    assert isinstance(X_batch, np.ndarray) and X_batch.dtype == np.float32
    assert isinstance(y_batch, np.ndarray) and np.issubdtype(y_batch.dtype, np.integer)
    orig_batch_size = X_batch.shape[0]

    # Strategic partial padding pattern
    pad_size = ((-orig_batch_size) % batch_multiple) if batch_multiple > 1 else 0

    # --- Feature Matrix (X) Requirements ---
    # 1. Zero-pad REQUIRED for SIMD-safe FP operations
    # 2. Must maintain feature dimension alignment
    X_pad = np.zeros((orig_batch_size + pad_size, X_batch.shape[1]),
                    dtype=np.float32, order='C')  # Contiguous for CL
    X_pad[:orig_batch_size, :] = X_batch  # Explicit copy prevents memory aliasing

    # --- Labels (y) Requirements ---
    # 1. Padded values can be undefined BUT must enforce guard clauses
    # 2. Full buffer allocation with partial initialization
    y_pad = np.empty(orig_batch_size + pad_size, dtype=np.int32)
    y_pad[:orig_batch_size] = y_batch

    assert (y_pad[orig_batch_size:].shape[0] == pad_size), \
        f"Mismatched label padding: {y_pad.shape} vs {orig_batch_size + pad_size}"

    return X_pad, y_pad, (orig_batch_size + pad_size)

def main():
    # Data preparation
    iris = load_iris()
    X = iris.data.astype(np.float32)
    y_true = iris.target.astype(np.int32)  # Integer labels (n_samples,)

    # OpenCL context setup
    ctx = cl.create_some_context()
    queue = cl.CommandQueue(ctx, properties=cl.command_queue_properties.PROFILING_ENABLE)
    device = ctx.devices[0]

    # Device configuration
    simd_width = select_simd_width(device)
    print(f"Training on {device.name} with SIMD-{simd_width}")
    batch_multiple = lcm(simd_width, min(device.max_work_group_size, next_pow2(HIDDEN_DIM // simd_width) * simd_width))

    # Pad features for SIMD alignment
    X_padded = pad_to_multiple(X, simd_width, axis=1)
    input_dim_padded = X_padded.shape[1]
    assert input_dim_padded % simd_width == 0, f"Input dim {input_dim_padded} not SIMD-{simd_width} aligned"

    # Calculate padded class dimension for outputs
    output_classes_padded = ((OUTPUT_CLASSES + simd_width - 1) // simd_width) * simd_width
    assert output_classes_padded >= OUTPUT_CLASSES,\
       f"Padded classes {output_classes_padded} < true classes {OUTPUT_CLASSES}"


    # Workgroup optimization
    wg_config = {
        'forward': (BATCH_SIZE, HIDDEN_DIM // simd_width),
        'gradients': (next_pow2(input_dim_padded // simd_width), next_pow2(HIDDEN_DIM)),
        'adam': max(1, (HIDDEN_DIM * input_dim_padded) // (simd_width * 256))
    }

    # Determine max padded batch size
    max_padded_batch = ((BATCH_SIZE + batch_multiple - 1) // batch_multiple) * batch_multiple

    # Kernel compilation
    kernel_src = []
    for fname in CL_KERNEL_FILES:
        with open(fname) as f:
            kernel_src.append(f.read())

    build_opts = [
        f"-D VECTOR_TYPE={'float'+str(simd_width) if simd_width>1 else 'float'}",
        f"-D SIMD_WIDTH={simd_width}",
        f"-D USE_FAST_MATH=1"
    ]

    if 'Mali' in device.vendor or 'Adreno' in device.vendor:
        build_opts.append("-cl-single-precision-constant -cl-opt-disable")

    program = cl.Program(ctx, "\n".join(kernel_src))
    try:
        program.build(options=" ".join(build_opts))
    except cl.RuntimeError as e:
        print(f"Build failed with OPTS: {' '.join(build_opts)}")
        if hasattr(e, 'log'):
            print("Kernel Build Log:\n", e.log)
        raise

    # Parameter initialization
    def init_param(shape, scale=0.1):
        return np.random.normal(0, scale, shape).astype(np.float32)

    weights = preprocess_weights(init_param((input_dim_padded, HIDDEN_DIM)), simd_width)
    biases = init_param(HIDDEN_DIM)
    exit_weights = preprocess_weights(init_param((NUM_EXITS, HIDDEN_DIM, OUTPUT_CLASSES)), simd_width)
    exit_biases = init_param((NUM_EXITS, OUTPUT_CLASSES))

    # Buffer creation
    hidden_padded_vectors = weights.shape[0]
    hidden_dim_padded = hidden_padded_vectors * simd_width
    assert HIDDEN_DIM <= hidden_dim_padded,\
       f"Hidden dim {HIDDEN_DIM} overflow ({hidden_dim_padded} padded)"

    buffers = {
        'input': cl.Buffer(ctx, cl.mem_flags.READ_WRITE, max_padded_batch * input_dim_padded * 4),
        'weights': create_aligned_buffer(ctx, weights, simd_width, 'rw'),
        'biases': create_aligned_buffer(ctx, biases, simd_width, 'rw'),
        'exit_weights': create_aligned_buffer(ctx, exit_weights, simd_width, 'rw'),
        'exit_biases': create_aligned_buffer(ctx, exit_biases, simd_width, 'rw'),
        'hidden': cl.Buffer(ctx, cl.mem_flags.READ_WRITE, max_padded_batch * hidden_dim_padded * 4),
        'targets': cl.Buffer(ctx, cl.mem_flags.READ_WRITE, max_padded_batch * 4),  # int32 labels
        'exit_probs': cl.Buffer(ctx, cl.mem_flags.READ_WRITE, max_padded_batch * NUM_EXITS * output_classes_padded * 4),
        'losses': cl.Buffer(ctx, cl.mem_flags.READ_WRITE, max_padded_batch * NUM_EXITS * 4)
    }

    # Gradient and moment buffers
    for param in ['weights', 'biases', 'exit_weights', 'exit_biases']:
        buffers[f'grad_{param}'] = cl.Buffer(ctx, cl.mem_flags.READ_WRITE, buffers[param].size)
        buffers[f'm1_{param}'], buffers[f'm2_{param}'] = [cl.Buffer(ctx, cl.mem_flags.READ_WRITE, buffers[param].size) for _ in range(2)]
        cl.enqueue_fill_buffer(queue, buffers[f'm1_{param}'], np.float32(0), 0, buffers[param].size)
        cl.enqueue_fill_buffer(queue, buffers[f'm2_{param}'], np.float32(0), 0, buffers[param].size)

    # Training loop
    global_step = 1
    for epoch in range(EPOCHS):
        shuffled_indices = np.random.permutation(len(X))
        num_batches = (len(X) + BATCH_SIZE - 1) // BATCH_SIZE

        epoch_loss = 0.0
        correct_predictions = 0

        for batch_idx in range(num_batches):
            batch_start = batch_idx * BATCH_SIZE
            batch_end = min(batch_start + BATCH_SIZE, len(X))
            actual_batch_size = batch_end - batch_start

            batch_indices = shuffled_indices[batch_start:batch_end]
            X_batch = X_padded[batch_indices]
            y_batch = pad_to_multiple(y_true[batch_indices], batch_multiple, axis=0)

            # Pad batch to the next multiple of batch_multiple
            X_batch, y_batch, padded_batch_size = pad_batch(X_batch, y_batch, batch_multiple)

            # Copy to device buffers
            cl.enqueue_copy(queue, buffers['input'], X_batch)
            cl.enqueue_copy(queue, buffers['targets'], y_batch.astype(np.int32))

            # Forward pass
            global_forward = (padded_batch_size, hidden_dim_padded // simd_width)
            local_forward = (min(256, padded_batch_size), 1)
            validate_workgroup(global_forward, local_forward, device)
            forward_event = program.forward_pass(queue, global_forward, local_forward,
                buffers['input'], buffers['weights'], buffers['biases'], buffers['hidden'],
                np.int32(actual_batch_size), np.int32(input_dim_padded),
                np.int32(HIDDEN_DIM), np.int32(hidden_dim_padded))
            forward_event.wait()

            # Early exits
            exit_events = []
            for exit_idx in range(NUM_EXITS):
                global_exit = (padded_batch_size,)
                local_exit = (min(256, padded_batch_size),)
                validate_workgroup(global_exit, local_exit, device)
                exit_event = program.compute_exit_probabilities(queue, global_exit, local_exit,
                    buffers['hidden'], buffers['exit_weights'], buffers['exit_biases'],
                    buffers['exit_probs'], buffers['losses'], buffers['targets'].as_type(cl.channel_type.INT),
                    np.int32(actual_batch_size), np.int32(HIDDEN_DIM), np.int32(hidden_dim_padded),
                    np.int32(OUTPUT_CLASSES), np.int32(output_classes_padded),
                    np.int32(NUM_EXITS), np.int32(exit_idx))
                exit_events.append(exit_event)
            cl.wait_for_events(exit_events)

            # Backpropagation
            global_grad = (input_dim_padded // simd_width, HIDDEN_DIM)
            local_grad = (min(16, input_dim_padded // simd_width), min(16, HIDDEN_DIM))
            validate_workgroup(global_grad, local_grad, device)
            grad_event = program.compute_gradients(queue, global_grad, local_grad,
                buffers['input'], buffers['hidden'], buffers['exit_probs'], buffers['exit_weights'],
                buffers['grad_weights'], buffers['grad_biases'], buffers['targets'].as_type(cl.channel_type.INT),
                np.int32(actual_batch_size), np.int32(input_dim_padded),
                np.int32(HIDDEN_DIM), np.int32(hidden_dim_padded),
                np.int32(OUTPUT_CLASSES), np.int32(output_classes_padded), np.int32(NUM_EXITS))
            grad_event.wait()

            # Adam update
            beta1_t = 1 / (1 - ADAM_BETA1 ** global_step)
            beta2_t = 1 / (1 - ADAM_BETA2 ** global_step)
            for param_name in ['weights', 'biases', 'exit_weights', 'exit_biases']:
                total_params = buffers[param_name].size // 4
                global_adam = (total_params,)
                local_adam = (min(256, total_params),)
                validate_workgroup(global_adam, local_adam, device)
                update_event = program.adam_update(queue, global_adam, local_adam,
                    buffers[f'grad_{param_name}'], buffers[param_name],
                    buffers[f'm1_{param_name}'], buffers[f'm2_{param_name}'],
                    np.float32(learning_rate), np.float32(ADAM_BETA1), np.float32(ADAM_BETA2),
                    np.float32(beta1_t), np.float32(beta2_t), np.float32(EPSILON), np.int32(total_params))
                update_event.wait()
            global_step += 1

            # Metrics
            exit_probs = np.empty((padded_batch_size, NUM_EXITS, output_classes_padded), np.float32)
            cl.enqueue_copy(queue, exit_probs, buffers['exit_probs'])
            valid_probs = exit_probs[:actual_batch_size, :, :OUTPUT_CLASSES]
            ensemble = np.mean(valid_probs, axis=1)
            batch_correct = np.sum(np.argmax(ensemble, axis=1) == y_batch[:actual_batch_size])
            correct_predictions += batch_correct

            losses = np.empty(padded_batch_size * NUM_EXITS, np.float32)
            cl.enqueue_copy(queue, losses, buffers['losses'])
            epoch_loss += np.mean(losses[:actual_batch_size * NUM_EXITS])

        train_acc = correct_predictions / len(X)
        avg_loss = epoch_loss / num_batches
        print(f"Epoch {epoch:3d} | Loss: {avg_loss:.4f} | Acc: {train_acc:.1%}")

if __name__ == "__main__":
    main()
