# iris_dynamic_cl.py
import pyopencl as cl
import numpy as np
import math
from sklearn.datasets import load_iris
from sklearn.preprocessing import OneHotEncoder

# Network Configuration
INPUT_DIM = 4
HIDDEN_DIM = 64
OUTPUT_CLASSES = 3
NUM_EXITS = 3
EPOCHS = 100
BATCH_SIZE = 128
CL_KERNEL_FILES = [
    'kernel_forward_pass.cl',
    'kernel_backpropagation.cl',
    'kernel_adam_update.cl',
    'kernel_multi_exit.cl'
]

def next_pow2(n):
    return 1 if n == 0 else 1 << (n - 1).bit_length()

def pad_to_multiple(arr, multiple, axis=-1):
    """Pad array to ensure SIMD alignment along specified axis"""
    pad_size = (-arr.shape[axis]) % multiple
    return np.pad(arr, [(0, pad_size) if i == axis else (0,0) for i in range(arr.ndim)],
                  mode='constant') if pad_size !=0 else arr

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

    # First compatible width that divides hidden dimension
    for sw in sorted(candidates, reverse=True):
        if HIDDEN_DIM % sw == 0:
            return sw
    return 1  # Scalar fallback

def preprocess_weights(w, simd_width):
    """Transform weights for optimal vectorized access patterns"""
    if w.ndim == 2:  # [input, hidden]
        w_padded = pad_to_multiple(w.T, simd_width)
        return w_padded.reshape(HIDDEN_DIM//simd_width, -1, simd_width)
    elif w.ndim == 3:  # [exits, hidden, classes]
        w_padded = pad_to_multiple(w, simd_width, axis=2)
        return w_padded.transpose(0, 2, 1).reshape(NUM_EXITS, w_padded.shape[2],
                                                  HIDDEN_DIM//simd_width, simd_width)
    raise ValueError(f"Unsupported weight dim {w.ndim}")

def create_aligned_buffer(ctx, host_data, simd_width, mode='r'):
    """Create device buffer with proper alignment and padding"""
    bytes_per_element = host_data.dtype.itemsize
    alignment = simd_width * bytes_per_element

    padded_shape = list(host_data.shape)
    padded_shape[-1] += (-host_data.shape[-1] % simd_width)

    padded_data = np.zeros(padded_shape, dtype=host_data.dtype)
    slices = tuple(slice(0, s) for s in host_data.shape)
    padded_data[slices] = host_data

    flags = cl.mem_flags.READ_ONLY if mode == 'r' else cl.mem_flags.READ_WRITE
    return cl.Buffer(ctx, flags | cl.mem_flags.COPY_HOST_PTR, hostbuf=padded_data)

def validate_workgroup(global_size, local_size, device, local_mem_per_item=0):
    """Comprehensive workgroup validation with memory checks"""
    # Size compatibility
    if not all(g % l == 0 for g,l in zip(global_size, local_size)):
        raise ValueError(f"Global {global_size} not divisible by local {local_size}")

    # Total workgroup size
    total_wg = np.prod(local_size)
    if total_wg > device.max_work_group_size:
        raise ValueError(f"Workgroup {total_wg} exceeds device limit {device.max_work_group_size}")

    # Local memory requirements
    required_local = total_wg * local_mem_per_item
    if required_local > device.local_mem_size:
        raise ValueError(f"Insufficient local memory: {required_local} > {device.local_mem_size}")

def main():
    # Data preparation
    iris = load_iris()
    X = iris.data.astype(np.float32)
    y_true = iris.target.astype(np.int32).reshape(-1,1)

    # One-Hot targets must be aligned and used in buffers
    y_onehot = OneHotEncoder(sparse_output=False).fit_transform(y_true)
    y_targets = pad_to_multiple(y_onehot, simd_width, axis=1).astype(np.float32)

    # OpenCL context setup
    ctx = cl.create_some_context()
    queue = cl.CommandQueue(ctx,
                          properties=cl.command_queue_properties.PROFILING_ENABLE)
    device = ctx.devices[0]

    # Device configuration
    simd_width = select_simd_width(device)
    max_wg = min(device.max_work_group_size, 1024)  # Sanitize driver reports
    print(f"Training on {device.name} with SIMD-{simd_width}")

    # Workgroup optimization
    # next_pow2() ensures coalesced memory access across wavefronts
    wg_config = {
        'forward': (BATCH_SIZE, HIDDEN_DIM//simd_width),
        'gradients': (next_pow2(X.shape[1]//simd_width), next_pow2(HIDDEN_DIM)),
        'adam': max(1, (HIDDEN_DIM * X.shape[1])//(simd_width * 256))
    }

    assert (HIDDEN_DIM % simd_width) == 0, \
        f"Hidden dim {HIDDEN_DIM} not aligned to SIMD-{simd_width}"

    # Kernel compilation
    kernel_src = []
    for fname in CL_KERNEL_FILES:
        with open(fname) as f:
            kernel_src.append(f.read())

    build_opts = [
        f"-D SIMD_WIDTH={simd_width}",
        f"-D FLOATV=float{simd_width if simd_width>1 else ''}",
        f"-D BATCH_SIZE={BATCH_SIZE}",
        f"-D HIDDEN_DIM={HIDDEN_DIM}",
        f"-D OUTPUT_CLASSES={OUTPUT_CLASSES}",
        f"-D NUM_EXITS={NUM_EXITS}",
        "-cl-mad-enable -cl-fast-relaxed-math"
    ]

    # Mobile GPU optimizations
    if 'Mali' in device.vendor or 'Adreno' in device.vendor:
        build_opts.append("-cl-single-precision-constant -cl-opt-disable")

    program = cl.Program(ctx, "\n".join(kernel_src)).build(" ".join(build_opts))

    # Data preparation with SIMD alignment
    X_padded = pad_to_multiple(X, simd_width)
    input_dim = X_padded.shape[1]
    assert input_dim % simd_width == 0, f"Input dim {input_dim} not SIMD-{simd_width} aligned"

    # Parameter initialization
    def init_param(shape, scale=0.1):
        arr = np.random.normal(0, scale, shape).astype(np.float32)
        return pad_to_multiple(arr, simd_width) if arr.ndim >=2 else arr

    weights = preprocess_weights(init_param((input_dim, HIDDEN_DIM)), simd_width)
    biases = init_param(HIDDEN_DIM)
    exit_weights = preprocess_weights(init_param((NUM_EXITS, HIDDEN_DIM, OUTPUT_CLASSES)), simd_width)
    exit_biases = init_param((NUM_EXITS, OUTPUT_CLASSES))

    # Buffer creation
    buffers = {
        'input': create_aligned_buffer(ctx, X_padded, simd_width, 'r'),
        'weights': create_aligned_buffer(ctx, weights, simd_width, 'rw'),
        'biases': create_aligned_buffer(ctx, biases, simd_width, 'rw'),
        'exit_weights': create_aligned_buffer(ctx, exit_weights, simd_width, 'rw'),
        'exit_biases': create_aligned_buffer(ctx, exit_biases, simd_width, 'rw'),
        'hidden': cl.Buffer(ctx, cl.mem_flags.READ_WRITE,
                          X_padded.nbytes * HIDDEN_DIM//input_dim),
        'targets': create_aligned_buffer(ctx, y_targets, simd_width, 'r'),  # Now proper one-hot
        'exit_probs': cl.Buffer(ctx, cl.mem_flags.READ_WRITE,
            BATCH_SIZE * NUM_EXITS * ((OUTPUT_CLASSES + simd_width-1)//simd_width*simd_width)*4),
        'losses': cl.Buffer(ctx, cl.mem_flags.READ_WRITE, BATCH_SIZE * NUM_EXITS *4)
    }

    # Gradient and moment buffers
    for param in ['weights', 'biases', 'exit_weights', 'exit_biases']:
        buffers[f'grad_{param}'] = cl.Buffer(ctx, cl.mem_flags.READ_WRITE,
                                           buffers[param].size)
        buffers[f'm1_{param}'], buffers[f'm2_{param}'] = [
            cl.Buffer(ctx, cl.mem_flags.READ_WRITE, buffers[param].size)
            for _ in range(2)
        ]
        buffer_elements = buffers[param].size // 4  # 4 bytes/float
        init_m1 = np.zeros(buffer_elements, dtype=np.float32)
        cl.enqueue_copy(queue, buffers[f'm1_{param}'], init_m1)
        cl.enqueue_copy(queue, buffers[f'm2_{param}'], init_m1)

    # Training loop
    global_step = 1
    for epoch in range(EPOCHS):
        epoch_loss = 0.0
        correct_predictions = 0

        # Shuffle dataset each epoch
        indices = np.random.permutation(len(X))
        for batch_start in range(0, len(indices), BATCH_SIZE):
            batch_end = min(batch_start + BATCH_SIZE, len(indices))
            actual_batch_size = batch_end - batch_start
            batch_indices = indices[batch_start:batch_end]

            # Get batch data (with SIMD padding if needed)
            X_batch = pad_to_multiple(X[batch_indices], simd_width)
            y_batch = y_true[batch_indices]

            # Update input buffer
            cl.enqueue_copy(queue, buffers['input'], X_batch)
            cl.enqueue_copy(queue, buffers['targets'], y_batch)
            queue.finish()

            # =========== Forward Pass ===========
            global_forward = (actual_batch_size, HIDDEN_DIM//simd_width)
            max_wg_x = min(device.max_work_group_size // simd_width,
              next_pow2(actual_batch_size))  # Align to batch-size
            local_forward_x = max_wg_x if actual_batch_size % max_wg_x ==0 \
                else next_pow2(math.gcd(actual_batch_size, max_wg_x))
            compute_units = device.max_compute_units
            local_forward_x_limited = min(local_forward_x, compute_units * 64)
            local_forward = (local_forward_x_limited, simd_width)
            validate_workgroup(global_forward, local_forward, device,
                             local_forward[0]*local_forward[1]*4)

            forward_event = program.forward_pass(
                queue, global_forward, local_forward,
                buffers['input'], buffers['weights'], buffers['biases'],
                buffers['hidden'], cl.LocalMemory(np.prod(local_forward)*4),
                np.int32(X_batch.shape[1]), np.int32(actual_batch_size)
            )
            forward_event.wait()

            # ============ Early Exits ===========
            exit_events = []
            for exit_idx in range(NUM_EXITS):
                global_exit = (actual_batch_size,)
                local_exit = (min(device.max_work_group_size,
                                next_pow2(actual_batch_size)),)
                validate_workgroup(global_exit, local_exit, device,
                                 local_exit[0]*simd_width*4)

                exit_event = program.compute_exit_probabilities(
                    queue, global_exit, local_exit,
                    buffers['hidden'], buffers['exit_weights'],
                    buffers['exit_biases'], buffers['targets'],
                    buffers['exit_probs'], buffers['losses'],
                    cl.LocalMemory(local_exit[0]*simd_width*4),
                    np.int32(exit_idx), np.int32(actual_batch_size)
                )
                exit_events.append(exit_event)
            cl.wait_for_events(exit_events)

            # ========== Backpropagation ==========
            global_grad = (X_batch.shape[1]//simd_width, HIDDEN_DIM)
            local_grad = (
                min(32, device.max_work_group_size//8),
                min(8, device.max_work_group_size//32)
            )
            validate_workgroup(global_grad, local_grad, device,
                             np.prod(local_grad)*simd_width*4)

            grad_event = program.compute_gradients(
                queue, global_grad, local_grad,
                buffers['input'], buffers['hidden'], buffers['exit_probs'],
                buffers['targets'], buffers['exit_weights'],
                buffers['grad_weights'], buffers['grad_biases'],
                cl.LocalMemory(np.prod(local_grad)*simd_width*4),
                np.int32(X_batch.shape[1]), np.int32(actual_batch_size)
            )
            grad_event.wait()

            # ============ Adam Update ============
            beta1_t = 1/(1 - 0.9**global_step)
            beta2_t = 1/(1 - 0.999**global_step)
            update_events = []

            for param in ['weights', 'biases', 'exit_weights', 'exit_biases']:
                total_params = buffers[param].size // 4
                # Round up to nearest multiple of workgroup size
                global_size = ((total_params + wg_config['adam'] -1) // wg_config['adam']) * wg_config['adam']
                vector_stride = max(1, global_size // wg_config['adam'])

                update_event = program.adam_update(
                    queue, (global_size,), (wg_config['adam'],),
                    buffers[f'grad_{param}'], buffers[param],
                    buffers[f'm1_{param}'], buffers[f'm2_{param}'],
                    np.float32(0.001), np.float32(0.9), np.float32(0.999),
                    np.float32(beta1_t), np.float32(beta2_t),
                    np.int32(vector_stride), np.int32(actual_batch_size)
                )
                update_events.append(update_event)
            cl.wait_for_events(update_events)

            # Calculate batch accuracy
            padded_classes = (OUTPUT_CLASSES + simd_width -1) // simd_width * simd_width
            exit_probs = np.empty((actual_batch_size, NUM_EXITS, padded_classes), np.float32)
            cl.enqueue_copy(queue, exit_probs, buffers['exit_probs'],
                size=actual_batch_size*NUM_EXITS*padded_classes*4).wait()
            valid_probs = exit_probs[:, :, :OUTPUT_CLASSES]

            ensemble = np.mean(valid_probs, axis=1)
            correct_predictions += np.sum(np.argmax(ensemble, axis=1) == y_batch.flatten())
            losses = np.empty(actual_batch_size * NUM_EXITS, dtype=np.float32)
            cl.enqueue_copy(queue, losses, buffers['losses'],
                size=actual_batch_size*NUM_EXITS*4).wait()
            epoch_loss += np.mean(losses)

            global_step += 1

        # Epoch statistics
        train_acc = correct_predictions / len(indices)
        num_batches = (len(indices) + BATCH_SIZE - 1) // BATCH_SIZE
        avg_loss = epoch_loss / num_batches
        print(f"Epoch {epoch:3d} | Loss: {avg_loss:.4f} | Acc: {train_acc:.1%}")


if __name__ == "__main__":
    main()
