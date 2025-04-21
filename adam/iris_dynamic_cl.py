# iris_dynamic_cl.py
import pyopencl as cl
import numpy as np
import math
from math import gcd
from sklearn.datasets import load_iris
from sklearn.preprocessing import OneHotEncoder

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

def pad_to_multiple(arr, multiple, axis=-1):
    """Pad array to ensure SIMD alignment along specified axis"""
    pad_size = (-arr.shape[axis]) % multiple
    padded = np.pad(arr, [(0, pad_size) if i == axis else (0,0) for i in range(arr.ndim)],
                  mode='constant') if pad_size !=0 else arr
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
    """Validate workgroup configuration with memory safety prioritization"""
    # Maintain original parameters as tuples for clear error messages
    global_size = tuple(global_size)
    local_size = tuple(local_size)

    # 1. Local Memory Validation (Critical Safety Gate)
    total_work_items = np.prod(local_size)
    required_local = total_work_items * local_mem_per_item

    if required_local > device.local_mem_size:
        raise ValueError(
            f"Local memory overflow: {required_local/1024:.1f}KB > "
            f"{device.local_mem_size/1024:.1f}KB\n"
            f"Calculation: {total_work_items} workers × "
            f"{local_mem_per_item} B/item = {required_local}B"
        )

    # 2. Total Workgroup Size Limit
    if total_work_items > device.max_work_group_size:
        raise ValueError(
            f"Workgroup size {total_work_items} exceeds device limit "
            f"{device.max_work_group_size}\n"
            f"Local size: {local_size} → {total_work_items} items"
        )

    # 3. Workgroup Divisibility Check
    for g_dim, l_dim in zip(global_size, local_size):
        if l_dim == 0:
            raise ValueError(f"Invalid local dimension: {l_dim}")
        if g_dim % l_dim != 0:
            raise ValueError(
                f"Global size {g_dim} not divisible by local {l_dim}\n"
                f"Required multiplier: {math.ceil(g_dim / l_dim)}"
            )

    # 4. ND-Range Alignment (Optional but recommended)
    if len(global_size) != len(local_size):
        raise ValueError(
            f"Dimension mismatch: Global {len(global_size)}D vs Local {len(local_size)}D"
        )

    return True  # Explicit success return for monitoring

def main():
    # Data preparation
    iris = load_iris()
    X = iris.data.astype(np.float32)
    y_true = iris.target.astype(np.int32).reshape(-1,1)

    # OpenCL context setup
    ctx = cl.create_some_context()
    queue = cl.CommandQueue(ctx,properties=cl.command_queue_properties.PROFILING_ENABLE)
    device = ctx.devices[0]

    # Device configuration
    simd_width = select_simd_width(device)
    max_wg = min(device.max_work_group_size, 1024)  # Sanitize driver reports
    print(f"Training on {device.name} with SIMD-{simd_width}")
    batch_multiple = lcm(
        simd_width,
        min(device.max_work_group_size,
            next_pow2(HIDDEN_DIM//simd_width) * simd_width)
    )

    # Create and align one-hot targets only once
    y_onehot = OneHotEncoder(sparse_output=False).fit_transform(y_true)
    y_targets = pad_to_multiple(y_onehot, simd_width, axis=1)

    # Workgroup optimization
    wg_config = {
        'forward': (BATCH_SIZE, HIDDEN_DIM//simd_width),
        'gradients': (next_pow2(X.shape[1]//simd_width), next_pow2(HIDDEN_DIM)),
        'adam': max(1, (HIDDEN_DIM * X.shape[1])//(simd_width * 256))
    }
    max_padded_batch = ((BATCH_SIZE + batch_multiple -1) // batch_multiple) * batch_multiple
    output_classes_padded = ((OUTPUT_CLASSES + simd_width-1) // simd_width) * simd_width

    assert (HIDDEN_DIM % simd_width) == 0, \
        f"Hidden dim {HIDDEN_DIM} not aligned to SIMD-{simd_width}"

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

    # Mobile GPU optimizations
    if 'Mali' in device.vendor or 'Adreno' in device.vendor:
        build_opts.append("-cl-single-precision-constant -cl-opt-disable")

    program = cl.Program(ctx, "\n".join(kernel_src))

    try:
        program.build(options=" ".join(build_opts))
    except cl.RuntimeError as e:
        print(f"Build failed with OPTS: {' '.join(build_opts)}")

        if hasattr(e, 'log'):
            print("Kernel Build Log:")
            print(e.log)

        print(f"Device Max WG Size: {device.max_work_group_size}")
        print(f"SIMD Configured: {simd_width}")
        raise

    # Data preparation with SIMD alignment
    X_padded = pad_to_multiple(X, simd_width, axis=1)
    input_dim = X_padded.shape[1]
    y_padded = pad_to_multiple(y_targets, batch_multiple, axis=0)  # Align batch dim to LCM
    assert input_dim % simd_width == 0, f"Input dim {input_dim} not SIMD-{simd_width} aligned"

    # Parameter initialization
    def init_param(shape, scale=0.1):
        """
        Ensure weights and biases are aligned both for input and output dimensions
        while accounting for SIMD vector width.
        """
        arr = np.random.normal(0, scale, shape).astype(np.float32)
        return pad_to_multiple(arr, simd_width) if arr.ndim >=2 else arr

    weights = preprocess_weights(init_param((input_dim, HIDDEN_DIM)), simd_width)
    biases = init_param(HIDDEN_DIM)
    exit_weights = preprocess_weights(init_param((NUM_EXITS, HIDDEN_DIM, OUTPUT_CLASSES)), simd_width)
    exit_biases = init_param((NUM_EXITS, OUTPUT_CLASSES))

    # Buffer creation
    buffers = {
        'input': create_aligned_buffer(ctx, X_padded, simd_width, 'r'),
        # DELETE EXCESS BUFFERS
        'weights': create_aligned_buffer(ctx, weights, simd_width, 'rw'),
        'biases': create_aligned_buffer(ctx, biases, simd_width, 'rw'),
        'exit_weights': create_aligned_buffer(ctx, exit_weights, simd_width, 'rw'),
        'exit_biases': create_aligned_buffer(ctx, exit_biases, simd_width, 'rw'),
        'hidden': cl.Buffer(ctx, cl.mem_flags.READ_WRITE, max_padded_batch * HIDDEN_DIM *4),
        'targets': cl.Buffer(ctx, cl.mem_flags.READ_WRITE, max_padded_batch * output_classes_padded *4),
        'exit_probs': cl.Buffer(ctx, cl.mem_flags.READ_WRITE,
            max_padded_batch * NUM_EXITS * output_classes_padded *4),
        'losses': cl.Buffer(ctx, cl.mem_flags.READ_WRITE, max_padded_batch * NUM_EXITS *4)
    }

    # ===== Buffer Allocation Tests =====
    buffer_meta = {
        'input': (max_padded_batch * input_dim, "batch × input features"),
        'targets': (max_padded_batch * output_classes_padded, "batch × classes"),
        'hidden': (max_padded_batch * HIDDEN_DIM, "batch × hidden units"),
        'exit_probs': (max_padded_batch * NUM_EXITS * output_classes_padded, "batch × exits × classes"),
        'losses': (max_padded_batch * NUM_EXITS, "batch × exits")
    }

    for buf_name, (expected_size, dim_desc) in buffer_meta.items():
        actual_size = buffers[buf_name].size // 4  # int32 to float32 conversion
        assert actual_size == expected_size, \
            f"Buffer {buf_name} ({dim_desc}) size mismatch: {actual_size} ≠ {expected_size}"

    # Gradient and moment buffers
    for param in ['weights', 'biases', 'exit_weights', 'exit_biases']:
        buffers[f'grad_{param}'] = cl.Buffer(ctx, cl.mem_flags.READ_WRITE,
                                           buffers[param].size)
        buffers[f'm1_{param}'], buffers[f'm2_{param}'] = [
            cl.Buffer(ctx, cl.mem_flags.READ_WRITE, buffers[param].size)
            for _ in range(2)
        ]

        cl.enqueue_fill_buffer(queue, buffers[f'm1_{param}'], np.float32(0), 0, buffers[param].size)
        cl.enqueue_fill_buffer(queue, buffers[f'm2_{param}'], np.float32(0), 0, buffers[param].size)

    # Training loop
    global_step = 1
    for epoch in range(EPOCHS):
        # Shuffle and pad full dataset once per epoch
        shuffled_indices = np.random.permutation(len(X))
        # Extract from X_padded to maintain feature padding
        shuffled_X_padded = X_padded[shuffled_indices]
        shuffled_y = y_padded[shuffled_indices]  # Shuffle targets while preserving padding
        shuffled_X_padded = pad_to_multiple(shuffled_X_padded, batch_multiple, axis=0)
        shuffled_y_padded = pad_to_multiple(shuffled_y, batch_multiple, axis=0)

        num_samples = len(shuffled_indices)
        num_batches = (num_samples + BATCH_SIZE - 1) // BATCH_SIZE

        epoch_loss = 0.0
        correct_predictions = 0

        for batch_idx in range(num_batches):
            batch_start = batch_idx * BATCH_SIZE
            batch_end = min(batch_start + BATCH_SIZE, num_samples)
            actual_batch_size = batch_end - batch_start

            # Get pre-padded batch data
            X_batch = shuffled_X_padded[batch_start:batch_end]
            y_batch = shuffled_y_padded[batch_start:batch_end]  # 1. Padded inputs
            valid_y_indices = shuffled_indices[batch_start:batch_end] # 2. Original targets
            valid_y = y_onehot[valid_y_indices]           # Directly access original one-hot         
            actual_batch_size = batch_end - batch_start  # True sample count
            padded_batch_size = X_batch.shape[0]         # With padding

            # ===== Batch Integrity Tests =====
            assert X_batch.shape == (padded_batch_size, input_dim), \
                f"X_batch shape {X_batch.shape} ≠ ({padded_batch_size}, {input_dim})"
            assert y_batch.shape == (padded_batch_size, output_classes_padded), \
                f"y_batch shape {y_batch.shape} ≠ ({padded_batch_size}, {output_classes_padded})"
            
            # Update device buffers
            cl.enqueue_copy(queue, buffers['input'], X_batch)
            cl.enqueue_copy(queue, buffers['targets'], y_batch,
                           size=padded_batch_size*output_classes_padded*4)
            queue.finish()

            padded_batch_size = X_batch.shape[0]
            global_forward = (padded_batch_size, HIDDEN_DIM//simd_width)

            # Calculate optimal local size
            max_wg_x = min(device.max_work_group_size // simd_width,
                         next_pow2(padded_batch_size))
            local_forward_x = max_wg_x if padded_batch_size % max_wg_x ==0 \
                            else next_pow2(math.gcd(padded_batch_size, max_wg_x))
            compute_units = device.max_compute_units
            
            wg_forward = min(local_forward_x, compute_units * 64)
            validate_workgroup(global_forward, (wg_forward, simd_width), device,
                             wg_forward*simd_width*4)

            forward_event = program.forward_pass(
                queue,
                (padded_batch_size, HIDDEN_DIM//simd_width),  # global_size
                (wg_forward,simd_width),  # local_size
                buffers['input'],
                buffers['weights'],
                buffers['biases'],
                buffers['hidden'],
                np.int32(padded_batch_size),
                np.int32(input_dim),
                np.int32(HIDDEN_DIM),
                cl.LocalMemory(wg_forward * simd_width * 4),
                np.int32(wg_forward)
            )

            forward_event.wait()

            # ============ Early Exits ===========
            exit_events = []
            for exit_idx in range(NUM_EXITS):
                max_local = min(device.max_work_group_size,
                        next_pow2(padded_batch_size))

                wg_exit = max_local if padded_batch_size % max_local == 0 \
                        else next_pow2(math.gcd(padded_batch_size, max_local))
                global_exit = ( (padded_batch_size + wg_exit -1) // wg_exit ) * wg_exit

                validate_workgroup((global_exit,), (wg_exit,), device,
                     wg_exit*4*simd_width)  # 4 floats per item

                exit_event = program.compute_exit_probabilities(
                    queue,
                    (global_exit,),  # global_size
                    (wg_exit,),      # local_size
                    buffers['hidden'],
                    buffers['exit_weights'],
                    buffers['exit_biases'],
                    buffers['targets'],
                    buffers['exit_probs'],
                    buffers['losses'],
                    np.int32(padded_batch_size),
                    np.int32(HIDDEN_DIM),
                    np.int32(OUTPUT_CLASSES),
                    np.int32(NUM_EXITS),
                    np.int32(exit_idx),
                    cl.LocalMemory(wg_exit * 4)  # reduction_buffer
                )

                exit_events.append(exit_event)
            cl.wait_for_events(exit_events)

            # ========== Backpropagation ==========
            
            wg_grad_x = min(32, device.max_work_group_size//8)
            wg_grad_y = min(8, device.max_work_group_size//32)

            input_chunks = X_batch.shape[1] // simd_width
            global_grad = (input_chunks, HIDDEN_DIM)

            validate_workgroup(global_grad, (wg_grad_x, wg_grad_y), device,
                  (wg_grad_x * wg_grad_y) * simd_width * 4)

            grad_event = program.compute_gradients(
                queue,
                (input_chunks, HIDDEN_DIM),  # global_size
                (wg_grad_x, wg_grad_y),      # local_size
                buffers['input'],
                buffers['hidden'],
                buffers['exit_probs'],
                buffers['targets'],
                buffers['exit_weights'],
                buffers['grad_weights'],
                buffers['grad_biases'],
                np.int32(padded_batch_size),
                np.int32(input_dim),
                np.int32(HIDDEN_DIM),
                np.int32(OUTPUT_CLASSES),
                np.int32(NUM_EXITS),
                cl.LocalMemory(wg_grad_x * wg_grad_y * simd_width * 4),
                cl.LocalMemory(wg_grad_x * wg_grad_y * 4)
            )

            grad_event.wait()

            # ============ Adam Update ============
            beta1_t = 1/(1 - 0.9**global_step)
            beta2_t = 1/(1 - 0.999**global_step)
            update_events = []

            for param in ['weights', 'biases', 'exit_weights', 'exit_biases']:
                total_params = buffers[param].size // 4
                wg_adam = min(wg_config['adam'], device.max_work_group_size)
                global_size = ((total_params + wg_adam -1) // wg_adam) * wg_adam

                update_event = program.adam_update(
                    queue,
                    (global_size,),  # global_size
                    (wg_adam,),      # local_size
                    buffers[f'grad_{param}'],
                    buffers[param],
                    buffers[f'm1_{param}'],
                    buffers[f'm2_{param}'],
                    np.float32(learning_rate),
                    np.float32(ADAM_BETA1),
                    np.float32(ADAM_BETA2),
                    np.float32(beta1_t),
                    np.float32(beta2_t),
                    np.int32(total_params)
                )

                update_events.append(update_event)

            global_step += 1

            cl.wait_for_events(update_events)

            # ========= Batch Metrics =========
            padded_classes = (OUTPUT_CLASSES + simd_width -1) // simd_width * simd_width
            exit_probs = np.empty((padded_batch_size, NUM_EXITS, padded_classes), np.float32)
            cl.enqueue_copy(queue, exit_probs, buffers['exit_probs'],
                size=padded_batch_size*NUM_EXITS*padded_classes*4).wait()

            # ===== Output Validation =====
            assert exit_probs.shape == (padded_batch_size, NUM_EXITS, padded_classes), \
                f"Exit probs shape {exit_probs.shape} ≠ ({padded_batch_size}, {NUM_EXITS}, {padded_classes})"
            valid_probs = exit_probs[:actual_batch_size, :, :OUTPUT_CLASSES]

            ensemble = np.mean(valid_probs, axis=1)
            batch_correct = np.sum(np.argmax(ensemble, axis=1) == np.argmax(valid_y, axis=1)) # y_onehot is unpadded on axis=1
            correct_predictions += batch_correct

            losses = np.empty(padded_batch_size * NUM_EXITS, dtype=np.float32)
            cl.enqueue_copy(queue, losses, buffers['losses'],
                size=padded_batch_size*NUM_EXITS*4).wait()
            epoch_loss += np.mean(losses[:actual_batch_size*NUM_EXITS])

        # Epoch statistics
        train_acc = correct_predictions / num_samples
        avg_loss = epoch_loss / num_batches
        print(f"Epoch {epoch:3d} | Loss: {avg_loss:.4f} | Acc: {train_acc:.1%}")

if __name__ == "__main__":
    main()
