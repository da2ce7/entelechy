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
LEARNING_RATE = 0.001
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
    raise ValueError(f"Unsupported weight dimension: {w.ndim}")

def validate_alignment(arr, simd_width, axis):
    if arr.shape[axis] % simd_width != 0:
        new_shape = list(arr.shape)
        new_shape[axis] = ((new_shape[axis] + simd_width - 1) // simd_width) * simd_width
        padded = np.zeros(new_shape, dtype=arr.dtype)
        slices = tuple(slice(0, s) for s in arr.shape)
        padded[slices] = arr
        return padded
    return arr

def create_aligned_buffer(ctx, host_data, simd_width, mode):
    aligned_data = validate_alignment(host_data, simd_width, axis=-1)
    flags = cl.mem_flags.READ_ONLY if mode == 'r' else cl.mem_flags.READ_WRITE
    return cl.Buffer(ctx, flags | cl.mem_flags.COPY_HOST_PTR, hostbuf=aligned_data)

def pad_batch(X_batch, y_batch, batch_multiple):
    orig_size = X_batch.shape[0]
    pad_size = (-orig_size % batch_multiple) if batch_multiple > 1 else 0
    X_pad = np.zeros((orig_size + pad_size, X_batch.shape[1]), dtype=np.float32)
    X_pad[:orig_size] = X_batch
    y_pad = np.zeros(orig_size + pad_size, dtype=np.int32)
    y_pad[:orig_size] = y_batch
    return X_pad, y_pad, orig_size + pad_size

def optimal_local_size(global_size, device_limits):
    max_wg = device_limits['max_work_group_size']
    max_dims = device_limits['max_work_item_sizes']

    def _compute_score(local, global_size):
        utilization = (local / (local * math.ceil(global_size/local)))
        pot_bonus = 0.1 if (local & (local-1)) == 0 else 0
        return local + pot_bonus + utilization

    if isinstance(global_size, int):
        valid = [s for s in range(1, min(max_wg, max_dims[0])+1) if global_size % s == 0]
        return max(valid, key=lambda x: (x, _compute_score(x, global_size)), default=1)
    else:
        gx, gy = global_size
        valid_x = [s for s in divisors(gx) if s <= max_dims[0]]
        valid_y = [s for s in divisors(gy) if s <= max_dims[1]]
        candidates = []
        for lx in valid_x:
            for ly in valid_y:
                if lx * ly > max_wg: continue
                score = lx * ly + (0.2 * ((lx & (lx-1) == 0) + (ly & (ly-1) == 0)))
                candidates.append((score, (lx, ly)))
        return max(candidates, key=lambda x: x[0])[1] if candidates else (1,1)

def select_simd_width(device):
    if 'Intel' in device.vendor and 'cl_intel_subgroups' in device.extensions:
        return 16
    elif 'AMD' in device.vendor:
        return 8 if any(s in device.name for s in ['CDNA', 'RDNA']) else 4
    elif 'NVIDIA' in device.vendor:
        return 4
    return 1

def validate_workgroup(global_size, local_size, device, local_mem_per_item):
    total_work_items = np.prod(local_size)
    required_local = total_work_items * local_mem_per_item
    if required_local > device.local_mem_size:
        raise ValueError(f"Local memory overflow: {required_local/1024:.1f}KB > {device.local_mem_size/1024:.1f}KB")
    if total_work_items > device.max_work_group_size:
        raise ValueError(f"Workgroup size {total_work_items} exceeds limit {device.max_work_group_size}")
    for g, l in zip(global_size, local_size):
        if g % l != 0 or l > g:
            raise ValueError(f"Invalid sizes: global={g}, local={l}")
    return True

def main():
    # Data preparation
    iris = load_iris()
    X = iris.data.astype(np.float32)
    y_true = iris.target.astype(np.int32)

    # OpenCL setup
    ctx = cl.create_some_context()
    transfer_queue = cl.CommandQueue(ctx)
    compute_queue = cl.CommandQueue(ctx, properties=cl.command_queue_properties.PROFILING_ENABLE)
    device = ctx.devices[0]

    # Device configuration
    device_limits = {
       'max_work_group_size': device.max_work_group_size,
       'max_work_item_sizes': device.max_work_item_sizes,
    }
    simd_width = select_simd_width(device)
    print(f"Training on {device.name} with SIMD-{simd_width}")
    batch_multiple = lcm(simd_width, min(device_limits['max_work_group_size'], next_pow2(HIDDEN_DIM // simd_width) * simd_width))

    # Pad features for SIMD alignment
    X_padded = pad_to_multiple(X, simd_width, axis=1)
    input_dim_padded = X_padded.shape[1]
    output_classes_padded = ((OUTPUT_CLASSES + simd_width - 1) // simd_width) * simd_width
    max_padded_batch = ((BATCH_SIZE + batch_multiple - 1) // batch_multiple) * batch_multiple

    # Kernel compilation
    kernel_src = []
    for fname in CL_KERNEL_FILES:
        with open(fname) as f:
            kernel_src.append(f.read())
    build_opts = [
        f"-D VECTOR_TYPE={'float' + str(simd_width) if simd_width > 1 else 'float'}",
        f"-D SIMD_WIDTH={simd_width}",
        f"-D USE_FAST_MATH=1"
    ]
    program = cl.Program(ctx, "\n".join(kernel_src)).build(options=" ".join(build_opts))

    # Parameter initialization
    weights = preprocess_weights(np.random.normal(0, 0.1, (input_dim_padded, HIDDEN_DIM)).astype(np.float32), simd_width)
    biases = np.random.normal(0, 0.1, HIDDEN_DIM).astype(np.float32)
    exit_weights = preprocess_weights(np.random.normal(0, 0.1, (NUM_EXITS, HIDDEN_DIM, OUTPUT_CLASSES)).astype(np.float32), simd_width)
    exit_biases = np.random.normal(0, 0.1, (NUM_EXITS, OUTPUT_CLASSES)).astype(np.float32)

    # Buffer creation
    hidden_dim_padded = weights.shape[0] * simd_width
    input_buf_size = max_padded_batch * input_dim_padded * 4
    targets_buf_size = max_padded_batch * 4

    buffers = {
        'weights': create_aligned_buffer(ctx, weights, simd_width, 'rw'),
        'biases': create_aligned_buffer(ctx, biases, simd_width, 'rw'),
        'exit_weights': create_aligned_buffer(ctx, exit_weights, simd_width, 'rw'),
        'exit_biases': create_aligned_buffer(ctx, exit_biases, simd_width, 'rw'),
        'hidden': cl.Buffer(ctx, cl.mem_flags.READ_WRITE, max_padded_batch * hidden_dim_padded * 4),
        'exit_probs': cl.Buffer(ctx, cl.mem_flags.READ_WRITE, max_padded_batch * NUM_EXITS * output_classes_padded * 4),
        'losses': cl.Buffer(ctx, cl.mem_flags.READ_WRITE, max_padded_batch * NUM_EXITS * 4)
    }

    # Gradient and moment buffers
    for param in ['weights', 'biases', 'exit_weights', 'exit_biases']:
        buffers[f'grad_{param}'] = cl.Buffer(ctx, cl.mem_flags.READ_WRITE, buffers[param].size)
        buffers[f'm1_{param}'] = cl.Buffer(ctx, cl.mem_flags.READ_WRITE, buffers[param].size)
        buffers[f'm2_{param}'] = cl.Buffer(ctx, cl.mem_flags.READ_WRITE, buffers[param].size)
        cl.enqueue_fill_buffer(compute_queue, buffers[f'm1_{param}'], np.float32(0), 0, buffers[param].size)
        cl.enqueue_fill_buffer(compute_queue, buffers[f'm2_{param}'], np.float32(0), 0, buffers[param].size)

    # Double buffering setup
    staging_input_A = cl.Buffer(ctx, cl.mem_flags.READ_WRITE | cl.mem_flags.ALLOC_HOST_PTR, size=input_buf_size)
    staging_input_B = cl.Buffer(ctx, cl.mem_flags.READ_WRITE | cl.mem_flags.ALLOC_HOST_PTR, size=input_buf_size)
    staging_targets_A = cl.Buffer(ctx, cl.mem_flags.READ_WRITE | cl.mem_flags.ALLOC_HOST_PTR, size=targets_buf_size)
    staging_targets_B = cl.Buffer(ctx, cl.mem_flags.READ_WRITE | cl.mem_flags.ALLOC_HOST_PTR, size=targets_buf_size)

    input_dev_A = cl.Buffer(ctx, cl.mem_flags.READ_WRITE, size=input_buf_size)
    input_dev_B = cl.Buffer(ctx, cl.mem_flags.READ_WRITE, size=input_buf_size)
    targets_dev_A = cl.Buffer(ctx, cl.mem_flags.READ_WRITE, size=targets_buf_size)
    targets_dev_B = cl.Buffer(ctx, cl.mem_flags.READ_WRITE, size=targets_buf_size)

    sets = [
        {'staging_input': staging_input_A, 'staging_targets': staging_targets_A, 'input_dev': input_dev_A, 'targets_dev': targets_dev_A, 'compute_events': []},
        {'staging_input': staging_input_B, 'staging_targets': staging_targets_B, 'input_dev': input_dev_B, 'targets_dev': targets_dev_B, 'compute_events': []}
    ]

    # Training loop
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
            X_pad, y_pad, padded_batch_size = pad_batch(X_batch, y_batch, batch_multiple)

            # Map staging buffers for host write using context managers
            with transfer_queue.map_buffer(
                current_set['staging_input'], cl.map_flags.WRITE_INVALIDATE_REGION,
                shape=(padded_batch_size, input_dim_padded), dtype=np.float32
            ) as host_input:
                host_input[:padded_batch_size, :input_dim_padded] = X_pad

            with transfer_queue.map_buffer(
                current_set['staging_targets'], cl.map_flags.WRITE_INVALIDATE_REGION,
                shape=(padded_batch_size,), dtype=np.int32
            ) as host_targets:
                host_targets[:padded_batch_size] = y_pad

            # Transfer to device buffers
            transfer_event_input = cl.enqueue_copy_buffer(
                transfer_queue,
                current_set['staging_input'],
                current_set['input_dev'],
                src_offset=0,
                dst_offset=0,
                size=padded_batch_size * input_dim_padded * 4,
                is_blocking=False
            )
            transfer_event_targets = cl.enqueue_copy_buffer(
                transfer_queue,
                current_set['staging_targets'],
                current_set['targets_dev'],
                src_offset=0,
                dst_offset=0,
                size=padded_batch_size * 4,
                is_blocking=False
            )

            # Forward pass
            global_forward = (padded_batch_size, hidden_dim_padded // simd_width)
            local_forward = optimal_local_size(global_forward, device_limits)
            validate_workgroup(global_forward, local_forward, device, 0)
            forward_event = program.forward_pass(
                compute_queue,
                global_forward,
                local_forward,
                current_set['input_dev'],
                buffers['weights'],
                buffers['biases'],
                buffers['hidden'],
                np.int32(padded_batch_size),
                np.int32(input_dim_padded),
                np.int32(HIDDEN_DIM),
                np.int32(hidden_dim_padded),
                wait_for=[transfer_event_input]
            )

            # Early exits
            exit_events = []
            for exit_idx in range(NUM_EXITS):
                global_exit = (padded_batch_size,)
                local_exit = (optimal_local_size(global_exit, device_limits),)
                validate_workgroup(global_exit, local_exit, device, 0)
                exit_event = program.compute_exit_probabilities(
                    compute_queue,
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
                    np.int32(exit_idx),
                    wait_for=[forward_event, transfer_event_targets]
                )
                exit_events.append(exit_event)

            # Backpropagation
            global_grad = (input_dim_padded // simd_width, HIDDEN_DIM)
            local_grad = optimal_local_size(global_grad, device_limits)
            validate_workgroup(global_grad, local_grad, device, 0)
            grad_event = program.compute_gradients(
                compute_queue,
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
                np.int32(NUM_EXITS),
                wait_for=exit_events
            )

            # Adam update
            beta1_t = 1 / (1 - ADAM_BETA1 ** global_step)
            beta2_t = 1 / (1 - ADAM_BETA2 ** global_step)
            update_events = []
            for param in ['weights', 'biases', 'exit_weights', 'exit_biases']:
                total_params = buffers[param].size // 4
                global_adam = (total_params,)
                local_adam = (optimal_local_size(global_adam, device_limits),)
                validate_workgroup(global_adam, local_adam, device, 0)
                update_event = program.adam_update(
                    compute_queue,
                    global_adam,
                    local_adam,
                    buffers[f'grad_{param}'],
                    buffers[param],
                    buffers[f'm1_{param}'],
                    buffers[f'm2_{param}'],
                    np.float32(LEARNING_RATE),
                    np.float32(ADAM_BETA1),
                    np.float32(ADAM_BETA2),
                    np.float32(beta1_t),
                    np.float32(beta2_t),
                    np.float32(EPSILON),
                    np.int32(total_params),
                    wait_for=[grad_event]
                )
                update_events.append(update_event)
            global_step += 1

            # Collect all compute events
            compute_events = [forward_event] + exit_events + [grad_event] + update_events
            current_set['compute_events'] = compute_events

            # Read losses and exit probabilities
            losses_host = np.empty(padded_batch_size * NUM_EXITS, dtype=np.float32)
            exit_probs_host = np.empty((padded_batch_size, NUM_EXITS, output_classes_padded), dtype=np.float32)
            cl.enqueue_copy(compute_queue, losses_host, buffers['losses'], wait_for=compute_events)
            cl.enqueue_copy(compute_queue, exit_probs_host, buffers['exit_probs'], wait_for=compute_events)

            # Compute batch metrics
            valid_losses = losses_host[:actual_batch_size * NUM_EXITS]
            valid_probs = exit_probs_host[:actual_batch_size, :, :OUTPUT_CLASSES]
            batch_loss = np.mean(valid_losses)
            epoch_loss += batch_loss * actual_batch_size
            ensemble = np.mean(valid_probs, axis=1)
            predicted_classes = np.argmax(ensemble, axis=1)
            batch_correct = np.sum(predicted_classes == y_batch)
            correct_predictions += batch_correct

            # Prefetch next batch
            if next_set:
                next_batch_start = (batch_idx + 1) * BATCH_SIZE
                next_batch_end = min(next_batch_start + BATCH_SIZE, len(X))
                next_actual_size = next_batch_end - next_batch_start
                next_indices = shuffled_indices[next_batch_start:next_batch_end]
                X_next = X_padded[next_indices]
                y_next = y_true[next_indices]
                X_next_pad, y_next_pad, next_padded_size = pad_batch(X_next, y_next, batch_multiple)

                with transfer_queue.map_buffer(
                    next_set['staging_input'], cl.map_flags.WRITE_INVALIDATE_REGION,
                    shape=(next_padded_size, input_dim_padded), dtype=np.float32
                ) as next_host_input:
                    next_host_input[:next_padded_size, :input_dim_padded] = X_next_pad

                with transfer_queue.map_buffer(
                    next_set['staging_targets'], cl.map_flags.WRITE_INVALIDATE_REGION,
                    shape=(next_padded_size,), dtype=np.int32
                ) as next_host_targets:
                    next_host_targets[:next_padded_size] = y_next_pad

                cl.enqueue_copy_buffer(
                    transfer_queue,
                    next_set['staging_input'],
                    next_set['input_dev'],
                    src_offset=0,
                    dst_offset=0,
                    size=next_padded_size * input_dim_padded * 4,
                    is_blocking=False
                )
                cl.enqueue_copy_buffer(
                    transfer_queue,
                    next_set['staging_targets'],
                    next_set['targets_dev'],
                    src_offset=0,
                    dst_offset=0,
                    size=next_padded_size * 4,
                    is_blocking=False
                )

        # Compute and print epoch metrics
        avg_loss = epoch_loss / len(X)
        train_acc = correct_predictions / len(X)
        print(f"Epoch {epoch:3d} | Loss: {avg_loss:.4f} | Acc: {train_acc:.1%}")

if __name__ == "__main__":
    main()
