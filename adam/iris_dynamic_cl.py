# iris_dynamic_cl.py
import pyopencl as cl
import numpy as np
from sklearn.datasets import load_iris
from sklearn.preprocessing import OneHotEncoder

# Configuration
INPUT_DIM = 4
HIDDEN_DIM = 8
OUTPUT_CLASSES = 3
NUM_EXITS = 3
EPOCHS = 100
BATCH_SIZE = 150
CL_KERNEL_FILES = [
    'kernel_forward_pass.cl',
    'kernel_backpropagation.cl',
    'kernel_adam_update.cl',
    'kernel_multi_exit.cl'
]

def pad_to_multiple(arr, multiple, axis=-1):
    pad_size = (-arr.shape[axis]) % multiple
    if pad_size == 0: return arr
    new_shape = list(arr.shape)
    new_shape[axis] += pad_size
    padded = np.zeros(new_shape, dtype=arr.dtype)
    slices = [slice(None)]*arr.ndim
    slices[axis] = slice(0, arr.shape[axis])
    padded[tuple(slices)] = arr
    return padded

def preprocess_weights(w, simd_width):
    if w.ndim == 2:  # [input, hidden]
        return w.T.reshape(HIDDEN_DIM//simd_width, INPUT_DIM//simd_width, simd_width)
    elif w.ndim == 3:  # [exits, hidden, classes]
        w_padded = pad_to_multiple(w, simd_width, axis=-1)
        return w_padded.transpose(0,2,1).reshape(
            NUM_EXITS, w_padded.shape[-1], HIDDEN_DIM//simd_width, simd_width
        )
    raise ValueError(f"Unsupported weight dim {w.ndim}")

def safe_init(shape, std=0.1, simd_width=4):
    arr = np.random.normal(0, std, shape).astype(np.float32)
    return pad_to_multiple(arr, simd_width) if arr.ndim >=2 else arr

def validate_wg(device, wg_type, wg_size):
    max_wg = device.max_work_group_size
    if isinstance(wg_size, tuple):
        product = wg_size[0] * wg_size[1]
        assert product <= max_wg, f"{wg_type} WG {product} > {max_wg}"
        assert (max_wg % wg_size[0]) >= wg_size[1], "WG Y-dim exceeds X-stride"
    else:
        assert wg_size <= max_wg, f"{wg_type} WG {wg_size} > {max_wg}"

def main():
    # Data preparation
    iris = load_iris()
    X = iris.data.astype(np.float32)
    y_true = iris.target.astype(np.int32)
    y_onehot = OneHotEncoder(sparse_output=False).fit_transform(y_true.reshape(-1,1))

    # OpenCL setup
    ctx = cl.create_some_context()
    queue = cl.CommandQueue(ctx, properties=cl.command_queue_properties.PROFILING_ENABLE)
    device = ctx.devices[0]

    # Device configuration
    simd_width = 4 if 'cl_khr_fp64' in device.extensions else 2
    max_wg = device.max_work_group_size
    wg_config = {
        "forward": min(max_wg, 2**int(np.log2(HIDDEN_DIM//simd_width))),
        "gradients": (min(32, max_wg//8), min(8, max_wg//32)),
        "adam": min(256, max_wg),
        "exit": min(64, max_wg)
    }
    for k,v in wg_config.items(): validate_wg(device, k, v)

    # Kernel compilation
    kernel_sources = []
    for fname in CL_KERNEL_FILES:
        with open(fname) as f:
            kernel_sources.append(f.read())

    build_opts = {
        'nvidia': '-cl-nv-verbose -cl-mad-enable',
        'amd': '-O2 -cl-fast-relaxed-math',
        'intel': '-cl-intel-no-simd-for-eu'
    }.get(device.vendor.lower(), '')
    build_opts += f" -D SIMD_WIDTH={simd_width}"

    program = cl.Program(ctx, "\n".join(kernel_sources)).build(build_opts)

    # Data preparation
    X_padded = pad_to_multiple(X, simd_width, axis=1)
    np_weights = safe_init((X_padded.shape[1], HIDDEN_DIM), simd_width)
    np_weights_vec = preprocess_weights(np_weights, simd_width)
    np_biases = safe_init(HIDDEN_DIM, simd_width)
    np_exit_weights = safe_init((NUM_EXITS, HIDDEN_DIM, OUTPUT_CLASSES), simd_width)
    np_exit_weights_vec = preprocess_weights(np_exit_weights, simd_width)
    np_exit_biases = safe_init((NUM_EXITS, OUTPUT_CLASSES), simd_width)

    # Buffer creation
    mf = cl.mem_flags
    buffers = {
        'input': cl.Buffer(ctx, mf.READ_ONLY|mf.COPY_HOST_PTR, hostbuf=X_padded),
        'weights': cl.Buffer(ctx, mf.READ_WRITE|mf.COPY_HOST_PTR, hostbuf=np_weights_vec),
        'biases': cl.Buffer(ctx, mf.READ_WRITE|mf.COPY_HOST_PTR, hostbuf=np_biases),
        'exit_w': cl.Buffer(ctx, mf.READ_WRITE|mf.COPY_HOST_PTR, hostbuf=np_exit_weights_vec),
        'exit_b': cl.Buffer(ctx, mf.READ_WRITE|mf.COPY_HOST_PTR, hostbuf=np_exit_biases),
        'hidden': cl.Buffer(ctx, mf.READ_WRITE, BATCH_SIZE * HIDDEN_DIM *4),
        'targets': cl.Buffer(ctx, mf.READ_ONLY|mf.COPY_HOST_PTR, hostbuf=y_true),
        'exit_probs': cl.Buffer(ctx, mf.READ_WRITE, BATCH_SIZE * NUM_EXITS * ((OUTPUT_CLASSES+3)//4*4)*4),
        'losses': cl.Buffer(ctx, mf.READ_WRITE, BATCH_SIZE * NUM_EXITS *4)
    }

    # Moment buffers initialization
    def create_moments(param):
        buf = [cl.Buffer(ctx, mf.READ_WRITE, param.nbytes) for _ in range(2)]
        for b in buf:
            cl.enqueue_fill_buffer(queue, b, np.float32(0), 0, param.nbytes)
        return buf
    moments = {
        'weights': create_moments(np_weights_vec),
        'biases': create_moments(np_biases),
        'exit_weights': create_moments(np_exit_weights_vec),
        'exit_biases': create_moments(np_exit_biases)
    }
    queue.finish()

    # Gradient buffers
    grads = {
        'weights': cl.Buffer(ctx, mf.READ_WRITE, np_weights_vec.nbytes),
        'biases': cl.Buffer(ctx, mf.READ_WRITE, np_biases.nbytes),
        'exit_weights': cl.Buffer(ctx, mf.READ_WRITE, np_exit_weights_vec.nbytes),
        'exit_biases': cl.Buffer(ctx, mf.READ_WRITE, np_exit_biases.nbytes)
    }

    # Training loop
    global_step = 1
    for epoch in range(EPOCHS):
        # Forward pass
        local_forward = cl.LocalMemory(wg_config['forward'] * simd_width * 4)
        event = program.forward_pass(queue,
            (BATCH_SIZE, HIDDEN_DIM//simd_width), (wg_config['forward'],),
            buffers['input'], buffers['weights'], buffers['biases'],
            buffers['hidden'], np.int32(X_padded.shape[1]), np.int32(HIDDEN_DIM),
            local_forward, np.int32(wg_config['forward']//simd_width)
        )
        event.wait()
        t_forward = 1e-6 * (event.profile.end - event.profile.start)

        # Early exits
        exit_events = []
        for exit_idx in range(NUM_EXITS):
            local_exit = cl.LocalMemory(wg_config['exit'] * 4)
            event = program.compute_exit_probabilities(queue, (BATCH_SIZE,), (wg_config['exit'],),
                buffers['hidden'], buffers['exit_w'], buffers['exit_b'], buffers['targets'],
                buffers['exit_probs'], buffers['losses'], np.int32(HIDDEN_DIM),
                np.int32(OUTPUT_CLASSES), np.int32(NUM_EXITS), np.int32(exit_idx),
                local_exit, np.int32((OUTPUT_CLASSES + simd_width -1)//simd_width)
            )
            exit_events.append(event)
        cl.wait_for_events(exit_events)
        t_exit = sum(1e-6*(e.profile.end-e.profile.start) for e in exit_events)

        # Backpropagation
        for buf in grads.values():
            cl.enqueue_fill_buffer(queue, buf, np.float32(0), 0, buf.size)

        local_grad = cl.LocalMemory(wg_config['gradients'][0] * wg_config['gradients'][1] * 16)
        local_db = cl.LocalMemory(wg_config['gradients'][1] * 4)
        event = program.compute_gradients(queue,
            (X_padded.shape[1]//simd_width, HIDDEN_DIM), wg_config['gradients'],
            buffers['input'], buffers['hidden'], buffers['exit_probs'], buffers['targets'],
            buffers['exit_w'], grads['weights'], grads['biases'],
            local_grad, local_db, np.int32(X_padded.shape[1]),
            np.int32(HIDDEN_DIM), np.int32(OUTPUT_CLASSES), np.int32(NUM_EXITS)
        )
        event.wait()
        t_backward = 1e-6 * (event.profile.end - event.profile.start)

        # Adam updates
        def run_adam(grad_buf, param_buf, m1, m2):
            vector_stride = (param_buf.size//4) // wg_config['adam']
            beta1_t = 1/(1 - 0.9**global_step)
            beta2_t = 1/(1 - 0.999**global_step)
            event = program.adam_update(queue, (wg_config['adam'],), None,
                grad_buf, param_buf, m1, m2, np.float32(0.001),
                np.float32(0.9), np.float32(0.999), np.float32(beta1_t),
                np.float32(beta2_t), np.int32(param_buf.size//4),
                np.int32(vector_stride)
            )
            return event

        adam_events = [
            run_adam(grads['weights'], buffers['weights'], *moments['weights']),
            run_adam(grads['biases'], buffers['biases'], *moments['biases']),
            run_adam(grads['exit_weights'], buffers['exit_w'], *moments['exit_weights']),
            run_adam(grads['exit_biases'], buffers['exit_b'], *moments['exit_biases'])
        ]
        cl.wait_for_events(adam_events)
        t_adam = sum(1e-6*(e.profile.end-e.profile.start) for e in adam_events)

        # Validation & monitoring
        if epoch % 10 == 0 or epoch == EPOCHS-1:
            # Copy exit probabilities for validation
            exit_probs = np.empty((BATCH_SIZE, NUM_EXITS, OUTPUT_CLASSES), np.float32)
            cl.enqueue_copy(queue, exit_probs, buffers['exit_probs'])

            # Ensemble predictions
            weights = np.array([0.4, 0.3, 0.3])
            ensemble = np.average(exit_probs, axis=1, weights=weights)
            preds = np.argmax(ensemble, axis=1)
            acc = np.mean(preds == y_true[:BATCH_SIZE])

            # Loss calculation
            losses = np.empty((BATCH_SIZE, NUM_EXITS), np.float32)
            cl.enqueue_copy(queue, losses, buffers['losses'])
            avg_loss = np.mean(losses)

            print(f"Epoch {epoch:3d} | Acc: {acc:.1%} | Loss: {avg_loss:.3f} | "
                  f"Time: {t_forward:.1f}+{t_exit:.1f}+{t_backward:.1f}+{t_adam:.1f}ms")

        global_step += 1

if __name__ == "__main__":
    main()
