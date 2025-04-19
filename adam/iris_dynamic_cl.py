# iris_dynamic_cl.py
import pyopencl as cl
import numpy as np
from sklearn.datasets import load_iris
from sklearn.preprocessing import OneHotEncoder

# Network Configuration
INPUT_DIM = 4
HIDDEN_DIM = 8    # Must be multiple of 4
OUTPUT_CLASSES = 3
NUM_EXITS = 3
EPOCHS = 100
BATCH_SIZE = 150
WORKGROUP_SIZE = 64  # Must be 64 or 128

# Adam Configuration
ADAM_WORKGROUP_SIZE = 256
ADAM_BETA1 = 0.9
ADAM_BETA2 = 0.999

CL_KERNEL_FILES = [
    'kernel_forward_pass.cl',
    'kernel_backpropagation.cl',
    'kernel_adam_optimizer.cl',
    'kernel_multi_exit.cl'
]

def pad_to_multiple(arr, multiple):
    pad_size = (-arr.shape[-1]) % multiple
    if pad_size == 0:
        return arr
    new_shape = list(arr.shape)
    new_shape[-1] += pad_size
    padded = np.zeros(new_shape, dtype=arr.dtype)
    padded[..., :arr.shape[-1]] = arr
    return padded

def preprocess_weights(w):
    """Reshape weights for vectorized OpenCL access"""
    if w.ndim == 2:  # Main weights [IN, HID]
        return w.T.reshape(HIDDEN_DIM, INPUT_DIM//4, 4)
    elif w.ndim == 3:  # Exit weights [EXIT, HID, OUT]
        return w.transpose(0, 2, 1).reshape(NUM_EXITS, OUTPUT_CLASSES, HIDDEN_DIM//4, 4)
    raise ValueError(f"Unsupported weight dim {w.ndim}")

def safe_init(shape, std=0.1):
    arr = np.random.normal(0, std, shape).astype(np.float32)
    if arr.ndim >= 2:  # Only pad weight matrices
        return pad_to_multiple(arr, 4)
    return arr

def main():
    iris = load_iris()
    X = pad_to_multiple(iris.data.astype(np.float32), 4)
    y_true = iris.target.astype(np.int32)
    y_onehot = OneHotEncoder(sparse_output=False).fit_transform(y_true.reshape(-1,1))

    # OpenCL Setup
    ctx = cl.create_some_context()
    queue = cl.CommandQueue(ctx)
    val_queue = cl.CommandQueue(ctx)

    # Load kernels
    kernel_sources = []
    for file in CL_KERNEL_FILES:
        with open(file) as f:
            kernel_sources.append(f.read())

    build_opts = [
        f"-D BATCH_SIZE={BATCH_SIZE}",
        f"-D WORKGROUP_SIZE={WORKGROUP_SIZE}",
        "-cl-mad-enable -cl-fast-relaxed-math"
    ]
    program = cl.Program(ctx, "\n".join(kernel_sources)).build(" ".join(build_opts))
    mf = cl.mem_flags

    # Parameter Initialization
    # Main network weights [IN, HID]
    np_weights = safe_init((INPUT_DIM, HIDDEN_DIM))
    np_weights = preprocess_weights(np_weights)  # [HID, IN/4, 4]
    np_biases = safe_init(HIDDEN_DIM)

    # Exit layers weights [EXITS, HID, OUT]
    np_exit_weights = safe_init((NUM_EXITS, HIDDEN_DIM, OUTPUT_CLASSES))
    np_exit_weights = preprocess_weights(np_exit_weights)  # [EXITS, OUT, HID/4,4]
    np_exit_biases = safe_init((NUM_EXITS, OUTPUT_CLASSES))

    # Validation checks
    assert HIDDEN_DIM % 4 == 0, "HIDDEN_DIM not SIMD-aligned"
    assert np_weights.shape == (HIDDEN_DIM, INPUT_DIM//4, 4), \
        f"Main weights shape invalid: {np_weights.shape}"
    assert np_exit_weights.shape == (NUM_EXITS, OUTPUT_CLASSES, HIDDEN_DIM//4,4), \
        f"Exit weights shape invalid: {np_exit_weights.shape}"

    # Create buffers
    cl_weights = cl.Buffer(ctx, mf.READ_WRITE | mf.COPY_HOST_PTR, hostbuf=np_weights)
    cl_biases = cl.Buffer(ctx, mf.READ_WRITE | mf.COPY_HOST_PTR, hostbuf=np_biases)
    cl_exit_weights = cl.Buffer(ctx, mf.READ_WRITE | mf.COPY_HOST_PTR, hostbuf=np_exit_weights)
    cl_exit_biases = cl.Buffer(ctx, mf.READ_WRITE | mf.COPY_HOST_PTR, hostbuf=np_exit_biases)

    # Training buffers
    cl_hidden = cl.Buffer(ctx, mf.READ_WRITE, BATCH_SIZE*HIDDEN_DIM*4)
    cl_X = cl.Buffer(ctx, mf.READ_ONLY | mf.COPY_HOST_PTR, hostbuf=X)
    cl_y = cl.Buffer(ctx, mf.READ_ONLY | mf.COPY_HOST_PTR, hostbuf=y_true)
    cl_all_exits = cl.Buffer(ctx, mf.READ_WRITE, BATCH_SIZE*NUM_EXITS*OUTPUT_CLASSES*4)
    cl_loss = cl.Buffer(ctx, mf.READ_WRITE, BATCH_SIZE*NUM_EXITS*4)

    # Moment buffers
    def create_moments(param):
        return (
            cl.Buffer(ctx, mf.READ_WRITE, param.nbytes),
            cl.Buffer(ctx, mf.READ_WRITE, param.nbytes)
        )
    hidden_w_m1, hidden_w_m2 = create_moments(np_weights)
    hidden_b_m1, hidden_b_m2 = create_moments(np_biases)
    exit_w_m1, exit_w_m2 = create_moments(np_exit_weights)
    exit_b_m1, exit_b_m2 = create_moments(np_exit_biases)

    # Gradient buffers
    cl_dW = cl.Buffer(ctx, mf.READ_WRITE, np_weights.nbytes)
    cl_db = cl.Buffer(ctx, mf.READ_WRITE, np_biases.nbytes)
    cl_exit_dW = cl.Buffer(ctx, mf.READ_WRITE, np_exit_weights.nbytes)
    cl_exit_db = cl.Buffer(ctx, mf.READ_WRITE, np_exit_biases.nbytes)

    learning_rate = 0.001
    global_step = 0

    for epoch in range(EPOCHS):
        # Forward Pass
        global_x = ((BATCH_SIZE + WORKGROUP_SIZE-1) // WORKGROUP_SIZE) * WORKGROUP_SIZE
        global_size = (global_x, HIDDEN_DIM//4)

        program.forward_pass(
            queue, global_size, (WORKGROUP_SIZE, 1),
            cl_X, cl_weights, cl_biases, cl_hidden,
            np.int32(INPUT_DIM), np.int32(HIDDEN_DIM)
        )

        # Exit Probability Computation
        events = []
        for exit_id in range(NUM_EXITS):
            event = program.compute_exit_probabilities(
                queue, (BATCH_SIZE,), None,
                cl_hidden, cl_exit_weights, cl_exit_biases,
                cl_all_exits, cl_loss, cl_y,
                np.int32(HIDDEN_DIM), np.int32(OUTPUT_CLASSES),
                np.int32(NUM_EXITS), np.int32(exit_id)
            )
            events.append(event)
        cl.wait_for_events(events)

        # Backpropagation
        cl.enqueue_fill_buffer(queue, cl_dW, np.float32(0), 0, np_weights.nbytes)
        cl.enqueue_fill_buffer(queue, cl_db, np.float32(0), 0, np_biases.nbytes)
        cl.enqueue_fill_buffer(queue, cl_exit_dW, np.float32(0), 0, np_exit_weights.nbytes)
        cl.enqueue_fill_buffer(queue, cl_exit_db, np.float32(0), 0, np_exit_biases.nbytes)
        queue.finish()

        # Backprop Kernel
        program.compute_backprop_gradients(
            queue,
            ((INPUT_DIM + 31)//32 * 32, HIDDEN_DIM),
            (32, 8),
            cl_X, cl_hidden, cl_all_exits, cl_y, cl_exit_weights,
            cl_dW, cl_db, cl_exit_dW, cl_exit_db,
            np.int32(INPUT_DIM), np.int32(HIDDEN_DIM),
            np.int32(np_exit_weights.shape[-2]*4),  # Account for padding
            np.int32(OUTPUT_CLASSES), np.int32(NUM_EXITS)
        )

        # Adam Updates
        def run_adam(buffer, grad, m1, m2, param_array):
            t = global_step + 1
            b1_t = 1/(1 - ADAM_BETA1**t)
            b2_t = 1/(1 - ADAM_BETA2**t)

            program.adam_optimize(
                queue, (param_array.size//4,), (ADAM_WORKGROUP_SIZE,),
                grad, buffer, m1, m2, np.float32(learning_rate),
                np.float32(ADAM_BETA1), np.float32(ADAM_BETA2),
                np.float32(b1_t), np.float32(b2_t),
                np.int32(param_array.size)
            )

        run_adam(cl_weights, cl_dW, hidden_w_m1, hidden_w_m2, np_weights)
        run_adam(cl_biases, cl_db, hidden_b_m1, hidden_b_m2, np_biases)
        run_adam(cl_exit_weights, cl_exit_dW, exit_w_m1, exit_w_m2, np_exit_weights)
        run_adam(cl_exit_biases, cl_exit_db, exit_b_m1, exit_b_m2, np_exit_biases)
        global_step += 1

        # Validation
        if epoch % 10 == 0:
            # Check numerical stability
            check_weights = np.empty_like(np_weights)
            cl.enqueue_copy(queue, check_weights, cl_weights)
            if not np.all(np.isfinite(check_weights)):
                print(f"Numerical instability at epoch {epoch}")
                break

            # Accuracy computation
            exit_probs = np.empty((BATCH_SIZE, NUM_EXITS, OUTPUT_CLASSES), np.float32)
            cl.enqueue_copy(queue, exit_probs, cl_all_exits)

            weights = np.array([0.4, 0.3, 0.3])  # Learned weights would be better
            ensemble = np.average(exit_probs, axis=1, weights=weights)
            acc = np.mean(np.argmax(ensemble, 1) == y_true[:BATCH_SIZE])

            # Loss retrieval
            losses = np.empty((BATCH_SIZE, NUM_EXITS), np.float32)
            cl.enqueue_copy(queue, losses, cl_loss)
            avg_losses = np.mean(losses, 0)

            print(f"Epoch {epoch:3d} | Acc: {acc:.1%} | Losses: {avg_losses.round(3)}")

if __name__ == "__main__":
    main()
