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
    'kernel_adam_update.cl',
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
    """Reshape weights for vectorized OpenCL access with SIMD padding"""
    if w.ndim == 2:  # Main weights [INPUT_DIM, HIDDEN_DIM]
        return w.T.reshape(HIDDEN_DIM//4, INPUT_DIM//4, 4)

    elif w.ndim == 3:  # Exit weights [NUM_EXITS, HIDDEN_DIM, OUTPUT_CLASSES]
        # Pad class dimension to multiple of 4 and reshape for vectorization
        w_padded = pad_to_multiple(w, 4, axis=-1)  # Pad class dimension
        return w_padded.transpose(0, 2, 1).reshape(
            NUM_EXITS,
            w_padded.shape[-1],  # Padded classes dimension
            HIDDEN_DIM//4,
            4
        )

    raise ValueError(f"Unsupported weight dim {w.ndim}")

def safe_init(shape, std=0.1):
    arr = np.random.normal(0, std, shape).astype(np.float32)
    # All 2D+ tensors get last dim padded to multiples of 4
    if arr.ndim >= 2:
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

    # Parameter Initialization -------------------------------------------------
    # Main network parameters
    np_weights_vec4 = preprocess_weights(safe_init((INPUT_DIM, HIDDEN_DIM)))
    np_biases = safe_init(HIDDEN_DIM)

    # Exit layer parameters
    np_exit_weights_vec4 = preprocess_weights(
        safe_init((NUM_EXITS, HIDDEN_DIM, OUTPUT_CLASSES))
    )
    np_exit_biases = safe_init((NUM_EXITS, OUTPUT_CLASSES))

    # Validation checks
    assert HIDDEN_DIM % 4 == 0, "HIDDEN_DIM not SIMD-aligned"
    assert np_weights_vec4.shape == (HIDDEN_DIM//4, INPUT_DIM//4, 4), \
        f"Main weights shape invalid: {np_weights_vec4.shape}"
    assert np_exit_weights_vec4.shape == (NUM_EXITS, OUTPUT_CLASSES, HIDDEN_DIM//4, 4), \
        f"Exit weights shape invalid: {np_exit_weights_vec4.shape}"
    assert np_exit_biases.shape == (NUM_EXITS, OUTPUT_CLASSES), \
        f"Exit biases shape invalid: {np_exit_biases.shape}"

    # Create OpenCL Buffers ----------------------------------------------------
    cl_input_vec4 = cl.Buffer(ctx, mf.READ_ONLY | mf.COPY_HOST_PTR,
                            hostbuf=X.reshape(BATCH_SIZE, INPUT_DIM//4, 4))
    cl_weights_vec4 = cl.Buffer(ctx, mf.READ_WRITE | mf.COPY_HOST_PTR,
                              hostbuf=np_weights_vec4)
    cl_biases = cl.Buffer(ctx, mf.READ_WRITE | mf.COPY_HOST_PTR, hostbuf=np_biases)
    cl_exit_weights_vec4 = cl.Buffer(ctx, mf.READ_WRITE | mf.COPY_HOST_PTR,
                                   hostbuf=np_exit_weights_vec4)
    cl_exit_biases = cl.Buffer(ctx, mf.READ_WRITE | mf.COPY_HOST_PTR,
                             hostbuf=np_exit_biases)

    # Training buffers
    cl_hidden_vec4 = cl.Buffer(ctx, mf.READ_WRITE,
                             BATCH_SIZE * (HIDDEN_DIM//4) * 16)  # float4: 4 elements * 4 bytes each * 4 bytes
    cl_targets = cl.Buffer(ctx, mf.READ_ONLY | mf.COPY_HOST_PTR, hostbuf=y_true)
    num_classes_padded = ((OUTPUT_CLASSES + 3) // 4) * 4
    cl_exit_probs = cl.Buffer(
        ctx, mf.READ_WRITE,
        BATCH_SIZE * NUM_EXITS * num_classes_padded * 4
    )
    cl_losses = cl.Buffer(ctx, mf.READ_WRITE, BATCH_SIZE*NUM_EXITS*4)

    # Moment buffers
    def create_moments_vec4(param_vec4):
        return (
            cl.Buffer(ctx, mf.READ_WRITE, param_vec4.nbytes),
            cl.Buffer(ctx, mf.READ_WRITE, param_vec4.nbytes)
        )
    hidden_w_m1, hidden_w_m2 = create_moments_vec4(np_weights_vec4)
    hidden_b_m1, hidden_b_m2 = create_moments_vec4(np_biases)
    exit_w_m1, exit_w_m2 = create_moments_vec4(np_exit_weights_vec4)
    exit_b_m1, exit_b_m2 = create_moments_vec4(np_exit_biases)

    # Gradient buffers
    cl_grad_weights_vec4 = cl.Buffer(ctx, mf.READ_WRITE, np_weights_vec4.nbytes)
    cl_grad_biases = cl.Buffer(ctx, mf.READ_WRITE, np_biases.nbytes)
    cl_grad_exit_weights_vec4 = cl.Buffer(ctx, mf.READ_WRITE, np_exit_weights_vec4.nbytes)
    cl_grad_exit_biases = cl.Buffer(ctx, mf.READ_WRITE, np_exit_biases.nbytes)

    learning_rate = 0.001
    global_step = 0

    # Training Loop ------------------------------------------------------------
    for epoch in range(EPOCHS):
        # Forward Pass
        program.forward_pass(
            queue,
            ((BATCH_SIZE + WORKGROUP_SIZE-1)//WORKGROUP_SIZE*WORKGROUP_SIZE, HIDDEN_DIM//4),
            (WORKGROUP_SIZE, 1),
            cl_input_vec4,
            cl_weights_vec4,
            cl_biases,
            cl_hidden_vec4,
            np.int32(INPUT_DIM),
            np.int32(HIDDEN_DIM)
        )

        # Multi-Exit Processing
        events = []
        for exit_idx in range(NUM_EXITS):
            event = program.compute_exit_probabilities(
                queue, (BATCH_SIZE,), None,
                cl_hidden_vec4,
                cl_exit_weights_vec4,
                cl_exit_biases,
                cl_exit_probs,
                cl_losses,
                cl_targets,
                np.int32(HIDDEN_DIM),
                np.int32(OUTPUT_CLASSES),
                np.int32(NUM_EXITS),
                np.int32(exit_idx)
            )
            events.append(event)
        cl.wait_for_events(events)

        # Backpropagation Setup
        cl.enqueue_fill_buffer(queue, cl_grad_weights_vec4, np.float32(0), 0, np_weights_vec4.nbytes)
        cl.enqueue_fill_buffer(queue, cl_grad_biases, np.float32(0), 0, np_biases.nbytes)
        cl.enqueue_fill_buffer(queue, cl_grad_exit_weights_vec4, np.float32(0), 0, np_exit_weights_vec4.nbytes)
        cl.enqueue_fill_buffer(queue, cl_grad_exit_biases, np.float32(0), 0, np_exit_biases.nbytes)
        queue.finish()

        # Compute Gradients
        program.compute_gradients(
            queue,
            ((INPUT_DIM + 31)//32 * 32, HIDDEN_DIM),
            (32, 8),
            cl_input_vec4,
            cl_hidden_vec4,
            cl_exit_probs,
            cl_targets,
            cl_exit_weights_vec4,
            cl_grad_weights_vec4,
            cl_grad_biases,
            np.int32(INPUT_DIM),
            np.int32(HIDDEN_DIM),
            np.int32(OUTPUT_CLASSES),
            np.int32(NUM_EXITS)
        )

        # Adam Updates for All Parameters
        def run_adam(param_vec4, grad_vec4, m1, m2):
            t = global_step + 1
            b1_t = 1/(1 - ADAM_BETA1**t)
            b2_t = 1/(1 - ADAM_BETA2**t)

            program.adam_update(
                queue,
                (param_vec4.size//4,),
                (ADAM_WORKGROUP_SIZE,),
                grad_vec4,
                param_vec4,
                m1,
                m2,
                np.float32(learning_rate),
                np.float32(ADAM_BETA1),
                np.float32(ADAM_BETA2),
                np.float32(b1_t),
                np.float32(b2_t),
                np.int32(param_vec4.size)
            )

        run_adam(cl_weights_vec4, cl_grad_weights_vec4, hidden_w_m1, hidden_w_m2)
        run_adam(cl_biases, cl_grad_biases, hidden_b_m1, hidden_b_m2)
        run_adam(cl_exit_weights_vec4, cl_grad_exit_weights_vec4, exit_w_m1, exit_w_m2)
        run_adam(cl_exit_biases, cl_grad_exit_biases, exit_b_m1, exit_b_m2)

        global_step += 1

        # Validation
        if epoch % 10 == 0:
            check_weights = np.empty_like(np_weights_vec4)
            cl.enqueue_copy(queue, check_weights, cl_weights_vec4)

            exit_probs = np.empty((BATCH_SIZE, NUM_EXITS, OUTPUT_CLASSES), np.float32)
            cl.enqueue_copy(queue, exit_probs, cl_exit_probs)

            ensemble = np.average(exit_probs, axis=1, weights=[0.4, 0.3, 0.3])
            acc = np.mean(np.argmax(ensemble, 1) == y_true[:BATCH_SIZE])

            print(f"Epoch {epoch:3d} | Acc: {acc:.1%}")

if __name__ == "__main__":
    main()
