# iris_dynamic_cl.py

import pyopencl as cl
import numpy as np
from sklearn.datasets import load_iris
from sklearn.preprocessing import OneHotEncoder

# Network Configuration
INPUT_DIM = 4
HIDDEN_DIM = 8    # Must be multiple of 4 for vectorization
OUTPUT_CLASSES = 3
NUM_EXITS = 3
EPOCHS = 100
BATCH_SIZE = 150
WORKGROUP_SIZE = 64
ADAM_WORKGROUP_SIZE = 256

# Kernel file list
CL_KERNEL_FILES = [
    'kernel_forward_pass.cl',
    'kernel_backpropagation.cl',
    'kernel_adam_optimizer.cl',
    'kernel_multi_exit.cl'
]

def pad_to_multiple(arr, multiple):
    """Pad array to be multiple of given size along last dimension"""
    pad_size = (-arr.shape[-1]) % multiple
    if pad_size == 0:
        return arr
    new_shape = list(arr.shape)
    new_shape[-1] += pad_size
    padded = np.zeros(new_shape, dtype=arr.dtype)
    padded[..., :arr.shape[-1]] = arr
    return padded

def main():
    # Data Preparation
    iris = load_iris()
    X = iris.data.astype(np.float32)
    y_true = iris.target.astype(np.int32)
    y_onehot = OneHotEncoder(sparse_output=False).fit_transform(y_true.reshape(-1,1))

    # OpenCL Setup
    ctx = cl.create_some_context()
    queue = cl.CommandQueue(ctx)
    val_queue = cl.CommandQueue(ctx)

    # Load and compile kernels
    kernel_sources = []
    for file in CL_KERNEL_FILES:
        with open(file) as f:
            kernel_sources.append(f.read())
    program = cl.Program(ctx, "\n".join(kernel_sources)).build()

    mf = cl.mem_flags

    # Parameter Initialization (ensure 4-byte alignment)
    def aligned_shape(shape, elements_per_float4=4):
        total = np.prod(shape)
        return total + (-total % elements_per_float4)

    # Hidden Layer Parameters (Input -> Hidden)
    np_weights = np.random.normal(0, 0.1, (INPUT_DIM, HIDDEN_DIM)).astype(np.float32)
    np_biases = np.zeros(HIDDEN_DIM, dtype=np.float32)

    # Exit Layer Parameters (pad output dimension for vectorization)
    np_exit_weights = np.random.normal(0, 0.01, (NUM_EXITS, HIDDEN_DIM, OUTPUT_CLASSES)).astype(np.float32)
    np_exit_biases = pad_to_multiple(np.zeros((NUM_EXITS, OUTPUT_CLASSES), dtype=np.float32), 4)

    # Create OpenCL buffers
    cl_weights = cl.Buffer(ctx, mf.READ_WRITE | mf.COPY_HOST_PTR, hostbuf=np_weights)
    cl_biases = cl.Buffer(ctx, mf.READ_WRITE | mf.COPY_HOST_PTR, hostbuf=np_biases)
    cl_exit_weights = cl.Buffer(ctx, mf.READ_WRITE | mf.COPY_HOST_PTR, hostbuf=np_exit_weights)
    cl_exit_biases = cl.Buffer(ctx, mf.READ_WRITE | mf.COPY_HOST_PTR, hostbuf=np_exit_biases)

    # Momentum buffers with aligned sizes
    def create_moment_buffers(arr):
        padded_size = (-arr.size % (ADAM_WORKGROUP_SIZE * 4)) + arr.size
        return (
            cl.Buffer(ctx, mf.READ_WRITE, size=padded_size*4),
            cl.Buffer(ctx, mf.READ_WRITE, size=padded_size*4)
        )

    hidden_w_m1, hidden_w_m2 = create_moment_buffers(np_weights)
    hidden_b_m1, hidden_b_m2 = create_moment_buffers(np_biases)
    exit_w_m1, exit_w_m2 = create_moment_buffers(np_exit_weights)
    exit_b_m1, exit_b_m2 = create_moment_buffers(np_exit_biases)

    # Training buffers
    cl_hidden = cl.Buffer(ctx, mf.READ_WRITE, size=BATCH_SIZE*HIDDEN_DIM*4)
    cl_dW = cl.Buffer(ctx, mf.READ_WRITE, size=np_weights.nbytes)
    cl_db = cl.Buffer(ctx, mf.READ_WRITE, size=np_biases.nbytes)
    cl_exit_dW = cl.Buffer(ctx, mf.READ_WRITE, size=np_exit_weights.nbytes)
    cl_exit_db = cl.Buffer(ctx, mf.READ_WRITE, size=np_exit_biases.nbytes)

    # Data buffers
    cl_X = cl.Buffer(ctx, mf.READ_ONLY | mf.COPY_HOST_PTR, hostbuf=X)
    cl_y = cl.Buffer(ctx, mf.READ_ONLY | mf.COPY_HOST_PTR, hostbuf=y_true)
    cl_all_exits = cl.Buffer(ctx, mf.READ_WRITE, size=BATCH_SIZE*NUM_EXITS*4*4)  # padded float4
    cl_loss = cl.Buffer(ctx, mf.READ_WRITE, size=BATCH_SIZE*NUM_EXITS*4)

    # Training parameters
    learning_rate = 0.001
    global_step = 0
    beta1, beta2 = 0.9, 0.999

    for epoch in range(EPOCHS):
        # 1. Forward Pass --------------------------------------------------------
        global_size = (HIDDEN_DIM, (BATCH_SIZE + WORKGROUP_SIZE-1) // WORKGROUP_SIZE)
        program.forward_pass(
            queue, global_size, (1, WORKGROUP_SIZE),
            cl_X, cl_weights, cl_biases, cl_hidden,
            np.int32(INPUT_DIM), np.int32(HIDDEN_DIM)
        )

        # 2. Compute Exit Probabilities ------------------------------------------
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

        # 3. Backpropagation -----------------------------------------------------
        local_mem_size = 2 * WORKGROUP_SIZE * 4  # float cache for 2 variables
        program.compute_backprop_gradients(
            queue, (HIDDEN_DIM,), (WORKGROUP_SIZE,),
            cl_X, cl_hidden, cl_all_exits, cl_y, cl_exit_weights,
            cl_dW, cl_db, cl_exit_dW, cl_exit_db,
            np.int32(INPUT_DIM), np.int32(HIDDEN_DIM),
            np.int32(OUTPUT_CLASSES), np.int32(NUM_EXITS),
            cl.LocalMemory(local_mem_size)
        )

        # 4. Adam Updates --------------------------------------------------------
        def run_adam(buffer, grad_buffer, m1, m2, size):
            inv_b1 = 1.0 / (1.0 - (beta1 ** (global_step + 1)))
            inv_b2 = 1.0 / (1.0 - (beta2 ** (global_step + 1)))

            program.adam_optimize(
                queue, (size // 4 + ADAM_WORKGROUP_SIZE -1,), (ADAM_WORKGROUP_SIZE,),
                grad_buffer, buffer, m1, m2,
                np.float32(learning_rate),
                np.float32(inv_b1),
                np.float32(inv_b2),
                np.int32(size)
            )

        # Update each parameter group
        run_adam(cl_weights, cl_dW, hidden_w_m1, hidden_w_m2, np_weights.size)
        run_adam(cl_biases, cl_db, hidden_b_m1, hidden_b_m2, np_biases.size)
        run_adam(cl_exit_weights, cl_exit_dW, exit_w_m1, exit_w_m2, np_exit_weights.size)
        run_adam(cl_exit_biases, cl_exit_db, exit_b_m1, exit_b_m2, np_exit_biases.size)

        global_step += 1

        # 5. Validation ----------------------------------------------------------
        if epoch % 10 == 0:
            exit_probs = np.empty((BATCH_SIZE, NUM_EXITS, 4), dtype=np.float32)  # padded
            exit_losses = np.empty((BATCH_SIZE, NUM_EXITS), dtype=np.float32)

            cl.enqueue_copy(val_queue, exit_probs, cl_all_exits)
            cl.enqueue_copy(val_queue, exit_losses, cl_loss)

            # Remove padding from exit probabilities
            valid_probs = exit_probs[..., :OUTPUT_CLASSES]
            ensemble_probs = np.mean(valid_probs, axis=1)
            pred_labels = np.argmax(ensemble_probs, axis=1)
            accuracy = np.mean(pred_labels == y_true[:BATCH_SIZE])
            avg_losses = np.mean(exit_losses, axis=0)

            print(f"Epoch {epoch:3d} | Accuracy: {accuracy:.1%}")
            print(f"  Exit Losses: {avg_losses.round(3)}")

if __name__ == "__main__":
    main()
