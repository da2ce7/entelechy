# iris_dynamic_cl.py

import pyopencl as cl
import numpy as np
from sklearn.datasets import load_iris
from sklearn.preprocessing import OneHotEncoder

# Network configuration
INPUT_DIM = 4
HIDDEN_DIM = 8
OUTPUT_CLASSES = 3
NUM_EXITS = 3
EPOCHS = 100
BATCH_SIZE = 150

# Kernel files list
CL_KERNEL_FILES = [
    'kernel_forward_pass.cl',
    'kernel_backpropagation.cl',
    'kernel_adam_optimizer.cl',
    'kernel_apply_updates.cl',
    'kernel_multi_exit.cl'
]

def main():
    # Data preparation
    iris = load_iris()
    X = iris.data.astype(np.float32)
    y_true = iris.target.astype(np.int32)
    y_onehot = OneHotEncoder(sparse_output=False).fit_transform(y_true.reshape(-1,1))

    # OpenCL initialization
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

    # ---- Parameter initialization ----
    # Hidden layer parameters
    np_weights = np.random.normal(0, 0.1, (INPUT_DIM, HIDDEN_DIM)).astype(np.float32)
    np_biases = np.zeros(HIDDEN_DIM, dtype=np.float32)

    # Exit layer parameters
    np_exit_weights = np.random.normal(0, 0.01, (NUM_EXITS, HIDDEN_DIM, OUTPUT_CLASSES)).astype(np.float32)
    np_exit_biases = np.zeros((NUM_EXITS, OUTPUT_CLASSES), dtype=np.float32)

    # Create OpenCL buffers
    cl_weights = cl.Buffer(ctx, mf.READ_WRITE | mf.COPY_HOST_PTR, hostbuf=np_weights)
    cl_biases = cl.Buffer(ctx, mf.READ_WRITE | mf.COPY_HOST_PTR, hostbuf=np_biases)
    cl_exit_weights = cl.Buffer(ctx, mf.READ_WRITE | mf.COPY_HOST_PTR, hostbuf=np_exit_weights)
    cl_exit_biases = cl.Buffer(ctx, mf.READ_WRITE | mf.COPY_HOST_PTR, hostbuf=np_exit_biases)

    # Momentum buffers (Adam)
    def create_momentum_buffers(shape):
        return [
            cl.Buffer(ctx, mf.READ_WRITE, size=np.prod(shape)*4),
            cl.Buffer(ctx, mf.READ_WRITE, size=np.prod(shape)*4)
        ]

    # Hidden layer momentum
    hidden_m1, hidden_m2 = create_momentum_buffers(np_weights.shape)
    hidden_bias_m1, hidden_bias_m2 = create_momentum_buffers((HIDDEN_DIM,))

    # Exit layer momentum
    exit_weights_m1, exit_weights_m2 = create_momentums(NUM_EXITS*HIDDEN_DIM*OUTPUT_CLASSES)
    exit_biases_m1, exit_biases_m2 = create_momentums(NUM_EXITS*OUTPUT_CLASSES)

    # Training buffers
    cl_hidden = cl.Buffer(ctx, mf.READ_WRITE, size=BATCH_SIZE*HIDDEN_DIM*4)
    cl_dW = cl.Buffer(ctx, mf.READ_WRITE, size=np_weights.nbytes)
    cl_db = cl.Buffer(ctx, mf.READ_WRITE, size=np_biases.nbytes)
    cl_exit_dW = cl.Buffer(ctx, mf.READ_WRITE, size=np_exit_weights.nbytes)
    cl_exit_db = cl.Buffer(ctx, mf.READ_WRITE, size=np_exit_biases.nbytes)

    # Data buffers
    cl_X = cl.Buffer(ctx, mf.READ_ONLY | mf.COPY_HOST_PTR, hostbuf=X)
    cl_y = cl.Buffer(ctx, mf.READ_ONLY | mf.COPY_HOST_PTR, hostbuf=y_true)
    cl_all_exits = cl.Buffer(ctx, mf.READ_WRITE, size=BATCH_SIZE*NUM_EXITS*OUTPUT_CLASSES*4)
    cl_loss = cl.Buffer(ctx, mf.READ_WRITE, size=BATCH_SIZE*NUM_EXITS*4)

    # Training parameters
    learning_rate = 0.001
    global_step = 0

    for epoch in range(EPOCHS):
        # 1. Forward pass
        program.forward_pass(
            queue, (BATCH_SIZE, HIDDEN_DIM), None,
            cl_X,
            cl_weights,
            cl_biases,
            cl_hidden,
            cl.Buffer(ctx, mf.READ_WRITE, size=BATCH_SIZE*HIDDEN_DIM*4),
            np.int32(INPUT_DIM),
            np.int32(HIDDEN_DIM)
        )

        # 2. Compute exit probabilities
        events = []
        for exit_id in range(NUM_EXITS):
            event = program.compute_exit_probabilities(
                queue, (BATCH_SIZE,), None,
                cl_hidden,
                cl_exit_weights,
                cl_exit_biases,
                cl_all_exits,
                cl_loss,
                cl_y,
                np.int32(HIDDEN_DIM),
                np.int32(OUTPUT_CLASSES),
                np.int32(NUM_EXITS),
                np.int32(exit_id)
            )
            events.append(event)
        cl.wait_for_events(events)

        # 3. Backpropagation
        program.compute_backprop_gradients(
            queue, (BATCH_SIZE, HIDDEN_DIM), None,
            cl_X,
            cl_hidden,
            cl_all_exits,
            cl_y,
            cl_exit_weights,
            cl_dW,
            cl_db,
            cl_exit_dW,
            cl_exit_db,
            None,
            np.int32(INPUT_DIM),
            np.int32(HIDDEN_DIM),
            np.int32(OUTPUT_CLASSES),
            np.int32(NUM_EXITS)
        )

        # 4. Parameter updates
        # Hidden layer updates
        program.adam_optimize(
            queue, (int(np.ceil(np_weights.size/64))*64,), (64,),
            cl_dW,
            cl_weights,
            hidden_m1,
            hidden_m2,
            np.float32(learning_rate),
            np.int32(global_step),
            np.int32(np_weights.size)
        )

        program.adam_optimize(
            queue, (int(np.ceil(HIDDEN_DIM/64))*64,), (64,),
            cl_db,
            cl_biases,
            hidden_bias_m1,
            hidden_bias_m2,
            np.float32(learning_rate),
            np.int32(global_step),
            np.int32(HIDDEN_DIM)
        )

        # Exit layer updates
        program.adam_optimize(
            queue, (int(np.ceil(NUM_EXITS*HIDDEN_DIM*OUTPUT_CLASSES/64))*64,), (64,),
            cl_exit_dW,
            cl_exit_weights,
            exit_weights_m1,
            exit_weights_m2,
            np.float32(learning_rate),
            np.int32(global_step),
            np.int32(NUM_EXITS*HIDDEN_DIM*OUTPUT_CLASSES)
        )

        program.adam_optimize(
            queue, (int(np.ceil(NUM_EXITS*OUTPUT_CLASSES/64))*64,), (64,),
            cl_exit_db,
            cl_exit_biases,
            exit_biases_m1,
            exit_biases_m2,
            np.float32(learning_rate),
            np.int32(global_step),
            np.int32(NUM_EXITS*OUTPUT_CLASSES)
        )

        global_step += 1

        # 5. Validation
        if epoch % 10 == 0:
            # Async data transfers for validation metrics
            exit_probs = np.empty((BATCH_SIZE, NUM_EXITS, OUTPUT_CLASSES), dtype=np.float32)
            exit_losses = np.empty((BATCH_SIZE, NUM_EXITS), dtype=np.float32)

            # Initiate async copies in parallel
            copy_events = [
                cl.enqueue_copy(val_queue, exit_probs, cl_all_exits, is_blocking=False),
                cl.enqueue_copy(val_queue, exit_losses, cl_loss, is_blocking=False)
            ]

            # Wait for both transfers to complete
            cl.wait_for_events(copy_events)

            # Calculate metrics
            ensemble_probs = np.mean(exit_probs, axis=1)
            pred_labels = np.argmax(ensemble_probs, axis=1)
            accuracy = np.mean(pred_labels == y_true[:BATCH_SIZE])
            avg_losses = np.mean(exit_losses, axis=0)

            print(f"Epoch {epoch:3d} | Accuracy: {accuracy:.1%}")
            print(f"  Exit Losses: {avg_losses.round(3)}")

if __name__ == "__main__":
    main()
