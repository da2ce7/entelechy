### **Architectural Concept: A Unified, Memory-Aware Streaming Engine**

#### **Guiding Principles**

1.  **Primacy of Memory Strategy:** The singular goal of the host-side orchestration is to ensure the core computation executes in the fastest possible memory tier (Registers > Local > Global).
2.  **Modular, "Dumb" Kernels:** Kernels are simple, single-purpose modules. The architecture avoids complex branching ("smart" kernels) and monolithic designs in favor of composability.
3.  **Trust the Driver:** Simple kernels are composed into a logical Directed Acyclic Graph (DAG). The architecture trusts the OpenCL driver to handle low-level optimizations like kernel fusion. The number of nodes in the DAG is a non-goal.
4.  **Unified Dataflow:** The architecture follows a single, logical "always stream" pipeline. Performance and efficiency for problems of any scale is an emergent property of this unified design, not a separate, hard-coded path.

---

### **Core Architectural Components**

#### **1. Modular, Chunk-Based Compute Kernels**

The architecture is built upon a foundation of modular, reusable kernels that operate on "chunks" of a larger problem. The system can chunk work across multiple dimensions (e.g., number of exits, batch size, number of classes) as needed. To ensure mathematical consistency across all scales, any kernel that computes gradients produces **partial results** which are passed to the aggregation engine. Kernels are kept simple, often with a single responsibility (e.g., mapping, reduction over a single dimension), which simplifies maintenance and exposes opportunities for parallel execution.

#### **2. The Generic, Tiered Aggregation Engine**

The heart of the architecture is a powerful, generic aggregation engine that replaces all specialized reduction logic. The host orchestrator chooses **one** of three specialist kernels based _only_ on the number of items (`N`) to be reduced. This engine is a pure, reusable component invoked multiple times within a single training step to consolidate various partial results (probabilities, losses, and gradients).

#### **3. Asynchronous Host Interaction**

The architecture decouples application-level latency from maximum GPU throughput by using an event-based synchronization model. Two key events per batch enable this:

1.  `inference_event`: Signals that final probabilities are available on the host.
2.  `final_batch_event`: Signals that all device-side computations for the batch are complete.

This allows a host application to act on inference results at the earliest possible moment while the GPU continues processing the backpropagation path at full efficiency.

#### **4. The Host Orchestrator**

The host logic is a sophisticated but straightforward orchestrator responsible for resource management and DAG construction. For each batch, it performs a series of strategic assessments:

1.  **Memory Assessment & Chunk Definition:** It compares the memory required for the complete problem against available device memory to determine the optimal chunking strategy. This includes defining `num_exit_chunks`, `num_batch_chunks`, and `num_class_chunks` to ensure all intermediate buffers fit in VRAM.
2.  **Intermediate Activation Strategy:** Crucially, it manages the lifecycle of intermediate hidden activations (`hidden_i`). Based on memory pressure, it selects the optimal strategy to balance performance and scalability:
    - **Cache (Space > Time):** If memory allows, it caches `hidden_i` buffers in VRAM for reuse during the backpropagation phase.
    - **Recompute (Time > Space):** Under extreme memory pressure (e.g., a massive shared layer), it discards `hidden_i` after its initial use and recomputes it on the fly during backpropagation. This guarantees scalability for any problem size.
3.  **DAG Construction & Parallel Dispatch:** It builds the multi-phase computational graph by enqueuing kernels. For logically independent tasks (e.g., the gradient calculations in Phase 8-10), it enqueues them back-to-back without synchronization, explicitly creating parallel workloads for the GPU to schedule.

---

### **Architectural Blueprint & Data Contracts**

#### **Detailed Dataflow Graph**

```mermaid
graph TD
    %% === Node Styling Definitions ===
    classDef kernel fill:#DDEBF7,stroke:#5B9BD5,stroke-width:2px
    classDef data fill:#E2F0D9,stroke:#70AD47,stroke-width:2px
    classDef param fill:#D9E1F2,stroke:#2F5496,stroke-width:2px
    classDef partial_data fill:#F8CBAD,stroke:#C00000,stroke-width:1.5px,stroke-dasharray: 4 4
    classDef final_data fill:#BDD7EE,stroke:#2F5597,stroke-width:2.5px
    classDef host_logic fill:#FFF2CC,stroke:#FFC000,stroke-width:2.5px
    classDef loop_box fill:#f5f5f5,stroke:#333,stroke-width:2px,stroke-dasharray: 10 5
    classDef sync_event fill:#ffc0cb,stroke:#C00000,stroke-width:2px,stroke-dasharray: 5 2
    classDef path_cce fill:#fef0e6,stroke:#C00000
    classDef path_bce fill:#e9eef7,stroke:#2F5496
    classDef transpose_kernel fill:#FBE5D6,stroke:#ED7D31,stroke-width:2px
    classDef full_intermediate fill:#FCE4D6,stroke:#F4B183,stroke-width:2px
    classDef parallel_group fill:#f5f5f5,stroke:#333,stroke-width:2px
    classDef grad_path_a fill:#f3e5f5,stroke:#8e24aa
    classDef grad_path_b fill:#e1f5fe,stroke:#0288d1
    classDef grad_path_c fill:#e8f5e9,stroke:#2e7d32


    %% Phase 0-3: Setup
    subgraph Phase 0-3: Host Setup & Global Params
        HL_0[Start Batch]:::host_logic --> HL_1["1. VRAM Budgeting & Chunking"]:::host_logic --> HL_2["2. Activation Lifecycle & Problem Type"]:::host_logic
        P_Shared[Shared Params]:::param; P_Exits[Exit Params]:::param; P_Temps[Temp Params]:::param; Targets[Targets]:::param
    end

    %% Phase 4: Shared Layer Forward Pass
    subgraph "Phase 4: Shared Layer Forward Pass"
        style "Phase 4: Shared Layer Forward Pass" loop_box
        note_hidden["Note: hidden_i lifecycle<br/>(Cache or Recompute)<br/>managed by Host"]
        HL_2 --> note_hidden
        K4["(4) forward_pass"]:::kernel --> hidden_i[Hidden Activations<br/>Chunk 'i']:::data
    end

    %% Phase 5-7: Conditional Exit Layer Forward Pass
    subgraph "Phase 5-7: Conditional Exit Layer Forward Pass"
        K5["(5) compute_logits_chunk"]:::kernel --> Full_Logits[Full Logits Buffer]:::full_intermediate
        hidden_i --> K5

        HL_3["(7) Host Selects Path<br/>based on problem_type"]:::host_logic
        Full_Logits --> HL_3

        subgraph CCE Path (Softmax)
            style "CCE Path (Softmax)" path_cce
            HL_3 --> K6["(6) reduce_logits_for_softmax"]:::kernel
            Full_Logits --> K6
            K6 --> Softmax_Params[Softmax Denominators]:::full_intermediate
            Softmax_Params --> K7_cce["(7a) compute_probs_loss_cce_chunk"]:::kernel
            Full_Logits --> K7_cce
            K7_cce --> FINAL_Loss_CCE[FINAL CCE Loss (no agg needed)]:::final_data
            K7_cce --> PARTIAL_Probs[PARTIAL Probabilities]:::partial_data
        end

        subgraph BCE Path (Sigmoid)
            style "BCE Path (Sigmoid)" path_bce
            HL_3 --> K7_bce["(7b) compute_probs_loss_bce_chunk"]:::kernel
            Full_Logits --> K7_bce
            K7_bce --> PARTIAL_Loss_BCE[PARTIAL BCE Loss]:::partial_data
            K7_bce --> PARTIAL_Probs
        end
    end

    %% Phase 8-10: Parallel Gradient Computation
    subgraph "Phase 8-10: Parallel Gradient Computation"
        style "Phase 8-10: Parallel Gradient Computation" parallel_group

        subgraph "Local Exit Gradients"
            style "Local Exit Gradients" grad_path_a
            K8["<b>(8) calculate_exit_param_grads_chunk</b>"]:::kernel
            PARTIAL_Probs & Targets & hidden_i --> K8
            K8 --> PARTIAL_Grad_ExitW[PARTIAL Grad_ExitW]:::partial_data & PARTIAL_Grad_ExitB[PARTIAL Grad_ExitB]:::partial_data
        end

        subgraph "Upstream Hidden Gradients"
            style "Upstream Hidden Gradients" grad_path_b
            K9["<b>(9) backprop_error_to_hidden_chunk</b>"]:::kernel
            PARTIAL_Probs & Targets & P_Exits --> K9
            K9 --> PARTIAL_Grad_H_AoS["PARTIAL Grad_H<br/>(AoS Layout)"]:::partial_data
        end

        subgraph "Temperature Gradients"
            style "Temperature Gradients" grad_path_c
            K10["<b>(10) calculate_chunk_temp_gradients</b>"]:::kernel
            PARTIAL_Probs & Full_Logits & Targets & P_Temps --> K10
            K10 --> PARTIAL_Grad_Temps[PARTIAL Grad_Temps]:::partial_data
        end
    end

    %% Phase 11: Data Layout Transformation
    subgraph "Phase 11: Data Layout Transformation"
        direction LR
        PARTIAL_Grad_H_AoS -- All Chunks --> K11["<b>(11) transpose_grad_h</b>"]:::transpose_kernel
        K11 --> PARTIAL_Grad_H_SoA["PARTIAL Grad_H<br/>(SoA Layout)"]:::partial_data
    end

    %% Phase 12: Primary Aggregation Engine
    subgraph Phase 12: Primary Aggregation
        K12["<b>(12) Aggregate Kernel</b>"]:::host_logic
        PARTIAL_Probs & PARTIAL_Loss_BCE & PARTIAL_Grad_ExitW & PARTIAL_Grad_ExitB & PARTIAL_Grad_Temps & PARTIAL_Grad_H_SoA -- All Partial Data --> K12
        K12 --> FINAL_Probs[Final Probs]:::final_data & FINAL_BCE_Loss[Final BCE Loss]:::final_data & FINAL_Grad_H[Final Grad_H]:::final_data
        K12 --> FINAL_Grad_ExitW[Final Grad_ExitW]:::final_data & FINAL_Grad_ExitB[Final Grad_ExitB]:::final_data & FINAL_Grad_Temps[Final Grad_Temps]:::final_data
    end

    %% Phase 13-14: Streaming Shared Layer Backprop
    subgraph "Phase 13-14: Streaming Shared Layer Backprop"
        style "Phase 13-14: Streaming Shared Layer Backprop" parallel_group
        Input_i[Input Chunk 'i']:::data --> K13["<b>(13) backprop_shared_weights_chunk</b>"]:::kernel
        K13 --> PARTIAL_Grad_SW_i[PARTIAL Grad_SW 'i']:::partial_data

        hidden_i --> K13 & K14
        FINAL_Grad_H -- slice --> K13 & K14

        K14["<b>(14) backprop_shared_biases_chunk</b>"]:::kernel --> PARTIAL_Grad_SB_i[PARTIAL Grad_SB 'i']:::partial_data
    end

    %% Phase 15: Final Aggregation
    subgraph Phase 15: Final Aggregation
        K15["<b>(15) Aggregate Kernel</b>"]:::host_logic
        PARTIAL_Grad_SW_i & PARTIAL_Grad_SB_i -- All Chunks --> K15
        K15 --> FINAL_Grad_SW[Final Grad_SW]:::final_data & FINAL_Grad_SB[Final Grad_SB]:::final_data
    end

    %% Phase 16-18: Finalization & Dispatch
    subgraph "Phase 16-18: Finalization & Dispatch"
        direction LR
        subgraph "A. Early Exit Path"
            K16["<b>(16) D2H Async Copy</b><br/>(Final Probs)"]:::data --> EV_Inference["<b>inference_event</b>"]:::sync_event
        end
        subgraph "B. Training Path (All Updates)"
            K17_shared["(17) adam_update (Shared)"]:::kernel; FINAL_Grad_SW & FINAL_Grad_SB --> K17_shared; K17_shared -- updates --> P_Shared
            K17_exits["(17) adam_update (Exits)"]:::kernel; FINAL_Grad_ExitW & FINAL_Grad_ExitB --> K17_exits; K17_exits -- updates --> P_Exits
            K17_temps["(17) adam_update (Temps)"]:::kernel; FINAL_Grad_Temps --> K17_temps; K17_temps -- updates --> P_Temps
            K17_temps --> K18["<b>(18) clamp_temps</b>"]:::kernel
            K18 --> EV_Final["<b>final_batch_event</b>"]:::sync_event
        end
    end

    %% Connections
    HL_1 --> K4;
    FINAL_Probs --> K16
    EV_Inference --> Host_Act["Host Acts on<br/>Early Result"]:::host_logic
    EV_Final --> Host_Wait_Final["Host Blocks for<br/>Full Batch"]:::host_logic
```

### **Final Kernel & Synchronization Contracts**

-   **(4) `forward_pass`**: Computes hidden activations for a chunk of the input batch.
-   **(5) `compute_logits_chunk`**: A streamable kernel computing raw logits for a chunk of exits and classes.
-   **(6) `reduce_logits_for_softmax`**: **CCE Path Only.** Synchronization kernel that computes stable Softmax normalization terms. Skipped on the BCE path.
-   **(7a) `compute_probs_loss_cce_chunk`**: **CCE Path Only.** Streamable kernel computing probabilities and final CCE loss (via scatter-write).
-   **(7b) `compute_probs_loss_bce_chunk`**: **BCE Path Only.** Streamable kernel computing probabilities and partial BCE loss.
-   ---
-   **(8) `calculate_exit_param_grads_chunk`**: A streamable kernel computing **partial** gradients for exit weights and biases (`Grad_ExitW`, `Grad_ExitB`) for a class chunk. It computes the `(prob - target)` error signal on the fly and performs a reduction over the batch dimension.
-   **(9) `backprop_error_to_hidden_chunk`**: A streamable kernel computing the **partial** upstream gradient for the hidden layer (`Grad_H`) for a class chunk. It computes the `(prob - target)` error signal on the fly and performs a reduction over the class dimension.
-   **(10) `calculate_chunk_temp_gradients`**: A streamable kernel computing **partial** gradients for the temperature parameters.
-   ---
-   **(11) `transpose_grad_h`**: A data layout transformation kernel. Reads the AoS-formatted `Grad_H` and writes it in an SoA layout for efficient aggregation.
-   **(12) `aggregate_*` kernels**: Generic, stateless kernel interface invoked to consolidate all partial results from the exit layer backpropagation (Probs, BCE Loss, Grads W, B, H, Temps).
-   **(13) `backprop_shared_weights_chunk`**: A streamable backpropagation kernel for shared layer weights, computing partial gradients for a batch chunk.
-   **(14) `backprop_shared_biases_chunk`**: A streamable backpropagation kernel for shared layer biases, computing partial gradients for a batch chunk.
-   **(15) `aggregate_*` kernels**: The same generic kernel interface, invoked to consolidate partial gradients from the shared layer (`Grad_SW`, `Grad_SB`).
-   **(16) `D2H Async Copy`**: A non-blocking Device-to-Host transfer of the `Final Probs` buffer, whose completion signals the `inference_event`.
-   **(17) `adam_update`**: Generic optimizer kernel, invoked multiple times for different parameter groups.
-   **(18) `clamp_temperatures`**: Final utility kernel for parameter constraint.
-   **Host/Device Synchronization Contracts**:
    -   `inference_event`: Guarantees `Final Probs` data is available on the host.
    -   `final_batch_event`: Guarantees all device computations and parameter updates for the batch are complete.

---

### **Validation Scenarios**

The architecture's unified and modular dataflow robustly handles all scenarios:

-   **Scenario: The Iris Case (Tiny Problem, `num_*_chunks=1`)**
    -   **Behavior:** The host sets all chunk counts to 1. All "streaming" loops run once. The parallel gradient kernels (8, 9, 10) are launched sequentially, and aggregation kernels (12, 15) become near-zero-cost `aggregate_identity` copies. The graph functions identically to a non-streaming design, proving its universality.

-   **Scenario: The Lexicon (Massive `output_classes`)**
    -   **Behavior:** The host identifies the class dimension as the memory bottleneck and sets `num_class_chunks` to a large value.
        -   **Forward Pass:** The three-stage softmax pipeline (5, 6, 7a) is critical. It materializes a massive `Full_Logits` buffer in chunks to enable the operation, with Node 6 acting as the essential synchronization point and bottleneck.
        -   **Backward Pass:** The parallel gradient kernels (8, 9, 10) are streamed over the class chunks. For example, `(9) backprop_error_to_hidden_chunk` performs many small, independent reductions to build up pieces of the final `Grad_H` buffer.
        -   **Aggregation:** The primary aggregation kernel (12) is called to sum a huge number of partial gradient results from all three parallel paths, demonstrating its scalability.

-   **Scenario: Maximizing GPU Throughput (The General Case)**
    -   **Behavior:** The architecture's modularity shines in Phase 8-10. The host enqueues kernels (8), (9), and (10) back-to-back without any intervening synchronization.
    -   **Parallel Execution:** This creates three independent, concurrent workloads. A proficient OpenCL driver and GPU scheduler will interleave the execution of work-groups from all three kernels. If a work-group from kernel (9) stalls on memory, a ready work-group from (8) or (10) can be scheduled, maximizing hardware occupancy and hiding memory latency. This improves overall throughput compared to a monolithic kernel that would have to execute its internal stages serially. A similar parallel execution occurs for kernels (13) and (14).

-   **Scenario: The Behemoth (Few Exits, Fat Network)**
    -   **Behavior:** The host determines the memory bottleneck is the intermediate `hidden_i` activation buffer. It streams along the batch dimension (`num_batch_chunks` > 1) and makes a strategic choice: **Recompute** `hidden_i` during backpropagation if VRAM is tight, or **Cache** it after Phase 4. This explicitly trades compute for memory, guaranteeing scalability.

-   **Scenario: The Live Dashboard (Low-Latency Reporting)**
    -   **Behavior:** The event-based design decouples latency from throughput. The host dispatches the full training batch, then immediately performs a non-blocking wait on the `inference_event`. The moment final probabilities are aggregated (end of Phase 12) and copied (Node 16), the host can fetch them. This happens in parallel while the GPU continues with the expensive shared layer backpropagation (Phases 13-18), delivering the earliest possible result to the application.