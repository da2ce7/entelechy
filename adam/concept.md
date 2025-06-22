### **Architectural Concept: A Unified, Memory-Aware Streaming Engine**

#### **Guiding Principles**

1.  **Primacy of Memory Strategy:** The singular goal of the host-side orchestration is to ensure the core computation executes in the fastest possible memory tier (Registers > Local > Global).
2.  **Modular, "Dumb" Kernels:** Kernels are simple, single-purpose modules. The architecture avoids complex branching ("smart" kernels) and monolithic designs in favor of composability.
3.  **Trust the Driver:** Simple kernels are composed into a logical Directed Acyclic Graph (DAG). The architecture trusts the OpenCL driver to handle low-level optimizations like kernel fusion. The number of nodes in the DAG is a non-goal.
4.  **Unified Dataflow:** The architecture follows a single, logical "always stream" pipeline. Performance and efficiency for problems of any scale is an emergent property of this unified design, not a separate, hard-coded path.
5.  **Architectural Hierarchy:** This system is developed under a strict hierarchy of artifacts to ensure conceptual integrity.
    *   **1. Design Document (This Document):** The highest authority and source of truth for conceptual correctness.
    *   **2. Kernel Header Contract:** The binding technical contract between host and device. It must be in deep resonance with the design document.
    *   **3. Host Code Implementation:** The lowest authority. It must be rigorously implemented to conform to the kernel header's contract. The header is *never* modified to suit the host code; the host code *always* yields to the contract.

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
    classDef fused_kernel_box fill:#f0f4c3,stroke:#afb42b,stroke-width:2px,stroke-dasharray: 5 2
    classDef logical_step fill:#ffffff,stroke:#757575,stroke-width:1px,stroke-dasharray: 2 2


    subgraph "Annotation Key: Fused Operations"
        direction LR
        subgraph " "
            LS_Alpha["α"]:::logical_step
            LS_Beta["β"]:::logical_step
            LS_Gamma["γ"]:::logical_step
            LS_Delta["δ"]:::logical_step
        end
        subgraph " "
            Desc_Alpha["Mat-Mul + Bias<br/>(Standard Op Fusion)"]
            Desc_Beta["Probability Calc + Loss Calc<br/>(Producer-Consumer Fusion)"]
            Desc_Gamma["Weight Grad + Bias Grad<br/>(Data Reuse Fusion)"]
            Desc_Delta["Mat-Mul + Bias + ReLU<br/>(Standard Op Fusion)"]
        end
    end

    %% Phase 0-3: Setup
    subgraph Phase 0-3: Host Setup & Global Params
        HL_0[Start Batch]:::host_logic --> HL_1["1. VRAM Budgeting & Chunking"]:::host_logic --> HL_2["2. Activation Lifecycle & Problem Type"]:::host_logic
        P_Shared[Shared Params]:::param; P_Exits[Exit Params]:::param; P_Temps[Temp Params]:::param;
        P_Step[Global Step 't']:::param
        Targets[Targets]:::param
    end

    %% Phase 4: Shared Layer Forward Pass (Fused)
    subgraph "Phase 4: Shared Layer Forward Pass"
        style "Phase 4: Shared Layer Forward Pass" loop_box
        note_hidden["Note: hidden_i lifecycle<br/>(Cache or Recompute)<br/>managed by Host"]
        HL_2 --> note_hidden

        subgraph K4["(4) forward_pass"]:::fused_kernel_box
            direction LR
            L4a["Mat-Mul δ"]:::logical_step --> L4b["Bias Add δ"]:::logical_step --> L4c["ReLU δ"]:::logical_step
        end
        HL_1 --> L4a
        L4c --> hidden_i[Hidden Activations<br/>Chunk 'i']:::data
    end

    %% Phase 5-7: Conditional Exit Layer Forward Pass (Fused)
    subgraph "Phase 5-7: Conditional Exit Layer Forward Pass"
        subgraph K5["(5) compute_logits_chunk"]:::fused_kernel_box
             direction LR
             L5a["Mat-Mul α"]:::logical_step --> L5b["Bias Add α"]:::logical_step
        end
        hidden_i --> L5a
        L5b --> Full_Logits[Full Logits Buffer]:::full_intermediate

        HL_3["(7) Host Selects Path<br/>based on problem_type"]:::host_logic
        Full_Logits --> HL_3

        subgraph CCE Path (Softmax)
            style "CCE Path (Softmax)" path_cce
            HL_3 --> K6["(6) reduce_logits_for_softmax"]:::kernel
            Full_Logits --> K6
            K6 --> Softmax_Params[Softmax Denominators]:::full_intermediate

            subgraph K7_cce["(7a) compute_probs_loss_cce_chunk"]:::fused_kernel_box
                L7a_prob["Prob Calc β"]:::logical_step --> L7a_loss["Loss Calc β"]:::logical_step
            end
            Softmax_Params & Full_Logits --> L7a_prob
            L7a_loss --> FINAL_Loss_CCE[FINAL CCE Loss (no agg needed)]:::final_data
            L7a_prob --> PARTIAL_Probs[PARTIAL Probabilities]:::partial_data
        end

        subgraph BCE Path (Sigmoid)
            style "BCE Path (Sigmoid)" path_bce
            subgraph K7_bce["(7b) compute_probs_loss_bce_chunk"]:::fused_kernel_box
                L7b_prob["Prob Calc β"]:::logical_step --> L7b_loss["Loss Calc β"]:::logical_step
            end
            HL_3 & Full_Logits --> L7b_prob
            L7b_loss --> PARTIAL_Loss_BCE[PARTIAL BCE Loss]:::partial_data
            L7b_prob --> PARTIAL_Probs
        end
    end

    %% Phase 8-11: Parallel Gradient Computation & Transformation
    subgraph "Phase 8-11: Parallel Gradient Path (Chunk-Based)"
        style "Phase 8-11: Parallel Gradient Path (Chunk-Based)" parallel_group

        subgraph "Local Exit Gradients"
            style "Local Exit Gradients" grad_path_a
            subgraph K8["<b>(8) calculate_exit_param_grads_chunk</b>"]:::fused_kernel_box
                L8_w["Weight Grad Calc γ"]:::logical_step
                L8_b["Bias Grad Calc γ"]:::logical_step
            end
            PARTIAL_Probs & Targets --> L8_w; PARTIAL_Probs & Targets --> L8_b
            hidden_i --> L8_w
            L8_w --> PARTIAL_Grad_ExitW[PARTIAL Grad_EW]:::partial_data
            L8_b --> PARTIAL_Grad_ExitB[PARTIAL Grad_EB]:::partial_data
        end

        subgraph "Upstream Hidden Gradients & Transformation"
            style "Upstream Hidden Gradients & Transformation" grad_path_b
            K9["<b>(9) backprop_error_to_hidden_chunk</b>"]:::kernel
            PARTIAL_Probs & Targets & P_Exits --> K9
            K9 --> PARTIAL_Grad_H_AoS["PARTIAL Grad_H<br/>(AoS Layout)"]:::partial_data
            PARTIAL_Grad_H_AoS --> K11["<b>(11) transpose_chunk</b><br/>(on partial Grad_H)"]:::transpose_kernel
            K11 --> PARTIAL_Grad_H_SoA["PARTIAL Grad_H<br/>(SoA Layout)"]:::partial_data
        end

        subgraph "Temperature Gradients"
            style "Temperature Gradients" grad_path_c
            K10["<b>(10) calculate_chunk_temp_gradients</b>"]:::kernel
            PARTIAL_Probs & Full_Logits & Targets & P_Temps --> K10
            K10 --> PARTIAL_Grad_Temps[PARTIAL Grad_Temps]:::partial_data
        end
    end


    %% Phase 12-18: Remainder of Graph
    subgraph Phase 12: Primary Aggregation
        K12["<b>(12) Aggregate Kernel</b>"]:::host_logic
        PARTIAL_Probs & PARTIAL_Loss_BCE & PARTIAL_Grad_ExitW & PARTIAL_Grad_ExitB & PARTIAL_Grad_Temps & PARTIAL_Grad_H_SoA -- All Partial Data --> K12
        K12 --> FINAL_Probs[Final Probs]:::final_data & FINAL_BCE_Loss[Final BCE Loss]:::final_data & FINAL_Grad_H[Final Grad_H]:::final_data
        K12 --> FINAL_Grad_ExitW[Final Grad_EW]:::final_data & FINAL_Grad_ExitB[Final Grad_EB]:::final_data & FINAL_Grad_Temps[Final Grad_Temps]:::final_data
    end

    subgraph "Phase 13-14: Streaming Shared Layer Backprop"
        style "Phase 13-14: Streaming Shared Layer Backprop" parallel_group
        Input_i[Input Chunk 'i']:::data --> K13["<b>(13) backprop_shared_weights_chunk</b>"]:::kernel
        K13 --> PARTIAL_Grad_SW_i[PARTIAL Grad_SW 'i']:::partial_data
        hidden_i --> K13 & K14
        FINAL_Grad_H -- slice --> K13 & K14
        K14["<b>(14) backprop_shared_biases_chunk</b>"]:::kernel --> PARTIAL_Grad_SB_i[PARTIAL Grad_SB 'i']:::partial_data
    end

    subgraph Phase 15: Final Aggregation
        K15["<b>(15) Aggregate Kernel</b>"]:::host_logic
        PARTIAL_Grad_SW_i & PARTIAL_Grad_SB_i -- All Chunks --> K15
        K15 --> FINAL_Grad_SW[Final Grad_SW]:::final_data & FINAL_Grad_SB[Final Grad_SB]:::final_data
    end

    subgraph "Phase 16-18: Finalization & Dispatch"
        direction LR
        subgraph "A. Early Exit Path"
            K16["<b>(16) D2H Async Copy</b><br/>(Final Probs)"]:::data --> EV_Inference["<b>inference_event</b>"]:::sync_event
        end
        subgraph "B. Training Path (All Updates)"
            K17_shared["(17) adam_update (Shared)"]:::kernel; FINAL_Grad_SW & FINAL_Grad_SB --> K17_shared; K17_shared -- updates --> P_Shared
            P_Step --> K17_shared
            K17_exits["(17) adam_update (Exits)"]:::kernel; FINAL_Grad_ExitW & FINAL_Grad_ExitB --> K17_exits; K17_exits -- updates --> P_Exits
            P_Step --> K17_exits
            K17_temps["(17) adam_update (Temps)"]:::kernel; FINAL_Grad_Temps --> K17_temps; K17_temps -- updates --> P_Temps
            P_Step --> K17_temps
            K17_temps --> K18["<b>(18) clamp_temps</b>"]:::kernel
            K18 --> EV_Final["<b>final_batch_event</b>"]:::sync_event
        end
    end

    %% Connections
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
-   **(11) `transpose_chunk`**: **(Generic Utility).** A streamable kernel that transposes a rectangular slice (chunk) of a matrix. Its fundamental architectural purpose is **latency hiding**. For the `Grad_H` backpropagation path, this kernel is enqueued on a **per-chunk** basis immediately following its corresponding `backprop_error_to_hidden_chunk` (9) call. This fine-grained dependency allows the GPU's out-of-order scheduler to overlap the memory-bound transpose of chunk *N* with the compute-bound gradient calculations of chunk *N+1*, maximizing hardware occupancy and scaling efficiency.
-   **(12) `aggregate_*` kernels**: Generic, stateless kernel interface invoked to consolidate all partial results. This includes Probs, BCE Loss, Grads W, B, Temps, and crucially, the **SoA-formatted** partial `Grad_H` results from the parallel transpose step.
-   **(13) `backprop_shared_weights_chunk`**: A streamable backpropagation kernel for shared layer weights, computing partial gradients for a batch chunk.
-   **(14) `backprop_shared_biases_chunk`**: A streamable backpropagation kernel for shared layer biases, computing partial gradients for a batch chunk.
-   **(15) `aggregate_*` kernels**: The same generic kernel interface, invoked to consolidate partial gradients from the shared layer (`Grad_SW`, `Grad_SB`).
-   **(16) `D2H Async Copy`**: A non-blocking Device-to-Host transfer of the `Final Probs` buffer, whose completion signals the `inference_event`.
-   **(17) `adam_update`**: Generic optimizer kernel, invoked multiple times for different parameter groups. This kernel is passed the global training step `t` and performs the Adam bias correction calculation internally (`sqrt(1 - beta2^t) / (1 - beta1^t)`). This approach guarantees numerical stability for training runs of any length (i.e., scaling across the epoch dimension) by avoiding the host-side calculation of `beta^t`, which can suffer from precision loss at large `t`.
-   **(18) `clamp_temperatures`**: Final utility kernel for parameter constraint.
-   **Host/Device Synchronization Contracts**:
    -   `inference_event`: Guarantees `Final Probs` data is available on the host.
    -   `final_batch_event`: Guarantees all device computations and parameter updates for the batch are complete.


### **Validation Scenarios**

The architecture's unified dataflow is validated by its robust and efficient handling of a wide spectrum of computational challenges, with each scenario probing a distinct scalability dimension.

-   **Scenario: The Iris Case (Tiny Problem)**
    -   **Insight:** Demonstrates **universality**. For a problem that fits entirely in memory, the "always stream" design gracefully degrades. With all `num_*_chunks=1`, streaming loops run once and aggregation becomes a near-zero-cost identity copy, proving the architecture functions efficiently at any scale without special-casing.

-   **Scenario: The Marathon (Massive `epochs`)**
    -   **Insight:** Guarantees **long-term stability**. The system avoids numerical underflow in the Adam optimizer by delegating the sensitive `beta**t` calculation to the `(17) adam_update` kernel. By passing the `global_step` `t` as a simple integer, the architecture ensures the optimizer is mathematically correct and stable indefinitely, proving scalability across the time dimension.

-   **Scenario: The Hydra (Massive `num_exits`)**
    -   **Insight:** Validates **scalability of the core multi-exit design**. When faced with a huge number of exit paths, the host orchestrator chunks the problem along the exit dimension (`num_exit_chunks > 1`). Kernels for logits, probabilities, and gradients (5-10) are designed to process these exit chunks in parallel streams, which are then consolidated by the aggregation engine.

-   **Scenario: The Behemoth (Massive `hidden_dim`)**
    -   **Insight:** Proves **scalability through strategic compute/memory trade-offs**. When a deep or wide shared layer makes the intermediate `hidden_i` buffer the memory bottleneck, the host makes a critical choice: **Cache** `hidden_i` for speed if memory allows, or **Recompute** it on-the-fly during backpropagation to guarantee scalability for any network size.

-   **Scenario: The Lexicon (Massive `output_classes`)**
    -   **Insight:** Achieves **maximum hardware occupancy via interleaved compute and memory operations**. When class count is the bottleneck, the host orchestrates a fine-grained parallel workload for each class chunk. The compute-bound gradient kernels (8, 9, 10) and the memory-bound `transpose_chunk` (11) for a given chunk are enqueued back-to-back without barriers. This allows the GPU's out-of-order scheduler to execute the transpose of chunk *N* concurrently with the gradient calculations of chunk *N+1*. This strategic interleaving of independent compute and memory tasks is the core mechanism for hiding latency and maximizing throughput.

-   **Scenario: The Data Tsunami (Massive `batch_size`)**
    -   **Insight:** Validates the **two-phase streaming backpropagation strategy**. For an extremely large batch, `backprop_shared_weights_chunk` (13) and `backprop_shared_biases_chunk` (14) stream over the batch dimension, computing partial gradients which are then averaged. This contrasts with earlier phases that reduce *over* the batch, showing a sophisticated dataflow tailored to memory access patterns.

-   **Scenario: The Colossus (Holistic Stress Test)**
    -   **Insight:** Tests the **synergy of all scaling strategies under compound memory pressure**. This case presents a network where no single dimension is the bottleneck, but the combination of large `num_exits`, `hidden_dim`, `output_classes`, and `batch_size` collectively exceeds VRAM. The host orchestrator is forced to **compose a multi-faceted execution plan**: it will apply chunking over exits/classes (Phases 5-12) and then apply chunking over the batch (Phases 13-14) sequentially within a single training step. It may also engage the `Recompute` strategy for hidden activations. This proves the system's ability to handle not just asymmetric but also large, symmetric workloads through a flexible, phased approach.

-   **Scenario: The Live Dashboard (Live Inference While Training)**
    -   **Insight:** Fulfills **application-level responsiveness requirements**. The event-based design (`inference_event`) decouples user-facing results from the full training step. An application can act on inference probabilities the moment they are available, while the GPU continues the expensive shared-layer backpropagation and parameter updates in the background.