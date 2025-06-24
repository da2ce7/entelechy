### **Architectural Concept: A Unified, Memory-Aware Streaming Classification Engine**

#### **Guiding Principles**

1.  **Primacy of Memory Strategy:** The singular goal of the host-side orchestration is to ensure the core computation executes in the fastest possible memory tier (Registers > Local > Global).
2.  **Modular, "Dumb" Kernels:** Kernels are simple, single-purpose modules. The architecture avoids complex branching ("smart" kernels) and monolithic designs in favor of composability.
3.  **Trust the Driver:** Simple kernels are composed into a logical Directed Acyclic Graph (DAG). The architecture trusts the OpenCL driver to handle low-level optimizations like kernel fusion. The number of nodes in the DAG is a non-goal.
4.  **Unified Dataflow:** The architecture follows a single, logical "always stream" pipeline. Performance and efficiency for problems of any scale is an emergent property of this unified design, not a separate, hard-coded path.
5.  **Architectural Hierarchy:** This system is developed under a strict hierarchy of artifacts to ensure conceptual integrity.
    - **1. Design Document (This Document):** The highest authority and source of truth for conceptual correctness.
    - **2. Kernel Header Contract:** The binding technical contract between host and device. It must be in deep resonance with the design document.
    - **3. Host Code Implementation:** The lowest authority. It must be rigorously implemented to conform to the kernel header's contract. The header is _never_ modified to suit the host code; the host code _always_ yields to the contract.

---

### **Core Architectural Components**

#### **1. Modular, Chunk-Based Compute Kernels**

The architecture is built upon a foundation of modular, reusable kernels that operate on "chunks" of a larger problem. The system can chunk work across multiple dimensions (e.g., number of modules, batch size, number of classes) as needed. Each `Classifier Module`, for instance, is configured with an **`Operating Mode`** (`CCE` for single-label or `BCE` for multi-label tasks), which instructs the Host Orchestrator on which specific computational path to construct using these kernels.

#### **2. The Recursive, Tiered Aggregation Engine**

The heart of the architecture is a powerful aggregation engine that implements a **recursive, multi-stage reduction tree**. Instead of generating all `N` partial results before aggregating, this model uses a classic divide-and-conquer strategy to keep the GPU saturated and VRAM usage minimal.

It is governed by a new host-configurable parameter, `K` (the **Reduction Batch Size**), which defines the width of the parallel kernel front at each reduction stage. The process is as follows:

1.  The Host Orchestrator renders the base `N` partial results in batches of `K`.
2.  After each batch of `K` partials is computed, they are **immediately reduced** by an `aggregate_*` kernel into a single "Level 1" intermediate result.
3.  This process is repeated `N/K` times, converting a large problem of `N` "Level 0" results into a much smaller problem of `N/K` "Level 1" results.
4.  The host then **recursively applies this same logic** to the buffer of "Level 1" results, reducing them in batches of `K` to create "Level 2" results, and so on, until a single final tensor remains.

This `log_K(N)` strategy breaks the `O(N)` memory dependency of a flat aggregation model and allows the system to scale to problems of arbitrary size.

#### **3. Asynchronous Host Interaction**

The architecture decouples application-level latency from maximum GPU throughput by using an event-based synchronization model. Two key events per batch enable this:

1.  `inference_event`: Signals that the **complete forward-pass results** are available on the host.
2.  `final_batch_event`: Signals that all device-side computations for the batch are complete.

This allows a host application to act on the complete forward-pass results as soon as they are aggregated, without blocking on the completion of the subsequent backpropagation and parameter update path.

#### **4. The Host Orchestrator**

The host logic is a sophisticated but straightforward orchestrator responsible for resource management and DAG construction. For each batch, it performs a series of strategic assessments:

1.  **Memory Assessment & Chunk Definition:** It compares the memory required for the complete problem against available device memory to determine the optimal chunking strategy. This includes defining `num_module_chunks`, `num_batch_chunks`, and `num_class_chunks`.
2.  **`Operating Mode` & Activation Lifecycle:** It determines the base computational graph by reading the `Classifier Module`'s configured `Operating Mode` (`CCE` or `BCE`). It then manages the lifecycle of intermediate hidden activations (`hidden_i`), selecting the optimal strategy:
    - **Cache (Space > Time):** If memory allows, it caches `hidden_i` buffers in VRAM for reuse during backpropagation.
    - **Recompute (Time > Space):** Under extreme memory pressure, it discards `hidden_i` and recomputes it on the fly during backpropagation.
3.  **SIMD-Aware Weight Layout:** To fully exploit vector processing capabilities on the device and ensure coalesced memory access, the Host Orchestrator is responsible for transforming the shared layer weights into a SIMD-friendly "Struct of Arrays" (SoA) layout before enqueuing kernel (4). This pre-processing step rearranges the weights from a logical `(hidden_dim, input_dim)` matrix to a physical `(hidden_dim/SIMD_WIDTH, input_dim, SIMD_WIDTH)` buffer. This data layout is a concrete expression of the **Primacy of Memory Strategy** principle and is a non-negotiable part of the contract with the `forward_pass` kernel.
4.  **DAG Construction & Parallel Dispatch:** It builds the multi-phase computational graph by enqueuing kernels according to the selected mode and chunking strategy. For logically independent tasks, it enqueues them back-to-back without synchronization to create parallel workloads.
5.  **Reduction Planning & Rendering:** For any task requiring aggregation over a large number of chunks (`N`), the host acts as a **Reduction Planner**. It analyzes the problem size and available VRAM to determine an optimal **Reduction Batch Size (`K`)**. It then renders the full `log_K(N)` reduction tree, orchestrating the iterative stages of partial computation and aggregation while managing the lifecycle of intermediate result buffers on the device.

### **Architectural Blueprint & Data Contracts for a Multi-Head Classifier**

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
    classDef specialized_kernel fill:#e1f5fe,stroke:#0288d1,stroke-width:2px


    subgraph "Annotation Keys"
      direction LR
        subgraph "Terminology Key"
            TermNote["Note: 'Module' refers to the system<br/>implementation of an ML 'Head'."]
        end
        subgraph "Fused Operation Key"
            direction LR
            subgraph " "
                LS_Alpha["α"]:::logical_step; LS_Beta["β"]:::logical_step
                LS_Gamma["γ"]:::logical_step; LS_Delta["δ"]:::logical_step
            end
            subgraph " "
                Desc_Alpha["Mat-Mul + Bias<br/>(Standard Op Fusion)"]
                Desc_Beta["Probability Calc + Loss Calc<br/>(Producer-Consumer Fusion)"]
                Desc_Gamma["Weight Grad + Bias Grad<br/>(Data Reuse Fusion)"]
                Desc_Delta["Mat-Mul + Bias + ReLU<br/>(Standard Op Fusion)"]
            end
        end
    end

    %% Phase 0-3: Setup
    subgraph Phase 0-3: Host Setup & Global Params
        HL_0[Start Batch]:::host_logic --> HL_1["1. VRAM Budgeting & Chunking"]:::host_logic --> HL_2["2. Read Mode, Manage Activations,<br/>& Plan Reductions"]:::host_logic
        P_Shared[Shared Params]:::param; P_ClassifierModule[Classifier Module Params]:::param; P_Temps[Temp Params]:::param;
        P_Step[Global Step 't']:::param
        Targets[Targets]:::param
        SampleMask[Sample Mask]:::param
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
        SampleMask --> L4a
        L4c --> hidden_i[Hidden Activations<br/>Chunk 'i']:::data
        L4c --> hidden_mask[Hidden Mask]:::data
    end

    %% Phase 5-7: Classifier Module Forward Pass (Fused)
    subgraph "Phase 5-7: Forward Pass for Classifier Head (as a Module)"
        subgraph K5["(5) compute_logits_chunk"]:::fused_kernel_box
             direction LR
             L5a["Mat-Mul α"]:::logical_step --> L5b["Bias Add α"]:::logical_step
        end
        hidden_i & hidden_mask --> L5a
        L5b --> Full_Logits[Full Logits Buffer]:::full_intermediate

        HL_3["(7) Host Selects Path<br/>based on Operating Mode (CCE/BCE)"]:::host_logic
        Full_Logits --> HL_3

        subgraph "Operating Mode: CCE (Softmax Path)"
            style "Operating Mode: CCE (Softmax Path)" path_cce
            HL_3 --> K6["(6) reduce_logits_for_softmax"]:::kernel
            Full_Logits --> K6
            K6 --> Softmax_Params[Softmax Denominators]:::full_intermediate

            subgraph K7_cce["(7a) compute_probs_loss_cce_chunk"]:::fused_kernel_box
                L7a_prob["Prob Calc β"]:::logical_step --> L7a_loss["Loss Calc β"]:::logical_step
            end
            Softmax_Params & Full_Logits --> L7a_prob
            SampleMask --> L7a_loss
            L7a_loss --> FINAL_Loss_CCE[FINAL CCE Loss (no agg needed)]:::final_data
            L7a_prob --> PARTIAL_Probs[PARTIAL Probabilities]:::partial_data
        end

        subgraph "Operating Mode: BCE (Sigmoid Path)"
            style "Operating Mode: BCE (Sigmoid Path)" path_bce
            subgraph K7_bce["(7b) compute_probs_loss_bce_chunk"]:::fused_kernel_box
                L7b_prob["Prob Calc β"]:::logical_step --> L7b_loss["Loss Calc β"]:::logical_step
            end
            HL_3 & Full_Logits --> L7b_prob
            SampleMask --> L7b_loss
            L7b_loss --> PARTIAL_Loss_BCE[PARTIAL BCE Loss]:::partial_data
            L7b_prob --> PARTIAL_Probs
        end
    end

    %% Phase 8-11: Parallel Gradient Computation & Transformation
    subgraph "Phase 8-11: Parallel Gradient Path (Chunk-Based)"
        style "Phase 8-11: Parallel Gradient Path (Chunk-Based)" parallel_group

        subgraph "Gradients for Classifier Module"
            style "Gradients for Classifier Module" grad_path_a
            subgraph K8["<b>(8) calculate_module_param_grads_chunk</b>"]:::fused_kernel_box
                L8_w["Weight Grad Calc γ"]:::logical_step
                L8_b["Bias Grad Calc γ"]:::logical_step
            end
            PARTIAL_Probs & Targets & SampleMask --> L8_w; PARTIAL_Probs & Targets & SampleMask --> L8_b
            hidden_i --> L8_w
            L8_w --> PARTIAL_Grad_ModW[PARTIAL Grad_ModW]:::partial_data
            L8_b --> PARTIAL_Grad_ModB[PARTIAL Grad_ModB]:::partial_data
        end

        subgraph "Upstream Hidden Gradients & Transformation"
            style "Upstream Hidden Gradients & Transformation" grad_path_b
            K9["<b>(9) backprop_error_to_hidden_chunk</b>"]:::kernel
            PARTIAL_Probs & Targets & P_ClassifierModule & SampleMask --> K9
            K9 --> PARTIAL_Grad_H_AoS["PARTIAL Grad_H<br/>(AoS Layout)"]:::partial_data
            PARTIAL_Grad_H_AoS --> K11["<b>(11) transpose_chunk</b><br/>(on partial Grad_H)"]:::transpose_kernel
            K11 --> PARTIAL_Grad_H_SoA["PARTIAL Grad_H<br/>(SoA Layout)"]:::partial_data
        end

        subgraph "Temperature Gradients"
            style "Temperature Gradients" grad_path_c
            K10["<b>(10) calculate_chunk_temp_gradients</b>"]:::kernel
            PARTIAL_Probs & Full_Logits & Targets & P_Temps & SampleMask --> K10
            K10 --> PARTIAL_Grad_Temps[PARTIAL Grad_Temps]:::partial_data
        end
    end

    %% Phase 12-19: Remainder of Graph
    subgraph "Phase 12: Recursive Aggregation Tree (log_K(N) Stages)"
        direction LR
        style "Phase 12: Recursive Aggregation Tree (log_K(N) Stages)" loop_box
        K_Recursive_12["(8, 9, 10, agg) ...<br/>Iterative Reduction"]:::host_logic
    end
    PARTIAL_Probs & PARTIAL_Loss_BCE & PARTIAL_Grad_ModW & PARTIAL_Grad_ModB & PARTIAL_Grad_Temps & PARTIAL_Grad_H_SoA -- All Partial Data --> K_Recursive_12
    K_Recursive_12 --> FINAL_Probs[Final Probs]:::final_data & FINAL_BCE_Loss[Final BCE Loss]:::final_data & AGG_Grad_H_SoA["Aggregated Grad_H<br/>(B*H, M Layout)"]:::full_intermediate
    K_Recursive_12 --> FINAL_Grad_ModW[Final Grad_ModW]:::final_data & FINAL_Grad_ModB[Final Grad_ModB]:::final_data & FINAL_Grad_Temps[Final Grad_Temps]:::final_data


    subgraph Phase 13: Specialized Grad_H Reduction
        K13["<b>(13) reduce_grad_h_over_modules</b><br/>(Specialized Reduction)"]:::specialized_kernel
        AGG_Grad_H_SoA --> K13
        K13 --> FINAL_Grad_H[Final Grad_H]:::final_data
    end

    subgraph "Phase 14-15: Streaming Shared Layer Backprop"
        style "Phase 14-15: Streaming Shared Layer Backprop" parallel_group
        Input_i[Input Chunk 'i']:::data & SampleMask --> K14["<b>(14) backprop_shared_weights_chunk</b>"]:::kernel
        K14 --> PARTIAL_Grad_SW_i[PARTIAL Grad_SW 'i']:::partial_data
        hidden_i --> K14 & K15
        FINAL_Grad_H -- slice --> K14 & K15
        SampleMask --> K15["<b>(15) backprop_shared_biases_chunk</b>"]:::kernel
        K15 --> PARTIAL_Grad_SB_i[PARTIAL Grad_SB 'i']:::partial_data
    end

    subgraph "Phase 16: Recursive Shared Gradient Aggregation"
        direction LR
        style "Phase 16: Recursive Shared Gradient Aggregation" loop_box
        K_Recursive_16["(agg) ...<br/>Iterative Reduction"]:::host_logic
    end
    PARTIAL_Grad_SW_i & PARTIAL_Grad_SB_i -- All Chunks --> K_Recursive_16
    K_Recursive_16 --> FINAL_Grad_SW[Final Grad_SW]:::final_data & FINAL_Grad_SB[Final Grad_SB]:::final_data


    subgraph "Phase 17-19: Finalization & Dispatch"
        direction LR
        subgraph "A. Asynchronous Result Retrieval"
            K17["<b>(17) D2H Async Copy</b><br/>(Final Probs)"]:::data --> EV_Inference["<b>inference_event</b>"]:::sync_event
        end
        subgraph "B. Training Path (All Updates)"
            K18_shared["(18) adam_update (Shared)"]:::kernel; FINAL_Grad_SW & FINAL_Grad_SB --> K18_shared; K18_shared -- updates --> P_Shared
            P_Step --> K18_shared
            K18_module["(18) adam_update (ClassifierModule)"]:::kernel; FINAL_Grad_ModW & FINAL_Grad_ModB --> K18_module; K18_module -- updates --> P_ClassifierModule
            P_Step --> K18_module
            K18_temps["(18) adam_update (Temps)"]:::kernel; FINAL_Grad_Temps --> K18_temps; K18_temps -- updates --> P_Temps
            K18_temps --> K19["<b>(19) clamp_temps</b>"]:::kernel
            K19 --> EV_Final["<b>final_batch_event</b>"]:::sync_event
        end
    end

    %% Connections
    FINAL_Probs --> K17
    EV_Inference --> Host_Act["Host Acts on<br/>Full Forward Result"]:::host_logic
    EV_Final --> Host_Wait_Final["Host Blocks for<br/>Full Batch"]:::host_logic
```

### **Final Kernel & Synchronization Contracts**

- **(4) `forward_pass`**: Computes hidden activations for a chunk of the input batch.
- **(5) `compute_logits_chunk`**: A streamable kernel computing raw logits. It is designed to be **robust to memory padding**, using physical strides to correctly navigate the module-level weight and bias buffers.
- **(6) `reduce_logits_for_softmax`**: **Invoked when `Operating Mode` is `CCE`.** A synchronization kernel that computes stable Softmax normalization terms. It is designed to be **robust to memory padding**, using physical strides to correctly navigate its input buffers.
- **(7a) `compute_probs_loss_cce_chunk`**: **Invoked when `OperatingMode` is `CCE`.** A streamable kernel computing probabilities and final CCE loss. It is designed to be **robust to memory padding**, using physical strides to correctly navigate its input buffers.
- **(7b) `compute_probs_loss_bce_chunk`**: **Invoked when `OperatingMode` is `BCE`.** A streamable kernel computing probabilities and partial BCE loss. It is designed to be **robust to memory padding**, using physical strides to correctly navigate its input buffers.

---

- **(8) `calculate_module_param_grads_chunk`**: A streamable kernel computing **partial** gradients for module weights and biases. It is designed to be **robust to memory padding**, using physical strides to navigate its input buffers. It computes the `(prob - target)` error signal on the fly.
- **(9) `backprop_error_to_hidden_chunk`**: A streamable kernel computing the **partial** upstream gradient for the hidden layer (`Grad_H`). It is designed to be **robust to memory padding**, using physical strides to navigate the module weight buffer.
- **(10) `calculate_chunk_temp_gradients`**: A streamable kernel computing **partial** gradients for the temperature parameters. It is designed to be **robust to memory padding**, using physical strides to correctly navigate its input buffers.

---

- **(11) `transpose_chunk`**: **(Generic Utility).** A streamable kernel whose purpose is **latency hiding**. It is enqueued on a per-chunk basis to overlap memory-bound transpose operations with compute-bound gradient calculations.
- **(12) `aggregate_*` kernels**: Generic, stateless kernel interface that serves as the engine for the **Recursive Halving Renderer**. It is invoked iteratively at each stage of the reduction tree to consolidate `K` partial results (`Level L`) into a single higher-level result (`Level L+1`). For `Grad_H`, this process produces an intermediate `Aggregated_Grad_H` buffer which requires its own separate reduction.
- **(13) `reduce_grad_h_over_modules`**: A specialized reduction kernel that sums the module-major `Aggregated Grad_H` buffer. It is designed to be **robust to memory padding**, using physical strides to correctly navigate the buffer and sum contributions across the logical module dimension, producing the final `Final_Grad_H`.
- **(14) `backprop_shared_weights_chunk`**: A streamable backpropagation kernel for shared layer weights, computing partial gradients for a batch chunk.
- **(15) `backprop_shared_biases_chunk`**: A streamable backpropagation kernel for shared layer biases, computing partial gradients for a batch chunk.
- **(16) `aggregate_*` kernels**: The same generic kernel interface, used by the **Recursive Halving Renderer** to consolidate partial gradients from the shared layer. It is invoked iteratively to reduce chunks from the streaming backpropagation phase (14, 15).
- **(17) `D2H Async Copy`**: A non-blocking Device-to-Host transfer of the `Final Probs` buffer, whose completion signals the `inference_event`.
- **(18) `adam_update`**: Generic optimizer kernel, invoked multiple times for different parameter groups. It accepts the global step `t` to guarantee numerical stability and relies on the host to provide **parameter, gradient, and momentum buffers with identical physical layouts** for correct operation.
- **(19) `clamp_temperatures`**: Final utility kernel for parameter constraint.
- **Host/Device Synchronization Contracts**:
  - `inference_event`: Guarantees the **complete, aggregated `Final Probs` tensor, representing results from all `Classifier Heads`**, is available on the host.
  - `final_batch_event`: Guarantees all device computations and parameter updates for the batch are complete.

### **Validation Scenarios**

- **Scenario: The Iris Case (Tiny Problem)**

  - **Insight:** Demonstrates **universality**. For a model with a single, small `Classifier Head`, the "always stream" design gracefully degrades. With `num_*_chunks=1`, the **Recursive Halving Renderer** simply executes a single reduction, becoming a near-zero-cost identity copy and proving efficient function at any scale.

- **Scenario: The Marathon (Massive `epochs`)**

  - **Insight:** Guarantees **long-term stability** by delegating the sensitive `beta**t` calculation to the `(18) adam_update` kernel, avoiding host-side precision loss and ensuring the optimizer is mathematically correct indefinitely.

- **Scenario: The Hydra (Massive `num_heads`)**

  - **Insight:** Validates the **scalability and intelligence of the Recursive Halving Renderer**. For a massive number of heads, the Host Orchestrator acts as a **Reduction Planner**. It assesses VRAM, chooses a suitable **Reduction Batch Size (`K`)**, and renders an efficient, multi-stage `log_K(N)` reduction tree over the module dimension, ensuring the GPU remains saturated without exhausting memory.

- **Scenario: The Behemoth (Massive `hidden_dim`)**

  - **Insight:** Proves **scalability through strategic compute/memory trade-offs**. When the intermediate `hidden_i` buffer is a bottleneck, the host chooses to **Cache** or **Recompute** it, guaranteeing that any network size can train.

- **Scenario: The Lexicon (Massive `output_classes`)**

  - **Insight:** Achieves **maximum hardware occupancy via the recursive renderer**. For a model with millions of classes, the Host Orchestrator acts as a **Reduction Planner**. It assesses VRAM, chooses a suitable **Reduction Batch Size (`K`)**, and renders an efficient, multi-stage `log_K(N)` reduction tree over the class dimension. The interleaved compute (8, 9, 10) and aggregation (`agg`) kernels ensure the GPU is fully occupied.

- **Scenario: The Data Tsunami (Massive `batch_size`)**

  - **Insight:** Validates the **two-phase streaming backpropagation strategy**. For large batches, backpropagation for the shared layer (14, 15) streams over the batch dimension, with partial results being fed into their own **Recursive Halving Renderer** (16), showing a sophisticated dataflow tailored to memory patterns.

- **Scenario: The Colossus (Holistic Stress Test)**

  - **Insight:** Tests the **synergy of all scaling strategies under compound memory pressure**. The host demonstrates its full intelligence by composing a multi-faceted plan. It applies chunking over the batch dimension (Phases 14-15) while simultaneously invoking the **Recursive Halving Renderer** for the module/class dimensions (Phases 5-13). It may also engage the `Recompute` strategy for activations, proving its ability to construct and execute complex, dynamic, and deeply memory-aware computational graphs.

- **Scenario: The Responsive Dashboard (Asynchronous Result Retrieval)**
  - **Insight:** Fulfills application-level needs by **decoupling forward-pass results from backward-pass latency**.
    - **Problem (ML Lens):** An application needs inference results from all `Classifier Heads` immediately.
    - **Solution (System Lens):** The `inference_event` fires as soon as the complete probability tensor is ready (17), while the GPU proceeds independently with backpropagation (14-16) and optimizer updates (18-19).
