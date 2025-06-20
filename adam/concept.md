### **Architectural Concept: A Unified, Memory-Aware Streaming Engine**

#### **Guiding Principles**

1.  **Primacy of Memory Strategy:** The singular goal of the host-side orchestration is to ensure the core computation executes in the fastest possible memory tier (Registers > Local > Global).
2.  **Modular, "Dumb" Kernels:** Kernels are simple, single-purpose modules. The architecture avoids complex branching ("smart" kernels) and monolithic designs in favor of composability.
3.  **Trust the Driver:** Simple kernels are composed into a logical Directed Acyclic Graph (DAG). The architecture trusts the OpenCL driver to handle low-level optimizations like kernel fusion. The number of nodes in the DAG is a non-goal.
4.  **Unified Dataflow:** The architecture follows a single, logical "always stream" pipeline. Performance and efficiency for problems of any scale is an emergent property of this unified design, not a separate, hard-coded path.

---

### **Core Architectural Components**

#### **1. Modular, Chunk-Based Compute Kernels**

The architecture is built upon a foundation of modular, reusable kernels that operate on "chunks" of a larger problem.

*   **Unified Chunk-Processing Kernels:** The core of the forward and backward pass is handled by a small set of unified kernels. These accept `chunk_id` and `chunk_size` parameters to operate on a slice of the data and use a runtime flag to select problem-specific math (e.g., CCE vs. BCE), eliminating the need for template-based code generation.
    *   `forward_pass`
    *   `compute_chunk_outputs`
    *   `calculate_chunk_gradients` (Outputs both final gradients for the chunk and partial gradients for shared layers).
    *   `calculate_chunk_temp_gradients`

*   **Generic Optimizer Kernel:** The `adam_update` kernel is a generic module that can be applied to any parameter buffer. This allows it to be used both for streaming updates inside the chunk-processing loop and for global parameter updates after aggregation is complete.

*   **Partial Result Data Structures:** The output of the chunk-processing loop is a set of temporary buffers holding intermediate data like `Partial_Probs`, `Partial_Loss`, and `Partial_Grad_H`. These structures are the inputs to the aggregation engine.

#### **2. The Generic, Tiered Aggregation Engine**

The heart of the architecture is a powerful, generic aggregation engine that replaces all specialized reduction logic.

*   **Tiered Kernel Implementations:** The host orchestrator chooses **one** of three specialist kernels based *only* on the number of items (`N`) to be reduced. These kernels can aggregate any type of partial result data.
    1.  `aggregate_identity`: **(if N = 1)**. A trivial-cost kernel that effectively performs a pointer swap or a fast buffer copy. It handles the case where no computational aggregation is needed.
    2.  `aggregate_register_reduce`: **(if 1 < N ≤ 64)**. A high-performance kernel where a single work-group performs the entire reduction in private registers. **Memory Strategy: Registers.**
    3.  `aggregate_local_reduce`: **(if N > 64)**. A scalable workhorse kernel where work-groups collaborate to reduce their assigned inputs using `__local` memory. **Memory Strategy: Local Memory.**

*   **Note on Hierarchical Reduction:** An extreme number of items (e.g., N > 100K) doesn't require a fourth kernel type. It is handled by the host as a two-stage pattern using the `aggregate_local_reduce` kernel twice: once on the full input to an intermediate buffer, and a second time on the small intermediate buffer.

#### **3. The Host Orchestrator**

The host logic is a sophisticated but straightforward orchestrator that builds the computational DAG for each batch.

*   **Unified DAG Construction:** The host operates on a single, unified dataflow model. It dynamically constructs the DAG for each batch based on resource availability.
*   **Orchestration Logic:**
    1.  **Memory Assessment:** The host calculates the memory required to process the entire problem at once.
    2.  **Chunk Determination:** It compares the required memory to the available VRAM to determine `num_chunks` and `chunk_size`. If the entire problem fits, `num_chunks` is simply `1`.
    3.  **Build Streaming DAG:** The host enters a `for` loop from `0` to `num_chunks-1` and enqueues the chunk-processing kernels for each chunk of data.
    4.  **Build Aggregation DAG:** After the loop, the host inspects `num_chunks` and enqueues the single, correct aggregation kernel from the Tiered Engine.
    5.  **Build Finalization DAG:** The host enqueues the final kernels (`finalize_backprop_activation`, etc.) to operate on the now-aggregated global results.

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

    %% Phase 0: Host Orchestrator Logic
    subgraph Phase 0: Host Orchestrator (Prepares DAG)
        direction TB
        HL_0[Start Batch]:::host_logic
        HL_0 --> HL_1["<b>1. VRAM Budgeting</b><br/><i>out: hidden_lifecycle_mode</i>"]:::host_logic
        HL_1 --> HL_2["<b>2. Chunk Definition</b><br/><i>out: num_chunks, chunk_size</i>"]:::host_logic
        HL_2 --> HL_3["<b>3. Aggregation Strategy</b><br/><i>out: agg_kernel_choice, agg_mode_flags</i>"]:::host_logic
    end

    %% Phase 1: Global Parameters
    subgraph Phase 1: Global Device Parameters
        P_Shared[Shared Params<br/>(W, B)]:::param
        P_Exits[Exit Params<br/>(W, B)]:::param
        P_Temps[Temp Params]:::param
    end

    %% Phase 2: Conditional Pre-computation (Host Choice)
    subgraph Phase 2: Pre-computation (if hidden_lifecycle_mode == PRECOMPUTE)
        K4_pre["<b>(4) forward_pass</b><br/>(Full Batch)"]:::kernel
        FULL_Hidden[FULL Batch<br/>Hidden Activations]:::data
        HL_1 -- "PRECOMPUTE" --> K4_pre
        K4_pre --> FULL_Hidden
    end

    %% Phase 3: The Universal Chunk Processing Loop
    subgraph "Phase 3: Universal Chunk Processing Loop (Host iterates 'num_chunks' times)"
        style "Phase 3: Universal Chunk Processing Loop (Host iterates 'num_chunks' times)" loop_box
        direction TB

        subgraph "A. Chunk-local Forward Pass"
            K4_loop["<b>(4) forward_pass</b><br/>(Chunked Batch)"]:::kernel
            HL_1 -- "STREAM" --> K4_loop
            hidden_i[Hidden Activations<br/>Chunk 'i']:::data
            K4_loop --> hidden_i
            FULL_Hidden -. slice .-> hidden_i
        end

        subgraph "B. Chunk-local Outputs & Gradients"
            K5["(5) compute_chunk_outputs"]:::kernel
            K6["(6) calculate_chunk_gradients"]:::kernel
            K7["(7) calculate_chunk_temp_gradients"]:::kernel
        end
        hidden_i --> K5 & K6

        %% Outputs from K5
        PARTIAL_Probs_i[Partial Probs 'i']:::partial_data
        PARTIAL_Loss_i[Partial Loss 'i']:::partial_data
        PARTIAL_Logits_i[Partial Logits 'i']:::partial_data
        K5 --> PARTIAL_Probs_i & PARTIAL_Loss_i & PARTIAL_Logits_i

        %% Dataflow for K6 and K7
        PARTIAL_Probs_i --> K6 & K7
        PARTIAL_Logits_i --> K7

        %% Outputs from K6 and K7
        PARTIAL_Grad_H_i[PARTIAL Grad_H 'i']:::partial_data
        GRAD_ExitW_i[Grad ExitW 'i']:::data
        GRAD_ExitB_i[Grad ExitB 'i']:::data
        K6 --> PARTIAL_Grad_H_i & GRAD_ExitW_i & GRAD_ExitB_i
        PARTIAL_Grad_Temps_i[PARTIAL Grad_Temps 'i']:::partial_data
        K7 --> PARTIAL_Grad_Temps_i

        subgraph "C. Streaming Parameter Update (Optional)"
            K12_exits["<b>(12) adam_update</b>"]:::kernel
            GRAD_ExitW_i & GRAD_ExitB_i --> K12_exits
            P_Exits -- IN/OUT for chunk 'i' --> K12_exits
        end
    end

    %% Phase 4: Aggregation
    subgraph Phase 4: Aggregation Engine (Runs after loop completes)
        K8["<b>(8) Aggregate Kernel</b><br/>(Chosen by Host, configured by mode flags)"]:::host_logic
        HL_3 --> K8
        PARTIAL_Probs_i & PARTIAL_Loss_i & PARTIAL_Grad_H_i & PARTIAL_Grad_Temps_i -- All Chunks --> K8

        FINAL_Probs[Final Probs]:::final_data
        FINAL_Loss[Final Loss]:::final_data
        FINAL_Grad_H[Final Grad_H]:::final_data
        FINAL_Grad_Temps[Final Grad_Temps]:::final_data
        K8 --> FINAL_Probs & FINAL_Loss & FINAL_Grad_H & FINAL_Grad_Temps
    end

    %% Phase 5: Final Global Backprop & Updates
    subgraph Phase 5: Final Global Operations
        K9["(9) finalize_backprop_activation"]:::kernel
        K10["(10) calculate_dense_layer_gradients"]:::kernel
        K11["(11) backprop_input_gradient"]:::kernel
        K12_shared["<b>(12) adam_update</b> (Shared)"]:::kernel
        K12_temps["<b>(12) adam_update</b> (Temps)"]:::kernel
        K13["(13) clamp_temperatures"]:::kernel

        FULL_Hidden -- from Phase 2 --> K9
        FINAL_Grad_H --> K9
        Grad_PreAct[Grad PreActivation]:::data
        K9 --> Grad_PreAct
        Grad_PreAct --> K10 & K11
        FINAL_Grad_SW[Final Grad SharedW]:::final_data
        FINAL_Grad_SB[Final Grad SharedB]:::final_data
        K10 --> FINAL_Grad_SW & FINAL_Grad_SB
        FINAL_Grad_SW & FINAL_Grad_SB --> K12_shared
        K12_shared -- updates --> P_Shared

        FINAL_Grad_Temps --> K12_temps
        K12_temps -- updates --> P_Temps
        P_Temps --> K13
    end
```

#### **Kernel Data Contracts**

*   **(4) `forward_pass`**: If in `STREAM` mode, operates on a slice of the batch. The host provides correct offsets.
*   **(5) `compute_chunk_outputs`**: `exit_*_buf` pointers are offset by the host to the start of the current chunk's parameters. `partial_*_out` buffers are written to with an offset based on `chunk_id`.
*   **(6) `calculate_chunk_gradients`**: Produces two types of output: **partial** `grad_h` (for aggregation) and **final** `grad_exit_*` (for immediate, streaming update).
*   **(7) `calculate_chunk_temp_gradients`**: Produces `partial_grad_temps` destined for the aggregation engine.
*   **(8) `aggregate_*` kernels**: A generic kernel interface is launched **multiple times** by the host, once for each data type that needs aggregation (Loss, Grad_H, etc.). The `reduction_mode_flag` (e.g., `SUM` vs. `AVERAGE`) configures the operation.
*   **(9) `finalize_backprop_activation`**: Requires `full_hidden_buf`, which must have been produced by the Phase 2 pre-computation step. This is a key dependency managed by the host.
*   **(10-13)**: The final utility kernels (`calculate_dense_layer_gradients`, `backprop_input_gradient`, `adam_update`, `clamp_temperatures`) operate on the `FINAL_*` data buffers produced by the aggregation engine.

---

### **Validation Scenarios**

The architecture's unified dataflow robustly handles all scenarios without special-casing:

*   **Scenario: The Iris Case (Tiny, `num_chunks=1`)**
    *   **Behavior:** The host sets `num_chunks=1`. The streaming loop runs once. The aggregation choice is `aggregate_identity`. The resulting DAG is computationally identical to a traditional non-streaming fast path, proving no performance is lost for small cases.

*   **Scenario: The Swarm (Many Exits, `num_chunks`=1000s)**
    *   **Behavior:** The host sets up a long stream. The aggregation choice is `aggregate_local_reduce`, potentially in a hierarchical pattern orchestrated by the host, correctly using a scalable memory strategy.

*   **Scenario: The Behemoth (Few Exits, Fat Network, `num_chunks`=20)**
    *   **Behavior:** The host sets up a short stream based on the shared layer's memory needs. The aggregation choice is `aggregate_register_reduce`, using the fastest possible memory strategy for a small number of items.