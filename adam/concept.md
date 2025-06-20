### **Architectural Concept: A Unified, Memory-Aware Streaming Engine**

#### **Guiding Principles**

1.  **Primacy of Memory Strategy:** The singular goal of the host-side orchestration is to ensure the core computation executes in the fastest possible memory tier (Registers > Local > Global).
2.  **Modular, "Dumb" Kernels:** Kernels are simple, single-purpose modules. The architecture avoids complex branching ("smart" kernels) and monolithic designs in favor of composability.
3.  **Trust the Driver:** Simple kernels are composed into a logical Directed Acyclic Graph (DAG). The architecture trusts the OpenCL driver to handle low-level optimizations like kernel fusion. The number of nodes in the DAG is a non-goal.
4.  **Unified Dataflow:** The architecture follows a single, logical "always stream" pipeline. Performance and efficiency for problems of any scale is an emergent property of this unified design, not a separate, hard-coded path.

---

### **Core Architectural Components**

#### **1. Modular, Chunk-Based Compute Kernels**

The architecture is built upon a foundation of modular, reusable kernels that operate on "chunks" of a larger problem. To ensure mathematical consistency across all scales, any kernel that computes gradients produces **partial results** which are passed to the aggregation engine. The concept of an "in-loop" parameter update is explicitly avoided, as it would alter the training algorithm based on memory constraints.

#### **2. The Generic, Tiered Aggregation Engine**

The heart of the architecture is a powerful, generic aggregation engine that replaces all specialized reduction logic. The host orchestrator chooses **one** of three specialist kernels based *only* on the number of items (`N`) to be reduced. This engine is a pure, reusable component invoked multiple times within a single training step to consolidate various partial results (probabilities, losses, and gradients).

#### **3. Asynchronous Host Interaction**

The architecture decouples application-level latency from maximum GPU throughput by using an event-based synchronization model. Two key events per batch enable this:
1.  `inference_event`: Signals that final probabilities are available on the host.
2.  `final_batch_event`: Signals that all device-side computations for the batch are complete.

This allows a host application to act on inference results at the earliest possible moment while the GPU continues processing the backpropagation path at full efficiency.

#### **4. The Host Orchestrator**

The host logic is a sophisticated but straightforward orchestrator responsible for resource management and DAG construction. For each batch, it performs a series of strategic assessments:

1.  **Memory Assessment & Chunk Definition:** It compares the memory required for the complete problem against available VRAM to determine the optimal `num_chunks` and `chunk_size`.
2.  **Intermediate Activation Strategy:** Crucially, it manages the lifecycle of intermediate hidden activations (`hidden_i`). Based on memory pressure, it selects the optimal strategy to balance performance and scalability:
    *   **Cache (Space > Time):** If memory allows, it caches `hidden_i` buffers in VRAM for reuse during the backpropagation phase.
    *   **Recompute (Time > Space):** Under extreme memory pressure (e.g., a massive shared layer), it discards `hidden_i` after its initial use and recomputes it on the fly during backpropagation. This guarantees scalability for any problem size.
3.  **DAG Construction:** It builds the multi-phase computational graph, ensuring all partial results are correctly routed to an aggregation stage before being consumed by subsequent kernels.

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
    classDef path_a fill:#fef0e6,stroke:#C00000
    classDef path_b fill:#e9eef7,stroke:#2F5496

    %% Phase 0-2: Setup
    subgraph Phase 0-2: Host Setup & Global Params
        HL_0[Start Batch]:::host_logic --> HL_1["1. VRAM Budgeting & Chunking"]:::host_logic --> HL_2["2. Activation Lifecycle Strategy"]:::host_logic
        P_Shared[Shared Params]:::param; P_Exits[Exit Params]:::param; P_Temps[Temp Params]:::param
    end

    %% Phase 3: Universal Chunk Processing (Forward)
    subgraph "Phase 3: Universal Chunk Processing (Forward Pass & Partial Grads)"
        style "Phase 3: Universal Chunk Processing (Forward Pass & Partial Grads)" loop_box

        note_hidden["Note: hidden_i lifecycle<br/>(Cache or Recompute)<br/>managed by Host"]

        K4["(4) forward_pass"]:::kernel --> hidden_i[Hidden Activations<br/>Chunk 'i']:::data
        hidden_i --> K5["(5) compute_chunk_outputs"]:::kernel & K6["(6) calculate_chunk_gradients"]:::kernel
        note_hidden --. hidden_i

        K5 --> PARTIAL_Probs_i[PARTIAL Probs 'i']:::partial_data & PARTIAL_Loss_i[PARTIAL Loss 'i']:::partial_data & PARTIAL_Logits_i[PARTIAL Logits 'i']:::partial_data
        PARTIAL_Probs_i & PARTIAL_Logits_i --> K7["(7) calculate_chunk_temp_gradients"]:::kernel --> PARTIAL_Grad_Temps_i[PARTIAL Grad_Temps 'i']:::partial_data

        K6 --> PARTIAL_Grad_H_i[PARTIAL Grad_H 'i']:::partial_data
        K6 --> PARTIAL_Grad_ExitW_i[PARTIAL Grad_ExitW 'i']:::partial_data & PARTIAL_Grad_ExitB_i[PARTIAL Grad_ExitB 'i']:::partial_data
    end

    %% Phase 4: Primary Aggregation Engine
    subgraph Phase 4: Primary Aggregation Engine
        K8["<b>(8) Aggregate Kernel</b>"]:::host_logic
        PARTIAL_Probs_i & PARTIAL_Loss_i & PARTIAL_Grad_H_i & PARTIAL_Grad_ExitW_i & PARTIAL_Grad_ExitB_i & PARTIAL_Grad_Temps_i -- All Chunks --> K8
        K8 --> FINAL_Probs[Final Probs]:::final_data & FINAL_Loss[Final Loss]:::final_data & FINAL_Grad_H[Final Grad_H]:::final_data
        K8 --> FINAL_Grad_ExitW[Final Grad_ExitW]:::final_data & FINAL_Grad_ExitB[Final Grad_ExitB]:::final_data & FINAL_Grad_Temps[Final Grad_Temps]:::final_data
    end

    %% Phase 5: Streaming Shared Layer Backprop
    subgraph "Phase 5: Streaming Shared Layer Backprop"
        style "Phase 5: Streaming Shared Layer Backprop" loop_box
        Input_i[Input Chunk 'i']:::data --> K9["<b>(9) backprop_shared_chunk</b>"]:::kernel
        hidden_i --> K9
        FINAL_Grad_H -- slice --> K9
        K9 --> PARTIAL_Grad_SW_i[PARTIAL Grad_SW 'i']:::partial_data & PARTIAL_Grad_SB_i[PARTIAL Grad_SB 'i']:::partial_data
    end

    %% Phase 6: Final Aggregation
    subgraph Phase 6: Final Aggregation
        K10["<b>(10) Aggregate Kernel</b>"]:::host_logic
        PARTIAL_Grad_SW_i & PARTIAL_Grad_SB_i -- All Chunks --> K10
        K10 --> FINAL_Grad_SW[Final Grad_SW]:::final_data & FINAL_Grad_SB[Final Grad_SB]:::final_data
    end

    %% Phase 7: Finalization & Dispatch
    subgraph "Phase 7: Finalization & Dispatch"
        direction LR
        subgraph "A. Early Exit Path"
            style "A. Early Exit Path" path_a
            K_D2H["<b>(11) D2H Async Copy</b><br/>(Final Probs)"]:::data --> EV_Inference["<b>inference_event</b>"]:::sync_event
        end
        subgraph "B. Training Path (All Updates)"
            style "B. Training Path (All Updates)" path_b
            K12_shared["(12) adam_update (Shared)"]:::kernel; FINAL_Grad_SW & FINAL_Grad_SB --> K12_shared; K12_shared -- updates --> P_Shared
            K12_exits["(12) adam_update (Exits)"]:::kernel; FINAL_Grad_ExitW & FINAL_Grad_ExitB --> K12_exits; K12_exits -- updates --> P_Exits
            K12_temps["(12) adam_update (Temps)"]:::kernel; FINAL_Grad_Temps --> K12_temps; K12_temps -- updates --> P_Temps
            K12_temps --> K13["(13) clamp_temps"]:::kernel
            K13 --> EV_Final["<b>final_batch_event</b>"]:::sync_event
        end
    end

    %% Connections
    HL_1 --> K4; HL_2 --> note_hidden
    FINAL_Probs --> K_D2H
    EV_Inference --> Host_Act["Host Acts on<br/>Early Result"]:::host_logic
    EV_Final --> Host_Wait_Final["Host Blocks for<br/>Full Batch"]:::host_logic
```

#### **Kernel & Synchronization Contracts**

*   **(4-7) Chunk-Processing Kernels**: Kernels designed for the forward pass and initial gradient computation. Their contract is to operate on a single chunk of data and produce **partial** results for all downstream consumers (probabilities, losses, and gradients).
*   **(8, 10) `aggregate_*` kernels**: A generic, stateless kernel interface invoked by the host whenever a set of partial results must be consolidated. Its behavior is defined by a `reduction_mode_flag` (e.g., SUM, AVERAGE).
*   **(9) `backprop_shared_chunk`**: A streamable backpropagation kernel. Its contract is to compute the partial gradients for the shared layer's weights and biases (`PARTIAL_Grad_SW`, `PARTIAL_Grad_SB`) for a single chunk. It consumes the final aggregated `Grad_H` and the chunk's corresponding input features and hidden activations.
*   **(11) `D2H Async Copy`**: A non-blocking Device-to-Host transfer of the `Final Probs` buffer, whose completion signals the `inference_event`.
*   **(12) `adam_update`**: A generic optimizer kernel. To ensure mathematical consistency, the architecture mandates that this kernel is only invoked in the final phase, *after* all gradients for the entire batch have been computed and fully aggregated.
*   **(13) `clamp_temperatures`**: A final utility kernel for parameter constraint.
*   **Host/Device Synchronization Contracts**:
    *   `inference_event`: Guarantees that the `Final Probs` data is available on the host for consumption.
    *   `final_batch_event`: Guarantees that all device-side computations for the batch are complete and all parameters have been updated.

---

### **Validation Scenarios**

The architecture's unified dataflow robustly handles all scenarios without special-casing:

*   **Scenario: The Iris Case (Tiny, `num_chunks=1`)**
    *   **Behavior:** The host sets `num_chunks=1`. Both streaming loops run exactly once. Both `Aggregate Kernel` calls (Node 8 and 10) are dispatched as `aggregate_identity`, which is a trivial, near-zero-cost pointer swap or buffer copy. The resulting computational graph dispatched to the device is functionally identical to a traditional non-streaming fast path, proving no performance is sacrificed for small problems.

*   **Scenario: The Swarm (Many Exits, `num_chunks`=1000s)**
    *   **Behavior:** The host sets up long streams for both loops. The aggregation engine correctly uses `aggregate_local_reduce`, potentially in a hierarchical pattern, to sum the thousands of partial results. Critically, all parameter updates occur after final aggregation, ensuring the training algorithm is standard mini-batch gradient descent, not an unintended and hardware-dependent SGD variant. The system scales predictably.

*   **Scenario: The Behemoth (Few Exits, Fat Network, `num_chunks`=20)**
    *   **Behavior:** The host determines that the memory bottleneck is the hidden activation buffer. It streams along the batch dimension. It then makes a strategic choice: if VRAM is extremely tight, it will `Recompute` hidden activations during Phase 5; if there is sufficient intermediate space, it will `Cache` them after Phase 3. The architecture correctly uses a fast `aggregate_register_reduce` for the small number of exit gradients, demonstrating its ability to select the optimal memory strategy for each aggregation task independently.

*   **Scenario: The Live Dashboard (Low-Latency Reporting)**
    *   **Behavior:** The architecture's event-based design shines. The host application dispatches the full training batch, then immediately performs a non-blocking wait on the `inference_event`. The moment final probabilities are aggregated (end of Phase 4) and copied (Node 11), the host can fetch them and update a UI. This occurs in parallel while the GPU continues with the much more expensive backpropagation work (Phases 5-7), successfully decoupling reporting latency from training throughput.