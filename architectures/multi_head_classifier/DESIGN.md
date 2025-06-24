
## **Architectural Concept: A Unified, Memory-Aware Streaming Classification Engine**

---

### **Guiding Principles**

1.  **Primacy of Memory Strategy:** The singular goal of the host-side orchestration is to ensure the core computation executes in the fastest possible memory tier (Registers > Local > Global). This principle drives all design decisions, prioritizing memory efficiency over computational complexity.
2.  **Modular, "Dumb" Kernels:** Kernels are simple, single-purpose modules. The architecture avoids complex branching ("smart" kernels) and monolithic designs in favor of composability, ensuring flexibility and ease of optimization.
3.  **Trust the Driver:** Simple kernels are composed into a logical Directed Acyclic Graph (DAG). The architecture trusts the OpenCL driver to handle low-level optimizations like kernel fusion. The number of nodes in the DAG is a non-goal, emphasizing adaptability over rigid structure.
4.  **Unified Execution Model:** All workflows follow Act (forward pass) then Learn (backpropagation) sequencing, manifesting as either **Sequential Execution Mode**—where Act-Learn phases execute contiguously for pre-labeled batches—or **Event-Triggered Execution Mode**—where Learn-phase execution awaits an external readiness signal post-Act. This split-phase approach ensures consistency across all use cases, enabling predictable behavior and optimized inter-phase pipelining.
5.  **Architectural Hierarchy:** This system is developed under a strict hierarchy of artifacts to ensure conceptual integrity:
    - **1. Design Document (This Document):** The highest authority and source of truth for conceptual correctness.
    - **2. Kernel Header Contract:** The binding technical contract between host and device, resonating deeply with the design document.
    - **3. Host Code Implementation:** The lowest authority, rigorously conforming to the kernel header's contract. The header is never modified to suit the host code; the host code always yields to the contract.
6.  **No Silent Monoliths:** All executions manifest externally as an Act/Learn split, even if phases are contiguous, ensuring consistency and enabling inter-phase optimization. This design choice eliminates silent assumptions of single-pass execution, reinforcing the architecture’s phased nature.

---

### **Core Architectural Components**

#### **1. Modular, Chunk-Based Compute Kernels**

The architecture is built upon a foundation of modular, reusable kernels that operate on "chunks" of a larger problem. Work is segmented across multiple dimensions—such as the number of modules, batch size, and number of classes—as dictated by resource constraints. Each `Classifier Module`, representing the system implementation of a machine learning 'Head,' is configured with an **Operating Mode** (`CCE` for single-label or `BCE` for multi-label tasks). This configuration instructs the Host Orchestrator on the specific computational path to construct, leveraging these kernels for tailored execution.

#### **2. The Recursive, Tiered Aggregation Engine**

At the core of the architecture lies a powerful aggregation engine that implements a **recursive, multi-stage reduction tree**. Rather than generating all `N` partial results before aggregation, this model employs a divide-and-conquer strategy to keep the GPU saturated while minimizing VRAM usage. Governed by a host-configurable parameter, `K`—termed the **Reduction Batch Size**—this engine defines the width of the parallel kernel front at each reduction stage. The process unfolds as follows:

1.  The Host Orchestrator renders the base `N` partial results in batches of `K`.
2.  After each batch of `K` partials is computed, they are immediately reduced by an `aggregate_*` kernel into a single "Level 1" intermediate result.
3.  This sequence repeats `N/K` times, transforming a large problem of `N` "Level 0" results into a smaller set of `N/K` "Level 1" results.
4.  The host recursively applies this logic to the "Level 1" results, reducing them in batches of `K` into "Level 2" results, continuing until a single final tensor emerges.

To further optimize memory access patterns, the engine employs a tiered selection of reduction kernels based on the number of partials (`N`) being aggregated at any given stage:

- **Tier 0 (N=1):** A zero-copy `aggregate_identity` pass-through is used, simply renaming a buffer handle or performing a direct copy if required.
- **Tier 1 (Small N):** An `aggregate_register_reduce` kernel is dispatched. This kernel is optimized for a small number of inputs that can be held and summed entirely in the GPU's registers, avoiding any local memory overhead.
- **Tier 2 (Large N):** The workhorse `aggregate_local_reduce` kernel is used. It performs a highly parallelized reduction using shared local memory, making it efficient for consolidating a large number of partial results.

The Host Orchestrator is responsible for selecting the optimal kernel at each step of the reduction, ensuring that even the aggregation logic itself adheres to the "Primacy of Memory" principle. This `log_K(N)` strategy breaks the `O(N)` memory dependency of a flat aggregation model, enabling scalability to problems of arbitrary size.

#### **3. Asynchronous Host Interaction**

The lifetime of an input batch spans two event-delimited phases:

- **Act Phase**: Concludes when complete, aggregated forward-pass results (`Final Probs`) become available on the host, signaled by the `inference_event`.
- **Learn Phase**: Triggered by the host when ground truth is ready, concluding when all device-side computations complete, signaled by the `final_batch_event`.

This phased structure enables scenarios where predictions are acted upon before label acquisition, allowing the host to pause between events to conserve VRAM or interleave unrelated tasks, enhancing throughput.

#### **4. The Host Orchestrator & Execution Policies**

The host logic serves as a sophisticated yet straightforward orchestrator, responsible for resource management and DAG construction. For each batch, it performs strategic assessments to optimize execution:

1.  **Memory Assessment & Chunk Definition:** The orchestrator compares the memory required for the complete problem against available device memory, determining an optimal chunking strategy. This includes defining `num_module_chunks`, `num_batch_chunks`, and `num_class_chunks` to balance compute and memory demands.
2.  **Operating Mode & Activation Lifecycle:** The orchestrator reads the `Classifier Module`'s configured `Operating Mode` (`CCE` or `BCE`) to shape the computational graph. It manages the lifecycle of intermediate hidden activations (`hidden_i`), selecting between:
    - **Cache Strategy (Space > Time):** Retains intermediate buffers in VRAM for reuse during the Learn phase, minimizing recomputation when memory allows.
    - **Recompute Strategy (Time > Space):** Discards intermediates post-Act to free VRAM, regenerating them during the Learn phase under memory pressure.
3.  **SIMD-Aware Weight Layout:** To exploit vector processing capabilities and ensure coalesced memory access, the orchestrator transforms shared layer weights into a SIMD-friendly "Struct of Arrays" (SoA) layout before enqueuing kernel (4). Weights shift from a logical `(hidden_dim, input_dim)` matrix to a physical `(hidden_dim/SIMD_WIDTH, input_dim, SIMD_WIDTH)` buffer, embodying the **Primacy of Memory Strategy** as a non-negotiable contract with the `forward_pass` kernel.
4.  **DAG Construction & Parallel Dispatch:** The orchestrator constructs the multi-phase computational graph, enqueuing kernels according to the selected mode and chunking strategy. Logically independent tasks are enqueued back-to-back without synchronization, fostering parallel workloads.
5.  **Reduction Planning & Rendering:** For tasks requiring aggregation over a large number of chunks (`N`), the orchestrator acts as a **Reduction Planner**, analyzing problem size and VRAM to set an optimal **Reduction Batch Size (`K`)**. It renders the full `log_K(N)` reduction tree, orchestrating iterative stages of computation and aggregation while managing intermediate buffer lifecycles.
6.  **Phase Staging Policies:** The orchestrator governs phase intervals through:
    - **Temporal Sequencing:** Backward kernels are emitted only after `inference_event`, ensuring phase separation.
    - **Transition Policy Selection:**
      - **Sequential Execution Mode:** Learn kernels are enqueued immediately post-`inference_event` for pre-labeled data, maintaining contiguous execution while preserving the Act/Learn split.
      - **Event-Triggered Execution Mode:** Learn kernels await an external readiness signal for live data, accommodating real-world delays.
    - **Buffer Lifetime Optimization:** The choice between Cache or Recompute strategies for intermediates (`hidden_i`, `Partial_Probs`) hinges on delay duration and VRAM pressure, balancing efficiency and resource availability.
    - **Task Interleaving:** Phases from distinct batches progress concurrently via priority queues, maximizing GPU utilization.

Enforcing an Act/Learn split even for pre-labeled data guarantees consistency in event handling across all use cases, ensuring predictable behavior and enabling optimized inter-phase pipelining.

---

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
        subgraph "Temporal Execution Keys & Buffer Lifetime"
            direction LR
            subgraph " "
                Solid_Arrow["→"]:::logical_step; Dashed_Arrow["-.->"]:::logical_step
            end
            subgraph " "
                Desc_Solid["Immediate Dependency"]
                Desc_Dashed["Host-Controlled Delay or Recompute"]
            end
            subgraph " "
                Cache_Note["[C]"]:::data; Recompute_Note["[R]"]:::data
            end
            subgraph " "
                Desc_Cache["Cache Supported"]
                Desc_Recompute["Recompute Supported"]
            end
        end
    end

    %% Phase 0-3: Setup
    subgraph "Phase 0-3: Host Setup & Global Params"
        HL_0[Start Batch]:::host_logic --> HL_1["1. VRAM Budgeting & Chunking"]:::host_logic --> HL_2["2. Read Mode, Manage Activations,<br/>& Plan Reductions"]:::host_logic
        P_Shared[Shared Params]:::param; P_ClassifierModule[Classifier Module Params]:::param; P_Temps[Temp Params]:::param;
        P_Step[Global Step 't']:::param
        Targets[Targets]:::param
        SampleMask[Sample Mask]:::param
    end

    %% Phase 4-17: Act Phase
    subgraph "PHASE 1: ACT"
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
            L4c --> hidden_i[Hidden Activations<br/>Chunk 'i' [C,R]]:::data
            L4c --> hidden_mask[Hidden Mask]:::data
        end

        subgraph "Phase 5-7: Forward Pass for Classifier Head (as a Module)"
            subgraph K5["(5) compute_logits_chunk"]:::fused_kernel_box
                direction LR
                L5a["Mat-Mul α"]:::logical_step --> L5b["Bias Add α"]:::logical_step
            end
            hidden_i & hidden_mask --> L5a
            L5b --> Full_Logits[Full Logits Buffer [C,R]]:::full_intermediate

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
                L7a_prob --> PARTIAL_Probs[PARTIAL Probabilities [C,R]]:::partial_data
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

        subgraph "Phase 17: Asynchronous Result Retrieval"
            direction LR
            K17["<b>(17) D2H Async Copy</b><br/>(Final Probs)"]:::data --> EV_Inference["<b>inference_event</b>"]:::sync_event
        end
    end

    FINAL_Probs --> K17
    EV_Inference --> Host_Act["Host Acts on<br/>Full Forward Result"]:::host_logic

    %% Phase 8-19: Learn Phase
    subgraph "PHASE 2: LEARN"
        Host_Act -.->|Host trigger| K8_Start["K8 Start Trigger"]:::host_logic

        subgraph "Phase 8-11: Per-Tile Streamable Gradient Computation"
            style "Phase 8-11: Per-Tile Streamable Gradient Computation" loop_box
            note_grad_loop["Note: Executes for each tile<br/>in the Execution Grid (N total tiles)"]:::host_logic

            subgraph "Parallel Gradient Path (Chunk-Based)"
                style "Parallel Gradient Path (Chunk-Based)" parallel_group

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
        end

        subgraph "Phase 12: Recursive Reduction Engine"
             style "Phase 12: Recursive Reduction Engine" loop_box
             K_Recursive_12["<b>(12) Recursive Reduction Engine</b><br/>(Processes N partials in batches of K)"]:::host_logic
        end

        PARTIAL_Probs & PARTIAL_Loss_BCE & PARTIAL_Grad_ModW & PARTIAL_Grad_ModB & PARTIAL_Grad_Temps & PARTIAL_Grad_H_SoA -- "Streamed Partial Results" --> K_Recursive_12

        K_Recursive_12 --> FINAL_Probs[Final Probs]:::final_data & FINAL_BCE_Loss[Final BCE Loss]:::final_data & AGG_Grad_H_SoA["Aggregated Grad_H<br/>(B*H, M Layout)"]:::full_intermediate
        K_Recursive_12 --> FINAL_Grad_ModW[Final Grad_ModW]:::final_data & FINAL_Grad_ModB[Final Grad_ModB]:::final_data & FINAL_Grad_Temps[Final Grad_Temps]:::final_data

        subgraph "Phase 13: Specialized Grad_H Reduction"
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

        subgraph "Phase 18-19: Training Path (All Updates)"
            direction LR
            K18_shared["(18) adam_update (Shared)"]:::kernel; FINAL_Grad_SW & FINAL_Grad_SB --> K18_shared; K18_shared -- updates --> P_Shared
            P_Step --> K18_shared
            K18_module["(18) adam_update (ClassifierModule)"]:::kernel; FINAL_Grad_ModW & FINAL_Grad_ModB --> K18_module; K18_module -- updates --> P_ClassifierModule
            P_Step --> K18_module
            K18_temps["(18) adam_update (Temps)"]:::kernel; FINAL_Grad_Temps --> K18_temps; K18_temps -- updates --> P_Temps
            K18_temps --> K19["<b>(19) clamp_temps</b>"]:::kernel
            K19 --> EV_Final["<b>final_batch_event</b>"]:::sync_event
        end
    end

    EV_Final --> Host_Wait_Final["Host Blocks for<br/>Full Batch"]:::host_logic
```

---

### Final Kernel & Synchronization Contracts

#### **Phase 0-3: Host Setup & Global Params**

- **(H0) `start_batch`**: Initiates batch processing by signaling the beginning of a new computational cycle.
  - **Contract:** Triggers the orchestration sequence, preparing the system for subsequent setup tasks. No device buffers are accessed at this stage.
  - **Phase Compatibility Note:** Executes prior to both Act and Learn phases, serving as the entry point for batch orchestration.
- **(H1) `vram_budgeting_chunking`**: Assesses available VRAM and determines the optimal chunking strategy for the batch.
  - **Contract:** Calculates memory requirements against device capabilities, defining `num_module_chunks`, `num_batch_chunks`, and `num_class_chunks` to balance compute and memory demands. Ensures all subsequent kernel enqueues respect these constraints.
  - **Phase Compatibility Note:** Executes prior to both Act and Learn phases, establishing the memory allocation framework.
- **(H2) `read_mode_manage_activations`**: Reads the `Classifier Module`'s operating mode and plans the activation lifecycle.
  - **Contract:** Configures the computational graph based on the `Operating Mode` (`CCE` or `BCE`) and selects between **Cache** (retaining `hidden_i` in VRAM) or **Recompute** (discarding and regenerating `hidden_i`) strategies, optimizing for VRAM or compute trade-offs.
  - **Phase Compatibility Note:** Executes prior to both Act and Learn phases, shaping the graph and buffer management strategy.
- **(H3) `select_operating_path`**: Selects the computational path based on the `Classifier Module`'s operating mode (`CCE` or `BCE`) for the forward pass.
  - **Contract:** Directs the enqueueing of either `CCE` (Softmax) or `BCE` (Sigmoid) kernels, ensuring the correct forward-pass logic is applied for subsequent Act phase operations.
  - **Phase Compatibility Note:** Executes at the transition to the Act phase, finalizing the forward-pass configuration.

#### **Act Phase Kernels**

- **(4) `forward_pass`**: Computes hidden activations for a chunk of the input batch.
  - **Contract:** Applies a (Weights \* Input + Bias) transform followed by a ReLU activation, producing `hidden_i` buffers that can be cached or recomputed based on host policy.
- **(5) `compute_logits_chunk`**: A streamable kernel computing raw logits. It is robust to memory padding, using physical strides to navigate module-level weight and bias buffers correctly.
  - **Contract:** Executes a (Weights \* Input + Bias) transform, yielding `Full_Logits` buffers that support both caching and recomputation strategies.
- **(6) `reduce_logits_for_softmax`**: Invoked when `Operating Mode` is `CCE`. A synchronization kernel computing stable Softmax normalization terms, robust to memory padding via physical strides.
  - **Contract:** Processes `Full_Logits` to produce `Softmax Denominators`, ensuring numerical stability for subsequent probability calculations.
- **(17) `D2H Async Copy`**: A non-blocking Device-to-Host transfer of the `Final Probs` buffer, signaling completion via the `inference_event`.
  - **Contract:** Transfers the aggregated `Final Probs` tensor to the host, enabling immediate action post-Act phase.

#### **Learn Phase Kernels**

- **(8) `calculate_module_param_grads_chunk`**: A streamable kernel computing partial gradients for module weights and biases, robust to memory padding via physical strides. It computes the `(prob - target)` error signal on the fly.
  - **Contract:** Utilizes `Partial_Probs` and `Targets` to generate gradient contributions, requiring ground truth presence.
- **(9) `backprop_error_to_hidden_chunk`**: A streamable kernel computing the partial upstream gradient for the hidden layer (`Grad_H`), robust to memory padding via physical strides.
  - **Contract:** Backpropagates errors using `Partial_Probs` and `Targets`, producing gradients for upstream layers.
- **(10) `calculate_chunk_temp_gradients`**: A streamable kernel computing partial gradients for temperature parameters, robust to memory padding via physical strides.
  - **Contract:** Derives temperature gradients from `Partial_Probs`, `Full_Logits`, and `Targets`, integrating ground truth data.
- **(13) `reduce_grad_h_over_modules`**: A specialized reduction kernel summing the module-major `Aggregated Grad_H` buffer, robust to memory padding via physical strides.
  - **Contract:** Consolidates gradient contributions across modules, producing the final `Final_Grad_H`.
- **(14) `backprop_shared_weights_chunk`**: A streamable backpropagation kernel for shared layer weights, computing partial gradients for a batch chunk.
  - **Contract:** Processes `Final_Grad_H` and `hidden_i` to update shared weights, requiring Learn-phase data.
- **(15) `backprop_shared_biases_chunk`**: A streamable backpropagation kernel for shared layer biases, computing partial gradients for a batch chunk.
  - **Contract:** Updates shared biases using `Final_Grad_H` and `hidden_i`, executed post-Act.
- **(18) `adam_update`**: Generic optimizer kernel, invoked multiple times for different parameter groups. It accepts the global step `t` for numerical stability and relies on the host to provide parameter, gradient, and momentum buffers with identical physical layouts.
  - **Contract:** Applies Adam optimization to update parameters, integrating Learn-phase gradients.
- **(19) `clamp_temperatures`**: Final utility kernel constraining temperature parameters.
  - **Contract:** Enforces bounds on temperature values post-optimization, concluding the Learn phase.

#### **Phase-Dependant Utility Kernels**

- **(7a) `compute_probs_loss_cce_chunk`**: Invoked when `Operating Mode` is `CCE`. A streamable kernel computing probabilities and final CCE loss, robust to memory padding via physical strides.
  - **Contract:** Computes probabilities from `Full_Logits` and `Softmax Denominators`, with loss calculation deferred to the Learn phase if ground truth is unavailable during Act.
  - **Phase Compatibility Note:** Designed for Act Phase execution for probability calculation; Learn Phase for loss calculation if ground truth is provided.
- **(7b) `compute_probs_loss_bce_chunk`**: Invoked when `Operating Mode` is `BCE`. A streamable kernel computing probabilities and partial BCE loss, robust to memory padding via physical strides.
  - **Contract:** Generates probabilities from `Full_Logits`, with loss calculation deferred to the Learn phase pending ground truth availability.
  - **Phase Compatibility Note:** Designed for Act Phase execution for probability calculation; Learn Phase for loss calculation if ground truth is provided.

#### **Phase-Agnostic Utility Kernels**

- **(11) `transpose_chunk`**: A streamable kernel designed for latency hiding, overlapping memory-bound transpose operations with compute-bound gradient calculations.
  - **Contract:** Rearranges data layouts (e.g., `Partial_Grad_H`) as needed across phases, supporting both Act and Learn operations.
  - **Phase Compatibility Note:** Designed for both Act and Learn Phase execution (phase-agnostic utility).
- **(12) `aggregate_*` kernels**: A suite of stateless reduction kernels serving as the engine for the **Recursive, Tiered Aggregation Engine**. The host selects the optimal kernel based on the number of partials (`N`) to reduce.
  - **Contract (Tier 0 - `aggregate_identity`):** For `N=1`, performs a no-op or direct copy, serving as the terminal base case for the reduction tree.
  - **Contract (Tier 1 - `aggregate_register_reduce`):** For small `N` (e.g., N <= 32), consolidates partials using register-only operations for minimal overhead.
  - **Contract (Tier 2 - `aggregate_local_reduce`):** For large `N`, uses shared local memory for scalable, parallel reduction of many partials.
    All kernels are robust to memory padding and can be configured for summation or averaging. They are designed for both Act and Learn Phase execution as a phase-agnostic utility.
- **(16) `aggregate_*` kernels**: The same generic kernel interface, consolidating partial gradients from the shared layer during streaming backpropagation (14, 15).
  - **Contract:** Reduces chunked gradients into final forms, specific to Learn-phase aggregation.
  - **Phase Compatibility Note:** Designed for Learn Phase execution.

#### **Host/Device Synchronization Contracts**

- **`inference_event`**: Guarantees the complete, aggregated `Final Probs` tensor, representing results from all `Classifier Heads`, is available on the host, marking the conclusion of the Act phase.
- **`final_batch_event`**: Guarantees all device computations and parameter updates for the batch are complete, signaling the end of the Learn phase.
- **Logical Sequencing of Events**: For any batch, `inference_event` precedes `final_batch_event`. The host can schedule non-dependent work between events. Learn-phase kernels remain unenqueued until ground truth resolution, with device-side buffers persisting only until explicitly released by the host.

---

#### Validation Scenarios

- **Scenario: The Iris Case (Sequential Execution Mode Validation)**

  - **Description:** A classic supervised learning task using the Iris dataset, where a batch of pre-labeled flower measurements is processed in a single training step. The system executes the Act phase (forward pass) and immediately triggers the Learn phase (backpropagation) with no delay.
  - **Validation Focus:** Measures the end-to-end latency of processing the batch, ensuring the split-phase model's overhead is minimal (e.g., less than 1% additional runtime compared to a monolithic baseline). This confirms efficiency for small, high-throughput problems.
  - **Key Insight:** Proves the success of the **unified execution model**. By demonstrating negligible overhead (<1%), it confirms that the Act/Learn split is not a costly abstraction but a foundational primitive that gracefully and efficiently handles both fully-labeled sequential batches and event-triggered asynchronous workflows under a single, consistent paradigm.

- **Scenario: The Real-Time Trader (Event-Triggered Execution Mode Validation)**

  - **Description:** A high-frequency trading system where market data streams in continuously. The system must make instant predictions (Act phase) to execute trades, while trade outcomes (labels) arrive after a delay (e.g., 500ms). The Learn phase is triggered only when labels become available.
  - **Validation Focus:** Assesses the system's ability to release VRAM after the Act phase, freeing GPU resources during the delay. When the Learn phase starts, it recomputes intermediates (e.g., `hidden_i`, `Partial_Probs`) efficiently for backpropagation. Success is gauged by balancing recomputation costs against memory availability.
  - **Key Insight:** Demonstrates the architecture's strength in real-time, event-driven scenarios, efficiently managing resources and enabling delayed learning without sacrificing performance.

- **Scenario: The Marathon (Massive `epochs`)**

  - **Insight:** Guarantees **long-term stability** by delegating the sensitive `beta**t` calculation to the `(18) adam_update` kernel, avoiding host-side precision loss and ensuring the optimizer remains mathematically correct indefinitely.

- **Scenario: The Hydra (Massive `num_heads`)**

  - **Insight:** Validates the **scalability and intelligence of the Recursive Halving Renderer**. For a massive number of heads, the Host Orchestrator acts as a **Reduction Planner**, assessing VRAM, choosing a suitable **Reduction Batch Size (`K`)**, and rendering an efficient `log_K(N)` reduction tree over the module dimension.

- **Scenario: The Behemoth (Massive `hidden_dim`)**

  - **Insight:** Proves **scalability through strategic compute/memory trade-offs**. When the `hidden_i` buffer is a bottleneck, the host opts to **Cache** or **Recompute** it, ensuring any network size can train.

- **Scenario: The Lexicon (Massive `output_classes`)**

  - **Insight:** Achieves **maximum hardware occupancy** via the recursive renderer. For millions of classes, the Host Orchestrator plans a `log_K(N)` reduction tree over the class dimension, interleaving compute and aggregation kernels for full GPU utilization.

- **Scenario: The Data Tsunami (Massive `batch_size`)**

  - **Insight:** Validates the **two-phase streaming backpropagation strategy**. For large batches, backpropagation streams over the batch dimension, feeding partial results into a **Recursive Halving Renderer**, optimizing memory patterns.

- **Scenario: The Colossus (Holistic Stress Test)**
  - **Insight:** Tests the **synergy of all scaling strategies** under compound memory pressure. The host composes a dynamic plan, chunking the batch dimension, invoking the **Recursive Halving Renderer** for module/class dimensions, and using the `Recompute` strategy for activations.
