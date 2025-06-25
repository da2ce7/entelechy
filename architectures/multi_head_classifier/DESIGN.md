## **Architectural Concept: A Unified, Memory-Aware Streaming Classification Engine (Revision 2)**

### **Guiding Principles**

1.  **Primacy of Memory Strategy:** The singular goal of the host-side orchestration is to ensure the core computation executes in the fastest possible memory tier (Registers > Local > Global). This principle drives all design decisions, prioritizing memory efficiency over computational complexity.
2.  **Modular, "Dumb" Kernels:** Kernels are simple, single-purpose modules. The architecture avoids complex branching ("smart" kernels) and monolithic designs in favor of composability. Where a complex data transformation is a critical-path bottleneck unsolved by generic tools, a **specialized, single-purpose kernel** will be employed. This specialist kernel remains "dumb"—stateless and reliant on the host for all contextual parameters.
3.  **Trust the Driver:** Simple kernels are composed into a logical Directed Acyclic Graph (DAG). The architecture trusts the OpenCL driver to handle low-level optimizations like kernel fusion. The number of nodes in the DAG is a non-goal, emphasizing adaptability over rigid structure.
4.  **Unified Execution Model:** All workflows follow Act (forward pass) then Learn (backpropagation) sequencing, manifesting as either **Sequential Execution Mode**—where Act-Learn phases execute contiguously for pre-labeled batches—or **Event-Triggered Execution Mode**—where Learn-phase execution awaits an external readiness signal post-Act. This split-phase approach ensures consistency across all use cases.
5.  **Architectural Hierarchy:** This system is developed under a strict hierarchy of artifacts to ensure conceptual integrity:
    - **1. Design Document (This Document):** The highest authority and source of truth for conceptual correctness.
    - **2. Kernel Header Contract:** The binding technical contract between host and device, resonating deeply with the design document.
    - **3. Host Code Implementation:** The lowest authority, rigorously conforming to the kernel header's contract. The header is never modified to suit the host code; the host code always yields to the contract.
6.  **No Silent Monoliths:** All executions manifest externally as an Act/Learn split, even if phases are contiguous, ensuring consistency and enabling inter-phase optimization. This eliminates silent assumptions of single-pass execution, reinforcing the architecture’s phased nature.

---

### **Core Architectural Components**

#### **1. Modular, Chunk-Based Compute Kernels**

The architecture is built upon a foundation of modular, reusable kernels that operate on "chunks" of a larger problem. Work is segmented across multiple dimensions—such as the number of modules, batch size, and number of classes—as dictated by resource constraints. Each `Classifier Module`, representing the system implementation of a machine learning 'Head,' is configured with an **Operating Mode** (`CCE` for single-label or `BCE` for multi-label tasks).

#### **2. The Recursive, Tiered Aggregation Engine**

At the core of the architecture lies a powerful aggregation engine that implements a **recursive, multi-stage reduction tree**. Governed by a host-configurable parameter, `K`—the **Reduction Batch Size**—this engine defines the width of the parallel kernel front at each reduction stage. This `log_K(N)` strategy breaks the `O(N)` memory dependency of a flat aggregation model, enabling scalability to problems of arbitrary size. The engine employs three tiers of `aggregate_*` kernels (Identity, Register, Local) selected by the Host Orchestrator based on the number of partials being reduced at any given stage.

#### **3. Asynchronous Host Interaction**

The lifetime of an input batch spans two event-delimited phases:

- **Act Phase**: Concludes when complete, aggregated forward-pass results (`Final Probs`) become available on the host, signaled by the `inference_event`.
- **Learn Phase**: Triggered by the host when ground truth is ready, concluding when all device-side computations complete, signaled by the `final_batch_event`.

#### **4. The Host Orchestrator & Execution Policies**

The host logic serves as a sophisticated orchestrator, responsible for resource management and DAG construction. For each batch, it performs strategic assessments to optimize execution:

1.  **Memory Assessment & Chunk Definition:** The orchestrator determines an optimal chunking strategy, defining `num_module_chunks`, `num_batch_chunks`, and `num_class_chunks` to balance compute and memory demands.
2.  **Activation Lifecycle & Streaming:** The orchestrator manages the lifecycle of intermediate hidden activations (`hidden_i`), which are always processed in **batch chunks**. It selects between:
    - **Cache Strategy (Space > Time):** Retains `hidden_i` chunks in VRAM for reuse during the Learn phase.
    - **Recompute Strategy (Time > Space):** Discards `hidden_i` chunks post-Act to free VRAM, regenerating them on-demand during the Learn phase. This enables true streaming backpropagation.
3.  **SIMD-Aware Weight Layout:** The orchestrator transforms shared layer weights into a SIMD-friendly "Struct of Arrays" (SoA) layout before enqueuing kernel (4).
4.  **DAG Construction & Parallel Dispatch:** The orchestrator constructs the multi-phase computational graph, enqueuing kernels according to the selected mode and chunking strategy.
5.  **Reduction Planning & Rendering:** The orchestrator acts as a **Reduction Planner**, analyzing problem size and VRAM to set an optimal **Reduction Batch Size (`K`)**. It renders the full `log_K(N)` reduction tree, managing intermediate buffer lifecycles.
6.  **Partial Result Placement & Indexing:** When dispatching `N` parallel kernels, the orchestrator provides each invocation with a unique **Placement Index** (`flat_tile_index`), guaranteeing that all `N` partials are written without data loss or race conditions.
7.  **Phase Staging & Optimizer State Management:** The orchestrator governs phase intervals through event-based sequencing. For the Adam optimizer, the host is responsible for computing the `beta1**t` and `beta2**t` bias correction terms in high precision (e.g., `double`) and passing the final numerical values to the update kernel, avoiding on-device precision loss and underflow during long training runs.

---

### **Architectural Blueprint & Data Contracts for a Multi-Head Classifier (Revision 2)**

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
    classDef specialized_kernel fill:#e1f5fe,stroke:#0288d1,stroke-width:2px
    classDef permute_kernel fill:#FBE5D6,stroke:#ED7D31,stroke-width:2px
    classDef full_intermediate fill:#FCE4D6,stroke:#F4B183,stroke-width:2px
    classDef parallel_group fill:#f5f5f5,stroke:#333,stroke-width:2px
    classDef grad_path_a fill:#f3e5f5,stroke:#8e24aa
    classDef grad_path_b fill:#e1f5fe,stroke:#0288d1
    classDef grad_path_c fill:#e8f5e9,stroke:#2e7d32
    classDef fused_kernel_box fill:#f0f4c3,stroke:#afb42b,stroke-width:2px,stroke-dasharray: 5 2
    classDef logical_step fill:#ffffff,stroke:#757575,stroke-width:1px,stroke-dasharray: 2 2

    subgraph "Annotation Keys"
      direction LR
      TermNote["Note: 'Module' refers to the system<br/>implementation of an ML 'Head'."]
      subgraph "Fused Operation Key"
          direction LR
          LS_Alpha["α"]:::logical_step; LS_Beta["β"]:::logical_step
          LS_Gamma["γ"]:::logical_step; LS_Delta["δ"]:::logical_step; LS_Epsilon["ε"]:::logical_step
          Desc_Alpha["Mat-Mul + Bias"]
          Desc_Beta["Prob Calc + Loss Calc"]
          Desc_Gamma["Weight Grad + Bias Grad"]
          Desc_Delta["Mat-Mul + Bias + ReLU"]
          Desc_Epsilon["Stable Softmax + Prob Calc + Loss Calc"]
      end
      subgraph "Temporal Execution Keys & Buffer Lifetime"
          direction LR
          Solid_Arrow["→"]:::logical_step; Dashed_Arrow["-.->"]:::logical_step;  Cache_Note["[C]"]:::data; Recompute_Note["[R]"]:::data
          Desc_Solid["Immediate Dependency"]; Desc_Dashed["Host-Controlled Delay"]; Desc_Cache["Cache Supported"]; Desc_Recompute["Recompute Supported"]
      end
    end

    %% Phase 0-3: Setup
    subgraph "Phase 0-3: Host Setup & Global Params"
        HL_0[Start Batch]:::host_logic --> HL_1["1. VRAM Budgeting & Chunking"]:::host_logic --> HL_2["2. Read Mode, Manage Activations,<br/>& Plan Reductions"]:::host_logic
        HL_2 --> HL_Placement["6. Partial Result Placement & Indexing"]:::host_logic
        P_Shared[Shared Params]:::param; P_ClassifierModule[Classifier Module Params]:::param; P_Temps[Temp Params]:::param;
        P_Adam_Host[Host-Computed Adam State<br/>(beta1_t, beta2_t)]:::param
        Targets[Targets]:::param; SampleMask[Sample Mask]:::param
    end

    %% Phase 4-18: Act Phase
    subgraph "PHASE 1: ACT"
        subgraph "Phase 4: Shared Layer Forward Pass (Batch-Chunked)"
            style "Phase 4: Shared Layer Forward Pass (Batch-Chunked)" loop_box
            note_hidden["Note: hidden_i lifecycle<br/>always chunked by batch,<br/>[C] or [R] policy by Host"]
            HL_2 --> note_hidden

            subgraph K4["(4) forward_pass"]:::fused_kernel_box
                direction LR
                L4["Mat-Mul + Bias + ReLU δ"]:::logical_step
            end
            HL_1 --> L4; SampleMask --> L4
            L4 --> hidden_i[Hidden Activations<br/>Chunk 'i' [C,R]]:::data
            L4 --> hidden_mask[Hidden Mask<br/>Chunk 'i' [C,R]]:::data
        end

        subgraph "Phase 5-7: Forward Pass for Classifier Head (as a Module)"
            subgraph K5["(5) compute_logits_chunk"]:::fused_kernel_box
                direction LR
                L5["Mat-Mul + Bias α"]:::logical_step
            end
            hidden_i & hidden_mask --> L5
            L5 --> Full_Logits[Full Logits Buffer [C,R]]:::full_intermediate

            HL_3["(7) Host Selects Path<br/>based on Operating Mode (CCE/BCE)"]:::host_logic
            Full_Logits --> HL_3

            subgraph "Operating Mode: CCE (Softmax Path) [Fused]"
              style "Operating Mode: CCE (Softmax Path) [Fused]" path_cce
              K6_cce["<b>(6) compute_probs_loss_cce_chunk</b><br/>[Fused Kernel]"]:::fused_kernel_box
              HL_3 & Full_Logits & P_Temps & SampleMask & Targets --> K6_cce
              K6_cce --> FINAL_Loss_CCE[FINAL CCE Loss (no agg needed)]:::final_data
              K6_cce --> PARTIALS_Probs["Collection of N<br/>PARTIAL Probabilities [C,R]"]:::partial_data
            end

            subgraph "Operating Mode: BCE (Sigmoid Path)"
                style "Operating Mode: BCE (Sigmoid Path)" path_bce
                subgraph K7_bce["(7) compute_probs_loss_bce_chunk"]:::fused_kernel_box
                    direction LR
                    L7_prob["Prob Calc β"]:::logical_step --> L7_loss["Loss Calc β"]:::logical_step
                end
                HL_3 & Full_Logits --> L7_prob
                SampleMask --> L7_loss
                L7_loss --> PARTIALS_Loss_BCE["Collection of N<br/>PARTIAL BCE Loss"]:::partial_data
                L7_prob --> PARTIALS_Probs
            end
        end

        subgraph "Phase 18: Asynchronous Result Retrieval"
            direction LR
            K18["<b>(18) D2H Async Copy</b><br/>(Final Probs)"]:::data --> EV_Inference["<b>inference_event</b>"]:::sync_event
        end
    end

    FINAL_Probs --> K18
    EV_Inference --> Host_Act["Host Acts on<br/>Full Forward Result"]:::host_logic

    %% Phase 8-20: Learn Phase
    subgraph "PHASE 2: LEARN"
        Host_Act -.->|Host trigger| K8_Start["K8 Start Trigger"]:::host_logic

        subgraph "Phase 8-10: Per-Tile Streamable Gradient Computation"
            style "Phase 8-10: Per-Tile Streamable Gradient Computation" loop_box
            note_grad_loop["Note: Executes for each tile<br/>in the Execution Grid (N total tiles)"]:::host_logic
            HL_Placement --> note_grad_loop

            subgraph "Parallel Gradient Path (Chunk-Based)"
                style "Parallel Gradient Path (Chunk-Based)" parallel_group

                subgraph "Gradients for Classifier Module"
                    style "Gradients for Classifier Module" grad_path_a
                    subgraph K8["<b>(8) calculate_module_param_grads_chunk</b>"]:::fused_kernel_box
                        L8_w["Weight Grad Calc γ"]:::logical_step; L8_b["Bias Grad Calc γ"]:::logical_step
                    end
                    PARTIALS_Probs & Targets & SampleMask & hidden_i --> L8_w
                    L8_w --> PARTIALS_Grad_ModW["Collection of N<br/>PARTIAL Grad_ModW"]:::partial_data
                    PARTIALS_Probs & Targets & SampleMask --> L8_b
                    L8_b --> PARTIALS_Grad_ModB["Collection of N<br/>PARTIAL Grad_ModB"]:::partial_data
                end

                subgraph "Upstream Hidden Gradients"
                    style "Upstream Hidden Gradients" grad_path_b
                    K9["<b>(9) backprop_error_to_hidden_chunk</b>"]:::kernel
                    PARTIALS_Probs & Targets & P_ClassifierModule & SampleMask --> K9
                    K9 --> PARTIALS_Grad_H_AoS["Collection of N<br/>PARTIAL Grad_H (AoS)"]:::partial_data
                end

                subgraph "Temperature Gradients"
                    style "Temperature Gradients" grad_path_c
                    K10["<b>(10) calculate_chunk_temp_gradients</b>"]:::kernel
                    PARTIALS_Probs & Full_Logits & Targets & P_Temps & SampleMask --> K10
                    K10 --> PARTIALS_Grad_Temps["Collection of N<br/>PARTIAL Grad_Temps"]:::partial_data
                end
            end
        end

        subgraph "Phase 11-12: Grad_H Permutation"
             K12["<b>(12) gather_and_permute_grad_h</b><br/>(Specialized Permutation)"]:::permute_kernel
             PARTIALS_Grad_H_AoS --> K12
             K12 --> Permuted_Grad_H_SoA["Permuted Grad_H<br/>(B*H, M Layout)"]:::full_intermediate
             K11["(11) transpose_chunk<br/>(General Utility, not on critical path)"]:::transpose_kernel
        end

        subgraph "Phase 13: Recursive Reduction Engine"
             style "Phase 13: Recursive Reduction Engine" loop_box
             K_Recursive_13["<b>(13) Recursive Reduction Engine</b><br/>(Processes N partials in batches of K)"]:::host_logic
        end

        PARTIALS_Probs & PARTIALS_Loss_BCE & PARTIALS_Grad_ModW & PARTIALS_Grad_ModB & PARTIALS_Grad_Temps -- "Input: Buffers of N discrete partials" --> K_Recursive_13

        K_Recursive_13 --> FINAL_Probs[Final Probs]:::final_data & FINAL_BCE_Loss[Final BCE Loss]:::final_data
        K_Recursive_13 --> FINAL_Grad_ModW[Final Grad_ModW]:::final_data & FINAL_Grad_ModB[Final Grad_ModB]:::final_data & FINAL_Grad_Temps[Final Grad_Temps]:::final_data

        subgraph "Phase 14: Specialized Grad_H Reduction"
            K14["<b>(14) reduce_grad_h_over_modules</b><br/>(Specialized Reduction)"]:::specialized_kernel
            Permuted_Grad_H_SoA --> K14
            K14 --> FINAL_Grad_H[Final Grad_H]:::final_data
        end

        subgraph "Phase 15-16: Streaming Shared Layer Backprop"
            style "Phase 15-16: Streaming Shared Layer Backprop" parallel_group
            Input_i[Input Chunk 'i']:::data & SampleMask --> K15["<b>(15) backprop_shared_weights_chunk</b>"]:::kernel
            K15 --> PARTIALS_Grad_SW_i["Collection of N<br/>PARTIAL Grad_SW 'i'"]:::partial_data
            hidden_i --> K15 & K16
            FINAL_Grad_H -- slice --> K15 & K16
            SampleMask --> K16["<b>(16) backprop_shared_biases_chunk</b>"]:::kernel
            K16 --> PARTIALS_Grad_SB_i["Collection of N<br/>PARTIAL Grad_SB 'i'"]:::partial_data
        end

        subgraph "Phase 17: Recursive Shared Gradient Aggregation"
            direction LR
            style "Phase 17: Recursive Shared Gradient Aggregation" loop_box
            K_Recursive_17["(agg) ...<br/>Iterative Reduction"]:::host_logic
        end
        PARTIALS_Grad_SW_i & PARTIALS_Grad_SB_i -- "Input: Buffers of N discrete partials" --> K_Recursive_17
        K_Recursive_17 --> FINAL_Grad_SW[Final Grad_SW]:::final_data & FINAL_Grad_SB[Final Grad_SB]:::final_data

        subgraph "Phase 19-20: Training Path (All Updates)"
            direction LR
            K19_shared["(19) adam_update (Shared)"]:::kernel; FINAL_Grad_SW & FINAL_Grad_SB & P_Adam_Host --> K19_shared; K19_shared -- updates --> P_Shared
            K19_module["(19) adam_update (ClassifierModule)"]:::kernel; FINAL_Grad_ModW & FINAL_Grad_ModB & P_Adam_Host --> K19_module; K19_module -- updates --> P_ClassifierModule
            K19_temps["(19) adam_update (Temps)"]:::kernel; FINAL_Grad_Temps & P_Adam_Host --> K19_temps; K19_temps -- updates --> P_Temps
            K19_temps --> K20["<b>(20) clamp_temps</b>"]:::kernel
            K20 --> EV_Final["<b>final_batch_event</b>"]:::sync_event
        end
    end

    EV_Final --> Host_Wait_Final["Host Blocks for<br/>Full Batch"]:::host_logic
```

---

### Final Kernel & Synchronization Contracts (Revision 2)

#### **The Partial Renderer Kernel Contract**

All kernels designated as "Partial Renderers" must accept a unique `flat_tile_index` integer parameter from the host. This index is used to calculate the write offset within its designated output buffer, ensuring each partial result is placed in its correct, discrete slot. This contract applies to: **(7), (8), (9), (10), (15), (16)**.

#### **Act Phase Kernels**

- **(4) `forward_pass`**: Computes hidden activations **for a specified batch chunk**.
  - **Contract:** Applies a (Weights \* Input + Bias + ReLU) transform, producing an `hidden_i` buffer for a single batch chunk. Supports both Cache and Recompute strategies.
- **(5) `compute_logits_chunk`**: A streamable kernel computing raw logits from a **batch chunk of hidden activations**.
  - **Contract:** Executes a (Weights \* Input + Bias) transform, yielding `Full_Logits` that support caching and recomputation.
  - **Contract:** Performs the **complete, temperature-aware, numerically stable Softmax calculation internally, guaranteeing mathematical correctness and memory locality.** The sequence of operations is indivisible:
    1. Scales logits by temperature: `scaled = logits / T`.
    2. Finds the maximum of the scaled logits for stability.
    3. Computes probabilities using the derived maximum.
    4. If `Targets` are present, computes final CCE loss.
  - **Outputs:** Adheres to the Partial Renderer Contract for `partial_probs_out`. The `final_loss_out` uses a scatter-write optimization as no aggregation is needed for CCE loss.
- **(7) `compute_probs_loss_bce_chunk`**: Invoked when `Operating Mode` is `BCE`. A streamable kernel computing probabilities and partial BCE loss.
  - **Contract:** Generates probabilities and partial loss from `Full_Logits`. Adheres to the Partial Renderer Contract for both `partial_loss_out` and `partial_probs_out`.
- **(18) `D2H Async Copy`**: A non-blocking Device-to-Host transfer of `Final Probs`, signaling completion via `inference_event`.

#### **Learn Phase Kernels**

- **(8) `calculate_module_param_grads_chunk`**: Computes partial gradients for module parameters from a **batch chunk of hidden activations**.
  - **Contract:** Utilizes `Partial_Probs`, `Targets`, and a **chunk** of `hidden_i` to generate partial `Grad_ModW` and `Grad_ModB`. Adheres to the Partial Renderer Contract.
- **(9) `backprop_error_to_hidden_chunk`**: Computes the partial upstream gradient (`Grad_H`) for a tile of the problem.
  - **Contract:** Backpropagates errors using `Partial_Probs` and `Targets`. Adheres to the Partial Renderer Contract.
- **(10) `calculate_chunk_temp_gradients`**: A streamable kernel computing partial gradients for temperature parameters.
  - **Contract:** Derives temperature gradients from `Partial_Probs`, `Full_Logits`, and `Targets`. Adheres to the Partial Renderer Contract.
- **(12) `gather_and_permute_grad_h`**: **[Specialized Kernel]** A global permutation kernel that gathers scattered `Grad_H` partials into a single, contiguous buffer.
  - **Contract:** Reads from the module-major `PARTIALS_Grad_H_AoS` collection and writes to a single batch-major `Permuted_Grad_H_SoA` buffer, preparing it for efficient reduction. This is a mandatory step before Node (14).
- **(14) `reduce_grad_h_over_modules`**: **[Specialized Kernel]** A specialized reduction kernel summing the `Permuted_Grad_H_SoA` buffer.
  - **Contract:** Consolidates gradient contributions across all modules, producing the `Final_Grad_H`.
- **(15) `backprop_shared_weights_chunk`**: Computes partial gradients for shared weights from a **batch chunk of hidden activations**.
  - **Contract:** Processes `Final_Grad_H` and a **chunk** of `hidden_i`. Adheres to the Partial Renderer Contract.
- **(16) `backprop_shared_biases_chunk`**: Computes partial gradients for shared biases from a **batch chunk of hidden activations**.
  - **Contract:** Processes `Final_Grad_H` and a **chunk** of `hidden_i`. Adheres to the Partial Renderer Contract.
- **(19) `adam_update`**: Generic optimizer kernel, invoked once per complete parameter group.
  - **Contract:** Applies the Adam update to an **entire parameter buffer in a single dispatch**. It is a stateless numerical function that receives `grad`, `m1`, `m2`, `learning_rate`, `epsilon`, and the **pre-computed, high-precision bias correction terms (`beta1_t`, `beta2_t`)** from the host. This contract forbids internal slicing and enforces host-side responsibility for long-term numerical stability.
- **(20) `clamp_temperatures`**: Final utility kernel constraining temperature parameters.

#### **Phase-Agnostic Utility Kernels**

- **(11) `transpose_chunk`**: A streamable utility kernel for transposing a contiguous 2D matrix slice.
  - **Phase Compatibility Note:** A general-purpose tool, not on the primary backpropagation path for `Grad_H`.
- **(13), (17) `aggregate_*` kernels**: A suite of stateless reduction kernels serving as the **Recursive, Tiered Aggregation Engine**.

---

#### **Validation Scenarios (Revised & Validated)**

- **Scenario: The Iris Case** - **Insight:** Unchanged. The unified model remains efficient for small, sequential tasks.
- **Scenario: The Real-Time Trader** - **Insight:** Unchanged. The system excels at event-driven workflows with delayed learning.
- **Scenario: The Marathon (Massive `epochs`)** - **Insight:** **[Corrected]** Guarantees **long-term stability** by offloading the `beta**t` calculation to the host, which can use high-precision arithmetic (`double`) to compute the bias correction terms before passing them to the kernel. This prevents on-device floating-point underflow and ensures the optimizer remains mathematically correct indefinitely.
- **Scenario: The Hydra (Massive `num_heads`)** - **Insight:** Unchanged. The recursive renderer efficiently handles aggregation over the module dimension.
- **Scenario: The Behemoth (Massive `hidden_dim`)** - **Insight:** Unchanged. The host can trade compute for memory by recomputing `hidden_i` chunks as needed.
- **Scenario: The Lexicon (Massive `output_classes`)** - **Insight:** Unchanged. The recursive renderer efficiently handles aggregation over the class dimension.
- **Scenario: The Data Tsunami (Massive `batch_size`)** - **Insight:** **[Corrected & Validated]** Validates the **two-phase streaming backpropagation strategy**. Gradient kernels (**8, 15, 16**) now operate on **chunks** of the `hidden_i` buffer, not the monolithic whole. This enables true streaming over the batch dimension, allowing the system to process arbitrarily large batches with a fixed VRAM footprint by caching or recomputing `hidden_i` chunks on demand.
- **Scenario: The Colossus (Holistic Stress Test)** - **Insight:** **[Validated]** Tests the synergy of all scaling strategies. The host composes a dynamic plan, **streaming over the batch dimension** (producing `hidden_i` chunks), invoking the **Recursive Halving Renderer** for module/class dimensions, and using the Recompute strategy. The system is now proven to be robust under compound memory pressure.
