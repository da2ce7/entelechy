# Averaging Ensembled Classifier: A Computational Organism

![Status: Stable](https://img.shields.io/badge/status-stable-brightgreen.svg)
![License: AGPL v3](https://img.shields.io/badge/License-AGPL%20v3-blue.svg)

This is not a static model; it is a computational organism. Born from the `Entelechy` framework's philosophy, it is an architecture that lives, breathes, and adapts. It does not merely execute commands; it senses its environment—the scale of the problem and the pressure on its memory—and makes a metabolic choice between raw speed and guaranteed survival. It is an engine forged to solve classification problems with principled elegance and profound resilience.

### The Three Pillars of Existence

The engine's vitality rests upon three foundational design principles, each a load-bearing pillar of the architecture.

#### I. Survival: The Primacy of Memory
The organism is not bound to a single execution model that fails when a problem exceeds the bounds of VRAM. Instead, it possesses a **metabolic switch**. Faced with a "data tsunami," it makes a strategic choice:
-   **Cache Mode:** When memory is plentiful, it operates at maximum velocity, caching intermediate results for the fastest possible learning.
-   **Recompute Mode:** When starved for memory, it gracefully transitions to a survival strategy, discarding and recomputing intermediate data on the fly. This guarantees its ability to process problems of literally any scale without failure, trading time for space in an elegant act of self-preservation.

#### II. Scale: The Indirection Contract
To command an ensemble of potentially thousands of classifier modules, the architecture rejects naive, memory-bound summation (`O(N)`). Instead, it commands a **Recursive Aggregation Engine**, a sophisticated, multi-stage reduction tree (`log_K(N)`) whose power is unlocked by its central nervous system: the **Indirection Contract**. By providing its kernels with a simple `offset_list` to gather scattered partial results, the engine avoids costly intermediate memory copies, maximizing available bandwidth for pure computation.

#### III. Stability: The Regulatory Funnel
The organism abandons simplistic, one-size-fits-all gradient clipping. It employs a sophisticated, **Policy-Driven Stabilization Strategy** that acts as an intelligent regulatory funnel:
-   It is more permissive at the leaves of its reduction tree, preserving the vital information in initial partial gradients.
-   It becomes progressively stricter as gradients are aggregated, ensuring the final, powerful sum remains stable and controllable.
-   It formally severs the concerns of **Algorithmic Regulation** (the user's desired `max_grad_norm`) from **Numerical Safety** (the hardware's physical limits), making it numerically robust by immutable design.

### The Heartbeat: A Stimulus-Response Cycle
The system's operation is formally separated into two phases, allowing it to natively support two fundamental modes of interaction with the world.

1.  **The "Act" Phase (Sensing):** Triggered by input data (`X`), the system performs a complete forward pass. This is a self-contained act of perception that produces a final, usable prediction and all intermediate state required for potential learning.
2.  **The "Learn" Phase (Adapting):** Triggered by the arrival of ground truth (`Y`), the system initiates backpropagation. It uses the state generated during a prior "Act" phase to update its internal parameters.

This decoupling is a core feature. It allows for both **sequential learning** (where the same `X` is used for `Act` and `Learn`) and **asynchronous learning** (where predictions are made on a live `X`, while learning occurs later on a different, historical `(X, Y)` pair).

The dataflow below illustrates the internal dependencies, where the Learn phase is contingent upon the state created by the Act phase.

```mermaid
graph TD
    %% --- Styling ---
    classDef phase fill:#f3e5f5,stroke:#7b1fa2,stroke-width:2px;
    classDef op fill:#e3f2fd,stroke:#1565c0,stroke-width:1.5px;
    classDef sync fill:#fffde7,stroke:#f57f17,stroke-width:2px,stroke-dasharray: 5 5;
    classDef data fill:#e8f5e9,stroke:#2e7d32;
    classDef data_in fill:#fff3e0,stroke:#ff9800,stroke-width:2px,stroke-dasharray: 8 4;

    %% --- External Inputs (Stimuli) ---
    Input_X["Input Data (X)"]:::data_in
    Input_Y["Ground Truth (Y)"]:::data_in

    %% --- Graph ---
    subgraph ActPhase ["Act Phase (Sensing)"]
        style ActPhase phase
        A1["(4,5) Forward Pass"]:::op --> D_Probs["Partial Probs"]:::data
        A1 --> D_Loss["Partial Loss"]:::data
    end

    Input_X --> ActPhase

    D_Probs & D_Loss --> S1["(14) Diagnostic Aggregation"]:::op
    S1 --> D_FinalProbs["Final Probs (to Host)"]:::data

    subgraph LearnPhase ["Learn Phase (Adapting)"]
        style LearnPhase phase
        L1["(8,9,10) Parallel Gradient Production"]:::op
        L1 --> L2["(11,19) Leaf-Level Clipping"]:::op
        L2 --> D_ClippedGrads["Clipped Partial Gradients"]:::data
    end

    %% --- The Critical Links ---
    Input_Y --> LearnPhase
    A1 --> L1

    subgraph AggUpdate ["Aggregation & Update"]
        style AggUpdate phase
        D_ClippedGrads --> M1["(15,20) Recursive Clip-Aggregation"]:::op
        D_ClippedGrads --> M2["(13,16) Specialized Grad_H Reduction"]:::op
        M1 & M2 --> S2["Batch-wide Sync Point"]:::sync
        S2 --> U1["(21,24,25) Normalize & Adam Update"]:::op
    end
```

### A Crucible of Constraint: The Power of OpenCL 1.2
This architecture was forged in a crucible of deliberate constraint: to target **OpenCL 1.2 *without* relying on Atomics**. This restriction rendered the simple path of `atomic_add` impassable. It *forced* the design to confront the problem of parallelism head-on, giving rise to the more sophisticated, elegant, and ultimately more robust parallel reduction patterns that form the very heart of the engine. This choice ensures maximum hardware compatibility, allowing the organism to thrive in the widest possible range of computational environments.

### Multi-Backend Execution
The engine executes through a **plan-as-data-structure** model: a shared orchestration layer constructs an immutable, backend-neutral execution plan, which each backend renders using its native execution model.

| Backend | Status | Execution Primitives |
| :--- | :--- | :--- |
| **OpenCL** | Reference implementation | `clEnqueueNDRange` + `cl.Event` synchronization |
| **CPU** | ✅ **Implemented** | SIMD-vectorized C kernels (`libcpu_kernels.so`) + persistent thread pool via `pool_dispatch_and_wait` |
| **Vulkan** | ✅ **Implemented** | SPIR-V compute shaders via `vkCmdDispatch` + pipeline barriers |

The CPU backend requires zero additional Python dependencies beyond `ctypes` (stdlib). It supports AVX-512, AVX2, SSE2, ARM NEON, and a scalar fallback, auto-detecting the optimal ISA at compile time. See [`CPU_BACKEND.md`](./CPU_BACKEND.md) for the full architecture.

---

### The Canonical Archives
This document provides the spirit of the architecture. To understand its flesh and bone, its laws and its logic, you must consult the canonical sources:

*   **The Philosophical Blueprint:** For the unabridged design philosophy, data contracts, validation scenarios, and the full computational graph, see [1: `CONCEPT.md`](./CONCEPT.md).
*   **The Implementation Blueprint:** For concrete data structures, module layout, build system, multi-backend design, test strategy, and the phase-gated migration plan, see [2: `DESIGN.md`](./DESIGN.md).
*   **The Book of Law:** For the inviolable Host-Device interface, parameter naming conventions, and memory layout contracts, see [3: `CONTRACT.md`](./CONTRACT.md).
*   **The Kernel Declarations:** For the ground-truth C-level function signatures that are the final authority for all kernel launches, see [`kernels/kernels.cl.h`](./kernels/kernels.cl.h).
*   **The CPU Architecture:** For the SIMD + multi-core execution model, threading design, reduction engine, and per-kernel analysis, see [`CPU_BACKEND.md`](./CPU_BACKEND.md).
*   **The Vulkan Architecture:** For the explicit GPU compute model with pre-compiled SPIR-V pipelines, see [`VULKAN_BACKEND.md`](./VULKAN_BACKEND.md).
