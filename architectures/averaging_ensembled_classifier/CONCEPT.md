## **Architectural Concept: A Unified, Memory-Aware Streaming Classification Engine (Revision 8)**

### **Guiding Principles**

#### 1. **Architectural Elegance Feedback**

Optimization pressure that violates these core principles shall be interpreted an useful signal that indicates incomplete architectural modeling, thus is no-justification for exceptions. The system must evolve its formal abstractions to subsume valid optimizations as first-class primitives, never compromise its contracts.

> **When emergent efficiency gains contradict current constraints:**
>
> 1. **Suspend implementation** of the optimization
> 2. **Formalize the pattern** as a documented architectural primitive
> 3. **Reify the optimization** through revised contracts & DAG extensions
>
> _Example:_ Discovering a kernel fusion opportunity that violates interface verifiability triggers:
>
> - Halting ad-hoc fusion attempts
> - Modeling fused operation in Design Document as new DAG node type
> - Defining strict fusion contracts in Kernel Headers
> - Implementing through host-controlled parameter switches, not hidden logic

#### 2. **Primacy of Memory Strategy**

The singular goal of the host-side orchestration is to ensure the core computation executes in the fastest possible memory tier (Registers > Local > Global). This principle drives all design decisions, prioritizing memory efficiency over computational complexity.

> **Data organization is inherently informed by the target hardware’s memory hierarchy and access patterns.** The system prioritizes layouts that harmonize with fundamental theoretical constraints of the hardware class (e.g., alignment principles, memory bank theory, access granularity axioms) to optimize computational pathways for throughput and scalable efficiency. This principle elevates hardware-class-aware design as a first-class concern, ensuring implementations achieve maximal bandwidth utilization and latency hiding through generalized hardware paradigms, not ad-hoc device-specific optimizations.
> **Storage precision is a first-class bandwidth lever.** The system decomposes numeric precision into three independent roles — storage, compute, and state — reflecting the distinct optimization targets of bandwidth, arithmetic fidelity, and long-term stability. Narrowing the storage format reduces buffer sizes and transfer bandwidth proportionally (FP16 halves FP32's footprint; FP8 halves again) without constraining the arithmetic precision used for computation or the precision of persistent optimizer state. This decoupling is a direct expression of the Primacy of Memory Strategy: data is stored in the narrowest format that preserves sufficient information for the downstream operation, and widened to the arithmetic format only at the point of computation.

**Storage-role FP64 is permitted but not optimised.** FP64 storage doubles FP32's bandwidth cost with no storage-compression benefit. The architecture permits storage-role FP64 for configurations where uniformity is preferred over bandwidth (e.g., `PrecisionConfig.float64()` for validation reference), but does not optimize for it. The expected production configurations place FP64 only in compute or state roles — e.g., `PrecisionConfig.mixed_f32_f64_state()` (FP32 storage, FP32 compute, FP64 state) for extended-stability training.

**FP8 is the ultimate expression of the Primacy of Memory Strategy.** FP8 (E4M3 or E5M2) storage achieves 4× bandwidth compression vs. FP32 and 2× vs. FP16. However, FP8's 3-bit mantissa (E4M3) or 2-bit mantissa (E5M2) is insufficient for compute or state roles: reductions saturate in one stage, and EMA updates round to zero for high β values. The architecture therefore permits FP8 **only in the storage role**, enforcing this constraint at `PrecisionConfig` construction time. Attempting to create a configuration with FP8 compute or FP8 state raises `ValueError` — there is no user-discipline escape hatch for non-functional training.

**FP8 quantization floor and mask strategy.** FP8 storage introduces a quantization floor at the format's minimum positive subnormal (2⁻⁹ ≈ 0.00195 for E4M3, 2⁻¹⁶ ≈ 1.53×10⁻⁵ for E5M2). When `storage_dtype != compute_dtype` (as with all FP8 configurations), the Policy tier automatically selects the `explicit` mask strategy: Node 4 writes the `hidden_mask` buffer capturing the compute-precision derivative truth (1.0 for active units, 0.0 for ReLU-zeroed) before the activation undergoes storage narrowing. Consuming kernels (Nodes 5, 17, 18) read this mask directly via the `src_scalar_FLAG_use_explicit_hidden_mask` flag. This preserves correct gradient gating for activations that were positive at compute precision but zeroed by storage quantization. When `storage_dtype == compute_dtype` (e.g., `PrecisionConfig.float32()`), the Policy tier selects the `recompute` strategy: the mask is bit-identical to `activation > 0` anyway, so no mask buffer is allocated and consumers derive the mask internally.

#### 3. **Modular, "Dumb" Kernels**

Kernels are simple, single-purpose modules. The architecture avoids complex branching ("smart" kernels) and monolithic designs in favor of composability. Where a complex data transformation is a critical-path bottleneck unsolved by generic tools, a **specialized, single-purpose kernel** will be employed. This specialist kernel remains "dumb"—stateless and reliant on the host for all contextual parameters.

> **Branching is orthogonal to modularity.** When a kernel's core operation diverges fundamentally between use cases (e.g., CCE vs. BCE loss gradients), this document permits two valid strategies:
>
> - **(A) Single Kernel with Host-Injected Flag:** A unified kernel uses a `FLAG__` scalar to toggle paths, provided the divergence is manageable.
> - **(B) Separate Kernels:** Distinct kernels are expected when divergence is complex or imposes conflicting memory patterns.

> **Inter-Dispatch Reduction Delegation.** When any kernel's internal reduction dimension is decomposed across multiple dispatches, each dispatch SHALL produce an independent partial stored in a unique slot. The summation across dispatches SHALL be delegated to the Recursive Reduction Engine. Temporal accumulation on a `dest_` flow buffer (`ZERO_REQUIRED_ADDITIVE` with overlapping write addresses across dispatches) is architecturally prohibited — it conflates kernel production with reduction, violating the `dest_` flow's pure-write semantics and the Placement Contract's non-overlap guarantee.

#### 4. **Backend-Neutral Plan Model**

Simple kernels are composed into a logical Directed Acyclic Graph (DAG) expressed as an immutable, backend-neutral execution plan—a data structure, not executable code. Each target backend (OpenCL, Vulkan, CPU) receives this plan and renders it using its native execution model. The architecture trusts each backend's rendering tier to handle low-level optimizations within its jurisdiction. The number of nodes in the plan is a non-goal, emphasizing adaptability over rigid structure.

#### 5. **Unified Execution Model**

All workflows follow Act (forward pass) then Learn (backpropagation) sequencing, manifesting as either **Sequential Execution Mode**—where Act-Learn phases execute contiguously for pre-labeled batches—or **Event-Triggered Execution Mode**—where Learn-phase execution awaits an external readiness signal post-Act. This split-phase approach ensures consistency across all use cases.

#### 6. **Architectural Hierarchy**

The system observes a strict tripartite authority structure to prevent circular dependencies and ensure traceable design decisions:

- **1. Conceptual (This Document):** Sovereign authority defining **what** must be achieved and **why**. Contains all principles, component definitions, and validation scenarios. Structural decisions that realize these mandates are recorded in the ADR chain (ADR-001 through ADR-018); once accepted, an ADR's decision is binding on all downstream layers.
  - _Example Mandate:_ "Optimizer implementations must avoid precision erosion across unbounded training steps"
  - _Example ADR:_ ADR-002 establishes the five-node plan type taxonomy as the closed vocabulary for the execution plan

- **2. Contractual (Kernel Headers & Contracts):** Binding authority formalizing **interface requirements**. Translates conceptual mandates into human/machine-verifiable API contracts, expressed as `KernelContract` frozen dataclasses and the `CONTRACT.md` articles.
  - _Example Enforcement:_ `adam_update` kernel's `KernelContract` requires `src_scalar_REAL_beta1_pow_t` parameter

- **3. Design (Implementations):** Subordinate authority implementing **how** to fulfill superior layers. Never influences higher layers.
  - _Example Execution:_ Host computes `beta1**t` using FP64 regardless of `COMPUTE_TYPE`

#### 7. **No Silent Monoliths**

All executions manifest externally as an Act/Learn split, even if phases are contiguous, ensuring consistency and enabling inter-phase optimization. This eliminates silent assumptions of single-pass execution, reinforcing the architecture’s phased nature.

---

### **Core Architectural Components**

#### **1. Modular, Chunk-Based Compute Kernels**

The architecture is built upon a foundation of modular, reusable kernels that operate on "chunks" of a larger problem. Work is segmented across multiple dimensions—such as the number of modules, batch size, and number of classes—as dictated by resource constraints. Each `Classifier Module`, representing the system implementation of a machine learning 'Head,' is configured with an **Operating Mode** (`CCE` for single-label or `BCE` for multi-label tasks).

#### **2. The Recursive, Tiered Aggregation Engine**

At the core of the architecture lies a powerful aggregation engine that implements a **recursive, multi-stage reduction tree**. Governed by a host-configurable parameter, `K`—the **Reduction Batch Size**—this engine defines the width of the parallel kernel front at each reduction stage. This `log_K(N)` strategy breaks the `O(N)` memory dependency of a flat aggregation model, enabling scalability to problems of arbitrary size. The engine employs three tiers of aggregation (Identity, Register, Local) selected by the Host Orchestrator based on the number of partials being reduced at any given stage. The **Identity** tier is an Orchestration-tier bypass: when `N=1`, no reduction kernel is dispatched and the single partial is copied directly to the destination buffer. The **Register** (`aggregate_register_reduce`) and **Local** (`aggregate_local_reduce`) tiers are Execution-tier kernels selected by a hardware-informed crossover heuristic based on reduction width.

It is critical to note that this 'Engine' is not a monolithic kernel. It is an emergent property of the Host Orchestrator intelligently composing numerous simple `aggregate_*` kernels in a `log_K(N)` tree structure. This composition is made safe and verifiable by two core contracts:

1. The **`Placement Contract`** governs how parallel kernels _write_ their scattered partial results.
2. The **`Indirection Contract`** governs how `aggregate_*` kernels _read_ these scattered partials.

This indirection-based model is a cornerstone of the **Primacy of Memory Strategy**. By providing the aggregation kernels with a list of memory offsets (`offset_list`), the Host Orchestrator eliminates the need for intermediate device-to-device memory copies at every stage of the reduction, maximizing bandwidth for computation.

**Module-Chunk Isolation Invariant (ADR-030).** For parameter-gradient reduction trees (Nodes 15 and associated optimizer paths), the offset lists constructed by the Host Orchestrator SHALL contain only tile offsets belonging to a **single module chunk**. Cross-module-chunk reduction is architecturally prohibited because it conflates gradients for disjoint parameter sets. This invariant is structurally enforced by the `ModuleChunkGather` primitive, which produces offset lists scoped to a single module chunk's tiles. When `num_module_chunks == 1`, the invariant is trivially satisfied — all tiles belong to the single module chunk.

`TiledGather` remains the correct primitive for non-parameter-gradient reductions (Node 14), where the output layout encodes the correct final position per tile and cross-module-chunk summation is mathematically valid. The hidden-gradient reduction (Node 16) is also exempt: it operates on a permuted SoA buffer that has already collapsed the module dimension via Node 13's class-chunk summation.

#### **3. Gradient Stabilization**

##### **3.1. The Methods of Gradient Clipping**

This architecture acknowledges degrees for Gradient Clipping, defined as Vector-Wise Scaling. The methods differ only in the **scope of their L2 norm calculation**. They are neutral tools; their purpose is determined by the context in which they are applied.

1.  **`Full-Group-Wise Clipping`**
    - **Mechanism:** This method operates on the principle of **"gather and regulate."** It treats the full group of gradients as a single, atomic vector. It computes one L2 norm over the entire pre-summed group and applies a single scaling factor, perfectly preserving the relative magnitudes of all vectors _within_ the group.
    - **Key Property:** Requires a full, synchronizing reduction of the group before it can be applied, making it architecturally expensive.

2.  **`Partial-Group-Wise Clipping`**
    - **Mechanism:** This method operates on the principle of **"assemble and scale."** It treats a practical sub-group of gradients as a single, partial vector. It computes one L2 norm over the pre-summed group and applies a single localized scaling factor, preserving the local-relative magnitudes of all vectors _within_ a sub-group.
    - **Key Property:** Requires a local barrier, synchronizing reduction of the sub-group before it can be applied, relatively cheap.

3.  **`Component-Wise Clipping`**
    - **Mechanism:** This method operates on the principle of **"divide and conquer."** It treats a logical group of gradients as a collection of independent components (tiles/chunks). It computes a local L2 norm for each component and applies a scaling factor to that component alone, preserving only the component's internal gradient direction.
    - **Key Property:** Massively parallel and efficient, requiring no cross-component synchronization, but alters the relative magnitudes _between_ components.

##### **3.2. The Goals of Clipping**

There are two distinct goals that a clipping method can be used to achieve:

1.  **Algorithmic Regulation:** The user-level goal of guiding the learning process by constraining gradient norms. This is governed by a user-defined hyperparameter, `T_algorithmic`.
2.  **Numerical Safety:** The system-level goal of preventing `NaN`/`Inf` overflow by enforcing the physical limits of the hardware's number format. This is governed by a high, mechanically-derived safety threshold, `T_safety`.

##### **3.3. The System's Policy-Driven Stabilization Strategy**

The architecture abandons a simplistic single-pass clipping model in favor of a sophisticated, **multi-stage clipping strategy** that operates in synergy with the `log_K(N)` Recursive Reduction Engine. Instead of applying a single, compromised threshold, the system employs a policy-driven approach to set an independent, optimal clipping threshold at each layer of the reduction tree.

The default strategy is the **Quadratic Scaling Policy**, which provides fine-grained control over the threshold schedule by scaling it based on the layer's distance from the final reduction.

The threshold `T_j` for a given layer `j` (where `j` is the number of aggregation steps away from the final root) is calculated as:

**`T_j = T_{algorithmic} + λ ⋅ j²`**

- **`T_algorithmic`**: The user's target `global_max_grad`. This serves as the **anchor point** of the policy, defining the final threshold to be met at the root of the reduction tree (`j=0`).
- **`j`**: The layer index relative to the root (`j=0` is the root, `j>0` are preceding layers).
- **`λ` (lambda)**: A user-defined, non-negative **Scale Parameter** that dictates the curvature of the policy funnel.
  - A **larger `λ`** creates a more permissive, wider funnel, prioritizing the preservation of gradient information from initial samples.
  - A **smaller `λ`** creates a more restrictive, narrower funnel, enforcing tighter constraints throughout the reduction.
  - When **`λ = 0`**, the policy gracefully degrades to a "Fixed Ceiling" strategy, applying `T_algorithmic` at all layers.

This policy-driven approach is computationally efficient, using the fast `Component-Wise` clipping method at each stage. It transforms stabilization from a single, coarse action into a nuanced, scheduled process.

##### **3.4. Orthogonal Enforcement of Numerical Safety**

The Quadratic Scaling Policy enables the formal architectural separation of **Algorithmic Regulation** from **Numerical Safety**. The two goals are no longer conflated by a `min()` function; they operate as orthogonal concerns.

1.  **The Algorithmic Policy (`T_algorithmic`, `λ`)** is calculated first. It defines the ideal, user-intended threshold curve for guiding the learning process. Its role is to define the "path."
2.  **The Safety Threshold (`T_safety`)** acts as a final, non-negotiable "boundary." It represents the physical hardware limit and is applied as a clamp to the policy's output.

This boundary, `T_safety`, is not a static value. It is a **dynamic budget** determined by the geometry of the reduction at each stage. To prevent summation-induced overflow, the safety threshold `T_safety_j` for a given stage `j` with a reduction fan-in of `K_j` is calculated as:

**`T_safety_j = COMPUTE_FP_FORMAT_MAX / K_j`**

- **`COMPUTE_FP_FORMAT_MAX`**: The maximum representable value of the **compute precision** format (e.g., `~3.4e38` for FP32 compute, `~6.5e4` for FP16 compute). The safety ceiling reflects the arithmetic precision that performs the summation, not the storage precision of the values being summed. In a configuration where all precision roles share a format, `COMPUTE_FP_FORMAT_MAX` equals that format's maximum.
- **`K_j`**: The number of partial results being summed by the `aggregate_*` kernel at stage `j`.

**Pre-Summation Amplification Factor.** The formula above assumes each input element to stage `j` has magnitude bounded by the prior stage's clipping threshold. However, some reduction stages receive inputs that have already undergone implicit summation. Node 16's stage 0 inputs, for example, are the output of Node 13, which performs an implicit summation across `num_class_chunks` class-chunk tiles. After Node 11 clips each tile to `T_pre` and Node 13 sums `C` class chunks, the per-element bound entering Node 16 is `C × T_pre`. The generalized safety ceiling formula that accounts for upstream summation fan-in is:

**`T_safety_j = COMPUTE_FP_FORMAT_MAX / (K_j × A_j)`**

where `A_j` is the **pre-summation amplification factor**: `1` when inputs are raw clipped partials (Nodes 14, 15, 20), and `num_class_chunks` for Node 16's first stage. The Host Orchestrator must incorporate Node 13's amplification when calculating `T_pre`:

**`T_pre ≤ COMPUTE_FP_FORMAT_MAX / num_class_chunks`**

**Execution-Tier Internal Amplification.** GPU backends (OpenCL, Vulkan) implement Node 16 using a "work-group per row" strategy where each thread first accumulates `⌈total_modules/workgroup_size⌉` elements before the staged reduction begins. The Orchestration tier accounts for this pre-accumulation when rendering Node 16's threshold schedule: it computes a pre-accumulation threshold from `compute_fp_format_max` and the dispatch workgroup size, and computes the staged reduction schedule over the `min(total_modules_count, workgroup_size)` post-accumulation intermediates. The kernel applies these host-prescribed thresholds without independent policy computation, consistent with the threshold injection pattern used by the generic reduction engine (Nodes 14/15/20).

The Host Orchestrator implements this separation with a clean, two-step logic for each layer `j`:

1.  **Calculate Policy Threshold:** `policy_threshold = T_algorithmic + λ * j²`
2.  **Enforce Safety Boundary:** `final_threshold_for_layer_j = min(T_safety, policy_threshold)`

This elegant separation ensures the user's algorithmic intent is maximally respected within the fundamental, non-negotiable constraints of the hardware, resolving the logical tension of the previous single-threshold model.

##### **3.5. Architectural Extensibility for Alternative Policies**

The system's default and recommended stabilization strategy employs a sophisticated, two-level approach that is formally described as follows:

1.  **Inter-Tile Strategy: `Component-Wise`**. The batch is decomposed into independent tiles (components). Each tile is processed in parallel and clipped independently of all others.
2.  **Intra-Tile Strategy: `Partial-Group-Wise (Tile Scope)`**. The crucial initial clipping operation (Node 11) within each tile is performed using a `Partial-Group-Wise` method. It computes a single L2 norm across the entire group of partial gradients (`Grad_ModW`, `Grad_ModB`, etc.) for that tile, preserving the complete gradient direction _within that component_.

This nested `Component-Wise` / `Partial-Group-Wise` combination is then followed by the **Quadratic Scaling Policy**, which is applied using `Component-Wise` clipping at each stage of the subsequent reduction tree.

Implementations may choose to make an natural extension to support a **`Per-Item Threshold Override`** for the computed gradients going into the `Partial-Group-Wise Clipping` leaf-pre-reduction-stage. This would offer fine-grained control for advanced strategies like non-uniform regularization, operating orthogonally to the policy applied during the subsequent reduction phase.

A Host Orchestrator could be configured to implement a true **`Group-Wise`** clipping policy for workloads that demand perfectly preserved relative gradient magnitudes. This would require extending the kernel set to include the necessary pre-summation and global norm calculations, and it would come at a significant host complexity, performance and synchronization cost. Such an extension is considered a specialized, scientific execution path.

##### **3.6. Zero-Propagation Theorem**

**Theorem.** *Given that* (a) the host initializes state-role padding to zero, (b) the optimizer applies Padding Zero-Preservation (Inductive) at all positions, (c) each kernel in the forward and backward chains correctly applies its declared padding invariant (Padding Zero-Establishment, Padding Zero-Preservation, or Padding Zero Propagation (Emergent) as mandated per CONTRACT Article 4.3), *then* padding positions carry zero values at every point in the DAG where a downstream consumer reads the full padded extent.

This theorem is not an independent guarantee — it is the emergent property of correct per-kernel invariant assignment. The following chain demonstrates how per-kernel invariants compose to establish the system-level property:

1. **State-role origin:** Host initializes weight and bias padding to zero. Optimizer kernels (Nodes 24, 25) process the full padded extent. At padding positions, the gradient is zero (guaranteed by the upstream chain), and moment vectors are host-initialized to zero. By the Adam recurrence, zero gradient and zero prior moments produce zero updates: m₁ ← β₁·0 + (1−β₁)·0 = 0, m₂ ← β₂·0 + (1−β₂)·0 = 0, Δw = 0. Parameters at padding positions remain at their host-initialized zero. This maintains state-role padding by induction over training steps, not by positional exclusion.
2. **Forward propagation:** Node 4 computes `W*x + b` over the padded input dimension (Padding Zero Propagation (Emergent)). Because input padding is host-zeroed and weight padding is state-zero, hidden-activation padding positions receive only bias contributions — which are themselves zero at padding indices. Node 5 consumes these zero-padded activations.
3. **Backward propagation:** Node 13's `ZERO_REQUIRED` output initialization ensures padded module positions carry zero (host is the guarantor; kernel applies Padding Zero-Preservation). Node 16 reduces across modules, propagating zeros at padded positions into `summed_grad_h`. Nodes 17 and 18 apply Padding Zero-Establishment for their output buffers (which have `Initialization Contract: NONE`), explicitly writing zeros to padding positions via their `input_count`/`hidden_count` boundary parameters. Node 8's output buffers have `ZERO_REQUIRED_ADDITIVE` initialization — the host zeros before first dispatch, and the kernel applies Padding Zero-Preservation (it accumulates only within the logical extent and never touches padding positions).

Any modification to a kernel's write pattern or a buffer's Initialization Contract must be verified against this theorem by confirming the invariant assignment remains sound.

#### **4. Asynchronous Host Interaction**

The processing of each batch spans two event-delimited phases whose temporal relationship is controlled by the user:

- **Act Phase**: Concludes when complete, aggregated forward-pass results (`Final Probs`) become available on the host, signaled by the `inference_event`. Results are delivered through a `RetrievalFuture`—a Protocol with `.wait()`, `.result()`, and `.release()` methods that adapts to each backend's native observation mechanism (zero-copy pointer access for CPU, async D2H transfer completion for GPU backends).
- **Learn Phase**: Triggered by the host when ground truth is ready, concluding when all device-side computations for the training batch complete, signaled by the `final_batch_event`. Device-side intermediate buffers are not retained between phases; the Learn plan recomputes any required activations, trading negligible redundant computation for immediate VRAM release and elimination of stale-activation risk.

The user-facing expression of this temporal split is the **WorkTicket**—a stateful object whose lifecycle mirrors the Act/Learn separation:

| Ticket State   | Transition                | Synchronization Point |
| :------------- | :------------------------ | :-------------------- |
| `PENDING`      | `engine.submit(x_data)`   | —                     |
| `ACT_COMPLETE` | `ticket.get_prediction()` | `inference_event`     |
| `RESOLVED`     | `ticket.resolve(y_data)`  | —                     |
| `CONSUMED`     | `learn_handle.wait()`     | `final_batch_event`   |

This model natively expresses both **Sequential Execution Mode** (resolve immediately after submit) and **Event-Triggered Execution Mode** (inspect prediction, defer resolution until ground truth arrives) through identical syntax, with no mode flag or conditional branching.

#### **5. The Three-Tier Jurisdictional Model**

The system's computational jurisdiction is partitioned into three tiers that correspond to naturally separable concerns:

- **Policy Tier (Shared Layer):** Produces the backend-neutral execution plan. All non-trivial shared computation—plan construction, contract validation, reduction tree planning, buffer lifecycle annotation, threshold scheduling, tiling, activation lifecycle decisions—occurs here. The Policy tier is the sole tier invariant across all backends and all node types.

- **Orchestration Tier (Per-Backend Rendering):** Receives the execution plan and renders it using the backend's native dispatch model. OpenCL enqueues per-tile `clEnqueueNDRange` calls with event chains. Vulkan records a command buffer with `vkCmdDispatch` and `vkCmdPipelineBarrier`. CPU calls `pool_dispatch_and_wait` per node. Kernel tier selection for reduction trees (register-reduce vs. local-reduce), buffer allocation via native allocators, and synchronization primitive management are Orchestration-tier concerns.

- **Execution Tier (Kernel Internals):** The algorithm inside a compiled kernel. Governed by the kernel contract's Behavioral Invariants specification, never prescribed by the plan or the rendering layer. Node 16 receives its threshold schedule from the Orchestration tier, consistent with the threshold injection pattern used by the generic reduction engine (Nodes 14/15/20). The kernel's specialization lies in its optimized single-launch reduction over contiguous SoA data — guaranteed by the upstream Node 13 synchronization point — and its row-wise reduction topology, not in independent policy computation. The Orchestration tier adapts the Policy tier's Quadratic Scaling Policy parameters to the backend's dispatch topology (e.g., accounting for GPU workgroup-size-limited pre-accumulation), then prescribes the concrete schedule.

This model provides a formal criterion for where new functionality belongs: if a computation must produce identical results across backends, it is a Policy concern and belongs in the shared plan. If it adapts to backend-specific capabilities, it is an Orchestration concern. If it is internal to a kernel's algorithm, it is an Execution concern.

#### **6. The Backend-Neutral Execution Plan**

The execution plan is a directed acyclic graph of typed, immutable node descriptors connected by explicit dependency edges. It replaces the implicit `cl.Event` chains of the monolithic implementation with a first-class, inspectable data structure.

The plan's node vocabulary is a **closed taxonomy** of five types:

| Node Type            | Semantics                                                                                                                                                                                                             |
| :------------------- | :-------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| `KernelDispatchNode` | A single logical kernel invocation with buffer bindings, scalar parameters, tile decomposition, and placement strategy.                                                                                               |
| `ReductionTreeNode`  | A multi-stage `log_K(N)` reduction tree, carrying fan-in, pre-computed threshold schedule, and initial offset lists. Rendered atomically by the backend, which selects kernel tiers and manages intermediate buffers. |
| `StreamingLoopNode`  | A parametric loop over a chunk-indexed body of `KernelDispatchNode`s. Specifies chunk count and per-chunk parameter strides; the backend instantiates per-chunk parameter values from base + index × stride.          |
| `BarrierNode`        | A named synchronization point that joins upstream dependency edges. Carries no dispatch payload—it is a pure sequencing construct.                                                                                    |
| `RetrievalNode`      | A host-accessible result extraction point, specifying the source buffer, expected shape, and the named event it signals.                                                                                              |

No additional node types may be introduced without a formal architectural decision. Per Principle §1 (Architectural Elegance Feedback): when an optimization requires a construct beyond this vocabulary, implementation is suspended and the pattern is formalized as a new first-class node type.

The plan's dependency edges are the concurrency specification. Nodes with no dependency relationship may execute in parallel; the backend is free to exploit this. No explicit concurrency annotations are required. A `KernelDispatchNode` with `tile_count=N` specifies logical parallelism—not dispatch granularity. Whether N tiles become N `clEnqueueNDRange` calls, one `vkCmdDispatch(N,1,1)`, or N thread-pool tasks is a backend rendering concern.

#### **7. The Kernel Contract Model**

Each kernel's interface is formalized as two complementary artifacts:

- **`KernelContract` (Shared Layer):** A frozen, backend-neutral dataclass carrying the kernel's complete interface specification—parameter names, types, shapes, padding contracts, placement strategies, and validation preconditions. The Policy tier uses this for pre-dispatch validation at plan-construction time. A plan is only valid if every `KernelDispatchNode`'s buffer bindings and scalar parameters satisfy its associated `KernelContract`.

Each buffer parameter in a `KernelContract` carries a `precision_role` declaration — one of `"storage"`, `"compute"`, or `"state"` — that identifies the buffer's element type under the active precision configuration. When a kernel reads a buffer whose precision role differs from `"compute"`, the kernel widens the value to compute precision before any arithmetic. When writing, the kernel narrows from compute precision to the buffer's role precision. These conversions are declared in the contract's `Behavioral Invariants` as a mandatory `Precision Boundary Conversion` invariant. The conversion pattern is structurally identical regardless of whether the role types happen to be equal — the kernel source has one codepath, and the compiler eliminates identity conversions.

- **`KernelBinding` (Per-Backend):** A backend-specific dispatch adapter that translates abstract placement keys and buffer references into native dispatch arguments. Tile index delivery—host-provided `flat_tile_index` scalar (OpenCL), `gl_WorkGroupID.x` (Vulkan), or `task_index` parameter (CPU)—is a binding concern, not a contract concern.

This split ensures that interface validation is shared (no duplication across backends) while dispatch mechanics remain backend-native (no lowest-common-denominator abstraction).

#### **8. The Buffer Lifecycle**

Every device-side buffer in the execution plan is described by a `BufferDescriptor` carrying:

- **`producing_node`**: The unique node that writes the buffer.
- **`consumers`**: The set of nodes that read the buffer.
- **`last_consumer`**: The final reader, after which the buffer may be freed.
- **`role`**: One of four lifecycle roles—`MODEL_STATE` (persists across batches), `BATCH_INPUT` (uploaded per batch), `BATCH_INTERMEDIATE` (temporary computation product), or `BATCH_OUTPUT` (returned to the host).

Every device-side buffer's `BufferDescriptor` carries a `precision_role` field — one of `"storage"`, `"compute"`, or `"state"` — that determines the buffer's element type and allocation size under the active `PrecisionConfig`. The `element_size_bytes` and `size_bytes` fields derive from the role-appropriate dtype. A storage-role activation buffer allocated at FP16 is half the size of a compute-role accumulation buffer at FP32, enabling the bandwidth benefits of narrow formats without affecting the fidelity of arithmetic operations.

Buffer lifetimes are plan-prescribed: the Policy tier annotates each buffer's lifetime interval, and the Orchestration tier manages allocation and deallocation according to these annotations. Each plan's buffer namespace is fully self-contained—no buffer persists across plan boundaries, enabling immediate VRAM release after each plan completes.

#### **9. Multi-Backend Architecture**

The system supports three compute backends:

- **OpenCL (PyOpenCL):** Runtime-compiled kernels from the architecture's `kernels/` directory. Per-tile imperative dispatch with `cl.Event` synchronization. The original and reference backend.
- **Vulkan (vulkan-python):** GLSL compute shaders compiled to SPIR-V at build time. Single-dispatch parallelism (`vkCmdDispatch(N,1,1)`) with pipeline barriers recorded into command buffers.
- **CPU (ctypes):** Native C implementations compiled to a shared library. Deterministic, single-threaded-per-task execution via `pool_dispatch_and_wait`. Struct layout verification at library load time prevents silent FFI corruption.

All backends implement from the same algorithmic specification (`kernels.cl.h`), which serves as the language-neutral reference for every kernel's mathematical operation. Per-backend source files reside in `src/backends/<name>/kernel_sources/`, with the shared orchestration layer in `src/shared/`. This physical isolation ensures that changes to one backend cannot affect another.

The build system (Meson with `meson-python`) conditionally enables each backend via `feature` options (`backend_opencl`, `backend_vulkan`, `backend_cpu`) with `auto` as the default—each backend is available when its toolchain is detected, absent otherwise, requiring no manual configuration. A generated `_build_config.py` module declares each backend's availability as a boolean, serving as the single authoritative source for runtime capability queries and test-tier skip logic.

The test strategy reflects this multi-backend structure through three tiers: **Tier 1** validates Policy-tier correctness (plan construction, contract validation) with no backend required; **Tier 2** validates per-backend kernel correctness against analytical/numpy reference fixtures; **Tier 3** validates cross-backend parity using the CPU backend as a deterministic reference oracle.

#### **10. The User-Facing API Surface**

The user-facing API expresses the architecture's Act/Learn temporal split through a `WorkTicket` lifecycle:

```python
# Sequential Execution Mode (pre-labeled batch)
result = engine.train_batch(X_train, y_train)

# Event-Triggered Execution Mode (temporal split)
ticket = engine.submit(x_data)         # returns immediately; Act plan dispatched
probs = ticket.get_prediction()        # blocks until Act completes
# ... time passes ...
learn_handle = ticket.resolve(y_data)  # returns immediately; Learn plan dispatched
learn_handle.wait()                    # blocks until Learn completes
```

Design decisions:

- **Future-based concurrency:** `submit()` and `resolve()` return immediately; the user decides when to block. No `asyncio` dependency. Compatible with synchronous scripts, Jupyter notebooks, and non-async frameworks.
- **Explicit batch submission:** One `submit()` produces one ticket representing one batch. The engine is stateless between ticket lifecycles.
- **Recompute (no cross-plan sharing):** The Learn plan recomputes the forward pass from the stored input data reference. No device buffers persist across plan boundaries. The ticket holds only host-side state—its arbitrary lifetime in `ACT_COMPLETE` carries zero VRAM cost.
- **Ephemeral data:** Each submission is a one-shot event. For multi-epoch training, the user resubmits the data each epoch, retaining full control over the training loop.

#### **11. The Host Orchestrator & Execution Policies**

**"All operations are spreadable except for the final gradient collect operation."**

The Policy tier's plan construction logic serves as a sophisticated orchestrator, responsible for resource assessment and plan-graph assembly. For each training batch, it performs strategic assessments that shape the execution plan:

1.  **Memory Assessment & Chunk Definition:** The orchestrator determines an optimal chunking strategy, defining `num_module_chunks`, `num_batch_chunks`, and `num_class_chunks` to balance compute and memory demands. These decisions manifest as tile counts on `KernelDispatchNode`s and chunk counts on `StreamingLoopNode`s. The device's capabilities are described by a `HardwareProfile` carrying `max_reduce_fan_in`, `simd_width`, `cache_line_bytes`, and `global_mem_bytes`—named for its plan-construction role, not its hardware origin. **Class-chunk amplification in collection buffers:** Node 8's `dest_buffer_GLOBAL_partial_grad_weights_module` is allocated with the full `padded_total_output_class_count` in its innermost dimension, but each tile writes only `classes_per_chunk` positions (the remainder is `ZERO_REQUIRED`-initialized). When `num_class_chunks` is large, the per-tile allocation is `num_class_chunks×` the actual write footprint — a deliberate trade-off enabling Node 11 to compute a contiguous-memory L2 norm without cross-tile gather. The chunking strategy must account for this amplification when budgeting device memory. **Batch-chunk amplification in collection buffers:** When `num_batch_chunks > 1`, Node 8's output buffers gain a `num_batch_chunks` dimension (per Principle §3, Inter-Dispatch Reduction Delegation). Each (tile, batch_chunk) pair writes to a unique slot; a `ReductionTreeNode` sums across batch chunks before Node 11 computes the joint L2 norm. When `num_batch_chunks == 1`, the batch dimension is trivially 1 and the Identity tier bypass applies — zero overhead. The chunking strategy adjusts `num_batch_chunks` to fit within the device memory budget, accepting the `num_batch_chunks×` amplification as the honest cost of batch-chunk parallelism.

    **Sample mask packing.** The host packs the per-sample boolean mask into a `uint` bitmask buffer (32 samples per word, LSB-first) before device upload. The `effective_batch_size` scalar for Node 21 is derived via integer popcount over the packed words, yielding an exact count for any batch size up to 2³².

    **Padded dimension synthesis.** When a padded dimension scalar (e.g., `padded_hidden_count`, `padded_total_output_class_count`) appears in the Tensor Shape of buffers with different precision roles, the Host Orchestrator computes its value as the least common multiple of all role-specific alignment requirements across every buffer that uses the scalar. For example, `padded_hidden_count` must simultaneously satisfy `padded_hidden_count × sizeof(STORAGE_TYPE) ≡ 0 (mod 128)` for CACHE-padded storage-role buffers, `padded_hidden_count ≡ 0 (mod SIMD_WIDTH)` for SIMD-padded state-role buffers, and any analogous constraint for compute-role buffers. The resulting LCM is configuration-dependent: under `PrecisionConfig.fp8_e4m3()`, the CACHE constraint (128 elements) typically dominates; under `PrecisionConfig.float32()`, the SIMD constraint may dominate. This synthesis is a Policy-tier concern; kernel contracts declare their individual requirements and are not aware of the cross-buffer union.
2.  **Activation Lifecycle & Streaming:** It selects between a Cache or Recompute strategy and manages **two distinct backpropagation streaming models**, expressed as plan-level structural choices:

    **Mask strategy selection.** The Policy tier evaluates `PrecisionConfig.mask_strategy` at plan-construction time and propagates the result as FLAG scalars on Nodes 4, 5, 17, and 18. Under `recompute` mode, the `hidden_mask` BufferDescriptor is omitted from the plan entirely — no allocation, no lifecycle tracking. Under `explicit` mode, the buffer is allocated and tracked as a `BATCH_INTERMEDIATE` with its `last_consumer` set to the final streaming loop iteration's Node 17 or 18 dispatch. The mask strategy is orthogonal to the activation Cache/Recompute strategy; all four combinations produce correct results (see ADR-031 §1.4).

- **Model A: Accumulate via Recompute (For `Grad_H` and `Grad_Mod*`):** Expressed as a `StreamingLoopNode` whose body recomputes `hidden_i` chunks and writes tile-indexed partials into a collection buffer for subsequent reduction. It maintains a minimal memory footprint at the cost of a parametric loop.
- **Model B: True Streaming (For `Grad_SW` & `Grad_SB`):** Expressed as a `StreamingLoopNode` (Nodes 17→18→19) that recomputes inputs, computes partial gradients, clips them immediately, and feeds the clipped partials into the reduction engine—all within a single parametric loop.

3.  **SIMD-Aware Weight Layout:** Transforms shared layer weights into a SIMD-friendly "Struct of Arrays" (SoA) layout before the Act phase.
4.  **Reduction Planning & Rendering:** Analyzes problem size and device capabilities (via `HardwareProfile`) to select an optimal Reduction Batch Size (`K`), rendering the full `log_K(N)` reduction tree as `ReductionTreeNode`s with pre-computed threshold schedules and initial offset lists. For Node 16, the Orchestration tier additionally renders a per-stage threshold schedule and pre-accumulation threshold, adapted to the backend's dispatch topology. The CPU backend renders the abstract `⌈log_K(M)⌉`-stage schedule directly (no pre-accumulation). GPU backends render a `⌈log_K(P)⌉`-stage schedule over `P = min(M, workgroup_size)` post-accumulation intermediates, with a pre-accumulation threshold of `compute_fp_format_max / K`. The rendered schedule is passed to the kernel as a buffer parameter, paralleling the per-stage threshold scalars passed to the generic `reduce_k_fan_in_and_clip` kernels.
5.  **Indirection List Construction:** For each reduction stage, constructs the `offset_list` buffer pointing to scattered partials, fulfilling the **Indirection Contract**.
6.  **Partial Result Placement & Indexing:** Provides each tile-parallel kernel invocation with a unique placement key via abstract placement strategies (e.g., `grid_mod_cls`, `linear_batch`), guaranteeing non-overlapping writes. The mechanism by which each tile resolves its index—host scalar, hardware intrinsic, or task parameter—is a `KernelBinding` concern.
7.  **Phase Staging & Optimizer State Management:** Governs phase intervals through named synchronization points (`BarrierNode`s and `RetrievalNode`s). For the Adam optimizer, the host computes `beta1**t` and `beta2**t` bias correction terms in high precision (FP64) and passes the final values to the update kernel, avoiding on-device precision loss. Precision is configured through a `PrecisionConfig` frozen dataclass carrying three independent role assignments: `storage_dtype` (bandwidth optimization), `compute_dtype` (arithmetic fidelity), and `state_dtype` (optimizer stability), along with derived constants `storage_fp_format_max`, `compute_fp_format_max`, and `compute_epsilon`. A configuration where all three roles share a type (e.g., `PrecisionConfig.float32()`) is a parameterization — not a distinct mode. The host's existing practice of computing Adam bias correction terms in FP64 is recognized as an instance of state-precision independence; the three-role model formalizes this pattern as a first-class concern rather than an ad-hoc exception.

**The state role supports extended precision for unbounded training stability through two mechanisms:**

1. **Host-side bias correction.** The host computes `beta1**t` and `beta2**t` in FP64 regardless of `COMPUTE_TYPE`, avoiding precision erosion in these geometrically-decaying terms. This is the established practice formalized by the three-role model.

2. **State-precision accumulation.** Stateful-update kernels with accumulative operations (Node 24) perform EMA updates in `ACCUM_TYPE = max(COMPUTE_TYPE, STATE_TYPE)`, ensuring moment vectors preserve state precision across unbounded training steps. When `STATE_TYPE > COMPUTE_TYPE`, the kernel widens gradients to state precision for the EMA computation rather than narrowing state values to compute precision. (Node 25's clamping is transformative, not accumulative — standard precision boundary conversion applies.)

Adam's EMA update ($\beta_1 \cdot m + (1 - \beta_1) \cdot g$) requires that the format represent the small difference $(1 - \beta_1) \cdot g$ without rounding it away. For $\beta_1 = 0.999$, the gradient contributes only $0.001$ of its magnitude per step. When state precision exceeds compute precision, state-precision accumulation ensures this contribution is captured at full state fidelity. FP64's 52-bit mantissa provides an order-of-magnitude more headroom than FP32's 23 bits — headroom that the architecture now preserves.

FP8 storage (E4M3 or E5M2) is supported for maximum bandwidth efficiency. Two FP8 variants are provided: **E4M3** (4 exponent bits, 3 mantissa bits, max value 448) prioritizes precision; **E5M2** (5 exponent bits, 2 mantissa bits, max value 57344) prioritizes dynamic range. E4M3 is the primary target for gradient and activation storage. The two formats differ in special-value semantics: **E4M3** (`float8_e4m3fn`) has no infinity representation but reserves 2 bit patterns for NaN (0x7F, 0xFF), leaving 254 finite values. **E5M2** follows IEEE-like conventions: exponent 0x1F encodes ±infinity (mantissa = 0) and NaN (mantissa ≠ 0), leaving 248 finite bit patterns. The store path saturates to the maximum finite value in both formats, so infinity and NaN bit patterns are never written to storage buffers. The host is responsible for scaling values to fit within FP8's representable range before storage; the existing Quadratic Scaling Policy already constrains gradients well below FP8 limits. `PrecisionConfig.fp8_e4m3()` provides E4M3 storage with FP32 compute and FP32 state; `PrecisionConfig.fp8_e4m3_f64()` adds FP64 compute and state for extended stability.

The Orchestrator's ability to construct valid execution plans for these complex streaming and reduction patterns is fundamentally underpinned by the **`Axiom of Interface Verifiability` (Contract Article 1.4)**. Because each kernel's `KernelContract` is a closed logical system with a self-contained `Calculability Proof`, the Policy tier possesses all information required to plan data flows without implicit or hidden knowledge, guaranteeing correctness.

---

### **Architectural Blueprint & Data Contracts for a Multi-Head Classifier**

```mermaid
graph TD
    %% === Node Styling Definitions ===
    classDef kernel fill:#DDEBF7,stroke:#5B9BD5,stroke-width:2px;
    classDef data fill:#E2F0D9,stroke:#70AD47,stroke-width:2px;
    classDef param fill:#D9E1F2,stroke:#2F5496,stroke-width:2px;
    classDef partial_data fill:#F8CBAD,stroke:#C00000,stroke-width:1.5px,stroke-dasharray: 4 4;
    classDef clipped_data fill:#f3e5f5,stroke:#8e24aa,stroke-width:2px,stroke-dasharray: 2 2;
    classDef summed_data fill:#fcf4dd,stroke:#c78a08,stroke-width:2px,stroke-dasharray: 2 2;
    classDef final_data fill:#BDD7EE,stroke:#2F5597,stroke-width:2.5px;
    classDef host_logic fill:#FFF2CC,stroke:#FFC000,stroke-width:2px;
    classDef specialization fill:#e1f5fe,stroke:#0288d1,stroke-width:2px;
    classDef sync_event fill:#ffc0cb,stroke:#C00000,stroke-width:2.5px,stroke-dasharray: 5 2;
    classDef barrier_node fill:#D9D9D9,stroke:#C00000,stroke-width:3px;
    classDef exec_domain fill:#f5f5f5,stroke:#757575,stroke-width:2.5px;
    classDef host_sm fill:#f5f5f5,stroke:#000000,stroke-width:3px;
    classDef host_state fill:#fffde7,stroke:#f57f17,stroke-width:2px,stroke-dasharray: 5 2;
    classDef phase_box fill:#fafafa,stroke:#616161,stroke-width:1.5px,stroke-dasharray: 2 2;
    classDef domain_instance fill:#e3f2fd,stroke:#1e88e5,stroke-width:2px,stroke-dasharray: 8 4;
    classDef comp_tile fill:#fce4ec,stroke:#d81b60,stroke-width:1.5px,stroke-dasharray: 5 3;
    classDef full_intermediate fill:#FCE4D6,stroke:#F4B183,stroke-width:2px;
    classDef streaming_loop fill:#e3f2fd,stroke:#1e88e5,stroke-width:2px,stroke-dasharray: 8 4;

    %% === HOST ORCHESTRATION LAYER ===
    subgraph HostStateMachine["WorkTicket Lifecycle"]
        style HostStateMachine host_sm
        HSM_Dispatch["submit(x)<br/>PENDING"]:::host_state --> HSM_Await_Inference["get_prediction()<br/>ACT_COMPLETE"]:::host_state
        HSM_Await_Inference --> HSM_Trigger_Learn["resolve(y)<br/>RESOLVED"]:::host_state
        HSM_Trigger_Learn --> HSM_Await_Batch["wait()<br/>CONSUMED"]:::host_state
    end

    %% === GLOBAL PARAMETERS (HOST-STAGED PRECONDITIONS) ===
    P_Shared["Shared Params"]:::param
    P_Module["Module Params"]:::param
    P_Temps["Temp Params"]:::param
    P_Adam["Host-Computed<br/>Adam State"]:::param
    P_Policy["Clipping Policy"]:::param
    P_Offsets["Indirection Lists"]:::param
    Targets["Targets"]:::param
    SampleMask["Sample Mask"]:::param


    %% === EXECUTION DOMAIN ===
    subgraph DeviceDAG[Device-Side Computational DAG]
        style DeviceDAG exec_domain
        direction TB

        %% === ACT PROCESS ===
        subgraph ActProcess["Act Process (Forward Pass & Loss)"]
            style ActProcess phase_box
            K4["(4) forward_pass"]:::kernel --> hidden_i["Hidden Activations 'i'"]:::data
            K4 --> hidden_mask["Hidden Mask 'i'"]:::data
            P_Shared & P_Module & SampleMask --> K4
            hidden_i & hidden_mask --> K5["(5) render_logits_chunk"]:::kernel
            P_Module & P_Temps & SampleMask --> K5
            K5 --> Full_Logits["Full Logits Buffer"]:::full_intermediate
            subgraph LossPath["Loss Path (CCE/BCE)"]
                K6["(6) compute_probs_loss_cce"]:::kernel
                K7["(7) compute_probs_loss_bce"]:::kernel
            end
            Full_Logits & Targets & SampleMask --> LossPath
            LossPath --> PARTIALS_Probs["Collection: PARTIAL Probs"]:::partial_data
            K7 --> PARTIALS_Loss_BCE["Collection: PARTIAL BCE Loss"]:::partial_data
            K14["(14) Reduction (Probs, BCE Loss)"]:::host_logic
            PARTIALS_Probs & PARTIALS_Loss_BCE & P_Offsets --> K14
            K14 --> FINAL_Probs["Final Probs"]:::final_data
            FINAL_Probs --> K23["(23) D2H Async Copy"]:::data
        end
        HSM_Dispatch --> ActProcess

        EV_Inference["<b>Host Sync Point</b><br/>inference_event"]:::sync_event
        K23 --> EV_Inference; EV_Inference --> HSM_Await_Inference

        %% === LEARN PROCESS ===
        subgraph BatchDomain ["Learn Process (Batch Processing Domain)"]
            direction TB
            K8_Start["(8) Learn Start Trigger"]:::sync_event
            HSM_Trigger_Learn --> K8_Start

            subgraph PhaseI["<b>Phase I:</b> Hierarchical Gradient Generation (N Instances)"]
                style PhaseI phase_box
                subgraph InstanceDomain ["Instance Processing Domain"]
                    style InstanceDomain domain_instance
                    direction TB
                    K8_Start -.-> comp_tile_trigger[" "]
                    subgraph ComputationalTile ["Computational Tile"]
                        style ComputationalTile comp_tile
                        comp_tile_trigger ~~~ K8_grad["(8) calc_module_grads"]:::kernel
                        comp_tile_trigger ~~~ K9_bprop["(9) backprop_error_hidden"]:::kernel
                        comp_tile_trigger ~~~ K10_temp["(10) calc_temp_grads"]:::kernel
                        hidden_i & PARTIALS_Probs & Targets & SampleMask --> K8_grad
                        PARTIALS_Probs & Targets & SampleMask --> K9_bprop
                        PARTIALS_Probs & Targets & SampleMask & Full_Logits --> K10_temp
                        P_Module --> K8_grad & K9_bprop
                        P_Temps --> K10_temp

                        K8_grad --> PARTIALS_Grad_Mod["PARTIAL Grad (Mod)"]:::partial_data
                        K9_bprop --> PARTIALS_Grad_H["PARTIAL Grad_H"]:::partial_data
                        K10_temp --> PARTIALS_Grad_Temps["PARTIAL Grad (Temps)"]:::partial_data

                        B11["<b>(11) Intra-Component Barrier</b><br/>clip_partial_gradients"]:::barrier_node
                        PARTIALS_Grad_Mod & PARTIALS_Grad_H & PARTIALS_Grad_Temps --> B11
                    end
                    B11 --> Clipped_PARTIALS_Mod["Clipped PARTIAL<br/>Grad (Mod)"]:::clipped_data
                    B11 --> Clipped_PARTIALS_H["Clipped PARTIAL<br/>Grad_H (AoS)"]:::clipped_data
                    B11 --> Clipped_PARTIALS_Temps["Clipped PARTIAL<br/>Grad (Temps)"]:::clipped_data

                    B13["<b>(13) Item Synchronization Barrier</b><br/>gather_and_permute_grad_h"]:::barrier_node
                    Clipped_PARTIALS_H --> B13
                end
            end
            B13 --> Permuted_Grad_H["Permuted Grad_H (SoA)"]:::full_intermediate
            Clipped_PARTIALS_Mod & Clipped_PARTIALS_Temps --> K15

            subgraph PhaseII["<b>Phase II:</b> Specialized & Collective Aggregation Pt. 1"]
                style PhaseII phase_box
                K16["(16) stabilize_and_reduce_grad_hidden"]:::specialization
                Permuted_Grad_H & P_Policy --> K16
                K16 --> Summed_Grad_H["Summed Grad_H"]:::summed_data

                K15["(15) Recursive Clip-Aggregation<br/>(Module & Temp Grads)"]:::host_logic
                P_Policy & P_Offsets --> K15
                K15 --> Summed_Grad_Mod["Summed Grad (Mod)"]:::summed_data
                K15 --> Summed_Grad_Temps["Summed Grad (Temps)"]:::summed_data
            end

            subgraph PhaseIII["<b>Phase III:</b> Dependent Gradient Generation (Streaming)"]
                style PhaseIII phase_box
                subgraph StreamingLoop["Streaming Backprop Loop (Chunk 'i')"]
                    style StreamingLoop streaming_loop
                    Input_i["Recomputed Input 'i'"]:::data --> K17["(17) backprop_shared_weights"]:::kernel
                    hidden_i --> K17
                    hidden_mask -. "FLAG" .-> K17
                    SampleMask --> K17
                    hidden_i --> K18["(18) backprop_shared_biases"]:::kernel
                    hidden_mask -. "FLAG" .-> K18
                    SampleMask --> K18
                    Summed_Grad_H -. "slice" .-> K17 & K18
                    K17 --> PARTIALS_Grad_SW["PARTIAL Grad_SW 'i'"]:::partial_data
                    K18 --> PARTIALS_Grad_SB["PARTIAL Grad_SB 'i'"]:::partial_data
                    K19["(19) clip_shared_gradients_chunk"]:::kernel
                    PARTIALS_Grad_SW & PARTIALS_Grad_SB --> K19
                    K19 --> Clipped_PARTIALS_S_i["Clipped PARTIAL Grad_S* 'i'"]:::clipped_data
                end
                Summed_Grad_H -- "triggers" --> StreamingLoop
                Clipped_PARTIALS_S_i --> Clipped_PARTIALS_S_Collection["Collection<br/>Clipped PARTIAL Grad_S*"]:::clipped_data
            end

             subgraph PhaseIV["<b>Phase IV:</b> Final Aggregation & Normalization"]
                style PhaseIV phase_box
                K20["(20) Recursive Clip-Aggregation<br/>(Shared Grads)"]:::host_logic
                Clipped_PARTIALS_S_Collection & P_Policy & P_Offsets --> K20
                K20 --> Summed_Grad_S["Summed Grad (Shared)"]:::summed_data

                K21["(21) normalize_gradients"]:::kernel
                Summed_Grad_Mod & Summed_Grad_Temps & Summed_Grad_S --> K21
                K21 --> FINAL_Grad_Mod["Final Avg Grad (Mod)"]:::final_data
                K21 --> FINAL_Grad_Temps["Final Avg Grad (Temps)"]:::final_data
                K21 --> FINAL_Grad_S["Final Avg Grad (Shared)"]:::final_data
            end

            subgraph PhaseV["<b>Phase V:</b> Parameter Update"]
                 style PhaseV phase_box
                 B22["<b>(22) Batch Synchronization Barrier</b>"]:::barrier_node
                 FINAL_Grad_Mod & FINAL_Grad_Temps & FINAL_Grad_S --> B22

                 K24_Shared["(24) adam_update (Shared)"]:::kernel
                 K24_Module["(24) adam_update (Module)"]:::kernel
                 K24_Temps["(24) adam_update (Temps)"]:::kernel
                 B22 & P_Adam --> K24_Shared & K24_Module & K24_Temps
                 P_Shared --> K24_Shared
                 P_Module --> K24_Module
                 P_Temps --> K24_Temps
                 K24_Shared -- "parameter updates" --> P_Shared
                 K24_Module -- "parameter updates" --> P_Module
                 K24_Temps -- "parameter updates" --> P_Temps

                 K24_Temps --> K25["(25) clamp_temperatures"]:::kernel
            end

            %% --- Inter-Phase Data Flow ---
            PhaseI --> PhaseII
            PhaseII --> PhaseIII
            PhaseIII --> PhaseIV
            PhaseIV --> PhaseV
        end

        EV_Final["<b>Host Sync Point</b><br/>final_batch_event"]:::sync_event
        K24_Shared & K24_Module & K25 --> EV_Final
        EV_Final --> HSM_Await_Batch
    end
```

---

### **Kernel & Synchronization Contracts**

Each kernel described below is formalized as a `KernelContract` frozen dataclass (§7) in the shared layer, validatable at plan-construction time without requiring any backend. Per-backend `KernelBinding` implementations translate these contracts into native dispatch arguments. The algorithmic reference for all kernels is `kernels.cl.h`, serving as the language-neutral specification from which each backend's kernel sources are independently implemented.

The CCE/BCE strategy divergence (CONCEPT.md §3, Principle "Modular, Dumb Kernels") is resolved through a mixed delegation model: **Strategy B** (separate kernels) for structurally divergent operations (Nodes 6 and 7, which have genuinely different interfaces and algorithms), and **Strategy A** (host-injected `FLAG__` scalar) for parametrically divergent operations (Nodes 8, 9, and 10, which share the same interface but toggle an internal branch).

#### **Fundamental Synchronization Barriers**

This architecture defines two distinct and fundamental types of synchronization points, which are properties of the computational graph itself, independent of the host's scheduling logic (e.g., "Epochs" or "Tickets"). In the execution plan (§6), these manifest as `BarrierNode`s (pure sequencing constructs) and `RetrievalNode`s (host-observable completion points).

- **Item Synchronization Point:** A barrier that resolves a data dependency _within the execution of a single learning item_. It ensures that all necessary partial results for that one item are available before a subsequent algorithmic step can proceed. Its purpose is to enforce the correctness of the core algorithm (e.g., backpropagation).
  - **Canonical Example:** Node **(13) `gather_and_permute_grad_h`** — represented as a `KernelDispatchNode` whose completion gates the Item Synchronization `BarrierNode`.

- **Batch Synchronization Point:** A synchronization property ensuring that parameter updates occur only after their gradient dependencies are fully resolved. When the module dimension is decomposed into chunks, this property is satisfied by per-sub-graph dependency edges rather than a single monolithic barrier: each `adam_update` dispatch's dependency on its own `normalize_gradients` output expresses the synchronization contract directly in the DAG structure. When `num_module_chunks == 1`, the property is trivially satisfied — a single dependency edge from each normalized gradient to its corresponding `adam_update` is equivalent to the former barrier.
  - The `final_batch_event` `RetrievalNode` remains the sole plan-wide synchronization point, collecting all terminal update nodes. It is the observable successor of the former Node 22 barrier.
  - **Canonical Example:** The per-module-chunk dependency edges from `normalize_gradients[m]` to `adam_update[m]` for each parameter group and module chunk index `m`, converging at `final_batch_event`.

The two host-visible synchronization events are represented as `RetrievalNode`s: `inference_event` (signaled after diagnostic aggregation, enabling host observation of `Final Probs`) and `final_batch_event` (signaled after all parameter updates complete).

#### **The Placement Contract**

All kernels designated as "Partial Renderers" must accept a unique `flat_tile_index` integer parameter from the host. This index is used to calculate the write offset within their designated output buffer, ensuring each partial result is placed in its correct, discrete slot. This contract applies to kernels: **(6), (7), (8), (9), (10), (11), (17), (18)**.

---

#### **Act Phase Kernels**

- **(4) `forward_pass`**: Computes hidden activations **for a specified data chunk**.
  - **Contract:** Applies a (Weights \* Input + Bias + ReLU) transform, producing an `hidden_i` buffer for a single data chunk. Supports both Cache and Recompute strategies.
- **(5) `render_logits_chunk`**: A streamable kernel computing raw logits from a **data chunk of hidden activations**.
  - **Contract:** Executes a (Weights \* Input + Bias) transform, yielding `Full_Logits` that support caching and recomputation.
- **(6) `compute_probs_loss_cce_chunk`**: Fused kernel for Softmax, probability, and CCE loss calculation.
  - **Contract:** Performs the **complete, temperature-aware, numerically stable Softmax calculation internally**, guaranteeing mathematical correctness and memory locality.
  - **Outputs:** Adheres to the Placement Contract for `partial_probs_out`.
  - **Loss Aggregation Strategy:** Employs a direct **scatter-write** for `final_loss_out`.
    - **Justification:** CCE loss produces a single scalar value per (module, sample) pair. This one-to-one mapping allows each parallel invocation to compute a unique, final write address and populate the destination buffer directly, eliminating the overhead of a reduction stage for the loss value itself.
- **(7) `compute_probs_loss_bce_chunk`**: Invoked when `Operating Mode` is `BCE`. A streamable kernel computing probabilities and partial BCE loss.
  - **Contract:** Performs the **complete, temperature-aware, numerically stable Sigmoid calculation internally** (two-branch form: positive/negative logit split to avoid `exp()` overflow), then computes binary cross-entropy loss. Generates probabilities and partial loss from `Full_Logits`.
  - **Loss Aggregation Strategy:** Produces `partial_loss_out` which adheres to the **Placement Contract**.
    - **Justification:** BCE loss is computed on a per-class basis. To derive the final loss for a given (module, sample) pair, these per-class values must be aggregated (summed). Therefore, this kernel is a **Partial Renderer** for loss, producing intermediate values that are contractually obligated to be processed by the **Recursive Clip-Aggregation Engine (Node 14)**.
  - **Contract:** Generates probabilities and partial loss from `Full_Logits`. Adheres to the Placement Contract for both `partial_loss_out` and `partial_probs_out`.

---

#### **Learn Phase Kernels: Gradient Production**

- **(8) `calculate_module_param_grads_chunk`**: Computes partial gradients for module parameters from a **data chunk of hidden activations**.
  - **Contract:** Utilizes `Partial_Probs`, `Targets`, and a **chunk** of `hidden_i` to generate partial `Grad_ModW` and `Grad_ModB`. Adheres to the Placement Contract.
- **(9) `backprop_error_to_hidden_chunk`**: Computes the partial upstream gradient (`Grad_H`) for a tile of the problem.
  - **Contract:** Backpropagates errors using `Partial_Probs` and `Targets`. Adheres to the Placement Contract.
- **(10) `calculate_chunk_temp_gradients`**: A streamable kernel computing partial gradients for temperature parameters.
  - **Contract:** Derives temperature gradients from `Partial_Probs`, `Full_Logits`, and `Targets`. Adheres to the Placement Contract.

---

#### **Learn Phase Kernels: Gradient Processing, Reduction & Synchronization**

- **(11) `clip_partial_gradients`**: **[Utility Kernel]** Performs a local, group-wise clip on the full gradient vector for a single parallel tile.
  - **Contract:** Invoked after all partial gradients for a single tile (`Nodes 8, 9, 10`) are generated. It performs the following indivisible operation:
    1.  It computes a **single L2 norm** over the logical concatenation of all input gradient buffers (`Grad_ModW`, `Grad_ModB`, `Grad_Temps`, and `Grad_H_AoS`).
    2.  If this single norm exceeds the threshold, the derived scaling factor is then applied **uniformly** to all four constituent buffers.
  - **Clarification of Clipping Strategy:** This kernel's behavior constitutes a **`Group-Wise` clip at the tile level**, preserving the internal directionality of the tile's total gradient. This is part of a broader system strategy that is **`Component-Wise` at the inter-tile level** (i.e., each tile is clipped independently of other tiles in the batch).
  - **Strategic Role:** By preserving the tile's gradient direction as a single unit, this operation is critical for stability. For the `Grad_H` vector, it serves as **Phase I (Safety) of a two-stage stabilization strategy**. It guarantees all raw partials are brought into a finite numerical range before their respective reductions, preventing `NaN`/`Inf` propagation. Consumes `PARTIAL_*` gradient buffers and produces `Clipped_PARTIAL_*` buffers.

- **(12)** _Reserved — retired. Its permutation functionality was subsumed by Node 13._
- **(13) `gather_and_permute_grad_h`**: **[Specialized Kernel]** Gathers and permutes the clipped `Grad_H` partials.
  - **Contract:** Reads from the `Clipped_PARTIALS_Grad_H_AoS` collection and writes to a single `Permuted_Grad_H_SoA` buffer. This is a mandatory step before Node (16) and serves as the canonical **Item Synchronization Point**.
- **The Recursive Clip-Aggregation Engine:** **[Architectural Concept, not a single kernel]**
  - **Description:** This refers to the intelligent composition of simple, single-purpose kernels by the Host Orchestrator to form a `log_K(N)` reduction tree. This is the physical implementation of the **Primacy of Memory Strategy**, using the **Indirection Contract** (`offset_list`) to sum scattered partials without intermediate copies.
  - **Contextual Composition:** The Host Orchestrator renders the engine differently based on the data being processed and the tree depth:
    - **For Single-Stage Trees** (`num_stages == 1`): The tree uses `aggregate_stage_j` (storage-entry or compute-entry variant, depending on the source buffer's `precision_role`), optionally followed by `clip_intermediate_grad` for gradient stabilization.
    - **For Multi-Stage Trees** (`num_stages > 1`): The tree's leaf stage (stage 0) uses `reduce_k_fan_in_and_clip` when the source collection is storage-role, or `reduce_k_fan_in_and_clip_from_compute` when the source collection is compute-role (e.g., BCE loss partials). All interior stages (stage ≥ 1) use `reduce_k_fan_in_and_clip_from_compute`, reading the prior stage's COMPUTE_TYPE output. This stage-typed variant selection is an Orchestration-tier concern, resolved by inspecting the source buffer's `precision_role` from its `BufferDescriptor` (ADR-026). Each stage produces `ceil(N/K)` independent output nodes, each formed by summing exactly K input partials and applying an independent per-node L2 clip. This fused kernel eliminates intermediate global memory traffic between the sum and clip operations.
    - **For Diagnostic Reduction (Node 14):** Clipping is disabled (the threshold sentinel bypasses the clip path), producing a simple summation of `PARTIAL_Probs` and `PARTIALS_Loss_BCE`.
    - **For Gradient Reduction (Nodes 15 & 20):** Clipping is enabled at each stage, implementing the `Gradient Stabilization` policy's Quadratic Scaling threshold schedule.

- **`aggregate_stage_j`**: **[Utility Kernel]** A stateless, all-to-one summation kernel.
  - **Note:** The `aggregate_stage_j` is a conceptual role fulfilled by a tiered set of concrete kernels (e.g., `aggregate_register_reduce`, `aggregate_local_reduce`) selected by the Host Orchestrator based on reduction width. Each concrete kernel exists in two **precision variants** (ADR-026): the storage-entry variant (e.g., `aggregate_register_reduce`) reads `STORAGE_TYPE` partials via `load_storage()`, while the compute-entry variant (e.g., `aggregate_register_reduce_from_compute`) reads `COMPUTE_TYPE` intermediates directly. The Orchestration tier selects the variant based on the source buffer's `precision_role`.

  - **Contract:** Accepts a generic memory pool (`partial_collection`) and an indirection table (`offset_list`) and reduces **all** referenced partials into a **single**, contiguous `Intermediate Sum` buffer of `partial_width` elements. It performs no other logic.
  - **Invocation:** Used for single-stage trees in Nodes (14), (15a), and (20a).

- **`clip_intermediate_grad`**: **[Utility Kernel]** A stateless, partial-group-wise clipping kernel.
  - **Contract:** Accepts a single, contiguous `Intermediate Sum` buffer (the output of `aggregate_stage_j`) and a scalar threshold `T_j`. It computes a single L2 norm for the entire buffer and applies a single scaling factor if the norm is exceeded.
  - **Invocation:** Used for single-stage gradient stabilization in Nodes (15b) and (20b), following an `aggregate_stage_j` dispatch.

- **`reduce_k_fan_in_and_clip`**: **[Utility Kernel]** A stateless, fused K-fan-in reduction-and-clip kernel (ADR-019).
  - **Note:** This kernel is the multi-stage counterpart to the `aggregate_stage_j` + `clip_intermediate_grad` pipeline. It processes `ceil(N/K)` independent reduction nodes in a single dispatch, fusing summation and per-node clipping to avoid intermediate global memory traffic. Like the aggregate kernels, it exists in two **precision variants** (ADR-026): the storage-entry variant reads `STORAGE_TYPE` partials via `load_storage()`, while the compute-entry variant (`reduce_k_fan_in_and_clip_from_compute`) reads `COMPUTE_TYPE` intermediates directly.

  - **Contract:** Accepts a generic memory pool (`partial_collection`), a flat offset list with K consecutive entries per node, and a scalar clipping threshold `T_j`. For each node: sums K partials via the Indirection Contract, computes a per-node L2 norm, and conditionally scales the node's output independently. A negative threshold bypasses clipping (diagnostic mode); zero clips to zero norm. Absent partials in the tail node use a sentinel offset (`0xFFFFFFFF`).
  - **Invocation:** The storage-entry variant is used for leaf-stage (stage 0) reduction of storage-role collections. The compute-entry variant (`_from_compute`) is used for interior stages (stage ≥ 1) reading prior COMPUTE_TYPE outputs, and for leaf stages whose source is natively compute-role (e.g., BCE loss partials).

- **(16) `stabilize_and_reduce_grad_hidden_activations`**: **[Specialized Kernel]** A parametric reduction engine that applies a host-prescribed stabilization schedule to the `Grad_H` vector.
  - **Contract:** Receives a pre-computed threshold schedule and pre-accumulation threshold from the Orchestration tier. Executes a multi-stage `sum-then-clip` reduction over the contiguous SoA buffer produced by Node (13), applying the prescribed threshold at each clip stage. The schedule encodes the **Quadratic Scaling Policy**, rendered by the Orchestration tier for the target backend's reduction topology.
  - **Justification for Specialization:** This kernel works in synergy with the mandatory leaf-level clipping from Node (11) to provide a complete, two-stage stabilization strategy. A generic engine is unsuitable because:
    1. **Algorithmic Fidelity:** The `Grad_H` vector is an intermediate error signal whose internal directionality must be preserved. Node (11) provides Phase I (Safety) coarse clipping; this kernel provides Phase II (Fidelity) by carefully aggregating the sanitized vectors with staged `sum-then-clip` using the prescribed schedule.
    2. **Performance:** It is optimized for the contiguous data block guaranteed by the **`(13) Item Synchronization Point`**, avoiding the latency and overhead of a host-driven recursive engine.
    3. **Architectural Coherence:** It resolves the logical conflict of applying a scatter-gather primitive to a pre-gathered, contiguous buffer.
    4. **Pattern Consistency:** It receives host-prescribed thresholds in the same manner as the generic reduction engine (Nodes 14/15/20), eliminating the former jurisdictional exception where this kernel computed its own policy schedule.
- **(17) `backprop_shared_weights_chunk`**: Computes partial gradients for shared weights.
  - **Contract:** Consumes `Input_i`, a slice of the `Summed_Grad_H`, and a **chunk** of recomputed `hidden_i`. Adheres to the Placement Contract.
  - **Lifecycle Note:** The `PARTIAL_Grad_SW_i` buffer produced by this kernel **must** be processed by **Node (19)** before being aggregated by Node (20).
- **(18) `backprop_shared_biases_chunk`**: Computes partial gradients for shared biases.
  - **Contract:** Consumes a slice of the `Summed_Grad_H` and a **chunk** of recomputed `hidden_i`. Adheres to the Placement Contract.
  - **Lifecycle Note:** The `PARTIAL_Grad_SB_i` buffer produced by this kernel **must** be processed by **Node (19)** before being aggregated by Node (20).
- **(19) `clip_shared_gradients_chunk`**: Applies gradient clipping to the partial gradients from the **streamed shared path**.
  - **Contract:** Invoked inside the host's streaming loop for each _chunk_ of shared layer backpropagation (`Nodes 17, 18`). It calculates the L2 norm of the `Grad_SW` and `Grad_SB` vectors for that chunk and applies scaling. Its interface is simple and accepts only the two relevant gradient buffers. This is a mandatory stability primitive for the streaming data path.

---

#### **Final Update & Retrieval Kernels**

- **(21) `normalize_gradients`**: **[Utility Kernel]** Generic, element-wise scaling kernel.
  - **Contract:** Performs an element-wise division of an input buffer by a scalar value (`N`, the number of items in the batch). It consumes all `Summed_*` parameter gradient buffers (`Summed_Grad_ModW`, `Summed_Grad_ModB`, `Summed_Grad_Temps`, `Summed_Grad_SW`, `Summed_Grad_SB`) and produces their corresponding **`Final_*`**, normalized gradient buffers, which are essential for numerically stable mini-batch training.
- **(23) `D2H Async Copy`**: A non-blocking Device-to-Host transfer of `Final Probs`, signaling completion via `inference_event`.
- **(24) `adam_update`**: Generic optimizer kernel, invoked once per complete parameter group **per training batch**, after the **Batch Synchronization Point (22)**.
  - **Contract:** Applies the Adam update to an **entire parameter buffer in a single dispatch**. It is a stateless numerical function that receives a **`Final_Grad_*`** buffer (the **fully reduced and normalized batch-wide gradient**), its corresponding moment vectors (`m1`, `m2`), and the host-computed hyperparameters.
- **(25) `clamp_temperatures`**: Final utility kernel constraining temperature parameters.
- **`final_batch_event`**: This logical node represents the final device-side event that signals the completion of all computations for the training batch, including all parameter updates.

---

### **Validation Scenarios**

- **Scenario: The Iris Case (Sequential Execution Mode Validation)**
  - **Description:** A classic supervised learning task using the Iris dataset, where a small training batch of pre-labeled flower measurements is processed.
  - **API Expression:**
    ```python
    engine = Engine(model_spec, hyperparams)
    for epoch in range(num_epochs):
        result = engine.train_batch(X_train, y_train)
    print(result.predictions)
    ```
  - **Validation Focus:** Measures the end-to-end latency, ensuring the overhead of the full `clipping -> reduction -> normalization -> update` pipeline is minimal for small batches and doesn't hinder high-throughput, low-latency use cases. Under 5-line ceremony threshold for the common case.
  - **Key Insight:** Proves the **graceful degradation** of the parallel machinery. The Act/Learn split and batch-wide processing model are not costly abstractions on small problems but a foundational primitive that scales from `N=1` to `N=Infinity`.

- **Scenario: The Real-Time Trader (Event-Triggered Execution Mode Validation)**
  - **Description:** A high-frequency trading system where the system must make instant predictions (`Act` phase) and the `Learn` phase is triggered later when trade outcomes (labels) arrive.
  - **API Expression:**
    ```python
    ticket = engine.submit(market_snapshot)        # returns immediately; Act plan dispatched
    prediction = ticket.get_prediction()           # blocks until Act completes
    execute_trade(prediction)
    # ... seconds to minutes pass ...
    learn_handle = ticket.resolve(actual_outcome)  # returns immediately; Learn plan dispatched
    learn_handle.wait()                            # blocks until Learn completes
    ```
  - **Validation Focus:** Assesses the system's ability to release VRAM after the `Act` phase (all device buffers freed when `get_prediction()` returns) and efficiently recompute intermediates for the `Learn` phase. The ticket captures `market_snapshot` at submit time—even if the original array is overwritten, the system-enforced Act→Learn association guarantees correctness.
  - **Key Insight:** Demonstrates the architecture's strength in real-time, event-driven scenarios. The temporal split between Act and Learn is a first-class concept, not a special mode. The ticket holds only host-side state; its arbitrary lifetime in `ACT_COMPLETE` carries zero VRAM cost.

- **Scenario: The Marathon (Massive `epochs`)**
  - **Insight:** Guarantees **long-term stability** by delegating the sensitive `beta**t` calculation to the host and passing the final values to the **`(24) adam_update`** kernel, avoiding on-device precision loss and ensuring the optimizer remains mathematically correct indefinitely.

- **Scenario: The Hydra (Massive `num_heads`)**
  - **Insight:** Validates the intelligence of the **Reduction Planner**. For a massive number of heads, the host renders an efficient `log_K(N)` reduction tree, not just for diagnostics like loss, but for the core accumulation of **clipped** `Grad_ModW` partials from `Node (11)`, proving the engine is the heart of the learning algorithm itself.

- **Scenario: The Behemoth (Massive `hidden_dim`)**
  - **Insight:** Proves **scalability through strategic memory trade-offs**. When the `hidden_i` buffer is a bottleneck, the host's policy to **Cache** or **Recompute** it remains a critical, orthogonal optimization that works in synergy with the batch accumulation model.

- **Scenario: The Lexicon (Massive `output_classes`)**
  - **Insight:** Achieves **maximum hardware occupancy** by interleaving partial gradient computation with the subsequent reduction stages. For millions of classes, the Host Orchestrator plans a `log_K(N)` reduction tree for the class-dimension gradients, demonstrating that the engine scales efficiently over any problem dimension.

- **Scenario: The Rodeo (Extreme Instability Resilience)**
  - **Description:** A training task using a combination of factors designed to induce extreme numerical instability: a noisy dataset, an aggressively high learning rate, and a low-precision format (FP16). This combination is known to produce exploding partial gradients that would immediately overflow standard FP16 arithmetic.
  - **Validation Focus:** Confirms that the training run completes without encountering `NaN` or `Inf` values in the loss or parameters. The training loss, while potentially high or volatile due to the chaotic inputs, must remain within a finite, observable range and demonstrate a general downward trend.
  - **Key Insight:** Proves the critical role of the **bifurcated clipping stage** as a non-optional, foundational stability primitive. By enforcing a maximum norm on every partial gradient—both the complex, tiled module gradients via **`(11) clip_partial_gradients`** and the streamed shared-layer gradients via **`(19) clip_shared_gradients_chunk`**—_before_ any reduction takes place, the architecture guarantees that intermediate values in the reduction tree and the final summed gradients cannot overflow the limited dynamic range of low-precision formats. This ensures the engine remains numerically robust not just by policy, but **by design**, even under the most adverse training conditions.

- **Scenario: The Scientist's Repeater (Numerical Correctness Validation)**
  - **Description:** Two training runs are executed with identical hyperparameters and learning items, but with different training batch sizes (e.g., `N=16` then `N=32`).
  - **Validation Focus:** Confirms that the per-parameter weight updates are of a similar magnitude in both runs.
  - **Key Insight:** Proves the correctness of the **`(21) normalize_gradients`** kernel. By demonstrating that the learning dynamics are independent of batch size, it validates that the system is computing the true _average_ gradient, not the sum, which is critical for predictable hyperparameter tuning and algorithmic stability.

- **Scenario: The Data Tsunami (Massive `batch_size`)**
  - **API Expression:**
    ```python
    ticket = engine.submit(X_massive)              # returns immediately; Act plan dispatched
    probs = ticket.get_prediction()                # blocks until Act completes; (N × C) matrix
    learn_handle = ticket.resolve(y_massive)       # returns immediately; Learn plan dispatched
    learn_handle.wait()                            # reduction tree handles log_K(N) aggregation
    ```
  - **Insight:** This is the canonical test of the full parallel learning model. One ticket, one Act plan, one Learn plan—no per-item overhead. The plan's `ReductionTreeNode` handles the aggregation. It validates that the host correctly processes a large training batch (`N` items) by:
    1.  Rendering `N` parallel gradient computation tasks.
    2.  Correctly inserting the **`(11) clip_partial_gradients`** step for each item's tiled module gradients.
    3.  Knowing the write locations of all `N` partials via the **Placement Contract**.
    4.  Generating a valid **`offset_list`** pointing to these scattered partials, fulfilling the **Indirection Contract**.
    5.  Planning and executing the `log_K(N)` reduction tree, which uses this `offset_list` to sum the **clipped** partial gradients without intermediate copies.
    6.  Correctly inserting the **`(21) normalize_gradients`** step.
    7.  Enforcing the strict **`(22) Batch Synchronization Point`**, ensuring the **`(24) adam_update`** kernel only runs _after_ all preceding steps are complete.

- **Scenario: The Colossus (Holistic Stress Test)**
  - **Insight:** Tests the synergy of all scaling strategies under extreme, compound memory pressure. This proves the orchestrator's robustness by forcing it to compose a single, valid execution plan that correctly handles:
    1.  Per-item gradient clipping for the module path (`Node 11`).
    2.  Per-chunk gradient clipping for the shared path (`Node 19`).
    3.  The **Item Synchronization Point** (`Node 13`) for the `Grad_H` calculation.
    4.  The **Batch Synchronization Point** (`Node 22`) for the final update.
    5.  The **Recursive Reduction Engine** applied over multiple, independent data paths.
    6.  The **dual streaming models** for recomputing inputs on demand.
        This demonstrates that the system's sophisticated, multi-stage processing and two-tier synchronization model is not just theoretical but robust in practice.

- **Scenario: The Cross-Backend Arbiter (Multi-Backend Parity Validation)**
  - **Description:** The same `ExecutionPlan` (Iris-scale, both CCE and BCE configurations) is rendered and executed on all enabled backends. Per-kernel outputs and end-to-end results (`Final Probs`, updated parameters) are compared across backends within per-kernel tolerance bounds derived from `PrecisionConfig`.
  - **Validation Focus:** Confirms that the three-tier jurisdictional model holds: the Policy tier produces one plan, and every backend's Orchestration + Execution tiers produce numerically equivalent results. The CPU backend serves as the deterministic reference oracle, with its correctness independently established by Tier 2 analytical/numpy fixtures.
  - **Key Insight:** Proves the **behavioral equivalence** of structurally different rendering paths. OpenCL's per-tile imperative dispatch, Vulkan's single-dispatch command buffer recording, and CPU's synchronous pool dispatch converge to the same mathematical outcome. This is the definitive validation that the backend abstraction boundary (§5) and the execution plan model (§6) faithfully preserve the algorithm's semantics across all execution paradigms.

- **Scenario: The Alchemist (Mixed-Precision Fidelity Validation)**
  - **Description:** The same training task is executed in two precision configurations: `PrecisionConfig.float32()` and `PrecisionConfig.mixed_f16_f32()`. Both use the same `PrecisionConfig` with three roles — there is no mode switch.
  - **Validation Focus:** Confirms that the mixed configuration produces loss curves and final parameters tracking the FP32 baseline within tolerance, while achieving memory footprint reduction (FP16 storage halves FP32's footprint). Validates that buffer allocation sizes reflect `storage_dtype`, that accumulation uses `compute_dtype`, that the stabilization policy's safety ceiling derives from `compute_fp_format_max`, and that optimizer state maintains `state_dtype` fidelity.
  - **Key Insight:** Proves the three-role decomposition achieves its stated goal: narrow-storage bandwidth without narrow-compute fidelity loss.

- **Scenario: The Bandwidth Extremist (FP8 Storage Fidelity)**
  - **Description:** A training task is executed with `PrecisionConfig.fp8_e4m3()` and compared against `PrecisionConfig.mixed_f16_f32()` baseline. Both configurations use identical compute (FP32) and state (FP32) precision.
  - **Validation Focus:** Confirms that FP8 storage produces convergent training within tolerance of the FP16 baseline. Validates that storage-role buffer sizes are halved (8 bits vs. 16 bits), that quantization error does not prevent convergence, and that the host's scaling machinery maintains numerical correctness.
  - **Key Insight:** Proves the architectural claim that storage precision is independent of training fidelity. The 2× bandwidth reduction vs. FP16 (4× vs. FP32) is achieved without degrading the loss curve, because compute and state remain at FP32.

- **Scenario: The Alchemist II (FP64 State Stability Validation)**
  - **Description:** A training task is executed for $10^5$ steps using three configurations: `PrecisionConfig.float32()` (FP32 state), `PrecisionConfig.mixed_f32_f64_state()` (FP64 state with FP32 compute), and a numpy reference implementation using FP64 throughout. All use identical hyperparameters ($\beta_1 = 0.999$, $\beta_2 = 0.9999$) and identical pre-generated gradient sequences of known precision.
  - **Validation Focus:** Confirms that the FP64-state configuration's moment vectors remain stable within $10^{-4}$ relative error over extended training, while FP32-state configurations show measurable precision degradation at scale. The EMA accumulation arithmetic operates in `ACCUM_TYPE = max(COMPUTE_TYPE, STATE_TYPE)` (FP64), preserving state precision; the tolerance bound reflects the FP32 precision of gradient contributions rather than the accumulation itself. Validates that state-precision accumulation correctly isolates moment fidelity from compute-path throughput.
  - **Key Insight:** Proves that the state role, combined with state-precision accumulation, achieves its stated goal: unbounded training stability through extended-precision moment vectors, while allowing narrower `COMPUTE_TYPE` for throughput in transformative operations. The configuration `mixed_f32_f64_state()` provides a distinct tradeoff from `mixed_f32_f64()`: the former preserves FP64 only where erosion compounds (moments), the latter uses FP64 throughout (higher fidelity, lower throughput).
