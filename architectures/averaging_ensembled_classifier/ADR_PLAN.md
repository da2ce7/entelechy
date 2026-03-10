# Architecture Decision Records: Multi-Backend Refactoring Plan

## Preamble

This document catalogues the Architecture Decision Records (ADRs) required for refactoring the `averaging_ensembled_classifier` from its current OpenCL-only implementation to a multi-backend architecture supporting OpenCL, Vulkan, and CPU execution.

The foundational decision—**ADR-001: Backend Abstraction Boundary**—has been accepted. The system will abstract at the DAG/Phase level: a shared orchestration layer produces a backend-neutral execution plan expressed as a data structure, and each backend receives this plan and renders it using its native execution model. The full decision record is in `adr/ADR-001-backend-abstraction-boundary.md`.

This choice resolves or significantly narrows many downstream decisions. The remaining ADRs are organized into tiers reflecting the actual dependency chain and work order. Within each tier, ADRs are independent of each other and may be resolved in parallel.

**Status Key:**
- `DECIDED` — Accepted decision, recorded in `adr/`.
- `NARROWED` — ADR-001 constrains this to a single viable direction. Requires formal confirmation, not open exploration.
- `OPEN` — Genuine design decision with multiple viable options remaining.

---

## Tier 0: Foundation

### ADR-001: Backend Abstraction Boundary

**Status:** DECIDED — Option B (DAG/Phase-level abstraction)
**Full Record:** `adr/ADR-001-backend-abstraction-boundary.md`

The shared orchestration layer produces a backend-neutral `ExecutionPlan` data structure. Each backend implements a `PlanRenderer` that consumes the plan and executes it using its native model—OpenCL enqueues imperatively with event chains, Vulkan records a command buffer with pipeline barriers, CPU calls `pool_dispatch_and_wait` sequentially.

**Boundary definition:**

```
┌─────────────────────────────────────────────────────────┐
│              Shared Orchestration Layer                  │
│                                                         │
│  ModelSpec, MemoryLayout, ParameterSpace, Tiling,       │
│  StabilizationPolicy, ReductionTreePlan,                │
│  KernelContract (validation), ExecutionPlanBuilder       │
│                                                         │
│  Produces: ExecutionPlan (backend-neutral data)         │
└───────────────────────┬─────────────────────────────────┘
                        │  plan data structure
┌───────────────────────┼─────────────────────────────────┐
│              Backend Execution Layer                     │
│                                                         │
│  Consumes: ExecutionPlan                                │
│  Owns: context/queue, buffer allocation, kernel         │
│        dispatch, synchronization, D2H transfer          │
│                                                         │
│  ┌──────────┐  ┌──────────┐  ┌──────────┐              │
│  │  OpenCL  │  │  Vulkan  │  │   CPU    │              │
│  │ Backend  │  │ Backend  │  │ Backend  │              │
│  └──────────┘  └──────────┘  └──────────┘              │
└─────────────────────────────────────────────────────────┘
```

**Design principle:** The plan boundary stops at each kernel's public interface. Plan nodes describe kernel identity, logical buffer bindings, tile decomposition, dependency edges, and policy parameters. Internal kernel algorithms (e.g., Node 16's multi-stage reduction) are never expressed in the plan.

---

## Tier 1: The Plan Data Structure

These ADRs define what the execution plan must be capable of expressing. They are the highest-priority design work because every subsequent tier depends on the plan vocabulary.

### ADR-002: Plan Node Types & Synchronization Structure

**Status:** NARROWED — Dependency edges in plan; no event objects in shared layer

**Context:**
ADR-001 eliminates event objects from the shared layer. The plan must instead express:
1. **Dependency edges** between nodes (which node's output feeds which node's input).
2. **Named synchronization points**: `inference_event` (Act→Host), `final_batch_event` (Learn→Host), the Item Synchronization Point (Node 13), and the Batch Synchronization Point (Node 22).
3. **Concurrency opportunities**: which nodes may execute in parallel (e.g., Nodes 8, 9, 10 within a tile).

Each backend maps this structure to its native sync model. OpenCL inserts `wait_for` event lists. Vulkan inserts `vkCmdPipelineBarrier` commands at recording time. CPU executes nodes sequentially (sync is free).

**Remaining Decision:**
What is the concrete taxonomy of plan node types? A minimal set:

| Node Type                | Semantics                                                                                          |
| :----------------------- | :------------------------------------------------------------------------------------------------- |
| `KernelDispatchNode`     | A single logical kernel invocation with tile count, buffer bindings, and scalar parameters.        |
| `ReductionTreeNode`      | A multi-stage reduction tree (stages, fan-ins, offset lists, per-stage thresholds). See ADR-003.   |
| `StreamingLoopNode`      | A parametric loop over a sub-DAG. See ADR-004.                                                     |
| `BarrierNode`            | A named synchronization point (Node 13, Node 22).                                                  |
| `RetrievalNode`          | A host-accessible result extraction point (`inference_event`, `final_batch_event`).                |

**Complexity Ceiling Constraint:** The plan is a **static, fully-typed DAG of the above node types**. It does not support conditional branches, dynamic dispatch, or nested loops beyond the single `StreamingLoopNode` primitive. If a future optimization requires constructs beyond this vocabulary, CONCEPT.md §1 (Architectural Elegance Feedback) mandates formalizing that construct as a new node type — not stretching the existing vocabulary.

**Tensions:**
- The plan must carry enough information for CONTRACT.md Article 1.4a pre-dispatch validation to remain in the shared layer. Each `KernelDispatchNode` must include the validated `KernelContract` data (shapes, preconditions) — not just a kernel name.
- Vulkan's single-dispatch-for-N-tiles model means a `KernelDispatchNode` with `tile_count=N` is rendered as one `vkCmdDispatch(N,1,1)`, while OpenCL renders it as N `clEnqueueNDRange` calls. The node must not prescribe the dispatch granularity.

---

### ADR-003: Reduction Tree Plan Representation

**Status:** NARROWED — Shared `ReductionTreePlan` embedded in the execution plan

**Context:**
The Recursive Clip-Aggregation Engine (Nodes 14, 15, 20) is the canonical example of why ADR-001 chose the plan-level boundary. The tree's mathematical structure — stage count, fan-in `K`, offset lists, Quadratic Scaling Policy thresholds (`T_j = T_algorithmic + λ·j²`) — is identical across all backends. Only the dispatch mechanics differ.

ADR-001 directly resolves this as a shared `ReductionTreePlan` embedded in the execution plan. The `StabilizationPolicy` module and offset-list construction remain in the shared layer. Each backend's renderer interprets the tree natively.

**Remaining Decision:**
The `ReductionTreePlan` must specify, per stage:
- Fan-in count and kernel tier selection (`aggregate_register_reduce` vs. `aggregate_local_reduce`).
- Offset list (integer array of memory displacements).
- Whether a `clip_intermediate_grad` step follows the aggregation (gradient paths) or not (diagnostic paths).
- The computed threshold `T_j` for the clip step.
- Intermediate buffer sizing.

The renderer is responsible for: uploading offset lists, allocating intermediate buffers, dispatching the kernels, and inserting synchronization between stages.

**Tensions:**
- The offset lists are pure integer arrays. The shared layer computes them; the backend uploads them in its native way (OpenCL: `cl.enqueue_copy` to a `cl.Buffer`; Vulkan: mapped staging buffer; CPU: direct pointer).
- For diagnostic reductions (Node 14), no clipping occurs — the plan must express "sum-only" vs. "sum-then-clip" per stage.

---

### ADR-004: Streaming Loop Representation (Phase III)

**Status:** OPEN

**Context:**
CONCEPT.md's Phase III (True Streaming for `Grad_SW` / `Grad_SB`) involves a host-side loop that iterates over chunks, dispatching a per-chunk kernel sequence (Nodes 17→18→19). The backends handle this loop radically differently:

- **OpenCL:** Host Python loop; per-chunk `clEnqueueNDRange` calls with event chains.
- **Vulkan:** Loop at *recording time*; per-chunk `vkCmdPushConstants` + `vkCmdDispatch` + `vkCmdPipelineBarrier`, baked into the command buffer.
- **CPU:** Host loop; per-chunk `pool_dispatch_and_wait`.

**Decision Required:**
How does the plan express the streaming loop?

- **(A) Abstract loop descriptor.** The plan carries a `StreamingLoopNode` with: chunk count, per-chunk parameter deltas (offsets, indices), and the sub-DAG body (a small sequence of `KernelDispatchNode`s). Each backend renders the loop natively — Vulkan records N iterations into one command buffer; OpenCL iterates in Python; CPU iterates with `pool_dispatch_and_wait`.

- **(B) Pre-expanded flat DAG.** The plan builder unrolls the loop into N concrete node sequences at plan-construction time. Simpler plan vocabulary, but: inflates plan size linearly with chunk count; prevents Vulkan from recognizing loop structure for potential batch optimization; loses the semantic signal that these N sequences are structurally identical.

**Tensions:**
- Option A demands the `StreamingLoopNode` be the **only** loop primitive — if it nests or generalizes, the plan risks becoming an IR. The constraint must be explicit: `StreamingLoopNode.body` is a flat sequence of `KernelDispatchNode`s, never containing another `StreamingLoopNode`.
- The chunk count varies per batch (determined by host memory assessment). Under both options the plan is constructed fresh each batch, but Option A keeps the plan compact and semantically clear.
- Vulkan's ability to record the entire loop body is a key performance characteristic that Option B would forfeit.

---

### ADR-005: Node 16 Opacity in the Plan

**Status:** NARROWED — Backend-owned; plan treats Node 16 as opaque

**Context:**
CONCEPT.md designates Node 16 (`stabilize_and_reduce_grad_hidden_activations`) as a Specialized Kernel with internal multi-stage reduction. Its internal structure — fan-in calculation from `get_local_size(0)`, per-stage threshold synthesis — is inherently hardware-specific.

ADR-001's design principle ("plan boundary stops at the kernel's public interface") directly resolves this. The plan specifies Node 16 as a single `KernelDispatchNode` with policy parameters (`T_algorithmic`, `λ`, `policy_max_k`, `fp_max`). The kernel contract (in `kernels.cl.h`) already specifies the internal algorithm in its `Behavioral Invariants` — this is a device-side concern, not a plan-level concern.

**Remaining Decision:**
None — this is resolved. The policy parameters are computed by the shared `StabilizationPolicy` module and embedded in the plan node's scalar parameter set. Each backend passes them to its native Node 16 implementation.

---

## Tier 2: Shared Layer Contracts

These ADRs define the interfaces that the shared orchestration layer depends on. They must be resolved before the shared layer can be purified of `pyopencl` imports.

### ADR-006: Hardware Profile

**Status:** NARROWED — Shared `HardwareProfile` dataclass

**Context:**
Plan construction requires hardware constants — `simd_width`, `c_tile_size`, `max_local_mem`, `cache_line_bytes` — to compute tiling, padding, reduction tree depth, and Node 16's `policy_max_k`. These must be available in the shared layer. Each backend fills the profile via its native discovery mechanism:

- **OpenCL:** Runtime discovery via `cl.device_info`.
- **Vulkan:** `VkPhysicalDeviceSubgroupProperties` and memory queries.
- **CPU:** Compile-time constants (`__AVX512F__` → `simd_width=16`, etc.) reported by the compiled shared library.

**Remaining Decision:**
The canonical constant set. A minimal proposal:

| Constant             | Type  | Source (OpenCL)          | Source (Vulkan)                        | Source (CPU)             |
| :------------------- | :---- | :----------------------- | :------------------------------------- | :----------------------- |
| `simd_width`         | `int` | `CL_KERNEL_PREFERRED_WORK_GROUP_SIZE_MULTIPLE` | `subgroupSize`        | Compile-time ISA flag    |
| `c_tile_size`        | `int` | Derived from simd_width  | Derived from subgroup size             | Derived from simd_width  |
| `max_local_mem`      | `int` | `CL_DEVICE_LOCAL_MEM_SIZE` | `maxComputeSharedMemorySize`         | N/A (no local memory)   |
| `max_work_group_size`| `int` | `CL_DEVICE_MAX_WORK_GROUP_SIZE` | `maxComputeWorkGroupSize[0]`  | Thread pool size         |
| `cache_line_bytes`   | `int` | `CL_DEVICE_GLOBAL_MEM_CACHELINE_SIZE` | Assumed 64           | Platform-specific        |
| `global_mem_bytes`   | `int` | `CL_DEVICE_GLOBAL_MEM_SIZE` | `VkPhysicalDeviceMemoryProperties` | `sysconf(_SC_PHYS_PAGES)` etc. |

`ParameterSpace`, `MemoryLayout`, `StabilizationPolicy`, and the plan builder consume this dataclass. These modules require zero backend-specific changes once `DiscoveredArchConstants` is replaced.

**Tensions:**
- The CPU backend's SIMD width is a compile-time constant. The `HardwareProfile` must accept pre-determined values without requiring a "discovery" phase.
- `max_local_mem` is meaningless for the CPU backend. The profile must tolerate `None` or sentinel values for inapplicable constants.

---

### ADR-007: KernelSignature Contract/Binding Split

**Status:** NARROWED — Split into shared `KernelContract` + backend-specific `KernelBinding`

**Context:**
The `KernelSignature` classes currently produce OpenCL argument lists and return `cl.Event`. ADR-001 demands that plan construction validates kernel contracts without touching backend types. CONTRACT.md Article 1.4 (Collaborative Interface Verifiability) requires that all validation occurs pre-dispatch — this validation is inherently shared.

The split:
- **`KernelContract`** (shared): Encodes shapes, padding contracts, calculability proofs, validation preconditions, and placement strategies from CONTRACT.md. Used by the plan builder to validate node correctness at plan-construction time.
- **`KernelBinding`** (backend-specific): Translates a validated contract into native dispatch arguments — positional `clSetKernelArg` calls (OpenCL), push constant structs + descriptor sets (Vulkan), typed arg structs (CPU).

**Remaining Decision:**
How does the `KernelContract` handle parameters that exist in some backends but not others (the `flat_tile_index` divergence)?

- **(A) Abstract placement key.** The contract specifies "this kernel is a Partial Renderer with N tiles using placement strategy `grid_mod_cls`." The *mechanism* — host scalar, `gl_WorkGroupID.x`, `task_index` — is a binding concern. The contract's calculability proof references the abstract placement key, not a specific parameter name.

- **(B) Superset parameter list with backend annotations.** The contract lists all parameters including `flat_tile_index`, annotated as `[OpenCL-only]`. Risks: pollutes the shared contract with backend concerns; must be extended for every new backend.

Option A is strongly favored. The Placement Contract (CONTRACT.md Article 3.2) already defines strategies abstractly (`grid_mod_cls`, `linear_batch`). The contract specifies the *strategy and key domain*; the binding specifies *how the key is communicated*.

**Tensions:**
- CONTRACT.md Article 1.4.1 demands that all calculability proof terms exist in the interface. If `flat_tile_index` is abstracted away, the proof must reference the abstract placement key. This requires a CONTRACT.md amendment — specifically, adding a "Backend Binding" section to Article 3.2.3 that acknowledges the mechanism is backend-specific while the strategy is universal.

---

### ADR-008: Precision Configuration

**Status:** NARROWED — Backend resolves native types independently

**Context:**
`PrecisionContext` currently carries `SCALAR_C_TYPE_NAME` (explicitly "for the OpenCL compiler"). Under the plan model, the shared layer works exclusively with numpy types (`numpy.float16`, `numpy.float32`). Backend renderers map to their native type systems.

The following values remain in the shared layer as they are precision-derived but backend-neutral:
- `FP_FORMAT_MAX`: Used by `StabilizationPolicy` for safety ceiling calculations (CONCEPT.md §3.4).
- `NUMERICAL_STABILITY_EPSILON`: Used by plan builder for kernel scalar parameters.
- `numpy_dtype`: Used by `MemoryLayout` for byte-size calculations.

**Remaining Decision:**
Does `PrecisionContext` retain a backend-neutral `bit_width` field that backends map from, or does each backend independently determine its type from the numpy dtype?

This is a minor design question. The practical answer: `PrecisionContext` carries `numpy_dtype`, `fp_format_max`, and `epsilon`. Backends derive everything else from `numpy_dtype` (`float16` → `half` for OpenCL, `VK_FORMAT_R16_SFLOAT` for Vulkan, `_Float16` or emulated for CPU).

**Tensions:**
- CPU FP16 is not universally hardware-accelerated. The CPU backend may need to declare FP16 as unsupported, which the shared layer must handle gracefully (e.g., by refusing to construct a plan for an unsupported precision).

---

## Tier 3: Backend Renderer Interface

These ADRs define the contract that every backend must satisfy. They depend on the plan vocabulary (Tier 1) and shared layer contracts (Tier 2).

### ADR-009: Buffer Lifecycle in the Plan Model

**Status:** NARROWED — `BufferHandle` as universal plan-level token; backend allocates

**Context:**
The plan references buffers by logical name (e.g., `hidden_activations`, `partial_grad_weights_module`). Plan construction never allocates — it declares "this node produces buffer X with shape Y and padding Z." The backend's `PlanRenderer` maps logical buffer names to physical allocations.

The existing `BufferHandle` token is the natural plan→renderer handoff mechanism. The plan builder assigns handles; the renderer allocates backing memory.

**Remaining Decision:**
Does the plan prescribe buffer *reuse* (e.g., "buffer A can be freed after Node 12 and its memory reused for buffer B"), or does it leave lifetime management entirely to the renderer?

- **(A) Plan prescribes lifetimes.** The plan annotates each buffer with its producing node and last-consuming node. The renderer uses this information to optimize memory reuse. This is the more principled approach — the shared layer has full DAG visibility and can compute optimal lifetimes.

- **(B) Renderer owns lifetimes.** The plan declares buffers but not their lifetimes. Each renderer analyzes the plan to determine reuse opportunities. Risk: duplicated analysis logic across backends.

Option A is favored. Buffer lifetime is a property of the DAG, not the dispatch model. The shared layer should compute it once.

**Tensions:**
- The activation lifecycle decision (Cache vs. Recompute, CONCEPT.md §5) affects buffer lifetimes and must be reflected in the plan. This is already a plan-construction concern.
- Vulkan backends may further optimize by suballocating from large `VkDeviceMemory` blocks. The plan's lifetime annotations enable this without prescribing it.

---

### ADR-010: D2H Transfer & Phase Sync Points

**Status:** OPEN

**Context:**
The plan's `RetrievalNode` describes a point where results become host-accessible. The renderer must return something to the shared orchestrator that represents "this data is ready." The backends diverge:

- **OpenCL:** `cl.enqueue_copy` → numpy array, signaled via `cl.Event`.
- **Vulkan:** `vkCmdCopyBuffer` to staging → `vkWaitForFences` → `memcpy` from mapped pointer.
- **CPU:** Direct pointer access — the buffer *is* host memory. Zero-cost.

**Decision Required:**
What does the renderer return for retrieval points?

- **(A) Synchronous `retrieve_results()` method.** The renderer exposes a blocking call that returns a numpy array. Simple, but forces synchronous blocking even where the host could overlap computation with the Learn phase trigger decision.

- **(B) `Future`-like handle.** The renderer returns a lightweight handle with `.wait()` and `.result()` methods. OpenCL wraps its event+buffer. Vulkan wraps its fence+staging pointer. CPU wraps an already-resolved value. This preserves temporal decoupling for Event-Triggered Execution Mode (CONCEPT.md §5).

**Tensions:**
- CONCEPT.md §4 requires that `inference_event` and `final_batch_event` be expressible as named synchronization points. A Future-like handle naturally represents "this named point has been reached."
- The Event-Triggered Execution Mode requires the host to read Act-phase results *before* deciding when to trigger Learn. A blocking API (Option A) forces the host to commit to retrieving results at a specific point. A Future (Option B) allows deferred retrieval.
- Option A is simpler; Option B is more faithful to the architecture's event-driven scenarios.

---

### ADR-011: CCE/BCE Strategy Delegation

**Status:** NARROWED — Backend-local decision

**Context:**
CONCEPT.md §3 explicitly permits both Strategy A (single kernel with runtime flag) and Strategy B (separate kernels) for CCE/BCE divergence. The existing `ProblemTypeStrategy` in `execution_plan.py` already abstracts this.

Under the plan model, the plan node for Nodes 6/7 carries a `problem_type` enum (`CCE` or `BCE`). The backend decides whether to:
- Dispatch a single kernel with a runtime flag (OpenCL's current approach).
- Select from pre-compiled pipeline variants via specialization constants (Vulkan).
- Call separate C functions (CPU).

**Remaining Decision:**
None — this is resolved. The plan conveys intent (`problem_type`), the renderer conveys mechanism.

---

## Tier 4: Physical Architecture & Tooling

These ADRs govern how the code is organized on disk, how it builds, and how Python talks to native backends. They depend on the logical architecture (Tiers 1–3) being settled.

### ADR-012: Module Factoring & Services Dissolution

**Status:** NARROWED — Two-layer physical split; `Services` dissolved

**Context:**
Under the plan model, the shared layer needs only `ModelSpec`, `HardwareProfile`, `PrecisionContext`, `MemoryLayout`, `ParameterSpace`, and `StabilizationPolicy` to construct the plan. It never needs a command queue, kernel executor, or buffer manager. The current `Services` dataclass — which bundles `cl.CommandQueue`, `KernelExecutor`, and `BufferManager` alongside `ModelSpec` — must be dissolved.

**Remaining Decision:**
The physical directory structure. A concrete proposal:

```
src/
├── __init__.py
├── model_spec.py              # Shared: model configuration
├── memory_layout.py           # Shared: padded shapes, buffer sizing
├── parameter_space.py         # Shared: parameter group definitions
├── stabilization_policy.py    # Shared: Quadratic Scaling thresholds
├── workload_primitives.py     # Shared: tiling, chunking
├── arch_primitives.py         # Shared: PrecisionContext, HardwareProfile
├── execution_plan.py          # Shared: plan data structures, plan builder
├── batch_processor.py         # Shared: plan construction orchestrator
├── kernel_contracts/          # Shared: validation logic (ex-kernel_signatures/)
│   ├── __init__.py
│   ├── phase_1_act.py
│   ├── phase_2_learn_A_production.py
│   └── ...
└── backends/
    ├── __init__.py            # Backend Protocol definition
    ├── opencl/
    │   ├── __init__.py
    │   ├── context_manager.py # cl.Context, cl.CommandQueue
    │   ├── buffer_manager.py  # cl.Buffer allocation, enqueue_copy
    │   ├── plan_renderer.py   # PlanRenderer implementation
    │   ├── kernel_bindings.py # KernelBinding: cl arg setting
    │   └── hardware_discovery.py
    ├── vulkan/
    │   └── ...
    └── cpu/
        └── ...
```

The current `graph_recipes.py` (984 lines) splits into:
- Plan-construction functions (move to `execution_plan.py` or `batch_processor.py`).
- OpenCL dispatch functions (move to `backends/opencl/plan_renderer.py`).

The current `launcher_infra.py` splits into:
- `BufferHandle` definition (stays shared in `execution_plan.py`).
- `BufferManager`, `KernelExecutor`, `Services` → dissolve into `backends/opencl/`.
- `HostView` → stays shared (it's a plan-level retrieval artifact).

The current `compute_patterns.py` (reduction dispatch logic) moves to `backends/opencl/plan_renderer.py`.

**Tensions:**
- The `kernel_signatures/` → `kernel_contracts/` rename reflects the Contract/Binding split (ADR-007). The shared code retains validation; the backend-specific binding code moves to each backend.
- `main_orchestrator.py` straddles both layers. Its plan-construction logic is shared; its `pyopencl` setup code moves to the OpenCL backend.

---

### ADR-013: Kernel Source Strategy

**Status:** OPEN

**Context:**
Three backends require three kernel implementations in three languages: OpenCL C, GLSL (→ SPIR-V), and C with SIMD intrinsics. All must honor the same CONCEPT.md DAG semantics and CONTRACT.md parameter grammar.

**Decision Required:**
How are kernel implementations organized and their contracts enforced?

- **(A) Shared header as reference spec, independent implementations.** `kernels.cl.h` remains the sole authoritative specification. GLSL shaders and C functions are written independently but must satisfy the same semantic contracts. Compliance is mechanically verified:
  - At plan-construction time by the shared `KernelContract` validation.
  - At test time by a shared numerical-equivalence test suite that runs identical inputs through all available backends.

- **(B) Three independent kernel sets with shared contract tests only.** Each backend maintains its own kernel source and its own contract documentation. Risk: contract drift between backends; `kernels.cl.h` loses its "single source of truth" status.

Option A is favored. `kernels.cl.h` has a rich, formally specified contract (`@kernel_contract` blocks, `@param` blocks with calculability proofs). This specification is language-neutral in substance even though its notation is OpenCL C. Elevating it to the role of "reference specification" — not "OpenCL-specific header" — is a natural evolution.

**Tensions:**
- CONTRACT.md Article 7 (Canonical Interface Instantiation) uses OpenCL C notation. A pragmatic amendment: Article 7 specifies the *semantic structure* of the contract; the OpenCL C syntax is the *reference notation*. GLSL and C implementations must satisfy the same semantic structure.
- The Vulkan backend eliminates `flat_tile_index` from the shader interface (replaced by `gl_WorkGroupID.x`). Per ADR-007, the contract specifies the abstract placement key; the GLSL shader's use of `gl_WorkGroupID.x` is a binding-level concern documented in the GLSL shader's own comments.

---

### ADR-014: Build System Integration

**Status:** OPEN

**Context:**
Meson is the build system. The OpenCL backend compiles kernels at runtime. New backends introduce build-time compilation:
- **Vulkan:** GLSL → SPIR-V via `glslc` or `glslangValidator`.
- **CPU:** C → shared library with ISA-specific flags (`-mavx2`, `-mavx512f`, `-mfpu=neon`).

**Decision Required:**
How does `meson.build` accommodate the new targets?

- **(A) Conditional backend targets.** Meson detects available toolchains and conditionally enables backends. Each backend's build artifacts are separate. The Python package exposes a `get_available_backends()` function.

- **(B) Backend as optional Meson subproject.** Each backend is a self-contained Meson subproject with its own `meson.build`. The top-level project includes them conditionally. Cleaner isolation but more build system complexity.

**Tensions:**
- STRUCTURE.md §2.4 (Build-Time System Guarantees) demands that a successful build guarantees correctness. If a backend builds, it must work. If a toolchain is missing, the backend must be cleanly absent — not half-built.
- The CPU backend requires ISA detection. Meson's `cc.has_argument('-mavx2')` and `cc.sizeof('__m256')` handle this.
- The Vulkan backend requires the Vulkan SDK. CI environments may not have it. The build must degrade gracefully.

---

### ADR-015: Python ↔ Native Backend Interop

**Status:** OPEN

**Context:**
The Python orchestrator builds the plan; native backends render it. The interop boundary is the `PlanRenderer.render(plan)` call. This is a significant architectural advantage of the plan model: the Python↔native crossing happens *once per batch*, not once per kernel dispatch.

**Decision Required:**
What is the interop mechanism?

- **(A) `cffi` for both backends.** The CPU backend compiles to a shared library with a C API; `cffi` calls it. The Vulkan backend wraps the Vulkan C API via `cffi`. Uniform mechanism, proven tooling, no C++ dependency.

- **(B) `nanobind`.** A thin C++ binding layer exposes each backend as a Python extension module. More Pythonic API, better type safety, but adds a C++ build dependency.

- **(C) `ctypes` (CPU) + `vulkan` Python package (Vulkan).** Use existing Python Vulkan bindings (`vulkan` or `vkbind`). Risk: Python Vulkan bindings may not support required extensions (`VK_KHR_push_descriptor`, push descriptor sets).

- **(D) Per-backend choice.** Allow each backend to choose its own interop mechanism. The shared layer defines only the `PlanRenderer` Python protocol; how the backend implements it is unconstrained.

**Tensions:**
- The CPU backend's hot path is entirely in C with SIMD intrinsics. Under the plan model, Python calls `render(plan)` once, and the C code processes the entire plan without crossing back into Python until the batch is done. Per-dispatch interop overhead is eliminated — this favors any mechanism that can efficiently pass the plan data structure.
- The Vulkan backend similarly receives the plan once and records a complete command buffer. The interop overhead is minimal.
- STRUCTURE.md §3.3 specifies a "Decoupled Runtime Bridge" via configured launcher scripts. The interop mechanism must be compatible with this model.

---

## Tier 5: Verification

### ADR-016: Test Strategy for Multi-Backend Correctness

**Status:** OPEN

**Context:**
The plan model introduces a new, powerful testing layer: **plan-level testing** that requires no backend at all. A given `ModelSpec` + `HardwareProfile` should produce a deterministic execution plan. This plan can be validated against CONCEPT.md and CONTRACT.md invariants purely as a data structure.

**Decision Required:**
How is correctness validated across backends?

- **(A) Layered strategy:**
  1.  **Plan tests** (no backend): Validate that plan construction produces correct DAG structure, reduction tree depth, threshold schedules, buffer lifetimes, and streaming loop parameters for known inputs.
  2.  **Backend unit tests** (per-backend): Validate dispatch mechanics, buffer allocation, synchronization correctness, and native kernel execution.
  3.  **Integration tests** (cross-backend): Run the same plan through all available backends and assert numerical equivalence within precision-aware tolerances.
  4.  **Scenario tests** (per CONCEPT.md): Run the full Validation Scenarios (Iris, Trader, Marathon, etc.) on all available backends.

- **(B) Parameterized-only.** Every test is parameterized over a `backend` fixture. Simpler, but loses the ability to test plan construction independently and cannot run in environments without any backend (e.g., CI without GPU).

- **(C) Reference oracle.** CPU with FP64 serves as ground truth. Other backends are validated against it. Handles cases where all backends might produce identical wrong results.

Option A is favored. Plan-level testing is a unique advantage of the plan model and should be exploited. Integration tests (A.3) subsume the parameterized approach (B). The oracle model (C) can be layered on top.

**Tensions:**
- FP16 results will differ between backends due to different intermediate precision handling and SIMD reduction order (floating-point associativity). Tolerances must be precision-aware and documented.
- Plan-level tests provide fast, device-free CI coverage. This is a significant practical benefit.

---

## Tier 6: Migration

### ADR-017: Incremental Migration Path

**Status:** OPEN

**Context:**
The refactoring touches nearly every module. ADR-001's decision constrains the migration to a specific path: the shared layer must be purified first, then the plan data structure introduced, then `graph_recipes.py` split, then new backends added.

**Proposed Phases:**

**Phase 0: Protocol Extraction.**
Extract backend-neutral `Protocol` types from current OpenCL code. Define `HardwareProfile`, `KernelContract` (the shared half of current `KernelSignature`), and `BufferHandle`. Current OpenCL code continues to work unchanged — Protocols are not enforced yet, just defined. No behavioral change. All tests pass.

**Phase 1: Shared Layer Purification.**
Remove all `pyopencl` imports from: `execution_plan.py`, `stabilization_policy.py`, `workload_primitives.py`, `model_spec.py`, `memory_layout.py`, `parameter_space.py`, `arch_primitives.py`. These modules depend only on Phase 0 abstractions and standard library types. `DiscoveredArchConstants` is replaced by `HardwareProfile`. All tests pass.

**Phase 2: Plan Data Structure.**
Define the plan node types (ADR-002), `ReductionTreePlan` (ADR-003), and `StreamingLoopNode` (ADR-004). Implement `ExecutionPlanBuilder` that produces a plan from `ModelSpec` + `HardwareProfile` + batch parameters. Write plan-level tests (ADR-016, layer 1). The plan builder exists alongside the current OpenCL execution path — it is not yet *used* for dispatch. All tests pass.

**Phase 3: OpenCL Plan Renderer.**
This is the critical step. Refactor `graph_recipes.py` and `compute_patterns.py` into an `OpenCLPlanRenderer` that consumes an `ExecutionPlan` and produces the same OpenCL dispatch sequence as the current code. The `BatchProcessor` switches from direct recipe calls to plan-build-then-render. The current behavior is preserved but the code path is fundamentally restructured. All existing integration tests validate the transition.

**Phase 4: CPU Backend.**
Implement `CPUPlanRenderer` against the established plan interface. Write cross-backend integration tests (ADR-016, layer 3). Validate with scenario tests.

**Phase 5: Vulkan Backend.**
Implement `VulkanPlanRenderer`, including SPIR-V compilation pipeline and command buffer recording. Validate with the same cross-backend test suite.

**Risk Assessment:**
- Phases 0–1: Zero behavioral change. Low risk. Mechanically verifiable.
- Phase 2: New code only — no existing code modified. Low risk.
- Phase 3: **Highest risk.** This restructures the core execution path. Mitigation: the plan-level tests from Phase 2 verify plan correctness independently. The integration tests verify end-to-end equivalence. A feature flag can toggle between old (direct dispatch) and new (plan-then-render) paths during transition.
- Phases 4–5: New backends against a stable interface. Moderate risk, isolated to the new backend.

---

## Summary Matrix

| ADR | Tier | Status | Core Question | Blocked By |
|:----|:-----|:-------|:--------------|:-----------|
| 001 | 0 | **DECIDED** | Where is the abstraction boundary? | — |
| 002 | 1 | NARROWED | What are the plan node types? | 001 |
| 003 | 1 | NARROWED | How are reduction trees represented? | 001 |
| 004 | 1 | OPEN | How are streaming loops represented? | 001 |
| 005 | 1 | NARROWED | Is Node 16 opaque in the plan? | 001 |
| 006 | 2 | NARROWED | How is the hardware profile shared? | 001 |
| 007 | 2 | NARROWED | How does KernelSignature split? | 001 |
| 008 | 2 | NARROWED | How does precision configuration flow? | 006 |
| 009 | 3 | NARROWED | How are buffers referenced in the plan? | 002 |
| 010 | 3 | OPEN | What does the renderer return for D2H? | 002, 009 |
| 011 | 3 | NARROWED | Is CCE/BCE strategy backend-local? | 007 |
| 012 | 4 | NARROWED | How are modules physically organized? | 007, 009 |
| 013 | 4 | OPEN | How are kernel sources organized? | 007 |
| 014 | 4 | OPEN | How does the build system accommodate backends? | 012 |
| 015 | 4 | OPEN | What is the Python↔native interop? | 012, 014 |
| 016 | 5 | OPEN | How is cross-backend correctness tested? | 013, 014 |
| 017 | 6 | OPEN | What is the phased migration strategy? | All |

## References

- [ADR-001 Full Record](adr/ADR-001-backend-abstraction-boundary.md)
- [CONCEPT.md](CONCEPT.md) — Governing architectural principles
- [CONTRACT.md](CONTRACT.md) — Host-device interface contract
- [CPU_BACKEND.md](CPU_BACKEND.md) — CPU backend architecture
- [VULKAN_BACKEND.md](VULKAN_BACKEND.md) — Vulkan backend architecture
