# Design Document: Averaging Ensembled Classifier

**Revision:** 2.1 — Post-Phase 3  
**Last Updated:** 2026-03-29  
**Scope:** Implementation design for the multi-backend averaging ensembled classifier architecture.

---

## Contents

| § | Section | Scope |
|:--|:--------|:------|
| 1 | [Document Role](#1-document-role) | Authority hierarchy, document positioning |
| 2 | [System Architecture Overview](#2-system-architecture-overview) | Three-tier model, backend abstraction boundary |
| 3 | [Execution Plan Data Structures](#3-execution-plan-data-structures) | Node taxonomy, plan types, hardware profile, precision, buffers |
| 4 | [Kernel Contract Model](#4-kernel-contract-model-adr-007-adr-011) | Contract/Binding split, CCE/BCE strategy delegation |
| 5 | [Kernel Specification Hierarchy](#5-three-tier-kernel-specification-hierarchy-adr-013) | Interface → Algorithm → Implementation |
| 6 | [Module Structure](#6-module-structure-adr-012) | Directory layout, services dissolution |
| 7 | [Backend Implementations](#7-backend-implementations) | OpenCL, Vulkan, CPU specifics |
| 8 | [Build System](#8-build-system-adr-014) | Meson structure, options, per-backend targets |
| 9 | [Test Strategy](#9-test-strategy-adr-016) | Three-tier framework, fixtures, tolerances, oracle model |
| 10 | [User-Facing API](#10-user-facing-api-adr-018) | WorkTicket lifecycle, design decisions |
| 11 | [Migration Path](#11-migration-path-adr-017) | Phase sequencing, dependency graph, rollback |
| 12 | [ADR Index](#12-adr-index) | Complete decision record reference |

---

## 1. Document Role

This document is the **design-level** authority in the three-tier architectural hierarchy (CONCEPT.md §6):

| Layer | Document | Authority |
| :--- | :--- | :--- |
| Conceptual | CONCEPT.md | **What** and **why** — principles, components, validation scenarios |
| Contractual | CONTRACT.md, `KernelContract` dataclasses | **Interface requirements** — parameter grammars, padding contracts, placement strategies |
| Design | **This document** | **How** — concrete data structures, module layout, build system, interop, test infrastructure, migration plan |

Every structural decision recorded here traces to an accepted ADR. The ADR chain (ADR-001 through ADR-018) is the authoritative record of each decision's rationale and trade-off analysis; this document synthesizes their outcomes into a coherent implementation blueprint.

---

## 2. System Architecture Overview

The system is a multi-backend execution engine for a single-hidden-layer averaging ensembled classifier. It is organized around a **plan-as-data-structure** model: a shared orchestration layer (the Policy tier) produces an immutable, backend-neutral execution plan, which each backend renders using its native execution model.

### 2.1 Three-Tier Jurisdictional Model (ADR-001)

| Tier | Responsibility | Invariant |
| :--- | :--- | :--- |
| **Policy** (Shared Layer) | Plan construction, contract validation, reduction tree planning, buffer lifecycle annotation, threshold scheduling, tiling | Identical across all backends and all node types |
| **Orchestration** (Per-Backend Rendering) | Plan traversal, native dispatch, kernel tier selection, buffer allocation, synchronization primitive management | Adapts the plan to backend-specific capabilities |
| **Execution** (Kernel Internals) | The compiled algorithm inside a kernel | Governed by `KernelContract` behavioral invariants; never prescribed by the plan |

Tiers may collapse: Node 16's internal multi-stage reduction is an Execution-tier concern because the upstream Item Synchronization Point (Node 13) guarantees contiguous input, eliminating Orchestration-tier inter-stage logistics (ADR-005).

### 2.2 Backend Abstraction Boundary (ADR-001)

The architecture splits into two layers with a clean data-structure boundary:

- **Shared Orchestration Layer** (`src/shared/`): Pure Python. Produces backend-neutral frozen dataclass plans. Contains no backend-specific types or imports.
- **Backend Execution Layer** (`src/backends/<name>/`): Per-backend rendering code that consumes plans and dispatches using native APIs.

Each backend exposes a `PlanRenderer` interface — the sole contract between the shared layer and any backend.

---

## 3. Execution Plan Data Structures

### 3.1 Plan Node Taxonomy (ADR-002)

The plan is a directed acyclic graph of typed, immutable node descriptors with explicit dependency edges. The node vocabulary is a **closed taxonomy** of five types:

| Node Type | Semantics | Key Fields |
| :--- | :--- | :--- |
| `KernelDispatchNode` | A single logical kernel invocation | `kernel_name`, `buffer_bindings`, `scalar_params`, `tile_count`, `local_work_size`, `contract`, `placement_strategy` |
| `ReductionTreeNode` | A multi-stage $\log_K(N)$ reduction tree rendered atomically by the backend | `reduction_plan: ReductionTreePlan` |
| `StreamingLoopNode` | A parametric loop over a chunk-indexed body of `KernelDispatchNode` references | `streaming_plan: StreamingLoopPlan` |
| `BarrierNode` | A named synchronization point joining upstream edges; no dispatch payload | `barrier_name` |
| `RetrievalNode` | A host-accessible result extraction point | `source_buffer`, `logical_shape`, `event_name` |

No additional node types may be introduced without a formal ADR. Dependency edges are the concurrency specification — nodes with no dependency relationship may execute in parallel at the backend's discretion.

### 3.2 ReductionTreePlan (ADR-003)

A parametric header consumed atomically by the backend's renderer. The Policy tier pre-computes the full threshold schedule; the backend selects kernel tiers (identity, register-reduce, local-reduce) per stage.

```python
@dataclass(frozen=True)
class ReductionTreePlan:
    num_partials: int
    fan_in_K: int
    num_stages: int
    elements_per_partial: int
    initial_offset_list: tuple[int, ...]
    tree_variant: Literal["sum", "sum_and_clip"]
    threshold_schedule: tuple[float | None, ...]   # len == num_stages; None for "sum" variant stages
    partial_width: int
    source_buffer: BufferHandle
    destination_buffer: BufferHandle
```

The threshold schedule follows the Quadratic Scaling Policy: $T_j = T_{\text{algorithmic}} + \lambda \cdot j^2$, clamped by $T_{\text{safety},\,j} = \text{FP\_FORMAT\_MAX} / K_j$.

### 3.3 StreamingLoopPlan (ADR-004)

A stride-based parametric specification. The body is a flat sequence of `KernelDispatchNode` references. Complexity ceiling: no nested loops.

```python
@dataclass(frozen=True)
class StreamingLoopPlan:
    iteration: IterationDimension                  # total_extent, chunk_count, chunk_size
    body: tuple[str, ...]                          # node IDs in the plan
    parameter_strides: tuple[ParameterStride, ...]
    scratch_buffers: tuple[ScratchBufferSpec, ...]
    constant_scalars: dict[str, float]             # scalars invariant across iterations
```

The backend instantiates per-chunk parameter values from `base + index × stride`.

### 3.4 HardwareProfile (ADR-006)

A frozen dataclass with fields named for their plan-construction role, not their hardware origin:

```python
@dataclass(frozen=True)
class HardwareProfile:
    simd_width: int
    cache_line_bytes: int
    max_reduce_fan_in: int
    max_local_mem_bytes: int | None                # None if backend has no local memory concept
    global_mem_bytes: int
```

Each backend computes `max_reduce_fan_in` from its native constraints. The Policy tier uses this value to determine the reduction batch size $K$.

### 3.5 PrecisionConfig (ADR-008)

Replaces the former `PrecisionContext` ABC hierarchy with a single frozen dataclass:

```python
@dataclass(frozen=True)
class PrecisionConfig:
    numpy_dtype: np.dtype
    fp_format_max: float
    epsilon: float
```

Constructed via `PrecisionConfig.float32()` and `PrecisionConfig.float16()` classmethods. `ModelSpec` consumes `PrecisionConfig` via composition; backward-compatible properties (`SCALAR_NP_TYPE`, `SCALAR_C_TYPE_NAME`) delegate to `self.precision`. Factory classmethods `ModelSpec.float32()` / `ModelSpec.float16()` and deprecated module-level `Float32ModelSpec()` / `Float16ModelSpec()` functions are available until Phase 6.

### 3.6 Buffer Lifecycle (ADR-009)

Every device-side buffer is described by a `BufferDescriptor`:

```python
@dataclass(frozen=True)
class BufferDescriptor:
    handle: BufferHandle                           # opaque int id
    logical_name: str                              # human-readable name
    padded_shape: tuple[int, ...]                  # SIMD/cache-padded dimensions
    element_size_bytes: int                        # bytes per element
    size_bytes: int                                # total allocation size
    role: BufferRole                               # MODEL_STATE | BATCH_INPUT | BATCH_INTERMEDIATE | BATCH_OUTPUT
    producing_node: str | None                     # None for MODEL_STATE / BATCH_INPUT
    consumers: frozenset[str]
    last_consumer: str | None
```

Buffer lifetimes are plan-prescribed. Each plan's buffer namespace is fully self-contained — no buffer persists across plan boundaries.

### 3.7 RetrievalFuture Protocol (ADR-010)

The sole host-facing type for observing device results:

```python
@runtime_checkable
class RetrievalFuture(Protocol):
    @property
    def node_id(self) -> str: ...              # originating RetrievalNode
    def wait(self) -> None: ...                # block until host-accessible
    def result(self) -> NDArray[np.floating]: ...  # unpadded numpy array
    def release(self) -> None: ...             # free device resources
```

The renderer owns padding-stripping. CPU backend degenerates to zero-copy.

---

## 4. Kernel Contract Model (ADR-007, ADR-011)

### 4.1 KernelContract / KernelBinding Split

| Artifact | Layer | Content |
| :--- | :--- | :--- |
| `KernelContract` | Shared (`src/shared/kernel_contracts/`) | Frozen dataclass: `BufferParamSpec`, `ScalarParamSpec`, `LocalMemorySpec`, `PlacementContract`. Used for pre-dispatch validation at plan-construction time. |
| `KernelBinding` | Per-backend (`src/backends/<name>/kernel_bindings/`) | Backend-specific dispatch adapter translating abstract placement keys and buffer references into native dispatch arguments. |

Abstract placement keys (e.g., `grid_mod_cls`, `linear_batch`) replace backend-specific tile index mechanisms. Tile index delivery — host-provided scalar (OpenCL), `gl_WorkGroupID.x` (Vulkan), `task_index` parameter (CPU) — is a binding concern.

### 4.2 CCE/BCE Strategy Delegation (ADR-011)

Mixed strategy:

| Strategy | Nodes | Mechanism | Rationale |
| :--- | :--- | :--- | :--- |
| **B** (separate kernels) | 6, 7 | Distinct `kernel_name` per variant | Structural divergence — genuinely different interfaces and algorithms |
| **A** (host-injected flag) | 8, 9, 10 | `src_scalar_FLAG_problem_type` in `scalar_params` | Parametric divergence — same interface, toggled internal branch |

Both strategies are expressed through existing ADR-002/ADR-007 primitives with no plan vocabulary extensions.

---

## 5. Three-Tier Kernel Specification Hierarchy (ADR-013)

| Tier | Artifact | Scope |
| :--- | :--- | :--- |
| **Interface** | `KernelContract` frozen dataclasses in `src/shared/kernel_contracts/` | *What* — parameter names, types, shapes, padding, placement, calculability proofs |
| **Algorithm** | `kernels.cl.h` in the architecture-root `kernels/` directory | *How conceptually* — computational strategy, mathematical formula, reduction structure, behavioral invariants |
| **Implementation** | Per-backend source files in `src/backends/<name>/kernel_sources/` | *How concretely* — language-native code using the backend's execution model |

`kernels.cl.h` serves dual roles: algorithmic specification (language-neutral content in comments and `@kernel_contract` blocks) and OpenCL compile infrastructure (`#ifdef` guarded). The existing `*.cl.c` files in `kernels/` are the canonical OpenCL reference implementations, consumed directly by the OpenCL backend.

Cross-backend algorithmic fidelity is ensured through: specification review against `kernels.cl.h`, `KernelContract` interface validation at plan-construction time, and cross-backend parity tests (§9, Tier 3).

---

## 6. Module Structure (ADR-012)

### 6.1 Physical Directory Layout

```
averaging_ensembled_classifier/
├── CONCEPT.md                               # Conceptual authority
├── CONTRACT.md                              # Contractual authority
├── DESIGN.md                                # This document
├── meson.build                              # Top-level build entry point
├── meson.options                            # Backend feature options
├── pyproject.toml                           # meson-python build backend
│
├── kernels/                                 # Specification + OpenCL reference impl (ADR-013)
│   ├── kernels.cl.h                         # Algorithmic specification
│   ├── phase_1_act.cl.c
│   ├── phase_2_learn_A_production.cl.c
│   ├── phase_2_learn_B_processing.cl.c
│   ├── phase_2_learn_C_reduction.cl.c
│   ├── phase_2_learn_D_backprop.cl.c
│   └── phase_3_update.cl.c
│
├── src/
│   ├── shared/                              # Policy tier — pure Python, no backend types
│   │   ├── __init__.py
│   │   ├── plan_types.py                    # Five node types (ADR-002)
│   │   ├── plan_builder.py                  # Plan construction logic
│   │   ├── buffer_lifecycle.py              # BufferHandle, BufferRole, BufferDescriptor (ADR-009)
│   │   ├── reduction_tree_plan.py           # ReductionTreePlan (ADR-003)
│   │   ├── streaming_loop_plan.py           # StreamingLoopPlan (ADR-004)
│   │   ├── retrieval_future.py              # RetrievalFuture Protocol (ADR-010)
│   │   ├── hardware_profile.py              # HardwareProfile (ADR-006)
│   │   ├── precision_config.py              # PrecisionConfig (ADR-008)
│   │   ├── model_spec.py                    # Model configuration
│   │   ├── parameter_space.py               # Learnable parameter management
│   │   ├── memory_layout.py                 # SIMD-aware layout and padding
│   │   ├── stabilization_policy.py          # Threshold scheduling
│   │   ├── workload_primitives.py           # Tiling and chunk decomposition
│   │   ├── problem_type_strategy.py         # CCE/BCE strategy delegation (ADR-011)
│   │   ├── work_ticket.py                   # WorkTicket + LearnHandle (ADR-018)
│   │   ├── engine.py                        # Engine: user-facing entry point (ADR-018)
│   │   └── kernel_contracts/                # KernelContract frozen dataclasses (ADR-007)
│   │       ├── __init__.py
│   │       ├── phase_1_act.py
│   │       ├── phase_2_learn_A_production.py
│   │       ├── phase_2_learn_B_processing.py
│   │       ├── phase_2_learn_C_reduction.py
│   │       ├── phase_2_learn_D_backprop.py
│   │       └── phase_3_update.py
│   │
│   ├── backends/
│   │   ├── opencl/                          # PyOpenCL backend — runtime kernel compilation
│   │   │   ├── renderer.py                  # PlanRenderer implementation
│   │   │   ├── retrieval.py                 # OpenCLRetrievalFuture
│   │   │   ├── buffer_allocator.py
│   │   │   ├── context.py
│   │   │   ├── discovery.py                 # HardwareProfile population
│   │   │   ├── type_mapping.py
│   │   │   └── kernel_bindings/             # KernelBinding dispatch code
│   │   │
│   │   ├── vulkan/                          # vulkan-python backend — SPIR-V at build time
│   │   │   ├── renderer.py
│   │   │   ├── retrieval.py
│   │   │   ├── buffer_allocator.py
│   │   │   ├── context.py
│   │   │   ├── discovery.py
│   │   │   ├── type_mapping.py
│   │   │   ├── kernel_bindings/
│   │   │   └── kernel_sources/              # GLSL compute shaders (*.comp)
│   │   │       └── common.glsl              # Shared specialization constant declarations
│   │   │
│   │   └── cpu/                             # ctypes FFI backend — compiled C shared library
│   │       ├── __init__.py                  # Public exports: CPUPlanRenderer, discover_hardware, etc.
│   │       ├── renderer.py                  # CPUPlanRenderer with pool_dispatch_and_wait
│   │       ├── retrieval.py                 # Zero-copy CPURetrievalFuture
│   │       ├── buffer_allocator.py          # SIMD-aligned numpy arrays
│   │       ├── discovery.py                 # Thread count, cache line size, SIMD width → HardwareProfile
│   │       ├── type_mapping.py              # PrecisionConfig → numpy dtype mapping
│   │       ├── _ffi_types.py                # ctypes struct definitions mirroring cpu_kernels.h (ADR-015)
│   │       ├── _loader.py                   # Library loading via importlib.resources + _verify_layouts()
│   │       ├── _dispatch_table.py           # kernel_name → (task_fn_ptr, args_struct_class) map
│   │       └── kernel_sources/              # C implementations (compiled to libcpu_kernels.so)
│   │           ├── cpu_simd.h               # SIMD abstraction (AVX-512/AVX2/SSE2/NEON/scalar)
│   │           ├── cpu_threads.h            # Thread pool interface
│   │           ├── cpu_threads.c            # Thread pool implementation (C11/pthreads)
│   │           ├── cpu_kernels.h            # Public ABI: task prototypes, argument structs, get_struct_size_*
│   │           ├── cpu_export.h             # Symbol visibility macros
│   │           ├── phase_1_act.c            # forward_pass, render_logits, cce/bce loss
│   │           ├── phase_2_learn_A_production.c  # module grads, backprop to hidden, temp grads
│   │           ├── phase_2_learn_B_processing.c  # clip partials, gather_and_permute
│   │           ├── phase_2_learn_C_reduction.c   # aggregate, clip intermediate, reduction tree, stabilize
│   │           ├── phase_2_learn_D_backprop.c    # shared weight/bias backprop, clip shared grads
│   │           └── phase_3_update.c              # normalize, adam_update, clamp_temperatures
│   │
│   ├── _build_config.py                     # Generated: BACKEND_OPENCL, BACKEND_VULKAN, BACKEND_CPU booleans
│   └── main_orchestrator.py                 # Training orchestration (migrating to Engine)
│
├── tests/                                   # Tiered test framework (ADR-016)
│   ├── conftest.py                          # _build_config-driven skip logic
│   ├── tolerance_config.py                  # Per-kernel tolerance tables (FP32 + backend-specific)
│   ├── tier1/                               # Host-side plan correctness (140 tests)
│   ├── tier2/                               # Per-backend kernel correctness
│   │   ├── fixtures/                        # Analytical + numpy reference implementations (shared)
│   │   │   ├── analytical.py                # Closed-form reference: clamp, normalize, clip, mask
│   │   │   ├── data_generators.py           # Deterministic RNG-seeded test data factories
│   │   │   ├── numpy_forward.py             # forward_pass, render_logits, softmax/sigmoid loss
│   │   │   ├── numpy_gradients.py           # module grads, hidden grads, temp grads
│   │   │   ├── numpy_reduction.py           # multi-stage reduction with clip
│   │   │   ├── numpy_backprop.py            # shared weight/bias backprop
│   │   │   └── numpy_update.py              # adam_update reference
│   │   ├── opencl/                          # OpenCL Tier 2 tests
│   │   └── cpu/                             # CPU Tier 2 tests (16 test modules)
│   │       ├── conftest.py                  # Session-scoped CPU fixtures
│   │       └── test_cpu_*.py                # Per-kernel correctness tests
│   └── tier3/                               # Cross-backend parity (CPU oracle)
│
└── adr/                                     # Architectural Decision Records (ADR-001 through ADR-018)
```

### 6.2 Services Dissolution (ADR-012)

The following legacy service modules are dissolved into stateless plan primitives and backend-internal modules:

| Dissolved Module | Replacement |
| :--- | :--- |
| `cl_context_manager.py` | `backends/opencl/context.py` (backend internal) |
| `compute_patterns.py` | Plan builder + per-backend renderer logic |
| `launcher_infra.py` | `kernel_bindings/` per backend |
| `arch_primitives.py` | `shared/hardware_profile.py`, `shared/precision_config.py`, `shared/problem_type_strategy.py` |
| `kernel_signatures/` | `shared/kernel_contracts/` + per-backend `kernel_bindings/` |
| `graph_recipes.py` | `shared/plan_builder.py` |
| `batch_processor.py` | Per-backend `renderer.py` |
| `execution_plan.py` | `shared/plan_types.py` + `shared/plan_builder.py` |

---

## 7. Backend Implementations

### 7.1 OpenCL Backend

- **Interop:** PyOpenCL (established core dependency).
- **Kernel compilation:** Runtime via `clBuildProgram`. Sources loaded from architecture-root `kernels/` directory, discovered via `importlib.resources`.
- **Dispatch model:** Per-tile imperative `clEnqueueNDRange` with `cl.Event` synchronization.
- **Tile index:** Host-provided `flat_tile_index` scalar per dispatch.
- **Local memory:** Runtime-sized via `cl.LocalMemory()`.

### 7.2 Vulkan Backend

- **Interop:** `vulkan-python` (optional dependency: `vulkan = ["vulkan-python>=0.2.0"]`).
- **Kernel compilation:** GLSL compute shaders compiled to SPIR-V at build time via `glslc --target-env=vulkan1.1`.
- **Dispatch model:** Single-dispatch parallelism — `vkCmdDispatch(N, 1, 1)` for N-tile kernels. Command buffers with `vkCmdPipelineBarrier` for sequencing.
- **Tile index:** Implicit via `gl_WorkGroupID.x`.
- **Local memory:** `shared` qualifier, compile-time sized via specialization constants.
- **Build-time constants:** Vulkan specialization constants replace `-D` preprocessor flags.

### 7.3 CPU Backend (ADR-015)

- **Interop:** `ctypes` (Python stdlib). Zero additional dependencies (`cpu = []`).
- **Kernel compilation:** C implementations compiled to `libcpu_kernels.so` by Meson's `shared_library()`. ISA selection via `-march=native` (overridable via `aec_cpu_isa_flags` build option). SIMD through `cpu_simd.h` abstraction layer supporting AVX-512, AVX2, SSE2, ARM NEON, and scalar fallback.
- **Dispatch model:** `pool_dispatch_and_wait(pool, task_fn, args, task_count)` — blocking dispatch of N independent tasks to a persistent thread pool. Atomic task claiming for natural load balancing.
- **ABI surface:** 18 `task_<kernel_name>` functions + `execute_reduction_tree` + `pool_create`/`pool_destroy`/`pool_dispatch_and_wait` + `get_simd_width` + 18 `get_struct_size_*` verification exports. Uniform task function signature: `void task_<kernel_name>(void* args, uint task_index, uint thread_id)`.
- **FFI types:** `src/backends/cpu/_ffi_types.py` — ctypes `Structure` subclasses mirroring `cpu_kernels.h` structs. `_loader.py` handles library discovery and `_verify_layouts()` at load time. `_dispatch_table.py` builds the `kernel_name → (task_fn_ptr, args_struct_class)` map.
- **Layout verification:** `_verify_layouts()` at library load time asserts Python-side struct sizes match C-side `get_struct_size_*()` exports. Catches struct drift before any dispatch. Same-size field reorderings caught by Tier 2 behavioral tests.
- **Library discovery:** `importlib.resources.files('averaging_ensembled_classifier.backends.cpu')` with platform-specific filename resolution.
- **Status:** ✅ **Implemented** (Phase 3A + 3B + 3C complete). 6 C source files + 4 headers. 9 Python FFI modules. 16 Tier 2 test modules.

---

## 8. Build System (ADR-014)

### 8.1 Structure

Top-level `meson.build` with `subdir()` delegation to per-backend `meson.build` files:

```meson
project('averaging_ensembled_classifier-sub', 'c', version: '0.1.0')
py = import('python').find_installation()

backend_vulkan = get_option('aec_backend_vulkan')
backend_cpu    = get_option('aec_backend_cpu')
glslc = find_program('glslc', required: backend_vulkan)

subdir('src/shared')                          # Pure Python install
subdir('src/backends/opencl')                 # Always enabled — runtime compilation only
subdir('kernels')                             # Install OpenCL kernel sources as package data

if backend_vulkan.allowed() and glslc.found()
  subdir('src/backends/vulkan')               # SPIR-V compilation
endif

if backend_cpu.allowed()
  subdir('src/backends/cpu')                  # C shared library
endif
```

### 8.2 Build Options

```meson
# meson.options
option('aec_backend_vulkan', type: 'feature', value: 'auto',
       description: 'Build Vulkan SPIR-V compute shaders (requires glslc)')
option('aec_backend_cpu', type: 'feature', value: 'auto',
       description: 'Build CPU SIMD kernel shared library')
option('aec_cpu_isa_flags', type: 'array', value: [],
       description: 'C compiler ISA flags (e.g., [\'\-mavx2\']. Empty = -march=native)')
```

Option names use the `aec_` prefix to namespace them within the Meson subproject.

**Feature flag semantics:** `auto` (default) probes for dependencies and enables if available; `enabled` makes the backend mandatory; `disabled` skips unconditionally. OpenCL is always enabled (no build-time compilation).

### 8.3 Per-Backend Build Targets

| Backend | Source | Build Action | Output | Installation |
| :--- | :--- | :--- | :--- | :--- |
| OpenCL | `kernels/*.cl.c`, `kernels/kernels.cl.h` | None | None | Package data via `importlib.resources` |
| Vulkan | `src/backends/vulkan/kernel_sources/*.comp` | `glslc` → SPIR-V | `*.spv` modules | Package data alongside backend Python |
| CPU | `src/backends/cpu/kernel_sources/*.c` | C compiler + ISA flags | `libcpu_kernels.so` | Alongside backend Python package |

### 8.4 Generated Build Configuration

`_build_config.py` declares `BACKEND_OPENCL`, `BACKEND_VULKAN`, `BACKEND_CPU` as booleans — the single authoritative source for runtime capability queries and test skip logic.

---

## 9. Test Strategy (ADR-016)

### 9.1 Three-Tier Framework

| Tier | Scope | Execution Requirement | Gate |
| :--- | :--- | :--- | :--- |
| **Tier 1** | Host-side plan correctness: plan construction, contract validation, buffer lifecycle, reduction tree plan, streaming loop plan, strategy delegation, memory layout, precision config, hardware profile | None — pure Python | Always runs (140 tests) |
| **Tier 2** | Per-backend kernel correctness against reference fixtures | Per-backend: `_build_config.BACKEND_<NAME> is True` | Per enabled backend (CPU: 16 test modules) |
| **Tier 3** | Cross-backend parity with CPU as reference oracle | CPU + ≥1 other backend | Falls back to GPU-vs-GPU if CPU unavailable |

### 9.2 Tier 2 Fixtures

- **Analytical fixtures** for closed-form kernels: `compute_hidden_mask`, `clamp_temperatures`, `normalize_gradients`, clipping kernels. The fixture *is* the mathematical definition. Located in `tests/tier2/fixtures/analytical.py`.
- **Numpy reference implementations** for complex kernels: `forward_pass`, `compute_probs_loss_cce/bce`, `calculate_module_param_grads`, `backprop_error_to_hidden`, `stabilize_reduce_grad_h`, `adam_update`, etc. Numpy code performs the mathematical operation without tiling, SIMD, or local memory. Organized across `tests/tier2/fixtures/numpy_forward.py`, `numpy_gradients.py`, `numpy_reduction.py`, `numpy_backprop.py`, and `numpy_update.py`.
- **Data generators** in `tests/tier2/fixtures/data_generators.py` — deterministic RNG-seeded helper functions for creating test inputs, weights, biases, and targets.

### 9.3 Tolerance Configuration

Base tolerances: FP32 `atol=1e-5, rtol=1e-5`; FP16 `atol=1e-2, rtol=1e-2`. Kernels with higher numerical sensitivity (Softmax, sigmoid near saturation, multi-stage reduction, Adam division) override to `atol=1e-4, rtol=1e-4` for FP32.

### 9.4 Oracle Model

The CPU backend is the Tier 3 reference oracle — deterministic, single-threaded-per-task, maximally inspectable. CPU correctness is established independently by Tier 2 against analytical/numpy fixtures, breaking the circular-dependency concern.

---

## 10. User-Facing API (ADR-018)

### 10.1 WorkTicket Lifecycle

```python
ticket = engine.submit(x_data)                 # → WorkTicket (PENDING); Act plan dispatched immediately
probs = ticket.get_prediction()                # blocks until Act completes; transitions to ACT_COMPLETE
learn_handle = ticket.resolve(y_data)          # → LearnHandle (RESOLVED); Learn plan dispatched immediately
learn_handle.wait()                            # blocks until Learn completes; transitions to CONSUMED
```

Convenience method for sequential batch training:

```python
result = engine.train_batch(X_train, y_train)  # combined Act+Learn; returns predictions
```

### 10.2 Design Decisions

| Choice | Decision | ADR-018 Reference |
| :--- | :--- | :--- |
| Unit of intent | Stateful ticket (`WorkTicket`) | Choice 1A |
| Concurrency | Future-based, non-blocking, sync-compatible; no `asyncio` dependency | Choice 2B |
| Batch composition | Explicit batch submission; one `submit()` = one ticket = one plan | Choice 3A |
| Cross-plan buffers | Recompute — Learn plan recomputes forward pass; no device buffers persist across plans | Choice 4A |
| Data persistence | Ephemeral — each submission is one-shot; user owns the training loop | Choice 5A |
| Migration positioning | Parallel workstream, Phase 4-adjacent | Choice 7C |

### 10.3 Internal Mechanism

1. **`engine.submit(x_data)`** — Constructs and renders an Act-only plan. Stores the `RetrievalFuture` and host-side input data reference in the ticket. Returns immediately.
2. **`ticket.get_prediction()`** — Calls `future.result()` (unpadded numpy), then `future.release()` (frees device buffers). Caches prediction. Subsequent calls return cached value.
3. **`ticket.resolve(y_data)`** — Constructs and renders a Learn-only plan using stored `x_data` and provided `y_data`. Learn plan recomputes `hidden_activations` from `x_data`. Returns `LearnHandle` wrapping the Learn plan's `RetrievalFuture`.
4. **`learn_handle.wait()`** — Blocks on the Learn plan's `RetrievalFuture.wait()`, then calls `.release()`.

---

## 11. Migration Path (ADR-017)

### 11.1 Phase Sequencing

All phases proceed in parallel behind `_build_config.py` feature flags. Each phase has a rollback gate defined by test tier outcomes.

### 11.2 Phase Structure

| Phase | Objective | Rollback Gate | Status |
| :--- | :--- | :--- | :--- |
| **0: Foundation** | Create directory structure; move modules to `src/shared/` + `src/backends/opencl/`; create `meson.options` and `_build_config.py` template | All existing tests green | **Complete** |
| **1: Plan Model** | Implement shared-layer plan data structures (node types, buffer lifecycle, reduction/streaming plans, retrieval protocol); write Tier 1 tests | Tier 1 green | **Complete** |
| **2: OpenCL Adapter** | Wrap existing PyOpenCL dispatch in `PlanRenderer` interface; write OpenCL Tier 2 tests | Tier 1 + OpenCL Tier 2 green | Not started |
| **3: CPU Backend** | Implement C kernel library, ctypes FFI, `CPUPlanRenderer`; write CPU Tier 2 tests | Tier 1 + CPU Tier 2 green | **Complete** |
| **4: Test Harness** | Full Tier 1/2/3 framework, fixtures, tolerance tables, oracle logic; can begin immediately | All enabled tiers green | Not started |
| **5: Vulkan Backend** | GLSL shaders, SPIR-V compilation, vulkan-python `PlanRenderer`; write Vulkan Tier 2 tests | Tier 1 + Vulkan Tier 2 + Tier 3 parity green | Not started |
| **User-Facing API** | `WorkTicket`, `LearnHandle`, `Engine`; parallel with Phase 4 | Tier 1 (ticket) + integration green | Not started |
| **6: Legacy Removal** | Delete dissolved modules; system operates exclusively through plan-model dispatch | Tier 3 parity green, all backends, FP32 + FP16 | Not started |

### 11.3 Phase Dependency Graph

```
Phase 0 (Foundation) ────────────── ✅ Complete (192 tests green)
  │
  ▼
Phase 1 (Plan Model) ──────────── ✅ Complete (332 tests green: 192 Phase 0 + 140 Tier 1)
  │
  ├──▶ Phase 2 (OpenCL Adapter) ─── Tier 1 + OpenCL Tier 2 gate
  ├──▶ Phase 3 (CPU Backend) ────── ✅ Complete (Tier 1 green + CPU Tier 2 green)
  ├──▶ Phase 5 (Vulkan Backend) ─── Tier 1 + Vulkan Tier 2 + Tier 3 gate
  ├──▶ User-Facing API ─────────── Tier 1 (ticket) + integration green
  │
  │    Phase 4 (Test Harness) ────── All enabled tiers green
  │    [can start in parallel with Phase 0]
  │
  ▼
Phase 6 (Legacy Removal) ───── Tier 3 parity green, all backends, FP32 + FP16
  [requires Phases 2, 3, 4, 5 complete]
```

Phases 2, 3, and 5 are independent workstreams. Phase 4 has no hard dependency. Phase 6 is a join point requiring all preceding phases.

### 11.4 Feature-Flag Lifecycle

| Stage | Flag Value | Meaning |
| :--- | :--- | :--- |
| Development | `auto` | Backend available when toolchain detected; CI skips if absent |
| Validated | `enabled` | Tier gate passed; CI requires the backend |
| Mandatory | `enabled` (enforced) | Phase 6 complete; backend is a required component |

### 11.5 Rollback Protocol

If a phase's tier gate regresses: revert the feature flag to `auto`, diagnose using tier test output, fix-forward or revert, re-promote once the gate is green.

---

## 12. ADR Index

| ADR | Title | Key Decision |
| :--- | :--- | :--- |
| [001](adr/ADR-001-backend-abstraction-boundary.md) | Backend Abstraction Boundary | Two-layer split; three-tier jurisdictional model; abstract at DAG/Phase level |
| [002](adr/ADR-002-plan-node-types-and-synchronization-structure.md) | Plan Node Types | Five closed node types; typed DAG with explicit dependency edges |
| [003](adr/ADR-003-reduction-tree-plan-representation.md) | Reduction Tree Plan | Parametric header with pre-computed threshold schedule |
| [004](adr/ADR-004-streaming-loop-plan-representation.md) | Streaming Loop Plan | Stride-based parametric specification; no nested loops |
| [005](adr/ADR-005-node-16-opacity-in-the-plan.md) | Node 16 Opacity | Single `KernelDispatchNode` with policy params; internal reduction is Execution-tier |
| [006](adr/ADR-006-hardware-profile.md) | Hardware Profile | Frozen dataclass with policy-input naming |
| [007](adr/ADR-007-kernel-signature-contract-binding-split.md) | Kernel Contract/Binding Split | `KernelContract` (shared validation) + `KernelBinding` (per-backend dispatch); abstract placement keys |
| [008](adr/ADR-008-precision-configuration.md) | Precision Configuration | `PrecisionConfig` frozen dataclass replacing ABC hierarchy |
| [009](adr/ADR-009-buffer-lifecycle-in-the-plan-model.md) | Buffer Lifecycle | Plan-prescribed lifetime intervals; `BufferDescriptor` with producer/consumer annotations |
| [010](adr/ADR-010-d2h-transfer-and-phase-sync-points.md) | D2H Transfer & Phase Sync | `RetrievalFuture` Protocol; padding-aware unpadded numpy result |
| [011](adr/ADR-011-cce-bce-strategy-delegation.md) | CCE/BCE Strategy Delegation | Mixed: Strategy B for Nodes 6/7; Strategy A for Nodes 8/9/10 |
| [012](adr/ADR-012-module-factoring-and-services-dissolution.md) | Module Factoring | `src/shared/` + `src/backends/<name>/`; services dissolution |
| [013](adr/ADR-013-kernel-source-strategy.md) | Kernel Source Strategy | `kernels.cl.h` as algorithmic reference; per-backend `kernel_sources/`; three-tier specification hierarchy |
| [014](adr/ADR-014-build-system-integration.md) | Build System Integration | Meson with `subdir()` delegation; `feature` options; `importlib.resources` discovery |
| [015](adr/ADR-015-python-native-backend-interop.md) | Python ↔ Native Interop | Per-backend FFI autonomy; CPU uses ctypes (stdlib) |
| [016](adr/ADR-016-test-strategy.md) | Test Strategy | Three-tier pytest framework; CPU oracle; analytical + numpy fixtures |
| [017](adr/ADR-017-incremental-migration-path.md) | Incremental Migration | Feature-flag gated phases; tier-based rollback gates |
| [018](adr/ADR-018-user-facing-api.md) | User-Facing API | `WorkTicket` lifecycle; future-based concurrency; explicit batch; recompute; ephemeral |
