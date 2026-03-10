# ADR-012: Module Factoring & Services Dissolution

**Status:** ACCEPTED  
**Date:** 2026-03-10  
**Deciders:** —  
**Supersedes:** —  
**Blocked by:** ADR-009  
**Blocks:** ADR-013, ADR-014, ADR-015

---

## Context

The current `src/` directory is a flat namespace where shared-layer logic, backend-specific dispatch, and infrastructure services co-exist. Fourteen Python modules sit at the same level, with no physical separation reflecting the jurisdictional boundaries established by ADR-001:

```
src/
├── __init__.py
├── arch_primitives.py          ← PrecisionContext ABC (dissolved by ADR-008)
├── batch_processor.py          ← mixes plan construction + cl.Event dispatch
├── cl_context_manager.py       ← DiscoveredArchConstants + OpenCL context (dissolved by ADR-006, ADR-008)
├── compute_patterns.py         ← ReductionPlan + AggregationManager (OpenCL-coupled)
├── execution_plan.py           ← DependencyProvider hierarchy (embeds cl.Event)
├── graph_recipes.py            ← 984 lines mixing DAG logic + OpenCL dispatch
├── kernel_signatures/          ← KernelSignature hierarchy (dissolved by ADR-007)
├── launcher_infra.py           ← Services, BufferManager, HostView, BufferHandle (OpenCL-coupled)
├── main_orchestrator.py
├── memory_layout.py            ← shared (backend-neutral)
├── model_spec.py               ← shared (backend-neutral after ADR-008)
├── parameter_space.py          ← shared (backend-neutral)
├── stabilization_policy.py     ← shared (backend-neutral)
└── workload_primitives.py      ← shared (backend-neutral)
```

The multi-backend refactoring (ADR-001 through ADR-011) requires a physical directory split that mirrors the two-layer jurisdictional model: a **shared layer** (Policy tier) containing backend-neutral types and plan construction, and a **backend layer** (Orchestration + Execution tiers) containing per-backend rendering, dispatch, and device management.

### The Services layer dissolution

Three modules constitute an informal "Services layer" that bundles backend-specific infrastructure:

**`cl_context_manager.py`** provides `OpenCLContextManager`, `DiscoveredArchConstants` (and its precision subclasses `Float32DiscoveredArchConstants`, `Float16DiscoveredArchConstants`), `CLBundle`, and `ComputeEnvironment`. ADR-006 (ACCEPTED) extracts the shared hardware constants into `HardwareProfile`, a standalone frozen dataclass. ADR-008 (ACCEPTED) replaces the `PrecisionContext` inheritance hierarchy with `PrecisionConfig`, a standalone frozen dataclass. What remains — `cl.Context` creation, `cl.CommandQueue` management, `cl.Program` compilation, and OpenCL device queries — is inherently backend-specific. Each backend must own its execution context lifecycle.

**`compute_patterns.py`** provides `ReductionPlan` (a frozen dataclass of stage counts and fan-ins) and `AggregationManager` (an OpenCL-coupled class that drives `clEnqueueNDRange` for reduction stages using `KernelExecutor` and `BufferManager`). ADR-003 (ACCEPTED) replaces `ReductionPlan` with `ReductionTreePlan`, a shared-layer frozen dataclass carrying pre-computed thresholds and offset lists. The `AggregationManager`'s dispatch logic is absorbed by each backend's `PlanRenderer`. The remaining tiling and padding helper functions (pure math) belong in the shared layer.

**`launcher_infra.py`** provides four types:

- **`Services`** — a frozen dataclass aggregating `cl.CommandQueue`, `BufferManager`, and `KernelExecutor`. This is a convenience bundle of OpenCL-specific infrastructure. It dissolves entirely — each backend owns its own service aggregation.
- **`BufferHandle`** — a backend-neutral opaque token. ADR-009 (ACCEPTED) extracts it into the shared layer as a frozen dataclass with an integer `id` field.
- **`BufferManager`** — coupled to `cl.Context` and `cl.Buffer`. It manages named buffer allocation (`create_named_buffer`), transient buffer pooling (`acquire_transient_buffer` / `release_transient_buffer`), and buffer handle resolution (`get_cl_buffer`). Under ADR-009, plan-level buffer allocation becomes a backend rendering concern. Each backend implements its own buffer allocator.
- **`HostView`** — couples `cl.enqueue_copy`, `cl.Event`, and numpy padding-stripping. ADR-010 (ACCEPTED) replaces it with the `RetrievalFuture` Protocol. The OpenCL backend's `_OpenCLRetrievalFuture` absorbs `HostView`'s functionality. The shared-layer import of `HostView` is eliminated.

### Additional module dissolution

**`arch_primitives.py`** hosts the `PrecisionContext` ABC, `Float32Context`, and `Float16Context`. ADR-008 eliminates the entire hierarchy, replacing it with the `PrecisionConfig` frozen dataclass. Any remaining non-precision utilities (e.g., `c_tile_extent`) move to the shared layer's workload primitives.

**`kernel_signatures/`** contains the `KernelSignature` subclass hierarchy. ADR-007 (ACCEPTED) splits each signature into a shared `KernelContract` (frozen dataclass) and per-backend `KernelBinding` implementations. The `kernel_signatures/` sub-package dissolves: contracts move to `shared/kernel_contracts/`; bindings move to `backends/<name>/kernel_bindings/`.

**`execution_plan.py`** hosts the `DependencyProvider` hierarchy (`CacheProvider`, `RecomputeProvider`, `ComputeOnceProvider`, `StagedComputationProvider`), each embedding `cl.Event` and `KernelExecutor.launch()` calls. ADR-002 (ACCEPTED) replaces this with declarative plan node types. The lifecycle decisions are expressed structurally in the plan DAG (presence or absence of recompute `StreamingLoopNode`s), not as executable provider objects.

**`graph_recipes.py`** mixes DAG-description logic and OpenCL dispatch. Under ADR-001, DAG construction moves to the shared layer's plan builder; OpenCL dispatch becomes the first backend renderer.

### The shared-layer module inventory

After all accepted ADRs are applied, the shared layer comprises:

| Module | Origin | Governing ADR |
| :--- | :--- | :--- |
| `ModelSpec` | `model_spec.py` — composes `PrecisionConfig` as a field | ADR-008 |
| `ParameterSpace` | `parameter_space.py` — unchanged | — |
| `MemoryLayout` | `memory_layout.py` — unchanged | — |
| `PrecisionConfig` | Extracted from `arch_primitives.py` | ADR-008 |
| `HardwareProfile` | Extracted from `cl_context_manager.py` | ADR-006 |
| `StabilizationPolicy` | `stabilization_policy.py` — unchanged | — |
| `BufferHandle`, `BufferRole`, `BufferDescriptor` | Extracted from `launcher_infra.py` | ADR-009 |
| `RetrievalFuture` Protocol | New, replaces `HostView` | ADR-010 |
| `KernelContract` hierarchy | Extracted from `kernel_signatures/` | ADR-007 |
| `ProblemTypeStrategy`, `CceStrategy`, `BceStrategy` | Extracted from `execution_plan.py` | ADR-011 |
| Plan node types | New, replaces `DependencyProvider` hierarchy | ADR-002 |
| Plan builder | Extracted from `batch_processor.py` + `graph_recipes.py` | ADR-001 |
| Workload primitives | `workload_primitives.py` + pure-math helpers from `compute_patterns.py` | — |

### The backend-layer module inventory

Each backend requires:

| Module | Responsibility | Governing ADR |
| :--- | :--- | :--- |
| Context/device management | `cl.Context`, `VkDevice`, thread pool | ADR-001 |
| Hardware discovery | Populates `HardwareProfile` from native API | ADR-006 |
| Type mapping | Maps `PrecisionConfig.numpy_dtype` → native types | ADR-008 |
| Buffer allocator | Allocates physical memory from `BufferDescriptor` | ADR-009 |
| Retrieval implementation | `RetrievalFuture` Protocol conformance | ADR-010 |
| Plan renderer | Consumes plan DAG, produces native dispatch | ADR-001 |
| Kernel bindings | `KernelContract` → native dispatch format | ADR-007 |

### The design question

The two-layer jurisdictional model (shared vs. backend) is established by ADR-001 and confirmed by every subsequent ADR. The remaining decision is the **concrete physical directory structure**: how do the shared-layer modules and per-backend modules map to Python packages and files on disk?

---

## Decision Drivers

1. **ADR-001 (Plan boundary — shared layer produces backend-neutral data).** The physical structure must enforce the plan boundary. Importing from the shared package must never transitively import backend types (`pyopencl`, `vulkan`, `ctypes` for CPU). The directory split must make this invariant mechanically verifiable (e.g., by linting for disallowed imports in the `shared/` tree).

2. **ADR-001 (Three-tier jurisdictional model).** The Policy tier is always shared; Orchestration and Execution are always backend-side. The physical structure must mirror this — Policy-tier modules live in one tree, Orchestration/Execution modules live in backend-specific trees. A module's jurisdictional tier must be determinable from its file path alone.

3. **ADR-006, ADR-007, ADR-008, ADR-009, ADR-010, ADR-011 (module placement contracts).** Each accepted ADR specifies where its artefacts live — `HardwareProfile` in the shared layer, `KernelContract` in the shared layer, `KernelBinding` per-backend, `PrecisionConfig` in the shared layer, `BufferDescriptor` in the shared layer, `RetrievalFuture` Protocol in the shared layer, `ProblemTypeStrategy` in the shared layer. The directory structure must accommodate all of these placements without ambiguity.

4. **CONCEPT.md §1 (Architectural Elegance Feedback).** The directory structure is a first-class architectural primitive. If a new module creates tension with the existing structure (e.g., a utility needed by both shared and backend code), the response is to formalize its placement — not to break the two-layer boundary.

5. **STRUCTURE.md §2 (Architectural Autonomy).** Each architecture is self-contained. The module factoring is internal to `averaging_ensembled_classifier/` — it must not impose constraints on the framework's top-level structure.

6. **Downstream dependencies.** ADR-013 (kernel source strategy) needs the physical layout to determine where kernel source files live. ADR-014 (build system integration) needs the package structure to define build targets. ADR-015 (Python ↔ native interop) needs the backend package structure to determine where FFI bindings are installed. All three are blocked by this ADR.

7. **Import clarity.** Third-party consumers (tests, the main orchestrator, future experiment scripts) must have clear, stable import paths. The structure must support both `from averaging_ensembled_classifier.shared.model_spec import ModelSpec` and `from averaging_ensembled_classifier.backends.opencl.renderer import OpenCLPlanRenderer` patterns.

---

## Options Considered

### Option A: Dedicated `shared/` sub-package with `backends/` sibling

All shared-layer modules move into `src/shared/`. All backend-specific modules live under `src/backends/<name>/`. A single `orchestrator.py` at `src/` top level assembles the shared layer with the selected backend.

```
src/
├── __init__.py
├── orchestrator.py
├── shared/
│   ├── __init__.py
│   ├── model_spec.py
│   ├── parameter_space.py
│   ├── memory_layout.py
│   ├── precision_config.py
│   ├── hardware_profile.py
│   ├── stabilization_policy.py
│   ├── plan_builder.py
│   ├── plan_types.py
│   ├── buffer_lifecycle.py
│   ├── retrieval_future.py
│   ├── problem_type_strategy.py
│   ├── workload_primitives.py
│   └── kernel_contracts/
│       ├── __init__.py
│       ├── phase_1_act.py
│       ├── phase_2_learn_A_production.py
│       ├── phase_2_learn_B_processing.py
│       ├── phase_2_learn_C_reduction.py
│       ├── phase_2_learn_D_backprop.py
│       └── phase_3_update.py
└── backends/
    ├── __init__.py
    ├── opencl/
    │   ├── __init__.py
    │   ├── renderer.py
    │   ├── context.py
    │   ├── discovery.py
    │   ├── type_mapping.py
    │   ├── buffer_allocator.py
    │   ├── retrieval.py
    │   └── kernel_bindings/
    │       ├── __init__.py
    │       ├── phase_1_act.py
    │       └── ...
    ├── vulkan/
    │   ├── __init__.py
    │   ├── renderer.py
    │   ├── context.py
    │   ├── discovery.py
    │   ├── type_mapping.py
    │   ├── buffer_allocator.py
    │   ├── retrieval.py
    │   └── kernel_bindings/
    │       └── ...
    └── cpu/
        ├── __init__.py
        ├── renderer.py
        ├── discovery.py
        ├── type_mapping.py
        ├── buffer_allocator.py
        ├── retrieval.py
        └── kernel_bindings/
            └── ...
```

**Advantages:**
- The Plan boundary is physically visible: `shared/` never imports from `backends/`; `backends/` imports from `shared/`. This invariant is mechanically verifiable by static analysis (e.g., a lint rule rejecting `from ..backends` in any `shared/` module).
- Each module's jurisdictional tier is determinable from its file path. A module in `shared/` is Policy-tier; a module in `backends/<name>/` is Orchestration/Execution-tier. No ambiguity.
- The `kernel_contracts/` sub-package within `shared/` mirrors the `kernel_bindings/` sub-package within each backend, making the ADR-007 contract/binding split visible in the directory structure.
- Import paths are self-documenting: `from .shared.plan_types import KernelDispatchNode` vs. `from .backends.opencl.renderer import OpenCLPlanRenderer`.
- Backend packages are fully autonomous — adding a new backend means adding a new directory under `backends/` with no changes to `shared/` or other backends.

**Disadvantages:**
- Every existing import path changes. All downstream test files and the main orchestrator must update their imports. This is a one-time mechanical cost.
- Two levels of package nesting (`src/shared/`, `src/backends/opencl/`) may feel heavy for the current module count (~15 shared modules, ~7 per backend).
- The `shared/` directory name is generic. It carries semantic weight only in the context of this architecture's jurisdictional model.

### Option B: Shared modules remain at `src/` top level; backends as sub-package

Shared-layer modules remain at the `src/` top level (where they are today). Only backend-specific modules move into `src/backends/<name>/`. Dissolved modules (`arch_primitives.py`, `cl_context_manager.py`, `compute_patterns.py`, `launcher_infra.py`, `kernel_signatures/`) are removed; their shared extractions become new top-level files.

```
src/
├── __init__.py
├── orchestrator.py
├── model_spec.py
├── parameter_space.py
├── memory_layout.py
├── precision_config.py
├── hardware_profile.py
├── stabilization_policy.py
├── plan_builder.py
├── plan_types.py
├── buffer_lifecycle.py
├── retrieval_future.py
├── problem_type_strategy.py
├── workload_primitives.py
├── kernel_contracts/
│   └── ...
└── backends/
    ├── opencl/
    │   └── ...
    ├── vulkan/
    │   └── ...
    └── cpu/
        └── ...
```

**Advantages:**
- Minimal import churn for existing shared-layer modules — `model_spec.py`, `parameter_space.py`, etc. retain their current import paths.
- Flatter structure — fewer nested packages.
- The "shared layer" is simply "everything at top level that isn't `backends/`."

**Disadvantages:**
- **The Plan boundary is not physically delimited.** There is no directory boundary between shared and backend code — only the convention that `backends/` is backend-specific and everything else is shared. A new module placed at the top level could be either shared or an accidental leak of backend logic. The boundary is enforceable only by convention, not by package structure.
- **Import lint rules are harder to express.** "No module at `src/` top level may import from `src/backends/`" is expressible but less obvious than "no module in `src/shared/` may import from `src/backends/`." The rule depends on enumerating which top-level modules are shared — a list that must be maintained as modules are added or renamed.
- **Dissolved source modules leave artefacts.** The current `cl_context_manager.py`, `compute_patterns.py`, etc. are removed, and their shared extractions become new top-level files (`precision_config.py`, `hardware_profile.py`, etc.). This makes the top level grow to ~16 files plus `kernel_contracts/` and `backends/`, approaching the same flat-namespace problem that motivated the refactoring.
- **Backend kernel_bindings have no structural counterpart.** The `kernel_contracts/` directory at the top level mirrors `backends/<name>/kernel_bindings/`, but the structural parallel is weaker without a `shared/` parent.

### Option C: Three-tier directory structure mirroring the jurisdictional model

Map ADR-001's three-tier jurisdictional model directly to three top-level directories: `policy/`, `orchestration/`, `execution/`. Policy-tier modules (shared) live in `policy/`. Orchestration-tier modules (backend rendering, dispatch sequencing) live in `orchestration/<backend>/`. Execution-tier artefacts (kernel source files) live in `execution/<backend>/`.

```
src/
├── __init__.py
├── orchestrator.py
├── policy/
│   ├── __init__.py
│   ├── model_spec.py
│   ├── plan_builder.py
│   ├── kernel_contracts/
│   └── ...
├── orchestration/
│   ├── opencl/
│   │   ├── renderer.py
│   │   ├── kernel_bindings/
│   │   └── ...
│   ├── vulkan/
│   │   └── ...
│   └── cpu/
│       └── ...
└── execution/
    ├── opencl/
    │   └── (kernel sources, .cl.c files)
    ├── vulkan/
    │   └── (GLSL shaders)
    └── cpu/
        └── (C source files)
```

**Advantages:**
- Directly embodies the three-tier jurisdictional model from ADR-001.
- The Orchestration/Execution split separates Python rendering code from kernel source files.

**Disadvantages:**
- **Over-specifies the tier model.** ADR-001 §Three-tier jurisdictional model explicitly states that the Orchestration and Execution tiers "redistribute freely depending on the node's structure and the backend's native capabilities." The physical separation of `orchestration/` and `execution/` reifies a boundary that is conceptually fluid — e.g., Node 16's kernel manages its own internal reduction (Execution absorbs Orchestration), but its binding code would still live in `orchestration/`. The directory path would contradict the actual tier assignment.
- **Duplicates backend identity across two trees.** Each backend has a directory in both `orchestration/` and `execution/`, doubling the backend namespace and complicating the build system (ADR-014). Backend-specific imports must cross two sibling packages: `from ..orchestration.opencl.renderer import ...` and `from ..execution.opencl import ...`.
- **Kernel source placement is an ADR-013 concern.** This ADR (ADR-012) governs the Python module factoring. Kernel source file organization is deferred to ADR-013. Prescribing an `execution/` directory for kernel sources would pre-empt ADR-013's decision space.
- **Unfamiliar naming.** `policy/` as a Python package name is unusual and may confuse contributors who are not intimate with the ADR-001 tier vocabulary. The `shared/` + `backends/` naming from Option A is more immediately understandable.

---

## Analysis

### Eliminating Option B

Option B is eliminated because it fails to physically delimit the Plan boundary. The central invariant of the multi-backend refactoring — established by ADR-001 and confirmed by every subsequent ADR — is the separation of shared (Policy-tier) code from backend-specific (Orchestration/Execution-tier) code. This boundary must be enforceable, not merely conventional.

The analysis parallels ADR-006's elimination of its Option A (raw hardware measurements with backend-category discriminator): Option A allowed backend-discriminated logic to appear in the shared layer, requiring runtime branching to disambiguate. Option B here allows backend-specific code to appear at the same directory level as shared code, requiring an external convention to disambiguate. In both cases, the failing option makes violation of the boundary structurally indistinguishable from correct behavior.

Concretely: a contributor adding a new module at `src/` top level has no structural signal that the module must be backend-neutral. Under Option A, the same contributor must choose between `src/shared/` (shared) and `src/backends/<name>/` (backend-specific) — the directory choice is a forcing function that prevents accidental boundary violations.

Furthermore, the dissolved modules (`cl_context_manager.py`, `compute_patterns.py`, `launcher_infra.py`, `arch_primitives.py`, `kernel_signatures/`) are replaced by new shared extractions (`precision_config.py`, `hardware_profile.py`, `buffer_lifecycle.py`, `retrieval_future.py`, `kernel_contracts/`) plus new top-level files (`plan_builder.py`, `plan_types.py`, `problem_type_strategy.py`). Under Option B, the top level grows from the current 14 modules to ~16 modules plus `kernel_contracts/` and `backends/`. This re-creates the flat-namespace problem that motivated the discussion.

### Eliminating Option C

Option C is eliminated because it over-specifies a boundary that ADR-001 explicitly identifies as fluid. ADR-001 §Three-tier jurisdictional model states:

> "The Orchestration and Execution tiers redistribute freely depending on the node's structure and the backend's native capabilities."

The three-tier model is an analytical framework for understanding how decisions flow — not a physical partitioning scheme. The Plan boundary (Policy vs. everything else) is the only boundary that is invariant across all node types and all backends. The Orchestration/Execution boundary varies:

- For host-orchestrated reduction trees (Nodes 14, 15, 20): Orchestration drives the stage loop; Execution is per-stage kernels.
- For Node 16: Orchestration collapses into Execution — the kernel manages its own internal reduction.
- For simple dispatches (Nodes 4–11, 17–19, 21, 24, 25): Orchestration is a single `dispatch()` call.

Encoding the Orchestration/Execution split as a physical directory structure forces every backend module into one tier or the other, contradicting the design's deliberate flexibility. A backend's `renderer.py` is Orchestration; its kernel source files are Execution; but its `buffer_allocator.py` serves both tiers (allocating plan-level buffers is Orchestration; allocating renderer-internal scratch is Orchestration serving Execution). The assignment is contextual, not categorical.

Additionally, Option C pre-empts ADR-013's decision space by prescribing an `execution/` directory for kernel source files. ADR-013 is specifically scoped to resolve kernel source organization; this ADR should provide the container structure (`backends/<name>/`) without dictating the internal layout of kernel source artefacts.

### Choosing Option A

Option A provides a single, unambiguous physical boundary at `shared/` vs. `backends/` — precisely mirroring the Plan boundary established by ADR-001. This boundary is:

- **Mechanically enforceable.** A lint rule can prohibit imports from `backends` in any module under `shared/`. No enumeration of "shared modules" is needed — the directory membership is the definition.
- **Self-documenting.** A module's file path communicates its jurisdictional tier. `shared/plan_types.py` is Policy-tier; `backends/opencl/renderer.py` is Orchestration-tier. No external context required.
- **Structurally parallel.** `shared/kernel_contracts/` mirrors `backends/<name>/kernel_bindings/`, making the ADR-007 contract/binding split visible in the directory tree. `shared/buffer_lifecycle.py` contains the declarations; `backends/<name>/buffer_allocator.py` contains the implementations. The naming convention reinforces the jurisdictional model at every level.
- **Extensible.** A new backend is added by creating `backends/<new_name>/` with the canonical module set (renderer, context, discovery, type_mapping, buffer_allocator, retrieval, kernel_bindings). No changes to `shared/` or other backends. This satisfies ADR-001's extensibility requirement: "new backends require implementing a renderer — not modifying the shared orchestration layer."

The one-time import path update is a mechanical cost justified by the permanent structural benefit. Every downstream consumer (tests, main orchestrator, experiment scripts) gains import paths that are explicit about the jurisdictional tier of the imported type.

---

## Decision

**Option A: Dedicated `shared/` sub-package with `backends/` sibling and top-level `orchestrator.py`.**

The `src/` directory is physically split into two sub-packages (`shared/`, `backends/`) plus a single top-level assembly module. All current modules are either relocated into the appropriate sub-package or dissolved, with their shared extractions placed in `shared/` and their backend-specific logic absorbed by `backends/opencl/`.

### The concrete directory structure

```
src/
├── __init__.py
├── orchestrator.py                   # Top-level assembly: shared + selected backend
├── shared/
│   ├── __init__.py
│   ├── model_spec.py                # ModelSpec composes PrecisionConfig (ADR-008)
│   ├── parameter_space.py
│   ├── memory_layout.py
│   ├── precision_config.py          # ADR-008 — PrecisionConfig frozen dataclass + factory
│   ├── hardware_profile.py          # ADR-006 — HardwareProfile frozen dataclass only
│   ├── stabilization_policy.py
│   ├── plan_builder.py              # Consumes above, produces plan DAG
│   ├── plan_types.py                # Node dataclasses from ADR-002 (incl. RetrievalNode)
│   ├── buffer_lifecycle.py          # ADR-009 — BufferHandle, BufferRole, BufferDescriptor
│   ├── retrieval_future.py          # ADR-010 — RetrievalFuture Protocol
│   ├── problem_type_strategy.py     # ADR-011 — ProblemTypeStrategy ABC, CceStrategy, BceStrategy
│   ├── workload_primitives.py       # TilingScheme, WorkTile, GatherPrimitive (pure math)
│   └── kernel_contracts/            # ADR-007 — KernelContract frozen dataclasses
│       ├── __init__.py              # Exports KernelContractBlock registry
│       ├── phase_1_act.py           # Node 4, 5; Node 6 (CCE) and Node 7 (BCE) as
│       │                            #   separate contracts per ADR-011 Strategy B
│       ├── phase_2_learn_A_production.py  # Nodes 8, 9, 10 — unified contracts with
│       │                            #   ScalarParamSpec(FLAG_problem_type) per ADR-011 Strategy A
│       ├── phase_2_learn_B_processing.py
│       ├── phase_2_learn_C_reduction.py
│       ├── phase_2_learn_D_backprop.py
│       └── phase_3_update.py
├── backends/
│   ├── __init__.py
│   ├── opencl/
│   │   ├── __init__.py
│   │   ├── renderer.py              # OpenCL PlanRenderer — render() returns Dict[str, RetrievalFuture]
│   │   ├── context.py               # cl.Context + queue management
│   │   ├── discovery.py             # Populates HardwareProfile from cl.device_info
│   │   ├── type_mapping.py          # Maps PrecisionConfig.numpy_dtype → OpenCL types
│   │   ├── buffer_allocator.py      # Allocates cl.Buffer from BufferDescriptor; reuse via lifetime intervals
│   │   ├── retrieval.py             # _OpenCLRetrievalFuture (absorbs HostView functionality)
│   │   └── kernel_bindings/         # ADR-007 — OpenCL KernelBinding implementations
│   │       ├── __init__.py
│   │       ├── phase_1_act.py       # Injects flat_tile_index, marshals cl.Buffer args
│   │       └── ...
│   ├── vulkan/
│   │   ├── __init__.py
│   │   ├── renderer.py
│   │   ├── context.py               # VkDevice + VkQueue management
│   │   ├── discovery.py             # Populates HardwareProfile from VkPhysicalDevice
│   │   ├── type_mapping.py          # Maps PrecisionConfig.numpy_dtype → Vulkan/GLSL types
│   │   ├── buffer_allocator.py      # Suballocates from VkDeviceMemory using lifetime intervals
│   │   ├── retrieval.py             # _VulkanRetrievalFuture (VkFence + staging buffer)
│   │   └── kernel_bindings/         # Push constants + descriptor sets
│   │       └── ...
│   └── cpu/
│       ├── __init__.py
│       ├── renderer.py
│       ├── discovery.py             # Populates HardwareProfile from ISA flags + OS queries
│       ├── type_mapping.py          # Maps PrecisionConfig.numpy_dtype → C types
│       ├── buffer_allocator.py      # malloc / arena allocation from BufferDescriptor
│       ├── retrieval.py             # _CPURetrievalFuture (zero-copy, zero-wait)
│       └── kernel_bindings/         # C function arg struct marshalling
│           └── ...
└── tests/                           # Existing test directory retained
```

### Module relocation schedule

The following table maps every current module to its destination in the new structure:

| Current module | Destination | Transformation |
| :--- | :--- | :--- |
| `model_spec.py` | `shared/model_spec.py` | Composes `PrecisionConfig` as field (ADR-008); removes `PrecisionContext` inheritance |
| `parameter_space.py` | `shared/parameter_space.py` | Unchanged |
| `memory_layout.py` | `shared/memory_layout.py` | Unchanged |
| `stabilization_policy.py` | `shared/stabilization_policy.py` | Unchanged (already backend-neutral) |
| `workload_primitives.py` | `shared/workload_primitives.py` | Absorbs pure-math helpers from `compute_patterns.py` |
| `arch_primitives.py` | **Dissolved** | `PrecisionConfig` → `shared/precision_config.py` (ADR-008). Remaining utilities (e.g., `c_tile_extent`) → `shared/workload_primitives.py` |
| `cl_context_manager.py` | **Dissolved** | `HardwareProfile` → `shared/hardware_profile.py` (ADR-006). `OpenCLContextManager` → `backends/opencl/context.py`. `DiscoveredArchConstants` hierarchy eliminated (ADR-006, ADR-008). `CLBundle` → `backends/opencl/context.py` internal |
| `compute_patterns.py` | **Dissolved** | `ReductionPlan` → superseded by `ReductionTreePlan` in `shared/plan_types.py` (ADR-003). `AggregationManager` dispatch → `backends/opencl/renderer.py`. Tiling/padding helpers → `shared/workload_primitives.py` |
| `launcher_infra.py` | **Dissolved** | `BufferHandle` → `shared/buffer_lifecycle.py` (ADR-009). `HostView` → absorbed by `backends/opencl/retrieval.py` (ADR-010). `BufferManager` → `backends/opencl/buffer_allocator.py`. `Services` → eliminated |
| `execution_plan.py` | **Dissolved** | `DependencyProvider` hierarchy → superseded by plan node types in `shared/plan_types.py` (ADR-002). `ProblemTypeStrategy` hierarchy → `shared/problem_type_strategy.py` (ADR-011) |
| `graph_recipes.py` | **Split** | DAG construction logic → `shared/plan_builder.py`. OpenCL dispatch logic → `backends/opencl/renderer.py` |
| `batch_processor.py` | **Split** | Plan construction → `shared/plan_builder.py`. Execution orchestration → `orchestrator.py` |
| `kernel_signatures/` | **Dissolved** | Contracts → `shared/kernel_contracts/` (ADR-007). OpenCL bindings → `backends/opencl/kernel_bindings/` (ADR-007) |
| `main_orchestrator.py` | `orchestrator.py` | Refactored to assemble shared layer + selected backend |

### The import boundary invariant

The physical structure enforces a strict, unidirectional import graph:

```
orchestrator.py  ──imports──▶  shared/*
       │
       └─────────imports──▶  backends/<selected>/*  ──imports──▶  shared/*
```

The following rules are mechanically enforceable by static analysis:

1. **No module in `shared/` may import from `backends/`.** This is the Plan boundary. The shared layer produces backend-neutral data structures; it never references backend-specific types.
2. **No backend may import from another backend.** `backends/opencl/` never imports from `backends/vulkan/` or `backends/cpu/`. Each backend is autonomous.
3. **Backends may import from `shared/`.** Every backend consumes shared types (`KernelContract`, `BufferDescriptor`, `HardwareProfile`, `PrecisionConfig`, `RetrievalFuture` Protocol, plan node types).
4. **`orchestrator.py` imports from both `shared/` and the selected `backends/<name>/`.** It is the assembly point that bridges the two layers.

### Kernel contract organization

The `shared/kernel_contracts/` directory maps one-to-one with the architecture's computational phases, carrying `KernelContract` frozen dataclass instances:

| File | Kernels (Nodes) | ADR-011 strategy | Notes |
| :--- | :--- | :--- | :--- |
| `phase_1_act.py` | `forward_pass` (4), `compute_hidden_mask` (5), `compute_probs_loss_cce_chunk` (6), `compute_probs_loss_bce_chunk` (7) | Nodes 6/7: **Strategy B** — two distinct `KernelContract` instances with different `kernel_name`, different targets buffer `BufferParamSpec`, and different loss output topology | Exports: `forward_pass_contract`, `compute_hidden_mask_contract`, `compute_probs_loss_cce_chunk_contract`, `compute_probs_loss_bce_chunk_contract` |
| `phase_2_learn_A_production.py` | `calculate_module_param_grads_chunk` (8), `backprop_error_to_hidden_chunk` (9), `calculate_chunk_temp_gradients` (10), `clip_partial_gradients_global_norm` (11) | Nodes 8/9/10: **Strategy A** — unified `KernelContract` per kernel with `ScalarParamSpec(FLAG_problem_type)` and conditional validation preconditions for the targets buffer | Exports per kernel: one `KernelContract` instance |
| `phase_2_learn_B_processing.py` | `gather_and_permute_grad_h` (13) | — | Global Barrier kernel |
| `phase_2_learn_C_reduction.py` | Aggregation kernels (`aggregate_identity`, `aggregate_register_reduce`, `aggregate_local_reduce`), `clip_intermediate_grad` | — | Used within `ReductionTreePlan` rendering |
| `phase_2_learn_D_backprop.py` | `stabilize_and_reduce_grad_hidden_activations` (16), streaming backprop kernels (17, 18, 19) | — | Node 16 is the specialized kernel with collapsed Orchestration tier |
| `phase_3_update.py` | `adam_update` (21), `normalize_updated_weights` (24), `normalize_updated_temperatures` (25) | — | |

The `shared/problem_type_strategy.py` module houses the `ProblemTypeStrategy` ABC, `CceStrategy`, and `BceStrategy`. The plan builder queries this strategy to:
- Select the correct `kernel_name` for Nodes 6/7 (Strategy B): `CceStrategy.loss_kernel_name = "compute_probs_loss_cce_chunk"`, `BceStrategy.loss_kernel_name = "compute_probs_loss_bce_chunk"`.
- Provide the correct FLAG value for Nodes 8/9/10 (Strategy A): `CceStrategy.problem_type_flag = 0`, `BceStrategy.problem_type_flag = 1`.
- Determine the DAG topology: `BceStrategy` includes the BCE loss reduction sub-tree in Node 14's inputs; `CceStrategy` omits it.

### Backend module contracts

Each backend's module set follows a canonical structure. The responsibilities are formally defined by upstream ADRs:

**`discovery.py`** — Populates `HardwareProfile` (ADR-006) from the backend's native hardware query mechanism. The `max_reduce_fan_in` derivation is backend-owned per ADR-006's decision: OpenCL derives it from `min(256, device.max_work_group_size)`; Vulkan from `maxComputeWorkGroupSize[0]`; CPU from L1 cache capacity.

**`type_mapping.py`** — Maps `PrecisionConfig.numpy_dtype` (ADR-008) to the backend's native type vocabulary. OpenCL: `np.float32` → `"float"`, `np.float16` → `"half"` (for `-D SCALAR_TYPE=...` compiler flags). Vulkan: `np.float32` → `float` (GLSL), `np.float16` → `float16_t`. CPU: `np.float32` → `float` (C), `np.float16` → `_Float16` or emulated.

**`buffer_allocator.py`** — Consumes the plan's `Tuple[BufferDescriptor, ...]` (ADR-009). Allocates physical memory from `size_bytes`, builds the `BufferHandle` → physical map, and optionally uses `role`, `producing_node`, and `last_consumer` annotations to optimize memory reuse. Vulkan uses lifetime intervals for suballocation packing; OpenCL allocates discrete `cl.Buffer`s; CPU uses `malloc` or arena allocation.

**`retrieval.py`** — Implements the `RetrievalFuture` Protocol (ADR-010). The OpenCL implementation (`_OpenCLRetrievalFuture`) absorbs `HostView`'s functionality: pre-allocated numpy host buffer, `cl.enqueue_copy`, `cl.Event` completion, padding-stripping via numpy slice in `.result()`. The CPU implementation wraps a zero-copy numpy view. The Vulkan implementation wraps `VkFence` + staging buffer. The `.release()` method bridges ADR-009's `last_consumer` semantics — the renderer retains `BATCH_OUTPUT` physical memory until the host signals consumption complete.

**`renderer.py`** — The `PlanRenderer` that consumes the plan DAG (ADR-001) and renders it using the backend's native execution model. Its `render()` method returns `Dict[str, RetrievalFuture]` — one future per `RetrievalNode`. Reduction tree rendering (ADR-003) and streaming loop rendering (ADR-004) are internal to the renderer.

**`kernel_bindings/`** — Per-kernel `KernelBinding` implementations (ADR-007) that translate validated `KernelContract` data to native dispatch format. For Strategy A kernels (Nodes 8/9/10), the OpenCL binding passes `src_scalar_FLAG_problem_type` as a positional scalar; the Vulkan binding delivers it as a specialization constant or push constant; the CPU binding passes it as a C function parameter (ADR-011 §Backend rendering responsibilities).

**`context.py`** (OpenCL, Vulkan only) — Backend-specific execution context lifecycle. OpenCL: `cl.Context` + `cl.CommandQueue` setup, kernel program compilation. Vulkan: `VkDevice` + `VkQueue` setup, pipeline cache. The CPU backend has no equivalent — its "context" is the thread pool, which may be initialized directly in `renderer.py`.

### Upstream ADR confirmation

The following table confirms that every accepted ADR's module placement requirements are satisfied by this structure:

| ADR | Shared-layer artefact | File | Backend artefact | File |
| :--- | :--- | :--- | :--- | :--- |
| ADR-002 | `KernelDispatchNode`, `ReductionTreeNode`, `StreamingLoopNode`, `BarrierNode`, `RetrievalNode` | `shared/plan_types.py` | — | — |
| ADR-003 | `ReductionTreePlan` | `shared/plan_types.py` | Rendering (ping-pong, offset-list upload) | `backends/<name>/renderer.py` |
| ADR-004 | `StreamingLoopNode`, `ScratchBufferSpec`, `ParameterStride` | `shared/plan_types.py` | Streaming loop execution | `backends/<name>/renderer.py` |
| ADR-006 | `HardwareProfile` frozen dataclass | `shared/hardware_profile.py` | Construction from native discovery | `backends/<name>/discovery.py` |
| ADR-007 | `KernelContract`, `BufferParamSpec`, `ScalarParamSpec`, `LocalMemorySpec`, `PlacementContract`, `KernelContractBlock` | `shared/kernel_contracts/` | `KernelBinding` implementations | `backends/<name>/kernel_bindings/` |
| ADR-008 | `PrecisionConfig` frozen dataclass + factory | `shared/precision_config.py` | `numpy_dtype` → native type mapping | `backends/<name>/type_mapping.py` |
| ADR-009 | `BufferHandle`, `BufferRole`, `BufferDescriptor` | `shared/buffer_lifecycle.py` | Physical allocation + reuse | `backends/<name>/buffer_allocator.py` |
| ADR-010 | `RetrievalFuture` Protocol | `shared/retrieval_future.py` | Protocol implementations | `backends/<name>/retrieval.py` |
| ADR-011 | `ProblemTypeStrategy` ABC, `CceStrategy`, `BceStrategy` | `shared/problem_type_strategy.py` | FLAG delivery mechanism | `backends/<name>/kernel_bindings/` |

---

## Consequences

### Positive

- **Mechanically enforceable Plan boundary.** The `shared/` directory is a closed package with no backend imports. Static analysis tools (mypy, ruff, a custom lint rule) can verify this invariant across every commit. Backend-specific types (`cl.Buffer`, `VkDevice`, C FFI handles) structurally cannot appear in any `shared/` module — they live in a different package subtree.

- **Self-documenting module placement.** A module's file path unambiguously communicates its jurisdictional tier. Every `shared/` module is backend-neutral Policy-tier code. Every `backends/<name>/` module is backend-specific Orchestration/Execution-tier code. No external convention or ADR cross-reference is required to determine a module's role.

- **Backend autonomy.** Adding a new backend requires creating `backends/<new_name>/` with the canonical module set (renderer, discovery, type_mapping, buffer_allocator, retrieval, kernel_bindings). No changes to `shared/`, no changes to other backends, no changes to `orchestrator.py` beyond adding the backend to the selection logic. This directly satisfies ADR-001's extensibility requirement.

- **ADR-007 contract/binding parallel.** The `shared/kernel_contracts/` ↔ `backends/<name>/kernel_bindings/` structural symmetry makes the contract/binding split navigable. A developer examining `shared/kernel_contracts/phase_1_act.py` knows to look for corresponding bindings in `backends/opencl/kernel_bindings/phase_1_act.py`, `backends/vulkan/kernel_bindings/phase_1_act.py`, and `backends/cpu/kernel_bindings/phase_1_act.py`.

- **Clean dissolution of the Services layer.** The four dissolved modules (`arch_primitives.py`, `cl_context_manager.py`, `compute_patterns.py`, `launcher_infra.py`) are not merely renamed — their responsibilities are redistributed to architecturally correct locations. Shared extractions go to `shared/`; backend-specific code goes to `backends/opencl/`. The dissolution is traceable through the module relocation schedule.

- **Unblocks downstream ADRs.** ADR-013 (kernel source strategy) can now specify source file locations within the `backends/<name>/` tree. ADR-014 (build system integration) can define Meson targets for `shared/` (pure Python, no compilation) and per-backend packages (with optional C/SPIR-V compilation). ADR-015 (Python ↔ native interop) can specify FFI binding installation within each backend package.

- **Testability.** Shared-layer unit tests import from `shared/` and never require a backend, a device, or a driver. They construct synthetic `HardwareProfile` and `PrecisionConfig` values, build plans, and validate contracts — all in pure Python with no hardware dependency. Backend-specific tests import from `backends/<name>/` and exercise device interaction. The directory structure makes the test boundary mirror the code boundary.

### Negative

- **Import path migration.** Every existing import of `model_spec`, `parameter_space`, `memory_layout`, `stabilization_policy`, `workload_primitives`, etc. must be updated to add the `shared.` prefix. Every import of `cl_context_manager`, `launcher_infra`, `compute_patterns`, `arch_primitives`, and `kernel_signatures` must be redirected to the appropriate new module. This is a one-time mechanical cost affecting all test files, the main orchestrator, and any internal tooling.

- **Package depth.** Backend modules are three levels deep: `src/backends/opencl/kernel_bindings/phase_1_act.py`. This is deeper than the current flat structure but is a necessary consequence of the two-layer split plus per-kernel binding files. The depth is bounded — there is no fourth level.

- **`shared/` naming is domain-specific.** The name "shared" carries semantic weight only in the context of the Policy/Orchestration jurisdictional split. A reader unfamiliar with the ADR series might not immediately understand what "shared" means. However, the primary audience for this directory structure is the project's developers, who are expected to be familiar with ADR-001's terminology. The `__init__.py` docstring and `STRUCTURE.md` can provide orientation.

- **Single `orchestrator.py` at top level breaks symmetry.** All other modules are in sub-packages, but `orchestrator.py` sits at the `src/` top level. This is deliberate — it is the assembly point that imports from both `shared/` and `backends/`, so it cannot belong to either. But it creates a minor structural asymmetry. If the orchestration logic grows, it may evolve into an `orchestration/` sub-package — that would be a future ADR concern.

### Migration implications

Per ADR-017's phasing:

**Phase 0 (Foundation — no behavioral change):**
- Create `shared/` and `backends/opencl/` directory skeletons.
- Move backend-neutral modules (`model_spec.py`, `parameter_space.py`, `memory_layout.py`, `stabilization_policy.py`, `workload_primitives.py`) into `shared/`.
- Extract `HardwareProfile` (ADR-006) → `shared/hardware_profile.py`.
- Extract `PrecisionConfig` (ADR-008) → `shared/precision_config.py`.
- Extract `KernelContract` hierarchy (ADR-007) → `shared/kernel_contracts/`.
- Extract `ProblemTypeStrategy` (ADR-011) → `shared/problem_type_strategy.py`.
- Move OpenCL-specific modules into `backends/opencl/` with thin shim adapters.
- All existing tests pass with updated import paths.

**Phase 1 (Plan Model — new shared layer, unused):**
- Implement plan node types → `shared/plan_types.py`.
- Implement `BufferHandle`, `BufferRole`, `BufferDescriptor` → `shared/buffer_lifecycle.py`.
- Implement `RetrievalFuture` Protocol → `shared/retrieval_future.py`.
- Implement plan builder → `shared/plan_builder.py`.
- The OpenCL backend is not yet wired to the plan model.

**Phase 2 (OpenCL Renderer — dual code path):**
- Implement OpenCL `PlanRenderer` → `backends/opencl/renderer.py`.
- Implement `_OpenCLRetrievalFuture` → `backends/opencl/retrieval.py`.
- Implement OpenCL `KernelBinding`s → `backends/opencl/kernel_bindings/`.
- Implement OpenCL `buffer_allocator.py`, `discovery.py`, `type_mapping.py`, `context.py`.
- Validation gate: bit-identical results via new and old code paths.

**Phase 4 (Old Code Path Removal):**
- Delete dissolved modules: `arch_primitives.py`, `cl_context_manager.py`, `compute_patterns.py`, `launcher_infra.py`, `execution_plan.py`, `kernel_signatures/`, `graph_recipes.py`, `batch_processor.py`.
- The physical directory structure is now in its final state.

---

## References

- [ADR-001: Backend Abstraction Boundary](ADR-001-backend-abstraction-boundary.md) — plan-as-data-structure principle; three-tier jurisdictional model (Policy / Orchestration / Execution); two-layer physical boundary (shared / backend); extensibility requirement
- [ADR-002: Plan Node Types & Synchronization Structure](ADR-002-plan-node-types-and-synchronization-structure.md) — `KernelDispatchNode`, `ReductionTreeNode`, `StreamingLoopNode`, `BarrierNode`, `RetrievalNode`; `DependencyProvider` dissolution
- [ADR-003: Reduction Tree Plan Representation](ADR-003-reduction-tree-plan-representation.md) — `ReductionTreePlan` in shared; rendering in backend; two-tier buffer scope formalization
- [ADR-004: Streaming Loop Plan Representation](ADR-004-streaming-loop-plan-representation.md) — `StreamingLoopNode`, `ScratchBufferSpec`, `ParameterStride` in shared; loop execution in backend
- [ADR-006: Hardware Profile](ADR-006-hardware-profile.md) — `HardwareProfile` frozen dataclass → `shared/hardware_profile.py`; backend-constructed via `discovery.py`
- [ADR-007: KernelSignature Contract/Binding Split](ADR-007-kernel-signature-contract-binding-split.md) — `KernelContract` → `shared/kernel_contracts/`; `KernelBinding` → `backends/<name>/kernel_bindings/`
- [ADR-008: Precision Configuration](ADR-008-precision-configuration.md) — `PrecisionConfig` → `shared/precision_config.py`; backend type mapping → `backends/<name>/type_mapping.py`; `arch_primitives.py` dissolution
- [ADR-009: Buffer Lifecycle in the Plan Model](ADR-009-buffer-lifecycle-in-the-plan-model.md) — `BufferHandle`, `BufferRole`, `BufferDescriptor` → `shared/buffer_lifecycle.py`; physical allocation → `backends/<name>/buffer_allocator.py`
- [ADR-010: D2H Transfer & Phase Sync Points](ADR-010-d2h-transfer-and-phase-sync-points.md) — `RetrievalFuture` Protocol → `shared/retrieval_future.py`; `HostView` dissolution → `backends/opencl/retrieval.py`
- [ADR-011: CCE/BCE Strategy Delegation](ADR-011-cce-bce-strategy-delegation.md) — `ProblemTypeStrategy` → `shared/problem_type_strategy.py`; mixed Strategy B (Nodes 6/7) / Strategy A (Nodes 8/9/10); kernel contract organization impact
- [CONCEPT.md](../CONCEPT.md) — §1 Architectural Elegance Feedback; §3 Modular Dumb Kernels
- [CONTRACT.md](../CONTRACT.md) — Article 1.4 Collaborative Interface Verifiability; Article 2 Parameter Lexical Mandate
- [STRUCTURE.md](../../STRUCTURE.md) — §2 Architectural Autonomy; §3 Specification of Implementation
- [CPU_BACKEND.md](../CPU_BACKEND.md) — threading model, ISA detection, cache-line constants
- [VULKAN_BACKEND.md](../VULKAN_BACKEND.md) — command buffer strategy, subgroup properties, staging buffers
