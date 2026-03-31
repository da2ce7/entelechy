# Phase 6: Legacy PyOpenCL Removal — Detailed Plan

**Status:** ✅ Complete  
**Phase:** 6 of 6 (terminal phase)  
**Objective:** Remove the legacy PyOpenCL code path, Phase 0 shim modules, deprecated compatibility APIs, and legacy test infrastructure. After this phase, the system operates exclusively through plan-model dispatch via `PlanRenderer` implementations. The physical directory structure reaches its final state (ADR-012). The `main_orchestrator.py` is rewritten to use the `Engine` / `WorkTicket` user-facing API (ADR-018) or, at minimum, direct `PlanRenderer` dispatch — with no references to the dissolved legacy modules.  
**Governing ADRs:** ADR-012 (module factoring — services dissolution; dissolved module inventory), ADR-001 (backend abstraction boundary — plan-as-data-structure; `PlanRenderer` as sole backend contract), ADR-007 (KernelContract/KernelBinding split — `kernel_signatures/` dissolution), ADR-008 (PrecisionConfig — deprecated `Float32ModelSpec`/`Float16ModelSpec` factory removal), ADR-014 (build system — feature flags promoted to `enabled`/mandatory), ADR-016 (test strategy — Tier 1/2/3 as sole test infrastructure), ADR-017 (migration path — Phase 6 definition; rollback gate; feature-flag lifecycle), ADR-018 (user-facing API — `Engine`/`WorkTicket` as orchestrator entry point)  
**Rollback gate:** Tier 3 parity green across all enabled backends for the full 22-kernel inventory at both FP32 and supported FP16 configurations. This is the most stringent gate — it requires that every backend produces equivalent results for every kernel at every supported precision before the legacy fallback is removed. Additionally: zero imports of dissolved module names anywhere in `src/` or `tests/`; no shim modules remain; `_build_config.py` flags for all three backends at `enabled`.  
**Dependencies:** All preceding phases must have passed their respective gates: Phase 2 (OpenCL Adapter — Tier 1 + OpenCL Tier 2 green), Phase 3 (CPU Backend — Tier 1 + CPU Tier 2 green), Phase 4 (Test Harness — all enabled tiers green), Phase 5 (Vulkan Backend — Tier 1 + Vulkan Tier 2 + Tier 3 parity green), User-Facing API workstream (Tier 1 ticket + integration green).

### Relationship to Other Phases

Phase 6 is the join point of the entire migration. It consumes the deliverables of every preceding phase and produces the system's final-state directory structure.

| Phase | Phase 6 interaction |
| :--- | :--- |
| Phase 0 (Foundation) | Phase 6 removes the shim modules that Phase 0 introduced for backward compatibility. |
| Phase 1 (Plan Model) | Phase 6 consumes the plan model as the sole execution mechanism. All plan types, contracts, and lifecycle are retained unchanged. |
| Phase 2 (OpenCL Adapter) | Phase 6 removes legacy OpenCL modules (`batch_processor.py`, `graph_recipes.py`, `execution_plan.py`, `launcher_infra.py`, `compute_patterns.py`, `kernel_compilation.py`) from `src/backends/opencl/`. The `OpenCLPlanRenderer` becomes the sole OpenCL dispatch path. |
| Phase 3 (CPU Backend) | Phase 6 does not modify the CPU backend. The CPU backend was built against the plan model from inception — no legacy code exists. |
| Phase 4 (Test Harness) | Phase 6 retires legacy `test_integration_*.py` and `bench_*.py` files from `tests/`. The Tier 1/2/3 framework becomes the sole test infrastructure. |
| Phase 5 (Vulkan Backend) | Phase 6 does not modify the Vulkan backend. The Vulkan backend was built against the plan model from inception — no legacy code exists. |
| User-Facing API | Phase 6 rewrites `main_orchestrator.py` to use the `Engine` / `WorkTicket` API surface, eliminating the legacy `TrainingOrchestrator` class and its direct OpenCL dependencies. |

---

## Table of Contents

1. [Scope & Constraints](#1-scope--constraints)
2. [Legacy Code Inventory](#2-legacy-code-inventory)
3. [Target Deliverables](#3-target-deliverables)
4. [Task Breakdown](#4-task-breakdown)
   - [Step 6.1: Verify all prerequisite phase gates are green](#step-61-verify-all-prerequisite-phase-gates-are-green)
   - [Step 6.2: Remove Phase 0 shim modules from `src/`](#step-62-remove-phase-0-shim-modules-from-src)
   - [Step 6.3: Remove legacy OpenCL modules from `src/backends/opencl/`](#step-63-remove-legacy-opencl-modules-from-srcbackendsopencl)
   - [Step 6.4: Remove `src/kernel_signatures/` shim package](#step-64-remove-srckernel_signatures-shim-package)
   - [Step 6.5: Remove `src/arch_primitives.py`](#step-65-remove-srcarch_primitivespy)
   - [Step 6.6: Remove deprecated `Float32ModelSpec` / `Float16ModelSpec` factory aliases](#step-66-remove-deprecated-float32modelspec--float16modelspec-factory-aliases)
   - [Step 6.7: Rewrite `src/main_orchestrator.py`](#step-67-rewrite-srcmain_orchestratorpy)
   - [Step 6.8: Clean `src/__init__.py`](#step-68-clean-src__init__py)
   - [Step 6.9: Retire legacy test files](#step-69-retire-legacy-test-files)
   - [Step 6.10: Update `tests/conftest.py` — remove legacy machinery](#step-610-update-testsconftestpy--remove-legacy-machinery)
   - [Step 6.11: Promote feature flags to `enabled` (mandatory)](#step-611-promote-feature-flags-to-enabled-mandatory)
   - [Step 6.12: Run full import audit](#step-612-run-full-import-audit)
   - [Step 6.13: Update documentation](#step-613-update-documentation)
   - [Step 6.14: Validate rollback gate](#step-614-validate-rollback-gate)
5. [Dissolved Module Traceability](#5-dissolved-module-traceability)
6. [Import Migration Table](#6-import-migration-table)
7. [Orchestrator Rewrite Specification](#7-orchestrator-rewrite-specification)
8. [Legacy Test Retirement Inventory](#8-legacy-test-retirement-inventory)
9. [Feature Flag Promotion Protocol](#9-feature-flag-promotion-protocol)
10. [Risk Register](#10-risk-register)

---

## 1. Scope & Constraints

### In scope

- Deleting Phase 0 shim modules from `src/`: `arch_primitives.py`, `cl_context_manager.py`, `compute_patterns.py`, `launcher_infra.py`, `execution_plan.py`, `graph_recipes.py`, `batch_processor.py` (ADR-012, ADR-017).
- Deleting the `src/kernel_signatures/` shim package and its re-export `__init__.py` (ADR-007, ADR-012).
- Deleting legacy OpenCL modules from `src/backends/opencl/` that are superseded by the `OpenCLPlanRenderer` and its associated infrastructure: `batch_processor.py`, `graph_recipes.py`, `execution_plan.py`, `launcher_infra.py`, `compute_patterns.py`, `kernel_compilation.py` (ADR-012).
- Removing deprecated `Float32ModelSpec()` / `Float16ModelSpec()` module-level factory functions from `src/shared/model_spec.py`, leaving only `ModelSpec.float32()` / `ModelSpec.float16()` classmethods and the `PrecisionConfig`-based constructor (ADR-008).
- Removing the `PrecisionContext` ABC from `src/arch_primitives.py` and any downstream references (ADR-008 — fully superseded by `PrecisionConfig`).
- Rewriting `src/main_orchestrator.py` to use the plan-model dispatch exclusively — either through the `Engine` / `WorkTicket` API (ADR-018) or through direct `PlanRenderer.render()` invocation. All references to legacy `Services`, `BatchProcessor`, `ExecutionPlan` (the legacy class), `DependencyProvider`, `ComputeOnceProvider`, `StagedComputationProvider`, `KernelExecutor`, `BufferManager`, `graph_recipes`, `ReductionPlan` (the legacy class), `OpenCLContextManager`, and `ComputeEnvironment` are eliminated.
- Retiring legacy test files from `tests/`: all `test_integration_*.py` files (9 files) and `bench_*.py` files (3 files) whose functionality is subsumed by the Tier 1/2/3 framework.
- Updating `tests/conftest.py` to remove the legacy OpenCL probe, the `requires_opencl` marker, the namespace-package `sys.modules` manipulation (if no longer needed), and any fixtures that reference dissolved types.
- Promoting all three backend feature flags (`backend_opencl`, `backend_cpu`, `backend_vulkan`) from `auto` to `enabled` in `meson.options`, establishing them as mandatory components (ADR-014, ADR-017 §Feature-flag lifecycle).
- Running a full import audit to confirm zero references to dissolved module names across `src/` and `tests/`.
- Updating `DESIGN.md`, `README.md`, and `STRUCTURE.md` to reflect the final directory structure.

### Out of scope

- Modifying kernel source files (`kernels/*.cl.c`, `kernels/kernels.cl.h`, `src/backends/*/kernel_sources/`).
- Modifying the shared-layer plan model (`src/shared/`), except removing deprecated factory aliases from `model_spec.py`.
- Modifying backend `PlanRenderer` implementations (the renderers are consumed, not changed).
- Modifying the Tier 1/2/3 test bodies (the test framework is consumed, not changed).
- Performance optimization — Phase 6 is a correctness-preserving deletion phase.
- FP8 or mixed-precision support (ADR-020 is PROPOSED, not yet accepted).
- Adding new features or functionality of any kind.

### Key constraint: deletion-only for source modules

Phase 6 is fundamentally a *deletion* phase. With the exception of `main_orchestrator.py` (which requires a rewrite) and `tests/conftest.py` (which requires cleanup), no existing code is modified — legacy modules are deleted in their entirety. The plan-model infrastructure (shared layer + backend renderers + test harness) is the production system; Phase 6 removes the scaffolding.

### Key constraint: no behavioral change

The system's observable behavior must be identical before and after Phase 6. The rollback gate (Tier 3 parity across all backends at FP32 + FP16) ensures this — the same kernel computations, producing the same numerical results, through the plan-model dispatch path that has been validated by Tiers 1–3 throughout the preceding phases.

### Key constraint: atomicity

Phase 6 should be executed as an atomic change set (single commit or tightly-coupled commit series). Partial deletion — removing some shims but leaving others — creates an inconsistent codebase where some legacy paths work and others are broken. The import audit (Step 6.12) serves as the mechanical verification that no dangling references remain.

### Key constraint: FP16 gate prerequisite

The rollback gate requires "FP32 and supported FP16 configurations." This means:
- FP32: all 22 kernels, all three backends, Tier 3 parity green.
- FP16: all 22 kernels, all backends that support FP16, Tier 3 parity green.

FP16 support availability by backend:
- **OpenCL:** Supported via `cl_khr_fp16` extension (device-dependent). The OpenCL renderer must pass Tier 2 + Tier 3 at FP16 on hardware that supports it.
- **CPU:** FP16 compute requires AVX-512 FP16 or ARM FP16 extensions. If the CI hardware lacks these, CPU FP16 Tier 2 is skipped (not failed) per ADR-016 skip logic.
- **Vulkan:** FP16 requires `VK_KHR_shader_float16_int8` and the `shaderFloat16` device feature. Device-dependent.

The gate is: "for every (backend, precision) combination where the backend's Tier 2 passes at that precision, Tier 3 parity also passes." Backends that do not support a given precision are skipped — their absence does not block the gate.

---

## 2. Legacy Code Inventory

### Phase 0 shim modules at `src/` top level

These are re-export shims created during Phase 0 that forward all public symbols to their canonical locations. Each contains a `# Remove in Phase 6` comment.

| Shim module | Canonical location | Forwards to |
| :--- | :--- | :--- |
| `src/arch_primitives.py` | Dissolved (ADR-008) | `PrecisionContext` ABC — standalone file, not a shim. Superseded by `src/shared/precision_config.py`. |
| `src/cl_context_manager.py` | `src/backends/opencl/context.py` | `from .backends.opencl.context import *` |
| `src/compute_patterns.py` | `src/backends/opencl/compute_patterns.py` | `from .backends.opencl.compute_patterns import *` |
| `src/launcher_infra.py` | `src/backends/opencl/launcher_infra.py` | `from .backends.opencl.launcher_infra import *` |
| `src/execution_plan.py` | `src/backends/opencl/execution_plan.py` + `src/shared/problem_type_strategy.py` | `from .backends.opencl.execution_plan import *` + `from .shared.problem_type_strategy import *` |
| `src/graph_recipes.py` | `src/backends/opencl/graph_recipes.py` | `from .backends.opencl.graph_recipes import *` |
| `src/batch_processor.py` | `src/backends/opencl/batch_processor.py` | `from .backends.opencl.batch_processor import *` |
| `src/kernel_signatures/` | `src/backends/opencl/kernel_bindings/` | `from ..backends.opencl.kernel_bindings import *` |

**Total:** 7 shim files + 1 shim package = 8 deletion targets at `src/` top level.

### Legacy OpenCL modules at `src/backends/opencl/`

These are the *actual* legacy implementations that the Phase 0 shims forward to. They predate the plan-model architecture and are superseded by the `OpenCLPlanRenderer` + `OpenCLKernelBinding` infrastructure built in Phases 2A/2B.

| Legacy module | Role | Superseded by |
| :--- | :--- | :--- |
| `src/backends/opencl/batch_processor.py` | Top-level Act/Learn orchestration with imperative `cl.Event` chaining | `OpenCLPlanRenderer.render()` (Phase 2A) |
| `src/backends/opencl/graph_recipes.py` | ~984 lines of mixed DAG construction + OpenCL dispatch | `PlanBuilder` (Phase 1) + `OpenCLPlanRenderer` (Phase 2A) + `OpenCLKernelBinding`s (Phase 2B) |
| `src/backends/opencl/execution_plan.py` | `DependencyProvider` hierarchy (`CacheProvider`, `ComputeOnceProvider`, `StagedComputationProvider`) embedding `cl.Event` | `ExecutionPlan` (Phase 1, `src/shared/plan_types.py`) — declarative plan DAG replaces executable providers |
| `src/backends/opencl/launcher_infra.py` | `Services`, `BufferManager`, `KernelExecutor`, `BufferHandle` (legacy), `HostView` | `OpenCLBufferAllocator` (Phase 2A), `OpenCLRetrievalFuture` (Phase 2A), shared `BufferHandle` (Phase 1) |
| `src/backends/opencl/compute_patterns.py` | `ReductionPlan` (legacy), `AggregationManager` with imperative `clEnqueueNDRange` | `ReductionTreePlan` (Phase 1, `src/shared/reduction_tree_plan.py`), `OpenCLPlanRenderer` reduction tree rendering (Phase 2A) |
| `src/backends/opencl/kernel_compilation.py` | OpenCL program compilation utilities | Absorbed into `OpenCLContext` (Phase 2A, `src/backends/opencl/context.py`) |

**Total:** 6 legacy modules to delete from `src/backends/opencl/`.

### Legacy test files

| Test file | Type | Subsumed by |
| :--- | :--- | :--- |
| `tests/test_integration_buffer_lifecycle.py` | Host-side integration | `tests/tier1/test_buffer_lifecycle.py` |
| `tests/test_integration_dag_orchestration.py` | Host-side integration | `tests/tier1/test_plan_builder.py` + `tests/tier1/test_plan_types.py` |
| `tests/test_integration_e2e_iris.py` | End-to-end with OpenCL | `tests/tier2/opencl/` + `tests/tier3/test_parity_e2e.py` |
| `tests/test_integration_execution_plan.py` | Host-side integration | `tests/tier1/test_plan_builder.py` |
| `tests/test_integration_model_memory.py` | Host-side integration | `tests/tier1/test_memory_layout.py` |
| `tests/test_integration_precision_chain.py` | Host-side integration | `tests/tier1/test_precision_config.py` |
| `tests/test_integration_scenario_validation.py` | Host-side integration | `tests/tier1/test_plan_builder.py` (scenario-level plan construction) |
| `tests/test_integration_stabilization.py` | Host-side integration | `tests/tier1/test_plan_builder.py` (stabilization policy in plan) |
| `tests/test_integration_workload_tiling.py` | Host-side integration | `tests/tier1/test_memory_layout.py` |
| `tests/bench_host_planning.py` | Benchmark | Retired — no performance gates (ADR-017 Option D) |
| `tests/bench_opencl_backend.py` | Benchmark (requires OpenCL) | Retired — functional coverage in Tier 2 OpenCL; no performance gates |
| `tests/bench_opencl_backend_learn.py` | Benchmark (requires OpenCL) | Retired — functional coverage in Tier 2 OpenCL; no performance gates |

**Total:** 12 legacy test files to delete.

### Deprecated API surface

| Deprecated API | Location | Replacement |
| :--- | :--- | :--- |
| `Float32ModelSpec()` function | `src/shared/model_spec.py` | `ModelSpec.float32()` classmethod |
| `Float16ModelSpec()` function | `src/shared/model_spec.py` | `ModelSpec.float16()` classmethod |
| `PrecisionContext` ABC | `src/arch_primitives.py` | `PrecisionConfig` frozen dataclass (`src/shared/precision_config.py`) |
| `Float32Context` / `Float16Context` classes | `src/arch_primitives.py` | `PrecisionConfig.float32()` / `PrecisionConfig.float16()` |
| `requires_opencl` marker | `tests/conftest.py` | `_build_config`-driven `pytest_collection_modifyitems` skip logic |

---

## 3. Target Deliverables

After Phase 6 completes, the source tree is in its final state:

```
src/
├── __init__.py                         # CLEANED: no shim re-exports
├── main_orchestrator.py                # REWRITTEN: uses Engine/WorkTicket or PlanRenderer exclusively
├── shared/
│   ├── __init__.py
│   ├── model_spec.py                  # MODIFIED: deprecated factory aliases removed
│   ├── parameter_space.py
│   ├── memory_layout.py
│   ├── precision_config.py
│   ├── hardware_profile.py
│   ├── stabilization_policy.py
│   ├── plan_builder.py
│   ├── plan_types.py
│   ├── plan_renderer.py
│   ├── buffer_lifecycle.py
│   ├── retrieval_future.py
│   ├── reduction_tree_plan.py
│   ├── streaming_loop_plan.py
│   ├── problem_type_strategy.py
│   ├── workload_primitives.py
│   ├── work_ticket.py                 # From User-Facing API workstream
│   ├── engine.py                      # From User-Facing API workstream
│   └── kernel_contracts/
│       ├── __init__.py
│       └── ... (per-phase contract modules)
└── backends/
    ├── __init__.py
    ├── opencl/
    │   ├── __init__.py
    │   ├── renderer.py                # Retained — OpenCLPlanRenderer
    │   ├── context.py                 # Retained — OpenCL context lifecycle
    │   ├── discovery.py               # Retained — HardwareProfile population
    │   ├── type_mapping.py            # Retained — PrecisionConfig → OpenCL flags
    │   ├── buffer_allocator.py        # Retained — BufferDescriptor → cl.Buffer
    │   ├── retrieval.py               # Retained — OpenCLRetrievalFuture
    │   ├── meson.build                # Retained
    │   └── kernel_bindings/           # Retained — per-kernel KernelBinding implementations
    │       ├── __init__.py
    │       └── ... (per-phase binding modules)
    ├── cpu/
    │   └── ... (unchanged from Phase 3)
    └── vulkan/
        └── ... (unchanged from Phase 5)
```

**Deleted from `src/` top level:**
- `arch_primitives.py`
- `cl_context_manager.py`
- `compute_patterns.py`
- `launcher_infra.py`
- `execution_plan.py`
- `graph_recipes.py`
- `batch_processor.py`
- `kernel_signatures/` (entire package)

**Deleted from `src/backends/opencl/`:**
- `batch_processor.py`
- `graph_recipes.py`
- `execution_plan.py`
- `launcher_infra.py`
- `compute_patterns.py`
- `kernel_compilation.py`

**Deleted from `tests/`:**
- `test_integration_buffer_lifecycle.py`
- `test_integration_dag_orchestration.py`
- `test_integration_e2e_iris.py`
- `test_integration_execution_plan.py`
- `test_integration_model_memory.py`
- `test_integration_precision_chain.py`
- `test_integration_scenario_validation.py`
- `test_integration_stabilization.py`
- `test_integration_workload_tiling.py`
- `bench_host_planning.py`
- `bench_opencl_backend.py`
- `bench_opencl_backend_learn.py`

**Total deletions:** 8 shim files + 6 legacy OpenCL modules + 12 test files = **26 files/packages removed**.

---

## 4. Task Breakdown

### Step 6.1: Verify all prerequisite phase gates are green

**Action:** Before any deletion, confirm that every preceding phase's rollback gate passes in the current CI environment. This is the precondition for Phase 6 — no code is deleted until the plan-model dispatch path is fully validated.

**Verification checklist:**

| Gate | Command | Required result |
| :--- | :--- | :--- |
| Tier 1 green | `pytest -m tier1` | All pass |
| CPU Tier 2 green | `pytest -m "tier2 and cpu"` | All pass |
| OpenCL Tier 2 green | `pytest -m "tier2 and opencl"` | All pass |
| Vulkan Tier 2 green | `pytest -m "tier2 and vulkan"` | All pass |
| Tier 3 parity green (FP32) | `pytest -m tier3` | All pass |
| Tier 3 parity green (FP16) | `pytest -m "tier3 and fp16"` | All pass (or skipped if no backend supports FP16 on this hardware) |
| User-Facing API integration green | `pytest tests/tier1/test_work_ticket.py tests/tier1/test_engine.py` | All pass |

**Gate:** All checks above pass. If any fail, Phase 6 is blocked until the failing phase is repaired.

**Risk:** If CI hardware lacks FP16 support on all backends, the FP16 Tier 3 gate is skipped. The decision to proceed with Phase 6 under this condition should be documented — the FP16 gate is satisfied on hardware that supports it, and `auto`-skipped otherwise (per ADR-017 §Distinguishing "skipped tier" from "failed tier").

---

### Step 6.2: Remove Phase 0 shim modules from `src/`

**Action:** Delete the seven shim files that were introduced in Phase 0 to maintain backward compatibility during the migration. Each file is a thin re-export wrapper containing `from .backends.opencl.<module> import *` — no logic resides in these shims.

**Files to delete:**

```
src/cl_context_manager.py
src/compute_patterns.py
src/launcher_infra.py
src/execution_plan.py
src/graph_recipes.py
src/batch_processor.py
```

**Verification:** After deletion, run:
```bash
grep -rn "from src\.cl_context_manager\|from src\.compute_patterns\|from src\.launcher_infra\|from src\.execution_plan\|from src\.graph_recipes\|from src\.batch_processor\|from \.cl_context_manager\|from \.compute_patterns\|from \.launcher_infra\|from \.execution_plan\|from \.graph_recipes\|from \.batch_processor" src/ tests/
```
Expected: zero matches. Any match indicates a dangling import that must be updated before proceeding.

**Sequencing note:** This step should precede Step 6.3 (legacy OpenCL module deletion). Removing shims first surfaces any test or module that still imports through the shim path — those must be updated to import from the canonical location (or, if the canonical module is also being deleted, to use the plan-model replacement).

---

### Step 6.3: Remove legacy OpenCL modules from `src/backends/opencl/`

**Action:** Delete the six legacy OpenCL implementation modules that are superseded by the `OpenCLPlanRenderer` infrastructure.

**Files to delete:**

```
src/backends/opencl/batch_processor.py
src/backends/opencl/graph_recipes.py
src/backends/opencl/execution_plan.py
src/backends/opencl/launcher_infra.py
src/backends/opencl/compute_patterns.py
src/backends/opencl/kernel_compilation.py
```

**Pre-deletion check:** Confirm that the `OpenCLPlanRenderer` (`src/backends/opencl/renderer.py`) does not import from any of these modules:

```bash
grep -n "from.*batch_processor\|from.*graph_recipes\|from.*execution_plan\|from.*launcher_infra\|from.*compute_patterns\|from.*kernel_compilation" src/backends/opencl/renderer.py
```
Expected: zero matches. The renderer was built in Phase 2A against the shared-layer plan model — it should have no dependencies on legacy modules.

**Update `src/backends/opencl/__init__.py`:** Remove any re-exports of symbols from deleted modules. The `__init__.py` should export only the plan-model infrastructure: `OpenCLPlanRenderer`, `OpenCLBufferAllocator`, `OpenCLRetrievalFuture`, `OpenCLContext`, `discover_hardware_profile`, `OpenCLTypeMapping`, and the `kernel_bindings` sub-package.

**Verification:** Run `pytest -m "tier2 and opencl"` — all OpenCL Tier 2 tests must remain green. These tests exercise the `OpenCLPlanRenderer` path, not the legacy path.

---

### Step 6.4: Remove `src/kernel_signatures/` shim package

**Action:** Delete the `src/kernel_signatures/` directory entirely. This shim package re-exports symbols from `src/backends/opencl/kernel_bindings/` via its `__init__.py`.

**Files to delete:**

```
src/kernel_signatures/__init__.py
src/kernel_signatures/tests/       (if present — the legacy sub-package test directory)
src/kernel_signatures/              (the directory itself)
```

**Verification:**
```bash
grep -rn "from src\.kernel_signatures\|from \.kernel_signatures\|import kernel_signatures" src/ tests/
```
Expected: zero matches.

---

### Step 6.5: Remove `src/arch_primitives.py`

**Action:** Delete the `PrecisionContext` ABC and its concrete subclasses (`Float32Context`, `Float16Context`). These are fully superseded by `PrecisionConfig` (ADR-008, `src/shared/precision_config.py`).

**File to delete:**

```
src/arch_primitives.py
```

**Verification:**
```bash
grep -rn "PrecisionContext\|Float32Context\|Float16Context\|from.*arch_primitives\|import arch_primitives" src/ tests/
```
Expected: zero matches in production code. If matches exist in legacy test files, those files should already be scheduled for deletion in Step 6.9.

**Note:** The `main_orchestrator.py` currently imports `Float32Context` and `Float16Context` from `src/backends/opencl/context.py` (not from `arch_primitives.py`). These context classes in the OpenCL backend may still reference `PrecisionContext`. If `src/backends/opencl/context.py` inherits from `PrecisionContext`, that inheritance must be removed and replaced with `PrecisionConfig` consumption before `arch_primitives.py` can be deleted. This refactoring is part of the Phase 2A context modernization — it should already be complete by the time Phase 6 begins.

---

### Step 6.6: Remove deprecated `Float32ModelSpec` / `Float16ModelSpec` factory aliases

**Action:** Remove the deprecated module-level factory functions from `src/shared/model_spec.py`. Users should use `ModelSpec.float32()` or `ModelSpec.float16()` classmethods, or construct `ModelSpec` directly with a `PrecisionConfig`.

**Change:** In `src/shared/model_spec.py`, delete the `Float32ModelSpec` and `Float16ModelSpec` function definitions and remove them from `__all__` (if present).

**Impact:** Any code that calls `Float32ModelSpec(...)` must be updated to `ModelSpec.float32(...)`. Check:
```bash
grep -rn "Float32ModelSpec\|Float16ModelSpec" src/ tests/
```

**Expected matches:** `tests/conftest.py` fixture definitions (updated in Step 6.10). If Tier 1/2/3 test files reference `Float32ModelSpec`, they must be updated to use `ModelSpec.float32()`.

**Note:** This is the one modification to shared-layer code in Phase 6. It is a pure API surface reduction — the classmethod produces the same `ModelSpec` instance.

---

### Step 6.7: Rewrite `src/main_orchestrator.py`

**Action:** Replace the current `TrainingOrchestrator` class — which directly constructs legacy `ExecutionPlan` objects, creates `Services` bundles, instantiates `BatchProcessor`, and manages OpenCL buffer allocation via `cl.enqueue_copy` — with a new orchestrator that uses the plan-model dispatch path exclusively.

**Current architecture (to be removed):**
```
main_orchestrator.py
  → imports: Services, BufferManager, KernelExecutor, BatchProcessor,
             ExecutionPlan, DependencyProvider, CacheProvider,
             ComputeOnceProvider, StagedComputationProvider,
             OpenCLContextManager, ComputeEnvironment, ReductionPlan,
             graph_recipes
  → TrainingOrchestrator.__init__: creates Services(q, ex, bm), calls cl.enqueue_copy
  → TrainingOrchestrator._create_execution_plan: builds legacy ExecutionPlan with providers
  → TrainingOrchestrator.train: creates BatchProcessor, calls processor.run()
```

**Target architecture:**

Two options, depending on whether the User-Facing API workstream (ADR-018) is complete:

#### Option A: Engine/WorkTicket API (preferred, if ADR-018 workstream is complete)

```python
from .shared.engine import Engine
from .shared.model_spec import ModelSpec
from .shared.parameter_space import ParameterSpace
from .shared.precision_config import PrecisionConfig
from .shared.stabilization_policy import StabilizationPolicy

# Backend selection via discovery
from .backends.opencl import create_renderer  # or cpu, vulkan

engine = Engine(renderer=create_renderer(), model_spec=spec, ...)
for epoch in range(epochs):
    ticket = engine.submit(X_train, y_train)
    learn_handle = ticket.learn_handle
    learn_handle.wait()
    probs = ticket.get_probabilities()
```

#### Option B: Direct PlanRenderer (fallback, if ADR-018 is not yet complete)

```python
from .shared.plan_builder import PlanBuilder
from .shared.plan_types import ExecutionPlan
from .backends.opencl.renderer import OpenCLPlanRenderer
from .backends.opencl.context import OpenCLContext
from .backends.opencl.discovery import discover_hardware_profile

context = OpenCLContext(kernel_source_dir=...)
hw_profile = discover_hardware_profile(context)
renderer = OpenCLPlanRenderer(context=context, hw_profile=hw_profile)

builder = PlanBuilder(model_spec=spec, hw_profile=hw_profile, ...)
plan = builder.build_training_batch(batch_size=batch_size, ...)
futures = renderer.render(plan)
futures["final_batch_event"].wait()
probs = futures["class_probabilities"].get()
```

**Key requirements for the rewrite:**
1. Zero imports from dissolved modules.
2. No direct `pyopencl` imports in the orchestrator — all OpenCL interaction is mediated through the backend's public API (`OpenCLContext`, `OpenCLPlanRenderer`, `discover_hardware_profile`).
3. The orchestrator is backend-agnostic — it should accept any `PlanRenderer` implementation. Backend selection is a configuration concern, not an orchestrator concern.
4. Buffer initialization (Xavier weight init, zero-init of optimizer state) is handled through the renderer's buffer upload mechanism, not raw `cl.enqueue_copy`.
5. The `if __name__ == "__main__"` block demonstrates the canonical usage pattern for the final architecture.

**Verification:** After rewrite, run:
```bash
grep -n "pyopencl\|cl\.enqueue_copy\|Services\|BatchProcessor\|KernelExecutor\|BufferManager\|DependencyProvider\|CacheProvider\|ComputeOnceProvider\|StagedComputationProvider\|graph_recipes\|ReductionPlan" src/main_orchestrator.py
```
Expected: zero matches.

---

### Step 6.8: Clean `src/__init__.py`

**Action:** Review and clean `src/__init__.py`. After shim deletion, this file should contain only clean re-exports of the public API surface. It must not import from any dissolved module.

**Target contents:** The `__init__.py` should export the user-facing API:
- From `shared`: `ModelSpec`, `PrecisionConfig`, `ParameterSpace`, `StabilizationPolicy`, `HardwareProfile`, `PlanBuilder`, `ExecutionPlan` (the shared-layer frozen dataclass).
- From `shared`: `Engine`, `WorkTicket`, `LearnHandle` (if ADR-018 is complete).
- Backend selection helpers (if defined).

**Verification:**
```bash
python -c "import src; print(dir(src))"
```
Must not raise `ImportError` for dissolved modules.

---

### Step 6.9: Retire legacy test files

**Action:** Delete all legacy test files whose coverage is subsumed by the Tier 1/2/3 framework.

**Files to delete:**

```
tests/test_integration_buffer_lifecycle.py
tests/test_integration_dag_orchestration.py
tests/test_integration_e2e_iris.py
tests/test_integration_execution_plan.py
tests/test_integration_model_memory.py
tests/test_integration_precision_chain.py
tests/test_integration_scenario_validation.py
tests/test_integration_stabilization.py
tests/test_integration_workload_tiling.py
tests/bench_host_planning.py
tests/bench_opencl_backend.py
tests/bench_opencl_backend_learn.py
```

**Pre-deletion audit:** For each legacy test file, verify that its test coverage is present in the Tier framework:

| Legacy test | Coverage in Tier framework |
| :--- | :--- |
| `test_integration_buffer_lifecycle.py` | `tier1/test_buffer_lifecycle.py` — `BufferDescriptor` construction, lifetime validation, role assignment |
| `test_integration_dag_orchestration.py` | `tier1/test_plan_builder.py` — DAG construction, topological ordering, dependency edges |
| `test_integration_e2e_iris.py` | `tier2/opencl/` (per-kernel) + `tier3/test_parity_e2e.py` (cross-backend end-to-end) |
| `test_integration_execution_plan.py` | `tier1/test_plan_builder.py` + `tier1/test_plan_types.py` — plan construction, node typing, validation |
| `test_integration_model_memory.py` | `tier1/test_memory_layout.py` — memory layout calculation, padding, alignment |
| `test_integration_precision_chain.py` | `tier1/test_precision_config.py` — `PrecisionConfig` factory, tolerance derivation |
| `test_integration_scenario_validation.py` | `tier1/test_plan_builder.py` — scenario-level plan construction validation |
| `test_integration_stabilization.py` | `tier1/test_plan_builder.py` — stabilization policy integration in plan construction |
| `test_integration_workload_tiling.py` | `tier1/test_memory_layout.py` — tiling scheme derivation |
| `bench_host_planning.py` | No replacement — performance benchmarks retired per ADR-017 Option D |
| `bench_opencl_backend.py` | No replacement — functional coverage in `tier2/opencl/` |
| `bench_opencl_backend_learn.py` | No replacement — functional coverage in `tier2/opencl/` |

**Sequencing:** Delete legacy test files *after* Steps 6.2–6.5 (shim and module deletion). The legacy tests import dissolved modules via shim paths — they will fail after shim deletion. Deleting them after confirms that no surviving test depends on legacy infrastructure.

---

### Step 6.10: Update `tests/conftest.py` — remove legacy machinery

**Action:** Modernize the top-level conftest to remove all legacy infrastructure. The Phase 4 test harness (already complete) provides the production skip logic; the legacy machinery is redundant.

**Changes:**

1. **Remove the runtime OpenCL probe block.** The `try: import pyopencl as cl` / `_has_opencl` / `_has_opencl_device` block is replaced by `_build_config`-driven detection (already present via `_load_build_config()`). Remove the legacy probe code and the `_has_opencl` / `_has_opencl_device` variables.

2. **Remove the `requires_opencl` marker.** The `requires_opencl = pytest.mark.skipif(...)` line is superseded by the `pytest_collection_modifyitems` hook that reads `BUILD_CONFIG["opencl"]`. All tests that used `@requires_opencl` should already be using the `@pytest.mark.opencl` marker (Phase 4 deliverable).

3. **Remove the `sys.modules["src"]` namespace-package manipulation** if it is no longer needed. Evaluate whether the `src` package's `__init__.py` can now be imported cleanly without triggering heavy `pyopencl` side effects. If the cleaned `__init__.py` (Step 6.8) is lightweight, the manipulation is unnecessary.

4. **Update fixture imports.** Replace `Float32ModelSpec` / `Float16ModelSpec` references with `ModelSpec.float32()` / `ModelSpec.float16()` in the fixture definitions (`fp32_iris_spec`, `fp16_iris_spec`, `fp32_hydra_spec`, `fp32_lexicon_spec`).

5. **Remove the `_load_build_config` fallback path.** The `except ImportError` branch that falls back to `_has_opencl_device` is a migration-era safety net. After Phase 6, `_build_config.py` is guaranteed to exist (backends are mandatory). The fallback can be removed, and a missing `_build_config.py` should raise a clear error.

**Verification:** Run the full test suite (`pytest`) — all Tier 1/2/3 tests remain green; no warnings about undefined markers or missing fixtures.

---

### Step 6.11: Promote feature flags to `enabled` (mandatory)

**Action:** Update `meson.options` to change all three backend feature options from `auto` to `enabled`. This is the operational definition of "Phase 6 complete" per ADR-017 §Feature-flag lifecycle: the backends transition from "available when detected" to "required — build fails if absent."

**Change in `meson.options`:**

```meson
# Before (development/validated stage):
option('aec_backend_cpu', type: 'feature', value: 'auto', description: '...')
option('aec_backend_vulkan', type: 'feature', value: 'auto', description: '...')
option('aec_backend_opencl', type: 'feature', value: 'auto', description: '...')

# After (mandatory stage):
option('aec_backend_cpu', type: 'feature', value: 'enabled', description: '...')
option('aec_backend_vulkan', type: 'feature', value: 'enabled', description: '...')
option('aec_backend_opencl', type: 'feature', value: 'enabled', description: '...')
```

**Impact:** After this change:
- The Meson build **fails** if the C compiler toolchain is absent (CPU backend).
- The Meson build **fails** if `glslc` is absent (Vulkan backend).
- The Meson build **fails** if PyOpenCL / an OpenCL-capable device is absent (OpenCL backend).
- `_build_config.py` always declares `BACKEND_CPU = True`, `BACKEND_VULKAN = True`, `BACKEND_OPENCL = True`.
- No test tier is ever skipped due to backend absence — all tiers are always exercised.

**CI implication:** CI environments must have all three toolchains installed. This is a higher infrastructure bar than the pre-Phase 6 state. Document the required CI packages:
- **CPU:** C compiler with C11 support (gcc ≥ 4.9, clang ≥ 3.1). Present on all CI runners.
- **Vulkan:** Vulkan SDK with `glslc` (via `vulkan-sdk` package or `shaderc` standalone).
- **OpenCL:** OpenCL ICD loader + at least one ICD (e.g., `pocl` for CPU-based OpenCL, or a GPU driver).

**Verification:** After `meson setup`, confirm:
```bash
python -c "from src._build_config import BACKEND_CPU, BACKEND_VULKAN, BACKEND_OPENCL; print(BACKEND_CPU, BACKEND_VULKAN, BACKEND_OPENCL)"
# Expected: True True True
```

---

### Step 6.12: Run full import audit

**Action:** Mechanically verify that zero references to dissolved modules remain anywhere in the codebase. This is the structural safety net for the entire phase.

**Audit commands:**

```bash
# 1. Dissolved module names as import targets
grep -rn \
  "from.*arch_primitives\|from.*cl_context_manager\|from.*compute_patterns\|from.*launcher_infra\|from.*execution_plan\|from.*graph_recipes\|from.*batch_processor\|from.*kernel_signatures\|from.*kernel_compilation" \
  src/ tests/ \
  --include="*.py"

# 2. Dissolved class/type names
grep -rn \
  "PrecisionContext\|Float32Context\|Float16Context\|DependencyProvider\|CacheProvider\|ComputeOnceProvider\|StagedComputationProvider\|AggregationManager\|KernelExecutor\b\|class Services\|HostView\b\|KernelSignature\b" \
  src/ tests/ \
  --include="*.py"

# 3. Legacy factory function names (after Step 6.6)
grep -rn "Float32ModelSpec\|Float16ModelSpec" src/ tests/ --include="*.py"

# 4. Direct pyopencl imports outside the OpenCL backend
grep -rn "import pyopencl\|from pyopencl" src/ --include="*.py" \
  | grep -v "src/backends/opencl/"
```

**Expected:** All four commands produce zero output. Any match is a bug that must be fixed before the rollback gate is evaluated.

---

### Step 6.13: Update documentation

**Action:** Update project documentation to reflect the final directory structure and API surface.

**Files to update:**

| Document | Changes |
| :--- | :--- |
| `DESIGN.md` §11.2 | Update Phase 6 status from "Not started" to "✅ Complete". Update dependency graph. |
| `DESIGN.md` §11.4 | Update feature-flag lifecycle table — all backends at "Mandatory" stage. |
| `README.md` | Update installation and usage instructions to reflect the `Engine` / `WorkTicket` API. Remove references to legacy `TrainingOrchestrator` or direct OpenCL usage. |
| `STRUCTURE.md` | Update the directory tree to reflect the final state (no shims, no legacy modules). |
| `CPU_BACKEND.md` | No changes expected — CPU backend documentation was written for the plan-model architecture. |
| `VULKAN_BACKEND.md` | No changes expected — Vulkan backend documentation was written for the plan-model architecture. |

---

### Step 6.14: Validate rollback gate

**Action:** Run the complete Tier 1/2/3 suite and confirm the Phase 6 rollback gate passes.

**Gate definition (ADR-017):** Tier 3 parity green across all enabled backends for the full 22-kernel inventory at both FP32 and supported FP16 configurations.

**Concrete test commands:**

```bash
# Full suite — captures all tiers
pytest 2>&1 | tee /tmp/phase6_gate.txt

# Verify specific gate components
grep -E "^(PASSED|FAILED|ERROR|tests/)" /tmp/phase6_gate.txt

# Tier breakdown
pytest -m tier1 --tb=no -q
pytest -m "tier2 and cpu" --tb=no -q
pytest -m "tier2 and opencl" --tb=no -q
pytest -m "tier2 and vulkan" --tb=no -q
pytest -m tier3 --tb=no -q
```

**Gate passes when:**
1. All Tier 1 tests pass (no regressions to plan model).
2. All CPU Tier 2 tests pass (FP32; FP16 if hardware supports it).
3. All OpenCL Tier 2 tests pass (FP32; FP16 if device supports `cl_khr_fp16`).
4. All Vulkan Tier 2 tests pass (FP32; FP16 if device supports `shaderFloat16`).
5. All Tier 3 parity tests pass for every (backend-pair, kernel, precision) combination where both backends in the pair have passing Tier 2 results.
6. Import audit (Step 6.12) produces zero matches.
7. No dissolved module names appear anywhere in `src/` or `tests/`.

**If the gate fails:** Do not proceed. Diagnose the failure using tier test output. The most likely failure modes are:
- A dangling import from a deleted module (fix: update the import).
- A Tier 3 parity regression exposed by removing the legacy code path (investigate: this indicates the plan-model dispatch path diverges from the legacy path, which should have been caught in preceding phases).
- A fixture reference to a deprecated factory function (fix: update to classmethod syntax).

---

## 5. Dissolved Module Traceability

Complete traceability from each dissolved module to its architectural replacements:

| Dissolved Module | ADR | Shared-Layer Replacement | Backend-Layer Replacement |
| :--- | :--- | :--- | :--- |
| `arch_primitives.py` | ADR-008 | `shared/precision_config.py` — `PrecisionConfig` frozen dataclass | — |
| `cl_context_manager.py` | ADR-006, ADR-008 | `shared/hardware_profile.py` — `HardwareProfile` frozen dataclass | `backends/opencl/context.py` — OpenCL context lifecycle |
| `compute_patterns.py` | ADR-003 | `shared/reduction_tree_plan.py` — `ReductionTreePlan` frozen dataclass | `backends/opencl/renderer.py` — reduction tree rendering |
| `launcher_infra.py` | ADR-009, ADR-010 | `shared/buffer_lifecycle.py` — `BufferHandle`, `BufferRole`, `BufferDescriptor` | `backends/opencl/buffer_allocator.py` + `backends/opencl/retrieval.py` |
| `execution_plan.py` | ADR-002, ADR-011 | `shared/plan_types.py` — `ExecutionPlan`, node types | — (declarative plan replaces executable providers) |
| `graph_recipes.py` | ADR-001 | `shared/plan_builder.py` — DAG construction | `backends/opencl/renderer.py` + `kernel_bindings/` — dispatch |
| `batch_processor.py` | ADR-001 | (absorbed into `Engine.submit()` / plan builder) | `backends/opencl/renderer.py` — `PlanRenderer.render()` |
| `kernel_signatures/` | ADR-007 | `shared/kernel_contracts/` — `KernelContract` hierarchy | `backends/opencl/kernel_bindings/` — `KernelBinding` implementations |
| `kernel_compilation.py` | ADR-014 | — | `backends/opencl/context.py` — program compilation absorbed into context |

---

## 6. Import Migration Table

For any remaining consumer that imported from a dissolved module, this table provides the replacement import:

| Legacy import | Replacement import |
| :--- | :--- |
| `from src.arch_primitives import PrecisionContext` | Remove — use `PrecisionConfig` |
| `from src.arch_primitives import Float32Context` | `from src.shared.precision_config import PrecisionConfig; PrecisionConfig.float32()` |
| `from src.arch_primitives import Float16Context` | `from src.shared.precision_config import PrecisionConfig; PrecisionConfig.float16()` |
| `from src.cl_context_manager import OpenCLContextManager` | `from src.backends.opencl.context import OpenCLContext` |
| `from src.cl_context_manager import ComputeEnvironment` | Remove — `HardwareProfile` + `PrecisionConfig` replace `ComputeEnvironment` |
| `from src.launcher_infra import Services` | Remove — no replacement; backends own their service aggregation |
| `from src.launcher_infra import BufferManager` | `from src.backends.opencl.buffer_allocator import OpenCLBufferAllocator` |
| `from src.launcher_infra import KernelExecutor` | Remove — absorbed into `OpenCLPlanRenderer` dispatch |
| `from src.launcher_infra import BufferHandle` | `from src.shared.buffer_lifecycle import BufferHandle` |
| `from src.launcher_infra import HostView` | `from src.backends.opencl.retrieval import OpenCLRetrievalFuture` |
| `from src.execution_plan import ExecutionPlan` | `from src.shared.plan_types import ExecutionPlan` |
| `from src.execution_plan import DependencyProvider` | Remove — declarative plan nodes replace providers |
| `from src.execution_plan import CacheProvider` | Remove |
| `from src.execution_plan import ComputeOnceProvider` | Remove |
| `from src.execution_plan import StagedComputationProvider` | Remove |
| `from src.compute_patterns import ReductionPlan` | `from src.shared.reduction_tree_plan import ReductionTreePlan` |
| `from src.graph_recipes import ...` | Remove — logic split between `PlanBuilder` and backend renderers |
| `from src.batch_processor import BatchProcessor` | Remove — use `Engine.submit()` or `PlanRenderer.render()` |
| `from src.kernel_signatures import ForwardPassSignature` | `from src.backends.opencl.kernel_bindings import ForwardPassBinding` |
| `from src.shared.model_spec import Float32ModelSpec` | `from src.shared.model_spec import ModelSpec; ModelSpec.float32(...)` |
| `from src.shared.model_spec import Float16ModelSpec` | `from src.shared.model_spec import ModelSpec; ModelSpec.float16(...)` |

---

## 7. Orchestrator Rewrite Specification

The `main_orchestrator.py` rewrite is the most complex task in Phase 6. It transforms a 400+ line OpenCL-coupled orchestrator into a backend-agnostic orchestrator that uses the plan-model dispatch path exclusively.

### Current responsibilities (to be preserved)

| Responsibility | Current implementation | Target implementation |
| :--- | :--- | :--- |
| Precision-aware model assembly | `Float32ModelSpec` / `Float16ModelSpec` construction | `ModelSpec.float32()` / `ModelSpec.float16()` |
| Compute environment setup | `OpenCLContextManager.build_and_discover()` | Backend-specific context factory + `discover_hardware_profile()` |
| Buffer allocation + initialization | `BufferManager.create_named_buffer()` + `cl.enqueue_copy()` | `PlanRenderer`-mediated allocation (buffers are `BufferDescriptor`s in the plan; physical allocation is the renderer's concern) or `Engine`-mediated upload |
| Execution plan construction | Legacy `ExecutionPlan` with `DependencyProvider` hierarchy | `PlanBuilder.build_training_batch()` producing shared-layer `ExecutionPlan` |
| Per-batch execution | `BatchProcessor(services, param_space, plan).run()` | `renderer.render(plan)` returning `dict[str, RetrievalFuture]`, or `engine.submit(data)` returning `WorkTicket` |
| Training loop | `for epoch in range(epochs): plan → processor → wait` | `for epoch in range(epochs): plan → render → wait` or `for epoch: ticket = engine.submit(...)` |
| Weight initialization | Xavier uniform via `np.random + cl.enqueue_copy` | Same Xavier logic, but upload via renderer's buffer upload mechanism or Engine's state initialization |
| Entry point | `if __name__ == "__main__"` with hardcoded Iris dataset | Same, modernized to use the final API |

### Architecture of the rewritten orchestrator

```python
# main_orchestrator.py — Final Architecture (Phase 6)

"""
The System's Strategist: backend-agnostic training orchestration.

Translates an experimental configuration into an immutable ExecutionPlan
and delegates execution to a PlanRenderer. All backend interaction is
mediated through the PlanRenderer Protocol — the orchestrator never
imports backend-specific modules directly (except for backend selection
in the entry point).
"""

from dataclasses import dataclass
from .shared.model_spec import ModelSpec
from .shared.parameter_space import ParameterSpace
from .shared.stabilization_policy import StabilizationPolicy
from .shared.plan_builder import PlanBuilder
from .shared.plan_renderer import PlanRenderer
from .shared.precision_config import PrecisionConfig
from .shared.hardware_profile import HardwareProfile
from .shared.problem_type_strategy import CceStrategy, BceStrategy
# ... (no pyopencl, no dissolved modules)
```

### Buffer initialization strategy

The current orchestrator performs buffer initialization with raw `cl.enqueue_copy`:
- Zero-init all buffers.
- Xavier uniform init for `shared_weights` and `module_weights`.
- Ones-init for `sample_mask` and `temperatures`.

In the plan-model architecture, buffers are declared as `BufferDescriptor`s with lifecycle metadata. Physical allocation is the renderer's concern. Initialization data must be provided through one of:

1. **`PlanRenderer` upload method** (if defined): `renderer.upload_buffer("shared_weights", xavier_init_array)`.
2. **`Engine` state initialization** (ADR-018): `engine.initialize_model_state(weights=..., biases=..., ...)`.
3. **Initial-value annotation on `BufferDescriptor`** (if supported by the plan model): the plan builder emits descriptors with `initial_value` fields, and the renderer populates them during allocation.

The specific mechanism depends on what Phases 2A and the User-Facing API workstream have defined. The orchestrator rewrite must use whatever mechanism is available. The requirement is: no raw `cl.enqueue_copy` or backend-specific buffer manipulation in the orchestrator.

---

## 8. Legacy Test Retirement Inventory

Detailed analysis of each legacy test file's coverage overlap with the Tier framework:

### `test_integration_buffer_lifecycle.py`

**Legacy coverage:** Tests `BufferManager.create_named_buffer()`, `acquire_transient_buffer()`, `release_transient_buffer()`, `get_cl_buffer()`, and handle resolution. Uses legacy `Services` pattern with live OpenCL context.

**Tier coverage:** `tests/tier1/test_buffer_lifecycle.py` tests `BufferDescriptor` construction, `BufferRole` assignment, `BufferHandle` uniqueness, lifetime annotations (`producing_node`, `last_consumer`, `consumers`), and validation invariants. This is host-side validation — no device required. Physical allocation correctness is covered by each backend's Tier 2.

**Safe to retire:** Yes — the legacy test's physical allocation testing is covered by Tier 2; the logical lifecycle testing is covered by Tier 1.

### `test_integration_dag_orchestration.py`

**Legacy coverage:** Tests `graph_recipes.py` recipe composition — verifies that recipe functions produce correct `cl.Event` dependency chains for reduction trees and streaming loops.

**Tier coverage:** `tests/tier1/test_plan_builder.py` tests DAG construction, topological ordering, dependency edge correctness, and node type validation. Tier 2 (per backend) validates that the renderer correctly traverses the DAG. Tier 3 validates cross-backend parity of the complete DAG execution.

**Safe to retire:** Yes — DAG construction is Tier 1; DAG execution correctness is Tier 2/3.

### `test_integration_e2e_iris.py`

**Legacy coverage:** End-to-end training loop with live OpenCL on Iris dataset. Verifies convergence (loss decreasing, accuracy increasing).

**Tier coverage:** `tests/tier3/test_parity_e2e.py` performs end-to-end cross-backend comparison. Per-kernel correctness covered by Tier 2. Convergence testing is not a Tier 1/2/3 concern — it is a model-level property, not a kernel-level property.

**Safe to retire:** Yes, with caveat — if convergence testing is desired, it should be added as a separate integration test in the Tier framework (non-blocking for Phase 6).

### `test_integration_execution_plan.py`

**Legacy coverage:** Tests legacy `ExecutionPlan` construction with `DataLifecyclePolicy`, `DependencyProvider` instances, and plan verification.

**Tier coverage:** `tests/tier1/test_plan_builder.py` + `tests/tier1/test_plan_types.py` — plan construction, node typing, topological ordering, validation.

**Safe to retire:** Yes — the legacy `ExecutionPlan` class is itself being removed.

### `test_integration_model_memory.py`

**Legacy coverage:** Tests `ParameterSpace.get_all_memory_layouts()` and buffer shape/dtype consistency.

**Tier coverage:** `tests/tier1/test_memory_layout.py` — memory layout calculation, SIMD padding, cache-line alignment, buffer shape derivation.

**Safe to retire:** Yes.

### `test_integration_precision_chain.py`

**Legacy coverage:** Tests the `PrecisionContext` → `ModelSpec` → `ParameterSpace` chain for type consistency at FP32 and FP16.

**Tier coverage:** `tests/tier1/test_precision_config.py` — `PrecisionConfig` factory methods, tolerance derivation, dtype propagation.

**Safe to retire:** Yes — the `PrecisionContext` chain it tests is itself being removed.

### `test_integration_scenario_validation.py`

**Legacy coverage:** Tests that various scenario configurations (CCE/BCE, CACHE/RECOMPUTE, various model sizes) produce valid execution plans.

**Tier coverage:** `tests/tier1/test_plan_builder.py` — parameterized across scenario configurations including strategy variants and model sizes.

**Safe to retire:** Yes.

### `test_integration_stabilization.py`

**Legacy coverage:** Tests `StabilizationPolicy` threshold computation and integration into the execution plan.

**Tier coverage:** `tests/tier1/test_plan_builder.py` includes stabilization policy validation in plan construction tests. Threshold arithmetic is pure math tested in the stabilization policy's own unit tests.

**Safe to retire:** Yes.

### `test_integration_workload_tiling.py`

**Legacy coverage:** Tests `TilingScheme` derivation from model dimensions.

**Tier coverage:** `tests/tier1/test_memory_layout.py` — tiling scheme is an input to memory layout; its derivation is tested as part of the layout computation chain.

**Safe to retire:** Yes.

---

## 9. Feature Flag Promotion Protocol

The transition from `auto` to `enabled` is the operational definition of Phase 6 completion per ADR-017 §Feature-flag lifecycle. This section details the protocol.

### Pre-promotion checklist

For each backend, before changing its flag from `auto` to `enabled`:

1. **Tier 2 gate passes consistently** — the backend's Tier 2 tests have passed in CI for at least N consecutive runs (suggested: 5) to exclude flaky passes.
2. **Tier 3 gate passes** — cross-backend parity with this backend included has passed consistently.
3. **CI environment has the toolchain** — all CI runners have the backend's required toolchain installed. A flag set to `enabled` causes a *build failure* on runners without the toolchain — not a test skip.
4. **Rollback path documented** — the one-line change to revert from `enabled` to `auto` is prepared and ready to execute if a regression surfaces post-promotion.

### Promotion order

The three backends can be promoted in any order. Suggested order based on maturity:

1. **CPU** — deterministic, no special hardware required, CI-universal. Promote first.
2. **OpenCL** — requires an OpenCL ICD but `pocl` provides CPU-based OpenCL on all CI runners. Promote second.
3. **Vulkan** — requires `glslc` and a Vulkan ICD. Promote last, after confirming CI has the Vulkan SDK.

### Post-promotion verification

After all three flags are set to `enabled`:
```bash
meson setup builddir --wipe
ninja -C builddir
python -c "from src._build_config import BACKEND_CPU, BACKEND_VULKAN, BACKEND_OPENCL; assert all([BACKEND_CPU, BACKEND_VULKAN, BACKEND_OPENCL])"
pytest 2>&1 | tee /tmp/phase6_promoted.txt
grep -c "PASSED\|FAILED\|ERROR" /tmp/phase6_promoted.txt
```

---

## 10. Risk Register

| # | Risk | Likelihood | Impact | Mitigation |
| :--- | :--- | :--- | :--- | :--- |
| R1 | Dangling import from deleted module causes `ImportError` at test collection | Medium | Low | Step 6.12 import audit catches all dangling references before gate evaluation. Run audit before running tests. |
| R2 | Legacy test covers a scenario not present in Tier framework | Low | Medium | Step 6.9 pre-deletion audit maps each legacy test to its Tier equivalent. Any gap is filled before deletion. |
| R3 | `main_orchestrator.py` rewrite introduces a behavioral regression | Medium | High | The orchestrator rewrite (Step 6.7) is validated by Tier 3 end-to-end parity tests. The rewrite produces the same numerical results because it uses the same plan model and renderer. |
| R4 | FP16 Tier 3 gate cannot be evaluated due to hardware limitations | Medium | Medium | ADR-017 §"Distinguishing skipped tier from failed tier" — FP16 tests are skipped (not failed) on hardware that lacks FP16 support. The gate is satisfied for FP32; FP16 is conditionally satisfied. Document the hardware limitation. |
| R5 | OpenCL backend retains hidden dependency on deleted module | Low | High | Step 6.3 pre-deletion grep confirms `renderer.py` has no imports from legacy modules. Step 6.12 catches any transitive dependency. |
| R6 | `src/__init__.py` breaks after shim removal | Medium | Low | Step 6.8 explicitly addresses `__init__.py` cleanup. Verify with `python -c "import src"`. |
| R7 | CI environment lacks Vulkan SDK after flag promotion | Medium | High | Step 6.11 documents CI requirements. Confirm Vulkan SDK availability on all runners before promotion. If not available, keep `backend_vulkan` at `auto` and document the exception. |
| R8 | `Float32ModelSpec`/`Float16ModelSpec` removal breaks downstream code outside this repository | Low | Medium | These are internal factory functions within the architecture package. No external consumers exist. The `ModelSpec.float32()` / `ModelSpec.float16()` classmethods were available since Phase 1. |
| R9 | User-Facing API workstream (ADR-018) is incomplete at Phase 6 start | Medium | Medium | Step 6.7 provides Option B (direct `PlanRenderer` usage) as a fallback. The orchestrator rewrite does not *require* the `Engine`/`WorkTicket` API — it can use `PlanBuilder` + `PlanRenderer.render()` directly. Option A is preferred but not blocking. |
| R10 | Phase 6 atomicity violated — partial deletion committed | Low | High | Execute all deletion steps in a single branch. Run the full gate (Step 6.14) before merging. Do not merge partial deletion states. |
