# Phase 0: Foundation — Detailed Plan

**Status: ✅ IMPLEMENTED** — All 192 tests pass. Rollback gate validated.  
**Phase:** 0 of 6  
**Objective:** Create the target directory structure and relocate existing modules without behavioral change. No new functionality is introduced.  
**Governing ADRs:** ADR-012 (module factoring), ADR-014 (build system), ADR-006 (HardwareProfile extraction), ADR-007 (KernelContract extraction), ADR-008 (PrecisionConfig extraction), ADR-011 (ProblemTypeStrategy extraction), ADR-013 (kernel source designation)  
**Rollback gate:** All existing tests green. Every test that passed before Phase 0 must pass after.  
**Dependencies:** None — Phase 0 is the starting point.

### Implementation Notes

- **Option name prefix:** Build options use an `aec_` prefix (`aec_backend_vulkan`, `aec_backend_cpu`, `aec_cpu_isa_flags`) to namespace them within the Meson subproject. DESIGN.md §8.2 reflects this.
- **Shim adapters:** Thin re-export shims placed at all original import paths (`src/cl_context_manager.py`, `src/compute_patterns.py`, etc.) to preserve backward compatibility. Removed in Phase 6.
- **`arch_primitives.py`:** Remains at `src/` root (not moved to `shared/`) because `shared/model_spec.py` inherits from `PrecisionContext`. Clean separation deferred to Phase 1 when `PrecisionConfig` replaces the ABC hierarchy.
- **`problem_type_strategy.py`:** Uses copy+shim approach — re-exports from `src.backends.opencl.execution_plan` because `CceStrategy`/`BceStrategy` reference OpenCL-specific `KernelSignature` types. Clean extraction deferred to Phase 1.

---

## Table of Contents

1. [Scope & Constraints](#1-scope--constraints)
2. [Current State Inventory](#2-current-state-inventory)
3. [Target Directory Structure](#3-target-directory-structure)
4. [Task Breakdown](#4-task-breakdown)
   - [Step 0.1: Create directory skeletons](#step-01-create-directory-skeletons)
   - [Step 0.2: Move backend-neutral modules → `src/shared/`](#step-02-move-backend-neutral-modules--srcshared)
   - [Step 0.3: Extract `HardwareProfile` → `src/shared/hardware_profile.py`](#step-03-extract-hardwareprofile--srcsharedhardware_profilepy)
   - [Step 0.4: Extract `PrecisionConfig` → `src/shared/precision_config.py`](#step-04-extract-precisionconfig--srcsharedprecision_configpy)
   - [Step 0.5: Extract `KernelContract` hierarchy → `src/shared/kernel_contracts/`](#step-05-extract-kernelcontract-hierarchy--srcsharedkernel_contracts)
   - [Step 0.6: Extract `ProblemTypeStrategy` → `src/shared/problem_type_strategy.py`](#step-06-extract-problemtypestrategy--srcsharedproblem_type_strategypy)
   - [Step 0.7: Move OpenCL-specific modules → `src/backends/opencl/`](#step-07-move-opencl-specific-modules--srcbackendsopencl)
   - [Step 0.8: Create `meson.options`](#step-08-create-mesonoptions)
   - [Step 0.9: Create `src/_build_config.py.in`](#step-09-create-src_build_configpyin)
   - [Step 0.10: Designate `kernels.cl.h` as the algorithmic specification](#step-010-designate-kernelsclh-as-the-algorithmic-specification)
   - [Step 0.11: Update Meson build](#step-011-update-meson-build)
   - [Step 0.12: Update all import paths](#step-012-update-all-import-paths)
   - [Step 0.13: Validate rollback gate](#step-013-validate-rollback-gate)
5. [Import Path Mapping](#5-import-path-mapping)
6. [Shim Adapter Strategy](#6-shim-adapter-strategy)
7. [Rollback Gate Procedure](#7-rollback-gate-procedure)
8. [Risk Register](#8-risk-register)

---

## 1. Scope & Constraints

### In scope

- Creating the `src/shared/` and `src/backends/opencl/` directory skeletons (ADR-012 Option A).
- Physically relocating backend-neutral modules into `src/shared/`.
- Extracting shared-layer types from dissolved modules into new `src/shared/` files.
- Moving OpenCL-coupled modules into `src/backends/opencl/` with thin shim adapters to preserve the existing API surface.
- Creating build system configuration files (`meson.options`, `_build_config.py.in`).
- Updating the architecture's `meson.build` to reflect the new directory layout, with `subdir()` delegation (ADR-014 Option C).
- Updating all internal import paths (source modules, test files, `__init__.py`).
- Designating `kernels.cl.h` as the algorithmic specification document (ADR-013).

### Out of scope

- Implementing the plan model (Phase 1).
- Implementing plan node types, `BufferHandle`/`BufferDescriptor`, `RetrievalFuture` Protocol, plan builder (Phase 1).
- Implementing any new `PlanRenderer` or `KernelBinding` types (Phases 2–5).
- Writing Tier 1/2/3 tests (Phases 1, 4).
- Any behavioral change to existing modules — Phase 0 is purely structural.
- Creating `src/backends/vulkan/` or `src/backends/cpu/` directories (those are created in their respective phases when they have content).

### Key constraint: atomicity

ADR-017 Consequences states: *"Phase 0 import churn [...] must be the first change, and it must be atomic — partial migration of import paths creates an inconsistent codebase."* All import-path changes within Phase 0 must be committed together. Intermediate commits may reorganize files, but the final Phase 0 commit must leave all imports consistent and all tests passing.

---

## 2. Current State Inventory

### Source modules (`src/`)

| Module | Classification | Phase 0 Destination |
| :--- | :--- | :--- |
| `__init__.py` | Package root | Remains at `src/__init__.py` (updated imports) |
| `model_spec.py` | Backend-neutral | → `src/shared/model_spec.py` |
| `parameter_space.py` | Backend-neutral | → `src/shared/parameter_space.py` |
| `memory_layout.py` | Backend-neutral | → `src/shared/memory_layout.py` |
| `stabilization_policy.py` | Backend-neutral | → `src/shared/stabilization_policy.py` |
| `workload_primitives.py` | Backend-neutral | → `src/shared/workload_primitives.py` |
| `arch_primitives.py` | Dissolved (ADR-008) | Extract `PrecisionConfig` → `src/shared/precision_config.py`; shim at original path |
| `cl_context_manager.py` | OpenCL-coupled | Extract `HardwareProfile` → `src/shared/hardware_profile.py`; remainder → `src/backends/opencl/context.py`; shim at original path |
| `compute_patterns.py` | OpenCL-coupled | → `src/backends/opencl/compute_patterns.py`; shim at original path |
| `execution_plan.py` | Mixed | Extract `ProblemTypeStrategy`/`CceStrategy`/`BceStrategy` → `src/shared/problem_type_strategy.py`; remainder → `src/backends/opencl/execution_plan.py`; shim at original path |
| `graph_recipes.py` | OpenCL-coupled | → `src/backends/opencl/graph_recipes.py`; shim at original path |
| `launcher_infra.py` | OpenCL-coupled | → `src/backends/opencl/launcher_infra.py`; shim at original path |
| `batch_processor.py` | OpenCL-coupled | → `src/backends/opencl/batch_processor.py`; shim at original path |
| `main_orchestrator.py` | Top-level orchestrator | Remains at `src/main_orchestrator.py` (updated imports) |
| `kernel_signatures/__init__.py` | OpenCL-coupled (ADR-007) | → `src/backends/opencl/kernel_bindings/__init__.py` |
| `kernel_signatures/phase_*.py` | OpenCL-coupled (ADR-007) | → `src/backends/opencl/kernel_bindings/phase_*.py` |

### Kernel source files (`kernels/`)

| File | Phase 0 Action |
| :--- | :--- |
| `kernels.cl.h` | Designate as algorithmic specification (ADR-013). No move. |
| `phase_1_act.cl.c` | No move. Remains at `kernels/`. |
| `phase_2_learn_A_production.cl.c` | No move. |
| `phase_2_learn_B_processing.cl.c` | No move. |
| `phase_2_learn_C_reduction.cl.c` | No move. |
| `phase_2_learn_D_backprop.cl.c` | No move. |
| `phase_3_update.cl.c` | No move. |

### Test files

| Location | Phase 0 Action |
| :--- | :--- |
| `src/tests/conftest.py` | Update import paths |
| `src/tests/test_*.py` (11 files) | Update import paths |
| `tests/conftest.py` | Update import paths |
| `tests/test_integration_*.py` (9 files) | Update import paths |
| `tests/bench_*.py` (3 files) | Update import paths |

---

## 3. Target Directory Structure

After Phase 0 completes, the directory tree under the architecture root shall be:

```
averaging_ensembled_classifier/
├── meson.build                     # Updated: subdir() delegation (ADR-014)
├── meson.options                   # NEW: backend_vulkan, backend_cpu options
├── pyproject.toml                  # Updated: optional-dependencies
├── kernels/                        # Unchanged — algorithmic specification (ADR-013)
│   ├── kernels.cl.h                #   Designated as spec document
│   ├── phase_1_act.cl.c
│   ├── phase_2_learn_A_production.cl.c
│   ├── phase_2_learn_B_processing.cl.c
│   ├── phase_2_learn_C_reduction.cl.c
│   ├── phase_2_learn_D_backprop.cl.c
│   └── phase_3_update.cl.c
├── src/
│   ├── __init__.py                 # Updated imports
│   ├── main_orchestrator.py        # Updated imports
│   ├── _build_config.py.in         # NEW: Meson configure_file template
│   ├── shared/
│   │   ├── __init__.py             # NEW
│   │   ├── model_spec.py           # MOVED from src/
│   │   ├── parameter_space.py      # MOVED from src/
│   │   ├── memory_layout.py        # MOVED from src/
│   │   ├── stabilization_policy.py # MOVED from src/
│   │   ├── workload_primitives.py  # MOVED from src/
│   │   ├── precision_config.py     # NEW: extracted from arch_primitives.py
│   │   ├── hardware_profile.py     # NEW: extracted from cl_context_manager.py
│   │   ├── problem_type_strategy.py# NEW: extracted from execution_plan.py
│   │   └── kernel_contracts/       # NEW: extracted from kernel_signatures/
│   │       ├── __init__.py
│   │       ├── phase_1_act.py
│   │       ├── phase_2_learn_A_production.py
│   │       ├── phase_2_learn_B_processing.py
│   │       ├── phase_2_learn_C_reduction.py
│   │       ├── phase_2_learn_D_backprop.py
│   │       └── phase_3_update.py
│   └── backends/
│       ├── __init__.py             # NEW
│       └── opencl/
│           ├── __init__.py         # NEW
│           ├── context.py          # MOVED from cl_context_manager.py (OpenCL parts)
│           ├── compute_patterns.py # MOVED from src/compute_patterns.py
│           ├── execution_plan.py   # MOVED from src/execution_plan.py (OpenCL parts)
│           ├── graph_recipes.py    # MOVED from src/graph_recipes.py
│           ├── launcher_infra.py   # MOVED from src/launcher_infra.py
│           ├── batch_processor.py  # MOVED from src/batch_processor.py
│           └── kernel_bindings/    # MOVED from src/kernel_signatures/
│               ├── __init__.py
│               ├── phase_1_act.py
│               ├── phase_2_learn_A_production.py
│               ├── phase_2_learn_B_processing.py
│               ├── phase_2_learn_C_reduction.py
│               ├── phase_2_learn_D_backprop.py
│               └── phase_3_update.py
├── src/tests/                      # Updated import paths
└── tests/                          # Updated import paths
```

### Thin shim adapters at original paths

To preserve backward compatibility during the transition, the original module paths gain thin shim files that re-export all public symbols from their new locations. These shims are temporary — they are removed in Phase 6. See [§6 Shim Adapter Strategy](#6-shim-adapter-strategy) for details.

Shim files:
- `src/arch_primitives.py` → re-exports from `src.shared.precision_config` (and retains original `PrecisionContext`/`Float32Context`/`Float16Context` until Phase 1 replaces them)
- `src/cl_context_manager.py` → re-exports from `src.backends.opencl.context`
- `src/compute_patterns.py` → re-exports from `src.backends.opencl.compute_patterns`
- `src/execution_plan.py` → re-exports from `src.backends.opencl.execution_plan` + `src.shared.problem_type_strategy`
- `src/graph_recipes.py` → re-exports from `src.backends.opencl.graph_recipes`
- `src/launcher_infra.py` → re-exports from `src.backends.opencl.launcher_infra`
- `src/batch_processor.py` → re-exports from `src.backends.opencl.batch_processor`
- `src/kernel_signatures/__init__.py` → re-exports from `src.backends.opencl.kernel_bindings`

---

## 4. Task Breakdown

### Step 0.1: Create directory skeletons

**Action:** Create the following directories and `__init__.py` files:

```
src/shared/__init__.py
src/shared/kernel_contracts/__init__.py
src/backends/__init__.py
src/backends/opencl/__init__.py
src/backends/opencl/kernel_bindings/__init__.py
```

**Details:**
- `src/shared/__init__.py` — exports the shared-layer public API (can be populated incrementally as modules are moved in subsequent steps).
- `src/backends/__init__.py` — minimal package marker; no exports.
- `src/backends/opencl/__init__.py` — minimal package marker; no public re-exports (backend modules are imported directly by the orchestrator).

**Validation:** Directory structure exists; Python can import the empty packages.

---

### Step 0.2: Move backend-neutral modules → `src/shared/`

**Action:** Move these five files physically into `src/shared/`:

| Source | Destination |
| :--- | :--- |
| `src/model_spec.py` | `src/shared/model_spec.py` |
| `src/parameter_space.py` | `src/shared/parameter_space.py` |
| `src/memory_layout.py` | `src/shared/memory_layout.py` |
| `src/stabilization_policy.py` | `src/shared/stabilization_policy.py` |
| `src/workload_primitives.py` | `src/shared/workload_primitives.py` |

**Internal import updates within moved files:**

- `model_spec.py` imports `PrecisionContext` from `arch_primitives.py`. After Phase 0, this import changes to either:
  - `from .precision_config import ...` (if `PrecisionConfig` is ready), or
  - `from ..arch_primitives import PrecisionContext, Float32Context, Float16Context` (temporary, via shim) until Phase 1 replaces the inheritance hierarchy.
  - **Decision:** Use the shim import initially (`from ..arch_primitives import ...`). This preserves the existing `ModelSpec(PrecisionContext, abc.ABC)` inheritance unchanged. The `PrecisionConfig` frozen dataclass replacement is a Phase 1 behavioral change.

- `parameter_space.py` imports `ModelSpec` from `model_spec`. Update to `from .model_spec import ModelSpec`.

- `stabilization_policy.py` has no internal imports from other src modules (only stdlib + numpy). No import changes needed.

- `memory_layout.py` has no internal imports from other src modules (only stdlib + numpy + enum). No import changes needed.

- `workload_primitives.py` has no internal imports from other src modules (only stdlib + numpy + abc). No import changes needed.

**Validation:** `python -c "from src.shared.model_spec import ModelSpec"` succeeds (from the architecture root with appropriate sys.path).

---

### Step 0.3: Extract `HardwareProfile` → `src/shared/hardware_profile.py`

**Action:** Create `src/shared/hardware_profile.py` containing the `HardwareProfile` frozen dataclass as specified by ADR-006.

**Details:**
- ADR-006 specifies `HardwareProfile` as a frozen dataclass with Policy-tier consumption-named fields: `cache_line_bytes`, `max_local_memory_bytes`, `max_work_group_size`, `simd_width`, `max_alloc_bytes`, `global_memory_bytes`.
- In Phase 0, this is a **stub extraction** — the `HardwareProfile` dataclass is created in the shared layer, but the existing `DiscoveredArchConstants` class in `cl_context_manager.py` is **not yet modified** to delegate to it. The `DiscoveredArchConstants` → `HardwareProfile` replacement is a behavioral change that occurs in Phase 1/2 when the plan model consumes `HardwareProfile`.
- Phase 0 creates the file and the type so downstream work (Phase 1) can import it immediately.

**File content outline:**
```python
# src/shared/hardware_profile.py
"""Backend-neutral hardware capability profile (ADR-006)."""
from dataclasses import dataclass

@dataclass(frozen=True)
class HardwareProfile:
    """Immutable snapshot of hardware capabilities, consumed by the plan builder.
    
    Each backend populates this from its native device queries.
    Field names describe Policy-tier consumption, not hardware origin.
    """
    cache_line_bytes: int
    max_local_memory_bytes: int
    max_work_group_size: int
    simd_width: int
    max_alloc_bytes: int
    global_memory_bytes: int
```

**Validation:** `from src.shared.hardware_profile import HardwareProfile` succeeds; `HardwareProfile(64, 65536, 256, 32, 1073741824, 4294967296)` constructs a frozen instance.

---

### Step 0.4: Extract `PrecisionConfig` → `src/shared/precision_config.py`

**Action:** Create `src/shared/precision_config.py` containing the `PrecisionConfig` frozen dataclass as specified by ADR-008.

**Details:**
- ADR-008 specifies `PrecisionConfig` as a frozen dataclass parameterized by `numpy_dtype`, with derived fields `fp_format_max` and `epsilon` computed at construction time.
- In Phase 0, this is a **stub extraction** — the new type is created, but the existing `PrecisionContext` ABC and its `Float32Context`/`Float16Context` subclasses in `arch_primitives.py` remain unchanged and operational. The `model_spec.py` → `PrecisionConfig` composition replacement is a Phase 1 behavioral change.
- The existing `arch_primitives.py` at its original path becomes a shim (Step 0.7) that re-exports the legacy classes. Modules that currently import from `arch_primitives` continue to work unchanged.

**File content outline:**
```python
# src/shared/precision_config.py
"""Backend-neutral precision configuration (ADR-008)."""
from dataclasses import dataclass
import numpy as np

@dataclass(frozen=True)
class PrecisionConfig:
    """Immutable precision configuration for plan construction.
    
    Replaces the PrecisionContext ABC hierarchy. Consumed by ModelSpec,
    StabilizationPolicy, and the plan builder.
    """
    numpy_dtype: np.dtype
    fp_format_max: float
    epsilon: float

    @classmethod
    def float32(cls) -> "PrecisionConfig":
        finfo = np.finfo(np.float32)
        return cls(numpy_dtype=np.dtype(np.float32),
                   fp_format_max=float(finfo.max),
                   epsilon=float(finfo.eps))

    @classmethod
    def float16(cls) -> "PrecisionConfig":
        finfo = np.finfo(np.float16)
        return cls(numpy_dtype=np.dtype(np.float16),
                   fp_format_max=float(finfo.max),
                   epsilon=float(finfo.eps))
```

**Validation:** `PrecisionConfig.float32()` and `PrecisionConfig.float16()` construct correctly; instances are immutable.

---

### Step 0.5: Extract `KernelContract` hierarchy → `src/shared/kernel_contracts/`

**Action:** Create `src/shared/kernel_contracts/` with one module per phase file, mirroring the existing `kernel_signatures/` structure.

**Details:**
- ADR-007 specifies that each `KernelSignature` subclass is split into a `KernelContract` (shared-layer frozen dataclass) and a `KernelBinding` (per-backend).
- In Phase 0, the `kernel_contracts/` directory is created with **stub files** that define the `KernelContract` base class and per-phase contract placeholders. The actual contract extraction (pulling validation logic out of each `KernelSignature`) is a **Phase 1 deliverable** — it requires the plan node types to exist.
- The stubs ensure the package structure exists and is importable, so Phase 1 can populate them.

**File inventory:**

| File | Content |
| :--- | :--- |
| `__init__.py` | Exports `KernelContract` base class |
| `phase_1_act.py` | Stub: `ForwardPassContract`, `ComputeHiddenMaskContract` |
| `phase_2_learn_A_production.py` | Stub: `ComputeProbsLoss{Cce,Bce}Contract`, `CalculateModuleParamGrads{Cce,Bce}Contract` |
| `phase_2_learn_B_processing.py` | Stub: `BackpropErrorToHiddenContract`, `CalculateChunkTempGradientsContract`, `ClipPartialGradientsContract` |
| `phase_2_learn_C_reduction.py` | Stub: `GatherAndPermuteContract`, `AggregateRegisterReduceContract`, `AggregateLocalReduceContract`, `ClipIntermediateGradContract`, `StabilizeReduceGradHContract` |
| `phase_2_learn_D_backprop.py` | Stub: `BackpropSharedWeightsContract`, `BackpropSharedBiasesContract`, `ClipSharedGradientsContract` |
| `phase_3_update.py` | Stub: `NormalizeGradientsContract`, `AdamUpdateContract`, `ClampTemperaturesContract` |

Each stub is a minimal frozen dataclass inheriting from `KernelContract` with a `kernel_name: str` field. The full field set is populated in Phase 1.

**Validation:** `from src.shared.kernel_contracts import KernelContract` imports successfully. Each per-phase module is importable.

---

### Step 0.6: Extract `ProblemTypeStrategy` → `src/shared/problem_type_strategy.py`

**Action:** Extract the `ProblemTypeStrategy` ABC, `CceStrategy`, and `BceStrategy` from `src/execution_plan.py` into `src/shared/problem_type_strategy.py`.

**Details:**
- ADR-011 specifies that `ProblemTypeStrategy` is a shared-layer abstraction consumed at plan-construction time. It must not import any OpenCL types.
- Currently, `CceStrategy` and `BceStrategy` in `execution_plan.py` reference `KernelSignature` subclasses — these are OpenCL-specific. In Phase 0, the extracted strategies retain their existing method signatures but the OpenCL-specific return types are imported from the new backend path (`src.backends.opencl.kernel_bindings`). This preserves behavior while establishing the shared-layer location.
- **Alternative (simpler):** If the Strategy methods return types that are too tightly coupled to OpenCL, the extraction in Phase 0 can be a **copy + shim** — the original classes remain in `execution_plan.py` and a shim in `src/shared/problem_type_strategy.py` re-exports from the backend path. The clean extraction happens in Phase 1 when `KernelContract` replaces `KernelSignature` in the strategy return types.

**Decision for Phase 0:** Use the copy + shim approach. The `ProblemTypeStrategy` ABC, `CceStrategy`, and `BceStrategy` are physically located in `src/shared/problem_type_strategy.py` but continue to import `KernelSignature` types from the OpenCL backend via the shim chain. This violates the Plan boundary temporarily — the clean separation occurs in Phase 1 when the strategies produce `KernelContract` instances instead of `KernelSignature` instances.

**Validation:** `from src.shared.problem_type_strategy import ProblemTypeStrategy, CceStrategy, BceStrategy` succeeds; existing behavioral tests pass.

---

### Step 0.7: Move OpenCL-specific modules → `src/backends/opencl/`

**Action:** Move the following files into `src/backends/opencl/`:

| Source | Destination |
| :--- | :--- |
| `src/cl_context_manager.py` | `src/backends/opencl/context.py` |
| `src/compute_patterns.py` | `src/backends/opencl/compute_patterns.py` |
| `src/execution_plan.py` | `src/backends/opencl/execution_plan.py` |
| `src/graph_recipes.py` | `src/backends/opencl/graph_recipes.py` |
| `src/launcher_infra.py` | `src/backends/opencl/launcher_infra.py` |
| `src/batch_processor.py` | `src/backends/opencl/batch_processor.py` |
| `src/kernel_signatures/` | `src/backends/opencl/kernel_bindings/` |

**Internal import updates within moved files:**
- Each moved module's internal `from .xxx import` references must be updated to reflect the new package depth. For example, `from .model_spec import ModelSpec` in `batch_processor.py` becomes `from ...shared.model_spec import ModelSpec`.
- OpenCL-internal cross-references (e.g., `graph_recipes.py` importing from `launcher_infra.py`) update from `from .launcher_infra import ...` to `from .launcher_infra import ...` (same, since they're still siblings within `backends/opencl/`).
- References to shared-layer modules go through `from ...shared.xxx import ...` (three levels up from `backends/opencl/` to `src/`, then into `shared/`).

**Create shim files at original paths:**
After moving each file, create a thin shim at the original location that re-exports all public symbols. See [§6 Shim Adapter Strategy](#6-shim-adapter-strategy).

**Validation:** All existing imports (both direct and via shims) resolve correctly. `python -c "from src.cl_context_manager import OpenCLContextManager"` still works (via shim).

---

### Step 0.8: Create `meson.options`

**Action:** Create `meson.options` at the architecture root per ADR-014.

**File content:**
```meson
# meson.options — Build feature options for averaging_ensembled_classifier

option('backend_vulkan', type: 'feature', value: 'auto',
       description: 'Build Vulkan SPIR-V compute shaders (requires glslc)')

option('backend_cpu', type: 'feature', value: 'auto',
       description: 'Build CPU SIMD kernel shared library')

option('cpu_isa_flags', type: 'array', value: [],
       description: 'C compiler ISA flags for CPU backend (e.g., [\'-mavx2\']. Empty = -march=native)')
```

**Notes:**
- No `backend_opencl` option — the OpenCL backend is always enabled (runtime kernel compilation, no build-time dependency beyond PyOpenCL).
- The `auto` default means zero-configuration for standard development: `meson setup builddir` probes the environment automatically.
- In Phase 0, neither Vulkan nor CPU backends exist yet, so these options have no effect. They are created now to establish the convention and make `_build_config.py` generation possible.

**Validation:** `meson setup` from the architecture root accepts the new options without error.

---

### Step 0.9: Create `src/_build_config.py.in`

**Action:** Create the Meson `configure_file` template per ADR-014.

**File content:**
```python
# _build_config.py.in — processed by Meson configure_file()
"""Build-time configuration for averaging_ensembled_classifier.

Generated by the Meson build system. Do not edit manually.
"""

BACKEND_OPENCL: bool = @BACKEND_OPENCL@
BACKEND_VULKAN: bool = @BACKEND_VULKAN@
BACKEND_CPU: bool = @BACKEND_CPU@
```

**Notes:**
- This template is processed by Meson's `configure_file()` to produce `_build_config.py` in the build directory.
- In Phase 0, `BACKEND_OPENCL` is always `True`; `BACKEND_VULKAN` and `BACKEND_CPU` depend on environment probing.
- The generated module is consumed by the orchestrator (for backend selection) and the test framework (for tier skip logic, Phase 4).

**Validation:** After `meson setup builddir`, `builddir/_build_config.py` exists and contains valid Python boolean assignments.

---

### Step 0.10: Designate `kernels.cl.h` as the algorithmic specification

**Action:** Add a header comment to `kernels/kernels.cl.h` explicitly designating it as the algorithmic specification per ADR-013.

**Details:**
- ADR-013 (Option A) designates `kernels.cl.h` as the human-readable, language-neutral algorithmic reference. Its `@kernel_contract` blocks and `@param` annotations constitute the authoritative specification that all backends implement against.
- No structural change to the file. Add a comment block at the top (if not already present) stating this designation, referencing ADR-013.
- The file remains at `kernels/kernels.cl.h` — it does not move.

**Validation:** Comment is present; no functional change.

---

### Step 0.11: Update Meson build

**Action:** Rewrite `meson.build` to reflect the new directory structure, using `subdir()` delegation per ADR-014.

**Top-level `meson.build` changes:**

1. Add dependency probing for `glslc` and CPU backend (guarded by feature options).
2. Replace the flat `py_sources` list with `subdir()` delegation:
   - `subdir('src/shared')` — pure Python install.
   - `subdir('src/backends/opencl')` — Python install for OpenCL backend.
3. Add `configure_file()` for `_build_config.py`.
4. Update the top-level `py.install_sources()` call for `src/__init__.py` and `src/main_orchestrator.py`.
5. Retain the existing `install_data()` for `cl_kernels` (kernel files at architecture root).
6. Add conditional `subdir()` calls for Vulkan and CPU (guarded, no-op in Phase 0 since directories don't exist yet).

**Create per-backend `meson.build` files:**

- `src/shared/meson.build` — `py.install_sources()` for all shared-layer Python files.
- `src/backends/opencl/meson.build` — `py.install_sources()` for all OpenCL backend Python files, including `kernel_bindings/`.

**Validation:** `meson setup builddir --wipe` succeeds; `meson compile -C builddir` succeeds; `meson install -C builddir` installs all files to correct package paths.

---

### Step 0.12: Update all import paths

**Action:** Update every file that imports from relocated modules. This is the largest mechanical step.

**Categories of import changes:**

#### A. Shared-layer modules importing from each other

| Module | Old Import | New Import |
| :--- | :--- | :--- |
| `shared/model_spec.py` | `from .arch_primitives import PrecisionContext` | `from ..arch_primitives import PrecisionContext` (via shim, temporary) |
| `shared/parameter_space.py` | `from .model_spec import ModelSpec` | `from .model_spec import ModelSpec` (same, now within `shared/`) |

#### B. OpenCL backend modules importing shared-layer types

| Module | Old Import | New Import |
| :--- | :--- | :--- |
| `backends/opencl/launcher_infra.py` | `from .model_spec import ModelSpec` | `from ...shared.model_spec import ModelSpec` |
| `backends/opencl/execution_plan.py` | `from .kernel_signatures import ...` | `from .kernel_bindings import ...` |
| (etc.) | | |

#### C. `main_orchestrator.py` imports

| Old Import | New Import |
| :--- | :--- |
| `from .model_spec import ...` | `from .shared.model_spec import ...` |
| `from .cl_context_manager import ...` | `from .backends.opencl.context import ...` |
| `from .parameter_space import ...` | `from .shared.parameter_space import ...` |
| `from .execution_plan import ...` | `from .backends.opencl.execution_plan import ...` + `from .shared.problem_type_strategy import ...` |
| `from .kernel_signatures import ...` | `from .backends.opencl.kernel_bindings import ...` |
| `from .launcher_infra import ...` | `from .backends.opencl.launcher_infra import ...` |
| `from .compute_patterns import ...` | `from .backends.opencl.compute_patterns import ...` |
| `from .workload_primitives import ...` | `from .shared.workload_primitives import ...` |
| `from .stabilization_policy import ...` | `from .shared.stabilization_policy import ...` |
| `from .batch_processor import ...` | `from .backends.opencl.batch_processor import ...` |
| `from . import graph_recipes` | `from .backends.opencl import graph_recipes` |

#### D. `src/__init__.py` imports

| Old Import | New Import |
| :--- | :--- |
| `from .main_orchestrator import ...` | `from .main_orchestrator import ...` (unchanged — `main_orchestrator.py` stays at `src/`) |
| `from .model_spec import ModelSpec` | `from .shared.model_spec import ModelSpec` |
| `from .parameter_space import ...` | `from .shared.parameter_space import ...` |
| `from .stabilization_policy import ...` | `from .shared.stabilization_policy import ...` |

#### E. Test files

All test files under `src/tests/` and `tests/` that use `from src.xxx import ...` patterns must be updated:

| Old Pattern | New Pattern |
| :--- | :--- |
| `from src.model_spec import ...` | `from src.shared.model_spec import ...` |
| `from src.parameter_space import ...` | `from src.shared.parameter_space import ...` |
| `from src.workload_primitives import ...` | `from src.shared.workload_primitives import ...` |
| `from src.stabilization_policy import ...` | `from src.shared.stabilization_policy import ...` |
| `from src.memory_layout import ...` | `from src.shared.memory_layout import ...` |
| `from src.execution_plan import ...` | `from src.backends.opencl.execution_plan import ...` (or via shim) |
| `from src.compute_patterns import ...` | `from src.backends.opencl.compute_patterns import ...` (or via shim) |
| `from src.graph_recipes import ...` | `from src.backends.opencl.graph_recipes import ...` (or via shim) |
| `from src.launcher_infra import ...` | `from src.backends.opencl.launcher_infra import ...` (or via shim) |

**Strategy decision: direct imports vs. shim imports in tests.**

Two options:
1. Update all test imports to point directly to the new locations.
2. Leave test imports pointing to the old shim paths; they will be updated in Phase 6.

**Decision:** Option 1 — update test imports directly. This exercises the new import paths immediately and catches any errors. The shims exist as a safety net for any external consumers, not as the primary import path for the project's own code.

**Validation:** `pytest src/tests/ tests/` — all tests pass with updated imports.

---

### Step 0.13: Validate rollback gate

**Action:** Run the complete existing test suite and confirm 100% green.

**Procedure:**
1. `cd architectures/averaging_ensembled_classifier`
2. `pytest src/tests/ tests/ -v` — all tests must pass.
3. `python -c "from src import TrainingOrchestrator, ModelSpec, ParameterSpace"` — public API imports work.
4. `meson setup builddir --wipe && meson compile -C builddir` — build succeeds.
5. If any test fails, diagnose: the failure must be an import error (fixable) or a pre-existing failure (document and exclude). No behavioral regression is acceptable.

---

## 5. Import Path Mapping

Complete mapping from old → new import paths for all public symbols:

| Symbol | Old path | New canonical path | Shim available? |
| :--- | :--- | :--- | :--- |
| `ModelSpec`, `Float32ModelSpec`, `Float16ModelSpec` | `src.model_spec` | `src.shared.model_spec` | No (file moved) |
| `ParameterSpace`, `ParameterFlowConfig` | `src.parameter_space` | `src.shared.parameter_space` | No (file moved) |
| `MemoryLayout`, `PaddingStrategy`, `PaddingType` | `src.memory_layout` | `src.shared.memory_layout` | No (file moved) |
| `StabilizationPolicy` | `src.stabilization_policy` | `src.shared.stabilization_policy` | No (file moved) |
| `WorkTile`, `TilingScheme`, `GatherPrimitive`, etc. | `src.workload_primitives` | `src.shared.workload_primitives` | No (file moved) |
| `HardwareProfile` | (new) | `src.shared.hardware_profile` | N/A |
| `PrecisionConfig` | (new) | `src.shared.precision_config` | N/A |
| `KernelContract` | (new) | `src.shared.kernel_contracts` | N/A |
| `ProblemTypeStrategy`, `CceStrategy`, `BceStrategy` | `src.execution_plan` | `src.shared.problem_type_strategy` | Yes (via `src.execution_plan` shim) |
| `PrecisionContext`, `Float32Context`, `Float16Context` | `src.arch_primitives` | `src.arch_primitives` (shim) | Yes |
| `OpenCLContextManager`, `ComputeEnvironment`, etc. | `src.cl_context_manager` | `src.backends.opencl.context` | Yes |
| `ReductionPlan`, `AggregationManager` | `src.compute_patterns` | `src.backends.opencl.compute_patterns` | Yes |
| `ExecutionPlan`, `DependencyProvider`, etc. | `src.execution_plan` | `src.backends.opencl.execution_plan` | Yes |
| `Services`, `BufferManager`, `KernelExecutor`, etc. | `src.launcher_infra` | `src.backends.opencl.launcher_infra` | Yes |
| `BatchProcessor` | `src.batch_processor` | `src.backends.opencl.batch_processor` | Yes |
| `ForwardPassSignature`, etc. | `src.kernel_signatures` | `src.backends.opencl.kernel_bindings` | Yes |
| `graph_recipes` (module) | `src.graph_recipes` | `src.backends.opencl.graph_recipes` | Yes |

---

## 6. Shim Adapter Strategy

### Purpose

Thin shim adapters are placed at the original import paths to ensure that any code not yet updated to the new paths continues to function. This provides a safety net during the transition. **Shims are removed in Phase 6.**

### Shim template

Each shim file follows this pattern:

```python
# src/<original_module>.py — SHIM (Phase 0)
# This module has moved to src/<new_path>. This shim re-exports all
# public symbols for backward compatibility. Remove in Phase 6.
#
# Canonical location: src/<new_path>

from src.<new_path> import *  # noqa: F401,F403
```

### Specific shims

**`src/arch_primitives.py`** (special case):
- The `PrecisionContext` ABC and its subclasses are *not moved* in Phase 0, because `model_spec.py` inherits from them. The file remains at `src/arch_primitives.py` as-is until Phase 1 replaces the inheritance hierarchy with `PrecisionConfig` composition.
- No shim needed — the file stays.

**`src/cl_context_manager.py`**:
```python
# SHIM — canonical location: src/backends/opencl/context.py
from .backends.opencl.context import *  # noqa: F401,F403
```

**`src/compute_patterns.py`**:
```python
# SHIM — canonical location: src/backends/opencl/compute_patterns.py
from .backends.opencl.compute_patterns import *  # noqa: F401,F403
```

**`src/execution_plan.py`**:
```python
# SHIM — canonical location: src/backends/opencl/execution_plan.py
# ProblemTypeStrategy family: src/shared/problem_type_strategy.py
from .backends.opencl.execution_plan import *  # noqa: F401,F403
from .shared.problem_type_strategy import *  # noqa: F401,F403
```

**`src/graph_recipes.py`**:
```python
# SHIM — canonical location: src/backends/opencl/graph_recipes.py
from .backends.opencl.graph_recipes import *  # noqa: F401,F403
```

**`src/launcher_infra.py`**:
```python
# SHIM — canonical location: src/backends/opencl/launcher_infra.py
from .backends.opencl.launcher_infra import *  # noqa: F401,F403
```

**`src/batch_processor.py`**:
```python
# SHIM — canonical location: src/backends/opencl/batch_processor.py
from .backends.opencl.batch_processor import *  # noqa: F401,F403
```

**`src/kernel_signatures/__init__.py`**:
```python
# SHIM — canonical location: src/backends/opencl/kernel_bindings/
from ..backends.opencl.kernel_bindings import *  # noqa: F401,F403
```

### Shim lifecycle

| Event | Shim status |
| :--- | :--- |
| Phase 0 complete | Shims created; existing code works via both old and new paths |
| Phases 1–5 | New code uses canonical (new) paths; shims untouched |
| Phase 6 (legacy removal) | Shims deleted |

---

## 7. Rollback Gate Procedure

### Gate definition (ADR-017)

> **Rollback gate:** All existing tests green. No tier gate — Tier 1 tests do not yet exist. The gate is the existing test suite: every test that passed before Phase 0 must pass after.

### Pre-Phase 0 baseline capture

Before any Phase 0 changes:

```bash
cd architectures/averaging_ensembled_classifier
pytest src/tests/ tests/ -v --tb=short 2>&1 | tee /tmp/phase0_baseline.txt
```

Record the count of passed, failed, skipped, and error tests. Any pre-existing failures are documented and excluded from the gate.

### Post-Phase 0 validation

After all Phase 0 steps are complete:

```bash
cd architectures/averaging_ensembled_classifier
pytest src/tests/ tests/ -v --tb=short 2>&1 | tee /tmp/phase0_post.txt
```

**Gate criteria:**
1. No new test failures (compared to baseline).
2. No new test errors.
3. All previously-passing tests still pass.
4. Skipped tests remain skipped (no new skips unless a pre-existing skip condition changed).
5. Build succeeds: `meson setup builddir --wipe && meson compile -C builddir`.

### Rollback procedure

If the gate fails and cannot be fixed within a reasonable timeframe:

1. `git stash` or `git checkout -- .` to revert all Phase 0 changes.
2. Verify baseline tests pass again.
3. Diagnose the failure (typically an import path error).
4. Re-attempt Phase 0.

---

## 8. Risk Register

| # | Risk | Likelihood | Impact | Mitigation |
| :--- | :--- | :--- | :--- | :--- |
| R1 | Circular import between `shared/` and `backends/opencl/` | Medium | High — breaks all imports | Phase 0 preserves the existing dependency direction: shared modules are consumed by backend modules, never the reverse. The `ProblemTypeStrategy` extraction (Step 0.6) is the only potential circular path — mitigated by the copy+shim approach. |
| R2 | `model_spec.py` inheritance from `PrecisionContext` creates cross-package dependency | High | Low — expected and managed | `arch_primitives.py` stays at `src/` level in Phase 0 (not moved). `model_spec.py` in `shared/` imports from `..arch_primitives` — this is a known temporary Plan-boundary violation, resolved in Phase 1. |
| R3 | Meson build fails with new directory structure | Medium | Medium — blocks installation | Validate `meson setup` and `meson compile` as part of Step 0.11. The build changes are additive (`subdir()` delegation); the existing install logic is preserved. |
| R4 | External tools/scripts import from old paths | Low | Low — shims handle it | Shim adapters re-export all public symbols from original paths. No behavioral change for external consumers. |
| R5 | Test imports fail due to sys.path manipulation | Medium | Medium — test suite breaks | The `src/tests/conftest.py` sys.path hack must be updated to account for the new package structure. Validate in Step 0.12. |
| R6 | `kernel_signatures/` → `kernel_bindings/` rename breaks OpenCL dispatch | Low | High — runtime failure | The existing signatures are moved as-is (only the directory name changes). Internal cross-references within the kernel_signatures package use relative imports that are updated to reflect the new parent path. Validated by running integration tests. |
| R7 | Merge conflicts with concurrent work | Low | Medium — manual resolution | Phase 0 should be completed as a single atomic change (one PR/branch). No concurrent work on the same files during the transition. |

---

## Appendix: Execution Order Summary

The steps must be executed in the following order, though some can be parallelized:

```
Step 0.1  (directory skeletons)
    │
    ├──► Step 0.2  (move backend-neutral modules)
    │       │
    │       ├──► Step 0.3  (extract HardwareProfile)
    │       ├──► Step 0.4  (extract PrecisionConfig)
    │       ├──► Step 0.5  (extract KernelContract stubs)
    │       └──► Step 0.6  (extract ProblemTypeStrategy)
    │
    ├──► Step 0.7  (move OpenCL modules + create shims)
    │
    ├──► Step 0.8  (meson.options)
    ├──► Step 0.9  (_build_config.py.in)
    └──► Step 0.10 (kernels.cl.h designation)
         │
         ▼
    Step 0.11 (update Meson build)  ← depends on 0.1–0.10
         │
         ▼
    Step 0.12 (update all import paths)  ← depends on 0.2, 0.7
         │
         ▼
    Step 0.13 (validate rollback gate)  ← depends on all above
```

Steps 0.3–0.6 are independent extractions that can proceed in parallel once Step 0.2 establishes the `src/shared/` directory.
Steps 0.8–0.10 are independent and can proceed in parallel with Steps 0.2–0.7.
Step 0.11 (Meson build) depends on all directory changes being complete.
Step 0.12 (import updates) is the atomic import-churn step.
Step 0.13 is the final validation gate.
