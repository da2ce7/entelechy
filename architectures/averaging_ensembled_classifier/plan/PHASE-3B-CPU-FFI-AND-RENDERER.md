# Phase 3B: CPU Python FFI Layer & CPUPlanRenderer — Detailed Plan

**Status: ✅ IMPLEMENTED** — 9 Python modules delivered in `src/backends/cpu/`. Library loads via ctypes, `_verify_layouts()` validates struct alignment at load time, dispatch table maps all kernel names to task function pointers. `CPUPlanRenderer` traverses complete `ExecutionPlan` DAGs including `KernelDispatchNode`, `ReductionTreeNode`, `StreamingLoopNode`, `BarrierNode`, and `RetrievalNode`. Zero-copy `CPURetrievalFuture` returns unpadded numpy views. SIMD-aligned buffer allocation via numpy. Hardware discovery populates `HardwareProfile` from OS queries. Tier 1 green (no regressions).  
**Phase:** 3B of 6 (sub-phase B of 3)  
**Objective:** Implement the Python-side integration layer for the CPU backend — ctypes struct definitions (`_ffi_types.py`), library loading via `importlib.resources`, struct layout verification (`_verify_layouts()`), the dispatch table mapping `kernel_name` → `(task_fn_ptr, args_struct_class)`, and the `CPUPlanRenderer` that traverses an `ExecutionPlan` DAG and dispatches via `pool_dispatch_and_wait`. Also implement the CPU buffer allocator (SIMD-aligned numpy arrays), CPU hardware discovery, CPU type mapping, and the zero-copy `CPURetrievalFuture`. After this sub-phase, the CPU backend can fully render any `ExecutionPlan` produced by the plan builder.  
**Governing ADRs:** ADR-001 (backend abstraction boundary — `PlanRenderer` Protocol), ADR-006 (HardwareProfile population), ADR-009 (buffer lifecycle — SIMD-aligned allocation), ADR-010 (RetrievalFuture — zero-copy implementation), ADR-012 (module factoring — `src/backends/cpu/` structure), ADR-014 (build system — `_build_config.py` update, `importlib.resources` discovery), ADR-015 (interop — ctypes FFI, `_ffi_types.py`, `_verify_layouts()`, dispatch table)  
**Rollback gate:** Tier 1 green (no regressions) + library loads successfully + `_verify_layouts()` passes + structural smoke tests for `CPUPlanRenderer` pass (renderer can traverse a simple plan without failing). Full CPU Tier 2 gate deferred to Phase 3C.  
**Dependencies:** Phase 3A (CPU kernel library) — provides `libcpu_kernels.so` with all exported symbols. Phase 1 (plan model) — provides `ExecutionPlan`, `PlanRenderer` Protocol, and all shared-layer types.

### Phase 3A Deliverables Consumed Here

Phase 3B is the Python consumer of the Phase 3A native library. The following artifacts cross the FFI boundary:

| C-Side Artifact | Python-Side Consumer |
| :--- | :--- |
| `libcpu_kernels.so` | `ctypes.CDLL` in library loader |
| `pool_create` / `pool_destroy` | `CPUPlanRenderer.__init__` / `__del__` |
| `pool_dispatch_and_wait` | `CPUPlanRenderer._render_kernel_dispatch()` |
| `task_<kernel_name>` function pointers | Dispatch table entries |
| `execute_reduction_tree` | `CPUPlanRenderer._render_reduction_tree()` |
| `get_struct_size_*` functions | `_verify_layouts()` |
| `ForwardPassArgs`, `AdamUpdateArgs`, etc. (struct layouts) | `_ffi_types.py` ctypes `Structure` subclasses |
| `ReductionTreePlan` (struct layout) | `_ffi_types.py` ctypes `Structure` subclass |

### Phase 1 Deliverables Consumed Here

The `CPUPlanRenderer` consumes the same shared-layer types as the OpenCL renderer (Phase 2A):

| Shared Type | Phase 3B Consumption |
| :--- | :--- |
| `ExecutionPlan` | Input to `CPUPlanRenderer.render()` |
| `KernelDispatchNode` | Dispatched via dispatch table → `pool_dispatch_and_wait` |
| `ReductionTreeNode` | Dispatched via `execute_reduction_tree` |
| `StreamingLoopNode` | Rendered by Python-side chunk iteration loop |
| `BarrierNode` | No-op — synchronization is implicit in blocking dispatch |
| `RetrievalNode` | Returns `CPURetrievalFuture` (zero-copy numpy view) |
| `BufferDescriptor` | Drives `CPUBufferAllocator` allocation (SIMD-aligned numpy) |
| `BufferHandle` | Mapped to numpy arrays by the allocator |
| `BufferRole` | Determines allocation strategy (persistent vs. per-batch) |
| `HardwareProfile` | Populated by `discovery.py` from OS/hardware queries |
| `PrecisionConfig` | Drives dtype selection for numpy arrays |
| `RetrievalFuture` | Protocol implemented by `CPURetrievalFuture` |
| `PlanRenderer` | Protocol implemented by `CPUPlanRenderer` |
| `ReductionTreePlan` | Translated to C struct for `execute_reduction_tree` |
| `StreamingLoopPlan` | Consumed by the Python-side streaming loop |

---

## Table of Contents

1. [Scope & Constraints](#1-scope--constraints)
2. [Current State After Phase 3A](#2-current-state-after-phase-3a)
3. [Target Deliverables](#3-target-deliverables)
4. [Task Breakdown](#4-task-breakdown)
   - [Step 3B.1: Implement `_ffi_types.py` — ctypes struct definitions](#step-3b1-implement-_ffi_typespy--ctypes-struct-definitions)
   - [Step 3B.2: Implement library loader](#step-3b2-implement-library-loader)
   - [Step 3B.3: Implement `_verify_layouts()`](#step-3b3-implement-_verify_layouts)
   - [Step 3B.4: Implement dispatch table](#step-3b4-implement-dispatch-table)
   - [Step 3B.5: Implement `discovery.py` — HardwareProfile population](#step-3b5-implement-discoverypy--hardwareprofile-population)
   - [Step 3B.6: Implement `type_mapping.py` — PrecisionConfig → C type mapping](#step-3b6-implement-type_mappingpy--precisionconfig--c-type-mapping)
   - [Step 3B.7: Implement `buffer_allocator.py` — SIMD-aligned numpy arrays](#step-3b7-implement-buffer_allocatorpy--simd-aligned-numpy-arrays)
   - [Step 3B.8: Implement `CPURetrievalFuture` — zero-copy](#step-3b8-implement-cpuretrievalfuture--zero-copy)
   - [Step 3B.9: Implement `CPUPlanRenderer` — DAG traversal + FFI dispatch](#step-3b9-implement-cpuplanrenderer--dag-traversal--ffi-dispatch)
   - [Step 3B.10: Update `_build_config.py` template](#step-3b10-update-_build_configpy-template)
   - [Step 3B.11: Update `src/backends/cpu/__init__.py` exports](#step-3b11-update-srcbackendscpu__init__py-exports)
   - [Step 3B.12: Write infrastructure smoke tests](#step-3b12-write-infrastructure-smoke-tests)
   - [Step 3B.13: Validate rollback gate](#step-3b13-validate-rollback-gate)
5. [FFI Architecture](#5-ffi-architecture)
6. [Struct Marshalling Discipline](#6-struct-marshalling-discipline)
7. [Buffer Allocation Strategy](#7-buffer-allocation-strategy)
8. [Dispatch Flow](#8-dispatch-flow)
9. [Risk Register](#9-risk-register)

---

## 1. Scope & Constraints

### In scope

- Implementing `src/backends/cpu/_ffi_types.py` with ctypes `Structure` subclasses mirroring every C argument struct in `cpu_kernels.h` (ADR-015). Each struct's field order, field types, and packing must exactly match the C layout.
- Implementing library loading via `importlib.resources.files('averaging_ensembled_classifier.backends.cpu')` with platform-specific filename resolution (`libcpu_kernels.so` / `.dylib` / `.dll`) and `ctypes.CDLL` (ADR-014, ADR-015).
- Implementing `_verify_layouts()` — a function called at library load time that asserts `ctypes.sizeof(PythonStruct) == lib.get_struct_size_*()`for every argument struct (ADR-015). Layout drift is caught here before any dispatch occurs.
- Implementing the dispatch table: a `dict[str, tuple[ctypes function pointer, ctypes.Structure subclass]]` mapping each `kernel_name` from the plan model to the corresponding C task function pointer and argument struct class (ADR-015).
- Implementing `src/backends/cpu/discovery.py` — populating `HardwareProfile` from OS queries: thread count via `os.cpu_count()` / `os.sched_getaffinity()`, cache line size via platform-specific detection, `max_reduce_fan_in` from CPU constraints, `max_local_mem_bytes=None`, `global_mem_bytes` from system memory queries.
- Implementing `src/backends/cpu/type_mapping.py` — mapping `PrecisionConfig.numpy_dtype` to numpy array allocation dtype. Initially FP32 only.
- Implementing `src/backends/cpu/buffer_allocator.py` — translating `BufferDescriptor` plan-level declarations into SIMD-aligned numpy arrays. Uses `numpy.empty` with `SIMD_ALIGNMENT`-aligned memory (via a thin wrapper around `posix_memalign` or numpy's built-in alignment guarantees).
- Implementing `src/backends/cpu/retrieval.py` — `CPURetrievalFuture` satisfying the `RetrievalFuture` Protocol via zero-copy: `.result()` returns an unpadded numpy view of the already-host-accessible buffer (no D2H transfer needed).
- Implementing `src/backends/cpu/renderer.py` — the `CPUPlanRenderer` class that satisfies the `PlanRenderer` Protocol, traverses `ExecutionPlan.topological_order`, constructs ctypes argument structs from plan node parameters, and dispatches via `pool_dispatch_and_wait`.
- Updating `_build_config.py.in` template / generated `_build_config.py` to include `BACKEND_CPU = True` when the CPU backend is built (ADR-014).
- Writing infrastructure smoke tests that validate library loading, layout verification, and basic plan traversal.

### Out of scope

- Modifying the C kernel library (Phase 3A).
- Writing per-kernel numerical correctness tests (Phase 3C).
- Modifying shared-layer code beyond updating `_build_config.py.in`.
- Implementing Vulkan or modifying the OpenCL backend.
- FP16 support (FP32 only initially).
- The `WorkTicket` / `LearnHandle` / `Engine` user-facing API.

### Key constraint: ctypes only — zero additional dependencies

ADR-014 specifies `cpu = []` in `pyproject.toml` — the CPU backend has zero additional Python dependencies beyond the standard library and numpy (which is already a core dependency). The FFI mechanism is `ctypes` from the Python standard library. No cffi, no Cython, no pybind11.

### Key constraint: layout verification as pre-dispatch gate

ADR-015 mandates `_verify_layouts()` at library load time. If any struct size mismatch is detected, the CPU backend raises an exception and refuses to initialize. This converts the most dangerous FFI failure mode (silent struct corruption producing wrong numerical results) into a deterministic startup error. The renderer cannot be constructed if layout verification fails.

### Key constraint: renderer is a pure consumer of the plan

Identical to the OpenCL renderer constraint (Phase 2A): the renderer receives an immutable `ExecutionPlan` and never modifies it. All plan construction occurs in the shared layer. The renderer's sole role is to translate plan nodes into `pool_dispatch_and_wait` calls via the FFI layer.

---

## 2. Current State After Phase 3A

### CPU backend (`src/backends/cpu/`)

| Component | Status |
| :--- | :--- |
| `kernel_sources/cpu_simd.h` | Complete — SIMD abstraction layer |
| `kernel_sources/cpu_threads.h/.c` | Complete — thread pool implementation |
| `kernel_sources/cpu_kernels.h` | Complete — ABI surface, all structs and function declarations |
| `kernel_sources/phase_*.c` | Complete — all ~22 kernel implementations |
| `kernel_sources/cpu_export.h` | Complete — symbol visibility macros |
| `meson.build` | Complete — `shared_library('cpu_kernels', ...)` target |
| `libcpu_kernels.so` (build output) | Complete — compiled and installable |
| `__init__.py` | Minimal stub |

### Not yet created (Phase 3B deliverables)

| Module | Purpose |
| :--- | :--- |
| `_ffi_types.py` | ctypes struct definitions |
| `_loader.py` | Library loading + layout verification |
| `_dispatch_table.py` | `kernel_name` → `(task_fn_ptr, args_struct_class)` |
| `discovery.py` | HardwareProfile population |
| `type_mapping.py` | PrecisionConfig → numpy dtype |
| `buffer_allocator.py` | SIMD-aligned numpy array allocation |
| `retrieval.py` | CPURetrievalFuture (zero-copy) |
| `renderer.py` | CPUPlanRenderer |

---

## 3. Target Deliverables

After Phase 3B completes:

```
src/backends/cpu/
├── __init__.py                         # UPDATED: exports CPUPlanRenderer, discover_hardware
├── _ffi_types.py                       # NEW: ctypes Structure subclasses
├── _loader.py                          # NEW: library loading + _verify_layouts()
├── _dispatch_table.py                  # NEW: kernel_name → (fn_ptr, struct_cls) mapping
├── discovery.py                        # NEW: HardwareProfile from OS queries
├── type_mapping.py                     # NEW: PrecisionConfig → numpy dtype
├── buffer_allocator.py                 # NEW: SIMD-aligned numpy buffer allocation
├── retrieval.py                        # NEW: CPURetrievalFuture (zero-copy)
├── renderer.py                         # NEW: CPUPlanRenderer
├── meson.build                         # UNCHANGED (from Phase 3A)
└── kernel_sources/                     # UNCHANGED (from Phase 3A)
    └── ... (all C sources and headers)
```

---

## 4. Task Breakdown

### Step 3B.1: Implement `_ffi_types.py` — ctypes struct definitions

**Action:** Create `src/backends/cpu/_ffi_types.py` with one `ctypes.Structure` subclass per C argument struct defined in `cpu_kernels.h`.

**Design:**

```python
import ctypes
from ctypes import (
    Structure, c_float, c_uint32, c_size_t,
    POINTER, CFUNCTYPE,
)

c_float_p = POINTER(c_float)
c_uint_p = POINTER(c_uint32)
c_int_p = POINTER(ctypes.c_int32)


class ForwardPassArgs(Structure):
    """Mirrors cpu_kernels.h ForwardPassArgs."""
    _fields_ = [
        # Buffer pointers (order matches C struct)
        ("input", c_float_p),
        ("sample_mask", c_float_p),
        ("weights_shared_simd_major", c_float_p),
        ("biases_shared", c_float_p),
        ("hidden_activations", c_float_p),
        ("hidden_mask", c_float_p),
        # Scalars
        ("batch_chunk_offset", c_uint32),
        ("batch_chunk_count", c_uint32),
        ("total_batch_count", c_uint32),
        ("padded_input_count", c_uint32),
        ("padded_hidden_count", c_uint32),
    ]


class CceChunkArgs(Structure):
    """Mirrors cpu_kernels.h CceChunkArgs."""
    _fields_ = [
        ("logits", c_float_p),
        ("temps", c_float_p),
        ("targets", c_int_p),
        ("sample_mask", c_float_p),
        ("partial_probs", c_float_p),
        ("final_loss", c_float_p),
        ("flat_tile_index_base", c_uint32),
        ("num_class_chunks", c_uint32),
        ("classes_per_chunk", c_uint32),
        ("modules_per_chunk", c_uint32),
        ("total_batch_count", c_uint32),
        ("total_output_class_count", c_uint32),
        ("padded_total_output_class_count", c_uint32),
        ("total_modules_count", c_uint32),
        ("total_tile_count", c_uint32),
    ]


class AdamUpdateArgs(Structure):
    """Mirrors cpu_kernels.h AdamUpdateArgs."""
    _fields_ = [
        ("final_grad", c_float_p),
        ("parameters", c_float_p),
        ("m1", c_float_p),
        ("m2", c_float_p),
        ("learning_rate", c_float),
        ("beta1_pow_t", c_float),
        ("beta2_pow_t", c_float),
        ("beta1", c_float),
        ("beta2", c_float),
        ("epsilon", c_float),
        ("parameter_count", c_uint32),
    ]


class ReductionTreePlanFFI(Structure):
    """Mirrors cpu_kernels.h ReductionTreePlan."""
    _fields_ = [
        ("partial_collection", c_float_p),
        ("offset_lists_flat", c_uint_p),
        ("stage_offsets_into_list", c_uint_p),
        ("stage_fan_in", c_uint_p),
        ("stage_node_counts", c_uint_p),
        ("staging_buffer_0", c_float_p),
        ("staging_buffer_1", c_float_p),
        ("output", c_float_p),
        ("partial_width", c_uint32),
        ("num_stages", c_uint32),
        ("t_algorithmic", c_float),
        ("lambda_", c_float),
        ("fp_max", c_float),
        ("epsilon", c_float),
    ]


# ... ClipPartialsArgs, BackpropSharedWeightsArgs, etc.
# One Structure subclass per C struct in cpu_kernels.h.


# Task function type for pool_dispatch_and_wait callback
TaskFuncType = CFUNCTYPE(None, ctypes.c_void_p, c_uint32, c_uint32)
```

**Critical discipline:** The `_fields_` order in each Python struct **must exactly match** the field order in the C struct. A field reordering that preserves the aggregate size (e.g., swapping two `c_uint32` fields) is not caught by `_verify_layouts()` — it is caught only by Tier 2 behavioral tests (ADR-015, ADR-016). This makes the struct definitions a high-trust artifact requiring careful manual review against `cpu_kernels.h`.

**`ReductionTreePlan` C struct note:** The C struct uses `float* staging_buffers[2]` (a two-element array of pointers). ctypes does not directly support arrays as struct fields in the same way C does. The Python struct flattens this to two separate pointer fields (`staging_buffer_0`, `staging_buffer_1`), or uses `c_float_p * 2` as the field type. The C struct must match whichever representation is chosen — coordinate with Phase 3A.

---

### Step 3B.2: Implement library loader

**Action:** Create `src/backends/cpu/_loader.py` that discovers and loads `libcpu_kernels.so` at runtime.

**Loading mechanism:**

```python
import ctypes
import importlib.resources
import platform
import sys


def _library_filename() -> str:
    """Resolve the platform-specific shared library filename."""
    system = platform.system()
    if system == "Linux":
        return "libcpu_kernels.so"
    elif system == "Darwin":
        return "libcpu_kernels.dylib"
    elif system == "Windows":
        return "cpu_kernels.dll"
    raise RuntimeError(f"Unsupported platform: {system}")


def load_cpu_library() -> ctypes.CDLL:
    """Load the CPU kernel shared library.

    Discovery via importlib.resources — the library is installed
    alongside the Python package by Meson (ADR-014).
    """
    package_files = importlib.resources.files(
        "averaging_ensembled_classifier.backends.cpu"
    )
    lib_resource = package_files / _library_filename()

    # importlib.resources may return a traversable that needs
    # extraction to a real filesystem path for ctypes.CDLL
    with importlib.resources.as_file(lib_resource) as lib_path:
        lib = ctypes.CDLL(str(lib_path))

    _set_function_signatures(lib)
    return lib


def _set_function_signatures(lib: ctypes.CDLL) -> None:
    """Declare argument and return types for all exported functions.

    This enables ctypes to perform type checking and automatic
    conversion on function calls.
    """
    # Thread pool
    lib.pool_create.argtypes = [ctypes.c_uint32]
    lib.pool_create.restype = ctypes.c_void_p

    lib.pool_destroy.argtypes = [ctypes.c_void_p]
    lib.pool_destroy.restype = None

    lib.pool_dispatch_and_wait.argtypes = [
        ctypes.c_void_p,                              # ThreadPool*
        ctypes.CFUNCTYPE(None, ctypes.c_void_p,       # fn pointer
                         ctypes.c_uint32, ctypes.c_uint32),
        ctypes.c_void_p,                              # args
        ctypes.c_uint32,                              # task_count
    ]
    lib.pool_dispatch_and_wait.restype = None

    # Reduction engine
    lib.execute_reduction_tree.argtypes = [
        ctypes.c_void_p,                              # ThreadPool*
        ctypes.c_void_p,                              # ReductionTreePlan*
    ]
    lib.execute_reduction_tree.restype = None

    # Layout verification
    for name in _STRUCT_SIZE_FUNCTIONS:
        getattr(lib, name).argtypes = []
        getattr(lib, name).restype = ctypes.c_size_t

    # Task functions — all have the same signature: (void*, uint, uint) → void
    for name in _TASK_FUNCTION_NAMES:
        fn = getattr(lib, name)
        fn.argtypes = [ctypes.c_void_p, ctypes.c_uint32, ctypes.c_uint32]
        fn.restype = None
```

**Design decisions:**

1. **`importlib.resources` over hardcoded paths.** The library path is resolved from the installed package, not from a filesystem path. This works for both editable installs (`pip install -e .`) and standard installs. ADR-014 specifies that the library is installed alongside the Python package.

2. **`as_file` context manager.** `importlib.resources.as_file` extracts the resource to a temporary path if needed (e.g., when installed inside a zip). For standard installs, it returns the real path directly.

3. **Function signature declarations.** Setting `argtypes` and `restype` on every exported function enables ctypes to catch type errors at call time rather than producing silent corruption.

---

### Step 3B.3: Implement `_verify_layouts()`

**Action:** Implement the layout verification function mandated by ADR-015.

```python
from . import _ffi_types as ffi


_LAYOUT_CHECKS: list[tuple[str, type]] = [
    ("get_struct_size_forward_pass_args", ffi.ForwardPassArgs),
    ("get_struct_size_cce_chunk_args", ffi.CceChunkArgs),
    ("get_struct_size_adam_update_args", ffi.AdamUpdateArgs),
    ("get_struct_size_reduction_tree_plan", ffi.ReductionTreePlanFFI),
    # ... one entry per struct
]


def _verify_layouts(lib: ctypes.CDLL) -> None:
    """Assert Python struct sizes match C struct sizes.

    Called at library load time. Raises RuntimeError if any
    mismatch is detected, preventing the CPU backend from
    initializing with a corrupted FFI layer (ADR-015).
    """
    mismatches = []
    for c_getter_name, py_struct_cls in _LAYOUT_CHECKS:
        c_size = getattr(lib, c_getter_name)()
        py_size = ctypes.sizeof(py_struct_cls)
        if c_size != py_size:
            mismatches.append(
                f"  {py_struct_cls.__name__}: "
                f"C={c_size} bytes, Python={py_size} bytes"
            )

    if mismatches:
        detail = "\n".join(mismatches)
        raise RuntimeError(
            f"CPU kernel library struct layout mismatch detected.\n"
            f"The Python ctypes struct definitions in _ffi_types.py "
            f"do not match the compiled C struct sizes in "
            f"libcpu_kernels.\n\n"
            f"Mismatches:\n{detail}\n\n"
            f"This likely means the C header (cpu_kernels.h) was "
            f"modified without updating _ffi_types.py, or the "
            f"library was compiled with different struct packing."
        )
```

**Integration point:** `_verify_layouts(lib)` is called immediately after `load_cpu_library()` returns, before the `CPUPlanRenderer` is constructed. The verification is mandatory — there is no way to skip it.

**Limitation acknowledged:** ADR-015 notes that same-size field reorderings (e.g., swapping two `uint` fields) are not caught by this check. Tier 2 behavioral tests (Phase 3C) serve as the safety net for this class of errors.

---

### Step 3B.4: Implement dispatch table

**Action:** Create `src/backends/cpu/_dispatch_table.py` containing the mapping from plan-model kernel names to C function pointers and argument struct classes.

```python
from . import _ffi_types as ffi


def build_dispatch_table(
    lib: ctypes.CDLL,
) -> dict[str, tuple[ctypes.c_void_p, type]]:
    """Build the kernel_name → (task_fn_ptr, args_struct_class) dispatch table.

    The kernel_name keys match KernelDispatchNode.kernel_name values
    from the plan model. The task_fn_ptr is resolved from the loaded
    library. The args_struct_class is the ctypes Structure subclass
    for constructing the argument struct.
    """
    return {
        # Act phase
        "forward_pass": (_get_fn_ptr(lib, "task_forward_pass"), ffi.ForwardPassArgs),
        "render_logits_chunk": (_get_fn_ptr(lib, "task_render_logits"), ffi.RenderLogitsArgs),
        # Learn phase A — production
        "compute_probs_loss_cce_chunk": (_get_fn_ptr(lib, "task_cce_probs_loss"), ffi.CceChunkArgs),
        "compute_probs_loss_bce_chunk": (_get_fn_ptr(lib, "task_bce_probs_loss"), ffi.BceChunkArgs),
        "calculate_module_param_grads_cce": (_get_fn_ptr(lib, "task_module_param_grads"), ffi.ModuleParamGradsArgs),
        "calculate_module_param_grads_bce": (_get_fn_ptr(lib, "task_module_param_grads"), ffi.ModuleParamGradsArgs),
        # Learn phase B — processing
        "backprop_error_to_hidden_cce": (_get_fn_ptr(lib, "task_backprop_to_hidden"), ffi.BackpropToHiddenArgs),
        "backprop_error_to_hidden_bce": (_get_fn_ptr(lib, "task_backprop_to_hidden"), ffi.BackpropToHiddenArgs),
        "calculate_chunk_temp_gradients_cce": (_get_fn_ptr(lib, "task_temp_gradients"), ffi.TempGradientsArgs),
        "calculate_chunk_temp_gradients_bce": (_get_fn_ptr(lib, "task_temp_gradients"), ffi.TempGradientsArgs),
        "clip_partial_gradients": (_get_fn_ptr(lib, "task_clip_partial_grads"), ffi.ClipPartialsArgs),
        # Learn phase C — reduction
        "gather_and_permute_grad_h": (_get_fn_ptr(lib, "task_gather_permute_grad_h"), ffi.GatherPermuteArgs),
        "stabilize_reduce_grad_h": (_get_fn_ptr(lib, "task_stabilize_reduce_grad_h"), ffi.StabilizeReduceArgs),
        # Learn phase D — backprop
        "backprop_shared_weights": (_get_fn_ptr(lib, "task_backprop_shared_weights"), ffi.BackpropSharedWeightsArgs),
        "backprop_shared_biases": (_get_fn_ptr(lib, "task_backprop_shared_biases"), ffi.BackpropSharedBiasesArgs),
        "clip_shared_gradients": (_get_fn_ptr(lib, "task_clip_shared_grads"), ffi.ClipSharedGradsArgs),
        # Update phase
        "normalize_gradients": (_get_fn_ptr(lib, "task_normalize_gradients"), ffi.NormalizeGradientsArgs),
        "adam_update": (_get_fn_ptr(lib, "task_adam_update"), ffi.AdamUpdateArgs),
        "clamp_temperatures": (_get_fn_ptr(lib, "task_clamp_temperatures"), ffi.ClampTemperaturesArgs),
    }


def _get_fn_ptr(lib: ctypes.CDLL, symbol_name: str) -> ctypes.c_void_p:
    """Resolve a task function's address from the loaded library."""
    fn = getattr(lib, symbol_name)
    return ctypes.cast(fn, ctypes.c_void_p)
```

**Strategy A kernels (Nodes 8, 9, 10):** The CCE and BCE variants of Strategy A kernels map to the **same** C task function (e.g., both `calculate_module_param_grads_cce` and `calculate_module_param_grads_bce` map to `task_module_param_grads`). The `problem_type` flag is a field in the argument struct, set by the renderer from `KernelDispatchNode.scalar_params["src_scalar_FLAG_problem_type"]`. This maintains ADR-011's Strategy A semantics: one kernel, host-injected flag.

**Strategy B kernels (Nodes 6, 7):** The CCE and BCE variants map to **different** C task functions (`task_cce_probs_loss` vs. `task_bce_probs_loss`) with different argument struct types. This maintains ADR-011's Strategy B semantics: distinct kernels.

---

### Step 3B.5: Implement `discovery.py` — HardwareProfile population

**Action:** Create `src/backends/cpu/discovery.py` that populates a `HardwareProfile` from OS and hardware queries.

```python
import os
import struct

from ...shared.hardware_profile import HardwareProfile


def discover_hardware() -> HardwareProfile:
    """Populate HardwareProfile from CPU hardware characteristics."""
    return HardwareProfile(
        simd_width=_detect_simd_width(),
        cache_line_bytes=_detect_cache_line_bytes(),
        max_reduce_fan_in=_compute_max_reduce_fan_in(),
        max_local_mem_bytes=None,        # CPU has no local memory concept
        global_mem_bytes=_detect_system_memory(),
    )
```

**Field population:**

| Field | Detection Method | Fallback |
| :--- | :--- | :--- |
| `simd_width` | Query the compiled library for its `SIMD_WIDTH` constant (via an exported getter, or compile-time constant embedded in the library). Alternatively, detect from platform at Python level. | 1 (scalar) |
| `cache_line_bytes` | Linux: `/sys/devices/system/cpu/cpu0/cache/index0/coherency_line_size`. macOS: `sysctl hw.cachelinesize`. | 64 (safe default for x86-64) |
| `max_reduce_fan_in` | Computed from CPU constraints — for CPU, the reduction fan-in is bounded by practical memory considerations, not work-group limits. A reasonable default is 256 (matching many OpenCL devices). | 256 |
| `max_local_mem_bytes` | `None` — CPU has no local memory concept (ADR-006). | N/A |
| `global_mem_bytes` | `os.sysconf('SC_PAGE_SIZE') * os.sysconf('SC_PHYS_PAGES')` on Linux/macOS. `ctypes.windll.kernel32.GlobalMemoryStatusEx` on Windows. | 4 GiB fallback |

**Thread count detection (for pool sizing, not in HardwareProfile):**

```python
def detect_thread_count() -> int:
    """Determine the optimal thread pool size."""
    try:
        return len(os.sched_getaffinity(0))
    except AttributeError:
        return os.cpu_count() or 1
```

`os.sched_getaffinity(0)` respects cgroup/taskset restrictions; `os.cpu_count()` does not. The renderer uses `detect_thread_count()` when creating the thread pool.

---

### Step 3B.6: Implement `type_mapping.py` — PrecisionConfig → C type mapping

**Action:** Create `src/backends/cpu/type_mapping.py`.

```python
import numpy as np

from ...shared.precision_config import PrecisionConfig


def get_numpy_dtype(precision: PrecisionConfig) -> np.dtype:
    """Map PrecisionConfig to the numpy dtype for buffer allocation."""
    return precision.numpy_dtype


def get_c_type_name(precision: PrecisionConfig) -> str:
    """Map PrecisionConfig to the C type name (for diagnostic purposes)."""
    if precision.numpy_dtype == np.float32:
        return "float"
    if precision.numpy_dtype == np.float16:
        return "half"
    raise ValueError(f"Unsupported precision: {precision.numpy_dtype}")
```

The CPU backend initially supports only FP32. The `type_mapping` module exists to maintain structural parity with the OpenCL backend's `type_mapping.py` and to provide the extension point for FP16 when hardware support is available.

---

### Step 3B.7: Implement `buffer_allocator.py` — SIMD-aligned numpy arrays

**Action:** Create `src/backends/cpu/buffer_allocator.py` implementing SIMD-aligned buffer allocation.

```python
import numpy as np

from ...shared.buffer_lifecycle import BufferHandle, BufferDescriptor, BufferRole


class CPUBufferAllocator:
    """Allocates SIMD-aligned numpy arrays from BufferDescriptors.

    On CPU, "device memory" is host memory — numpy arrays serve as
    both the host and device representation. The critical requirement
    is SIMD alignment: the array's data pointer must be aligned to
    SIMD_ALIGNMENT bytes so that simd_load/simd_store in the C
    kernels can use aligned operations.
    """

    def __init__(self, simd_alignment: int, dtype: np.dtype) -> None:
        self._alignment = simd_alignment
        self._dtype = dtype
        self._buffers: dict[BufferHandle, np.ndarray] = {}

    def allocate(self, descriptor: BufferDescriptor) -> None:
        """Allocate a SIMD-aligned numpy array for the given descriptor."""
        total_elements = 1
        for dim in descriptor.padded_shape:
            total_elements *= dim

        buf = self._allocate_aligned(total_elements)
        self._buffers[descriptor.handle] = buf

    def get_buffer(self, handle: BufferHandle) -> np.ndarray:
        """Retrieve the numpy array for a given buffer handle."""
        return self._buffers[handle]

    def get_data_pointer(self, handle: BufferHandle):
        """Return the ctypes-compatible data pointer for FFI dispatch."""
        return self._buffers[handle].ctypes.data_as(
            ctypes.POINTER(ctypes.c_float)
        )

    def release(self, handle: BufferHandle) -> None:
        """Release a buffer (allows garbage collection)."""
        self._buffers.pop(handle, None)

    def release_all(self) -> None:
        """Release all buffers."""
        self._buffers.clear()

    def _allocate_aligned(self, num_elements: int) -> np.ndarray:
        """Allocate a numpy array with guaranteed SIMD alignment."""
        # numpy's default allocator typically aligns to 64 bytes on
        # modern platforms, but this is not guaranteed. Use a
        # deliberate over-allocation + alignment strategy:
        arr = np.empty(num_elements, dtype=self._dtype)

        # Verify alignment
        if arr.ctypes.data % self._alignment != 0:
            # Fallback: over-allocate and slice to aligned offset
            extra = self._alignment // arr.itemsize
            padded = np.empty(num_elements + extra, dtype=self._dtype)
            offset = (self._alignment
                      - padded.ctypes.data % self._alignment) // arr.itemsize
            arr = padded[offset:offset + num_elements]

        return arr
```

**Zero-copy for MODEL_STATE buffers:** Model state buffers (weights, biases, optimizer moments) are numpy arrays owned by the Python-side parameter manager. The allocator records a reference to these existing arrays rather than allocating new ones. The `ctypes.data` pointer from the existing array is passed directly to the C kernels — no data copy occurs.

**Design decision — numpy alignment guarantees:** Modern numpy (≥1.20) allocates arrays with 64-byte alignment when using the default allocator. The allocator verifies alignment and falls back to over-allocation + slicing only if the default allocation is under-aligned. This avoids the complexity of a custom allocator while maintaining the SIMD alignment guarantee.

---

### Step 3B.8: Implement `CPURetrievalFuture` — zero-copy

**Action:** Create `src/backends/cpu/retrieval.py` implementing the `RetrievalFuture` Protocol for CPU.

```python
import numpy as np
from numpy.typing import NDArray

from ...shared.retrieval_future import RetrievalFuture


class CPURetrievalFuture:
    """Zero-copy RetrievalFuture for the CPU backend.

    On CPU, all computation is synchronous — by the time render()
    returns, all results are already in host memory. The "future"
    is immediately resolved. No D2H transfer is needed.

    .result() returns an unpadded view of the output buffer,
    stripping SIMD/cache padding to return the logical shape.
    """

    def __init__(
        self,
        node_id: str,
        padded_buffer: np.ndarray,
        logical_shape: tuple[int, ...],
    ) -> None:
        self._node_id = node_id
        self._padded_buffer = padded_buffer
        self._logical_shape = logical_shape
        self._released = False

    @property
    def node_id(self) -> str:
        return self._node_id

    def wait(self) -> None:
        """No-op — CPU computation is synchronous."""
        pass

    def result(self) -> NDArray[np.floating]:
        """Return unpadded numpy array with logical shape."""
        if self._released:
            raise RuntimeError(
                f"RetrievalFuture for '{self._node_id}' has been released"
            )
        # Slice the padded buffer to the logical shape
        slices = tuple(slice(0, dim) for dim in self._logical_shape)
        return self._padded_buffer.reshape(-1)[:np.prod(self._logical_shape)].reshape(
            self._logical_shape
        )

    def release(self) -> None:
        """Release the reference to the backing buffer."""
        self._padded_buffer = None
        self._released = True
```

**Key property:** `wait()` is a no-op. CPU dispatch is blocking — `pool_dispatch_and_wait` returns only when all tasks complete. There is no asynchronous completion to wait on. The "future" pattern exists solely to satisfy the `RetrievalFuture` Protocol, maintaining the backend-neutral interface.

**Padding stripping:** The `result()` method strips SIMD/cache padding by slicing the flat buffer to the product of logical dimensions, then reshaping. This mirrors the OpenCL backend's `OpenCLRetrievalFuture.result()` which strips padding during the D2H transfer.

---

### Step 3B.9: Implement `CPUPlanRenderer` — DAG traversal + FFI dispatch

**Action:** Create `src/backends/cpu/renderer.py` implementing the `PlanRenderer` Protocol.

**Renderer structure:**

```python
import ctypes
from typing import Any

from ...shared.plan_types import (
    ExecutionPlan,
    KernelDispatchNode,
    ReductionTreeNode,
    StreamingLoopNode,
    BarrierNode,
    RetrievalNode,
)
from ...shared.retrieval_future import RetrievalFuture
from .buffer_allocator import CPUBufferAllocator
from .retrieval import CPURetrievalFuture
from ._loader import load_cpu_library, _verify_layouts
from ._dispatch_table import build_dispatch_table
from ._ffi_types import ReductionTreePlanFFI


class CPUPlanRenderer:
    """Plan renderer for the CPU backend (ADR-001, ADR-015).

    Traverses an ExecutionPlan DAG in topological order and dispatches
    each node via the CPU kernel library's pool_dispatch_and_wait.
    """

    def __init__(self, thread_count: int | None = None) -> None:
        # Load and verify the native library
        self._lib = load_cpu_library()
        _verify_layouts(self._lib)

        # Build the dispatch table
        self._dispatch_table = build_dispatch_table(self._lib)

        # Create the thread pool
        if thread_count is None:
            from .discovery import detect_thread_count
            thread_count = detect_thread_count()
        self._pool = self._lib.pool_create(thread_count)
        self._thread_count = thread_count

    def __del__(self) -> None:
        if hasattr(self, "_pool") and self._pool:
            self._lib.pool_destroy(self._pool)
            self._pool = None

    def render(self, plan: ExecutionPlan) -> dict[str, RetrievalFuture]:
        """Render an execution plan via CPU dispatch.

        Allocates SIMD-aligned numpy buffers, traverses the plan's
        topological order, and dispatches each node type:

        - KernelDispatchNode → construct args struct, pool_dispatch_and_wait
        - ReductionTreeNode  → translate to C ReductionTreePlan, execute_reduction_tree
        - StreamingLoopNode  → iterate chunks, dispatch body nodes per chunk
        - BarrierNode        → no-op (synchronization is implicit)
        - RetrievalNode      → create CPURetrievalFuture (zero-copy)
        """
        # Allocate buffers
        allocator = CPUBufferAllocator(
            simd_alignment=plan.hardware.cache_line_bytes,
            dtype=plan.precision.numpy_dtype,
        )
        for descriptor in plan.buffers.values():
            allocator.allocate(descriptor)

        # Traverse DAG
        futures: dict[str, RetrievalFuture] = {}

        for node_id in plan.topological_order:
            node = plan.nodes[node_id]

            if isinstance(node, KernelDispatchNode):
                self._render_kernel_dispatch(node, allocator, plan)

            elif isinstance(node, ReductionTreeNode):
                self._render_reduction_tree(node, allocator, plan)

            elif isinstance(node, StreamingLoopNode):
                self._render_streaming_loop(node, allocator, plan)

            elif isinstance(node, BarrierNode):
                pass  # No-op — synchronization is implicit

            elif isinstance(node, RetrievalNode):
                future = CPURetrievalFuture(
                    node_id=node.node_id,
                    padded_buffer=allocator.get_buffer(node.source_buffer),
                    logical_shape=node.logical_shape,
                )
                futures[node.event_name] = future

        return futures

    def _render_kernel_dispatch(
        self,
        node: KernelDispatchNode,
        allocator: CPUBufferAllocator,
        plan: ExecutionPlan,
    ) -> None:
        """Dispatch a KernelDispatchNode via pool_dispatch_and_wait."""
        fn_ptr, struct_cls = self._dispatch_table[node.kernel_name]

        # Construct the argument struct
        args = self._marshal_args(struct_cls, node, allocator, plan)

        # Dispatch all tiles
        self._lib.pool_dispatch_and_wait(
            self._pool,
            ctypes.cast(fn_ptr, ctypes.CFUNCTYPE(
                None, ctypes.c_void_p, ctypes.c_uint32, ctypes.c_uint32
            )),
            ctypes.byref(args),
            node.tile_count,
        )

    def _marshal_args(
        self,
        struct_cls: type,
        node: KernelDispatchNode,
        allocator: CPUBufferAllocator,
        plan: ExecutionPlan,
    ) -> ctypes.Structure:
        """Construct a ctypes argument struct from plan node parameters.

        Maps buffer_bindings to data pointers via the allocator,
        and scalar_params to struct fields.
        """
        args = struct_cls()

        # Set buffer pointer fields
        for field_name, field_type in struct_cls._fields_:
            if hasattr(field_type, '_type_') and field_type._type_ == 'f':
                # This is a POINTER(c_float) — look up in buffer_bindings
                if field_name in node.buffer_bindings:
                    handle = node.buffer_bindings[field_name]
                    setattr(args, field_name,
                            allocator.get_data_pointer(handle))

        # Set scalar fields
        for param_name, value in node.scalar_params.items():
            # Map contract param names to struct field names
            field_name = self._contract_param_to_field(param_name)
            if hasattr(args, field_name):
                setattr(args, field_name, value)

        return args

    def _render_reduction_tree(
        self,
        node: ReductionTreeNode,
        allocator: CPUBufferAllocator,
        plan: ExecutionPlan,
    ) -> None:
        """Render a ReductionTreeNode via execute_reduction_tree."""
        rtp = node.reduction_plan

        # Translate the shared-layer ReductionTreePlan to the C struct
        c_plan = ReductionTreePlanFFI()
        c_plan.partial_collection = allocator.get_data_pointer(
            rtp.source_buffer
        )
        c_plan.output = allocator.get_data_pointer(rtp.destination_buffer)
        c_plan.partial_width = rtp.partial_width
        c_plan.num_stages = rtp.num_stages
        c_plan.t_algorithmic = rtp.threshold_schedule[0] or 0.0
        # ... populate remaining fields from rtp

        self._lib.execute_reduction_tree(self._pool, ctypes.byref(c_plan))

    def _render_streaming_loop(
        self,
        node: StreamingLoopNode,
        allocator: CPUBufferAllocator,
        plan: ExecutionPlan,
    ) -> None:
        """Render a StreamingLoopNode by iterating chunks."""
        sp = node.streaming_plan
        for chunk_index in range(sp.iteration.chunk_count):
            for body_node_id in sp.body:
                body_node = plan.nodes[body_node_id]
                if isinstance(body_node, KernelDispatchNode):
                    # Apply per-chunk parameter strides
                    adjusted_node = self._apply_strides(
                        body_node, sp, chunk_index
                    )
                    self._render_kernel_dispatch(
                        adjusted_node, allocator, plan
                    )
```

**Key design decisions:**

1. **Blocking dispatch = implicit synchronization.** Every `pool_dispatch_and_wait` call blocks until completion. The renderer traverses nodes in topological order, and each node's dispatch completes before the next begins. There is no need for an explicit event graph — the blocking model provides the synchronization guarantee.

2. **Argument marshalling via contract-to-field mapping.** The renderer maps plan-model parameter names (e.g., `src_scalar_NATURAL_padded_hidden_count`) to struct field names (e.g., `padded_hidden_count`). The mapping strips the CONTRACT.md naming prefix to derive the C struct field name. This is a thin translation layer — the struct field names were chosen (in Phase 3A) to correspond to the contract's `[ContextAndUsage]` suffix.

3. **Reduction tree translation.** The shared-layer `ReductionTreePlan` frozen dataclass is translated to the C `ReductionTreePlan` struct. This involves:
   - Converting Python tuples to ctypes arrays for offset lists.
   - Allocating staging buffers (ping-pong) as SIMD-aligned numpy arrays.
   - Setting policy parameters (`t_algorithmic`, `lambda`, `fp_max`, `epsilon`) from the plan.

4. **Streaming loop iteration in Python.** Unlike the OpenCL renderer (which may batch streaming loop iterations into command buffers), the CPU renderer iterates chunks in Python. This is acceptable because `pool_dispatch_and_wait` is the dominant cost — the Python loop overhead is negligible compared to the C dispatch overhead per chunk.

---

### Step 3B.10: Update `_build_config.py` template

**Action:** Ensure the `_build_config.py.in` Meson template and generated `_build_config.py` include `BACKEND_CPU`.

The template (`src/_build_config.py.in`) should include:

```python
BACKEND_CPU: bool = @BACKEND_CPU@
```

And the `meson.build` configuration should populate this based on whether the CPU backend was built:

```meson
build_config_data = configuration_data()
build_config_data.set('BACKEND_CPU', backend_cpu.allowed() ? 'True' : 'False')
# ... existing BACKEND_OPENCL, BACKEND_VULKAN entries
```

**Validation:** After a build with `-Daec_backend_cpu=enabled`, `_build_config.BACKEND_CPU` is `True`. After a build with `-Daec_backend_cpu=disabled`, it is `False`.

---

### Step 3B.11: Update `src/backends/cpu/__init__.py` exports

**Action:** Update the CPU backend's `__init__.py` to export the public API surface.

```python
"""CPU backend for the averaging ensembled classifier.

Provides CPUPlanRenderer — a PlanRenderer implementation that
dispatches execution plans via a native C kernel library using
ctypes FFI and a persistent thread pool.
"""
from .renderer import CPUPlanRenderer
from .discovery import discover_hardware, detect_thread_count

__all__ = ["CPUPlanRenderer", "discover_hardware", "detect_thread_count"]
```

---

### Step 3B.12: Write infrastructure smoke tests

**Action:** Create smoke tests that validate the CPU backend's infrastructure without full per-kernel correctness testing (that's Phase 3C).

**Test categories:**

1. **Library loading test:** Verify that `load_cpu_library()` succeeds and the returned `CDLL` object has all expected symbols.

2. **Layout verification test:** Verify that `_verify_layouts()` passes (sizes match). Also test the failure path: intentionally define a struct with the wrong size and assert the error is raised.

3. **Thread pool lifecycle test:** Create a pool, dispatch a trivial task (e.g., increment a counter), verify completion, destroy the pool.

4. **Buffer allocator test:** Allocate a buffer from a `BufferDescriptor`, verify the numpy array has the correct shape and dtype, verify SIMD alignment of the data pointer.

5. **CPURetrievalFuture test:** Create a future from a padded buffer, verify `.wait()` is a no-op, verify `.result()` returns the unpadded logical shape, verify `.release()` prevents subsequent `.result()` calls.

6. **Renderer plan traversal test:** Construct a minimal `ExecutionPlan` with a single `BarrierNode` and a `RetrievalNode` (no kernel dispatch). Verify the renderer traverses without error and returns the expected `RetrievalFuture` mapping.

7. **HardwareProfile population test:** Call `discover_hardware()` and verify all fields have reasonable values (`simd_width > 0`, `cache_line_bytes > 0`, `max_local_mem_bytes is None`).

**Test location:** `tests/tier2/cpu/test_cpu_infrastructure.py` (or `tests/tier2/cpu/conftest.py` + test files, following the Phase 2A pattern).

**Skip logic:** `pytest.mark.skipif(not BACKEND_CPU, reason="CPU backend not available")`.

---

### Step 3B.13: Validate rollback gate

**Action:** Verify the Phase 3B rollback gate:

1. **Tier 1 green** — no regressions to the 332 existing tests.
2. **Library loads** — `load_cpu_library()` succeeds on the build platform.
3. **Layout verification passes** — `_verify_layouts()` raises no errors.
4. **Infrastructure smoke tests pass** — the ~7 smoke tests from Step 3B.12 are green.
5. **`_build_config.BACKEND_CPU` is `True`** — the build configuration correctly reflects the enabled CPU backend.
6. **`isinstance(CPUPlanRenderer(), PlanRenderer)` returns `True`** — the renderer satisfies the Protocol.

Full per-kernel numerical correctness (CPU Tier 2) is validated in Phase 3C.

---

## 5. FFI Architecture

The CPU backend's FFI layer follows a strict layered design:

```
┌──────────────────────────────────────────────┐
│  CPUPlanRenderer (renderer.py)               │
│  Traverses ExecutionPlan DAG                 │
│  Calls dispatch table entries                │
├──────────────────────────────────────────────┤
│  Dispatch Table (_dispatch_table.py)         │
│  kernel_name → (task_fn_ptr, struct_cls)     │
├──────────────────────────────────────────────┤
│  Argument Marshalling (in renderer.py)       │
│  plan node → ctypes struct instance          │
├──────────────────────────────────────────────┤
│  ctypes Struct Definitions (_ffi_types.py)   │
│  Python-side mirror of C argument structs    │
├──────────────────────────────────────────────┤
│  Layout Verification (_loader.py)            │
│  sizeof(Python) == get_struct_size_*(C)      │
├──────────────────────────────────────────────┤
│  Library Loading (_loader.py)                │
│  importlib.resources → ctypes.CDLL           │
├──────────────────────────────────────────────┤
│  libcpu_kernels.so (Phase 3A)               │
│  Thread pool + task functions + reduction    │
└──────────────────────────────────────────────┘
```

Each layer has a single responsibility:
- **Library loading** — find and open the platform-specific shared library.
- **Layout verification** — catch struct drift before any dispatch.
- **Struct definitions** — provide Python-side type descriptions matching C.
- **Argument marshalling** — translate plan parameters to struct fields.
- **Dispatch table** — map plan-model kernel names to C function pointers.
- **Renderer** — orchestrate the DAG traversal and dispatch sequence.

---

## 6. Struct Marshalling Discipline

The argument marshalling process translates a `KernelDispatchNode` into a populated ctypes struct. The translation has two phases:

**Phase 1: Buffer pointer resolution.**

For each buffer pointer field in the struct, the renderer:
1. Looks up the buffer name in `node.buffer_bindings` → `BufferHandle`.
2. Resolves the handle to a numpy array via `allocator.get_buffer(handle)`.
3. Extracts the data pointer via `.ctypes.data_as(POINTER(c_float))`.
4. Assigns the pointer to the struct field.

**Phase 2: Scalar value assignment.**

For each scalar field in the struct, the renderer:
1. Looks up the scalar name in `node.scalar_params`.
2. Converts the value to the appropriate ctypes type (`c_uint32` for `NATURAL` scalars, `c_float` for `REAL` scalars, `c_uint32` for `FLAG` scalars).
3. Assigns the value to the struct field.

**Contract name → struct field name mapping:**

The plan model uses CONTRACT.md Article 2 naming (e.g., `src_scalar_NATURAL_padded_hidden_count`). The C struct uses abbreviated field names (e.g., `padded_hidden_count`). The renderer strips the `src_scalar_NATURAL_` / `src_scalar_REAL_` / `src_scalar_FLAG_` / `src_buffer_GLOBAL_` prefix to derive the struct field name.

This mapping is a deterministic string transformation, not a lookup table — reducing maintenance burden when new parameters are added. If a mapping fails (no matching struct field), the renderer raises a descriptive error during marshalling, not during dispatch.

---

## 7. Buffer Allocation Strategy

The CPU buffer allocator differs fundamentally from the OpenCL allocator:

| Aspect | OpenCL Allocator | CPU Allocator |
| :--- | :--- | :--- |
| Backing store | `cl.Buffer` (device memory) | `numpy.ndarray` (host memory) |
| Alignment | Implicit (driver-managed) | Explicit: `SIMD_ALIGNMENT` bytes |
| D2H transfer | Required (async read event) | Not needed (zero-copy) |
| Upload | `cl.enqueue_write_buffer` | Direct array assignment |
| Pointer for dispatch | `cl.Buffer` object | `.ctypes.data_as(POINTER(c_float))` |

**MODEL_STATE buffers (weights, biases, optimizer moments):** These exist as numpy arrays in the Python parameter manager. The CPU allocator registers a reference to the existing array rather than copying — the C kernel reads/writes the same memory. This is the zero-copy model described in DESIGN.md §7.3.

**BATCH_INPUT buffers:** The input data numpy array is registered directly. No copy.

**BATCH_INTERMEDIATE buffers:** Newly allocated SIMD-aligned numpy arrays. Freed after the plan rendering completes.

**BATCH_OUTPUT buffers:** Newly allocated. The `CPURetrievalFuture` holds a reference; freed when `.release()` is called.

---

## 8. Dispatch Flow

A complete kernel dispatch through the CPU backend proceeds as:

```
1. CPUPlanRenderer.render(plan)
   │
   2. For each node in plan.topological_order:
   │   │
   │   3. [KernelDispatchNode]
   │   │   ├── Look up (task_fn_ptr, struct_cls) in dispatch_table
   │   │   ├── Construct struct_cls instance
   │   │   ├── Set buffer pointer fields from allocator
   │   │   ├── Set scalar value fields from node.scalar_params
   │   │   └── Call: lib.pool_dispatch_and_wait(pool, task_fn_ptr, &args, tile_count)
   │   │         │
   │   │         4. [Inside C library]
   │   │         │   ├── Wake worker threads
   │   │         │   ├── Workers atomically claim task indices
   │   │         │   ├── Each worker calls: task_fn(args, task_index, thread_id)
   │   │         │   │   ├── Cast args to typed struct pointer
   │   │         │   │   ├── Execute SIMD-vectorized kernel logic
   │   │         │   │   └── Write results to output buffer pointers
   │   │         │   └── Last worker signals completion
   │   │         └── pool_dispatch_and_wait returns (all tasks done)
   │   │
   │   5. [ReductionTreeNode]
   │   │   ├── Translate plan's ReductionTreePlan → C ReductionTreePlan struct
   │   │   ├── Allocate staging buffers
   │   │   └── Call: lib.execute_reduction_tree(pool, &c_plan)
   │   │         └── Internally performs staged pool_dispatch_and_wait calls
   │   │
   │   6. [StreamingLoopNode]
   │   │   ├── For each chunk in iteration.chunk_count:
   │   │   │   ├── Apply parameter strides to body node scalar_params
   │   │   │   └── Dispatch body nodes via step 3 above
   │   │   └── (All chunks complete)
   │   │
   │   7. [BarrierNode] → no-op
   │   │
   │   8. [RetrievalNode]
   │       └── Create CPURetrievalFuture(padded_buffer, logical_shape)
   │
   9. Return: {event_name → CPURetrievalFuture}
```

---

## 9. Risk Register

| Risk | Likelihood | Impact | Mitigation |
| :--- | :--- | :--- | :--- |
| Struct field order mismatch between `_ffi_types.py` and `cpu_kernels.h` | Medium | **Critical** — silent numerical corruption | `_verify_layouts()` catches size mismatches; Tier 2 tests (Phase 3C) catch field reorderings; careful manual review of struct definitions against the header |
| numpy alignment not guaranteed on all platforms | Low | High — segfault on `simd_load` of unaligned data | Explicit alignment check in `_allocate_aligned()` with over-allocation fallback |
| `importlib.resources` fails in editable install | Low | Medium — library not discoverable | Test both editable and standard install paths; `as_file` context manager handles extraction if needed |
| Library symbol resolution fails on Windows | Low | Medium — CPU backend unavailable on Windows | Meson handles platform-specific naming; `cpu_export.h` provides `__declspec(dllexport/dllimport)` |
| Thread pool creation fails (insufficient OS resources) | Low | Medium — CPU backend unusable | Graceful error message with fallback to single-threaded dispatch (pool of 1 thread) |
| Pointer lifetime: numpy array garbage-collected while C kernel is running | Low | **Critical** — use-after-free / segfault | The allocator holds strong references to all arrays; arrays are not freed until `release_all()` is called after rendering completes |
| Streaming loop Python iteration overhead | Low | Low — Python loop is ~µs per chunk; C dispatch is ~ms | Acceptable; optimization (batched dispatch) can be added post-Phase 3 if profiling shows overhead |
| `os.sched_getaffinity` not available on macOS/Windows | Medium | Low — falls back to `os.cpu_count()` | `try/except AttributeError` in `detect_thread_count()` |
