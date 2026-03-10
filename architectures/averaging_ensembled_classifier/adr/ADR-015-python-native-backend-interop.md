# ADR-015: Python ↔ Native Backend Interop

**Status:** ACCEPTED  
**Date:** 2026-03-10  
**Deciders:** —  
**Supersedes:** —  
**Blocked by:** ADR-012, ADR-013, ADR-014  
**Blocks:** ADR-017

---

## Context

The three target backends each require a Python ↔ native code interop mechanism. The backends differ fundamentally in how their native code is compiled, loaded, and invoked:

- **OpenCL** — kernel source code is compiled at runtime by the OpenCL driver via `clBuildProgram`. The established Python binding is PyOpenCL, which wraps the OpenCL C API with Pythonic objects (`cl.Context`, `cl.CommandQueue`, `cl.Buffer`, `cl.Event`). PyOpenCL is already a core dependency in `pyproject.toml`. No FFI decision is needed — the interop mechanism is PyOpenCL and has been since the architecture's inception.

- **Vulkan** — GLSL compute shaders are compiled to SPIR-V at build time (ADR-014). The Python backend loads SPIR-V modules and interacts with the Vulkan API via `vulkan-python`, already specified as an optional dependency in ADR-014's `pyproject.toml` extension (`vulkan = ["vulkan-python>=0.2.0"]`). `vulkan-python` provides typed Python bindings generated from the Vulkan XML registry, wrapping `VkDevice`, `VkQueue`, `VkBuffer`, etc. No FFI decision is needed — the interop mechanism is `vulkan-python`.

- **CPU** — C kernel implementations are compiled into a shared library (`libcpu_kernels.so` / `libcpu_kernels.dylib` / `cpu_kernels.dll`) at build time (ADR-014). The library exports a stable C ABI. No established Python binding package exists for this custom library. The CPU backend's `renderer.py` must load this library at runtime and call its exported functions from Python. **This is the open FFI decision.**

### The CPU ABI surface

ADR-013 establishes the CPU kernel source layout in `src/backends/cpu/kernel_sources/`, with `cpu_kernels.h` defining the C ABI surface. ADR-014 specifies the shared library build: Meson's `shared_library('cpu_kernels', ...)` with hidden symbol visibility and explicit `CPU_KERNELS_EXPORT` macros on public entry points. CPU_BACKEND.md provides the complete interface specification.

The exported ABI comprises three categories of symbols:

**1. Thread pool lifecycle (~3 symbols).**

| Symbol | Signature | Purpose |
| :--- | :--- | :--- |
| `pool_create` | `ThreadPool* pool_create(uint num_threads)` | Create a persistent thread pool |
| `pool_destroy` | `void pool_destroy(ThreadPool* pool)` | Destroy the thread pool and join all threads |
| `pool_dispatch_and_wait` | `void pool_dispatch_and_wait(ThreadPool* pool, void (*fn)(void*, uint, uint), void* args, uint task_count)` | Submit N independent tasks and block until all complete |

**2. Per-kernel task functions (~20 symbols).**

Each kernel follows the uniform signature convention from CPU_BACKEND.md:

```c
void task_<kernel_name>(void* args, uint task_index, uint thread_id);
```

The `void* args` parameter is a pointer to a typed argument struct specific to that kernel (e.g., `ForwardPassArgs`, `CceChunkArgs`, `AdamUpdateArgs`). The full inventory from `cpu_kernels.h`:

| Symbol | Argument struct | DAG node(s) |
| :--- | :--- | :--- |
| `task_forward_pass` | `ForwardPassArgs` | Node 4 |
| `task_render_logits` | (per-kernel struct) | Node 5 |
| `task_cce_probs_loss` | `CceChunkArgs` | Node 6 (Strategy B) |
| `task_bce_probs_loss` | (per-kernel struct) | Node 7 (Strategy B) |
| `task_module_param_grads` | (per-kernel struct) | Node 8 (Strategy A) |
| `task_backprop_to_hidden` | (per-kernel struct) | Node 9 (Strategy A) |
| `task_temp_gradients` | (per-kernel struct) | Node 10 (Strategy A) |
| `task_clip_partial_grads` | `ClipPartialsArgs` | Node 11 |
| `task_gather_permute_grad_h` | (per-kernel struct) | Node 13 |
| `task_reduce_one_node` | (per-stage struct) | Nodes 14/15/20 |
| `task_clip_one_node` | (per-stage struct) | Nodes 15/20 |
| `task_stabilize_reduce_grad_h` | (per-kernel struct) | Node 16 |
| `task_backprop_shared_weights` | (per-kernel struct) | Node 17 |
| `task_backprop_shared_biases` | (per-kernel struct) | Node 18 |
| `task_clip_shared_grads` | (per-kernel struct) | Node 19 |
| `task_normalize_gradients` | (per-kernel struct) | Node 21 |
| `task_adam_update` | `AdamUpdateArgs` | Node 24 |
| `task_clamp_temperatures` | (per-kernel struct) | Node 25 |

**3. Reduction engine (~1 symbol).**

| Symbol | Signature | Purpose |
| :--- | :--- | :--- |
| `execute_reduction_tree` | `void execute_reduction_tree(ThreadPool* pool, ReductionTreePlan* plan)` | Execute a complete staged reduction with per-stage clipping |

The total exported ABI surface is **~24 symbols**. This is a moderately-sized, stable ABI — the symbol count is bounded by the kernel inventory from ADR-013 and the thread pool contract from CPU_BACKEND.md.

### Argument struct marshalling

The `task_*` functions accept `void* args`, which is cast internally to the appropriate typed struct pointer. Each struct contains two kinds of fields:

- **Buffer pointers** (`const float*`, `float*`, `const int*`, `const uint*`) — point into numpy-managed memory on the Python side. The `PlanRenderer` allocates numpy arrays as the buffer backing store (via `buffer_allocator.py`) and passes their data pointers through the FFI layer.
- **Scalar values** (`uint`, `float`) — plan-derived parameters (batch counts, dimensions, thresholds, flag values).

The `ForwardPassArgs` struct from CPU_BACKEND.md illustrates the pattern:

```c
typedef struct {
    const float* input;                      // → numpy array .ctypes.data
    const float* sample_mask;                // → numpy array .ctypes.data
    const float* weights_shared_simd_major;  // → numpy array .ctypes.data
    const float* biases_shared;              // → numpy array .ctypes.data
    float*       hidden_activations;         // → numpy array .ctypes.data
    float*       hidden_mask;                // → numpy array .ctypes.data
    uint batch_chunk_offset;
    uint batch_chunk_count;
    uint total_batch_count;
    uint padded_input_count;
    uint padded_hidden_count;
} ForwardPassArgs;
```

The Python side must construct these structs with:
1. Correct field order and types matching the C memory layout.
2. Valid pointers to SIMD-aligned numpy array memory (alignment guaranteed by the buffer allocator).
3. Correct scalar values derived from the execution plan's `KernelDispatchNode.scalar_params`.

The `ReductionTreePlan` struct is the most complex, containing pointers to flattened offset arrays and staging buffers, plus scalar policy parameters (`t_algorithmic`, `lambda`, `fp_max`). Its Python-side construction mirrors the shared layer's `ReductionTreePlan` frozen dataclass (ADR-003), translated to C struct layout.

### Strategy A FLAG marshalling

ADR-011 established Strategy A (unified kernel with `FLAG_problem_type` scalar) for Nodes 8, 9, and 10. In the CPU backend, this flag is a `uint` field in the kernel's argument struct. The Python side sets this field from the `KernelDispatchNode.scalar_params["src_scalar_FLAG_problem_type"]` value. No special FFI handling is needed — the flag is an ordinary unsigned integer scalar marshalled identically to any other `uint` field.

The marshalling discipline is important here: if the Python struct definition has the `FLAG_problem_type` field at a different offset than the C definition, the kernel receives an incorrect flag value. This would manifest as CCE logic executing for a BCE problem or vice versa — a silent correctness failure. The FFI layer must verify struct layout consistency (see §Decision, Layout verification).

### ADR-012 isolation boundary

ADR-012 confines each backend's Python-facing code behind the `PlanRenderer` interface. The shared layer's `orchestrator.py` and `plan_builder.py` never see which FFI library is in use. The FFI mechanism is entirely internal to `src/backends/cpu/`:

```
src/backends/cpu/
├── renderer.py           ← PlanRenderer: walks plan DAG, dispatches via FFI
├── buffer_allocator.py   ← Allocates SIMD-aligned numpy arrays
├── discovery.py          ← Populates HardwareProfile (thread count, L1 size)
├── type_mapping.py       ← PrecisionConfig.numpy_dtype → C type mapping
├── retrieval.py          ← RetrievalFuture via zero-copy numpy view
└── kernel_bindings/      ← KernelBinding: struct construction + dispatch call
```

The FFI choice affects only `renderer.py` (library loading, pool lifecycle) and `kernel_bindings/` (struct construction, function calls). No module outside `src/backends/cpu/` is aware of the FFI mechanism.

### ADR-014 packaging constraints

ADR-014 specifies:
- The shared library is installed to `averaging_ensembled_classifier/backends/cpu/` via `py.get_install_dir()`.
- Runtime discovery via `importlib.resources.files('averaging_ensembled_classifier.backends.cpu')` with platform-specific filename resolution.
- The `_build_config.BACKEND_CPU` boolean gates whether the CPU backend is available.
- `pyproject.toml` specifies `cpu = []` — no additional Python dependencies for the CPU backend.

The last point is significant: **ADR-014 established that the CPU backend has zero additional Python dependencies.** Any FFI mechanism that requires adding a package to `cpu = [...]` departs from this established contract.

---

## Decision Drivers

1. **ADR-012 (Backend isolation boundary).** The FFI mechanism is an internal concern of `src/backends/cpu/`, invisible to `src/shared/` and `orchestrator.py`. The PlanRenderer interface abstracts all dispatch mechanics. This means the FFI choice has no cross-cutting impact — it is a local decision within the CPU backend.

2. **ADR-014 (Zero additional CPU dependencies).** ADR-014's `pyproject.toml` specifies `cpu = []`. Introducing a Python dependency for the FFI mechanism (e.g., `cffi`) contradicts this established contract. If cffi is chosen, ADR-014's dependency declaration must be amended — a cross-ADR modification.

3. **ADR-014 (Pre-built library, runtime loading).** The shared library is pre-built by Meson and installed as a binary artifact. The FFI mechanism must load a pre-built `.so`/`.dylib`/`.dll` at runtime without any compilation step at import time. This eliminates FFI mechanisms that require source compilation at load time.

4. **ADR-013 (`cpu_kernels.h` ABI surface).** The ABI surface is defined by `cpu_kernels.h`: ~20 `task_*` functions with uniform `(void*, uint, uint)` signature, typed argument structs, thread pool lifecycle functions, and the reduction engine entry point. The FFI mechanism must faithfully marshal these types.

5. **CPU_BACKEND.md (`pool_dispatch_and_wait` function pointer pattern).** The `pool_dispatch_and_wait` function accepts a function pointer of type `void (*fn)(void*, uint, uint)`. The Python-side renderer needs to pass C function pointers (the `task_*` symbols from the same library) to this function. The FFI mechanism must support passing C function pointers obtained from the loaded library.

6. **ADR-001 (Per-backend FFI autonomy).** ADR-001 and ADR-012 establish that each backend's internals are its own concern. The OpenCL backend uses PyOpenCL; the Vulkan backend uses `vulkan-python`. There is no architectural requirement for FFI mechanism uniformity across backends. The CPU backend is free to choose the mechanism most natural for its ABI surface.

7. **CONCEPT.md §1 (Architectural Elegance Feedback).** Complexity should not be introduced speculatively. If the chosen FFI mechanism creates maintenance pressure (e.g., struct layout drift, marshalling errors), the response is to formalize the solution as an architectural primitive — not to preemptively adopt a heavier mechanism.

8. **ADR-016 (Test strategy, pending).** Tier 2 per-backend tests exercise each kernel independently. Struct marshalling errors (wrong field order, wrong type size, misaligned offsets) would be caught by Tier 2 tests as incorrect numerical results. The test strategy provides a behavioral verification layer for FFI correctness.

---

## Options Considered

### Option A: cffi ABI mode

Use cffi's ABI (Application Binary Interface) mode to load the pre-built `libcpu_kernels.so` at runtime. The Python side declares the C types and function signatures using cffi's `ffi.cdef()` and loads the library with `ffi.dlopen()`:

```python
from cffi import FFI
ffi = FFI()

ffi.cdef("""
typedef struct {
    const float* input;
    const float* sample_mask;
    const float* weights_shared_simd_major;
    const float* biases_shared;
    float*       hidden_activations;
    float*       hidden_mask;
    unsigned int batch_chunk_offset;
    unsigned int batch_chunk_count;
    unsigned int total_batch_count;
    unsigned int padded_input_count;
    unsigned int padded_hidden_count;
} ForwardPassArgs;

typedef ... ThreadPool;

ThreadPool* pool_create(unsigned int num_threads);
void pool_destroy(ThreadPool* pool);
void pool_dispatch_and_wait(ThreadPool* pool,
                            void (*fn)(void*, unsigned int, unsigned int),
                            void* args, unsigned int task_count);

void task_forward_pass(void* args, unsigned int task_index, unsigned int thread_id);
// ... ~20 more task_* declarations
""")

lib = ffi.dlopen(str(lib_path))
```

Usage:

```python
args = ffi.new("ForwardPassArgs *")
args.input = ffi.cast("const float *", input_array.ctypes.data)
args.batch_chunk_count = node.scalar_params["batch_chunk_count"]
# ...
lib.pool_dispatch_and_wait(pool, lib.task_forward_pass, ffi.cast("void *", args), task_count)
```

**Advantages:**
- Loads pre-built library directly — no compile step at import time. Compatible with ADR-014's packaging.
- Struct construction via `ffi.new()` returns garbage-collected objects, simplifying lifetime management.
- The `ffi.cdef()` declarations provide a human-readable specification of the expected ABI in the Python code itself. A developer can compare the `cdef` block against `cpu_kernels.h` to verify correctness.
- Function pointer passing (`lib.task_forward_pass` as a callable in `pool_dispatch_and_wait`) is handled natively by cffi's function pointer support.
- cffi provides better error messages than ctypes when type mismatches occur (e.g., passing a `float*` where `int*` is expected).

**Disadvantages:**
- **Requires `cffi` as a dependency.** The CPU optional dependency group would change from `cpu = []` to `cpu = ["cffi"]`, contradicting ADR-014's established contract. While cffi is a widely-used, well-maintained package, it is an external dependency where none was previously specified.
- **`ffi.cdef()` declarations must be maintained in sync with `cpu_kernels.h`.** Changes to struct layouts, function signatures, or new kernel additions require updating both the C header and the Python-side `cdef` block. This is the same synchronization burden as any FFI mechanism, but cffi does not automate it in ABI mode.
- **`ffi.cast()` for numpy pointers.** Passing numpy array data pointers requires explicit casting: `ffi.cast("float *", array.ctypes.data)`. This is verbose but not error-prone.
- **Opaque pointer limitation.** `ThreadPool*` is an opaque pointer (the Python side never dereferences it). cffi handles this via `typedef ... ThreadPool;` (incomplete type), which works but requires the `...` syntax.

### Option B: cffi API mode (build-time header parsing)

Use cffi's API mode to parse `cpu_kernels.h` at build time, generating a compiled Python extension module (`_cpu_kernels_cffi.so`) that embeds the type definitions and library bindings:

```python
# build script (invoked by Meson custom_target or setup.py)
from cffi import FFI
ffi = FFI()
ffi.cdef(open("cpu_kernels.h").read())  # Parse actual header
ffi.set_source("_cpu_kernels_cffi", '#include "cpu_kernels.h"',
               libraries=["cpu_kernels"],
               library_dirs=[...])
ffi.compile()
```

**Advantages:**
- **Strongest type safety.** The C header is parsed directly — no manual `cdef` block to maintain. Struct layout, field types, and function signatures are extracted from the authoritative source.
- Compile-time verification catches header/binding mismatches before runtime. If `cpu_kernels.h` adds a field to `ForwardPassArgs`, the cffi extension is regenerated to match.

**Disadvantages:**
- **Adds a Meson build integration step.** The cffi compilation must be coordinated with the `libcpu_kernels.so` build. Meson must invoke cffi's `ffi.compile()` after the shared library is built, producing a second artifact (`_cpu_kernels_cffi.so`). This adds a `custom_target` to `src/backends/cpu/meson.build` with a dependency on the `cpu_kernels_lib` target.
- **Two compiled artifacts.** The CPU backend produces both `libcpu_kernels.so` (the kernel library) and `_cpu_kernels_cffi.so` (the cffi wrapper). Both must be installed, discovered at runtime, and version-matched. This doubles the binary artifact surface for one backend.
- **Requires `cffi` as both build and runtime dependency.** `cffi` must be added to `[build-system] requires` in `pyproject.toml`, not just optional runtime deps. This affects every build environment, including those targeting only the OpenCL backend.
- **Header parsing limitations.** cffi's C parser is not a full C preprocessor. It cannot evaluate `#ifdef` directives, expand macros from `cpu_simd.h`, or handle platform-specific type definitions. The `cpu_kernels.h` header would need a cffi-friendly subset or a preprocessing step to produce parseable declarations.
- **Meson-python interaction complexity.** `meson-python` manages the build lifecycle. Introducing a cffi compile step that depends on a Meson-built shared library creates a circular build ordering challenge: `meson-python` drives Meson, which builds `libcpu_kernels.so`, then a cffi step (outside Meson? inside?) builds the wrapper.

### Option C: ctypes (stdlib)

Use Python's standard library `ctypes` module to load the pre-built `libcpu_kernels.so` at runtime. The Python side defines struct layouts as `ctypes.Structure` subclasses and loads the library via `ctypes.CDLL`:

```python
import ctypes

class ForwardPassArgs(ctypes.Structure):
    _fields_ = [
        ("input",                      ctypes.c_void_p),
        ("sample_mask",                ctypes.c_void_p),
        ("weights_shared_simd_major",  ctypes.c_void_p),
        ("biases_shared",              ctypes.c_void_p),
        ("hidden_activations",         ctypes.c_void_p),
        ("hidden_mask",                ctypes.c_void_p),
        ("batch_chunk_offset",         ctypes.c_uint),
        ("batch_chunk_count",          ctypes.c_uint),
        ("total_batch_count",          ctypes.c_uint),
        ("padded_input_count",         ctypes.c_uint),
        ("padded_hidden_count",        ctypes.c_uint),
    ]

lib = ctypes.CDLL(str(lib_path))
```

Usage:

```python
TASK_FN = ctypes.CFUNCTYPE(None, ctypes.c_void_p, ctypes.c_uint, ctypes.c_uint)

lib.pool_dispatch_and_wait.argtypes = [
    ctypes.c_void_p,   # ThreadPool*
    TASK_FN,            # void (*fn)(void*, uint, uint)
    ctypes.c_void_p,   # void* args
    ctypes.c_uint,      # uint task_count
]
lib.pool_dispatch_and_wait.restype = None

args = ForwardPassArgs()
args.input = input_array.ctypes.data
args.batch_chunk_count = node.scalar_params["batch_chunk_count"]
# ...
task_fn = TASK_FN(("task_forward_pass", lib))
lib.pool_dispatch_and_wait(pool, task_fn, ctypes.byref(args), task_count)
```

**Advantages:**
- **Zero external dependency.** ctypes is part of the Python standard library (since Python 2.5). The CPU optional dependency group remains `cpu = []`, fully consistent with ADR-014's established contract.
- **Runtime loading of pre-built library.** `ctypes.CDLL(path)` loads the shared library at runtime without any compilation step. Fully compatible with ADR-014's packaging and `importlib.resources` discovery.
- **numpy integration is mature.** Every numpy array exposes `.ctypes.data` (a `ctypes.c_void_p`-compatible integer) for direct pointer passing to C functions. The `_fields_` definitions use `ctypes.c_void_p` for all pointer fields, and the Python side assigns `array.ctypes.data` values. This is the standard pattern for numpy ↔ C interop via ctypes.
- **Function pointer passing is supported.** `ctypes.CFUNCTYPE` defines callback types, and `CFUNCTYPE(("symbol_name", lib))` retrieves a typed function pointer from the loaded library. The `pool_dispatch_and_wait` pattern is expressible.
- **No build system integration.** No additional Meson targets, no `custom_target`, no cffi compile step. The ctypes code is pure Python that loads the library produced by ADR-014's existing `shared_library` target.
- **Portable across all platforms.** ctypes handles platform-specific library loading (`LoadLibrary` on Windows, `dlopen` on POSIX) and calling conventions transparently.

**Disadvantages:**
- **No compile-time type checking.** Struct `_fields_` definitions are Python code with no mechanical connection to `cpu_kernels.h`. A field added, removed, or reordered in the C header requires a manual update to the Python struct definition. Type mismatches (e.g., `c_uint` vs. `c_int`, or a missing field) produce incorrect memory layouts that manifest as silent data corruption, not error messages.
- **Verbose struct definitions.** Each argument struct requires a `ctypes.Structure` subclass with an explicit `_fields_` list. For ~20 kernel argument structs, this produces substantial boilerplate — each struct averaging 8–15 fields yields ~200–300 lines of type definitions.
- **Weaker error diagnostics.** ctypes provides minimal error checking at the Python level. Passing the wrong number of arguments to a C function crashes the process. Passing a `c_uint` where `c_float` is expected silently reinterprets bits. `argtypes` declarations mitigate this for top-level function calls, but struct field assignments are unchecked.
- **`void*` pointer casting is manual.** Buffer pointers are all `ctypes.c_void_p` — the Python type system does not distinguish `const float*` from `float*` from `const uint*`. The programmer must ensure the correct numpy array is assigned to the correct struct field. This is the same issue in cffi ABI mode, but ctypes' `c_void_p` obscures it more than cffi's typed pointers.

### Option D: nanobind / pybind11

Use nanobind (lightweight successor to pybind11) or pybind11 to create a Python extension module that wraps the CPU kernel library:

```cpp
// _cpu_bindings.cpp
#include <nanobind/nanobind.h>
#include "cpu_kernels.h"

namespace nb = nanobind;

NB_MODULE(_cpu_bindings, m) {
    m.def("pool_create", &pool_create);
    m.def("pool_destroy", &pool_destroy);
    m.def("dispatch_forward_pass", [](intptr_t pool, nb::ndarray<float> input, ...) {
        ForwardPassArgs args;
        args.input = input.data();
        // ...
        pool_dispatch_and_wait((ThreadPool*)pool, task_forward_pass, &args, count);
    });
    // ... per-kernel dispatch wrappers
}
```

**Advantages:**
- **Highest type safety.** The C++ wrapper provides full type checking at the Python ↔ C boundary. Buffer dimensions can be validated in the wrapper code. `null` pointer protection, range checks, and type coercion are expressible.
- **Native numpy integration.** nanobind's `nb::ndarray` provides zero-copy numpy array access with compile-time shape and dtype constraints.
- **Idiomatic Python API.** The extension module exposes Python functions with keyword arguments, docstrings, and proper exceptions — not raw ctypes structs.

**Disadvantages:**
- **Heaviest build dependency.** Requires a C++ compiler (in addition to the C compiler for the kernel library), nanobind or pybind11 headers, and a `custom_target` in Meson to compile the wrapper. This is the largest build-chain expansion of any option.
- **C++ for a pure C ABI.** The CPU kernel library is pure C. Wrapping it with C++ introduces a language boundary inside the backend that serves no computational purpose. The wrapper's sole role is FFI marshalling — a responsibility that ctypes and cffi handle without C++.
- **Two compiled artifacts.** Like Option B, this produces both `libcpu_kernels.so` and `_cpu_bindings.so`. Both must be built, installed, and version-matched.
- **Mismatched scope.** nanobind/pybind11 are designed for rich C++ interfaces with classes, overloaded functions, and complex type hierarchies. The CPU ABI is a flat set of C functions with `void*` argument passing — the simplest C ABI pattern. nanobind/pybind11's strengths (class wrapping, template instantiation, inheritance mapping) are unused.
- **Wrapper code duplication.** Each kernel's dispatch wrapper lambda repeats the struct construction logic (assign fields from Python arguments). With ~20 kernels, this wrapper file approaches the verbosity of ctypes struct definitions but requires C++ compilation.

---

## Analysis

### Eliminating Option D (nanobind / pybind11)

Option D is eliminated because it introduces the heaviest build dependency for the simplest ABI pattern. The CPU kernel library exports flat C functions with `void*` arguments — no classes, no templates, no overloads, no inheritance. nanobind/pybind11's design strengths are entirely orthogonal to this use case.

The C++ wrapper adds a compiled artifact, a C++ compiler requirement, and a language boundary inside the CPU backend. The wrapper's sole purpose — marshalling struct fields from Python arguments and calling C functions — is exactly what ctypes and cffi provide without compilation. Introducing C++ compilation for marshalling contradicts CONCEPT.md §1: the complexity (C++ build chain) creates no benefit that the simpler mechanisms cannot provide.

The analysis parallels ADR-007's elimination of Option C (abstract binding interface with embedded contracts): just as Option C conflated validation and dispatch into a single opaque class, Option D conflates FFI marshalling and struct construction into compiled C++ wrappers where they cannot be inspected or modified without recompilation. The Python-side transparency of ctypes and cffi is preferable for a moderate ABI surface.

### Eliminating Option B (cffi API mode)

Option B is eliminated because its build integration requirements are disproportionate to the ABI surface's complexity and create structural tension with the existing Meson build chain.

The decisive issue is the **build ordering dependency**. ADR-014's Meson build produces `libcpu_kernels.so`. cffi API mode must then compile a second artifact (`_cpu_kernels_cffi.so`) that links against the first. This requires either:
1. A Meson `custom_target` that invokes cffi's Python-based compilation after the library is built — interleaving Python execution with C compilation inside the Meson build graph.
2. A post-build step outside Meson, which `meson-python` would need to orchestrate.

Neither integrates cleanly with ADR-014's `subdir()` delegation model, where `src/backends/cpu/meson.build` declares a single `shared_library` target. Adding a Python-dependent `custom_target` that imports `cffi`, reads `cpu_kernels.h`, and produces a second compiled artifact fundamentally changes the backend's build complexity.

Furthermore, cffi's C parser cannot process the raw `cpu_kernels.h` header: the `#include "cpu_simd.h"` directive, `#ifdef` ISA detection cascade, and `SIMD_WIDTH`-dependent type definitions require full C preprocessing that cffi does not provide. A preprocessed header or a cffi-specific subset would need to be generated, adding a third build step.

This analysis mirrors ADR-014's elimination of Option B (per-backend Meson subprojects): both introduce structural complexity that is disproportionate to the problem. Option B here distributes compilation across two systems (Meson for the library, cffi for the wrapper) where a single mechanism suffices.

### Choosing between Options A and C

Options A (cffi ABI mode) and C (ctypes) are both capable of loading a pre-built shared library at runtime and marshalling the CPU ABI surface. They differ in three dimensions:

**Dependency cost.**

| | Option A (cffi ABI mode) | Option C (ctypes) |
| :--- | :--- | :--- |
| External dependency | `cffi` package | None (stdlib) |
| ADR-014 impact | `cpu = ["cffi"]` (amends ADR-014) | `cpu = []` (consistent with ADR-014) |
| Build system impact | None (runtime only) | None |
| Install size | ~200 KB (cffi + pycparser) | 0 KB |

Option C preserves ADR-014's established contract. Option A requires amending ADR-014's `pyproject.toml` and introducing a dependency that is specific to the FFI marshalling layer — not to the computation or the domain.

**Type safety.** Both mechanisms require Python-side type declarations that must match `cpu_kernels.h`. Neither verifies this match mechanically at load time (only cffi API mode does, which is eliminated). Both rely on the developer to maintain synchronization between C and Python type definitions.

cffi offers nominally better diagnostics: `ffi.cdef()` declarations use C syntax, so a field type mismatch between `cdef` and the actual library may produce a cffi-specific error. ctypes offers no such check — a wrong field type in `_fields_` silently produces incorrect memory layout.

However, the practical difference is marginal for this ABI surface. The argument structs are composed of two types: `void*` (all buffer pointers) and `unsigned int` / `float` (all scalars). There is no complex nested struct hierarchy, no union types, no bitfields. The marshalling errors that cffi's diagnostics would catch — e.g., confusing `uint` with `int`, or adding a field — are equally catchable by the struct size assertion mechanism described in §Decision.

**Struct construction ergonomics.**

cffi:
```python
args = ffi.new("ForwardPassArgs *")
args.input = ffi.cast("const float *", array.ctypes.data)
args.batch_chunk_count = count
```

ctypes:
```python
args = ForwardPassArgs()
args.input = array.ctypes.data
args.batch_chunk_count = count
```

The ergonomic difference is minimal. cffi's `ffi.cast()` for pointer fields is slightly more verbose than ctypes' direct integer assignment. ctypes' `ctypes.Structure` subclasses are more verbose to define but use standard Python class syntax.

**Function pointer marshalling.**

Both mechanisms support retrieving C function pointers from a loaded library and passing them to other C functions. The `pool_dispatch_and_wait(pool, task_fn, args, count)` pattern is expressible in both:

cffi: `lib.pool_dispatch_and_wait(pool, lib.task_forward_pass, ffi.cast("void *", args), count)`

ctypes:
```python
TASK_FN = ctypes.CFUNCTYPE(None, ctypes.c_void_p, ctypes.c_uint, ctypes.c_uint)
task_fn = TASK_FN(("task_forward_pass", lib))
lib.pool_dispatch_and_wait(pool, task_fn, ctypes.byref(args), count)
```

ctypes requires the explicit `CFUNCTYPE` declaration; cffi infers the type from the `cdef` signature. Both work correctly.

### The decisive factors

**ADR-014 consistency.** ADR-014 was accepted with `cpu = []`. This was not an oversight — it was a deliberate statement that the CPU backend requires no additional Python packages. Introducing `cffi` as a dependency for a marshalling layer contradicts this accepted contract. Amending an accepted ADR to accommodate a convenience library for FFI marshalling is a disproportionate architectural change for a marginal ergonomic benefit.

**Proportionality.** The CPU ABI surface is ~24 exported symbols with simple `void*`-based argument passing. The struct types use only `c_void_p`, `c_uint`, and `c_float` — no unions, no bitfields, no nested structs beyond flat pointer/scalar compositions. ctypes handles this comfortably. cffi's advantages (better error messages, `ffi.new()` garbage collection) are real but modest for this complexity level. Introducing a dependency for modest ergonomic gains contravenes CONCEPT.md §1.

**CONCEPT.md §1 (Architectural Elegance Feedback).** If ctypes' limitations create genuine maintenance pressure — for example, if the ABI surface grows to 100+ functions with deeply nested argument structs and recurrent layout-drift bugs — the response is to formalize cffi as an architectural primitive by amending ADR-014's dependency contract and revisiting this ADR. This is the prescribed escalation path, not a preemptive adoption of cffi "just in case."

---

## Decision

**Per-backend FFI autonomy with ctypes for the CPU backend.**

Each backend chooses the interop mechanism most natural for its platform binding:

| Backend | Interop mechanism | Source | Governed by |
| :--- | :--- | :--- | :--- |
| OpenCL | PyOpenCL | `pyopencl` package | Established (core dependency) |
| Vulkan | vulkan-python | `vulkan-python` package | ADR-014 (`vulkan = ["vulkan-python>=0.2.0"]`) |
| CPU | ctypes (stdlib) | `ctypes` (Python stdlib) | This ADR |

The CPU backend uses Python's standard library `ctypes` module to load `libcpu_kernels.so` and call its exported C functions. No additional Python dependencies are introduced.

### Library loading

The CPU backend's `renderer.py` loads the shared library at initialization using the discovery path from ADR-014:

```python
# src/backends/cpu/renderer.py
import ctypes
import importlib.resources
import sys

def _load_cpu_kernels() -> ctypes.CDLL:
    """Load the CPU kernels shared library."""
    backend_dir = importlib.resources.files(
        'averaging_ensembled_classifier.backends.cpu'
    )
    if sys.platform == 'win32':
        lib_name = 'cpu_kernels.dll'
    elif sys.platform == 'darwin':
        lib_name = 'libcpu_kernels.dylib'
    else:
        lib_name = 'libcpu_kernels.so'

    lib_path = backend_dir / lib_name
    return ctypes.CDLL(str(lib_path))
```

This is consistent with ADR-014's `_cpu_kernels_lib_path()` pattern but uses `ctypes.CDLL` directly, which loads the library and resolves symbols on demand.

### FFI type definitions module

A dedicated module `src/backends/cpu/_ffi_types.py` contains all ctypes struct definitions and function signature declarations. This module is the single source of truth for the Python-side ABI contract:

```python
# src/backends/cpu/_ffi_types.py
"""ctypes struct definitions mirroring cpu_kernels.h.

This module defines the Python-side representation of every C struct
and function signature exported by libcpu_kernels. Each struct's _fields_
MUST match the field order, names, and types in cpu_kernels.h exactly.

Layout verification assertions (see _verify_layout()) catch
drift between Python and C definitions during development and testing.
"""
import ctypes

# ── Common type aliases ────────────────────────────────────────────
c_uint = ctypes.c_uint
c_float = ctypes.c_float
c_void_p = ctypes.c_void_p

# ── Task function pointer type ─────────────────────────────────────
# void (*fn)(void* args, uint task_index, uint thread_id)
TASK_FN = ctypes.CFUNCTYPE(None, c_void_p, c_uint, c_uint)


# ── Argument structs ───────────────────────────────────────────────

class ForwardPassArgs(ctypes.Structure):
    _fields_ = [
        ("input",                      c_void_p),
        ("sample_mask",                c_void_p),
        ("weights_shared_simd_major",  c_void_p),
        ("biases_shared",              c_void_p),
        ("hidden_activations",         c_void_p),
        ("hidden_mask",                c_void_p),
        ("batch_chunk_offset",         c_uint),
        ("batch_chunk_count",          c_uint),
        ("total_batch_count",          c_uint),
        ("padded_input_count",         c_uint),
        ("padded_hidden_count",        c_uint),
    ]

# ... (remaining ~19 kernel argument structs)

class ReductionTreePlan(ctypes.Structure):
    _fields_ = [
        ("partial_collection",         c_void_p),
        ("offset_lists_flat",          c_void_p),
        ("stage_offsets_into_list",    c_void_p),
        ("stage_fan_in",               c_void_p),
        ("stage_node_counts",          c_void_p),
        ("staging_buffer_0",           c_void_p),
        ("staging_buffer_1",           c_void_p),
        ("output",                     c_void_p),
        ("partial_width",              c_uint),
        ("num_stages",                 c_uint),
        ("t_algorithmic",              c_float),
        ("lambda_",                    c_float),
        ("fp_max",                     c_float),
        ("epsilon",                    c_float),
    ]
```

All buffer pointer fields use `ctypes.c_void_p`. The Python side assigns `numpy_array.ctypes.data` to these fields, which returns the array's data pointer as a Python integer — directly assignable to `c_void_p`. This is the standard numpy ↔ ctypes interop pattern.

### Function signature declarations

The `_ffi_types.py` module also declares `argtypes` and `restype` for all exported functions. These declarations enable ctypes' argument type checking at call time:

```python
def configure_library(lib: ctypes.CDLL) -> None:
    """Set argtypes and restype for all exported functions.

    Must be called once after loading the library. Enables
    ctypes type checking at each function call.
    """
    # Thread pool lifecycle
    lib.pool_create.argtypes = [c_uint]
    lib.pool_create.restype = c_void_p

    lib.pool_destroy.argtypes = [c_void_p]
    lib.pool_destroy.restype = None

    lib.pool_dispatch_and_wait.argtypes = [c_void_p, TASK_FN, c_void_p, c_uint]
    lib.pool_dispatch_and_wait.restype = None

    # Reduction engine
    lib.execute_reduction_tree.argtypes = [c_void_p, ctypes.POINTER(ReductionTreePlan)]
    lib.execute_reduction_tree.restype = None

    # Per-kernel task functions (all share the same signature)
    task_fn_argtypes = [c_void_p, c_uint, c_uint]
    for name in KERNEL_TASK_NAMES:
        fn = getattr(lib, name)
        fn.argtypes = task_fn_argtypes
        fn.restype = None
```

The `KERNEL_TASK_NAMES` list enumerates all exported `task_*` symbols, providing a single inventory that the renderer iterates for function pointer retrieval.

### Function pointer retrieval

The CPU `PlanRenderer` retrieves typed function pointers for all `task_*` symbols during initialization and stores them in a dispatch table keyed by `KernelContract.kernel_name`:

```python
# src/backends/cpu/renderer.py

class CPUPlanRenderer:
    def __init__(self) -> None:
        self._lib = _load_cpu_kernels()
        configure_library(self._lib)
        self._pool = self._lib.pool_create(os.cpu_count() or 4)

        # Build dispatch table: kernel_name → (task_fn_ptr, args_struct_class)
        self._dispatch_table: dict[str, tuple[TASK_FN, type]] = {
            "forward_pass": (
                TASK_FN(("task_forward_pass", self._lib)),
                ForwardPassArgs,
            ),
            "compute_probs_loss_cce_chunk": (
                TASK_FN(("task_cce_probs_loss", self._lib)),
                CceChunkArgs,
            ),
            # ... remaining kernel mappings
        }

    def _dispatch_kernel(self, node: KernelDispatchNode, buffers: dict) -> None:
        """Dispatch a single kernel node via pool_dispatch_and_wait."""
        task_fn, args_cls = self._dispatch_table[node.contract.kernel_name]
        args = args_cls()
        # Populate args from node.scalar_params and buffer pointers
        _populate_args(args, node, buffers)
        self._lib.pool_dispatch_and_wait(
            self._pool, task_fn, ctypes.byref(args), node.task_count
        )
```

The dispatch table is constructed once at initialization. Each `KernelDispatchNode` in the plan is dispatched by looking up its `kernel_name` in the table, constructing the appropriate argument struct, and calling `pool_dispatch_and_wait`. This is the fine-grained FFI pattern — Python walks the plan DAG node by node, and each node results in one `pool_dispatch_and_wait` call through ctypes.

### Layout verification

The most significant risk of ctypes (vs. cffi API mode) is silent struct layout drift — a field added, removed, or reordered in `cpu_kernels.h` without a corresponding update to `_ffi_types.py`. To mitigate this, the C library exports a layout verification function:

```c
// cpu_kernels.h — layout verification support
typedef struct {
    const char* struct_name;
    size_t      struct_size;
    size_t      num_fields;
    // ... optionally: per-field offsets
} StructLayoutInfo;

CPU_KERNELS_EXPORT size_t get_struct_size_ForwardPassArgs(void);
CPU_KERNELS_EXPORT size_t get_struct_size_CceChunkArgs(void);
CPU_KERNELS_EXPORT size_t get_struct_size_ReductionTreePlan(void);
// ... one per struct
```

The Python side verifies layout at library load time:

```python
def _verify_layouts(lib: ctypes.CDLL) -> None:
    """Assert that Python struct sizes match C struct sizes.

    Called once during library initialization. Raises RuntimeError
    if any struct has a size mismatch, indicating layout drift
    between _ffi_types.py and cpu_kernels.h.
    """
    checks = [
        (ForwardPassArgs,   lib.get_struct_size_ForwardPassArgs),
        (CceChunkArgs,      lib.get_struct_size_CceChunkArgs),
        (ReductionTreePlan, lib.get_struct_size_ReductionTreePlan),
        # ... remaining structs
    ]
    for py_struct, c_size_fn in checks:
        c_size_fn.restype = ctypes.c_size_t
        c_size = c_size_fn()
        py_size = ctypes.sizeof(py_struct)
        if c_size != py_size:
            raise RuntimeError(
                f"Struct layout mismatch for {py_struct.__name__}: "
                f"C size={c_size}, Python size={py_size}. "
                f"Update _ffi_types.py to match cpu_kernels.h."
            )
```

This verification catches:
- Added or removed fields (size changes).
- Type changes that alter field size (e.g., `uint` → `uint64_t`).
- Padding differences due to alignment.

It does **not** catch field reordering within the same type roster (e.g., swapping two `uint` fields). This class of error is caught by ADR-016 Tier 2 tests, which verify numerical correctness per kernel.

The `get_struct_size_*` functions add ~20 trivial symbols to the exported ABI. These are `CPU_KERNELS_EXPORT`-annotated and carry zero runtime cost (they return compile-time constants). The verification runs once at backend initialization and adds negligible startup latency.

### Thread pool lifecycle

The thread pool is an opaque resource managed by the `CPUPlanRenderer`:

- **Creation:** `pool_create(num_threads)` is called in `CPUPlanRenderer.__init__()`. The thread count is derived from the `HardwareProfile` (ADR-006), defaulting to `os.cpu_count()`.
- **Dispatch:** Every `pool_dispatch_and_wait` call uses the renderer's pool handle. The pool handle is an opaque `c_void_p` — Python never dereferences it.
- **Destruction:** `pool_destroy(pool)` is called in `CPUPlanRenderer.__del__()` or via an explicit `close()` method. This joins all worker threads and frees pool resources.

The pool handle's lifetime spans the renderer's lifetime. Since the renderer is typically created once per training session, the pool is created once and reused across all `render()` calls.

### Buffer lifetime discipline

The `buffer_allocator.py` creates numpy arrays as the underlying buffer storage. These arrays must remain alive (not garbage-collected) while the C code accesses their memory through the `void*` pointers in argument structs. The discipline is:

1. **Buffer allocator retains all arrays.** The `CPUBufferAllocator` holds references to all allocated numpy arrays in a `dict[BufferHandle, np.ndarray]`. As long as the allocator is alive, no buffer is collected.
2. **Argument struct references are transient.** The ctypes `Structure` objects are created per dispatch call, populated with `array.ctypes.data` values, passed to `pool_dispatch_and_wait`, and discarded after the function returns. Since `pool_dispatch_and_wait` blocks until all tasks complete, the argument struct only needs to be alive for the duration of the call.
3. **The GC cannot intervene during dispatch.** `pool_dispatch_and_wait` is a blocking C call. Python's GIL is released during the call (ctypes releases the GIL for all `CDLL` calls), but the buffer allocator's reference to the numpy arrays prevents collection regardless of GIL state.

This discipline ensures no dangling pointer can occur through normal usage. The only violation path is if the buffer allocator is destroyed while a dispatch is in progress — a programming error that would also affect any other FFI mechanism.

### Interaction with future multi-ISA dispatch

ADR-014 documents a future extension where multiple ISA-specific library variants (`libcpu_kernels_avx512.so`, `libcpu_kernels_avx2.so`, `libcpu_kernels_sse2.so`) are compiled and the FFI layer selects the appropriate variant at runtime.

Under this ADR's ctypes approach, multi-ISA dispatch is a straightforward extension:

```python
def _select_isa_library(backend_dir: Path) -> Path:
    """Select the highest-ISA library available on this CPU."""
    # Check CPU feature flags (e.g., via CPUID or /proc/cpuinfo)
    for name in ['libcpu_kernels_avx512.so',
                 'libcpu_kernels_avx2.so',
                 'libcpu_kernels_sse2.so']:
        path = backend_dir / name
        if path.exists():
            return path
    raise RuntimeError("No compatible CPU kernel library found.")
```

The selected library is loaded via `ctypes.CDLL(path)` — the same loading, layout verification, and dispatch table construction applies regardless of which ISA variant is loaded. All variants export the same ABI surface; they differ only in the compiled ISA instructions within each function body.

This extension requires no changes to `_ffi_types.py`, `configure_library()`, or the `CPUPlanRenderer` dispatch logic. The only change is in the library path resolution function.

---

## Consequences

### Positive

- **Zero additional dependencies for the CPU backend.** The CPU optional dependency group remains `cpu = []`, fully consistent with ADR-014's established contract. No `pyproject.toml` amendment is required. The dependency footprint for a CPU-only deployment is `numpy` + stdlib.

- **No build system integration.** The ctypes FFI layer is pure Python code with no compilation step. ADR-014's `src/backends/cpu/meson.build` produces a single `shared_library` target — no additional `custom_target` for FFI wrapper generation. The build system complexity is unchanged.

- **Library loading is consistent with ADR-014.** The `importlib.resources` discovery path and platform-specific library name resolution are identical to ADR-014's specification. The only addition is `ctypes.CDLL(path)` at the end of the discovery chain.

- **Layout verification catches struct drift at startup.** The `get_struct_size_*` verification functions provide an early, clear error message when `_ffi_types.py` is out of sync with `cpu_kernels.h`. This converts the most dangerous ctypes failure mode (silent struct layout corruption) into a deterministic startup error.

- **Fine-grained dispatch preserves PlanRenderer pattern.** The Python-side renderer walks the plan DAG and calls `pool_dispatch_and_wait` per node, consistent with how the OpenCL renderer calls `clEnqueueNDRange` per node and the Vulkan renderer records `vkCmdDispatch` per node. Debugging a CPU execution means inspecting the same plan traversal logic that governs all backends.

- **Per-backend FFI autonomy is architecturally sound.** The decision confirms that ADR-012's isolation boundary extends to the FFI mechanism. Each backend's interop technology is an internal concern — invisible to the shared layer, the orchestrator, and other backends. This is consistent with ADR-001's principle that backends are autonomous within their package.

- **Multi-ISA extensibility is trivial.** Swapping the loaded library path to a different ISA variant requires no changes to type definitions, dispatch logic, or the PlanRenderer. The ABI surface is ISA-invariant; only the machine code within each function changes.

### Negative

- **~200–300 lines of boilerplate in `_ffi_types.py`.** Each of the ~20 kernel argument structs requires an explicit `ctypes.Structure` subclass with individually typed `_fields_` entries. This is verbose compared to cffi's `ffi.cdef()` (which uses C syntax) but is offset by the zero-dependency benefit and is a one-time authoring cost that changes only when the C ABI changes.

- **No compile-time type checking.** Struct field definitions in `_ffi_types.py` are not mechanically linked to `cpu_kernels.h`. A developer who modifies the C struct must remember to update the Python definition. The layout verification assertion catches size changes but not same-size reorderings (e.g., swapping two `uint` fields). This gap is covered by ADR-016 Tier 2 tests, which verify numerical correctness and would produce wrong results for a reordered struct.

- **Weaker error diagnostics than cffi.** A ctypes type mismatch (e.g., assigning a string to a `c_uint` field) produces a ctypes-specific error, but the diagnostics are less informative than cffi's. Setting `argtypes` on all exported functions mitigates this for direct function calls but does not help for struct field assignments.

- **Function pointer boilerplate.** Retrieving `task_*` function pointers via `CFUNCTYPE(("symbol_name", lib))` and building the dispatch table requires ~40 lines of initialization code. cffi would handle this implicitly. The boilerplate is concentrated in `CPUPlanRenderer.__init__()` and is written once.

- **GIL release during dispatch.** `ctypes.CDLL` calls release the GIL, which is correct (it allows Python threads to run while C code executes). However, this means that if Python code modifies a numpy buffer while a dispatch is in progress, data corruption occurs. The buffer lifetime discipline (§Decision) prevents this in normal usage, but the absence of compile-time protection makes it a potential pitfall for future maintenance.

### Migration implications

Per ADR-017's phasing:

**Phase 3 (CPU Backend):**
1. Create `src/backends/cpu/_ffi_types.py` with ctypes struct definitions mirroring `cpu_kernels.h`.
2. Add `get_struct_size_*` verification functions to `cpu_kernels.h` and the corresponding C source.
3. Implement library loading in `renderer.py` using `importlib.resources` + `ctypes.CDLL`.
4. Implement layout verification (`_verify_layouts`) in the renderer's initialization.
5. Implement the dispatch table mapping `kernel_name` → `(task_fn_ptr, args_struct_class)`.
6. Implement `CPUPlanRenderer._dispatch_kernel()` using `pool_dispatch_and_wait` with ctypes function pointers.
7. ADR-016 Tier 2 tests validate each kernel's numerical correctness — implicitly verifying that struct marshalling is correct.
8. ADR-016 Tier 3 tests compare CPU vs. OpenCL outputs — catching any structural FFI error that produces numerically wrong results.

**No build system changes.** ADR-014's `src/backends/cpu/meson.build` is unchanged — the `shared_library('cpu_kernels', ...)` target already produces the library that ctypes loads. The only addition is the `get_struct_size_*` symbols in the C source, which are trivial functions added to the existing compilation unit.

**No `pyproject.toml` changes.** The CPU optional dependency group remains `cpu = []`. This is the deliberately zero-cost outcome of choosing ctypes over cffi.

---

## References

- [ADR-001: Backend Abstraction Boundary](ADR-001-backend-abstraction-boundary.md) — `PlanRenderer` interface; per-backend autonomy; fine-grained dispatch pattern
- [ADR-007: KernelSignature Contract/Binding Split](ADR-007-kernel-signature-contract-binding-split.md) — `KernelContract.kernel_name` as dispatch table key; abstract placement keys
- [ADR-012: Module Factoring & Services Dissolution](ADR-012-module-factoring-and-services-dissolution.md) — backend isolation boundary; `src/backends/cpu/` module inventory; `PlanRenderer` as FFI abstraction point
- [ADR-013: Kernel Source Strategy](ADR-013-kernel-source-strategy.md) — CPU kernel source layout in `src/backends/cpu/kernel_sources/`; `task_*` function signature convention; `cpu_kernels.h` ABI surface; three-tier specification hierarchy
- [ADR-014: Build System Integration](ADR-014-build-system-integration.md) — `libcpu_kernels.so` build output; hidden visibility + `CPU_KERNELS_EXPORT`; `importlib.resources` discovery; installation path; `cpu = []` dependency declaration; single-ISA strategy with multi-ISA extension path
- [ADR-016: Test Strategy](ADR-016-test-strategy.md) — Tier 2 per-backend kernel tests catch FFI marshalling errors; Tier 3 cross-backend parity tests validate end-to-end correctness
- [ADR-017: Incremental Migration Path](ADR-017-incremental-migration-path-stub.md) — Phase 3 (CPU Backend) requires the FFI mechanism decided here
- [CONCEPT.md](../CONCEPT.md) — §1 Architectural Elegance Feedback (proportional complexity; formalize when pressure arises)
- [CONTRACT.md](../CONTRACT.md) — Article 6 Mandatory Build-Time Symbols; Article 1.4 Collaborative Interface Verifiability
- [CPU_BACKEND.md](../CPU_BACKEND.md) — `cpu_simd.h` ISA detection; `task_<kernel_name>` function signatures; `pool_dispatch_and_wait` threading model; `cpu_kernels.h` / `kernels_interface_cpu.h` complete ABI specification
