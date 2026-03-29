# Phase 3A: CPU Kernel Library & Build Infrastructure — Detailed Plan

**Status: ✅ IMPLEMENTED** — `libcpu_kernels.so` builds successfully via Meson `shared_library()`. All `get_struct_size_*` verification exports resolve. SIMD abstraction layer supports AVX-512, AVX2, SSE2, ARM NEON, and scalar fallback. Thread pool implements C11 threads with pthreads fallback. 18 task functions + `execute_reduction_tree` implemented across 6 C source files + 4 headers. Tier 1 green (no regressions).  
**Phase:** 3A of 6 (sub-phase A of 3)  
**Objective:** Implement the CPU backend's native C kernel library — the `cpu_simd.h` SIMD abstraction layer, the `cpu_threads.h` thread pool, the `cpu_kernels.h` ABI surface with all ~22 `task_<kernel_name>` function implementations, and the Meson `shared_library()` build target — producing `libcpu_kernels.so` as the sole build artifact. This sub-phase delivers the compiled native library; Python-side FFI integration is Phase 3B.  
**Governing ADRs:** ADR-013 (kernel source strategy — `kernels.cl.h` as algorithmic reference), ADR-014 (build system — Meson `shared_library()`, `aec_backend_cpu` feature option, ISA flags), ADR-015 (interop — ctypes; ABI surface definition; `task_<kernel_name>` uniform signature; `get_struct_size_*` verification exports), ADR-001 (three-tier jurisdictional model — Execution tier)  
**Rollback gate:** Library compiles on at least one ISA tier (SSE2/AVX2/AVX-512/NEON/scalar fallback) with zero warnings at `-Wall -Wextra`. All `get_struct_size_*` exports resolve. Tier 1 green (no regressions). Full CPU Tier 2 correctness deferred to Phase 3C.  
**Dependencies:** Phase 1 (plan model) — **COMPLETE** (332 tests green). Independent of Phase 2 (OpenCL adapter).

### Implementation Summary

Delivered artifacts in `src/backends/cpu/kernel_sources/`:

| File | Role |
| :--- | :--- |
| `cpu_simd.h` | Compile-time ISA detection cascade (AVX-512 → AVX2 → SSE2 → NEON → scalar). Uniform macro API: `simd_load`, `simd_store`, `simd_fmadd`, `simd_reduce_add`, `simd_select`, etc. |
| `cpu_threads.h` / `cpu_threads.c` | Persistent thread pool with `pool_create()`, `pool_destroy()`, `pool_dispatch_and_wait()`. Atomic task claiming for natural load balancing. C11 `<threads.h>` with pthreads fallback. |
| `cpu_kernels.h` | Public ABI header: all `task_<kernel_name>` prototypes, per-kernel argument structs, `ReductionTreePlan` C struct, `get_struct_size_*` layout verification exports, `get_simd_width()`. |
| `cpu_export.h` | Symbol visibility macros (`CPU_EXPORT`). |
| `phase_1_act.c` | `task_forward_pass`, `task_render_logits_chunk`, `task_compute_probs_loss_cce`, `task_compute_probs_loss_bce` |
| `phase_2_learn_A_production.c` | `task_calculate_module_param_grads_chunk`, `task_backprop_error_to_hidden`, `task_calculate_temp_gradients` |
| `phase_2_learn_B_processing.c` | `task_clip_partial_gradients`, `task_gather_and_permute_grad_h` |
| `phase_2_learn_C_reduction.c` | `task_aggregate_partials`, `task_clip_intermediate_grad`, `execute_reduction_tree`, `task_stabilize_reduce_grad_h` |
| `phase_2_learn_D_backprop.c` | `task_backprop_shared_weights`, `task_backprop_shared_biases`, `task_clip_shared_gradients` |
| `phase_3_update.c` | `task_normalize_gradients`, `task_adam_update`, `task_clamp_temperatures` |

Meson build target: `shared_library('cpu_kernels', ...)` with ISA flag handling from `aec_cpu_isa_flags` option (defaults to `-march=native`). Math and thread library dependencies auto-detected.

### Phase 1 Deliverables Consumed Here

Phase 3A implements the Execution tier (C kernels) that will ultimately be driven by the Phase 1 plan model through the Phase 3B Python FFI layer. The C code does not import the plan model directly — it implements the algorithms specified in `kernels.cl.h` using the CPU's native execution model. The plan model's influence is indirect:

| Shared-Layer Concept | C-Side Manifestation |
| :--- | :--- |
| `KernelDispatchNode.kernel_name` | `task_<kernel_name>` exported function |
| `KernelDispatchNode.scalar_params` | Fields in per-kernel argument structs |
| `KernelDispatchNode.buffer_bindings` | Pointer fields in per-kernel argument structs |
| `KernelDispatchNode.tile_count` | `task_count` parameter to `pool_dispatch_and_wait` |
| `ReductionTreePlan` | `ReductionTreePlan` C struct + `execute_reduction_tree()` |
| `HardwareProfile.simd_width` | Compile-time `SIMD_WIDTH` from `cpu_simd.h` |
| `BufferDescriptor.padded_shape` | SIMD-aligned buffer dimensions in argument structs |
| `PrecisionConfig` | `SCALAR_TYPE` typedef (initially `float` only; FP16 deferred) |

---

## Table of Contents

1. [Scope & Constraints](#1-scope--constraints)
2. [CPU Kernel Architecture Overview](#2-cpu-kernel-architecture-overview)
3. [Target Deliverables](#3-target-deliverables)
4. [Task Breakdown](#4-task-breakdown)
   - [Step 3A.1: Create directory skeleton](#step-3a1-create-directory-skeleton)
   - [Step 3A.2: Implement `cpu_simd.h` — SIMD abstraction layer](#step-3a2-implement-cpu_simdh--simd-abstraction-layer)
   - [Step 3A.3: Implement `cpu_threads.h` / `cpu_threads.c` — thread pool](#step-3a3-implement-cpu_threadsh--cpu_threadsc--thread-pool)
   - [Step 3A.4: Implement `cpu_kernels.h` — ABI surface and argument structs](#step-3a4-implement-cpu_kernelsh--abi-surface-and-argument-structs)
   - [Step 3A.5: Implement Act-phase kernel task functions](#step-3a5-implement-act-phase-kernel-task-functions)
   - [Step 3A.6: Implement Learn-phase gradient production task functions](#step-3a6-implement-learn-phase-gradient-production-task-functions)
   - [Step 3A.7: Implement Learn-phase reduction engine](#step-3a7-implement-learn-phase-reduction-engine)
   - [Step 3A.8: Implement Learn-phase streaming backprop task functions](#step-3a8-implement-learn-phase-streaming-backprop-task-functions)
   - [Step 3A.9: Implement Learn-phase update task functions](#step-3a9-implement-learn-phase-update-task-functions)
   - [Step 3A.10: Implement `get_struct_size_*` verification exports](#step-3a10-implement-get_struct_size_-verification-exports)
   - [Step 3A.11: Create `src/backends/cpu/meson.build`](#step-3a11-create-srcbackendscpumesonbuild)
   - [Step 3A.12: Update architecture `meson.build` for CPU delegation](#step-3a12-update-architecture-mesonbuild-for-cpu-delegation)
   - [Step 3A.13: Validate build on available ISA tiers](#step-3a13-validate-build-on-available-isa-tiers)
5. [SIMD Abstraction Layer Design](#5-simd-abstraction-layer-design)
6. [Thread Pool Design](#6-thread-pool-design)
7. [Argument Struct Inventory](#7-argument-struct-inventory)
8. [Kernel Implementation Strategy](#8-kernel-implementation-strategy)
9. [ABI Stability & Symbol Visibility](#9-abi-stability--symbol-visibility)
10. [Risk Register](#10-risk-register)

---

## 1. Scope & Constraints

### In scope

- Creating the `src/backends/cpu/kernel_sources/` directory with all C source and header files.
- Implementing `cpu_simd.h` — the compile-time ISA detection cascade supporting AVX-512, AVX2, SSE2, ARM NEON, and a scalar fallback (CPU_BACKEND.md). Provides a uniform macro API (`simd_load`, `simd_store`, `simd_fmadd`, `simd_reduce_add`, etc.) and `SIMD_WIDTH`, `SIMD_ALIGNMENT` constants.
- Implementing `cpu_threads.h` / `cpu_threads.c` — a persistent thread pool with C11 threads (falling back to pthreads) exposing `pool_create()`, `pool_destroy()`, and `pool_dispatch_and_wait()`. Worker threads use atomic task claiming for natural load balancing (CPU_BACKEND.md).
- Implementing `cpu_kernels.h` — the public ABI header declaring all `task_<kernel_name>` function signatures, per-kernel argument structs, the `ReductionTreePlan` C struct, and `get_struct_size_*` layout verification exports (ADR-015).
- Implementing all ~22 kernel task functions in C, organized into per-phase source files that mirror the `kernels/*.cl.c` structure, developed against `kernels.cl.h` as the algorithmic reference (ADR-013).
- Implementing the `execute_reduction_tree()` function — the staged reduction with per-stage parallel sum, per-stage parallel clip, and ping-pong buffer management (CPU_BACKEND.md).
- Implementing `get_struct_size_*()` exported functions for every argument struct, enabling Python-side layout verification at library load time (ADR-015).
- Creating `src/backends/cpu/meson.build` with the `shared_library('cpu_kernels', ...)` target, ISA flag handling from `aec_cpu_isa_flags`, and symbol visibility configuration.
- Updating the architecture's top-level `meson.build` to conditionally delegate to the CPU backend via `subdir('src/backends/cpu')` when `aec_backend_cpu` is allowed (ADR-014 Option C).

### Out of scope

- Python-side FFI integration (`_ffi_types.py`, library loading, dispatch table, `CPUPlanRenderer`) — Phase 3B.
- CPU Tier 2 tests — Phase 3C.
- Modifying shared-layer code (plan model, plan builder, contracts).
- Modifying the OpenCL backend or kernel source files (`kernels/*.cl.c`, `kernels/kernels.cl.h`).
- Implementing the Vulkan backend.
- FP16 kernel implementations — the CPU backend targets FP32 initially. FP16 support (requiring AVX-512 FP16 or ARM FP16 extensions) is a post-Phase 3 enhancement gated by hardware availability.
- The `WorkTicket` / `LearnHandle` / `Engine` user-facing API.

### Key constraint: algorithmic fidelity to `kernels.cl.h`

Every C kernel implementation must produce mathematically equivalent results to the algorithm specified in `kernels.cl.h`. The C code implements the same mathematical operation using CPU-native primitives (SIMD intrinsics, scalar loops) instead of OpenCL work-item/work-group primitives. Structural differences (SIMD width, loop structure, memory access patterns) are expected; numerical differences beyond FP accumulation ordering are bugs.

Specifically:
- The `_simd_major` weight layout described in CONTRACT.md and `kernels.cl.h` is preserved — the CPU backend consumes the same padded, SIMD-interleaved memory layout as the GPU backends.
- Padding contracts (`CACHE`, `SIMD`, `NONE`) from CONTRACT.md Article 3.1 are honored — the C code assumes SIMD-aligned, cache-line-padded buffer dimensions.
- Placement strategies (`grid_mod_cls`, `linear_batch`, `linear_generic`) from CONTRACT.md Article 3.2.3 are translated from host-provided tile indices to buffer offsets using the same arithmetic as the OpenCL kernels.

### Key constraint: uniform task function signature

All kernel task functions follow the uniform signature from CPU_BACKEND.md and ADR-015:

```c
void task_<kernel_name>(void* args, uint task_index, uint thread_id);
```

- `args` — pointer to the kernel-specific argument struct (cast internally).
- `task_index` — the work unit index (analogous to `flat_tile_index` for tiled kernels, or batch chunk index for streaming kernels).
- `thread_id` — the executing thread's index within the pool (enables per-thread scratch access).

This uniformity is the CPU backend's analog to the OpenCL `clEnqueueNDRange` dispatch model — the Python renderer calls `pool_dispatch_and_wait(pool, task_fn, args, task_count)` with the same pattern for every kernel.

### Key constraint: no local memory concept

The CPU backend has no equivalent of OpenCL `__local` memory or Vulkan `shared` memory. GPU local memory maps to either:
- **Per-thread stack allocation** — for scratch buffers used within a single task.
- **Pre-allocated per-thread scratch memory** — indexed by `thread_id` for larger temporaries (CPU_BACKEND.md).

`HardwareProfile.max_local_mem_bytes` is `None` for the CPU backend. The plan builder handles this — it produces plans where `local_work_size` in `KernelDispatchNode` is `None`, and the CPU renderer ignores local memory specifications.

---

## 2. CPU Kernel Architecture Overview

The CPU backend maps GPU parallelism to CPU parallelism:

| GPU Concept | CPU Equivalent |
| :--- | :--- |
| N parallel kernel dispatches (tiles) | N tasks submitted to a thread pool |
| Work-group | One task's execution on one core |
| Work-items within a group | SIMD lanes within a task |
| Command queue ordering | `pool_dispatch_and_wait` return = barrier |
| `cl_event` between kernels | Sequential function calls |
| `barrier(CLK_LOCAL_MEM_FENCE)` | Not needed — one thread owns the whole "work-group" |

The execution model is:
1. The Python renderer (Phase 3B) walks the plan DAG in topological order.
2. For each `KernelDispatchNode`, it constructs the argument struct and calls `pool_dispatch_and_wait(pool, task_fn, args, tile_count)`.
3. `pool_dispatch_and_wait` blocks until all tasks complete — this is the synchronization primitive.
4. For `ReductionTreeNode`, the renderer calls `execute_reduction_tree(pool, plan)` which internally performs staged `pool_dispatch_and_wait` calls.
5. For `StreamingLoopNode`, the renderer iterates chunks and dispatches per-chunk.
6. `BarrierNode` is a no-op — synchronization is implicit in the blocking dispatch.
7. `RetrievalNode` returns a zero-copy view of the output numpy array.

---

## 3. Target Deliverables

After Phase 3A completes:

```
src/backends/cpu/
├── __init__.py                         # Minimal (exists from Phase 0 or created here)
├── meson.build                         # NEW: shared_library build target
└── kernel_sources/
    ├── cpu_simd.h                      # NEW: SIMD abstraction layer
    ├── cpu_threads.h                   # NEW: Thread pool declarations
    ├── cpu_threads.c                   # NEW: Thread pool implementation
    ├── cpu_kernels.h                   # NEW: ABI surface — structs + function declarations
    ├── cpu_export.h                    # NEW: CPU_KERNELS_EXPORT visibility macro
    ├── phase_1_act.c                   # NEW: forward_pass, render_logits_chunk
    ├── phase_2_learn_A_production.c    # NEW: cce_probs_loss, bce_probs_loss, module_param_grads
    ├── phase_2_learn_B_processing.c    # NEW: backprop_to_hidden, temp_gradients, clip_partial_grads
    ├── phase_2_learn_C_reduction.c     # NEW: gather_permute, reduction engine, clip_intermediate
    ├── phase_2_learn_D_backprop.c      # NEW: backprop_shared_weights/biases, clip_shared_grads
    └── phase_3_update.c               # NEW: normalize_gradients, adam_update, clamp_temperatures
```

Build output (in `builddir/`):
```
builddir/src/backends/cpu/
└── libcpu_kernels.so                   # (or .dylib / .dll per platform)
```

---

## 4. Task Breakdown

### Step 3A.1: Create directory skeleton

**Action:** Create `src/backends/cpu/kernel_sources/` directory. Ensure `src/backends/cpu/__init__.py` exists (may already exist as a stub from Phase 0).

**Files created:**
- `src/backends/cpu/__init__.py` (if not present)
- `src/backends/cpu/kernel_sources/` (directory)

**Validation:** Directory exists and is empty.

---

### Step 3A.2: Implement `cpu_simd.h` — SIMD abstraction layer

**Action:** Create `src/backends/cpu/kernel_sources/cpu_simd.h` implementing the compile-time ISA detection cascade from CPU_BACKEND.md.

**ISA detection priority:**

```
__AVX512F__  → SIMD_WIDTH=16, SIMD_ALIGNMENT=64, __m512 / __m512i
__AVX2__     → SIMD_WIDTH=8,  SIMD_ALIGNMENT=32, __m256 / __m256i
__SSE2__     → SIMD_WIDTH=4,  SIMD_ALIGNMENT=16, __m128 / __m128i
__ARM_NEON   → SIMD_WIDTH=4,  SIMD_ALIGNMENT=16, float32x4_t
(fallback)   → SIMD_WIDTH=1,  SIMD_ALIGNMENT=4,  scalar float
```

**Macro API surface (uniform across all ISAs):**

| Macro | Semantics |
| :--- | :--- |
| `simd_zero()` | Return zero vector |
| `simd_set1(x)` | Broadcast scalar to all lanes |
| `simd_load(p)` | Aligned load from `p` |
| `simd_loadu(p)` | Unaligned load from `p` |
| `simd_store(p, v)` | Aligned store `v` to `p` |
| `simd_add(a, b)` | Element-wise addition |
| `simd_mul(a, b)` | Element-wise multiplication |
| `simd_fmadd(a, b, c)` | Fused multiply-add: `a*b + c` |
| `simd_max(a, b)` | Element-wise maximum |
| `simd_min(a, b)` | Element-wise minimum |
| `simd_reduce_add(v)` | Horizontal sum → scalar |
| `simd_select(a, b, mask)` | Conditional select: `mask > 0 ? b : a` |

**Aligned allocation utilities:**

```c
static inline void* simd_alloc(size_t bytes);  // posix_memalign / _aligned_malloc
static inline void  simd_free(void* ptr);       // free / _aligned_free
```

**Design decisions:**

1. **Compile-time ISA selection, not runtime dispatch.** The library is compiled once with the target ISA flags (`-mavx2`, `-march=native`, etc.). Runtime multi-ISA dispatch (CPU feature detection + function pointer tables) is a complexity cost not justified at this stage — ADR-014's `aec_cpu_isa_flags` build option provides the configuration point.

2. **No FP16 SIMD initially.** FP16 requires AVX-512 FP16 (`__AVX512FP16__`) or ARM FP16 extensions. The initial implementation targets FP32 only. FP16 can be added as a second `cpu_simd_fp16.h` header when hardware availability justifies it.

3. **`simd_fmadd` falls back to `mul + add` on SSE2.** SSE2 lacks native FMA. The macro expands to `simd_add(simd_mul(a, b), c)`. This introduces a minor numerical divergence vs. FMA-capable ISAs — Tier 2 tolerances account for this.

**Validation:** The header compiles without warnings under `-Wall -Wextra` for each ISA tier. `SIMD_WIDTH` and `SIMD_ALIGNMENT` are defined to expected values.

---

### Step 3A.3: Implement `cpu_threads.h` / `cpu_threads.c` — thread pool

**Action:** Create the persistent thread pool implementation from CPU_BACKEND.md.

**Public API (in `cpu_threads.h`):**

```c
typedef struct ThreadPool ThreadPool;

// Create a persistent thread pool with num_threads worker threads.
ThreadPool* pool_create(uint num_threads);

// Destroy the thread pool, joining all worker threads.
void pool_destroy(ThreadPool* pool);

// Submit task_count independent tasks and block until all complete.
// fn(args, task_index, thread_id) is called for task_index in [0, task_count).
void pool_dispatch_and_wait(ThreadPool* pool,
                            void (*fn)(void*, uint, uint),
                            void* args,
                            uint task_count);
```

**Internal design (in `cpu_threads.c`):**

```c
typedef struct {
    void (*function)(void* args, uint task_index, uint thread_id);
    void*  args;
    uint   task_count;
    atomic_uint next_task;
    atomic_uint tasks_completed;
} TaskBatch;

struct ThreadPool {
    thrd_t*    threads;         // C11 threads array
    uint       thread_count;
    TaskBatch* current_batch;
    mtx_t      wake_mutex;
    cnd_t      wake_cond;
    cnd_t      done_cond;
    int        shutdown;
};
```

**Worker loop:** Each worker thread atomically claims task indices via `atomic_fetch_add(&batch->next_task, 1)`. The last thread to complete signals `done_cond`. This provides natural load balancing — tasks of varying duration are distributed as threads become available, avoiding the pathological case of pre-partitioning where one thread is assigned all the expensive tasks.

**Threading API fallback:** Prefer C11 `<threads.h>` (`thrd_t`, `mtx_t`, `cnd_t`). If `<threads.h>` is unavailable (some older compilers), fall back to POSIX `<pthread.h>` (`pthread_t`, `pthread_mutex_t`, `pthread_cond_t`). The fallback is selected via preprocessor detection:

```c
#if defined(__STDC_VERSION__) && __STDC_VERSION__ >= 201112L && !defined(__STDC_NO_THREADS__)
    #include <threads.h>
    #define THREADS_USE_C11
#else
    #include <pthread.h>
    #define THREADS_USE_POSIX
#endif
```

**Validation:** `pool_create(4)` → 4 threads; `pool_dispatch_and_wait` with a trivial task function completes; `pool_destroy` joins cleanly. No memory leaks under Valgrind/AddressSanitizer.

---

### Step 3A.4: Implement `cpu_kernels.h` — ABI surface and argument structs

**Action:** Create the public ABI header declaring all exported types and functions. This header is the definitive C-side contract that the Python FFI layer (Phase 3B) binds against.

**Header structure:**

```c
#ifndef CPU_KERNELS_H
#define CPU_KERNELS_H

#include "cpu_simd.h"
#include "cpu_threads.h"
#include "cpu_export.h"

#include <stdint.h>
#include <string.h>
#include <math.h>

// ============================================================
// Build Configuration Constants
// ============================================================
#define LOCAL_MEM_BANK_PADDING     1
#define NUMERICAL_STABILITY_EPSILON 1e-7f
#define CACHE_LINE_BYTES           64

#ifndef C_TILE_SIZE
#define C_TILE_SIZE SIMD_WIDTH
#endif

#define PROBLEM_TYPE_CCE  0
#define PROBLEM_TYPE_BCE  1

typedef uint32_t uint;

// ============================================================
// Argument Structures (one per kernel)
// ============================================================
// ... ForwardPassArgs, CceChunkArgs, ClipPartialsArgs, AdamUpdateArgs, etc.

// ============================================================
// Task Functions (uniform signature)
// ============================================================
CPU_KERNELS_EXPORT void task_forward_pass(void* args, uint task_index, uint thread_id);
CPU_KERNELS_EXPORT void task_render_logits(void* args, uint task_index, uint thread_id);
// ... all ~18 task functions

// ============================================================
// Reduction Engine
// ============================================================
CPU_KERNELS_EXPORT void execute_reduction_tree(ThreadPool* pool, ReductionTreePlan* plan);

// ============================================================
// Thread Pool Lifecycle
// ============================================================
CPU_KERNELS_EXPORT ThreadPool* pool_create(uint num_threads);
CPU_KERNELS_EXPORT void pool_destroy(ThreadPool* pool);
CPU_KERNELS_EXPORT void pool_dispatch_and_wait(ThreadPool* pool,
                                                void (*fn)(void*, uint, uint),
                                                void* args, uint task_count);

// ============================================================
// Layout Verification Exports
// ============================================================
CPU_KERNELS_EXPORT size_t get_struct_size_forward_pass_args(void);
CPU_KERNELS_EXPORT size_t get_struct_size_cce_chunk_args(void);
// ... one per argument struct
CPU_KERNELS_EXPORT size_t get_struct_size_reduction_tree_plan(void);

#endif // CPU_KERNELS_H
```

**`cpu_export.h` — symbol visibility macro:**

```c
#ifndef CPU_EXPORT_H
#define CPU_EXPORT_H

#if defined(_WIN32) || defined(__CYGWIN__)
    #ifdef CPU_KERNELS_BUILDING
        #define CPU_KERNELS_EXPORT __declspec(dllexport)
    #else
        #define CPU_KERNELS_EXPORT __declspec(dllimport)
    #endif
#elif defined(__GNUC__) || defined(__clang__)
    #define CPU_KERNELS_EXPORT __attribute__((visibility("default")))
#else
    #define CPU_KERNELS_EXPORT
#endif

#endif // CPU_EXPORT_H
```

**Argument struct inventory:** See §7 for the complete list with field definitions. Each struct's fields are ordered to match the plan model's `KernelDispatchNode.buffer_bindings` and `scalar_params` — buffer pointers first (in contract parameter order), then scalar values.

**Validation:** Header compiles without warnings under all target compilers. All `CPU_KERNELS_EXPORT` symbols are visible in the compiled library (`nm -D libcpu_kernels.so | grep ' T '`).

---

### Step 3A.5: Implement Act-phase kernel task functions

**Action:** Create `src/backends/cpu/kernel_sources/phase_1_act.c` implementing Act-phase kernels.

**Kernels implemented:**

| Task Function | DAG Node | Algorithm Reference | SIMD Value |
| :--- | :--- | :--- | :--- |
| `task_forward_pass` | Node 4 | `kernels.cl.h` `forward_pass` | **Critical** — tiled matrix-vector multiply with SIMD-major weight layout |
| `task_render_logits` | Node 5 | `kernels.cl.h` `render_logits_chunk` | **Critical** — tiled matrix-vector multiply for output logits |

**`task_forward_pass` implementation strategy:**

The forward pass is a matrix-vector multiplication with ReLU activation and sample masking. The `_simd_major` weight layout from CONTRACT.md places `SIMD_WIDTH` weights for different hidden units but the same input contiguously. This maps directly to a vectorized dot product:

```
For each hidden block (SIMD_WIDTH hidden units):
    accum = simd_load(biases[h_offset])
    For each input dimension i:
        accum = simd_fmadd(weights[hb][i], broadcast(input[i]), accum)
    activated = simd_max(zero, accum)        // ReLU
    activated = simd_mul(activated, mask)     // sample mask
    simd_store(hidden[h_offset], activated)
    relu_mask = simd_select(zero, mask, accum)  // hidden mask
    simd_store(hidden_mask[h_offset], relu_mask)
```

Each task processes one sample (`task_index` = sample index within the batch chunk). All tasks are independent — no cross-sample synchronization.

**`task_render_logits` implementation strategy:**

Similar tiled matrix-vector multiply, computing `W_output × hidden_activations` to produce per-module logits. Temperature-scaled division is applied per-module.

**Design decision — tile index mapping:** On CPU, `task_index` maps directly to the sample or tile being processed. For `grid_mod_cls` placement (used by gradient kernels), the task function decomposes `flat_tile_index` into `(module_chunk, class_chunk)` using the same modular arithmetic as the OpenCL kernels:

```c
uint module_chunk = flat_tile_index / num_class_chunks;
uint class_chunk  = flat_tile_index % num_class_chunks;
```

**Validation:** Functions compile. Output shape and buffer access patterns match `kernels.cl.h` specification. Correctness validated in Phase 3C.

---

### Step 3A.6: Implement Learn-phase gradient production task functions

**Action:** Create two source files covering Learn-phase gradient production and processing:

**`phase_2_learn_A_production.c`:**

| Task Function | DAG Node | Strategy | Notes |
| :--- | :--- | :--- | :--- |
| `task_cce_probs_loss` | Node 6 | B (separate kernel) | Softmax + cross-entropy loss |
| `task_bce_probs_loss` | Node 7 | B (separate kernel) | Sigmoid + binary cross-entropy loss |
| `task_module_param_grads` | Node 8 | A (FLAG dispatch) | `PROBLEM_TYPE_CCE` / `PROBLEM_TYPE_BCE` flag selects gradient formula |

**`phase_2_learn_B_processing.c`:**

| Task Function | DAG Node | Strategy | Notes |
| :--- | :--- | :--- | :--- |
| `task_backprop_to_hidden` | Node 9 | A (FLAG dispatch) | Backprop error through output weights to hidden layer |
| `task_temp_gradients` | Node 10 | A (FLAG dispatch) | Temperature gradient computation |
| `task_clip_partial_grads` | Node 11 | — | Partial-Group-Wise clipping (L2 norm + conditional scale) |

**Implementation notes:**

- **Strategy A kernels** (Nodes 8, 9, 10) receive a `uint problem_type` field in their argument struct. The task function uses an `if (args->problem_type == PROBLEM_TYPE_CCE)` branch — the OpenCL analog is `src_scalar_FLAG_problem_type`. The branch occurs once per task invocation, not per element, so branch prediction is not a concern.

- **Softmax numerics** (Node 6): The CCE loss kernel computes `exp(logit - max_logit)` for numerical stability, followed by normalization. The CPU implementation uses `expf()` from `<math.h>`. SIMD exponentiation is not attempted — the `exp` call is scalar within a SIMD-width loop. This is acceptable because Node 6 is mixed compute/memory-bound, not purely compute-bound.

- **Sigmoid numerics** (Node 7): The BCE loss kernel uses a two-branch numerically stable sigmoid: for `z ≥ 0`, compute `1 / (1 + exp(-z))`; for `z < 0`, compute `exp(z) / (1 + exp(z))`. This avoids `exp()` overflow on large-magnitude logits. Both the CPU kernel (`task_bce_probs_loss`) and the OpenCL kernel (`compute_probs_loss_bce_chunk`) use this identical two-branch form, matching the numpy reference (`_sigmoid` in `numpy_forward.py`).

- **`task_clip_partial_grads`** implements Partial-Group-Wise clipping within each tile: compute L2 norm across all partial gradient buffers for that tile, compare against threshold, conditionally scale. The SIMD implementation vectorizes the norm computation and the scaling loop.

---

### Step 3A.7: Implement Learn-phase reduction engine

**Action:** Create `src/backends/cpu/kernel_sources/phase_2_learn_C_reduction.c` implementing the gather/permute, staged reduction, and clipping kernels.

**Kernels implemented:**

| Task Function | DAG Node | Description |
| :--- | :--- | :--- |
| `task_gather_permute_grad_h` | Node 13 | Item Synchronization Point — scatter/gather from tiled partial layout to contiguous per-sample layout |
| `task_reduce_one_node` | Nodes 14/15/20 | Sum K partials into one output (one reduction node) |
| `task_clip_one_node` | Nodes 15/20 | Component-wise clip one reduced buffer (one reduction node) |
| `task_stabilize_reduce_grad_h` | Node 16 | Specialized grad_h reduction — parallel across rows; internal multi-stage reduction (ADR-005 Node 16 opacity) |

**`execute_reduction_tree()` — the staged reduction engine:**

```c
CPU_KERNELS_EXPORT void execute_reduction_tree(ThreadPool* pool,
                                                ReductionTreePlan* plan);
```

This function orchestrates the full multi-stage reduction with per-stage clipping, as specified in CPU_BACKEND.md:

```
For each stage s in [0, num_stages):
    1. Sum phase — all nodes at this stage are independent:
       pool_dispatch_and_wait(pool, task_reduce_one_node, &stage_args, num_nodes)
    2. Clip phase — all nodes at this stage are independent:
       pool_dispatch_and_wait(pool, task_clip_one_node, &stage_args, num_nodes)
    3. Swap ping-pong buffers (current dst becomes next src)
```

**Threshold calculation per stage (Quadratic Scaling Policy):**

```c
uint j = plan->num_stages - 1 - stage;
float t_policy = plan->t_algorithmic + plan->lambda * (float)(j * j);
float t_safety = plan->fp_max / (float)plan->stage_fan_in[stage];
float threshold = fminf(t_policy, t_safety);
```

This implements CONCEPT.md §3.3's policy: $T_j = T_{\text{algorithmic}} + \lambda \cdot j^2$, clamped by $T_{\text{safety},j} = \text{FP\_FORMAT\_MAX} / K_j$.

**`ReductionTreePlan` C struct:**

```c
typedef struct {
    const float* partial_collection;       // All partials in one buffer
    const uint*  offset_lists_flat;        // Flattened offset lists per stage
    const uint*  stage_offsets_into_list;   // Start index for each stage's offsets
    const uint*  stage_fan_in;             // K per stage
    const uint*  stage_node_counts;        // Number of reduction nodes per stage
    float*       staging_buffers[2];       // Ping-pong intermediate buffers
    float*       output;                   // Final reduced result
    uint         partial_width;            // Elements per partial
    uint         num_stages;
    float        t_algorithmic;
    float        lambda;
    float        fp_max;
    float        epsilon;
} ReductionTreePlan;
```

**Node 16 (stabilize_reduce_grad_h):** Per ADR-005, Node 16's internal multi-stage reduction is an Execution-tier concern. The upstream Item Synchronization Point (Node 13) guarantees contiguous input, so the reduction logic is self-contained within the task function. The CPU implementation performs a per-row reduction with SIMD, followed by stabilization clipping. No `execute_reduction_tree` call is needed — the task function handles everything internally.

**Validation:** `execute_reduction_tree` compiles and links. Offset list traversal logic matches the plan model's `ReductionTreePlan.initial_offset_list` semantics. Correctness validated in Phase 3C.

---

### Step 3A.8: Implement Learn-phase streaming backprop task functions

**Action:** Create `src/backends/cpu/kernel_sources/phase_2_learn_D_backprop.c` implementing streaming backpropagation kernels.

**Kernels implemented:**

| Task Function | DAG Node | Description |
| :--- | :--- | :--- |
| `task_backprop_shared_weights` | Node 17 | Outer product: grad_h × input^T, accumulated per batch chunk |
| `task_backprop_shared_biases` | Node 18 | Reduction: sum grad_h across batch chunk |
| `task_clip_shared_grads` | Node 19 | Component-wise clip per batch chunk |

These are streaming loop body kernels — the Python renderer (Phase 3B) iterates batch chunks and dispatches per-chunk via `pool_dispatch_and_wait`. Each task within a chunk processes a subset of the batch chunk (e.g., one sample or one output dimension).

**`task_backprop_shared_weights`** is the second most compute-intensive kernel after `forward_pass` — it performs a vectorized outer product using the SIMD-major weight layout. The implementation follows the same `simd_fmadd` accumulation pattern as `task_forward_pass` but in the backward direction.

**Validation:** Functions compile. Buffer access patterns are consistent with the streaming loop's per-chunk parameter stride semantics.

---

### Step 3A.9: Implement Learn-phase update task functions

**Action:** Create `src/backends/cpu/kernel_sources/phase_3_update.c` implementing the final update-phase kernels.

**Kernels implemented:**

| Task Function | DAG Node | Description |
| :--- | :--- | :--- |
| `task_normalize_gradients` | Node 21 | Element-wise gradient normalization by batch count |
| `task_adam_update` | Node 24 | Adam optimizer step: m₁, m₂ moments + bias-corrected update |
| `task_clamp_temperatures` | Node 25 | Clamp temperature parameters to valid range |

**`task_adam_update` implementation notes:**

The Adam update follows the standard formulation:
```
m1 = beta1 * m1 + (1 - beta1) * grad
m2 = beta2 * m2 + (1 - beta2) * grad²
m1_hat = m1 / (1 - beta1^t)
m2_hat = m2 / (1 - beta2^t)
param -= lr * m1_hat / (sqrt(m2_hat) + epsilon)
```

The `beta1_pow_t` and `beta2_pow_t` values are computed on the Python side using FP64 arithmetic (per CONCEPT.md §6 hierarchical design: the host computes `beta1**t` at higher precision) and passed as `float` scalars in the argument struct.

The SIMD implementation vectorizes the element-wise operations. The `sqrtf` is called per-element within a SIMD-width loop (no horizontal SIMD sqrt needed).

**`task_clamp_temperatures`** is trivial — element-wise clamp of temperature parameters to `[T_min, T_max]`. On CPU this is a single-threaded task (Node 25 in the plan has `tile_count=1` for the temperature vector).

---

### Step 3A.10: Implement `get_struct_size_*` verification exports

**Action:** Add one `get_struct_size_*()` function per argument struct to the library's public ABI.

These functions return `sizeof(StructName)` for each argument struct. The Python FFI layer (Phase 3B) calls these at library load time and asserts that the Python-side ctypes struct definitions have matching sizes (ADR-015 `_verify_layouts()`).

**Implementation (appended to `cpu_kernels.h` implementation file or a dedicated `cpu_verify.c`):**

```c
CPU_KERNELS_EXPORT size_t get_struct_size_forward_pass_args(void) {
    return sizeof(ForwardPassArgs);
}
CPU_KERNELS_EXPORT size_t get_struct_size_cce_chunk_args(void) {
    return sizeof(CceChunkArgs);
}
// ... one per argument struct
CPU_KERNELS_EXPORT size_t get_struct_size_reduction_tree_plan(void) {
    return sizeof(ReductionTreePlan);
}
```

**Total exports:** One per argument struct (~12 structs) + 1 for `ReductionTreePlan` = ~13 verification functions.

**Validation:** All `get_struct_size_*` symbols are exported and callable. Return values match `sizeof()` for the target platform.

---

### Step 3A.11: Create `src/backends/cpu/meson.build`

**Action:** Create the Meson build file for the CPU backend's shared library.

**Target `meson.build`:**

```meson
# src/backends/cpu/meson.build

cpu_kernel_sources = files(
    'kernel_sources/cpu_threads.c',
    'kernel_sources/phase_1_act.c',
    'kernel_sources/phase_2_learn_A_production.c',
    'kernel_sources/phase_2_learn_B_processing.c',
    'kernel_sources/phase_2_learn_C_reduction.c',
    'kernel_sources/phase_2_learn_D_backprop.c',
    'kernel_sources/phase_3_update.c',
)

# ISA flags: use aec_cpu_isa_flags if provided, else -march=native
cpu_isa_flags = get_option('aec_cpu_isa_flags')
if cpu_isa_flags.length() == 0
    cpu_isa_flags = ['-march=native']
endif

cpu_c_args = [
    '-DCPU_KERNELS_BUILDING',
    '-fvisibility=hidden',
] + cpu_isa_flags

cpu_lib = shared_library(
    'cpu_kernels',
    cpu_kernel_sources,
    c_args: cpu_c_args,
    include_directories: include_directories('kernel_sources'),
    install: true,
    install_dir: py.get_install_dir() / 'averaging_ensembled_classifier' / 'backends' / 'cpu',
    gnu_symbol_visibility: 'hidden',
)

# Install Python source files for the CPU backend
cpu_py_sources = files(
    '__init__.py',
)
py.install_sources(
    cpu_py_sources,
    subdir: 'averaging_ensembled_classifier' / 'backends' / 'cpu',
)
```

**Key build configuration:**

- **`-DCPU_KERNELS_BUILDING`** — activates `__declspec(dllexport)` on Windows via `cpu_export.h`.
- **`gnu_symbol_visibility: 'hidden'`** — all symbols hidden by default; only `CPU_KERNELS_EXPORT`-decorated symbols are visible. This produces a clean ABI surface of ~24 symbols.
- **`-march=native`** (default when `aec_cpu_isa_flags` is empty) — lets the compiler auto-detect the host CPU's ISA capabilities. The `cpu_simd.h` `#ifdef` cascade then selects the appropriate SIMD tier.
- **Math library linking** — `-lm` may be needed on some platforms for `sqrtf`, `expf`, `logf`, `fminf`, `fmaxf`. Meson detects this via `cc.find_library('m', required: false)`.

---

### Step 3A.12: Update architecture `meson.build` for CPU delegation

**Action:** Add conditional `subdir('src/backends/cpu')` to the architecture's top-level `meson.build`, guarded by the `aec_backend_cpu` feature option (ADR-014 Option C).

**Change to `meson.build`:**

```meson
backend_cpu = get_option('aec_backend_cpu')

# ... existing OpenCL subdir() ...

if backend_cpu.allowed()
    subdir('src/backends/cpu')
endif
```

The `aec_backend_cpu` option is already declared in `meson.options` (created in Phase 0):

```meson
option('aec_backend_cpu', type: 'feature', value: 'auto',
       description: 'Build CPU SIMD kernel shared library')
```

When `auto` (default), Meson includes the CPU backend if a C compiler is available. When `enabled`, the build fails if no C compiler is found. When `disabled`, the CPU backend is skipped unconditionally.

**Validation:** `meson setup builddir -Daec_backend_cpu=enabled` succeeds and produces `libcpu_kernels.so` in the build output. `meson setup builddir -Daec_backend_cpu=disabled` succeeds without compiling any CPU kernel sources.

---

### Step 3A.13: Validate build on available ISA tiers

**Action:** Verify that the library compiles cleanly under multiple ISA flag configurations:

| Configuration | Flags | Expected `SIMD_WIDTH` |
| :--- | :--- | :--- |
| Native (default) | `-march=native` | Platform-dependent |
| Explicit AVX2 | `-mavx2 -mfma` | 8 |
| Explicit SSE2 | `-msse2` | 4 |
| Scalar fallback | `-mno-sse -mno-avx` (if feasible) | 1 |

For cross-platform validation (ARM NEON), a CI runner with ARM hardware is required — this is a CI infrastructure concern, not a blocking validation step.

**Validation criteria:**
1. Zero compiler warnings at `-Wall -Wextra -Wpedantic`.
2. All `CPU_KERNELS_EXPORT` symbols are present in the shared library (`nm -D` / `dumpbin /EXPORTS`).
3. `get_struct_size_*` functions return non-zero values.
4. Library loads via `dlopen` / `ctypes.CDLL` without errors.

---

## 5. SIMD Abstraction Layer Design

The `cpu_simd.h` layer provides a common API across four ISA tiers. The design prioritizes:

1. **Zero overhead** — all macros expand directly to intrinsics. No function call overhead, no vtable dispatch.
2. **Uniform semantics** — code written against the macro API produces correct results on all tiers. The only difference is performance (wider SIMD = more parallelism).
3. **Compile-time resolution** — ISA selection is a compile-time decision. No runtime feature detection. The `aec_cpu_isa_flags` build option controls the target.

### ISA-specific considerations

| ISA | FMA Support | Horizontal Sum | Select/Blend |
| :--- | :--- | :--- | :--- |
| AVX-512 | Native (`_mm512_fmadd_ps`) | Native (`_mm512_reduce_add_ps`) | Mask blend (`_mm512_mask_blend_ps`) |
| AVX2 | Native with FMA3 (`_mm256_fmadd_ps`) | Manual: extract + hadd | `_mm256_blendv_ps` |
| SSE2 | Emulated: `mul + add` | Manual: movehdup + shuffle | Not used (scalar fallback for select) |
| NEON | Native (`vfmaq_f32`) | Native (`vaddvq_f32`) | `vbslq_f32` |
| Scalar | Direct arithmetic | Identity | Ternary operator |

### Alignment contract

All buffer pointers passed to task functions must be `SIMD_ALIGNMENT`-aligned. The CPU buffer allocator (Phase 3B) guarantees this using `posix_memalign` / `_aligned_malloc`. The CACHE padding contract from CONTRACT.md Article 3.1 ensures buffer dimensions are multiples of `SIMD_WIDTH`, eliminating scalar cleanup loops.

---

## 6. Thread Pool Design

### Sizing

The pool is created once per `CPUPlanRenderer` lifetime (Phase 3B). Default thread count: `std::thread::hardware_concurrency()` equivalent, queried via platform-specific API or `sysconf(_SC_NPROCESSORS_ONLN)`. The Python-side `discovery.py` determines the thread count and passes it through the FFI layer.

### Synchronization model

`pool_dispatch_and_wait` is blocking — the calling thread waits on `done_cond` until all tasks complete. This provides the DAG edge contract: when `pool_dispatch_and_wait` returns, all tasks for that node are complete, and the next node's inputs are ready.

### Edge case: `task_count = 0`

When `task_count` is 0 (e.g., a degenerate plan with no tiles), `pool_dispatch_and_wait` returns immediately without signaling workers.

### Edge case: `task_count = 1`

For single-task dispatches (e.g., Node 25 `clamp_temperatures`), the thread pool has overhead vs. a direct call. This overhead is negligible — the task is trivial and the mutex round-trip is ~100ns.

---

## 7. Argument Struct Inventory

Each kernel has a dedicated argument struct. Fields are ordered: buffer pointers first, then scalar values.

| Struct Name | Kernel | Buffer Fields | Scalar Fields |
| :--- | :--- | :--- | :--- |
| `ForwardPassArgs` | `task_forward_pass` | `input`, `sample_mask`, `weights_shared_simd_major`, `biases_shared`, `hidden_activations`, `hidden_mask` | `batch_chunk_offset`, `batch_chunk_count`, `total_batch_count`, `padded_input_count`, `padded_hidden_count` |
| `RenderLogitsArgs` | `task_render_logits` | `hidden_activations`, `output_weights_simd_major`, `output_biases`, `temps`, `logits` | Per-module geometry scalars |
| `CceChunkArgs` | `task_cce_probs_loss` | `logits`, `temps`, `targets`, `sample_mask`, `partial_probs`, `final_loss` | Tiling, geometry, and count scalars |
| `BceChunkArgs` | `task_bce_probs_loss` | Similar to CCE with sigmoid-specific buffers | Similar geometry scalars |
| `ModuleParamGradsArgs` | `task_module_param_grads` | Input, hidden, probs/targets, output gradient buffers | Geometry + `problem_type` flag |
| `BackpropToHiddenArgs` | `task_backprop_to_hidden` | Output weights, error signal, hidden mask, grad_h output | Geometry + `problem_type` flag |
| `TempGradientsArgs` | `task_temp_gradients` | Logits, probs, targets, partial_grad_temps output | Geometry + `problem_type` flag |
| `ClipPartialsArgs` | `task_clip_partial_grads` | Partial gradient buffers (weights, biases, temps, hidden) + clipped output buffers | Threshold, epsilon, geometry |
| `GatherPermuteArgs` | `task_gather_permute_grad_h` | Source tiled grad_h buffer, destination contiguous grad_h buffer | Geometry scalars |
| `StabilizeReduceArgs` | `task_stabilize_reduce_grad_h` | Input grad_h, output stabilized grad_h | Threshold, geometry |
| `BackpropSharedWeightsArgs` | `task_backprop_shared_weights` | grad_h, input, partial_grad_shared_weights | Chunk geometry |
| `BackpropSharedBiasesArgs` | `task_backprop_shared_biases` | grad_h, partial_grad_shared_biases | Chunk geometry |
| `ClipSharedGradsArgs` | `task_clip_shared_grads` | Shared gradient buffers | Threshold, epsilon |
| `NormalizeGradientsArgs` | `task_normalize_gradients` | Gradient buffers, output normalized buffers | Batch count, parameter count |
| `AdamUpdateArgs` | `task_adam_update` | `final_grad`, `parameters`, `m1`, `m2` | `learning_rate`, `beta1_pow_t`, `beta2_pow_t`, `beta1`, `beta2`, `epsilon`, `parameter_count` |
| `ClampTemperaturesArgs` | `task_clamp_temperatures` | `temperatures` | `temp_min`, `temp_max`, `count` |
| `ReductionTreePlan` | `execute_reduction_tree` | `partial_collection`, `offset_lists_flat`, staging buffers, `output` | `partial_width`, `num_stages`, policy parameters |

---

## 8. Kernel Implementation Strategy

### Principle: extraction, not rewrite

The C kernel implementations follow the mathematical operations specified in `kernels.cl.h`. The translation from OpenCL to C follows a mechanical mapping:

| OpenCL Concept | C Equivalent |
| :--- | :--- |
| `get_global_id(0)` | `task_index` (one task per logical work-unit) |
| `get_local_id(0)` | Not applicable — one thread owns the full "work-group" |
| `__local float shared[N]` | Stack-allocated `float scratch[N]` or pre-allocated TLS |
| `barrier(CLK_LOCAL_MEM_FENCE)` | Not needed — single-threaded per task |
| `SCALAR_TYPE` | `float` (FP32 only initially) |
| `SIMD_WIDTH * i` | `SIMD_WIDTH * i` (same constant, from `cpu_simd.h`) |
| `clamp(x, lo, hi)` | `fminf(fmaxf(x, lo), hi)` |
| `native_exp(x)` | `expf(x)` |
| `native_log(x)` | `logf(x)` |

### Hot loop vectorization

Compute-bound kernels (Nodes 4, 5, 8, 9, 17) use explicit SIMD in their inner loops. The pattern is:

```c
for (uint e = 0; e < padded_dimension; e += SIMD_WIDTH) {
    simd_float a = simd_load(&src[e]);
    simd_float b = simd_load(&weights[e]);
    accum = simd_fmadd(a, b, accum);
}
```

Memory-bound kernels (Nodes 11, 13, 21, 24, 25) use SIMD for element-wise operations but derive less benefit.

### Transcendental functions

`expf`, `logf`, `sqrtf`, `fabsf` are called per-element within SIMD-width loops. SIMD transcendental approximations (e.g., polynomial `exp` approximation) are not implemented initially — they introduce numerical divergence that complicates Tier 2 validation. Standard `<math.h>` functions are sufficient for correctness; SIMD transcendentals can be added as a post-Phase 3 optimization once the baseline is validated.

---

## 9. ABI Stability & Symbol Visibility

### Exported symbol count

The library exports exactly:
- 3 thread pool functions (`pool_create`, `pool_destroy`, `pool_dispatch_and_wait`)
- ~18 `task_*` kernel functions
- 1 `execute_reduction_tree`
- ~13 `get_struct_size_*` verification functions

Total: **~35 symbols**. All other symbols are hidden via `-fvisibility=hidden` and `gnu_symbol_visibility: 'hidden'`.

### ABI versioning

No explicit ABI versioning is implemented at this stage. The `_verify_layouts()` mechanism (Phase 3B) provides a behavioral compatibility check that catches struct drift. If ABI stability becomes a concern post-Phase 3, a versioned ABI (e.g., `cpu_kernels_get_abi_version()`) can be added as a new export — this is an additive change per CONCEPT.md §1.

### Platform-specific library naming

| Platform | Library Name | Extension |
| :--- | :--- | :--- |
| Linux | `libcpu_kernels.so` | `.so` |
| macOS | `libcpu_kernels.dylib` | `.dylib` |
| Windows | `cpu_kernels.dll` | `.dll` |

Meson handles the naming convention automatically. The Python-side `importlib.resources` discovery (Phase 3B) resolves the platform-specific name at runtime.

---

## 10. Risk Register

| Risk | Likelihood | Impact | Mitigation |
| :--- | :--- | :--- | :--- |
| ISA detection cascade has gaps on exotic platforms | Low | Medium | Scalar fallback guarantees compilation everywhere; CI testing on x86-64 + ARM covers primary targets |
| C11 `<threads.h>` unavailable on target compiler | Medium | Low | POSIX pthreads fallback implemented; preprocessor detection selects automatically |
| Struct padding differences between C compiler and Python ctypes | Medium | High | `get_struct_size_*` exports + Phase 3B `_verify_layouts()` catches size mismatches; Tier 2 catches field reordering |
| Numerical divergence from OpenCL due to FMA availability | Medium | Low | SSE2's emulated FMA introduces ~1 ULP difference; Tier 2 tolerances account for this (FP32 `atol=1e-5`) |
| Thread pool contention on high-core-count systems | Low | Medium | Atomic task claiming has proven scalability; lock-free pattern avoids mutex bottleneck |
| `-march=native` produces non-portable binary | Low | Low | This is expected — CPU backend is compiled per-machine. Explicit `aec_cpu_isa_flags` overrides for CI/packaging |
| `expf`/`logf` precision varies across platforms | Low | Medium | Using standard `<math.h>` guarantees IEEE 754 compliance; platform-specific ULP differences within Tier 2 tolerances |
| Library fails to link due to missing `-lm` | Low | Low | Meson build includes `cc.find_library('m', required: false)` to detect and link math library |
