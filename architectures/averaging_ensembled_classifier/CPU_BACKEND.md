# CPU Back-End: Architecture for SIMD + Multi-Core

> **Implementation Status: ✅ Complete** — Phase 3 implemented; Phase 7 mixed-precision migration complete; Phase 9C FP8 CPU backend complete; Phase 10 state-precision accumulation complete. All components described in this document have been realized in `src/backends/cpu/` (10 Python modules + 1 generated module, 2 C source files, 7 C headers) with a full Tier 2 test suite (21 test modules in `tests/tier2/cpu/`). The Meson `shared_library('cpu_kernels', ...)` build target produces `libcpu_kernels.so` with up to 32 precision-variant instantiations via the three-axis `STORAGE_T`/`COMPUTE_T`/`STATE_T` macro system (ADR-023, ADR-025): 5 FP32/FP64-storage variants (unconditional), 8 FP8 E4M3/E5M2 variants with FP32/FP64 compute (unconditional), and 9 FP16-storage variants plus 10 FP8 variants with FP16 compute/state (conditional on `_Float16` availability). See [PHASE-3A](plan/PHASE-3A-CPU-KERNEL-LIBRARY-AND-BUILD.md), [PHASE-3B](plan/PHASE-3B-CPU-FFI-AND-RENDERER.md), [PHASE-3C](plan/PHASE-3C-CPU-TIER2-TESTS.md) for base implementation, [PHASE-7C](plan/PHASE-7C-MIXED-PRECISION-CPU-VULKAN-AND-FINAL.md) for the precision migration, [PHASE-9C](plan/PHASE-9C-FP8-CPU-BACKEND.md) for FP8 support, and [PHASE-10](plan/PHASE-10-STATE-PRECISION-ACCUMULATION.md) for state-precision accumulation.

## Design Constraints

Two hard requirements for the CPU back-end design:

1. **SIMD** — inner loops must be vectorized, not scalar
2. **Multi-core** — work decomposition must map to threads, not just loop iterations

This means the CPU back-end isn't "run the kernel logic sequentially" — it's a parallel execution engine in its own right, just with different primitives than OpenCL or Vulkan.

---

## Threading Model

### The GPU→CPU Mapping

The architecture already decomposes work into independent tiles. The mapping is direct:

| GPU Concept                          | CPU Multi-Core Equivalent                           |
| ------------------------------------ | --------------------------------------------------- |
| N parallel kernel dispatches (tiles) | N tasks submitted to a thread pool                  |
| Work-group                           | One task's execution on one core                    |
| Work-items within a group            | SIMD lanes within a task                            |
| Command queue ordering               | Task dependency graph in the scheduler              |
| `cl_event` between kernels           | Countdown latch / barrier between task batches      |
| `barrier(CLK_LOCAL_MEM_FENCE)`       | Not needed — one thread owns the whole "work-group" |

### Thread Pool, Not Thread-Per-Dispatch

Spawning threads per kernel dispatch is catastrophic for the reduction engine, which may dispatch hundreds of small kernels. A persistent thread pool is mandatory:

```cpp
// cpu_thread_pool.h

#include <stdatomic.h>
#include <threads.h>  // C11 threads, or pthreads

typedef struct {
    void (*function)(void* args, uint task_index, uint thread_id);
    void*  args;
    uint   task_count;
    atomic_uint next_task;
    atomic_uint tasks_completed;
} TaskBatch;

typedef struct {
    thrd_t*    threads;
    uint       thread_count;
    TaskBatch* current_batch;
    mtx_t      wake_mutex;
    cnd_t      wake_cond;
    cnd_t      done_cond;
    int        shutdown;
} ThreadPool;

// Submit N independent tasks — blocks until all complete
void pool_dispatch_and_wait(ThreadPool* pool,
                            void (*fn)(void*, uint, uint),
                            void* args,
                            uint task_count);
```

The worker loop is simple — each thread atomically claims task indices:

```cpp
static int worker_main(void* arg) {
    WorkerContext* ctx = (WorkerContext*)arg;
    ThreadPool* pool = ctx->pool;
    uint thread_id = ctx->thread_id;

    while (!pool->shutdown) {
        mtx_lock(&pool->wake_mutex);
        while (!pool->current_batch && !pool->shutdown)
            cnd_wait(&pool->wake_cond, &pool->wake_mutex);
        mtx_unlock(&pool->wake_mutex);

        if (pool->shutdown) break;

        TaskBatch* batch = pool->current_batch;
        uint task;
        while ((task = atomic_fetch_add(&batch->next_task, 1))
               < batch->task_count)
        {
            batch->function(batch->args, task, thread_id);
        }

        if (atomic_fetch_add(&batch->tasks_completed, 1)
            == batch->task_count - 1)
        {
            cnd_signal(&pool->done_cond);  // Last thread signals completion
        }
    }
    return 0;
}
```

**Why work-stealing atomics instead of pre-partitioning?** Because kernel work isn't uniform. In `calculate_module_param_grads_chunk`, tiles with masked-out samples finish faster than valid ones. Atomic task claiming gives natural load balancing.

### DAG Execution on the Thread Pool

Each node in the DAG becomes a `pool_dispatch_and_wait` call. The "and_wait" provides the synchronization — when it returns, all tasks for that node are complete, fulfilling the DAG edge contract:

```cpp
void execute_learn_phase(PipelineContext* ctx, ThreadPool* pool) {

    // Phase I: All tiles are independent — full parallelism
    // Each task processes one flat_tile_index
    pool_dispatch_and_wait(pool, task_calculate_module_param_grads_chunk,
                           ctx, ctx->total_tile_count);      // Node 8
    pool_dispatch_and_wait(pool, task_backprop_error_to_hidden_chunk,
                           ctx, ctx->total_tile_count);      // Node 9
    pool_dispatch_and_wait(pool, task_calculate_chunk_temp_gradients,
                           ctx, ctx->total_tile_count);      // Node 10
    pool_dispatch_and_wait(pool, task_clip_partial_gradients,
                           ctx, ctx->total_tile_count);      // Node 11

    // Node 13: Item Synchronization Barrier
    // (dispatches across output elements, reads all tiles)
    uint grad_h_elements = ctx->total_batch_count * ctx->padded_hidden_count;
    pool_dispatch_and_wait(pool, task_gather_and_permute_grad_hidden_activations,
                           ctx, grad_h_elements);            // Node 13

    // Phase II: Reduction of module/temp grads
    execute_reduction_tree(pool, ctx, GRAD_MOD_WEIGHTS);     // Node 15
    execute_reduction_tree(pool, ctx, GRAD_MOD_BIASES);      // Node 15
    execute_reduction_tree(pool, ctx, GRAD_TEMPS);           // Node 15

    // Specialized Grad_H reduction — parallel across rows
    pool_dispatch_and_wait(pool, task_stabilize_and_reduce_grad_hidden_activations,
                           ctx, grad_h_elements);            // Node 16

    // Phase III: Streaming shared backprop — each chunk is a task
    pool_dispatch_and_wait(pool, task_backprop_shared_weights_chunk,
                           ctx, ctx->num_batch_chunks);      // Node 17
    pool_dispatch_and_wait(pool, task_backprop_shared_biases_chunk,
                           ctx, ctx->num_batch_chunks);      // Node 18
    pool_dispatch_and_wait(pool, task_clip_shared_gradients_chunk,
                           ctx, ctx->num_batch_chunks);      // Node 19

    // Phase IV: Final reduction & normalize
    execute_reduction_tree(pool, ctx, GRAD_SHARED_WEIGHTS);  // Node 20
    execute_reduction_tree(pool, ctx, GRAD_SHARED_BIASES);   // Node 20

    pool_dispatch_and_wait(pool, task_normalize_gradients,
                           ctx, ctx->parameter_count);       // Node 21

    // Phase V: Update — one task per parameter element
    pool_dispatch_and_wait(pool, task_adam_update,
                           ctx, ctx->parameter_count);       // Node 24

    pool_dispatch_and_wait(pool, task_clamp_temperatures,
                           ctx, ctx->total_modules_count);   // Node 25
}
```

Notice: `pool_dispatch_and_wait` is the universal synchronization primitive. It replaces OpenCL events, Vulkan pipeline barriers, and explicit thread joins. Every DAG edge is a return from this function.

---

## SIMD Strategy

### Why the Architecture Already Helps

The `_simd_major` weight layout is the critical enabler. Recall from the contract:

```
Tensor Shape: (padded_hidden_count/SIMD_WIDTH, padded_input_count, SIMD_WIDTH)
```

This means SIMD_WIDTH weights for _different hidden units but the same input_ are contiguous in memory. On CPU, this maps perfectly to a vectorized dot product:

```
Memory layout:     [h0_i0, h1_i0, h2_i0, ..., h7_i0, h0_i1, h1_i1, ...]
                    \___________ one SIMD load ____________/

AVX2 register:     [h0,    h1,    h2,    h3,    h4,    h5,    h6,    h7]
                    × broadcast(input[i0])
                    = [h0*x0, h1*x0, h2*x0, h3*x0, h4*x0, h5*x0, h6*x0, h7*x0]

                    Accumulate across all input dimensions →
                    8 output hidden activations simultaneously
```

### SIMD Abstraction Layer

Rather than scattering intrinsics throughout kernel code, define a thin abstraction that targets multiple ISAs:

```c
// cpu_simd.h — Compile-time SIMD dispatch

#ifndef CPU_SIMD_H
#define CPU_SIMD_H

#include <stdint.h>

// ============================================================
// Lane Width Selection (matches Article 6: SIMD_WIDTH)
// ============================================================

#if defined(__AVX512F__)
    #include <immintrin.h>
    #define SIMD_WIDTH       16
    #define SIMD_ALIGNMENT   64
    typedef __m512  simd_float;
    typedef __m512i simd_int;

    #define simd_zero()            _mm512_setzero_ps()
    #define simd_set1(x)           _mm512_set1_ps(x)
    #define simd_load(p)           _mm512_load_ps(p)
    #define simd_loadu(p)          _mm512_loadu_ps(p)
    #define simd_store(p, v)       _mm512_store_ps(p, v)
    #define simd_add(a, b)         _mm512_add_ps(a, b)
    #define simd_mul(a, b)         _mm512_mul_ps(a, b)
    #define simd_fmadd(a, b, c)    _mm512_fmadd_ps(a, b, c)
    #define simd_max(a, b)         _mm512_max_ps(a, b)
    #define simd_min(a, b)         _mm512_min_ps(a, b)
    #define simd_reduce_add(v)     _mm512_reduce_add_ps(v)

    static inline simd_float simd_select(simd_float a, simd_float b,
                                         simd_float mask) {
        __mmask16 k = _mm512_cmp_ps_mask(mask, _mm512_setzero_ps(),
                                          _CMP_GT_OQ);
        return _mm512_mask_blend_ps(k, a, b);
    }

#elif defined(__AVX2__)
    #include <immintrin.h>
    #define SIMD_WIDTH       8
    #define SIMD_ALIGNMENT   32
    typedef __m256  simd_float;
    typedef __m256i simd_int;

    #define simd_zero()            _mm256_setzero_ps()
    #define simd_set1(x)           _mm256_set1_ps(x)
    #define simd_load(p)           _mm256_load_ps(p)
    #define simd_loadu(p)          _mm256_loadu_ps(p)
    #define simd_store(p, v)       _mm256_store_ps(p, v)
    #define simd_add(a, b)         _mm256_add_ps(a, b)
    #define simd_mul(a, b)         _mm256_mul_ps(a, b)
    #define simd_fmadd(a, b, c)    _mm256_fmadd_ps(a, b, c)
    #define simd_max(a, b)         _mm256_max_ps(a, b)
    #define simd_min(a, b)         _mm256_min_ps(a, b)

    static inline float simd_reduce_add(simd_float v) {
        __m128 hi  = _mm256_extractf128_ps(v, 1);
        __m128 lo  = _mm256_castps256_ps128(v);
        __m128 sum = _mm_add_ps(lo, hi);
        sum = _mm_hadd_ps(sum, sum);
        sum = _mm_hadd_ps(sum, sum);
        return _mm_cvtss_f32(sum);
    }

    static inline simd_float simd_select(simd_float a, simd_float b,
                                         simd_float mask) {
        simd_float cmp = _mm256_cmp_ps(mask, _mm256_setzero_ps(),
                                        _CMP_GT_OQ);
        return _mm256_blendv_ps(a, b, cmp);
    }

#elif defined(__SSE2__)
    #include <xmmintrin.h>
    #include <emmintrin.h>
    #define SIMD_WIDTH       4
    #define SIMD_ALIGNMENT   16
    typedef __m128  simd_float;
    typedef __m128i simd_int;

    #define simd_zero()            _mm_setzero_ps()
    #define simd_set1(x)           _mm_set1_ps(x)
    #define simd_load(p)           _mm_load_ps(p)
    #define simd_store(p, v)       _mm_store_ps(p, v)
    #define simd_add(a, b)         _mm_add_ps(a, b)
    #define simd_mul(a, b)         _mm_mul_ps(a, b)
    #define simd_max(a, b)         _mm_max_ps(a, b)
    #define simd_min(a, b)         _mm_min_ps(a, b)

    #define simd_fmadd(a, b, c)    simd_add(simd_mul(a, b), c)

    static inline float simd_reduce_add(simd_float v) {
        __m128 shuf = _mm_movehdup_ps(v);
        __m128 sums = _mm_add_ps(v, shuf);
        shuf = _mm_movehl_ps(shuf, sums);
        sums = _mm_add_ss(sums, shuf);
        return _mm_cvtss_f32(sums);
    }

#elif defined(__ARM_NEON)
    #include <arm_neon.h>
    #define SIMD_WIDTH       4
    #define SIMD_ALIGNMENT   16
    typedef float32x4_t simd_float;

    #define simd_zero()            vdupq_n_f32(0.0f)
    #define simd_set1(x)           vdupq_n_f32(x)
    #define simd_load(p)           vld1q_f32(p)
    #define simd_store(p, v)       vst1q_f32(p, v)
    #define simd_add(a, b)         vaddq_f32(a, b)
    #define simd_mul(a, b)         vmulq_f32(a, b)
    #define simd_fmadd(a, b, c)    vfmaq_f32(c, a, b)
    #define simd_max(a, b)         vmaxq_f32(a, b)
    #define simd_min(a, b)         vminq_f32(a, b)

    static inline float simd_reduce_add(simd_float v) {
        return vaddvq_f32(v);
    }

#else
    // Scalar fallback
    #define SIMD_WIDTH       1
    #define SIMD_ALIGNMENT   4
    typedef float simd_float;

    #define simd_zero()            0.0f
    #define simd_set1(x)           (x)
    #define simd_load(p)           (*(p))
    #define simd_store(p, v)       (*(p) = (v))
    #define simd_add(a, b)         ((a) + (b))
    #define simd_mul(a, b)         ((a) * (b))
    #define simd_fmadd(a, b, c)    ((a) * (b) + (c))
    #define simd_max(a, b)         ((a) > (b) ? (a) : (b))
    #define simd_min(a, b)         ((a) < (b) ? (a) : (b))
    #define simd_reduce_add(v)     (v)
#endif

// ============================================================
// Aligned allocation (all buffers must use this)
// ============================================================
#include <stdlib.h>

static inline void* simd_alloc(size_t bytes) {
#if defined(_MSC_VER)
    return _aligned_malloc(bytes, SIMD_ALIGNMENT);
#else
    void* ptr = NULL;
    posix_memalign(&ptr, SIMD_ALIGNMENT, bytes);
    return ptr;
#endif
}

static inline void simd_free(void* ptr) {
#if defined(_MSC_VER)
    _aligned_free(ptr);
#else
    free(ptr);
#endif
}

#endif // CPU_SIMD_H
```

### The Hot Kernel: `forward_pass` with SIMD + Threading

This is the most performance-critical kernel — it's a matrix multiply. Here's how it looks with both SIMD and multi-core:

```c
// cpu_kernels/forward_pass.c
#include "cpu_simd.h"
#include "kernels_interface_cpu.h"

// This function is called by the thread pool with a task_index
// that represents which sample to process.
void task_forward_pass(void* raw_args, uint task_index, uint thread_id) {
    ForwardPassArgs* args = (ForwardPassArgs*)raw_args;
    (void)thread_id;

    const uint sample = args->batch_chunk_offset + task_index;
    if (task_index >= args->batch_chunk_count) return;

    const float  mask_val       = args->sample_mask[sample];
    const float* input_row      = args->input + sample * args->padded_input_count;
    float*       hidden_row     = args->hidden_activations
                                + sample * args->padded_hidden_count;
    float*       hidden_msk_row = args->hidden_mask
                                + sample * args->padded_hidden_count;

    const uint padded_input  = args->padded_input_count;
    const uint padded_hidden = args->padded_hidden_count;

    // Weights are in SIMD-major layout:
    //   weights[h_block][input_dim][simd_lane]
    // where h_block = padded_hidden / SIMD_WIDTH
    //
    // For each block of SIMD_WIDTH hidden units, we:
    //   1. Load bias vector (SIMD_WIDTH biases)
    //   2. For each input dimension, broadcast input[i] and FMA with weight row
    //   3. Apply ReLU + sample mask
    //   4. Store SIMD_WIDTH results

    const simd_float v_zero = simd_zero();
    const simd_float v_mask = simd_set1(mask_val);

    const uint num_h_blocks = padded_hidden / SIMD_WIDTH;

    for (uint hb = 0; hb < num_h_blocks; hb++) {
        const uint h_offset = hb * SIMD_WIDTH;

        // Load biases for this block
        simd_float accum = simd_load(&args->biases_shared[h_offset]);

        // Weight pointer for this h_block:
        // weights_simd_major[hb * padded_input * SIMD_WIDTH + i * SIMD_WIDTH]
        const float* w_block = args->weights_shared_simd_major
                              + hb * padded_input * SIMD_WIDTH;

        // Dot product across all input dimensions
        for (uint i = 0; i < padded_input; i++) {
            simd_float x_broadcast = simd_set1(input_row[i]);
            simd_float w_vec       = simd_load(&w_block[i * SIMD_WIDTH]);
            accum = simd_fmadd(w_vec, x_broadcast, accum);
        }

        // ReLU: max(0, accum)
        simd_float activated = simd_max(v_zero, accum);

        // Apply sample mask
        activated = simd_mul(activated, v_mask);

        // Store hidden activations
        simd_store(&hidden_row[h_offset], activated);

        // Store hidden mask: 1.0 where pre-ReLU > 0 AND sample is valid
        // hidden_mask[h] = (accum[h] > 0) ? mask_val : 0
        simd_float relu_mask = simd_select(v_zero, v_mask, accum);
        simd_store(&hidden_msk_row[h_offset], relu_mask);
    }
}

// Dispatch entry point
void dispatch_forward_pass(ThreadPool* pool, ForwardPassArgs* args) {
    pool_dispatch_and_wait(pool, task_forward_pass, args,
                           args->batch_chunk_count);
}
```

**Performance characteristics:**

- Each task processes one sample — independent, no synchronization
- Inner loop is fully vectorized: one SIMD FMA per input dimension per hidden block
- Memory access is sequential for both input (broadcast) and weights (stride-1 SIMD load)
- Zero branching in the hot loop (select replaces conditional)
- Padding eliminates scalar cleanup

### The Compute-Bound Kernels vs Memory-Bound Kernels

Not all kernels benefit equally from SIMD. The architecture's kernels fall into two categories:

| Kernel                                                    | Bottleneck              | SIMD Value   | Threading Value |
| --------------------------------------------------------- | ----------------------- | ------------ | --------------- |
| `forward_pass` (Node 4)                                   | Compute (matmul)        | **Critical** | **Critical**    |
| `render_logits_chunk` (Node 5)                            | Compute (matmul)        | **Critical** | **Critical**    |
| `compute_probs_loss_cce_chunk` (Node 6)                   | Mixed (exp, log)        | High         | High            |
| `compute_probs_loss_bce_chunk` (Node 7)                   | Mixed (sigmoid, log)    | High         | High            |
| `calculate_module_param_grads_chunk` (Node 8)             | Compute (outer product) | **Critical** | **Critical**    |
| `backprop_error_to_hidden_chunk` (Node 9)                 | Compute (matmul)        | **Critical** | **Critical**    |
| `calculate_chunk_temp_gradients` (Node 10)                | Mixed                   | Medium       | High            |
| `clip_partial_gradients` (Node 11)                        | Memory (norm scan)      | Medium       | High            |
| `gather_and_permute_grad_hidden_activations` (Node 13)    | Memory (scatter/gather) | Low-Medium   | **Critical**    |
| `aggregate_register_reduce` / `aggregate_local_reduce` (Node 14/15a/20a) | Memory (streaming sum)  | Medium       | Medium          |
| `clip_intermediate_grad` (Node 15b/20b)                   | Memory (norm + scale)   | Medium       | Low-Medium      |
| `stabilize_and_reduce_grad_hidden_activations` (Node 16)  | Mixed                   | High         | High            |
| `backprop_shared_weights_chunk` (Node 17)                 | Compute (outer product) | **Critical** | **Critical**    |
| `backprop_shared_biases_chunk` (Node 18)                  | Memory (reduction)      | Medium       | High            |
| `clip_shared_gradients_chunk` (Node 19)                   | Memory (norm + scale)   | Medium       | High            |
| `normalize_gradients` (Node 21)                           | Memory (element-wise)   | Low          | High            |
| `adam_update` (Node 24)                                   | Memory (element-wise)   | Medium       | High            |
| `clamp_temperatures` (Node 25)                            | Memory (element-wise)   | Low          | High            |

The matmul kernels (4, 5, 8, 9, 17) dominate runtime. These are where SIMD investment pays off most.

---

## The Reduction Engine on Multi-Core CPU

This is the most architecturally interesting problem. The GPU reduction engine uses scattered partials + indirection lists. On CPU, we need to preserve the staged clipping policy while exploiting parallelism.

### The Key Tension

The clipping policy requires sequential stages — you can't clip stage `j` until stage `j+1`'s sums are complete. But _within_ a stage, all reductions are independent.

### Solution: Parallel Stages with Barrier Between

```c
// cpu_reduction_engine.c
#include "cpu_simd.h"

typedef struct {
    const float*  partial_collection;    // All partials in one buffer
    const uint*   offset_lists;          // Flattened offset lists per stage
    const uint*   stage_offset_counts;   // How many offsets per reduction node
    const uint*   stage_node_counts;     // How many nodes at each stage
    float*        intermediate_buffers;  // Double-buffered stage results
    float*        output;                // Final reduced result

    uint   partial_width;      // Elements per partial
    uint   num_stages;         // Total tree depth
    float  t_algorithmic;      // Policy anchor
    float  lambda;             // Policy curvature
    float  fp_max;             // Hardware safety ceiling
    float  epsilon;            // Numerical stability
} ReductionTreePlan;

// --- Per-node task: sum K partials into one output ---
typedef struct {
    ReductionTreePlan* plan;
    uint               stage;
    const float*       src_buffer;
    float*             dst_buffer;
} ReductionStageArgs;

void task_reduce_one_node(void* raw_args, uint node_index, uint thread_id) {
    ReductionStageArgs* args = (ReductionStageArgs*)raw_args;
    ReductionTreePlan*  plan = args->plan;
    (void)thread_id;

    const uint width = plan->partial_width;
    const uint K = plan->stage_offset_counts[args->stage];

    // Compute where this node's offset list starts
    // (host pre-computed as a flat array of offsets)
    const uint* offsets = /* ... locate via node_index ... */;

    float* dest = args->dst_buffer + node_index * width;

    // Initialize from first partial
    const float* first = args->src_buffer + offsets[0];
    for (uint e = 0; e < width; e += SIMD_WIDTH) {
        simd_store(&dest[e], simd_load(&first[e]));
    }

    // Accumulate remaining partials
    for (uint k = 1; k < K; k++) {
        const float* src = args->src_buffer + offsets[k];
        for (uint e = 0; e < width; e += SIMD_WIDTH) {
            simd_float a = simd_load(&dest[e]);
            simd_float b = simd_load(&src[e]);
            simd_store(&dest[e], simd_add(a, b));
        }
    }
}

// --- Per-node task: clip one reduced buffer ---
void task_clip_one_node(void* raw_args, uint node_index, uint thread_id) {
    ReductionStageArgs* args = (ReductionStageArgs*)raw_args;
    ReductionTreePlan*  plan = args->plan;
    (void)thread_id;

    const uint width = plan->partial_width;
    float* buf = args->dst_buffer + node_index * width;

    // Compute L2 norm (vectorized)
    simd_float sum_sq = simd_zero();
    for (uint e = 0; e < width; e += SIMD_WIDTH) {
        simd_float v = simd_load(&buf[e]);
        sum_sq = simd_fmadd(v, v, sum_sq);
    }
    float norm = sqrtf(simd_reduce_add(sum_sq));

    // Policy threshold for this stage
    uint j = plan->num_stages - 1 - args->stage;
    float t_policy = plan->t_algorithmic + plan->lambda * (float)(j * j);
    float K_actual = (float)plan->stage_offset_counts[args->stage];
    float t_safety = plan->fp_max / K_actual;
    float threshold = fminf(t_policy, t_safety);

    // Conditional scale
    if (norm > threshold) {
        float scale = threshold / (norm + plan->epsilon);
        simd_float v_scale = simd_set1(scale);
        for (uint e = 0; e < width; e += SIMD_WIDTH) {
            simd_store(&buf[e], simd_mul(simd_load(&buf[e]), v_scale));
        }
    }
}

// --- Execute the full tree ---
void execute_reduction_tree(ThreadPool* pool, ReductionTreePlan* plan) {
    ReductionStageArgs stage_args;
    stage_args.plan = plan;

    // First stage reads from partial_collection
    stage_args.src_buffer = plan->partial_collection;
    stage_args.dst_buffer = plan->intermediate_buffers;

    for (uint s = 0; s < plan->num_stages; s++) {
        stage_args.stage = s;
        uint num_nodes = plan->stage_node_counts[s];

        // Sum phase — all nodes at this stage are independent
        pool_dispatch_and_wait(pool, task_reduce_one_node,
                               &stage_args, num_nodes);

        // Clip phase — all nodes at this stage are independent
        pool_dispatch_and_wait(pool, task_clip_one_node,
                               &stage_args, num_nodes);

        // Swap buffers for next stage
        // (current dst becomes next src)
        stage_args.src_buffer = stage_args.dst_buffer;
        stage_args.dst_buffer = /* alternate buffer */;
    }

    // Final result is in src_buffer after last swap
    memcpy(plan->output, stage_args.src_buffer,
           plan->partial_width * sizeof(float));
}
```

The parallelism structure per stage:

```
Stage 2 (leaves):   [node0][node1][node2][node3]  ← 4 tasks, parallel
                            barrier
                     [clip0][clip1][clip2][clip3]  ← 4 tasks, parallel
                            barrier
Stage 1 (mid):      [node0][node1]                ← 2 tasks, parallel
                            barrier
                     [clip0][clip1]                ← 2 tasks, parallel
                            barrier
Stage 0 (root):     [node0]                       ← 1 task
                            barrier
                     [clip0]                       ← 1 task
```

For large numbers of partials, the leaf stages have high parallelism. The root is sequential, but it processes one buffer — the cost is trivial.

### Precision-Typed Variants (ADR-026)

Under mixed-precision configurations (`PrecisionConfig.mixed_f16_f32()`), the reduction engine must handle two entry point types differentiated by the source buffer's `precision_role`:

| Variant | Source Buffer Role | Load Mechanism | When Used |
| :--- | :--- | :--- | :--- |
| **Storage-entry** (existing) | `"storage"` | Widening conversion (`STORAGE_TYPE` → `COMPUTE_TYPE`) | Leaf stage reading from STORAGE_TYPE partial collection |
| **Compute-entry** (ADR-026) | `"compute"` | Direct `COMPUTE_TYPE` read | Interior stages; leaf stage reading from COMPUTE_TYPE collection (e.g., BCE loss) |

**Variant selection rule:** The Orchestration tier queries the source buffer's `BufferDescriptor.precision_role`:

```python
# Multi-stage tree
for stage in range(num_stages):
    if stage == 0 and source_desc.precision_role == "storage":
        dispatch execute_reduction_tree_<suffix>           # storage-entry
    else:
        dispatch execute_reduction_tree_from_compute_<suffix>  # compute-entry
```

The C library exposes both entry points per precision suffix (`s32x32`, `s16x16`, `s16x32`):

| Entry Point | Parameter Type |
| :--- | :--- |
| `execute_reduction_tree_<suffix>` | `STORAGE_TYPE* partial_collection` |
| `execute_reduction_tree_from_compute_<suffix>` | `COMPUTE_TYPE* partial_collection` |

Both share identical algorithm, clipping logic, and output type (`COMPUTE_TYPE`). The divergence is exactly one parameter's C type.

---

## FP8 Support (Phase 9C, ADR-025)

The CPU backend supports FP8 E4M3 (`float8_e4m3fn`) and FP8 E5M2 (`float8_e5m2`) as storage-only types. FP8 values never participate directly in arithmetic — they are widened to compute precision on load and narrowed back on store.

### FP8 Type Representation

FP8 values are wrapped in C structs to prevent accidental arithmetic on raw bits:

```c
// cpu_fp8.h
typedef struct { uint8_t bits; } cpu_fp8_e4m3;  // 1+4+3, bias=7, max=448.0
typedef struct { uint8_t bits; } cpu_fp8_e5m2;  // 1+5+2, bias=15, max=57344.0
```

**E4M3** (`float8_e4m3fn`): No infinity. Two NaN patterns (0x7F, 0xFF). Overflow saturates to ±448.0.
**E5M2** (`float8_e5m2`): IEEE-like infinity/NaN at exponent 0x1F. Overflow saturates to ±57344.0 (max finite).

### Conversion Strategy

- **FP8 → float (decode):** LUT-based. Two 256-entry lookup tables (`cpu_fp8_e4m3_to_float_lut`, `cpu_fp8_e5m2_to_float_lut`) generated at build time by Phase 9A. Single table lookup per value.
- **float → FP8 (encode):** Algorithmic with round-to-nearest-even (RTE). Handles NaN→zero, overflow→saturation, subnormal rounding. Approximately 15 integer operations per conversion.
- **double → FP8:** Clamps to float range first, then delegates to the float→FP8 encoder.
- **_Float16 → FP8 / FP8 → _Float16:** Conditionally compiled when `HAS_FLOAT16=1` (detected by Meson at configure time).

### SIMD Dispatch

No CPU architecture has native FP8 SIMD instructions. The SIMD load/store wrappers (`simd_load_real_fp8e4m3`, `simd_store_real_fp8e4m3`, etc.) perform element-wise scalar conversion in a loop:

```c
// Scalar loop through SIMD_WIDTH elements:
//   load:  fp8[i] → LUT → float → simd_lane[i]
//   store: simd_lane[i] → encode → fp8[i]
```

This is the correct baseline — AVX-512 has no FP8 instructions, so scalar-loop conversion is architecturally required. Future ISAs with native FP8 support would add new branches to the `#ifdef` chain in `cpu_fp8.h`.

### FP8 Precision Suffix Variants

FP8 introduces 18 new three-axis suffix variants (ADR-025 §5.3):

| Suffix | Storage | Compute | State | Condition |
| :--- | :--- | :--- | :--- | :--- |
| `s8e4c32x32` | E4M3 | FP32 | FP32 | Always |
| `s8e4c32x64` | E4M3 | FP32 | FP64 | Always |
| `s8e4c64x32` | E4M3 | FP64 | FP32 | Always |
| `s8e4c64x64` | E4M3 | FP64 | FP64 | Always |
| `s8e5c32x32` | E5M2 | FP32 | FP32 | Always |
| `s8e5c32x64` | E5M2 | FP32 | FP64 | Always |
| `s8e5c64x32` | E5M2 | FP64 | FP32 | Always |
| `s8e5c64x64` | E5M2 | FP64 | FP64 | Always |
| `s8e4c16x16` | E4M3 | FP16 | FP16 | `_Float16` available |
| `s8e4c16x32` | E4M3 | FP16 | FP32 | `_Float16` available |
| `s8e4c16x64` | E4M3 | FP16 | FP64 | `_Float16` available |
| `s8e4c32x16` | E4M3 | FP32 | FP16 | `_Float16` available |
| `s8e4c64x16` | E4M3 | FP64 | FP16 | `_Float16` available |
| `s8e5c16x16` | E5M2 | FP16 | FP16 | `_Float16` available |
| `s8e5c16x32` | E5M2 | FP16 | FP32 | `_Float16` available |
| `s8e5c16x64` | E5M2 | FP16 | FP64 | `_Float16` available |
| `s8e5c32x16` | E5M2 | FP32 | FP16 | `_Float16` available |
| `s8e5c64x16` | E5M2 | FP64 | FP16 | `_Float16` available |

The 8 FP32/FP64-only compute/state variants are always compiled. The 10 FP16 compute or state variants are conditional on `_Float16` support (C23 / GCC 12+ / Clang 15+ with `-std=c2x`), detected at Meson configure time via `cc.compiles()`.

### `_Float16` Availability

`_Float16` detection is the single source of truth for FP16 compute availability:

1. **Meson detects** `_Float16` support via `cc.compiles('_Float16 x = 1.0f16; x = x * 2.0f16;', args: ['-std=c2x'])`
2. **C flag:** `-DHAS_FLOAT16=1` or `-DHAS_FLOAT16=0` passed to all kernel source compilation
3. **Generated Python:** `_fp8_variants.py` (from `_fp8_variants.py.in`) exposes `HAS_FLOAT16: bool` and `AVAILABLE_FP8_VARIANTS: list[str]` to the Python FFI layer
4. **Runtime:** `_ffi_types.py` imports `_fp8_variants` to conditionally include the FP16 compute/state suffixes in `ALL_PRECISION_SUFFIXES`

**Cross-backend note:** A `PrecisionConfig` like `fp8_e4m3_f16()` (E4M3 storage, FP16 compute) may work on OpenCL/Vulkan but fail on CPU if the C compiler lacks `_Float16` support. The CPU backend gracefully omits these variants rather than failing at build time.

---

## Cache Considerations

### False Sharing Prevention

When multiple threads write to adjacent memory, cache line bouncing destroys performance. The architecture's Placement Contract protects against this — each thread writes to a separate region — but the _size_ of that region matters:

```c
// BAD: If partial_width < cache line size (64 bytes / 4 = 16 floats),
// adjacent partials share a cache line
//
// partial[0]: [thread 0 writes here.......][thread 1 writes here]
//              \______ same cache line ______/
//               → FALSE SHARING

// SOLUTION: Pad each partial's width to a cache line boundary
#define CACHE_LINE_BYTES  64
#define CACHE_LINE_FLOATS (CACHE_LINE_BYTES / sizeof(float))

static inline uint pad_to_cache_line(uint element_count) {
    return ((element_count + CACHE_LINE_FLOATS - 1)
            / CACHE_LINE_FLOATS) * CACHE_LINE_FLOATS;
}
```

This aligns with the architecture's existing `{Type: CACHE}` padding contract — the host orchestrator already does this. On CPU, it's just more critical because the penalty is immediate.

### Per-Thread Scratch Memory

On the GPU, `__local` memory is per-work-group. On CPU, the equivalent is per-thread stack allocation. Pre-allocate per-thread scratch buffers to avoid contention on `malloc`:

```c
typedef struct {
    float* scratch;          // General-purpose aligned scratch
    uint   scratch_capacity; // In elements
} ThreadLocalStorage;

typedef struct {
    ThreadPool*        pool;
    ThreadLocalStorage tls[];  // One per thread, indexed by thread_id
} CPUBackend;

CPUBackend* cpu_backend_create(uint num_threads, uint scratch_elements) {
    CPUBackend* be = malloc(sizeof(CPUBackend)
                          + num_threads * sizeof(ThreadLocalStorage));
    be->pool = pool_create(num_threads);
    for (uint t = 0; t < num_threads; t++) {
        be->tls[t].scratch = simd_alloc(scratch_elements * sizeof(float));
        be->tls[t].scratch_capacity = scratch_elements;
    }
    return be;
}
```

The `thread_id` parameter in every task function enables this:

```c
void task_some_kernel(void* args, uint task_index, uint thread_id) {
    CPUBackend* be = ((TaskArgs*)args)->backend;
    float* my_scratch = be->tls[thread_id].scratch;
    // Use my_scratch freely — no other thread touches it
}
```

---

## The CPU Interface Header

Given the performance requirements, the CPU header is more than just function signatures. It includes the SIMD abstraction and threading types:

```c
// kernels_interface_cpu.h
#ifndef KERNELS_INTERFACE_CPU_H
#define KERNELS_INTERFACE_CPU_H

#include "cpu_simd.h"      // SIMD abstraction (defines SIMD_WIDTH)
#include "cpu_threads.h"   // ThreadPool, pool_dispatch_and_wait

#include <stdint.h>
#include <string.h>
#include <math.h>

// ============================================================
// Build Configuration
// ============================================================
#define LOCAL_MEM_BANK_PADDING     1  // Retained for layout compatibility
#define NUMERICAL_STABILITY_EPSILON 1e-7f
#define CACHE_LINE_BYTES           64

#ifndef C_TILE_SIZE
#define C_TILE_SIZE SIMD_WIDTH
#endif

#define PROBLEM_TYPE_CCE  0
#define PROBLEM_TYPE_BCE  1
#define AGG_MODE_SUM      0
#define AGG_MODE_AVERAGE  1

typedef uint32_t uint;

// ============================================================
// Argument Structures (one per kernel)
// ============================================================
// These replace OpenCL's flat arg lists and Vulkan's push constants.
// Each struct is passed as the void* to the thread pool task function.

typedef struct {
    // Buffers
    const float* input;
    const float* sample_mask;
    const float* weights_shared_simd_major;
    const float* biases_shared;
    float*       hidden_activations;
    float*       hidden_mask;
    // Scalars
    uint batch_chunk_offset;
    uint batch_chunk_count;
    uint total_batch_count;
    uint padded_input_count;
    uint padded_hidden_count;
} ForwardPassArgs;

typedef struct {
    const float* logits;
    const float* temps;
    const int*   targets;
    const float* sample_mask;
    float*       partial_probs;
    float*       final_loss;
    uint flat_tile_index_base;  // task_index is added to this
    uint num_class_chunks;
    uint classes_per_chunk;
    uint modules_per_chunk;
    uint total_batch_count;
    uint total_output_class_count;
    uint padded_total_output_class_count;
    uint total_modules_count;
    uint total_tile_count;
} CceChunkArgs;

typedef struct {
    // Sources
    const float* partial_grad_weights_module;
    const float* partial_grad_biases_module;
    const float* partial_grad_temps;
    const float* partial_grad_hidden_aos;
    const float* threshold_per_item;  // NULL if global threshold
    // Destinations
    float* clipped_grad_weights_module;
    float* clipped_grad_biases_module;
    float* clipped_grad_temps;
    float* clipped_grad_hidden_aos;
    // Config
    uint  use_per_item_norm;
    float clipping_threshold;
    float epsilon;
    uint  num_class_chunks;
    uint  classes_per_chunk;
    uint  modules_per_chunk;
    uint  total_batch_count;
    uint  padded_hidden_count;
    uint  total_tile_count;
} ClipPartialsArgs;

typedef struct {
    const float* final_grad;
    float*       parameters;
    float*       m1;
    float*       m2;
    float learning_rate;
    float beta1_pow_t;
    float beta2_pow_t;
    float beta1;
    float beta2;
    float epsilon;
    uint  parameter_count;
} AdamUpdateArgs;

// ... (remaining kernels follow the same pattern)

// ============================================================
// Task Functions (called by thread pool)
// ============================================================
// Each takes (void* args, uint task_index, uint thread_id)
//
// Naming Convention: task_<kernel_name> where <kernel_name> is the
// exact kernel name from kernels.cl.h. Each task function implements
// the CPU-parallelized equivalent of the corresponding OpenCL kernel.

void task_forward_pass(void* args, uint task_index, uint thread_id);
void task_render_logits_chunk(void* args, uint task_index, uint thread_id);
void task_compute_probs_loss_cce_chunk(void* args, uint task_index, uint thread_id);
void task_compute_probs_loss_bce_chunk(void* args, uint task_index, uint thread_id);
void task_calculate_module_param_grads_chunk(void* args, uint task_index, uint thread_id);
void task_backprop_error_to_hidden_chunk(void* args, uint task_index, uint thread_id);
void task_calculate_chunk_temp_gradients(void* args, uint task_index, uint thread_id);
void task_clip_partial_gradients(void* args, uint task_index, uint thread_id);
void task_gather_and_permute_grad_hidden_activations(void* args, uint task_index, uint thread_id);
void task_stabilize_and_reduce_grad_hidden_activations(void* args, uint task_index, uint thread_id);
void task_backprop_shared_weights_chunk(void* args, uint task_index, uint thread_id);
void task_backprop_shared_biases_chunk(void* args, uint task_index, uint thread_id);
void task_clip_shared_gradients_chunk(void* args, uint task_index, uint thread_id);
void task_normalize_gradients(void* args, uint task_index, uint thread_id);
void task_adam_update(void* args, uint task_index, uint thread_id);
void task_clamp_temperatures(void* args, uint task_index, uint thread_id);

// ============================================================
// Reduction Engine
// ============================================================

typedef struct {
    const float* partial_collection;
    const uint*  offset_lists_flat;
    const uint*  stage_offsets_into_list;
    const uint*  stage_fan_in;
    const uint*  stage_node_counts;
    float*       staging_buffers[2];
    float*       output;
    uint   partial_width;
    uint   num_stages;
    float  t_algorithmic;
    float  lambda;
    float  fp_max;
    float  epsilon;
} ReductionTreePlan;

void execute_reduction_tree(ThreadPool* pool, ReductionTreePlan* plan);

#endif // KERNELS_INTERFACE_CPU_H
```

---

## CPU-Specific Naming Conventions

The CPU backend uses a formal naming convention that maintains direct correspondence with the kernel names defined in `kernels.cl.h`, while also introducing backend-specific constructs.

### Task Function Naming

Task functions follow the pattern `task_<kernel_name>` where `<kernel_name>` is the exact kernel name from `kernels.cl.h`. This naming convention ensures:

1. **Traceability**: Each task function can be trivially mapped to its corresponding OpenCL kernel
2. **Consistency**: Code review and debugging benefit from identical naming across backends
3. **Documentation alignment**: References to kernel contracts in `kernels.cl.h` apply directly to their CPU task counterparts

| OpenCL Kernel (`kernels.cl.h`)                  | CPU Task Function                                             |
| ----------------------------------------------- | ------------------------------------------------------------- |
| `forward_pass`                                  | `task_forward_pass`                                           |
| `render_logits_chunk`                           | `task_render_logits_chunk`                                    |
| `compute_probs_loss_cce_chunk`                  | `task_compute_probs_loss_cce_chunk`                           |
| `compute_probs_loss_bce_chunk`                  | `task_compute_probs_loss_bce_chunk`                           |
| `calculate_module_param_grads_chunk`            | `task_calculate_module_param_grads_chunk`                     |
| `backprop_error_to_hidden_chunk`                | `task_backprop_error_to_hidden_chunk`                         |
| `calculate_chunk_temp_gradients`                | `task_calculate_chunk_temp_gradients`                         |
| `clip_partial_gradients`                        | `task_clip_partial_gradients`                                 |
| `gather_and_permute_grad_hidden_activations`    | `task_gather_and_permute_grad_hidden_activations`             |
| `stabilize_and_reduce_grad_hidden_activations`  | `task_stabilize_and_reduce_grad_hidden_activations`           |
| `backprop_shared_weights_chunk`                 | `task_backprop_shared_weights_chunk`                          |
| `backprop_shared_biases_chunk`                  | `task_backprop_shared_biases_chunk`                           |
| `clip_shared_gradients_chunk`                   | `task_clip_shared_gradients_chunk`                            |
| `normalize_gradients`                           | `task_normalize_gradients`                                    |
| `adam_update`                                   | `task_adam_update`                                            |
| `clamp_temperatures`                            | `task_clamp_temperatures`                                     |

### CPU-Specific Primitives

The following are backend-specific primitives without direct OpenCL kernel counterparts:

| CPU-Specific Name           | Purpose                                                        | Replaces                                           |
| --------------------------- | -------------------------------------------------------------- | -------------------------------------------------- |
| `execute_reduction_tree`    | Multi-stage reduction with staged clipping (tree execution)   | `aggregate_*` + `clip_intermediate_grad` sequences |
| `task_reduce_one_node`      | Per-node partial summation within a reduction stage            | Internal to `aggregate_register_reduce` / `aggregate_local_reduce` |
| `task_clip_one_node`        | Per-node L2 clip within a reduction stage                      | Internal to `clip_intermediate_grad`               |
| `pool_dispatch_and_wait`    | Thread pool dispatch primitive (synchronization barrier)       | `clEnqueueNDRangeKernel` + event wait              |
| `ThreadPool` / `TaskBatch`  | Thread pool management structures                              | OpenCL command queue                               |

**Design rationale**: The reduction engine kernels (`aggregate_register_reduce`, `aggregate_local_reduce`, `reduce_k_fan_in_and_clip`, etc.) operate on individual nodes in a reduction tree. On the CPU backend, the tree traversal is explicit via `execute_reduction_tree`, which dispatches `task_reduce_one_node` and `task_clip_one_node` across nodes at each stage. This refactoring exploits CPU threading semantics (barriers via function return) rather than emulating GPU work-group synchronization.

---

## Architecture Summary: Three Back-Ends Compared

```
              ┌──────────────────────────────────┐
              │    Host Orchestrator (shared)     │
              │  DAG planning, reduction trees,   │
              │  clipping policy, Adam bias calc  │
              └──────────┬───────────────────────┘
                         │
          ┌──────────────┼──────────────────┐
          │              │                  │
    ┌─────┴─────┐  ┌────┴──────┐  ┌───────┴────────┐
    │  OpenCL   │  │  Vulkan   │  │      CPU       │
    │  Backend  │  │  Backend  │  │    Backend     │
    ├───────────┤  ├───────────┤  ├────────────────┤
    │ dispatch: │  │ dispatch: │  │ dispatch:      │
    │ clEnqueue │  │ vkCmd     │  │ pool_dispatch  │
    │ NDRange   │  │ Dispatch  │  │ _and_wait      │
    ├───────────┤  ├───────────┤  ├────────────────┤
    │ sync:     │  │ sync:     │  │ sync:          │
    │ cl_event  │  │ VkBarrier │  │ function       │
    │           │  │ VkFence   │  │ return         │
    ├───────────┤  ├───────────┤  ├────────────────┤
    │ memory:   │  │ memory:   │  │ memory:        │
    │ clCreate  │  │ VMA /     │  │ simd_alloc     │
    │ Buffer    │  │ vkAlloc   │  │ (aligned)      │
    ├───────────┤  ├───────────┤  ├────────────────┤
    │ SIMD:     │  │ SIMD:     │  │ SIMD:          │
    │ wavefront │  │ subgroup  │  │ AVX/SSE/NEON   │
    │ (implicit)│  │ ops       │  │ (explicit)     │
    ├───────────┤  ├───────────┤  ├────────────────┤
    │ local mem:│  │ local mem:│  │ local mem:     │
    │ __local   │  │ shared    │  │ stack / TLS    │
    │ (host     │  │ (shader   │  │ (pre-alloc)    │
    │  alloc)   │  │  decl)    │  │                │
    ├───────────┤  ├───────────┤  ├────────────────┤
    │ kernels:  │  │ shaders:  │  │ tasks:         │
    │ kernels.cl│  │ *.comp    │  │ cpu_kernels/*.c│
    └───────────┘  └───────────┘  └────────────────┘
```

| Concern               | OpenCL                     | Vulkan                                            | CPU                                                      |
| --------------------- | -------------------------- | ------------------------------------------------- | -------------------------------------------------------- |
| Interface contract    | `kernels.cl.h` (flat args) | `kernels_interface.h` (bindings + push constants) | `kernels_interface_cpu.h` (arg structs + task functions) |
| Parallelism unit      | Work-item                  | Invocation                                        | Loop iteration / SIMD lane                               |
| Parallelism grouping  | Work-group                 | Work-group                                        | Thread pool task                                         |
| Barrier cost          | Moderate (HW-managed)      | Low (explicit, minimal)                           | Near-zero (function return)                              |
| SIMD control          | Implicit (hardware)        | Semi-explicit (subgroups)                         | Fully explicit (intrinsics)                              |
| Weight layout benefit | Coalesced memory access    | Coalesced memory access                           | Aligned SIMD loads                                       |
| Padding benefit       | Wavefront alignment        | Subgroup alignment                                | SIMD register fill, cache line alignment                 |

The `_simd_major` layout and the SIMD-aligned padding contracts serve all three back-ends. The architecture's memory strategy wasn't GPU-specific — it was _parallel hardware_-specific, and a modern CPU is exactly that.
