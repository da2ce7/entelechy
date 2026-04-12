# Vulkan Back-End: Architecture for Explicit GPU Compute

> **Implementation Status: ✅ Complete** — Phase 5 implemented; Phase 7 mixed-precision migration complete; Phase 9D FP8 Vulkan backend complete; Phase 10 state-precision accumulation complete. All components described in this document have been realized in `src/backends/vulkan/` with a Vulkan renderer, GLSL compute shaders compiled to SPIR-V (three precision variants per storage/state-role-bearing shader: `_fp32`, `_s16fp32`, `_fp16`, plus eight FP8 variants: `_s8e4c32x32`, `_s8e4c16x32`, `_s8e4c32x64`, `_s8e4c64x64`, `_s8e5c32x32`, `_s8e5c16x32`, `_s8e5c32x64`, `_s8e5c64x64`), and Tier 2/3 test coverage. The Meson build target produces the Vulkan shared library. See [PHASE-5A](plan/PHASE-5A-VULKAN-SHADER-LIBRARY-AND-BUILD.md), [PHASE-5B](plan/PHASE-5B-VULKAN-RENDERER-AND-INFRASTRUCTURE.md), [PHASE-5C](plan/PHASE-5C-VULKAN-TIER2-TESTS.md) for base implementation, [PHASE-7C](plan/PHASE-7C-MIXED-PRECISION-CPU-VULKAN-AND-FINAL.md) for the precision migration, [PHASE-9D](plan/PHASE-9D-FP8-VULKAN-BACKEND.md) for FP8 support, and [PHASE-10](plan/PHASE-10-STATE-PRECISION-ACCUMULATION.md) for state-precision accumulation. Dispatch geometry for `linear_generic` placement strategy kernels (`normalize_gradients`, `adam_update`, `clamp_temperatures`, `stabilize_reduce_grad_h`) uses per-kernel workgroup count resolvers rather than `tile_count`.

## Design Constraints

Two hard requirements for the Vulkan back-end design:

1. **Explicit everything** — every barrier, resource binding, and memory transfer is host-controlled
2. **Pre-compiled pipelines** — all shader variants are compiled to SPIR-V before the first dispatch

This means the Vulkan back-end isn't "OpenCL with a different API" — it's a fully explicit execution engine where the host assumes total responsibility for synchronization correctness, memory hazard resolution, and resource lifetime management. In exchange, it gains fine-grained control over exactly what happens and when.

---

## Execution Model

### The Architecture→Vulkan Mapping

| Architecture Concept                  | Vulkan Equivalent                                                           |
| ------------------------------------- | --------------------------------------------------------------------------- |
| Kernel                                | Compute Pipeline (`VkPipeline`)                                             |
| N parallel kernel dispatches (tiles)  | **Single** `vkCmdDispatch(N, 1, 1)` — `gl_WorkGroupID` serves as tile index |
| Work-group                            | Workgroup (`local_size` in GLSL)                                            |
| Work-items within a group             | Invocations                                                                 |
| Subgroup / Wavefront                  | Subgroup (`VK_KHR_shader_subgroup`)                                         |
| `cl_event` between kernels            | `vkCmdPipelineBarrier` (compute → compute)                                  |
| `__local` memory                      | `shared` storage qualifier in GLSL                                          |
| `__global` buffer                     | Storage Buffer (SSBO)                                                       |
| `__constant` / `DEVICE_CONST_` buffer | Uniform Buffer (UBO) or `readonly` SSBO                                     |
| Build-time `-D` symbols               | Specialization Constants                                                    |
| Kernel arguments (scalars)            | Push Constants                                                              |
| Kernel arguments (buffers)            | Descriptor Set Bindings                                                     |
| Command queue                         | `VkQueue` + `VkCommandBuffer`                                               |
| `inference_event`                     | `VkFence` (host CPU waits)                                                  |
| `final_batch_event`                   | `VkFence` (host CPU waits)                                                  |
| Kernel arg setting (`clSetKernelArg`) | `vkCmdPushDescriptorSetKHR` / `vkUpdateDescriptorSets`                      |

### Single-Dispatch Parallelism

The most significant structural difference from OpenCL: **N independent tiles are a single dispatch, not N separate enqueues.** When the architecture says "dispatch N parallel kernels for gradient computation," this maps to:

```c
vkCmdDispatch(cmd, total_tile_count, 1, 1);
```

Inside the shader, `gl_WorkGroupID.x` replaces the host-provided `flat_tile_index`. The host no longer loops over tiles to enqueue — the GPU's hardware scheduler distributes workgroups across compute units.

For 2D decomposition (e.g., `forward_pass` across samples × hidden blocks):

```c
uint group_x = batch_chunk_count;
uint group_y = padded_hidden_count / SIMD_WIDTH;
vkCmdDispatch(cmd, group_x, group_y, 1);
// gl_WorkGroupID.x → sample index
// gl_WorkGroupID.y → hidden block index
```

This eliminates the per-tile dispatch overhead that plagues OpenCL implementations with many small tiles.

### Command Buffer Strategy

The architecture's Act/Learn split maps to two independently recorded command buffers:

```
                    ┌─────────────────────┐
                    │  Record act_cmd     │
                    │  (Nodes 4→5→6/7→23) │
                    └────────┬────────────┘
                             │
                    vkQueueSubmit(act_cmd) → signal act_fence
                             │
                    host: vkWaitForFences(act_fence)
                    host: read staging buffer (Final Probs)
                    host: ← inference_event equivalent
                             │
                    host: await ground truth...
                             │
                    ┌─────────────────────┐
                    │  Record learn_cmd   │
                    │  (Nodes 8→...→25)   │
                    └────────┬────────────┘
                             │
                    vkQueueSubmit(learn_cmd) → signal learn_fence
                             │
                    host: vkWaitForFences(learn_fence)
                    host: ← final_batch_event equivalent
```

**Why two command buffers, not one?** The architecture's Event-Triggered Execution Mode requires the host to read inference results and potentially wait for external events before triggering learning. A single command buffer would force the device to idle during this host-side decision window. Splitting allows the Act command buffer to complete and signal independently.

**Why not pre-record and replay?** The Learn phase's reduction tree depth, streaming chunk count, and clipping thresholds may change between batches. The command buffers are re-recorded each cycle. This is not a performance concern — Vulkan command recording is CPU-side and fast.

### DAG Execution via Recorded Command Buffers

Each DAG edge becomes a pipeline barrier. The critical Vulkan optimization: **dispatches with no mutual data hazards need no barrier between them.** The architect can reason about read/write sets and eliminate unnecessary synchronization.

```c
// vulkan_dag_recording.c

static void record_learn_phase(VkCommandBuffer cmd, VulkanBackend* vk,
                                PipelineContext* ctx) {
    // ================================================================
    // Phase I: Parallel Gradient Generation
    // Nodes 8, 9, 10 read SAME inputs, write DIFFERENT outputs
    // → Zero barriers between them (no RAW/WAW hazards)
    // ================================================================

    // Node 8: module param grads
    bind_and_dispatch(cmd, &vk->module_param_grads,
                      ctx->desc_node8, &ctx->push_node8,
                      ctx->total_tile_count, 1, 1);

    // Node 9: backprop to hidden
    bind_and_dispatch(cmd, &vk->backprop_to_hidden,
                      ctx->desc_node9, &ctx->push_node9,
                      ctx->total_tile_count, 1, 1);

    // Node 10: temp grads
    bind_and_dispatch(cmd, &vk->temp_grads,
                      ctx->desc_node10, &ctx->push_node10,
                      ctx->total_tile_count, 1, 1);

    // Single barrier: Nodes {8,9,10} → Node 11
    // ALL four output buffers must be visible before clipping
    compute_write_read_barrier(cmd, 4, (VkBuffer[]){
        ctx->buf_partial_grad_mod_w, ctx->buf_partial_grad_mod_b,
        ctx->buf_partial_grad_temps, ctx->buf_partial_grad_h_aos
    });

    // Node 11: clip partial gradients (one workgroup per tile)
    bind_and_dispatch(cmd, &vk->clip_partial,
                      ctx->desc_node11, &ctx->push_node11,
                      ctx->total_tile_count, 1, 1);

    // Barrier: Node 11 → Node 13
    compute_write_read_barrier(cmd, 1, (VkBuffer[]){
        ctx->buf_clipped_grad_h_aos
    });

    // Node 13: gather and permute (global barrier — reads all tiles)
    uint grad_h_rows = ctx->total_batch_count * ctx->padded_hidden_count;
    bind_and_dispatch(cmd, &vk->gather_permute,
                      ctx->desc_node13, &ctx->push_node13,
                      (grad_h_rows + vk->workgroup_size - 1) / vk->workgroup_size,
                      1, 1);

    // Barrier: Node 13 → Nodes {15, 16} (both read from different outputs)
    compute_write_read_barrier(cmd, 3, (VkBuffer[]){
        ctx->buf_permuted_grad_h_soa,
        ctx->buf_clipped_grad_mod_w,  // For Node 15
        ctx->buf_clipped_grad_temps   // For Node 15
    });

    // ================================================================
    // Phase II: Independent Reductions (can overlap)
    // Node 15 (module/temp grads) and Node 16 (Grad_H) operate on
    // disjoint buffers → no barrier between them
    // ================================================================

    // Node 15: Recursive clip-aggregation for module & temp grads
    record_reduction_tree(cmd, vk, &ctx->tree_grad_mod_w);
    record_reduction_tree(cmd, vk, &ctx->tree_grad_mod_b);
    record_reduction_tree(cmd, vk, &ctx->tree_grad_temps);

    // Node 16: Specialized Grad_H reduction
    bind_and_dispatch(cmd, &vk->stabilize_reduce_grad_h,
                      ctx->desc_node16, &ctx->push_node16,
                      grad_h_rows, 1, 1);

    // Barrier: Phase II → Phase III
    compute_write_read_barrier(cmd, 4, (VkBuffer[]){
        ctx->buf_summed_grad_mod_w, ctx->buf_summed_grad_mod_b,
        ctx->buf_summed_grad_temps, ctx->buf_summed_grad_h
    });

    // ================================================================
    // Phase III: Streaming Shared Backprop
    // Each chunk: dispatch 17+18 → barrier → dispatch 19
    // ================================================================
    for (uint c = 0; c < ctx->num_batch_chunks; c++) {
        // Update push constants for this chunk
        SharedBackpropPush push17 = make_shared_weights_push(ctx, c);
        SharedBackpropPush push18 = make_shared_biases_push(ctx, c);

        // Nodes 17 & 18: independent (different outputs)
        push_and_dispatch(cmd, &vk->backprop_shared_weights,
                          &push17, 1, 1, 1);
        push_and_dispatch(cmd, &vk->backprop_shared_biases,
                          &push18, 1, 1, 1);

        compute_write_read_barrier(cmd, 2, (VkBuffer[]){
            ctx->buf_partial_grad_sw, ctx->buf_partial_grad_sb
        });

        // Node 19: clip shared grads for this chunk
        ClipSharedPush push19 = make_clip_shared_push(ctx, c);
        push_and_dispatch(cmd, &vk->clip_shared_grads,
                          &push19, 1, 1, 1);

        compute_write_read_barrier(cmd, 2, (VkBuffer[]){
            ctx->buf_clipped_grad_sw, ctx->buf_clipped_grad_sb
        });
    }

    // ================================================================
    // Phase IV: Final Aggregation & Normalization
    // ================================================================
    record_reduction_tree(cmd, vk, &ctx->tree_grad_sw);
    record_reduction_tree(cmd, vk, &ctx->tree_grad_sb);

    compute_write_read_barrier(cmd, 2, (VkBuffer[]){
        ctx->buf_summed_grad_sw, ctx->buf_summed_grad_sb
    });

    // Node 21: normalize all parameter groups
    for (uint g = 0; g < NUM_PARAM_GROUPS; g++) {
        NormalizePush push21 = make_normalize_push(ctx, g);
        push_and_dispatch(cmd, &vk->normalize_grads,
                          &push21, dispatch_size(ctx, g), 1, 1);
    }

    compute_write_read_barrier(cmd, NUM_PARAM_GROUPS,
                                ctx->all_final_grad_buffers);

    // ================================================================
    // Phase V: Parameter Update
    // ================================================================
    for (uint g = 0; g < NUM_PARAM_GROUPS; g++) {
        AdamPush push24 = make_adam_push(ctx, g);
        push_and_dispatch(cmd, &vk->adam_update,
                          &push24, dispatch_size(ctx, g), 1, 1);
    }

    compute_write_read_barrier(cmd, 1, (VkBuffer[]){
        ctx->buf_temps
    });

    // Node 25: clamp temperatures
    bind_and_dispatch(cmd, &vk->clamp_temps,
                      ctx->desc_node25, &ctx->push_node25,
                      (ctx->total_modules_count + vk->workgroup_size - 1)
                       / vk->workgroup_size, 1, 1);
}
```

Notice: the barrier structure reveals the **inherent parallelism** in the DAG. Nodes 8/9/10 have no mutual barriers. Nodes 15 and 16 have no mutual barriers. The host explicitly encodes this information, which is implicit (and potentially missed by the driver) in OpenCL's single-queue model.

The helper `compute_write_read_barrier` encapsulates the standard compute→compute barrier:

```c
static void compute_write_read_barrier(VkCommandBuffer cmd,
                                        uint buffer_count,
                                        VkBuffer* buffers) {
    VkMemoryBarrier2 barrier = {
        .sType         = VK_STRUCTURE_TYPE_MEMORY_BARRIER_2,
        .srcStageMask  = VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT,
        .srcAccessMask = VK_ACCESS_2_SHADER_WRITE_BIT,
        .dstStageMask  = VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT,
        .dstAccessMask = VK_ACCESS_2_SHADER_READ_BIT,
    };
    VkDependencyInfo dep = {
        .sType                    = VK_STRUCTURE_TYPE_DEPENDENCY_INFO,
        .memoryBarrierCount       = 1,
        .pMemoryBarriers          = &barrier,
    };
    vkCmdPipelineBarrier2(cmd, &dep);
}
```

**Why a single global memory barrier, not per-buffer barriers?** For compute-only workloads on a single queue, a global memory barrier flushes all outstanding writes. Buffer-granular barriers (`VkBufferMemoryBarrier2`) add host-side complexity without meaningful driver-side benefit — the GPU's L2 cache is a unified domain. Per-buffer barriers matter for queue ownership transfers and image layout transitions, neither of which applies here.

---

## Resource Management

### Memory Allocation Strategy

All compute buffers reside in device-local memory. The architecture's staging requirement (Node 23: D2H copy of `Final Probs`) requires one host-visible staging buffer.

Using the [Vulkan Memory Allocator (VMA)](https://github.com/GPUOpen-LibrariesAndSDKs/VulkanMemoryAllocator) eliminates manual memory type selection, sub-allocation, and aliasing:

```c
// vulkan_memory.c

typedef struct {
    VkBuffer       buffer;
    VmaAllocation  allocation;
    VkDeviceSize   size;
} GpuBuffer;

static GpuBuffer create_device_buffer(VmaAllocator vma, VkDeviceSize size,
                                       VkBufferUsageFlags usage) {
    GpuBuffer buf = {0};
    buf.size = size;

    VkBufferCreateInfo buffer_info = {
        .sType = VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO,
        .size  = size,
        .usage = usage,
    };
    VmaAllocationCreateInfo alloc_info = {
        .usage = VMA_MEMORY_USAGE_AUTO_PREFER_DEVICE,
        .flags = 0,
    };
    vmaCreateBuffer(vma, &buffer_info, &alloc_info,
                    &buf.buffer, &buf.allocation, NULL);
    return buf;
}

static GpuBuffer create_staging_buffer(VmaAllocator vma, VkDeviceSize size) {
    GpuBuffer buf = {0};
    buf.size = size;

    VkBufferCreateInfo buffer_info = {
        .sType = VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO,
        .size  = size,
        .usage = VK_BUFFER_USAGE_TRANSFER_DST_BIT,
    };
    VmaAllocationCreateInfo alloc_info = {
        .usage = VMA_MEMORY_USAGE_AUTO_PREFER_HOST,
        .flags = VMA_ALLOCATION_CREATE_HOST_ACCESS_RANDOM_BIT
               | VMA_ALLOCATION_CREATE_MAPPED_BIT,
    };
    vmaCreateBuffer(vma, &buffer_info, &alloc_info,
                    &buf.buffer, &buf.allocation, NULL);
    return buf;
}
```

Buffer usage flags follow directly from the architecture's data flow:

| Architecture Data Role                        | Vulkan Usage Flags                      |
| --------------------------------------------- | --------------------------------------- |
| Learnable parameters (Weights, Biases, Temps) | `STORAGE_BUFFER` (read + write by Adam) |
| Intermediate activations, partials, gradients | `STORAGE_BUFFER`                        |
| Optimizer state (m1, m2)                      | `STORAGE_BUFFER`                        |
| Final Probs (for D2H)                         | `STORAGE_BUFFER \| TRANSFER_SRC`        |
| Staging readback buffer                       | `TRANSFER_DST`                          |
| Offset lists (indirection tables)             | `STORAGE_BUFFER`                        |

### Descriptor Set Strategy

The architecture's kernels have diverse buffer signatures — `forward_pass` uses 6 buffers, `clip_partial_gradients` uses 10. Rather than managing a complex descriptor set hierarchy, the backend employs two complementary strategies:

**Fixed pipelines (main kernels):** Pre-allocated descriptor sets. The bindings don't change within a batch, so one descriptor set per kernel, updated once per batch:

```c
// One-time setup for forward_pass descriptor set
VkDescriptorSetLayoutBinding bindings[] = {
    {0, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, 1, VK_SHADER_STAGE_COMPUTE_BIT, NULL}, // input
    {1, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, 1, VK_SHADER_STAGE_COMPUTE_BIT, NULL}, // sample_mask
    {2, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, 1, VK_SHADER_STAGE_COMPUTE_BIT, NULL}, // weights
    {3, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, 1, VK_SHADER_STAGE_COMPUTE_BIT, NULL}, // biases
    {4, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, 1, VK_SHADER_STAGE_COMPUTE_BIT, NULL}, // hidden_out
    {5, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, 1, VK_SHADER_STAGE_COMPUTE_BIT, NULL}, // mask_out
};
```

**Dynamic pipelines (reduction engine):** `VK_KHR_push_descriptor` for inline binding updates. The aggregation and clipping kernels are dispatched many times per batch with different source/destination buffers at each reduction stage. Pre-allocating descriptor sets for every stage combo is wasteful:

```c
// Inside record_reduction_tree, per-stage:
VkDescriptorBufferInfo src_info = { stage_src_buffer, 0, VK_WHOLE_SIZE };
VkDescriptorBufferInfo off_info = { offset_list_buf,  0, VK_WHOLE_SIZE };
VkDescriptorBufferInfo dst_info = { stage_dst_buffer, 0, VK_WHOLE_SIZE };

VkWriteDescriptorSet writes[] = {
    {VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET, NULL, VK_NULL_HANDLE,
     0, 0, 1, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, NULL, &src_info, NULL},
    {VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET, NULL, VK_NULL_HANDLE,
     1, 0, 1, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, NULL, &off_info, NULL},
    {VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET, NULL, VK_NULL_HANDLE,
     2, 0, 1, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, NULL, &dst_info, NULL},
};

vkCmdPushDescriptorSetKHR(cmd, VK_PIPELINE_BIND_POINT_COMPUTE,
                           vk->aggregate_layout, /*set=*/0,
                           3, writes);
```

This records the binding update directly into the command buffer — no pre-allocation, no pool management, no per-stage descriptor sets.

**Precision-Typed Variants (ADR-026):** Under mixed-precision configurations, the reduction tree renderer must select between storage-entry and compute-entry pipeline variants:

| Stage | Source Role | Pipeline |
| :--- | :--- | :--- |
| Stage 0 | `"storage"` | `aggregate_partials` (reads via `load_storage()`) |
| Stage 0 | `"compute"` | `aggregate_partials_from_compute` (direct COMPUTE_TYPE read) |
| Stage ≥ 1 | always `"compute"` | `aggregate_partials_from_compute` |

The `_from_compute` GLSL shader shares identical algorithm but declares `COMPUTE_TYPE` for its source buffer binding rather than `STORAGE_TYPE`. Pipeline selection occurs per-stage in `record_reduction_tree`:

```c
// Inside record_reduction_tree, per-stage:
VkPipeline pipeline;
if (stage == 0 && source_role == PRECISION_ROLE_STORAGE)
    pipeline = vk->aggregate_partials;           // storage-entry
else
    pipeline = vk->aggregate_partials_from_compute;  // compute-entry

vkCmdBindPipeline(cmd, VK_PIPELINE_BIND_POINT_COMPUTE, pipeline);
```

### FP8 Software Conversion (Phase 9D, ADR-025 §7)

GLSL and SPIR-V have no native FP8 type. FP8 storage buffers use `uint8_t` (via `VK_KHR_8bit_storage` extension) with software conversion to `COMPUTE_TYPE` in the shader. The conversion is arithmetic (~12–15 ALU ops per value), not LUT-based — GLSL lacks OpenCL's `__constant` memory for embedding lookup tables, and the cost is negligible for bandwidth-bound workloads.

**Vulkan extension requirements:**

| Extension / Feature | Role |
| :--- | :--- |
| `VK_KHR_8bit_storage` | `uint8_t` in storage buffer declarations |
| `VK_KHR_shader_float16_int8` | `uint8_t` in function parameters/returns; required when `COMPUTE_TYPE_IS_HALF` |
| `shaderFloat64` device feature | Required when `COMPUTE_TYPE_IS_DOUBLE` |

`VulkanContext` detects and enables these extensions at device creation. `check_precision_requirements(precision)` validates device capabilities before dispatch, raising `RuntimeError` with a clear message if requirements are unmet.

**Compile-time flag scheme:**

FP8 variants are selected by `-D` flags injected via `glslc`, not specialization constants (the GLSL preprocessor cannot evaluate specialization constants):

```
-DSTORAGE_TYPE_IS_FP8=1 -DSTORAGE_TYPE_IS_E4M3=1 -DCOMPUTE_TYPE_IS_FLOAT=1
```

When `STORAGE_TYPE_IS_FP8=1`, `common.glsl`:
- Enables `GL_EXT_shader_8bit_storage` and `GL_EXT_shader_explicit_arithmetic_types_int8`
- Defines `STORAGE_TYPE` as `uint8_t` (instead of `float`/`float16_t`/`double`)
- Overrides `read_storage()` and `write_storage()` to route through `load_storage_fp8()` and `store_storage_fp8()`
- No `.comp` files were modified — the existing `{ STORAGE_TYPE data[]; }` buffer declarations and `read_storage()`/`write_storage()` abstractions handle FP8 transparently

**SPIR-V variant suffixes:**

| Suffix | Storage | Compute | State |
| :--- | :--- | :--- | :--- |
| `_s8e4c32x32` | E4M3 | FP32 | FP32 |
| `_s8e4c16x32` | E4M3 | FP16 | FP32 |
| `_s8e4c32x64` | E4M3 | FP32 | FP64 |
| `_s8e4c64x64` | E4M3 | FP64 | FP64 |
| `_s8e5c32x32` | E5M2 | FP32 | FP32 |
| `_s8e5c16x32` | E5M2 | FP16 | FP32 |
| `_s8e5c32x64` | E5M2 | FP32 | FP64 |
| `_s8e5c64x64` | E5M2 | FP64 | FP64 |

**Conversion characteristics:**
- **E4M3:** sign(1) + exponent(4) + mantissa(3), bias=7, max=448, min subnormal=2⁻⁹
- **E5M2:** sign(1) + exponent(5) + mantissa(2), bias=15, max=57344, min subnormal=2⁻¹⁶
- **Rounding:** Round-to-nearest-even on store
- **Overflow:** Saturates to format max (±448 for E4M3, ±57344 for E5M2)
- **NaN handling:** NaN inputs store as zero (E4M3 has no NaN; E5M2 preserves NaN only for inf/NaN inputs)

`_pipeline_cache.py` maps `PrecisionConfig` → SPIR-V variant suffix via `precision_to_suffix()` from `src/shared/precision_suffix.py`, keyed by `(storage_dtype, compute_dtype, state_dtype)`.

### Push Constant Design

All scalar parameters from the architecture's kernel interfaces map to push constants. Vulkan guarantees a minimum of 128 bytes; every kernel in the architecture fits within this budget:

```c
// Push constant structs mirror the OpenCL scalar parameter lists

typedef struct {
    uint32_t batch_chunk_offset;
    uint32_t batch_chunk_count;
    uint32_t total_batch_count;
    uint32_t padded_input_count;
    uint32_t padded_hidden_count;
} ForwardPassPush;  // 20 bytes

typedef struct {
    uint32_t flat_tile_index_base;  // Not needed — gl_WorkGroupID.x replaces this
    uint32_t num_class_chunks;
    uint32_t classes_per_chunk;
    uint32_t modules_per_chunk;
    uint32_t total_batch_count;
    uint32_t total_output_class_count;
    uint32_t padded_total_output_class_count;
    uint32_t total_modules_count;
    uint32_t total_tile_count;
} ProbsLossPush;  // 36 bytes (flat_tile_index removed — see below)

typedef struct {
    uint32_t use_per_item_norm;       // FLAG
    float    clipping_threshold;
    float    epsilon;
    uint32_t num_class_chunks;
    uint32_t classes_per_chunk;
    uint32_t modules_per_chunk;
    uint32_t total_batch_count;
    uint32_t padded_hidden_count;
    uint32_t total_tile_count;
} ClipPartialsPush;  // 36 bytes

typedef struct {
    float    learning_rate;
    float    beta1_pow_t;
    float    beta2_pow_t;
    float    beta1;
    float    beta2;
    float    epsilon;
    uint32_t parameter_count;
} AdamUpdatePush;  // 28 bytes
```

**The `flat_tile_index` elimination.** In the OpenCL backend, the host provides `flat_tile_index` as a scalar because each tile is a separate kernel enqueue. In Vulkan, N tiles are a single dispatch — `gl_WorkGroupID.x` serves as the tile index natively. This eliminates one parameter and the associated per-tile host-side loop.

For the few cases where a push constant value changes per-dispatch within a recording loop (e.g., streaming chunk index in Phase III), the host simply calls `vkCmdPushConstants` before each dispatch:

```c
for (uint c = 0; c < num_batch_chunks; c++) {
    SharedBackpropPush push = { .batch_chunk_index = c, ... };
    vkCmdPushConstants(cmd, layout, VK_SHADER_STAGE_COMPUTE_BIT,
                       0, sizeof(push), &push);
    vkCmdDispatch(cmd, groups_x, groups_y, 1);
}
```

---

## Shader Architecture

### Specialization Constants

The architecture's mandatory build-time symbols (Contract Article 6) map to Vulkan specialization constants. These are baked into the SPIR-V at pipeline creation time, enabling the compiler to optimize for the specific hardware profile:

```glsl
// Common specialization constant block (shared by all shaders)
layout(constant_id = 0) const uint SPEC_SIMD_WIDTH             = 8;
layout(constant_id = 1) const uint SPEC_LOCAL_MEM_BANK_PADDING  = 1;
layout(constant_id = 2) const uint SPEC_C_TILE_SIZE             = 8;
layout(constant_id = 3) const uint SPEC_PROBLEM_TYPE            = 0;  // 0=CCE, 1=BCE
```

On the host, these are set at pipeline creation:

```c
VkSpecializationMapEntry entries[] = {
    {0, offsetof(SpecConstants, simd_width),      sizeof(uint32_t)},
    {1, offsetof(SpecConstants, bank_padding),     sizeof(uint32_t)},
    {2, offsetof(SpecConstants, tile_size),         sizeof(uint32_t)},
    {3, offsetof(SpecConstants, problem_type),      sizeof(uint32_t)},
};

SpecConstants spec = {
    .simd_width   = device_props.subgroup_size,
    .bank_padding = 1,
    .tile_size    = device_props.subgroup_size,
    .problem_type = PROBLEM_TYPE_CCE,
};

VkSpecializationInfo spec_info = {
    .mapEntryCount = 4,
    .pMapEntries   = entries,
    .dataSize      = sizeof(spec),
    .pData         = &spec,
};
```

The `SPEC_PROBLEM_TYPE` constant deserves special attention. In the OpenCL backend, kernels 8/9/10 use a runtime flag (`src_scalar_FLAG_problem_type`) to select CCE vs BCE paths. In Vulkan, this becomes a **specialization constant**, meaning the compiler eliminates the dead branch entirely. The cost: two pipeline variants per kernel that uses it. The benefit: zero divergent branching on the GPU.

| Kernel                           | CCE Pipeline             | BCE Pipeline |
| -------------------------------- | ------------------------ | ------------ |
| `compute_probs_loss_*`           | Separate shaders already | (Same)       |
| `calculate_module_param_grads`   | Variant A                | Variant B    |
| `backprop_error_to_hidden`       | Variant A                | Variant B    |
| `calculate_chunk_temp_gradients` | Variant A                | Variant B    |

Total pipelines: ~22 (including SUM/AVG variants for aggregation). This is a trivial one-time cost.

### Subgroup Operations

Vulkan's subgroup operations (`VK_KHR_shader_subgroup`, core since Vulkan 1.1) are the backend's primary advantage for reduction-heavy kernels. They replace the OpenCL pattern of shared-memory tree reduction with hardware-native warp/wavefront operations:

```glsl
// OpenCL pattern (shared memory tree reduction):
//   local[lid] = value;
//   barrier(CLK_LOCAL_MEM_FENCE);
//   for (uint s = local_size/2; s > 0; s >>= 1) {
//       if (lid < s) local[lid] += local[lid + s];
//       barrier(CLK_LOCAL_MEM_FENCE);
//   }                                          // log2(N) barriers, N shared mem ops

// Vulkan subgroup pattern:
float warp_sum = subgroupAdd(value);          // Single instruction, zero barriers
```

For workgroups larger than one subgroup, a two-level reduction bridges the gap:

```glsl
#extension GL_KHR_shader_subgroup_arithmetic : require

shared float cross_subgroup[32];  // Max 32 subgroups per workgroup typical

float workgroup_reduce_add(float value) {
    // Level 1: Subgroup-native reduction
    float subgroup_sum = subgroupAdd(value);

    // Level 2: Cross-subgroup via shared memory
    if (subgroupElect()) {
        cross_subgroup[gl_SubgroupID] = subgroup_sum;
    }
    barrier();

    // Level 3: First subgroup reduces the cross-subgroup results
    float total = 0.0;
    if (gl_SubgroupID == 0) {
        float val = (gl_SubgroupInvocationID < gl_NumSubgroups)
                    ? cross_subgroup[gl_SubgroupInvocationID] : 0.0;
        total = subgroupAdd(val);
    }

    // Broadcast result to all invocations
    if (gl_SubgroupID == 0 && subgroupElect()) {
        cross_subgroup[0] = total;
    }
    barrier();
    return cross_subgroup[0];
}
```

This replaces **all** shared-memory reductions in the kernel set — L2 norm calculations (Nodes 11, 15b, 16, 19), aggregation (Nodes 14, 15a, 20a), and the dot-product accumulation in matmul kernels.

### The Hot Kernel: `forward_pass.comp`

```glsl
#version 450
#extension GL_KHR_shader_subgroup_arithmetic : require

// ---- Specialization Constants (Article 6) ----
layout(constant_id = 0) const uint SPEC_SIMD_WIDTH            = 8;
layout(constant_id = 1) const uint SPEC_LOCAL_MEM_BANK_PADDING = 1;

layout(local_size_x_id = 0) in;  // local_size_x = SPEC_SIMD_WIDTH

// ---- Push Constants (scalar parameters) ----
layout(push_constant) uniform PushConstants {
    uint batch_chunk_offset;
    uint batch_chunk_count;
    uint total_batch_count;
    uint padded_input_count;
    uint padded_hidden_count;
} push;

// ---- Descriptor Bindings (buffer parameters) ----
layout(set = 0, binding = 0) readonly buffer InputBuf {
    float data[];
} src_input;

layout(set = 0, binding = 1) readonly buffer MaskBuf {
    float data[];
} src_sample_mask;

layout(set = 0, binding = 2) readonly buffer WeightsBuf {
    float data[];   // SIMD-major layout: [hb][i][lane]
} src_weights;

layout(set = 0, binding = 3) readonly buffer BiasesBuf {
    float data[];
} src_biases;

layout(set = 0, binding = 4) writeonly buffer HiddenBuf {
    float data[];
} dest_hidden;

layout(set = 0, binding = 5) writeonly buffer HiddenMaskBuf {
    float data[];
} dest_hidden_mask;

// ---- Shared Memory Tile (bank-conflict avoidance) ----
shared float simd_tile[SPEC_SIMD_WIDTH][SPEC_SIMD_WIDTH + SPEC_LOCAL_MEM_BANK_PADDING];

void main() {
    // Dispatch: vkCmdDispatch(batch_chunk_count, padded_hidden/SIMD_WIDTH, 1)
    uint sample = gl_WorkGroupID.x + push.batch_chunk_offset;
    uint hb     = gl_WorkGroupID.y;   // Hidden block index
    uint lid    = gl_LocalInvocationID.x;

    if (sample >= push.total_batch_count) return;

    uint h_idx      = hb * SPEC_SIMD_WIDTH + lid;
    float mask_val  = src_sample_mask.data[sample];
    uint input_base = sample * push.padded_input_count;
    uint w_base     = hb * push.padded_input_count * SPEC_SIMD_WIDTH;

    // Initialize accumulator with bias
    float accum = src_biases.data[h_idx];

    // Tiled dot product across input dimension
    for (uint it = 0; it < push.padded_input_count; it += SPEC_SIMD_WIDTH) {

        // Cooperatively load weight tile into shared memory
        // Each invocation loads one row: weights[hb][(it+lid)][0..SIMD_WIDTH-1]
        for (uint k = 0; k < SPEC_SIMD_WIDTH; k++) {
            simd_tile[lid][k] = src_weights.data[
                w_base + (it + lid) * SPEC_SIMD_WIDTH + k
            ];
        }
        barrier();

        // Compute: each invocation reads column `lid` from tile
        // Column-wise access → bank-conflict-free due to +1 padding
        for (uint j = 0; j < SPEC_SIMD_WIDTH; j++) {
            float x = src_input.data[input_base + it + j];
            accum = fma(simd_tile[j][lid], x, accum);
        }
        barrier();
    }

    // ReLU activation + sample mask
    float activated = max(0.0, accum) * mask_val;
    float rmask     = (accum > 0.0) ? mask_val : 0.0;

    // Store results
    uint out_idx = sample * push.padded_hidden_count + h_idx;
    dest_hidden.data[out_idx]      = activated;
    dest_hidden_mask.data[out_idx] = rmask;
}
```

**Performance characteristics:**

- Memory access: weight loads are coalesced (contiguous `lid` → contiguous addresses); input broadcast leverages L1 cache (all invocations in a subgroup read the same address)
- Shared memory: weights are tiled into shared memory to enable column-wise access by all invocations; bank padding guarantees conflict-free access. Input is read directly from global memory, relying on L1 cache for broadcast efficiency.
- Zero branching in the hot loop — `max()` and ternary compile to predicated instructions
- Padding guarantees: every dimension is SIMD-aligned, so no scalar tail-loop cleanup

### The Bottleneck Table

| Kernel                             | Bottleneck                  | Subgroup Value                   | Dispatch Parallelism                       |
| ---------------------------------- | --------------------------- | -------------------------------- | ------------------------------------------ |
| `forward_pass` (4)                 | Compute (matmul)            | Low (dot product dominates)      | **Critical** — N samples × M hidden blocks |
| `render_logits_chunk` (5)          | Compute (matmul)            | Low                              | **Critical**                               |
| `compute_probs_loss_cce` (6)       | Mixed (exp, log, reduction) | **High** (workgroup max/sum)     | High                                       |
| `compute_probs_loss_bce` (7)       | Mixed (sigmoid, log)        | Medium                           | High                                       |
| `calc_module_param_grads` (8)      | Compute (outer product)     | Medium (bias sum reduction)      | **Critical**                               |
| `backprop_error_to_hidden` (9)     | Compute (matmul)            | Low                              | **Critical**                               |
| `calc_temp_gradients` (10)         | Mixed                       | **High** (workgroup reduction)   | High                                       |
| `clip_partial_gradients` (11)      | Memory (norm scan)          | **High** (L2 norm reduction)     | High                                       |
| `gather_and_permute` (13)          | Memory (scatter/gather)     | Low                              | **Critical**                               |
| `aggregate_*` (14/15/20)           | Memory (streaming sum)      | **High** (partial sum reduction) | Medium                                     |
| `clip_intermediate_grad` (15b/20b) | Memory (norm + scale)       | **High** (L2 norm)               | Low-Medium                                 |
| `stabilize_reduce_grad_h` (16)     | Mixed                       | **Critical** (staged reduction)  | High                                       |
| `backprop_shared_weights` (17)     | Compute (outer product)     | Medium                           | **Critical**                               |
| `backprop_shared_biases` (18)      | Memory (reduction)          | **High**                         | High                                       |
| `normalize_gradients` (21)         | Memory (element-wise)       | Low                              | High                                       |
| `adam_update` (24)                 | Memory (element-wise)       | Low                              | High                                       |

The "Subgroup Value" column is the key differentiator from the OpenCL backend. Any kernel that performs a workgroup-level reduction benefits enormously from `subgroupAdd` / `subgroupMax`, replacing shared-memory barrier chains with single-cycle hardware operations.

---

## The Reduction Engine in Vulkan

### Dispatch-Barrier Staging

The host-driven recursive clip-aggregation engine (Nodes 15, 20) maps to a sequence of dispatches interleaved with pipeline barriers. Each stage of the `log_K(N)` tree is a dispatch-barrier-dispatch-barrier cycle:

```c
// vulkan_reduction_engine.c

typedef struct {
    GpuBuffer   staging_buffers[2];  // Double-buffered for ping-pong
    GpuBuffer*  offset_list_buffers; // One per stage
    uint32_t*   stage_node_counts;   // Nodes at each stage
    uint32_t*   stage_fan_in;        // K at each stage
    uint32_t    partial_width;       // Elements per partial
    uint32_t    num_stages;
    float       t_algorithmic;
    float       lambda;
    float       fp_max;
    float       epsilon;
} ReductionTreePlan;

static void record_reduction_tree(VkCommandBuffer cmd,
                                   VulkanBackend* vk,
                                   ReductionTreePlan* plan) {
    // First stage reads from the partial collection
    VkBuffer src_buffer = plan->staging_buffers[0].buffer;  // partial_collection
    VkBuffer dst_buffer = plan->staging_buffers[1].buffer;

    for (uint32_t s = 0; s < plan->num_stages; s++) {
        uint32_t num_nodes = plan->stage_node_counts[s];
        uint32_t K         = plan->stage_fan_in[s];

        // ---- Sum phase: aggregate_* ----
        // Select tier based on fan-in
        ComputeKernel* agg_kernel = (K <= vk->subgroup_size)
            ? &vk->aggregate_register
            : &vk->aggregate_local;

        AggregatePush agg_push = {
            .partial_offset_list_count = K,
            .partial_width             = plan->partial_width,
            .operation_type            = AGG_MODE_SUM,
        };

        // Push descriptors: src, offset_list, dst change every stage
        VkDescriptorBufferInfo src_info = { src_buffer, 0, VK_WHOLE_SIZE };
        VkDescriptorBufferInfo off_info = { plan->offset_list_buffers[s].buffer,
                                            0, VK_WHOLE_SIZE };
        VkDescriptorBufferInfo dst_info = { dst_buffer, 0, VK_WHOLE_SIZE };

        VkWriteDescriptorSet writes[3];
        fill_aggregate_writes(writes, &src_info, &off_info, &dst_info);

        vkCmdPushDescriptorSetKHR(cmd, VK_PIPELINE_BIND_POINT_COMPUTE,
                                   agg_kernel->layout, 0, 3, writes);
        vkCmdPushConstants(cmd, agg_kernel->layout,
                           VK_SHADER_STAGE_COMPUTE_BIT,
                           0, sizeof(agg_push), &agg_push);
        vkCmdBindPipeline(cmd, VK_PIPELINE_BIND_POINT_COMPUTE,
                           agg_kernel->pipeline);

        // One workgroup per reduction node, each handles partial_width elements
        uint32_t groups = num_nodes
            * ((plan->partial_width + vk->workgroup_size - 1) / vk->workgroup_size);
        vkCmdDispatch(cmd, groups, 1, 1);

        // Barrier: sum → clip
        compute_write_read_barrier(cmd, 1, &dst_buffer);

        // ---- Clip phase: clip_intermediate_grad ----
        uint32_t j = plan->num_stages - 1 - s;
        float t_policy = plan->t_algorithmic + plan->lambda * (float)(j * j);
        float t_safety = plan->fp_max / (float)K;
        float threshold = fminf(t_policy, t_safety);

        ClipIntermediatePush clip_push = {
            .clipping_threshold = threshold,
            .epsilon            = plan->epsilon,
            .parameter_count    = plan->partial_width * num_nodes,
        };

        VkDescriptorBufferInfo clip_info = { dst_buffer, 0, VK_WHOLE_SIZE };
        VkWriteDescriptorSet clip_write;
        fill_clip_write(&clip_write, &clip_info);

        vkCmdPushDescriptorSetKHR(cmd, VK_PIPELINE_BIND_POINT_COMPUTE,
                                   vk->clip_intermediate.layout, 0,
                                   1, &clip_write);
        vkCmdPushConstants(cmd, vk->clip_intermediate.layout,
                           VK_SHADER_STAGE_COMPUTE_BIT,
                           0, sizeof(clip_push), &clip_push);
        vkCmdBindPipeline(cmd, VK_PIPELINE_BIND_POINT_COMPUTE,
                           vk->clip_intermediate.pipeline);

        uint32_t clip_groups = (clip_push.parameter_count + vk->workgroup_size - 1)
                               / vk->workgroup_size;
        vkCmdDispatch(cmd, clip_groups, 1, 1);

        // Barrier: clip → next stage
        compute_write_read_barrier(cmd, 1, &dst_buffer);

        // Ping-pong: this stage's dst becomes next stage's src
        VkBuffer tmp = src_buffer;
        src_buffer = dst_buffer;
        dst_buffer = tmp;
    }
}
```

The parallelism structure, visualized:

```
Stage 2 (leaves):  ┌────────────────────────────────────────────┐
                   │  vkCmdDispatch(4 nodes)  ← one dispatch    │
                   └────────────────────────────────────────────┘
                   vkCmdPipelineBarrier
                   ┌────────────────────────────────────────────┐
                   │  vkCmdDispatch(clip, 4 nodes)              │
                   └────────────────────────────────────────────┘
                   vkCmdPipelineBarrier

Stage 1 (mid):     ┌───────────────────────┐
                   │  vkCmdDispatch(2 nodes)│
                   └───────────────────────┘
                   vkCmdPipelineBarrier
                   ┌───────────────────────┐
                   │  vkCmdDispatch(clip, 2)│
                   └───────────────────────┘
                   vkCmdPipelineBarrier

Stage 0 (root):    ┌─────────────┐
                   │  dispatch(1) │
                   └─────────────┘
                   barrier
                   ┌─────────────┐
                   │  clip(1)     │
                   └─────────────┘
```

Contrast with the CPU backend: on CPU, each node is a separate task submitted to the thread pool. On Vulkan, all nodes at a stage are a **single dispatch** — the GPU hardware scheduler maps workgroups to compute units. This eliminates the per-node dispatch overhead that would dominate for large reduction trees.

### Subgroup-Accelerated Aggregation Shader

```glsl
#version 450
#extension GL_KHR_shader_subgroup_arithmetic : require

layout(constant_id = 0) const uint SPEC_SIMD_WIDTH = 8;

layout(local_size_x_id = 0) in;

layout(push_constant) uniform PushConstants {
    uint partial_offset_list_count;  // K: number of partials to sum
    uint partial_width;              // Elements per partial
    uint operation_type;             // 0 = SUM, 1 = AVERAGE
} push;

layout(set = 0, binding = 0) readonly buffer PartialCollection {
    float data[];
} src_collection;

layout(set = 0, binding = 1) readonly buffer OffsetList {
    uint data[];
} src_offsets;

layout(set = 0, binding = 2) writeonly buffer DestPartial {
    float data[];
} dest;

shared float cross_subgroup_accum[32];  // For cross-subgroup reduction

void main() {
    // Each workgroup reduces one output element range
    // gl_WorkGroupID.x maps to the output element position
    uint elem = gl_WorkGroupID.x * gl_WorkGroupSize.x
              + gl_LocalInvocationID.x;

    if (elem >= push.partial_width) return;

    // Phase 1: Each invocation accumulates a strided subset of partials
    float accum = 0.0;
    for (uint k = gl_LocalInvocationID.x; k < push.partial_offset_list_count;
         k += gl_WorkGroupSize.x) {
        uint offset = src_offsets.data[k];
        accum += src_collection.data[offset + elem];
    }

    // Phase 2: Workgroup-wide reduction via subgroup ops
    float subgroup_sum = subgroupAdd(accum);

    if (subgroupElect()) {
        cross_subgroup_accum[gl_SubgroupID] = subgroup_sum;
    }
    barrier();

    float total = 0.0;
    if (gl_SubgroupID == 0) {
        float val = (gl_SubgroupInvocationID < gl_NumSubgroups)
                    ? cross_subgroup_accum[gl_SubgroupInvocationID] : 0.0;
        total = subgroupAdd(val);
    }
    barrier();

    if (gl_SubgroupID == 0 && subgroupElect()) {
        cross_subgroup_accum[0] = total;
    }
    barrier();

    total = cross_subgroup_accum[0];

    // Apply averaging if requested
    if (push.operation_type == 1) {
        total /= float(push.partial_offset_list_count);
    }

    // One invocation writes the result
    if (gl_LocalInvocationID.x == 0) {
        dest.data[elem] = total;
    }
}
```

**Note on the register-tier variant:** For small K (≤ subgroup size), the shader simplifies — each invocation loads one partial, `subgroupAdd` completes the full reduction, and no shared memory is needed. The host selects the appropriate tier at dispatch time, matching the architecture's tiered aggregation model.

---

## Host Synchronization & Transfer

### D2H Transfer (Node 23)

The architecture's `D2H Async Copy` maps to a buffer copy command followed by a transfer barrier, recorded as the final Act-phase command:

```c
// At the end of record_act_phase:

// Barrier: compute → transfer (probs buffer)
VkBufferMemoryBarrier2 xfer_barrier = {
    .sType         = VK_STRUCTURE_TYPE_BUFFER_MEMORY_BARRIER_2,
    .srcStageMask  = VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT,
    .srcAccessMask = VK_ACCESS_2_SHADER_WRITE_BIT,
    .dstStageMask  = VK_PIPELINE_STAGE_2_TRANSFER_BIT,
    .dstAccessMask = VK_ACCESS_2_TRANSFER_READ_BIT,
    .buffer        = ctx->buf_final_probs,
    .size          = VK_WHOLE_SIZE,
};

VkDependencyInfo dep = {
    .sType                     = VK_STRUCTURE_TYPE_DEPENDENCY_INFO,
    .bufferMemoryBarrierCount  = 1,
    .pBufferMemoryBarriers     = &xfer_barrier,
};
vkCmdPipelineBarrier2(cmd, &dep);

// Copy to staging buffer
VkBufferCopy region = { 0, 0, ctx->probs_size_bytes };
vkCmdCopyBuffer(cmd, ctx->buf_final_probs, ctx->staging_probs, 1, &region);

// Submit and signal fence
vkEndCommandBuffer(cmd);
VkSubmitInfo submit = { ..., .commandBufferCount = 1, .pCommandBuffers = &cmd };
vkQueueSubmit(vk->compute_queue, 1, &submit, vk->act_fence);

// Host-side: wait and read
vkWaitForFences(vk->device, 1, &vk->act_fence, VK_TRUE, UINT64_MAX);
vkResetFences(vk->device, 1, &vk->act_fence);

// Staging buffer is persistently mapped — direct pointer access
memcpy(host_probs, ctx->staging_mapped_ptr, ctx->probs_size_bytes);
// ← This IS the inference_event
```

The staging buffer's persistent `VMA_ALLOCATION_CREATE_MAPPED_BIT` mapping eliminates per-transfer map/unmap overhead — the host simply reads from the mapped pointer after the fence signals.

---

## The Vulkan Interface Header

```c
// kernels_interface_vk.h
#ifndef KERNELS_INTERFACE_VK_H
#define KERNELS_INTERFACE_VK_H

#include <vulkan/vulkan.h>
#include <vk_mem_alloc.h>
#include <stdint.h>

// ============================================================
// Build Configuration (mirrors System Contract)
// ============================================================
#define LOCAL_MEM_BANK_PADDING      1
#define NUMERICAL_STABILITY_EPSILON 1e-7f
#define PROBLEM_TYPE_CCE            0
#define PROBLEM_TYPE_BCE            1
#define AGG_MODE_SUM                0
#define AGG_MODE_AVERAGE            1

// ============================================================
// Core Structures
// ============================================================

typedef struct {
    VkPipeline          pipeline;
    VkPipelineLayout    layout;
    VkDescriptorSetLayout desc_layout;
} ComputeKernel;

typedef struct {
    VkBuffer       buffer;
    VmaAllocation  allocation;
    VkDeviceSize   size;
} GpuBuffer;

typedef struct {
    VkDevice         device;
    VkQueue          compute_queue;
    uint32_t         queue_family_index;
    VmaAllocator     vma;

    // Hardware profile (queried at init)
    uint32_t         subgroup_size;       // → SPEC_SIMD_WIDTH
    uint32_t         max_workgroup_size;
    VkDeviceSize     max_push_constant_size;

    // ---- Compute Pipelines (one per kernel type × variant) ----

    // Act phase
    ComputeKernel    forward_pass;
    ComputeKernel    render_logits;
    ComputeKernel    cce_probs_loss;
    ComputeKernel    bce_probs_loss;

    // Learn phase — gradient generation
    ComputeKernel    module_param_grads;    // [CCE variant]
    ComputeKernel    module_param_grads_bce;// [BCE variant]
    ComputeKernel    backprop_to_hidden;
    ComputeKernel    backprop_to_hidden_bce;
    ComputeKernel    temp_grads;
    ComputeKernel    temp_grads_bce;

    // Learn phase — clipping & permutation
    ComputeKernel    clip_partial;
    ComputeKernel    gather_permute;

    // Learn phase — reduction engine
    ComputeKernel    aggregate_register;
    ComputeKernel    aggregate_local;
    ComputeKernel    aggregate_register_avg;  // AVG variant
    ComputeKernel    aggregate_local_avg;
    ComputeKernel    clip_intermediate;

    // Learn phase — specialized & streaming
    ComputeKernel    stabilize_reduce_grad_h;
    ComputeKernel    backprop_shared_weights;
    ComputeKernel    backprop_shared_biases;
    ComputeKernel    clip_shared_grads;

    // Learn phase — finalization
    ComputeKernel    normalize_grads;
    ComputeKernel    adam_update;
    ComputeKernel    clamp_temps;

    // ---- Synchronization ----
    VkCommandPool    cmd_pool;
    VkCommandBuffer  act_cmd;
    VkCommandBuffer  learn_cmd;
    VkFence          act_fence;
    VkFence          learn_fence;

    // ---- Staging (D2H) ----
    GpuBuffer        staging_probs;
    void*            staging_probs_mapped;

    // ---- Descriptor Pool ----
    VkDescriptorPool descriptor_pool;

} VulkanBackend;

// ============================================================
// Push Constant Structures (one per kernel family)
// ============================================================

typedef struct {
    uint32_t batch_chunk_offset;
    uint32_t batch_chunk_count;
    uint32_t total_batch_count;
    uint32_t padded_input_count;
    uint32_t padded_hidden_count;
} ForwardPassPush;

typedef struct {
    uint32_t batch_chunk_offset;
    uint32_t batch_chunk_count;
    uint32_t module_chunk_offset;
    uint32_t module_chunk_count;
    uint32_t class_chunk_offset;
    uint32_t class_chunk_count;
    uint32_t total_batch_count;
    uint32_t hidden_count;
    uint32_t padded_hidden_count;
    uint32_t total_output_class_count;
    uint32_t padded_total_output_class_count;
    uint32_t total_modules_count;
} RenderLogitsPush;

typedef struct {
    uint32_t num_class_chunks;
    uint32_t classes_per_chunk;
    uint32_t modules_per_chunk;
    uint32_t total_batch_count;
    uint32_t total_output_class_count;
    uint32_t padded_total_output_class_count;
    uint32_t total_modules_count;
    uint32_t total_tile_count;
} ProbsLossPush;

typedef struct {
    uint32_t batch_chunk_offset;
    uint32_t batch_chunk_count;
    uint32_t num_class_chunks;
    uint32_t classes_per_chunk;
    uint32_t modules_per_chunk;
    uint32_t total_batch_count;
    uint32_t hidden_count;
    uint32_t padded_hidden_count;
    uint32_t total_output_class_count;
    uint32_t padded_total_output_class_count;
    uint32_t total_modules_count;
    uint32_t total_tile_count;
} GradientTilePush;

typedef struct {
    uint32_t use_per_item_norm;
    float    clipping_threshold;
    float    epsilon;
    uint32_t num_class_chunks;
    uint32_t classes_per_chunk;
    uint32_t modules_per_chunk;
    uint32_t total_batch_count;
    uint32_t padded_hidden_count;
    uint32_t total_tile_count;
} ClipPartialsPush;

typedef struct {
    uint32_t total_batch_count;
    uint32_t hidden_count;
    uint32_t padded_hidden_count;
    uint32_t total_modules_count;
    uint32_t padded_total_modules_count;
    uint32_t num_module_chunks;
    uint32_t modules_per_chunk;
    uint32_t num_class_chunks;
    uint32_t total_tile_count;
} GatherPermutePush;

typedef struct {
    uint32_t partial_offset_list_count;
    uint32_t partial_width;
    uint32_t operation_type;
} AggregatePush;

typedef struct {
    float    clipping_threshold;
    float    epsilon;
    uint32_t parameter_count;
} ClipIntermediatePush;

typedef struct {
    float    fp_max;
    float    policy_t_algorithmic;
    float    policy_lambda;
    float    epsilon;
    uint32_t policy_max_k;
    uint32_t total_batch_count;
    uint32_t padded_hidden_count;
    uint32_t total_modules_count;
    uint32_t padded_total_modules_count;
} StabilizeReducePush;

typedef struct {
    uint32_t batch_chunk_offset;
    uint32_t batch_chunk_count;
    uint32_t batch_chunk_index;
    uint32_t total_batch_count;
    uint32_t num_batch_chunks;
    uint32_t padded_input_count;
    uint32_t padded_hidden_count;
    uint32_t final_grad_hidden_activations_total_count;
} SharedBackpropWeightsPush;

typedef struct {
    uint32_t batch_chunk_offset;
    uint32_t batch_chunk_count;
    uint32_t batch_chunk_index;
    uint32_t total_batch_count;
    uint32_t num_batch_chunks;
    uint32_t padded_hidden_count;
    uint32_t final_grad_hidden_activations_total_count;
} SharedBackpropBiasesPush;

typedef struct {
    float    clipping_threshold;
    float    epsilon;
    uint32_t weights_parameter_count;
    uint32_t biases_parameter_count;
    uint32_t weights_write_offset;
    uint32_t biases_write_offset;
    uint32_t num_batch_chunks;
} ClipSharedPush;

// Note: effective_batch_size and epsilon are COMPUTE_TYPE per the kernel contract.
// Precision-specific variants (NormalizePushFP64, NormalizePushFP16) are defined
// in _push_constants.py. The FP32 variant uses float.
typedef struct {
    COMPUTE_TYPE effective_batch_size;
    COMPUTE_TYPE epsilon;
    uint32_t     parameter_count;
} NormalizePush;

typedef struct {
    float    learning_rate;
    float    beta1_pow_t;
    float    beta2_pow_t;
    float    beta1;
    float    beta2;
    float    epsilon;
    uint32_t parameter_count;
} AdamUpdatePush;

typedef struct {
    float    min_value;
    float    max_value;
    uint32_t total_modules_count;
} ClampTempsPush;

// ============================================================
// Reduction Engine Plan
// ============================================================

typedef struct {
    GpuBuffer   staging_buffers[2];
    GpuBuffer*  offset_list_buffers;
    uint32_t*   stage_node_counts;
    uint32_t*   stage_fan_in;
    uint32_t    partial_width;
    uint32_t    num_stages;
    float       t_algorithmic;
    float       lambda;
    float       fp_max;
    float       epsilon;
    GpuBuffer   partial_collection;
    GpuBuffer   output;
} ReductionTreePlan;

// ============================================================
// Backend Lifecycle
// ============================================================

VulkanBackend* vk_backend_create(VkDevice device, VkPhysicalDevice phys_device,
                                  uint32_t queue_family_index,
                                  uint32_t problem_type);
void           vk_backend_destroy(VulkanBackend* vk);

// ============================================================
// Pipeline Creation (called once at initialization)
// ============================================================

ComputeKernel vk_create_compute_kernel(VulkanBackend* vk,
                                        const char* spirv_path,
                                        uint32_t push_constant_size,
                                        uint32_t binding_count,
                                        const VkDescriptorSetLayoutBinding* bindings,
                                        const VkSpecializationInfo* spec_info);

// ============================================================
// Command Recording
// ============================================================

void vk_record_act_phase(VkCommandBuffer cmd, VulkanBackend* vk,
                          PipelineContext* ctx);
void vk_record_learn_phase(VkCommandBuffer cmd, VulkanBackend* vk,
                            PipelineContext* ctx);
void record_reduction_tree(VkCommandBuffer cmd, VulkanBackend* vk,
                            ReductionTreePlan* plan);

// ============================================================
// Helpers
// ============================================================

static inline void bind_and_dispatch(VkCommandBuffer cmd, ComputeKernel* kernel,
                                      VkDescriptorSet desc_set,
                                      const void* push_data, uint32_t push_size,
                                      uint32_t gx, uint32_t gy, uint32_t gz) {
    vkCmdBindPipeline(cmd, VK_PIPELINE_BIND_POINT_COMPUTE, kernel->pipeline);
    vkCmdBindDescriptorSets(cmd, VK_PIPELINE_BIND_POINT_COMPUTE,
                            kernel->layout, 0, 1, &desc_set, 0, NULL);
    vkCmdPushConstants(cmd, kernel->layout, VK_SHADER_STAGE_COMPUTE_BIT,
                       0, push_size, push_data);
    vkCmdDispatch(cmd, gx, gy, gz);
}

static inline void compute_write_read_barrier(VkCommandBuffer cmd) {
    VkMemoryBarrier2 barrier = {
        .sType         = VK_STRUCTURE_TYPE_MEMORY_BARRIER_2,
        .srcStageMask  = VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT,
        .srcAccessMask = VK_ACCESS_2_SHADER_WRITE_BIT,
        .dstStageMask  = VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT,
        .dstAccessMask = VK_ACCESS_2_SHADER_READ_BIT,
    };
    VkDependencyInfo dep = {
        .sType               = VK_STRUCTURE_TYPE_DEPENDENCY_INFO,
        .memoryBarrierCount  = 1,
        .pMemoryBarriers     = &barrier,
    };
    vkCmdPipelineBarrier2(cmd, &dep);
}

#endif // KERNELS_INTERFACE_VK_H
```

---

## Architecture Summary: Vulkan's Position

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
    └───────────┘  └───────────┘  └────────────────┘
```

| Concern                 | OpenCL                        | Vulkan                                          | CPU                                 |
| ----------------------- | ----------------------------- | ----------------------------------------------- | ----------------------------------- |
| Parallelism model       | Implicit (driver schedules)   | **Explicit (host records, GPU executes)**       | Explicit (thread pool)              |
| N tiles dispatch        | N × `clEnqueueNDRange`        | **1 × `vkCmdDispatch(N,1,1)`**                  | 1 × `pool_dispatch(N tasks)`        |
| Sync granularity        | Per-event (coarse)            | **Per-barrier (fine-grained, host-controlled)** | Per-function-return (implicit)      |
| Dead branch elimination | Runtime divergence            | **Specialization constants (compile-time)**     | Compiler optimization               |
| Reduction primitive     | Shared memory + barrier chain | **`subgroupAdd` + minimal shared mem**          | SIMD intrinsics + `simd_reduce_add` |
| Resource binding        | `clSetKernelArg` per arg      | **Descriptor Sets + Push Constants**            | Arg structs                         |
| Memory management       | Driver-managed                | **VMA / explicit allocation**                   | `simd_alloc` (aligned)              |
| D2H transfer            | `clEnqueueReadBuffer`         | **Staging buffer + fence**                      | Direct pointer (shared memory)      |
| Command recording       | Enqueue = execute             | **Record ≠ execute (batch submission)**         | Call = execute                      |

The Vulkan backend's defining property is **batched command recording with explicit synchronization.** Where OpenCL relies on the driver to infer dependencies from queue ordering, and the CPU backend relies on `pool_dispatch_and_wait` returning, Vulkan records the entire DAG into a command buffer with precisely placed barriers. The GPU executes the recorded commands autonomously, with zero host interaction until the fence signals. This makes Vulkan the optimal backend for sustained, high-throughput training workloads where per-dispatch host overhead must be minimized.
