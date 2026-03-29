# Phase 5A: GLSL Compute Shader Library & SPIR-V Build Infrastructure — Detailed Plan

**Status:** Not started  
**Phase:** 5A of 6 (sub-phase A of 3)  
**Objective:** Implement the Vulkan backend's GLSL compute shaders in `src/backends/vulkan/kernel_sources/`, set up the Meson `glslc` compilation pipeline to produce SPIR-V modules, and validate that all shaders compile to valid `.spv` binaries. This sub-phase delivers the compiled shader library; Python-side Vulkan integration is Phase 5B.  
**Governing ADRs:** ADR-013 (kernel source strategy — `kernels.cl.h` as algorithmic reference; three-tier specification hierarchy), ADR-014 (build system — Meson `custom_target` for SPIR-V compilation, `aec_backend_vulkan` feature option), ADR-001 (three-tier jurisdictional model — Execution tier), ADR-007 (KernelContract/KernelBinding split — abstract placement keys replaced by `gl_WorkGroupID`), ADR-011 (CCE/BCE strategy — specialization constant variants)  
**Rollback gate:** All shaders compile via `glslc --target-env=vulkan1.1` with zero errors. SPIR-V validation passes (`spirv-val`). Tier 1 green (no regressions). No behavioral testing — shader correctness deferred to Phase 5C.  
**Dependencies:** Phase 1 (plan model) — **COMPLETE** (plan types define the kernel inventory and parameter contracts). Independent of Phase 2 (OpenCL) and Phase 3 (CPU).

### Relationship to Other Phases

| Phase | Phase 5A interaction |
| :--- | :--- |
| Phase 0 (Foundation) | Phase 5A creates the `src/backends/vulkan/` directory skeleton (first content in the Vulkan backend). |
| Phase 1 (Plan Model) | Phase 5A consumes `KernelContract` definitions to derive push constant layouts and descriptor binding counts. |
| Phase 2 (OpenCL) | Independent. No interaction. |
| Phase 3 (CPU) | Independent. Algorithmic parity ensured via shared `kernels.cl.h` specification. |
| Phase 4 (Test Harness) | Phase 5A has no test dependency. Phase 4 provides Vulkan Tier 2 stubs consumed by Phase 5C. |
| Phase 5B | Phase 5B consumes the `.spv` modules produced here. |
| Phase 5C | Phase 5C validates shader correctness via Tier 2 tests through the renderer built in 5B. |

---

## Table of Contents

1. [Scope & Constraints](#1-scope--constraints)
2. [Vulkan Shader Architecture Overview](#2-vulkan-shader-architecture-overview)
3. [Target Deliverables](#3-target-deliverables)
4. [Task Breakdown](#4-task-breakdown)
   - [Step 5A.1: Create directory skeleton](#step-5a1-create-directory-skeleton)
   - [Step 5A.2: Implement `common.glsl` — shared declarations](#step-5a2-implement-commonglsl--shared-declarations)
   - [Step 5A.3: Implement Act-phase shaders](#step-5a3-implement-act-phase-shaders)
   - [Step 5A.4: Implement Learn-phase gradient production shaders](#step-5a4-implement-learn-phase-gradient-production-shaders)
   - [Step 5A.5: Implement Learn-phase clipping & permutation shaders](#step-5a5-implement-learn-phase-clipping--permutation-shaders)
   - [Step 5A.6: Implement Learn-phase reduction engine shaders](#step-5a6-implement-learn-phase-reduction-engine-shaders)
   - [Step 5A.7: Implement Learn-phase streaming backprop shaders](#step-5a7-implement-learn-phase-streaming-backprop-shaders)
   - [Step 5A.8: Implement Learn-phase finalization shaders](#step-5a8-implement-learn-phase-finalization-shaders)
   - [Step 5A.9: Create `src/backends/vulkan/kernel_sources/meson.build`](#step-5a9-create-srcbackendsvulkankernel_sourcesmesonbuild)
   - [Step 5A.10: Update architecture `meson.build` for Vulkan delegation](#step-5a10-update-architecture-mesonbuild-for-vulkan-delegation)
   - [Step 5A.11: Validate SPIR-V compilation and validation](#step-5a11-validate-spir-v-compilation-and-validation)
   - [Step 5A.12: Validate rollback gate](#step-5a12-validate-rollback-gate)
5. [Specialization Constant Design](#5-specialization-constant-design)
6. [Push Constant Layout Inventory](#6-push-constant-layout-inventory)
7. [Descriptor Binding Inventory](#7-descriptor-binding-inventory)
8. [Subgroup Operation Strategy](#8-subgroup-operation-strategy)
9. [Shader Pipeline Variant Matrix](#9-shader-pipeline-variant-matrix)
10. [Risk Register](#10-risk-register)

---

## 1. Scope & Constraints

### In scope

- Creating the `src/backends/vulkan/` and `src/backends/vulkan/kernel_sources/` directory skeletons with `__init__.py` files.
- Implementing `common.glsl` — the shared include file declaring specialization constants (mapping Contract Article 6 build-time symbols to Vulkan specialization constants), the `workgroup_reduce_add` / `workgroup_reduce_max` subgroup-accelerated helpers, and common type/constant definitions.
- Implementing all GLSL compute shaders (`.comp` files) for the complete kernel inventory from ADR-013, developed against `kernels.cl.h` as the algorithmic reference. Each shader implements a single kernel's computation using Vulkan's execution model:
  - `gl_WorkGroupID.x` replaces host-provided `flat_tile_index` (single-dispatch parallelism).
  - Push constants replace OpenCL scalar arguments.
  - Storage buffer descriptor bindings replace OpenCL `__global` buffer arguments.
  - `shared` qualifier replaces OpenCL `__local` memory.
  - Specialization constants replace OpenCL `-D` preprocessor flags.
  - Subgroup operations (`subgroupAdd`, `subgroupMax`) replace shared-memory barrier-chain reductions.
- Creating CCE/BCE pipeline variants via the `SPEC_PROBLEM_TYPE` specialization constant for kernels that use Strategy A (Nodes 8, 9, 10), eliminating runtime branch divergence (VULKAN_BACKEND.md §Shader Architecture).
- Creating `src/backends/vulkan/kernel_sources/meson.build` with `custom_target()` rules invoking `glslc --target-env=vulkan1.1` for each `.comp` file, producing `.spv` outputs installed as package data.
- Creating `src/backends/vulkan/meson.build` with `subdir('kernel_sources')` and Python module installation for the backend package.
- Updating the architecture's top-level `meson.build` to conditionally delegate to the Vulkan backend via `subdir('src/backends/vulkan')` when `aec_backend_vulkan` is allowed and `glslc` is found (ADR-014).

### Out of scope

- Python-side Vulkan integration (`vulkan-python` context, renderer, buffer allocator, retrieval, discovery) — Phase 5B.
- Vulkan Tier 2 tests — Phase 5C.
- Modifying shared-layer code (plan model, plan builder, contracts).
- Modifying the OpenCL backend or CPU backend.
- Modifying kernel specification files (`kernels/*.cl.c`, `kernels/kernels.cl.h`).
- FP16 shader variants — the Vulkan backend targets FP32 initially. FP16 requires `VK_KHR_shader_float16_int8` and `shaderFloat16` feature, gated by hardware availability.
- The `WorkTicket` / `LearnHandle` / `Engine` user-facing API.

### Key constraint: algorithmic fidelity to `kernels.cl.h`

Every GLSL shader must produce mathematically equivalent results to the algorithm specified in `kernels.cl.h`. The GLSL code implements the same mathematical operation using Vulkan's execution model — subgroup operations, specialization constants, push constants, descriptor bindings — instead of OpenCL's work-item/work-group and preprocessor model. Structural differences are expected and encouraged (single dispatch vs. per-tile enqueue, specialization constants vs. preprocessor macros, `subgroupAdd` vs. shared-memory tree reduction). Numerical differences beyond FP accumulation ordering are bugs.

### Key constraint: single-dispatch parallelism

The defining structural difference from OpenCL: N independent tiles map to a **single** `vkCmdDispatch(N, 1, 1)` call, not N separate enqueues. Inside each shader, `gl_WorkGroupID.x` serves as the tile index. The host does not loop over tiles — the GPU hardware scheduler distributes workgroups. This eliminates per-tile dispatch overhead and is the primary performance advantage of the Vulkan backend.

For 2D decompositions (e.g., `forward_pass` across batch samples × hidden blocks), the dispatch uses two group dimensions:

```c
vkCmdDispatch(cmd, batch_chunk_count, padded_hidden / SIMD_WIDTH, 1);
// gl_WorkGroupID.x → sample index
// gl_WorkGroupID.y → hidden block index
```

### Key constraint: subgroup operations as first-class primitive

Vulkan's `VK_KHR_shader_subgroup` (core since Vulkan 1.1) provides `subgroupAdd`, `subgroupMax`, `subgroupBroadcast`, and related operations. These replace **all** shared-memory tree reductions from the OpenCL kernel set:

| OpenCL pattern | Vulkan replacement |
| :--- | :--- |
| `local[lid] = val; barrier(); for (s = size/2; ...) { if (lid < s) local[lid] += local[lid+s]; barrier(); }` | `subgroupAdd(val)` (intra-subgroup) + minimal shared-memory cross-subgroup bridge |

For workgroups larger than one subgroup, a two-level reduction strategy is used: (1) `subgroupAdd` within each subgroup, (2) shared-memory exchange between subgroups using `subgroupElect()` to write representatives, (3) `subgroupAdd` in the first subgroup to produce the final result. This is detailed in `common.glsl`.

### Key constraint: specialization constants for CCE/BCE branching

Kernels using Strategy A (host-injected flag) for CCE/BCE selection — `calculate_module_param_grads` (Node 8), `backprop_error_to_hidden` (Node 9), `calculate_temp_gradients` (Node 10) — use the `SPEC_PROBLEM_TYPE` specialization constant rather than a runtime flag. The Vulkan compiler eliminates the dead branch at pipeline creation time, producing distinct pipeline objects for CCE and BCE. This costs one extra pipeline per variant kernel but eliminates all divergent branching on the GPU.

---

## 2. Vulkan Shader Architecture Overview

### Execution Model Mapping

| Architecture Concept | Vulkan Shader Manifestation |
| :--- | :--- |
| Kernel | GLSL compute shader (`.comp` → `.spv`) compiled to `VkPipeline` |
| N parallel tiles | Single `vkCmdDispatch(N, 1, 1)` — `gl_WorkGroupID.x` = tile index |
| Work-group | Workgroup (`layout(local_size_x = ...) in;`) |
| Work-items | Invocations within a workgroup |
| Subgroup/Wavefront | Subgroup (`GL_KHR_shader_subgroup_arithmetic`) |
| `__local` memory | `shared` storage qualifier |
| `__global` buffer | Storage Buffer (SSBO) via `layout(set=0, binding=N)` |
| `__constant` buffer | Uniform Buffer (UBO) or `readonly` SSBO |
| `-D` compile flags | Specialization constants (`layout(constant_id=N)`) |
| Scalar kernel args | Push constants (`layout(push_constant) uniform`) |
| Buffer kernel args | Descriptor set bindings |

### Shader File Organization

```
src/backends/vulkan/kernel_sources/
├── common.glsl                          # Shared: spec constants, subgroup helpers, types
├── forward_pass.comp                    # Node 4
├── render_logits.comp                   # Node 5
├── compute_probs_loss_cce.comp          # Node 6 (Strategy B — separate shader)
├── compute_probs_loss_bce.comp          # Node 7 (Strategy B — separate shader)
├── calculate_module_param_grads.comp    # Node 8 (Strategy A — SPEC_PROBLEM_TYPE)
├── backprop_error_to_hidden.comp        # Node 9 (Strategy A — SPEC_PROBLEM_TYPE)
├── calculate_temp_gradients.comp        # Node 10 (Strategy A — SPEC_PROBLEM_TYPE)
├── clip_partial_gradients.comp          # Node 11
├── gather_and_permute.comp              # Node 13
├── aggregate_partials.comp              # Nodes 14/15a/20a (tiered: register + local)
├── clip_intermediate_grad.comp          # Nodes 15b/20b
├── stabilize_reduce_grad_h.comp         # Node 16
├── backprop_shared_weights.comp         # Node 17
├── backprop_shared_biases.comp          # Node 18
├── clip_shared_gradients.comp           # Node 19
├── normalize_gradients.comp             # Node 21
├── adam_update.comp                     # Node 24
└── clamp_temperatures.comp              # Node 25
```

Total: 1 shared include + 18 compute shaders → ~22 pipeline variants (including CCE/BCE and register/local tier splits).

---

## 3. Target Deliverables

| Deliverable | Location | Description |
| :--- | :--- | :--- |
| `common.glsl` | `src/backends/vulkan/kernel_sources/` | Shared declarations: specialization constants, `workgroup_reduce_add`, `workgroup_reduce_max`, type definitions, constant aliases |
| 18 `.comp` files | `src/backends/vulkan/kernel_sources/` | Complete kernel inventory as GLSL compute shaders |
| 18 `.spv` files | `builddir/src/backends/vulkan/kernel_sources/` (build output) | Compiled SPIR-V modules |
| `meson.build` (kernel_sources) | `src/backends/vulkan/kernel_sources/` | `custom_target()` rules for `glslc` compilation |
| `meson.build` (vulkan) | `src/backends/vulkan/` | Sub-directory delegation + Python package installation |
| `__init__.py` files | `src/backends/vulkan/`, `src/backends/vulkan/kernel_sources/` | Package markers for `importlib.resources` discovery |

---

## 4. Task Breakdown

### Step 5A.1: Create directory skeleton

Create the Vulkan backend directory structure:

```
src/backends/vulkan/
├── __init__.py                    # Empty initially (populated in Phase 5B)
└── kernel_sources/
    ├── __init__.py                # Package marker for importlib.resources
    └── (shader files follow)
```

**Acceptance criteria:**
- Directories exist with `__init__.py` files.
- No import errors from the parent package.
- Tier 1 green.

---

### Step 5A.2: Implement `common.glsl` — shared declarations

Create the shared GLSL include file consumed by all compute shaders via `#include "common.glsl"` (requires `glslc -I` include path).

**Contents:**

```glsl
#ifndef COMMON_GLSL
#define COMMON_GLSL

#extension GL_KHR_shader_subgroup_arithmetic : require
#extension GL_KHR_shader_subgroup_ballot     : require

// ── Specialization Constants (Contract Article 6) ──
layout(constant_id = 0) const uint SPEC_SIMD_WIDTH              = 8;
layout(constant_id = 1) const uint SPEC_LOCAL_MEM_BANK_PADDING   = 1;
layout(constant_id = 2) const uint SPEC_C_TILE_SIZE              = 8;
layout(constant_id = 3) const uint SPEC_PROBLEM_TYPE             = 0;  // 0=CCE, 1=BCE

// ── Derived Constants ──
const float NUMERICAL_STABILITY_EPSILON = 1e-7;
const uint  PROBLEM_TYPE_CCE = 0;
const uint  PROBLEM_TYPE_BCE = 1;

// ── Subgroup-Accelerated Workgroup Reduction ──
shared float _cross_subgroup_scratch[32];

float workgroup_reduce_add(float value) { ... }
float workgroup_reduce_max(float value) { ... }

#endif
```

The `workgroup_reduce_add` function implements the three-level pattern from VULKAN_BACKEND.md:
1. **Intra-subgroup:** `subgroupAdd(value)` — single instruction, zero barriers.
2. **Cross-subgroup exchange:** `subgroupElect()` writes to shared memory; barrier.
3. **Final reduction:** First subgroup reduces representatives via `subgroupAdd`; broadcast result.

`workgroup_reduce_max` follows the same structure using `subgroupMax`.

**Acceptance criteria:**
- `common.glsl` parses without errors when included by any `.comp` shader.
- Specialization constant IDs are stable and documented.
- Reduction helpers are correct for workgroups containing 1–32 subgroups.

---

### Step 5A.3: Implement Act-phase shaders

Implement the four Act-phase compute shaders, developed against `kernels.cl.h` §Phase 1 Act:

| Shader | DAG Node | Dispatch Geometry | Key Features |
| :--- | :--- | :--- | :--- |
| `forward_pass.comp` | 4 | `(batch_chunks, hidden_blocks, 1)` | Tiled dot product with shared-memory weight tile; ReLU + sample mask; coalesced weight loads |
| `render_logits.comp` | 5 | `(tile_count, 1, 1)` | Per-module logits computation; weight × hidden dot product |
| `compute_probs_loss_cce.comp` | 6 | `(tile_count, 1, 1)` | Softmax with numerical stability (workgroup max + sum via subgroup ops); CCE loss; Strategy B — separate shader |
| `compute_probs_loss_bce.comp` | 7 | `(tile_count, 1, 1)` | Sigmoid + BCE loss; Strategy B — separate shader |

**Implementation notes:**

- `forward_pass.comp` uses 2D dispatch: `gl_WorkGroupID.x` = sample index, `gl_WorkGroupID.y` = hidden block index. The shared-memory tile follows the bank-conflict-avoidance layout from VULKAN_BACKEND.md (`shared float simd_tile[SPEC_SIMD_WIDTH][SPEC_SIMD_WIDTH + SPEC_LOCAL_MEM_BANK_PADDING]`).
- `compute_probs_loss_cce.comp` requires a full workgroup reduction for the numerically stable softmax: `max_logit = workgroup_reduce_max(logit)`, then `sum_exp = workgroup_reduce_add(exp(logit - max_logit))`. This is the primary beneficiary of subgroup acceleration.
- Nodes 6 and 7 are Strategy B (separate shaders), not Strategy A (flag-toggled). No specialization constant branching needed.

**Acceptance criteria:**
- All four shaders compile to `.spv` via `glslc --target-env=vulkan1.1`.
- Push constant layouts match the structures in VULKAN_BACKEND.md §Push Constant Design.
- Descriptor binding counts match the buffer parameter lists in `KernelContract` definitions.
- `spirv-val` reports no errors.

---

### Step 5A.4: Implement Learn-phase gradient production shaders

Implement the three gradient production shaders — these use Strategy A (specialization constant branching):

| Shader | DAG Node | Strategy | Key Features |
| :--- | :--- | :--- | :--- |
| `calculate_module_param_grads.comp` | 8 | A (`SPEC_PROBLEM_TYPE`) | Outer product accumulation; writes to partial grad collection; `Placement Contract: grid_mod_cls(gl_WorkGroupID.x)` |
| `backprop_error_to_hidden.comp` | 9 | A (`SPEC_PROBLEM_TYPE`) | Transposed matmul; partial grad_h production |
| `calculate_temp_gradients.comp` | 10 | A (`SPEC_PROBLEM_TYPE`) | Temperature gradient computation; workgroup reduction for per-module temperature sums |

**Implementation notes:**

- All three shaders branch on `SPEC_PROBLEM_TYPE` to select CCE or BCE gradient formulas. The specialization constant ensures the dead branch is eliminated at pipeline creation time.
- Node 8 writes scattered partials governed by the `grid_mod_cls` Placement Contract. In Vulkan, `gl_WorkGroupID.x` replaces `flat_tile_index`. The shader decomposes `gl_WorkGroupID.x` into `(module_chunk, class_chunk)` using `gl_WorkGroupID.x / num_class_chunks` and `gl_WorkGroupID.x % num_class_chunks`, matching the OpenCL `grid_mod_cls` strategy.
- DAG concurrency: Nodes 8, 9, 10 read the same inputs and write to disjoint output buffers. The renderer (Phase 5B) records all three dispatches without intermediate barriers.

**Acceptance criteria:**
- All three shaders compile for both `SPEC_PROBLEM_TYPE=0` (CCE) and `SPEC_PROBLEM_TYPE=1` (BCE).
- Push constant sizes fit within the guaranteed 128-byte minimum.
- `spirv-val` reports no errors for both variants.

---

### Step 5A.5: Implement Learn-phase clipping & permutation shaders

| Shader | DAG Node | Key Features |
| :--- | :--- | :--- |
| `clip_partial_gradients.comp` | 11 | Partial-Group-Wise clipping: L2 norm across all partial gradient groups via `workgroup_reduce_add(sq_sum)`; single scaling factor; uses `SPEC_PROBLEM_TYPE` for per-item norm toggle |
| `gather_and_permute.comp` | 13 | Global barrier kernel — gathers scattered AoS partial grad_h into contiguous SoA layout for downstream reduction; reads from all tiles |

**Implementation notes:**

- Node 11 (`clip_partial_gradients`) is the first major consumer of the subgroup-accelerated L2 norm reduction: compute per-invocation squared sums, `workgroup_reduce_add` to get group norm, scale all elements.
- Node 13 (`gather_and_permute`) is a Global Barrier in the DAG — it reads outputs from all upstream tiles. Dispatch geometry is based on the total element count, not the tile count.

**Acceptance criteria:**
- Both shaders compile and pass `spirv-val`.
- `clip_partial_gradients.comp` handles both `use_per_item_norm=0` and `use_per_item_norm=1` paths.

---

### Step 5A.6: Implement Learn-phase reduction engine shaders

| Shader | DAG Nodes | Key Features |
| :--- | :--- | :--- |
| `aggregate_partials.comp` | 14/15a/20a | Tiered aggregation: register-reduce (K ≤ subgroup size, no shared memory) and local-reduce (K > subgroup size, cross-subgroup bridge); SUM and AVERAGE modes via push constant; indirection-based source addressing via offset list |
| `clip_intermediate_grad.comp` | 15b/20b | Per-stage clipping with threshold from host push constant; element-wise L2 norm + scale |

**Implementation notes:**

- `aggregate_partials.comp` is dispatched many times per batch with different source/destination buffers at each reduction stage. The renderer uses `VK_KHR_push_descriptor` to update bindings inline in the command buffer, avoiding pre-allocated descriptor set management.
- Two pipeline variants exist for `aggregate_partials.comp` — differing only in local work size and reduction path. Tier selection (register vs. local) is a host-side decision based on the stage's fan-in K relative to the device's subgroup size.
- `clip_intermediate_grad.comp` is simpler — it receives the clipping threshold as a push constant (pre-computed by the host from the Quadratic Scaling Policy: $T_j = T_{\text{algorithmic}} + \lambda \cdot j^2$, clamped by $T_{\text{safety},j} = \text{FP\_FORMAT\_MAX} / K_j$).

**Acceptance criteria:**
- `aggregate_partials.comp` compiles for both SUM and AVERAGE operation types.
- Offset list addressing is correct for arbitrary fan-in K values.
- `clip_intermediate_grad.comp` compiles and passes `spirv-val`.

---

### Step 5A.7: Implement Learn-phase streaming backprop shaders

| Shader | DAG Node | Key Features |
| :--- | :--- | :--- |
| `stabilize_reduce_grad_h.comp` | 16 | Specialized Grad_H reduction — internal multi-stage reduction within a single dispatch; exploits contiguous input guarantee from Node 13 (ADR-005); subgroup-native staged reduction |
| `backprop_shared_weights.comp` | 17 | Outer product: per-chunk shared weight gradients; dispatched per streaming chunk |
| `backprop_shared_biases.comp` | 18 | Per-chunk shared bias gradients; reduction along batch dimension |
| `clip_shared_gradients.comp` | 19 | Clip shared weight and bias gradients per chunk; L2 norm + scale |

**Implementation notes:**

- Node 16 (`stabilize_reduce_grad_h`) is the most complex shader. Per ADR-005 (Node 16 opacity), this kernel performs its own internal multi-stage reduction. The plan exposes it as a single `KernelDispatchNode` with policy parameters (thresholds, Grad_H dimensions) as push constants. The shader implements the full stabilized reduction internally using subgroup operations.
- Nodes 17, 18, 19 are dispatched iteratively in a streaming loop. The host re-records push constants (specifically `batch_chunk_index`) before each dispatch via `vkCmdPushConstants`. The dispatch-barrier-dispatch pattern within the streaming loop is: `dispatch(17) + dispatch(18) → barrier → dispatch(19) → barrier → next chunk`.

**Acceptance criteria:**
- All four shaders compile and pass `spirv-val`.
- `stabilize_reduce_grad_h.comp` handles workloads with varying `total_modules_count`, `padded_hidden_count`, and `total_batch_count`.
- Streaming shaders accept per-chunk push constant updates.

---

### Step 5A.8: Implement Learn-phase finalization shaders

| Shader | DAG Node | Key Features |
| :--- | :--- | :--- |
| `normalize_gradients.comp` | 21 | Element-wise gradient normalization by effective batch size |
| `adam_update.comp` | 24 | Adam optimizer: first/second moment updates, bias-corrected parameter step |
| `clamp_temperatures.comp` | 25 | Element-wise temperature clamping to `[min, max]` range |

**Implementation notes:**

- These are element-wise or streaming kernels with minimal control flow. They represent the simplest shaders in the inventory.
- `adam_update.comp` receives `beta1_pow_t` and `beta2_pow_t` as push constants — these are computed on the host in FP64 per ADR (Architectural Hierarchy: Host computes `beta1**t` using FP64 regardless of `SCALAR_TYPE`).
- Nodes 21 and 24 are dispatched per parameter group (weights, biases, temperatures, shared weights, shared biases) with different buffer bindings. Descriptor updates via `vkUpdateDescriptorSets` or push descriptors.

**Acceptance criteria:**
- All three shaders compile and pass `spirv-val`.
- Push constant layouts for `adam_update.comp` accommodate all required optimizer parameters.

---

### Step 5A.9: Create `src/backends/vulkan/kernel_sources/meson.build`

Define `custom_target()` rules for SPIR-V compilation:

```meson
glslc = find_program('glslc')

shader_sources = files(
    'forward_pass.comp',
    'render_logits.comp',
    'compute_probs_loss_cce.comp',
    'compute_probs_loss_bce.comp',
    'calculate_module_param_grads.comp',
    'backprop_error_to_hidden.comp',
    'calculate_temp_gradients.comp',
    'clip_partial_gradients.comp',
    'gather_and_permute.comp',
    'aggregate_partials.comp',
    'clip_intermediate_grad.comp',
    'stabilize_reduce_grad_h.comp',
    'backprop_shared_weights.comp',
    'backprop_shared_biases.comp',
    'clip_shared_gradients.comp',
    'normalize_gradients.comp',
    'adam_update.comp',
    'clamp_temperatures.comp',
)

spirv_targets = []
foreach shader : shader_sources
    name = '@BASENAME@'
    spirv_targets += custom_target(
        name + '.spv',
        input: shader,
        output: name + '.spv',
        command: [glslc,
                  '--target-env=vulkan1.1',
                  '-I', meson.current_source_dir(),
                  '-o', '@OUTPUT@',
                  '@INPUT@'],
        install: true,
        install_dir: py.get_install_dir() / 'averaging_ensembled_classifier' / 'backends' / 'vulkan' / 'kernel_sources',
    )
endforeach
```

**Key decisions:**
- `--target-env=vulkan1.1` ensures subgroup operations are available (core since Vulkan 1.1).
- `-I meson.current_source_dir()` allows `#include "common.glsl"` from compute shaders.
- SPIR-V modules are installed alongside the Python package via `py.get_install_dir()`, discoverable via `importlib.resources`.

**Acceptance criteria:**
- `ninja -C builddir` compiles all shaders when `aec_backend_vulkan` is allowed.
- `.spv` files appear in the build directory.
- `ninja install` places `.spv` files in the correct package data location.

---

### Step 5A.10: Update architecture `meson.build` for Vulkan delegation

The existing `meson.build` already contains the conditional delegation structure (DESIGN.md §8.1):

```meson
if backend_vulkan.allowed() and glslc.found()
  subdir('src/backends/vulkan')
endif
```

Create `src/backends/vulkan/meson.build` to delegate to the kernel sources sub-directory and install the Python backend package:

```meson
subdir('kernel_sources')

py.install_sources(
    '__init__.py',
    subdir: 'averaging_ensembled_classifier' / 'backends' / 'vulkan',
)
```

Additional Python sources will be added in Phase 5B as they are created.

**Acceptance criteria:**
- The top-level build correctly skips Vulkan delegation when `glslc` is absent.
- The top-level build correctly processes Vulkan delegation when `glslc` is present and `aec_backend_vulkan` is allowed.
- `_build_config.py` correctly reports `BACKEND_VULKAN = True/False` based on the build configuration.

---

### Step 5A.11: Validate SPIR-V compilation and validation

Run the full build and validation pipeline:

1. `meson setup builddir --reconfigure` (or fresh setup with Vulkan enabled).
2. `ninja -C builddir` — all 18 shaders compile to `.spv` with zero `glslc` errors.
3. `spirv-val` on every `.spv` output — zero validation errors.
4. Inspect `.spv` sizes for sanity (non-zero, reasonable range).

**Acceptance criteria:**
- Zero compilation errors.
- Zero SPIR-V validation errors.
- All 18 `.spv` files are non-empty.

---

### Step 5A.12: Validate rollback gate

**Gate condition:** All shaders compile. `spirv-val` passes. Tier 1 green.

Run `pytest tests/tier1/ -v` to confirm no regressions from directory structure changes or `meson.build` modifications.

**Acceptance criteria:**
- Tier 1 unchanged (140+ tests pass).
- `_build_config.py` correctly reflects Vulkan availability.

---

## 5. Specialization Constant Design

| Constant ID | Name | Type | Default | Contract Article 6 Symbol |
| :--- | :--- | :--- | :--- | :--- |
| 0 | `SPEC_SIMD_WIDTH` | `uint` | 8 | `SIMD_WIDTH` |
| 1 | `SPEC_LOCAL_MEM_BANK_PADDING` | `uint` | 1 | `LOCAL_MEM_BANK_PADDING` (Article 5) |
| 2 | `SPEC_C_TILE_SIZE` | `uint` | 8 | `C_TILE_SIZE` |
| 3 | `SPEC_PROBLEM_TYPE` | `uint` | 0 | `SCALAR_IS_HALF` repurposed¹ |

¹ `SPEC_PROBLEM_TYPE` encodes CCE (0) vs BCE (1), replacing the runtime `FLAG__problem_type` scalar. `SCALAR_TYPE` and `SCALAR_IS_HALF` are not directly needed — Vulkan shaders operate on `float` at the GLSL type level; FP16 would require `float16_t` via the `VK_KHR_shader_float16_int8` extension (future work).

**Host-side specialization at pipeline creation:**

```c
VkSpecializationMapEntry entries[] = {
    {0, offsetof(SpecConstants, simd_width),      sizeof(uint32_t)},
    {1, offsetof(SpecConstants, bank_padding),     sizeof(uint32_t)},
    {2, offsetof(SpecConstants, tile_size),         sizeof(uint32_t)},
    {3, offsetof(SpecConstants, problem_type),      sizeof(uint32_t)},
};
```

`SPEC_SIMD_WIDTH` is populated from the device's `subgroupSize` property (queried via `VkPhysicalDeviceSubgroupProperties`). This ensures the shader's `layout(local_size_x_id = 0) in;` directive sets the workgroup width to match the hardware's subgroup width.

---

## 6. Push Constant Layout Inventory

Every push constant struct fits within the Vulkan-guaranteed 128-byte minimum:

| Shader | Push Constant Struct | Size (bytes) | Fields |
| :--- | :--- | :--- | :--- |
| `forward_pass` | `ForwardPassPush` | 20 | `batch_chunk_offset`, `batch_chunk_count`, `total_batch_count`, `padded_input_count`, `padded_hidden_count` |
| `render_logits` | `RenderLogitsPush` | 48 | Module/class chunk offsets and counts, batch/hidden/class dimensions |
| `compute_probs_loss_cce/bce` | `ProbsLossPush` | 32 | `num_class_chunks`, `classes_per_chunk`, `modules_per_chunk`, batch/class dimensions, `total_tile_count` |
| `calculate_module_param_grads` | `GradientTilePush` | 40 | Chunk dimensions, batch/hidden/class/module counts, `total_tile_count` |
| `backprop_error_to_hidden` | `GradientTilePush` | 40 | (shares layout with Node 8) |
| `calculate_temp_gradients` | `GradientTilePush` | 40 | (shares layout with Node 8) |
| `clip_partial_gradients` | `ClipPartialsPush` | 36 | `use_per_item_norm`, `clipping_threshold`, `epsilon`, chunk dimensions |
| `gather_and_permute` | `GatherPermutePush` | 36 | Batch/hidden/module dimensions, chunk counts |
| `aggregate_partials` | `AggregatePush` | 12 | `partial_offset_list_count`, `partial_width`, `operation_type` |
| `clip_intermediate_grad` | `ClipIntermediatePush` | 12 | `clipping_threshold`, `epsilon`, `parameter_count` |
| `stabilize_reduce_grad_h` | `StabilizeReducePush` | 36 | Threshold policy params, dimensions |
| `backprop_shared_weights` | `SharedBackpropWeightsPush` | 32 | Chunk indices, batch/dimension counts |
| `backprop_shared_biases` | `SharedBackpropBiasesPush` | 28 | Chunk indices, batch/dimension counts |
| `clip_shared_gradients` | `ClipSharedPush` | 28 | `clipping_threshold`, `epsilon`, parameter counts, offsets |
| `normalize_gradients` | `NormalizePush` | 12 | `effective_batch_size`, `epsilon`, `parameter_count` |
| `adam_update` | `AdamUpdatePush` | 28 | `learning_rate`, `beta1/2_pow_t`, `beta1/2`, `epsilon`, `parameter_count` |
| `clamp_temperatures` | `ClampTempsPush` | 12 | `min_value`, `max_value`, `total_modules_count` |

All structures are ≤48 bytes — well within the 128-byte guarantee.

---

## 7. Descriptor Binding Inventory

| Shader | Binding Count | Buffer Roles |
| :--- | :--- | :--- |
| `forward_pass` | 6 | input, sample_mask, weights, biases, hidden_out, hidden_mask_out |
| `render_logits` | 5 | hidden, hidden_mask, weights, biases, logits_out |
| `compute_probs_loss_cce` | 4 | logits, targets, probs_out, loss_out |
| `compute_probs_loss_bce` | 4 | logits, targets, probs_out, loss_out |
| `calculate_module_param_grads` | 6 | hidden, hidden_mask, probs, targets, grad_mod_w_out, grad_mod_b_out |
| `backprop_error_to_hidden` | 5 | probs, targets, weights, grad_h_out, hidden_mask |
| `calculate_temp_gradients` | 5 | logits, probs, targets, grad_temps_out, module_temps |
| `clip_partial_gradients` | 10 | 4 partial grad inputs, 4 clipped outputs, offset_list, scratch |
| `gather_and_permute` | 4 | clipped_grad_h_aos, permuted_grad_h_soa, clipped_grad_mod_w, clipped_grad_temps |
| `aggregate_partials` | 3 | src_collection, offset_list, dest_partial |
| `clip_intermediate_grad` | 1 | grad_buffer (in-place or read→write) |
| `stabilize_reduce_grad_h` | 3 | permuted_grad_h_soa, final_grad_h_out, scratch |
| `backprop_shared_weights` | 4 | final_grad_h, hidden, partial_grad_sw_out, sample_mask |
| `backprop_shared_biases` | 3 | final_grad_h, partial_grad_sb_out, sample_mask |
| `clip_shared_gradients` | 4 | partial_grad_sw, partial_grad_sb, clipped_grad_sw_out, clipped_grad_sb_out |
| `normalize_gradients` | 1 | grad_buffer (in-place) |
| `adam_update` | 4 | params, grads, m1, m2 |
| `clamp_temperatures` | 1 | temperatures (in-place) |

---

## 8. Subgroup Operation Strategy

| Subgroup Operation | GLSL Extension | Kernels Using It |
| :--- | :--- | :--- |
| `subgroupAdd(float)` | `GL_KHR_shader_subgroup_arithmetic` | All reduction-bearing kernels: Nodes 6, 10, 11, 13¹, 14/15a/20a, 15b/20b, 16, 18, 19 |
| `subgroupMax(float)` | `GL_KHR_shader_subgroup_arithmetic` | Node 6 (softmax numerically stable max) |
| `subgroupElect()` | `GL_KHR_shader_subgroup_ballot` | Cross-subgroup bridge (all reduction kernels using `workgroup_reduce_add/max`) |
| `subgroupBroadcast(float, uint)` | `GL_KHR_shader_subgroup_ballot` | Optional optimization for broadcasting reduction results |

¹ Node 13 (`gather_and_permute`) may use subgroup operations for coordinated scatter patterns, depending on implementation.

Both `GL_KHR_shader_subgroup_arithmetic` and `GL_KHR_shader_subgroup_ballot` are core since Vulkan 1.1 — no optional extension negotiation required.

---

## 9. Shader Pipeline Variant Matrix

| Shader | CCE Pipeline | BCE Pipeline | Register Tier | Local Tier | SUM Mode | AVG Mode | Total Pipelines |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| `forward_pass` | — | — | — | — | — | — | 1 |
| `render_logits` | — | — | — | — | — | — | 1 |
| `compute_probs_loss_cce` | ✓ | — | — | — | — | — | 1 |
| `compute_probs_loss_bce` | — | ✓ | — | — | — | — | 1 |
| `calculate_module_param_grads` | ✓ | ✓ | — | — | — | — | 2 |
| `backprop_error_to_hidden` | ✓ | ✓ | — | — | — | — | 2 |
| `calculate_temp_gradients` | ✓ | ✓ | — | — | — | — | 2 |
| `clip_partial_gradients` | — | — | — | — | — | — | 1 |
| `gather_and_permute` | — | — | — | — | — | — | 1 |
| `aggregate_partials` | — | — | ✓ | ✓ | ✓ | ✓ | 4 |
| `clip_intermediate_grad` | — | — | — | — | — | — | 1 |
| `stabilize_reduce_grad_h` | — | — | — | — | — | — | 1 |
| `backprop_shared_weights` | — | — | — | — | — | — | 1 |
| `backprop_shared_biases` | — | — | — | — | — | — | 1 |
| `clip_shared_gradients` | — | — | — | — | — | — | 1 |
| `normalize_gradients` | — | — | — | — | — | — | 1 |
| `adam_update` | — | — | — | — | — | — | 1 |
| `clamp_temperatures` | — | — | — | — | — | — | 1 |
| **Total** | | | | | | | **~22** |

Pipeline creation is a one-time cost at backend initialization. ~22 pipelines is trivial.

---

## 10. Risk Register

| Risk | Likelihood | Impact | Mitigation |
| :--- | :--- | :--- | :--- |
| `glslc` not available on CI runners | Medium | Build: Vulkan shaders not compiled | `aec_backend_vulkan` defaults to `auto` — graceful skip when `glslc` absent. CI adds `glslc` to its toolchain image. |
| Subgroup size varies across GPU vendors (16 on NVIDIA, 32/64 on AMD) | High | Correctness: workgroup reduction assumes specific subgroup count | `common.glsl` uses `gl_NumSubgroups` and `gl_SubgroupSize` for dynamic sizing. Shared-memory scratch array sized to 32 (maximum subgroups per workgroup). |
| Push constant size exceeds 128-byte minimum on exotic hardware | Very Low | Dispatch: pipeline creation failure | All structs are ≤48 bytes. Even the most constrained hardware provides ≥128 bytes. |
| Bank conflict patterns differ between GPU architectures | Medium | Performance: not correctness | Bank padding via `SPEC_LOCAL_MEM_BANK_PADDING` (specialization constant) allows tuning per device without shader recompilation. |
| `#include "common.glsl"` not supported by all `glslc` versions | Low | Build: include resolution failure | `glslc` (from the Vulkan SDK's `shaderc`) has supported `#include` since 2017. Mitigation: document minimum `glslc` version in build requirements. |
| SPIR-V validation divergence from runtime behavior | Low | Correctness: shader that validates may still fail on a specific driver | Phase 5C Tier 2 tests catch runtime correctness issues. `spirv-val` is a necessary but not sufficient gate. |
| FP32 accumulation order differences vs. OpenCL/CPU | High | Parity: Tier 3 tests may show cross-backend numerical divergence | Expected. Tier 3 tolerance configuration (Phase 4) accommodates implementation-specific accumulation ordering. Not a Phase 5A risk — it surfaces in Phase 5C. |
