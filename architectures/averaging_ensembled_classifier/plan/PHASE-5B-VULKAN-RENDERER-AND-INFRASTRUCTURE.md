# Phase 5B: Vulkan Python Integration Layer & VulkanPlanRenderer — Detailed Plan

**Status:** Not started  
**Phase:** 5B of 6 (sub-phase B of 3)  
**Objective:** Implement the Python-side integration layer for the Vulkan backend — `vulkan-python` device context management, hardware discovery (`HardwareProfile` population), buffer allocation via VMA-style management, descriptor set and push constant marshalling, command buffer recording, the `VulkanPlanRenderer` that traverses an `ExecutionPlan` DAG and records dispatches into Vulkan command buffers, and the `VulkanRetrievalFuture` using fence-gated staging buffer readback. After this sub-phase, the Vulkan backend can fully render any `ExecutionPlan` produced by the plan builder.  
**Governing ADRs:** ADR-001 (backend abstraction boundary — `PlanRenderer` Protocol), ADR-006 (HardwareProfile population — subgroup size, shared memory limits), ADR-009 (buffer lifecycle — device-local allocation, staging buffer), ADR-010 (RetrievalFuture — fence-gated D2H transfer with padding-aware unpadding), ADR-012 (module factoring — `src/backends/vulkan/` structure), ADR-014 (build system — `_build_config.py` update, `importlib.resources` discovery of `.spv` modules), ADR-015 (interop — `vulkan-python` typed bindings)  
**Rollback gate:** Tier 1 green (no regressions) + `vulkan-python` context initializes successfully on available hardware + SPIR-V modules load into compute pipelines + structural smoke tests pass (renderer can traverse a simple plan without failing). Full Vulkan Tier 2 gate deferred to Phase 5C.  
**Dependencies:** Phase 5A (shader library) — provides compiled `.spv` modules. Phase 1 (plan model) — provides `ExecutionPlan`, `PlanRenderer` Protocol, and all shared-layer types.

### Phase 5A Deliverables Consumed Here

Phase 5B is the Python consumer of the Phase 5A shader library. The following artifacts cross the boundary:

| Build Artifact | Python-Side Consumer |
| :--- | :--- |
| `*.spv` SPIR-V modules (18 shaders) | `_pipeline_cache.py` → `vkCreateComputePipelines` |
| Specialization constant layout (IDs 0–3) | Pipeline creation with `VkSpecializationInfo` |
| Push constant struct sizes | `vkCmdPushConstants` calls during command recording |
| Descriptor binding counts per shader | `VkDescriptorSetLayout` creation |

### Phase 1 Deliverables Consumed Here

The `VulkanPlanRenderer` consumes the same shared-layer types as the OpenCL (Phase 2A) and CPU (Phase 3B) renderers:

| Shared Type | Phase 5B Consumption |
| :--- | :--- |
| `ExecutionPlan` | Input to `VulkanPlanRenderer.render()` |
| `KernelDispatchNode` | Recorded as `vkCmdDispatch` into command buffer via pipeline + descriptor set + push constants |
| `ReductionTreeNode` | Recorded as staged dispatch-barrier-dispatch sequence via `_record_reduction_tree()` |
| `StreamingLoopNode` | Recorded as per-chunk dispatch loop with push constant updates |
| `BarrierNode` | Recorded as `vkCmdPipelineBarrier2` (compute → compute) |
| `RetrievalNode` | Triggers compute→transfer barrier + `vkCmdCopyBuffer` to staging + fence signal → `VulkanRetrievalFuture` |
| `BufferDescriptor` | Drives `VulkanBufferAllocator` allocation (device-local SSBOs) |
| `BufferHandle` | Mapped to `VkBuffer` + `VmaAllocation` objects by the allocator |
| `BufferRole` | Determines allocation strategy (persistent model state vs. per-batch intermediates) |
| `HardwareProfile` | Populated by `discovery.py` from Vulkan device property queries |
| `PrecisionConfig` | Drives specialization constant selection (FP32 initially) |
| `RetrievalFuture` | Protocol implemented by `VulkanRetrievalFuture` |
| `PlanRenderer` | Protocol implemented by `VulkanPlanRenderer` |
| `ReductionTreePlan` | Consumed by `_record_reduction_tree()` to generate staged command sequences |
| `StreamingLoopPlan` | Consumed by `_record_streaming_loop()` for per-chunk push constant iteration |

---

## Table of Contents

1. [Scope & Constraints](#1-scope--constraints)
2. [Current State After Phase 5A](#2-current-state-after-phase-5a)
3. [Target Deliverables](#3-target-deliverables)
4. [Task Breakdown](#4-task-breakdown)
   - [Step 5B.1: Implement `context.py` — Vulkan device/instance lifecycle](#step-5b1-implement-contextpy--vulkan-deviceinstance-lifecycle)
   - [Step 5B.2: Implement `discovery.py` — HardwareProfile population](#step-5b2-implement-discoverypy--hardwareprofile-population)
   - [Step 5B.3: Implement `type_mapping.py` — PrecisionConfig → Vulkan type mapping](#step-5b3-implement-type_mappingpy--precisionconfig--vulkan-type-mapping)
   - [Step 5B.4: Implement `buffer_allocator.py` — device-local buffer management](#step-5b4-implement-buffer_allocatorpy--device-local-buffer-management)
   - [Step 5B.5: Implement `_pipeline_cache.py` — SPIR-V loading and pipeline creation](#step-5b5-implement-_pipeline_cachepy--spir-v-loading-and-pipeline-creation)
   - [Step 5B.6: Implement `_descriptor_manager.py` — descriptor set/layout management](#step-5b6-implement-_descriptor_managerpy--descriptor-setlayout-management)
   - [Step 5B.7: Implement `_push_constants.py` — push constant struct marshalling](#step-5b7-implement-_push_constantspy--push-constant-struct-marshalling)
   - [Step 5B.8: Implement `retrieval.py` — VulkanRetrievalFuture](#step-5b8-implement-retrievalpy--vulkanretrievalfuture)
   - [Step 5B.9: Implement `renderer.py` — VulkanPlanRenderer (command buffer recording)](#step-5b9-implement-rendererpy--vulkanplanrenderer-command-buffer-recording)
   - [Step 5B.10: Update `_build_config.py` template](#step-5b10-update-_build_configpy-template)
   - [Step 5B.11: Update `src/backends/vulkan/__init__.py` exports](#step-5b11-update-srcbackendsvulkan__init__py-exports)
   - [Step 5B.12: Update `src/backends/vulkan/meson.build` for new Python modules](#step-5b12-update-srcbackendsvulkanmesonbuild-for-new-python-modules)
   - [Step 5B.13: Write infrastructure smoke tests](#step-5b13-write-infrastructure-smoke-tests)
   - [Step 5B.14: Validate rollback gate](#step-5b14-validate-rollback-gate)
5. [Module Inventory](#5-module-inventory)
6. [Vulkan Object Lifecycle Management](#6-vulkan-object-lifecycle-management)
7. [Command Buffer Recording Architecture](#7-command-buffer-recording-architecture)
8. [Buffer Allocation Strategy](#8-buffer-allocation-strategy)
9. [Descriptor Set Strategy](#9-descriptor-set-strategy)
10. [Dispatch Flow](#10-dispatch-flow)
11. [Risk Register](#11-risk-register)

---

## 1. Scope & Constraints

### In scope

- Implementing `src/backends/vulkan/context.py` — Vulkan instance creation, physical device selection (prioritizing discrete GPU), logical device creation with a compute queue, command pool creation. Context teardown destroys all Vulkan objects in reverse creation order.
- Implementing `src/backends/vulkan/discovery.py` — populating `HardwareProfile` from Vulkan device property queries: `subgroupSize` → `simd_width`, `maxComputeSharedMemorySize` → `max_local_mem_bytes`, device memory heap sizes → `global_mem_bytes`, computed `max_reduce_fan_in` from shared memory and subgroup constraints, cache line deduction from subgroup granularity → `cache_line_bytes`.
- Implementing `src/backends/vulkan/type_mapping.py` — mapping `PrecisionConfig` to Vulkan/GLSL type information and specialization constant values. Initially FP32 only.
- Implementing `src/backends/vulkan/buffer_allocator.py` — translating `BufferDescriptor` plan-level declarations into device-local `VkBuffer` + memory allocations (via VMA or manual `vkAllocateMemory` with memory type selection). Manages the staging buffer for D2H readback.
- Implementing `src/backends/vulkan/_pipeline_cache.py` — loading `.spv` modules via `importlib.resources`, creating `VkShaderModule` objects, creating compute pipelines with specialization constants, caching pipelines keyed by `(shader_name, specialization_values)`.
- Implementing `src/backends/vulkan/_descriptor_manager.py` — managing `VkDescriptorSetLayout` creation per kernel, `VkDescriptorPool` allocation, descriptor set updates for fixed-binding kernels, and `vkCmdPushDescriptorSetKHR` for dynamic-binding kernels (reduction engine).
- Implementing `src/backends/vulkan/_push_constants.py` — `ctypes.Structure` definitions for each push constant struct (mirroring the C-side structs from VULKAN_BACKEND.md), plus marshalling helpers that convert plan node `scalar_params` dicts into packed push constant bytes.
- Implementing `src/backends/vulkan/retrieval.py` — `VulkanRetrievalFuture` satisfying the `RetrievalFuture` Protocol: `.wait()` blocks on `vkWaitForFences(act_fence)`, `.result()` reads from the persistently-mapped staging buffer and strips padding, `.release()` resets the fence and frees per-batch staging resources.
- Implementing `src/backends/vulkan/renderer.py` — the `VulkanPlanRenderer` class satisfying the `PlanRenderer` Protocol. Records the Act and Learn phases into separate `VkCommandBuffer` objects. Traverses `ExecutionPlan.topological_order` and records:
  - `KernelDispatchNode` → bind pipeline + bind descriptors + push constants + `vkCmdDispatch`
  - `ReductionTreeNode` → staged `_record_reduction_tree()` with ping-pong buffers and per-stage barriers
  - `StreamingLoopNode` → per-chunk dispatch loop with `vkCmdPushConstants` updates
  - `BarrierNode` → `vkCmdPipelineBarrier2` (compute write → compute read)
  - `RetrievalNode` → compute→transfer barrier + `vkCmdCopyBuffer` to staging
- Submits command buffers via `vkQueueSubmit` with fence signaling.
- Updating `_build_config.py.in` to include `BACKEND_VULKAN = True` when the Vulkan backend is built.
- Writing infrastructure smoke tests validating context creation, pipeline loading, buffer allocation, and basic plan traversal.

### Out of scope

- Writing or modifying GLSL shaders (Phase 5A).
- Writing per-kernel Tier 2 correctness tests (Phase 5C).
- Modifying shared-layer code beyond `_build_config.py.in` updates.
- Modifying the OpenCL backend or CPU backend.
- FP16 support (FP32 only initially).
- Performance profiling or optimization.
- VMA library integration as a C dependency — Python-side memory management uses `vulkan-python`'s direct Vulkan memory API calls. VMA-style allocation logic (memory type selection, sub-allocation) is implemented in Python.
- The `WorkTicket` / `LearnHandle` / `Engine` user-facing API.

### Key constraint: `vulkan-python` as sole native dependency

The Vulkan backend uses `vulkan-python` (optional dependency: `vulkan = ["vulkan-python>=0.2.0"]`) for all Vulkan API interactions. Unlike the CPU backend (which uses `ctypes` for a custom shared library), the Vulkan backend delegates to the Vulkan driver via `vulkan-python`'s generated bindings. No custom C/C++ code is needed on the Python side — the native code lives entirely in the SPIR-V shaders.

### Key constraint: command buffer recording ≠ execution

The Vulkan backend's defining characteristic is **deferred execution**: the renderer records the entire DAG into command buffers (CPU-side, fast) then submits them for GPU execution. The GPU processes all commands autonomously with zero host interaction until the fence signals. This means:

1. Command buffer recording is deterministic and fast — it can be profiled and debugged independently of GPU execution.
2. Errors in barrier placement or descriptor binding manifest as validation layer warnings (during development) or incorrect results (during testing), not as crashes during recording.
3. The Act and Learn phases use separate command buffers (VULKAN_BACKEND.md §Command Buffer Strategy) to support the Event-Triggered Execution Mode — the host reads Act results and waits for external events before triggering Learn.

### Key constraint: renderer is a pure consumer of the plan

Same constraint as OpenCL (Phase 2A) and CPU (Phase 3B) renderers: the `VulkanPlanRenderer` receives an immutable `ExecutionPlan` and never modifies it. All plan construction occurs in the shared layer.

---

## 2. Current State After Phase 5A

### Vulkan backend (`src/backends/vulkan/`)

| Artifact | Status | Notes |
| :--- | :--- | :--- |
| `__init__.py` | Exists (5A) | Empty — populated in this phase |
| `kernel_sources/__init__.py` | Exists (5A) | Package marker for `importlib.resources` |
| `kernel_sources/common.glsl` | ✅ (5A) | Shared declarations, subgroup helpers |
| `kernel_sources/*.comp` | ✅ (5A) | 18 compute shaders |
| `kernel_sources/*.spv` | ✅ (5A build) | 18 compiled SPIR-V modules |
| `kernel_sources/meson.build` | ✅ (5A) | `custom_target()` rules for `glslc` |
| `meson.build` | ✅ (5A) | Delegates to `kernel_sources/`; installs `__init__.py` |
| `context.py` | ❌ Not started | This phase |
| `discovery.py` | ❌ Not started | This phase |
| `type_mapping.py` | ❌ Not started | This phase |
| `buffer_allocator.py` | ❌ Not started | This phase |
| `_pipeline_cache.py` | ❌ Not started | This phase |
| `_descriptor_manager.py` | ❌ Not started | This phase |
| `_push_constants.py` | ❌ Not started | This phase |
| `retrieval.py` | ❌ Not started | This phase |
| `renderer.py` | ❌ Not started | This phase |

---

## 3. Target Deliverables

| Deliverable | Location | Description |
| :--- | :--- | :--- |
| `context.py` | `src/backends/vulkan/` | Vulkan instance, device, queue, command pool lifecycle |
| `discovery.py` | `src/backends/vulkan/` | `HardwareProfile` population from `VkPhysicalDevice` properties |
| `type_mapping.py` | `src/backends/vulkan/` | `PrecisionConfig` → specialization constant values, buffer format info |
| `buffer_allocator.py` | `src/backends/vulkan/` | `BufferDescriptor` → `VkBuffer` + device memory allocation; staging buffer management |
| `_pipeline_cache.py` | `src/backends/vulkan/` | SPIR-V loading, `VkShaderModule` / `VkPipeline` creation with specialization, caching |
| `_descriptor_manager.py` | `src/backends/vulkan/` | Descriptor set layouts, pool, fixed + push descriptor management |
| `_push_constants.py` | `src/backends/vulkan/` | `ctypes.Structure` push constant definitions + marshalling from plan scalars |
| `retrieval.py` | `src/backends/vulkan/` | `VulkanRetrievalFuture` — fence-gated staging readback with unpadding |
| `renderer.py` | `src/backends/vulkan/` | `VulkanPlanRenderer` — command buffer recording, submission, DAG traversal |
| Updated `__init__.py` | `src/backends/vulkan/` | Public exports: `VulkanPlanRenderer`, `discover_hardware`, etc. |
| Updated `meson.build` | `src/backends/vulkan/` | Installs new Python modules |

---

## 4. Task Breakdown

### Step 5B.1: Implement `context.py` — Vulkan device/instance lifecycle

Create the Vulkan context manager encapsulating the entire Vulkan device lifecycle:

**Responsibilities:**
- `VkInstance` creation with validation layers enabled in debug mode (`VK_LAYER_KHRONOS_validation`).
- Physical device enumeration and selection — prefer `VK_PHYSICAL_DEVICE_TYPE_DISCRETE_GPU`, fall back to integrated GPU, fail if no compute-capable device found.
- Logical device creation with a single compute queue family.
- Command pool creation (`VK_COMMAND_POOL_CREATE_RESET_COMMAND_BUFFER_BIT`).
- Two command buffers: `act_cmd` and `learn_cmd`.
- Two fences: `act_fence` and `learn_fence` (created in unsignaled state).
- Context cleanup: destroy all Vulkan objects in reverse creation order.

**Interface:**

```python
class VulkanContext:
    def __init__(self, *, enable_validation: bool = False) -> None: ...

    @property
    def device(self) -> VkDevice: ...
    @property
    def physical_device(self) -> VkPhysicalDevice: ...
    @property
    def compute_queue(self) -> VkQueue: ...
    @property
    def queue_family_index(self) -> int: ...
    @property
    def command_pool(self) -> VkCommandPool: ...
    def allocate_command_buffer(self) -> VkCommandBuffer: ...
    def create_fence(self, *, signaled: bool = False) -> VkFence: ...
    def destroy(self) -> None: ...
```

**Required `vulkan-python` APIs:** `vkCreateInstance`, `vkEnumeratePhysicalDevices`, `vkGetPhysicalDeviceProperties2`, `vkGetPhysicalDeviceMemoryProperties`, `vkCreateDevice`, `vkGetDeviceQueue`, `vkCreateCommandPool`, `vkAllocateCommandBuffers`, `vkCreateFence`, and teardown counterparts.

**Acceptance criteria:**
- Context initializes on a machine with a Vulkan-capable GPU.
- Context initializes with validation layers when `enable_validation=True`.
- Context `destroy()` frees all resources without validation layer warnings.
- Context creation fails gracefully with a clear error on machines without Vulkan support.
- Tier 1 green.

---

### Step 5B.2: Implement `discovery.py` — HardwareProfile population

Populate `HardwareProfile` from Vulkan device properties:

| HardwareProfile Field | Vulkan Source |
| :--- | :--- |
| `simd_width` | `VkPhysicalDeviceSubgroupProperties.subgroupSize` |
| `cache_line_bytes` | Estimated from subgroup size × 4 bytes (FP32); or device-specific heuristic |
| `max_reduce_fan_in` | Computed: `min(subgroup_size * max_workgroup_size_x / subgroup_size, shared_mem_limit)` — similar to OpenCL computation but using Vulkan properties |
| `max_local_mem_bytes` | `VkPhysicalDeviceLimits.maxComputeSharedMemorySize` |
| `global_mem_bytes` | Largest `VkMemoryHeap` with `VK_MEMORY_HEAP_DEVICE_LOCAL_BIT` |

**Required queries:**
- `vkGetPhysicalDeviceProperties2` with `VkPhysicalDeviceSubgroupProperties` chained into `pNext`.
- `vkGetPhysicalDeviceMemoryProperties`.
- `VkPhysicalDeviceLimits` from the base properties structure.

**Acceptance criteria:**
- Returns a valid `HardwareProfile` on any Vulkan-capable device.
- `simd_width` is a power of 2 (typical: 16, 32, 64).
- `max_local_mem_bytes` is non-None (all Vulkan compute devices have shared memory).

---

### Step 5B.3: Implement `type_mapping.py` — PrecisionConfig → Vulkan type mapping

Map `PrecisionConfig` (initially FP32 only) to Vulkan-relevant type information:

| Property | FP32 Value | FP16 Value (future) |
| :--- | :--- | :--- |
| Specialization constant for `SCALAR_IS_HALF` | 0 | 1 |
| Buffer element size (bytes) | 4 | 2 |
| `VkFormat` for validation | `VK_FORMAT_R32_SFLOAT` | `VK_FORMAT_R16_SFLOAT` |
| numpy dtype for host-side arrays | `np.float32` | `np.float16` |

**Acceptance criteria:**
- FP32 mapping is complete and correct.
- FP16 mapping raises `NotImplementedError` with a clear message.

---

### Step 5B.4: Implement `buffer_allocator.py` — device-local buffer management

Translate `BufferDescriptor` plan-level declarations into physical Vulkan buffer allocations.

**Design:**
- **Device-local buffers** (all compute data): `VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT`. Usage flags: `VK_BUFFER_USAGE_STORAGE_BUFFER_BIT`. For buffers that source a D2H copy, add `VK_BUFFER_USAGE_TRANSFER_SRC_BIT`.
- **Staging buffer** (D2H readback for `RetrievalNode`): `VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT`. Usage: `VK_BUFFER_USAGE_TRANSFER_DST_BIT`. Persistently mapped at allocation time.
- **Memory type selection:** Scan `VkPhysicalDeviceMemoryProperties.memoryTypes` for the first type satisfying the required property flags within a heap that has sufficient free space.
- **Lifetime management:** Buffers allocated by `BufferRole`:
  - `MODEL_STATE`: Long-lived, persist across batches (weights, biases, temperatures, optimizer state).
  - `BATCH_INPUT` / `BATCH_INTERMEDIATE` / `BATCH_OUTPUT`: Per-batch, freed after the batch completes.

**Interface:**

```python
class VulkanBufferAllocator:
    def __init__(self, context: VulkanContext, precision: PrecisionConfig) -> None: ...

    def allocate(self, descriptor: BufferDescriptor) -> VulkanBuffer: ...
    def allocate_staging(self, size_bytes: int) -> VulkanStagingBuffer: ...
    def release_batch_buffers(self) -> None: ...
    def destroy(self) -> None: ...

@dataclass
class VulkanBuffer:
    buffer: VkBuffer
    memory: VkDeviceMemory
    size_bytes: int
    descriptor: BufferDescriptor

@dataclass
class VulkanStagingBuffer:
    buffer: VkBuffer
    memory: VkDeviceMemory
    mapped_ptr: ctypes.c_void_p
    size_bytes: int
```

**Host-to-device data upload** (for input data and initial model state): uses a transient staging buffer, `vkCmdCopyBuffer`, and a transfer→compute barrier. Alternatively, for small uploads, uses `VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT` (resizable BAR) if available.

**Acceptance criteria:**
- Device-local buffers are allocated with correct size and usage flags.
- Staging buffer is persistently mapped and readable from the host.
- `release_batch_buffers()` frees only per-batch buffers, leaving model state intact.
- `destroy()` frees all remaining buffers with no Vulkan validation warnings.

---

### Step 5B.5: Implement `_pipeline_cache.py` — SPIR-V loading and pipeline creation

Load compiled SPIR-V modules and create compute pipelines with specialization constants.

**Responsibilities:**
- SPIR-V module discovery via `importlib.resources.files('averaging_ensembled_classifier.backends.vulkan.kernel_sources')`.
- `VkShaderModule` creation from `.spv` bytes.
- `VkPipelineLayout` creation from descriptor set layouts + push constant ranges.
- `VkComputePipeline` creation with `VkSpecializationInfo` (specialization constants from `HardwareProfile` and problem type).
- Pipeline caching: key is `(shader_name, problem_type)` since `SPEC_SIMD_WIDTH` and `SPEC_C_TILE_SIZE` are fixed per device.

**Pipeline creation flow:**

```python
def create_pipeline(
    self,
    shader_name: str,
    descriptor_layout: VkDescriptorSetLayout,
    push_constant_size: int,
    specialization: SpecConstants,
) -> ComputePipeline: ...
```

Where `SpecConstants` is a frozen dataclass:

```python
@dataclass(frozen=True)
class SpecConstants:
    simd_width: int
    bank_padding: int = 1
    tile_size: int = 8
    problem_type: int = 0  # 0=CCE, 1=BCE
```

**Total pipelines created:** ~22 (see Phase 5A §9).

**Acceptance criteria:**
- All 18 `.spv` files load successfully.
- Pipeline creation succeeds for all ~22 variants.
- Specialization constants are correctly applied (verifiable via validation layers).
- `VkShaderModule` objects are destroyed after pipeline creation (not needed post-creation).

---

### Step 5B.6: Implement `_descriptor_manager.py` — descriptor set/layout management

Manage descriptor set layouts, the descriptor pool, and per-kernel descriptor set updates.

**Two-track strategy (VULKAN_BACKEND.md §Descriptor Set Strategy):**

1. **Fixed pipelines** (most kernels): Pre-allocated descriptor sets, updated once per batch via `vkUpdateDescriptorSets`. The buffer bindings don't change within a batch.

2. **Dynamic pipelines** (reduction engine — `aggregate_partials`, `clip_intermediate_grad`): `VK_KHR_push_descriptor` via `vkCmdPushDescriptorSetKHR`. Bindings change every reduction stage — push descriptors record updates directly into the command buffer without pre-allocation.

**Interface:**

```python
class VulkanDescriptorManager:
    def __init__(self, context: VulkanContext) -> None: ...

    def create_layout(self, binding_count: int) -> VkDescriptorSetLayout: ...
    def allocate_set(self, layout: VkDescriptorSetLayout) -> VkDescriptorSet: ...
    def update_set(self, desc_set: VkDescriptorSet,
                   bindings: list[tuple[int, VulkanBuffer]]) -> None: ...
    def destroy(self) -> None: ...
```

**Descriptor pool sizing:**
- Estimate total descriptor sets from the pipeline variant count (~22 fixed sets, plus overhead).
- Pool type: `VK_DESCRIPTOR_TYPE_STORAGE_BUFFER` — all buffer bindings are SSBOs.
- Maximum bindings per set: 10 (the largest is `clip_partial_gradients` with 10 bindings).

**Push descriptor requirements:**
- Check for `VK_KHR_push_descriptor` support during device creation.
- If `VK_KHR_push_descriptor` is unavailable, fall back to pre-allocated descriptor sets with per-stage updates (slower but functionally equivalent).

**Acceptance criteria:**
- Descriptor layouts match the binding counts from Phase 5A §7.
- Descriptor set allocation succeeds for all fixed pipelines.
- Push descriptor commands record correctly into command buffers.
- No validation warnings about descriptor set mismatches.

---

### Step 5B.7: Implement `_push_constants.py` — push constant struct marshalling

Define `ctypes.Structure` subclasses for each push constant struct from VULKAN_BACKEND.md, plus marshalling helpers:

```python
class ForwardPassPush(ctypes.Structure):
    _fields_ = [
        ("batch_chunk_offset", ctypes.c_uint32),
        ("batch_chunk_count", ctypes.c_uint32),
        ("total_batch_count", ctypes.c_uint32),
        ("padded_input_count", ctypes.c_uint32),
        ("padded_hidden_count", ctypes.c_uint32),
    ]

class AdamUpdatePush(ctypes.Structure):
    _fields_ = [
        ("learning_rate", ctypes.c_float),
        ("beta1_pow_t", ctypes.c_float),
        ("beta2_pow_t", ctypes.c_float),
        ("beta1", ctypes.c_float),
        ("beta2", ctypes.c_float),
        ("epsilon", ctypes.c_float),
        ("parameter_count", ctypes.c_uint32),
    ]

# ... (one struct per push constant layout)
```

**Marshalling helper:** Converts `KernelDispatchNode.scalar_params` (a `dict[str, int | float]`) into the packed push constant bytes for the corresponding kernel:

```python
def marshal_push_constants(kernel_name: str, scalar_params: dict[str, int | float]) -> bytes:
    """Convert plan scalar_params into Vulkan push constant bytes."""
    ...
```

The mapping from plan `scalar_params` keys to push constant struct fields is the binding layer's responsibility — each kernel has a known mapping from abstract parameter names to struct fields.

**Acceptance criteria:**
- All push constant structs match the sizes in Phase 5A §6.
- Marshalling produces correctly packed bytes (little-endian, no padding surprises).
- All plan scalar parameter names map to push constant struct fields for every kernel.

---

### Step 5B.8: Implement `retrieval.py` — VulkanRetrievalFuture

Implement the `RetrievalFuture` Protocol for the Vulkan backend:

```python
class VulkanRetrievalFuture:
    def __init__(
        self,
        context: VulkanContext,
        fence: VkFence,
        staging: VulkanStagingBuffer,
        node_id: str,
        logical_shape: tuple[int, ...],
        padded_shape: tuple[int, ...],
        dtype: np.dtype,
    ) -> None: ...

    @property
    def node_id(self) -> str: ...

    def wait(self) -> None:
        """Block until the GPU signals the fence."""
        # vkWaitForFences(device, 1, [fence], True, UINT64_MAX)
        ...

    def result(self) -> NDArray[np.floating]:
        """Read from staging buffer, strip padding, return unpadded numpy array."""
        self.wait()
        # Read from staging.mapped_ptr into numpy array
        # Slice to logical_shape (strip padding)
        ...

    def release(self) -> None:
        """Reset fence, free staging buffer."""
        # vkResetFences(device, 1, [fence])
        ...
```

**Key design details:**
- The staging buffer is persistently mapped (allocated with `VMA_ALLOCATION_CREATE_MAPPED_BIT` equivalent). After the fence signals, the host reads directly from the mapped pointer — no map/unmap overhead.
- Padding stripping: the staging buffer contains the padded result (SIMD-aligned dimensions). `result()` copies the data into a numpy array and slices to the logical (unpadded) shape.
- Thread safety: `wait()` is idempotent. Multiple calls are safe — `vkWaitForFences` returns immediately if the fence is already signaled.

**Acceptance criteria:**
- Implements the `RetrievalFuture` Protocol (passes `isinstance` check at runtime).
- `wait()` blocks until GPU work completes.
- `result()` returns an unpadded numpy array with the correct shape and dtype.
- `release()` resets the fence without validation warnings.

---

### Step 5B.9: Implement `renderer.py` — VulkanPlanRenderer (command buffer recording)

The core module — implements `PlanRenderer` Protocol by recording `ExecutionPlan` nodes into Vulkan command buffers and submitting them for GPU execution.

**Architecture:**

```python
class VulkanPlanRenderer:
    def __init__(
        self,
        context: VulkanContext,
        hardware_profile: HardwareProfile,
        precision: PrecisionConfig,
        problem_type: int,  # 0=CCE, 1=BCE
    ) -> None:
        # Initialize buffer allocator, pipeline cache, descriptor manager
        # Create all ~22 pipelines
        ...

    def render(self, plan: ExecutionPlan) -> list[RetrievalFuture]:
        """Record and submit the plan, returning futures for retrieval nodes."""
        ...

    def destroy(self) -> None:
        """Clean up all Vulkan resources."""
        ...
```

**Rendering flow:**

1. **Allocate buffers:** Iterate `plan.buffer_descriptors`, call `buffer_allocator.allocate()` for each. Upload input data and model state to device-local buffers via staging.
2. **Update descriptor sets:** For each pipeline's fixed descriptor set, bind the allocated buffers.
3. **Record command buffer:** Begin recording → traverse `plan.topological_order`:
   - **`KernelDispatchNode`:** Look up pipeline from cache → bind pipeline → bind descriptor set → push constants → `vkCmdDispatch(tile_count, 1, 1)` (or 2D dispatch for `forward_pass`).
   - **`ReductionTreeNode`:** Call `_record_reduction_tree()` — staged dispatch-barrier-dispatch loop with ping-pong buffers and push descriptor updates per stage.
   - **`StreamingLoopNode`:** Call `_record_streaming_loop()` — per-chunk dispatch loop: update push constants with chunk index → dispatch body nodes → barrier → next chunk.
   - **`BarrierNode`:** Call `_record_barrier()` — `vkCmdPipelineBarrier2` with compute→compute memory barrier.
   - **`RetrievalNode`:** Call `_record_retrieval()` — compute→transfer barrier → `vkCmdCopyBuffer` to staging buffer.
4. **End recording, submit:** `vkEndCommandBuffer` → `vkQueueSubmit` → return `VulkanRetrievalFuture` for each `RetrievalNode`.

**DAG concurrency exploitation:**

The plan's dependency edges encode concurrency — nodes with no mutual dependency can be dispatched without intermediate barriers. The renderer records dispatches in topological order. When consecutive nodes have no data hazards (e.g., Nodes 8, 9, 10 all read the same inputs and write disjoint outputs), no barrier is recorded between them. The GPU hardware scheduler overlaps their execution naturally.

Barrier insertion follows the plan's explicit `BarrierNode` placements. The plan builder (shared layer) is responsible for placing barriers at the correct synchronization points. The renderer trusts the plan's barrier placement — it does not infer data hazards.

**Reduction tree recording:**

```python
def _record_reduction_tree(self, cmd: VkCommandBuffer,
                            tree: ReductionTreePlan) -> None:
    """Record a staged reduction into the command buffer."""
    src_buf, dst_buf = self._ping_pong_buffers(tree)

    for stage in range(tree.num_stages):
        # Select tier: register (K <= subgroup_size) or local
        pipeline = self._select_aggregate_pipeline(tree, stage)

        # Push descriptors: src, offset_list, dst
        self._push_descriptor_aggregate(cmd, pipeline, src_buf,
                                         tree.offset_list_buffer(stage),
                                         dst_buf, tree, stage)
        # Dispatch aggregate
        groups = self._compute_aggregate_groups(tree, stage)
        vkCmdDispatch(cmd, groups, 1, 1)

        # Barrier: sum → clip
        self._record_barrier(cmd)

        # Dispatch clip with computed threshold
        threshold = self._compute_stage_threshold(tree, stage)
        self._push_constants_clip(cmd, threshold, tree, stage)
        vkCmdDispatch(cmd, self._compute_clip_groups(tree, stage), 1, 1)

        # Barrier: clip → next stage
        self._record_barrier(cmd)

        # Ping-pong
        src_buf, dst_buf = dst_buf, src_buf
```

**Acceptance criteria:**
- Renders a complete Act plan and Learn plan without validation warnings.
- Returns `VulkanRetrievalFuture` for each `RetrievalNode`.
- Barrier placement matches the DAG's explicit synchronization points.
- Reduction tree stages execute in correct order with correct thresholds.
- Streaming loops iterate over all chunks with correct push constant values.
- `destroy()` frees all Vulkan resources without warnings.

---

### Step 5B.10: Update `_build_config.py` template

Update `_build_config.py.in` so that `BACKEND_VULKAN` is set to `True` when the Vulkan backend is built:

```python
BACKEND_VULKAN = @BACKEND_VULKAN@
```

The Meson `configure_file()` substitution replaces `@BACKEND_VULKAN@` with `True` or `False` based on whether the Vulkan build target was enabled.

**Acceptance criteria:**
- `_build_config.BACKEND_VULKAN` is `True` when `aec_backend_vulkan` is enabled and `glslc` was found.
- `_build_config.BACKEND_VULKAN` is `False` otherwise.
- Existing `BACKEND_OPENCL` and `BACKEND_CPU` flags are unaffected.

---

### Step 5B.11: Update `src/backends/vulkan/__init__.py` exports

Expose the public interface:

```python
from .renderer import VulkanPlanRenderer
from .discovery import discover_hardware
from .context import VulkanContext
from .buffer_allocator import VulkanBufferAllocator
from .retrieval import VulkanRetrievalFuture

__all__ = [
    "VulkanPlanRenderer",
    "VulkanContext",
    "VulkanBufferAllocator",
    "VulkanRetrievalFuture",
    "discover_hardware",
]
```

**Acceptance criteria:**
- `from averaging_ensembled_classifier.backends.vulkan import VulkanPlanRenderer` resolves when the Vulkan backend is available.
- Import fails gracefully with a clear message when `vulkan-python` is not installed.

---

### Step 5B.12: Update `src/backends/vulkan/meson.build` for new Python modules

Add all new Python modules to the install list:

```meson
py.install_sources(
    '__init__.py',
    'context.py',
    'discovery.py',
    'type_mapping.py',
    'buffer_allocator.py',
    '_pipeline_cache.py',
    '_descriptor_manager.py',
    '_push_constants.py',
    'retrieval.py',
    'renderer.py',
    subdir: 'averaging_ensembled_classifier' / 'backends' / 'vulkan',
)

subdir('kernel_sources')
```

**Acceptance criteria:**
- `ninja install` places all Python modules in the correct package directory.
- All modules are importable after installation.

---

### Step 5B.13: Write infrastructure smoke tests

Structural tests that validate the integration layer works without testing kernel correctness:

| Test | Validates |
| :--- | :--- |
| `test_vulkan_context_creates` | `VulkanContext()` initializes without errors; `destroy()` cleans up |
| `test_vulkan_discovery_returns_profile` | `discover_hardware()` returns a `HardwareProfile` with valid fields |
| `test_vulkan_pipeline_cache_loads_shaders` | All `.spv` modules load; all ~22 pipelines create successfully |
| `test_vulkan_buffer_allocator_lifecycle` | Buffer allocate → release cycle completes without validation warnings |
| `test_vulkan_simple_plan_traversal` | Renderer records a minimal plan (1 `KernelDispatchNode` + 1 `RetrievalNode`) into a command buffer without errors |
| `test_vulkan_retrieval_future_protocol` | `VulkanRetrievalFuture` passes `isinstance(x, RetrievalFuture)` check |

All smoke tests are marked `@pytest.mark.vulkan` and skip when `_build_config.BACKEND_VULKAN is not True`.

**Acceptance criteria:**
- Smoke tests pass on machines with Vulkan-capable GPUs and `vulkan-python` installed.
- Smoke tests skip cleanly on machines without Vulkan support.
- No validation layer warnings during smoke test execution.

---

### Step 5B.14: Validate rollback gate

**Gate condition:** Tier 1 green + smoke tests pass + `_build_config.BACKEND_VULKAN` correctly reflects availability.

Run:
1. `pytest tests/tier1/ -v` — 140+ tests unchanged.
2. `pytest tests/ -k vulkan --co` — Vulkan smoke tests collected (not skipped due to missing backend).
3. `pytest tests/ -k vulkan -v` — smoke tests pass.

**Acceptance criteria:**
- Tier 1 unchanged.
- Vulkan smoke tests pass on Vulkan-capable hardware.
- Existing CPU Tier 2 and OpenCL Tier 2 tests unaffected.

---

## 5. Module Inventory

| Module | Responsibility | Lines (est.) | Dependencies |
| :--- | :--- | :--- | :--- |
| `context.py` | Vulkan instance/device/queue/pool lifecycle | ~200 | `vulkan-python` |
| `discovery.py` | `HardwareProfile` from Vulkan device queries | ~80 | `vulkan-python`, `shared.hardware_profile` |
| `type_mapping.py` | `PrecisionConfig` → Vulkan type info | ~40 | `shared.precision_config` |
| `buffer_allocator.py` | `BufferDescriptor` → `VkBuffer` + memory | ~250 | `vulkan-python`, `context.py`, `shared.buffer_lifecycle` |
| `_pipeline_cache.py` | SPIR-V loading, pipeline creation, caching | ~200 | `vulkan-python`, `context.py`, `importlib.resources` |
| `_descriptor_manager.py` | Descriptor layouts, pool, set management | ~180 | `vulkan-python`, `context.py` |
| `_push_constants.py` | `ctypes.Structure` push constant defs + marshalling | ~250 | `ctypes`, `shared.plan_types` |
| `retrieval.py` | `VulkanRetrievalFuture` — fence-gated readback | ~80 | `vulkan-python`, `context.py`, `shared.retrieval_future` |
| `renderer.py` | `VulkanPlanRenderer` — command recording + submission | ~450 | All above modules, `shared.plan_types`, `shared.plan_renderer` |
| **Total** | | **~1,730** | |

---

## 6. Vulkan Object Lifecycle Management

All Vulkan objects follow strict creation-order tracked destruction:

```
VkInstance
  └── VkPhysicalDevice (enumerated, not created/destroyed)
  └── VkDevice
        ├── VkQueue (obtained, not created/destroyed)
        ├── VkCommandPool
        │     ├── act_cmd (VkCommandBuffer)
        │     └── learn_cmd (VkCommandBuffer)
        ├── VkFence (act_fence, learn_fence)
        ├── VkDescriptorPool
        │     └── VkDescriptorSet(s) (freed with pool)
        ├── VkPipelineLayout(s)
        │     └── VkComputePipeline(s)
        ├── VkDescriptorSetLayout(s)
        ├── VkShaderModule(s) → destroyed after pipeline creation
        ├── VkBuffer(s) + VkDeviceMemory (via buffer_allocator)
        └── Staging VkBuffer + mapped memory
```

Destruction order: buffers → pipelines → pipeline layouts → descriptor sets (via pool) → descriptor pool → descriptor set layouts → fences → command pool → device → instance.

The `VulkanContext.destroy()` method and `VulkanPlanRenderer.destroy()` method coordinate to ensure no double-free and no use-after-destroy. The renderer owns pipeline and descriptor resources; the context owns device-level resources.

---

## 7. Command Buffer Recording Architecture

### Act Phase Command Buffer

```
vkBeginCommandBuffer(act_cmd)
  │
  ├── Node 4: forward_pass       vkCmdDispatch(batch_chunks, hidden_blocks, 1)
  ├── barrier                     compute → compute
  ├── Node 5: render_logits       vkCmdDispatch(tile_count, 1, 1)
  ├── barrier                     compute → compute
  ├── Node 6/7: probs_loss        vkCmdDispatch(tile_count, 1, 1)
  ├── barrier                     compute → compute
  ├── Node 23: retrieval          compute → transfer barrier
  │                               vkCmdCopyBuffer(probs → staging)
  │
vkEndCommandBuffer(act_cmd)
vkQueueSubmit(act_cmd, signal: act_fence)
```

### Learn Phase Command Buffer

```
vkBeginCommandBuffer(learn_cmd)
  │
  ├── Recompute forward activations (Nodes 4, 5)
  ├── barrier
  ├── Nodes 8,9,10: grad production (no mutual barriers — disjoint outputs)
  ├── barrier                     {8,9,10} → 11
  ├── Node 11: clip_partial       vkCmdDispatch(tile_count, 1, 1)
  ├── barrier                     11 → 13
  ├── Node 13: gather_permute     vkCmdDispatch(rows, 1, 1)
  ├── barrier                     13 → {15, 16}
  ├── Node 15: reduction trees    _record_reduction_tree × 3 (mod_w, mod_b, temps)
  ├── Node 16: stabilize_grad_h   vkCmdDispatch(rows, 1, 1)
  │   (15 and 16 have no mutual barrier — disjoint data)
  ├── barrier                     {15, 16} → streaming phase
  ├── Streaming loop: for each chunk c:
  │     ├── Nodes 17,18: backprop (no mutual barrier)
  │     ├── barrier               {17,18} → 19
  │     ├── Node 19: clip_shared
  │     └── barrier               19 → next chunk
  ├── Reduction trees: grad_sw, grad_sb
  ├── barrier
  ├── Node 21: normalize × per param group
  ├── barrier
  ├── Node 24: adam_update × per param group
  ├── barrier
  ├── Node 25: clamp_temps
  │
vkEndCommandBuffer(learn_cmd)
vkQueueSubmit(learn_cmd, signal: learn_fence)
```

---

## 8. Buffer Allocation Strategy

| Architecture Buffer Role | Vulkan Memory Type | Usage Flags | Lifetime |
| :--- | :--- | :--- | :--- |
| `MODEL_STATE` (weights, biases, temps, optimizer m1/m2) | Device-local | `STORAGE_BUFFER` | Persistent (across batches) |
| `BATCH_INPUT` (x_data, y_data) | Device-local | `STORAGE_BUFFER` | Per-batch (upload, use, free) |
| `BATCH_INTERMEDIATE` (activations, partials, gradients) | Device-local | `STORAGE_BUFFER` | Per-batch |
| `BATCH_OUTPUT` (final probs) | Device-local | `STORAGE_BUFFER \| TRANSFER_SRC` | Per-batch |
| Staging (D2H readback) | Host-visible, host-coherent | `TRANSFER_DST` | Persistent (reused across batches) |
| Upload staging (H2D) | Host-visible, host-coherent | `TRANSFER_SRC` | Transient (freed after upload) |
| Offset lists (reduction indirection) | Device-local | `STORAGE_BUFFER` | Per-plan (freed with batch) |
| Ping-pong staging (reduction engine) | Device-local | `STORAGE_BUFFER` | Per-plan (two buffers, alternating src/dst) |

---

## 9. Descriptor Set Strategy

| Category | Kernels | Descriptor Strategy | Rationale |
| :--- | :--- | :--- | :--- |
| Fixed (main kernels) | All except aggregation/clipping | Pre-allocated `VkDescriptorSet`, updated once per batch | Buffer bindings constant within a batch |
| Dynamic (reduction engine) | `aggregate_partials`, `clip_intermediate_grad` | `vkCmdPushDescriptorSetKHR` | Source/dest buffers change every reduction stage; pre-allocating per-stage sets is wasteful |
| Per-chunk (streaming) | `backprop_shared_weights/biases`, `clip_shared_gradients` | Same fixed set, push constants updated per chunk | Buffers don't change; only chunk index changes (via push constants) |

---

## 10. Dispatch Flow

End-to-end flow for a single batch:

```
╔══════════════════ HOST ═══════════════════╗   ╔═════════════ GPU ══════════════╗
║                                           ║   ║                                ║
║  1. plan = plan_builder.build_act_plan()  ║   ║                                ║
║  2. renderer.render(plan)                 ║   ║                                ║
║     ├── allocate_buffers(plan)            ║   ║                                ║
║     ├── upload_inputs(staging → device)   ║──▶║  3. Execute copy commands       ║
║     ├── record_act_cmd(plan)              ║   ║                                ║
║     └── vkQueueSubmit(act_cmd)            ║──▶║  4. Execute Act DAG             ║
║                                           ║   ║     (forward → logits → loss    ║
║  5. future.wait()                         ║   ║      → copy to staging)         ║
║     └── vkWaitForFences(act_fence)        ║◀──║     signal act_fence            ║
║  6. probs = future.result()               ║   ║                                ║
║     └── read staging_mapped_ptr           ║   ║                                ║
║     └── strip padding → numpy array       ║   ║                                ║
║                                           ║   ║                                ║
║  7. plan = plan_builder.build_learn_plan()║   ║                                ║
║  8. renderer.render(plan)                 ║   ║                                ║
║     ├── record_learn_cmd(plan)            ║   ║                                ║
║     └── vkQueueSubmit(learn_cmd)          ║──▶║  9. Execute Learn DAG           ║
║                                           ║   ║     (grads → reduce → update)   ║
║ 10. learn_future.wait()                   ║   ║                                ║
║     └── vkWaitForFences(learn_fence)      ║◀──║     signal learn_fence          ║
║                                           ║   ║                                ║
╚═══════════════════════════════════════════╝   ╚════════════════════════════════╝
```

---

## 11. Risk Register

| Risk | Likelihood | Impact | Mitigation |
| :--- | :--- | :--- | :--- |
| `vulkan-python` API surface insufficient for required Vulkan calls | Low | Block: missing bindings for push descriptors, subgroup properties | `vulkan-python` generates from the Vulkan XML registry — full API coverage. Verify during Step 5B.1 that all required functions are accessible. |
| `VK_KHR_push_descriptor` unavailable on target hardware | Low | Performance: reduction engine falls back to pre-allocated descriptor sets | Fallback implemented in `_descriptor_manager.py`. Functional parity maintained; slight allocation overhead. |
| Validation layer not available in CI (headless GPU) | Medium | Quality: Vulkan errors undetected in CI | CI image includes Vulkan SDK with validation layers. Alternatively, structural smoke tests + Phase 5C Tier 2 tests catch functional regressions. |
| Memory type selection fails on exotic hardware (no device-local heap) | Very Low | Block: buffer allocation fails | `buffer_allocator.py` scans all memory types with required properties. Falls back to any available type that satisfies property flags. Error message lists available types for diagnosis. |
| Staging buffer persistent mapping not supported | Very Low | Block: `VulkanRetrievalFuture.result()` cannot read | `HOST_VISIBLE` + `HOST_COHERENT` memory types with `MAPPED_BIT` are universally supported for transfer-destination buffers. |
| Resource leak from improper Vulkan object destruction order | Medium | Quality: validation warnings, potential crashes | Strict creation-order tracking (§6). `context.destroy()` and `renderer.destroy()` follow documented reverse-creation-order teardown. Smoke tests validate clean teardown. |
| Command buffer recording overhead for large Learn plans | Low | Performance: not correctness | Command recording is CPU-side and fast. The Learn phase has ~50–80 recorded commands even for large models. Not a bottleneck. |
| `importlib.resources` cannot locate `.spv` files after install | Medium | Block: pipeline creation fails (no shader modules) | Same discovery pattern used by the CPU backend for `libcpu_kernels.so` (proven in Phase 3B). Directory structure mirrors CPU pattern. |
| Cross-backend numerical divergence exceeds tolerance | High | Parity: Tier 3 failures | Expected. Vulkan's subgroup operations may use different accumulation order than OpenCL/CPU. Tier 3 tolerances (Phase 4) configured separately for Vulkan. Not a Phase 5B risk — surfaces in Phase 5C. |
