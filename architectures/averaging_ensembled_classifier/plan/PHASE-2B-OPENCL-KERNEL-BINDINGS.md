# Phase 2B: OpenCL KernelBinding Adaptation — Detailed Plan

**Status:** Not started  
**Phase:** 2B of 6 (sub-phase B of 3)  
**Objective:** Refactor the existing `KernelSignature` hierarchy into `KernelBinding` dispatch adapters that consume `KernelContract`s and translate plan-level node descriptions into concrete PyOpenCL `clEnqueueNDRange` calls. Implement the reduction tree and streaming loop rendering logic. After this sub-phase, the `OpenCLPlanRenderer` can fully dispatch any `ExecutionPlan` produced by the plan builder.  
**Governing ADRs:** ADR-001 (three-tier jurisdictional model), ADR-002 (plan node types), ADR-003 (reduction tree rendering), ADR-004 (streaming loop rendering), ADR-005 (Node 16 opacity), ADR-007 (KernelContract/KernelBinding split), ADR-011 (CCE/BCE strategy delegation — mixed model), ADR-013 (kernel source strategy — OpenCL reference implementations)  
**Rollback gate:** Tier 1 green + OpenCL renderer can dispatch complete Act and Learn plans without error. Full per-kernel numerical correctness deferred to Phase 2C.  
**Dependencies:** Phase 2A (renderer infrastructure) — provides `OpenCLPlanRenderer`, `OpenCLBufferAllocator`, `OpenCLContext`, `OpenCLRetrievalFuture`, `discover_hardware`, `build_compiler_flags`.

---

## Table of Contents

1. [Scope & Constraints](#1-scope--constraints)
2. [Current State After Phase 2A](#2-current-state-after-phase-2a)
3. [Target Deliverables](#3-target-deliverables)
4. [Task Breakdown](#4-task-breakdown)
   - [Step 2B.1: Define `KernelBinding` interface](#step-2b1-define-kernelbinding-interface)
   - [Step 2B.2: Implement per-kernel `KernelBinding` subclasses](#step-2b2-implement-per-kernel-kernelbinding-subclasses)
   - [Step 2B.3: Build the binding dispatch table](#step-2b3-build-the-binding-dispatch-table)
   - [Step 2B.4: Implement `_render_kernel_dispatch()` in the renderer](#step-2b4-implement-_render_kernel_dispatch-in-the-renderer)
   - [Step 2B.5: Implement `_render_reduction_tree()` in the renderer](#step-2b5-implement-_render_reduction_tree-in-the-renderer)
   - [Step 2B.6: Implement `_render_streaming_loop()` in the renderer](#step-2b6-implement-_render_streaming_loop-in-the-renderer)
   - [Step 2B.7: Kernel compilation integration](#step-2b7-kernel-compilation-integration)
   - [Step 2B.8: End-to-end renderer integration test](#step-2b8-end-to-end-renderer-integration-test)
   - [Step 2B.9: Validate rollback gate](#step-2b9-validate-rollback-gate)
5. [KernelBinding Architecture](#5-kernelbinding-architecture)
6. [Binding Extraction from KernelSignature](#6-binding-extraction-from-kernelsignature)
7. [Reduction Tree Rendering](#7-reduction-tree-rendering)
8. [Streaming Loop Rendering](#8-streaming-loop-rendering)
9. [Kernel Inventory and Binding Map](#9-kernel-inventory-and-binding-map)
10. [Risk Register](#10-risk-register)

---

## 1. Scope & Constraints

### In scope

- Defining the `KernelBinding` base class / interface in `src/backends/opencl/kernel_bindings/` — the per-backend dispatch adapter that translates abstract plan-level parameters into concrete OpenCL kernel arguments (ADR-007).
- Refactoring each existing `KernelSignature` subclass into a `KernelBinding` subclass. The binding consumes the node's `buffer_bindings`, `scalar_params`, `tile_count`, and `placement_strategy` and produces `clEnqueueNDRange` calls with correctly ordered arguments, grid dimensions, and local memory allocations.
- Building the binding dispatch table: a `dict[str, KernelBinding]` mapping `kernel_name` → binding instance, injected into the `OpenCLPlanRenderer` at initialization.
- Implementing the `_render_kernel_dispatch()` method in the renderer — the per-tile imperative dispatch loop for `KernelDispatchNode`s.
- Implementing `_render_reduction_tree()` — the multi-stage rendering loop for `ReductionTreeNode`s, including kernel tier selection (identity / register-reduce / local-reduce), intermediate ping-pong buffer management, offset list upload, and per-stage `cl.Event` chaining.
- Implementing `_render_streaming_loop()` — the per-chunk parametric dispatch loop for `StreamingLoopNode`s, including per-chunk `ParameterStride` instantiation and scratch buffer reuse.
- Integrating kernel source compilation (`cl.Program` from `kernels/*.cl.c` + `kernels/kernels.cl.h`) with the build flags from `type_mapping.py`.
- Writing end-to-end integration tests that dispatch complete Act and Learn plans through the renderer and verify structural completion (not numerical correctness — that's Phase 2C).

### Out of scope

- Writing per-kernel numerical correctness tests (Phase 2C).
- Modifying kernel source files (`kernels/*.cl.c`, `kernels/kernels.cl.h`).
- Modifying shared-layer code (plan model, plan builder, contracts).
- Implementing Vulkan or CPU backends.
- Removing the legacy `KernelSignature` classes — they coexist until Phase 6.
- Implementing the `WorkTicket` / `LearnHandle` / `Engine` user-facing API.

### Key constraint: extraction, not rewrite

The existing `KernelSignature` subclasses contain production-validated argument marshalling logic. Phase 2B extracts this logic into `KernelBinding` classes — it does not rewrite it. The argument ordering, `numpy` type coercion, `cl.LocalMemory` sizing, and grid computation are preserved verbatim. The only structural change is that the binding receives abstract `BufferHandle` → `cl.Buffer` mappings (from the allocator) instead of calling `BufferManager.get_cl_buffer()` directly.

### Key constraint: legacy coexistence

The legacy `KernelSignature` classes in `src/backends/opencl/kernel_bindings/` remain intact. The new `KernelBinding` classes are placed alongside them. Both paths are functional: `batch_processor.py` continues to use `KernelSignature`; `OpenCLPlanRenderer` uses `KernelBinding`. Phase 6 removes the legacy path.

---

## 2. Current State After Phase 2A

### Renderer state

| Component | Status |
| :--- | :--- |
| `OpenCLPlanRenderer` | DAG traversal loop implemented; `BarrierNode` and `RetrievalNode` handling functional; `KernelDispatchNode`, `ReductionTreeNode`, `StreamingLoopNode` dispatch methods are stubs |
| `OpenCLBufferAllocator` | Functional — allocates `cl.Buffer` from `BufferDescriptor` |
| `OpenCLRetrievalFuture` | Functional — wraps async read events |
| `discover_hardware()` | Functional — populates `HardwareProfile` |
| `build_compiler_flags()` | Functional — generates `-D` flags |
| `OpenCLContext` | Functional — holds context, queue, device |

### Legacy `KernelSignature` classes (`src/backends/opencl/kernel_bindings/`)

The existing `kernel_bindings/` directory contains `KernelSignature` subclasses for every kernel. Each class fuses validation (from `__post_init__`) and OpenCL argument marshalling (from `get_args()` and `get_grid()`). The validation logic is now redundant with the `KernelContract` validation performed at plan-construction time by the plan builder. The argument marshalling logic is the extraction target.

| Phase File | Signature Classes | Kernels |
| :--- | :--- | :--- |
| `phase_1_act.py` | `ForwardPassSignature`, `ComputeHiddenMaskSignature`, `RenderLogitsChunkSignature` | `forward_pass`, `compute_hidden_mask`, `render_logits_chunk` |
| `phase_2_learn_A_production.py` | `ComputeProbsLossCceChunkSignature`, `ComputeProbsLossBceChunkSignature`, `CalculateModuleParamGradsCceSignature`, `CalculateModuleParamGradsBceSignature` | `compute_probs_loss_cce_chunk`, `compute_probs_loss_bce_chunk`, `calculate_module_param_grads_cce`, `calculate_module_param_grads_bce` |
| `phase_2_learn_B_processing.py` | `BackpropErrorToHiddenCceSignature`, `BackpropErrorToHiddenBceSignature`, `CalculateChunkTempGradientsCceSignature`, `CalculateChunkTempGradientsBceSignature`, `ClipPartialGradientsSignature` | `backprop_error_to_hidden_cce/bce`, `calculate_chunk_temp_gradients_cce/bce`, `clip_partial_gradients` |
| `phase_2_learn_C_reduction.py` | `GatherAndPermuteGradHSignature`, `AggregateRegisterReduceSignature`, `AggregateLocalReduceSignature`, `ClipIntermediateGradSignature`, `StabilizeReduceGradHSignature` | `gather_and_permute_grad_h`, `aggregate_register_reduce`, `aggregate_local_reduce`, `clip_intermediate_grad`, `stabilize_reduce_grad_h` |
| `phase_2_learn_D_backprop.py` | `BackpropSharedWeightsSignature`, `BackpropSharedBiasesSignature`, `ClipSharedGradientsSignature` | `backprop_shared_weights`, `backprop_shared_biases`, `clip_shared_gradients` |
| `phase_3_update.py` | `NormalizeGradientsSignature`, `AdamUpdateSignature`, `ClampTemperaturesSignature` | `normalize_gradients`, `adam_update`, `clamp_temperatures` |

---

## 3. Target Deliverables

After Phase 2B completes:

```
src/backends/opencl/
├── renderer.py                         # UPDATED: full dispatch logic (no more stubs)
├── kernel_bindings/
│   ├── __init__.py                     # UPDATED: exports KernelBinding + dispatch table
│   ├── base.py                         # NEW: KernelBinding base class
│   ├── binding_phase_1_act.py          # NEW: Act-phase bindings
│   ├── binding_phase_2_learn_A.py      # NEW: Learn-A bindings
│   ├── binding_phase_2_learn_B.py      # NEW: Learn-B bindings
│   ├── binding_phase_2_learn_C.py      # NEW: Learn-C bindings (incl. aggregation)
│   ├── binding_phase_2_learn_D.py      # NEW: Learn-D bindings
│   ├── binding_phase_3_update.py       # NEW: Update-phase bindings
│   ├── dispatch_table.py              # NEW: kernel_name → KernelBinding mapping
│   ├── phase_1_act.py                 # UNCHANGED (legacy KernelSignature)
│   ├── phase_2_learn_A_production.py  # UNCHANGED (legacy)
│   ├── phase_2_learn_B_processing.py  # UNCHANGED (legacy)
│   ├── phase_2_learn_C_reduction.py   # UNCHANGED (legacy)
│   ├── phase_2_learn_D_backprop.py    # UNCHANGED (legacy)
│   └── phase_3_update.py             # UNCHANGED (legacy)
└── ... (Phase 2A modules unchanged)
```

---

## 4. Task Breakdown

### Step 2B.1: Define `KernelBinding` interface

**Action:** Create `src/backends/opencl/kernel_bindings/base.py` containing the `KernelBinding` base class for the OpenCL backend.

**Design:**

```python
from abc import ABC, abstractmethod
from typing import Any

import pyopencl as cl

from ....shared.buffer_lifecycle import BufferHandle


class KernelBinding(ABC):
    """Base class for OpenCL kernel dispatch adapters (ADR-007).

    Translates abstract plan-level parameters (buffer handles, scalar
    values, tile index) into concrete OpenCL kernel arguments and
    dispatch dimensions.

    Each subclass corresponds to one kernel_name and implements:
    - get_kernel_name() → the cl.Program kernel attribute name
    - compute_grid(tile_index, scalar_params) → (global_size, local_size)
    - marshal_args(allocator, buffer_bindings, scalar_params, tile_index)
        → ordered list of cl kernel arguments
    """

    @abstractmethod
    def get_kernel_name(self) -> str:
        """Return the kernel function name as it appears in kernels.cl.h."""
        ...

    @abstractmethod
    def compute_grid(
        self,
        tile_index: int,
        scalar_params: dict[str, int | float],
        hardware_simd_width: int,
    ) -> tuple[tuple[int, ...], tuple[int, ...]]:
        """Compute (global_size, local_size) for one tile dispatch.

        Returns the NDRange dimensions for a single clEnqueueNDRange call.
        """
        ...

    @abstractmethod
    def marshal_args(
        self,
        get_buffer: "Callable[[BufferHandle], cl.Buffer]",
        buffer_bindings: dict[str, BufferHandle],
        scalar_params: dict[str, int | float],
        tile_index: int,
    ) -> list[Any]:
        """Marshal the ordered argument list for clEnqueueNDRange.

        The argument list includes cl.Buffer objects, numpy scalars, and
        cl.LocalMemory objects, in the order expected by the kernel's
        function signature.

        get_buffer: callable that resolves BufferHandle → cl.Buffer
                    (from OpenCLBufferAllocator.get_buffer)
        """
        ...
```

**Key design decisions:**

1. **`get_buffer` callable instead of direct allocator reference:** The binding receives a `get_buffer` function rather than the full `OpenCLBufferAllocator`. This keeps the binding decoupled from the allocator's internal state and enables testing with mock buffer providers.

2. **Tile index as explicit parameter:** The binding receives `tile_index` as an explicit parameter. It translates this into the `flat_tile_index` scalar argument that the OpenCL kernel expects (the host-provided tile index from DESIGN.md §7.1). This is the OpenCL-specific tile delivery mechanism — Vulkan uses `gl_WorkGroupID.x`, CPU uses `task_index`.

3. **No validation in bindings:** All contract validation (shape checks, padding, calculability proofs) occurred at plan-construction time via `KernelContract`. The binding trusts that the plan is valid. This avoids duplicating validation logic across backends and honors ADR-007's split.

---

### Step 2B.2: Implement per-kernel `KernelBinding` subclasses

**Action:** For each kernel in the system, create a `KernelBinding` subclass that extracts the argument marshalling logic from the corresponding legacy `KernelSignature` class.

**Extraction process for each kernel:**

1. Open the legacy `KernelSignature` subclass (e.g., `ForwardPassSignature`).
2. Identify `get_args()` — this becomes `marshal_args()`.
3. Identify `get_grid()` — this becomes `compute_grid()`.
4. Replace `self.buffer_manager.get_cl_buffer(handle)` calls with `get_buffer(buffer_bindings[param_name])`.
5. Replace `self.arch_constants.simd_width` references with the `hardware_simd_width` parameter.
6. Replace `numpy` type coercion for scalars (e.g., `np.uint32(value)`) preserving exact types.
7. Replace `cl.LocalMemory(size)` calls preserving exact size expressions.
8. Remove `__post_init__` validation logic (moved to `KernelContract` in Phase 1).

**Binding file organization (one file per DAG phase):**

| Binding File | Bindings | Source Signatures |
| :--- | :--- | :--- |
| `binding_phase_1_act.py` | `ForwardPassBinding`, `RenderLogitsChunkBinding` | `ForwardPassSignature`, `RenderLogitsChunkSignature` |
| `binding_phase_2_learn_A.py` | `ComputeProbsLossCceBinding`, `ComputeProbsLossBceBinding`, `CalculateModuleParamGradsCceBinding`, `CalculateModuleParamGradsBceBinding` | Corresponding `Signature` classes |
| `binding_phase_2_learn_B.py` | `BackpropErrorToHiddenBinding`, `CalculateChunkTempGradientsBinding`, `ClipPartialGradientsBinding` | Note: `BackpropErrorToHidden` and `CalculateChunkTempGradients` use Strategy A (host-injected `FLAG__problem_type` scalar); a single binding handles both CCE/BCE via the flag value in `scalar_params` |
| `binding_phase_2_learn_C.py` | `GatherAndPermuteGradHBinding`, `AggregateRegisterReduceBinding`, `AggregateLocalReduceBinding`, `ClipIntermediateGradBinding`, `StabilizeReduceGradHBinding` | Corresponding `Signature` classes |
| `binding_phase_2_learn_D.py` | `BackpropSharedWeightsBinding`, `BackpropSharedBiasesBinding`, `ClipSharedGradientsBinding` | Corresponding `Signature` classes |
| `binding_phase_3_update.py` | `NormalizeGradientsBinding`, `AdamUpdateBinding`, `ClampTemperaturesBinding` | Corresponding `Signature` classes |

**CCE/BCE Strategy handling in bindings:**

Per ADR-011's mixed delegation model:
- **Strategy B (Nodes 6/7):** Separate bindings for CCE and BCE (`ComputeProbsLossCceBinding`, `ComputeProbsLossBceBinding`) because the kernels have genuinely different interfaces and parameter sets.
- **Strategy A (Nodes 8/9/10):** Single binding per kernel (e.g., `BackpropErrorToHiddenBinding`) that reads the `src_scalar_FLAG_problem_type` value from `scalar_params`. The flag value is set by the plan builder; the binding passes it through to the OpenCL kernel unchanged.

**Reduction engine bindings (Phase 2 Learn C):**

The reduction-related bindings are special because they are consumed by the reduction tree renderer (Step 2B.5), not directly by `_render_kernel_dispatch()`:
- `AggregateRegisterReduceBinding` — for small fan-in stages ($K \leq$ register threshold).
- `AggregateLocalReduceBinding` — for large fan-in stages ($K >$ register threshold).
- `ClipIntermediateGradBinding` — for inter-stage clipping in `"sum_and_clip"` trees.

The renderer selects between register and local bindings per stage based on `fan_in_K` and a heuristic crossover threshold.

**Validation:** Each binding's `marshal_args()` produces an argument list of the correct length and type. Unit tests compare binding output against legacy `KernelSignature.get_args()` output for identical inputs.

---

### Step 2B.3: Build the binding dispatch table

**Action:** Create `src/backends/opencl/kernel_bindings/dispatch_table.py` containing the dispatch table factory.

```python
from .base import KernelBinding
from .binding_phase_1_act import ForwardPassBinding, RenderLogitsChunkBinding
from .binding_phase_2_learn_A import (
    ComputeProbsLossCceBinding, ComputeProbsLossBceBinding,
    CalculateModuleParamGradsCceBinding, CalculateModuleParamGradsBceBinding,
)
# ... all other bindings


def build_dispatch_table() -> dict[str, KernelBinding]:
    """Construct the kernel_name → KernelBinding dispatch table.

    This table is injected into OpenCLPlanRenderer at initialization.
    Each entry maps a kernel_name (as it appears in KernelDispatchNode.kernel_name
    and in kernels.cl.h) to the KernelBinding instance that can dispatch it.
    """
    return {
        "forward_pass": ForwardPassBinding(),
        "render_logits_chunk": RenderLogitsChunkBinding(),
        "compute_probs_loss_cce_chunk": ComputeProbsLossCceBinding(),
        "compute_probs_loss_bce_chunk": ComputeProbsLossBceBinding(),
        "calculate_module_param_grads_cce": CalculateModuleParamGradsCceBinding(),
        "calculate_module_param_grads_bce": CalculateModuleParamGradsBceBinding(),
        "backprop_error_to_hidden": BackpropErrorToHiddenBinding(),
        "calculate_chunk_temp_gradients": CalculateChunkTempGradientsBinding(),
        "clip_partial_gradients": ClipPartialGradientsBinding(),
        "gather_and_permute_grad_h": GatherAndPermuteGradHBinding(),
        "stabilize_reduce_grad_h": StabilizeReduceGradHBinding(),
        "backprop_shared_weights": BackpropSharedWeightsBinding(),
        "backprop_shared_biases": BackpropSharedBiasesBinding(),
        "clip_shared_gradients": ClipSharedGradientsBinding(),
        "normalize_gradients": NormalizeGradientsBinding(),
        "adam_update": AdamUpdateBinding(),
        "clamp_temperatures": ClampTemperaturesBinding(),
        # Reduction engine bindings (consumed by _render_reduction_tree, not dispatch table):
        # "aggregate_register_reduce" and "aggregate_local_reduce" are internal to the
        # reduction tree renderer and not in the plan's KernelDispatchNode vocabulary.
    }
```

**Reduction engine bindings note:** The `aggregate_register_reduce`, `aggregate_local_reduce`, and `clip_intermediate_grad` bindings are not in the main dispatch table because they are not referenced by `KernelDispatchNode`s in the plan. `ReductionTreeNode`s are rendered atomically by the renderer — the internal kernel selection is the renderer's decision. These bindings are accessed directly by `_render_reduction_tree()`.

---

### Step 2B.4: Implement `_render_kernel_dispatch()` in the renderer

**Action:** Implement the full `KernelDispatchNode` dispatch logic in `OpenCLPlanRenderer`.

**Algorithm:**

```python
def _render_kernel_dispatch(
    self,
    node: KernelDispatchNode,
    wait_for: list[cl.Event],
) -> cl.Event:
    """Dispatch a KernelDispatchNode via per-tile imperative enqueue.

    For tile_count=N, loops N times, calling clEnqueueNDRange with
    per-tile flat_tile_index. Returns a marker event joining all tile
    completion events.
    """
    binding = self._bindings[node.kernel_name]
    kernel = getattr(self._program, binding.get_kernel_name())

    tile_events = []
    for tile_idx in range(node.tile_count):
        args = binding.marshal_args(
            get_buffer=self._allocator.get_buffer,
            buffer_bindings=node.buffer_bindings,
            scalar_params=node.scalar_params,
            tile_index=tile_idx,
        )
        global_size, local_size = binding.compute_grid(
            tile_index=tile_idx,
            scalar_params=node.scalar_params,
            hardware_simd_width=self._hardware.simd_width,
        )
        event = cl.enqueue_nd_range_kernel(
            self._queue, kernel, global_size, local_size,
            *args, wait_for=wait_for,
        )
        tile_events.append(event)

    # Aggregate tile events into a single marker for downstream nodes
    if len(tile_events) == 1:
        return tile_events[0]
    return cl.enqueue_marker(self._queue, wait_for=tile_events)
```

**Tile-count=1 optimization:** Kernels with `tile_count=1` (e.g., `normalize_gradients`, `clamp_temperatures`, `gather_and_permute_grad_h`) skip the loop and dispatch once with `tile_index=0`.

**Argument passing convention:** PyOpenCL's `enqueue_nd_range_kernel` accepts positional arguments after the kernel. The `marshal_args()` return value is splatted: `kernel(queue, global_size, local_size, *args, wait_for=wait_for)`. This matches the existing `KernelExecutor.launch()` calling convention.

---

### Step 2B.5: Implement `_render_reduction_tree()` in the renderer

**Action:** Implement the multi-stage reduction tree rendering loop in `OpenCLPlanRenderer`, consuming `ReductionTreePlan` from `ReductionTreeNode`.

**This is the most complex rendering logic in Phase 2B.** It mirrors the existing `_execute_reduction_pipeline` in `graph_recipes.py` but operates on the plan model's `ReductionTreePlan` instead of ad-hoc parameters.

**Algorithm (from DESIGN.md §3.2, ADR-003, CONCEPT.md §2):**

```
def _render_reduction_tree(
    self,
    node: ReductionTreeNode,
    wait_for: list[cl.Event],
) -> cl.Event:
    plan = node.reduction_plan
    N = plan.num_partials
    K = plan.fan_in_K

    # 1. Upload initial offset list to device
    offset_buf = upload_offset_list(plan.initial_offset_list)

    # 2. Allocate ping-pong intermediate buffers (renderer-internal)
    ping = allocate_intermediate(plan)
    pong = allocate_intermediate(plan)

    # 3. For each stage (leaf → root):
    source = get_buffer(plan.source_buffer)
    current_N = N
    prev_event = wait_for

    for stage in range(plan.num_stages):
        # a. Select kernel tier
        if current_N <= 1:
            # Identity — skip this stage
            continue
        elif current_N <= REGISTER_REDUCE_THRESHOLD:
            binding = self._register_reduce_binding
        else:
            binding = self._local_reduce_binding

        # b. Compute output count
        output_N = ceil(current_N / K)

        # c. Dispatch aggregation kernel
        agg_event = dispatch_reduction_stage(
            binding, source, ping, offset_buf, current_N, K,
            plan.elements_per_partial, prev_event,
        )

        # d. If "sum_and_clip" and threshold is not None: dispatch clip
        if plan.tree_variant == "sum_and_clip" and plan.threshold_schedule[stage] is not None:
            clip_event = dispatch_clip_stage(
                ping, plan.threshold_schedule[stage],
                output_N, plan.elements_per_partial,
                wait_for=[agg_event],
            )
            prev_event = [clip_event]
        else:
            prev_event = [agg_event]

        # e. Swap ping-pong buffers
        source = ping
        ping, pong = pong, ping

        # f. Compute contiguous offset list for next stage
        offset_buf = make_contiguous_offsets(output_N)
        current_N = output_N

    # 4. Copy final result to destination buffer
    copy_event = enqueue_copy(source, get_buffer(plan.destination_buffer), prev_event)
    return copy_event
```

**Kernel tier selection:**

| Tier | Condition | Kernel | Mechanism |
| :--- | :--- | :--- | :--- |
| Identity | `current_N == 1` | None | Skip stage — source is already the result |
| Register-reduce | `current_N <= REGISTER_REDUCE_THRESHOLD` | `aggregate_register_reduce` | Each work-item reduces $K$ partials using register accumulation; no local memory |
| Local-reduce | `current_N > REGISTER_REDUCE_THRESHOLD` | `aggregate_local_reduce` | Workgroup loads $K$ partials into local memory and performs tree reduction with barriers |

`REGISTER_REDUCE_THRESHOLD` is a heuristic constant (typically 16–32) derived from the existing `graph_recipes.py` logic (`max_reg_agg = 16`). It is a rendering-tier decision — not plan-level.

**Intermediate buffer management:**

The ping-pong buffers are renderer-internal (ADR-009 two-tier scope). They are not plan-level `BufferDescriptor`s. The renderer allocates them at the start of `_render_reduction_tree()` and frees them when the tree completes.

**Offset list handling:**

- **Stage 0:** `plan.initial_offset_list` is uploaded to device memory as a `cl.Buffer`. These offsets point to scattered partial results in the source buffer (the Indirection Contract from CONCEPT.md §2).
- **Stage > 0:** Intermediate results are contiguous (the ping-pong buffers are laid out sequentially). The renderer computes trivial contiguous offsets: `[0, elements_per_partial, 2 * elements_per_partial, ...]`.

**Extraction from `graph_recipes.py`:**

The existing `_execute_reduction_pipeline` function in `graph_recipes.py` implements this exact logic using the legacy `Services`/`BufferManager`/`KernelExecutor` infrastructure. Phase 2B extracts the algorithmic core (stage loop, tier selection, ping-pong, offset management) and rewrites the dispatch calls to use `KernelBinding` and `OpenCLBufferAllocator`.

---

### Step 2B.6: Implement `_render_streaming_loop()` in the renderer

**Action:** Implement the streaming loop rendering logic in `OpenCLPlanRenderer`, consuming `StreamingLoopPlan` from `StreamingLoopNode`.

**Algorithm (from DESIGN.md §3.3, ADR-004):**

```
def _render_streaming_loop(
    self,
    node: StreamingLoopNode,
    wait_for: list[cl.Event],
) -> cl.Event:
    plan = node.streaming_plan

    # 1. Allocate scratch buffers (renderer-internal, reused across chunks)
    scratch = {
        spec.logical_name: allocate_scratch(spec.size_bytes)
        for spec in plan.scratch_buffers
    }

    chunk_events = wait_for

    # 2. For each chunk:
    for chunk_idx in range(plan.iteration.chunk_count):
        # a. Compute per-chunk scalar parameters from strides
        chunk_scalars = dict(plan.constant_scalars)
        for stride in plan.parameter_strides:
            chunk_scalars[stride.param_name] = stride.base + chunk_idx * stride.stride

        # b. Dispatch each body node in sequence
        body_events = chunk_events
        for body_node_id in plan.body:
            body_node = self._plan.nodes[body_node_id]
            assert isinstance(body_node, KernelDispatchNode)

            # Override scalar_params with chunk-specific values
            merged_scalars = {**body_node.scalar_params, **chunk_scalars}
            modified_node = body_node  # conceptual — actual override via binding

            event = self._render_kernel_dispatch_with_overrides(
                body_node, merged_scalars, scratch, body_events,
            )
            body_events = [event]

        chunk_events = body_events

    return chunk_events[0] if chunk_events else wait_for[0]
```

**Per-chunk parameter instantiation:**

The `ParameterStride` mechanism from ADR-004:
- Each `ParameterStride` defines `(param_name, base, stride)`.
- For chunk index `i`, the parameter value is `base + i * stride`.
- Common use case: `batch_chunk_offset = 0 + i * chunk_size` for streaming over batch chunks.

**Scratch buffer reuse:**

Scratch buffers from `ScratchBufferSpec` are allocated once before the loop and reused each iteration. The renderer maps the scratch buffer's `logical_name` to a physical `cl.Buffer`, which is provided to body node bindings via the buffer resolution mechanism.

**Body node sequencing:**

Within each chunk iteration, body nodes are dispatched sequentially. The plan builder guarantees that `plan.body` is in valid execution order (no dependency violations within the body). Between chunks, the entire body sequence completes before the next chunk begins — this is the streaming model.

**Extraction from `graph_recipes.py`:**

The existing streaming backprop logic in `graph_recipes.py` (`_build_streaming_module_grad_path` and related functions) implements this pattern for Nodes 17→18→19. Phase 2B extracts the generic loop structure and parameterizes it via `StreamingLoopPlan`.

---

### Step 2B.7: Kernel compilation integration

**Action:** Integrate OpenCL kernel source loading and compilation into the renderer's initialization.

**Kernel source discovery:** Kernel source files reside in `kernels/` at the architecture root (ADR-013). The renderer loads them via `importlib.resources`:

```python
import importlib.resources

def load_and_compile_kernels(
    context: cl.Context,
    device: cl.Device,
    compiler_flags: list[str],
) -> cl.Program:
    """Load kernel sources from the architecture's kernels/ directory and compile."""
    # Load header
    header_src = importlib.resources.files(
        "averaging_ensembled_classifier.kernels"
    ).joinpath("kernels.cl.h").read_text()

    # Load all phase source files
    phase_files = [
        "phase_1_act.cl.c",
        "phase_2_learn_A_production.cl.c",
        "phase_2_learn_B_processing.cl.c",
        "phase_2_learn_C_reduction.cl.c",
        "phase_2_learn_D_backprop.cl.c",
        "phase_3_update.cl.c",
    ]
    sources = [header_src]
    for fname in phase_files:
        sources.append(
            importlib.resources.files("averaging_ensembled_classifier.kernels")
            .joinpath(fname).read_text()
        )

    program = cl.Program(context, "\n".join(sources))
    program.build(options=" ".join(compiler_flags), devices=[device])
    return program
```

**Compiler flag source:** `build_compiler_flags()` from `type_mapping.py` (Phase 2A).

**Build once:** The program is compiled once at renderer initialization and stored in the `OpenCLContext`. Re-used across all `render()` calls. Kernel function handles are obtained via `getattr(program, kernel_name)`.

---

### Step 2B.8: End-to-end renderer integration test

**Action:** Write integration tests that construct an `ExecutionPlan` (via the plan builder), render it through `OpenCLPlanRenderer`, and verify structural completion.

**Test location:** `tests/tier2/opencl/test_renderer_e2e.py`

**Test cases:**

1. **`test_render_act_plan`** — Build an Act plan for a small model (e.g., Iris-scale: 4 inputs, 8 hidden, 3 classes, 2 modules); render via `OpenCLPlanRenderer`; verify `render()` returns a `dict` with `"inference_retrieval"` key mapping to a valid `RetrievalFuture`; verify `future.wait()` completes without timeout.

2. **`test_render_learn_plan`** — Build a Learn plan; render; verify `"final_batch_retrieval"` future completes.

3. **`test_render_act_then_learn`** — Render Act plan, retrieve predictions, then render Learn plan. Validates the full batch lifecycle.

4. **`test_render_with_reduction_tree`** — Build a Learn plan with a model geometry that produces multi-stage reduction trees (e.g., `num_modules=16`, requiring $\log_K(16)$ stages); render; verify completion.

5. **`test_render_with_streaming_loop`** — Build a Learn plan with multiple batch chunks (triggering Phase III streaming loop); render; verify completion.

**Note:** These tests verify structural completion (no crashes, no hangs, futures resolve). Numerical correctness of kernel outputs is Phase 2C.

**Skip logic:** `@pytest.mark.skipif(not _build_config.BACKEND_OPENCL, ...)`.

---

### Step 2B.9: Validate rollback gate

**Procedure:**

1. `pytest tests/tier1/ -v` — all Tier 1 tests pass.
2. `pytest src/tests/ tests/ -v` — all existing tests (332) pass.
3. `pytest tests/tier2/opencl/test_renderer_infrastructure.py tests/tier2/opencl/test_renderer_e2e.py -v` — infrastructure + e2e tests pass.

---

## 5. KernelBinding Architecture

The `KernelBinding` sits at the Orchestration tier — between the plan's abstract node description and the OpenCL runtime's concrete dispatch API:

```
┌─────────────────────────────────────────────────────────────┐
│  Plan Layer (Policy Tier)                                   │
│                                                             │
│  KernelDispatchNode                                         │
│    kernel_name: "forward_pass"                              │
│    buffer_bindings: {"input": BH(0), "weights": BH(1), ...}│
│    scalar_params: {"batch_count": 32, "hidden_count": 64}   │
│    tile_count: 4                                            │
│    placement_strategy: "grid_mod_cls"                       │
└──────────────────────────┬──────────────────────────────────┘
                           │ consumed by
                           ▼
┌─────────────────────────────────────────────────────────────┐
│  KernelBinding (Orchestration Tier)                         │
│                                                             │
│  ForwardPassBinding                                         │
│    marshal_args(get_buffer, bindings, scalars, tile_idx)    │
│      → [cl.Buffer, cl.Buffer, ..., np.uint32(tile_idx),    │
│         np.uint32(batch_count), cl.LocalMemory(size), ...]  │
│    compute_grid(tile_idx, scalars, simd_width)              │
│      → ((global_width,), (local_width,))                    │
└──────────────────────────┬──────────────────────────────────┘
                           │ dispatched to
                           ▼
┌─────────────────────────────────────────────────────────────┐
│  OpenCL Runtime (Execution Tier)                            │
│                                                             │
│  clEnqueueNDRange(queue, forward_pass_kernel,               │
│                   global_size, local_size, *args)           │
└─────────────────────────────────────────────────────────────┘
```

---

## 6. Binding Extraction from KernelSignature

The extraction follows a mechanical process for each kernel:

**Example: `ForwardPassSignature` → `ForwardPassBinding`**

**Legacy `get_args()` (simplified):**
```python
def get_args(self):
    return [
        self.buffer_manager.get_cl_buffer(self.input_handle),
        self.buffer_manager.get_cl_buffer(self.sample_mask_handle),
        self.buffer_manager.get_cl_buffer(self.weights_shared_handle),
        self.buffer_manager.get_cl_buffer(self.biases_shared_handle),
        self.buffer_manager.get_cl_buffer(self.hidden_act_handle),
        self.buffer_manager.get_cl_buffer(self.hidden_mask_handle),
        np.uint32(self.flat_tile_index),
        np.uint32(self.batch_chunk_offset),
        np.uint32(self.batch_chunk_count),
        np.uint32(self.total_batch_count),
        np.uint32(self.padded_input_count),
        np.uint32(self.padded_hidden_count),
    ]
```

**New `marshal_args()`:**
```python
def marshal_args(self, get_buffer, buffer_bindings, scalar_params, tile_index):
    return [
        get_buffer(buffer_bindings["input"]),
        get_buffer(buffer_bindings["sample_mask"]),
        get_buffer(buffer_bindings["weights_shared_simd_major"]),
        get_buffer(buffer_bindings["biases_shared"]),
        get_buffer(buffer_bindings["hidden_activations"]),
        get_buffer(buffer_bindings["hidden_mask"]),
        np.uint32(tile_index),                          # flat_tile_index
        np.uint32(scalar_params["batch_chunk_offset"]),
        np.uint32(scalar_params["batch_chunk_count"]),
        np.uint32(scalar_params["total_batch_count"]),
        np.uint32(scalar_params["padded_input_count"]),
        np.uint32(scalar_params["padded_hidden_count"]),
    ]
```

The transformation is systematic:
- `self.buffer_manager.get_cl_buffer(self.X_handle)` → `get_buffer(buffer_bindings["X"])`
- `self.flat_tile_index` → `tile_index` (explicit parameter)
- `self.scalar_field` → `scalar_params["field_name"]`
- `cl.LocalMemory(...)` preserved verbatim
- `np.uint32(...)` / `np.float32(...)` coercions preserved verbatim

---

## 7. Reduction Tree Rendering

The reduction tree renderer implements the rendering contract documented in Phase 1's `ReductionTreePlan` specification:

```
Input:  ReductionTreePlan (from ReductionTreeNode)
        ├── num_partials (N)
        ├── fan_in_K (K)
        ├── num_stages
        ├── initial_offset_list
        ├── tree_variant ("sum" or "sum_and_clip")
        ├── threshold_schedule [T₀, T₁, ..., T_{stages-1}]
        ├── source_buffer
        └── destination_buffer

Process:
  Stage 0: N partials → ⌈N/K⌉ partials (via aggregate kernel)
           Optional clip at T₀ (if "sum_and_clip")
  Stage 1: ⌈N/K⌉ partials → ⌈⌈N/K⌉/K⌉ partials
           Optional clip at T₁
  ...
  Stage S-1: K or fewer partials → 1 result
           Optional clip at T_{S-1}

Output: Single result in destination_buffer
```

**Kernel tier crossover heuristic:**

The renderer chooses between `aggregate_register_reduce` and `aggregate_local_reduce` per stage:

| Criterion | Register Reduce | Local Reduce |
| :--- | :--- | :--- |
| Partials at stage | $\leq$ `MAX_REG_AGG` (typically 16) | $>$ `MAX_REG_AGG` |
| Mechanism | One work-item sums $K$ partials in registers | Workgroup loads partials into `__local`, tree reduction with barriers |
| Local memory | None | $2 \times K \times$ `elements_per_partial` $\times$ `element_size` |

This mirrors the existing `graph_recipes.py` logic exactly. The crossover point is the `max_reg_agg` constant.

---

## 8. Streaming Loop Rendering

The streaming loop renderer implements the Phase III backpropagation pattern:

```
StreamingLoopPlan
├── iteration: {total_extent: B, chunk_count: C, chunk_size: B/C}
├── body: ["node_17", "node_18", "node_19"]
├── parameter_strides:
│   ├── ParameterStride("batch_chunk_offset", base=0, stride=chunk_size)
│   └── ParameterStride("batch_chunk_count", base=chunk_size, stride=0)
├── scratch_buffers:
│   ├── ScratchBufferSpec("partial_grad_sw", ..., (padded_hidden, padded_input))
│   └── ScratchBufferSpec("partial_grad_sb", ..., (padded_hidden,))
└── constant_scalars: {"padded_hidden_count": 64, ...}
```

For each chunk `i = 0, 1, ..., C-1`:
1. Compute `batch_chunk_offset = 0 + i × chunk_size`.
2. Dispatch `node_17` (backprop_shared_weights) with chunk-specific offset.
3. Dispatch `node_18` (backprop_shared_biases) with chunk-specific offset.
4. Dispatch `node_19` (clip_shared_gradients) with chunk-specific scalars.
5. Barrier between chunks (implicit via `wait_for` chaining).

The scratch buffers (`partial_grad_sw`, `partial_grad_sb`) are overwritten each chunk — no accumulation across chunks. The clipped partials from each chunk are written to distinct positions in a collection buffer via the placement contract.

---

## 9. Kernel Inventory and Binding Map

Complete mapping from plan `kernel_name` to `KernelBinding` and DAG node:

| `kernel_name` | KernelBinding | DAG Node(s) | Strategy | Notes |
| :--- | :--- | :--- | :--- | :--- |
| `forward_pass` | `ForwardPassBinding` | 4 | — | |
| `render_logits_chunk` | `RenderLogitsChunkBinding` | 5 | — | |
| `compute_probs_loss_cce_chunk` | `ComputeProbsLossCceBinding` | 6 | B (separate) | CCE-only |
| `compute_probs_loss_bce_chunk` | `ComputeProbsLossBceBinding` | 7 | B (separate) | BCE-only |
| `calculate_module_param_grads_cce` | `CalculateModuleParamGradsCceBinding` | 8 | B (separate) | CCE variant |
| `calculate_module_param_grads_bce` | `CalculateModuleParamGradsBceBinding` | 8 | B (separate) | BCE variant |
| `backprop_error_to_hidden` | `BackpropErrorToHiddenBinding` | 9 | A (flag) | `FLAG__problem_type` |
| `calculate_chunk_temp_gradients` | `CalculateChunkTempGradientsBinding` | 10 | A (flag) | `FLAG__problem_type` |
| `clip_partial_gradients` | `ClipPartialGradientsBinding` | 11 | — | |
| `gather_and_permute_grad_h` | `GatherAndPermuteGradHBinding` | 13 | — | |
| `stabilize_reduce_grad_h` | `StabilizeReduceGradHBinding` | 16 | — | Node 16 opacity (ADR-005) |
| `backprop_shared_weights` | `BackpropSharedWeightsBinding` | 17 | — | Streaming body |
| `backprop_shared_biases` | `BackpropSharedBiasesBinding` | 18 | — | Streaming body |
| `clip_shared_gradients` | `ClipSharedGradientsBinding` | 19 | — | Streaming body |
| `normalize_gradients` | `NormalizeGradientsBinding` | 21 | — | |
| `adam_update` | `AdamUpdateBinding` | 24 | — | Per parameter group |
| `clamp_temperatures` | `ClampTemperaturesBinding` | 25 | — | |

**Reduction tree internal bindings (not in dispatch table):**

| Kernel | Binding | Used by |
| :--- | :--- | :--- |
| `aggregate_register_reduce` | `AggregateRegisterReduceBinding` | `_render_reduction_tree()` |
| `aggregate_local_reduce` | `AggregateLocalReduceBinding` | `_render_reduction_tree()` |
| `clip_intermediate_grad` | `ClipIntermediateGradBinding` | `_render_reduction_tree()` |

---

## 10. Risk Register

| Risk | Likelihood | Impact | Mitigation |
| :--- | :--- | :--- | :--- |
| Binding argument order mismatch with kernel parameter order | Medium | High | Automated comparison test: `binding.marshal_args()` output length and types compared against legacy `signature.get_args()` for identical inputs |
| Grid dimension mismatch causing incorrect results or crashes | Medium | High | `compute_grid()` output validated against legacy `signature.get_grid()` for identical inputs |
| Reduction tree stage count off-by-one | Low | High | Validated against existing `graph_recipes.py` stage loop; Tier 2 tests (Phase 2C) verify numerical correctness |
| Streaming loop last-chunk edge case (smaller chunk) | Medium | Medium | `ParameterStride` arithmetic tested with non-divisible total_extent; last chunk's effective count derived correctly |
| Legacy `KernelSignature` and new `KernelBinding` drift during coexistence | Low | Medium | Phase 6 removes `KernelSignature` classes. During coexistence, no modifications to legacy classes; extraction is one-directional. |
| `cl.Program` attribute lookup fails for kernel names with underscores | Low | Low | Verified: PyOpenCL converts underscores in `getattr(program, name)` correctly; kernel names match `kernels.cl.h` function names exactly. |
| `importlib.resources` kernel source loading fails in editable install | Medium | Low | Verified in Phase 0 test infrastructure; editable install uses path-based fallback. |
