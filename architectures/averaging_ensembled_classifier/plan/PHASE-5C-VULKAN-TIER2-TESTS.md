# Phase 5C: Vulkan Tier 2 Tests & Rollback Gate — Detailed Plan

**Status:** Not started  
**Phase:** 5C of 6 (sub-phase C of 3)  
**Objective:** Write the complete Vulkan Tier 2 test suite — per-kernel numerical correctness tests against the shared analytical and numpy reference fixtures — and validate the Phase 5 rollback gate (Tier 1 green + Vulkan Tier 2 green + Tier 3 cross-backend parity green). After this sub-phase, the Vulkan backend's plan-driven dispatch path is fully validated and the Phase 5 gate is closed.  
**Governing ADRs:** ADR-016 (test strategy — three-tier framework, CPU oracle, analytical + numpy fixtures), ADR-008 (precision configuration — tolerance parameterization), ADR-011 (CCE/BCE strategy — combinatorial test matrix; specialization constant variants for Vulkan), ADR-013 (kernel source hierarchy — specification fidelity; `kernels.cl.h` as algorithmic reference), ADR-007 (contract/binding split — Tier 2 validates the full stack from plan → renderer → GPU dispatch → retrieval)  
**Rollback gate:** Tier 1 green (no regressions) + Vulkan Tier 2 green (all per-kernel correctness tests pass for FP32) + Tier 3 cross-backend parity green (CPU-vs-Vulkan parity within configured tolerances, if CPU backend available). Phase 5's rollback gate is the most comprehensive of all backend phases because it includes Tier 3 as a requirement.  
**Dependencies:** Phase 5B (renderer & infrastructure) — provides fully functional `VulkanPlanRenderer` with pipeline cache, buffer allocator, and descriptor management. Phase 5A (shader library) — provides compiled SPIR-V modules. Phase 1 (plan model) — provides `ExecutionPlan` and shared-layer types. Phase 4 (test harness) — provides Tier 3 framework and oracle selection logic (if available; Phase 5C can implement Vulkan-specific Tier 3 tests independently if Phase 4 is incomplete).

---

## Table of Contents

1. [Scope & Constraints](#1-scope--constraints)
2. [Tier 2 Test Architecture](#2-tier-2-test-architecture)
3. [Target Deliverables](#3-target-deliverables)
4. [Task Breakdown](#4-task-breakdown)
   - [Step 5C.1: Create Vulkan Tier 2 test infrastructure](#step-5c1-create-vulkan-tier-2-test-infrastructure)
   - [Step 5C.2: Write Act-phase kernel tests](#step-5c2-write-act-phase-kernel-tests)
   - [Step 5C.3: Write Learn-phase gradient production tests](#step-5c3-write-learn-phase-gradient-production-tests)
   - [Step 5C.4: Write Learn-phase clipping & permutation tests](#step-5c4-write-learn-phase-clipping--permutation-tests)
   - [Step 5C.5: Write Learn-phase reduction & aggregation tests](#step-5c5-write-learn-phase-reduction--aggregation-tests)
   - [Step 5C.6: Write Learn-phase streaming backprop tests](#step-5c6-write-learn-phase-streaming-backprop-tests)
   - [Step 5C.7: Write Learn-phase finalization tests](#step-5c7-write-learn-phase-finalization-tests)
   - [Step 5C.8: Write end-to-end plan rendering correctness tests](#step-5c8-write-end-to-end-plan-rendering-correctness-tests)
   - [Step 5C.9: Configure Vulkan-specific tolerance tables](#step-5c9-configure-vulkan-specific-tolerance-tables)
   - [Step 5C.10: Implement Tier 3 parity tests (CPU-vs-Vulkan)](#step-5c10-implement-tier-3-parity-tests-cpu-vs-vulkan)
   - [Step 5C.11: FP16 parameterization (conditional)](#step-5c11-fp16-parameterization-conditional)
   - [Step 5C.12: Validate Phase 5 rollback gate](#step-5c12-validate-phase-5-rollback-gate)
5. [Fixture Architecture](#5-fixture-architecture)
6. [Per-Kernel Test Matrix](#6-per-kernel-test-matrix)
7. [Tolerance Configuration](#7-tolerance-configuration)
8. [Vulkan-Specific Test Considerations](#8-vulkan-specific-test-considerations)
9. [Tier 3 Architecture](#9-tier-3-architecture)
10. [Risk Register](#10-risk-register)

---

## 1. Scope & Constraints

### In scope

- Replacing the existing Vulkan Tier 2 test stub (`tests/tier2/vulkan/test_vulkan_kernels.py`) with a complete suite of per-kernel correctness tests, following the pattern established by `tests/tier2/cpu/` and `tests/tier2/opencl/`.
- Creating full Vulkan Tier 2 infrastructure: `tests/tier2/vulkan/conftest.py` with session-scoped `VulkanPlanRenderer` and `VulkanContext` fixtures, Vulkan-specific skip logic, per-kernel test modules.
- **Reusing** the existing shared reference fixtures from `tests/tier2/fixtures/` — `analytical.py`, `numpy_forward.py`, `numpy_gradients.py`, `numpy_reduction.py`, `numpy_backprop.py`, `numpy_update.py`, `data_generators.py`. These are backend-neutral and shared across all Tier 2 test suites.
- Writing per-kernel correctness tests that:
  1. Construct input data (deterministic RNG or handcrafted edge cases).
  2. Compute reference output via the shared numpy/analytical fixture.
  3. Build a minimal `ExecutionPlan` containing the kernel under test.
  4. Load inputs into Vulkan device-local buffers via the buffer allocator.
  5. Render through `VulkanPlanRenderer`.
  6. Retrieve output via `VulkanRetrievalFuture.result()`.
  7. Compare output vs. reference within configured tolerances via `np.testing.assert_allclose`.
- Writing end-to-end plan rendering tests exercising complete Act and Learn plans.
- Configuring Vulkan-specific tolerance tables in `tests/tolerance_config.py`.
- Implementing Tier 3 cross-backend parity tests (CPU-vs-Vulkan) as part of the Phase 5 rollback gate.
- Testing both CCE and BCE specialization constant variants for Strategy A kernels (Nodes 8, 9, 10).
- Testing both register-reduce and local-reduce tiers for the aggregation kernel.
- Testing correct barrier placement by validating multi-node sub-plans with data dependencies.

### Out of scope

- Writing Tier 1 tests (already complete in Phase 1).
- Modifying the shared reference fixtures (unless a Vulkan-specific edge case reveals a fixture gap, in which case the fixture is updated in the shared location to benefit all backends).
- Modifying GLSL shaders (Phase 5A) or renderer code (Phase 5B) — except for bug fixes discovered during testing, which are implemented and documented as part of the test-driven development cycle.
- Performance benchmarking (separate concern from correctness testing).
- Modifying the OpenCL or CPU backends.
- FP16 testing — deferred until the Vulkan backend gains FP16 shader support.

### Key constraint: fixtures are the mathematical definition

Same constraint as CPU Tier 2 (Phase 3C): the shared fixtures define mathematical truth. If the fixture and the Vulkan kernel disagree, the Vulkan kernel is wrong. The fixture is never modified to match a backend's output.

### Key constraint: shared fixtures, backend-specific tests

The reference fixtures in `tests/tier2/fixtures/` are backend-neutral. The Vulkan-specific parts are:

1. **Test infrastructure** — `conftest.py` differs from CPU/OpenCL: manages `VulkanContext`, `VulkanPlanRenderer`, GPU buffer lifecycle, fence-based synchronization.
2. **Tolerance tables** — Vulkan may have different tolerances due to:
   - Subgroup operation accumulation order (vendor-specific).
   - FMA instruction availability (most GPUs have native FMA).
   - Specialization constant branch elimination altering intermediate precision.
3. **Skip conditions** — Vulkan tests skip on `not BACKEND_VULKAN`.
4. **Variant coverage** — Vulkan tests must cover specialization constant variants (CCE/BCE) that don't exist in the CPU backend.

### Key constraint: Tier 3 is a Phase 5 rollback gate requirement

Per DESIGN.md §11.2, Phase 5's rollback gate includes "Tier 3 parity green." This is uniquely stringent — no other backend phase requires Tier 3 as part of its gate. The rationale: the Vulkan backend is the second GPU backend (after OpenCL), and cross-GPU parity validation is essential before the backend can be promoted to `enabled` status.

If the CPU backend is available at Phase 5C time, Tier 3 tests compare CPU-vs-Vulkan results. If only OpenCL and Vulkan are available, Tier 3 compares GPU-vs-GPU (per ADR-016 Option C fallback).

### Key constraint: deterministic test inputs

Test inputs use deterministic RNG seeds (via `data_generators.py`) or handcrafted values to ensure reproducibility. GPU execution order is non-deterministic at the hardware level, but the mathematical operations are deterministic given the same inputs and the same barrier placement. Tests must be reproducible across runs.

---

## 2. Tier 2 Test Architecture

Vulkan Tier 2 tests follow the same consistent flow as CPU (Phase 3C) and OpenCL (Phase 2C):

```
┌─────────────────────────────────┐
│  Test Case                      │
│                                 │
│  1. Construct input data        │  ← deterministic RNG or handcrafted
│  2. Compute reference output    │  ← shared numpy fixture / analytical formula
│  3. Build minimal plan          │  ← plan builder or manual plan construction
│  4. Upload inputs to GPU        │  ← VulkanBufferAllocator (staging → device copy)
│  5. Render plan                 │  ← VulkanPlanRenderer.render()
│  6. Retrieve output (D2H)      │  ← VulkanRetrievalFuture.result()
│  7. Compare output vs reference │  ← np.testing.assert_allclose(atol, rtol)
└─────────────────────────────────┘
```

**Key difference from CPU Tier 2:** Steps 4 and 6 involve actual device transfers (H2D upload via staging buffer, D2H download via staging buffer + fence). The CPU backend's zero-copy semantics are replaced by fence-gated asynchronous transfers.

**Key difference from OpenCL Tier 2:** The Vulkan renderer records commands into a command buffer and submits them as a batch, rather than enqueuing individual kernel dispatches. This means each test exercises the full command recording → submit → fence wait → staging readback pipeline.

---

## 3. Target Deliverables

| Deliverable | Location | Description |
| :--- | :--- | :--- |
| `conftest.py` | `tests/tier2/vulkan/` | Session-scoped `VulkanContext`, `VulkanPlanRenderer`, `HardwareProfile` fixtures; Vulkan skip logic |
| `test_vulkan_act_forward_pass.py` | `tests/tier2/vulkan/` | `forward_pass` kernel correctness |
| `test_vulkan_act_render_logits.py` | `tests/tier2/vulkan/` | `render_logits_chunk` kernel correctness |
| `test_vulkan_act_loss_cce.py` | `tests/tier2/vulkan/` | `compute_probs_loss_cce_chunk` kernel correctness |
| `test_vulkan_act_loss_bce.py` | `tests/tier2/vulkan/` | `compute_probs_loss_bce_chunk` kernel correctness |
| `test_vulkan_learn_module_grads.py` | `tests/tier2/vulkan/` | `calculate_module_param_grads` correctness (CCE + BCE variants) |
| `test_vulkan_learn_hidden_grads.py` | `tests/tier2/vulkan/` | `backprop_error_to_hidden` correctness (CCE + BCE variants) |
| `test_vulkan_learn_temp_grads.py` | `tests/tier2/vulkan/` | `calculate_temp_gradients` correctness (CCE + BCE variants) |
| `test_vulkan_learn_clip_partials.py` | `tests/tier2/vulkan/` | `clip_partial_gradients` correctness |
| `test_vulkan_learn_gather_permute.py` | `tests/tier2/vulkan/` | `gather_and_permute_grad_h` correctness |
| `test_vulkan_learn_reduction.py` | `tests/tier2/vulkan/` | `aggregate_partials` (register + local tiers) + `clip_intermediate_grad` correctness |
| `test_vulkan_learn_stabilize_grad_h.py` | `tests/tier2/vulkan/` | `stabilize_reduce_grad_h` correctness |
| `test_vulkan_learn_streaming_backprop.py` | `tests/tier2/vulkan/` | `backprop_shared_weights` + `backprop_shared_biases` + `clip_shared_gradients` correctness |
| `test_vulkan_learn_normalize.py` | `tests/tier2/vulkan/` | `normalize_gradients` correctness |
| `test_vulkan_learn_adam_update.py` | `tests/tier2/vulkan/` | `adam_update` correctness |
| `test_vulkan_learn_clamp_temps.py` | `tests/tier2/vulkan/` | `clamp_temperatures` correctness |
| `test_vulkan_plan_rendering_correctness.py` | `tests/tier2/vulkan/` | End-to-end Act + Learn plan rendering |
| Vulkan tolerance entries | `tests/tolerance_config.py` | Vulkan-specific `atol`/`rtol` per kernel |
| Tier 3 parity tests | `tests/tier3/` | CPU-vs-Vulkan (or OpenCL-vs-Vulkan fallback) per-kernel parity |

**Total: 16 Vulkan Tier 2 test modules** (matching the CPU Tier 2 count) + Tier 3 parity additions.

---

## 4. Task Breakdown

### Step 5C.1: Create Vulkan Tier 2 test infrastructure

Replace the existing stub in `tests/tier2/vulkan/` with full test infrastructure.

**`tests/tier2/vulkan/conftest.py`:**

```python
"""Vulkan Tier 2 test configuration and fixtures."""
from __future__ import annotations

import pytest
import numpy as np

from averaging_ensembled_classifier._build_config import BACKEND_VULKAN

# Skip entire directory if Vulkan backend not available
pytestmark = [
    pytest.mark.tier2,
    pytest.mark.vulkan,
    pytest.mark.skipif(not BACKEND_VULKAN, reason="Vulkan backend not built"),
]


@pytest.fixture(scope="session")
def vulkan_context():
    """Session-scoped Vulkan context — one GPU context for all tests."""
    from averaging_ensembled_classifier.backends.vulkan import VulkanContext
    ctx = VulkanContext(enable_validation=True)
    yield ctx
    ctx.destroy()


@pytest.fixture(scope="session")
def vulkan_hardware_profile(vulkan_context):
    """HardwareProfile populated from the Vulkan device."""
    from averaging_ensembled_classifier.backends.vulkan import discover_hardware
    return discover_hardware(vulkan_context)


@pytest.fixture(scope="session")
def vulkan_renderer_cce(vulkan_context, vulkan_hardware_profile):
    """Session-scoped VulkanPlanRenderer for CCE problem type."""
    from averaging_ensembled_classifier.backends.vulkan import VulkanPlanRenderer
    from averaging_ensembled_classifier.shared.precision_config import PrecisionConfig
    renderer = VulkanPlanRenderer(
        context=vulkan_context,
        hardware_profile=vulkan_hardware_profile,
        precision=PrecisionConfig.float32(),
        problem_type=0,  # CCE
    )
    yield renderer
    renderer.destroy()


@pytest.fixture(scope="session")
def vulkan_renderer_bce(vulkan_context, vulkan_hardware_profile):
    """Session-scoped VulkanPlanRenderer for BCE problem type."""
    from averaging_ensembled_classifier.backends.vulkan import VulkanPlanRenderer
    from averaging_ensembled_classifier.shared.precision_config import PrecisionConfig
    renderer = VulkanPlanRenderer(
        context=vulkan_context,
        hardware_profile=vulkan_hardware_profile,
        precision=PrecisionConfig.float32(),
        problem_type=1,  # BCE
    )
    yield renderer
    renderer.destroy()
```

**Key design decisions:**
- Session-scoped fixtures: `VulkanContext` and `VulkanPlanRenderer` are expensive to create (GPU initialization, pipeline compilation). One instance per test session avoids redundant setup.
- Separate CCE and BCE renderers: Vulkan pipelines are specialized at creation time — CCE and BCE variants are different pipeline objects. Tests that exercise Strategy A kernels use the appropriate renderer.
- Validation layers enabled: during testing, validation layers catch synchronization errors, descriptor mismatches, and resource leaks.

**Acceptance criteria:**
- `conftest.py` creates session-scoped Vulkan resources.
- Tests skip cleanly when `BACKEND_VULKAN is False`.
- Validation layers are active during test execution.
- `conftest.py` teardown destroys all Vulkan resources without warnings.

---

### Step 5C.2: Write Act-phase kernel tests

Four test modules covering the Act-phase kernels:

| Test Module | Kernel Under Test | Reference Fixture | Parameterization |
| :--- | :--- | :--- | :--- |
| `test_vulkan_act_forward_pass.py` | `forward_pass` (Node 4) | `numpy_forward.forward_pass` | Batch sizes (1, small, padded); input/hidden dimensions; zero inputs; sample mask patterns |
| `test_vulkan_act_render_logits.py` | `render_logits_chunk` (Node 5) | `numpy_forward.render_logits` | Module/class/batch chunk configurations; edge cases (single module, single class) |
| `test_vulkan_act_loss_cce.py` | `compute_probs_loss_cce_chunk` (Node 6) | `numpy_forward.compute_probs_loss_cce` | Tile counts; class counts near/at SIMD boundary; extreme logit values |
| `test_vulkan_act_loss_bce.py` | `compute_probs_loss_bce_chunk` (Node 7) | `numpy_forward.compute_probs_loss_bce` | Sigmoid saturation points; multi-label target configurations |

**Vulkan-specific test dimensions:**
- **2D dispatch geometry:** `forward_pass` uses `vkCmdDispatch(batch_chunks, hidden_blocks, 1)`. Tests verify correctness across various `gl_WorkGroupID.x/y` combinations.
- **Subgroup reduction correctness:** `compute_probs_loss_cce` exercises the `workgroup_reduce_max` / `workgroup_reduce_add` helpers through the numerically stable softmax. Tests include configurations where the workgroup spans multiple subgroups.
- **SIMD boundary alignment:** Test with input dimensions that are exactly SIMD-aligned and dimensions that require padding, verifying that padding elements don't pollute results.

**Acceptance criteria:**
- All four Act-phase test modules pass for FP32.
- Edge cases (extreme values, boundary dimensions) are covered.
- No Vulkan validation warnings during test execution.

---

### Step 5C.3: Write Learn-phase gradient production tests

Three test modules for Strategy A kernels, covering both CCE and BCE specialization variants:

| Test Module | Kernel Under Test | Reference Fixture | Variants |
| :--- | :--- | :--- | :--- |
| `test_vulkan_learn_module_grads.py` | `calculate_module_param_grads` (Node 8) | `numpy_gradients.calculate_module_param_grads` | CCE (`vulkan_renderer_cce`), BCE (`vulkan_renderer_bce`) |
| `test_vulkan_learn_hidden_grads.py` | `backprop_error_to_hidden` (Node 9) | `numpy_gradients.backprop_error_to_hidden` | CCE, BCE |
| `test_vulkan_learn_temp_grads.py` | `calculate_temp_gradients` (Node 10) | `numpy_gradients.calculate_temp_gradients` | CCE, BCE |

**Vulkan-specific test dimensions:**
- **Specialization constant coverage:** Each test is parameterized over `(vulkan_renderer_cce, vulkan_renderer_bce)` to exercise both pipeline variants. This is unique to the Vulkan backend — CPU and OpenCL use a runtime flag.
- **Placement contract verification:** Node 8 writes scattered partials via `grid_mod_cls`. Tests verify that `gl_WorkGroupID.x` correctly decomposes into `(module_chunk, class_chunk)` by checking the output at expected offsets in the partial collection buffer.
- **DAG concurrency:** Tests exercise Nodes 8, 9, 10 dispatched as a group (in a single command buffer with no mutual barriers) to validate that disjoint output buffers prevent data hazards.

**Acceptance criteria:**
- All three modules pass for both CCE and BCE variants.
- Gradient values match numpy reference within tolerance.
- Partial collection buffer write offsets are correct (Placement Contract validated).

---

### Step 5C.4: Write Learn-phase clipping & permutation tests

| Test Module | Kernel Under Test | Reference Fixture | Key Scenarios |
| :--- | :--- | :--- | :--- |
| `test_vulkan_learn_clip_partials.py` | `clip_partial_gradients` (Node 11) | `numpy_gradients.clip_partial_gradients` | Group L2 norm below threshold (no clipping); above threshold (scaling); zero gradients; `use_per_item_norm=0/1` |
| `test_vulkan_learn_gather_permute.py` | `gather_and_permute_grad_h` (Node 13) | `numpy_gradients.gather_and_permute` | Single tile; multiple tiles; alignment edge cases; AoS→SoA layout verification |

**Vulkan-specific test dimensions:**
- **Workgroup reduction for L2 norm:** Node 11's `workgroup_reduce_add` computes the group L2 norm. Tests include configurations where the group norm crosses subgroup boundaries to validate the cross-subgroup bridge.
- **Global barrier semantics:** Node 13 reads from all upstream tiles. The test must ensure the barrier between Node 11 and Node 13 is correctly placed (command buffer recording order matters).

**Acceptance criteria:**
- Both modules pass with correct clipping behavior.
- AoS→SoA permutation produces the expected layout.
- No validation warnings from barrier-related issues.

---

### Step 5C.5: Write Learn-phase reduction & aggregation tests

| Test Module | Kernel Under Test | Reference Fixture | Configuration |
| :--- | :--- | :--- | :--- |
| `test_vulkan_learn_reduction.py` | `aggregate_partials` (register + local) + `clip_intermediate_grad` | `numpy_reduction.multi_stage_reduction` | 1-stage, 2-stage, 3-stage trees; fan-in K ≤ subgroup size (register) and K > subgroup size (local); threshold schedule computation |

**Vulkan-specific test dimensions:**
- **Tier selection:** Tests exercise both register-tier (K ≤ subgroup_size, no shared memory) and local-tier (K > subgroup_size, cross-subgroup bridge). The tier boundary depends on the device's actual subgroup size.
- **Push descriptor updates:** The reduction engine uses `vkCmdPushDescriptorSetKHR` to update source/destination buffers per stage. Tests verify the ping-pong buffer swap produces correct results at each stage.
- **Threshold schedule verification:** Per-stage clipping thresholds follow $T_j = T_{\text{algorithmic}} + \lambda \cdot j^2$, clamped by $T_{\text{safety},j} = \text{FP\_FORMAT\_MAX} / K_j$. Tests verify the correct threshold is applied at each stage by comparing against known reduction outputs.
- **SUM and AVERAGE modes:** Tests cover both `operation_type=0` (SUM) and `operation_type=1` (AVERAGE) via push constants.

**Acceptance criteria:**
- Multi-stage reductions produce correct results for 1, 2, and 3-stage trees.
- Both register and local reduction tiers produce correct results.
- Threshold schedule is correctly applied per stage.
- Ping-pong buffer management does not corrupt data.

---

### Step 5C.6: Write Learn-phase streaming backprop tests

| Test Module | Kernels Under Test | Reference Fixture |
| :--- | :--- | :--- |
| `test_vulkan_learn_stabilize_grad_h.py` | `stabilize_reduce_grad_h` (Node 16) | `numpy_reduction.stabilize_reduce_grad_h` |
| `test_vulkan_learn_streaming_backprop.py` | `backprop_shared_weights` (17) + `backprop_shared_biases` (18) + `clip_shared_gradients` (19) | `numpy_backprop.backprop_shared_weights`, `numpy_backprop.backprop_shared_biases`, `numpy_gradients.clip_shared_gradients` |

**Vulkan-specific test dimensions:**
- **Node 16 internal reduction:** `stabilize_reduce_grad_h` performs its own staged reduction internally (ADR-005). Tests verify the full internal reduction behaves correctly, particularly with varying `padded_hidden_count` and `total_modules_count` values that affect the workgroup-level reduction depth.
- **Streaming loop command recording:** Nodes 17/18/19 are dispatched in a per-chunk loop within the command buffer. Tests verify:
  - Push constant `batch_chunk_index` is updated correctly per chunk.
  - Barrier between `{17,18}` and `19` is correctly placed per chunk.
  - Results accumulate correctly across chunks.
- **Per-chunk partial accumulation:** Across multiple streaming chunks, each chunk contributes a partial that is subsequently reduced. Tests verify the end-to-end streaming chain (dispatch per chunk → reduce → normalize) produces the correct final result.

**Acceptance criteria:**
- Node 16 produces correct stabilized Grad_H reduction.
- Streaming loop correctly iterates over chunks with per-chunk push constant updates.
- Per-chunk barrier placement within the command buffer is correct.

---

### Step 5C.7: Write Learn-phase finalization tests

| Test Module | Kernel Under Test | Reference Fixture |
| :--- | :--- | :--- |
| `test_vulkan_learn_normalize.py` | `normalize_gradients` (Node 21) | `numpy_update.normalize_gradients` |
| `test_vulkan_learn_adam_update.py` | `adam_update` (Node 24) | `numpy_update.adam_update` |
| `test_vulkan_learn_clamp_temps.py` | `clamp_temperatures` (Node 25) | `analytical.clamp_temperatures` |

These are the simplest Tier 2 tests — element-wise operations with minimal control flow.

**Test scenarios:**
- `normalize_gradients`: zero batch size (edge), typical batch sizes, large batch sizes.
- `adam_update`: initial step (`t=1`, where bias correction has maximal effect), later steps, near-zero gradients, large gradients. Verify `beta1_pow_t` and `beta2_pow_t` push constants are correctly consumed (host-computed in FP64, passed as FP32 push constants).
- `clamp_temperatures`: values below min, above max, within range; min > max edge case (if applicable).

**Acceptance criteria:**
- All three modules pass for FP32 with correct element-wise behavior.
- Adam optimizer correctly applies bias correction.
- Temperature clamping boundary conditions are handled correctly.

---

### Step 5C.8: Write end-to-end plan rendering correctness tests

**`test_vulkan_plan_rendering_correctness.py`:**

Exercises complete Act and Learn plans through the full pipeline:

| Test | Scenario | Validation |
| :--- | :--- | :--- |
| `test_act_plan_small_cce` | 4 samples × 8 features × 16 hidden × 2 modules × 4 classes (CCE) | Act plan end-to-end: forward → logits → softmax probs. Compare vs. numpy orchestration. |
| `test_act_plan_small_bce` | Same dimensions, BCE problem type | Act plan end-to-end with sigmoid probs. |
| `test_act_plan_padded_dimensions` | Dimensions requiring SIMD padding | Verify padding doesn't pollute results. |
| `test_learn_plan_small_cce` | Full Learn plan with 1-stage reduction tree | Complete gradient pipeline: grads → clip → permute → reduce → backprop → normalize → Adam. |
| `test_learn_plan_multi_stage_reduction` | Large enough for 2-stage reduction tree | Validates multi-stage reduction within full Learn plan. |
| `test_learn_plan_streaming_chunks` | Configuration requiring multiple streaming chunks | Validates streaming loop execution within full Learn plan. |
| `test_act_learn_round_trip` | Act plan → extract probs → Learn plan | Full training step; verifies model state is updated correctly. |

**Validation method:** Numpy reference implementations orchestrate the same computation (end-to-end), and the final outputs are compared within tolerance. This differs from per-kernel tests: end-to-end tests accumulate numerical divergence across the full DAG, so tolerances may be slightly broader.

**Acceptance criteria:**
- All end-to-end tests pass for both CCE and BCE.
- Model state updates are numerically correct after a Learn plan.
- Act → Learn round-trip demonstrates functional training behavior.

---

### Step 5C.9: Configure Vulkan-specific tolerance tables

Add Vulkan entries to `tests/tolerance_config.py`:

```python
VULKAN_FP32_TOLERANCES = {
    # Base tolerances (element-wise kernels)
    "default": {"atol": 1e-5, "rtol": 1e-5},

    # Kernels with reduction (subgroup operations have vendor-specific ordering)
    "compute_probs_loss_cce": {"atol": 1e-4, "rtol": 1e-4},
    "compute_probs_loss_bce": {"atol": 1e-4, "rtol": 1e-4},
    "clip_partial_gradients": {"atol": 1e-4, "rtol": 1e-4},
    "aggregate_partials": {"atol": 1e-4, "rtol": 1e-4},
    "stabilize_reduce_grad_h": {"atol": 1e-4, "rtol": 1e-4},

    # Adam optimizer (division, sqrt, multiple accumulations)
    "adam_update": {"atol": 1e-4, "rtol": 1e-4},

    # End-to-end plans (accumulated divergence across full DAG)
    "e2e_act": {"atol": 5e-4, "rtol": 5e-4},
    "e2e_learn": {"atol": 1e-3, "rtol": 1e-3},
}

# Tier 3 cross-backend tolerances (CPU-vs-Vulkan)
TIER3_CPU_VS_VULKAN_FP32 = {
    "default": {"atol": 1e-4, "rtol": 1e-4},
    "aggregate_partials": {"atol": 5e-4, "rtol": 5e-4},
    "stabilize_reduce_grad_h": {"atol": 5e-4, "rtol": 5e-4},
    "adam_update": {"atol": 5e-4, "rtol": 5e-4},
    "e2e_learn": {"atol": 2e-3, "rtol": 2e-3},
}
```

**Rationale for Vulkan-specific tolerances:**
- Subgroup reduction operations (`subgroupAdd`, `subgroupMax`) have vendor-specific accumulation ordering within a subgroup. This may produce slightly different intermediate results compared to the sequential numpy reference or the CPU's SIMD reduction.
- The cross-subgroup bridge (shared-memory exchange + second `subgroupAdd`) introduces an additional accumulation boundary.
- FMA instructions on GPU hardware may fuse multiply-add operations differently than the numpy reference (which uses separate multiply and add).
- Tier 3 cross-backend tolerances are broader than per-backend Tier 2 tolerances because they account for implementation-specific differences between two execution engines (CPU SIMD vs. GPU subgroup ops).

**Acceptance criteria:**
- Tolerance configuration entries exist for all Vulkan kernels.
- Tier 3 cross-backend tolerances are calibrated during initial test runs and documented.
- No tolerance is looser than necessary — tighten after initial runs stabilize.

---

### Step 5C.10: Implement Tier 3 parity tests (CPU-vs-Vulkan)

Tier 3 validates cross-backend parity — the same plan produces numerically equivalent results on different backends.

**Test structure (`tests/tier3/test_parity_cpu_vulkan.py`):**

```python
@pytest.mark.tier3
@pytest.mark.skipif(
    not (BACKEND_CPU and BACKEND_VULKAN),
    reason="Tier 3 requires CPU + Vulkan backends",
)
class TestCpuVulkanParity:
    """Cross-backend parity: CPU-vs-Vulkan."""

    def test_forward_pass_parity(self, cpu_renderer, vulkan_renderer, ...):
        """Same input → same output (within tolerance) on CPU and Vulkan."""
        plan = build_minimal_forward_pass_plan(...)
        cpu_result = render_and_extract(cpu_renderer, plan)
        vulkan_result = render_and_extract(vulkan_renderer, plan)
        np.testing.assert_allclose(
            cpu_result, vulkan_result,
            **TIER3_CPU_VS_VULKAN_FP32["forward_pass"]
        )

    # Per-kernel parity tests for all kernels in the inventory...
    # End-to-end Act plan parity
    # End-to-end Learn plan parity
```

**Oracle model (ADR-016):**
- **Primary:** CPU as reference oracle (CPU correctness established by CPU Tier 2 against numpy fixtures; circle broken).
- **Fallback:** If CPU is unavailable, GPU-vs-GPU comparison (OpenCL-vs-Vulkan) — same inputs, compare outputs within cross-GPU tolerance.

**Tier 3 test matrix:**

| Test | CPU Oracle | Vulkan Subject | Comparison |
| :--- | :--- | :--- | :--- |
| Per-kernel parity (19 kernels) | CPU Tier 2 output | Vulkan Tier 2 output | `assert_allclose` per kernel |
| Act plan parity | CPU act plan output | Vulkan act plan output | Final probs comparison |
| Learn plan parity | CPU learn plan output | Vulkan learn plan output | Post-update model state comparison |
| Full training step parity | CPU act+learn | Vulkan act+learn | Probs + model state comparison |

**Acceptance criteria:**
- All Tier 3 parity tests pass within the configured cross-backend tolerance.
- Parity tests skip cleanly when only one backend is available.
- No systematic bias (e.g., Vulkan consistently higher/lower than CPU) — random divergence is expected, systematic divergence indicates a bug.

---

### Step 5C.11: FP16 parameterization (conditional)

FP16 testing is deferred until the Vulkan backend implements FP16 shader support (requires `VK_KHR_shader_float16_int8` extension and `shaderFloat16` device feature). When available:

1. Add FP16 specialization constant handling to the shader library (Phase 5A amendment).
2. Add FP16 pipeline variants to the pipeline cache (Phase 5B amendment).
3. Parameterize Tier 2 tests over `[PrecisionConfig.float32(), PrecisionConfig.float16()]`.
4. Add FP16-specific tolerance entries to `tolerance_config.py`.

**Current status:** FP32 only. FP16 is explicitly out of scope for the initial Phase 5 delivery. The test infrastructure (parameterization, tolerance lookup, precision fixture) is designed to accommodate FP16 when it arrives.

---

### Step 5C.12: Validate Phase 5 rollback gate

**Gate conditions (the most comprehensive rollback gate in the migration):**

1. **Tier 1 green:** All 140+ plan model tests pass (`pytest tests/tier1/ -v`).
2. **Vulkan Tier 2 green:** All per-kernel correctness tests pass for FP32 (`pytest tests/tier2/vulkan/ -v`).
3. **Tier 3 parity green:** Cross-backend parity tests pass (`pytest tests/tier3/ -v`).
4. **Existing backends unaffected:** CPU Tier 2 green (`pytest tests/tier2/cpu/ -v`), OpenCL Tier 2 green if available (`pytest tests/tier2/opencl/ -v`).
5. **Legacy tests unaffected:** Existing `tests/test_integration_*.py` pass.

**Full validation command:**

```bash
pytest tests/ -v --tb=short
```

This runs all tiers, all backends, all legacy tests. The Phase 5 gate is green when every test that should pass does pass, and every test that should skip does skip.

**Acceptance criteria:**
- Zero failures.
- Vulkan tests show as "passed" (not "skipped").
- Tier 3 tests show as "passed" (not "skipped").
- `_build_config.BACKEND_VULKAN is True`.
- Feature flag can be promoted from `auto` to `enabled` in CI.

---

## 5. Fixture Architecture

### Shared Fixtures (from `tests/tier2/fixtures/`)

| Fixture Module | Kernels Covered | Phase 5C Consumer |
| :--- | :--- | :--- |
| `analytical.py` | `clamp_temperatures`, `compute_hidden_mask`, `normalize_gradients`, clip kernels | `test_vulkan_learn_clamp_temps`, `test_vulkan_learn_normalize`, clip tests |
| `data_generators.py` | All (input data generation) | All test modules |
| `numpy_forward.py` | `forward_pass`, `render_logits`, `compute_probs_loss_cce/bce` | Act-phase tests |
| `numpy_gradients.py` | Nodes 8–11, 13, 19 | Gradient production and clipping tests |
| `numpy_reduction.py` | Nodes 14/15/20, 16 | Reduction and stabilization tests |
| `numpy_backprop.py` | Nodes 17, 18 | Streaming backprop tests |
| `numpy_update.py` | Nodes 21, 24 | Normalization and Adam tests |

Phase 5C **consumes** these fixtures without modification. If a Vulkan-specific edge case reveals a fixture gap, the fixture is updated in the shared location (benefiting all backends).

### Vulkan-Specific Fixtures

The Vulkan `conftest.py` provides backend-specific fixtures:

| Fixture | Scope | Purpose |
| :--- | :--- | :--- |
| `vulkan_context` | Session | Shared `VulkanContext` with validation layers |
| `vulkan_hardware_profile` | Session | `HardwareProfile` from device queries |
| `vulkan_renderer_cce` | Session | `VulkanPlanRenderer` with CCE specialization |
| `vulkan_renderer_bce` | Session | `VulkanPlanRenderer` with BCE specialization |

---

## 6. Per-Kernel Test Matrix

Each kernel is tested across multiple dimensions:

| Dimension | Values | Kernels Affected |
| :--- | :--- | :--- |
| Problem type | CCE, BCE | Nodes 6/7 (separate shaders), Nodes 8/9/10 (specialization variants) |
| Batch size | 1, small (4–8), SIMD-aligned, SIMD-unaligned | All |
| Feature/hidden dimensions | Small, SIMD-aligned, SIMD-unaligned, large | Nodes 4, 5, 9, 16, 17, 18 |
| Module/class counts | 1, small, medium | Nodes 5, 6, 7, 8, 10 |
| Reduction stages | 1-stage, 2-stage, 3-stage | Nodes 14/15/20 |
| Reduction fan-in K | K ≤ subgroup_size (register tier), K > subgroup_size (local tier) | Nodes 14/15/20 |
| Streaming chunks | 1 chunk, multiple chunks | Nodes 17, 18, 19 |
| Threshold values | Below threshold (no clipping), above threshold (clipping), edge (exactly at threshold) | Nodes 11, 15b/20b, 19 |
| Zero/extreme inputs | Zero inputs, very large gradients, NaN guard scenarios | All |

---

## 7. Tolerance Configuration

### Tier 2: Vulkan-vs-Numpy Tolerances

| Kernel Category | FP32 atol | FP32 rtol | Rationale |
| :--- | :--- | :--- | :--- |
| Element-wise (normalize, clamp, simple clip) | 1e-5 | 1e-5 | Minimal accumulation; matches CPU baseline |
| Dot product (forward_pass, render_logits, backprop) | 1e-5 | 1e-5 | GPU FMA may slightly differ but within 1 ULP |
| Reduction-bearing (softmax, L2 norm, aggregate) | 1e-4 | 1e-4 | Subgroup accumulation order is vendor-specific |
| Multi-stage (stabilize_reduce, streaming backprop total) | 1e-4 | 1e-4 | Accumulated divergence across stages |
| Adam optimizer | 1e-4 | 1e-4 | Division + sqrt + multiple accumulations |
| End-to-end Act plan | 5e-4 | 5e-4 | Accumulated across full forward pass |
| End-to-end Learn plan | 1e-3 | 1e-3 | Accumulated across full gradient pipeline |

### Tier 3: CPU-vs-Vulkan Cross-Backend Tolerances

| Kernel Category | FP32 atol | FP32 rtol | Rationale |
| :--- | :--- | :--- | :--- |
| Element-wise | 1e-5 | 1e-5 | Should be bit-identical or near-identical |
| Dot product | 1e-4 | 1e-4 | SIMD reduction vs. subgroup reduction ordering |
| Reduction-bearing | 5e-4 | 5e-4 | CPU uses `simd_reduce_add`; Vulkan uses `subgroupAdd` — fundamentally different tree shapes |
| Multi-stage reduction | 5e-4 | 5e-4 | Stage ordering is identical (plan-prescribed), but per-stage accumulation differs |
| End-to-end Learn | 2e-3 | 2e-3 | Maximum accumulated divergence across all stages |

---

## 8. Vulkan-Specific Test Considerations

### Validation Layer Integration

All Vulkan Tier 2 tests run with `VK_LAYER_KHRONOS_validation` enabled (via `VulkanContext(enable_validation=True)` in the session fixture). The validation layer catches:

| Issue Category | Detection |
| :--- | :--- |
| Missing barriers (RAW/WAW hazards) | `VUID-vkCmdDispatch-commandBuffer-xxxxx` |
| Descriptor set mismatch | `VUID-vkCmdDispatch-descriptorType-xxxxx` |
| Push constant size mismatch | `VUID-vkCmdPushConstants-size-xxxxx` |
| Out-of-bounds buffer access | `VK_EXT_robustness2` (if available) |
| Resource leaks | Validation layer destruction tracking |

Tests should hook into the validation layer's debug messenger to convert validation warnings into test failures. This ensures barrier correctness is actively verified, not just assumed.

### GPU Non-Determinism

GPU execution order within a workgroup/subgroup is deterministic (hardware-defined). Execution order *between* independent workgroups is non-deterministic. The architecture guarantees correctness via explicit barriers:

- Independent dispatches (no barrier) → no data hazard by construction (disjoint write sets).
- Dependent dispatches → barrier between them.

Tests verify functional correctness (correct results), not execution ordering. Tests are deterministic given the same inputs, same barrier placement, and same plan structure.

### Command Buffer Recording vs. Execution Errors

If a test fails, the failure may be in:
1. **Command recording** (Python-side): wrong pipeline bound, wrong descriptors, wrong push constants, wrong dispatch dimensions.
2. **Shader execution** (GPU-side): algorithm bug, shared memory race, subgroup assumption violated.
3. **Readback** (Python-side): staging buffer not flushed, padding not stripped, wrong dtype.

The validation layer catches category 1. Numerical comparison against the reference fixture catches categories 2 and 3. Diagnostic output should include the dispatch dimensions, push constant values, and buffer sizes to aid debugging.

---

## 9. Tier 3 Architecture

### Oracle Selection (ADR-016)

```
CPU available?
├── Yes → CPU is oracle, Vulkan is subject
│         Compare: cpu_output vs vulkan_output
│         Tolerance: TIER3_CPU_VS_VULKAN_FP32
│
└── No → GPU-vs-GPU fallback
          OpenCL available?
          ├── Yes → OpenCL is reference, Vulkan is subject
          │         Tolerance: TIER3_GPU_VS_GPU_FP32 (broader)
          └── No → Skip Tier 3 (only one backend available)
```

### Tier 3 Test Infrastructure

```python
# tests/tier3/conftest.py

@pytest.fixture(scope="session")
def oracle_renderer():
    """Select the reference oracle backend."""
    if BACKEND_CPU:
        from averaging_ensembled_classifier.backends.cpu import CPUPlanRenderer
        # ... create CPU renderer
    elif BACKEND_OPENCL:
        from averaging_ensembled_classifier.backends.opencl import OpenCLPlanRenderer
        # ... create OpenCL renderer
    else:
        pytest.skip("No oracle backend available for Tier 3")

@pytest.fixture(scope="session")
def subject_renderer():
    """The backend under parity test."""
    if BACKEND_VULKAN:
        from averaging_ensembled_classifier.backends.vulkan import VulkanPlanRenderer
        # ... create Vulkan renderer
    else:
        pytest.skip("Vulkan backend not available")
```

### Tier 3 Test Coverage

| Parity Test | Plan Size | Validation |
| :--- | :--- | :--- |
| Per-kernel parity (×19) | Single-kernel sub-plans | Element-wise comparison per kernel |
| Act plan parity | Full Act plan | Final probs comparison |
| Learn plan parity | Full Learn plan | Post-update model state comparison (all 5 parameter groups) |
| Multi-batch parity | 3 consecutive Act+Learn cycles | Model state after 3 training steps |
| Reduction parity | Isolated 3-stage reduction tree | Reduction output comparison |

---

## 10. Risk Register

| Risk | Likelihood | Impact | Mitigation |
| :--- | :--- | :--- | :--- |
| GPU not available in CI (headless environment) | Medium | Block: Vulkan tests cannot run | CI uses GPU-enabled runners (e.g., NVIDIA T4 instances). Tests skip gracefully on runners without GPU. |
| Validation layer warnings treated as errors cause false failures | Medium | Noise: tests fail for non-functional validation warnings | Filter known benign warnings (if any). Treat all unknown warnings as failures. |
| Subgroup size mismatch between test fixture assumptions and hardware | Medium | Correctness: tests may not exercise both register and local reduction tiers | Query actual `subgroupSize` in `conftest.py`; parameterize reduction tests to force both tiers by adjusting fan-in K. |
| Vulkan driver bugs on specific GPU vendors | Low | Block: tests fail due to driver issues, not shader bugs | Run on multiple GPU vendors in CI (NVIDIA, AMD, Intel if available). Document known driver issues. |
| Tier 3 tolerance calibration is too tight or too loose | High | Quality: false failures (too tight) or missed bugs (too loose) | Calibrate during initial test runs against all available backends. Tighten iteratively. Document per-kernel tolerance rationale. |
| Test session leaked GPU resources on abnormal exit | Medium | Quality: GPU memory exhaustion on subsequent test runs | Use `pytest` finalizers and try/finally in fixtures. Session-scoped fixtures call `destroy()` in `yield` cleanup. `atexit` handler as fallback. |
| Phase 4 (Test Harness) not yet complete when Phase 5C starts | Medium | Scope: Tier 3 framework not available | Phase 5C implements its own Tier 3 tests in `tests/tier3/`. If Phase 4 later provides shared Tier 3 infrastructure, migrate tests to use it. |
| Command buffer recording errors produce silent correctness failures | Medium | Quality: tests pass but barriers are wrong → intermittent failures on different hardware | Validation layers catch barrier issues. Run tests on ≥2 GPU architectures to expose driver-assumption dependencies. |
