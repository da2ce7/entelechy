# Phase 2C: OpenCL Tier 2 Tests & Rollback Gate — Detailed Plan

**Status:** Not started  
**Phase:** 2C of 6 (sub-phase C of 3)  
**Objective:** Write the complete OpenCL Tier 2 test suite — per-kernel numerical correctness tests against analytical and numpy reference fixtures — and validate the Phase 2 rollback gate (Tier 1 + OpenCL Tier 2 green). After this sub-phase, the OpenCL backend's plan-driven dispatch path is fully validated and the Phase 2 gate is closed.  
**Governing ADRs:** ADR-016 (test strategy — three-tier framework), ADR-008 (precision configuration — tolerance parameterization), ADR-011 (CCE/BCE strategy — combinatorial test matrix), ADR-013 (kernel source hierarchy — specification fidelity), ADR-007 (contract/binding split — Tier 2 validates the full stack)  
**Rollback gate:** Tier 1 green (no regressions) + OpenCL Tier 2 green (all per-kernel correctness tests pass for FP32; FP16 tests pass at FP16 tolerances).  
**Dependencies:** Phase 2B (kernel bindings) — provides fully functional `OpenCLPlanRenderer` with all `KernelBinding`s and rendering logic.

---

## Table of Contents

1. [Scope & Constraints](#1-scope--constraints)
2. [Tier 2 Test Architecture](#2-tier-2-test-architecture)
3. [Target Deliverables](#3-target-deliverables)
4. [Task Breakdown](#4-task-breakdown)
   - [Step 2C.1: Create Tier 2 test infrastructure](#step-2c1-create-tier-2-test-infrastructure)
   - [Step 2C.2: Implement analytical reference fixtures](#step-2c2-implement-analytical-reference-fixtures)
   - [Step 2C.3: Implement numpy reference implementations](#step-2c3-implement-numpy-reference-implementations)
   - [Step 2C.4: Write Act-phase kernel tests](#step-2c4-write-act-phase-kernel-tests)
   - [Step 2C.5: Write Learn-phase gradient production tests](#step-2c5-write-learn-phase-gradient-production-tests)
   - [Step 2C.6: Write Learn-phase reduction & aggregation tests](#step-2c6-write-learn-phase-reduction--aggregation-tests)
   - [Step 2C.7: Write Learn-phase streaming backprop tests](#step-2c7-write-learn-phase-streaming-backprop-tests)
   - [Step 2C.8: Write Learn-phase update kernel tests](#step-2c8-write-learn-phase-update-kernel-tests)
   - [Step 2C.9: Write end-to-end plan rendering correctness tests](#step-2c9-write-end-to-end-plan-rendering-correctness-tests)
   - [Step 2C.10: Configure tolerance tables](#step-2c10-configure-tolerance-tables)
   - [Step 2C.11: FP16 parameterization](#step-2c11-fp16-parameterization)
   - [Step 2C.12: Validate Phase 2 rollback gate](#step-2c12-validate-phase-2-rollback-gate)
5. [Fixture Architecture](#5-fixture-architecture)
6. [Per-Kernel Test Matrix](#6-per-kernel-test-matrix)
7. [Tolerance Configuration](#7-tolerance-configuration)
8. [Test Infrastructure Details](#8-test-infrastructure-details)
9. [Risk Register](#9-risk-register)

---

## 1. Scope & Constraints

### In scope

- Creating the Tier 2 test directory structure (`tests/tier2/opencl/`) with conftest, fixtures, and per-kernel test modules.
- Implementing **analytical reference fixtures** for kernels with closed-form mathematical definitions: `compute_hidden_mask`, `clamp_temperatures`, `normalize_gradients`, clipping kernels.
- Implementing **numpy reference implementations** for complex kernels: `forward_pass`, `render_logits_chunk`, `compute_probs_loss_cce/bce`, `calculate_module_param_grads`, `backprop_error_to_hidden`, `calculate_chunk_temp_gradients`, `backprop_shared_weights/biases`, `stabilize_reduce_grad_h`, `aggregate_*_reduce`, `adam_update`.
- Writing per-kernel correctness tests that:
  1. Construct a minimal `ExecutionPlan` containing the kernel under test (isolated sub-plan).
  2. Render through `OpenCLPlanRenderer`.
  3. Retrieve the output via `RetrievalFuture`.
  4. Compare against the reference fixture within configured tolerances.
- Writing end-to-end plan rendering tests that exercise complete Act and Learn plans and verify final outputs against numpy reference orchestration.
- Configuring per-kernel tolerance tables (ADR-008, ADR-016 §9.3).
- Parameterizing the test suite for FP16 (where the device supports `cl_khr_fp16`).
- Validating the Phase 2 rollback gate: Tier 1 + OpenCL Tier 2 all green.

### Out of scope

- Tier 3 cross-backend parity tests (Phase 4 / when CPU backend exists).
- Tier 1 tests (already complete in Phase 1).
- Implementing Vulkan or CPU backends.
- Performance benchmarking (separate from correctness testing).
- Modifying shared-layer code or kernel sources.

### Key constraint: fixtures are the mathematical definition

Per ADR-016: "Analytical fixtures for closed-form kernels: the fixture *is* the mathematical definition." The fixture code expresses the kernel's intended mathematical operation using standard numpy operations (no tiling, no SIMD, no local memory). The kernel under test must produce results that match the fixture within tolerance. If the fixture and the kernel disagree, the kernel is wrong — the fixture defines truth.

### Key constraint: CPU oracle deferred

ADR-016 designates the CPU backend as the Tier 3 reference oracle. The CPU backend does not exist yet (Phase 3). Phase 2C validates the OpenCL backend against numpy fixtures, not against another backend. Tier 3 parity tests are written when ≥ 2 backends exist.

### Key constraint: plan-driven dispatch path only

Tier 2 tests exercise the new plan-driven render path (`OpenCLPlanRenderer` → `KernelBinding`), not the legacy `BatchProcessor` → `graph_recipes` → `KernelSignature` path. The legacy path remains functional but is not the subject of Tier 2 validation.

---

## 2. Tier 2 Test Architecture

Tier 2 tests follow a consistent pattern:

```
┌─────────────────────────────────┐
│  Test Case                      │
│                                 │
│  1. Construct input data        │  ← deterministic RNG or handcrafted
│  2. Compute reference output    │  ← numpy fixture / analytical formula
│  3. Build minimal plan          │  ← plan builder or manual plan construction
│  4. Upload inputs               │  ← via OpenCLBufferAllocator
│  5. Render plan                 │  ← OpenCLPlanRenderer.render()
│  6. Retrieve output             │  ← RetrievalFuture.result()
│  7. Compare output vs reference │  ← np.testing.assert_allclose(atol, rtol)
└─────────────────────────────────┘
```

**Isolation strategy:** Where possible, each kernel is tested in isolation — the test constructs a plan containing only the nodes necessary to exercise that kernel. This avoids cascading failures: if `forward_pass` has a bug, the `render_logits_chunk` test does not fail because of upstream corruption.

For kernels that inherently depend on upstream outputs (e.g., `adam_update` requires finalized gradients), the test either:
- (A) Manually uploads pre-computed intermediate buffers, bypassing upstream kernels.
- (B) Uses a minimal chain: compute upstream via numpy, upload as if they were intermediate buffers, then dispatch only the kernel under test.

**Parameterization axes:**

| Axis | Values | Mechanism |
| :--- | :--- | :--- |
| Precision | FP32, FP16 | `@pytest.mark.parametrize("precision", [PrecisionConfig.float32(), PrecisionConfig.float16()])` |
| Problem type | CCE, BCE | `@pytest.mark.parametrize("strategy", ["cce", "bce"])` for Strategy A/B kernels |
| Model geometry | Small (Iris-like), Medium | Fixture factory with configurable dimensions |
| Edge cases | Single module, single class, single batch item, maximal tile count | Explicit parameterization |

---

## 3. Target Deliverables

After Phase 2C completes:

```
tests/
├── tier2/
│   ├── __init__.py
│   ├── opencl/
│   │   ├── __init__.py
│   │   ├── conftest.py                    # Tier 2 OpenCL fixtures & session-scoped setup
│   │   ├── test_renderer_infrastructure.py # FROM Phase 2A (unchanged)
│   │   ├── test_renderer_e2e.py           # FROM Phase 2B (unchanged)
│   │   ├── test_act_forward_pass.py       # Node 4: forward_pass
│   │   ├── test_act_render_logits.py      # Node 5: render_logits_chunk
│   │   ├── test_act_loss_cce.py           # Node 6: compute_probs_loss_cce_chunk
│   │   ├── test_act_loss_bce.py           # Node 7: compute_probs_loss_bce_chunk
│   │   ├── test_learn_module_grads.py     # Node 8: calculate_module_param_grads
│   │   ├── test_learn_hidden_grads.py     # Node 9: backprop_error_to_hidden
│   │   ├── test_learn_temp_grads.py       # Node 10: calculate_chunk_temp_gradients
│   │   ├── test_learn_clip_partials.py    # Node 11: clip_partial_gradients
│   │   ├── test_learn_gather_permute.py   # Node 13: gather_and_permute_grad_h
│   │   ├── test_learn_reduction.py        # Nodes 14/15/20: aggregate_* + clip
│   │   ├── test_learn_stabilize_grad_h.py # Node 16: stabilize_reduce_grad_h
│   │   ├── test_learn_streaming_backprop.py # Nodes 17/18/19: streaming loop
│   │   ├── test_learn_normalize.py        # Node 21: normalize_gradients
│   │   ├── test_learn_adam_update.py       # Node 24: adam_update
│   │   ├── test_learn_clamp_temps.py       # Node 25: clamp_temperatures
│   │   └── test_plan_rendering_correctness.py # Full Act+Learn end-to-end
│   │
│   └── fixtures/
│       ├── __init__.py
│       ├── analytical.py              # Closed-form reference functions
│       ├── numpy_forward.py           # Numpy reference: forward pass, logits, loss
│       ├── numpy_gradients.py         # Numpy reference: gradient computation
│       ├── numpy_reduction.py         # Numpy reference: reduction tree, gradient norm
│       ├── numpy_backprop.py          # Numpy reference: shared weight/bias backprop
│       ├── numpy_update.py            # Numpy reference: Adam update, temperature clamp
│       └── data_generators.py         # Deterministic input data generation
│
├── tolerance_config.py                # NEW: per-kernel tolerance tables
└── ... (existing test infrastructure)
```

---

## 4. Task Breakdown

### Step 2C.1: Create Tier 2 test infrastructure

**Action:** Create the `tests/tier2/opencl/` directory structure and `conftest.py` with session-scoped OpenCL fixtures.

**conftest.py contents:**

```python
import pytest
import pyopencl as cl

from averaging_ensembled_classifier.src._build_config import BACKEND_OPENCL
from averaging_ensembled_classifier.src.shared import (
    HardwareProfile, PrecisionConfig, ModelSpec,
)
from averaging_ensembled_classifier.src.backends.opencl.discovery import discover_hardware
from averaging_ensembled_classifier.src.backends.opencl.renderer import OpenCLPlanRenderer
from averaging_ensembled_classifier.src.backends.opencl.type_mapping import build_compiler_flags


pytestmark = pytest.mark.skipif(
    not BACKEND_OPENCL, reason="OpenCL backend not available"
)


@pytest.fixture(scope="session")
def cl_context():
    """Session-scoped OpenCL context."""
    ctx = cl.create_some_context(interactive=False)
    yield ctx


@pytest.fixture(scope="session")
def cl_queue(cl_context):
    """Session-scoped OpenCL command queue."""
    device = cl_context.devices[0]
    queue = cl.CommandQueue(cl_context, device)
    yield queue


@pytest.fixture(scope="session")
def cl_device(cl_context):
    return cl_context.devices[0]


@pytest.fixture(scope="session")
def hardware_profile(cl_device):
    return discover_hardware(cl_device)


@pytest.fixture(scope="session")
def renderer_fp32(cl_context, cl_queue, cl_device, hardware_profile):
    """Session-scoped renderer for FP32 tests."""
    precision = PrecisionConfig.float32()
    flags = build_compiler_flags(precision, hardware_profile, c_tile_size=32)
    program = load_and_compile_kernels(cl_context, cl_device, flags)
    bindings = build_dispatch_table()
    return OpenCLPlanRenderer(cl_context, cl_queue, program, bindings)


@pytest.fixture
def small_model_spec():
    """Iris-like model for unit tests: 4 inputs, 8 hidden, 3 classes, 2 modules."""
    return ModelSpec.float32(
        num_modules=2, num_classes=3,
        input_features=4, hidden_features=8,
    )
```

**Session scoping rationale:** OpenCL context creation and kernel compilation are expensive (~100ms–1s). Session-scoped fixtures amortize this cost across all Tier 2 tests.

---

### Step 2C.2: Implement analytical reference fixtures

**Action:** Create `tests/tier2/fixtures/analytical.py` with closed-form reference functions for kernels whose outputs have simple mathematical definitions.

**Analytical fixtures:**

| Kernel | Mathematical Definition | Fixture |
| :--- | :--- | :--- |
| `compute_hidden_mask` | `mask[i] = 1.0 if activations[i] > 0.0 else 0.0` | `ref_hidden_mask(activations) → NDArray` |
| `clamp_temperatures` | `temps[i] = clamp(temps[i], T_min, T_max)` | `ref_clamp_temperatures(temps, t_min, t_max) → NDArray` |
| `normalize_gradients` | `grads[i] = grads[i] / batch_size` | `ref_normalize_gradients(grads, batch_size) → NDArray` |
| `clip_partial_gradients` | Component-wise L2 norm clip: `if ‖g‖ > T: g ← g × (T / ‖g‖)` | `ref_clip_partial_gradients(grads, threshold) → NDArray` |
| `clip_intermediate_grad` | Same as clip_partial but per-partial | `ref_clip_intermediate(partials, threshold) → NDArray` |
| `clip_shared_gradients` | Same clipping pattern for shared weight/bias gradients | `ref_clip_shared_gradients(grad_sw, grad_sb, threshold) → NDArray` |

These are definitive — the kernel must match them exactly (within floating-point tolerance). "The fixture *is* the mathematical definition" (ADR-016).

---

### Step 2C.3: Implement numpy reference implementations

**Action:** Create numpy reference implementation modules for complex kernels that cannot be expressed as one-line analytical formulas.

**`fixtures/numpy_forward.py`:**

| Reference Function | Kernel(s) | Mathematical Operation |
| :--- | :--- | :--- |
| `ref_forward_pass(input, weights_simd, biases, sample_mask)` | Node 4 | $h = \text{ReLU}(W_s \cdot x + b_s)$; mask channels by sample_mask |
| `ref_render_logits_chunk(hidden_act, module_weights, module_biases, temps)` | Node 5 | $z_{m,c} = (W_m \cdot h + b_m) / \tau_m$ |
| `ref_compute_probs_loss_cce(logits, targets, sample_mask)` | Node 6 | Softmax → cross-entropy loss per module |
| `ref_compute_probs_loss_bce(logits, targets, sample_mask)` | Node 7 | Sigmoid → binary cross-entropy loss per module per class |

**`fixtures/numpy_gradients.py`:**

| Reference Function | Kernel(s) | Mathematical Operation |
| :--- | :--- | :--- |
| `ref_module_param_grads_cce(probs, targets, hidden_act, ...)` | Node 8 (CCE) | $\nabla_{W_m} \mathcal{L}$, $\nabla_{b_m} \mathcal{L}$ for CCE |
| `ref_module_param_grads_bce(probs, targets, hidden_act, ...)` | Node 8 (BCE) | Same for BCE |
| `ref_backprop_error_to_hidden(probs, targets, module_weights, ...)` | Node 9 | $\nabla_h \mathcal{L} = \sum_m W_m^T \cdot \delta_m$ |
| `ref_temp_gradients_cce(probs, targets, logits, temps, ...)` | Node 10 (CCE) | $\nabla_{\tau_m} \mathcal{L}$ for CCE |
| `ref_temp_gradients_bce(probs, targets, logits, temps, ...)` | Node 10 (BCE) | Same for BCE |
| `ref_gather_and_permute_grad_h(grad_h_tiles)` | Node 13 | AoS→SoA permutation of scattered Grad_H partials |

**`fixtures/numpy_reduction.py`:**

| Reference Function | Kernel(s) | Mathematical Operation |
| :--- | :--- | :--- |
| `ref_reduction_tree_sum(partials, fan_in_K)` | Nodes 14/15/20 (`"sum"` variant) | Multi-stage $\log_K(N)$ summation |
| `ref_reduction_tree_sum_and_clip(partials, fan_in_K, threshold_schedule)` | Nodes 15/20 (`"sum_and_clip"` variant) | Staged sum + per-stage component-wise clip |
| `ref_stabilize_reduce_grad_h(grad_h_soa, policy_params)` | Node 16 | Specialized single-kernel Grad_H reduction with internal clipping |

**`fixtures/numpy_backprop.py`:**

| Reference Function | Kernel(s) | Mathematical Operation |
| :--- | :--- | :--- |
| `ref_backprop_shared_weights(grad_h, input_data, ...)` | Node 17 | $\nabla_{W_s} = h \cdot x^T$ (per batch chunk) |
| `ref_backprop_shared_biases(grad_h, ...)` | Node 18 | $\nabla_{b_s} = \sum h$ (per batch chunk) |

**`fixtures/numpy_update.py`:**

| Reference Function | Kernel(s) | Mathematical Operation |
| :--- | :--- | :--- |
| `ref_adam_update(params, grads, m, v, lr, beta1, beta2, eps, t)` | Node 24 | Adam optimizer: $m_t, v_t, \hat{m}_t, \hat{v}_t, \theta_{t+1}$ |

**`fixtures/data_generators.py`:**

Deterministic input generators using `numpy.random.Generator` with fixed seeds:

```python
def make_input_data(rng, batch_size, input_features, dtype=np.float32):
    """Generate deterministic input data for testing."""
    return rng.standard_normal((batch_size, input_features)).astype(dtype)

def make_model_weights(rng, spec, dtype=np.float32):
    """Generate deterministic model weights (shared + per-module)."""
    ...

def make_targets_cce(rng, batch_size, num_classes):
    """Generate random one-hot CCE targets."""
    ...

def make_targets_bce(rng, batch_size, num_classes):
    """Generate random multi-label BCE targets."""
    ...
```

All generators accept a `numpy.random.Generator` instance initialized with a fixed seed, ensuring reproducibility across runs and platforms.

---

### Step 2C.4: Write Act-phase kernel tests

**Action:** Create per-kernel test modules for Act-phase kernels.

**`test_act_forward_pass.py`:**

| Test | Description |
| :--- | :--- |
| `test_forward_pass_basic` | Small model, single tile; compare hidden activations and mask against `ref_forward_pass` |
| `test_forward_pass_multi_tile` | Multi-tile dispatch; verify all tiles produce correct partial results |
| `test_forward_pass_relu_activation` | Verify ReLU gating: negative pre-activations → zero activations, zero mask |
| `test_forward_pass_sample_mask` | Verify masked samples produce zero activations |
| `test_forward_pass_simd_padding` | Verify SIMD-padded dimensions don't corrupt logical output region |

**`test_act_render_logits.py`:**

| Test | Description |
| :--- | :--- |
| `test_render_logits_basic` | Small model; compare logits against `ref_render_logits_chunk` |
| `test_render_logits_temperature_scaling` | Verify logit division by temperature: $z / \tau$ |
| `test_render_logits_multi_module` | Verify each module's logits are independent |

**`test_act_loss_cce.py` and `test_act_loss_bce.py`:**

| Test | Description |
| :--- | :--- |
| `test_cce_probs_softmax` | Verify softmax probabilities sum to 1.0 per sample |
| `test_cce_loss_basic` | Compare CCE loss against `ref_compute_probs_loss_cce` |
| `test_cce_loss_numerical_stability` | Large logit magnitudes; verify no NaN/Inf |
| `test_bce_probs_sigmoid` | Verify sigmoid probabilities ∈ [0, 1] |
| `test_bce_loss_basic` | Compare BCE loss against `ref_compute_probs_loss_bce` |
| `test_bce_loss_saturation` | Extreme targets (all 0, all 1); verify numerical stability |

---

### Step 2C.5: Write Learn-phase gradient production tests

**Action:** Test Nodes 8–11 (gradient computation and partial clipping).

**`test_learn_module_grads.py`:**

| Test | Description |
| :--- | :--- |
| `test_module_grads_cce_basic` | Strategy B CCE variant; compare against `ref_module_param_grads_cce` |
| `test_module_grads_bce_basic` | Strategy B BCE variant; compare against `ref_module_param_grads_bce` |
| `test_module_grads_multi_tile` | Verify per-tile placement: each tile writes to its designated region |
| `test_module_grads_partial_placement` | Verify `grid_mod_cls` placement contract: `(module_idx, class_chunk_idx)` decode |

**`test_learn_hidden_grads.py`:**

| Test | Description |
| :--- | :--- |
| `test_hidden_grads_cce` | Strategy A with `FLAG_problem_type=CCE`; compare against `ref_backprop_error_to_hidden` |
| `test_hidden_grads_bce` | Strategy A with `FLAG_problem_type=BCE`; same reference |
| `test_hidden_grads_gradient_flow` | Verify gradients propagate through ReLU mask correctly: zero where mask is 0 |

**`test_learn_temp_grads.py`:**

| Test | Description |
| :--- | :--- |
| `test_temp_grads_cce` | Temperature gradient for CCE; compare against `ref_temp_gradients_cce` |
| `test_temp_grads_bce` | Temperature gradient for BCE; compare against `ref_temp_gradients_bce` |

**`test_learn_clip_partials.py`:**

| Test | Description |
| :--- | :--- |
| `test_clip_below_threshold` | Gradients below threshold unchanged |
| `test_clip_above_threshold` | Gradients above threshold scaled to threshold norm |
| `test_clip_preserves_direction` | Clipped gradient direction matches original direction |
| `test_clip_zero_gradient` | Zero gradient remains zero (no division by zero) |

---

### Step 2C.6: Write Learn-phase reduction & aggregation tests

**Action:** Test the reduction tree rendering (Nodes 14/15/20) and Node 16 (specialized Grad_H reduction).

**`test_learn_reduction.py`:**

| Test | Description |
| :--- | :--- |
| `test_reduction_sum_single_stage` | $N \leq K$; single-stage sum; compare against `ref_reduction_tree_sum` |
| `test_reduction_sum_multi_stage` | $N > K$; multi-stage; compare against `ref_reduction_tree_sum` |
| `test_reduction_sum_and_clip_single` | Single-stage sum-and-clip; verify clipping at threshold |
| `test_reduction_sum_and_clip_multi` | Multi-stage; verify per-stage threshold schedule applied correctly |
| `test_reduction_threshold_schedule` | Verify $T_j = T_{\text{alg}} + \lambda \cdot j^2$ schedule correctness |
| `test_reduction_safety_clamp` | Verify safety ceiling $T_{\text{safe}} = \text{FP\_MAX} / K$ applied |
| `test_reduction_register_vs_local` | Same input, same K; verify register-reduce and local-reduce produce identical results |
| `test_reduction_non_power_of_K` | $N$ not a power of $K$; verify correct handling of partial last stage |
| `test_reduction_probs_diagnostic` | Node 14 (diagnostic probs aggregation, `"sum"` variant); verify averaged probs match |

**`test_learn_gather_permute.py`:**

| Test | Description |
| :--- | :--- |
| `test_gather_permute_basic` | AoS→SoA permutation; compare against `ref_gather_and_permute_grad_h` |
| `test_gather_permute_multi_tile` | Multi-tile input → contiguous SoA output; verify no interleaving errors |

**`test_learn_stabilize_grad_h.py`:**

| Test | Description |
| :--- | :--- |
| `test_stabilize_reduce_basic` | Small model; compare against `ref_stabilize_reduce_grad_h` |
| `test_stabilize_reduce_with_clipping` | Verify internal multi-stage clipping within Node 16 (ADR-005 opacity) |
| `test_stabilize_reduce_row_independence` | Each row of Grad_H reduced independently; verify no cross-row contamination |

---

### Step 2C.7: Write Learn-phase streaming backprop tests

**Action:** Test the streaming loop (Nodes 17→18→19) rendered by `_render_streaming_loop()`.

**`test_learn_streaming_backprop.py`:**

| Test | Description |
| :--- | :--- |
| `test_shared_weights_backprop_basic` | Single chunk; compare `∇W_s` against `ref_backprop_shared_weights` |
| `test_shared_biases_backprop_basic` | Single chunk; compare `∇b_s` against `ref_backprop_shared_biases` |
| `test_streaming_multi_chunk` | Multiple chunks; verify each chunk writes to correct placement in collection buffer |
| `test_streaming_clip_per_chunk` | Verify per-chunk clipping at Node 19; compare against `ref_clip_shared_gradients` |
| `test_streaming_final_reduction` | After streaming loop, verify reduction (Node 20) produces correct aggregated `∇W_s`, `∇b_s` |
| `test_streaming_stride_arithmetic` | Verify `ParameterStride` correctly computes `batch_chunk_offset` per chunk |
| `test_streaming_last_chunk_smaller` | Non-divisible `total_extent / chunk_count`; verify last chunk handles remainder |

---

### Step 2C.8: Write Learn-phase update kernel tests

**Action:** Test Nodes 21, 24, 25 (normalization, Adam update, temperature clamping).

**`test_learn_normalize.py`:**

| Test | Description |
| :--- | :--- |
| `test_normalize_basic` | `grads / batch_size`; compare against `ref_normalize_gradients` |
| `test_normalize_preserves_sign` | Verify normalization preserves gradient sign |

**`test_learn_adam_update.py`:**

| Test | Description |
| :--- | :--- |
| `test_adam_update_basic` | Single step; compare parameters against `ref_adam_update` |
| `test_adam_update_bias_correction` | Verify bias correction terms $\hat{m}_t = m_t / (1 - \beta_1^t)$, $\hat{v}_t = v_t / (1 - \beta_2^t)$ |
| `test_adam_update_multi_step` | Multiple steps; accumulate momentum/variance; compare against reference |
| `test_adam_update_fp64_bias_terms` | Verify host-computed `beta1**t`, `beta2**t` in FP64 match device results (precision invariant from CONCEPT.md §11) |
| `test_adam_update_per_param_group` | Separate updates for shared, module, temp parameters; verify independence |

**`test_learn_clamp_temps.py`:**

| Test | Description |
| :--- | :--- |
| `test_clamp_basic` | Temperatures within range unchanged |
| `test_clamp_below_min` | Below-minimum temperatures raised to `T_min` |
| `test_clamp_above_max` | Above-maximum temperatures lowered to `T_max` |

---

### Step 2C.9: Write end-to-end plan rendering correctness tests

**Action:** Create `test_plan_rendering_correctness.py` with tests that exercise complete Act and Learn plans, verifying final outputs against a numpy reference orchestration that computes the full forward + backward pass.

**`test_plan_rendering_correctness.py`:**

| Test | Description |
| :--- | :--- |
| `test_act_plan_final_probs_cce` | Full Act plan (CCE); verify `final_probs` matches numpy forward+softmax+aggregate |
| `test_act_plan_final_probs_bce` | Full Act plan (BCE); verify `final_probs` matches numpy forward+sigmoid+aggregate |
| `test_learn_plan_parameter_update_cce` | Full Learn plan (CCE); verify updated parameters match numpy reference forward+backward+Adam |
| `test_learn_plan_parameter_update_bce` | Full Learn plan (BCE); same for BCE |
| `test_act_then_learn_roundtrip` | Submit Act plan → retrieve predictions → submit Learn plan → verify parameter update |
| `test_multi_batch_convergence` | Run 5 consecutive batch cycles; verify loss decreases (sanity convergence check, not strict tolerance) |

**Numpy reference orchestration:**

The end-to-end reference computes the entire forward + backward pass in numpy:

```python
def numpy_reference_train_step(x, y, params, spec, strategy, policy):
    """Full reference implementation of one Act+Learn cycle.

    Returns: (predictions, updated_params)
    """
    # Forward pass
    hidden = relu(x @ params.weights_shared.T + params.biases_shared)
    logits = {}
    for m in range(spec.num_modules):
        logits[m] = (hidden @ params.module_weights[m].T + params.module_biases[m]) / params.temps[m]

    # Probabilities + loss
    if strategy == "cce":
        probs = {m: softmax(logits[m]) for m in range(spec.num_modules)}
        loss = ...
    else:
        probs = {m: sigmoid(logits[m]) for m in range(spec.num_modules)}
        loss = ...

    # Aggregate probs (average across modules)
    final_probs = sum(probs.values()) / spec.num_modules

    # Backward pass (gradient computation, reduction, update)
    ...

    return final_probs, updated_params
```

This reference is intentionally straightforward — no tiling, no streaming, no reduction trees. It computes the mathematically correct answer. The plan-driven dispatch must converge to the same answer within tolerance.

---

### Step 2C.10: Configure tolerance tables

**Action:** Create `tests/tolerance_config.py` with per-kernel tolerance overrides.

**Base tolerances (DESIGN.md §9.3):**

| Precision | Default `atol` | Default `rtol` |
| :--- | :--- | :--- |
| FP32 | `1e-5` | `1e-5` |
| FP16 | `1e-2` | `1e-2` |

**Per-kernel overrides (kernels with higher numerical sensitivity):**

| Kernel | FP32 Override | Rationale |
| :--- | :--- | :--- |
| `compute_probs_loss_cce_chunk` | `atol=1e-4, rtol=1e-4` | Softmax `exp()` and `log()` accumulate error |
| `compute_probs_loss_bce_chunk` | `atol=1e-4, rtol=1e-4` | Sigmoid near saturation |
| `stabilize_reduce_grad_h` | `atol=1e-4, rtol=1e-4` | Multi-stage internal reduction accumulates rounding |
| `adam_update` | `atol=1e-4, rtol=1e-4` | Division by $\sqrt{\hat{v}_t} + \epsilon$; sensitive near zero |
| Reduction tree (multi-stage) | `atol=1e-4, rtol=1e-4` | Summation order differs between reference (sequential) and kernel (tree-structured) |

**Configuration structure:**

```python
from dataclasses import dataclass

@dataclass(frozen=True)
class TolerancePair:
    atol: float
    rtol: float

# Default tolerances
FP32_DEFAULT = TolerancePair(atol=1e-5, rtol=1e-5)
FP16_DEFAULT = TolerancePair(atol=1e-2, rtol=1e-2)

# Per-kernel overrides
KERNEL_TOLERANCES: dict[str, dict[str, TolerancePair]] = {
    "compute_probs_loss_cce_chunk": {
        "fp32": TolerancePair(atol=1e-4, rtol=1e-4),
        "fp16": TolerancePair(atol=5e-2, rtol=5e-2),
    },
    "compute_probs_loss_bce_chunk": {
        "fp32": TolerancePair(atol=1e-4, rtol=1e-4),
        "fp16": TolerancePair(atol=5e-2, rtol=5e-2),
    },
    "stabilize_reduce_grad_h": {
        "fp32": TolerancePair(atol=1e-4, rtol=1e-4),
    },
    "adam_update": {
        "fp32": TolerancePair(atol=1e-4, rtol=1e-4),
    },
    "aggregate_local_reduce": {
        "fp32": TolerancePair(atol=1e-4, rtol=1e-4),
    },
    "aggregate_register_reduce": {
        "fp32": TolerancePair(atol=1e-4, rtol=1e-4),
    },
}


def get_tolerance(kernel_name: str, precision_label: str) -> TolerancePair:
    """Look up tolerance for a kernel, falling back to defaults."""
    if kernel_name in KERNEL_TOLERANCES:
        overrides = KERNEL_TOLERANCES[kernel_name]
        if precision_label in overrides:
            return overrides[precision_label]
    return FP32_DEFAULT if precision_label == "fp32" else FP16_DEFAULT
```

---

### Step 2C.11: FP16 parameterization

**Action:** Parameterize the Tier 2 test suite for FP16 precision.

**Strategy:**

1. All test modules accept a `precision` fixture parameter (`PrecisionConfig.float32()` / `PrecisionConfig.float16()`).
2. FP16 tests are marked `@pytest.mark.fp16` for selective execution.
3. FP16 device support is checked at collection time: `cl.device_info.EXTENSIONS` must contain `"cl_khr_fp16"`. If absent, FP16 tests are skipped.
4. Tolerance lookup uses `precision_label` (`"fp32"` or `"fp16"`) to select the appropriate tolerance pair.

**FP16-specific concerns:**

| Concern | Handling |
| :--- | :--- |
| Overflow in intermediate computation | Numpy reference uses FP32 internally; kernel operates in FP16. Tolerance accounts for range difference. |
| Softmax/sigmoid saturation | FP16 `exp()` overflows earlier; loss values may clip to `±65504` (max half). Test verifies no NaN/Inf rather than exact value match. |
| Adam update $\epsilon$ | `PrecisionConfig.float16().epsilon` is larger (`~1e-3` vs. `1e-7`); Adam denominator stability is tested. |
| Reduction tree safety bound | $T_{\text{safe}} = 65504 / K$; FP16 safety bound is much tighter. Verify threshold schedule respects this. |

---

### Step 2C.12: Validate Phase 2 rollback gate

**Action:** Run the complete test suite and confirm the Phase 2 gate is green.

**Procedure:**

1. **Tier 1 green (no regressions):**
   ```
   pytest tests/tier1/ -v
   ```
   Expected: all 140 Tier 1 tests pass.

2. **Existing tests green (no regressions):**
   ```
   pytest src/tests/ tests/ -v --ignore=tests/tier1 --ignore=tests/tier2
   ```
   Expected: all 192 Phase 0 tests pass.

3. **OpenCL Tier 2 green (FP32):**
   ```
   pytest tests/tier2/opencl/ -v -m "not fp16"
   ```
   Expected: all FP32 Tier 2 tests pass.

4. **OpenCL Tier 2 FP16 (if device supports cl_khr_fp16):**
   ```
   pytest tests/tier2/opencl/ -v -m fp16
   ```
   Expected: all FP16 tests pass or skip (if device lacks FP16).

5. **Full gate summary:**
   ```
   pytest tests/ src/tests/ -v
   ```
   Expected: Tier 1 (140) + Phase 0 (192) + Phase 2A infrastructure + Phase 2B e2e + Phase 2C Tier 2 = all green.

**Gate criteria (ADR-017):**

- Tier 1 green: no regressions from shared-layer plan model tests.
- OpenCL Tier 2 green: every kernel correctness test passes at configured tolerances.
- No NaN/Inf values in any kernel output for valid input.
- Existing legacy test suite unaffected.

**Post-gate promotion:** Upon green gate, the OpenCL renderer path is validated. The `aec_backend_opencl` feature flag semantic is `auto` → `enabled` (CI requires the backend). This does not affect legacy code — both paths coexist until Phase 6.

---

## 5. Fixture Architecture

The fixture architecture follows a layered design:

```
┌─────────────────────────────────────────────────────────┐
│ Test Case                                               │
│   Constructs input data (data_generators.py)            │
│   Calls reference function (fixtures/)                  │
│   Builds plan and renders (OpenCLPlanRenderer)          │
│   Compares: np.testing.assert_allclose(actual, ref)     │
└────────────────────────────┬────────────────────────────┘
                             │ consumes
┌────────────────────────────┼────────────────────────────┐
│ Fixture Layer              │                            │
│                            ▼                            │
│  ┌─────────────────┐  ┌────────────────────┐           │
│  │  Analytical      │  │  Numpy Reference   │           │
│  │  (closed-form)   │  │  (complex kernels) │           │
│  │  analytical.py   │  │  numpy_*.py        │           │
│  └─────────────────┘  └────────────────────┘           │
│                                                         │
│  ┌─────────────────────────────────────────────┐       │
│  │  Data Generator                              │       │
│  │  (deterministic RNG, fixed seeds)            │       │
│  │  data_generators.py                          │       │
│  └─────────────────────────────────────────────┘       │
└─────────────────────────────────────────────────────────┘
```

**Fixture independence:** Each fixture function is self-contained — it takes input arrays and model parameters, computes the reference output using only numpy, and returns the result. No fixture depends on OpenCL or any backend.

**Fixture reuse:** The same fixtures are consumed by Tier 2 tests for all three backends (OpenCL in Phase 2C, CPU in Phase 3, Vulkan in Phase 5). They are placed in `tests/tier2/fixtures/` (not `tests/tier2/opencl/fixtures/`) for cross-backend reuse.

---

## 6. Per-Kernel Test Matrix

Complete matrix of kernels × test dimensions:

| Kernel | Node | Strategy | Tests (FP32) | Tests (FP16) | Tolerance Override |
| :--- | :--- | :--- | :--- | :--- | :--- |
| `forward_pass` | 4 | — | 5 | 5 | No |
| `render_logits_chunk` | 5 | — | 3 | 3 | No |
| `compute_probs_loss_cce_chunk` | 6 | B | 3 | 3 | Yes |
| `compute_probs_loss_bce_chunk` | 7 | B | 3 | 3 | Yes |
| `calculate_module_param_grads_cce` | 8 | B | 4 | 4 | No |
| `calculate_module_param_grads_bce` | 8 | B | 4 | 4 | No |
| `backprop_error_to_hidden` | 9 | A | 3 | 3 | No |
| `calculate_chunk_temp_gradients` | 10 | A | 2 | 2 | No |
| `clip_partial_gradients` | 11 | — | 4 | 4 | No |
| `gather_and_permute_grad_h` | 13 | — | 2 | 2 | No |
| `aggregate_register_reduce` | 14/15/20 | — | 3 | 3 | Yes |
| `aggregate_local_reduce` | 14/15/20 | — | 3 | 3 | Yes |
| `clip_intermediate_grad` | 15/20 | — | 2 | 2 | No |
| `stabilize_reduce_grad_h` | 16 | — | 3 | 3 | Yes |
| `backprop_shared_weights` | 17 | — | 2 | 2 | No |
| `backprop_shared_biases` | 18 | — | 2 | 2 | No |
| `clip_shared_gradients` | 19 | — | 2 | 2 | No |
| `normalize_gradients` | 21 | — | 2 | 2 | No |
| `adam_update` | 24 | — | 5 | 5 | Yes |
| `clamp_temperatures` | 25 | — | 3 | 3 | No |
| **End-to-end** | All | CCE+BCE | 6 | 6 | Various |
| **Total** | | | **~65** | **~65** | |

Estimated total: **~130 tests** (65 FP32 + 65 FP16, with FP16 conditionally skipped).

---

## 7. Tolerance Configuration

### Philosophy

Floating-point comparison tolerances serve two roles:

1. **Correctness assertion:** The kernel must produce results that are numerically equivalent to the mathematical reference, up to floating-point representation and ordering differences.
2. **Regression signal:** Tolerance values are tight enough that genuine algorithmic bugs cause failures, but loose enough that non-deterministic floating-point ordering (e.g., different reduction tree traversal vs. sequential numpy summation) does not cause spurious failures.

### Tolerance table governance

The tolerance table in `tolerance_config.py` is a controlled artifact:
- **Relaxation requires justification.** If a test fails and the fix is to relax the tolerance, the justification must be documented (e.g., "reduction tree summation order differs from sequential numpy; accumulated rounding error is $O(\log N \cdot \epsilon)$").
- **Tightening is always welcome.** If a kernel consistently passes with tighter tolerances, the table should be updated.
- **FP16 tolerances are inherently looser** because the format has only 10 mantissa bits (vs. 23 for FP32). The 2 orders-of-magnitude tolerance increase (1e-5 → 1e-2) reflects this.

### Numerical stability assertions

In addition to closeness checks, every test asserts:
```python
assert not np.any(np.isnan(actual)), "Kernel produced NaN"
assert not np.any(np.isinf(actual)), "Kernel produced Inf"
```

These are hard failures with no tolerance — NaN/Inf in any output is always a bug.

---

## 8. Test Infrastructure Details

### Session-scoped resource management

```
Session Start
│
├── cl.create_some_context()          # once per test session
├── cl.CommandQueue()                 # once
├── discover_hardware()               # once
├── load_and_compile_kernels(fp32)    # once
├── build_dispatch_table()            # once
├── OpenCLPlanRenderer(fp32)          # once
│
├── [FP16 path, if device supports]:
│   ├── load_and_compile_kernels(fp16)
│   └── OpenCLPlanRenderer(fp16)
│
├── Run all Tier 2 tests
│   ├── test_act_forward_pass.py (uses renderer_fp32)
│   ├── test_act_loss_cce.py
│   ├── ...
│   └── test_plan_rendering_correctness.py
│
└── Session Teardown
    └── cl resources released
```

### Deterministic random state

Every test that uses randomized input data creates a fresh `numpy.random.Generator(numpy.random.PCG64(seed=<test_specific_seed>))`. The seed is:
- Deterministic (same across runs).
- Unique per test (avoids correlation between tests).
- Derived from the test name: `seed = hash(test_name) % 2**31`.

### Test markers

```python
# Tier markers
pytest.mark.tier2       # Tier 2 kernel correctness
pytest.mark.opencl      # OpenCL backend required

# Precision markers
pytest.mark.fp16        # FP16 variant (conditionally skipped)

# Strategy markers
pytest.mark.cce         # CCE-specific test
pytest.mark.bce         # BCE-specific test
```

Selective execution examples:
```bash
pytest tests/tier2/opencl/ -m "tier2 and opencl and not fp16"    # FP32 only
pytest tests/tier2/opencl/ -m "tier2 and opencl and fp16"        # FP16 only
pytest tests/tier2/opencl/ -k "test_learn_adam"                  # Adam tests only
```

---

## 9. Risk Register

| Risk | Likelihood | Impact | Mitigation |
| :--- | :--- | :--- | :--- |
| Numpy reference implementation has a bug, leading to false positive test results | Medium | High | Cross-validate numpy references against known analytical solutions where possible; review references against `kernels.cl.h` specification; Tier 3 (Phase 4) provides independent cross-backend validation |
| FP32 tolerance too tight, causing flaky CI on different OpenCL devices | Medium | Medium | Test on multiple devices in CI (Intel, AMD, NVIDIA); loosen per-kernel tolerances only with documented justification; use `rtol` (relative) in preference to `atol` (absolute) for outputs with varying magnitude |
| FP16 device support absent in CI; FP16 tests never run | Medium | Medium | At least one CI runner must have FP16-capable OpenCL device; if unavailable, FP16 coverage is tracked as a known gap |
| Session-scoped fixtures mask per-test state leaks | Low | Medium | Each test allocates and releases its own plan buffers; session-scoped context is read-only after construction |
| Large model geometries cause OOM on CI test devices | Low | Medium | Test model geometries are deliberately small (Iris-like: 4→8→3, 2 modules); medium geometries only for edge-case tests |
| Reduction tree test with different tree traversal order produces tolerance-failing results | Medium | Medium | Known issue: tree-structured reduction has different summation order than sequential numpy. Multi-stage reduction tests use the relaxed tolerance (1e-4). If still failing, compare against a tree-structured numpy reference that matches the kernel's traversal order. |
| End-to-end tests too slow for CI | Low | Low | Small model geometry keeps dispatch count low (~50 kernel dispatches for a full Learn plan). Target: each e2e test completes in < 5 seconds. |
