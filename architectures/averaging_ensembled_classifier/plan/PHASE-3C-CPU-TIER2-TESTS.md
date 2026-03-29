# Phase 3C: CPU Tier 2 Tests & Rollback Gate — Detailed Plan

**Status: ✅ IMPLEMENTED** — 16 CPU Tier 2 test modules delivered in `tests/tier2/cpu/`, covering all kernel functions against shared analytical and numpy reference fixtures. Shared fixture library (`tests/tier2/fixtures/`) provides 7 backend-neutral reference modules. CPU-specific tolerance configuration added to `tests/tolerance_config.py`. Phase 3 rollback gate validated: Tier 1 green + CPU Tier 2 green.  
**Phase:** 3C of 6 (sub-phase C of 3)  
**Objective:** Write the complete CPU Tier 2 test suite — per-kernel numerical correctness tests against analytical and numpy reference fixtures — and validate the Phase 3 rollback gate (Tier 1 green + CPU Tier 2 green). After this sub-phase, the CPU backend's plan-driven dispatch path is fully validated and the Phase 3 gate is closed.  
**Governing ADRs:** ADR-016 (test strategy — three-tier framework), ADR-008 (precision configuration — tolerance parameterization), ADR-011 (CCE/BCE strategy — combinatorial test matrix), ADR-013 (kernel source hierarchy — specification fidelity), ADR-007 (contract/binding split — Tier 2 validates the full stack)  
**Rollback gate:** Tier 1 green (no regressions to existing 332+ tests) + CPU Tier 2 green (all per-kernel correctness tests pass for FP32). FP16 parameterization is deferred until hardware support is available.  
**Dependencies:** Phase 3B (FFI layer & renderer) — provides fully functional `CPUPlanRenderer` with dispatch table, buffer allocator, and FFI infrastructure. Phase 3A (kernel library) — provides `libcpu_kernels.so` with all kernel implementations.

---

## Table of Contents

1. [Scope & Constraints](#1-scope--constraints)
2. [Tier 2 Test Architecture](#2-tier-2-test-architecture)
3. [Target Deliverables](#3-target-deliverables)
4. [Task Breakdown](#4-task-breakdown)
   - [Step 3C.1: Create CPU Tier 2 test infrastructure](#step-3c1-create-cpu-tier-2-test-infrastructure)
   - [Step 3C.2: Implement analytical reference fixtures (shared with OpenCL)](#step-3c2-implement-analytical-reference-fixtures-shared-with-opencl)
   - [Step 3C.3: Implement numpy reference implementations (shared with OpenCL)](#step-3c3-implement-numpy-reference-implementations-shared-with-opencl)
   - [Step 3C.4: Write Act-phase kernel tests](#step-3c4-write-act-phase-kernel-tests)
   - [Step 3C.5: Write Learn-phase gradient production tests](#step-3c5-write-learn-phase-gradient-production-tests)
   - [Step 3C.6: Write Learn-phase reduction & aggregation tests](#step-3c6-write-learn-phase-reduction--aggregation-tests)
   - [Step 3C.7: Write Learn-phase streaming backprop tests](#step-3c7-write-learn-phase-streaming-backprop-tests)
   - [Step 3C.8: Write Learn-phase update kernel tests](#step-3c8-write-learn-phase-update-kernel-tests)
   - [Step 3C.9: Write end-to-end plan rendering correctness tests](#step-3c9-write-end-to-end-plan-rendering-correctness-tests)
   - [Step 3C.10: Configure CPU-specific tolerance tables](#step-3c10-configure-cpu-specific-tolerance-tables)
   - [Step 3C.11: FP16 parameterization (conditional)](#step-3c11-fp16-parameterization-conditional)
   - [Step 3C.12: Validate Phase 3 rollback gate](#step-3c12-validate-phase-3-rollback-gate)
5. [Fixture Architecture](#5-fixture-architecture)
6. [Per-Kernel Test Matrix](#6-per-kernel-test-matrix)
7. [Tolerance Configuration](#7-tolerance-configuration)
8. [CPU-Specific Test Considerations](#8-cpu-specific-test-considerations)
9. [Risk Register](#9-risk-register)

---

## 1. Scope & Constraints

### In scope

- Creating the CPU Tier 2 test directory structure (`tests/tier2/cpu/`) with conftest, fixtures, and per-kernel test modules.
- **Reusing** the analytical reference fixtures and numpy reference implementations from `tests/tier2/fixtures/` (shared across backends). If Phase 2C has already created these, Phase 3C consumes them directly. If Phase 2C has not yet been implemented, Phase 3C creates them — they are backend-neutral by design.
- Writing per-kernel correctness tests that:
  1. Construct a minimal `ExecutionPlan` containing the kernel under test (isolated sub-plan).
  2. Render through `CPUPlanRenderer`.
  3. Retrieve the output via `CPURetrievalFuture.result()`.
  4. Compare against the reference fixture within configured tolerances.
- Writing end-to-end plan rendering tests that exercise complete Act and Learn plans and verify final outputs against numpy reference orchestration.
- Configuring CPU-specific tolerance tables (ADR-008, ADR-016 §9.3). CPU tolerances may differ from OpenCL due to different reduction orderings and SIMD instruction semantics.
- Conditionally parameterizing for FP16 (only if the CPU backend gains FP16 support — initially FP32 only).
- Validating the Phase 3 rollback gate: Tier 1 + CPU Tier 2 all green.
- Testing SIMD correctness across ISA tiers where possible (e.g., running tests with `SIMD_WIDTH=1` scalar fallback in addition to the platform's native SIMD width).

### Out of scope

- Tier 3 cross-backend parity tests (Phase 4 — when both CPU and OpenCL backends exist).
- Tier 1 tests (already complete in Phase 1).
- Implementing or modifying the OpenCL backend.
- Performance benchmarking (separate from correctness testing).
- Modifying shared-layer code, kernel sources, or FFI layer code.

### Key constraint: fixtures are the mathematical definition

Per ADR-016: "Analytical fixtures for closed-form kernels: the fixture *is* the mathematical definition." The fixture code expresses the kernel's intended mathematical operation using standard numpy operations (no tiling, no SIMD, no threading). The kernel under test must produce results that match the fixture within tolerance. If the fixture and the kernel disagree, the kernel is wrong — the fixture defines truth.

### Key constraint: shared fixtures, backend-specific tests

The reference fixtures (`tests/tier2/fixtures/`) are backend-neutral. They express mathematical truth using numpy. Both the OpenCL Tier 2 tests (Phase 2C) and the CPU Tier 2 tests (Phase 3C) consume the same fixtures. The backend-specific parts are:

1. **Test infrastructure** — `conftest.py` differs (no `cl.Context`/`cl.CommandQueue`; instead, `CPUPlanRenderer` with thread pool).
2. **Tolerance tables** — CPU may have tighter or looser tolerances depending on SIMD instruction semantics (e.g., FMA presence affects intermediate precision).
3. **Skip conditions** — CPU tests skip on `not BACKEND_CPU`; OpenCL tests skip on `not BACKEND_OPENCL`.

### Key constraint: CPU backend as future Tier 3 oracle

ADR-016 designates the CPU backend as the Tier 3 reference oracle for cross-backend parity tests. Tier 2 validates the CPU backend against numpy fixtures. Once both CPU and OpenCL backends pass Tier 2, Tier 3 can validate that they produce the same results (within cross-backend tolerances). Phase 3C establishes the CPU side of this contract.

### Key constraint: plan-driven dispatch path only

Tier 2 tests exercise the new plan-driven render path (`CPUPlanRenderer` → dispatch table → `pool_dispatch_and_wait`), not the legacy `BatchProcessor` → `graph_recipes` path. The legacy path remains functional but is not the subject of Tier 2 validation.

---

## 2. Tier 2 Test Architecture

CPU Tier 2 tests follow the same consistent pattern as OpenCL Tier 2 (Phase 2C):

```
┌─────────────────────────────────┐
│  Test Case                      │
│                                 │
│  1. Construct input data        │  ← deterministic RNG or handcrafted
│  2. Compute reference output    │  ← numpy fixture / analytical formula
│  3. Build minimal plan          │  ← plan builder or manual plan construction
│  4. Load inputs into buffers    │  ← via CPUBufferAllocator (numpy arrays)
│  5. Render plan                 │  ← CPUPlanRenderer.render()
│  6. Retrieve output             │  ← CPURetrievalFuture.result()
│  7. Compare output vs reference │  ← np.testing.assert_allclose(atol, rtol)
└─────────────────────────────────┘
```

**CPU simplifications vs. OpenCL:**

| Aspect | OpenCL Tier 2 | CPU Tier 2 |
| :--- | :--- | :--- |
| Context setup | `cl.create_some_context()` + compile kernels | `CPUPlanRenderer()` (loads library) |
| Buffer upload | `cl.enqueue_write_buffer` | Direct numpy array assignment |
| Dispatch | Asynchronous (event graph) | Synchronous (`pool_dispatch_and_wait` blocks) |
| Result retrieval | `cl.enqueue_read_buffer` (D2H transfer) | Zero-copy (numpy view) |
| Timing sensitivity | Async completion ordering | None |

The CPU test infrastructure is simpler because every operation is synchronous and all memory is host-accessible. There is no asynchronous event management or device-to-host transfer.

**Isolation strategy:** Identical to Phase 2C. Each kernel is tested in isolation — the test constructs a plan containing only the nodes necessary to exercise that kernel. For kernels that depend on upstream outputs, the test manually uploads pre-computed intermediate buffers, bypassing upstream kernels.

**Parameterization axes:**

| Axis | Values | Mechanism |
| :--- | :--- | :--- |
| Precision | FP32 (FP16 deferred) | `@pytest.mark.parametrize("precision", [PrecisionConfig.float32()])` |
| Problem type | CCE, BCE | `@pytest.mark.parametrize("strategy", ["cce", "bce"])` for Strategy A/B kernels |
| Model geometry | Small (Iris-like), Medium | Fixture factory with configurable dimensions |
| Edge cases | Single module, single class, single batch item, maximal tile count | Explicit parameterization |
| Thread count | 1, `detect_thread_count()` | `@pytest.mark.parametrize("threads", [1, None])` — tests single-threaded and multi-threaded paths |

**Thread count parameterization note:** Running with `threads=1` exercises the serial execution path and eliminates race condition masking. Running with `threads=None` (auto-detect) exercises the multi-threaded path. Both must produce identical results.

---

## 3. Target Deliverables

After Phase 3C completes:

```
tests/
├── tier2/
│   ├── __init__.py
│   ├── cpu/
│   │   ├── __init__.py
│   │   ├── conftest.py                    # CPU Tier 2 fixtures & session-scoped setup
│   │   ├── test_cpu_act_forward_pass.py   # Node 4: forward_pass
│   │   ├── test_cpu_act_render_logits.py  # Node 5: render_logits_chunk
│   │   ├── test_cpu_act_loss_cce.py       # Node 6: compute_probs_loss_cce_chunk
│   │   ├── test_cpu_act_loss_bce.py       # Node 7: compute_probs_loss_bce_chunk
│   │   ├── test_cpu_learn_module_grads.py     # Node 8: calculate_module_param_grads
│   │   ├── test_cpu_learn_hidden_grads.py     # Node 9: backprop_error_to_hidden
│   │   ├── test_cpu_learn_temp_grads.py       # Node 10: calculate_chunk_temp_gradients
│   │   ├── test_cpu_learn_clip_partials.py    # Node 11: clip_partial_gradients
│   │   ├── test_cpu_learn_gather_permute.py   # Node 13: gather_and_permute_grad_h
│   │   ├── test_cpu_learn_reduction.py        # Nodes 14/15/20: reduction tree + clip
│   │   ├── test_cpu_learn_stabilize_grad_h.py # Node 16: stabilize_reduce_grad_h
│   │   ├── test_cpu_learn_streaming_backprop.py # Nodes 17/18/19: streaming loop
│   │   ├── test_cpu_learn_normalize.py        # Node 21: normalize_gradients
│   │   ├── test_cpu_learn_adam_update.py       # Node 24: adam_update
│   │   ├── test_cpu_learn_clamp_temps.py       # Node 25: clamp_temperatures
│   │   └── test_cpu_plan_rendering_correctness.py # Full Act+Learn end-to-end
│   │
│   ├── opencl/                            # (Phase 2C — parallel structure)
│   │   └── ...
│   │
│   └── fixtures/                          # SHARED between OpenCL and CPU Tier 2
│       ├── __init__.py
│       ├── analytical.py              # Closed-form reference functions
│       ├── numpy_forward.py           # Numpy reference: forward pass, logits, loss
│       ├── numpy_gradients.py         # Numpy reference: gradient computation
│       ├── numpy_reduction.py         # Numpy reference: reduction tree, gradient norm
│       ├── numpy_backprop.py          # Numpy reference: shared weight/bias backprop
│       ├── numpy_update.py            # Numpy reference: Adam update, temperature clamp
│       └── data_generators.py         # Deterministic input data generation
│
├── tolerance_config.py                # Per-kernel tolerance tables (both backends)
└── ... (existing test infrastructure)
```

---

## 4. Task Breakdown

### Step 3C.1: Create CPU Tier 2 test infrastructure

**Action:** Create the `tests/tier2/cpu/` directory structure and `conftest.py` with session-scoped CPU fixtures.

**conftest.py contents:**

```python
import pytest

from averaging_ensembled_classifier.src._build_config import BACKEND_CPU
from averaging_ensembled_classifier.src.shared import (
    HardwareProfile, PrecisionConfig, ModelSpec,
)
from averaging_ensembled_classifier.src.backends.cpu.discovery import (
    discover_hardware, detect_thread_count,
)
from averaging_ensembled_classifier.src.backends.cpu.renderer import CPUPlanRenderer


pytestmark = pytest.mark.skipif(
    not BACKEND_CPU, reason="CPU backend not available"
)


@pytest.fixture(scope="session")
def hardware_profile():
    """Session-scoped CPU hardware profile."""
    return discover_hardware()


@pytest.fixture(scope="session")
def thread_count():
    """Auto-detected thread count for the test host."""
    return detect_thread_count()


@pytest.fixture(scope="session")
def renderer_fp32():
    """Session-scoped CPUPlanRenderer for FP32 tests.

    Uses auto-detected thread count. Session scoping amortizes
    the library load and layout verification cost across all tests.
    """
    return CPUPlanRenderer()


@pytest.fixture
def renderer_single_thread():
    """Per-test CPUPlanRenderer with a single worker thread.

    Uses threads=1 to exercise the serial execution path.
    Useful for isolating race conditions vs. logic errors.
    """
    return CPUPlanRenderer(thread_count=1)


@pytest.fixture
def small_model_spec():
    """Iris-like model for unit tests: 4 inputs, 8 hidden, 3 classes, 2 modules."""
    return ModelSpec.float32(
        num_modules=2, num_classes=3,
        input_features=4, hidden_features=8,
    )


@pytest.fixture
def medium_model_spec():
    """Medium model for integration tests: 32 inputs, 128 hidden, 10 classes, 4 modules."""
    return ModelSpec.float32(
        num_modules=4, num_classes=10,
        input_features=32, hidden_features=128,
    )
```

**Session scoping rationale:** Unlike the OpenCL backend (which has expensive context creation and kernel compilation), the CPU backend's session scoping amortizes library loading, layout verification, and thread pool creation. While these operations are cheaper than OpenCL's, session scoping is still preferred to avoid per-test overhead from repeated `pool_create`/`pool_destroy` calls.

**Skip mechanism:** `pytest.mark.skipif(not BACKEND_CPU, ...)` causes all CPU Tier 2 tests to be skipped when the CPU backend is not built. This allows CI to run on platforms without a C compiler while still collecting the test count.

---

### Step 3C.2: Implement analytical reference fixtures (shared with OpenCL)

**Action:** Create or reuse `tests/tier2/fixtures/analytical.py` with closed-form reference functions for kernels whose outputs have simple mathematical definitions.

**Note on shared fixture ownership:** If Phase 2C has already implemented these fixtures, Phase 3C reuses them without modification. If Phase 3C executes before Phase 2C (which is valid — ADR-017 specifies Phase 2 and Phase 3 as independent branches), Phase 3C creates the fixtures, and Phase 2C will reuse them later.

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

### Step 3C.3: Implement numpy reference implementations (shared with OpenCL)

**Action:** Create or reuse `tests/tier2/fixtures/numpy_*.py` with reference implementations for complex kernels.

**Same shared-fixture ownership rule as Step 3C.2.** The fixtures are backend-neutral numpy implementations.

**Reference implementations:**

| Module | Kernels Covered | Strategy |
| :--- | :--- | :--- |
| `numpy_forward.py` | `forward_pass`, `render_logits_chunk` | Matmul + activation, logits computation |
| `numpy_gradients.py` | `compute_probs_loss_cce/bce`, `calculate_module_param_grads`, `backprop_error_to_hidden`, `calculate_chunk_temp_gradients` | Softmax, cross-entropy loss, gradient derivation |
| `numpy_reduction.py` | `gather_and_permute_grad_h`, `stabilize_reduce_grad_h`, reduction tree | Multi-stage reduction, stabilization transforms |
| `numpy_backprop.py` | `backprop_shared_weights`, `backprop_shared_biases` | Weight/bias gradient accumulation |
| `numpy_update.py` | `adam_update`, `clamp_temperatures` | Adam optimizer step, parameter clamping |
| `data_generators.py` | All | Deterministic input data generation with configurable dimensions |

**Data generator design:**

```python
import numpy as np


class ModelGeometry:
    """Describes a model configuration for test data generation."""

    def __init__(
        self, *,
        num_modules: int,
        num_classes: int,
        input_features: int,
        hidden_features: int,
        batch_size: int,
    ) -> None:
        self.num_modules = num_modules
        self.num_classes = num_classes
        self.input_features = input_features
        self.hidden_features = hidden_features
        self.batch_size = batch_size
        # Padded dimensions (SIMD-aligned)
        self.padded_input = _pad_to_simd(input_features)
        self.padded_hidden = _pad_to_simd(hidden_features)
        self.padded_classes = _pad_to_simd(num_classes)


def generate_forward_pass_inputs(
    geom: ModelGeometry, *, seed: int = 42,
) -> dict[str, np.ndarray]:
    """Generate deterministic inputs for forward_pass kernel testing."""
    rng = np.random.default_rng(seed)
    return {
        "input": rng.standard_normal(
            (geom.batch_size, geom.padded_input), dtype=np.float32
        ),
        "weights_shared_simd_major": rng.standard_normal(
            (geom.padded_input, geom.padded_hidden), dtype=np.float32
        ) * 0.01,
        "biases_shared": rng.standard_normal(
            (geom.padded_hidden,), dtype=np.float32
        ) * 0.01,
        "sample_mask": np.ones(geom.batch_size, dtype=np.float32),
    }
```

**Deterministic RNG:** All data generators use `np.random.default_rng(seed)` with explicit seeds. Tests are reproducible across platforms and runs. The seed is a test parameter, not a global state.

---

### Step 3C.4: Write Act-phase kernel tests

**Action:** Write per-kernel correctness tests for the Act phase (Nodes 4 and 5).

**test_cpu_act_forward_pass.py** — Tests for `forward_pass` (Node 4):

```python
import numpy as np
import pytest

from tests.tier2.fixtures.data_generators import ModelGeometry, generate_forward_pass_inputs
from tests.tier2.fixtures.numpy_forward import ref_forward_pass
from tests.tolerance_config import CPU_TOLERANCES


class TestCpuForwardPass:
    """Validate forward_pass kernel against numpy reference."""

    @pytest.mark.parametrize("threads", [1, None])
    def test_small_geometry(
        self, renderer_fp32, renderer_single_thread,
        small_model_spec, threads,
    ):
        renderer = renderer_single_thread if threads == 1 else renderer_fp32
        geom = ModelGeometry.from_model_spec(small_model_spec, batch_size=8)
        inputs = generate_forward_pass_inputs(geom)
        expected = ref_forward_pass(inputs, geom)

        plan = build_forward_pass_plan(inputs, geom)
        futures = renderer.render(plan)
        actual = futures["hidden_activations"].result()

        tol = CPU_TOLERANCES["forward_pass"]
        np.testing.assert_allclose(actual, expected, atol=tol.atol, rtol=tol.rtol)

    @pytest.mark.parametrize("batch_size", [1, 7, 32, 128])
    def test_batch_size_variations(self, renderer_fp32, small_model_spec, batch_size):
        """Test correctness across different batch sizes, including non-power-of-2."""
        geom = ModelGeometry.from_model_spec(small_model_spec, batch_size=batch_size)
        inputs = generate_forward_pass_inputs(geom)
        expected = ref_forward_pass(inputs, geom)

        plan = build_forward_pass_plan(inputs, geom)
        futures = renderer_fp32.render(plan)
        actual = futures["hidden_activations"].result()

        tol = CPU_TOLERANCES["forward_pass"]
        np.testing.assert_allclose(actual, expected, atol=tol.atol, rtol=tol.rtol)

    def test_zero_input(self, renderer_fp32, small_model_spec):
        """Forward pass with all-zero input should produce bias-only activations."""
        geom = ModelGeometry.from_model_spec(small_model_spec, batch_size=4)
        inputs = generate_forward_pass_inputs(geom, seed=0)
        inputs["input"][:] = 0.0

        expected = ref_forward_pass(inputs, geom)
        plan = build_forward_pass_plan(inputs, geom)
        futures = renderer_fp32.render(plan)
        actual = futures["hidden_activations"].result()

        tol = CPU_TOLERANCES["forward_pass"]
        np.testing.assert_allclose(actual, expected, atol=tol.atol, rtol=tol.rtol)
```

**test_cpu_act_render_logits.py** — Tests for `render_logits_chunk` (Node 5):

Same test pattern as `forward_pass`: construct inputs, compute numpy reference, build plan, render, compare. Additional test for streaming loop iteration: verify that multi-chunk logit computation produces the same result as single-chunk.

---

### Step 3C.5: Write Learn-phase gradient production tests

**Action:** Write per-kernel correctness tests for Learn Phase A → B (Nodes 6–11).

**Test modules:**

| Module | Kernel(s) | Strategy A/B Notes |
| :--- | :--- | :--- |
| `test_cpu_act_loss_cce.py` | `compute_probs_loss_cce_chunk` (Node 6) | Strategy B: dedicated CCE kernel |
| `test_cpu_act_loss_bce.py` | `compute_probs_loss_bce_chunk` (Node 7) | Strategy B: dedicated BCE kernel |
| `test_cpu_learn_module_grads.py` | `calculate_module_param_grads` (Node 8) | Strategy A: same kernel, CCE+BCE parameterized via flag |
| `test_cpu_learn_hidden_grads.py` | `backprop_error_to_hidden` (Node 9) | Strategy A: same kernel, CCE+BCE parameterized via flag |
| `test_cpu_learn_temp_grads.py` | `calculate_chunk_temp_gradients` (Node 10) | Strategy A: same kernel, CCE+BCE parameterized via flag |
| `test_cpu_learn_clip_partials.py` | `clip_partial_gradients` (Node 11) | Unified (no CCE/BCE distinction) |

**Strategy A parameterization:**

```python
@pytest.mark.parametrize("problem_type", ["cce", "bce"])
def test_module_param_grads(self, renderer_fp32, problem_type, small_model_spec):
    """Validate module param grads for both CCE and BCE problem types.

    Per ADR-011, calculate_module_param_grads uses a single kernel with
    a host-injected problem type flag. Both CCE and BCE variants must
    produce results matching their respective numpy references.
    """
    ...
```

**Strategy B parameterization:**

```python
class TestCpuLossCce:
    """Tests for compute_probs_loss_cce_chunk (Strategy B — dedicated kernel)."""

class TestCpuLossBce:
    """Tests for compute_probs_loss_bce_chunk (Strategy B — dedicated kernel)."""
```

**Pre-computed intermediate buffer pattern:** For `backprop_error_to_hidden` (Node 9), which depends on `forward_pass` (Node 4) output:

```python
def test_backprop_to_hidden_isolated(self, renderer_fp32, small_model_spec):
    """Test backprop_error_to_hidden with manually injected upstream outputs."""
    geom = ModelGeometry.from_model_spec(small_model_spec, batch_size=8)

    # Compute upstream outputs via numpy
    fwd_inputs = generate_forward_pass_inputs(geom)
    hidden_activations = ref_forward_pass(fwd_inputs, geom)

    # Compute gradient production via numpy
    grad_inputs = {
        "hidden_activations": hidden_activations,
        "error_deltas": generate_error_deltas(geom),
        "hidden_mask": ref_hidden_mask(hidden_activations),
    }
    expected = ref_backprop_to_hidden(grad_inputs, geom)

    # Build plan with ONLY the backprop node — inject upstream as input buffers
    plan = build_isolated_backprop_plan(grad_inputs, geom)
    futures = renderer_fp32.render(plan)
    actual = futures["grad_hidden"].result()

    tol = CPU_TOLERANCES["backprop_error_to_hidden"]
    np.testing.assert_allclose(actual, expected, atol=tol.atol, rtol=tol.rtol)
```

---

### Step 3C.6: Write Learn-phase reduction & aggregation tests

**Action:** Write per-kernel correctness tests for the reduction pipeline (Nodes 13–16, 20).

**Test modules:**

| Module | Kernel(s) | Key Complexity |
| :--- | :--- | :--- |
| `test_cpu_learn_gather_permute.py` | `gather_and_permute_grad_h` (Node 13) | Data layout transformation |
| `test_cpu_learn_reduction.py` | Reduction tree (Nodes 14/15/20) | Multi-stage `execute_reduction_tree` |
| `test_cpu_learn_stabilize_grad_h.py` | `stabilize_reduce_grad_h` (Node 16) | Numerical stabilization transform |

**Reduction tree testing strategy:**

The CPU reduction tree (`execute_reduction_tree`) implements multi-stage fan-in reduction using the thread pool. Testing requires validating:

1. **Single-stage reduction** — one level of fan-in, verifying that partial aggregation produces the correct sum/mean.
2. **Multi-stage reduction** — multiple levels, verifying that staging buffer ping-pong does not corrupt intermediate results.
3. **Non-uniform fan-in** — different fan-in values at different stages, verifying that the offset list indexing is correct.
4. **Edge case: single partial** — degenerate case where `num_partials = 1`, reduction should be a no-op (pass-through).

```python
class TestCpuReductionTree:
    """Validate execute_reduction_tree against numpy staged reduction."""

    @pytest.mark.parametrize("num_partials,fan_in", [
        (1, 1),      # Degenerate: pass-through
        (4, 4),      # Single-stage, all fit in one fan-in
        (8, 4),      # Two-stage reduction
        (16, 4),     # Three-stage reduction
        (100, 8),    # Multi-stage with non-power-of-2 count
    ])
    def test_reduction_correctness(
        self, renderer_fp32, num_partials, fan_in, small_model_spec,
    ):
        """Test reduction tree for various partial counts and fan-in values."""
        ...

    @pytest.mark.parametrize("threads", [1, None])
    def test_reduction_thread_safety(
        self, renderer_fp32, renderer_single_thread, threads,
    ):
        """Verify multi-threaded and single-threaded reduction produce identical results."""
        ...
```

---

### Step 3C.7: Write Learn-phase streaming backprop tests

**Action:** Write tests for the streaming loop body — `backprop_shared_weights` (Node 17), `backprop_shared_biases` (Node 18), `clip_shared_gradients` (Node 19).

**Test module:** `test_cpu_learn_streaming_backprop.py`

**Streaming loop testing strategy:**

The streaming loop iterates over module chunks, dispatching the backprop kernels per chunk with per-chunk parameter strides (e.g., `batch_chunk_offset` increments by `chunk_size`). Testing validates:

1. **Single-chunk execution** — model with `num_modules = 1`, streaming loop has one iteration.
2. **Multi-chunk execution** — model with `num_modules = 4`, verify that per-chunk strides correctly index into the gradient buffers.
3. **Gradient accumulation** — verify that backprop results from multiple chunks accumulate correctly into the shared weight/bias gradient buffers.

```python
class TestCpuStreamingBackprop:

    @pytest.mark.parametrize("num_modules", [1, 2, 4])
    def test_shared_weight_backprop(self, renderer_fp32, num_modules):
        """Validate backprop_shared_weights across different module counts."""
        geom = ModelGeometry(
            num_modules=num_modules, num_classes=3,
            input_features=4, hidden_features=8, batch_size=8,
        )
        # Compute full reference via numpy
        inputs = generate_backprop_inputs(geom)
        expected_grad_sw = ref_backprop_shared_weights(inputs, geom)

        # Build plan with streaming loop
        plan = build_streaming_backprop_plan(inputs, geom)
        futures = renderer_fp32.render(plan)
        actual_grad_sw = futures["grad_shared_weights"].result()

        tol = CPU_TOLERANCES["backprop_shared_weights"]
        np.testing.assert_allclose(
            actual_grad_sw, expected_grad_sw,
            atol=tol.atol, rtol=tol.rtol,
        )
```

---

### Step 3C.8: Write Learn-phase update kernel tests

**Action:** Write per-kernel correctness tests for update-phase kernels (Nodes 21, 24, 25).

**Test modules:**

| Module | Kernel | Reference |
| :--- | :--- | :--- |
| `test_cpu_learn_normalize.py` | `normalize_gradients` (Node 21) | Analytical: `grad / batch_size` |
| `test_cpu_learn_adam_update.py` | `adam_update` (Node 24) | Numpy Adam implementation |
| `test_cpu_learn_clamp_temps.py` | `clamp_temperatures` (Node 25) | Analytical: `clamp(t, T_min, T_max)` |

**Adam update testing:**

```python
class TestCpuAdamUpdate:

    def test_single_step(self, renderer_fp32, small_model_spec):
        """Verify one Adam step matches numpy reference."""
        inputs = generate_adam_inputs(
            num_params=64, learning_rate=0.001,
            beta1=0.9, beta2=0.999, epsilon=1e-8, t=1,
        )
        expected = ref_adam_update(inputs)

        plan = build_adam_update_plan(inputs)
        futures = renderer_fp32.render(plan)
        actual_params = futures["parameters"].result()

        tol = CPU_TOLERANCES["adam_update"]
        np.testing.assert_allclose(
            actual_params, expected["parameters"],
            atol=tol.atol, rtol=tol.rtol,
        )

    def test_multiple_steps_accumulation(self, renderer_fp32, small_model_spec):
        """Verify repeated Adam steps converge consistent with numpy."""
        ...

    def test_zero_gradient(self, renderer_fp32, small_model_spec):
        """Adam step with zero gradient should update only via momentum."""
        ...

    def test_beta_power_schedule(self, renderer_fp32, small_model_spec):
        """Verify beta1^t, beta2^t decay correctly over multiple steps."""
        ...
```

---

### Step 3C.9: Write end-to-end plan rendering correctness tests

**Action:** Write full Act and Learn end-to-end tests that exercise the complete plan DAG.

**Test module:** `test_cpu_plan_rendering_correctness.py`

**End-to-end strategy:**

1. **Act plan test:** Build an `ExecutionPlan` for a complete forward pass (Node 4 → Node 5 → loss computation). Render through `CPUPlanRenderer`. Compare all retrievable outputs against the numpy reference orchestration (which computes the same sequence in numpy).

2. **Learn plan test:** Build an `ExecutionPlan` for a complete learning step (gradient production → reduction → backprop → update). Render through `CPUPlanRenderer`. Compare final gradients and updated parameters against the numpy reference.

3. **Full Act+Learn cycle test:** Construct an Act plan, render it, feed the results into a Learn plan, render that, verify the final parameter state matches a complete numpy orchestration.

```python
class TestCpuPlanRenderingCorrectness:

    def test_act_plan_correctness(self, renderer_fp32, small_model_spec):
        """Full Act plan produces correct predictions."""
        data = generate_full_act_data(small_model_spec, batch_size=16)
        expected_predictions = ref_full_act(data, small_model_spec)

        plan = build_full_act_plan(data, small_model_spec)
        futures = renderer_fp32.render(plan)
        actual_predictions = futures["predictions"].result()

        tol = CPU_TOLERANCES["act_plan_e2e"]
        np.testing.assert_allclose(
            actual_predictions, expected_predictions,
            atol=tol.atol, rtol=tol.rtol,
        )

    def test_learn_plan_correctness(self, renderer_fp32, small_model_spec):
        """Full Learn plan produces correct parameter updates."""
        ...

    def test_act_learn_cycle(self, renderer_fp32, small_model_spec):
        """Complete Act → Learn cycle matches numpy orchestration."""
        ...

    @pytest.mark.parametrize("threads", [1, None])
    def test_determinism(self, renderer_fp32, renderer_single_thread, threads):
        """Verify that multi-threaded rendering produces deterministic results.

        Run the same plan twice and verify bit-exact output. CPU FP32
        with a fixed reduction order should produce deterministic results
        regardless of thread scheduling.
        """
        renderer = renderer_single_thread if threads == 1 else renderer_fp32
        plan = build_full_act_plan(...)
        futures_1 = renderer.render(plan)
        futures_2 = renderer.render(plan)
        np.testing.assert_array_equal(
            futures_1["predictions"].result(),
            futures_2["predictions"].result(),
        )
```

**Determinism note:** The CPU backend uses a fixed reduction tree plan (ADR-003), which produces a deterministic reduction order. Combined with IEEE 754 compliance of the C compiler's FP32 operations, the CPU backend should produce bit-exact results across runs. This is tested explicitly — determinism is a valuable property for debugging and is not guaranteed by the OpenCL backend.

---

### Step 3C.10: Configure CPU-specific tolerance tables

**Action:** Add CPU-specific entries to `tests/tolerance_config.py`.

```python
from dataclasses import dataclass


@dataclass(frozen=True)
class Tolerance:
    atol: float
    rtol: float


# CPU Tier 2 tolerances — FP32
# CPU operations are IEEE 754 compliant, so tolerances can be tighter
# than OpenCL (which may use -cl-fast-relaxed-math or vendor-specific
# precision). However, SIMD reduction ordering differences between
# ISA tiers may introduce small numerical differences.
CPU_TOLERANCES: dict[str, Tolerance] = {
    # Act phase
    "forward_pass": Tolerance(atol=1e-6, rtol=1e-5),
    "render_logits_chunk": Tolerance(atol=1e-6, rtol=1e-5),
    "compute_probs_loss_cce": Tolerance(atol=1e-5, rtol=1e-4),
    "compute_probs_loss_bce": Tolerance(atol=1e-5, rtol=1e-4),

    # Learn phase — gradient production
    "calculate_module_param_grads": Tolerance(atol=1e-5, rtol=1e-4),
    "backprop_error_to_hidden": Tolerance(atol=1e-5, rtol=1e-4),
    "calculate_chunk_temp_gradients": Tolerance(atol=1e-5, rtol=1e-4),
    "clip_partial_gradients": Tolerance(atol=1e-6, rtol=1e-5),

    # Learn phase — reduction
    "gather_and_permute_grad_h": Tolerance(atol=1e-6, rtol=1e-5),
    "stabilize_reduce_grad_h": Tolerance(atol=1e-5, rtol=1e-4),
    "reduction_tree": Tolerance(atol=1e-5, rtol=1e-4),

    # Learn phase — streaming backprop
    "backprop_shared_weights": Tolerance(atol=1e-5, rtol=1e-4),
    "backprop_shared_biases": Tolerance(atol=1e-5, rtol=1e-4),
    "clip_shared_gradients": Tolerance(atol=1e-6, rtol=1e-5),

    # Update phase
    "normalize_gradients": Tolerance(atol=1e-7, rtol=1e-6),
    "adam_update": Tolerance(atol=1e-5, rtol=1e-4),
    "clamp_temperatures": Tolerance(atol=0.0, rtol=0.0),  # Exact

    # End-to-end
    "act_plan_e2e": Tolerance(atol=1e-5, rtol=1e-4),
    "learn_plan_e2e": Tolerance(atol=1e-4, rtol=1e-3),
}
```

**Tolerance rationale:**

| Category | CPU Factor | Expected Tolerance |
| :--- | :--- | :--- |
| Simple element-wise | IEEE 754 exact | Tighter than OpenCL (atol=1e-7) |
| Matrix multiply (forward_pass) | Accumulation order differs from numpy | atol=1e-6, rtol=1e-5 |
| Softmax + log (loss) | exp/log approximation depends on compiler | atol=1e-5, rtol=1e-4 |
| Multi-stage reduction | Staging order fixed but differs from numpy | atol=1e-5, rtol=1e-4 |
| Adam update | Division by sqrt + epsilon | atol=1e-5, rtol=1e-4 |
| Clamp (exact) | No floating-point arithmetic | atol=0.0, rtol=0.0 |
| End-to-end | Accumulated across full pipeline | atol=1e-4, rtol=1e-3 |

**CPU vs. OpenCL tolerance comparison:** CPU tolerances are generally tighter than OpenCL tolerances because:
1. CPU FP32 operations are IEEE 754 compliant (no `-cl-fast-relaxed-math`).
2. Reduction order is fixed and deterministic (not hardware-scheduled).
3. No local memory → global memory precision loss from OpenCL implementations.

---

### Step 3C.11: FP16 parameterization (conditional)

**Action:** Document the FP16 parameterization path without implementing it.

The CPU backend initially supports FP32 only. FP16 support requires:
1. A C compiler with `_Float16` support (GCC 12+, Clang 15+).
2. CPU hardware with FP16 SIMD instructions (ARMv8.2 FP16, Intel AVX-512 FP16).
3. Build-time detection of FP16 capability in `meson.build`.
4. FP16 kernel variants in the C library.
5. FP16 ctypes struct variants in `_ffi_types.py`.

When FP16 is available, the test suite should be parameterized:

```python
@pytest.mark.parametrize("precision", [
    PrecisionConfig.float32(),
    pytest.param(
        PrecisionConfig.float16(),
        marks=pytest.mark.skipif(
            not CPU_FP16_AVAILABLE,
            reason="CPU FP16 not available"
        ),
    ),
])
def test_forward_pass(self, renderer_factory, precision, small_model_spec):
    renderer = renderer_factory(precision=precision)
    tol = CPU_TOLERANCES[("forward_pass", precision.name)]
    ...
```

**FP16 tolerance expectations:** When available, FP16 CPU tolerances should be:
- `atol=1e-2` to `1e-3` (depending on kernel complexity).
- Tighter than OpenCL FP16 (CPU has no vendor-specific precision relaxation).

**This step produces no code in Phase 3C.** It documents the extension point for future FP16 enablement.

---

### Step 3C.12: Validate Phase 3 rollback gate

**Action:** Verify the Phase 3 rollback gate (ADR-017).

The Phase 3 rollback gate requires:

1. **Tier 1 green** — all 332+ existing tests pass, no regressions.
2. **CPU Tier 2 green** — all per-kernel correctness tests pass for FP32.
3. **Infrastructure smoke tests green** — library loading, layout verification, plan traversal (from Phase 3B).

**Gate validation procedure:**

```bash
# 1. Run Tier 1 (shared-layer) tests
pytest tests/tier1/ -v --tb=short
# Expect: 332+ tests passed, 0 failed, 0 errors

# 2. Run CPU Tier 2 tests
pytest tests/tier2/cpu/ -v --tb=short
# Expect: all tests passed (count depends on parameterization)

# 3. Run infrastructure smoke tests
pytest tests/tier2/cpu/test_cpu_infrastructure.py -v --tb=short
# Expect: ~7 tests passed

# 4. Verify build configuration
python -c "from averaging_ensembled_classifier.src._build_config import BACKEND_CPU; assert BACKEND_CPU"
# Expect: no assertion error

# 5. Combined gate check
pytest tests/tier1/ tests/tier2/cpu/ -v --tb=short -x
# Expect: all green, exit code 0
```

**Gate failure policy:** If any Tier 1 or CPU Tier 2 test fails, the gate is not passed. The failure must be investigated and resolved before Phase 4 (cross-backend parity) can begin.

---

## 5. Fixture Architecture

The fixture architecture is designed for cross-backend reuse:

```
tests/tier2/fixtures/          (backend-neutral — shared)
├── analytical.py              ← closed-form definitions
├── numpy_forward.py           ← forward pass reference
├── numpy_gradients.py         ← gradient production reference
├── numpy_reduction.py         ← reduction pipeline reference
├── numpy_backprop.py          ← shared weight/bias backprop reference
├── numpy_update.py            ← Adam + clamp reference
└── data_generators.py         ← deterministic input generation

tests/tier2/cpu/conftest.py    (CPU-specific — infrastructure)
├── session-scoped CPUPlanRenderer
├── per-test single-thread renderer
└── model spec fixtures

tests/tier2/opencl/conftest.py (OpenCL-specific — infrastructure)
├── session-scoped cl.Context + cl.CommandQueue
├── compiled kernel program
└── OpenCLPlanRenderer
```

**Fixture ownership rules:**

1. **Analytical fixtures** (`analytical.py`) are the mathematical specification. They may not be modified to accommodate backend behavior — if a backend disagrees with the fixture, the backend is wrong.

2. **Numpy reference implementations** (`numpy_*.py`) use idiomatic numpy operations (broadcasting, matmul, etc.) without tiling, SIMD, or threading. They express the algorithm at the logical level, not the implementation level.

3. **Data generators** (`data_generators.py`) produce deterministic inputs parameterized by `ModelGeometry`. The same generator is used by both backend test suites.

4. **Backend infrastructure** (`conftest.py`) is backend-specific — it handles renderer construction, buffer allocation, and skip conditions.

---

## 6. Per-Kernel Test Matrix

The following matrix enumerates every kernel and its test dimensions:

| Kernel | Node | Test Module | CCE/BCE | Batch Sizes | Thread Counts | Edge Cases |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| `forward_pass` | 4 | `test_cpu_act_forward_pass.py` | N/A | 1, 7, 32, 128 | 1, auto | Zero input |
| `render_logits_chunk` | 5 | `test_cpu_act_render_logits.py` | N/A | 1, 8, 64 | 1, auto | Single class |
| `compute_probs_loss_cce` | 6 | `test_cpu_act_loss_cce.py` | CCE only | 1, 8, 32 | 1, auto | Single class, all-same target |
| `compute_probs_loss_bce` | 7 | `test_cpu_act_loss_bce.py` | BCE only | 1, 8, 32 | 1, auto | All-0 target, all-1 target |
| `calculate_module_param_grads` | 8 | `test_cpu_learn_module_grads.py` | Both | 1, 8, 32 | 1, auto | Single module |
| `backprop_error_to_hidden` | 9 | `test_cpu_learn_hidden_grads.py` | Both | 1, 8, 32 | 1, auto | Zero error deltas |
| `calculate_chunk_temp_gradients` | 10 | `test_cpu_learn_temp_grads.py` | Both | 1, 8, 32 | 1, auto | Zero gradient |
| `clip_partial_gradients` | 11 | `test_cpu_learn_clip_partials.py` | N/A | 1, 8, 32 | 1, auto | Below threshold, at threshold, above threshold |
| `gather_and_permute_grad_h` | 13 | `test_cpu_learn_gather_permute.py` | N/A | 8, 32 | 1, auto | Identity permutation |
| Reduction tree | 14/15/20 | `test_cpu_learn_reduction.py` | N/A | N/A | 1, auto | 1 partial, non-power-of-2, large fan-in |
| `stabilize_reduce_grad_h` | 16 | `test_cpu_learn_stabilize_grad_h.py` | N/A | 8, 32 | 1, auto | Near-zero gradient |
| `backprop_shared_weights` | 17 | `test_cpu_learn_streaming_backprop.py` | N/A | 8, 32 | 1, auto | Single chunk |
| `backprop_shared_biases` | 18 | `test_cpu_learn_streaming_backprop.py` | N/A | 8, 32 | 1, auto | Single chunk |
| `clip_shared_gradients` | 19 | `test_cpu_learn_streaming_backprop.py` | N/A | 8, 32 | 1, auto | All below threshold |
| `normalize_gradients` | 21 | `test_cpu_learn_normalize.py` | N/A | 1, 8, 128 | 1, auto | batch_size=1 |
| `adam_update` | 24 | `test_cpu_learn_adam_update.py` | N/A | N/A | 1, auto | Zero gradient, t=1, t=1000 |
| `clamp_temperatures` | 25 | `test_cpu_learn_clamp_temps.py` | N/A | N/A | 1, auto | Within bounds, below min, above max |

**Estimated test count:**

- 18 kernel groups × ~4 test methods per kernel × ~3 parameterization values ≈ **~216 test cases** (FP32 only).
- Plus ~10 end-to-end test cases.
- Plus ~7 infrastructure smoke tests (from Phase 3B).
- **Total Phase 3 test target: ~233 tests.**

---

## 7. Tolerance Configuration

### CPU tolerance properties

The CPU backend has several properties that affect tolerance selection differently from OpenCL:

1. **IEEE 754 compliance.** C compilers targeting x86-64 with `-fno-fast-math` (the default) produce IEEE 754 compliant FP32 operations. No relaxed-precision shortcuts.

2. **Deterministic reduction order.** The reduction tree plan specifies a fixed evaluation order. Unlike OpenCL, where work-group-level reductions may execute in hardware-dependent order, the CPU uses a Python-orchestrated staging sequence. Same inputs → same outputs.

3. **SIMD lane processing order.** SIMD operations process lanes in a fixed order (lane 0, 1, ..., N-1). The tiling scheme maps tiles to SIMD lanes deterministically. No out-of-order lane execution.

4. **FMA availability.** If the CPU supports FMA (Fused Multiply-Add), the compiler may use it for operations like `a * b + c`. FMA produces a more accurate result than separate multiply + add (single rounding instead of double rounding). This means FMA-enabled builds may be one ULP more accurate than non-FMA builds — and both may differ from numpy (which uses whatever the platform provides). Tolerances must accommodate this difference.

5. **Compiler optimization level.** At `-O2` or `-O3`, the compiler may reorder associative floating-point operations for ILP. With strict IEEE compliance (`-fno-fast-math`), this reordering is prohibited for FP operations. The Meson build should not enable `-ffast-math`.

### Tolerance escalation strategy

If a test fails at the configured tolerance:

1. **Investigate the failure.** Is the error systematic (wrong algorithm) or numerical (precision difference)?
2. **Check FMA influence.** If the difference is exactly 1 ULP on FMA-capable hardware, the tolerance is correct — widen by 1 ULP.
3. **Check reduction order.** If the numpy reference uses a different summation order, the difference may be inherent. Increase `atol` by the expected summation error.
4. **Never silence a large discrepancy.** If `atol > 1e-3` for an FP32 kernel, the kernel implementation likely has a bug — investigate rather than widening tolerance.

---

## 8. CPU-Specific Test Considerations

### Thread safety validation

The CPU backend's multi-threaded dispatch introduces race condition risks that do not exist in the OpenCL backend (where the runtime handles thread safety). Key thread safety tests:

1. **Concurrent write detection.** Kernels that write to shared buffers (e.g., gradient accumulation) must use atomic operations or partitioned writes. Test by running with `threads=1` and `threads=auto` and verifying identical results.

2. **Buffer aliasing.** Ensure no two concurrent tasks write to overlapping buffer regions. The plan model guarantees non-overlapping tile assignments, but the test validates this property.

3. **Thread pool reuse.** The session-scoped renderer reuses the same thread pool across all tests. Verify that no state leaks between dispatches — each `render()` call starts with fresh buffer contents.

### SIMD alignment validation

The CPU backend requires SIMD-aligned buffer data pointers. Tests should verify:

1. **Alignment of allocated buffers.** Every buffer from `CPUBufferAllocator` has a data pointer aligned to `SIMD_ALIGNMENT`.
2. **Alignment after padding.** Padded dimensions are multiples of `SIMD_WIDTH`, ensuring that SIMD loads/stores at tile boundaries do not cross alignment boundaries.

These are tested in the infrastructure smoke tests (Phase 3B Step 3B.12) but can be spot-checked in Tier 2 tests via assertions:

```python
def assert_aligned(arr: np.ndarray, alignment: int) -> None:
    assert arr.ctypes.data % alignment == 0, (
        f"Array at {arr.ctypes.data:#x} is not {alignment}-byte aligned"
    )
```

### Scalar fallback regression test

To validate that the scalar fallback path (`SIMD_WIDTH=1`) produces correct results, at least one test configuration should exercise it. This can be achieved by:

1. Building the library with `SIMD_WIDTH=1` (via `aec_cpu_isa_flags=''` — no ISA flags, falling back to scalar).
2. Running the full CPU Tier 2 suite against the scalar-fallback build.

This is a CI-level configuration test, not a per-test-method parameterization. It validates that the SIMD abstraction layer (`cpu_simd.h`) correctly handles the scalar case.

### Memory leak detection (optional)

For debug builds, consider running CPU Tier 2 tests under a memory sanitizer (ASan/MSan via Meson's `b_sanitize=address,undefined`). This catches:
- Buffer overflows in C kernels.
- Use-after-free from incorrect buffer lifecycle management.
- Uninitialized memory reads.

This is a CI configuration concern, not a test code concern. The tests themselves do not need modification to support sanitizer builds.

---

## 9. Risk Register

| Risk | Likelihood | Impact | Mitigation |
| :--- | :--- | :--- | :--- |
| Fixture and kernel disagree on edge case behavior (e.g., `exp(-inf)`, `log(0)`) | Medium | Medium — test failure, not a correctness bug per se | Explicitly test edge cases; document expected behavior for boundary values in fixture comments |
| FMA presence causes systematic 1 ULP difference on some platforms | Medium | Low — tolerance widening needed | FMA-aware tolerance adjustment; detect FMA at build time and select tolerance table variant |
| Thread race condition produces non-deterministic test failures | Medium | High — flaky CI, hard to diagnose | Always test with `threads=1` in addition to `threads=auto`; run with TSan (`b_sanitize=thread`) in CI |
| Fixture creation order conflict (Phase 3C runs before Phase 2C) | Low | Low — duplicated fixture creation effort | Fixtures are in `tests/tier2/fixtures/` (shared). Whichever phase runs first creates them; the other reuses. |
| Reduction tree correctness depends on RNG seed for tile ordering | Low | Medium — test appears to pass with one seed and fail with another | Test with multiple seeds; verify determinism property separately |
| numpy reference uses different algorithm than C kernel (e.g., different softmax stabilization) | Medium | High — systematic tolerance violations | Carefully align numpy reference with the C kernel's documented algorithm; review CONTRACT.md and kernel header comments |
| Test suite too slow due to thread pool create/destroy overhead | Low | Low — longer CI times | Session-scoped renderer amortizes pool lifecycle; per-test renderer only for thread count parameterization |
| Tolerance table needs per-ISA variants (AVX-512 vs. SSE2 produce different results) | Low | Medium — false failures on some platforms | Initially use the loosest reasonable tolerance; tighten per-ISA if CI data supports it |
| Infrastructure smoke tests pass but Tier 2 tests fail (library loads but produces wrong results) | Medium | Low — expected; that's why Tier 2 exists | Infrastructure tests are a subset; Tier 2 is the authoritative correctness gate |
| Padding stripping in CPURetrievalFuture introduces off-by-one in logical shape | Low | High — tests pass but user-visible outputs are mis-shaped | Explicit shape assertion in every test: `assert actual.shape == expected.shape` before `assert_allclose` |
