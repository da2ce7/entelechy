# Phase 4: Test Harness — Detailed Plan

**Status:** Not started  
**Phase:** 4 of 6  
**Objective:** Bring the full Tier 1/2/3 test framework to operational status, providing validation infrastructure that serves all backend development phases. The test harness grows its coverage incrementally as backends become available — its gate is satisfied when all enabled tiers are green.  
**Governing ADRs:** ADR-016 (test strategy — layered pytest framework, CPU oracle, analytical + numpy fixtures), ADR-014 (build system — `_build_config.py` backend flags), ADR-008 (precision configuration — per-kernel tolerances), ADR-013 (kernel source strategy — kernel inventory), ADR-015 (CPU FFI — layout verification as pre-gate; Tier 2 as FFI behavioral verification), ADR-011 (CCE/BCE strategy delegation — Strategy A/B test matrix), ADR-005 (Node 16 opacity — fixture complexity)  
**Rollback gate:** Full Tier 1/2/3 framework operational; all enabled tiers green. This gate is satisfied incrementally — the framework reports correct results for whatever backends exist at the time.  
**Dependencies:** ADR-016 (decided). Phase 4 can begin in parallel with all other phases. It consumes infrastructure from Phases 1–3 and 5 as that infrastructure becomes available.

### Relationship to Other Phases

Phase 4 is unique in the migration: it has no hard dependency on any single phase and proceeds concurrently with all workstreams. Its role is to provide validation infrastructure *during* development, not *after*:

| Phase | Phase 4 interaction |
| :--- | :--- |
| Phase 0 (Foundation) | Phase 4 consumes the directory skeleton and `_build_config.py.in` template (already complete). |
| Phase 1 (Plan Model) | Phase 4 consumes the plan model types for Tier 1 testing. Tier 1 tests were a Phase 1 deliverable but the test infrastructure (fixtures, conftest, tolerance config) is Phase 4 scope. |
| Phase 2 (OpenCL Adapter) | Phase 4 provides OpenCL Tier 2 test files. OpenCL Tier 2 tests validate the renderer against fixtures as it develops. |
| Phase 3 (CPU Backend) | Phase 4 provides CPU Tier 2 test files. CPU Tier 2 tests validate FFI correctness (ADR-015) and establish the oracle's trustworthiness for Tier 3. |
| Phase 5 (Vulkan Backend) | Phase 4 provides Vulkan Tier 2 test files and Tier 3 parity tests. |
| User-Facing API | Phase 4 cross-pollinates with the ticket API workstream — ticket lifecycle tests exercise plan builder usage patterns. |

### Current State Assessment

Several Phase 4 deliverables already exist due to organic development during Phases 1–3:

| Deliverable | Status | Location |
| :--- | :--- | :--- |
| Tier 1 test directory and files | ✅ Complete (Phase 1) | `tests/tier1/` — 11 test files, conftest |
| Tier 2 CPU test files | ✅ Complete (Phase 3C) | `tests/tier2/cpu/` — 16 test files, conftest |
| Tier 2 OpenCL test files | ✅ Complete (Phase 2C) | `tests/tier2/opencl/` — 17 test files, conftest |
| Analytical fixtures | ✅ Complete | `tests/tier2/fixtures/analytical.py` |
| Numpy reference: forward | ✅ Complete | `tests/tier2/fixtures/numpy_forward.py` |
| Numpy reference: backprop | ✅ Complete | `tests/tier2/fixtures/numpy_backprop.py` |
| Numpy reference: gradients | ✅ Complete | `tests/tier2/fixtures/numpy_gradients.py` |
| Numpy reference: reduction | ✅ Complete | `tests/tier2/fixtures/numpy_reduction.py` |
| Numpy reference: update | ✅ Complete | `tests/tier2/fixtures/numpy_update.py` |
| Data generators | ✅ Complete | `tests/tier2/fixtures/data_generators.py` |
| Tolerance configuration | ✅ Complete | `tests/tolerance_config.py` |
| Top-level conftest (legacy) | ⚠️ Partial | `tests/conftest.py` — legacy probe-based skip logic; not yet `_build_config`-driven collection modifier |
| Tier 3 directory and files | ❌ Not started | `tests/tier3/` does not exist |
| Tier 3 oracle selection | ❌ Not started | — |
| `--all-pairs` mode | ❌ Not started | — |
| Tier 2 Vulkan test files | ❌ Not started (Phase 5 dependency) | — |
| pytest marker registration | ⚠️ Partial | `tier1`, `tier2` markers used implicitly; `tier3`, `cpu`, `opencl`, `vulkan` not formally registered |

---

## Table of Contents

1. [Scope & Constraints](#1-scope--constraints)
2. [Current State Inventory](#2-current-state-inventory)
3. [Target Deliverables](#3-target-deliverables)
4. [Task Breakdown](#4-task-breakdown)
   - [Step 4.1: Register pytest markers formally](#step-41-register-pytest-markers-formally)
   - [Step 4.2: Modernize top-level `conftest.py` with `_build_config`-driven skip logic](#step-42-modernize-top-level-conftestpy-with-_build_config-driven-skip-logic)
   - [Step 4.3: Create `tests/tier3/` directory skeleton](#step-43-create-teststier3-directory-skeleton)
   - [Step 4.4: Implement Tier 3 conftest — oracle selection and multi-backend fixtures](#step-44-implement-tier-3-conftest--oracle-selection-and-multi-backend-fixtures)
   - [Step 4.5: Implement per-kernel parity tests](#step-45-implement-per-kernel-parity-tests)
   - [Step 4.6: Implement end-to-end parity tests](#step-46-implement-end-to-end-parity-tests)
   - [Step 4.7: Implement `--all-pairs` optional mode](#step-47-implement---all-pairs-optional-mode)
   - [Step 4.8: Add Tier 3 tolerance configuration](#step-48-add-tier-3-tolerance-configuration)
   - [Step 4.9: Create Vulkan Tier 2 test stubs](#step-49-create-vulkan-tier-2-test-stubs)
   - [Step 4.10: Validate framework against available backends](#step-410-validate-framework-against-available-backends)
   - [Step 4.11: Validate rollback gate](#step-411-validate-rollback-gate)
5. [Tier 3 Test Architecture](#5-tier-3-test-architecture)
6. [Oracle Selection Logic](#6-oracle-selection-logic)
7. [Multi-Backend Fixture Lifecycle](#7-multi-backend-fixture-lifecycle)
8. [Canonical Test Configurations](#8-canonical-test-configurations)
9. [CLI Usage Patterns](#9-cli-usage-patterns)
10. [Test Target Inventory](#10-test-target-inventory)
11. [Risk Register](#11-risk-register)

---

## 1. Scope & Constraints

### In scope

- Modernizing `tests/conftest.py` to use `_build_config.py`-driven `pytest_collection_modifyitems` skip logic (ADR-016), replacing the legacy OpenCL-only probe.
- Formally registering all pytest markers (`tier1`, `tier2`, `tier3`, `opencl`, `vulkan`, `cpu`) in `pyproject.toml` or `conftest.py` to eliminate marker warnings.
- Implementing the complete `tests/tier3/` directory structure with per-kernel parity tests and end-to-end parity tests (ADR-016).
- Implementing oracle selection logic: CPU as the reference oracle when available; fallback to direct GPU-vs-GPU comparison when CPU is absent (ADR-016 Option C).
- Implementing the `--all-pairs` optional mode for supplementary GPU-vs-GPU comparison (ADR-016).
- Adding Tier 3-specific tolerance configuration to `tests/tolerance_config.py` — cross-backend tolerances may differ from per-backend Tier 2 tolerances due to implementation-specific accumulation order differences.
- Creating Vulkan Tier 2 test stubs in `tests/tier2/vulkan/` — structural placeholders that become functional when Phase 5 delivers the Vulkan renderer.
- Validating the framework end-to-end against whatever backends are currently available.

### Out of scope

- Implementing backend renderers (`PlanRenderer` implementations) — Phases 2, 3, 5.
- Implementing kernel sources — Phases 2, 3, 5.
- Writing new Tier 1 tests — Tier 1 is a Phase 1 deliverable (already complete).
- Writing new Tier 2 CPU or OpenCL kernel test bodies — these are Phase 3C and Phase 2C deliverables respectively (already complete).
- Modifying shared-layer code (plan model, plan builder, contracts).
- Performance benchmarking — ADR-017 chose tier-based gates (Option D) with no performance thresholds.
- FP16 test infrastructure — deferred until a backend supports FP16. The tolerance table and fixture factories accept precision parameters and will work for FP16 when a backend enables it.
- The `WorkTicket` / `LearnHandle` / `Engine` user-facing API and its associated integration tests (Phase 4-adjacent workstream per ADR-018).

### Key constraint: framework-only, not test-body

Phase 4 delivers the test *framework* — infrastructure, fixtures, skip logic, oracle selection, directory structure. The test *bodies* (per-kernel correctness assertions, fixture wiring for specific kernels) are delivered by each backend's respective phase. Phase 4 provides the structural scaffolding that those test bodies plug into.

The exception is Tier 3: both the framework *and* the test bodies are Phase 4 deliverables, because Tier 3 tests are cross-backend by nature and belong to no single backend phase.

### Key constraint: incremental satisfaction

Phase 4's rollback gate is "all enabled tiers green." This is inherently incremental:
- With only the CPU backend available: Tier 1 green + CPU Tier 2 green.
- With CPU + OpenCL: Tier 1 green + CPU Tier 2 green + OpenCL Tier 2 green + Tier 3 (CPU-vs-OpenCL) green.
- With all three backends: full matrix.

The framework must report correct pass/skip/fail status at every intermediate state. A tier that cannot run because its backend is absent is *skipped*, not *failed* (ADR-016, ADR-017 §"Distinguishing skipped tier from failed tier").

### Key constraint: backward compatibility with legacy tests

The existing `tests/test_integration_*.py` files (9 files) and `tests/bench_*.py` files (3 files) must continue to run. The modernized `conftest.py` must not break the legacy import machinery or fixture chain. Legacy tests coexist with the tier framework until Phase 6 retires them.

---

## 2. Current State Inventory

### Test directory structure (post Phases 1–3)

```
tests/
├── __init__.py
├── conftest.py                             # Legacy: probe-based OpenCL skip; shared fixtures
├── tolerance_config.py                     # ✅ Per-kernel + CPU-specific tolerance tables
├── bench_host_planning.py                  # Legacy benchmark
├── bench_opencl_backend.py                 # Legacy benchmark (requires OpenCL)
├── bench_opencl_backend_learn.py           # Legacy benchmark (requires OpenCL)
├── test_integration_buffer_lifecycle.py    # Legacy integration test
├── test_integration_dag_orchestration.py   # Legacy integration test
├── test_integration_e2e_iris.py            # Legacy integration test
├── test_integration_execution_plan.py      # Legacy integration test
├── test_integration_model_memory.py        # Legacy integration test
├── test_integration_precision_chain.py     # Legacy integration test
├── test_integration_scenario_validation.py # Legacy integration test
├── test_integration_stabilization.py       # Legacy integration test
├── test_integration_workload_tiling.py     # Legacy integration test
├── tier1/                                  # ✅ Complete (Phase 1)
│   ├── __init__.py
│   ├── conftest.py
│   ├── test_buffer_lifecycle.py
│   ├── test_hardware_profile.py
│   ├── test_kernel_contracts.py
│   ├── test_memory_layout.py
│   ├── test_plan_builder.py
│   ├── test_plan_types.py
│   ├── test_precision_config.py
│   ├── test_problem_type_strategy.py
│   ├── test_reduction_tree_plan.py
│   └── test_streaming_loop_plan.py
├── tier2/                                  # ⚠️ Partial — CPU and OpenCL complete; Vulkan absent
│   ├── __init__.py
│   ├── fixtures/                           # ✅ Complete
│   │   ├── __init__.py
│   │   ├── analytical.py
│   │   ├── data_generators.py
│   │   ├── numpy_backprop.py
│   │   ├── numpy_forward.py
│   │   ├── numpy_gradients.py
│   │   ├── numpy_reduction.py
│   │   └── numpy_update.py
│   ├── cpu/                                # ✅ Complete (Phase 3C)
│   │   ├── __init__.py
│   │   ├── conftest.py
│   │   └── test_cpu_*.py (16 files)
│   └── opencl/                             # ✅ Complete (Phase 2C)
│       ├── __init__.py
│       ├── conftest.py
│       └── test_*.py (17 files)
└── tier3/                                  # ❌ Not started
```

### `tests/conftest.py` — current state

The top-level conftest uses a runtime probe for OpenCL availability (`pyopencl` import + `cl.create_some_context(interactive=False)`). It does not read `_build_config.py`. It registers a `requires_opencl` marker but does not register `tier1`, `tier2`, `tier3`, `cpu`, `vulkan`, or `opencl` markers. It does not implement `pytest_collection_modifyitems`.

The conftest provides:
- `fp32_iris_spec` / `fp16_iris_spec` — canonical small model fixtures using legacy `Float32ModelSpec`/`Float16ModelSpec` factories.
- `fp32_hydra_spec` / `fp32_lexicon_spec` — stress-test configurations.
- `iris_param_space` / `iris_tiling` / `iris_stabilization` — derived fixtures.
- `sys.path` manipulation to import `src.*` modules without `pyopencl` side effects.

### `tests/tolerance_config.py` — current state

Complete. Provides:
- `TolerancePair` frozen dataclass with `atol` and `rtol`.
- `FP32_DEFAULT` / `FP16_DEFAULT` baseline tolerances.
- `KERNEL_TOLERANCES` — shared per-kernel overrides keyed by `(kernel_name, precision_label)`.
- `CPU_KERNEL_TOLERANCES` — CPU-specific tighter tolerances due to IEEE 754 compliance.
- `get_tolerance()` / `get_cpu_tolerance()` — lookup functions with fallback to defaults.

No Tier 3-specific tolerances exist yet.

### Backend-specific Tier 2 conftest modules

**`tests/tier2/cpu/conftest.py`:**
- Module-level skip via `_build_config.BACKEND_CPU`.
- Session-scoped fixtures: `hardware_profile`, `thread_count`, `precision_fp32`, `renderer_fp32`, `renderer_single_thread`.

**`tests/tier2/opencl/conftest.py`:**
- Module-level skip via `pyopencl` import + device probe.
- Session-scoped fixtures: `cl_context`, `cl_device`, `cl_queue`, `hardware_profile`, `precision_fp32`, `compiled_program_fp32`, OpenCL renderer.

Both conftest modules are self-contained and do not rely on the top-level conftest's skip logic for backend gating.

---

## 3. Target Deliverables

After Phase 4 completes, the test directory gains:

```
tests/
├── conftest.py                             # MODIFIED: _build_config-driven skip logic,
│                                           #   pytest_collection_modifyitems, marker registration
├── tolerance_config.py                     # MODIFIED: Tier 3 cross-backend tolerances added
├── tier2/
│   └── vulkan/                             # NEW: Vulkan Tier 2 stubs
│       ├── __init__.py
│       ├── conftest.py                     # Vulkan session fixtures (stub)
│       └── test_vulkan_kernels.py          # Parameterized kernel tests (stub)
└── tier3/                                  # NEW: Cross-backend parity
    ├── __init__.py
    ├── conftest.py                         # Oracle selection, multi-backend fixtures
    ├── test_parity_per_kernel.py           # Per-kernel cross-backend comparison
    └── test_parity_e2e.py                  # End-to-end Act+Learn parity
```

Additionally:
- `pyproject.toml` (or `conftest.py`) gains formal marker registration for `tier1`, `tier2`, `tier3`, `cpu`, `opencl`, `vulkan`.
- The `--all-pairs` CLI option is registered via a `conftest.py` plugin hook.

---

## 4. Task Breakdown

### Step 4.1: Register pytest markers formally

**Action:** Add marker definitions so that `pytest --strict-markers` does not produce warnings. Markers are registered in `pyproject.toml` under `[tool.pytest.ini_options]`.

**Markers to register:**

| Marker | Description |
| :--- | :--- |
| `tier1` | Host-side plan correctness — always runs, no backend required |
| `tier2` | Per-backend kernel correctness — runs per enabled backend |
| `tier3` | Cross-backend parity — runs when ≥ 2 backends available |
| `cpu` | Requires CPU backend (`_build_config.BACKEND_CPU`) |
| `opencl` | Requires OpenCL backend (device probe) |
| `vulkan` | Requires Vulkan backend (`_build_config.BACKEND_VULKAN`) |
| `slow` | Long-running test (stress configurations) |

**Location:** `architectures/averaging_ensembled_classifier/pyproject.toml` under `[tool.pytest.ini_options]`:

```toml
[tool.pytest.ini_options]
markers = [
    "tier1: Host-side plan correctness (no backend required)",
    "tier2: Per-backend kernel correctness (requires enabled backend)",
    "tier3: Cross-backend parity (requires >= 2 backends)",
    "cpu: Requires CPU backend",
    "opencl: Requires OpenCL backend",
    "vulkan: Requires Vulkan backend",
    "slow: Long-running test",
]
```

**Validation:** `pytest --strict-markers --collect-only` reports zero marker warnings.

---

### Step 4.2: Modernize top-level `conftest.py` with `_build_config`-driven skip logic

**Action:** Extend `tests/conftest.py` with a `pytest_collection_modifyitems` hook that reads `_build_config.py` and applies skip markers to Tier 2/3 tests whose backend requirements are not met.

**Design:**

The existing probe-based OpenCL detection and legacy fixtures are preserved for backward compatibility with `test_integration_*.py` files. The new logic is *additive*:

```python
def _load_build_config() -> dict[str, bool]:
    """Load the build manifest; return a dict of backend availability."""
    try:
        from src._build_config import BACKEND_OPENCL, BACKEND_VULKAN, BACKEND_CPU
        return {
            "opencl": BACKEND_OPENCL,
            "vulkan": BACKEND_VULKAN,
            "cpu": BACKEND_CPU,
        }
    except ImportError:
        # Pre-migration fallback: only OpenCL via runtime probe.
        return {"opencl": _has_opencl_device, "vulkan": False, "cpu": False}

BUILD_CONFIG = _load_build_config()

def pytest_collection_modifyitems(config, items):
    """Skip tests whose backend requirements are not met (ADR-016)."""
    for item in items:
        # Tier 2: skip per-backend tests if backend unavailable
        for backend in ("opencl", "vulkan", "cpu"):
            if backend in item.keywords and not BUILD_CONFIG[backend]:
                item.add_marker(pytest.mark.skip(
                    reason=f"Backend '{backend}' not available (_build_config)"
                ))

        # Tier 3: skip parity tests if < 2 backends available
        if "tier3" in item.keywords:
            available = sum(BUILD_CONFIG.values())
            if available < 2:
                item.add_marker(pytest.mark.skip(
                    reason=f"Tier 3 requires >= 2 backends ({available} available)"
                ))
```

**Key decisions:**

1. **`_build_config` import path.** The CPU conftest already imports from `src._build_config`. The collection modifier uses the same path for consistency.
2. **Fallback to probe.** If `_build_config` is not yet generated (e.g., running tests from source without a Meson build), the existing `_has_opencl_device` probe is used as fallback. This preserves pre-Phase 0 behavior.
3. **Legacy `requires_opencl` marker.** Retained for existing `test_integration_*.py` files that use it. Not replaced — coexists with the new `opencl` marker.

**Backward compatibility check:** Run all existing tests after modification; verify identical pass/skip/fail outcomes.

---

### Step 4.3: Create `tests/tier3/` directory skeleton

**Action:** Create the Tier 3 test directory with `__init__.py`, `conftest.py`, and test file placeholders.

**Files to create:**

```
tests/tier3/
├── __init__.py               # Empty package marker
├── conftest.py               # Oracle selection + multi-backend fixture lifecycle
├── test_parity_per_kernel.py # Per-kernel cross-backend parity
└── test_parity_e2e.py        # End-to-end Act+Learn parity
```

---

### Step 4.4: Implement Tier 3 conftest — oracle selection and multi-backend fixtures

**Action:** Implement `tests/tier3/conftest.py` with the oracle selection logic and session-scoped multi-backend fixtures.

**Oracle selection logic (ADR-016 Option C):**

1. If `BUILD_CONFIG["cpu"]` is `True`, the CPU backend is the oracle.
2. If the CPU backend is unavailable but ≥ 2 other backends are available, fall back to direct comparison (no privileged oracle — all pairs compared).
3. If < 2 backends are available, Tier 3 is skipped entirely (handled by `pytest_collection_modifyitems` in Step 4.2).

```python
def _select_oracle() -> str | None:
    """Select the Tier 3 oracle backend.

    Returns the oracle backend name, or None if no oracle is available
    (triggering all-pairs fallback).
    """
    if BUILD_CONFIG.get("cpu", False):
        return "cpu"
    return None  # No oracle — all-pairs fallback

def _get_comparison_backends(oracle: str | None) -> list[str]:
    """Return backends to compare against the oracle (or all pairs if no oracle)."""
    available = [b for b in ("opencl", "vulkan", "cpu") if BUILD_CONFIG.get(b, False)]
    if oracle is not None:
        return [b for b in available if b != oracle]
    return available
```

**Multi-backend fixtures:**

Session-scoped fixtures must create renderer instances for **all available backends simultaneously**. This is necessary because Tier 3 compares outputs from different backends for the same input:

```python
@pytest.fixture(scope="session")
def oracle_renderer():
    """Session-scoped renderer for the oracle backend."""
    oracle = _select_oracle()
    if oracle == "cpu":
        from src.backends.cpu.renderer import CPUPlanRenderer
        return CPUPlanRenderer()
    return None  # No oracle — test functions handle this

@pytest.fixture(scope="session")
def renderer_factory():
    """Factory that creates a renderer for a named backend."""
    def _create(backend_name: str):
        if backend_name == "cpu":
            from src.backends.cpu.renderer import CPUPlanRenderer
            return CPUPlanRenderer()
        elif backend_name == "opencl":
            from src.backends.opencl.renderer import OpenCLPlanRenderer
            return OpenCLPlanRenderer()
        elif backend_name == "vulkan":
            from src.backends.vulkan.renderer import VulkanPlanRenderer
            return VulkanPlanRenderer()
        raise ValueError(f"Unknown backend: {backend_name}")
    return _create
```

**Fixture lifecycle:**
- Session-scoped renderer creation at the start of Tier 3 collection.
- Each renderer manages its own device context (OpenCL queue, Vulkan device, CPU thread pool).
- Cleanup via finalizers registered on the session-scoped fixtures.
- Shared `ModelSpec` and `PrecisionConfig` fixtures from Tier 1's conftest are reused.

**Plan construction fixtures:**
- `iris_act_plan` — a pre-built Act-phase `ExecutionPlan` for the Iris-scale FP32 model.
- `iris_learn_plan` — a pre-built Learn-phase `ExecutionPlan` for the Iris-scale FP32 model.
- `iris_full_plan` — combined Act+Learn plan.
- `single_kernel_plan_factory` — factory fixture that builds a minimal `ExecutionPlan` containing a single `KernelDispatchNode` for isolation testing.

---

### Step 4.5: Implement per-kernel parity tests

**Action:** Implement `tests/tier3/test_parity_per_kernel.py` — parameterized tests that dispatch individual kernels on both the oracle and comparison backends, comparing outputs within tolerance.

**Test design (ADR-016):**

```python
@pytest.mark.tier3
@pytest.mark.parametrize("kernel_name", KERNEL_INVENTORY)
@pytest.mark.parametrize("comparison_backend", _get_comparison_backends(_select_oracle()))
def test_parity_per_kernel(
    oracle_renderer,
    renderer_factory,
    comparison_backend,
    kernel_name,
    single_kernel_plan_factory,
    tolerance_config,
):
    """Verify per-kernel numerical parity between oracle and comparison backend."""
    comparison_renderer = renderer_factory(comparison_backend)
    plan = single_kernel_plan_factory(kernel_name)

    oracle_output = _execute_and_extract(oracle_renderer, plan)
    comparison_output = _execute_and_extract(comparison_renderer, plan)

    tol = tolerance_config.get_tier3_tolerance(kernel_name)
    np.testing.assert_allclose(
        comparison_output, oracle_output,
        atol=tol.atol, rtol=tol.rtol,
        err_msg=f"Parity failure: {comparison_backend} vs oracle for {kernel_name}",
    )
```

**Kernel inventory (from ADR-013, ADR-016):**

The test is parameterized over the full kernel inventory from the phase files:

| Phase | Kernels | Strategy variants |
| :--- | :--- | :--- |
| `phase_1_act` | `forward_pass`, `render_logits_chunk`, `compute_probs_loss_cce_chunk`, `compute_probs_loss_bce_chunk` | B (problem type) |
| `phase_2_learn_A_production` | `calculate_module_param_grads`, `backprop_error_to_hidden`, `calculate_temp_gradients` | A (FLAG) |
| `phase_2_learn_B_processing` | `clip_partial_gradients`, `gather_and_permute_grad_h` | — |
| `phase_2_learn_C_reduction` | `aggregate_register_reduce`, `aggregate_local_reduce`, `clip_intermediate_grad` | — |
| `phase_2_learn_D_backprop` | `stabilize_reduce_grad_h`, `backprop_shared_weights`, `backprop_shared_biases`, `clip_shared_gradients` | — |
| `phase_3_update` | `normalize_gradients`, `adam_update`, `clamp_temperatures` | — |

Strategy A kernels receive additional parameterization over `problem_type ∈ {"CCE", "BCE"}`. Strategy B kernels (`compute_probs_loss_cce_chunk`, `compute_probs_loss_bce_chunk`) are already distinct kernel names.

**Total per-kernel parity test cases:** ~22 kernels × (backends − 1) comparisons. With Strategy A expansion: ~25 test cases per comparison pair.

---

### Step 4.6: Implement end-to-end parity tests

**Action:** Implement `tests/tier3/test_parity_e2e.py` — tests that execute a complete `ExecutionPlan` (Act + Learn phases) on both the oracle and comparison backends, comparing plan-level outputs.

**Test design (ADR-016):**

```python
@pytest.mark.tier3
@pytest.mark.parametrize("comparison_backend", _get_comparison_backends(_select_oracle()))
@pytest.mark.parametrize("problem_type", ["CCE", "BCE"])
def test_parity_e2e_act_learn(
    oracle_renderer,
    renderer_factory,
    comparison_backend,
    problem_type,
    iris_full_plan_factory,
):
    """End-to-end parity: complete Act+Learn cycle."""
    plan = iris_full_plan_factory(problem_type=problem_type)
    comparison_renderer = renderer_factory(comparison_backend)

    oracle_futures = oracle_renderer.render(plan)
    comparison_futures = comparison_renderer.render(plan)

    # Compare Act-phase outputs (Final Probs)
    oracle_probs = oracle_futures["inference_event"].result()
    comp_probs = comparison_futures["inference_event"].result()
    np.testing.assert_allclose(
        comp_probs, oracle_probs, atol=1e-4, rtol=1e-4,
        err_msg=f"Act-phase parity failure: {comparison_backend} vs oracle ({problem_type})",
    )

    # Compare Learn-phase outputs (updated parameters)
    oracle_params = oracle_futures["final_batch_event"].result()
    comp_params = comparison_futures["final_batch_event"].result()
    for name in oracle_params:
        np.testing.assert_allclose(
            comp_params[name], oracle_params[name], atol=1e-4, rtol=1e-4,
            err_msg=f"Learn-phase parity failure: {comparison_backend} vs oracle, param '{name}' ({problem_type})",
        )
```

**End-to-end configurations:**

| Configuration | Purpose | Dimensions |
| :--- | :--- | :--- |
| `iris_fp32` | Primary correctness | `input=4, hidden=32, classes=3, modules=8, batch=150` |
| `stress_fp32` | Scale sensitivity | `input=128, hidden=512, classes=100, modules=32, batch=1024` |

The `stress_fp32` configuration is marked `@pytest.mark.slow` and excluded from default runs. It exercises:
- Deeper reduction trees (more modules → more reduction stages).
- Larger streaming loop chunk counts.
- Buffer lifecycle at scale (more buffers, larger allocations).
- Numerical accumulation across more operations (error propagation stress).

---

### Step 4.7: Implement `--all-pairs` optional mode

**Action:** Register a `--all-pairs` CLI option that enables exhaustive GPU-vs-GPU comparison in addition to oracle-based comparison. This is a supplementary check (ADR-016) — not the default mode.

**Implementation:**

In `tests/conftest.py`:

```python
def pytest_addoption(parser):
    parser.addoption(
        "--all-pairs",
        action="store_true",
        default=False,
        help="Run Tier 3 parity tests with all-pairs comparison (GPU-vs-GPU in addition to oracle)",
    )
```

In `tests/tier3/conftest.py`:

```python
def _get_tier3_pairs(config) -> list[tuple[str, str]]:
    """Return (oracle_or_left, comparison) pairs for Tier 3 tests.

    Default: CPU-oracle vs each GPU backend.
    --all-pairs: adds GPU-vs-GPU pairs.
    """
    available = [b for b in ("cpu", "opencl", "vulkan") if BUILD_CONFIG.get(b, False)]
    oracle = _select_oracle()

    pairs = []
    if oracle is not None:
        # Default: oracle-based comparison
        for b in available:
            if b != oracle:
                pairs.append((oracle, b))

    if config.getoption("--all-pairs") or oracle is None:
        # All-pairs: every combination
        from itertools import combinations
        for left, right in combinations(available, 2):
            if (left, right) not in pairs and (right, left) not in pairs:
                pairs.append((left, right))

    return pairs
```

When `--all-pairs` is active, the per-kernel and end-to-end parity tests are parameterized over all backend pairs, not just oracle-vs-comparison. This provides the Option F-equivalent check as a supplementary diagnostic without requiring it for the gate.

**Validation:** `pytest -m tier3 --all-pairs --collect-only` shows the expanded parameter matrix.

---

### Step 4.8: Add Tier 3 tolerance configuration

**Action:** Add cross-backend tolerance entries to `tests/tolerance_config.py`.

Tier 3 tolerances may be wider than Tier 2 tolerances because they account for implementation-specific differences across backends (accumulation order, fused multiply-add availability, denormal handling). The oracle-based comparison aggregates two sources of divergence: oracle-vs-reference *and* comparison-vs-reference.

**Additions to `tolerance_config.py`:**

```python
# Cross-backend (Tier 3) tolerance overrides.
# Wider than Tier 2 because both backends independently diverge
# from the mathematical reference in different directions.
TIER3_KERNEL_TOLERANCES: dict[str, dict[str, TolerancePair]] = {
    "compute_probs_loss_cce_chunk": {
        "fp32": TolerancePair(atol=5e-4, rtol=5e-4),
    },
    "compute_probs_loss_bce_chunk": {
        "fp32": TolerancePair(atol=5e-4, rtol=5e-4),
    },
    "stabilize_reduce_grad_h": {
        "fp32": TolerancePair(atol=5e-4, rtol=5e-4),
    },
    "adam_update": {
        "fp32": TolerancePair(atol=5e-4, rtol=5e-4),
    },
    "aggregate_local_reduce": {
        "fp32": TolerancePair(atol=5e-4, rtol=5e-4),
    },
    "aggregate_register_reduce": {
        "fp32": TolerancePair(atol=5e-4, rtol=5e-4),
    },
}

TIER3_FP32_DEFAULT = TolerancePair(atol=1e-4, rtol=1e-4)
TIER3_FP16_DEFAULT = TolerancePair(atol=5e-2, rtol=5e-2)

def get_tier3_tolerance(kernel_name: str, precision_label: str = "fp32") -> TolerancePair:
    """Look up Tier 3 cross-backend tolerance, falling back to Tier 3 defaults."""
    if kernel_name in TIER3_KERNEL_TOLERANCES:
        overrides = TIER3_KERNEL_TOLERANCES[kernel_name]
        if precision_label in overrides:
            return overrides[precision_label]
    return TIER3_FP32_DEFAULT if precision_label == "fp32" else TIER3_FP16_DEFAULT
```

**Rationale for wider tolerances:**

| Kernel | Tier 2 atol | Tier 3 atol | Reason |
| :--- | :--- | :--- | :--- |
| `compute_probs_loss_cce_chunk` | `1e-4` | `5e-4` | `exp()` / `log()` implementation differences between CPU (libm) and GPU (transcendental approximation) |
| `stabilize_reduce_grad_h` | `1e-4` | `5e-4` | Multi-stage reduction: CPU sequential vs GPU parallel reduction order |
| `adam_update` | `1e-4` | `5e-4` | Division by `sqrt(v_hat + eps)` amplifies small differences; FMA availability varies |
| `aggregate_local_reduce` | `1e-4` | `5e-4` | Summation order varies (CPU linear, GPU tree) |

Kernels with exact mathematical operations (`clamp_temperatures`, `compute_hidden_mask`, `normalize_gradients`) use the default Tier 3 tolerance, which is already generous enough to absorb cross-backend differences.

---

### Step 4.9: Create Vulkan Tier 2 test stubs

**Action:** Create `tests/tier2/vulkan/` with structural placeholders that become functional when Phase 5 delivers the Vulkan renderer.

**Files:**

```
tests/tier2/vulkan/
├── __init__.py
├── conftest.py               # Module-level skip when Vulkan unavailable; session fixtures
└── test_vulkan_kernels.py    # Parameterized kernel test stubs
```

**`conftest.py` design:**

```python
import pytest

try:
    from src._build_config import BACKEND_VULKAN
except ImportError:
    BACKEND_VULKAN = False

if not BACKEND_VULKAN:
    pytest.skip("Vulkan backend not available", allow_module_level=True)

# Session-scoped Vulkan fixtures will be populated in Phase 5.
# The renderer import is deferred until the module is not skipped.
```

**`test_vulkan_kernels.py` design:**

A parameterized test function mirroring the structure of `test_cpu_*.py` and `test_*.py` (OpenCL), but with Vulkan-specific renderer fixtures. The test body follows the same pattern as OpenCL and CPU Tier 2 tests — dispatch a single kernel, compare against the fixture, assert within tolerance.

The test file imports from `tests/tier2/fixtures/` for reference implementations and from `tests/tolerance_config.py` for tolerance lookup — the same infrastructure used by CPU and OpenCL Tier 2 tests.

**Validation:** `pytest --collect-only tests/tier2/vulkan/` shows the test stubs are collected and skipped (because `BACKEND_VULKAN` is `False` until Phase 5).

---

### Step 4.10: Validate framework against available backends

**Action:** Run the full test suite with the new framework and verify:

1. **Tier 1:** All 140+ Tier 1 tests pass (no regressions from Phase 1).
2. **Tier 2 CPU:** All CPU Tier 2 tests pass (no regressions from Phase 3C).
3. **Tier 2 OpenCL:** All OpenCL Tier 2 tests pass where an OpenCL device is available; cleanly skipped otherwise.
4. **Tier 2 Vulkan:** Cleanly skipped (backend not yet available).
5. **Tier 3:** Runs if ≥ 2 backends are available. If CPU + OpenCL are both available, Tier 3 per-kernel and end-to-end parity tests execute against the CPU oracle. If only CPU is available, Tier 3 is cleanly skipped.
6. **Legacy tests:** All `test_integration_*.py` and `bench_*.py` files pass or skip as before.
7. **No marker warnings:** `pytest --strict-markers` produces zero warnings.

**CLI validation commands:**

```bash
# Verify marker registration
pytest --strict-markers --collect-only 2>&1 | grep -c "PytestUnknownMarkWarning"

# Run Tier 1 only
pytest -m tier1 -v

# Run Tier 2 for available backends
pytest -m tier2 -v

# Run Tier 3 (if applicable)
pytest -m tier3 -v

# Run all tiers
pytest -v

# Verify skip logic for absent backends
pytest -m vulkan --collect-only  # Should show all skipped

# Full suite with legacy tests
pytest -v
```

---

### Step 4.11: Validate rollback gate

**Action:** Confirm the Phase 4 rollback gate: "Full Tier 1/2/3 framework operational; all enabled tiers green."

**Gate criteria:**

| Condition | Verification |
| :--- | :--- |
| Tier 1 green | `pytest -m tier1` — 0 failures, 0 errors |
| Tier 2 green (per enabled backend) | `pytest -m tier2` — 0 failures for enabled backends; absent backends cleanly skipped |
| Tier 3 green (if ≥ 2 backends) | `pytest -m tier3` — 0 failures for available pairs; skipped if < 2 backends |
| No regressions | Total test count ≥ previous total; no previously-passing test now fails |
| Skip/fail distinction correct | Absent backends produce `SKIPPED`, not `FAILED` |
| Marker registration clean | `pytest --strict-markers` — zero warnings |
| Legacy tests unaffected | `test_integration_*.py` pass/skip outcomes unchanged |

---

## 5. Tier 3 Test Architecture

### Comparison model

Tier 3 operates at two granularities (ADR-016):

1. **Per-kernel parity.** Isolates a single `KernelDispatchNode` in a minimal `ExecutionPlan`. Both the oracle and comparison backend execute the same plan and produce outputs that are compared element-wise within tolerance. This catches:
   - Algorithmic divergence in individual kernel implementations.
   - Buffer layout misinterpretation (e.g., treating row-major as column-major).
   - Precision handling differences (e.g., denormal flushing vs. propagation).

2. **End-to-end parity.** Executes a complete Act+Learn `ExecutionPlan` on both backends and compares the plan-level observable outputs: `Final Probs` (post-Act), updated parameter tensors (post-Learn). This catches:
   - Error accumulation across the full DAG.
   - Buffer lifecycle errors (stale data, double-free, use-after-release).
   - Synchronization point mishandling (barrier ordering, retrieval timing).
   - Interaction effects between kernels that per-kernel tests miss.

### Data flow

```
                   ┌──────────────────────┐
                   │  Plan Builder        │
                   │  (shared layer)      │
                   └──────┬───────────────┘
                          │ ExecutionPlan
                          ▼
              ┌───────────┴───────────┐
              │                       │
    ┌─────────▼─────────┐  ┌─────────▼─────────┐
    │  Oracle Renderer   │  │  Comparison        │
    │  (CPU by default)  │  │  Renderer          │
    └─────────┬─────────┘  └─────────┬──────────┘
              │ RetrievalFutures     │ RetrievalFutures
              ▼                      ▼
    ┌─────────────────┐  ┌──────────────────────┐
    │  oracle_output   │  │  comparison_output    │
    └────────┬────────┘  └──────────┬────────────┘
             │                      │
             └──────────┬───────────┘
                        ▼
              ┌─────────────────┐
              │  assert_allclose │
              │  (tolerance)     │
              └─────────────────┘
```

Both renderers receive the *same* immutable `ExecutionPlan` and *identical* initial buffer contents (seeded from the same deterministic RNG). The comparison is between the *outputs* — the plan's observable results as extracted through `RetrievalFuture.result()`.

### Input reproducibility

Deterministic input generation is critical for Tier 3 — both backends must operate on identical data. The `data_generators.py` module (already implemented) provides:

- `make_rng(seed=42)` — deterministic `numpy.random.Generator` via `PCG64`.
- `make_input_data()`, `make_weights()`, `make_biases()`, `make_sample_mask()` — all accept an `rng` parameter.

Each Tier 3 test creates a fresh `rng` with a fixed seed, generates inputs, and feeds the *same* numpy arrays to both renderers. Buffer allocation and device transfer are handled by each renderer's own `BufferAllocator` — the renderer receives logical numpy arrays and produces padded, SIMD-aligned device buffers internally.

---

## 6. Oracle Selection Logic

### Decision tree

```
Is BACKEND_CPU available?
├── Yes → Oracle = CPU
│         Comparisons: CPU vs OpenCL, CPU vs Vulkan (if available)
└── No → Is --all-pairs set OR is this the only option?
          ├── Yes → All-pairs: compare every available pair
          └── No → All-pairs: compare every available pair (no oracle = all-pairs fallback)
```

### Justification for CPU as oracle (ADR-016 Option C)

1. **Deterministic execution.** CPU backend uses single-threaded-per-task execution (ADR-015) — no wavefront scheduling variance, no warp divergence. The same input always produces the same output.
2. **Independently verified.** CPU Tier 2 tests use analytical/numpy fixtures that are generated without executing any backend. CPU correctness is established before the oracle role is exercised.
3. **Maximally inspectable.** ADR-015's fine-grained `pool_dispatch_and_wait` dispatch means every kernel invocation can be individually logged and debugged.
4. **ADR-013 alignment.** ADR-013 explicitly identifies the CPU backend as the natural oracle candidate.

### Oracle unavailability

If `_build_config.BACKEND_CPU` is `False`, the oracle is `None` and all-pairs mode is implicitly activated. This is expected in GPU-only CI environments. Diagnostic quality decreases — a two-way disagreement provides no tie-breaker — but coverage is maintained.

---

## 7. Multi-Backend Fixture Lifecycle

### Session-scoped renderer management

Each backend's renderer is created once per session and reused across all Tier 3 tests. This amortizes expensive initialization costs:

| Backend | Initialization cost |
| :--- | :--- |
| CPU | Library load (`ctypes.CDLL`), layout verification (`_verify_layouts()`), thread pool creation |
| OpenCL | Context creation, kernel compilation, command queue setup |
| Vulkan | Device enumeration, pipeline creation, SPIR-V loading, descriptor pool setup |

Session scoping is safe because `PlanRenderer.render()` is stateless with respect to the renderer — it receives an immutable plan, allocates transient buffers, dispatches, and returns futures. No renderer-level mutable state carries between test functions.

### Fixture cleanup

Renderers that manage device resources (OpenCL context, Vulkan device, CPU thread pool) must release them at session end. This is implemented via pytest finalizers:

```python
@pytest.fixture(scope="session")
def cpu_renderer(request):
    renderer = CPUPlanRenderer()
    request.addfinalizer(renderer.close)
    return renderer
```

Each backend's renderer implements a `close()` method (or equivalent) that releases device resources. The `PlanRenderer` Protocol does not mandate `close()` — it is a per-backend implementation detail — but Tier 3 fixtures call it if present.

---

## 8. Canonical Test Configurations

### Model configurations

| Configuration | Name | Parameters | Purpose |
| :--- | :--- | :--- | :--- |
| Iris FP32 | `iris_fp32` | `input=4, hidden=32, classes=3, modules=8, batch=150` | Primary correctness — small enough for fast iteration |
| Iris FP16 | `iris_fp16` | Same geometry, FP16 | Precision-sensitivity (when backends support FP16) |
| Stress FP32 | `stress_fp32` | `input=128, hidden=512, classes=100, modules=32, batch=1024` | Scale-sensitivity — deep reduction trees, large streaming loops |

The `iris_fp32` configuration is used for *all* Tier 3 tests by default. The `stress_fp32` configuration is an additional test case marked `@pytest.mark.slow`, excluded from default runs but included in full regression CI.

### Hardware profile for Tier 3

Tier 3 tests use the *actual* hardware profile from each backend's device discovery, not a synthetic fixture. This is necessary because the plan builder uses `HardwareProfile` fields (e.g., `max_reduce_fan_in`, `simd_width`) to determine plan structure — and different backends may report different hardware capabilities for the same physical machine.

This means two backends may receive slightly different `ExecutionPlan` instances if their hardware profiles differ (e.g., different `simd_width`). Tier 3 handles this by comparing *observable outputs*, not plan structures — the plan may have different internal structure as long as the mathematical result is equivalent within tolerance.

**Exception:** Per-kernel parity tests use a *shared* hardware profile (the CPU backend's profile, or a synthetic profile if CPU is unavailable) to ensure both backends receive the same plan. This isolates kernel-level divergence from plan-structure divergence.

---

## 9. CLI Usage Patterns

### Basic tier execution

```bash
# Tier 1 only (always works, no device needed)
pytest -m tier1

# Tier 1 + Tier 2 for all available backends
pytest -m "tier1 or tier2"

# Tier 2 for CPU only
pytest -m "tier2 and cpu"

# Tier 2 for OpenCL only
pytest -m "tier2 and opencl"

# Tier 3 only (requires >= 2 backends)
pytest -m tier3

# All tiers
pytest
```

### Advanced options

```bash
# Tier 3 with all-pairs GPU-vs-GPU comparison
pytest -m tier3 --all-pairs

# Full regression with stress tests
pytest -m "not slow"   # Default: exclude slow tests
pytest                  # Include slow tests

# Strict marker mode (CI)
pytest --strict-markers

# Collect-only mode for debugging skip logic
pytest --collect-only -q
```

### Tier-specific filtering

```bash
# Per-kernel parity for a specific kernel
pytest tests/tier3/test_parity_per_kernel.py -k "forward_pass"

# End-to-end parity for CCE only
pytest tests/tier3/test_parity_e2e.py -k "CCE"

# Vulkan Tier 2 tests (when available)
pytest -m "tier2 and vulkan"
```

---

## 10. Test Target Inventory

### Tier 3 per-kernel parity test matrix

| Phase | Kernel | Strategy | Per-pair tests |
| :--- | :--- | :--- | :--- |
| Act | `forward_pass` | — | 1 |
| Act | `render_logits_chunk` | — | 1 |
| Act | `compute_probs_loss_cce_chunk` | B | 1 |
| Act | `compute_probs_loss_bce_chunk` | B | 1 |
| Learn A | `calculate_module_param_grads` | A (CCE, BCE) | 2 |
| Learn A | `backprop_error_to_hidden` | A (CCE, BCE) | 2 |
| Learn A | `calculate_temp_gradients` | A (CCE, BCE) | 2 |
| Learn B | `clip_partial_gradients` | — | 1 |
| Learn B | `gather_and_permute_grad_h` | — | 1 |
| Learn C | `aggregate_register_reduce` | — | 1 |
| Learn C | `aggregate_local_reduce` | — | 1 |
| Learn C | `clip_intermediate_grad` | — | 1 |
| Learn D | `stabilize_reduce_grad_h` | — | 1 |
| Learn D | `backprop_shared_weights` | — | 1 |
| Learn D | `backprop_shared_biases` | — | 1 |
| Learn D | `clip_shared_gradients` | — | 1 |
| Update | `normalize_gradients` | — | 1 |
| Update | `adam_update` | — | 1 |
| Update | `clamp_temperatures` | — | 1 |

**Total per-kernel tests per comparison pair:** 22

### Tier 3 end-to-end parity test matrix

| Configuration | Problem types | Per-pair tests |
| :--- | :--- | :--- |
| `iris_fp32` | CCE, BCE | 2 |
| `stress_fp32` (slow) | CCE, BCE | 2 |

**Total end-to-end tests per comparison pair:** 4 (2 default + 2 slow)

### Comparison pair counts

| Available backends | Oracle | Default pairs | `--all-pairs` adds |
| :--- | :--- | :--- | :--- |
| CPU + OpenCL | CPU | 1 (CPU↔OpenCL) | 0 |
| CPU + Vulkan | CPU | 1 (CPU↔Vulkan) | 0 |
| CPU + OpenCL + Vulkan | CPU | 2 (CPU↔OpenCL, CPU↔Vulkan) | 1 (OpenCL↔Vulkan) |
| OpenCL + Vulkan (no CPU) | None | 1 (OpenCL↔Vulkan) | 0 |

### Total Tier 3 test count (CPU + OpenCL + Vulkan, all-pairs)

- Per-kernel: 22 × 3 pairs = 66 tests (default: 22 × 2 = 44)
- End-to-end: 4 × 3 pairs = 12 tests (default: 4 × 2 = 8)
- **Total: 78** (default: **52**)

---

## 11. Risk Register

| Risk | Likelihood | Impact | Mitigation |
| :--- | :--- | :--- | :--- |
| **Tier 3 tolerance tuning.** Cross-backend tolerances may be too tight (false failures) or too loose (missed bugs) until empirical data is available. | High | Medium | Start with conservative (wide) tolerances from §Step 4.8. Tighten iteratively as backends mature and empirical divergence data accumulates. Document the rationale for each override. |
| **Plan-structure divergence.** Different backends report different `HardwareProfile` values, producing different `ExecutionPlan` structures. Per-kernel parity tests assume identical plans. | Medium | Medium | Per-kernel tests use a shared hardware profile (§8). End-to-end tests accept structural divergence and compare only observable outputs. |
| **Session-scoped renderer contention.** Two backends sharing the same GPU (e.g., OpenCL and Vulkan on the same device) may contend for device resources, causing non-deterministic timing failures. | Medium | Low | Renderers are session-scoped and tests execute sequentially within `pytest`. No parallel dispatch across backends within a single test. If contention is observed, add a mutex fixture or separate device allocation. |
| **CPU oracle unavailability in GPU-only CI.** If the CI environment lacks a C compiler, `BACKEND_CPU` is `False` and Tier 3 has no privileged oracle. | Medium | Medium | All-pairs fallback activates automatically. Diagnostic quality decreases but coverage is maintained. Recommend CI environments include a C compiler (ADR-014's `auto` default enables CPU when toolchain is present). |
| **Vulkan Tier 2 stubs become stale.** Stub test files may not match the actual Vulkan renderer API when Phase 5 delivers. | Medium | Low | Stubs are minimal — they contain only the skeleton structure and import pattern. Phase 5 populates the test bodies. Stubs are explicitly marked as placeholders in comments. |
| **Legacy conftest interaction.** Modernizing `tests/conftest.py` may break the `sys.path` manipulation or fixture chain used by `test_integration_*.py` files. | Medium | Medium | The modernization is additive — new functions are added, existing functions are not removed. Run full legacy test suite after each conftest change. The `_load_build_config()` fallback preserves pre-migration behavior when `_build_config` is absent. |
| **Node 16 Tier 3 divergence.** Node 16 (`stabilize_reduce_grad_h`) uses fundamentally different reduction strategies across backends (CPU: sequential SIMD, OpenCL: local memory + barriers, Vulkan: subgroup operations). Cross-backend divergence may exceed tolerance. | Medium | Medium | Node 16 has an explicit Tier 3 tolerance override in §Step 4.8. The tolerance is set wider than other kernels. If empirical divergence exceeds even the wider tolerance, investigate whether the mathematical equivalence class is correctly defined — per CONCEPT.md §1, this may signal incomplete architectural modeling of the reduction operation's numerical contract. |
| **`--all-pairs` mode combinatorial explosion.** With 3 backends and `--all-pairs`, the test count increases by 50% (from 2 pairs to 3). This is manageable, but future backend additions (e.g., Metal, CUDA) would grow quadratically. | Low | Low | The current 3-backend scenario produces only 3 pairs — manageable. If the backend count grows, consider promoting `--all-pairs` to a CI-only configuration rather than a developer-facing option. |

---

## References

- [ADR-001: Backend Abstraction Boundary](../adr/ADR-001-backend-abstraction-boundary.md) — plan-as-data-structure; `PlanRenderer` Protocol; three-tier jurisdictional model
- [ADR-005: Node 16 Opacity in the Plan](../adr/ADR-005-node-16-opacity-in-the-plan.md) — Node 16's internal reduction is opaque to the plan; affects fixture complexity and Tier 3 tolerance
- [ADR-008: Precision Configuration](../adr/ADR-008-precision-configuration.md) — `PrecisionConfig`; precision-derived tolerances for numerical comparison
- [ADR-011: CCE/BCE Strategy Delegation](../adr/ADR-011-cce-bce-strategy-delegation.md) — Strategy A/B variants; test matrix expansion
- [ADR-013: Kernel Source Strategy](../adr/ADR-013-kernel-source-strategy.md) — kernel inventory; three-tier specification hierarchy; CPU as natural oracle
- [ADR-014: Build System Integration](../adr/ADR-014-build-system-integration.md) — `_build_config.py` manifest; `auto`/`enabled`/`disabled` feature options; CI self-configuration
- [ADR-015: Python ↔ Native Backend Interop](../adr/ADR-015-python-native-backend-interop.md) — `_verify_layouts()` pre-gate; Tier 2 as behavioral FFI verification; CPU dispatch determinism
- [ADR-016: Test Strategy](../adr/ADR-016-test-strategy.md) — layered pytest framework; Tier 1/2/3 definitions; CPU oracle (Option C); analytical + numpy fixtures (Option E); per-kernel tolerance tables; `pytest_collection_modifyitems` skip logic; `--all-pairs` mode
- [ADR-017: Incremental Migration Path](../adr/ADR-017-incremental-migration-path.md) — Phase 4 definition; rollback gate; dependency graph; feature-flag lifecycle
- [CONCEPT.md](../CONCEPT.md) — §1 Architectural Elegance Feedback; §5 Unified Execution Model
