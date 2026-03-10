# ADR-016: Test Strategy

**Status:** ACCEPTED  
**Date:** 2026-03-10  
**Deciders:** —  
**Supersedes:** —  
**Blocked by:** ADR-013, ADR-014  
**Blocks:** ADR-017

---

## Context

The multi-backend architecture (OpenCL, Vulkan, CPU) requires a layered test strategy that addresses:

1. **Host-side plan construction.** Verify that `ExecutionPlan` construction, `KernelContract` validation, `BufferContract` lifecycle, and `MemoryLayout` placement are correct independently of any backend.
2. **Per-backend kernel correctness.** Verify that each backend's compiled kernels produce numerically correct results for a known input/output pair.
3. **Cross-backend parity.** Verify that all backends produce equivalent outputs (within ADR-008 precision tolerances) for the same `ExecutionPlan`.

ADR-013 (ACCEPTED) resolves kernel source locations and establishes a **three-tier specification hierarchy** that directly structures the parity assurance model:

| Tier | Artifact | Verification mechanism |
| :--- | :--- | :--- |
| **Interface** | `KernelContract` (ADR-007) | Plan-construction-time validation — type/shape/padding/placement checks |
| **Algorithm** | `kernels.cl.h` in `kernels/` | Code review against algorithmic reference during development |
| **Implementation** | Per-backend sources in `src/backends/<name>/kernel_sources/` | Cross-backend oracle tests (this ADR, Tier 3) |

ADR-014 (ACCEPTED) resolves conditional backend enablement and provides a generated `_build_config.py` module that declares `BACKEND_OPENCL`, `BACKEND_VULKAN`, and `BACKEND_CPU` as booleans. This module is the authoritative source for determining which backends are available at runtime and, consequently, which test tiers can execute in a given environment:

- **Tier 1** (host-side plan correctness) — always runs; no backend required.
- **Tier 2** (per-backend kernel correctness) — runs for each backend where `_build_config.BACKEND_<NAME>` is `True`.
- **Tier 3** (cross-backend parity) — runs when two or more backends have `True` values. ADR-014's `auto` default for `backend_vulkan` and `backend_cpu` means CI environments get the tiers their hardware supports with no manual configuration.

ADR-015 (ACCEPTED) resolves the CPU backend's FFI mechanism as ctypes with struct layout verification. This has direct implications for the test strategy:

- **Layout verification as a pre-test gate.** ADR-015's `_verify_layouts()` function asserts that Python-side ctypes struct definitions match the C-side struct sizes at library load time. If layout drift exists, the CPU backend fails to initialize — the test suite never reaches Tier 2 with a corrupted FFI layer. This converts the most dangerous FFI failure mode (silent struct corruption) into a deterministic startup error that precedes test execution.
- **Tier 2 as implicit FFI validation.** ADR-015 notes that struct field reordering within the same type roster (e.g., swapping two `uint` fields) is not caught by size-based layout verification. Tier 2 per-kernel correctness tests are the behavioral safety net: a reordered struct produces numerically wrong results, which Tier 2 detects. This makes Tier 2 tests load-bearing for FFI correctness — not just algorithmic correctness.
- **CPU as oracle viability.** ADR-015's fine-grained dispatch pattern (Python walks the plan DAG, calls `pool_dispatch_and_wait` per node) means CPU execution is maximally inspectable. Combined with the deterministic, single-threaded-per-task execution model (CPU_BACKEND.md), the CPU backend remains the natural Tier 3 oracle candidate identified by ADR-013.

The kernel inventory is known (ADR-013 §Decision, file table): ~20 kernels across 6 phase files, with Strategy A/B variants (ADR-011) expanding the effective test matrix. Each kernel has a `KernelContract` (ADR-007) that declares its interface and a reference algorithm in `kernels.cl.h`.

### Existing test infrastructure

The current test suite in `tests/` uses pytest and provides:

- Host-side integration tests (`test_integration_execution_plan.py`, `test_integration_buffer_lifecycle.py`, `test_integration_dag_orchestration.py`, etc.) that exercise `ExecutionPlan` assembly, buffer lifecycle, strategy composition, and workload tiling without requiring a device.
- A `conftest.py` with canonical model configurations (`fp32_iris_spec`) and a `requires_opencl` marker for device-dependent tests.
- OpenCL benchmark/integration tests (`bench_opencl_backend.py`, `bench_opencl_backend_learn.py`) that require a live OpenCL device.

This existing infrastructure covers a subset of what Tier 1 requires but uses the pre-refactoring class hierarchy (`Float32ModelSpec`, `PrecisionContext` inheritance). The new test framework must accommodate both the legacy test structure (for continuity during migration) and the target architecture's `PrecisionConfig`, `HardwareProfile`, and multi-backend `PlanRenderer` abstractions.

### Open questions resolved by this ADR

1. **Test framework structure.** How should tests be organized across tiers, backends, and kernel variants?
2. **Parity comparison strategy.** How should cross-backend numerical equivalence be verified — all-vs-all or oracle-based?
3. **CPU Tier 2 fixture generation.** How should expected outputs for CPU kernel correctness tests be generated without circular dependency?
4. **Per-kernel tolerance configuration.** How should ADR-008's precision tolerances map to per-kernel comparison bounds?
5. **Test collection and skip logic.** How should the test framework discover enabled backends and conditionally execute tiers?

---

## Decision Drivers

1. **ADR-001 (Backend Abstraction Boundary — three-tier jurisdictional model).** The Policy tier is the only tier invariant across all node types and backends. Tier 1 tests validate Policy-tier correctness (plan construction, contract validation, buffer lifecycle) independently of any backend. Tier 2 tests validate the Orchestration + Execution tiers per backend. Tier 3 tests verify that the Policy-tier's invariant — identical semantics regardless of backend — holds.

2. **ADR-007 (`KernelContract` as interface specification).** The `KernelContract` frozen dataclass carries all information needed for Tier 1 validation: parameter names, types, shapes, padding contracts, placement strategies, calculability proofs, and validation preconditions. Tier 1 tests exercise this validation machinery without dispatching any kernel.

3. **ADR-008 (Precision Configuration).** Precision tolerances vary by precision format (`PrecisionConfig.numpy_dtype`). FP32 and FP16 have fundamentally different numerical ranges and accumulation error profiles. Tier 3 comparison bounds must be parameterized by precision — and may need per-kernel refinement where specific kernels (e.g., Softmax with its `exp()` and `log()` operations, or the stabilization policy's threshold arithmetic) exhibit higher sensitivity to implementation differences.

4. **ADR-009 (Buffer Lifecycle in the Plan Model).** `BufferDescriptor` lifetime annotations (`producing_node`, `last_consumer`, `consumers`) are plan-level data that Tier 1 must validate: every buffer has exactly one producer, all consumers reference live buffers, and no buffer is read after its last consumer.

5. **ADR-011 (CCE/BCE Strategy Delegation).** Strategy B (Nodes 6/7) and Strategy A (Nodes 8/9/10) create a combinatorial test surface. Strategy B requires separate test cases per variant (CCE, BCE) because the kernels have genuinely different interfaces and algorithms. Strategy A requires testing both flag values (`PROBLEM_TYPE_CCE`, `PROBLEM_TYPE_BCE`) for each kernel.

6. **ADR-013 (Kernel Source Strategy — three-tier specification hierarchy).** The fidelity assurance model establishes three layers: specification review (`kernels.cl.h`), interface validation (`KernelContract`), and behavioral verification (this ADR, Tier 3). This ADR provides the definitive behavioral layer.

7. **ADR-014 (Build System Integration — `_build_config.py` manifest).** The generated build manifest is the single authoritative source for backend availability. Test collection must read this module to determine which tiers are executable. The `auto` default means CI environments self-configure — the test framework must not assume any backend's presence.

8. **ADR-015 (Python ↔ Native Backend Interop — layout verification).** ADR-015's `_verify_layouts()` catches struct size drift at library load time. Same-size field reorderings are caught only by Tier 2 behavioral tests. The test strategy must treat CPU Tier 2 tests as load-bearing for FFI correctness.

9. **CONCEPT.md §1 (Architectural Elegance Feedback).** If a test reveals a pattern that cannot be expressed within the current tier model — e.g., a test that needs to validate an optimization spanning multiple backends simultaneously — the response is to extend the tier vocabulary as a documented architectural primitive, not to bypass the framework.

10. **CONCEPT.md §5 (Unified Execution Model).** Act and Learn phases follow a fixed sequencing regardless of backend. End-to-end tests must validate this sequencing through both phases, ensuring that phase sync points (`inference_event`, `final_batch_event`) are honored.

---

## Options Considered

### Test framework structure

#### Option A: Layered pytest framework

pytest markers and fixtures define each tier. `--backend` CLI flags control which tiers execute. Tier 1 always runs. Tier 2 runs for each backend where `_build_config.BACKEND_<NAME>` is `True`. Tier 3 runs if ≥ 2 backends are available. Skip logic reads `_build_config` at collection time.

**Advantages:**
- Each tier is a distinct test directory or marker, making it trivial to run a single tier in isolation: `pytest -m tier1`, `pytest -m "tier2 and opencl"`.
- Fixtures for backend lifecycle (context creation, library loading, pool initialization) are scoped to the session or module level, amortizing setup cost across all tests in a tier.
- New backends are added by registering a new marker and fixture — no test body changes.
- pytest's built-in `skipif` and parametrize decorators handle the combinatorial test matrix naturally.

**Disadvantages:**
- Requires discipline to maintain tier marker consistency — a test missing its marker silently runs in the wrong context.
- Tier 3 tests need access to multiple backend fixtures simultaneously, complicating fixture scoping.

#### Option B: Parameterized test matrix

A single test body parameterized across backends × kernels × strategies. Each test function is a generic "dispatch this kernel on this backend with these inputs and compare."

**Advantages:**
- Maximum code reuse — one test function exercises all combinations.
- Adding a kernel or backend automatically expands coverage.

**Disadvantages:**
- Scales combinatorially: 22 kernels × 3 backends × 2 strategies × 2 precisions = 264 test cases from a single parameterized function. Slow test suites, complex skip logic, and difficult failure diagnosis.
- Collapses the tier distinction — Tier 1 (host-only), Tier 2 (per-backend), and Tier 3 (cross-backend) have fundamentally different test structures that do not parameterize cleanly into a single body.
- Per-kernel tolerance tables and fixture generation strategies do not compose naturally with a flat parameterization.

### Parity comparison strategy

#### Option C: CPU reference oracle

Designate the CPU backend (deterministic, no GPU required) as the golden reference and run all parity comparisons against it. Tier 3 reduces to pairwise comparisons: OpenCL-vs-CPU and Vulkan-vs-CPU.

**Advantages:**
- Simplifies Tier 3 from all-pairs (N choose 2) to N−1 comparisons.
- CPU is deterministic (single-threaded per task, no warp/wavefront scheduling variance), making it the most reproducible backend.
- CPU execution is maximally inspectable — ADR-015's fine-grained `pool_dispatch_and_wait` dispatch means every kernel invocation can be individually logged and debugged.
- Aligns with ADR-013's identification of CPU as the natural oracle and ADR-015's confirmation of its inspectable dispatch model.

**Disadvantages:**
- Assumes CPU correctness is established independently. Circular dependency risk if CPU Tier 2 is insufficient.
- CPU backend may not support all precisions (FP16 requires AVX-512 FP16 or ARM FP16); Tier 3 is limited to precisions all compared backends support.
- If the CPU backend has a systematic bug, all parity tests pass — the oracle is wrong.

#### Option F: All-pairs comparison (no oracle)

Every pair of enabled backends is compared. With N backends, this produces N×(N−1)/2 comparisons per kernel.

**Advantages:**
- No single backend is privileged — a systematic CPU bug does not mask failures.
- Every backend pair is independently verified.

**Disadvantages:**
- Quadratic growth: 3 backends × 2 = 3 pairs per kernel. For the full kernel inventory this is manageable, but the conceptual complexity (which comparison failed?) increases.
- Harder to diagnose — when a comparison fails, which backend is wrong requires additional investigation.

### CPU Tier 2 fixture generation

#### Option D: Independent golden fixtures (numpy reference implementations)

Generate expected outputs from a trusted reference implementation — a manually-verified numpy computation for each kernel. CPU Tier 2 compares against these fixtures.

**Advantages:**
- Independent verification — CPU correctness does not depend on CPU execution.
- Fixtures are human-auditable Python/numpy code.

**Disadvantages:**
- Substantial development cost — ~22 numpy reference implementations, one per kernel.
- Risk of bugs in the reference implementation itself (though numpy is well-tested).
- Fixture maintenance burden as kernel algorithms evolve.

#### Option E: Analytical fixtures (closed-form + numpy for complex kernels)

For kernels with closed-form solutions (ReLU, element-wise scaling, Adam update with known inputs), derive expected outputs analytically. For kernels without closed-form solutions (tiled matmul, staged reduction), use numpy reference implementations with known-correct logic.

**Advantages:**
- Closed-form fixtures are trivially correct (e.g., `ReLU(x) = max(0, x)` — the fixture *is* the definition).
- Numpy reference implementations for complex kernels are substantially simpler than the kernels themselves — no tiling, no SIMD, no local memory, just the mathematical operation. Correctness is easier to audit.
- Independently verifiable — no dependency on any backend's execution.

**Disadvantages:**
- Some kernels' expected behavior is complex enough that the numpy reference approaches the kernel's own complexity (e.g., the tiled matrix-vector multiplication with SIMD padding, or the multi-stage reduction with per-stage clipping). The reference must reproduce the *mathematical* behavior, not the *implementation* structure — but the two can be difficult to separate for kernels with implementation-specific numerical accumulation patterns.
- FP32 numpy results may not exactly match FP32 GPU results due to instruction ordering, fused multiply-add availability, and denormal handling. Tolerances must account for this.

---

## Analysis

### Choosing Option A (layered pytest framework)

Option A is chosen over Option B. The three test tiers have fundamentally different structures:

- **Tier 1** constructs plan data structures and validates their properties — no dispatch, no backend, no numerical comparison. These are pure Python unit/integration tests.
- **Tier 2** dispatches individual kernels on a specific backend and compares outputs against reference fixtures. These require backend initialization, buffer allocation, and dispatch machinery.
- **Tier 3** dispatches the same `ExecutionPlan` on multiple backends and compares their outputs. These require simultaneous access to multiple backend contexts.

Option B's flat parameterization collapses these structural differences into a single test body, which produces complex skip logic and poor diagnostic quality. Option A preserves the natural structure: each tier has its own test files, fixtures, and execution requirements.

### Choosing Option C (CPU reference oracle)

Option C is chosen over Option F. The decisive factors:

**Oracle trustworthiness.** The CPU backend's correctness is established by Tier 2 tests using Option E's analytical/numpy fixtures. These fixtures are independently generated — no backend execution is involved in their creation. Once CPU Tier 2 passes, the CPU backend's outputs are trustworthy to the tolerance of the Tier 2 fixtures. This breaks the circular dependency concern.

**Diagnostic clarity.** When a CPU-vs-GPU comparison fails in Tier 3, the diagnostic path is clear: the GPU backend diverges from the CPU oracle. Under Option F, a three-way disagreement requires determining which backend(s) are correct — additional investigation that Option C's asymmetric comparison avoids.

**ADR-013 and ADR-015 alignment.** ADR-013 explicitly identifies the CPU backend as the natural oracle. ADR-015 confirms its deterministic, single-threaded dispatch model makes it maximally inspectable. The architectural documents converge on this choice.

**Fallback for GPU-only environments.** If the CPU backend is not available (ADR-014's `backend_cpu=disabled`), Tier 3 degrades to direct comparison between available GPU backends — effectively Option F for the available subset. The framework supports this degradation without structural changes to the test code.

**Mitigation of the systematic-CPU-bug risk.** The risk that a CPU bug masks all parity failures is mitigated by:
1. CPU Tier 2 uses independently-derived fixtures, not CPU-generated outputs.
2. Any CPU bug that affects Tier 3 comparisons (wrong numerical results) would also fail the CPU's own Tier 2 tests — the bug is caught before it reaches the oracle role.
3. In environments with all three backends, an optional `--all-pairs` mode can run Option F as a supplementary check. This is a CI configuration option, not a structural framework decision.

### Choosing Option E (analytical + numpy fixtures)

Option E is chosen over Option D. The distinction is subtle but meaningful: Option E explicitly identifies which kernels have closed-form expected values and which require numpy reference computation, whereas Option D treats all kernels uniformly as "golden fixture" generators. This matters because:

- Closed-form kernels (ReLU activation, mask application, element-wise scaling, temperature clamping, gradient normalization) have expected outputs that are trivially derivable from the definition. Writing a numpy "reference implementation" for `max(0, x)` is unnecessary verbosity — the fixture *is the formula*.
- Complex kernels (tiled matmul, staged reduction with per-stage clipping, Adam update with FP64 exponentiation) genuinely need a numpy reference implementation. For these kernels, the numpy code performs the mathematical operation without implementation-specific structure (no tiling, no SIMD, no local memory), providing a clean behavioral specification.

The combination produces the highest confidence at the lowest maintenance cost: trivially-correct analytical fixtures for simple kernels, and auditable numpy references for complex kernels.

---

## Decision

**Option A (layered pytest framework) + Option C (CPU reference oracle) + Option E (analytical + numpy fixtures).**

The test framework is organized into three tiers with pytest markers, conditional execution based on `_build_config.py`, CPU as the Tier 3 reference oracle, and analytical/numpy fixtures for Tier 2 CPU correctness.

### Tier definitions

#### Tier 1: Host-Side Plan Correctness

**Scope:** Validates the shared orchestration layer — `ExecutionPlan` construction, `KernelContract` validation, `BufferDescriptor` lifecycle, `MemoryLayout` placement, `StabilizationPolicy` threshold scheduling, `ReductionTreePlan` structure, `StreamingLoopPlan` structure, and strategy delegation (CCE/BCE).

**Execution requirement:** None. Tier 1 is pure Python with no device or backend dependency. It always runs in every environment.

**Test structure:** Tests exercise plan-builder functions with known model configurations and assert structural properties of the resulting plan:

- Every `KernelDispatchNode` has a valid `KernelContract` reference.
- Every buffer in `buffer_bindings` has a corresponding `BufferDescriptor` in the plan's buffer namespace.
- `BufferDescriptor` lifetime annotations (`producing_node`, `last_consumer`, `consumers`) are consistent with the DAG topology: every buffer has exactly one producing node, all consumers are downstream of the producer, and no buffer is consumed after its `last_consumer`.
- `ReductionTreePlan` stage counts and fan-in values are consistent with `StabilizationPolicy`'s threshold schedule: `len(thresholds) == num_stages`, and each threshold satisfies `T_j = T_algorithmic + λ * j²` clamped by `T_safety_j = FP_FORMAT_MAX / K_j`.
- `StreamingLoopPlan` chunk counts and parameter strides are consistent with the tiling scheme.
- Strategy A nodes (8, 9, 10) carry `src_scalar_FLAG_problem_type` in their `scalar_params`.
- Strategy B nodes (6, 7) have distinct `kernel_name` values for CCE and BCE.
- Named synchronization points (`inference_event`, `final_batch_event`) are present in the plan.
- `PrecisionConfig` fields (`numpy_dtype`, `fp_format_max`, `epsilon`) are consistent with each other and with the plan's scalar parameter values.

**Marker:** `@pytest.mark.tier1`

**File organization:**

```
tests/
├── tier1/
│   ├── __init__.py
│   ├── test_plan_construction.py          # ExecutionPlan assembly for known ModelSpecs
│   ├── test_kernel_contract_validation.py # KernelContract field validation, calculability proofs
│   ├── test_buffer_lifecycle.py           # BufferDescriptor lifetime consistency
│   ├── test_reduction_tree_plan.py        # Stage counts, fan-in, threshold schedules
│   ├── test_streaming_loop_plan.py        # Chunk counts, parameter strides
│   ├── test_strategy_delegation.py        # CCE/BCE strategy selection and plan-level expression
│   ├── test_memory_layout.py             # MemoryLayout placement, padding, alignment
│   └── test_precision_config.py          # PrecisionConfig construction, factory validation
```

#### Tier 2: Per-Backend Kernel Correctness

**Scope:** Validates that each backend's compiled kernel produces numerically correct results for known inputs. Each kernel is tested in isolation: the test constructs a minimal `KernelDispatchNode`, allocates buffers with known data, dispatches the kernel through the backend's `PlanRenderer`, and compares the output against the expected fixture.

**Execution requirement:** Per-backend. A Tier 2 test for backend `X` runs only if `_build_config.BACKEND_X` is `True`.

**Fixture generation (Option E):**

*Analytical fixtures* — used for kernels with closed-form expected outputs:

| Kernel | Fixture derivation |
| :--- | :--- |
| `compute_hidden_mask` (Node 5) | Element-wise `mask[i] = activation[i] > 0 ? 1.0 : 0.0` |
| `clamp_temperatures` (Node 25) | Element-wise `clamp(T, T_min, T_max)` |
| `normalize_gradients` (Node 21) | Element-wise `grad / batch_count` |
| `clip_partial_gradients` (Node 11) | L2 norm + conditional scaling: `if ‖g‖ > T: g *= T / ‖g‖` |
| `clip_intermediate_grad` (reduction) | Same clipping formula as Node 11, applied at reduction stage |
| `clip_shared_gradients` (Node 19) | Same clipping formula, applied to shared-weight gradients |

*Numpy reference implementations* — used for kernels with non-trivial computation:

| Kernel | Numpy reference description |
| :--- | :--- |
| `forward_pass` (Node 4) | `hidden = max(0, X @ W.T + b)` — standard matmul + ReLU, no tiling |
| `compute_probs_loss_cce` (Node 6) | Softmax + categorical cross-entropy: `probs = softmax(logits); loss = -log(probs[target])` |
| `compute_probs_loss_bce` (Node 7) | Sigmoid + binary cross-entropy: `probs = sigmoid(logits); loss = -[y*log(p) + (1-y)*log(1-p)]` |
| `calculate_module_param_grads` (Node 8) | `d_loss_d_logit` computation (CCE/BCE branch) + outer product for weight grads |
| `backprop_error_to_hidden` (Node 9) | `d_loss_d_hidden = d_loss_d_logit @ W` — matmul backprop |
| `calculate_chunk_temp_gradients` (Node 10) | Temperature gradient via chain rule through Softmax/Sigmoid |
| `gather_and_permute_grad_h` (Node 13) | Permutation of gradient tensor dimensions — index arithmetic |
| `aggregate_register_reduce` / `aggregate_local_reduce` | Sum of partials at specified offsets from an offset list |
| `stabilize_reduce_grad_h` (Node 16) | Stabilized reduction with internal `log_K(M)` loop |
| `backprop_shared_weights` (Node 17) | Outer product accumulation for shared-weight gradients |
| `backprop_shared_biases` (Node 18) | Sum along batch dimension for bias gradients |
| `adam_update` (Node 24) | Standard Adam: `m = β₁m + (1-β₁)g; v = β₂v + (1-β₂)g²; θ -= α·m̂/√(v̂+ε)` |

Each numpy reference operates on standard numpy arrays with no SIMD padding, no tiling, and no local memory simulation. The reference computes the *mathematical* result; the test compares the backend's output (extracted from padded, SIMD-aligned buffers) against this reference within tolerance.

**Tolerance configuration:**

Per-kernel absolute and relative tolerances are defined in a tolerance table. Base tolerances derive from `PrecisionConfig`:

| Precision | Default `atol` | Default `rtol` |
| :--- | :--- | :--- |
| FP32 | `1e-5` | `1e-5` |
| FP16 | `1e-2` | `1e-2` |

Kernels with higher numerical sensitivity override the defaults:

| Kernel | Tolerance override | Rationale |
| :--- | :--- | :--- |
| `compute_probs_loss_cce` | `atol=1e-4, rtol=1e-4` (FP32) | `exp()` and `log()` accumulation in Softmax |
| `compute_probs_loss_bce` | `atol=1e-4, rtol=1e-4` (FP32) | `sigmoid()` near saturation |
| `stabilize_reduce_grad_h` | `atol=1e-4, rtol=1e-4` (FP32) | Multi-stage internal reduction |
| `adam_update` | `atol=1e-4, rtol=1e-4` (FP32) | Division by `√(v̂+ε)` amplifies small differences |
| `aggregate_local_reduce` | `atol=1e-4, rtol=1e-4` (FP32) | Multi-element summation order sensitivity |

The tolerance table is a Python dict keyed by `(kernel_name, numpy_dtype)`, falling back to the precision-level defaults. It lives in a shared test module (`tests/tolerance_config.py`) consumed by both Tier 2 and Tier 3.

**Marker:** `@pytest.mark.tier2`; additionally `@pytest.mark.opencl`, `@pytest.mark.vulkan`, `@pytest.mark.cpu` per backend.

**File organization:**

```
tests/
├── tier2/
│   ├── __init__.py
│   ├── conftest.py                        # Backend fixture factories, tolerance loader
│   ├── fixtures/
│   │   ├── __init__.py
│   │   ├── analytical.py                  # Closed-form expected-output generators
│   │   └── numpy_reference.py             # Numpy reference implementations per kernel
│   ├── test_opencl_kernels.py             # Per-kernel tests for OpenCL backend
│   ├── test_vulkan_kernels.py             # Per-kernel tests for Vulkan backend
│   └── test_cpu_kernels.py                # Per-kernel tests for CPU backend
```

Each `test_<backend>_kernels.py` file is parameterized over the kernel inventory. A single parameterized test function per backend dispatches each kernel with known inputs and compares against the fixture:

```python
@pytest.mark.tier2
@pytest.mark.cpu
@pytest.mark.parametrize("kernel_name,fixture_fn", KERNEL_FIXTURE_TABLE)
def test_cpu_kernel(cpu_renderer, kernel_name, fixture_fn, tolerance_config):
    inputs, expected = fixture_fn(model_spec, precision_config)
    actual = dispatch_single_kernel(cpu_renderer, kernel_name, inputs)
    assert_close(actual, expected, **tolerance_config[kernel_name])
```

Strategy A kernels are parameterized over problem type:

```python
@pytest.mark.tier2
@pytest.mark.parametrize("problem_type", ["CCE", "BCE"])
@pytest.mark.parametrize("kernel_name", STRATEGY_A_KERNELS)
def test_cpu_strategy_a_kernel(cpu_renderer, kernel_name, problem_type, ...):
    ...
```

#### Tier 3: Cross-Backend Parity

**Scope:** Validates that all enabled backends produce equivalent outputs for the same `ExecutionPlan`. The CPU backend is the reference oracle; each GPU backend is compared against it.

**Execution requirement:** The CPU backend must be enabled (`_build_config.BACKEND_CPU is True`) plus at least one other backend. If the CPU backend is not available but two or more GPU backends are available, Tier 3 falls back to direct GPU-vs-GPU comparison (no oracle — equivalent to Option F for the available subset).

**Comparison granularity:** Tier 3 operates at two levels:

1. **Per-kernel parity.** The same `KernelDispatchNode` is executed on the oracle and the comparison backend. Individual kernel outputs are compared within the tolerance table. This catches per-kernel algorithmic divergence.

2. **End-to-end parity.** A complete `ExecutionPlan` (Act + Learn phases) is executed on both backends for a canonical model configuration (Iris-scale). The plan-level outputs — `Final Probs` (post-Act), updated parameters (post-Learn) — are compared. This catches interaction effects that per-kernel tests might miss (e.g., error accumulation across the DAG, buffer lifecycle errors, synchronization point mishandling).

**Oracle dispatch pattern:**

```python
@pytest.mark.tier3
@pytest.mark.parametrize("comparison_backend", get_non_oracle_backends())
def test_parity_per_kernel(oracle_renderer, comparison_renderer, comparison_backend,
                           kernel_name, tolerance_config):
    plan_node = build_single_kernel_plan(kernel_name, model_spec)
    oracle_output = oracle_renderer.execute(plan_node)
    comparison_output = comparison_renderer.execute(plan_node)
    assert_close(oracle_output, comparison_output, **tolerance_config[kernel_name])
```

**End-to-end test:**

```python
@pytest.mark.tier3
@pytest.mark.parametrize("comparison_backend", get_non_oracle_backends())
@pytest.mark.parametrize("problem_type", ["CCE", "BCE"])
def test_parity_e2e(oracle_renderer, comparison_renderer, comparison_backend,
                    problem_type, iris_plan):
    oracle_probs, oracle_params = oracle_renderer.execute_full(iris_plan)
    comp_probs, comp_params = comparison_renderer.execute_full(iris_plan)
    assert_close(oracle_probs, comp_probs, atol=1e-4, rtol=1e-4)
    for name in oracle_params:
        assert_close(oracle_params[name], comp_params[name], atol=1e-4, rtol=1e-4)
```

**Marker:** `@pytest.mark.tier3`

**File organization:**

```
tests/
├── tier3/
│   ├── __init__.py
│   ├── conftest.py                        # Multi-backend fixture setup, oracle selection
│   ├── test_parity_per_kernel.py          # Per-kernel cross-backend comparison
│   └── test_parity_e2e.py                # End-to-end Act+Learn parity
```

### Test collection and skip logic

Skip logic is implemented through pytest conftest fixtures and markers that read `_build_config` at collection time:

```python
# tests/conftest.py (top-level)

import pytest

def _load_build_config():
    """Load the build manifest; return a dict of backend availability."""
    try:
        from averaging_ensembled_classifier import _build_config
        return {
            "opencl": getattr(_build_config, "BACKEND_OPENCL", False),
            "vulkan": getattr(_build_config, "BACKEND_VULKAN", False),
            "cpu": getattr(_build_config, "BACKEND_CPU", False),
        }
    except ImportError:
        # Pre-migration: _build_config not yet generated.
        # Fall back to runtime probing for OpenCL only.
        return {"opencl": _probe_opencl(), "vulkan": False, "cpu": False}

BUILD_CONFIG = _load_build_config()

def pytest_collection_modifyitems(config, items):
    """Skip tests whose backend requirements are not met."""
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
                    reason=f"Tier 3 requires ≥ 2 backends ({available} available)"
                ))
```

**CLI usage:**

```bash
# Run only Tier 1 (host-side, always works)
pytest -m tier1

# Run Tier 1 + Tier 2 for all available backends
pytest -m "tier1 or tier2"

# Run Tier 2 for CPU only
pytest -m "tier2 and cpu"

# Run all tiers
pytest

# Run Tier 3 with all-pairs mode (optional supplementary check)
pytest -m tier3 --all-pairs
```

### Canonical test model configurations

Test configurations mirror the existing `conftest.py` Iris-scale model but use the target architecture's dataclasses:

| Configuration | Purpose | Parameters |
| :--- | :--- | :--- |
| `iris_fp32` | Primary correctness | `input_dim=4, hidden_dim=32, output_classes=3, num_modules=8, batch_size=150, FP32` |
| `iris_fp16` | Precision-sensitivity | Same geometry as `iris_fp32`, `FP16` |
| `stress_fp32` | Scale-sensitivity | `input_dim=128, hidden_dim=512, output_classes=100, num_modules=32, batch_size=1024, FP32` |

The `iris_fp32` configuration is used for all Tier 2 and Tier 3 tests. The `stress_fp32` configuration is used for Tier 3 end-to-end tests to stress-test reduction tree depths, streaming loop chunk counts, and buffer lifecycle at scale. The `iris_fp16` configuration is used where the backend supports FP16 (gated by `PrecisionConfig` + backend capability check).

### Test target inventory

The kernel inventory from ADR-013 gives a concrete test surface per backend:

| Phase file | Kernels | Strategy variants | Tier 2 tests per backend |
| :--- | :--- | :--- | :--- |
| `phase_1_act` | `forward_pass`, `compute_hidden_mask`, `compute_probs_loss_cce`, `compute_probs_loss_bce` | B (problem_type) | 4 |
| `phase_2_learn_A_production` | `calculate_module_param_grads`, `backprop_error_to_hidden`, `calculate_chunk_temp_gradients` | A (FLAG) | 3 × 2 = 6 |
| `phase_2_learn_B_processing` | `clip_partial_gradients`, `gather_and_permute_grad_h` | — | 2 |
| `phase_2_learn_C_reduction` | `aggregate_register_reduce`, `aggregate_local_reduce`, `clip_intermediate_grad` | — | 3 |
| `phase_2_learn_D_backprop` | `stabilize_reduce_grad_h`, `backprop_shared_weights`, `backprop_shared_biases`, `clip_shared_gradients` | — | 4 |
| `phase_3_update` | `normalize_gradients`, `adam_update`, `clamp_temperatures` | — | 3 |

This yields **22 Tier 2 test cases per backend**, and **22 × (backends − 1) Tier 3 per-kernel comparisons** under the CPU oracle model.

### Relationship to existing tests

The existing `tests/` directory contains pre-refactoring integration tests that exercise the monolithic PyOpenCL code path. During migration (ADR-017), these tests remain operational:

- **Phase 1–2 (ADR-017):** Existing tests continue to run against the legacy code path. New Tier 1 tests are written alongside them — the two test sets coexist.
- **Phase 3 (CPU backend):** New Tier 2 CPU tests exercise the new backend. Legacy OpenCL integration tests continue to validate that existing functionality is unbroken.
- **Phase 4 (test harness):** The full Tier 1/2/3 framework is operational. Legacy tests may be migrated into the new tier structure or retained as regression tests.
- **Phase 6 (legacy removal):** Legacy tests are retired. The Tier 1/2/3 framework is the sole test infrastructure.

The existing `conftest.py` is preserved during migration. The new tier-specific `conftest.py` files extend (not replace) the shared fixture infrastructure.

### Node 16 test considerations

ADR-005 (Node 16 Opacity in the Plan) establishes that Node 16 (`stabilize_and_reduce_grad_hidden_activations`) is opaque to the plan — it manages its own internal multi-stage reduction. This affects testing:

- **Tier 1:** Node 16 appears as a single `KernelDispatchNode` in the plan. Tier 1 validates its `KernelContract` and buffer bindings but does not inspect its internal reduction structure (which is invisible to the plan).
- **Tier 2:** Node 16's numpy reference implementation must reproduce the full internal `log_K(M)` reduction with per-stage stabilization. This is the most complex fixture — it requires simulating the kernel's internal loop structure at the mathematical level. The fixture's correctness is independently auditable (it is numpy code, not device code).
- **Tier 3:** Node 16 is compared across backends as any other kernel. The CPU implementation uses explicit SIMD vectorization with no barriers (single-threaded per workgroup); the OpenCL implementation uses local memory with barriers; the Vulkan implementation uses subgroup operations. All three must converge to the same mathematical result within tolerance.

---

## Consequences

### Positive

- **Complete coverage of the three-tier specification hierarchy.** Tier 1 validates the Interface tier (`KernelContract`), Tier 2 validates the Implementation tier (per-backend source correctness), and Tier 3 validates cross-implementation equivalence. The Algorithm tier (`kernels.cl.h`) is verified transitively — any algorithmic divergence that affects behavior is caught by Tier 2 or Tier 3.

- **ADR-015 FFI correctness is load-bearing on Tier 2.** CPU Tier 2 tests behaviorally verify that ctypes struct marshalling is correct. ADR-015's `_verify_layouts()` catches size-level drift; Tier 2 catches same-size field reorderings. Together they provide comprehensive FFI verification without a separate FFI test suite.

- **Graceful degradation across CI environments.** Tier 1 always runs. Tier 2 runs for whatever backends are available. Tier 3 runs when ≥ 2 backends are present. A CPU-only CI machine runs Tier 1 + CPU Tier 2. A GPU CI machine runs all tiers. No manual configuration is needed — `_build_config.py`'s `auto` defaults handle this.

- **Independent fixture verification breaks circular dependency.** CPU Tier 2 tests use analytical/numpy fixtures that are generated without executing any backend. CPU correctness is established independently before the CPU backend serves as the Tier 3 oracle. The oracle's trustworthiness is grounded in Tier 2, not in self-reference.

- **Per-kernel tolerance tables enable precision-aware comparison.** Kernels with higher numerical sensitivity (Softmax, Adam, multi-stage reductions) have wider tolerances. Kernels with exact mathematical operations (mask, clamp, normalize) have tighter tolerances. This prevents false positives from expected numerical divergence while maintaining sensitivity to actual bugs.

- **Migration-compatible.** The tier structure is additive — new tiers and backends are added alongside existing tests without disruption. The existing `conftest.py` and integration tests remain operational until ADR-017's Phase 6 retires them.

- **Unblocks ADR-017.** ADR-017's rollback gates can now be formally defined in terms of tier outcomes: Phase 1 requires Tier 1 green; Phase 3 requires Tier 1 + CPU Tier 2 green; Phase 5 requires Tier 1 + Vulkan Tier 2 green; Phase 6 requires Tier 3 parity green across all enabled backends.

### Negative

- **Fixture development cost.** ~22 numpy reference implementations and analytical fixtures must be written and audited. This is a substantial upfront investment. The cost is mitigated by the fact that each fixture is a simple, flat numpy computation — no tiling, no SIMD, no local memory — making them faster to write and easier to audit than the kernel implementations themselves.

- **Tolerance table maintenance.** Per-kernel tolerance overrides must be maintained as kernel algorithms evolve. A tightened kernel (e.g., improved numerical stability in Softmax) may require updating the tolerance entry. This is an ongoing, low-frequency maintenance task.

- **Node 16 fixture complexity.** The numpy reference for Node 16 must reproduce a multi-stage internal reduction with per-stage stabilization — the most complex fixture in the inventory. This reference itself requires careful verification, though its numpy-only nature makes it auditable.

- **CPU-as-oracle limitation for GPU-only environments.** If the CPU backend is unavailable, Tier 3 falls back to GPU-vs-GPU comparison with no privileged oracle. Failure diagnosis in this mode is harder — a two-way disagreement provides no tie-breaker. This is acceptable because GPU-only CI environments are expected to be rare (ADR-014's `backend_cpu` auto-enables when a C compiler is present).

- **FP16 Tier 3 coverage depends on backend capability.** CPU FP16 support requires specific hardware (AVX-512 FP16, ARM FP16). If the CPU backend does not support FP16, Tier 3 for FP16 falls back to GPU-vs-GPU comparison. This is a hardware-contingent limitation, not a framework limitation.

### Migration implications (ADR-017)

This decision enables formal gate definitions for ADR-017's migration phases:

| Phase | Rollback gate |
| :--- | :--- |
| **1** (Host-side plan model) | Tier 1 green — all plan construction, contract validation, buffer lifecycle, and strategy delegation tests pass |
| **2** (PyOpenCL backend adapter) | Tier 1 green + OpenCL Tier 2 green — existing kernel correctness preserved under the new plan-model dispatch |
| **3** (CPU backend) | Tier 1 green + CPU Tier 2 green — CPU kernels produce correct results via ctypes FFI |
| **4** (Test harness) | Full Tier 1/2/3 framework operational; all enabled tiers green |
| **5** (Vulkan backend) | Tier 1 green + Vulkan Tier 2 green + Tier 3 parity (Vulkan-vs-oracle) green |
| **6** (Legacy removal) | Tier 3 parity green across all enabled backends for the full kernel inventory at both FP32 and supported FP16 configurations |

---

## References

- [ADR-001: Backend Abstraction Boundary](ADR-001-backend-abstraction-boundary.md) — three-tier jurisdictional model (Policy / Orchestration / Execution); plan-as-data-structure; backend rendering contract
- [ADR-005: Node 16 Opacity in the Plan](ADR-005-node-16-opacity-in-the-plan.md) — Node 16's internal reduction is opaque to the plan; affects Tier 2 fixture complexity
- [ADR-007: KernelSignature Contract/Binding Split](ADR-007-kernel-signature-contract-binding-split.md) — `KernelContract` interface specification; plan-time validation tested by Tier 1
- [ADR-008: Precision Configuration](ADR-008-precision-configuration.md) — `PrecisionConfig` frozen dataclass; precision-derived tolerances for numerical comparison
- [ADR-009: Buffer Lifecycle in the Plan Model](ADR-009-buffer-lifecycle-in-the-plan-model.md) — `BufferDescriptor` lifetime annotations validated by Tier 1
- [ADR-011: CCE/BCE Strategy Delegation](ADR-011-cce-bce-strategy-delegation.md) — Strategy A/B variants expanding the test matrix; Strategy A FLAG parameterization in Tier 2
- [ADR-013: Kernel Source Strategy](ADR-013-kernel-source-strategy.md) — three-tier specification hierarchy; kernel inventory; CPU as natural oracle candidate; cross-backend fidelity assurance model
- [ADR-014: Build System Integration](ADR-014-build-system-integration.md) — `_build_config.py` manifest; conditional backend enablement; `auto` defaults for CI self-configuration
- [ADR-015: Python ↔ Native Backend Interop](ADR-015-python-native-backend-interop.md) — ctypes FFI with `_verify_layouts()` pre-gate; Tier 2 as behavioral FFI verification; CPU dispatch inspectability
- [ADR-017: Incremental Migration Path](ADR-017-incremental-migration-path-stub.md) — migration phase rollback gates defined by tier outcomes
- [CONCEPT.md](../CONCEPT.md) — §1 Architectural Elegance Feedback; §3 Modular Dumb Kernels; §3.4 Safety ceiling calculations; §5 Unified Execution Model
- [CONTRACT.md](../CONTRACT.md) — Article 1.4 Collaborative Interface Verifiability; Article 3.2 Placement Contract; Article 5 Architectural Constants
- [CPU_BACKEND.md](../CPU_BACKEND.md) — deterministic single-threaded-per-task execution model; `pool_dispatch_and_wait` dispatch
