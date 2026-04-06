# Phase 8E: Double Precision — FP64 Tier 2 Tests & Alchemist II Validation

**Status: ✅ COMPLETED**  
**Phase:** 8E of 8  
**Prerequisite:** Phases 8A–8D complete and all rollback gates passed.  
**Objective:** Add comprehensive Tier 2 test coverage for all four FP64 `PrecisionConfig` factories across all three backends. Implement the Alchemist II validation scenario (ADR-024 §8.2). Verify backend capability checks fail fast on unsupported hardware. Run full migration completion verification. The phase ends when all FP64 test scenarios pass on at least one backend (CPU) and gracefully skip on backends lacking FP64 support.  
**Governing ADR:** ADR-024 §8  
**Rollback gate:** All prior tests pass. FP64 factory tests pass on the CPU backend. Capability-gated tests skip cleanly on Vulkan/OpenCL devices lacking FP64. The Alchemist II scenario produces the expected precision-erosion differential. `grep -r "cpu_compute_t\|_IS_DOUBLE" src/` returns results only in the expected locations (type_mapping, kernels.cl.h, common.glsl).  
**Dependencies:** Phases 8A–8D complete.

---

## Table of Contents

1. [Scope & Constraints](#1-scope--constraints)
2. [Pre-Condition Inventory](#2-pre-condition-inventory)
3. [Task Breakdown](#3-task-breakdown)
   - [Step 8E.1: Extend `conftest.py` — FP64 precision fixtures](#step-8e1-extend-conftestpy--fp64-precision-fixtures)
   - [Step 8E.2: Add FP64 `PrecisionConfig` invariant tests](#step-8e2-add-fp64-precisionconfig-invariant-tests)
   - [Step 8E.3: Add FP64 `StabilizationPolicy` tests](#step-8e3-add-fp64-stabilizationpolicy-tests)
   - [Step 8E.4: Add FP64 `ModelSpec` padding tests](#step-8e4-add-fp64-modelspec-padding-tests)
   - [Step 8E.5: Add FP64 plan builder tests](#step-8e5-add-fp64-plan-builder-tests)
   - [Step 8E.6: Add FP64 CPU backend integration tests](#step-8e6-add-fp64-cpu-backend-integration-tests)
   - [Step 8E.7: Add FP64 OpenCL backend integration tests](#step-8e7-add-fp64-opencl-backend-integration-tests)
   - [Step 8E.8: Add FP64 Vulkan backend integration tests](#step-8e8-add-fp64-vulkan-backend-integration-tests)
   - [Step 8E.9: Implement the Alchemist II validation scenario](#step-8e9-implement-the-alchemist-ii-validation-scenario)
   - [Step 8E.10: Full migration completion verification](#step-8e10-full-migration-completion-verification)
4. [Numerical Tolerance Strategy](#4-numerical-tolerance-strategy)
5. [Risk Register](#5-risk-register)

---

## 1. Scope & Constraints

### In scope

- Test fixtures for FP64 `PrecisionConfig` factories.
- Unit tests validating `PrecisionConfig` invariants, derived constants, and construction error cases for FP64.
- `StabilizationPolicy` tests at FP64 `compute_fp_format_max` — verifying safety ceiling computation doesn't overflow.
- `ModelSpec` padding tests with FP64 storage (8-byte elements).
- Plan builder tests constructing plans with FP64 configs — verifying buffer sizes, element sizes, precision roles.
- CPU backend integration tests exercising FP64 kernel dispatch (via the FFI layer) for at least `PrecisionConfig.float64()` and `PrecisionConfig.mixed_f32_f64_state()`.
- OpenCL backend integration tests (capability-gated: `@pytest.mark.skipif` when `cl_khr_fp64` unavailable).
- Vulkan backend integration tests (capability-gated: `@pytest.mark.skipif` when `shaderFloat64` unavailable).
- The Alchemist II validation scenario (ADR-024 §8.2): long-run Adam convergence comparison between FP32-state and FP64-state.
- Full migration completion verification per ADR-024 Migration Path checklist.

### Out of scope

- Performance benchmarking of FP64 vs FP32 kernel throughput.
- FP8 factories or tests (ADR-020 §5 preconditions not yet met).
- Numerical certification workloads (beyond the Alchemist II scenario).

---

## 2. Pre-Condition Inventory

| File | Relevant current state |
|:---|:---|
| `src/tests/conftest.py` | `PRECISION_CONFIGS` list with three factories. `MODEL_SPEC_FACTORIES` list. |
| `src/tests/test_mixed_precision.py` | Alchemist scenario for FP16/FP32/mixed. Multi-config parametrization. |
| `src/tests/test_stabilization_policy.py` | Tests with `compute_fp_format_max` for FP32. |
| `src/tests/test_model_spec.py` | Padding tests with FP32/FP16 element sizes. |
| All backends | FP64 code paths compiled (Phases 8B–8D) but not yet exercised by tests. |

---

## 3. Task Breakdown

---

### Step 8E.1: Extend `conftest.py` — FP64 precision fixtures

**File:** `src/tests/conftest.py`

Add FP64 factory fixtures:

```python
FP64_PRECISION_CONFIGS = [
    PrecisionConfig.float64(),
    PrecisionConfig.mixed_f32_f64_state(),
    PrecisionConfig.mixed_f16_f64_state(),
    PrecisionConfig.mixed_f32_f64(),
]

ALL_PRECISION_CONFIGS = PRECISION_CONFIGS + FP64_PRECISION_CONFIGS
```

Add pytest fixtures:

```python
@pytest.fixture(params=FP64_PRECISION_CONFIGS, ids=lambda p: _config_id(p))
def fp64_precision(request):
    return request.param

@pytest.fixture(params=ALL_PRECISION_CONFIGS, ids=lambda p: _config_id(p))
def any_precision(request):
    return request.param
```

Where `_config_id()` produces human-readable strings like `"float64"`, `"mixed_f32_f64_state"`, etc.

---

### Step 8E.2: Add FP64 `PrecisionConfig` invariant tests

**File:** `src/tests/test_mixed_precision.py` (or a new `test_fp64_precision.py`)

```python
class TestFP64PrecisionConfig:
    def test_float64_factory_construction(self):
        p = PrecisionConfig.float64()
        assert p.storage_dtype == np.dtype(np.float64)
        assert p.compute_dtype == np.dtype(np.float64)
        assert p.state_dtype == np.dtype(np.float64)
        assert p.compute_fp_format_max == float(np.finfo(np.float64).max)
        assert p.compute_epsilon == 1e-15

    def test_mixed_f32_f64_state_factory(self):
        p = PrecisionConfig.mixed_f32_f64_state()
        assert p.storage_dtype == np.dtype(np.float32)
        assert p.compute_dtype == np.dtype(np.float32)
        assert p.state_dtype == np.dtype(np.float64)
        assert p.state_dtype.itemsize > p.compute_dtype.itemsize  # state > compute is valid

    def test_mixed_f16_f64_state_factory(self):
        p = PrecisionConfig.mixed_f16_f64_state()
        assert p.storage_dtype == np.dtype(np.float16)
        assert p.compute_dtype == np.dtype(np.float32)
        assert p.state_dtype == np.dtype(np.float64)

    def test_mixed_f32_f64_factory(self):
        p = PrecisionConfig.mixed_f32_f64()
        assert p.storage_dtype == np.dtype(np.float32)
        assert p.compute_dtype == np.dtype(np.float64)
        assert p.state_dtype == np.dtype(np.float64)
        assert p.compute_fp_format_max == float(np.finfo(np.float64).max)
        assert p.compute_epsilon == 1e-15

    def test_invalid_fp64_storage_fp32_compute_rejected(self):
        """FP64 storage with FP32 compute violates storage <= compute invariant."""
        with pytest.raises(AssertionError, match="storage precision must not be wider"):
            PrecisionConfig(
                storage_dtype=np.dtype(np.float64),
                compute_dtype=np.dtype(np.float32),
                state_dtype=np.dtype(np.float64),
                storage_fp_format_max=float(np.finfo(np.float64).max),
                compute_fp_format_max=float(np.finfo(np.float32).max),
                compute_epsilon=float(np.finfo(np.float32).eps),
            )

    def test_state_wider_than_compute_is_valid(self):
        """State role is independent of compute — state > compute is permitted."""
        p = PrecisionConfig.mixed_f32_f64_state()
        assert p.state_dtype.itemsize > p.compute_dtype.itemsize
        # No assertion error: this is the purpose of the FP64 state extension.

    def test_existing_factories_unchanged(self):
        """Regression: existing factories produce identical values."""
        p32 = PrecisionConfig.float32()
        assert p32.compute_fp_format_max == float(np.finfo(np.float32).max)
        assert p32.compute_epsilon == float(np.finfo(np.float32).eps)
        p16 = PrecisionConfig.float16()
        assert p16.compute_fp_format_max == float(np.finfo(np.float16).max)
```

---

### Step 8E.3: Add FP64 `StabilizationPolicy` tests

**File:** `src/tests/test_stabilization_policy.py`

```python
def test_fp64_safety_ceiling_no_overflow():
    """At FP64, the safety ceiling is astronomically large but finite."""
    policy = StabilizationPolicy(
        t_algorithmic=1.0,
        lambda_=1.0,
        compute_fp_format_max=float(np.finfo(np.float64).max),
    )
    for k in [2, 32, 64, 128, 1024]:
        ceiling = policy.compute_fp_format_max / k
        assert math.isfinite(ceiling)
        assert ceiling > 0
        assert ceiling > 1e300  # Sanity: still astronomically large
```

---

### Step 8E.4: Add FP64 `ModelSpec` padding tests

**File:** `src/tests/test_model_spec.py`

Parametrize existing padding tests with FP64 factories. For `PrecisionConfig.float64()`:
- `storage_dtype.itemsize == 8`
- `padded_input_dim` aligns to cache-line boundary using 8-byte elements
- Buffer sizes are double those of the equivalent FP32 configuration

---

### Step 8E.5: Add FP64 plan builder tests

**File:** `src/tests/test_buffer_plumbing.py` or `test_mixed_precision.py`

Construct a plan with `PrecisionConfig.mixed_f32_f64_state()` and verify:
1. Storage-role `BufferDescriptor` objects have `element_size_bytes == 4` (FP32 storage).
2. State-role `BufferDescriptor` objects have `element_size_bytes == 8` (FP64 state).
3. Compute-role `BufferDescriptor` objects have `element_size_bytes == 4` (FP32 compute).
4. The precision role assignments match the kernel-contract `BufferParamSpec` tables.

---

### Step 8E.6: Add FP64 CPU backend integration tests

**File:** `src/tests/test_fp64_cpu.py` (new)

Tests that exercise the CPU FFI layer with FP64 configs:

```python
@pytest.mark.skipif(not _cpu_available(), reason="CPU backend not available")
class TestFP64CPU:
    def test_dispatch_table_resolves_fp64_suffixes(self):
        """All FP64 factories resolve to valid CPU kernel suffixes."""
        for config in FP64_PRECISION_CONFIGS:
            suffix = _select_suffix(config)
            assert suffix in PRECISION_SUFFIXES

    def test_struct_layout_verification_fp64(self):
        """ctypes struct sizes match C struct sizes for FP64 variants."""
        # Calls get_struct_size_* exports for each FP64 suffix
        ...

    def test_forward_pass_float64_produces_finite_output(self):
        """Smoke test: forward_pass with PrecisionConfig.float64() produces finite values."""
        ...

    def test_adam_update_mixed_f32_f64_state(self):
        """Adam update with FP64 state stores double-precision moment vectors."""
        ...
```

---

### Step 8E.7: Add FP64 OpenCL backend integration tests

**File:** `src/tests/test_fp64_opencl.py` (new)

```python
@pytest.mark.skipif(
    not _opencl_fp64_available(),
    reason="OpenCL device lacks cl_khr_fp64 support"
)
class TestFP64OpenCL:
    def test_compiler_flags_include_double_symbols(self):
        """build_compiler_flags() emits COMPUTE_TYPE_IS_DOUBLE and STATE_TYPE_IS_DOUBLE."""
        flags = build_compiler_flags(PrecisionConfig.float64(), hw, tile)
        assert "-DCOMPUTE_TYPE=double" in flags
        assert "-DCOMPUTE_TYPE_IS_DOUBLE=1" in flags
        assert "-DSTATE_TYPE_IS_DOUBLE=1" in flags

    def test_kernel_compilation_fp64(self):
        """OpenCL kernels compile successfully with FP64 flags."""
        ...
```

---

### Step 8E.8: Add FP64 Vulkan backend integration tests

**File:** `src/tests/test_fp64_vulkan.py` (new)

```python
@pytest.mark.skipif(
    not _vulkan_fp64_available(),
    reason="Vulkan device lacks shaderFloat64 support"
)
class TestFP64Vulkan:
    def test_spv_variant_suffix_resolution(self):
        """All FP64 factories resolve to valid SPIR-V variant suffixes."""
        for config in FP64_PRECISION_CONFIGS:
            suffix = _spv_variant_suffix(config)
            assert suffix.startswith("_s")

    def test_spv_files_exist_for_fp64_variants(self):
        """SPIR-V artifacts exist for FP64 variants."""
        ...

    def test_capability_check_fails_gracefully(self):
        """PrecisionConfig.float64() on a device without shaderFloat64 raises RuntimeError."""
        ...
```

---

### Step 8E.9: Implement the Alchemist II validation scenario

**Governing authority:** ADR-024 §8.2  
**File:** `src/tests/test_mixed_precision.py`

```python
@pytest.mark.slow
@pytest.mark.skipif(not _cpu_available(), reason="CPU backend required for Alchemist II")
def test_alchemist_ii_fp64_state_stability():
    """ADR-024 §8.2: FP64 state tracks reference; FP32 state diverges."""
    beta1, beta2 = 0.999, 0.9999
    steps = 100_000  # Reduced from 10^6 for CI; sufficient to demonstrate divergence.

    # Reference: pure FP64 numpy Adam
    ref_m1, ref_m2 = np.float64(0.0), np.float64(0.0)
    for t in range(1, steps + 1):
        g = np.float64(1e-4)  # Constant small gradient
        ref_m1 = beta1 * ref_m1 + (1 - beta1) * g
        ref_m2 = beta2 * ref_m2 + (1 - beta2) * g * g

    # FP32-state configuration
    m1_f32, m2_f32 = np.float32(0.0), np.float32(0.0)
    for t in range(1, steps + 1):
        g = np.float32(1e-4)
        m1_f32 = np.float32(beta1) * m1_f32 + np.float32(1 - beta1) * g
        m2_f32 = np.float32(beta2) * m2_f32 + np.float32(1 - beta2) * g * g

    # FP64-state configuration (mixed_f32_f64_state: compute=FP32, state=FP64)
    m1_f64, m2_f64 = np.float64(0.0), np.float64(0.0)
    for t in range(1, steps + 1):
        g = np.float32(1e-4)  # Compute is FP32
        m1_f64 = np.float64(beta1) * m1_f64 + np.float64(1 - beta1) * np.float64(g)
        m2_f64 = np.float64(beta2) * m2_f64 + np.float64(1 - beta2) * np.float64(g) * np.float64(g)

    # FP64-state tracks reference within FP64 tolerance
    assert abs(m1_f64 - ref_m1) / abs(ref_m1) < 1e-14
    assert abs(m2_f64 - ref_m2) / abs(ref_m2) < 1e-14

    # FP32-state diverges measurably from reference
    f32_rel_err_m1 = abs(float(m1_f32) - float(ref_m1)) / abs(float(ref_m1))
    assert f32_rel_err_m1 > 1e-7, (
        f"Expected FP32 state to diverge from FP64 reference, "
        f"but relative error was {f32_rel_err_m1}"
    )
```

This test validates the core thesis of ADR-024 §Context.1: FP64 state maintains precision where FP32 state accumulates rounding errors over long training runs.

---

### Step 8E.10: Full migration completion verification

Run the complete ADR-024 Migration Path verification:

```bash
cd architectures/averaging_ensembled_classifier

echo "=== Phase 8 Migration Verification ==="

# 1. PrecisionConfig factories
python -c "
from src.shared.precision_config import PrecisionConfig
for name in ['float64', 'mixed_f32_f64_state', 'mixed_f16_f64_state', 'mixed_f32_f64']:
    p = getattr(PrecisionConfig, name)()
    print(f'{name}: s={p.storage_dtype}, c={p.compute_dtype}, x={p.state_dtype}')
print('OK: All FP64 factories construct')
"

# 2. OpenCL type mapping emits _IS_DOUBLE
python -c "
from src.backends.opencl.type_mapping import build_compiler_flags
from src.shared.precision_config import PrecisionConfig
from src.shared.hardware_profile import HardwareProfile
hw = HardwareProfile(simd_width=8, cache_line_bytes=64, max_reduce_fan_in=64, max_local_mem_bytes=49152, global_mem_bytes=8*1024**3)
flags = build_compiler_flags(PrecisionConfig.float64(), hw, 8)
assert '-DCOMPUTE_TYPE_IS_DOUBLE=1' in flags
assert '-DSTATE_TYPE_IS_DOUBLE=1' in flags
assert '-DCOMPUTE_TYPE=double' in flags
print('OK: OpenCL flags correct for FP64')
"

# 3. CPU suffix resolution
python -c "
from src.backends.cpu._dispatch_table import _select_suffix
from src.shared.precision_config import PrecisionConfig
for name in ['float64', 'mixed_f32_f64_state', 'mixed_f16_f64_state', 'mixed_f32_f64']:
    p = getattr(PrecisionConfig, name)()
    print(f'{name} -> {_select_suffix(p)}')
print('OK: CPU suffix resolution')
"

# 4. CPU library exports all 11 suffixes
nm -D builddir/src/libcpu_kernels.so | grep "task_forward_pass_" | sort

# 5. Vulkan SPIR-V variant count
ls builddir-vulkan/src/backends/vulkan/kernel_sources/forward_pass_*.spv 2>/dev/null | wc -l

# 6. Full test suite
python -m pytest src/tests/ -q 2>&1 | tee /tmp/phase8e_final_tests.txt
grep -E "passed|failed|error|skipped" /tmp/phase8e_final_tests.txt
```

All criteria must pass. Any failure is a blocker.

---

## 4. Numerical Tolerance Strategy

| `PrecisionConfig` | `compute_dtype` | Test tolerance source | Expected tolerance |
|:---|:---|:---|:---|
| `float64()` | `np.float64` | `compute_epsilon = 1e-15` | `< 1e-14` (relative) |
| `mixed_f32_f64_state()` | `np.float32` | `compute_epsilon = np.finfo(np.float32).eps` | Standard FP32 tolerance |
| `mixed_f16_f64_state()` | `np.float32` | `compute_epsilon = np.finfo(np.float32).eps` | Standard FP32 tolerance |
| `mixed_f32_f64()` | `np.float64` | `compute_epsilon = 1e-15` | `< 1e-14` (relative) |

Tolerance is always derived from `compute_dtype`, not `storage_dtype` or `state_dtype`. Storage rounding affects input precision but not the arithmetic fidelity. State precision affects long-term stability but not per-step comparison.

---

## 5. Risk Register

| Risk | Likelihood | Mitigation |
|:---|:---|:---|
| Alchemist II test is too slow for CI at $10^6$ steps | High | Use $10^5$ steps in CI. The precision-erosion differential is visible at $10^5$ for $\beta_1 = 0.999$. Add a `@pytest.mark.slow` variant at $10^6$ for nightly runs. |
| FP64 CPU kernel produces slightly different results than the Python reference due to compiler optimization flags | Medium | Use `-O2` (not `-ffast-math`) for FP64 compilation. `-ffast-math` allows non-IEEE reordering that breaks FP64 precision guarantees. Verify Meson compiler flags. |
| FP64 tests on OpenCL/Vulkan are permanently skipped in CI (no FP64-capable device) | Medium | Ensure at least one CI runner has an FP64-capable GPU. CPU backend tests provide baseline FP64 coverage regardless. |
| FP64 state tests show no FP32→FP64 divergence for small step counts | Low | The Alchemist II uses $\beta_1 = 0.999$ (aggressive) and a constant small gradient. Divergence is demonstrable at $10^4$ steps. If not, increase `beta1` to 0.9999 or reduce gradient magnitude. |
