# tests/tier2/cpu/test_cpu_adam_accum_precision.py
"""Tier 2 CPU tests: adam_update ACCUM_TYPE derivation (ADR-027 regression).

This test validates that the ACCUM_TYPE = max(COMPUTE_TYPE, STATE_TYPE) logic
is correctly implemented in the native CPU kernel. The bug being tested:

  - The current #if logic only catches double > float/half
  - It MISSES the float > half case entirely
  - For fp8_e4m3_f16() / fp8_e5m2_f16() (FP16 compute, FP32 state):
    - ACCUM_TYPE should be float (FP32)
    - Buggy code sets ACCUM_TYPE = half (FP16), silently narrowing FP32 moments

The test executes the actual native kernel via FFI, not a Python simulation.
"""
from __future__ import annotations

import ctypes
import numpy as np
import pytest

try:
    from src._build_config import BACKEND_CPU
except ImportError:
    BACKEND_CPU = False

# Check if the FP16 variants are available.
# s16c16x32 (FP16 storage, FP16 compute, FP32 state) and s16c16x16 (all FP16)
# are compiled when _Float16 is available — both are in the base PRECISION_SUFFIXES.
HAS_FP16_COMPUTE_FP32_STATE = False
HAS_FP16_COMPUTE_FP16_STATE = False
try:
    from src.backends.cpu._ffi_types import ALL_PRECISION_SUFFIXES
    HAS_FP16_COMPUTE_FP32_STATE = "s16c16x32" in ALL_PRECISION_SUFFIXES
    HAS_FP16_COMPUTE_FP16_STATE = "s16c16x16" in ALL_PRECISION_SUFFIXES
except ImportError:
    pass

pytestmark = [
    pytest.mark.skipif(not BACKEND_CPU, reason="CPU backend not available"),
    pytest.mark.cpu,
]


@pytest.fixture(scope="module")
def cpu_lib():
    """Load the CPU kernel library once per module."""
    from src.backends.cpu._loader import load_cpu_library
    return load_cpu_library()


@pytest.fixture(scope="module")
def thread_pool(cpu_lib):
    """Create a thread pool for kernel dispatch."""
    pool = cpu_lib.pool_create(4)
    yield pool
    cpu_lib.pool_destroy(pool)


def _np_fp16_to_ctypes_uint16(arr: np.ndarray) -> np.ndarray:
    """View FP16 array as uint16 for FFI (ctypes has no _Float16)."""
    return arr.view(np.uint16)


def _ref_adam_update_correct_accum(
    params: np.ndarray,
    grads_f16: np.ndarray,
    m1: np.ndarray,
    m2: np.ndarray,
    lr: float,
    beta1: float,
    beta2: float,
    epsilon: float,
    beta1_pow_t: float,
    beta2_pow_t: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Reference Adam with correct ACCUM_TYPE = max(compute, state) = FP32.
    
    For FP16 compute + FP32 state:
    - Hyperparameters arrive in FP16 (COMPUTE_T), widen to FP32 for EMA
    - Gradients arrive in FP16, widen to FP32 for EMA
    - Moments (m1, m2) are FP32, EMA in FP32, store as FP32
    - Bias correction in FP16 (compute type)
    - Final param update in FP32 (accumulative)
    """
    # Widen FP16 grads to FP32 for EMA (ACCUM_TYPE = FP32)
    g_accum = grads_f16.astype(np.float32)
    # Hyperparameters are COMPUTE_T (FP16), then widened to ACCUM_T (FP32)
    beta1_accum = np.float32(np.float16(beta1))
    beta2_accum = np.float32(np.float16(beta2))
    
    # EMA in FP32 (state precision preserved)
    m1_new = beta1_accum * m1 + (np.float32(1.0) - beta1_accum) * g_accum
    m2_new = beta2_accum * m2 + (np.float32(1.0) - beta2_accum) * (g_accum * g_accum)
    
    # Bias correction in FP16 (compute type) — narrow from accum
    # Use errstate to suppress expected edge-case warnings (e.g., t=1 → divide by zero)
    with np.errstate(divide="ignore", invalid="ignore"):
        m_hat_f16 = (m1_new.astype(np.float16) / 
                     np.float16(1.0 - np.float16(beta1_pow_t)))
        v_hat_f16 = (m2_new.astype(np.float16) / 
                     np.float16(1.0 - np.float16(beta2_pow_t)))
        
        # Parameter delta in FP16
        delta_f16 = (np.float16(lr) * m_hat_f16 / 
                     (np.sqrt(v_hat_f16.astype(np.float32)).astype(np.float16) + np.float16(epsilon)))
        
        # Final subtraction in FP32 (accumulative)
        params_new = params - delta_f16.astype(np.float32)
    
    return params_new, m1_new, m2_new


def _ref_adam_update_buggy_accum(
    params: np.ndarray,
    grads_f16: np.ndarray,
    m1: np.ndarray,
    m2: np.ndarray,
    lr: float,
    beta1: float,
    beta2: float,
    epsilon: float,
    beta1_pow_t: float,
    beta2_pow_t: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Reference Adam with BUGGY ACCUM_TYPE = COMPUTE_TYPE = FP16.
    
    This simulates the bug where load_state_for_accum() narrows FP32 → FP16.
    """
    # BUG: narrow FP32 moments to FP16 for EMA (wrong ACCUM_TYPE)
    m1_f16 = m1.astype(np.float16)
    m2_f16 = m2.astype(np.float16)
    
    g_f16 = grads_f16.astype(np.float16)
    beta1_f16 = np.float16(beta1)
    beta2_f16 = np.float16(beta2)
    
    # EMA in FP16 (precision destroyed)
    m1_new_f16 = beta1_f16 * m1_f16 + (np.float16(1.0) - beta1_f16) * g_f16
    m2_new_f16 = beta2_f16 * m2_f16 + (np.float16(1.0) - beta2_f16) * (g_f16 * g_f16)
    
    # Store back as FP32 (widening, but damage is done)
    m1_new = m1_new_f16.astype(np.float32)
    m2_new = m2_new_f16.astype(np.float32)
    
    # Bias correction in FP16
    # Use errstate to suppress expected edge-case warnings (e.g., t=1 → divide by zero)
    with np.errstate(divide="ignore", invalid="ignore"):
        m_hat = m1_new_f16 / np.float16(1.0 - np.float16(beta1_pow_t))
        v_hat = m2_new_f16 / np.float16(1.0 - np.float16(beta2_pow_t))
        
        # Parameter delta
        delta = (np.float16(lr) * m_hat / 
                 (np.sqrt(v_hat.astype(np.float32)).astype(np.float16) + np.float16(epsilon)))
        
        # Final subtraction
        params_new = params - delta.astype(np.float32)
    
    return params_new, m1_new, m2_new


@pytest.mark.skipif(not HAS_FP16_COMPUTE_FP32_STATE, reason="s16c16x32 variant not available")
class TestAdamAccumTypePrecision:
    """Validates ACCUM_TYPE = max(COMPUTE_TYPE, STATE_TYPE) for FP16c/FP32s."""
    
    # Precision suffix for FP16 storage, FP16 compute, FP32 state.
    # This configuration triggers the ACCUM_TYPE derivation bug:
    # - STATE_TYPE (float) > COMPUTE_TYPE (half)  
    # - Correct ACCUM_TYPE = float (FP32)
    # - Buggy ACCUM_TYPE = half (FP16, same as COMPUTE_TYPE)
    SUFFIX = "s16c16x32"
    
    @pytest.fixture
    def structs(self):
        """Get precision-specific struct types."""
        from src.backends.cpu._ffi_types import PRECISION_STRUCTS
        return PRECISION_STRUCTS[self.SUFFIX]
    
    @pytest.fixture
    def adam_fn_ptr(self, cpu_lib):
        """Get the adam_update task function pointer."""
        return ctypes.cast(
            getattr(cpu_lib, f"task_adam_update_{self.SUFFIX}"),
            ctypes.c_void_p
        )
    
    def test_adam_update_preserves_fp32_state_precision(
        self, cpu_lib, thread_pool, structs, adam_fn_ptr
    ):
        """Native kernel must preserve FP32 moment precision during FP16-compute EMA.
        
        This test will FAIL if the ACCUM_TYPE derivation bug exists, because:
        - The buggy kernel narrows FP32 moments to FP16 during load
        - EMA is computed in FP16 (losing precision)
        - Results differ significantly from correct FP32-EMA reference
        """
        n = 64
        rng = np.random.default_rng(42)
        
        # Initialize with values that expose precision differences
        # Use values that are representable in FP16 but where EMA
        # accumulation benefits from FP32 intermediate precision
        params = (rng.standard_normal(n) * 0.1).astype(np.float32)
        grads_f32 = (rng.standard_normal(n) * 0.01).astype(np.float32)
        grads_f16 = grads_f32.astype(np.float16)
        
        # Non-zero initial moments to test precision preservation
        m1 = (rng.standard_normal(n) * 0.001).astype(np.float32)
        m2 = (np.abs(rng.standard_normal(n)) * 0.0001).astype(np.float32)
        
        # Adam hyperparameters
        lr, beta1, beta2, epsilon = 0.001, 0.9, 0.999, 1e-7
        t = 10  # non-trivial step count
        beta1_pow_t = float(np.float64(beta1) ** t)
        beta2_pow_t = float(np.float64(beta2) ** t)
        
        # Compute correct reference (FP32 accumulation)
        ref_params, ref_m1, ref_m2 = _ref_adam_update_correct_accum(
            params.copy(), grads_f16, m1.copy(), m2.copy(),
            lr, beta1, beta2, epsilon, beta1_pow_t, beta2_pow_t
        )
        
        # Compute buggy reference (FP16 accumulation) for comparison
        buggy_params, buggy_m1, buggy_m2 = _ref_adam_update_buggy_accum(
            params.copy(), grads_f16, m1.copy(), m2.copy(),
            lr, beta1, beta2, epsilon, beta1_pow_t, beta2_pow_t
        )
        
        # Prepare buffers for native kernel
        # final_grad is COMPUTE_TYPE (FP16) — view as uint16 for FFI
        final_grad_buf = np.ascontiguousarray(grads_f16.view(np.uint16))
        params_buf = np.ascontiguousarray(params.copy())
        m1_buf = np.ascontiguousarray(m1.copy())
        m2_buf = np.ascontiguousarray(m2.copy())
        
        # Construct args struct
        AdamUpdateArgs = structs["AdamUpdateArgs"]
        args = AdamUpdateArgs()
        args.final_grad = final_grad_buf.ctypes.data_as(ctypes.POINTER(ctypes.c_uint16))
        args.parameters = params_buf.ctypes.data_as(ctypes.POINTER(ctypes.c_float))
        args.m1 = m1_buf.ctypes.data_as(ctypes.POINTER(ctypes.c_float))
        args.m2 = m2_buf.ctypes.data_as(ctypes.POINTER(ctypes.c_float))
        # Hyperparams are COMPUTE_TYPE (FP16) — pass as uint16 bit pattern
        args.learning_rate = _float32_to_float16_bits(lr)
        args.beta1_pow_t = _float32_to_float16_bits(beta1_pow_t)
        args.beta2_pow_t = _float32_to_float16_bits(beta2_pow_t)
        args.beta1 = _float32_to_float16_bits(beta1)
        args.beta2 = _float32_to_float16_bits(beta2)
        args.epsilon = _float32_to_float16_bits(epsilon)
        args.parameter_count = n
        
        # Dispatch kernel
        cpu_lib.pool_dispatch_and_wait(
            thread_pool,
            adam_fn_ptr,
            ctypes.byref(args),
            n
        )
        
        # Verify results match CORRECT reference (FP32 accumulation)
        # If the kernel has the bug, it will match the BUGGY reference instead.
        
        # Check m1 (first moment) — most sensitive to precision
        m1_vs_correct = np.max(np.abs(m1_buf - ref_m1))
        m1_vs_buggy = np.max(np.abs(m1_buf - buggy_m1))
        
        # Check m2 (second moment)
        m2_vs_correct = np.max(np.abs(m2_buf - ref_m2))
        m2_vs_buggy = np.max(np.abs(m2_buf - buggy_m2))
        
        # If the kernel is correct, output should be closer to correct reference.
        # If the kernel is buggy, output will be closer to (or equal to) the buggy reference.
        # 
        # We use a diagnostic assertion: the kernel must be SIGNIFICANTLY closer
        # to the correct reference than the buggy one. If the bug exists, the
        # kernel output will match the buggy reference almost exactly.
        
        if m1_vs_buggy < m1_vs_correct:
            pytest.fail(
                f"ACCUM_TYPE derivation bug detected!\n"
                f"Kernel m1 is closer to BUGGY (FP16-accum) reference:\n"
                f"  Distance to correct (FP32-accum): {m1_vs_correct:.2e}\n"
                f"  Distance to buggy (FP16-accum):   {m1_vs_buggy:.2e}\n"
                f"The kernel is narrowing FP32 state moments to FP16 during EMA."
            )
        
        if m2_vs_buggy < m2_vs_correct:
            pytest.fail(
                f"ACCUM_TYPE derivation bug detected!\n"
                f"Kernel m2 is closer to BUGGY (FP16-accum) reference:\n"
                f"  Distance to correct (FP32-accum): {m2_vs_correct:.2e}\n"
                f"  Distance to buggy (FP16-accum):   {m2_vs_buggy:.2e}\n"
                f"The kernel is narrowing FP32 state moments to FP16 during EMA."
            )
        
        # Also verify reasonable absolute tolerance from correct reference
        # FP32 mantissa should preserve ~7 decimal digits
        np.testing.assert_allclose(
            m1_buf, ref_m1, 
            rtol=1e-5, atol=1e-8,
            err_msg="m1 diverged from correct FP32-accumulation reference"
        )
        np.testing.assert_allclose(
            m2_buf, ref_m2, 
            rtol=1e-5, atol=1e-8,
            err_msg="m2 diverged from correct FP32-accumulation reference"
        )
    
    def test_multi_step_precision_accumulation(
        self, cpu_lib, thread_pool, structs, adam_fn_ptr
    ):
        """Multiple Adam steps: precision errors compound if ACCUM_TYPE is wrong.
        
        With BUGGY FP16 accumulation, errors grow faster over many steps.
        With CORRECT FP32 accumulation, FP32 precision is maintained.
        """
        n = 32
        num_steps = 100
        rng = np.random.default_rng(123)
        
        # Initialize
        params = (rng.standard_normal(n) * 0.1).astype(np.float32)
        m1 = np.zeros(n, dtype=np.float32)
        m2 = np.zeros(n, dtype=np.float32)
        
        # Reference copies
        ref_params, ref_m1, ref_m2 = params.copy(), m1.copy(), m2.copy()
        
        # Kernel buffers
        kern_params = np.ascontiguousarray(params.copy())
        kern_m1 = np.ascontiguousarray(m1.copy())
        kern_m2 = np.ascontiguousarray(m2.copy())
        
        lr, beta1, beta2, epsilon = 0.001, 0.999, 0.999, 1e-8  # High β exacerbates precision issues
        
        AdamUpdateArgs = structs["AdamUpdateArgs"]
        
        for t in range(1, num_steps + 1):
            grad_f32 = (rng.standard_normal(n) * 0.01).astype(np.float32)
            grad_f16 = grad_f32.astype(np.float16)
            
            beta1_pow_t = float(np.float64(beta1) ** t)
            beta2_pow_t = float(np.float64(beta2) ** t)
            
            # Update reference (correct FP32 accumulation)
            ref_params, ref_m1, ref_m2 = _ref_adam_update_correct_accum(
                ref_params, grad_f16, ref_m1, ref_m2,
                lr, beta1, beta2, epsilon, beta1_pow_t, beta2_pow_t
            )
            
            # Update kernel
            final_grad_buf = np.ascontiguousarray(grad_f16.view(np.uint16))
            
            args = AdamUpdateArgs()
            args.final_grad = final_grad_buf.ctypes.data_as(ctypes.POINTER(ctypes.c_uint16))
            args.parameters = kern_params.ctypes.data_as(ctypes.POINTER(ctypes.c_float))
            args.m1 = kern_m1.ctypes.data_as(ctypes.POINTER(ctypes.c_float))
            args.m2 = kern_m2.ctypes.data_as(ctypes.POINTER(ctypes.c_float))
            args.learning_rate = _float32_to_float16_bits(lr)
            args.beta1_pow_t = _float32_to_float16_bits(beta1_pow_t)
            args.beta2_pow_t = _float32_to_float16_bits(beta2_pow_t)
            args.beta1 = _float32_to_float16_bits(beta1)
            args.beta2 = _float32_to_float16_bits(beta2)
            args.epsilon = _float32_to_float16_bits(epsilon)
            args.parameter_count = n
            
            cpu_lib.pool_dispatch_and_wait(
                thread_pool,
                adam_fn_ptr,
                ctypes.byref(args),
                n
            )
        
        # After many steps, FP32-correct accumulation should be maintained
        # FP16-buggy accumulation would show significant drift
        m1_error = np.max(np.abs(kern_m1 - ref_m1) / (np.abs(ref_m1) + 1e-10))
        m2_error = np.max(np.abs(kern_m2 - ref_m2) / (np.abs(ref_m2) + 1e-10))
        
        # With correct FP32 accumulation, relative error should stay small
        # With buggy FP16 accumulation, error would be ~1e-3 or worse
        assert m1_error < 1e-5, (
            f"m1 relative error {m1_error:.2e} after {num_steps} steps\n"
            f"Expected < 1e-5 with correct FP32 accumulation\n"
            f"ACCUM_TYPE bug would cause ~1e-3 or worse"
        )
        assert m2_error < 1e-5, (
            f"m2 relative error {m2_error:.2e} after {num_steps} steps\n"
            f"Expected < 1e-5 with correct FP32 accumulation"
        )


def _float32_to_float16_bits(val: float) -> int:
    """Convert float to FP16 bit pattern as uint16."""
    return int(np.array([val], dtype=np.float32).astype(np.float16).view(np.uint16)[0])


def _ref_adam_update_fp16_fp16(
    params: np.ndarray,
    grads_f16: np.ndarray,
    m1_f16: np.ndarray,
    m2_f16: np.ndarray,
    lr: float,
    beta1: float,
    beta2: float,
    epsilon: float,
    beta1_pow_t: float,
    beta2_pow_t: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Reference Adam for FP16 compute + FP16 state (s16c16x16 variant).
    
    All intermediate operations are in FP16:
    - Hyperparameters: FP16 (COMPUTE_T)
    - Gradients: FP16 (COMPUTE_T)
    - Moments: FP16 (STATE_T)
    - ACCUM_TYPE = max(FP16, FP16) = FP16
    - Parameters: FP16 (STATE_T for state buffers)
    """
    # All in FP16
    g_f16 = grads_f16.astype(np.float16)
    beta1_f16 = np.float16(beta1)
    beta2_f16 = np.float16(beta2)
    lr_f16 = np.float16(lr)
    eps_f16 = np.float16(epsilon)
    
    # EMA in FP16 (ACCUM_TYPE = FP16)
    m1_new = beta1_f16 * m1_f16 + (np.float16(1.0) - beta1_f16) * g_f16
    m2_new = beta2_f16 * m2_f16 + (np.float16(1.0) - beta2_f16) * (g_f16 * g_f16)
    
    # Bias correction in FP16
    m_hat = m1_new / np.float16(1.0 - np.float16(beta1_pow_t))
    v_hat = m2_new / np.float16(1.0 - np.float16(beta2_pow_t))
    
    # Parameter delta in FP16
    # sqrt computed via FP32 intermediate then back to FP16 (matches kernel behavior)
    delta = lr_f16 * m_hat / (np.sqrt(v_hat.astype(np.float32)).astype(np.float16) + eps_f16)
    
    # Final subtraction in FP16 (params are STATE_T = FP16)
    params_new = params.astype(np.float16) - delta
    
    return params_new, m1_new, m2_new


@pytest.mark.skipif(not HAS_FP16_COMPUTE_FP16_STATE, reason="s16c16x16 variant not available")
class TestAdamFP16ComputeFP16State:
    """Validates adam_update with full FP16 compute and state (s16c16x16)."""
    
    SUFFIX = "s16c16x16"
    
    @pytest.fixture
    def structs(self):
        """Get precision-specific struct types."""
        from src.backends.cpu._ffi_types import PRECISION_STRUCTS
        return PRECISION_STRUCTS[self.SUFFIX]
    
    @pytest.fixture
    def adam_fn_ptr(self, cpu_lib):
        """Get the adam_update task function pointer."""
        return ctypes.cast(
            getattr(cpu_lib, f"task_adam_update_{self.SUFFIX}"),
            ctypes.c_void_p
        )
    
    def test_adam_update_fp16_compute_fp16_state(
        self, cpu_lib, thread_pool, structs, adam_fn_ptr
    ):
        """Verify adam_update kernel works correctly with full FP16.
        
        Tests s16c16x16: FP16 storage, FP16 compute, FP16 state.
        ACCUM_TYPE = max(FP16, FP16) = FP16 — no precision widening needed.
        """
        n = 64
        rng = np.random.default_rng(42)
        
        # Use small values representable in FP16
        params = (rng.standard_normal(n) * 0.1).astype(np.float16)
        grads = (rng.standard_normal(n) * 0.01).astype(np.float16)
        
        # Initial moments (FP16 state)
        m1 = (rng.standard_normal(n) * 0.001).astype(np.float16)
        m2 = (np.abs(rng.standard_normal(n)) * 0.0001).astype(np.float16)
        
        # Adam hyperparameters (will be truncated to FP16 in kernel)
        lr, beta1, beta2, epsilon = 0.001, 0.9, 0.999, 1e-4  # larger epsilon for FP16 stability
        t = 10
        beta1_pow_t = float(np.float64(beta1) ** t)
        beta2_pow_t = float(np.float64(beta2) ** t)
        
        # Reference computation
        ref_params, ref_m1, ref_m2 = _ref_adam_update_fp16_fp16(
            params.copy(), grads, m1.copy(), m2.copy(),
            lr, beta1, beta2, epsilon, beta1_pow_t, beta2_pow_t
        )
        
        # Prepare buffers for native kernel
        # All buffers are FP16, viewed as uint16 for FFI
        grads_buf = np.ascontiguousarray(grads.view(np.uint16))
        params_buf = np.ascontiguousarray(params.view(np.uint16).copy())
        m1_buf = np.ascontiguousarray(m1.view(np.uint16).copy())
        m2_buf = np.ascontiguousarray(m2.view(np.uint16).copy())
        
        # Construct args struct
        AdamUpdateArgs = structs["AdamUpdateArgs"]
        args = AdamUpdateArgs()
        args.final_grad = grads_buf.ctypes.data_as(ctypes.POINTER(ctypes.c_uint16))
        args.parameters = params_buf.ctypes.data_as(ctypes.POINTER(ctypes.c_uint16))
        args.m1 = m1_buf.ctypes.data_as(ctypes.POINTER(ctypes.c_uint16))
        args.m2 = m2_buf.ctypes.data_as(ctypes.POINTER(ctypes.c_uint16))
        args.learning_rate = _float32_to_float16_bits(lr)
        args.beta1_pow_t = _float32_to_float16_bits(beta1_pow_t)
        args.beta2_pow_t = _float32_to_float16_bits(beta2_pow_t)
        args.beta1 = _float32_to_float16_bits(beta1)
        args.beta2 = _float32_to_float16_bits(beta2)
        args.epsilon = _float32_to_float16_bits(epsilon)
        args.parameter_count = n
        
        # Dispatch kernel
        cpu_lib.pool_dispatch_and_wait(
            thread_pool,
            adam_fn_ptr,
            ctypes.byref(args),
            n
        )
        
        # Convert kernel output back to FP16 for comparison
        kern_params = params_buf.view(np.float16)
        kern_m1 = m1_buf.view(np.float16)
        kern_m2 = m2_buf.view(np.float16)
        
        # Verify results match reference (FP16 tolerance)
        # FP16 has ~3 decimal digits of precision
        np.testing.assert_allclose(
            kern_m1, ref_m1,
            rtol=1e-2, atol=1e-4,
            err_msg="m1 diverged from FP16 reference"
        )
        np.testing.assert_allclose(
            kern_m2, ref_m2,
            rtol=1e-2, atol=1e-4,
            err_msg="m2 diverged from FP16 reference"
        )
        np.testing.assert_allclose(
            kern_params, ref_params,
            rtol=1e-2, atol=1e-4,
            err_msg="params diverged from FP16 reference"
        )
    
    def test_multi_step_fp16_accumulation(
        self, cpu_lib, thread_pool, structs, adam_fn_ptr
    ):
        """Multiple Adam steps with full FP16.
        
        FP16 has limited precision, so some drift is expected after many steps.
        This test verifies the kernel doesn't catastrophically diverge.
        """
        n = 32
        num_steps = 50  # fewer steps than FP32 test due to FP16 precision limits
        rng = np.random.default_rng(123)
        
        # Initialize (all FP16)
        params = (rng.standard_normal(n) * 0.1).astype(np.float16)
        m1 = np.zeros(n, dtype=np.float16)
        m2 = np.zeros(n, dtype=np.float16)
        
        # Reference copies
        ref_params, ref_m1, ref_m2 = params.copy(), m1.copy(), m2.copy()
        
        # Kernel buffers (as uint16)
        kern_params = np.ascontiguousarray(params.view(np.uint16).copy())
        kern_m1 = np.ascontiguousarray(m1.view(np.uint16).copy())
        kern_m2 = np.ascontiguousarray(m2.view(np.uint16).copy())
        
        # Use smaller beta for FP16 to avoid precision issues
        lr, beta1, beta2, epsilon = 0.001, 0.9, 0.99, 1e-4
        
        AdamUpdateArgs = structs["AdamUpdateArgs"]
        
        for t in range(1, num_steps + 1):
            grad = (rng.standard_normal(n) * 0.01).astype(np.float16)
            
            beta1_pow_t = float(np.float64(beta1) ** t)
            beta2_pow_t = float(np.float64(beta2) ** t)
            
            # Update reference
            ref_params, ref_m1, ref_m2 = _ref_adam_update_fp16_fp16(
                ref_params, grad, ref_m1, ref_m2,
                lr, beta1, beta2, epsilon, beta1_pow_t, beta2_pow_t
            )
            
            # Update kernel
            grad_buf = np.ascontiguousarray(grad.view(np.uint16))
            
            args = AdamUpdateArgs()
            args.final_grad = grad_buf.ctypes.data_as(ctypes.POINTER(ctypes.c_uint16))
            args.parameters = kern_params.ctypes.data_as(ctypes.POINTER(ctypes.c_uint16))
            args.m1 = kern_m1.ctypes.data_as(ctypes.POINTER(ctypes.c_uint16))
            args.m2 = kern_m2.ctypes.data_as(ctypes.POINTER(ctypes.c_uint16))
            args.learning_rate = _float32_to_float16_bits(lr)
            args.beta1_pow_t = _float32_to_float16_bits(beta1_pow_t)
            args.beta2_pow_t = _float32_to_float16_bits(beta2_pow_t)
            args.beta1 = _float32_to_float16_bits(beta1)
            args.beta2 = _float32_to_float16_bits(beta2)
            args.epsilon = _float32_to_float16_bits(epsilon)
            args.parameter_count = n
            
            cpu_lib.pool_dispatch_and_wait(
                thread_pool,
                adam_fn_ptr,
                ctypes.byref(args),
                n
            )
        
        # Convert kernel output back to FP16
        kern_m1_fp16 = kern_m1.view(np.float16)
        kern_m2_fp16 = kern_m2.view(np.float16)
        
        # After many steps, check relative error (FP16 tolerance)
        # Use masked comparison to avoid div-by-zero for near-zero values
        m1_nonzero = np.abs(ref_m1) > 1e-6
        m2_nonzero = np.abs(ref_m2) > 1e-6
        
        if np.any(m1_nonzero):
            m1_rel_err = np.max(np.abs(kern_m1_fp16[m1_nonzero] - ref_m1[m1_nonzero]) / np.abs(ref_m1[m1_nonzero]))
            assert m1_rel_err < 0.05, (
                f"m1 relative error {m1_rel_err:.2e} after {num_steps} steps\n"
                f"Expected < 5% with FP16 precision"
            )
        
        if np.any(m2_nonzero):
            m2_rel_err = np.max(np.abs(kern_m2_fp16[m2_nonzero] - ref_m2[m2_nonzero]) / np.abs(ref_m2[m2_nonzero]))
            assert m2_rel_err < 0.05, (
                f"m2 relative error {m2_rel_err:.2e} after {num_steps} steps\n"
                f"Expected < 5% with FP16 precision"
            )
