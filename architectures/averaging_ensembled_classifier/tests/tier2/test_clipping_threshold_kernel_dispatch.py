# tests/tier2/test_clipping_threshold_kernel_dispatch.py
"""Tier 2: Direct kernel dispatch tests for clipping threshold semantics.

These tests actually dispatch the clip_intermediate_grad kernel on each
backend with negative, zero, and positive thresholds, verifying that the
real kernel implementations conform to the contract (ADR-019, ADR-026):
  - threshold < 0: bypass clipping (diagnostic mode)
  - threshold = 0: clip to zero norm (zeros all gradients)
  - threshold > 0: standard L2-norm clipping

Unlike the reference-only tests in test_learn_clip_partials.py, these tests
exercise the actual compiled kernel code.
"""
from __future__ import annotations

import os
from pathlib import Path

import numpy as np
import pytest

from tests.tier2.fixtures.analytical import ref_clip_l2_norm
from tests.tier2.fixtures.data_generators import make_rng

# Architecture root for kernel sources
_ARCH_ROOT = Path(__file__).parent.parent.parent
_KERNEL_DIR = _ARCH_ROOT / "kernels"


# =============================================================================
# OpenCL Backend Tests
# =============================================================================

@pytest.mark.skipif(
    not os.environ.get("RUN_OPENCL_TESTS", "1") == "1",
    reason="OpenCL tests disabled",
)
class TestOpenCLClipThresholdSemantics:
    """Direct OpenCL kernel dispatch tests for threshold semantics."""

    @pytest.fixture(scope="class")
    def cl_env(self):
        """Set up OpenCL environment for tests."""
        cl = pytest.importorskip("pyopencl")

        try:
            ctx = cl.create_some_context(interactive=False)
            if not ctx.devices:
                pytest.skip("No OpenCL device available")
        except Exception:
            pytest.skip("OpenCL context creation failed")

        from src.backends.opencl.discovery import discover_hardware
        from src.backends.opencl.context import load_and_compile_kernels_from_path
        from src.backends.opencl.type_mapping import build_compiler_flags
        from src.shared.precision_config import PrecisionConfig

        device = ctx.devices[0]
        queue = cl.CommandQueue(ctx, device)
        hw = discover_hardware(device)
        prec = PrecisionConfig.float32()
        flags = build_compiler_flags(prec, hw, c_tile_size=16)
        program = load_and_compile_kernels_from_path(ctx, device, flags, str(_KERNEL_DIR))

        # Cache kernel to avoid RepeatedKernelRetrieval warning
        kernel = cl.Kernel(program, "clip_intermediate_grad")

        return {"cl": cl, "ctx": ctx, "queue": queue, "kernel": kernel}

    def _dispatch_clip_intermediate(self, cl_env, input_data, threshold, epsilon=1e-7):
        """Dispatch clip_intermediate_grad kernel and return result."""
        cl = cl_env["cl"]
        ctx = cl_env["ctx"]
        queue = cl_env["queue"]
        kernel = cl_env["kernel"]

        n = len(input_data)
        # Create input buffer (read-write, kernel modifies in-place)
        buf = cl.Buffer(
            ctx,
            cl.mem_flags.READ_WRITE | cl.mem_flags.COPY_HOST_PTR,
            hostbuf=input_data.astype(np.float32),
        )

        # Local memory for reduction (256 work items * 4 bytes)
        local_mem = cl.LocalMemory(256 * 4)

        kernel.set_args(
            local_mem,
            buf,
            np.float32(threshold),
            np.float32(epsilon),
            np.uint32(n),
        )

        # Dispatch with work group size 256
        global_size = ((n + 255) // 256) * 256
        cl.enqueue_nd_range_kernel(queue, kernel, (global_size,), (256,))
        queue.finish()

        # Read back result
        output = np.empty(n, dtype=np.float32)
        cl.enqueue_copy(queue, output, buf)
        queue.finish()
        return output

    @pytest.mark.parametrize("threshold", [-1.0, -0.001, -100.0])
    def test_negative_threshold_bypasses_clipping(self, cl_env, threshold):
        """Negative threshold bypasses clipping — gradients unchanged."""
        rng = make_rng(seed=400)
        grads = rng.standard_normal(64).astype(np.float32) * 10.0

        result = self._dispatch_clip_intermediate(cl_env, grads, threshold)
        expected = ref_clip_l2_norm(grads, threshold)

        np.testing.assert_allclose(
            result, expected, atol=1e-6,
            err_msg=f"OpenCL: threshold={threshold} should bypass clipping"
        )
        # Also verify unchanged from input
        np.testing.assert_allclose(result, grads, atol=1e-6)

    def test_zero_threshold_zeros_all_gradients(self, cl_env):
        """Zero threshold clips to zero norm — all zeros."""
        rng = make_rng(seed=401)
        grads = rng.standard_normal(64).astype(np.float32) * 10.0

        result = self._dispatch_clip_intermediate(cl_env, grads, threshold=0.0)
        expected = ref_clip_l2_norm(grads, 0.0)

        np.testing.assert_allclose(
            result, expected, atol=1e-6,
            err_msg="OpenCL: threshold=0 should zero all gradients"
        )
        np.testing.assert_allclose(result, np.zeros_like(grads), atol=1e-6)

    @pytest.mark.parametrize("threshold", [0.1, 1.0, 10.0])
    def test_positive_threshold_clips_normally(self, cl_env, threshold):
        """Positive threshold applies standard L2-norm clipping."""
        rng = make_rng(seed=402)
        # Use large gradients to ensure clipping is applied
        grads = rng.standard_normal(64).astype(np.float32) * 100.0

        result = self._dispatch_clip_intermediate(cl_env, grads, threshold)
        expected = ref_clip_l2_norm(grads, threshold)

        np.testing.assert_allclose(result, expected, atol=1e-5, rtol=1e-5)

        # Verify norm is clipped
        result_norm = np.sqrt(np.sum(result * result))
        assert result_norm <= threshold + 1e-5


# =============================================================================
# CPU Backend Tests
# =============================================================================

class TestCPUClipThresholdSemantics:
    """Direct CPU kernel dispatch tests for threshold semantics."""

    @pytest.fixture(scope="class")
    def cpu_lib(self):
        """Load CPU kernel library and get FP32 struct definitions."""
        try:
            from src._build_config import BACKEND_CPU  # type: ignore[import-not-found]
            if not BACKEND_CPU:
                pytest.skip("CPU backend not available")
        except ImportError:
            pytest.skip("Build config not available")

        from src.backends.cpu._loader import load_cpu_library
        from src.backends.cpu._ffi_types import ClipIntermediateArgs

        lib = load_cpu_library()
        return {"lib": lib, "ClipIntermediateArgs": ClipIntermediateArgs}

    def _dispatch_clip_intermediate(self, cpu_lib, input_data, threshold, epsilon=1e-7):
        """Dispatch CPU task_clip_intermediate via FFI (s32c32x32 precision)."""
        import ctypes

        lib = cpu_lib["lib"]
        ClipArgs = cpu_lib["ClipIntermediateArgs"]

        # Create mutable buffer (in-place kernel)
        data = input_data.astype(np.float32).copy()
        n = len(data)

        # Build args struct
        args = ClipArgs()
        args.intermediate_grad = data.ctypes.data_as(ctypes.POINTER(ctypes.c_float))
        args.clipping_threshold_t_j = np.float32(threshold)
        args.epsilon = np.float32(epsilon)
        args.parameter_count = n

        # Dispatch single task (task_index=0, thread_id=0)
        # Suffix s32c32x32 = storage32, compute32, state32 (pure FP32)
        lib.task_clip_intermediate_s32c32x32(ctypes.byref(args), 0, 0)

        return data

    @pytest.mark.parametrize("threshold", [-1.0, -0.001, -100.0])
    def test_negative_threshold_bypasses_clipping(self, cpu_lib, threshold):
        """Negative threshold bypasses clipping — gradients unchanged."""
        rng = make_rng(seed=500)
        grads = rng.standard_normal(64).astype(np.float32) * 10.0

        result = self._dispatch_clip_intermediate(cpu_lib, grads, threshold)
        expected = ref_clip_l2_norm(grads, threshold)

        np.testing.assert_allclose(
            result, expected, atol=1e-6,
            err_msg=f"CPU: threshold={threshold} should bypass clipping"
        )
        np.testing.assert_allclose(result, grads, atol=1e-6)

    def test_zero_threshold_zeros_all_gradients(self, cpu_lib):
        """Zero threshold clips to zero norm — all zeros."""
        rng = make_rng(seed=501)
        grads = rng.standard_normal(64).astype(np.float32) * 10.0

        result = self._dispatch_clip_intermediate(cpu_lib, grads, threshold=0.0)
        expected = ref_clip_l2_norm(grads, 0.0)

        np.testing.assert_allclose(
            result, expected, atol=1e-6,
            err_msg="CPU: threshold=0 should zero all gradients"
        )
        np.testing.assert_allclose(result, np.zeros_like(grads), atol=1e-6)

    @pytest.mark.parametrize("threshold", [0.1, 1.0, 10.0])
    def test_positive_threshold_clips_normally(self, cpu_lib, threshold):
        """Positive threshold applies standard L2-norm clipping."""
        rng = make_rng(seed=502)
        grads = rng.standard_normal(64).astype(np.float32) * 100.0

        result = self._dispatch_clip_intermediate(cpu_lib, grads, threshold)
        expected = ref_clip_l2_norm(grads, threshold)

        np.testing.assert_allclose(result, expected, atol=1e-5, rtol=1e-5)

        # Verify norm constraint
        result_norm = np.sqrt(np.sum(result * result))
        assert result_norm <= threshold + 1e-5


# =============================================================================
# Cross-Backend Consistency
# =============================================================================

class TestCrossBackendClipConsistency:
    """Verify OpenCL and CPU backends produce identical results."""

    @pytest.fixture(scope="class")
    def opencl_env(self):
        """Set up OpenCL environment."""
        try:
            cl = pytest.importorskip("pyopencl")
            ctx = cl.create_some_context(interactive=False)
            if not ctx.devices:
                pytest.skip("No OpenCL devices")
        except Exception:
            pytest.skip("OpenCL not available")

        from src.backends.opencl.discovery import discover_hardware
        from src.backends.opencl.context import load_and_compile_kernels_from_path
        from src.backends.opencl.type_mapping import build_compiler_flags
        from src.shared.precision_config import PrecisionConfig

        device = ctx.devices[0]
        queue = cl.CommandQueue(ctx, device)
        hw = discover_hardware(device)
        prec = PrecisionConfig.float32()
        flags = build_compiler_flags(prec, hw, c_tile_size=16)
        program = load_and_compile_kernels_from_path(ctx, device, flags, str(_KERNEL_DIR))
        # Cache kernel to avoid RepeatedKernelRetrieval warning
        kernel = cl.Kernel(program, "clip_intermediate_grad")
        return {"cl": cl, "ctx": ctx, "queue": queue, "kernel": kernel}

    @pytest.fixture(scope="class")
    def cpu_env(self):
        """Set up CPU environment."""
        try:
            from src._build_config import BACKEND_CPU  # type: ignore[import-not-found]
            if not BACKEND_CPU:
                pytest.skip("CPU backend not available")
        except ImportError:
            pytest.skip("Build config not available")

        from src.backends.cpu._loader import load_cpu_library
        from src.backends.cpu._ffi_types import ClipIntermediateArgs

        lib = load_cpu_library()
        return {"lib": lib, "ClipIntermediateArgs": ClipIntermediateArgs}

    def _dispatch_opencl(self, env, input_data, threshold, epsilon=1e-7):
        cl = env["cl"]
        ctx = env["ctx"]
        queue = env["queue"]
        kernel = env["kernel"]

        n = len(input_data)
        buf = cl.Buffer(
            ctx, cl.mem_flags.READ_WRITE | cl.mem_flags.COPY_HOST_PTR,
            hostbuf=input_data.astype(np.float32),
        )
        local_mem = cl.LocalMemory(256 * 4)
        kernel.set_args(local_mem, buf, np.float32(threshold), np.float32(epsilon), np.uint32(n))
        global_size = ((n + 255) // 256) * 256
        cl.enqueue_nd_range_kernel(queue, kernel, (global_size,), (256,))
        queue.finish()
        output = np.empty(n, dtype=np.float32)
        cl.enqueue_copy(queue, output, buf)
        queue.finish()
        return output

    def _dispatch_cpu(self, env, input_data, threshold, epsilon=1e-7):
        import ctypes

        lib = env["lib"]
        ClipArgs = env["ClipIntermediateArgs"]

        data = input_data.astype(np.float32).copy()
        n = len(data)

        args = ClipArgs()
        args.intermediate_grad = data.ctypes.data_as(ctypes.POINTER(ctypes.c_float))
        args.clipping_threshold_t_j = np.float32(threshold)
        args.epsilon = np.float32(epsilon)
        args.parameter_count = n

        lib.task_clip_intermediate_s32c32x32(ctypes.byref(args), 0, 0)
        return data

    @pytest.mark.parametrize("threshold", [-1.0, 0.0, 1.0])
    def test_opencl_cpu_consistency(self, opencl_env, cpu_env, threshold):
        """OpenCL and CPU produce identical results for each threshold class."""
        rng = make_rng(seed=600)
        grads = rng.standard_normal(64).astype(np.float32) * 10.0

        ocl_result = self._dispatch_opencl(opencl_env, grads, threshold)
        cpu_result = self._dispatch_cpu(cpu_env, grads, threshold)

        np.testing.assert_allclose(
            ocl_result, cpu_result, atol=1e-5, rtol=1e-5,
            err_msg=f"OpenCL vs CPU mismatch for threshold={threshold}"
        )

        # Both should match reference
        expected = ref_clip_l2_norm(grads, threshold)
        np.testing.assert_allclose(ocl_result, expected, atol=1e-5, rtol=1e-5)
        np.testing.assert_allclose(cpu_result, expected, atol=1e-5, rtol=1e-5)
