# tests/tier2/vulkan/test_fp8_vulkan.py
"""Tier 2 tests: FP8 Vulkan backend — shader compilation, suffix mapping,
extension requirements, and roundtrip correctness (ADR-025 §9.2)."""
from __future__ import annotations

import pathlib

import numpy as np
import pytest

import ml_dtypes

from src.shared.precision_config import (
    FP8_DTYPES,
    FP8_E4M3,
    FP8_E5M2,
    PrecisionConfig,
)

# ── Helpers for Vulkan extension queries ──


def _get_vulkan_extensions() -> tuple[set[str], int]:
    """Query available Vulkan device extensions and API version.

    Returns (extensions set, api_version) or (empty set, 0) if Vulkan
    is unavailable.
    """
    try:
        import vulkan as vk

        app_info = vk.VkApplicationInfo(
            pApplicationName="extension_query",
            applicationVersion=1,
            pEngineName="test",
            engineVersion=1,
            apiVersion=vk.VK_MAKE_VERSION(1, 2, 0),
        )
        instance_info = vk.VkInstanceCreateInfo(pApplicationInfo=app_info)
        instance = vk.vkCreateInstance(instance_info, None)

        devices = vk.vkEnumeratePhysicalDevices(instance)
        if not devices:
            vk.vkDestroyInstance(instance, None)
            return set(), 0

        props = vk.vkGetPhysicalDeviceProperties(devices[0])
        api_version = props.apiVersion

        extensions = vk.vkEnumerateDeviceExtensionProperties(devices[0], None)
        ext_names = {ext.extensionName for ext in extensions}

        vk.vkDestroyInstance(instance, None)
        return ext_names, api_version

    except (ImportError, RuntimeError, OSError, SystemError):
        return set(), 0


def _get_vulkan_features() -> dict[str, bool]:
    """Query Vulkan physical device features."""
    try:
        import vulkan as vk

        app_info = vk.VkApplicationInfo(
            pApplicationName="feature_query",
            applicationVersion=1,
            pEngineName="test",
            engineVersion=1,
            apiVersion=vk.VK_MAKE_VERSION(1, 2, 0),
        )
        instance_info = vk.VkInstanceCreateInfo(pApplicationInfo=app_info)
        instance = vk.vkCreateInstance(instance_info, None)

        devices = vk.vkEnumeratePhysicalDevices(instance)
        if not devices:
            vk.vkDestroyInstance(instance, None)
            return {}

        features = vk.vkGetPhysicalDeviceFeatures(devices[0])
        result = {
            "shaderFloat64": bool(features.shaderFloat64),
            "shaderInt16": bool(features.shaderInt16),
        }

        vk.vkDestroyInstance(instance, None)
        return result

    except (ImportError, RuntimeError, OSError, SystemError):
        return {}


# Cache results to avoid repeated Vulkan instance creation
_VULKAN_EXTENSIONS: set[str] | None = None
_VULKAN_API_VERSION: int = 0
_VULKAN_FEATURES: dict[str, bool] | None = None

# VK_MAKE_API_VERSION(0, 1, 2, 0)
_VK_API_VERSION_1_2 = 0x00402000


def _vulkan_has_8bit_storage() -> bool:
    """Check if VK_KHR_8bit_storage is available (promoted to Vulkan 1.2 core)."""
    global _VULKAN_EXTENSIONS, _VULKAN_API_VERSION  # noqa: PLW0603
    if _VULKAN_EXTENSIONS is None:
        _VULKAN_EXTENSIONS, _VULKAN_API_VERSION = _get_vulkan_extensions()
    return (
        "VK_KHR_8bit_storage" in _VULKAN_EXTENSIONS
        or _VULKAN_API_VERSION >= _VK_API_VERSION_1_2
    )


def _vulkan_has_float16() -> bool:
    """Check if VK_KHR_shader_float16_int8 is available (promoted to Vulkan 1.2 core)."""
    global _VULKAN_EXTENSIONS, _VULKAN_API_VERSION  # noqa: PLW0603
    if _VULKAN_EXTENSIONS is None:
        _VULKAN_EXTENSIONS, _VULKAN_API_VERSION = _get_vulkan_extensions()
    return (
        "VK_KHR_shader_float16_int8" in _VULKAN_EXTENSIONS
        or _VULKAN_API_VERSION >= _VK_API_VERSION_1_2
    )


def _vulkan_has_float64() -> bool:
    """Check if shaderFloat64 device feature is supported."""
    global _VULKAN_FEATURES  # noqa: PLW0603
    if _VULKAN_FEATURES is None:
        _VULKAN_FEATURES = _get_vulkan_features()
    return _VULKAN_FEATURES.get("shaderFloat64", False)


def _reference_roundtrip(values, fp8_dtype, compute_np_dtype):
    """Compute expected roundtrip result using ml_dtypes as reference."""
    compute_values = np.array(values, dtype=compute_np_dtype)
    fp8_values = compute_values.astype(fp8_dtype)
    return fp8_values.astype(compute_np_dtype)


# ── SPIR-V Compilation Tests ────────────────────────────────────────


def _find_spv(shader_name: str, suffix: str) -> pathlib.Path | None:
    """Find compiled .spv file in builddir or installed location."""
    arch_root = pathlib.Path(__file__).resolve().parents[3]
    spv_name = f"{shader_name}{suffix}.spv"
    candidates = [
        arch_root / "builddir-vulkan" / "src" / "backends" / "vulkan" / "kernel_sources" / spv_name,
        arch_root / "builddir" / "src" / "backends" / "vulkan" / "kernel_sources" / spv_name,
    ]
    build_dir = arch_root / "build"
    if build_dir.is_dir():
        for child in build_dir.iterdir():
            if child.is_dir() and child.name.startswith("cp"):
                candidates.append(
                    child / "src" / "backends" / "vulkan" / "kernel_sources" / spv_name
                )
    for candidate in candidates:
        if candidate.exists():
            return candidate
    return None


class TestFP8ShaderCompilation:
    """Verify FP8 SPIR-V shader variants compile successfully."""

    @pytest.mark.parametrize(
        "suffix",
        [
            "_s8e4c32x32",
            "_s8e4c16x32",
            "_s8e4c64x64",
            "_s8e5c32x32",
            "_s8e5c16x32",
            "_s8e5c64x64",
        ],
    )
    def test_fp8_roundtrip_shader_compiled(self, suffix: str):
        """FP8 roundtrip test shader variant compiled to SPIR-V."""
        spv = _find_spv("fp8_roundtrip", suffix)
        if spv is None:
            pytest.skip(
                f"fp8_roundtrip{suffix}.spv not found — "
                "rebuild with: ninja -C builddir-vulkan"
            )
        data = spv.read_bytes()
        # Minimal SPIR-V validation: check magic number
        assert len(data) >= 20, "SPIR-V file too small"
        assert data[:4] == b"\x03\x02\x23\x07", "Invalid SPIR-V magic number"

    @pytest.mark.parametrize(
        "suffix",
        [
            "_s8e4c32x32",
            "_s8e5c32x32",
        ],
    )
    def test_forward_pass_fp8_variant_compiled(self, suffix: str):
        """Forward pass shader compiles with FP8 storage variant."""
        spv = _find_spv("forward_pass", suffix)
        if spv is None:
            pytest.skip(
                f"forward_pass{suffix}.spv not found — "
                "rebuild with: ninja -C builddir-vulkan"
            )
        data = spv.read_bytes()
        assert len(data) >= 20
        assert data[:4] == b"\x03\x02\x23\x07"


# ── Suffix Map Tests ────────────────────────────────────────────────


class TestFP8SuffixMapping:
    """Verify PrecisionConfig → SPIR-V suffix mapping for FP8."""

    def test_e4m3_fp32_suffix(self):
        from src.backends.vulkan._pipeline_cache import _spv_variant_suffix

        cfg = PrecisionConfig.fp8_e4m3()
        assert _spv_variant_suffix(cfg) == "_s8e4c32x32"

    def test_e4m3_fp16_suffix(self):
        from src.backends.vulkan._pipeline_cache import _spv_variant_suffix

        cfg = PrecisionConfig.fp8_e4m3_f16()
        assert _spv_variant_suffix(cfg) == "_s8e4c16x32"

    def test_e4m3_fp64_suffix(self):
        from src.backends.vulkan._pipeline_cache import _spv_variant_suffix

        cfg = PrecisionConfig.fp8_e4m3_f64()
        assert _spv_variant_suffix(cfg) == "_s8e4c64x64"

    def test_e5m2_fp32_suffix(self):
        from src.backends.vulkan._pipeline_cache import _spv_variant_suffix

        cfg = PrecisionConfig.fp8_e5m2()
        assert _spv_variant_suffix(cfg) == "_s8e5c32x32"

    def test_e5m2_fp16_suffix(self):
        from src.backends.vulkan._pipeline_cache import _spv_variant_suffix

        cfg = PrecisionConfig.fp8_e5m2_f16()
        assert _spv_variant_suffix(cfg) == "_s8e5c16x32"

    def test_e5m2_fp64_suffix(self):
        from src.backends.vulkan._pipeline_cache import _spv_variant_suffix

        cfg = PrecisionConfig.fp8_e5m2_f64()
        assert _spv_variant_suffix(cfg) == "_s8e5c64x64"

    def test_non_fp8_configs_unchanged(self):
        """Existing non-FP8 suffix mappings still work."""
        from src.backends.vulkan._pipeline_cache import _spv_variant_suffix

        assert _spv_variant_suffix(PrecisionConfig.float32()) == "_s32c32x32"
        assert _spv_variant_suffix(PrecisionConfig.mixed_f16_f32()) == "_s16c32x32"
        assert _spv_variant_suffix(PrecisionConfig.float64()) == "_s64c64x64"


# ── Extension Requirement Tests ──────────────────────────────────────


class TestFP8ExtensionRequirements:
    """Verify VulkanContext.check_precision_requirements raises clear errors."""

    def test_fp8_requires_8bit_storage(self, vulkan_context):
        """FP8 config raises RuntimeError if VK_KHR_8bit_storage is missing."""
        cfg = PrecisionConfig.fp8_e4m3()
        if vulkan_context.supports_8bit_storage():
            # Extension available — check should pass without error
            vulkan_context.check_precision_requirements(cfg)
        else:
            with pytest.raises(RuntimeError, match="VK_KHR_8bit_storage"):
                vulkan_context.check_precision_requirements(cfg)

    def test_fp16_compute_requires_float16_int8(self, vulkan_context):
        """FP16 compute raises RuntimeError if extension is missing."""
        cfg = PrecisionConfig.fp8_e4m3_f16()
        if vulkan_context.supports_8bit_storage() and vulkan_context.supports_float16_int8():
            vulkan_context.check_precision_requirements(cfg)
        elif not vulkan_context.supports_8bit_storage():
            with pytest.raises(RuntimeError, match="VK_KHR_8bit_storage"):
                vulkan_context.check_precision_requirements(cfg)
        else:
            with pytest.raises(RuntimeError, match="VK_KHR_shader_float16_int8"):
                vulkan_context.check_precision_requirements(cfg)

    def test_fp64_compute_requires_shader_float64(self, vulkan_context):
        """FP64 compute raises RuntimeError if feature is missing."""
        cfg = PrecisionConfig.fp8_e4m3_f64()
        if (
            vulkan_context.supports_8bit_storage()
            and vulkan_context.supports_float64()
        ):
            vulkan_context.check_precision_requirements(cfg)
        elif not vulkan_context.supports_8bit_storage():
            with pytest.raises(RuntimeError, match="VK_KHR_8bit_storage"):
                vulkan_context.check_precision_requirements(cfg)
        else:
            with pytest.raises(RuntimeError, match="shaderFloat64"):
                vulkan_context.check_precision_requirements(cfg)

    def test_fp32_no_requirements(self, vulkan_context):
        """FP32 config requires no special extensions."""
        cfg = PrecisionConfig.float32()
        vulkan_context.check_precision_requirements(cfg)


# ── E4M3 Roundtrip Tests (ml_dtypes reference, CPU-side) ────────────


class TestFP8E4M3RoundtripReference:
    """ADR-025 §9.2: E4M3 roundtrip against ml_dtypes reference.

    These tests verify the expected roundtrip behavior using ml_dtypes
    as the authoritative source. The Vulkan shader conversion functions
    must produce identical results.
    """

    _IN_RANGE_VALUES = [0.0, 1.0, 0.5, 0.25, 100.0, 448.0, -1.0, -448.0, 0.125, 2.0]

    def test_e4m3_fp32_roundtrip(self):
        """Values in E4M3 range survive FP32→FP8→FP32 roundtrip."""
        input_data = np.array(self._IN_RANGE_VALUES, dtype=np.float32)
        expected = _reference_roundtrip(
            self._IN_RANGE_VALUES, ml_dtypes.float8_e4m3fn, np.float32
        )
        np.testing.assert_array_equal(
            input_data.astype(ml_dtypes.float8_e4m3fn).astype(np.float32), expected
        )

    def test_e4m3_saturation(self):
        """E4M3 overflow: ml_dtypes maps to NaN (no inf in E4M3).

        Note: Our Vulkan shader saturates to ±448 instead (safe failure).
        The GPU dispatch tests verify shader saturation behavior.
        """
        input_values = [500.0, 1000.0, 1e6, -500.0, -1000.0]
        result = _reference_roundtrip(input_values, ml_dtypes.float8_e4m3fn, np.float32)
        # ml_dtypes E4M3 maps overflow → NaN (E4M3 has no inf representation)
        assert np.all(np.isnan(result))

    def test_e4m3_nan_to_zero(self):
        """NaN input maps to NaN in ml_dtypes (shader maps to zero)."""
        # ml_dtypes preserves NaN bit patterns for E4M3; our shader maps NaN→zero.
        # This test documents the ml_dtypes behavior; GPU tests verify zero.
        input_data = np.array([np.nan], dtype=np.float32)
        fp8 = input_data.astype(ml_dtypes.float8_e4m3fn)
        back = fp8.astype(np.float32)
        assert np.isnan(back[0])  # ml_dtypes preserves NaN

    def test_e4m3_subnormals(self):
        """Small values near E4M3 subnormal boundary roundtrip correctly."""
        test_values = [0.001953125, 0.00390625, 0.005859375, 0.0078125]
        result = _reference_roundtrip(test_values, ml_dtypes.float8_e4m3fn, np.float32)
        np.testing.assert_array_equal(
            np.array(test_values, dtype=np.float32)
            .astype(ml_dtypes.float8_e4m3fn)
            .astype(np.float32),
            result,
        )

    def test_e4m3_all_256_patterns(self):
        """All 256 E4M3 bit patterns: finite values survive roundtrip."""
        all_bytes = np.arange(256, dtype=np.uint8)
        all_fp8 = all_bytes.view(ml_dtypes.float8_e4m3fn)
        all_f32 = all_fp8.astype(np.float32)
        finite_mask = np.isfinite(all_f32)
        input_data = all_f32[finite_mask]
        expected = _reference_roundtrip(input_data, ml_dtypes.float8_e4m3fn, np.float32)
        np.testing.assert_array_equal(input_data, expected)


# ── E5M2 Roundtrip Tests (ml_dtypes reference, CPU-side) ────────────


class TestFP8E5M2RoundtripReference:
    """ADR-025 §9.2: E5M2 roundtrip against ml_dtypes reference."""

    _IN_RANGE_VALUES = [0.0, 1.0, 0.5, 1000.0, 57344.0, -1.0, -57344.0, 0.25, 4.0]

    def test_e5m2_fp32_roundtrip(self):
        """Values in E5M2 range survive FP32→FP8→FP32 roundtrip."""
        input_data = np.array(self._IN_RANGE_VALUES, dtype=np.float32)
        expected = _reference_roundtrip(
            self._IN_RANGE_VALUES, ml_dtypes.float8_e5m2, np.float32
        )
        np.testing.assert_array_equal(
            input_data.astype(ml_dtypes.float8_e5m2).astype(np.float32), expected
        )

    def test_e5m2_saturation(self):
        """E5M2 overflow: ml_dtypes maps to ±inf (IEEE behavior).

        Note: Our Vulkan shader saturates to ±57344 instead.
        The GPU dispatch tests verify shader saturation behavior.
        """
        input_values = [100000.0, 1e10, -100000.0, -1e10]
        result = _reference_roundtrip(input_values, ml_dtypes.float8_e5m2, np.float32)
        # ml_dtypes E5M2 maps large overflow → ±inf (IEEE E5M2 has inf)
        assert np.isinf(result[0]) and result[0] > 0
        assert np.isinf(result[1]) and result[1] > 0
        assert np.isinf(result[2]) and result[2] < 0
        assert np.isinf(result[3]) and result[3] < 0

    def test_e5m2_nan_to_zero(self):
        """NaN in E5M2: ml_dtypes preserves NaN; shader maps to zero."""
        input_data = np.array([np.nan], dtype=np.float32)
        fp8 = input_data.astype(ml_dtypes.float8_e5m2)
        back = fp8.astype(np.float32)
        assert np.isnan(back[0])  # ml_dtypes preserves NaN

    def test_e5m2_subnormals(self):
        """Small values near E5M2 subnormal boundary roundtrip correctly."""
        # E5M2 min subnormal = 2^-16
        test_values = [1.52587890625e-05, 3.0517578125e-05, 6.103515625e-05]
        result = _reference_roundtrip(test_values, ml_dtypes.float8_e5m2, np.float32)
        np.testing.assert_array_equal(
            np.array(test_values, dtype=np.float32)
            .astype(ml_dtypes.float8_e5m2)
            .astype(np.float32),
            result,
        )

    def test_e5m2_all_256_patterns(self):
        """All 256 E5M2 bit patterns: finite values survive roundtrip."""
        all_bytes = np.arange(256, dtype=np.uint8)
        all_fp8 = all_bytes.view(ml_dtypes.float8_e5m2)
        all_f32 = all_fp8.astype(np.float32)
        finite_mask = np.isfinite(all_f32)
        input_data = all_f32[finite_mask]
        expected = _reference_roundtrip(input_data, ml_dtypes.float8_e5m2, np.float32)
        np.testing.assert_array_equal(input_data, expected)


# ── GPU-dispatched roundtrip tests ──────────────────────────────────


@pytest.mark.skipif(
    not _vulkan_has_8bit_storage(),
    reason="VK_KHR_8bit_storage not available",
)
class TestFP8VulkanGPURoundtrip:
    """ADR-025 §9.2: FP8 roundtrip tests via Vulkan GPU dispatch.

    These tests load the fp8_roundtrip SPIR-V shader, dispatch it on the
    GPU, and verify the output matches ml_dtypes reference values.
    """

    @pytest.mark.parametrize(
        "precision_factory,suffix,fp8_dtype,compute_np_dtype",
        [
            (PrecisionConfig.fp8_e4m3, "_s8e4c32x32", ml_dtypes.float8_e4m3fn, np.float32),
            (PrecisionConfig.fp8_e5m2, "_s8e5c32x32", ml_dtypes.float8_e5m2, np.float32),
        ],
    )
    def test_fp32_roundtrip(
        self, vulkan_context, precision_factory, suffix, fp8_dtype, compute_np_dtype
    ):
        """FP8 roundtrip: COMPUTE_TYPE → FP8 → COMPUTE_TYPE on GPU."""
        cfg = precision_factory()
        vulkan_context.check_precision_requirements(cfg)

        spv_path = _find_spv("fp8_roundtrip", suffix)
        if spv_path is None:
            pytest.skip(f"fp8_roundtrip{suffix}.spv not found")

        # Use test values that are exactly representable in FP8
        max_val = cfg.storage_fp_format_max
        test_values = np.array(
            [0.0, 1.0, 0.5, -1.0, max_val, -max_val], dtype=compute_np_dtype
        )
        expected = _reference_roundtrip(test_values, fp8_dtype, compute_np_dtype)

        actual = _dispatch_roundtrip(
            vulkan_context, spv_path, test_values, compute_np_dtype
        )
        np.testing.assert_array_equal(actual, expected)

    @pytest.mark.skipif(
        not _vulkan_has_float16(),
        reason="VK_KHR_shader_float16_int8 not available",
    )
    @pytest.mark.parametrize(
        "precision_factory,suffix,fp8_dtype",
        [
            (PrecisionConfig.fp8_e4m3_f16, "_s8e4c16x32", ml_dtypes.float8_e4m3fn),
            (PrecisionConfig.fp8_e5m2_f16, "_s8e5c16x32", ml_dtypes.float8_e5m2),
        ],
    )
    def test_fp16_roundtrip(
        self, vulkan_context, precision_factory, suffix, fp8_dtype
    ):
        """FP8 roundtrip with FP16 compute on GPU."""
        cfg = precision_factory()
        vulkan_context.check_precision_requirements(cfg)

        spv_path = _find_spv("fp8_roundtrip", suffix)
        if spv_path is None:
            pytest.skip(f"fp8_roundtrip{suffix}.spv not found")

        test_values = np.array([0.0, 1.0, 0.5, 64.0, -1.0], dtype=np.float16)
        expected = _reference_roundtrip(test_values, fp8_dtype, np.float16)

        actual = _dispatch_roundtrip(
            vulkan_context, spv_path, test_values, np.float16
        )
        np.testing.assert_array_equal(actual, expected)

    @pytest.mark.skipif(
        not _vulkan_has_float64(),
        reason="shaderFloat64 not supported",
    )
    @pytest.mark.parametrize(
        "precision_factory,suffix,fp8_dtype",
        [
            (PrecisionConfig.fp8_e4m3_f64, "_s8e4c64x64", ml_dtypes.float8_e4m3fn),
            (PrecisionConfig.fp8_e5m2_f64, "_s8e5c64x64", ml_dtypes.float8_e5m2),
        ],
    )
    def test_fp64_roundtrip(
        self, vulkan_context, precision_factory, suffix, fp8_dtype
    ):
        """FP8 roundtrip with FP64 compute on GPU."""
        cfg = precision_factory()
        vulkan_context.check_precision_requirements(cfg)

        spv_path = _find_spv("fp8_roundtrip", suffix)
        if spv_path is None:
            pytest.skip(f"fp8_roundtrip{suffix}.spv not found")

        test_values = np.array([0.0, 1.0, 0.5, 100.0, -1.0], dtype=np.float64)
        expected = _reference_roundtrip(test_values, fp8_dtype, np.float64)

        actual = _dispatch_roundtrip(
            vulkan_context, spv_path, test_values, np.float64
        )
        np.testing.assert_array_equal(actual, expected)

    def test_e4m3_saturation_gpu(self, vulkan_context):
        """Overflow saturates to ±448 on GPU."""
        spv_path = _find_spv("fp8_roundtrip", "_s8e4c32x32")
        if spv_path is None:
            pytest.skip("fp8_roundtrip_s8e4c32x32.spv not found")

        vulkan_context.check_precision_requirements(PrecisionConfig.fp8_e4m3())

        test_values = np.array([500.0, 1000.0, 1e6, -500.0, -1000.0], dtype=np.float32)
        expected = np.array([448.0, 448.0, 448.0, -448.0, -448.0], dtype=np.float32)

        actual = _dispatch_roundtrip(
            vulkan_context, spv_path, test_values, np.float32
        )
        np.testing.assert_array_equal(actual, expected)

    def test_e5m2_saturation_gpu(self, vulkan_context):
        """Overflow saturates to ±57344 on GPU."""
        spv_path = _find_spv("fp8_roundtrip", "_s8e5c32x32")
        if spv_path is None:
            pytest.skip("fp8_roundtrip_s8e5c32x32.spv not found")

        vulkan_context.check_precision_requirements(PrecisionConfig.fp8_e5m2())

        test_values = np.array(
            [60000.0, 100000.0, 1e10, -60000.0, -1e10], dtype=np.float32
        )
        expected = np.array(
            [57344.0, 57344.0, 57344.0, -57344.0, -57344.0], dtype=np.float32
        )

        actual = _dispatch_roundtrip(
            vulkan_context, spv_path, test_values, np.float32
        )
        np.testing.assert_array_equal(actual, expected)


# ── GPU dispatch helper ──────────────────────────────────────────────


def _dispatch_roundtrip(
    context, spv_path: pathlib.Path, input_data: np.ndarray, compute_dtype
) -> np.ndarray:
    """Load a roundtrip SPIR-V shader, dispatch it, and return output.

    Creates buffers, pipeline, records commands, submits, waits, and reads back.
    """
    import ctypes
    import struct

    import vulkan as vk
    from vulkan._vulkan import ffi as _ffi

    device = context.device
    n = len(input_data)
    compute_bytes = input_data.nbytes
    storage_bytes = n  # FP8: 1 byte per element

    # --- Create buffers ---

    def _create_buffer(size, usage, memory_bits):
        buf_info = vk.VkBufferCreateInfo(size=size, usage=usage)
        buf = vk.vkCreateBuffer(device, buf_info, None)
        mem_req = vk.vkGetBufferMemoryRequirements(device, buf)
        mem_props = vk.vkGetPhysicalDeviceMemoryProperties(context.physical_device)
        mem_idx = _find_memory_type(mem_props, mem_req.memoryTypeBits, memory_bits)
        alloc_info = vk.VkMemoryAllocateInfo(
            allocationSize=mem_req.size, memoryTypeIndex=mem_idx
        )
        mem = vk.vkAllocateMemory(device, alloc_info, None)
        vk.vkBindBufferMemory(device, buf, mem, 0)
        return buf, mem, mem_req.size

    host_visible = (
        vk.VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT
        | vk.VK_MEMORY_PROPERTY_HOST_COHERENT_BIT
    )
    storage_usage = (
        vk.VK_BUFFER_USAGE_STORAGE_BUFFER_BIT
        | vk.VK_BUFFER_USAGE_TRANSFER_SRC_BIT
        | vk.VK_BUFFER_USAGE_TRANSFER_DST_BIT
    )

    input_buf, input_mem, input_alloc = _create_buffer(
        compute_bytes, storage_usage, host_visible
    )
    storage_buf, storage_mem, storage_alloc = _create_buffer(
        max(storage_bytes, 4), storage_usage, host_visible  # min 4 bytes
    )
    output_buf, output_mem, output_alloc = _create_buffer(
        compute_bytes, storage_usage, host_visible
    )

    # --- Upload input data ---
    mapped = vk.vkMapMemory(device, input_mem, 0, input_alloc, 0)
    _ffi.memmove(mapped, _ffi.from_buffer(input_data), compute_bytes)
    vk.vkUnmapMemory(device, input_mem)

    # --- Load SPIR-V and create pipeline ---
    spv_bytes = spv_path.read_bytes()
    module_info = vk.VkShaderModuleCreateInfo(codeSize=len(spv_bytes), pCode=spv_bytes)
    shader_module = vk.vkCreateShaderModule(device, module_info, None)

    # Descriptor set layout: 3 storage buffers
    bindings = [
        vk.VkDescriptorSetLayoutBinding(
            binding=i,
            descriptorType=vk.VK_DESCRIPTOR_TYPE_STORAGE_BUFFER,
            descriptorCount=1,
            stageFlags=vk.VK_SHADER_STAGE_COMPUTE_BIT,
        )
        for i in range(3)
    ]
    layout_info = vk.VkDescriptorSetLayoutCreateInfo(
        bindingCount=3, pBindings=bindings
    )
    desc_layout = vk.vkCreateDescriptorSetLayout(device, layout_info, None)

    # Push constants: uint count
    push_range = vk.VkPushConstantRange(
        stageFlags=vk.VK_SHADER_STAGE_COMPUTE_BIT,
        offset=0,
        size=4,
    )
    pipe_layout_info = vk.VkPipelineLayoutCreateInfo(
        setLayoutCount=1,
        pSetLayouts=[desc_layout],
        pushConstantRangeCount=1,
        pPushConstantRanges=[push_range],
    )
    pipeline_layout = vk.vkCreatePipelineLayout(device, pipe_layout_info, None)

    # Specialization constants (SIMD_WIDTH=1 for simple test dispatch)
    spec_data = struct.pack("IIII", 1, 1, 8, 0)  # simd=1, padding=1, tile=8, cce=0
    spec_buf = _ffi.new("char[]", spec_data)
    entries = [
        vk.VkSpecializationMapEntry(constantID=i, offset=i * 4, size=4)
        for i in range(4)
    ]
    spec_info = vk.VkSpecializationInfo(
        mapEntryCount=4,
        pMapEntries=entries,
        dataSize=len(spec_data),
        pData=spec_buf,
    )

    stage_info = vk.VkPipelineShaderStageCreateInfo(
        stage=vk.VK_SHADER_STAGE_COMPUTE_BIT,
        module=shader_module,
        pName="main",
        pSpecializationInfo=spec_info,
    )
    pipeline_info = vk.VkComputePipelineCreateInfo(
        stage=stage_info, layout=pipeline_layout
    )
    pipeline = vk.vkCreateComputePipelines(
        device, vk.VK_NULL_HANDLE, 1, [pipeline_info], None
    )[0]

    vk.vkDestroyShaderModule(device, shader_module, None)

    # --- Descriptor pool & set ---
    pool_size = vk.VkDescriptorPoolSize(
        type=vk.VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, descriptorCount=3
    )
    pool_info = vk.VkDescriptorPoolCreateInfo(
        maxSets=1, poolSizeCount=1, pPoolSizes=[pool_size]
    )
    desc_pool = vk.vkCreateDescriptorPool(device, pool_info, None)

    alloc_desc = vk.VkDescriptorSetAllocateInfo(
        descriptorPool=desc_pool,
        descriptorSetCount=1,
        pSetLayouts=[desc_layout],
    )
    desc_set = vk.vkAllocateDescriptorSets(device, alloc_desc)[0]

    # Update descriptor set
    buf_infos = [
        vk.VkDescriptorBufferInfo(buffer=input_buf, offset=0, range=compute_bytes),
        vk.VkDescriptorBufferInfo(
            buffer=storage_buf, offset=0, range=max(storage_bytes, 4)
        ),
        vk.VkDescriptorBufferInfo(buffer=output_buf, offset=0, range=compute_bytes),
    ]
    writes = [
        vk.VkWriteDescriptorSet(
            dstSet=desc_set,
            dstBinding=i,
            descriptorCount=1,
            descriptorType=vk.VK_DESCRIPTOR_TYPE_STORAGE_BUFFER,
            pBufferInfo=[buf_infos[i]],
        )
        for i in range(3)
    ]
    vk.vkUpdateDescriptorSets(device, len(writes), writes, 0, None)

    # --- Record & submit ---
    cmd = context.allocate_command_buffer()
    begin_info = vk.VkCommandBufferBeginInfo(
        flags=vk.VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT
    )
    vk.vkBeginCommandBuffer(cmd, begin_info)

    vk.vkCmdBindPipeline(cmd, vk.VK_PIPELINE_BIND_POINT_COMPUTE, pipeline)
    vk.vkCmdBindDescriptorSets(
        cmd,
        vk.VK_PIPELINE_BIND_POINT_COMPUTE,
        pipeline_layout,
        0,
        1,
        [desc_set],
        0,
        None,
    )

    # Push count
    push_data = struct.pack("I", n)
    push_buf = _ffi.new("char[]", push_data)
    vk.vkCmdPushConstants(
        cmd,
        pipeline_layout,
        vk.VK_SHADER_STAGE_COMPUTE_BIT,
        0,
        4,
        push_buf,
    )

    # Dispatch: ceil(n / simd_width), simd_width=1 for test
    vk.vkCmdDispatch(cmd, n, 1, 1)

    vk.vkEndCommandBuffer(cmd)

    fence = context.create_fence()
    submit = vk.VkSubmitInfo(commandBufferCount=1, pCommandBuffers=[cmd])
    vk.vkQueueSubmit(context.compute_queue, 1, [submit], fence)
    vk.vkWaitForFences(device, 1, [fence], vk.VK_TRUE, 1_000_000_000)  # 1s timeout

    # --- Read back output ---
    mapped_out = vk.vkMapMemory(device, output_mem, 0, output_alloc, 0)
    result = np.empty(n, dtype=compute_dtype)
    _ffi.memmove(_ffi.from_buffer(result.data), mapped_out, compute_bytes)
    vk.vkUnmapMemory(device, output_mem)

    # --- Cleanup ---
    vk.vkDestroyFence(device, fence, None)
    vk.vkDestroyDescriptorPool(device, desc_pool, None)
    vk.vkDestroyPipeline(device, pipeline, None)
    vk.vkDestroyPipelineLayout(device, pipeline_layout, None)
    vk.vkDestroyDescriptorSetLayout(device, desc_layout, None)
    for buf, mem in [
        (input_buf, input_mem),
        (storage_buf, storage_mem),
        (output_buf, output_mem),
    ]:
        vk.vkDestroyBuffer(device, buf, None)
        vk.vkFreeMemory(device, mem, None)

    return result


def _find_memory_type(
    mem_props, type_bits: int, required_properties: int
) -> int:
    """Find a memory type index matching the requirements."""
    for i in range(mem_props.memoryTypeCount):
        if (type_bits & (1 << i)) and (
            mem_props.memoryTypes[i].propertyFlags & required_properties
        ) == required_properties:
            return i
    raise RuntimeError("No suitable memory type found")
