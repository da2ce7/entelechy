# src/tests/test_mixed_precision.py
"""
Multi-configuration precision tests (Phase 7C §15, ADR-020 §4.6).

Validates the three-role precision decomposition across all PrecisionConfig
factories: float32(), float16(), and mixed_f16_f32(). Includes the
"Alchemist" validation scenario (CONCEPT.md §Validation Scenarios).
"""
from __future__ import annotations

import numpy as np
import pytest

from src.shared.buffer_lifecycle import BufferDescriptor, BufferHandle, BufferRole
from src.shared.hardware_profile import HardwareProfile
from src.shared.model_spec import ModelSpec
from src.shared.precision_config import PrecisionConfig
from src.shared.stabilization_policy import StabilizationPolicy
from src.shared.plan_builder import build_act_plan
from src.shared.problem_type_strategy import PlanCceStrategy

from .conftest import IRIS, HYDRA, PRECISION_CONFIGS, MODEL_SPEC_FACTORIES


def _make_hw(simd_width: int = 8) -> HardwareProfile:
    return HardwareProfile(
        simd_width=simd_width,
        cache_line_bytes=64,
        max_reduce_fan_in=64,
        max_local_mem_bytes=49152,
        global_mem_bytes=4 * 1024**3,
    )


# =========================================================================
# 1. PrecisionConfig three-role invariants
# =========================================================================


class TestPrecisionConfigInvariants:
    """Verify structural invariants hold for all three factory methods."""

    @pytest.mark.parametrize("factory", PRECISION_CONFIGS)
    def test_storage_not_wider_than_compute(self, factory) -> None:
        pc = factory()
        assert pc.storage_dtype.itemsize <= pc.compute_dtype.itemsize

    @pytest.mark.parametrize("factory", PRECISION_CONFIGS)
    def test_storage_not_wider_than_state(self, factory) -> None:
        pc = factory()
        assert pc.storage_dtype.itemsize <= pc.state_dtype.itemsize

    @pytest.mark.parametrize("factory", PRECISION_CONFIGS)
    def test_derived_constants_consistent(self, factory) -> None:
        pc = factory()
        assert pc.storage_fp_format_max == float(np.finfo(pc.storage_dtype).max)
        assert pc.compute_fp_format_max == float(np.finfo(pc.compute_dtype).max)
        assert pc.compute_epsilon == float(np.finfo(pc.compute_dtype).eps)

    @pytest.mark.parametrize("factory", PRECISION_CONFIGS)
    def test_all_fields_finite_and_positive(self, factory) -> None:
        pc = factory()
        assert np.isfinite(pc.storage_fp_format_max) and pc.storage_fp_format_max > 0
        assert np.isfinite(pc.compute_fp_format_max) and pc.compute_fp_format_max > 0
        assert np.isfinite(pc.compute_epsilon) and pc.compute_epsilon > 0

    def test_float32_uniform(self) -> None:
        pc = PrecisionConfig.float32()
        assert pc.storage_dtype == pc.compute_dtype == pc.state_dtype
        assert pc.storage_dtype == np.dtype(np.float32)

    def test_float16_uniform(self) -> None:
        pc = PrecisionConfig.float16()
        assert pc.storage_dtype == pc.compute_dtype == pc.state_dtype
        assert pc.storage_dtype == np.dtype(np.float16)

    def test_mixed_f16_f32_split(self) -> None:
        pc = PrecisionConfig.mixed_f16_f32()
        assert pc.storage_dtype == np.dtype(np.float16)
        assert pc.compute_dtype == np.dtype(np.float32)
        assert pc.state_dtype == np.dtype(np.float32)

    def test_invalid_config_rejected(self) -> None:
        """storage wider than compute must be rejected."""
        with pytest.raises(AssertionError):
            PrecisionConfig(
                storage_dtype=np.dtype(np.float32),
                compute_dtype=np.dtype(np.float16),
                state_dtype=np.dtype(np.float32),
                storage_fp_format_max=float(np.finfo(np.float32).max),
                compute_fp_format_max=float(np.finfo(np.float16).max),
                compute_epsilon=float(np.finfo(np.float16).eps),
            )


# =========================================================================
# 2. ModelSpec padding under all precisions
# =========================================================================


class TestModelSpecMultiPrecision:
    """Padding calculations must be correct for all three precision configs."""

    @pytest.mark.parametrize("factory", MODEL_SPEC_FACTORIES)
    def test_padded_dims_ge_logical(self, factory) -> None:
        spec = factory(**IRIS)
        assert spec.padded_input_dim >= spec.input_dim
        assert spec.padded_hidden_dim >= spec.hidden_dim
        assert spec.padded_class_dim >= spec.output_classes
        assert spec.padded_module_dim >= spec.num_modules

    @pytest.mark.parametrize("factory", MODEL_SPEC_FACTORIES)
    def test_cache_aligned_padding(self, factory) -> None:
        spec = factory(**IRIS)
        item = spec.precision.storage_dtype.itemsize
        cl = spec.cache_line_bytes
        assert (spec.padded_input_dim * item) % cl == 0
        assert (spec.padded_class_dim * item) % cl == 0
        assert (spec.padded_module_dim * item) % cl == 0

    def test_fp16_wider_element_count_than_fp32(self) -> None:
        """FP16 storage needs more elements per cache line → wider padded dims."""
        fp32 = ModelSpec.float32(**IRIS)
        fp16 = ModelSpec.float16(**IRIS)
        # 2-byte elements need 32 per 64-byte line vs 16 for 4-byte elements
        assert fp16.padded_input_dim >= fp32.padded_input_dim

    def test_mixed_uses_storage_dtype_for_padding(self) -> None:
        """mixed_f16_f32 pads using storage_dtype (FP16), same as float16."""
        fp16 = ModelSpec.float16(**IRIS)
        mixed = ModelSpec.mixed_f16_f32(**IRIS)
        assert mixed.padded_input_dim == fp16.padded_input_dim
        assert mixed.padded_class_dim == fp16.padded_class_dim
        assert mixed.padded_module_dim == fp16.padded_module_dim


# =========================================================================
# 3. StabilizationPolicy under all precisions
# =========================================================================


class TestStabilizationPolicyMultiPrecision:
    """Safety ceiling must derive from compute_fp_format_max, not storage."""

    @pytest.mark.parametrize("factory", PRECISION_CONFIGS)
    def test_leaf_safety_positive_and_finite(self, factory) -> None:
        pc = factory()
        policy = StabilizationPolicy(
            t_algorithmic=1.0,
            lambda_=1.0,
            compute_fp_format_max=pc.compute_fp_format_max,
        )
        t = policy.get_leaf_safety_threshold()
        assert t > 0
        assert np.isfinite(t)

    @pytest.mark.parametrize("factory", PRECISION_CONFIGS)
    def test_reduction_tree_safe_k(self, factory) -> None:
        pc = factory()
        policy = StabilizationPolicy(
            t_algorithmic=1.0,
            lambda_=1.0,
            compute_fp_format_max=pc.compute_fp_format_max,
        )
        safe_k, stages = policy.plan_uniform_reduction_tree(
            num_partials=64, hardware_max_fan_in=128,
        )
        assert safe_k >= 2
        assert safe_k ** stages >= 64

    def test_mixed_uses_fp32_safety_ceiling(self) -> None:
        """mixed_f16_f32: safety ceiling = FP32 max, not FP16 max."""
        mixed = PrecisionConfig.mixed_f16_f32()
        fp32 = PrecisionConfig.float32()
        assert mixed.compute_fp_format_max == fp32.compute_fp_format_max
        assert mixed.compute_fp_format_max > mixed.storage_fp_format_max


# =========================================================================
# 4. Plan construction under all precisions
# =========================================================================


class TestPlanMultiPrecision:
    """Plan builder must produce valid plans for all precision configs."""

    @pytest.mark.parametrize("factory", MODEL_SPEC_FACTORIES)
    def test_act_plan_constructs(self, factory) -> None:
        spec = factory(**IRIS)
        hw = _make_hw(spec.simd_width)
        strategy = PlanCceStrategy()
        plan = build_act_plan(spec, hw, strategy, batch_size=32)
        assert len(plan.nodes) > 0
        assert len(plan.buffers) > 0

    @pytest.mark.parametrize("factory", MODEL_SPEC_FACTORIES)
    def test_buffer_sizes_reflect_precision(self, factory) -> None:
        """Storage-role buffers must use storage_dtype element size."""
        spec = factory(**IRIS)
        hw = _make_hw(spec.simd_width)
        strategy = PlanCceStrategy()
        plan = build_act_plan(spec, hw, strategy, batch_size=32)

        # Target buffers are integer indices (always 4B), not FP data.
        integer_buffers = {"targets_cce", "targets_bce"}

        for desc in plan.buffers.values():
            if desc.logical_name in integer_buffers:
                continue
            if desc.precision_role == "storage":
                assert desc.element_size_bytes == spec.precision.storage_dtype.itemsize, (
                    f"Buffer '{desc.logical_name}': expected storage element size "
                    f"{spec.precision.storage_dtype.itemsize}, got {desc.element_size_bytes}"
                )
            elif desc.precision_role == "state":
                assert desc.element_size_bytes == spec.precision.state_dtype.itemsize, (
                    f"Buffer '{desc.logical_name}': expected state element size "
                    f"{spec.precision.state_dtype.itemsize}, got {desc.element_size_bytes}"
                )
            elif desc.precision_role == "compute":
                assert desc.element_size_bytes == spec.precision.compute_dtype.itemsize, (
                    f"Buffer '{desc.logical_name}': expected compute element size "
                    f"{spec.precision.compute_dtype.itemsize}, got {desc.element_size_bytes}"
                )


# =========================================================================
# 5. The Alchemist — Mixed-Precision Fidelity Validation (ADR-020 §2.6)
# =========================================================================


class TestAlchemistMixedPrecisionFidelity:
    """
    Scenario: The Alchemist (CONCEPT.md §Validation Scenarios).

    Validates that the mixed configuration achieves narrow-storage bandwidth
    without compromising compute-precision fidelity:

    1. Storage buffer allocation sizes in the mixed plan are half those of FP32.
    2. The stabilization policy's safety ceiling derives from compute_fp_format_max
       (FP32 max ≈ 3.4e38), not from storage_fp_format_max (FP16 max ≈ 65504).
    3. Compute-role buffers remain at FP32 size regardless of storage precision.
    """

    def _build_plan(self, spec: ModelSpec):
        hw = _make_hw(spec.simd_width)
        strategy = PlanCceStrategy()
        return build_act_plan(spec, hw, strategy, batch_size=32)

    def test_storage_buffers_half_size(self) -> None:
        """Mixed-precision storage buffers must be half the FP32 size."""
        fp32_plan = self._build_plan(ModelSpec.float32(**IRIS))
        mixed_plan = self._build_plan(ModelSpec.mixed_f16_f32(**IRIS))

        fp32_storage = {
            d.logical_name: d
            for d in fp32_plan.buffers.values()
            if d.precision_role == "storage"
        }
        mixed_storage = {
            d.logical_name: d
            for d in mixed_plan.buffers.values()
            if d.precision_role == "storage"
        }

        # Every storage-role buffer present in both plans should be half the
        # byte size in the mixed plan (FP16 = 2B vs FP32 = 4B per element).
        common_names = set(fp32_storage) & set(mixed_storage)
        assert len(common_names) > 0, "No common storage buffers found"

        for name in sorted(common_names):
            fp32_desc = fp32_storage[name]
            mixed_desc = mixed_storage[name]
            # Element counts may differ (FP16 padding ≥ FP32 padding), but
            # element_size_bytes must be exactly half.
            assert mixed_desc.element_size_bytes == fp32_desc.element_size_bytes // 2, (
                f"Buffer '{name}': mixed element_size={mixed_desc.element_size_bytes}, "
                f"fp32 element_size={fp32_desc.element_size_bytes}"
            )

    def test_safety_ceiling_is_fp32_not_fp16(self) -> None:
        """Mixed precision must use FP32 compute ceiling, not FP16 storage ceiling."""
        mixed = PrecisionConfig.mixed_f16_f32()
        policy = StabilizationPolicy(
            t_algorithmic=1.0,
            lambda_=1.0,
            compute_fp_format_max=mixed.compute_fp_format_max,
        )
        # FP32 max ≈ 3.4e38; FP16 max ≈ 65504
        assert policy.compute_fp_format_max > 1e10, (
            f"Safety ceiling {policy.compute_fp_format_max} looks like FP16, not FP32"
        )
        assert policy.compute_fp_format_max == float(np.finfo(np.float32).max)

    def test_compute_buffers_remain_fp32(self) -> None:
        """Compute-role buffers must be FP32 in the mixed configuration."""
        mixed_plan = self._build_plan(ModelSpec.mixed_f16_f32(**IRIS))
        for desc in mixed_plan.buffers.values():
            if desc.precision_role == "compute":
                assert desc.element_size_bytes == 4, (
                    f"Compute buffer '{desc.logical_name}': expected 4B (FP32), "
                    f"got {desc.element_size_bytes}B"
                )

    def test_state_buffers_remain_fp32(self) -> None:
        """State-role buffers must be FP32 in the mixed configuration."""
        mixed_plan = self._build_plan(ModelSpec.mixed_f16_f32(**IRIS))
        for desc in mixed_plan.buffers.values():
            if desc.precision_role == "state":
                assert desc.element_size_bytes == 4, (
                    f"State buffer '{desc.logical_name}': expected 4B (FP32), "
                    f"got {desc.element_size_bytes}B"
                )

    def test_three_configs_are_parameterizations_not_modes(self) -> None:
        """All three configs produce plans with identical DAG topology."""
        fp32_plan = self._build_plan(ModelSpec.float32(**IRIS))
        fp16_plan = self._build_plan(ModelSpec.float16(**IRIS))
        mixed_plan = self._build_plan(ModelSpec.mixed_f16_f32(**IRIS))

        assert fp32_plan.topological_order == fp16_plan.topological_order == mixed_plan.topological_order, (
            "Plan topology must be identical across precision configurations"
        )

    def test_mixed_reduction_tree_matches_fp32(self) -> None:
        """Mixed precision uses FP32 compute, so reduction tree planning should
        match the FP32 baseline (same compute_fp_format_max → same safe_k)."""
        fp32 = PrecisionConfig.float32()
        mixed = PrecisionConfig.mixed_f16_f32()

        fp32_policy = StabilizationPolicy(
            t_algorithmic=1.0, lambda_=1.0,
            compute_fp_format_max=fp32.compute_fp_format_max,
        )
        mixed_policy = StabilizationPolicy(
            t_algorithmic=1.0, lambda_=1.0,
            compute_fp_format_max=mixed.compute_fp_format_max,
        )

        k32, s32 = fp32_policy.plan_uniform_reduction_tree(64, 128)
        km, sm = mixed_policy.plan_uniform_reduction_tree(64, 128)
        assert k32 == km, f"FP32 safe_k={k32} != mixed safe_k={km}"
        assert s32 == sm, f"FP32 stages={s32} != mixed stages={sm}"


# =========================================================================
# 6. Buffer plumbing under all precisions
# =========================================================================


class TestBufferPlumbingMultiPrecision:
    """Buffer naming and shape consistency must hold across all configs."""

    @pytest.mark.parametrize("factory", MODEL_SPEC_FACTORIES)
    def test_all_recipe_buffers_exist(self, factory) -> None:
        from src.shared.parameter_space import ParameterSpace
        from src.shared.workload_primitives import TilingScheme

        spec = factory(**IRIS)
        ps = ParameterSpace(spec=spec)
        grid = TilingScheme(
            num_module_chunks=(spec.num_modules + 15) // 16,
            num_class_chunks=(spec.output_classes + 15) // 16,
            total_modules=spec.num_modules,
            total_classes=spec.output_classes,
        )
        layouts = ps.get_all_memory_layouts(
            batch_size=150, grid=grid, num_batch_chunks=4,
        )

        core_buffers = [
            "input", "sample_mask", "shared_weights", "shared_biases",
            "hidden_activations", "hidden_mask",
            "module_weights", "module_biases", "logits", "temperatures",
            "partial_probs", "final_probs",
            "final_loss", "partial_loss",
            "targets_cce", "targets_bce",
        ]
        missing = [n for n in core_buffers if n not in layouts]
        assert not missing, f"Missing buffers: {missing}"


# =========================================================================
# 7. OpenCL type mapping under all precisions
# =========================================================================


class TestOpenCLTypeMappingMultiPrecision:

    @pytest.mark.parametrize("factory", PRECISION_CONFIGS)
    def test_compiler_flags_complete(self, factory) -> None:
        from src.shared.hardware_profile import HardwareProfile
        from src.backends.opencl.type_mapping import build_compiler_flags

        pc = factory()
        hw = _make_hw(8)
        flags = build_compiler_flags(pc, hw, c_tile_size=8)

        flag_str = " ".join(flags)
        assert "-DSTORAGE_TYPE=" in flag_str
        assert "-DCOMPUTE_TYPE=" in flag_str
        assert "-DSTATE_TYPE=" in flag_str
        assert "-DSTORAGE_TYPE_IS_HALF=" in flag_str
        assert "-DCOMPUTE_TYPE_IS_HALF=" in flag_str
        assert "-DSIMD_WIDTH=8" in flag_str
        assert "-DC_TILE_SIZE=8" in flag_str
        assert "-DNUMERICAL_STABILITY_EPSILON=" in flag_str

    def test_mixed_flags_correct_types(self) -> None:
        from src.shared.hardware_profile import HardwareProfile
        from src.backends.opencl.type_mapping import build_compiler_flags

        pc = PrecisionConfig.mixed_f16_f32()
        hw = _make_hw(8)
        flags = build_compiler_flags(pc, hw, c_tile_size=8)
        flag_str = " ".join(flags)

        assert "-DSTORAGE_TYPE=half" in flag_str
        assert "-DCOMPUTE_TYPE=float" in flag_str
        assert "-DSTATE_TYPE=float" in flag_str
        assert "-DSTORAGE_TYPE_IS_HALF=1" in flag_str
        assert "-DCOMPUTE_TYPE_IS_HALF=0" in flag_str

    @pytest.mark.parametrize("factory", PRECISION_CONFIGS)
    def test_no_transitional_aliases(self, factory) -> None:
        """Transitional SCALAR_TYPE aliases must not be present (Phase 7C §14)."""
        from src.shared.hardware_profile import HardwareProfile
        from src.backends.opencl.type_mapping import build_compiler_flags

        pc = factory()
        hw = _make_hw(8)
        flags = build_compiler_flags(pc, hw, c_tile_size=8)
        flag_str = " ".join(flags)
        assert "SCALAR_TYPE" not in flag_str
        assert "SCALAR_IS_HALF" not in flag_str
        assert "SCALAR_ZERO" not in flag_str


# =========================================================================
# 8. CPU backend type mapping under all precisions
# =========================================================================


class TestCPUTypeMappingMultiPrecision:

    @pytest.mark.parametrize("factory", PRECISION_CONFIGS)
    def test_role_specific_dtypes(self, factory) -> None:
        from src.backends.cpu.type_mapping import (
            get_storage_dtype, get_compute_dtype, get_state_dtype,
        )

        pc = factory()
        assert get_storage_dtype(pc) == pc.storage_dtype
        assert get_compute_dtype(pc) == pc.compute_dtype
        assert get_state_dtype(pc) == pc.state_dtype


# =========================================================================
# 9. Vulkan backend type mapping under all precisions
# =========================================================================


class TestVulkanTypeMappingMultiPrecision:

    @pytest.mark.parametrize("factory", PRECISION_CONFIGS)
    def test_role_specific_dtypes(self, factory) -> None:
        from src.backends.vulkan.type_mapping import (
            get_storage_dtype, get_compute_dtype, get_state_dtype,
            get_storage_element_size, get_storage_is_half,
        )

        pc = factory()
        assert get_storage_dtype(pc) == pc.storage_dtype
        assert get_compute_dtype(pc) == pc.compute_dtype
        assert get_state_dtype(pc) == pc.state_dtype
        assert get_storage_element_size(pc) == pc.storage_dtype.itemsize
        expected_is_half = 1 if pc.storage_dtype == np.dtype(np.float16) else 0
        assert get_storage_is_half(pc) == expected_is_half


# =========================================================================
# 10. Vulkan pipeline cache variant selection
# =========================================================================


class TestVulkanPipelineVariantSelection:

    def test_fp32_selects_fp32_variant(self) -> None:
        from src.backends.vulkan._pipeline_cache import _spv_variant_suffix

        assert _spv_variant_suffix(PrecisionConfig.float32()) == "_fp32"

    def test_mixed_selects_s16fp32_variant(self) -> None:
        from src.backends.vulkan._pipeline_cache import _spv_variant_suffix

        assert _spv_variant_suffix(PrecisionConfig.mixed_f16_f32()) == "_s16fp32"

    def test_fp16_selects_fp16_variant(self) -> None:
        from src.backends.vulkan._pipeline_cache import _spv_variant_suffix

        assert _spv_variant_suffix(PrecisionConfig.float16()) == "_fp16"

    def test_fp32_and_mixed_select_different_variants(self) -> None:
        from src.backends.vulkan._pipeline_cache import _spv_variant_suffix

        fp32_suffix = _spv_variant_suffix(PrecisionConfig.float32())
        mixed_suffix = _spv_variant_suffix(PrecisionConfig.mixed_f16_f32())
        assert fp32_suffix != mixed_suffix


# =========================================================================
# 11. CPU FFI dispatch suffix selection
# =========================================================================


class TestCPUDispatchSuffix:

    @pytest.mark.parametrize("factory,expected_suffix", [
        (PrecisionConfig.float32, "s32x32"),
        (PrecisionConfig.float16, "s16x16"),
        (PrecisionConfig.mixed_f16_f32, "s16x32"),
    ])
    def test_suffix_selection(self, factory, expected_suffix) -> None:
        from src.backends.cpu._ffi_types import PRECISION_SUFFIXES

        pc = factory()
        s = pc.storage_dtype
        t = pc.state_dtype
        if s == np.dtype(np.float32) and t == np.dtype(np.float32):
            result = "s32x32"
        elif s == np.dtype(np.float16) and t == np.dtype(np.float16):
            result = "s16x16"
        elif s == np.dtype(np.float16) and t == np.dtype(np.float32):
            result = "s16x32"
        else:
            pytest.fail(f"No suffix for storage={s}, state={t}")

        assert result == expected_suffix
        assert result in PRECISION_SUFFIXES
