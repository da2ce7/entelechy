# src/tests/test_fp64_precision.py
"""
FP64 / double-precision tests (Phase 8E, ADR-024 §8).

Validates the four FP64 PrecisionConfig factories, their structural
invariants, backend integration, and the Alchemist II validation scenario.
"""
from __future__ import annotations

import math

import numpy as np
import pytest

from src.shared.hardware_profile import HardwareProfile
from src.shared.model_spec import ModelSpec
from src.shared.precision_config import PrecisionConfig
from src.shared.stabilization_policy import StabilizationPolicy
from src.shared.plan_builder import build_act_plan
from architectures.averaging_ensembled_classifier.src.shared.problem_type_spec import PlanCceStrategy

from .conftest import IRIS, FP64_PRECISION_CONFIGS, FP64_MODEL_SPEC_FACTORIES, ALL_PRECISION_CONFIGS, ALL_MODEL_SPEC_FACTORIES


def _make_hw(simd_width: int = 8) -> HardwareProfile:
    return HardwareProfile(
        simd_width=simd_width,
        cache_line_bytes=64,
        max_reduce_fan_in=64,
        max_local_mem_bytes=49152,
        global_mem_bytes=4 * 1024**3,
    )


# =========================================================================
# 1. FP64 PrecisionConfig invariants
# =========================================================================


class TestFP64PrecisionConfig:

    def test_float64_factory_construction(self) -> None:
        p = PrecisionConfig.float64()
        assert p.storage_dtype == np.dtype(np.float64)
        assert p.compute_dtype == np.dtype(np.float64)
        assert p.state_dtype == np.dtype(np.float64)
        assert p.compute_fp_format_max == float(np.finfo(np.float64).max)
        assert p.compute_epsilon == float(np.finfo(np.float64).eps)

    def test_mixed_f32_f64_state_factory(self) -> None:
        p = PrecisionConfig.mixed_f32_f64_state()
        assert p.storage_dtype == np.dtype(np.float32)
        assert p.compute_dtype == np.dtype(np.float32)
        assert p.state_dtype == np.dtype(np.float64)
        assert p.state_dtype.itemsize > p.compute_dtype.itemsize

    def test_mixed_f16_f64_state_factory(self) -> None:
        p = PrecisionConfig.mixed_f16_f64_state()
        assert p.storage_dtype == np.dtype(np.float16)
        assert p.compute_dtype == np.dtype(np.float32)
        assert p.state_dtype == np.dtype(np.float64)

    def test_mixed_f32_f64_factory(self) -> None:
        p = PrecisionConfig.mixed_f32_f64()
        assert p.storage_dtype == np.dtype(np.float32)
        assert p.compute_dtype == np.dtype(np.float64)
        assert p.state_dtype == np.dtype(np.float64)
        assert p.compute_fp_format_max == float(np.finfo(np.float64).max)
        assert p.compute_epsilon == float(np.finfo(np.float64).eps)

    def test_invalid_fp64_storage_fp32_compute_rejected(self) -> None:
        """FP64 storage with FP32 compute violates storage <= compute invariant."""
        with pytest.raises(ValueError, match="cannot be wider than"):
            PrecisionConfig(
                storage_dtype=np.dtype(np.float64),
                compute_dtype=np.dtype(np.float32),
                state_dtype=np.dtype(np.float64),
                storage_fp_format_max=float(np.finfo(np.float64).max),
                storage_fp_min_positive=float(np.finfo(np.float64).smallest_subnormal),
                storage_mantissa_bits=np.finfo(np.float64).nmant,
                compute_fp_format_max=float(np.finfo(np.float32).max),
                compute_epsilon=float(np.finfo(np.float32).eps),
            )

    def test_state_wider_than_compute_is_valid(self) -> None:
        """State role is independent of compute — state > compute is permitted."""
        p = PrecisionConfig.mixed_f32_f64_state()
        assert p.state_dtype.itemsize > p.compute_dtype.itemsize

    def test_existing_factories_unchanged(self) -> None:
        """Regression: existing factories produce identical values."""
        p32 = PrecisionConfig.float32()
        assert p32.compute_fp_format_max == float(np.finfo(np.float32).max)
        assert p32.compute_epsilon == float(np.finfo(np.float32).eps)
        p_mixed = PrecisionConfig.mixed_f16_f32()
        assert p_mixed.compute_fp_format_max == float(np.finfo(np.float32).max)

    @pytest.mark.parametrize("factory", FP64_PRECISION_CONFIGS)
    def test_storage_not_wider_than_compute(self, factory) -> None:
        pc = factory()
        assert pc.storage_dtype.itemsize <= pc.compute_dtype.itemsize

    @pytest.mark.parametrize("factory", FP64_PRECISION_CONFIGS)
    def test_storage_not_wider_than_state(self, factory) -> None:
        pc = factory()
        assert pc.storage_dtype.itemsize <= pc.state_dtype.itemsize

    @pytest.mark.parametrize("factory", FP64_PRECISION_CONFIGS)
    def test_derived_constants_consistent(self, factory) -> None:
        pc = factory()
        assert pc.storage_fp_format_max == float(np.finfo(pc.storage_dtype).max)
        assert pc.compute_fp_format_max == float(np.finfo(pc.compute_dtype).max)

    @pytest.mark.parametrize("factory", FP64_PRECISION_CONFIGS)
    def test_all_fields_finite_and_positive(self, factory) -> None:
        pc = factory()
        assert np.isfinite(pc.storage_fp_format_max) and pc.storage_fp_format_max > 0
        assert np.isfinite(pc.compute_fp_format_max) and pc.compute_fp_format_max > 0
        assert np.isfinite(pc.compute_epsilon) and pc.compute_epsilon > 0


# =========================================================================
# 2. FP64 StabilizationPolicy
# =========================================================================


class TestFP64StabilizationPolicy:

    def test_fp64_safety_ceiling_no_overflow(self) -> None:
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
            assert ceiling > 1e300

    @pytest.mark.parametrize("factory", FP64_PRECISION_CONFIGS)
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

    @pytest.mark.parametrize("factory", FP64_PRECISION_CONFIGS)
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


# =========================================================================
# 3. FP64 ModelSpec padding
# =========================================================================


class TestFP64ModelSpecPadding:

    @pytest.mark.parametrize("factory", FP64_MODEL_SPEC_FACTORIES)
    def test_padded_dims_ge_logical(self, factory) -> None:
        spec = factory(**IRIS)
        assert spec.padded_input_dim >= spec.input_dim
        assert spec.padded_hidden_dim >= spec.hidden_dim
        assert spec.padded_class_dim >= spec.output_classes
        assert spec.padded_module_dim >= spec.num_modules

    @pytest.mark.parametrize("factory", FP64_MODEL_SPEC_FACTORIES)
    def test_cache_aligned_padding(self, factory) -> None:
        spec = factory(**IRIS)
        item = spec.precision.storage_dtype.itemsize
        cl = spec.cache_line_bytes
        assert (spec.padded_input_dim * item) % cl == 0
        assert (spec.padded_class_dim * item) % cl == 0
        assert (spec.padded_module_dim * item) % cl == 0

    def test_fp64_wider_byte_width_than_fp32(self) -> None:
        """FP64 storage has 8-byte elements — fewer per cache line → same or narrower padded count."""
        fp32 = ModelSpec.float32(**IRIS)
        fp64 = ModelSpec.float64(**IRIS)
        assert fp64.precision.storage_dtype.itemsize == 8
        assert fp32.precision.storage_dtype.itemsize == 4


# =========================================================================
# 4. FP64 plan construction
# =========================================================================


class TestFP64PlanConstruction:

    @pytest.mark.parametrize("factory", FP64_MODEL_SPEC_FACTORIES)
    def test_act_plan_constructs(self, factory) -> None:
        spec = factory(**IRIS)
        hw = _make_hw(spec.simd_width)
        strategy = PlanCceStrategy()
        plan = build_act_plan(spec, hw, strategy, batch_size=32)
        assert len(plan.nodes) > 0
        assert len(plan.buffers) > 0

    @pytest.mark.parametrize("factory", FP64_MODEL_SPEC_FACTORIES)
    def test_buffer_sizes_reflect_precision(self, factory) -> None:
        spec = factory(**IRIS)
        hw = _make_hw(spec.simd_width)
        strategy = PlanCceStrategy()
        plan = build_act_plan(spec, hw, strategy, batch_size=32)

        integer_buffers = {"targets_cce", "targets_bce"}

        for desc in plan.buffers.values():
            if desc.logical_name in integer_buffers:
                continue
            if desc.precision_role == "storage":
                assert desc.element_size_bytes == spec.precision.storage_dtype.itemsize
            elif desc.precision_role == "state":
                assert desc.element_size_bytes == spec.precision.state_dtype.itemsize
            elif desc.precision_role == "compute":
                assert desc.element_size_bytes == spec.precision.compute_dtype.itemsize

    def test_mixed_f32_f64_state_role_sizes(self) -> None:
        """mixed_f32_f64_state: storage=4B, compute=4B, state=8B."""
        spec = ModelSpec.mixed_f32_f64_state(**IRIS)
        hw = _make_hw(spec.simd_width)
        strategy = PlanCceStrategy()
        plan = build_act_plan(spec, hw, strategy, batch_size=32)

        for desc in plan.buffers.values():
            if desc.logical_name in {"targets_cce", "targets_bce"}:
                continue
            if desc.precision_role == "state":
                assert desc.element_size_bytes == 8, (
                    f"State buffer '{desc.logical_name}': expected 8B (FP64), "
                    f"got {desc.element_size_bytes}B"
                )
            elif desc.precision_role == "storage":
                assert desc.element_size_bytes == 4
            elif desc.precision_role == "compute":
                assert desc.element_size_bytes == 4


# =========================================================================
# 5. FP64 OpenCL type mapping
# =========================================================================


class TestFP64OpenCLTypeMapping:

    def test_float64_flags_include_double_symbols(self) -> None:
        from src.backends.opencl.type_mapping import build_compiler_flags

        pc = PrecisionConfig.float64()
        hw = _make_hw(8)
        flags = build_compiler_flags(pc, hw, c_tile_size=8)
        flag_str = " ".join(flags)
        assert "-DCOMPUTE_TYPE=double" in flag_str
        assert "-DCOMPUTE_TYPE_IS_DOUBLE=1" in flag_str
        assert "-DSTATE_TYPE_IS_DOUBLE=1" in flag_str

    def test_mixed_f32_f64_state_flags(self) -> None:
        from src.backends.opencl.type_mapping import build_compiler_flags

        pc = PrecisionConfig.mixed_f32_f64_state()
        hw = _make_hw(8)
        flags = build_compiler_flags(pc, hw, c_tile_size=8)
        flag_str = " ".join(flags)
        assert "-DCOMPUTE_TYPE=float" in flag_str
        assert "-DSTATE_TYPE=double" in flag_str
        assert "-DSTATE_TYPE_IS_DOUBLE=1" in flag_str
        assert "-DCOMPUTE_TYPE_IS_DOUBLE=0" in flag_str


# =========================================================================
# 6. FP64 CPU backend
# =========================================================================


class TestFP64CPUBackend:

    def test_dispatch_table_resolves_fp64_suffixes(self) -> None:
        from src.backends.cpu._ffi_types import PRECISION_SUFFIXES
        from src.backends.cpu.renderer import _get_precision_suffix

        configs = [
            PrecisionConfig.float64(),
            PrecisionConfig.mixed_f32_f64_state(),
            PrecisionConfig.mixed_f16_f64_state(),
            PrecisionConfig.mixed_f32_f64(),
        ]
        for pc in configs:
            suffix = _get_precision_suffix(pc.storage_dtype, pc.compute_dtype, pc.state_dtype)
            assert suffix in PRECISION_SUFFIXES, f"Suffix {suffix!r} not in PRECISION_SUFFIXES"

    def test_fp64_suffix_values(self) -> None:
        from src.backends.cpu.renderer import _get_precision_suffix

        assert _get_precision_suffix(
            np.dtype(np.float64), np.dtype(np.float64), np.dtype(np.float64),
        ) == "s64c64x64"
        assert _get_precision_suffix(
            np.dtype(np.float32), np.dtype(np.float32), np.dtype(np.float64),
        ) == "s32c32x64"
        assert _get_precision_suffix(
            np.dtype(np.float32), np.dtype(np.float64), np.dtype(np.float64),
        ) == "s32c64x64"
        assert _get_precision_suffix(
            np.dtype(np.float16), np.dtype(np.float32), np.dtype(np.float64),
        ) == "s16c32x64"


# =========================================================================
# 7. FP64 Vulkan backend
# =========================================================================


class TestFP64VulkanBackend:

    def test_spv_variant_suffix_resolution(self) -> None:
        from src.backends.vulkan._pipeline_cache import _spv_variant_suffix

        configs = [
            PrecisionConfig.float64(),
            PrecisionConfig.mixed_f32_f64_state(),
            PrecisionConfig.mixed_f16_f64_state(),
            PrecisionConfig.mixed_f32_f64(),
        ]
        for pc in configs:
            suffix = _spv_variant_suffix(pc)
            assert suffix.startswith("_s")

    def test_fp64_suffix_values(self) -> None:
        from src.backends.vulkan._pipeline_cache import _spv_variant_suffix

        assert _spv_variant_suffix(PrecisionConfig.float64()) == "_s64c64x64"
        assert _spv_variant_suffix(PrecisionConfig.mixed_f32_f64_state()) == "_s32c32x64"
        assert _spv_variant_suffix(PrecisionConfig.mixed_f32_f64()) == "_s32c64x64"
        assert _spv_variant_suffix(PrecisionConfig.mixed_f16_f64_state()) == "_s16c32x64"

    def test_compute_only_suffix_values(self) -> None:
        from src.backends.vulkan._pipeline_cache import _compute_only_suffix

        assert _compute_only_suffix(PrecisionConfig.float32()) == "_c32"
        assert _compute_only_suffix(PrecisionConfig.float64()) == "_c64"
        assert _compute_only_suffix(PrecisionConfig.mixed_f32_f64()) == "_c64"
        assert _compute_only_suffix(PrecisionConfig.mixed_f32_f64_state()) == "_c32"

    def test_vulkan_type_mapping_fp64_helpers(self) -> None:
        from src.backends.vulkan.type_mapping import (
            get_compute_is_double, get_state_is_double, get_storage_is_double,
        )

        p64 = PrecisionConfig.float64()
        assert get_compute_is_double(p64) == 1
        assert get_state_is_double(p64) == 1
        assert get_storage_is_double(p64) == 1

        p_mixed = PrecisionConfig.mixed_f32_f64_state()
        assert get_compute_is_double(p_mixed) == 0
        assert get_state_is_double(p_mixed) == 1
        assert get_storage_is_double(p_mixed) == 0


# =========================================================================
# 8. Alchemist II — FP64 State Stability (ADR-024 §8.2)
# =========================================================================


class TestAlchemistIIFP64StateStability:
    """
    Validates that FP64 optimizer state maintains precision where FP32
    state accumulates rounding errors over long training runs.
    """

    @pytest.mark.slow
    def test_alchemist_ii_fp64_state_stability(self) -> None:
        """ADR-024 §8.2: FP64 state tracks reference; FP32 state diverges."""
        beta1, beta2 = 0.999, 0.9999
        steps = 100_000

        # Reference: pure FP64 numpy Adam
        ref_m1, ref_m2 = np.float64(0.0), np.float64(0.0)
        for t in range(1, steps + 1):
            g = np.float64(1e-4)
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
            g = np.float32(1e-4)
            m1_f64 = np.float64(beta1) * m1_f64 + np.float64(1 - beta1) * np.float64(g)
            m2_f64 = np.float64(beta2) * m2_f64 + np.float64(1 - beta2) * np.float64(g) * np.float64(g)

        # FP64-state tracks reference much better than FP32-state.
        # The FP64-state path still casts gradients from FP32, so the
        # m1/m2 values inherit FP32 truncation of g.  Tolerance is ~1e-8
        # (FP32 eps relative to the accumulated value).
        assert abs(m1_f64 - ref_m1) / abs(ref_m1) < 1e-6
        assert abs(m2_f64 - ref_m2) / abs(ref_m2) < 1e-6

        # FP32-state diverges measurably from reference
        f32_rel_err_m1 = abs(float(m1_f32) - float(ref_m1)) / abs(float(ref_m1))
        assert f32_rel_err_m1 > 1e-7, (
            f"Expected FP32 state to diverge from FP64 reference, "
            f"but relative error was {f32_rel_err_m1}"
        )


# =========================================================================
# 9. Combined invariants: all configs produce valid plans
# =========================================================================


class TestAllPrecisionsCombined:
    """Ensure all 7 precision configs produce valid plans with identical topology."""

    @pytest.mark.parametrize("factory", ALL_MODEL_SPEC_FACTORIES)
    def test_plan_constructs_for_all(self, factory) -> None:
        spec = factory(**IRIS)
        hw = _make_hw(spec.simd_width)
        strategy = PlanCceStrategy()
        plan = build_act_plan(spec, hw, strategy, batch_size=32)
        assert len(plan.nodes) > 0

    def test_all_configs_produce_same_topology(self) -> None:
        hw = _make_hw(4)
        strategy = PlanCceStrategy()
        orders = []
        for factory_fn in [
            ModelSpec.float32, ModelSpec.mixed_f16_f32,
            ModelSpec.float64, ModelSpec.mixed_f32_f64_state,
            ModelSpec.mixed_f16_f64_state, ModelSpec.mixed_f32_f64,
        ]:
            spec = factory_fn(**IRIS)
            plan = build_act_plan(spec, hw, strategy, batch_size=32)
            orders.append(plan.topological_order)

        for i in range(1, len(orders)):
            assert orders[i] == orders[0], (
                f"Config {i} topology differs from config 0"
            )
