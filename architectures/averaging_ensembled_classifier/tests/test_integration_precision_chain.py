# tests/test_integration_precision_chain.py
from __future__ import annotations

"""
Integration Tests: End-to-End Precision Context Chain.

These tests verify the foundational type-safety contract that flows through
the entire system:

  PrecisionContext -> ModelSpec -> ParameterSpace -> MemoryLayout -> physical bytes

The architecture mandates that a precision choice made at the top level
(Float32Context vs Float16Context) must deterministically propagate to:
  • padded dimension calculations
  • memory layout byte sizes
  • scalar type consistency

Target CONCEPT.md Principle:
  - "Architectural Hierarchy" (Article 6): the precision chain is an
    implementation of the Conceptual → Contractual → Design hierarchy.

No OpenCL device is required.
"""

from typing import Any

import numpy as np
import pytest

from src.arch_primitives import Float16Context, Float32Context
from src.shared.model_spec import Float16ModelSpec, Float32ModelSpec, ModelSpec
from src.shared.parameter_space import ParameterSpace
from src.shared.workload_primitives import TilingScheme

# =========================================================================
# Helpers
# =========================================================================


def _make_spec(cls: type[ModelSpec], **overrides: Any) -> ModelSpec:
    defaults = dict(
        input_dim=4,
        hidden_dim=32,
        output_classes=3,
        num_modules=8,
        simd_width=4,
        cache_line_bytes=64,
    )
    defaults.update(overrides)
    return cls(**defaults)


def _make_tiling(spec: ModelSpec) -> TilingScheme:
    return TilingScheme(
        num_module_chunks=(spec.num_modules + 15) // 16,
        num_class_chunks=(spec.output_classes + 15) // 16,
        total_modules=spec.num_modules,
        total_classes=spec.output_classes,
    )


# =========================================================================
# 1. Precision Context Exhaustive Coverage
# =========================================================================


class TestPrecisionContextContract:
    """Verify the ABC contract for all precision contexts."""

    @pytest.mark.parametrize(
        "ctx_cls,expected_np,expected_c",
        [
            (Float32Context, np.float32, "float"),
            (Float16Context, np.float16, "half"),
        ],
    )
    def test_context_properties(self, ctx_cls: type, expected_np: type, expected_c: str) -> None:
        ctx = ctx_cls()
        assert ctx.SCALAR_NP_TYPE is expected_np
        assert ctx.SCALAR_C_TYPE_NAME == expected_c


# =========================================================================
# 2. ModelSpec Inherits Correct Precision
# =========================================================================


class TestModelSpecPrecisionInheritance:
    """Verify that ModelSpec subclasses correctly inherit precision."""

    def test_fp32_spec_has_fp32_type(self):
        spec = _make_spec(Float32ModelSpec)
        assert spec.SCALAR_NP_TYPE is np.float32
        assert spec.SCALAR_C_TYPE_NAME == "float"

    def test_fp16_spec_has_fp16_type(self):
        spec = _make_spec(Float16ModelSpec)
        assert spec.SCALAR_NP_TYPE is np.float16
        assert spec.SCALAR_C_TYPE_NAME == "half"

    def test_spec_is_frozen(self):
        spec = _make_spec(Float32ModelSpec)
        with pytest.raises(AttributeError):
            spec.input_dim = 999  # type: ignore[misc]


# =========================================================================
# 3. Precision Propagation Through Layout Pipeline
# =========================================================================


class TestPrecisionPropagation:
    """
    Verify that FP16 and FP32 produce different physical layouts for
    the same logical specification.
    """

    def test_total_byte_sizes_differ(self):
        """FP16 buffers should be ~half the byte size of FP32 buffers."""
        spec32 = _make_spec(Float32ModelSpec)
        spec16 = _make_spec(Float16ModelSpec)
        ps32 = ParameterSpace(spec=spec32)
        ps16 = ParameterSpace(spec=spec16)

        grid32 = _make_tiling(spec32)
        grid16 = _make_tiling(spec16)

        layouts32 = ps32.get_all_memory_layouts(batch_size=150, grid=grid32, num_batch_chunks=4)
        layouts16 = ps16.get_all_memory_layouts(batch_size=150, grid=grid16, num_batch_chunks=4)

        # Compare total byte sizes for canonical buffers.
        # FP16 padding doubles elements-per-cache-line, which can compensate
        # the halved element size for some buffer shapes (e.g. module_weights).
        # The contract is: FP16 never *exceeds* FP32, and at least one buffer
        # is strictly smaller.
        any_strictly_smaller = False
        for name in ["shared_weights", "module_weights", "hidden_activations"]:
            if name in layouts32 and name in layouts16:
                shape32 = layouts32[name].get_padded_shape(np.dtype(np.float32))
                shape16 = layouts16[name].get_padded_shape(np.dtype(np.float16))
                bytes32 = int(np.prod(shape32)) * 4
                bytes16 = int(np.prod(shape16)) * 2
                assert bytes16 <= bytes32, f"{name}: FP16 ({bytes16}B) > FP32 ({bytes32}B)"
                if bytes16 < bytes32:
                    any_strictly_smaller = True
        assert any_strictly_smaller, "Expected at least one buffer to be strictly smaller in FP16"

    def test_padded_dims_reflect_element_size(self):
        """Cache-line padding should produce more elements for FP16."""
        spec32 = _make_spec(Float32ModelSpec, input_dim=10)
        spec16 = _make_spec(Float16ModelSpec, input_dim=10)
        # Same cache line (64 bytes): FP32 = 64/4 = 16 elements, FP16 = 64/2 = 32 elements
        assert spec16.padded_input_dim >= spec32.padded_input_dim


# =========================================================================
# 4. Edge Cases in Precision
# =========================================================================


class TestPrecisionEdgeCases:
    """Edge cases related to precision boundaries."""

    def test_fp16_max_in_stabilization(self):
        """Verify that FP16's limited range is correctly captured."""
        fp16_max = float(np.finfo(np.float16).max)
        fp32_max = float(np.finfo(np.float32).max)
        assert fp16_max < fp32_max
        assert fp16_max == pytest.approx(65504.0, rel=1e-3)

    def test_scalar_type_consistency_through_pipeline(self):
        """The dtype used for padding must match the spec's SCALAR_NP_TYPE."""
        test_cases: list[tuple[type[ModelSpec], type]] = [
            (Float32ModelSpec, np.float32),
            (Float16ModelSpec, np.float16),
        ]
        for spec_cls, expected_dtype in test_cases:
            spec = _make_spec(spec_cls)
            ps = ParameterSpace(spec=spec)
            grid = _make_tiling(spec)
            layouts = ps.get_all_memory_layouts(batch_size=10, grid=grid, num_batch_chunks=2)

            dtype = np.dtype(spec.SCALAR_NP_TYPE)
            assert dtype == np.dtype(expected_dtype)

            # All layouts should accept this dtype without error
            for name, layout in layouts.items():
                shape = layout.get_padded_shape(dtype)
                assert all(d >= 0 for d in shape), f"Invalid shape for {name}"
