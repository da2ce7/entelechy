# tests/test_integration_model_memory.py
from __future__ import annotations

"""
Integration Tests: ModelSpec + ParameterSpace + MemoryLayout Pipeline.

These tests verify the end-to-end contract chain from a logical model
specification through the ParameterSpace manifest to the final, padded
physical memory layouts.  They exercise the interplay between:

  ModelSpec -> padded_*_dim properties
  ParameterSpace -> get_all_memory_layouts()
  MemoryLayout -> get_padded_shape()

No OpenCL device is required.  The tests are parameterized across precision
contexts (FP32 / FP16) and model scales (Iris, Hydra, Lexicon) to cover
the CONCEPT.md validation scenarios for graceful degradation, massive
num_heads, and massive output_classes.
"""

import numpy as np
import pytest

from src.memory_layout import MemoryLayout, PaddingStrategy, PaddingType
from src.model_spec import Float16ModelSpec, Float32ModelSpec, ModelSpec
from src.parameter_space import ParameterSpace
from src.workload_primitives import TilingScheme

# =========================================================================
# Helper: build a TilingScheme consistent with the orchestrator's logic
# =========================================================================


def _make_tiling(spec: ModelSpec) -> TilingScheme:
    return TilingScheme(
        num_module_chunks=(spec.num_modules + 15) // 16,
        num_class_chunks=(spec.output_classes + 15) // 16,
        total_modules=spec.num_modules,
        total_classes=spec.output_classes,
    )


# =========================================================================
# 1. ModelSpec Padding Contracts
# =========================================================================


class TestModelSpecPadding:
    """Verify that padded dimension properties are SIMD/cache-aligned."""

    def test_padded_hidden_dim_is_simd_aligned(self, fp32_iris_spec: Float32ModelSpec) -> None:
        spec = fp32_iris_spec
        assert spec.padded_hidden_dim % spec.simd_width == 0

    def test_padded_input_dim_is_cache_aligned(self, fp32_iris_spec: Float32ModelSpec) -> None:
        spec = fp32_iris_spec
        byte_stride = spec.padded_input_dim * np.dtype(spec.SCALAR_NP_TYPE).itemsize
        assert byte_stride % spec.cache_line_bytes == 0

    def test_padded_class_dim_is_cache_aligned(self, fp32_iris_spec: Float32ModelSpec) -> None:
        spec = fp32_iris_spec
        byte_stride = spec.padded_class_dim * np.dtype(spec.SCALAR_NP_TYPE).itemsize
        assert byte_stride % spec.cache_line_bytes == 0

    def test_padded_dims_are_ge_logical(self, fp32_iris_spec: Float32ModelSpec) -> None:
        spec = fp32_iris_spec
        assert spec.padded_hidden_dim >= spec.hidden_dim
        assert spec.padded_input_dim >= spec.input_dim
        assert spec.padded_class_dim >= spec.output_classes
        assert spec.padded_module_dim >= spec.num_modules

    def test_fp16_padding_differs_from_fp32(
        self, fp32_iris_spec: Float32ModelSpec, fp16_iris_spec: Float16ModelSpec
    ) -> None:
        """FP16's smaller element size should yield wider padded dims (in elements)."""
        # Same cache line, half the bytes per element -> twice the elements per row
        assert fp16_iris_spec.padded_input_dim >= fp32_iris_spec.padded_input_dim

    @pytest.mark.parametrize("hidden_dim", [1, 3, 7, 16, 31, 64, 129])
    def test_padded_hidden_dim_always_aligned(self, hidden_dim: int) -> None:
        spec = Float32ModelSpec(
            input_dim=4,
            hidden_dim=hidden_dim,
            output_classes=3,
            num_modules=2,
            simd_width=4,
            cache_line_bytes=64,
        )
        assert spec.padded_hidden_dim % spec.simd_width == 0
        assert spec.padded_hidden_dim >= hidden_dim


# =========================================================================
# 2. ParameterSpace – Layout Generation
# =========================================================================


class TestParameterSpaceLayouts:
    """Verify that ParameterSpace produces a complete, consistent memory plan."""

    def test_all_expected_buffers_present(
        self, fp32_iris_spec: Float32ModelSpec, iris_param_space: ParameterSpace
    ) -> None:
        grid = _make_tiling(fp32_iris_spec)
        layouts = iris_param_space.get_all_memory_layouts(
            batch_size=150,
            grid=grid,
            num_batch_chunks=4,
        )

        # Core learnable parameters
        for name in [
            "shared_weights",
            "shared_biases",
            "module_weights",
            "module_biases",
            "temperatures",
        ]:
            assert name in layouts, f"Missing buffer: {name}"

        # Optimizer state (m1, m2) for non-specialized flows
        for flow in iris_param_space:
            if not flow.specialized_reduction:
                assert flow.m1_buffer_name in layouts, f"Missing m1 for {flow.name}"
                assert flow.m2_buffer_name in layouts, f"Missing m2 for {flow.name}"

        # Partial gradient collection buffers
        for stem in [
            "module_weights",
            "module_biases",
            "temps",
            "hidden_activations",
            "shared_weights",
            "shared_biases",
        ]:
            # Just check partial_grad_* exists (clipped mirrors are auto-generated)
            assert f"partial_grad_{stem}" in layouts, f"Missing buffer: partial_grad_{stem}"

    def test_optimizer_state_shapes_match_param_shapes(
        self, fp32_iris_spec: Float32ModelSpec, iris_param_space: ParameterSpace
    ) -> None:
        """Contract: m1 and m2 shapes must equal their parent parameter shapes."""
        grid = _make_tiling(fp32_iris_spec)
        layouts = iris_param_space.get_all_memory_layouts(
            batch_size=150,
            grid=grid,
            num_batch_chunks=4,
        )
        dtype = np.dtype(fp32_iris_spec.SCALAR_NP_TYPE)
        for flow in iris_param_space:
            if flow.specialized_reduction or not flow.m1_buffer_name:
                continue
            param_shape = layouts[flow.param_buffer_name].get_padded_shape(dtype)
            m1_shape = layouts[flow.m1_buffer_name].get_padded_shape(dtype)
            m2_shape = layouts[flow.m2_buffer_name].get_padded_shape(dtype)
            assert m1_shape == param_shape, f"m1 shape mismatch for {flow.name}"
            assert m2_shape == param_shape, f"m2 shape mismatch for {flow.name}"

    def test_clipped_shapes_equal_partial_shapes(
        self, fp32_iris_spec: Float32ModelSpec, iris_param_space: ParameterSpace
    ) -> None:
        """Contract: clipped partials have identical shape to raw partials."""
        grid = _make_tiling(fp32_iris_spec)
        layouts = iris_param_space.get_all_memory_layouts(
            batch_size=150,
            grid=grid,
            num_batch_chunks=4,
        )
        dtype = np.dtype(fp32_iris_spec.SCALAR_NP_TYPE)
        for key in list(layouts.keys()):
            if key.startswith("partial_grad_"):
                clipped_key = key.replace("partial_grad_", "clipped_partial_grad_")
                if clipped_key in layouts:
                    assert layouts[key].get_padded_shape(dtype) == layouts[clipped_key].get_padded_shape(
                        dtype
                    ), f"Shape mismatch: {key} vs {clipped_key}"

    def test_shared_weights_shape_uses_padded_hidden_major_layout(
        self, fp32_iris_spec: Float32ModelSpec, iris_param_space: ParameterSpace
    ) -> None:
        """Contract: shared_weights is (padded_hidden_dim, padded_input_dim) — hidden-major."""
        grid = _make_tiling(fp32_iris_spec)
        layouts = iris_param_space.get_all_memory_layouts(
            batch_size=150,
            grid=grid,
            num_batch_chunks=4,
        )
        sw_shape = layouts["shared_weights"].logical_shape
        assert sw_shape[0] == fp32_iris_spec.padded_hidden_dim
        assert sw_shape[1] == fp32_iris_spec.padded_input_dim

    def test_partial_grad_shared_weights_first_dim_equals_num_batch_chunks(
        self, fp32_iris_spec: Float32ModelSpec, iris_param_space: ParameterSpace
    ) -> None:
        """The streaming model writes one partial per batch chunk."""
        grid = _make_tiling(fp32_iris_spec)
        num_batch_chunks = 4
        layouts = iris_param_space.get_all_memory_layouts(
            batch_size=150,
            grid=grid,
            num_batch_chunks=num_batch_chunks,
        )
        pg_sw_shape = layouts["partial_grad_shared_weights"].logical_shape
        assert pg_sw_shape[0] == num_batch_chunks


# =========================================================================
# 3. Cross-Precision Integration (FP32 / FP16 Contract Consistency)
# =========================================================================


class TestCrossPrecisionConsistency:
    """Verify that the model-to-layout pipeline is self-consistent for both precisions."""

    @pytest.mark.parametrize(
        "spec_cls,ctx_cls",
        [
            (Float32ModelSpec, "fp32"),
            (Float16ModelSpec, "fp16"),
        ],
    )
    def test_layout_pipeline_end_to_end(self, spec_cls: type[ModelSpec], ctx_cls: str) -> None:
        spec = spec_cls(
            input_dim=4,
            hidden_dim=32,
            output_classes=3,
            num_modules=8,
            simd_width=4,
            cache_line_bytes=64,
        )
        ps = ParameterSpace(spec=spec)
        grid = _make_tiling(spec)
        layouts = ps.get_all_memory_layouts(batch_size=150, grid=grid, num_batch_chunks=4)

        dtype = np.dtype(spec.SCALAR_NP_TYPE)
        for name, layout in layouts.items():
            shape = layout.get_padded_shape(dtype)
            assert all(d > 0 for d in shape), f"Zero-dim detected in {name}: {shape}"
            total_bytes = int(np.prod(shape)) * dtype.itemsize
            assert total_bytes > 0, f"Zero-byte allocation for {name}"


# =========================================================================
# 4. Scaling Scenarios (Hydra / Lexicon / Behemoth)
# =========================================================================


class TestScalingScenarios:
    """Verify that the memory planning pipeline handles extreme configurations."""

    def test_hydra_massive_modules(self, fp32_hydra_spec: Float32ModelSpec) -> None:
        """Hydra: 256 modules.  All buffers must be allocated without error."""
        ps = ParameterSpace(spec=fp32_hydra_spec)
        grid = _make_tiling(fp32_hydra_spec)
        layouts = ps.get_all_memory_layouts(batch_size=32, grid=grid, num_batch_chunks=4)
        dtype = np.dtype(fp32_hydra_spec.SCALAR_NP_TYPE)

        # Module-weight partials scale with total_tiles
        pg_mw = layouts["partial_grad_module_weights"]
        shape = pg_mw.logical_shape
        assert shape[0] == grid.total_tiles
        for name, layout in layouts.items():
            padded = layout.get_padded_shape(dtype)
            assert all(d > 0 for d in padded), f"Bad shape for {name}: {padded}"

    def test_lexicon_massive_classes(self, fp32_lexicon_spec: Float32ModelSpec) -> None:
        """Lexicon: 10,000 output classes.  Tiling and layout must succeed."""
        ps = ParameterSpace(spec=fp32_lexicon_spec)
        grid = _make_tiling(fp32_lexicon_spec)
        layouts = ps.get_all_memory_layouts(batch_size=16, grid=grid, num_batch_chunks=4)
        dtype = np.dtype(fp32_lexicon_spec.SCALAR_NP_TYPE)

        assert grid.num_class_chunks > 1, "Lexicon should have multiple class chunks"
        for name, layout in layouts.items():
            padded = layout.get_padded_shape(dtype)
            assert all(d > 0 for d in padded), f"Bad shape for {name}: {padded}"

    def test_single_sample_batch(self, fp32_iris_spec: Float32ModelSpec) -> None:
        """Graceful degradation: batch_size=1 must still produce valid layouts."""
        ps = ParameterSpace(spec=fp32_iris_spec)
        grid = _make_tiling(fp32_iris_spec)
        layouts = ps.get_all_memory_layouts(batch_size=1, grid=grid, num_batch_chunks=1)
        dtype = np.dtype(fp32_iris_spec.SCALAR_NP_TYPE)

        for name, layout in layouts.items():
            shape = layout.get_padded_shape(dtype)
            assert all(d > 0 for d in shape), f"Bad shape for {name}: {shape}"


# =========================================================================
# 5. MemoryLayout Composition
# =========================================================================


class TestMemoryLayoutComposition:
    """Verify that MemoryLayout correctly composes padding strategies."""

    def test_element_count_padding(self):
        layout = MemoryLayout((10, 30))
        layout.add_strategy(PaddingStrategy(type=PaddingType.ELEMENT_COUNT, value=16, target_dim_idx=-1))
        shape = layout.get_padded_shape(np.dtype(np.float32))
        assert shape == (10, 32)

    def test_byte_alignment_padding_fp32(self):
        # 4 elements * 4 bytes = 16 bytes, pad to 64 -> 16 elements
        layout = MemoryLayout((5, 4))
        layout.add_strategy(PaddingStrategy(type=PaddingType.BYTE_ALIGNMENT, value=64, target_dim_idx=-1))
        shape = layout.get_padded_shape(np.dtype(np.float32))
        assert shape[1] * 4 % 64 == 0

    def test_byte_alignment_padding_fp16(self):
        # 4 elements * 2 bytes = 8 bytes, pad to 64 -> 32 elements
        layout = MemoryLayout((5, 4))
        layout.add_strategy(PaddingStrategy(type=PaddingType.BYTE_ALIGNMENT, value=64, target_dim_idx=-1))
        shape = layout.get_padded_shape(np.dtype(np.float16))
        assert shape[1] * 2 % 64 == 0
        assert shape[1] == 32  # 64 / 2

    def test_compound_padding(self):
        """SIMD alignment then byte alignment in sequence."""
        layout = MemoryLayout((5, 3))
        layout.add_strategy(PaddingStrategy(type=PaddingType.ELEMENT_COUNT, value=8, target_dim_idx=-1))
        layout.add_strategy(PaddingStrategy(type=PaddingType.BYTE_ALIGNMENT, value=64, target_dim_idx=-1))
        shape = layout.get_padded_shape(np.dtype(np.float32))
        assert shape[1] % 8 == 0
        assert shape[1] * 4 % 64 == 0

    def test_no_strategy_returns_logical_shape(self):
        layout = MemoryLayout((10, 20, 30))
        assert layout.get_padded_shape(np.dtype(np.float32)) == (10, 20, 30)

    def test_byte_alignment_on_non_last_dim_raises(self):
        layout = MemoryLayout((10, 20))
        layout.add_strategy(PaddingStrategy(type=PaddingType.BYTE_ALIGNMENT, value=64, target_dim_idx=0))
        with pytest.raises(ValueError, match="last dimension"):
            layout.get_padded_shape(np.dtype(np.float32))

    def test_negative_shape_rejected(self):
        with pytest.raises(ValueError):
            MemoryLayout((-1, 20))
