# kernel_signatures/tests/test_phase_2_learn_B_processing.py

"""
Unit tests for Phase 2B (Gradient Processing) kernel signatures: Nodes 11, 13.

Covers:
  - ClipPartialGradientsGlobalNormSignature  (clip_partial_gradients, flag=0)
  - ClipPartialGradientsPerItemNormSignature (clip_partial_gradients, flag=1)
  - GatherAndPermuteGradHiddenActivationsSignature (gather_and_permute_grad_hidden_activations)
"""

from __future__ import annotations

import numpy as np
import pytest

from .conftest import (
    BATCH,
    CLASSES_PER_CHUNK,
    HIDDEN_NATURAL,
    HIDDEN_PADDED,
    MODULES,
    MODULES_PER_CHUNK,
    NUM_CLASS_CHUNKS,
    NUM_MODULE_CHUNKS,
    PADDED_MODULES,
    TILES,
    MockArchConsts,
    MockBufferManager,
    ParamCategory,
    classify_python_arg,
    make_tile,
)

from src.kernel_signatures.phase_2_learn_B_processing import (
    ClipPartialGradientsGlobalNormSignature,
    ClipPartialGradientsPerItemNormSignature,
    GradientHandles,
    GatherAndPermuteGradHiddenActivationsSignature,
)


def _make_gradient_handles(bm: MockBufferManager) -> GradientHandles:
    return GradientHandles(
        grad_weights_module=bm.make_handle(
            (TILES, MODULES_PER_CHUNK, HIDDEN_PADDED, CLASSES_PER_CHUNK)
        ),
        grad_biases_module=bm.make_handle(
            (TILES, MODULES_PER_CHUNK, CLASSES_PER_CHUNK)
        ),
        grad_temps=bm.make_handle((TILES, MODULES_PER_CHUNK)),
        grad_hidden_activations_aos=bm.make_handle(
            (TILES, MODULES_PER_CHUNK, BATCH, HIDDEN_PADDED)
        ),
        clipped_grad_weights_module=bm.make_handle(
            (TILES, MODULES_PER_CHUNK, HIDDEN_PADDED, CLASSES_PER_CHUNK)
        ),
        clipped_grad_biases_module=bm.make_handle(
            (TILES, MODULES_PER_CHUNK, CLASSES_PER_CHUNK)
        ),
        clipped_grad_temps=bm.make_handle((TILES, MODULES_PER_CHUNK)),
        clipped_grad_hidden_activations_aos=bm.make_handle(
            (TILES, MODULES_PER_CHUNK, BATCH, HIDDEN_PADDED)
        ),
    )


# =========================================================================
# Node 11: clip_partial_gradients (Global Norm variant)
# =========================================================================


class TestClipPartialGradientsGlobalNorm:

    @pytest.fixture
    def sig(self, bm: MockBufferManager, ac: MockArchConsts):
        return ClipPartialGradientsGlobalNormSignature(
            _buffer_mgr=bm,
            _arch_consts=ac,
            handles=_make_gradient_handles(bm),
            tile=make_tile(),
            epsilon=np.float32(1e-6),
            clipping_threshold_global=np.float32(1.0),
        )

    def test_kernel_name(self, sig):
        assert sig.kernel_name == "clip_partial_gradients"

    def test_derived_fields(self, sig):
        assert sig.total_tile_count == np.uint32(TILES)
        assert sig.total_batch_count == np.uint32(BATCH)
        assert sig.padded_hidden_count == np.uint32(HIDDEN_PADDED)
        assert sig.padded_class_count == np.uint32(CLASSES_PER_CHUNK)

    def test_get_args_count(self, sig):
        # C header: 1 local + 10 global + 1 flag + 2 float + 7 uint = 21
        assert len(sig.get_args()) == 21

    def test_get_args_type_order(self, sig):
        cats = [classify_python_arg(a) for a in sig.get_args()]
        expected = (
            [ParamCategory.LOCAL_MEM]            # local reduction tile
            + [ParamCategory.GLOBAL_BUFFER] * 9   # 4 src + 1 conditional (None) + 4 dest
            + [ParamCategory.UINT_SCALAR]         # flag: use_per_item_norm
            + [ParamCategory.FLOAT_SCALAR]        # clipping threshold
            + [ParamCategory.FLOAT_SCALAR]        # epsilon
            + [ParamCategory.UINT_SCALAR] * 8     # tile_idx, class_chunks, classes_per_chunk, mods_per_chunk, batch, hidden, classes, tiles
        )
        assert cats == expected

    def test_global_flag_is_zero(self, sig):
        """The use_per_item_norm flag must be 0 for Global variant."""
        args = sig.get_args()
        # Position 10 (after 1 local + 9 global)
        assert args[10] == np.uint32(0)

    def test_null_per_item_buffer(self, sig):
        """The per-item threshold buffer must be None (NULL pointer) for global."""
        args = sig.get_args()
        # Position 5: the conditional per-item buffer
        assert args[5] is None


# =========================================================================
# Node 11: clip_partial_gradients (Per-Item Norm variant)
# =========================================================================


class TestClipPartialGradientsPerItemNorm:

    @pytest.fixture
    def sig(self, bm: MockBufferManager, ac: MockArchConsts):
        return ClipPartialGradientsPerItemNormSignature(
            _buffer_mgr=bm,
            _arch_consts=ac,
            handles=_make_gradient_handles(bm),
            tile=make_tile(),
            epsilon=np.float32(1e-6),
            clipping_threshold_per_item_ref=bm.make_handle((TILES,)),
        )

    def test_per_item_flag_is_one(self, sig):
        """The use_per_item_norm flag must be 1 for Per-Item variant."""
        args = sig.get_args()
        # Position 10 (after 1 local + 9 global)
        assert args[10] == np.uint32(1)

    def test_per_item_buffer_is_not_none(self, sig):
        """The per-item threshold buffer must be a real buffer, not None."""
        args = sig.get_args()
        assert args[5] is not None

    def test_get_args_count(self, sig):
        assert len(sig.get_args()) == 21


# =========================================================================
# Node 13: gather_and_permute_grad_hidden_activations
# =========================================================================


class TestGatherAndPermuteGradHiddenActivations:

    @pytest.fixture
    def sig(self, bm: MockBufferManager, ac: MockArchConsts):
        return GatherAndPermuteGradHiddenActivationsSignature(
            _buffer_mgr=bm,
            _arch_consts=ac,
            clipped_partials_aos_ref=bm.make_handle(
                (TILES, MODULES_PER_CHUNK, BATCH, HIDDEN_PADDED)
            ),
            permuted_soa_out_ref=bm.make_handle(
                (BATCH * HIDDEN_PADDED, PADDED_MODULES)
            ),
            total_modules_count=np.uint32(MODULES),
            hidden_count=np.uint32(HIDDEN_NATURAL),
            total_batch_count=np.uint32(BATCH),
            num_module_chunks=np.uint32(NUM_MODULE_CHUNKS),
            modules_per_chunk=np.uint32(MODULES_PER_CHUNK),
            num_class_chunks=np.uint32(NUM_CLASS_CHUNKS),
        )

    def test_kernel_name(self, sig):
        assert sig.kernel_name == "gather_and_permute_grad_hidden_activations"

    def test_derived_fields(self, sig):
        assert sig.total_tile_count == np.uint32(TILES)
        assert sig.padded_hidden_count == np.uint32(HIDDEN_PADDED)
        assert sig.padded_total_modules_count == np.uint32(PADDED_MODULES)

    def test_get_args_count(self, sig):
        # C header: 2 global + 9 uint = 11
        assert len(sig.get_args()) == 11

    def test_get_args_type_order(self, sig):
        cats = [classify_python_arg(a) for a in sig.get_args()]
        expected = (
            [ParamCategory.GLOBAL_BUFFER] * 2
            + [ParamCategory.UINT_SCALAR] * 9
        )
        assert cats == expected

    def test_plan_validation(self, bm: MockBufferManager, ac: MockArchConsts):
        """Verify that plan inconsistencies are caught by __post_init__ assertions."""
        with pytest.raises(AssertionError, match="Host plan inconsistency"):
            GatherAndPermuteGradHiddenActivationsSignature(
                _buffer_mgr=bm,
                _arch_consts=ac,
                clipped_partials_aos_ref=bm.make_handle(
                    (TILES, MODULES_PER_CHUNK, BATCH, HIDDEN_PADDED)
                ),
                permuted_soa_out_ref=bm.make_handle(
                    (BATCH * HIDDEN_PADDED, PADDED_MODULES)
                ),
                total_modules_count=np.uint32(MODULES),
                hidden_count=np.uint32(HIDDEN_NATURAL),
                total_batch_count=np.uint32(BATCH),
                # Intentionally wrong: 2 * 2 = 4 != TILES (1)
                num_module_chunks=np.uint32(2),
                modules_per_chunk=np.uint32(MODULES_PER_CHUNK),
                num_class_chunks=np.uint32(2),
            )
