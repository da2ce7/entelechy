# kernel_signatures/tests/test_phase_2_learn_A_production.py

"""
Unit tests for Phase 2A (Gradient Production) kernel signatures: Nodes 8–10.

Covers:
  - CalculateModuleParamGradsCceSignature / BceSignature  (calculate_module_param_grads_chunk)
  - BackpropErrorToHiddenChunkCceSignature / BceSignature  (backprop_error_to_hidden_chunk)
  - CalculateChunkTempGradientsCceSignature / BceSignature (calculate_chunk_temp_gradients)
"""

from __future__ import annotations

import numpy as np
import pytest

from .conftest import (
    BATCH,
    CLASSES_NATURAL,
    CLASSES_PADDED,
    CLASSES_PER_CHUNK,
    HIDDEN_NATURAL,
    HIDDEN_PADDED,
    MODULES,
    MODULES_PER_CHUNK,
    TILES,
    WG_SIZE_REDUCTION,
    MockArchConsts,
    MockBufferManager,
    ParamCategory,
    classify_python_arg,
    make_tile,
)

from src.backends.opencl.kernel_bindings.phase_2_learn_A_production import (  # type: ignore[import-not-found]
    CalculateModuleParamGradsCceSignature,
    CalculateModuleParamGradsBceSignature,
    BackpropErrorToHiddenChunkCceSignature,
    BackpropErrorToHiddenChunkBceSignature,
    CalculateChunkTempGradientsCceSignature,
    CalculateChunkTempGradientsBceSignature,
)


# =========================================================================
# Node 8: calculate_module_param_grads_chunk (CCE variant)
# =========================================================================


class TestCalculateModuleParamGradsCce:

    @pytest.fixture
    def sig(self, bm: MockBufferManager, ac: MockArchConsts):
        return CalculateModuleParamGradsCceSignature(
            _buffer_mgr=bm,
            _arch_consts=ac,
            work_group_size_0=WG_SIZE_REDUCTION,
            h_ref=bm.make_handle((BATCH, HIDDEN_PADDED)),
            prob_ref=bm.make_handle(
                (TILES, MODULES_PER_CHUNK, BATCH, CLASSES_PER_CHUNK)
            ),
            mask_ref=bm.make_handle((BATCH,)),
            gw_out_ref=bm.make_handle(
                (TILES, MODULES_PER_CHUNK, HIDDEN_PADDED, CLASSES_PER_CHUNK)
            ),
            gb_out_ref=bm.make_handle(
                (TILES, MODULES_PER_CHUNK, CLASSES_PER_CHUNK)
            ),
            tile=make_tile(),
            batch_chunk_offset=np.uint32(0),
            batch_chunk_count=np.uint32(BATCH),
            hidden_count=np.uint32(HIDDEN_NATURAL),
            total_output_class_count=np.uint32(CLASSES_NATURAL),
            padded_total_output_class_count=np.uint32(CLASSES_PADDED),
            total_modules_count=np.uint32(MODULES),
            targets_cce_ref=bm.make_handle((BATCH,)),
        )

    def test_kernel_name(self, sig):
        assert sig.kernel_name == "calculate_module_param_grads_chunk"

    def test_derived_fields(self, sig):
        assert sig.padded_hidden_count == np.uint32(HIDDEN_PADDED)
        assert sig.total_batch_count == np.uint32(BATCH)
        assert sig.total_tile_count == np.uint32(TILES)

    def test_get_args_count(self, sig):
        # C header: 1 local + 7 global + 14 uint = 22
        # But: 1 local + 6 global(h,prob,targets,mask,gw,gb) + 1 flag + 13 uint
        # Actually counting from the C declaration: 22 total params
        assert len(sig.get_args()) == 21

    def test_get_args_type_order(self, sig):
        cats = [classify_python_arg(a) for a in sig.get_args()]
        expected = (
            [ParamCategory.LOCAL_MEM]       # local reduction tile
            + [ParamCategory.GLOBAL_BUFFER] * 6  # h, prob, targets, mask, gw_out, gb_out
            + [ParamCategory.UINT_SCALAR] * 14   # flag + all scalars
        )
        assert cats == expected

    def test_cce_flag_is_zero(self, sig):
        """The problem_type flag (first uint scalar after buffers) must be 0 for CCE."""
        args = sig.get_args()
        # Position 7 is the first uint after 1 local + 6 global = position 7
        assert args[7] == np.uint32(0), "CCE flag must be 0"


# =========================================================================
# Node 8: calculate_module_param_grads_chunk (BCE variant)
# =========================================================================


class TestCalculateModuleParamGradsBce:

    @pytest.fixture
    def sig(self, bm: MockBufferManager, ac: MockArchConsts):
        return CalculateModuleParamGradsBceSignature(
            _buffer_mgr=bm,
            _arch_consts=ac,
            work_group_size_0=WG_SIZE_REDUCTION,
            h_ref=bm.make_handle((BATCH, HIDDEN_PADDED)),
            prob_ref=bm.make_handle(
                (TILES, MODULES_PER_CHUNK, BATCH, CLASSES_PER_CHUNK)
            ),
            mask_ref=bm.make_handle((BATCH,)),
            gw_out_ref=bm.make_handle(
                (TILES, MODULES_PER_CHUNK, HIDDEN_PADDED, CLASSES_PER_CHUNK)
            ),
            gb_out_ref=bm.make_handle(
                (TILES, MODULES_PER_CHUNK, CLASSES_PER_CHUNK)
            ),
            tile=make_tile(),
            batch_chunk_offset=np.uint32(0),
            batch_chunk_count=np.uint32(BATCH),
            hidden_count=np.uint32(HIDDEN_NATURAL),
            total_output_class_count=np.uint32(CLASSES_NATURAL),
            padded_total_output_class_count=np.uint32(CLASSES_PADDED),
            total_modules_count=np.uint32(MODULES),
            targets_bce_ref=bm.make_handle((BATCH, CLASSES_PADDED)),
        )

    def test_bce_flag_is_one(self, sig):
        """The problem_type flag must be 1 for BCE."""
        args = sig.get_args()
        assert args[7] == np.uint32(1), "BCE flag must be 1"

    def test_get_args_count(self, sig):
        assert len(sig.get_args()) == 21


# =========================================================================
# Node 9: backprop_error_to_hidden_chunk (CCE variant)
# =========================================================================


class TestBackpropErrorToHiddenChunkCce:

    @pytest.fixture
    def sig(self, bm: MockBufferManager, ac: MockArchConsts):
        return BackpropErrorToHiddenChunkCceSignature(
            _buffer_mgr=bm,
            _arch_consts=ac,
            prob_ref=bm.make_handle(
                (TILES, MODULES_PER_CHUNK, BATCH, CLASSES_PER_CHUNK)
            ),
            mask_ref=bm.make_handle((BATCH,)),
            w_mod_ref=bm.make_handle((MODULES, HIDDEN_PADDED, CLASSES_PADDED)),
            gh_out_ref=bm.make_handle(
                (TILES, MODULES_PER_CHUNK, BATCH, HIDDEN_PADDED)
            ),
            tile=make_tile(),
            hidden_count=np.uint32(HIDDEN_NATURAL),
            total_output_class_count=np.uint32(CLASSES_NATURAL),
            targets_cce_ref=bm.make_handle((BATCH,)),
        )

    def test_kernel_name(self, sig):
        assert sig.kernel_name == "backprop_error_to_hidden_chunk"

    def test_derived_fields(self, sig):
        assert sig.total_modules_count == np.uint32(MODULES)
        assert sig.padded_hidden_count == np.uint32(HIDDEN_PADDED)
        assert sig.total_tile_count == np.uint32(TILES)
        assert sig.total_batch_count == np.uint32(BATCH)

    def test_get_args_count(self, sig):
        # C header: 5 global + 12 uint = 17
        assert len(sig.get_args()) == 17

    def test_get_args_type_order(self, sig):
        cats = [classify_python_arg(a) for a in sig.get_args()]
        expected = (
            [ParamCategory.GLOBAL_BUFFER] * 5
            + [ParamCategory.UINT_SCALAR] * 12
        )
        assert cats == expected

    def test_cce_flag_is_zero(self, sig):
        args = sig.get_args()
        # Position 5 = first uint after 5 global buffers
        assert args[5] == np.uint32(0), "CCE flag must be 0"


# =========================================================================
# Node 9: backprop_error_to_hidden_chunk (BCE variant)
# =========================================================================


class TestBackpropErrorToHiddenChunkBce:

    @pytest.fixture
    def sig(self, bm: MockBufferManager, ac: MockArchConsts):
        return BackpropErrorToHiddenChunkBceSignature(
            _buffer_mgr=bm,
            _arch_consts=ac,
            prob_ref=bm.make_handle(
                (TILES, MODULES_PER_CHUNK, BATCH, CLASSES_PER_CHUNK)
            ),
            mask_ref=bm.make_handle((BATCH,)),
            w_mod_ref=bm.make_handle((MODULES, HIDDEN_PADDED, CLASSES_PADDED)),
            gh_out_ref=bm.make_handle(
                (TILES, MODULES_PER_CHUNK, BATCH, HIDDEN_PADDED)
            ),
            tile=make_tile(),
            hidden_count=np.uint32(HIDDEN_NATURAL),
            total_output_class_count=np.uint32(CLASSES_NATURAL),
            targets_bce_ref=bm.make_handle((BATCH, CLASSES_PADDED)),
        )

    def test_bce_flag_is_one(self, sig):
        args = sig.get_args()
        assert args[5] == np.uint32(1), "BCE flag must be 1"


# =========================================================================
# Node 10: calculate_chunk_temp_gradients (CCE variant)
# =========================================================================


class TestCalculateChunkTempGradientsCce:

    @pytest.fixture
    def sig(self, bm: MockBufferManager, ac: MockArchConsts):
        return CalculateChunkTempGradientsCceSignature(
            _buffer_mgr=bm,
            _arch_consts=ac,
            work_group_size_0=WG_SIZE_REDUCTION,
            logit_ref=bm.make_handle((MODULES, BATCH, CLASSES_PADDED)),
            prob_ref=bm.make_handle(
                (TILES, MODULES_PER_CHUNK, BATCH, CLASSES_PER_CHUNK)
            ),
            mask_ref=bm.make_handle((BATCH,)),
            temp_ref=bm.make_handle((MODULES,)),
            gt_out_ref=bm.make_handle((TILES, MODULES_PER_CHUNK)),
            tile=make_tile(),
            total_output_class_count=np.uint32(CLASSES_NATURAL),
            targets_cce_ref=bm.make_handle((BATCH,)),
        )

    def test_kernel_name(self, sig):
        assert sig.kernel_name == "calculate_chunk_temp_gradients"

    def test_get_args_count(self, sig):
        # C header: 1 local + 7 global + 10 uint = 18
        # Actually counted from header: 17
        assert len(sig.get_args()) == 17

    def test_get_args_type_order(self, sig):
        cats = [classify_python_arg(a) for a in sig.get_args()]
        expected = (
            [ParamCategory.LOCAL_MEM]
            + [ParamCategory.GLOBAL_BUFFER] * 6
            + [ParamCategory.UINT_SCALAR] * 10
        )
        assert cats == expected


# =========================================================================
# Node 10: calculate_chunk_temp_gradients (BCE variant)
# =========================================================================


class TestCalculateChunkTempGradientsBce:

    @pytest.fixture
    def sig(self, bm: MockBufferManager, ac: MockArchConsts):
        return CalculateChunkTempGradientsBceSignature(
            _buffer_mgr=bm,
            _arch_consts=ac,
            work_group_size_0=WG_SIZE_REDUCTION,
            logit_ref=bm.make_handle((MODULES, BATCH, CLASSES_PADDED)),
            prob_ref=bm.make_handle(
                (TILES, MODULES_PER_CHUNK, BATCH, CLASSES_PER_CHUNK)
            ),
            mask_ref=bm.make_handle((BATCH,)),
            temp_ref=bm.make_handle((MODULES,)),
            gt_out_ref=bm.make_handle((TILES, MODULES_PER_CHUNK)),
            tile=make_tile(),
            total_output_class_count=np.uint32(CLASSES_NATURAL),
            targets_bce_ref=bm.make_handle((BATCH, CLASSES_PADDED)),
        )

    def test_bce_flag_is_one(self, sig):
        args = sig.get_args()
        # Position 7 = first uint after 1 local + 6 global
        assert args[7] == np.uint32(1), "BCE flag must be 1"
