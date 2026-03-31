# kernel_signatures/tests/test_phase_1_act.py

"""
Unit tests for Phase 1 (Act) kernel signatures: Nodes 4–7.

Covers:
  - ForwardPassSignature           (forward_pass)
  - RenderLogitsChunkSignature     (render_logits_chunk)
  - ComputeProbsLossCceChunkSignature  (compute_probs_loss_cce_chunk)
  - ComputeProbsLossBceChunkSignature  (compute_probs_loss_bce_chunk)
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
    INPUT_PADDED,
    MODULES,
    MODULES_PER_CHUNK,
    TILES,
    MockArchConsts,
    MockBufferManager,
    ParamCategory,
    classify_python_arg,
    make_tile,
)

from src.backends.opencl.kernel_bindings.phase_1_act import (  # type: ignore[import-not-found]
    ForwardPassSignature,
    RenderLogitsChunkSignature,
    ComputeProbsLossCceChunkSignature,
    ComputeProbsLossBceChunkSignature,
)


# =========================================================================
# Node 4: forward_pass
# =========================================================================


class TestForwardPassSignature:

    @pytest.fixture
    def sig(self, bm: MockBufferManager, ac: MockArchConsts):
        return ForwardPassSignature(
            _buffer_mgr=bm,
            _arch_consts=ac,
            in_ref=bm.make_handle((BATCH, INPUT_PADDED)),
            mask_ref=bm.make_handle((BATCH,)),
            w_ref=bm.make_handle(
                (HIDDEN_PADDED // ac.simd_width, INPUT_PADDED, ac.simd_width)
            ),
            b_ref=bm.make_handle((HIDDEN_PADDED,)),
            h_out_ref=bm.make_handle((BATCH, HIDDEN_PADDED)),
            h_mask_out_ref=bm.make_handle((BATCH, HIDDEN_PADDED)),
            batch_chunk_offset=np.uint32(0),
            batch_chunk_count=np.uint32(BATCH),
        )

    def test_kernel_name(self, sig):
        assert sig.kernel_name == "forward_pass"

    def test_derived_fields(self, sig):
        assert sig.total_batch_count == np.uint32(BATCH)
        assert sig.padded_input_count == np.uint32(INPUT_PADDED)
        assert sig.padded_hidden_count == np.uint32(HIDDEN_PADDED)

    def test_get_args_count(self, sig):
        # C header: 1 local + 6 global + 5 uint = 12
        assert len(sig.get_args()) == 12

    def test_get_args_type_order(self, sig):
        cats = [classify_python_arg(a) for a in sig.get_args()]
        expected = (
            [ParamCategory.LOCAL_MEM]
            + [ParamCategory.GLOBAL_BUFFER] * 6
            + [ParamCategory.UINT_SCALAR] * 5
        )
        assert cats == expected

    def test_grid_wellformed(self, sig):
        g, l = sig.get_grid()
        assert len(g) == 2
        assert l is not None and len(l) == 2
        assert all(d > 0 for d in g)
        assert all(d > 0 for d in l)


# =========================================================================
# Node 5: render_logits_chunk
# =========================================================================


class TestRenderLogitsChunkSignature:

    @pytest.fixture
    def sig(self, bm: MockBufferManager, ac: MockArchConsts):
        return RenderLogitsChunkSignature(
            _buffer_mgr=bm,
            _arch_consts=ac,
            h_ref=bm.make_handle((BATCH, HIDDEN_PADDED)),
            h_mask_ref=bm.make_handle((BATCH, HIDDEN_PADDED)),
            w_ref=bm.make_handle((MODULES, HIDDEN_PADDED, CLASSES_PADDED)),
            b_ref=bm.make_handle((MODULES, CLASSES_PADDED)),
            logit_out_ref=bm.make_handle((MODULES, BATCH, CLASSES_PADDED)),
            batch_chunk_offset=np.uint32(0),
            batch_chunk_count=np.uint32(BATCH),
            module_chunk_offset=np.uint32(0),
            module_chunk_count=np.uint32(MODULES_PER_CHUNK),
            class_chunk_offset=np.uint32(0),
            class_chunk_count=np.uint32(CLASSES_PER_CHUNK),
            hidden_count=np.uint32(HIDDEN_NATURAL),
            total_output_class_count=np.uint32(CLASSES_NATURAL),
        )

    def test_kernel_name(self, sig):
        assert sig.kernel_name == "render_logits_chunk"

    def test_derived_fields(self, sig):
        assert sig.total_batch_count == np.uint32(BATCH)
        assert sig.padded_hidden_count == np.uint32(HIDDEN_PADDED)
        assert sig.total_modules_count == np.uint32(MODULES)
        assert sig.padded_total_output_class_count == np.uint32(CLASSES_PADDED)

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

    def test_grid_3d(self, sig):
        g, l = sig.get_grid()
        assert len(g) == 3
        assert l is None


# =========================================================================
# Node 6: compute_probs_loss_cce_chunk
# =========================================================================


class TestComputeProbsLossCceChunkSignature:

    @pytest.fixture
    def sig(self, bm: MockBufferManager, ac: MockArchConsts):
        return ComputeProbsLossCceChunkSignature(
            _buffer_mgr=bm,
            _arch_consts=ac,
            logit_ref=bm.make_handle((MODULES, BATCH, CLASSES_PADDED)),
            temp_ref=bm.make_handle((MODULES,)),
            target_ref=bm.make_handle((BATCH,)),
            mask_ref=bm.make_handle((BATCH,)),
            prob_out_ref=bm.make_handle(
                (TILES, MODULES_PER_CHUNK, BATCH, CLASSES_PER_CHUNK)
            ),
            loss_out_ref=bm.make_handle((MODULES, BATCH)),
            tile=make_tile(),
            total_output_class_count=np.uint32(CLASSES_NATURAL),
        )

    def test_kernel_name(self, sig):
        assert sig.kernel_name == "compute_probs_loss_cce_chunk"

    def test_derived_fields(self, sig):
        assert sig.total_modules_count == np.uint32(MODULES)
        assert sig.total_batch_count == np.uint32(BATCH)
        assert sig.padded_total_output_class_count == np.uint32(CLASSES_PADDED)
        assert sig.total_tile_count == np.uint32(TILES)

    def test_get_args_count(self, sig):
        # C header: 6 global + 9 uint = 15
        assert len(sig.get_args()) == 15

    def test_get_args_type_order(self, sig):
        cats = [classify_python_arg(a) for a in sig.get_args()]
        expected = (
            [ParamCategory.GLOBAL_BUFFER] * 6
            + [ParamCategory.UINT_SCALAR] * 9
        )
        assert cats == expected


# =========================================================================
# Node 7: compute_probs_loss_bce_chunk
# =========================================================================


class TestComputeProbsLossBceChunkSignature:

    @pytest.fixture
    def sig(self, bm: MockBufferManager, ac: MockArchConsts):
        return ComputeProbsLossBceChunkSignature(
            _buffer_mgr=bm,
            _arch_consts=ac,
            logit_ref=bm.make_handle((MODULES, BATCH, CLASSES_PADDED)),
            temp_ref=bm.make_handle((MODULES,)),
            target_ref=bm.make_handle((BATCH, CLASSES_PADDED)),
            mask_ref=bm.make_handle((BATCH,)),
            prob_out_ref=bm.make_handle(
                (TILES, MODULES_PER_CHUNK, BATCH, CLASSES_PER_CHUNK)
            ),
            partial_loss_out_ref=bm.make_handle(
                (TILES, MODULES_PER_CHUNK, BATCH)
            ),
            tile=make_tile(),
            total_output_class_count=np.uint32(CLASSES_NATURAL),
        )

    def test_kernel_name(self, sig):
        assert sig.kernel_name == "compute_probs_loss_bce_chunk"

    def test_get_args_count(self, sig):
        # Same structure as CCE: 6 global + 9 uint = 15
        assert len(sig.get_args()) == 15

    def test_get_args_type_order(self, sig):
        cats = [classify_python_arg(a) for a in sig.get_args()]
        expected = (
            [ParamCategory.GLOBAL_BUFFER] * 6
            + [ParamCategory.UINT_SCALAR] * 9
        )
        assert cats == expected
