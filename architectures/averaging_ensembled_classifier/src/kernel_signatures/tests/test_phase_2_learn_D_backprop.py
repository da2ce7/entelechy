# kernel_signatures/tests/test_phase_2_learn_D_backprop.py

"""
Unit tests for Phase 2D (Shared Layer Backprop) kernel signatures: Nodes 17–19.

Covers:
  - BackpropSharedWeightsChunkSignature  (backprop_shared_weights_chunk)
  - BackpropSharedBiasesChunkSignature   (backprop_shared_biases_chunk)
  - ClipSharedGradientsChunkSignature    (clip_shared_gradients_chunk)
"""

from __future__ import annotations

import numpy as np
import pytest

from .conftest import (
    BATCH,
    BATCH_CHUNKS,
    HIDDEN_PADDED,
    INPUT_PADDED,
    MockArchConsts,
    MockBufferManager,
    ParamCategory,
    classify_python_arg,
)

from src.kernel_signatures.phase_2_learn_D_backprop import (  # type: ignore[import-not-found]
    BackpropSharedWeightsChunkSignature,
    BackpropSharedBiasesChunkSignature,
    SharedGradientHandles,
    ClipSharedGradientsChunkSignature,
)


# =========================================================================
# Node 17: backprop_shared_weights_chunk
# =========================================================================


class TestBackpropSharedWeightsChunkSignature:

    @pytest.fixture
    def sig(self, bm: MockBufferManager, ac: MockArchConsts):
        return BackpropSharedWeightsChunkSignature(
            _buffer_mgr=bm,
            _arch_consts=ac,
            input_ref=bm.make_handle((BATCH, INPUT_PADDED)),
            h_ref=bm.make_handle((BATCH, HIDDEN_PADDED)),
            grad_h_ref=bm.make_handle((BATCH, HIDDEN_PADDED)),
            mask_ref=bm.make_handle((BATCH,)),
            partial_gsw_out_ref=bm.make_handle(
                (BATCH_CHUNKS, INPUT_PADDED, HIDDEN_PADDED)
            ),
            batch_chunk_offset=np.uint32(0),
            batch_chunk_count=np.uint32(BATCH // BATCH_CHUNKS),
            batch_chunk_index=np.uint32(0),
            num_batch_chunks_count=np.uint32(BATCH_CHUNKS),
        )

    def test_kernel_name(self, sig):
        assert sig.kernel_name == "backprop_shared_weights_chunk"

    def test_derived_fields(self, sig):
        assert sig.total_batch_count == np.uint32(BATCH)
        assert sig.padded_input_count == np.uint32(INPUT_PADDED)
        assert sig.padded_hidden_count == np.uint32(HIDDEN_PADDED)
        assert sig.final_grad_hidden_total_element_count == np.uint32(
            BATCH * HIDDEN_PADDED
        )

    def test_get_args_count(self, sig):
        # C header: 1 local + 5 global + 8 uint = 14
        assert len(sig.get_args()) == 14

    def test_get_args_type_order(self, sig):
        cats = [classify_python_arg(a) for a in sig.get_args()]
        expected = (
            [ParamCategory.LOCAL_MEM]
            + [ParamCategory.GLOBAL_BUFFER] * 5
            + [ParamCategory.UINT_SCALAR] * 8
        )
        assert cats == expected


# =========================================================================
# Node 18: backprop_shared_biases_chunk
# =========================================================================


class TestBackpropSharedBiasesChunkSignature:

    @pytest.fixture
    def sig(self, bm: MockBufferManager, ac: MockArchConsts):
        return BackpropSharedBiasesChunkSignature(
            _buffer_mgr=bm,
            _arch_consts=ac,
            h_ref=bm.make_handle((BATCH, HIDDEN_PADDED)),
            grad_h_ref=bm.make_handle((BATCH, HIDDEN_PADDED)),
            mask_ref=bm.make_handle((BATCH,)),
            partial_gsb_out_ref=bm.make_handle((BATCH_CHUNKS, HIDDEN_PADDED)),
            batch_chunk_offset=np.uint32(0),
            batch_chunk_count=np.uint32(BATCH // BATCH_CHUNKS),
            batch_chunk_index=np.uint32(0),
            num_batch_chunks_count=np.uint32(BATCH_CHUNKS),
        )

    def test_kernel_name(self, sig):
        assert sig.kernel_name == "backprop_shared_biases_chunk"

    def test_derived_fields(self, sig):
        assert sig.total_batch_count == np.uint32(BATCH)
        assert sig.padded_hidden_count == np.uint32(HIDDEN_PADDED)
        assert sig.final_grad_hidden_total_element_count == np.uint32(
            BATCH * HIDDEN_PADDED
        )

    def test_get_args_count(self, sig):
        # C header: 1 local + 4 global + 7 uint = 12
        assert len(sig.get_args()) == 12

    def test_get_args_type_order(self, sig):
        cats = [classify_python_arg(a) for a in sig.get_args()]
        expected = (
            [ParamCategory.LOCAL_MEM]
            + [ParamCategory.GLOBAL_BUFFER] * 4
            + [ParamCategory.UINT_SCALAR] * 7
        )
        assert cats == expected


# =========================================================================
# Node 19: clip_shared_gradients_chunk
# =========================================================================


class TestClipSharedGradientsChunkSignature:

    @pytest.fixture
    def sig(self, bm: MockBufferManager, ac: MockArchConsts):
        return ClipSharedGradientsChunkSignature(
            _buffer_mgr=bm,
            _arch_consts=ac,
            handles=SharedGradientHandles(
                grad_weights_shared_chunk=bm.make_handle(
                    (INPUT_PADDED, HIDDEN_PADDED)
                ),
                grad_biases_shared_chunk=bm.make_handle((HIDDEN_PADDED,)),
                clipped_grad_weights_shared_collection=bm.make_handle(
                    (BATCH_CHUNKS, INPUT_PADDED, HIDDEN_PADDED)
                ),
                clipped_grad_biases_shared_collection=bm.make_handle(
                    (BATCH_CHUNKS, HIDDEN_PADDED)
                ),
            ),
            clipping_threshold_global=np.float32(1.0),
            epsilon=np.float32(1e-6),
            dest_weights_write_offset_elements=np.uint32(0),
            dest_biases_write_offset_elements=np.uint32(0),
            num_batch_chunks=np.uint32(BATCH_CHUNKS),
        )

    def test_kernel_name(self, sig):
        assert sig.kernel_name == "clip_shared_gradients_chunk"

    def test_derived_fields(self, sig):
        assert sig.weights_param_count == np.uint32(INPUT_PADDED * HIDDEN_PADDED)
        assert sig.biases_param_count == np.uint32(HIDDEN_PADDED)

    def test_get_args_count(self, sig):
        # C header: 1 local + 4 global + 2 float + 5 uint = 12
        assert len(sig.get_args()) == 12

    def test_get_args_type_order(self, sig):
        cats = [classify_python_arg(a) for a in sig.get_args()]
        expected = (
            [ParamCategory.LOCAL_MEM]
            + [ParamCategory.GLOBAL_BUFFER] * 4
            + [ParamCategory.FLOAT_SCALAR] * 2
            + [ParamCategory.UINT_SCALAR] * 5
        )
        assert cats == expected
