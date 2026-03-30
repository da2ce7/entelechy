# kernel_signatures/tests/test_phase_2_learn_C_reduction.py

"""
Unit tests for Phase 2C (Reduction & Aggregation) kernel signatures: Nodes 14–16.

Covers:
  - AggregateRegisterReduceSignature      (aggregate_register_reduce)
  - AggregateLocalReduceSignature         (aggregate_local_reduce)
  - ClipIntermediateGradSignature         (clip_intermediate_grad)
  - StabilizeAndReduceGradHiddenActivationsSignature (stabilize_and_reduce_grad_hidden_activations)
"""

from __future__ import annotations

import numpy as np
import pytest

from .conftest import (
    BATCH,
    HIDDEN_PADDED,
    MODULES,
    PADDED_MODULES,
    MockArchConsts,
    MockBufferManager,
    ParamCategory,
    classify_python_arg,
)

from src.kernel_signatures.phase_2_learn_C_reduction import (  # type: ignore[import-not-found]
    AggregateRegisterReduceSignature,
    AggregateLocalReduceSignature,
    ClipIntermediateGradSignature,
    StabilizeAndReduceGradHiddenActivationsSignature,
)


# =========================================================================
# Nodes 14, 15a, 20a (Tier 1): aggregate_register_reduce
# =========================================================================


class TestAggregateRegisterReduceSignature:

    @pytest.fixture
    def sig(self, bm: MockBufferManager, ac: MockArchConsts):
        return AggregateRegisterReduceSignature(
            _buffer_mgr=bm,
            _arch_consts=ac,
            partial_collection_ref=bm.make_handle((128,)),
            partial_offset_list_ref=bm.make_handle((4,)),
            dest_ref=bm.make_handle((32,)),
            partial_offset_list_count=np.uint32(4),
            partial_width=np.uint32(32),
            operation_type=np.uint32(0),
        )

    def test_kernel_name(self, sig):
        assert sig.kernel_name == "aggregate_register_reduce"

    def test_get_args_count(self, sig):
        # C header: 3 global + 3 uint = 6
        assert len(sig.get_args()) == 6

    def test_get_args_type_order(self, sig):
        cats = [classify_python_arg(a) for a in sig.get_args()]
        expected = (
            [ParamCategory.GLOBAL_BUFFER] * 3
            + [ParamCategory.UINT_SCALAR] * 3
        )
        assert cats == expected

    def test_grid_1d(self, sig):
        g, l = sig.get_grid()
        assert len(g) == 1
        assert l is None


# =========================================================================
# Nodes 14, 15a, 20a (Tier 2): aggregate_local_reduce
# =========================================================================


class TestAggregateLocalReduceSignature:

    @pytest.fixture
    def sig(self, bm: MockBufferManager, ac: MockArchConsts):
        return AggregateLocalReduceSignature(
            _buffer_mgr=bm,
            _arch_consts=ac,
            partial_collection_ref=bm.make_handle((128,)),
            partial_offset_list_ref=bm.make_handle((4,)),
            dest_ref=bm.make_handle((32,)),
            partial_offset_list_count=np.uint32(4),
            partial_width=np.uint32(32),
            operation_type=np.uint32(0),
        )

    def test_kernel_name(self, sig):
        assert sig.kernel_name == "aggregate_local_reduce"

    def test_get_args_count(self, sig):
        # C header: 1 local + 3 global + 3 uint = 7
        assert len(sig.get_args()) == 7

    def test_get_args_type_order(self, sig):
        cats = [classify_python_arg(a) for a in sig.get_args()]
        expected = (
            [ParamCategory.LOCAL_MEM]
            + [ParamCategory.GLOBAL_BUFFER] * 3
            + [ParamCategory.UINT_SCALAR] * 3
        )
        assert cats == expected


# =========================================================================
# Nodes 15b, 20b: clip_intermediate_grad
# =========================================================================


class TestClipIntermediateGradSignature:

    @pytest.fixture
    def sig(self, bm: MockBufferManager, ac: MockArchConsts):
        return ClipIntermediateGradSignature(
            _buffer_mgr=bm,
            _arch_consts=ac,
            intermediate_grad_ref=bm.make_handle((32,)),
            clipping_threshold_t_j=np.float32(1.0),
            epsilon=np.float32(1e-6),
        )

    def test_kernel_name(self, sig):
        assert sig.kernel_name == "clip_intermediate_grad"

    def test_derived_parameter_count(self, sig):
        assert sig.parameter_count == np.uint32(32)

    def test_get_args_count(self, sig):
        # C header: 1 local + 1 global + 2 float + 1 uint = 5
        assert len(sig.get_args()) == 5

    def test_get_args_type_order(self, sig):
        cats = [classify_python_arg(a) for a in sig.get_args()]
        expected = [
            ParamCategory.LOCAL_MEM,
            ParamCategory.GLOBAL_BUFFER,
            ParamCategory.FLOAT_SCALAR,
            ParamCategory.FLOAT_SCALAR,
            ParamCategory.UINT_SCALAR,
        ]
        assert cats == expected


# =========================================================================
# Node 16: stabilize_and_reduce_grad_hidden_activations
# =========================================================================


class TestStabilizeAndReduceGradHiddenActivations:

    @pytest.fixture
    def sig(self, bm: MockBufferManager, ac: MockArchConsts):
        return StabilizeAndReduceGradHiddenActivationsSignature(
            _buffer_mgr=bm,
            _arch_consts=ac,
            permuted_soa_in_ref=bm.make_handle(
                (BATCH * HIDDEN_PADDED, PADDED_MODULES)
            ),
            final_grad_h_out_ref=bm.make_handle((BATCH, HIDDEN_PADDED)),
            fp_max=np.float32(3.4e38),
            policy_t_algorithmic=np.float32(1.0),
            policy_lambda=np.float32(0.1),
            policy_max_k=np.uint32(16),
            epsilon=np.float32(1e-6),
            total_batch_count=np.uint32(BATCH),
            padded_hidden_count=np.uint32(HIDDEN_PADDED),
            total_modules_count=np.uint32(MODULES),
            padded_total_modules_count=np.uint32(PADDED_MODULES),
        )

    def test_kernel_name(self, sig):
        assert sig.kernel_name == "stabilize_and_reduce_grad_hidden_activations"

    def test_get_args_count(self, sig):
        # C header: 1 local + 2 global + 4 float + 1 uint + 1 float + 4 uint = 13
        # Counting from C: local, global_soa_in, global_dest, fp_max, t_alg, lambda,
        #                   policy_max_k, epsilon, batch, hidden, modules, padded_modules
        # = 1 local + 2 global + 3 float + 1 uint + 1 float + 4 uint = 12
        assert len(sig.get_args()) == 12

    def test_get_args_type_order(self, sig):
        cats = [classify_python_arg(a) for a in sig.get_args()]
        expected = [
            ParamCategory.LOCAL_MEM,       # local reduction tile
            ParamCategory.GLOBAL_BUFFER,   # soa in
            ParamCategory.GLOBAL_BUFFER,   # dest
            ParamCategory.FLOAT_SCALAR,    # fp_max
            ParamCategory.FLOAT_SCALAR,    # policy_t_algorithmic
            ParamCategory.FLOAT_SCALAR,    # policy_lambda
            ParamCategory.UINT_SCALAR,     # policy_max_k
            ParamCategory.FLOAT_SCALAR,    # epsilon
            ParamCategory.UINT_SCALAR,     # total_batch_count
            ParamCategory.UINT_SCALAR,     # padded_hidden_count
            ParamCategory.UINT_SCALAR,     # total_modules_count
            ParamCategory.UINT_SCALAR,     # padded_total_modules_count
        ]
        assert cats == expected

    def test_buffer_spec_validation(self, bm: MockBufferManager, ac: MockArchConsts):
        """Plan-to-buffer inconsistency must be caught at construction time."""
        with pytest.raises(AssertionError, match="Buffer spec mismatch"):
            StabilizeAndReduceGradHiddenActivationsSignature(
                _buffer_mgr=bm,
                _arch_consts=ac,
                permuted_soa_in_ref=bm.make_handle(
                    (999, PADDED_MODULES)  # wrong row count
                ),
                final_grad_h_out_ref=bm.make_handle((BATCH, HIDDEN_PADDED)),
                fp_max=np.float32(3.4e38),
                policy_t_algorithmic=np.float32(1.0),
                policy_lambda=np.float32(0.1),
                policy_max_k=np.uint32(16),
                epsilon=np.float32(1e-6),
                total_batch_count=np.uint32(BATCH),
                padded_hidden_count=np.uint32(HIDDEN_PADDED),
                total_modules_count=np.uint32(MODULES),
                padded_total_modules_count=np.uint32(PADDED_MODULES),
            )
