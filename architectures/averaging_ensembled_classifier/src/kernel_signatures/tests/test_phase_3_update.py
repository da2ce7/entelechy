# kernel_signatures/tests/test_phase_3_update.py

"""
Unit tests for Phase 3 (Update) kernel signatures: Nodes 21, 24, 25.

Covers:
  - NormalizeGradientsSignature  (normalize_gradients)
  - AdamUpdateSignature          (adam_update)
  - ClampTemperaturesSignature   (clamp_temperatures)
"""

from __future__ import annotations

import numpy as np
import pytest

from .conftest import (
    BATCH,
    MODULES,
    MockArchConsts,
    MockBufferManager,
    ParamCategory,
    classify_python_arg,
)

from src.backends.opencl.kernel_bindings.phase_3_update import (  # type: ignore[import-not-found]
    NormalizeGradientsSignature,
    AdamParameterGroup,
    AdamUpdateSignature,
    ClampTemperaturesSignature,
)


# =========================================================================
# Node 21: normalize_gradients
# =========================================================================


class TestNormalizeGradientsSignature:

    @pytest.fixture
    def sig(self, bm: MockBufferManager, ac: MockArchConsts):
        return NormalizeGradientsSignature(
            _buffer_mgr=bm,
            _arch_consts=ac,
            summed_grad_ref=bm.make_handle((32,)),
            final_grad_out_ref=bm.make_handle((32,)),
            effective_batch_size=np.float32(BATCH),
            epsilon=np.float32(1e-6),
        )

    def test_kernel_name(self, sig):
        assert sig.kernel_name == "normalize_gradients"

    def test_derived_element_count(self, sig):
        assert sig.element_count == np.uint32(32)

    def test_get_args_count(self, sig):
        # C header: 2 global + 2 float + 1 uint = 5
        assert len(sig.get_args()) == 5

    def test_get_args_type_order(self, sig):
        cats = [classify_python_arg(a) for a in sig.get_args()]
        expected = [
            ParamCategory.GLOBAL_BUFFER,  # summed_grad
            ParamCategory.GLOBAL_BUFFER,  # final_grad
            ParamCategory.FLOAT_SCALAR,   # effective_batch_size
            ParamCategory.FLOAT_SCALAR,   # epsilon
            ParamCategory.UINT_SCALAR,    # parameter_count
        ]
        assert cats == expected

    def test_grid_1d(self, sig):
        g, l = sig.get_grid()
        assert len(g) == 1
        assert g[0] == 32
        assert l is None


# =========================================================================
# Node 24: adam_update
# =========================================================================


class TestAdamUpdateSignature:

    @pytest.fixture
    def sig(self, bm: MockBufferManager, ac: MockArchConsts):
        return AdamUpdateSignature(
            _buffer_mgr=bm,
            _arch_consts=ac,
            param_group=AdamParameterGroup(
                param_ref=bm.make_handle((32,)),
                grad_ref=bm.make_handle((32,)),
                m1_state_ref=bm.make_handle((32,)),
                m2_state_ref=bm.make_handle((32,)),
            ),
            learning_rate=np.float32(0.001),
            beta1=np.float32(0.9),
            beta2=np.float32(0.999),
            epsilon=np.float32(1e-8),
            beta1_pow_t=np.float32(0.9),
            beta2_pow_t=np.float32(0.999),
        )

    def test_kernel_name(self, sig):
        assert sig.kernel_name == "adam_update"

    def test_derived_parameter_count(self, sig):
        assert sig.parameter_count == np.uint32(32)

    def test_get_args_count(self, sig):
        # C header: 4 global + 6 float + 1 uint = 11
        assert len(sig.get_args()) == 11

    def test_get_args_type_order(self, sig):
        cats = [classify_python_arg(a) for a in sig.get_args()]
        expected = (
            [ParamCategory.GLOBAL_BUFFER] * 4   # grad, params, m1, m2
            + [ParamCategory.FLOAT_SCALAR] * 6  # lr, beta1_pow_t, beta2_pow_t, beta1, beta2, eps
            + [ParamCategory.UINT_SCALAR]        # parameter_count
        )
        assert cats == expected

    def test_grid_element_wise(self, sig):
        g, l = sig.get_grid()
        assert g == (32,)
        assert l is None


# =========================================================================
# Node 25: clamp_temperatures
# =========================================================================


class TestClampTemperaturesSignature:

    @pytest.fixture
    def sig(self, bm: MockBufferManager, ac: MockArchConsts):
        return ClampTemperaturesSignature(
            _buffer_mgr=bm,
            _arch_consts=ac,
            temps_ref=bm.make_handle((MODULES,)),
            min_val=np.float32(0.1),
            max_val=np.float32(10.0),
        )

    def test_kernel_name(self, sig):
        assert sig.kernel_name == "clamp_temperatures"

    def test_derived_element_count(self, sig):
        assert sig.element_count == np.uint32(MODULES)

    def test_get_args_count(self, sig):
        # C header: 1 global + 2 float + 1 uint = 4
        assert len(sig.get_args()) == 4

    def test_get_args_type_order(self, sig):
        cats = [classify_python_arg(a) for a in sig.get_args()]
        expected = [
            ParamCategory.GLOBAL_BUFFER,  # temps
            ParamCategory.FLOAT_SCALAR,   # min_value
            ParamCategory.FLOAT_SCALAR,   # max_value
            ParamCategory.UINT_SCALAR,    # total_modules_count
        ]
        assert cats == expected

    def test_grid_element_wise(self, sig):
        g, l = sig.get_grid()
        assert g == (MODULES,)
        assert l is None
