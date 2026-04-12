# tests/oracle/test_precision_oracle.py
"""Oracle D (NumPy Precision Oracle) test matrix.

Implements the comparison axes between Oracle D and Oracles A/B/C:

  1. D (FP64) vs C — forward parity: at FP64 precision with no storage
     narrowing, D and C must agree on loss and probabilities.
  2. D (FP64) vs C — multi-step parity: at FP64, parameter trajectories
     must match (no clipping in either oracle).
  3. D self-test — known-solution convergence: D alone must reduce loss
     on separable problems across all precision configurations.
  4. D precision-gap isolation — FP32 vs FP16 vs FP64 trajectories:
     verify that narrower storage produces measurably different (but
     still convergent) trajectories.
  5. D mask strategy — explicit vs recompute consistency: at FP32 (where
     both strategies should agree) and FP8 (where they diverge).

Oracle D is unique among the oracles: it has **no torch dependency**.
Its authority is precision-limited convergence rate.  When D and C agree
at FP64, D's precision-parameterized variants provide the ground truth
for what the engine should achieve under each precision configuration.

Reference: doc_archive/Oracle.md §Option D: NumPy Precision Oracle
"""
from __future__ import annotations

from typing import TYPE_CHECKING, Any

import numpy as np
import pytest

if TYPE_CHECKING:
    import torch
else:
    torch = pytest.importorskip("torch", reason="Oracle D cross-validation requires PyTorch")

from .conftest import (
    _generate_xor_data,
    _loss_close,
    init_with_seed,
    make_unclipped_config,
    oracle_config_to_d_config,
    sync_oracle_d_from_torch,
)
from .convergence_oracle import ConvergenceOracle
from .oracle_config import OracleConfig
from .precision_oracle import (
    NumpyPrecisionSpec,
    NumPyPrecisionOracle,
    OracleDConfig,
)


# =====================================================================
# Tolerance constants
# =====================================================================

# D vs C, both at FP64, single step — limited by NumPy vs PyTorch
# accumulation order and transcendental function implementations.
ATOL_D_VS_C = 1e-9
RTOL_D_VS_C = 1e-9

# D vs C, multi-step drift tolerance grows with steps.
ATOL_MULTI_STEP_BASE = 1e-9

# D self-convergence thresholds.
CONVERGENCE_LOSS_RATIO = 0.5  # final/initial must be < this


# =====================================================================
# Helpers
# =====================================================================


def _make_oracle_d_fp64(
    cfg: OracleConfig,
    **kwargs: Any,
) -> NumPyPrecisionOracle:
    """Create Oracle D at FP64 (matching Oracles A/B/C's precision)."""
    d_cfg = oracle_config_to_d_config(cfg)
    return NumPyPrecisionOracle(
        d_cfg,
        NumpyPrecisionSpec.float64(),
        model_gradient_storage=False,
        model_logit_storage=False,
        model_probability_storage=False,
        **kwargs,
    )


def _sync_d_from_c(
    oracle_d: NumPyPrecisionOracle,
    oracle_c: ConvergenceOracle,
) -> None:
    """Sync Oracle D state from Oracle C."""
    sync_oracle_d_from_torch(oracle_d, oracle_c.export_state())


def _torch_to_numpy(X: torch.Tensor) -> np.ndarray:
    return X.detach().cpu().numpy()


# =====================================================================
# 1. D (FP64) vs C — Forward parity (single step)
# =====================================================================


class TestDvsCForwardParity:
    """At FP64 with no storage narrowing, D and C must agree.

    Oracle D uses NumPy; Oracle C uses PyTorch.  Both operate at FP64
    with no clipping.  Differences arise only from accumulation order
    and transcendental function implementations.
    """

    def test_loss_parity_cce(
        self,
        small_cce_config: OracleConfig,
        small_cce_data: tuple[torch.Tensor, torch.Tensor],
    ) -> None:
        cfg = make_unclipped_config(small_cce_config)
        oracle_c = ConvergenceOracle(cfg)
        oracle_d = _make_oracle_d_fp64(cfg)
        init_with_seed(oracle_c)
        _sync_d_from_c(oracle_d, oracle_c)

        X_t, targets_t = small_cce_data
        X_np = _torch_to_numpy(X_t)
        targets_np = _torch_to_numpy(targets_t)

        oracle_c.step(X_t, targets_t)
        oracle_d.step(X_np, targets_np)

        assert _loss_close(oracle_c.last_loss, oracle_d.last_loss, atol=ATOL_D_VS_C, rtol=RTOL_D_VS_C), (
            f"Loss mismatch: C={oracle_c.last_loss}, D={oracle_d.last_loss}"
        )

    def test_loss_parity_bce(
        self,
        small_bce_config: OracleConfig,
        small_bce_data: tuple[torch.Tensor, torch.Tensor],
    ) -> None:
        cfg = make_unclipped_config(small_bce_config)
        oracle_c = ConvergenceOracle(cfg)
        oracle_d = _make_oracle_d_fp64(cfg)
        init_with_seed(oracle_c)
        _sync_d_from_c(oracle_d, oracle_c)

        X_t, targets_t = small_bce_data
        X_np = _torch_to_numpy(X_t)
        targets_np = _torch_to_numpy(targets_t)

        oracle_c.step(X_t, targets_t)
        oracle_d.step(X_np, targets_np)

        assert _loss_close(oracle_c.last_loss, oracle_d.last_loss, atol=ATOL_D_VS_C, rtol=RTOL_D_VS_C), (
            f"Loss mismatch: C={oracle_c.last_loss}, D={oracle_d.last_loss}"
        )

    def test_loss_parity_xor(
        self,
        xor_config: OracleConfig,
        xor_data: tuple[torch.Tensor, torch.Tensor],
    ) -> None:
        cfg = make_unclipped_config(xor_config)
        oracle_c = ConvergenceOracle(cfg)
        oracle_d = _make_oracle_d_fp64(cfg)
        init_with_seed(oracle_c)
        _sync_d_from_c(oracle_d, oracle_c)

        X_t, targets_t = xor_data
        X_np = _torch_to_numpy(X_t)
        targets_np = _torch_to_numpy(targets_t)

        oracle_c.step(X_t, targets_t)
        oracle_d.step(X_np, targets_np)

        assert _loss_close(oracle_c.last_loss, oracle_d.last_loss, atol=ATOL_D_VS_C, rtol=RTOL_D_VS_C), (
            f"Loss mismatch: C={oracle_c.last_loss}, D={oracle_d.last_loss}"
        )

    def test_param_parity_cce(
        self,
        small_cce_config: OracleConfig,
        small_cce_data: tuple[torch.Tensor, torch.Tensor],
    ) -> None:
        """After one step, D and C parameters must match at FP64."""
        cfg = make_unclipped_config(small_cce_config)
        oracle_c = ConvergenceOracle(cfg)
        oracle_d = _make_oracle_d_fp64(cfg)
        init_with_seed(oracle_c)
        _sync_d_from_c(oracle_d, oracle_c)

        X_t, targets_t = small_cce_data
        oracle_c.step(X_t, targets_t)
        oracle_d.step(_torch_to_numpy(X_t), _torch_to_numpy(targets_t))

        state_c = oracle_c.export_state()
        state_d = oracle_d.export_state()

        for name in ("W_shared", "b_shared", "W_module", "b_module", "temps"):
            c_vals = state_c[name].detach().cpu().numpy()
            d_vals = state_d[name]
            max_diff = float(np.max(np.abs(c_vals - d_vals)))
            assert max_diff < ATOL_D_VS_C, (
                f"Param {name} mismatch after 1 step: max_diff={max_diff:.2e}"
            )

    def test_param_parity_xor(
        self,
        xor_config: OracleConfig,
        xor_data: tuple[torch.Tensor, torch.Tensor],
    ) -> None:
        """After one step, D and C parameters must match at FP64 on XOR."""
        cfg = make_unclipped_config(xor_config)
        oracle_c = ConvergenceOracle(cfg)
        oracle_d = _make_oracle_d_fp64(cfg)
        init_with_seed(oracle_c)
        _sync_d_from_c(oracle_d, oracle_c)

        X_t, targets_t = xor_data
        oracle_c.step(X_t, targets_t)
        oracle_d.step(_torch_to_numpy(X_t), _torch_to_numpy(targets_t))

        state_c = oracle_c.export_state()
        state_d = oracle_d.export_state()

        for name in ("W_shared", "b_shared", "W_module", "b_module", "temps"):
            c_vals = state_c[name].detach().cpu().numpy()
            d_vals = state_d[name]
            max_diff = float(np.max(np.abs(c_vals - d_vals)))
            assert max_diff < ATOL_D_VS_C, (
                f"Param {name} mismatch after 1 step: max_diff={max_diff:.2e}"
            )


# =====================================================================
# 2. D (FP64) vs C — Multi-step trajectory parity
# =====================================================================


class TestDvsCMultiStep:
    """Multi-step trajectory parity at FP64, no clipping.

    Over N steps on a fixed dataset, D and C must track within a
    tolerance that grows slowly with step count (accumulated
    per-step accumulation-order divergence).
    """

    @pytest.mark.parametrize("steps", [5, 50])
    def test_trajectory_parity_cce(
        self,
        small_cce_config: OracleConfig,
        small_cce_data: tuple[torch.Tensor, torch.Tensor],
        steps: int,
    ) -> None:
        cfg = make_unclipped_config(small_cce_config)
        oracle_c = ConvergenceOracle(cfg)
        oracle_d = _make_oracle_d_fp64(cfg)
        init_with_seed(oracle_c)
        _sync_d_from_c(oracle_d, oracle_c)

        X_t, targets_t = small_cce_data
        X_np, targets_np = _torch_to_numpy(X_t), _torch_to_numpy(targets_t)

        for _ in range(steps):
            oracle_c.step(X_t, targets_t)
            oracle_d.step(X_np, targets_np)

        # Tolerance grows with sqrt(steps) for accumulation drift
        tol = ATOL_MULTI_STEP_BASE * (1 + steps**0.5)
        state_c = oracle_c.export_state()
        state_d = oracle_d.export_state()

        for name in ("W_shared", "b_shared", "W_module", "b_module", "temps"):
            c_vals = state_c[name].detach().cpu().numpy()
            d_vals = state_d[name]
            max_diff = float(np.max(np.abs(c_vals - d_vals)))
            assert max_diff < tol, (
                f"D vs C drift on {name} after {steps} steps: {max_diff:.2e} > {tol:.2e}"
            )

    @pytest.mark.parametrize("steps", [5, 50])
    def test_trajectory_parity_bce(
        self,
        small_bce_config: OracleConfig,
        small_bce_data: tuple[torch.Tensor, torch.Tensor],
        steps: int,
    ) -> None:
        cfg = make_unclipped_config(small_bce_config)
        oracle_c = ConvergenceOracle(cfg)
        oracle_d = _make_oracle_d_fp64(cfg)
        init_with_seed(oracle_c)
        _sync_d_from_c(oracle_d, oracle_c)

        X_t, targets_t = small_bce_data
        X_np, targets_np = _torch_to_numpy(X_t), _torch_to_numpy(targets_t)

        for _ in range(steps):
            oracle_c.step(X_t, targets_t)
            oracle_d.step(X_np, targets_np)

        tol = ATOL_MULTI_STEP_BASE * (1 + steps**0.5)
        state_c = oracle_c.export_state()
        state_d = oracle_d.export_state()

        for name in ("W_shared", "b_shared", "W_module", "b_module", "temps"):
            c_vals = state_c[name].detach().cpu().numpy()
            d_vals = state_d[name]
            max_diff = float(np.max(np.abs(c_vals - d_vals)))
            assert max_diff < tol, (
                f"D vs C drift on {name} after {steps} steps: {max_diff:.2e} > {tol:.2e}"
            )

    @pytest.mark.parametrize("steps", [5, 50])
    def test_trajectory_parity_xor(
        self,
        xor_config: OracleConfig,
        xor_data: tuple[torch.Tensor, torch.Tensor],
        steps: int,
    ) -> None:
        cfg = make_unclipped_config(xor_config)
        oracle_c = ConvergenceOracle(cfg)
        oracle_d = _make_oracle_d_fp64(cfg)
        init_with_seed(oracle_c)
        _sync_d_from_c(oracle_d, oracle_c)

        X_t, targets_t = xor_data
        X_np, targets_np = _torch_to_numpy(X_t), _torch_to_numpy(targets_t)

        for _ in range(steps):
            oracle_c.step(X_t, targets_t)
            oracle_d.step(X_np, targets_np)

        tol = ATOL_MULTI_STEP_BASE * (1 + steps**0.5)
        state_c = oracle_c.export_state()
        state_d = oracle_d.export_state()

        for name in ("W_shared", "b_shared", "W_module", "b_module", "temps"):
            c_vals = state_c[name].detach().cpu().numpy()
            d_vals = state_d[name]
            max_diff = float(np.max(np.abs(c_vals - d_vals)))
            assert max_diff < tol, (
                f"D vs C drift on {name} after {steps} steps: {max_diff:.2e} > {tol:.2e}"
            )


# =====================================================================
# 3. D self-test — known-solution convergence per precision
# =====================================================================


PRECISION_CONFIGS_CONVERGENCE = [
    pytest.param(NumpyPrecisionSpec.float64(), id="fp64"),
    pytest.param(NumpyPrecisionSpec.float32(), id="fp32"),
    pytest.param(NumpyPrecisionSpec.mixed_f16_f32(), id="f16_f32"),
    pytest.param(NumpyPrecisionSpec.mixed_f32_f64_state(), id="f32_f64state"),
    pytest.param(NumpyPrecisionSpec.mixed_f32_f64(), id="f32_f64"),
]


class TestKnownSolutionConvergencePrecision:
    """Oracle D must converge on separable problems for each precision config.

    This is D's primary authority: for a given precision spec, what
    convergence trajectory should we expect?
    """

    @pytest.mark.parametrize("prec", PRECISION_CONFIGS_CONVERGENCE)
    def test_separable_cce(self, prec: NumpyPrecisionSpec) -> None:
        num_classes = 3
        input_dim = 8
        d_cfg = OracleDConfig(
            input_dim=input_dim,
            hidden_dim=16,
            output_classes=num_classes,
            num_modules=2,
            mode="CCE",
            learning_rate=0.01,
        )
        oracle_d = NumPyPrecisionOracle.with_xavier_init(d_cfg, prec, seed=7)

        X, targets = NumPyPrecisionOracle.make_separable_clusters_cce(
            num_classes=num_classes,
            input_dim=input_dim,
            num_samples_per_class=30,
            separation=3.0,
            seed=42,
        )

        trace = oracle_d.train_n_steps(X, targets, n=500, record_every=10)

        assert trace.converged, (
            f"CCE/{prec} failed to converge: "
            f"initial={trace.initial_loss:.4f}, final={trace.final_loss:.4f}"
        )
        assert trace.is_stable, f"NaN/Inf during CCE/{prec} convergence"
        assert trace.loss_reduction_ratio < CONVERGENCE_LOSS_RATIO, (
            f"CCE/{prec} loss ratio {trace.loss_reduction_ratio:.4f} >= {CONVERGENCE_LOSS_RATIO}"
        )

    @pytest.mark.parametrize("prec", PRECISION_CONFIGS_CONVERGENCE)
    def test_independent_bce(self, prec: NumpyPrecisionSpec) -> None:
        num_classes = 4
        input_dim = 8
        d_cfg = OracleDConfig(
            input_dim=input_dim,
            hidden_dim=16,
            output_classes=num_classes,
            num_modules=2,
            mode="BCE",
            learning_rate=0.01,
        )
        oracle_d = NumPyPrecisionOracle.with_xavier_init(d_cfg, prec, seed=7)

        X, targets = NumPyPrecisionOracle.make_independent_bce_no_scipy(
            num_classes=num_classes,
            input_dim=input_dim,
            num_samples=100,
            seed=42,
        )

        trace = oracle_d.train_n_steps(X, targets, n=500, record_every=10)

        assert trace.converged, (
            f"BCE/{prec} failed to converge: "
            f"initial={trace.initial_loss:.4f}, final={trace.final_loss:.4f}"
        )
        assert trace.is_stable, f"NaN/Inf during BCE/{prec} convergence"
        assert trace.loss_reduction_ratio < CONVERGENCE_LOSS_RATIO, (
            f"BCE/{prec} loss ratio {trace.loss_reduction_ratio:.4f} >= {CONVERGENCE_LOSS_RATIO}"
        )

    @pytest.mark.parametrize("prec", PRECISION_CONFIGS_CONVERGENCE)
    def test_xor_bce(self, prec: NumpyPrecisionSpec) -> None:
        """Oracle D must converge on XOR at each precision config."""
        d_cfg = OracleDConfig(
            input_dim=2,
            hidden_dim=16,
            output_classes=1,
            num_modules=8,
            mode="BCE",
            learning_rate=0.01,
        )
        oracle_d = NumPyPrecisionOracle.with_xavier_init(d_cfg, prec, seed=7)

        X, targets = _generate_xor_data()

        trace = oracle_d.train_n_steps(X, targets, n=500, record_every=10)

        assert trace.converged, (
            f"XOR/{prec} failed to converge: "
            f"initial={trace.initial_loss:.4f}, final={trace.final_loss:.4f}"
        )
        assert trace.is_stable, f"NaN/Inf during XOR/{prec} convergence"
        assert trace.loss_reduction_ratio < CONVERGENCE_LOSS_RATIO, (
            f"XOR/{prec} loss ratio {trace.loss_reduction_ratio:.4f} >= {CONVERGENCE_LOSS_RATIO}"
        )


# =====================================================================
# 4. Precision-gap isolation — trajectory divergence across precisions
# =====================================================================


class TestPrecisionGapIsolation:
    """Narrower storage formats produce measurably different trajectories.

    Confirms that D's precision modeling is non-trivial: FP16 storage
    should produce a different (typically worse) loss curve than FP32,
    which in turn differs from FP64.
    """

    def test_cce_fp16_vs_fp64_diverges(self) -> None:
        """FP16 storage trajectory must differ measurably from FP64."""
        num_classes = 3
        input_dim = 8
        d_cfg = OracleDConfig(
            input_dim=input_dim,
            hidden_dim=16,
            output_classes=num_classes,
            num_modules=2,
            mode="CCE",
            learning_rate=0.01,
        )
        X, targets = NumPyPrecisionOracle.make_separable_clusters_cce(
            num_classes=num_classes, input_dim=input_dim,
            num_samples_per_class=30, separation=3.0, seed=42,
        )

        oracle_fp64 = NumPyPrecisionOracle.with_xavier_init(
            d_cfg, NumpyPrecisionSpec.float64(), seed=7,
        )
        oracle_fp16 = NumPyPrecisionOracle.with_xavier_init(
            d_cfg, NumpyPrecisionSpec.mixed_f16_f32(), seed=7,
        )

        trace_64 = oracle_fp64.train_n_steps(X, targets, n=200, record_every=10)
        trace_16 = oracle_fp16.train_n_steps(X, targets, n=200, record_every=10)

        # Both must converge
        assert trace_64.converged, "FP64 baseline failed to converge"
        assert trace_16.converged, "FP16 failed to converge"

        # Trajectories must differ — FP16 should have higher final loss
        # or different per-step behavior. At minimum the loss curves
        # shouldn't be identical.
        assert trace_64.final_loss != trace_16.final_loss, (
            "FP64 and FP16 produced identical final loss — "
            "precision modeling may be ineffective"
        )

    def test_cce_fp32_vs_fp64_state_diverges(self) -> None:
        """FP32 state vs FP64 state: state precision affects optimizer trajectory."""
        num_classes = 3
        input_dim = 8
        d_cfg = OracleDConfig(
            input_dim=input_dim,
            hidden_dim=16,
            output_classes=num_classes,
            num_modules=2,
            mode="CCE",
            learning_rate=0.01,
        )
        X, targets = NumPyPrecisionOracle.make_separable_clusters_cce(
            num_classes=num_classes, input_dim=input_dim,
            num_samples_per_class=30, separation=3.0, seed=42,
        )

        oracle_f32 = NumPyPrecisionOracle.with_xavier_init(
            d_cfg, NumpyPrecisionSpec.float32(), seed=7,
        )
        oracle_f64s = NumPyPrecisionOracle.with_xavier_init(
            d_cfg, NumpyPrecisionSpec.mixed_f32_f64_state(), seed=7,
        )

        trace_32 = oracle_f32.train_n_steps(X, targets, n=200, record_every=10)
        trace_64s = oracle_f64s.train_n_steps(X, targets, n=200, record_every=10)

        assert trace_32.converged, "FP32 failed to converge"
        assert trace_64s.converged, "FP32/FP64-state failed to converge"

        # The mechanistic effect of f64 state precision is higher fidelity
        # in moment accumulation and parameter updates.  This manifests in
        # the parameters themselves, not necessarily in the f32-observed
        # loss (which can converge to bitwise-identical values on easy
        # problems).  Assert that at least one parameter array diverges
        # when compared at full f64 precision.
        any_diverged = any(
            not np.array_equal(
                p32.astype(np.float64),
                p64s.astype(np.float64),
            )
            for (_, p32), (_, p64s) in zip(
                oracle_f32._named_params(),
                oracle_f64s._named_params(),
            )
        )
        assert any_diverged, (
            "FP32 and FP32/FP64-state produced identical parameters — "
            "state precision had no effect"
        )

    def test_xor_fp16_vs_fp64_diverges(self) -> None:
        """XOR BCE: FP16 storage trajectory must differ from FP64."""
        d_cfg = OracleDConfig(
            input_dim=2, hidden_dim=16, output_classes=1,
            num_modules=8, mode="BCE", learning_rate=0.01,
        )
        X, targets = _generate_xor_data()

        oracle_fp64 = NumPyPrecisionOracle.with_xavier_init(
            d_cfg, NumpyPrecisionSpec.float64(), seed=7,
        )
        oracle_fp16 = NumPyPrecisionOracle.with_xavier_init(
            d_cfg, NumpyPrecisionSpec.mixed_f16_f32(), seed=7,
        )

        trace_64 = oracle_fp64.train_n_steps(X, targets, n=200, record_every=10)
        trace_16 = oracle_fp16.train_n_steps(X, targets, n=200, record_every=10)

        assert trace_64.converged, "FP64 baseline failed to converge on XOR"
        assert trace_16.converged, "FP16 failed to converge on XOR"

        assert trace_64.final_loss != trace_16.final_loss, (
            "FP64 and FP16 produced identical final loss on XOR — "
            "precision modeling may be ineffective"
        )


# =====================================================================
# 5. Mask strategy — explicit vs recompute
# =====================================================================


class TestMaskStrategy:
    """At FP32/FP64 (where all positive activations survive storage),
    explicit and recompute mask strategies must agree.
    """

    @pytest.mark.parametrize(
        "prec",
        [
            pytest.param(NumpyPrecisionSpec.float32(), id="fp32"),
            pytest.param(NumpyPrecisionSpec.float64(), id="fp64"),
        ],
    )
    def test_strategies_agree_cce(self, prec: NumpyPrecisionSpec) -> None:
        num_classes = 3
        input_dim = 8
        d_cfg = OracleDConfig(
            input_dim=input_dim,
            hidden_dim=16,
            output_classes=num_classes,
            num_modules=2,
            mode="CCE",
            learning_rate=0.01,
        )

        oracle_explicit = NumPyPrecisionOracle.with_xavier_init(
            d_cfg, prec, seed=7, mask_strategy="explicit",
        )
        oracle_recompute = NumPyPrecisionOracle.with_xavier_init(
            d_cfg, prec, seed=7, mask_strategy="recompute",
        )

        X, targets = NumPyPrecisionOracle.make_separable_clusters_cce(
            num_classes=num_classes, input_dim=input_dim,
            num_samples_per_class=20, separation=3.0, seed=42,
        )

        trace_e = oracle_explicit.train_n_steps(X, targets, n=50, record_every=10)
        trace_r = oracle_recompute.train_n_steps(X, targets, n=50, record_every=10)

        for i, (le, lr) in enumerate(
            zip(trace_e.loss_history, trace_r.loss_history),
        ):
            assert abs(le - lr) < 1e-10, (
                f"Mask strategies diverged at step {i}: "
                f"explicit={le:.6e}, recompute={lr:.6e}"
            )

    @pytest.mark.parametrize(
        "prec",
        [
            pytest.param(NumpyPrecisionSpec.float32(), id="fp32"),
            pytest.param(NumpyPrecisionSpec.float64(), id="fp64"),
        ],
    )
    def test_strategies_agree_bce(self, prec: NumpyPrecisionSpec) -> None:
        num_classes = 4
        input_dim = 8
        d_cfg = OracleDConfig(
            input_dim=input_dim,
            hidden_dim=16,
            output_classes=num_classes,
            num_modules=2,
            mode="BCE",
            learning_rate=0.01,
        )

        oracle_explicit = NumPyPrecisionOracle.with_xavier_init(
            d_cfg, prec, seed=7, mask_strategy="explicit",
        )
        oracle_recompute = NumPyPrecisionOracle.with_xavier_init(
            d_cfg, prec, seed=7, mask_strategy="recompute",
        )

        X, targets = NumPyPrecisionOracle.make_independent_bce_no_scipy(
            num_classes=num_classes, input_dim=input_dim,
            num_samples=100, seed=42,
        )

        trace_e = oracle_explicit.train_n_steps(X, targets, n=50, record_every=10)
        trace_r = oracle_recompute.train_n_steps(X, targets, n=50, record_every=10)

        for i, (le, lr) in enumerate(
            zip(trace_e.loss_history, trace_r.loss_history),
        ):
            assert abs(le - lr) < 1e-10, (
                f"Mask strategies diverged at step {i}: "
                f"explicit={le:.6e}, recompute={lr:.6e}"
            )

    @pytest.mark.parametrize(
        "prec",
        [
            pytest.param(NumpyPrecisionSpec.float32(), id="fp32"),
            pytest.param(NumpyPrecisionSpec.float64(), id="fp64"),
        ],
    )
    def test_strategies_agree_xor(self, prec: NumpyPrecisionSpec) -> None:
        """XOR BCE: explicit and recompute mask strategies must agree."""
        d_cfg = OracleDConfig(
            input_dim=2, hidden_dim=16, output_classes=1,
            num_modules=8, mode="BCE", learning_rate=0.01,
        )

        oracle_explicit = NumPyPrecisionOracle.with_xavier_init(
            d_cfg, prec, seed=7, mask_strategy="explicit",
        )
        oracle_recompute = NumPyPrecisionOracle.with_xavier_init(
            d_cfg, prec, seed=7, mask_strategy="recompute",
        )

        X, targets = _generate_xor_data()

        trace_e = oracle_explicit.train_n_steps(X, targets, n=50, record_every=10)
        trace_r = oracle_recompute.train_n_steps(X, targets, n=50, record_every=10)

        for i, (le, lr) in enumerate(
            zip(trace_e.loss_history, trace_r.loss_history),
        ):
            assert abs(le - lr) < 1e-10, (
                f"Mask strategies diverged at step {i} on XOR: "
                f"explicit={le:.6e}, recompute={lr:.6e}"
            )


# =====================================================================
# 6. Gradient storage modeling
# =====================================================================


class TestGradientStorageModeling:
    """Gradient storage round-trips should be a no-op at FP32/FP64 but
    produce measurable effects at FP16 storage.
    """

    def test_gradient_storage_noop_fp64(self) -> None:
        """At FP64, enabling gradient storage modeling changes nothing."""
        d_cfg = OracleDConfig(
            input_dim=8, hidden_dim=16, output_classes=3,
            num_modules=2, mode="CCE", learning_rate=0.01,
        )
        X, targets = NumPyPrecisionOracle.make_separable_clusters_cce(
            num_classes=3, input_dim=8,
            num_samples_per_class=20, separation=3.0, seed=42,
        )

        prec = NumpyPrecisionSpec.float64()
        oracle_off = NumPyPrecisionOracle.with_xavier_init(
            d_cfg, prec, seed=7, model_gradient_storage=False,
        )
        oracle_on = NumPyPrecisionOracle.with_xavier_init(
            d_cfg, prec, seed=7, model_gradient_storage=True,
        )

        trace_off = oracle_off.train_n_steps(X, targets, n=20, record_every=5)
        trace_on = oracle_on.train_n_steps(X, targets, n=20, record_every=5)

        for lo, ln in zip(trace_off.loss_history, trace_on.loss_history):
            assert abs(lo - ln) < 1e-12, (
                f"Gradient storage changed FP64 loss: off={lo:.6e}, on={ln:.6e}"
            )

    def test_gradient_storage_affects_fp16(self) -> None:
        """At FP16 storage, gradient storage round-trip changes the trajectory."""
        d_cfg = OracleDConfig(
            input_dim=8, hidden_dim=16, output_classes=3,
            num_modules=2, mode="CCE", learning_rate=0.01,
        )
        X, targets = NumPyPrecisionOracle.make_separable_clusters_cce(
            num_classes=3, input_dim=8,
            num_samples_per_class=20, separation=3.0, seed=42,
        )

        prec = NumpyPrecisionSpec.mixed_f16_f32()
        oracle_off = NumPyPrecisionOracle.with_xavier_init(
            d_cfg, prec, seed=7, model_gradient_storage=False,
        )
        oracle_on = NumPyPrecisionOracle.with_xavier_init(
            d_cfg, prec, seed=7, model_gradient_storage=True,
        )

        trace_off = oracle_off.train_n_steps(X, targets, n=100, record_every=10)
        trace_on = oracle_on.train_n_steps(X, targets, n=100, record_every=10)

        # Both should converge
        assert trace_off.converged, "FP16 without gradient storage failed"
        assert trace_on.converged, "FP16 with gradient storage failed"

        # Trajectories should differ
        assert trace_off.final_loss != trace_on.final_loss, (
            "Gradient storage round-trip had no effect at FP16"
        )

    def test_gradient_storage_noop_xor_fp64(self) -> None:
        """At FP64, enabling gradient storage modeling changes nothing on XOR."""
        d_cfg = OracleDConfig(
            input_dim=2, hidden_dim=16, output_classes=1,
            num_modules=8, mode="BCE", learning_rate=0.01,
        )
        X, targets = _generate_xor_data()

        prec = NumpyPrecisionSpec.float64()
        oracle_off = NumPyPrecisionOracle.with_xavier_init(
            d_cfg, prec, seed=7, model_gradient_storage=False,
        )
        oracle_on = NumPyPrecisionOracle.with_xavier_init(
            d_cfg, prec, seed=7, model_gradient_storage=True,
        )

        trace_off = oracle_off.train_n_steps(X, targets, n=20, record_every=5)
        trace_on = oracle_on.train_n_steps(X, targets, n=20, record_every=5)

        for lo, ln in zip(trace_off.loss_history, trace_on.loss_history):
            assert abs(lo - ln) < 1e-12, (
                f"Gradient storage changed FP64 loss on XOR: off={lo:.6e}, on={ln:.6e}"
            )

    def test_gradient_storage_affects_xor_fp16(self) -> None:
        """At FP16 storage, gradient storage round-trip changes the XOR trajectory."""
        d_cfg = OracleDConfig(
            input_dim=2, hidden_dim=16, output_classes=1,
            num_modules=8, mode="BCE", learning_rate=0.01,
        )
        X, targets = _generate_xor_data()

        prec = NumpyPrecisionSpec.mixed_f16_f32()
        oracle_off = NumPyPrecisionOracle.with_xavier_init(
            d_cfg, prec, seed=7, model_gradient_storage=False,
        )
        oracle_on = NumPyPrecisionOracle.with_xavier_init(
            d_cfg, prec, seed=7, model_gradient_storage=True,
        )

        trace_off = oracle_off.train_n_steps(X, targets, n=100, record_every=10)
        trace_on = oracle_on.train_n_steps(X, targets, n=100, record_every=10)

        assert trace_off.converged, "FP16 without gradient storage failed on XOR"
        assert trace_on.converged, "FP16 with gradient storage failed on XOR"

        assert trace_off.final_loss != trace_on.final_loss, (
            "Gradient storage round-trip had no effect at FP16 on XOR"
        )
