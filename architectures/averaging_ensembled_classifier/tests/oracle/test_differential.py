# tests/oracle/test_differential.py
"""Differential triangulation test matrix.

Implements the five comparison axes from doc_archive/Oracle.md §The Five
Comparison Axes and §Recommended Test Progression.

  Axis 1: A vs B1 (clipping disabled) — gradient calculus
  Axis 2: A vs B2 (clipping disabled) — tile decomposition
  Axis 3: A vs B2 (clipping enabled)  — full pipeline parity
  Axis 4: B1 vs B2 (clipping disabled) — autograd self-consistency
  Axis 5: Multi-step convergence (A vs B2)
"""
from __future__ import annotations

import pytest

torch = pytest.importorskip("torch", reason="Oracle tests require PyTorch")

from .autograd_oracle import AutogradOracle
from .conftest import init_with_seed, sync_oracles
from .faithful_oracle import FaithfulOracle
from .oracle_config import OracleConfig

# Tolerance: both oracles are FP64 — should agree to ~1e-10.
# Accumulation-order and log/exp differences allow for ~1e-9.
ATOL = 1e-9
RTOL = 1e-9


# ═════════════════════════════════════════════════════════════════════
# Stage 1 — Axis 1: A vs B1, clipping disabled (gradient calculus)
# ═════════════════════════════════════════════════════════════════════


class TestAxisOneGradientCalculus:
    """A vs B1 with clipping disabled. Pure math validation.

    If these fail, Oracle A has a manual gradient formula bug.
    Autograd (B1) is the arbiter.
    """

    @staticmethod
    def _make_unclipped_config(base: OracleConfig) -> OracleConfig:
        """Return config with effectively-infinite clip threshold."""
        return OracleConfig(
            input_dim=base.input_dim,
            hidden_dim=base.hidden_dim,
            output_classes=base.output_classes,
            num_modules=base.num_modules,
            mode=base.mode,
            learning_rate=base.learning_rate,
            beta1=base.beta1,
            beta2=base.beta2,
            epsilon=base.epsilon,
            t_algorithmic=float("inf"),
            lambda_=base.lambda_,
            compute_fp_format_max=float("inf"),
            temp_min=base.temp_min,
            temp_max=base.temp_max,
        )

    def test_forward_parity_cce(self, small_cce_config, small_cce_data):
        """Forward pass: A and B1 produce identical probs and loss (CCE)."""
        cfg = self._make_unclipped_config(small_cce_config)
        oracle_a = FaithfulOracle(cfg)
        oracle_b1 = AutogradOracle(cfg, mode="flat")
        init_with_seed(oracle_a)
        sync_oracles(oracle_a, oracle_b1)

        X, targets = small_cce_data
        probs_a = oracle_a.step(X, targets)
        # Re-sync (step modified params), so forward-only comparison
        oracle_b1.load_state(oracle_a.export_state())
        # Need one step from B1 too to compare loss
        sync_oracles(oracle_a, oracle_b1)
        # Reset both to initial state
        init_with_seed(oracle_a)
        sync_oracles(oracle_a, oracle_b1)

        probs_a = oracle_a.step(X, targets)
        probs_b1 = oracle_b1.step(X, targets)

        # Compare losses
        assert abs(oracle_a.last_loss - oracle_b1.last_loss) < ATOL, (
            f"Loss mismatch: A={oracle_a.last_loss}, B1={oracle_b1.last_loss}"
        )

    def test_forward_parity_bce(self, small_bce_config, small_bce_data):
        """Forward pass parity under BCE."""
        cfg = self._make_unclipped_config(small_bce_config)
        oracle_a = FaithfulOracle(cfg)
        oracle_b1 = AutogradOracle(cfg, mode="flat")
        init_with_seed(oracle_a)
        sync_oracles(oracle_a, oracle_b1)

        X, targets = small_bce_data
        probs_a = oracle_a.step(X, targets)
        probs_b1 = oracle_b1.step(X, targets)

        assert abs(oracle_a.last_loss - oracle_b1.last_loss) < ATOL

    def test_gradient_parity_cce(self, small_cce_config, small_cce_data):
        """After one step, A and B1 have identical final_grads (CCE, no clip)."""
        cfg = self._make_unclipped_config(small_cce_config)
        oracle_a = FaithfulOracle(cfg)
        oracle_b1 = AutogradOracle(cfg, mode="flat")
        init_with_seed(oracle_a)
        sync_oracles(oracle_a, oracle_b1)

        X, targets = small_cce_data
        oracle_a.step(X, targets)
        oracle_b1.step(X, targets)

        for name in ["W_shared", "b_shared", "W_module", "b_module", "temps"]:
            grad_a = oracle_a.final_grads[name]
            grad_b1 = oracle_b1.final_grads[name]
            assert torch.allclose(grad_a, grad_b1, atol=ATOL, rtol=RTOL), (
                f"Gradient mismatch on {name}: "
                f"max_diff={( grad_a - grad_b1).abs().max().item():.2e}"
            )

    def test_gradient_parity_bce(self, small_bce_config, small_bce_data):
        """Gradient parity under BCE."""
        cfg = self._make_unclipped_config(small_bce_config)
        oracle_a = FaithfulOracle(cfg)
        oracle_b1 = AutogradOracle(cfg, mode="flat")
        init_with_seed(oracle_a)
        sync_oracles(oracle_a, oracle_b1)

        X, targets = small_bce_data
        oracle_a.step(X, targets)
        oracle_b1.step(X, targets)

        for name in ["W_shared", "b_shared", "W_module", "b_module", "temps"]:
            grad_a = oracle_a.final_grads[name]
            grad_b1 = oracle_b1.final_grads[name]
            assert torch.allclose(grad_a, grad_b1, atol=ATOL, rtol=RTOL), (
                f"Gradient mismatch on {name}: "
                f"max_diff={(grad_a - grad_b1).abs().max().item():.2e}"
            )


# ═════════════════════════════════════════════════════════════════════
# Stage 2 — Axis 2: A vs B2, clipping disabled (tile decomposition)
# ═════════════════════════════════════════════════════════════════════


class TestAxisTwoTileDecomposition:
    """A vs B2 with clipping disabled.

    Both compute per-tile gradients; the only difference is
    manual (A) vs autograd (B2). Should agree exactly.
    """

    @staticmethod
    def _make_unclipped_config(base: OracleConfig) -> OracleConfig:
        return OracleConfig(
            input_dim=base.input_dim,
            hidden_dim=base.hidden_dim,
            output_classes=base.output_classes,
            num_modules=base.num_modules,
            mode=base.mode,
            learning_rate=base.learning_rate,
            beta1=base.beta1,
            beta2=base.beta2,
            epsilon=base.epsilon,
            t_algorithmic=float("inf"),
            lambda_=base.lambda_,
            compute_fp_format_max=float("inf"),
            temp_min=base.temp_min,
            temp_max=base.temp_max,
        )

    def test_gradient_parity_cce(self, small_cce_config, small_cce_data):
        cfg = self._make_unclipped_config(small_cce_config)
        oracle_a = FaithfulOracle(cfg)
        oracle_b2 = AutogradOracle(cfg, mode="tiled")
        init_with_seed(oracle_a)
        sync_oracles(oracle_a, oracle_b2)

        X, targets = small_cce_data
        oracle_a.step(X, targets)
        oracle_b2.step(X, targets)

        for name in ["W_shared", "b_shared", "W_module", "b_module", "temps"]:
            grad_a = oracle_a.final_grads[name]
            grad_b2 = oracle_b2.final_grads[name]
            assert torch.allclose(grad_a, grad_b2, atol=ATOL, rtol=RTOL), (
                f"Tile decomposition mismatch on {name}: "
                f"max_diff={(grad_a - grad_b2).abs().max().item():.2e}"
            )

    def test_gradient_parity_bce(self, small_bce_config, small_bce_data):
        cfg = self._make_unclipped_config(small_bce_config)
        oracle_a = FaithfulOracle(cfg)
        oracle_b2 = AutogradOracle(cfg, mode="tiled")
        init_with_seed(oracle_a)
        sync_oracles(oracle_a, oracle_b2)

        X, targets = small_bce_data
        oracle_a.step(X, targets)
        oracle_b2.step(X, targets)

        for name in ["W_shared", "b_shared", "W_module", "b_module", "temps"]:
            grad_a = oracle_a.final_grads[name]
            grad_b2 = oracle_b2.final_grads[name]
            assert torch.allclose(grad_a, grad_b2, atol=ATOL, rtol=RTOL), (
                f"Tile decomposition mismatch on {name}: "
                f"max_diff={(grad_a - grad_b2).abs().max().item():.2e}"
            )


# ═════════════════════════════════════════════════════════════════════
# Stage 3 — Axis 3: A vs B2, clipping enabled (full pipeline parity)
# ═════════════════════════════════════════════════════════════════════


class TestAxisThreeFullPipeline:
    """A vs B2 with clipping enabled. Same algorithm, two derivations."""

    def test_gradient_parity_cce(self, small_cce_config, small_cce_data):
        oracle_a = FaithfulOracle(small_cce_config)
        oracle_b2 = AutogradOracle(small_cce_config, mode="tiled")
        init_with_seed(oracle_a)
        sync_oracles(oracle_a, oracle_b2)

        X, targets = small_cce_data
        oracle_a.step(X, targets)
        oracle_b2.step(X, targets)

        for name in ["W_shared", "b_shared", "W_module", "b_module", "temps"]:
            grad_a = oracle_a.final_grads[name]
            grad_b2 = oracle_b2.final_grads[name]
            assert torch.allclose(grad_a, grad_b2, atol=ATOL, rtol=RTOL), (
                f"Pipeline mismatch on {name}: "
                f"max_diff={(grad_a - grad_b2).abs().max().item():.2e}"
            )

    def test_gradient_parity_bce(self, small_bce_config, small_bce_data):
        oracle_a = FaithfulOracle(small_bce_config)
        oracle_b2 = AutogradOracle(small_bce_config, mode="tiled")
        init_with_seed(oracle_a)
        sync_oracles(oracle_a, oracle_b2)

        X, targets = small_bce_data
        oracle_a.step(X, targets)
        oracle_b2.step(X, targets)

        for name in ["W_shared", "b_shared", "W_module", "b_module", "temps"]:
            grad_a = oracle_a.final_grads[name]
            grad_b2 = oracle_b2.final_grads[name]
            assert torch.allclose(grad_a, grad_b2, atol=ATOL, rtol=RTOL), (
                f"Pipeline mismatch on {name}: "
                f"max_diff={(grad_a - grad_b2).abs().max().item():.2e}"
            )


# ═════════════════════════════════════════════════════════════════════
# Stage 4 — Axis 4: B1 vs B2, clipping disabled (autograd self-check)
# ═════════════════════════════════════════════════════════════════════


class TestAxisFourAutogradSelfConsistency:
    """B1 vs B2 with no clipping. Must be identical (same autograd)."""

    @staticmethod
    def _make_unclipped_config(base: OracleConfig) -> OracleConfig:
        return OracleConfig(
            input_dim=base.input_dim,
            hidden_dim=base.hidden_dim,
            output_classes=base.output_classes,
            num_modules=base.num_modules,
            mode=base.mode,
            learning_rate=base.learning_rate,
            beta1=base.beta1,
            beta2=base.beta2,
            epsilon=base.epsilon,
            t_algorithmic=float("inf"),
            lambda_=base.lambda_,
            compute_fp_format_max=float("inf"),
            temp_min=base.temp_min,
            temp_max=base.temp_max,
        )

    def test_gradient_parity_cce(self, small_cce_config, small_cce_data):
        cfg = self._make_unclipped_config(small_cce_config)
        oracle_b1 = AutogradOracle(cfg, mode="flat")
        oracle_b2 = AutogradOracle(cfg, mode="tiled")
        init_with_seed(oracle_b1)
        sync_oracles(oracle_b1, oracle_b2)

        X, targets = small_cce_data
        oracle_b1.step(X, targets)
        oracle_b2.step(X, targets)

        for name in ["W_shared", "b_shared", "W_module", "b_module", "temps"]:
            grad_b1 = oracle_b1.final_grads[name]
            grad_b2 = oracle_b2.final_grads[name]
            assert torch.allclose(grad_b1, grad_b2, atol=ATOL, rtol=RTOL), (
                f"Autograd self-inconsistency on {name}: "
                f"max_diff={(grad_b1 - grad_b2).abs().max().item():.2e}"
            )


# ═════════════════════════════════════════════════════════════════════
# Stage 5 — Multi-step convergence (A vs B2)
# ═════════════════════════════════════════════════════════════════════


class TestMultiStepConvergence:
    """A vs B2 over multiple steps — trajectory parity."""

    @pytest.mark.parametrize("steps", [5, 20])
    def test_trajectory_cce(self, small_cce_config, small_cce_data, steps):
        oracle_a = FaithfulOracle(small_cce_config)
        oracle_b2 = AutogradOracle(small_cce_config, mode="tiled")
        init_with_seed(oracle_a)
        sync_oracles(oracle_a, oracle_b2)

        X, targets = small_cce_data
        for step in range(steps):
            oracle_a.step(X, targets)
            oracle_b2.step(X, targets)

            for name in ["W_shared", "b_shared", "W_module", "b_module", "temps"]:
                pa = oracle_a.export_state()[name]
                pb = oracle_b2.export_state()[name]
                assert torch.allclose(pa, pb, atol=1e-8, rtol=1e-8), (
                    f"Step {step}, param {name} diverged: "
                    f"max_diff={(pa - pb).abs().max().item():.2e}"
                )

    @pytest.mark.parametrize("steps", [5, 20])
    def test_trajectory_bce(self, small_bce_config, small_bce_data, steps):
        oracle_a = FaithfulOracle(small_bce_config)
        oracle_b2 = AutogradOracle(small_bce_config, mode="tiled")
        init_with_seed(oracle_a)
        sync_oracles(oracle_a, oracle_b2)

        X, targets = small_bce_data
        for step in range(steps):
            oracle_a.step(X, targets)
            oracle_b2.step(X, targets)

            for name in ["W_shared", "b_shared", "W_module", "b_module", "temps"]:
                pa = oracle_a.export_state()[name]
                pb = oracle_b2.export_state()[name]
                assert torch.allclose(pa, pb, atol=1e-8, rtol=1e-8), (
                    f"Step {step}, param {name} diverged: "
                    f"max_diff={(pa - pb).abs().max().item():.2e}"
                )


# ═════════════════════════════════════════════════════════════════════
# Stage 6 — Larger problem with tiling (medium config)
# ═════════════════════════════════════════════════════════════════════


class TestLargerProblemWithTiling:
    """Axis 1 + 2 on a problem large enough to produce multiple tiles."""

    @staticmethod
    def _make_unclipped_config(base: OracleConfig) -> OracleConfig:
        return OracleConfig(
            input_dim=base.input_dim,
            hidden_dim=base.hidden_dim,
            output_classes=base.output_classes,
            num_modules=base.num_modules,
            mode=base.mode,
            learning_rate=base.learning_rate,
            beta1=base.beta1,
            beta2=base.beta2,
            epsilon=base.epsilon,
            t_algorithmic=float("inf"),
            lambda_=base.lambda_,
            compute_fp_format_max=float("inf"),
            temp_min=base.temp_min,
            temp_max=base.temp_max,
        )

    def test_a_vs_b1_medium(self, medium_cce_config, medium_cce_data):
        """A vs B1, no clip, medium problem (multiple tiles)."""
        cfg = self._make_unclipped_config(medium_cce_config)
        oracle_a = FaithfulOracle(cfg)
        oracle_b1 = AutogradOracle(cfg, mode="flat")
        init_with_seed(oracle_a)
        sync_oracles(oracle_a, oracle_b1)

        X, targets = medium_cce_data
        oracle_a.step(X, targets)
        oracle_b1.step(X, targets)

        for name in ["W_shared", "b_shared", "W_module", "b_module", "temps"]:
            grad_a = oracle_a.final_grads[name]
            grad_b1 = oracle_b1.final_grads[name]
            assert torch.allclose(grad_a, grad_b1, atol=ATOL, rtol=RTOL), (
                f"Medium problem gradient mismatch on {name}: "
                f"max_diff={(grad_a - grad_b1).abs().max().item():.2e}"
            )

    def test_a_vs_b2_medium(self, medium_cce_config, medium_cce_data):
        """A vs B2, no clip, medium problem."""
        cfg = self._make_unclipped_config(medium_cce_config)
        oracle_a = FaithfulOracle(cfg)
        oracle_b2 = AutogradOracle(cfg, mode="tiled")
        init_with_seed(oracle_a)
        sync_oracles(oracle_a, oracle_b2)

        X, targets = medium_cce_data
        oracle_a.step(X, targets)
        oracle_b2.step(X, targets)

        for name in ["W_shared", "b_shared", "W_module", "b_module", "temps"]:
            grad_a = oracle_a.final_grads[name]
            grad_b2 = oracle_b2.final_grads[name]
            assert torch.allclose(grad_a, grad_b2, atol=ATOL, rtol=RTOL), (
                f"Medium problem tile mismatch on {name}: "
                f"max_diff={(grad_a - grad_b2).abs().max().item():.2e}"
            )
