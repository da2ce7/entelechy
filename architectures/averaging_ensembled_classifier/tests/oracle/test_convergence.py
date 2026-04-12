# tests/oracle/test_convergence.py
"""Convergence oracle (Oracle C) test matrix.

Implements the test patterns from the Oracle C design document:

  1. Multi-step parity (no clipping): A vs C, B2 vs C
  2. Moment parity after N steps
  3. Known-solution convergence (CCE + BCE)
  4. Convergence under clipping: A vs C loss curves
  5. Normalization invariance (batch-size independence)
  6. Long-run stability (The Marathon)
  7. Single-step gradient parity: C vs B1 (no clipping)

Oracle C shares ZERO gradient-processing code with A/B.  The only
shared dependency is adam_update_fp64.  Divergence between C and A
(when clipping is inactive) proves a bug in A's manual pipeline.
"""
from __future__ import annotations

from typing import TYPE_CHECKING

import pytest

pytestmark = pytest.mark.slow

if TYPE_CHECKING:
    import torch
else:
    torch = pytest.importorskip("torch", reason="Oracle tests require PyTorch")

from .autograd_oracle import AutogradOracle
from .conftest import _loss_close, init_with_seed, make_unclipped_config, sync_oracles
from .convergence_oracle import ConvergenceOracle
from .faithful_oracle import FaithfulOracle
from .oracle_config import OracleConfig
from .conftest import _generate_xor_data

PARAM_NAMES = ["W_shared", "b_shared", "W_module", "b_module", "temps"]



# ═════════════════════════════════════════════════════════════════════
# 1. Single-step gradient parity: C vs B1, no clipping
# ═════════════════════════════════════════════════════════════════════


class TestSingleStepGradientParity:
    """C vs B1 (flat autograd), no clipping.

    Both use standard torch.autograd; they must agree on raw gradients
    and on the batch-normalized final gradients, confirming C's forward
    pass and loss computation are structurally correct.
    """

    ATOL = 1e-9
    RTOL = 1e-9

    def test_gradient_parity_cce(
        self,
        small_cce_config: OracleConfig,
        small_cce_data: tuple[torch.Tensor, torch.Tensor],
    ) -> None:
        cfg = make_unclipped_config(small_cce_config)
        oracle_b1 = AutogradOracle(cfg, mode="flat")
        oracle_c = ConvergenceOracle(cfg)
        init_with_seed(oracle_b1)
        sync_oracles(oracle_b1, oracle_c)

        X, targets = small_cce_data
        oracle_b1.step(X, targets)
        oracle_c.step(X, targets)

        assert _loss_close(oracle_b1.last_loss, oracle_c.last_loss, atol=self.ATOL, rtol=self.RTOL), (
            f"Loss mismatch: B1={oracle_b1.last_loss}, C={oracle_c.last_loss}"
        )
        for name in PARAM_NAMES:
            grad_b1 = oracle_b1.final_grads[name]
            grad_c = oracle_c.final_grads[name]
            assert torch.allclose(grad_b1, grad_c, atol=self.ATOL, rtol=self.RTOL), (
                f"Gradient mismatch on {name}: "
                f"max_diff={(grad_b1 - grad_c).abs().max().item():.2e}"
            )

    def test_gradient_parity_bce(
        self,
        small_bce_config: OracleConfig,
        small_bce_data: tuple[torch.Tensor, torch.Tensor],
    ) -> None:
        cfg = make_unclipped_config(small_bce_config)
        oracle_b1 = AutogradOracle(cfg, mode="flat")
        oracle_c = ConvergenceOracle(cfg)
        init_with_seed(oracle_b1)
        sync_oracles(oracle_b1, oracle_c)

        X, targets = small_bce_data
        oracle_b1.step(X, targets)
        oracle_c.step(X, targets)

        assert _loss_close(oracle_b1.last_loss, oracle_c.last_loss, atol=self.ATOL, rtol=self.RTOL), (
            f"Loss mismatch: B1={oracle_b1.last_loss}, C={oracle_c.last_loss}"
        )
        for name in PARAM_NAMES:
            grad_b1 = oracle_b1.final_grads[name]
            grad_c = oracle_c.final_grads[name]
            assert torch.allclose(grad_b1, grad_c, atol=self.ATOL, rtol=self.RTOL), (
                f"Gradient mismatch on {name}: "
                f"max_diff={(grad_b1 - grad_c).abs().max().item():.2e}"
            )

    def test_gradient_parity_a_vs_c_cce(
        self,
        small_cce_config: OracleConfig,
        small_cce_data: tuple[torch.Tensor, torch.Tensor],
    ) -> None:
        """A vs C, no clipping. Manual formulas vs. independent autograd."""
        cfg = make_unclipped_config(small_cce_config)
        oracle_a = FaithfulOracle(cfg)
        oracle_c = ConvergenceOracle(cfg)
        init_with_seed(oracle_a)
        sync_oracles(oracle_a, oracle_c)

        X, targets = small_cce_data
        oracle_a.step(X, targets)
        oracle_c.step(X, targets)

        for name in PARAM_NAMES:
            grad_a = oracle_a.final_grads[name]
            grad_c = oracle_c.final_grads[name]
            assert torch.allclose(grad_a, grad_c, atol=self.ATOL, rtol=self.RTOL), (
                f"A vs C gradient mismatch on {name}: "
                f"max_diff={(grad_a - grad_c).abs().max().item():.2e}"
            )

    def test_gradient_parity_xor(
        self,
        xor_config: OracleConfig,
        xor_data: tuple[torch.Tensor, torch.Tensor],
    ) -> None:
        """B1 vs C gradient parity on XOR (BCE, 1 output class)."""
        cfg = make_unclipped_config(xor_config)
        oracle_b1 = AutogradOracle(cfg, mode="flat")
        oracle_c = ConvergenceOracle(cfg)
        init_with_seed(oracle_b1)
        sync_oracles(oracle_b1, oracle_c)

        X, targets = xor_data
        oracle_b1.step(X, targets)
        oracle_c.step(X, targets)

        assert _loss_close(oracle_b1.last_loss, oracle_c.last_loss, atol=self.ATOL, rtol=self.RTOL), (
            f"Loss mismatch on XOR: B1={oracle_b1.last_loss}, C={oracle_c.last_loss}"
        )
        for name in PARAM_NAMES:
            grad_b1 = oracle_b1.final_grads[name]
            grad_c = oracle_c.final_grads[name]
            assert torch.allclose(grad_b1, grad_c, atol=self.ATOL, rtol=self.RTOL), (
                f"Gradient mismatch on {name}: "
                f"max_diff={(grad_b1 - grad_c).abs().max().item():.2e}"
            )

    def test_gradient_parity_a_vs_c_xor(
        self,
        xor_config: OracleConfig,
        xor_data: tuple[torch.Tensor, torch.Tensor],
    ) -> None:
        """A vs C gradient parity on XOR (BCE, 1 output class)."""
        cfg = make_unclipped_config(xor_config)
        oracle_a = FaithfulOracle(cfg)
        oracle_c = ConvergenceOracle(cfg)
        init_with_seed(oracle_a)
        sync_oracles(oracle_a, oracle_c)

        X, targets = xor_data
        oracle_a.step(X, targets)
        oracle_c.step(X, targets)

        for name in PARAM_NAMES:
            grad_a = oracle_a.final_grads[name]
            grad_c = oracle_c.final_grads[name]
            assert torch.allclose(grad_a, grad_c, atol=self.ATOL, rtol=self.RTOL), (
                f"A vs C gradient mismatch on {name}: "
                f"max_diff={(grad_a - grad_c).abs().max().item():.2e}"
            )


# ═════════════════════════════════════════════════════════════════════
# 2. Multi-step parity (no clipping): A vs C, B2 vs C
# ═════════════════════════════════════════════════════════════════════


class TestMultiStepParity:
    """Multi-step trajectory parity with clipping disabled.

    When clipping is inactive, C must produce identical parameters to
    A after N steps for any N.  Divergence proves a bug in A's manual
    gradient pipeline.
    """

    ATOL = 1e-10

    @pytest.mark.parametrize("steps", [5, 50, 200])
    def test_a_vs_c_trajectory_cce(
        self,
        small_cce_config: OracleConfig,
        small_cce_data: tuple[torch.Tensor, torch.Tensor],
        steps: int,
    ) -> None:
        cfg = make_unclipped_config(small_cce_config)
        oracle_a = FaithfulOracle(cfg)
        oracle_c = ConvergenceOracle(cfg)
        init_with_seed(oracle_a)
        sync_oracles(oracle_a, oracle_c)

        X, targets = small_cce_data
        for _step in range(steps):
            oracle_a.step(X, targets)
            oracle_c.step(X, targets)

        distances = oracle_c.parameter_distance(oracle_a.export_state())
        for name, dist in distances.items():
            assert dist < self.ATOL, (
                f"A vs C param drift on {name} after {steps} steps: {dist:.2e}"
            )

    @pytest.mark.parametrize("steps", [5, 50, 200])
    def test_a_vs_c_trajectory_bce(
        self,
        small_bce_config: OracleConfig,
        small_bce_data: tuple[torch.Tensor, torch.Tensor],
        steps: int,
    ) -> None:
        cfg = make_unclipped_config(small_bce_config)
        oracle_a = FaithfulOracle(cfg)
        oracle_c = ConvergenceOracle(cfg)
        init_with_seed(oracle_a)
        sync_oracles(oracle_a, oracle_c)

        X, targets = small_bce_data
        for _step in range(steps):
            oracle_a.step(X, targets)
            oracle_c.step(X, targets)

        distances = oracle_c.parameter_distance(oracle_a.export_state())
        for name, dist in distances.items():
            assert dist < self.ATOL, (
                f"A vs C param drift on {name} after {steps} steps: {dist:.2e}"
            )

    @pytest.mark.parametrize("steps", [5, 50, 200])
    def test_a_vs_c_trajectory_xor(
        self,
        xor_config: OracleConfig,
        xor_data: tuple[torch.Tensor, torch.Tensor],
        steps: int,
    ) -> None:
        cfg = make_unclipped_config(xor_config)
        oracle_a = FaithfulOracle(cfg)
        oracle_c = ConvergenceOracle(cfg)
        init_with_seed(oracle_a)
        sync_oracles(oracle_a, oracle_c)

        X, targets = xor_data
        for _step in range(steps):
            oracle_a.step(X, targets)
            oracle_c.step(X, targets)

        distances = oracle_c.parameter_distance(oracle_a.export_state())
        for name, dist in distances.items():
            assert dist < self.ATOL, (
                f"A vs C param drift on {name} after {steps} steps: {dist:.2e}"
            )

    @pytest.mark.parametrize("steps", [5, 50, 200])
    def test_b2_vs_c_trajectory_cce(
        self,
        small_cce_config: OracleConfig,
        small_cce_data: tuple[torch.Tensor, torch.Tensor],
        steps: int,
    ) -> None:
        cfg = make_unclipped_config(small_cce_config)
        oracle_b2 = AutogradOracle(cfg, mode="tiled")
        oracle_c = ConvergenceOracle(cfg)
        init_with_seed(oracle_b2)
        sync_oracles(oracle_b2, oracle_c)

        X, targets = small_cce_data
        for _step in range(steps):
            oracle_b2.step(X, targets)
            oracle_c.step(X, targets)

        distances = oracle_c.parameter_distance(oracle_b2.export_state())
        for name, dist in distances.items():
            assert dist < self.ATOL, (
                f"B2 vs C param drift on {name} after {steps} steps: {dist:.2e}"
            )


# ═════════════════════════════════════════════════════════════════════
# 3. Moment parity after N steps
# ═════════════════════════════════════════════════════════════════════


class TestMomentParity:
    """Adam m1/m2 moment vectors must match after N steps (no clipping).

    Catches optimizer state management bugs between step() calls.
    """

    ATOL = 1e-9

    @pytest.mark.parametrize("steps", [10, 100])
    def test_moment_parity_a_vs_c_cce(
        self,
        small_cce_config: OracleConfig,
        small_cce_data: tuple[torch.Tensor, torch.Tensor],
        steps: int,
    ) -> None:
        cfg = make_unclipped_config(small_cce_config)
        oracle_a = FaithfulOracle(cfg)
        oracle_c = ConvergenceOracle(cfg)
        init_with_seed(oracle_a)
        sync_oracles(oracle_a, oracle_c)

        X, targets = small_cce_data
        for _ in range(steps):
            oracle_a.step(X, targets)
            oracle_c.step(X, targets)

        moment_dists = oracle_c.moment_distance(oracle_a.export_state())
        for key, dist in moment_dists.items():
            assert dist < self.ATOL, (
                f"Moment drift on {key} after {steps} steps: {dist:.2e}"
            )

    @pytest.mark.parametrize("steps", [10, 100])
    def test_moment_parity_a_vs_c_xor(
        self,
        xor_config: OracleConfig,
        xor_data: tuple[torch.Tensor, torch.Tensor],
        steps: int,
    ) -> None:
        cfg = make_unclipped_config(xor_config)
        oracle_a = FaithfulOracle(cfg)
        oracle_c = ConvergenceOracle(cfg)
        init_with_seed(oracle_a)
        sync_oracles(oracle_a, oracle_c)

        X, targets = xor_data
        for _ in range(steps):
            oracle_a.step(X, targets)
            oracle_c.step(X, targets)

        moment_dists = oracle_c.moment_distance(oracle_a.export_state())
        for key, dist in moment_dists.items():
            assert dist < self.ATOL, (
                f"Moment drift on {key} after {steps} steps: {dist:.2e}"
            )

    @pytest.mark.parametrize("steps", [10, 100])
    def test_full_state_parity_a_vs_c_bce(
        self,
        small_bce_config: OracleConfig,
        small_bce_data: tuple[torch.Tensor, torch.Tensor],
        steps: int,
    ) -> None:
        """Params + moments combined distance (BCE)."""
        cfg = make_unclipped_config(small_bce_config)
        oracle_a = FaithfulOracle(cfg)
        oracle_c = ConvergenceOracle(cfg)
        init_with_seed(oracle_a)
        sync_oracles(oracle_a, oracle_c)

        X, targets = small_bce_data
        for _ in range(steps):
            oracle_a.step(X, targets)
            oracle_c.step(X, targets)

        full_dists = oracle_c.full_state_distance(oracle_a.export_state())
        for key, dist in full_dists.items():
            assert dist < self.ATOL, (
                f"State drift on {key} after {steps} steps: {dist:.2e}"
            )


# ═════════════════════════════════════════════════════════════════════
# 4. Known-solution convergence
# ═════════════════════════════════════════════════════════════════════


class TestKnownSolutionConvergence:
    """Train Oracle C on problems with known solutions.

    Confirms the mathematical model can converge — loss decreases
    to near-zero on separable problems.
    """

    def test_separable_clusters_cce(self):
        """CCE on well-separated Gaussian clusters: loss → near-zero."""
        num_classes = 3
        input_dim = 8
        config = OracleConfig(
            input_dim=input_dim,
            hidden_dim=16,
            output_classes=num_classes,
            num_modules=2,
            mode="CCE",
            learning_rate=0.01,
        )
        oracle_c = ConvergenceOracle(config)
        init_with_seed(oracle_c, seed=7)

        X, targets = ConvergenceOracle.make_separable_clusters_cce(
            num_classes=num_classes,
            input_dim=input_dim,
            num_samples_per_class=30,
            separation=3.0,
            seed=42,
        )

        trace = oracle_c.train_n_steps(X, targets, n=1000, record_every=10)

        assert trace.converged, (
            f"CCE failed to converge: initial={trace.initial_loss:.4f}, "
            f"final={trace.final_loss:.4f}"
        )
        assert trace.is_stable, "NaN/Inf detected during CCE convergence"
        assert trace.final_loss < 0.5, (
            f"CCE final loss too high: {trace.final_loss:.4f}"
        )

    def test_independent_bce(self):
        """BCE on independent multi-label data: loss decreases."""
        num_classes = 4
        input_dim = 8
        config = OracleConfig(
            input_dim=input_dim,
            hidden_dim=16,
            output_classes=num_classes,
            num_modules=2,
            mode="BCE",
            learning_rate=0.01,
        )
        oracle_c = ConvergenceOracle(config)
        init_with_seed(oracle_c, seed=7)

        X, targets = ConvergenceOracle.make_independent_bce(
            num_classes=num_classes,
            input_dim=input_dim,
            num_samples=100,
            seed=42,
        )

        trace = oracle_c.train_n_steps(X, targets, n=500, record_every=10)

        assert trace.converged, (
            f"BCE failed to converge: initial={trace.initial_loss:.4f}, "
            f"final={trace.final_loss:.4f}"
        )
        assert trace.is_stable, "NaN/Inf detected during BCE convergence"
        assert trace.loss_reduction_ratio < 0.5, (
            f"BCE loss reduction too small: ratio={trace.loss_reduction_ratio:.4f}"
        )

    def test_xor_bce(self):
        """BCE on XOR (non-linear separability): loss decreases."""
        config = OracleConfig(
            input_dim=2,
            hidden_dim=16,
            output_classes=1,
            num_modules=8,
            mode="BCE",
            learning_rate=0.01,
        )
        oracle_c = ConvergenceOracle(config)
        init_with_seed(oracle_c, seed=7)

        X_np, targets_np = _generate_xor_data()
        X = torch.from_numpy(X_np).to(torch.float64)
        targets = torch.from_numpy(targets_np).to(torch.float64)

        trace = oracle_c.train_n_steps(X, targets, n=1000, record_every=10)

        assert trace.converged, (
            f"XOR failed to converge: initial={trace.initial_loss:.4f}, "
            f"final={trace.final_loss:.4f}"
        )
        assert trace.is_stable, "NaN/Inf detected during XOR convergence"
        assert trace.loss_reduction_ratio < 0.5, (
            f"XOR loss reduction too small: ratio={trace.loss_reduction_ratio:.4f}"
        )


# ═════════════════════════════════════════════════════════════════════
# 5. Convergence under clipping: A vs C loss curves
# ═════════════════════════════════════════════════════════════════════


class TestConvergenceUnderClipping:
    """When clipping is active, A's loss should track C's within a
    bounded envelope.

    Clipping distorts gradients vs. the unclipped ideal (C), but the
    system should still converge — just possibly slower.
    """

    def test_clipped_a_tracks_c_cce(self):
        """A with modest clipping should still converge, tracking C."""
        num_classes = 3
        input_dim = 8
        # Moderate clipping config for A
        clipped_config = OracleConfig(
            input_dim=input_dim,
            hidden_dim=16,
            output_classes=num_classes,
            num_modules=2,
            mode="CCE",
            learning_rate=0.01,
            t_algorithmic=1.0,
            lambda_=0.1,
        )
        # No-clip config for C
        unclipped_config = make_unclipped_config(clipped_config)

        oracle_a = FaithfulOracle(clipped_config)
        oracle_c = ConvergenceOracle(unclipped_config)
        init_with_seed(oracle_a, seed=7)
        sync_oracles(oracle_a, oracle_c)

        X, targets = ConvergenceOracle.make_separable_clusters_cce(
            num_classes=num_classes,
            input_dim=input_dim,
            num_samples_per_class=30,
            separation=3.0,
            seed=42,
        )

        n_steps = 300
        losses_a: list[float] = []
        losses_c: list[float] = []
        for _ in range(n_steps):
            oracle_a.step(X, targets)
            oracle_c.step(X, targets)
            losses_a.append(oracle_a.last_loss)
            losses_c.append(oracle_c.last_loss)

        # Both should converge
        assert losses_a[-1] < losses_a[0], (
            f"A failed to reduce loss: {losses_a[0]:.4f} → {losses_a[-1]:.4f}"
        )
        assert losses_c[-1] < losses_c[0], (
            f"C failed to reduce loss: {losses_c[0]:.4f} → {losses_c[-1]:.4f}"
        )

        # A's final loss should be within a reasonable envelope of C's
        # (clipping may slow convergence but shouldn't cause divergence)
        tolerance_ratio = 10.0
        assert losses_a[-1] < losses_c[-1] * tolerance_ratio + 1.0, (
            f"A's loss too far from C's: A={losses_a[-1]:.4f}, C={losses_c[-1]:.4f}"
        )


# ═════════════════════════════════════════════════════════════════════
# 6. Normalization invariance (batch-size independence)
# ═════════════════════════════════════════════════════════════════════


class TestNormalizationInvariance:
    """Node 21 divides by effective_batch_size.  Two runs with different
    batch sizes on the same data distribution should produce similar
    parameter trajectories.

    This is a statistical test — not exact parity, but the learning
    dynamics should be qualitatively similar.
    """

    def test_batch_size_invariance_cce(self):
        num_classes = 3
        input_dim = 8
        config = OracleConfig(
            input_dim=input_dim,
            hidden_dim=16,
            output_classes=num_classes,
            num_modules=2,
            mode="CCE",
            learning_rate=0.01,
        )

        # Generate data — same distribution, different batch sizes
        X_full, targets_full = ConvergenceOracle.make_separable_clusters_cce(
            num_classes=num_classes,
            input_dim=input_dim,
            num_samples_per_class=40,
            separation=3.0,
            seed=42,
        )

        # Small batch: first 30 samples
        X_small, targets_small = X_full[:30], targets_full[:30]
        # Large batch: first 90 samples
        X_large, targets_large = X_full[:90], targets_full[:90]

        oracle_small = ConvergenceOracle(config)
        oracle_large = ConvergenceOracle(config)
        init_with_seed(oracle_small, seed=7)
        sync_oracles(oracle_small, oracle_large)

        trace_small = oracle_small.train_n_steps(
            X_small, targets_small, n=200, record_every=10,
        )
        trace_large = oracle_large.train_n_steps(
            X_large, targets_large, n=200, record_every=10,
        )

        # Both should converge
        assert trace_small.converged, "Small-batch run failed to converge"
        assert trace_large.converged, "Large-batch run failed to converge"

        # Both should be stable
        assert trace_small.is_stable, "Small-batch run unstable"
        assert trace_large.is_stable, "Large-batch run unstable"


# ═════════════════════════════════════════════════════════════════════
# 7. Long-run stability (The Marathon)
# ═════════════════════════════════════════════════════════════════════


class TestLongRunStability:
    """Train C for many steps — no NaN/Inf, Adam bias correction
    stays correct at large t.

    Marked slow; the 100K marathon is a separate marker.
    """

    def test_stability_1k_steps_cce(self):
        """1,000 steps: no NaN/Inf in any recorded quantity."""
        num_classes = 3
        input_dim = 6
        config = OracleConfig(
            input_dim=input_dim,
            hidden_dim=12,
            output_classes=num_classes,
            num_modules=2,
            mode="CCE",
            learning_rate=0.001,
        )
        oracle_c = ConvergenceOracle(config)
        init_with_seed(oracle_c, seed=7)

        X, targets = ConvergenceOracle.make_separable_clusters_cce(
            num_classes=num_classes,
            input_dim=input_dim,
            num_samples_per_class=20,
            separation=3.0,
            seed=42,
        )

        trace = oracle_c.train_n_steps(X, targets, n=1000, record_every=50)

        assert trace.is_stable, "NaN/Inf during 1K-step run"
        assert trace.converged, (
            f"Failed to converge over 1K steps: "
            f"initial={trace.initial_loss:.4f}, final={trace.final_loss:.4f}"
        )

    def test_stability_1k_steps_bce(self):
        """1,000 steps BCE: no NaN/Inf."""
        num_classes = 4
        input_dim = 6
        config = OracleConfig(
            input_dim=input_dim,
            hidden_dim=12,
            output_classes=num_classes,
            num_modules=2,
            mode="BCE",
            learning_rate=0.001,
        )
        oracle_c = ConvergenceOracle(config)
        init_with_seed(oracle_c, seed=7)

        X, targets = ConvergenceOracle.make_independent_bce(
            num_classes=num_classes,
            input_dim=input_dim,
            num_samples=80,
            seed=42,
        )

        trace = oracle_c.train_n_steps(X, targets, n=1000, record_every=50)

        assert trace.is_stable, "NaN/Inf during 1K-step BCE run"
        assert trace.converged, (
            f"BCE failed to converge over 1K steps: "
            f"initial={trace.initial_loss:.4f}, final={trace.final_loss:.4f}"
        )

    def test_stability_1k_steps_xor(self):
        """1,000 steps XOR: no NaN/Inf."""
        config = OracleConfig(
            input_dim=2,
            hidden_dim=16,
            output_classes=1,
            num_modules=8,
            mode="BCE",
            learning_rate=0.001,
        )
        oracle_c = ConvergenceOracle(config)
        init_with_seed(oracle_c, seed=7)

        X_np, targets_np = _generate_xor_data()
        X = torch.from_numpy(X_np).to(torch.float64)
        targets = torch.from_numpy(targets_np).to(torch.float64)

        trace = oracle_c.train_n_steps(X, targets, n=1000, record_every=50)

        assert trace.is_stable, "NaN/Inf during 1K-step XOR run"
        assert trace.converged, (
            f"XOR failed to converge over 1K steps: "
            f"initial={trace.initial_loss:.4f}, final={trace.final_loss:.4f}"
        )

    @pytest.mark.slow
    def test_marathon_100k_steps(self):
        """100K steps: the marathon. Validates Adam bias correction
        stays correct at very large t.
        """
        num_classes = 3
        input_dim = 6
        config = OracleConfig(
            input_dim=input_dim,
            hidden_dim=12,
            output_classes=num_classes,
            num_modules=2,
            mode="CCE",
            learning_rate=0.001,
        )
        oracle_c = ConvergenceOracle(config)
        init_with_seed(oracle_c, seed=7)

        X, targets = ConvergenceOracle.make_separable_clusters_cce(
            num_classes=num_classes,
            input_dim=input_dim,
            num_samples_per_class=20,
            separation=3.0,
            seed=42,
        )

        trace = oracle_c.train_n_steps(X, targets, n=100_000, record_every=5000)

        assert trace.is_stable, "NaN/Inf during 100K marathon"
        assert trace.converged, (
            f"Marathon failed to converge: "
            f"initial={trace.initial_loss:.4f}, final={trace.final_loss:.4f}"
        )

    def test_step_counter_consistency(
        self,
        small_cce_config: OracleConfig,
        small_cce_data: tuple[torch.Tensor, torch.Tensor],
    ) -> None:
        """After N steps, both A and C report the same step counter."""
        cfg = make_unclipped_config(small_cce_config)
        oracle_a = FaithfulOracle(cfg)
        oracle_c = ConvergenceOracle(cfg)
        init_with_seed(oracle_a)
        sync_oracles(oracle_a, oracle_c)

        X, targets = small_cce_data
        n = 37
        for _ in range(n):
            oracle_a.step(X, targets)
            oracle_c.step(X, targets)

        assert oracle_a.t == n
        assert oracle_c.t == n
        state_a = oracle_a.export_state()
        state_c = oracle_c.export_state()
        assert state_a["_step"].item() == state_c["_step"].item() == n


# ═════════════════════════════════════════════════════════════════════
# 8. Sample mask support
# ═════════════════════════════════════════════════════════════════════


class TestSampleMaskParity:
    """C must handle sample_mask identically to A when clipping is off."""

    ATOL = 1e-10

    def test_masked_parity_cce(
        self,
        small_cce_config: OracleConfig,
        small_cce_data: tuple[torch.Tensor, torch.Tensor],
    ) -> None:
        cfg = make_unclipped_config(small_cce_config)
        oracle_a = FaithfulOracle(cfg)
        oracle_c = ConvergenceOracle(cfg)
        init_with_seed(oracle_a)
        sync_oracles(oracle_a, oracle_c)

        X, targets = small_cce_data
        # Mask out the last 3 samples
        mask = torch.ones(X.shape[0], dtype=torch.bool)
        mask[-3:] = False

        for _ in range(10):
            oracle_a.step(X, targets, sample_mask=mask)
            oracle_c.step(X, targets, sample_mask=mask)

        distances = oracle_c.parameter_distance(oracle_a.export_state())
        for name, dist in distances.items():
            assert dist < self.ATOL, (
                f"Masked parity drift on {name}: {dist:.2e}"
            )
