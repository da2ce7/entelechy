"""Oracle C: Convergence Oracle — end-to-end PyTorch reference.

A standard PyTorch implementation of the same model architecture, trained
with manual FP64 Adam.  No tiling, no clipping, no reduction trees, no
manual gradient formulas.  Shares ZERO gradient-processing code with
Oracles A/B.

Authority: end-to-end convergence correctness over multi-step training.

Detects inter-node convergence defects invisible to per-step oracles:
  1. Accumulation errors that compound across training steps but stay
     within per-step FP tolerance
  2. Optimizer state management bugs between step() calls (moment
     corruption, step counter drift)
  3. Normalization/scaling errors whose per-step effect is within
     tolerance but whose multi-step effect diverges
  4. Structural model errors (dimension transpositions, wrong
     activation placement, etc.)
  5. Shared bugs in Oracles A/B's manual reduction/clipping code
     that cause both to agree with each other but diverge from
     correctness

Validation modes:
  - Per-step parity: when clipping is inactive, Oracle A/B and C must
    produce identical parameters after N steps for any N.
  - Known-solution convergence: Oracle C must converge to near-zero
    loss on separable problems.
  - Loss curve monitoring: loss must decrease; no NaN/Inf.

Weakness: cannot model per-tile clipping or staged reduction — these
are engine-specific decomposition artifacts.  Oracle C validates the
mathematical model; Oracles A/B validate the decomposed implementation.
When clipping activates, C provides the "unclipped ideal" trajectory.

Reference: doc_archive/Oracle.md §Option C: Convergence Oracle
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field

import torch
import torch.nn as nn
import torch.nn.functional as F

from .oracle_config import OracleConfig
from .reference_utils import adam_update_fp64


# =====================================================================
# Diagnostics
# =====================================================================


@dataclass
class ConvergenceTrace:
    """Diagnostics recorded during a multi-step training run.

    All histories are indexed by recording step (controlled by
    ``record_every`` in :meth:`ConvergenceOracle.train_n_steps`).
    """

    loss_history: list[float] = field(default_factory=lambda: [])
    grad_norm_history: dict[str, list[float]] = field(default_factory=lambda: {})
    param_norm_history: dict[str, list[float]] = field(default_factory=lambda: {})

    @property
    def num_recorded(self) -> int:
        """Number of recorded diagnostic snapshots."""
        return len(self.loss_history)

    @property
    def initial_loss(self) -> float:
        return self.loss_history[0] if self.loss_history else float("nan")

    @property
    def final_loss(self) -> float:
        return self.loss_history[-1] if self.loss_history else float("nan")

    @property
    def loss_reduction_ratio(self) -> float:
        """``final_loss / initial_loss``.  Values < 1 indicate improvement."""
        if (
            len(self.loss_history) < 2
            or self.loss_history[0] == 0.0
        ):
            return float("nan")
        return self.loss_history[-1] / self.loss_history[0]

    @property
    def converged(self) -> bool:
        """Loss decreased overall with no NaN/Inf anywhere."""
        if len(self.loss_history) < 2:
            return False
        return self.is_stable and self.loss_history[-1] < self.loss_history[0]

    @property
    def is_stable(self) -> bool:
        """No NaN or Inf in any recorded quantity."""
        for v in self.loss_history:
            if math.isnan(v) or math.isinf(v):
                return False
        for norms in self.param_norm_history.values():
            for v in norms:
                if math.isnan(v) or math.isinf(v):
                    return False
        for norms in self.grad_norm_history.values():
            for v in norms:
                if math.isnan(v) or math.isinf(v):
                    return False
        return True

    @property
    def loss_monotonicity_violations(self) -> int:
        """Count of steps where loss increased vs. the previous step.

        A small number of violations is normal due to learning rate
        overshoot; many violations suggest instability.
        """
        count = 0
        for i in range(1, len(self.loss_history)):
            if self.loss_history[i] > self.loss_history[i - 1]:
                count += 1
        return count


# =====================================================================
# Oracle C
# =====================================================================


class ConvergenceOracle:
    """Standard PyTorch reference for multi-step convergence validation.

    Implements the same mathematical model as the engine::

        Input → ReLU(X @ W_shared^T + b_shared)
              → per-module (H @ W_module[m] + b_module[m])
              → temperature scaling (logits / T[m])
              → softmax (CCE) or sigmoid (BCE)
              → loss

    Gradients via standard ``loss.backward()``.  Optimizer via manual
    FP64 Adam (matching Oracles A/B and the engine's host-computed bias
    correction).

    **No tiling.  No clipping.  No reduction trees.  No manual gradient
    formulas.**  The only shared dependency with A/B is
    ``adam_update_fp64``.
    """

    def __init__(self, config: OracleConfig) -> None:
        self.config = config
        self.dtype = torch.float64

        # --- Learnable parameters (nn.Parameter for autograd) ---
        self.W_shared = nn.Parameter(
            torch.zeros(
                config.hidden_dim, config.input_dim, dtype=self.dtype,
            ),
        )
        self.b_shared = nn.Parameter(
            torch.zeros(config.hidden_dim, dtype=self.dtype),
        )
        self.W_module = nn.Parameter(
            torch.zeros(
                config.num_modules, config.hidden_dim,
                config.output_classes, dtype=self.dtype,
            ),
        )
        self.b_module = nn.Parameter(
            torch.zeros(
                config.num_modules, config.output_classes,
                dtype=self.dtype,
            ),
        )
        self.temps = nn.Parameter(
            torch.ones(config.num_modules, dtype=self.dtype),
        )
        self._all_params = [
            self.W_shared, self.b_shared,
            self.W_module, self.b_module, self.temps,
        ]
        self._param_names = [
            "W_shared", "b_shared", "W_module", "b_module", "temps",
        ]

        # --- Manual FP64 Adam state ---
        self.m1: dict[str, torch.Tensor] = {}
        self.m2: dict[str, torch.Tensor] = {}
        for name, param in zip(self._param_names, self._all_params):
            self.m1[name] = torch.zeros_like(param.data)
            self.m2[name] = torch.zeros_like(param.data)
        self.t = 0

        # --- Per-step diagnostics ---
        self.last_loss: float = float("nan")
        self.last_grad_norms: dict[str, float] = {}
        # Stores the normalized (post-Node-21) gradients for cross-oracle
        # comparison.  Same dict structure as Oracles A/B.
        self.final_grads: dict[str, torch.Tensor] = {}
        # Stores the raw (pre-normalization) autograd gradients for
        # cross-validation against Oracle B1.
        self.autograd_raw_grads: dict[str, torch.Tensor] = {}

    def named_params(self) -> list[tuple[str, nn.Parameter]]:
        return list(zip(self._param_names, self._all_params))

    def export_state(self) -> dict[str, torch.Tensor]:
        """Return a deep copy of all learnable state (params + Adam moments)."""
        state: dict[str, torch.Tensor] = {}
        for name, param in self.named_params():
            state[name] = param.data.clone()
            state[f"m1_{name}"] = self.m1[name].clone()
            state[f"m2_{name}"] = self.m2[name].clone()
        state["_step"] = torch.tensor(self.t, dtype=torch.int64)
        return state

    def load_state(self, state: dict[str, torch.Tensor]) -> None:
        """Load state from a dict (e.g. produced by another oracle's export_state)."""
        for name, param in self.named_params():
            param.data.copy_(state[name])
            self.m1[name].copy_(state[f"m1_{name}"])
            self.m2[name].copy_(state[f"m2_{name}"])
        self.t = int(state["_step"].item())

    # =================================================================
    # Public API
    # =================================================================

    @torch.enable_grad()
    def step(
        self,
        X: torch.Tensor,
        targets: torch.Tensor,
        sample_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """One training step using standard PyTorch autograd.

        Args:
            X: (batch, input_dim) float64.
            targets: CCE: (batch,) int64; BCE: (batch, output_classes) float64.
            sample_mask: optional (batch,) bool tensor.  None = all valid
                         (ADR-031).

        Returns:
            probs: (num_modules, batch, output_classes) detached.
        """
        X = X.to(self.dtype)  # pyright: ignore[reportConstantRedefinition]
        batch_size = X.shape[0]

        if sample_mask is None:
            sample_mask = torch.ones(batch_size, dtype=torch.bool)
        else:
            sample_mask = sample_mask.to(torch.bool)
        effective_batch_size = int(sample_mask.sum().item())
        mask_float = sample_mask.to(self.dtype)

        # --- Zero grads ---
        for p in self._all_params:
            if p.grad is not None:
                p.grad.zero_()

        # --- Forward + loss (autograd-tracked) ---
        probs, loss = self._forward_and_loss(X, targets, mask_float)
        self.last_loss = loss.item()

        # --- Backward ---
        loss.backward()  # pyright: ignore[reportUnknownMemberType]

        # --- Record raw autograd gradients ---
        self.autograd_raw_grads = {}
        self.last_grad_norms = {}
        for name, param in self.named_params():
            assert param.grad is not None, (
                f"No gradient for {name} — forward graph may be broken"
            )
            self.autograd_raw_grads[name] = param.grad.data.detach().clone()
            self.last_grad_norms[name] = float(param.grad.data.norm(2))  # pyright: ignore[reportUnknownMemberType, reportUnknownArgumentType]

        # --- Node 21: normalize by effective batch size ---
        eps = self.config.epsilon
        N = float(effective_batch_size)
        self.final_grads = {}
        for name, param in self.named_params():
            assert param.grad is not None
            self.final_grads[name] = param.grad.data / (N + eps)

        # --- Node 24: Adam update ---
        self.t += 1
        beta1_pow_t = self.config.beta1 ** self.t
        beta2_pow_t = self.config.beta2 ** self.t

        for name, param in self.named_params():
            param_new, self.m1[name], self.m2[name] = adam_update_fp64(
                param.data,
                self.final_grads[name],
                self.m1[name],
                self.m2[name],
                self.config.learning_rate,
                self.config.beta1,
                self.config.beta2,
                self.config.epsilon,
                beta1_pow_t,
                beta2_pow_t,
            )
            param.data.copy_(param_new)

        # --- Node 25: temperature clamp ---
        self.temps.data.clamp_(self.config.temp_min, self.config.temp_max)

        return probs.detach()

    def train_n_steps(
        self,
        X: torch.Tensor,
        targets: torch.Tensor,
        n: int,
        sample_mask: torch.Tensor | None = None,
        record_every: int = 1,
    ) -> ConvergenceTrace:
        """Train for *n* steps on the same batch, recording diagnostics.

        The same ``(X, targets)`` are presented every step — this tests
        the system's ability to overfit a fixed dataset, which is the
        simplest convergence criterion.

        Args:
            X: (batch, input_dim).
            targets: CCE (batch,) int64 or BCE (batch, classes) float64.
            n: number of training steps.
            sample_mask: optional sample validity mask.
            record_every: record diagnostics every N steps.
                          The final step is always recorded.

        Returns:
            ConvergenceTrace with loss and norm histories.
        """
        trace = ConvergenceTrace()
        for name in self._param_names:
            trace.grad_norm_history[name] = []
            trace.param_norm_history[name] = []

        for step_idx in range(n):
            self.step(X, targets, sample_mask)

            if step_idx % record_every == 0 or step_idx == n - 1:
                trace.loss_history.append(self.last_loss)
                for name, param in self.named_params():
                    trace.grad_norm_history[name].append(
                        self.last_grad_norms.get(name, 0.0),
                    )
                    trace.param_norm_history[name].append(
                        float(param.data.norm(2)),  # pyright: ignore[reportUnknownMemberType, reportUnknownArgumentType]
                    )

        return trace

    @torch.no_grad()
    def predict(
        self,
        X: torch.Tensor,
        sample_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Forward pass only (no learning).

        Returns:
            probs: (num_modules, batch, output_classes).
        """
        X = X.to(self.dtype)  # pyright: ignore[reportConstantRedefinition]
        batch_size = X.shape[0]
        if sample_mask is None:
            mask_float = torch.ones(batch_size, dtype=self.dtype)
        else:
            mask_float = sample_mask.to(self.dtype)

        M = self.config.num_modules

        # Node 4
        H = F.relu(X @ self.W_shared.data.T + self.b_shared.data)
        # Node 5
        logits = (
            torch.einsum("bh,mhc->mbc", H, self.W_module.data)
            + self.b_module.data.unsqueeze(1)
        )
        logits = logits * mask_float[None, :, None]

        probs = torch.zeros_like(logits)
        if self.config.mode == "CCE":
            for m in range(M):
                probs[m] = torch.softmax(
                    logits[m] / self.temps.data[m], dim=-1,
                )
        else:
            for m in range(M):
                probs[m] = torch.sigmoid(
                    logits[m] / self.temps.data[m],
                )
        return probs

    # =================================================================
    # Cross-oracle comparison utilities
    # =================================================================

    def parameter_distance(
        self,
        other_state: dict[str, torch.Tensor],
    ) -> dict[str, float]:
        """L2 distance between this oracle's parameters and another state.

        Usage::

            distance = oracle_c.parameter_distance(oracle_a.export_state())
            assert all(d < tol for d in distance.values())
        """
        distances: dict[str, float] = {}
        for name, param in self.named_params():
            other = other_state[name].to(self.dtype)
            distances[name] = float((param.data - other).norm(2))  # pyright: ignore[reportUnknownMemberType, reportUnknownArgumentType]
        return distances

    def moment_distance(
        self,
        other_state: dict[str, torch.Tensor],
    ) -> dict[str, float]:
        """L2 distance between Adam moment vectors and another state's."""
        moments = {"m1": self.m1, "m2": self.m2}
        distances: dict[str, float] = {}
        for mk_name, mk_dict in moments.items():
            for param_name in self._param_names:
                key = f"{mk_name}_{param_name}"
                other = other_state[key].to(self.dtype)
                distances[key] = float(
                    (mk_dict[param_name] - other).norm(2),  # pyright: ignore[reportUnknownMemberType, reportUnknownArgumentType]
                )
        return distances

    def full_state_distance(
        self,
        other_state: dict[str, torch.Tensor],
    ) -> dict[str, float]:
        """Combined parameter + moment distance for convenience."""
        d = self.parameter_distance(other_state)
        d.update(self.moment_distance(other_state))
        return d

    # =================================================================
    # Test data generators
    # =================================================================

    @staticmethod
    def make_separable_clusters_cce(
        num_classes: int,
        input_dim: int,
        num_samples_per_class: int = 50,
        separation: float = 3.0,
        seed: int = 42,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Generate well-separated Gaussian clusters for CCE convergence.

        Each class is a Gaussian blob centered at a distinct point in
        input space.  With ``separation=3.0`` and unit variance, clusters
        are ~6σ apart — trivially separable by a single hidden layer.

        Returns:
            (X, targets) where X is (N, input_dim) float64 and
            targets is (N,) int64.
        """
        gen = torch.Generator().manual_seed(seed)
        dtype = torch.float64

        # Class centers: random directions scaled by separation
        centers = torch.randn(
            num_classes, input_dim, generator=gen, dtype=dtype,
        ) * separation

        x_parts: list[torch.Tensor] = []
        y_parts: list[torch.Tensor] = []
        for c in range(num_classes):
            noise = torch.randn(
                num_samples_per_class, input_dim,
                generator=gen, dtype=dtype,
            ) * 0.5
            x_parts.append(centers[c].unsqueeze(0) + noise)
            y_parts.append(
                torch.full(
                    (num_samples_per_class,), c, dtype=torch.long,
                ),
            )

        X = torch.cat(x_parts, dim=0)
        y = torch.cat(y_parts, dim=0)

        # Shuffle
        perm = torch.randperm(X.shape[0], generator=gen)
        return X[perm], y[perm]

    @staticmethod
    def make_independent_bce(
        num_classes: int,
        input_dim: int,
        num_samples: int = 200,
        active_probability: float = 0.3,
        separation: float = 2.0,
        seed: int = 42,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Generate multi-label data for BCE convergence testing.

        Each class label is generated independently: a random
        hyperplane separates active/inactive, with Gaussian noise
        on both sides.

        Returns:
            (X, targets) where X is (N, input_dim) float64 and
            targets is (N, num_classes) float64 binary.
        """
        gen = torch.Generator().manual_seed(seed)
        dtype = torch.float64

        X = torch.randn(
            num_samples, input_dim, generator=gen, dtype=dtype,
        )

        # Random hyperplanes for each class
        normals: torch.Tensor = torch.randn(
            num_classes, input_dim, generator=gen, dtype=dtype,
        )
        normals = normals / normals.norm(dim=1, keepdim=True)  # pyright: ignore[reportUnknownMemberType, reportUnknownVariableType]

        # Bias to achieve approximate active_probability
        # P(w·x > b) ≈ active_probability for x ~ N(0,I)
        bias = torch.tensor(
            torch.distributions.Normal(0, 1).icdf(  # pyright: ignore[reportUnknownMemberType]
                torch.tensor(1.0 - active_probability),
            ),
            dtype=dtype,
        )
        projections: torch.Tensor = X @ normals.T  # pyright: ignore[reportUnknownMemberType, reportUnknownVariableType]
        targets: torch.Tensor = (projections > bias).to(dtype) * separation  # pyright: ignore[reportUnknownMemberType, reportUnknownVariableType]
        targets = (targets > 0).to(dtype)  # pyright: ignore[reportUnknownMemberType, reportUnknownVariableType]
        return X, targets  # pyright: ignore[reportUnknownVariableType]

    # =================================================================
    # Internal: forward pass
    # =================================================================

    def _forward_and_loss(
        self,
        X: torch.Tensor,
        targets: torch.Tensor,
        mask_float: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Standard PyTorch forward pass (autograd-tracked).

        Implements Nodes 4→5→6/7 using ``F.relu``, ``einsum``,
        ``F.log_softmax`` / ``F.logsigmoid``.  No manual gradient
        formulas.  The autograd graph flows from parameters through
        to the scalar loss.

        Returns:
            (probs, loss) where probs has shape
            (num_modules, batch, output_classes).
        """
        M = self.config.num_modules
        B = X.shape[0]

        # Node 4: shared layer
        H = F.relu(X @ self.W_shared.T + self.b_shared)

        # Node 5: module logits + sample masking
        logits = (
            torch.einsum("bh,mhc->mbc", H, self.W_module)
            + self.b_module.unsqueeze(1)
        )
        logits = logits * mask_float[None, :, None]

        # Nodes 6/7: probabilities + loss
        if self.config.mode == "CCE":
            total_loss = torch.zeros(1, dtype=self.dtype)
            probs = torch.zeros_like(logits)
            for m in range(M):
                scaled = logits[m] / self.temps[m]
                log_p = F.log_softmax(scaled, dim=-1)
                probs[m] = torch.exp(log_p)
                per_sample = -log_p[torch.arange(B), targets]
                total_loss = total_loss + (
                    per_sample * mask_float
                ).sum()
            return probs, total_loss.squeeze()
        else:  # BCE
            total_loss = torch.zeros(1, dtype=self.dtype)
            probs = torch.zeros_like(logits)
            for m in range(M):
                scaled = logits[m] / self.temps[m]
                probs[m] = torch.sigmoid(scaled)
                log_p = F.logsigmoid(scaled)
                log_1mp = F.logsigmoid(-scaled)
                loss_per_elem = -(
                    targets * log_p + (1.0 - targets) * log_1mp
                )
                total_loss = total_loss + (
                    loss_per_elem * mask_float[:, None]
                ).sum()
            return probs, total_loss.squeeze()
