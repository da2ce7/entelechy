"""Oracle B: Autograd Oracle — gradient computation via torch.autograd.

Two modes:
  B1 (flat):  Standard loss.backward(). No per-tile or per-stage clipping.
              Authority: gradient calculus correctness.
              Weakness: no clipping — diverges from engine when any clipping
              activates.  Useful as a pure-calculus reference: if the sum of
              Oracle A's per-tile contributions (before clipping) doesn't
              match B1's gradients, there is a formula error.

  B2 (tiled): Full autograd for W_module/b_module (sliced per tile, valid
              because their gradient decomposes by class column) + manual
              per-tile decomposition for temps/grad_H (which require
              class-chunk-level granularity that autograd totals cannot
              provide).  Same group-wise clip + staged reduction as Oracle A.
              Authority: calculus (autograd validates all parameter groups)
              AND pipeline fidelity.  Full autograd reference gradients are
              stored in ``autograd_reference_grads`` for cross-validation.

Backend modeling note: Node 16's reduction topology follows the CPU
backend's rendering — a flat ⌈log_K(M)⌉-stage schedule without
pre-accumulation (CONCEPT §11).

Reference: doc_archive/Oracle.md §Option B: Autograd Oracle
"""
from __future__ import annotations

import math
from typing import Literal

import torch
import torch.nn as nn
import torch.nn.functional as F

from .oracle_config import OracleConfig
from .reference_utils import adam_update_fp64, group_wise_clip


class AutogradOracle:
    """Autograd-based FP64 implementation of the training pipeline.

    Use ``mode='flat'`` for B1, ``mode='tiled'`` for B2.
    """

    def __init__(
        self,
        config: OracleConfig,
        mode: Literal["flat", "tiled"] = "flat",
    ) -> None:
        self.config = config
        self.oracle_mode = mode
        self.dtype = torch.float64
        self.tiling = config.tiling

        # nn.Parameters for autograd tracking
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
                config.num_modules, config.hidden_dim, config.output_classes,
                dtype=self.dtype,
            ),
        )
        self.b_module = nn.Parameter(
            torch.zeros(
                config.num_modules, config.output_classes, dtype=self.dtype,
            ),
        )
        self.temps = nn.Parameter(
            torch.ones(config.num_modules, dtype=self.dtype),
        )
        self.all_params = [
            self.W_shared, self.b_shared,
            self.W_module, self.b_module, self.temps,
        ]
        self.param_names = [
            "W_shared", "b_shared", "W_module", "b_module", "temps",
        ]

        # Manual Adam state (NOT torch.optim — we need FP64 bias correction)
        self.m1: dict[str, torch.Tensor] = {}
        self.m2: dict[str, torch.Tensor] = {}
        for name, param in zip(self.param_names, self.all_params):
            self.m1[name] = torch.zeros_like(param.data)
            self.m2[name] = torch.zeros_like(param.data)
        self.t = 0

        self.last_loss: float = float("nan")

        # Full autograd reference gradients, stored each step for
        # cross-validation by tests.  Tests can compare the sum of
        # manual per-tile contributions against these totals.
        self.autograd_reference_grads: dict[str, torch.Tensor] = {}
        # dL/dH (post-ReLU): autograd total for validating per-tile
        # grad_H manual formulas.  Only populated by B2.
        self.autograd_reference_grad_H: torch.Tensor | None = None

    def named_params(self) -> list[tuple[str, nn.Parameter]]:
        return list(zip(self.param_names, self.all_params))

    def export_state(self) -> dict[str, torch.Tensor]:
        state: dict[str, torch.Tensor] = {}
        for name, param in self.named_params():
            state[name] = param.data.clone()
            state[f"m1_{name}"] = self.m1[name].clone()
            state[f"m2_{name}"] = self.m2[name].clone()
        state["_step"] = torch.tensor(self.t, dtype=torch.int64)
        return state

    def load_state(self, state: dict[str, torch.Tensor]) -> None:
        for name, param in self.named_params():
            param.data.copy_(state[name])
            self.m1[name].copy_(state[f"m1_{name}"])
            self.m2[name].copy_(state[f"m2_{name}"])
        self.t = int(state["_step"].item())

    # =================================================================
    # Public API
    # =================================================================

    def step(
        self,
        X: torch.Tensor,
        targets: torch.Tensor,
        sample_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """One full training step.

        Args:
            X: (batch, input_dim) float64.
            targets: CCE: (batch,) int64; BCE: (batch, output_classes) float64.
            sample_mask: optional (batch,) bool tensor.  True = valid sample,
                         False = padding/invalid.  When None, all samples are
                         treated as valid (ADR-031).

        Returns:
            probs: (num_modules, batch, output_classes).
        """
        X = X.to(self.dtype)  # pyright: ignore[reportConstantRedefinition]
        batch_size = X.shape[0]

        if sample_mask is None:
            sample_mask = torch.ones(batch_size, dtype=torch.bool)
        else:
            sample_mask = sample_mask.to(torch.bool)
        effective_batch_size = int(sample_mask.sum().item())
        mask_float = sample_mask.to(self.dtype)

        if self.oracle_mode == "flat":
            return self._step_flat(
                X, targets, batch_size,
                sample_mask, mask_float, effective_batch_size,
            )
        else:
            return self._step_tiled(
                X, targets, batch_size,
                sample_mask, mask_float, effective_batch_size,
            )

    # =================================================================
    # B1: Flat autograd (no clipping)
    # =================================================================

    def _step_flat(
        self,
        X: torch.Tensor,
        targets: torch.Tensor,
        batch_size: int,
        sample_mask: torch.Tensor,
        mask_float: torch.Tensor,
        effective_batch_size: int,
    ) -> torch.Tensor:
        """B1: full autograd backward, no per-tile or per-stage clipping.

        Produces mathematically exact gradients (no clipping distortion).
        Useful as a calculus reference: if the sum of Oracle A's per-tile
        contributions (before any clipping) doesn't match B1's gradients,
        there is a formula error in the manual gradient computation.
        """
        # Zero grads
        for p in self.all_params:
            if p.grad is not None:
                p.grad.zero_()

        # Autograd forward
        probs, loss = self._autograd_forward(X, targets, mask_float)
        self.last_loss = loss.item()

        # Backward via autograd
        loss.backward()  # pyright: ignore[reportUnknownMemberType]

        # Store reference gradients for cross-validation
        self.autograd_reference_grads = {
            name: param.grad.data.detach().clone()  # pyright: ignore[reportOptionalMemberAccess]
            for name, param in self.named_params()
        }

        # Node 21: normalize by effective batch size
        eps = self.config.epsilon
        N = float(effective_batch_size)
        self.final_grads: dict[str, torch.Tensor] = {}
        for name, param in self.named_params():
            assert param.grad is not None
            self.final_grads[name] = param.grad.data.clone() / (N + eps)

        # Adam + clamp
        self._optimizer_step()

        return probs.detach()

    # =================================================================
    # B2: Tiled autograd with clipping pipeline
    # =================================================================

    def _step_tiled(
        self,
        X: torch.Tensor,
        targets: torch.Tensor,
        batch_size: int,
        sample_mask: torch.Tensor,
        mask_float: torch.Tensor,
        effective_batch_size: int,
    ) -> torch.Tensor:
        """B2: per-tile clipping pipeline matching Oracle A.

        Uses full-loss autograd for all 5 parameter groups.  W_module and
        b_module gradients are sliced from the autograd total (valid because
        their gradient decomposes by class column — each W_module[m,h,c]
        only receives gradient from the (m,c) pair in d_logits).  Temps and
        grad_H use manual per-tile decomposition (necessary because autograd
        totals can't be split into per-class-chunk contributions for these
        coupled parameters).

        Full autograd reference gradients are stored in
        ``self.autograd_reference_grads`` and
        ``self.autograd_reference_grad_H`` for test cross-validation.
        """
        # --- Autograd forward ---
        probs, total_loss, H_tracked = self._autograd_forward_with_H(
            X, targets, mask_float,
        )
        self.last_loss = total_loss.item()

        # --- Full autograd gradients for all 5 params + H ---
        all_grad_targets = self.all_params + [H_tracked]
        full_grads = torch.autograd.grad(
            total_loss,
            all_grad_targets,
            retain_graph=False,
        )
        # Store reference gradients for cross-validation
        self.autograd_reference_grads = {
            name: full_grads[i].detach().clone()
            for i, name in enumerate(self.param_names)
        }
        self.autograd_reference_grad_H = full_grads[-1].detach().clone()

        # W_module and b_module: slice from autograd (column-decomposable)
        autograd_grad_Wmod = full_grads[2].detach()  # (M, hidden, classes)
        autograd_grad_bmod = full_grads[3].detach()  # (M, classes)

        # --- Manual intermediates (detached, for per-tile decomposition) ---
        H_data = F.relu(X @ self.W_shared.data.T + self.b_shared.data)
        hidden_mask = (H_data > 0.0).to(self.dtype)
        logits = (
            torch.einsum("bh,mhc->mbc", H_data, self.W_module.data)
            + self.b_module.data.unsqueeze(1)
        )
        logits *= mask_float[None, :, None]  # Node 5 sample masking

        # --- Phase I: Per-tile gradient generation ---
        clipped_grad_Wmod_tiles: list[torch.Tensor] = []
        clipped_grad_bmod_tiles: list[torch.Tensor] = []
        clipped_grad_temp_tiles: list[torch.Tensor] = []
        clipped_grad_H_tiles: list[torch.Tensor] = []

        for tile_idx in range(self.tiling.total_tiles):
            mc = tile_idx // self.tiling.num_class_chunks
            cc = tile_idx % self.tiling.num_class_chunks

            m_start = mc * self.tiling.modules_per_chunk
            m_end = min(
                m_start + self.tiling.modules_per_chunk,
                self.config.num_modules,
            )
            c_start = cc * self.tiling.classes_per_chunk
            c_end = min(
                c_start + self.tiling.classes_per_chunk,
                self.config.output_classes,
            )

            # W_module, b_module: slice from full autograd gradient.
            # Valid because dL/dW_module[m,h,c] decomposes by (m,c) pair.
            grad_Wmod = autograd_grad_Wmod[
                m_start:m_end, :, c_start:c_end
            ].clone()
            grad_bmod = autograd_grad_bmod[
                m_start:m_end, c_start:c_end
            ].clone()

            # Temps and grad_H: manual per-tile decomposition.
            # Autograd totals can't be split into per-class-chunk
            # contributions because softmax (CCE) couples all classes.
            probs_tile = probs.detach()[m_start:m_end, :, c_start:c_end]
            d_logits_pre_temp = self._compute_d_logits_tile(
                probs_tile, targets, batch_size,
                m_start, m_end, c_start, c_end,
            )
            # Zero masked samples' gradient contributions
            d_logits_pre_temp *= mask_float[None, :, None]

            # Node 10: temperature gradient (uses d_logits WITHOUT /T)
            logits_tile = logits[m_start:m_end, :, c_start:c_end]
            temp_grad_sum = torch.einsum(
                "mbc,mbc->m", d_logits_pre_temp, logits_tile,
            )
            grad_temps = temp_grad_sum * (
                -1.0 / (self.temps.data[m_start:m_end] ** 2)
            )

            # Node 9: grad_H per tile (uses d_logits WITH /T)
            inv_temps = (
                (1.0 / self.temps.data[m_start:m_end])
                .unsqueeze(1)
                .unsqueeze(2)
            )
            d_logits_tile = d_logits_pre_temp * inv_temps
            W_tile = self.W_module.data[m_start:m_end, :, c_start:c_end]
            grad_H_tile = torch.einsum(
                "mbc,mhc->mbh", d_logits_tile, W_tile,
            )

            # Node 11: group-wise clip
            threshold = self.config.node11_clip_threshold()
            clipped = group_wise_clip(
                [grad_Wmod, grad_bmod, grad_temps, grad_H_tile],
                threshold,
            )
            clipped_grad_Wmod_tiles.append(clipped[0])
            clipped_grad_bmod_tiles.append(clipped[1])
            clipped_grad_temp_tiles.append(clipped[2])
            clipped_grad_H_tiles.append(clipped[3])

        # --- Phase II: Gather + Reduce ---

        # Node 13: sum clipped Grad_H across class chunks per module
        summed_grad_H_per_module = torch.zeros(
            self.config.num_modules, batch_size, self.config.hidden_dim,
            dtype=self.dtype,
        )
        for tile_idx, grad_H_tile in enumerate(clipped_grad_H_tiles):
            mc = tile_idx // self.tiling.num_class_chunks
            m_start = mc * self.tiling.modules_per_chunk
            m_end = min(
                m_start + self.tiling.modules_per_chunk,
                self.config.num_modules,
            )
            tile_m = m_end - m_start
            summed_grad_H_per_module[m_start:m_end] += grad_H_tile[:tile_m]

        # Node 16: staged reduction across modules
        summed_grad_H = self._staged_reduce_node16(
            summed_grad_H_per_module, batch_size,
        )

        # Node 15: per-module-chunk reduction for module & temp gradients.
        # Per ADR-030, each module chunk's tiles are reduced independently.
        # Tiles are zero-padded to full class width before staged reduction.
        summed_grad_Wmod = torch.zeros_like(self.W_module.data)
        summed_grad_bmod = torch.zeros_like(self.b_module.data)
        summed_grad_temps = torch.zeros(
            self.config.num_modules, dtype=self.dtype,
        )

        num_mc = self.tiling.num_module_chunks
        for mc in range(num_mc):
            m_start = mc * self.tiling.modules_per_chunk
            m_end = min(
                m_start + self.tiling.modules_per_chunk,
                self.config.num_modules,
            )
            tile_m = m_end - m_start

            mc_wmod_tiles: list[torch.Tensor] = []
            mc_bmod_tiles: list[torch.Tensor] = []
            mc_temp_tiles: list[torch.Tensor] = []

            for cc in range(self.tiling.num_class_chunks):
                tile_idx = mc * self.tiling.num_class_chunks + cc
                c_start = cc * self.tiling.classes_per_chunk
                c_end = min(
                    c_start + self.tiling.classes_per_chunk,
                    self.config.output_classes,
                )

                wmod_full = torch.zeros(
                    tile_m, self.config.hidden_dim,
                    self.config.output_classes,
                    dtype=self.dtype,
                )
                wmod_full[:, :, c_start:c_end] = (
                    clipped_grad_Wmod_tiles[tile_idx]
                )
                mc_wmod_tiles.append(wmod_full)

                bmod_full = torch.zeros(
                    tile_m, self.config.output_classes, dtype=self.dtype,
                )
                bmod_full[:, c_start:c_end] = (
                    clipped_grad_bmod_tiles[tile_idx]
                )
                mc_bmod_tiles.append(bmod_full)

                mc_temp_tiles.append(clipped_grad_temp_tiles[tile_idx])

            summed_grad_Wmod[m_start:m_end] = (
                self._reduce_tiles_sum_and_clip(mc_wmod_tiles, "W_module")
            )
            summed_grad_bmod[m_start:m_end] = (
                self._reduce_tiles_sum_and_clip(mc_bmod_tiles, "b_module")
            )
            summed_grad_temps[m_start:m_end] = (
                self._reduce_tiles_sum_and_clip(mc_temp_tiles, "temps")
            )

        # --- Phase III: Streaming shared backprop (Nodes 17→18→19) ---
        num_batch_chunks: int = getattr(
            self.tiling, "num_batch_chunks", batch_size,
        )
        batch_chunk_size = math.ceil(batch_size / num_batch_chunks)

        clipped_sw_chunks: list[torch.Tensor] = []
        clipped_sb_chunks: list[torch.Tensor] = []

        for chunk_idx in range(num_batch_chunks):
            b_start = chunk_idx * batch_chunk_size
            b_end = min(b_start + batch_chunk_size, batch_size)

            grad_sw_chunk = torch.zeros(
                self.config.hidden_dim, self.config.input_dim,
                dtype=self.dtype,
            )
            grad_sb_chunk = torch.zeros(
                self.config.hidden_dim, dtype=self.dtype,
            )

            for b in range(b_start, b_end):
                if not sample_mask[b]:
                    continue
                grad_h_b = summed_grad_H[b]
                d_act = hidden_mask[b]
                dL_dZ = grad_h_b * d_act
                grad_sw_chunk += torch.outer(dL_dZ, X[b])
                grad_sb_chunk += dL_dZ

            threshold_shared = self.config.leaf_safety_threshold()
            clipped = group_wise_clip(
                [grad_sw_chunk, grad_sb_chunk], threshold_shared,
            )
            clipped_sw_chunks.append(clipped[0])
            clipped_sb_chunks.append(clipped[1])

        # --- Phase IV: Final aggregation + normalization ---

        # Node 20: reduce shared grads with staged clip
        summed_grad_sw = self._reduce_chunks_sum_and_clip(
            clipped_sw_chunks, "shared_weights",
        )
        summed_grad_sb = self._reduce_chunks_sum_and_clip(
            clipped_sb_chunks, "shared_biases",
        )

        # Node 21: normalize by effective batch size
        eps = self.config.epsilon
        N = float(effective_batch_size)
        self.final_grads = {
            "W_module": summed_grad_Wmod / (N + eps),
            "b_module": summed_grad_bmod / (N + eps),
            "temps": summed_grad_temps / (N + eps),
            "W_shared": summed_grad_sw / (N + eps),
            "b_shared": summed_grad_sb / (N + eps),
        }

        # --- Phase V: Adam + clamp ---
        self._optimizer_step()

        return probs.detach()

    # =================================================================
    # Autograd forward passes
    # =================================================================

    def _autograd_forward(
        self,
        X: torch.Tensor,
        targets: torch.Tensor,
        mask_float: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Autograd-tracked forward: Nodes 4→5→6/7.

        Returns (probs, loss).  Used by B1.
        """
        probs, loss, _ = self._autograd_forward_core(X, targets, mask_float)
        return probs, loss

    def _autograd_forward_with_H(
        self,
        X: torch.Tensor,
        targets: torch.Tensor,
        mask_float: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Autograd-tracked forward returning H for gradient extraction.

        Returns (probs, loss, H_tracked).  Used by B2 to extract
        dL/dH via torch.autograd.grad.
        """
        return self._autograd_forward_core(X, targets, mask_float)

    def _autograd_forward_core(
        self,
        X: torch.Tensor,
        targets: torch.Tensor,
        mask_float: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Shared autograd forward implementation.

        Returns (probs, loss, H) where H is the post-ReLU hidden
        activation tensor (part of the autograd graph, extractable
        as a gradient target by B2).
        """
        M = self.config.num_modules
        B = X.shape[0]

        # Node 4: shared layer
        H = F.relu(X @ self.W_shared.T + self.b_shared)

        # Node 5: logits with sample masking
        logits = (
            torch.einsum("bh,mhc->mbc", H, self.W_module)
            + self.b_module.unsqueeze(1)
        )
        logits = logits * mask_float[None, :, None]

        # Nodes 6/7: probs and loss
        if self.config.mode == "CCE":
            probs = torch.zeros_like(logits)
            total_loss = torch.zeros(1, dtype=self.dtype)
            for m in range(M):
                scaled = logits[m] / self.temps[m]
                log_p = F.log_softmax(scaled, dim=-1)
                probs[m] = torch.exp(log_p)
                # Loss only for valid samples
                per_sample_loss = -log_p[torch.arange(B), targets]
                total_loss = total_loss + (
                    per_sample_loss * mask_float
                ).sum()
            return probs, total_loss.squeeze(), H
        else:  # BCE
            probs = torch.zeros_like(logits)
            total_loss = torch.zeros(1, dtype=self.dtype)
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
            return probs, total_loss.squeeze(), H

    # =================================================================
    # d_logits (manual, matching Oracle A)
    # =================================================================

    def _compute_d_logits_tile(
        self,
        probs_tile: torch.Tensor,
        targets: torch.Tensor,
        batch_size: int,
        m_start: int,
        m_end: int,
        c_start: int,
        c_end: int,
    ) -> torch.Tensor:
        """dL/d(scaled_logits) for a tile, BEFORE temperature chain rule.

        For CCE: d_logits[m, b, c] = probs[m, b, c] - 1_{c == target[b]}
        For BCE: d_logits[m, b, c] = probs[m, b, c] - targets[b, c]

        Sample masking is NOT applied here — the caller zeroes masked
        samples' contributions after this returns.

        Matches FaithfulOracle._compute_d_logits_tile exactly.
        """
        tile_M = m_end - m_start

        if self.config.mode == "CCE":
            d_logits = probs_tile.clone()
            for m_local in range(tile_M):
                for b in range(batch_size):
                    c = int(targets[b].item())
                    if c_start <= c < c_end:
                        d_logits[m_local, b, c - c_start] -= 1.0
            return d_logits
        else:  # BCE
            targets_slice = targets[:, c_start:c_end]
            d_logits = probs_tile - targets_slice.unsqueeze(0).expand_as(
                probs_tile,
            )
            return d_logits

    # =================================================================
    # Reduction (same algorithms as revised Faithful Oracle)
    # =================================================================

    def _staged_reduce_node16(
        self,
        grad_H_per_module: torch.Tensor,
        batch_size: int,
    ) -> torch.Tensor:
        """Node 16: staged reduction across modules with per-stage clipping.

        Models the CPU backend's flat ⌈log_K(M)⌉-stage schedule
        (CONCEPT §11).  Includes Pre-Summation Amplification Factor
        A_j = num_class_chunks for the first stage (CONCEPT §3.4).
        """
        M = self.config.num_modules
        if M <= 1:
            return grad_H_per_module.squeeze(0)

        K, num_stages = self._plan_reduction_tree(M)
        num_class_chunks = self.tiling.num_class_chunks

        flat = grad_H_per_module.permute(1, 2, 0).reshape(-1, M)
        result = torch.zeros(flat.shape[0], dtype=self.dtype)

        for row_idx in range(flat.shape[0]):
            current: list[float] = flat[row_idx].tolist()  # pyright: ignore[reportUnknownMemberType, reportUnknownVariableType]
            num_items = len(current)

            for s in range(num_stages):
                j = num_stages - 1 - s  # distance from root
                T_policy = (
                    self.config.t_algorithmic
                    + self.config.lambda_ * (j * j)
                )
                K_actual = min(K, num_items)
                # Pre-Summation Amplification Factor (CONCEPT §3.4):
                # first stage inputs carry Node 13's class-chunk sum
                A_j = num_class_chunks if s == 0 else 1
                denominator = K_actual * A_j
                T_safety = (
                    self.config.compute_fp_format_max / denominator
                    if denominator > 0
                    else self.config.compute_fp_format_max
                )
                threshold = min(T_policy, T_safety)

                new_items: list[float] = []
                for g in range(0, num_items, K):
                    end = min(g + K, num_items)
                    s_val = sum(current[g:end])
                    norm = abs(s_val)
                    if norm > threshold:
                        s_val *= threshold / (norm + self.config.epsilon)
                    new_items.append(s_val)
                current = new_items
                num_items = len(current)

            result[row_idx] = current[0]

        return result.reshape(batch_size, self.config.hidden_dim)

    def _reduce_tiles_sum_and_clip(
        self,
        tiles: list[torch.Tensor],
        param_name: str,
    ) -> torch.Tensor:
        """Node 15: reduction tree with per-stage clipping for tiled partials.

        Args:
            tiles: list of equal-shaped gradient tensors to reduce.
            param_name: identifier for diagnostics (unused in computation).
        """
        if not tiles:
            raise ValueError("No tiles to reduce")
        if len(tiles) == 1:
            return tiles[0].clone()

        flat_tiles = [t.flatten() for t in tiles]
        N = len(flat_tiles)
        K, num_stages = self._plan_reduction_tree(N)

        current = flat_tiles
        for s in range(num_stages):
            j = num_stages - 1 - s
            threshold = self._get_threshold_for_stage(j, K)
            next_level: list[torch.Tensor] = []
            for i in range(0, len(current), K):
                group = current[i : i + K]
                summed = torch.stack(group).sum(dim=0)
                norm_val = float(summed.norm(2))  # pyright: ignore[reportUnknownMemberType, reportUnknownArgumentType]
                if threshold >= 0.0 and norm_val > threshold:
                    summed = summed * (
                        threshold / (norm_val + self.config.epsilon)
                    )
                next_level.append(summed)
            current = next_level

        return current[0].reshape(tiles[0].shape)

    def _reduce_chunks_sum_and_clip(
        self,
        chunks: list[torch.Tensor],
        param_name: str,
    ) -> torch.Tensor:
        """Node 20: reduction tree for streaming shared grad chunks."""
        return self._reduce_tiles_sum_and_clip(chunks, param_name)

    def _plan_reduction_tree(self, num_partials: int) -> tuple[int, int]:
        """Determine (K, num_stages) for a uniform reduction tree.

        Matches revised FaithfulOracle._plan_reduction_tree.

        Returns:
            (K, num_stages) where K is the fan-in.
            For num_partials <= 1: (2, 0) — identity bypass, no reduction.
            For num_partials >= 2: (K, num_stages) where num_stages >= 1.
        """
        if num_partials <= 1:
            return 2, 0  # Identity tier: no reduction needed
        max_fan_in = 256
        T_root = self.config.t_algorithmic

        if (
            math.isinf(self.config.compute_fp_format_max)
            or math.isinf(T_root)
            or T_root <= 0
        ):
            safe_k = max_fan_in
        else:
            safe_k_root = self.config.compute_fp_format_max / T_root
            safe_k = min(max_fan_in, int(safe_k_root))

        safe_k = max(2, safe_k)
        num_stages = math.ceil(
            math.log(num_partials) / math.log(safe_k),
        )
        return safe_k, max(1, num_stages)

    def _get_threshold_for_stage(
        self, stage_j: int, K: int,
    ) -> float:
        """Compute final clipping threshold for generic reduction stage j.

        Implements CONCEPT §3.4 two-step logic:
            policy_threshold = T_algorithmic + λ·j²
            final_threshold  = min(policy_threshold, T_safety)
        where T_safety = COMPUTE_FP_FORMAT_MAX / K.

        No floor — matches revised FaithfulOracle.
        """
        T_policy = (
            self.config.t_algorithmic
            + self.config.lambda_ * (stage_j ** 2)
        )
        T_safety = (
            self.config.compute_fp_format_max / K
            if K > 0
            else self.config.compute_fp_format_max
        )
        return min(T_policy, T_safety)

    # =================================================================
    # Optimizer
    # =================================================================

    def _optimizer_step(self) -> None:
        """Node 24 + 25: Manual Adam + temperature clamp."""
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

        self.temps.data.clamp_(self.config.temp_min, self.config.temp_max)
