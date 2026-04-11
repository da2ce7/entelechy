"""Oracle A: Faithful Oracle — manual forward/backward with per-tile clipping.

Replicates the engine's exact node sequence (Nodes 4→25) in FP64 PyTorch
with torch.no_grad(). Zero autograd. All operations are manual.

Authority: pipeline fidelity (matches the engine's tiled decomposition,
group-wise clip, staged reduction, and Adam update sequence).

Weakness: manual gradient formulas could contain errors — Oracle B
(autograd) is the arbiter of calculus correctness.

Reference: doc_archive/Oracle.md §Option A: Faithful Oracle

Backend modeling note: Node 16's reduction topology follows the CPU
backend's rendering — a flat ⌈log_K(M)⌉-stage schedule without
pre-accumulation (CONCEPT §11). GPU backends use a two-phase structure
(pre-accumulation + staged binary tree over min(M, workgroup_size)
intermediates) that may produce numerically different results when
intermediate clipping fires, because clipping is nonlinear and the
summation grouping differs.
"""
from __future__ import annotations

import math

import torch

from .oracle_config import OracleConfig
from .reference_utils import adam_update_fp64, group_wise_clip


class FaithfulOracle:
    """Manual FP64 implementation of the full training pipeline.

    State is maintained across steps: parameters, Adam moments, step counter.
    All tensors are torch.float64 on CPU.
    """

    def __init__(self, config: OracleConfig) -> None:
        self.config = config
        self.dtype = torch.float64
        self.tiling = config.tiling

        # --- Learnable parameters ---
        self.W_shared = torch.zeros(
            config.hidden_dim, config.input_dim, dtype=self.dtype,
        )
        self.b_shared = torch.zeros(config.hidden_dim, dtype=self.dtype)
        self.W_module = torch.zeros(
            config.num_modules, config.hidden_dim, config.output_classes,
            dtype=self.dtype,
        )
        self.b_module = torch.zeros(
            config.num_modules, config.output_classes, dtype=self.dtype,
        )
        self.temps = torch.ones(config.num_modules, dtype=self.dtype)

        # --- Adam state ---
        self.m1: dict[str, torch.Tensor] = {}
        self.m2: dict[str, torch.Tensor] = {}
        for name, param in self.named_params():
            self.m1[name] = torch.zeros_like(param)
            self.m2[name] = torch.zeros_like(param)
        self.t = 0

        # --- Last-step diagnostics ---
        self.last_loss: float = float("nan")
        self.clip_stats = _ClipStats()

    def named_params(self) -> list[tuple[str, torch.Tensor]]:
        return [
            ("W_shared", self.W_shared),
            ("b_shared", self.b_shared),
            ("W_module", self.W_module),
            ("b_module", self.b_module),
            ("temps", self.temps),
        ]

    def export_state(self) -> dict[str, torch.Tensor]:
        """Return a deep copy of all learnable state (params + Adam moments)."""
        state: dict[str, torch.Tensor] = {}
        for name, param in self.named_params():
            state[name] = param.clone()
            state[f"m1_{name}"] = self.m1[name].clone()
            state[f"m2_{name}"] = self.m2[name].clone()
        state["_step"] = torch.tensor(self.t, dtype=torch.int64)
        return state

    def load_state(self, state: dict[str, torch.Tensor]) -> None:
        """Load state from a dict (e.g. produced by another oracle's export_state)."""
        for name, param in self.named_params():
            param.copy_(state[name])
            self.m1[name].copy_(state[f"m1_{name}"])
            self.m2[name].copy_(state[f"m2_{name}"])
        self.t = int(state["_step"].item())

    # =================================================================
    # Public API
    # =================================================================

    @torch.no_grad()
    def step(
        self,
        X: torch.Tensor,
        targets: torch.Tensor,
        sample_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Execute one full training step (Act + Learn).

        Args:
            X: input data, shape (batch_size, input_dim), float64.
            targets: CCE: (batch_size,) int64 class indices.
                     BCE: (batch_size, output_classes) float64 binary targets.
            sample_mask: optional (batch_size,) bool tensor. True = valid
                         sample, False = padding/invalid. When None, all
                         samples are treated as valid. Mirrors the engine's
                         packed sample_mask bitmask (ADR-031).

        Returns:
            predictions: (num_modules, batch_size, output_classes) probabilities.
                         Masked samples have probabilities derived from zeroed
                         logits (uniform for CCE, 0.5 for BCE), matching
                         Node 5's sample masking behavior.
        """
        X = X.to(self.dtype)  # pyright: ignore[reportConstantRedefinition]
        batch_size = X.shape[0]

        if sample_mask is None:
            sample_mask = torch.ones(batch_size, dtype=torch.bool)
        else:
            sample_mask = sample_mask.to(torch.bool)
        effective_batch_size = int(sample_mask.sum().item())
        # Float mask for tensor multiplication (1.0 valid, 0.0 masked)
        mask_float = sample_mask.to(self.dtype)  # (batch,)

        # --- Node 4: forward pass (shared layer) ---
        H = torch.relu(X @ self.W_shared.T + self.b_shared)  # (batch, hidden)
        hidden_mask = (H > 0.0).to(self.dtype)

        # --- Node 5: render logits (all modules, all classes) ---
        # logits[m, b, c] = H[b] @ W_module[m] + b_module[m]
        logits = (
            torch.einsum("bh,mhc->mbc", H, self.W_module)
            + self.b_module.unsqueeze(1)
        )
        # Node 5 Sample Masking: zero logits for invalid samples
        logits *= mask_float[None, :, None]

        # --- Nodes 6/7: probabilities and loss ---
        probs, loss = self._compute_probs_loss(logits, targets, sample_mask)
        self.last_loss = loss.item()

        # ===== Backward: Phase I (per-tile gradient generation) =====
        self.clip_stats = _ClipStats()

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

            probs_tile = probs[m_start:m_end, :, c_start:c_end]

            # dL/d(z_scaled) = p - target, BEFORE /T chain rule.
            # Computed once, reused by Nodes 8, 9 (after /T) and Node 10
            # (without /T), eliminating the prior redundant recomputation.
            d_logits_pre_temp = self._compute_d_logits_tile(
                probs_tile, targets, batch_size,
                m_start, m_end, c_start, c_end,
            )

            # Zero masked samples' gradient contributions.  d_logits is
            # the single source feeding Nodes 8, 9, and 10 — zeroing here
            # guarantees zero gradient contribution from masked samples
            # across all three downstream paths.
            d_logits_pre_temp *= mask_float[None, :, None]

            # Node 10: temperature gradient (uses d_logits without /T)
            # dL/dT[m] = -1/T² · Σ_{b,c} (dL/dz_s)[m,b,c] · z[m,b,c]
            logits_tile = logits[m_start:m_end, :, c_start:c_end]
            temp_grad_sum = torch.einsum(
                "mbc,mbc->m", d_logits_pre_temp, logits_tile,
            )
            grad_temps = temp_grad_sum * (
                -1.0 / (self.temps[m_start:m_end] ** 2)
            )

            # Apply temperature chain rule: dL/dz = dL/dz_s · (1/T)
            inv_temps = (
                (1.0 / self.temps[m_start:m_end])
                .unsqueeze(1)
                .unsqueeze(2)
            )
            d_logits_tile = d_logits_pre_temp * inv_temps

            # Node 8: module parameter gradients
            grad_Wmod = torch.einsum("mbc,bh->mhc", d_logits_tile, H)
            grad_bmod = d_logits_tile.sum(dim=1)  # (tile_M, tile_C)

            # Node 9: backprop error to hidden
            W_tile = self.W_module[m_start:m_end, :, c_start:c_end]
            grad_H_tile = torch.einsum("mbc,mhc->mbh", d_logits_tile, W_tile)

            # Node 11: group-wise clip (single L2 norm over all 4 buffers)
            threshold = self.config.node11_clip_threshold()
            clipped = group_wise_clip(
                [grad_Wmod, grad_bmod, grad_temps, grad_H_tile],
                threshold,
            )
            self.clip_stats.record(
                [grad_Wmod, grad_bmod, grad_temps, grad_H_tile],
                threshold,
            )

            clipped_grad_Wmod_tiles.append(clipped[0])
            clipped_grad_bmod_tiles.append(clipped[1])
            clipped_grad_temp_tiles.append(clipped[2])
            clipped_grad_H_tiles.append(clipped[3])

        # ===== Phase II: Gather + Reduce =====

        # Node 13: gather_and_permute_grad_h — sum clipped Grad_H tiles
        # across class chunks for each (module, batch, hidden) position.
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

        # Node 16: stabilize_and_reduce_grad_hidden_activations
        summed_grad_H = self._staged_reduce_node16(
            summed_grad_H_per_module, batch_size,
        )

        # Node 15: Recursive Clip-Aggregation for module & temp gradients.
        # Per ADR-030, each module chunk's tiles are reduced independently
        # through the staged tree.  Tiles are zero-padded to full class width
        # (matching the engine's ZERO_REQUIRED_ADDITIVE initialisation) so
        # the group-wise L2 clip sees the joint contribution from all class
        # chunks assembled so far at each stage.
        summed_grad_Wmod = torch.zeros_like(self.W_module)
        summed_grad_bmod = torch.zeros_like(self.b_module)
        summed_grad_temps = torch.zeros_like(self.temps)

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

                # Zero-pad to full class width for this module chunk
                wmod_full = torch.zeros(
                    tile_m, self.config.hidden_dim, self.config.output_classes,
                    dtype=self.dtype,
                )
                wmod_full[:, :, c_start:c_end] = clipped_grad_Wmod_tiles[
                    tile_idx
                ]
                mc_wmod_tiles.append(wmod_full)

                bmod_full = torch.zeros(
                    tile_m, self.config.output_classes, dtype=self.dtype,
                )
                bmod_full[:, c_start:c_end] = clipped_grad_bmod_tiles[tile_idx]
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

        # ===== Phase III: Streaming shared backprop =====
        # Nodes 17→18→19 per batch chunk.  The engine's StreamingLoopNode
        # may use batch_chunk_count > 1 per dispatch; each chunk produces
        # a sum across its samples which is then jointly clipped.  Since
        # clipping is nonlinear, chunk granularity affects the result.
        # Default: one sample per chunk (matching the most conservative
        # decomposition).
        num_batch_chunks: int = getattr(
            self.tiling, "num_batch_chunks", batch_size,
        )
        batch_chunk_size = math.ceil(batch_size / num_batch_chunks)

        clipped_sw_chunks: list[torch.Tensor] = []
        clipped_sb_chunks: list[torch.Tensor] = []

        for chunk_idx in range(num_batch_chunks):
            b_start = chunk_idx * batch_chunk_size
            b_end = min(b_start + batch_chunk_size, batch_size)

            # Accumulate gradient contributions across samples in this chunk
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
                # Node 17: grad_W_shared contribution from sample b
                grad_h_b = summed_grad_H[b]  # (hidden,)
                d_act = hidden_mask[b]  # (hidden,)
                dL_dZ = grad_h_b * d_act  # (hidden,)
                x_b = X[b]  # (input,)
                grad_sw_chunk += torch.outer(dL_dZ, x_b)

                # Node 18: grad_b_shared contribution from sample b
                grad_sb_chunk += dL_dZ

            # Node 19: clip shared gradients for this chunk
            threshold_shared = self.config.leaf_safety_threshold()
            clipped = group_wise_clip(
                [grad_sw_chunk, grad_sb_chunk], threshold_shared,
            )
            clipped_sw_chunks.append(clipped[0])
            clipped_sb_chunks.append(clipped[1])

        # ===== Phase IV: Final aggregation + normalization =====

        # Node 20: reduce shared grads with staged clip
        summed_grad_sw = self._reduce_chunks_sum_and_clip(
            clipped_sw_chunks, "shared_weights",
        )
        summed_grad_sb = self._reduce_chunks_sum_and_clip(
            clipped_sb_chunks, "shared_biases",
        )

        # Node 21: normalize all gradients by effective batch size
        eps = self.config.epsilon
        N = float(effective_batch_size)
        final_grad_Wmod = summed_grad_Wmod / (N + eps)
        final_grad_bmod = summed_grad_bmod / (N + eps)
        final_grad_temps = summed_grad_temps / (N + eps)
        final_grad_sw = summed_grad_sw / (N + eps)
        final_grad_sb = summed_grad_sb / (N + eps)

        self.final_grads = {
            "W_module": final_grad_Wmod,
            "b_module": final_grad_bmod,
            "temps": final_grad_temps,
            "W_shared": final_grad_sw,
            "b_shared": final_grad_sb,
        }

        # ===== Phase V: Parameter update =====

        # Node 24: Adam update
        self.t += 1
        beta1_pow_t = self.config.beta1 ** self.t
        beta2_pow_t = self.config.beta2 ** self.t

        for name, param in self.named_params():
            param_new, self.m1[name], self.m2[name] = adam_update_fp64(
                param,
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
            param.copy_(param_new)

        # Node 25: clamp temperatures
        self.temps.clamp_(self.config.temp_min, self.config.temp_max)

        return probs

    # =================================================================
    # Internal: forward pass helpers
    # =================================================================

    def _compute_probs_loss(
        self,
        logits: torch.Tensor,
        targets: torch.Tensor,
        sample_mask: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Nodes 6/7: compute probabilities and loss.

        Args:
            logits: (num_modules, batch, output_classes) unscaled.
                    Masked samples have zeroed logits (from Node 5).
            targets: CCE int64 (batch,) or BCE float64 (batch, classes).
            sample_mask: (batch,) bool tensor.

        Returns:
            (probs, loss) where probs has same shape as logits.
            Loss is summed only over valid (unmasked) samples.
        """
        M = self.config.num_modules
        B = logits.shape[1]
        eps = 1e-30

        if self.config.mode == "CCE":
            probs = torch.zeros_like(logits)
            total_loss = torch.tensor(0.0, dtype=self.dtype)
            for m in range(M):
                scaled = logits[m] / self.temps[m]  # (batch, classes)
                probs[m] = _temperature_scaled_softmax_manual(scaled)
                for b in range(B):
                    if not sample_mask[b]:
                        continue
                    c = int(targets[b].item())
                    total_loss += -torch.log(probs[m, b, c] + eps)
            return probs, total_loss
        else:  # BCE
            probs = torch.zeros_like(logits)
            total_loss = torch.tensor(0.0, dtype=self.dtype)
            for m in range(M):
                scaled = logits[m] / self.temps[m]
                probs[m] = torch.sigmoid(scaled)
                for b in range(B):
                    if not sample_mask[b]:
                        continue
                    p = probs[m, b]  # (classes,)
                    y = targets[b]  # (classes,)
                    log_p = torch.log(p.clamp(min=eps))
                    log_1mp = torch.log((1.0 - p).clamp(min=eps))
                    total_loss += -(y * log_p + (1.0 - y) * log_1mp).sum()
            return probs, total_loss

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
        """Compute dL/d(scaled_logits) for a tile, BEFORE temperature chain rule.

        For CCE: d_logits[m, b, c] = probs[m, b, c] - 1_{c == target[b]}
        For BCE: d_logits[m, b, c] = probs[m, b, c] - targets[b, c]

        Sample masking is NOT applied here — the caller zeroes masked
        samples' contributions after this returns.

        Returns:
            (tile_modules, batch, tile_classes) tensor.
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
            targets_slice = targets[:, c_start:c_end]  # (batch, tile_C)
            d_logits = probs_tile - targets_slice.unsqueeze(0).expand_as(
                probs_tile,
            )
            return d_logits

    # =================================================================
    # Internal: reduction with per-stage clipping
    # =================================================================

    def _staged_reduce_node16(
        self,
        grad_H_per_module: torch.Tensor,
        batch_size: int,
    ) -> torch.Tensor:
        """Node 16: stabilize_and_reduce_grad_hidden_activations.

        Reduces across modules (dim=0) with per-stage clipping.
        Input: (num_modules, batch, hidden).
        Output: (batch, hidden).

        Models the CPU backend's rendering: a flat ⌈log_K(M)⌉-stage
        schedule without pre-accumulation (CONCEPT §11).  GPU backends
        use a two-phase structure that may produce numerically different
        results when intermediate clipping fires.

        The first reduction stage accounts for the Pre-Summation
        Amplification Factor (CONCEPT §3.4): Node 13 sums across
        num_class_chunks tiles per element, so A_j = num_class_chunks
        for stage 0 (leaf) and A_j = 1 for subsequent stages.
        """
        M = self.config.num_modules
        if M <= 1:
            return grad_H_per_module.squeeze(0)

        K, num_stages = self._plan_reduction_tree(M)
        num_class_chunks = self.tiling.num_class_chunks

        # Work per-row: flatten to (batch*hidden, modules), then reduce
        flat = grad_H_per_module.permute(1, 2, 0).reshape(-1, M)  # (B*H, M)
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
                # first stage (s=0) inputs carry Node 13's class-chunk sum
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

        # Flatten tiles to 1D vectors for norm computation
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
                norm_val = float(summed.norm(2))  # pyright: ignore[reportUnknownMemberType]
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

        Mirrors the Orchestration tier's tree planning logic.

        Returns:
            (K, num_stages) where K is the fan-in.
            For num_partials <= 1: (2, 0) — identity bypass, no reduction.
            For num_partials >= 2: (K, num_stages) where num_stages >= 1.
        """
        if num_partials <= 1:
            return 2, 0  # Identity tier: no reduction needed
        max_fan_in = 256
        # Root threshold: j=0 → T_algorithmic + λ·0² = T_algorithmic
        T_root = self.config.t_algorithmic

        if (
            math.isinf(self.config.compute_fp_format_max)
            or math.isinf(T_root)
            or T_root <= 0
        ):
            # When T_root is zero/negative, clipping zeroes all gradients
            # anyway — overflow is impossible.  Use maximum fan-in.
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

        For Nodes 15 and 20, inputs are raw clipped partials with no
        upstream amplification (A_j = 1).
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


# =====================================================================
# Helpers
# =====================================================================


def _temperature_scaled_softmax_manual(
    scaled_logits: torch.Tensor,
) -> torch.Tensor:
    """Numerically stable softmax (input is already temperature-scaled)."""
    shifted = scaled_logits - scaled_logits.max(dim=-1, keepdim=True).values
    exp_vals = torch.exp(shifted)
    return exp_vals / exp_vals.sum(dim=-1, keepdim=True)


class _ClipStats:
    """Diagnostics for per-tile clipping."""

    def __init__(self) -> None:
        self.total_tiles = 0
        self.num_clipped = 0
        self._norm_ratios: list[float] = []

    def record(
        self, grad_list: list[torch.Tensor], threshold: float,
    ) -> None:
        self.total_tiles += 1
        if threshold <= 0:
            return
        concat = torch.cat([g.flatten() for g in grad_list])
        norm_val: float = float(concat.norm(2))  # pyright: ignore[reportUnknownMemberType, reportUnknownArgumentType]
        ratio: float = norm_val / threshold if threshold > 0 else 0.0
        self._norm_ratios.append(ratio)
        if norm_val > threshold:
            self.num_clipped += 1

    @property
    def clip_rate(self) -> float:
        return self.num_clipped / max(1, self.total_tiles)

    @property
    def mean_norm_over_threshold(self) -> float:
        return sum(self._norm_ratios) / max(1, len(self._norm_ratios))
