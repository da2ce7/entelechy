# tests/oracle/reference_utils.py
"""Pure utility functions shared by all oracle variants.

These are the canonical mathematical operations that Oracle A, B1, and B2
all agree on. They operate in FP64 throughout (torch.float64).

Reference: doc_archive/Oracle.md §Shared Infrastructure
"""
from __future__ import annotations

import torch


def temperature_scaled_softmax(logits: torch.Tensor, temps: torch.Tensor) -> torch.Tensor:
    """Numerically stable softmax with per-module temperature scaling.

    Args:
        logits: (batch, classes) or (batch, modules, classes) — unscaled.
        temps: scalar or broadcastable temperature tensor.

    Returns:
        Probability tensor, same shape as logits.
    """
    scaled = logits / temps
    shifted = scaled - scaled.max(dim=-1, keepdim=True).values
    exp_shifted = torch.exp(shifted)
    return exp_shifted / exp_shifted.sum(dim=-1, keepdim=True)


def temperature_scaled_sigmoid(logits: torch.Tensor, temps: torch.Tensor) -> torch.Tensor:
    """Numerically stable sigmoid with per-module temperature scaling.

    Args:
        logits: unscaled logits.
        temps: broadcastable temperature tensor.

    Returns:
        Probability tensor, same shape as logits.
    """
    scaled = logits / temps
    return torch.sigmoid(scaled)


def group_wise_clip(
    grad_list: list[torch.Tensor],
    threshold: float,
    eps: float = 1e-12,
) -> list[torch.Tensor]:
    """Node 11 / Node 19: single L2 norm over concatenated gradients, single scale.

    Computes a single L2 norm across the logical concatenation of all tensors
    in grad_list. If the norm exceeds threshold, all tensors are uniformly
    scaled down.

    Threshold semantics (matches kernel behavior):
      - threshold < 0: bypass clipping (diagnostic mode)
      - threshold == 0: zero all gradients
      - threshold > 0: standard L2-norm clipping

    Args:
        grad_list: list of gradient tensors forming the "virtual vector".
        threshold: clipping threshold.
        eps: numerical stability epsilon.

    Returns:
        New list of (possibly scaled) gradient tensors, same shapes.
    """
    if threshold < 0.0:
        return [g.clone() for g in grad_list]
    if threshold == 0.0:
        return [torch.zeros_like(g) for g in grad_list]

    concat = torch.cat([g.flatten() for g in grad_list])
    norm_val = float(concat.norm(2))  # pyright: ignore[reportUnknownMemberType, reportUnknownArgumentType]
    if norm_val > threshold:
        scale = threshold / (norm_val + eps)
        return [g * scale for g in grad_list]
    return [g.clone() for g in grad_list]


def adam_update_fp64(
    param: torch.Tensor,
    grad: torch.Tensor,
    m1: torch.Tensor,
    m2: torch.Tensor,
    lr: float,
    beta1: float,
    beta2: float,
    eps: float,
    beta1_pow_t: float,
    beta2_pow_t: float,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Manual Adam with FP64 bias correction throughout.

    Matches the engine's kernel behavior: bias correction powers are
    pre-computed on the host in FP64 and passed in.

    Args:
        param: current parameter values.
        grad: final (normalized) gradient.
        m1: first moment estimate.
        m2: second moment estimate.
        lr: learning rate.
        beta1, beta2: EMA decay rates.
        eps: Adam epsilon.
        beta1_pow_t: beta1 ** t (pre-computed in FP64).
        beta2_pow_t: beta2 ** t (pre-computed in FP64).

    Returns:
        (updated_param, updated_m1, updated_m2)
    """
    m1_new = beta1 * m1 + (1.0 - beta1) * grad
    m2_new = beta2 * m2 + (1.0 - beta2) * grad * grad
    m1_hat = m1_new / (1.0 - beta1_pow_t)
    m2_hat = m2_new / (1.0 - beta2_pow_t)
    param_new = param - lr * m1_hat / (torch.sqrt(m2_hat) + eps)
    return param_new, m1_new, m2_new
