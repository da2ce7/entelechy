# tests/tier2/fixtures/numpy_forward.py
"""Numpy reference implementations for Act-phase kernels (Nodes 4-7)."""
from __future__ import annotations

from typing import Any

import numpy as np
from numpy.typing import NDArray


def ref_forward_pass(
    input_data: NDArray[np.floating[Any]],
    weights_shared: NDArray[np.floating[Any]],
    biases_shared: NDArray[np.floating[Any]],
    sample_mask: NDArray[np.floating[Any]],
) -> tuple[NDArray[np.floating[Any]], NDArray[np.floating[Any]]]:
    """Reference: forward_pass kernel (Node 4).

    Returns: (hidden_activations, hidden_mask)
        hidden_activations: ReLU(input @ weights^T + biases) * sample_mask
        hidden_mask: 1.0 where pre-activation > 0, else 0.0
    """
    # weights_shared is (hidden_dim, input_dim) stored SIMD-major
    # but for reference we use standard (hidden, input)
    pre_act = input_data @ weights_shared.T + biases_shared  # (batch, hidden)
    hidden_mask = (pre_act > 0.0).astype(input_data.dtype)
    hidden_act = pre_act * hidden_mask
    # Apply sample mask: zero out masked samples
    mask_2d = sample_mask.reshape(-1, 1)
    hidden_act = hidden_act * mask_2d
    return hidden_act, hidden_mask


def ref_render_logits_chunk(
    hidden_act: NDArray[np.floating[Any]],
    module_weights: NDArray[np.floating[Any]],
    module_biases: NDArray[np.floating[Any]],
    temperature: float,
) -> NDArray[np.floating[Any]]:
    """Reference: render_logits_chunk kernel (Node 5).

    Returns: logits = (hidden @ W_m^T + b_m) / temperature
    """
    logits = hidden_act @ module_weights.T + module_biases
    return logits / temperature


def _softmax(logits: NDArray[np.floating[Any]]) -> NDArray[np.floating[Any]]:
    """Numerically stable softmax along last axis."""
    shifted = logits - np.max(logits, axis=-1, keepdims=True)
    exp_vals = np.exp(shifted)
    return exp_vals / np.sum(exp_vals, axis=-1, keepdims=True)


def _sigmoid(logits: NDArray[np.floating[Any]]) -> NDArray[np.floating[Any]]:
    """Numerically stable sigmoid."""
    return np.where(
        logits >= 0,
        1.0 / (1.0 + np.exp(-logits)),
        np.exp(logits) / (1.0 + np.exp(logits)),
    )


def ref_compute_probs_loss_cce(
    logits: NDArray[np.floating[Any]],
    targets: NDArray[np.floating[Any]],
    sample_mask: NDArray[np.floating[Any]],
    epsilon: float = 1e-7,
) -> tuple[NDArray[np.floating[Any]], float]:
    """Reference: compute_probs_loss_cce_chunk (Node 6).

    Returns: (probs, loss)
        probs: softmax(logits)
        loss: mean cross-entropy over active samples
    """
    probs = _softmax(logits)
    # Cross-entropy: -sum(target * log(prob)) per sample
    log_probs = np.log(probs + epsilon)
    per_sample_loss = -np.sum(targets * log_probs, axis=-1)
    active_count = np.sum(sample_mask)
    if active_count > 0:
        loss = np.sum(per_sample_loss * sample_mask) / active_count
    else:
        loss = 0.0
    return probs, float(loss)


def ref_compute_probs_loss_bce(
    logits: NDArray[np.floating[Any]],
    targets: NDArray[np.floating[Any]],
    sample_mask: NDArray[np.floating[Any]],
    epsilon: float = 1e-7,
) -> tuple[NDArray[np.floating[Any]], float]:
    """Reference: compute_probs_loss_bce_chunk (Node 7).

    Returns: (probs, loss)
        probs: sigmoid(logits)
        loss: mean binary cross-entropy over active samples
    """
    probs = _sigmoid(logits)
    # BCE: -[target * log(prob) + (1-target) * log(1-prob)] per class per sample
    log_p = np.log(probs + epsilon)
    log_1mp = np.log(1.0 - probs + epsilon)
    per_sample_loss = -np.sum(targets * log_p + (1.0 - targets) * log_1mp, axis=-1)
    active_count = np.sum(sample_mask)
    if active_count > 0:
        loss = np.sum(per_sample_loss * sample_mask) / active_count
    else:
        loss = 0.0
    return probs, float(loss)
