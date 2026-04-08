# tests/tier2/fixtures/numpy_gradients.py
"""Numpy reference implementations for gradient computation kernels (Nodes 8-11, 13)."""
from __future__ import annotations

from typing import Any

import numpy as np
from numpy.typing import NDArray


def ref_module_param_grads_cce(
    probs: NDArray[np.floating[Any]],
    targets: NDArray[np.floating[Any]],
    hidden_act: NDArray[np.floating[Any]],
    sample_mask: NDArray[np.floating[Any]],
) -> tuple[NDArray[np.floating[Any]], NDArray[np.floating[Any]]]:
    """Reference: calculate_module_param_grads_cce (Node 8, Strategy B CCE).

    Returns: (grad_weights, grad_biases)
    """
    _batch_size = probs.shape[0]
    # delta = probs - targets for CCE (softmax output)
    delta = probs - targets  # (batch, classes)
    # Apply sample mask
    masked_delta = delta * sample_mask.reshape(-1, 1)  # (batch, classes)
    # grad_weights = delta^T @ hidden_act  (classes, hidden)
    grad_weights = masked_delta.T @ hidden_act
    # grad_biases = sum(delta, axis=0)  (classes,)
    grad_biases = np.sum(masked_delta, axis=0)
    return grad_weights, grad_biases


def ref_module_param_grads_bce(
    probs: NDArray[np.floating[Any]],
    targets: NDArray[np.floating[Any]],
    hidden_act: NDArray[np.floating[Any]],
    sample_mask: NDArray[np.floating[Any]],
) -> tuple[NDArray[np.floating[Any]], NDArray[np.floating[Any]]]:
    """Reference: calculate_module_param_grads_bce (Node 8, Strategy B BCE).

    Returns: (grad_weights, grad_biases)
    """
    # delta = probs - targets for BCE (sigmoid output)
    delta = probs - targets  # (batch, classes)
    masked_delta = delta * sample_mask.reshape(-1, 1)
    grad_weights = masked_delta.T @ hidden_act
    grad_biases = np.sum(masked_delta, axis=0)
    return grad_weights, grad_biases


def ref_backprop_error_to_hidden(
    probs: NDArray[np.floating[Any]],
    targets: NDArray[np.floating[Any]],
    module_weights: NDArray[np.floating[Any]],
    hidden_mask: NDArray[np.floating[Any]],
    sample_mask: NDArray[np.floating[Any]],
) -> NDArray[np.floating[Any]]:
    """Reference: backprop_error_to_hidden (Node 9, Strategy A).

    Returns: grad_hidden (batch, hidden)
    """
    delta = probs - targets  # (batch, classes)
    masked_delta = delta * sample_mask.reshape(-1, 1)
    # Backprop through module layer: delta @ W_m → (batch, hidden)
    grad_h = masked_delta @ module_weights
    # Apply ReLU mask
    grad_h = grad_h * hidden_mask
    return grad_h


def ref_temp_gradients_cce(
    probs: NDArray[np.floating[Any]],
    targets: NDArray[np.floating[Any]],
    logits_unscaled: NDArray[np.floating[Any]],
    temperature: float,
    sample_mask: NDArray[np.floating[Any]],
) -> float:
    """Reference: calculate_chunk_temp_gradients (Node 10, CCE).

    Returns: scalar gradient w.r.t. temperature
    """
    delta = probs - targets  # (batch, classes)
    # dL/dT = -1/T^2 * sum(delta * z_unscaled) for CCE
    per_sample = np.sum(delta * logits_unscaled, axis=-1)
    masked = per_sample * sample_mask
    return float(-np.sum(masked) / (temperature * temperature))


def ref_temp_gradients_bce(
    probs: NDArray[np.floating[Any]],
    targets: NDArray[np.floating[Any]],
    logits_unscaled: NDArray[np.floating[Any]],
    temperature: float,
    sample_mask: NDArray[np.floating[Any]],
) -> float:
    """Reference: calculate_chunk_temp_gradients (Node 10, BCE).

    Returns: scalar gradient w.r.t. temperature
    """
    delta = probs - targets
    per_sample = np.sum(delta * logits_unscaled, axis=-1)
    masked = per_sample * sample_mask
    return float(-np.sum(masked) / (temperature * temperature))


def ref_gather_and_permute_grad_h(
    grad_h_tiles: list[NDArray[np.floating[Any]]],
) -> NDArray[np.floating[Any]]:
    """Reference: gather_and_permute_grad_h (Node 13).

    Concatenates scattered Grad_H tile partials into a contiguous
    SoA (Structure-of-Arrays) layout.

    Returns: permuted_grad_h
    """
    return np.concatenate(grad_h_tiles, axis=0)
