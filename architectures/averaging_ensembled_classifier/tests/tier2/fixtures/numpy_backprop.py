# tests/tier2/fixtures/numpy_backprop.py
"""Numpy reference implementations for shared-layer backprop kernels (Nodes 17-19)."""
from __future__ import annotations

from typing import Any

import numpy as np
from numpy.typing import NDArray


def ref_backprop_shared_weights(
    grad_h: NDArray[np.floating[Any]],
    input_data: NDArray[np.floating[Any]],
    sample_mask: NDArray[np.floating[Any]],
) -> NDArray[np.floating[Any]]:
    """Reference: backprop_shared_weights_chunk (Node 17).

    Computes partial gradient of shared weights for one batch chunk.

    Returns: grad_weights_shared (hidden_dim, input_dim)
    """
    # Apply sample mask to grad_h
    masked_grad_h = grad_h * sample_mask.reshape(-1, 1)
    # grad_W = grad_h^T @ input  (hidden, input)
    return masked_grad_h.T @ input_data


def ref_backprop_shared_biases(
    grad_h: NDArray[np.floating[Any]],
    sample_mask: NDArray[np.floating[Any]],
) -> NDArray[np.floating[Any]]:
    """Reference: backprop_shared_biases_chunk (Node 18).

    Computes partial gradient of shared biases for one batch chunk.

    Returns: grad_biases_shared (hidden_dim,)
    """
    masked_grad_h = grad_h * sample_mask.reshape(-1, 1)
    return np.sum(masked_grad_h, axis=0)
