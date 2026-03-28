# tests/tier2/fixtures/numpy_update.py
"""Numpy reference implementations for update-phase kernels (Nodes 21, 24, 25)."""
from __future__ import annotations

from typing import Any

import numpy as np
from numpy.typing import NDArray


def ref_adam_update(
    params: NDArray[np.floating[Any]],
    grads: NDArray[np.floating[Any]],
    m1: NDArray[np.floating[Any]],
    m2: NDArray[np.floating[Any]],
    learning_rate: float,
    beta1: float,
    beta2: float,
    epsilon: float,
    beta1_pow_t: float,
    beta2_pow_t: float,
) -> tuple[NDArray[np.floating[Any]], NDArray[np.floating[Any]], NDArray[np.floating[Any]]]:
    """Reference: adam_update kernel (Node 24).

    Computes one Adam optimizer step.
    beta1_pow_t and beta2_pow_t are pre-computed by the host in float64.

    Returns: (updated_params, updated_m1, updated_m2)
    """
    # Update biased first moment estimate
    m1_new = beta1 * m1 + (1.0 - beta1) * grads
    # Update biased second moment estimate
    m2_new = beta2 * m2 + (1.0 - beta2) * (grads * grads)
    # Bias-corrected estimates
    m1_hat = m1_new / (1.0 - beta1_pow_t)
    m2_hat = m2_new / (1.0 - beta2_pow_t)
    # Update parameters
    params_new = params - learning_rate * m1_hat / (np.sqrt(m2_hat) + epsilon)
    return params_new.astype(params.dtype), m1_new.astype(m1.dtype), m2_new.astype(m2.dtype)
