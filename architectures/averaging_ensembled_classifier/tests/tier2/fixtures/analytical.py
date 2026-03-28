# tests/tier2/fixtures/analytical.py
"""Analytical (closed-form) reference fixtures for Tier 2 tests.

These are the mathematical definitions — if the kernel disagrees,
the kernel is wrong (ADR-016).
"""
from __future__ import annotations

from typing import Any

import numpy as np
from numpy.typing import NDArray


def ref_hidden_mask(activations: NDArray[np.floating[Any]]) -> NDArray[np.floating[Any]]:
    """Reference: compute_hidden_mask — binary ReLU mask."""
    return (activations > 0.0).astype(activations.dtype)


def ref_clamp_temperatures(
    temps: NDArray[np.floating[Any]], t_min: float, t_max: float,
) -> NDArray[np.floating[Any]]:
    """Reference: clamp_temperatures — element-wise clamp."""
    return np.clip(temps, t_min, t_max)


def ref_normalize_gradients(
    summed_grads: NDArray[np.floating[Any]],
    effective_batch_size: float,
    epsilon: float = 1e-7,
) -> NDArray[np.floating[Any]]:
    """Reference: normalize_gradients — divide by effective batch size."""
    return summed_grads / (effective_batch_size + epsilon)


def ref_clip_l2_norm(
    grads: NDArray[np.floating[Any]],
    threshold: float,
    epsilon: float = 1e-7,
) -> NDArray[np.floating[Any]]:
    """Reference: component-wise L2-norm clipping.

    Used by clip_partial_gradients, clip_intermediate_grad,
    clip_shared_gradients.
    """
    norm = np.sqrt(np.sum(grads * grads) + epsilon)
    if norm > threshold:
        return grads * (threshold / norm)
    return grads.copy()


def ref_clip_partial_gradients(
    grads: NDArray[np.floating[Any]],
    threshold: float,
    epsilon: float = 1e-7,
) -> NDArray[np.floating[Any]]:
    """Reference: clip_partial_gradients — per-partial L2-norm clip."""
    return ref_clip_l2_norm(grads, threshold, epsilon)


def ref_clip_intermediate(
    partials: NDArray[np.floating[Any]],
    threshold: float,
    epsilon: float = 1e-7,
) -> NDArray[np.floating[Any]]:
    """Reference: clip_intermediate_grad — L2-norm clip on buffer."""
    return ref_clip_l2_norm(partials, threshold, epsilon)


def ref_clip_shared_gradients(
    grad_sw: NDArray[np.floating[Any]],
    grad_sb: NDArray[np.floating[Any]],
    threshold: float,
    epsilon: float = 1e-7,
) -> tuple[NDArray[np.floating[Any]], NDArray[np.floating[Any]]]:
    """Reference: clip_shared_gradients — joint L2-norm clip on concatenated grads."""
    combined = np.concatenate([grad_sw.ravel(), grad_sb.ravel()])
    norm = np.sqrt(np.sum(combined * combined) + epsilon)
    if norm > threshold:
        scale = threshold / norm
        return grad_sw * scale, grad_sb * scale
    return grad_sw.copy(), grad_sb.copy()
