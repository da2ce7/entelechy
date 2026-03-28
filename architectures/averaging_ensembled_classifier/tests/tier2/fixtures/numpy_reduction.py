# tests/tier2/fixtures/numpy_reduction.py
"""Numpy reference implementations for reduction tree and aggregation kernels."""
from __future__ import annotations

from typing import Any

import numpy as np
from numpy.typing import NDArray


def ref_reduction_tree_sum(
    partials: list[NDArray[np.floating[Any]]],
    fan_in_K: int,
) -> NDArray[np.floating[Any]]:
    """Reference: multi-stage sum reduction tree.

    Sums N partials using a K-way tree structure. The result should match
    regardless of tree structure, since floating-point addition is the only
    operation and we use the sequential sum as ground truth.

    Returns: single reduced result array
    """
    return sum(partials[1:], partials[0].copy())


def ref_reduction_tree_sum_and_clip(
    partials: list[NDArray[np.floating[Any]]],
    fan_in_K: int,
    threshold_schedule: list[float | None],
    epsilon: float = 1e-7,
) -> NDArray[np.floating[Any]]:
    """Reference: multi-stage sum-and-clip reduction tree.

    Simulates the staged reduction with per-stage clipping.

    Returns: single reduced and clipped result array
    """
    current = list(partials)
    stage = 0
    while len(current) > 1:
        next_level: list[NDArray[np.floating[Any]]] = []
        for i in range(0, len(current), fan_in_K):
            group = current[i : i + fan_in_K]
            summed = sum(group[1:], group[0].copy())
            next_level.append(summed)

        # Apply clip if threshold is defined for this stage
        if stage < len(threshold_schedule) and threshold_schedule[stage] is not None:
            threshold = threshold_schedule[stage]
            clipped: list[NDArray[np.floating[Any]]] = []
            for arr in next_level:
                norm = np.sqrt(np.sum(arr * arr) + epsilon)
                if norm > threshold:
                    arr = arr * (threshold / norm)
                clipped.append(arr)
            next_level = clipped

        current = next_level
        stage += 1

    return current[0]


def ref_stabilize_reduce_grad_h(
    grad_h_partials: list[NDArray[np.floating[Any]]],
    clip_threshold: float | None = None,
    epsilon: float = 1e-7,
) -> NDArray[np.floating[Any]]:
    """Reference: stabilize_reduce_grad_h (Node 16).

    Single-kernel specialized Grad_H reduction with optional internal clipping.
    Per ADR-005, this is opaque in the plan — the test validates the
    mathematical result.

    Returns: reduced grad_h
    """
    result = sum(grad_h_partials[1:], grad_h_partials[0].copy())
    if clip_threshold is not None:
        norm = np.sqrt(np.sum(result * result) + epsilon)
        if norm > clip_threshold:
            result = result * (clip_threshold / norm)
    return result
