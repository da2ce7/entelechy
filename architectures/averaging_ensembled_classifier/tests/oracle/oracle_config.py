# tests/oracle/oracle_config.py
"""Shared configuration for all oracle variants.

Mirrors the engine's ModelSpec + StabilizationPolicy + OptimizerConfig
but expressed as a single, test-oriented configuration object.
"""
from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class TilingGeometry:
    """Mirrors _make_tiling() from plan_builder.py.

    chunk_size=16 matches the engine's hardcoded tiling constant.
    """
    num_modules: int
    output_classes: int
    chunk_size: int = 16

    @property
    def num_module_chunks(self) -> int:
        return max(1, (self.num_modules + self.chunk_size - 1) // self.chunk_size)

    @property
    def num_class_chunks(self) -> int:
        return max(1, (self.output_classes + self.chunk_size - 1) // self.chunk_size)

    @property
    def modules_per_chunk(self) -> int:
        return (self.num_modules + self.num_module_chunks - 1) // self.num_module_chunks

    @property
    def classes_per_chunk(self) -> int:
        return (self.output_classes + self.num_class_chunks - 1) // self.num_class_chunks

    @property
    def total_tiles(self) -> int:
        return self.num_module_chunks * self.num_class_chunks


@dataclass(frozen=True)
class OracleConfig:
    """Complete configuration for an oracle instance.

    All oracle variants receive the same OracleConfig to ensure they
    are solving the same problem.
    """
    input_dim: int
    hidden_dim: int
    output_classes: int
    num_modules: int
    mode: str  # "CCE" or "BCE"

    # Optimizer
    learning_rate: float = 0.001
    beta1: float = 0.9
    beta2: float = 0.999
    epsilon: float = 1e-7

    # Stabilization policy
    t_algorithmic: float = 1.0
    lambda_: float = 0.1
    compute_fp_format_max: float = 3.4028235e+38  # FP32 max

    # Temperature clamp
    temp_min: float = 0.01
    temp_max: float = 100.0

    @property
    def tiling(self) -> TilingGeometry:
        return TilingGeometry(
            num_modules=self.num_modules,
            output_classes=self.output_classes,
        )

    def leaf_safety_threshold(self) -> float:
        """Matches StabilizationPolicy.get_leaf_safety_threshold()."""
        return self.compute_fp_format_max * 0.9

    def node11_clip_threshold(self) -> float:
        """Pre-summation threshold for Node 11 (clip_partial_gradients).

        Matches plan_builder.py: get_leaf_safety_threshold() / num_class_chunks.
        """
        return self.leaf_safety_threshold() / self.tiling.num_class_chunks
