# src/shared/model_spec.py
"""Backend-neutral model specification (ADR-008, ADR-020, ADR-022 §2).

Describes a classifier model's logical dimensions and precision
configuration.  Padded (physical) dimensions are derived by delegating
to ``memory_layout.resolve_model_dimensions()`` — the single source of
truth for LCM-based dimension synthesis (CONCEPT.md §11).

Design decisions
────────────────
**Padding delegation.**  ``ModelSpec`` does not compute padded
dimensions itself.  It delegates to ``memory_layout.py``, which
applies the Padded Dimension Synthesis mandate: each dimension's padded
extent is the LCM of all alignment constraints across every precision
role and buffer that shares that dimension.  The ``padded_*`` properties
are thin accessors into the resolved ``ModelDimensions``.

**Hardware parameters are bundled.**  ``simd_width`` and
``cache_line_bytes`` are hardware-profile concerns, not intrinsic
model properties.  They are bundled here because padded dimensions
depend on them, and ``ModelSpec`` is the natural integration point
where logical model + hardware constraints produce resolved geometry.
A future revision may separate a pure ``LogicalModelSpec`` from the
hardware-resolved layout.

**Construction validation.**  ``__post_init__`` enforces that all
dimensions are strictly positive and that the SIMD-major weight layout
structural invariant (``padded_hidden_dim % simd_width == 0``) holds.

**Factory classmethods.**  Named constructors for common precision
configurations mirror ``PrecisionConfig``'s factory vocabulary.
The deprecated ``float16()`` factory is retained with a warning;
new code should use ``mixed_f16_f32()``.

Authoritative sources
─────────────────────
ADR-008           — ModelSpec as backend-neutral model description
ADR-020, ADR-022  — Three-role precision model
CONCEPT.md §11    — Padded dimension synthesis mandate
memory_layout.py  — Constraint-driven LCM padding implementation
"""

from __future__ import annotations

import warnings
from dataclasses import dataclass, field
from typing import Any

from .memory_layout import ModelDimensions, resolve_model_dimensions
from .precision_config import PrecisionConfig

__all__ = [
    "ModelSpec",
]


@dataclass(frozen=True)
class ModelSpec:
    """Backend-neutral model configuration.

    Combines a logical model description (dimensions, module count,
    precision) with hardware layout parameters (SIMD width, cache line
    size) to produce resolved padded dimensions via
    ``memory_layout.resolve_model_dimensions()``.

    Parameters
    ----------
    precision:
        Three-role precision configuration governing storage, compute,
        and state dtypes.
    input_dim:
        Logical input feature count (≥ 1).
    hidden_dim:
        Logical hidden-layer unit count (≥ 1).
    output_classes:
        Logical output class count per module (≥ 1).
    num_modules:
        Number of classifier modules / heads (≥ 1).
    simd_width:
        Hardware SIMD lane count (≥ 1).  Governs SIMD-alignment
        padding on the hidden and class dimensions, and the
        shared-weight SoA reshape divisor.
    cache_line_bytes:
        Hardware cache line size in bytes (≥ 1).  Governs
        CACHE-alignment padding on all dimensions.

    Raises
    ------
    ValueError
        If any dimension or hardware parameter is < 1, or if the
        resolved ``padded_hidden_dim`` violates the SIMD-major
        divisibility invariant.
    """

    # ── Primary fields (user-supplied) ────────────────────────────

    precision: PrecisionConfig
    input_dim: int
    hidden_dim: int
    output_classes: int
    num_modules: int
    simd_width: int
    cache_line_bytes: int

    # ── Derived (resolved from primaries, never user-supplied) ────

    _dims: ModelDimensions = field(
        init=False, repr=False, compare=False, hash=False,
    )

    def __post_init__(self) -> None:
        # ── Validate primary fields ───────────────────────────────
        for name, value in (
            ("input_dim", self.input_dim),
            ("hidden_dim", self.hidden_dim),
            ("output_classes", self.output_classes),
            ("num_modules", self.num_modules),
            ("simd_width", self.simd_width),
            ("cache_line_bytes", self.cache_line_bytes),
        ):
            if value < 1:
                raise ValueError(f"{name} must be ≥ 1, got {value}")

        # ── Resolve padded dimensions (single source of truth) ────
        dims = resolve_model_dimensions(
            input_dim=self.input_dim,
            hidden_dim=self.hidden_dim,
            output_classes=self.output_classes,
            num_modules=self.num_modules,
            precision=self.precision,
            simd_width=self.simd_width,
            cache_line_bytes=self.cache_line_bytes,
        )
        object.__setattr__(self, "_dims", dims)

        # ── SIMD-major structural invariant ───────────────────────
        #
        # Node 4's weight layout reshapes to
        #   (padded_hidden/SIMD_WIDTH, padded_input, SIMD_WIDTH)
        # and parameter_space.py's shared_weights geometry divides
        # by simd_width.  memory_layout's simd_constraint on the
        # hidden dimension guarantees this; we verify defensively.
        if dims.hidden.padded % self.simd_width != 0:
            raise ValueError(
                f"padded_hidden_dim ({dims.hidden.padded}) must be "
                f"divisible by simd_width ({self.simd_width}) for "
                f"SIMD-major layout — this indicates a bug in "
                f"memory_layout constraint specification"
            )

    # ── Resolved dimension accessors ──────────────────────────────
    #
    # Thin projections into the resolved ModelDimensions.  Preserve
    # the existing API consumed by parameter_space.py, plan builders,
    # and other downstream modules.

    @property
    def dimensions(self) -> ModelDimensions:
        """Full resolved dimensions with constraint provenance.

        Provides alignment metadata, padding element counts, and
        diagnostic inspection for plan-builder tooling.
        """
        return self._dims

    @property
    def padded_input_dim(self) -> int:
        """``padded_input_count`` — CACHE-aligned input feature extent.

        Constraints: CACHE@storage (Node 4 input, Node 17 input/output),
        CACHE@state (Node 4 SIMD-major weight dim[1]).
        """
        return self._dims.input.padded

    @property
    def padded_hidden_dim(self) -> int:
        """``padded_hidden_count`` — CACHE+SIMD-aligned hidden extent.

        Constraints: CACHE@storage (Nodes 4, 5, 8, 9, 11, 13, 18
        activation/gradient buffers), CACHE@state (Node 5 module
        weight dim[1]), SIMD (Node 4 shared weight/bias layout).
        """
        return self._dims.hidden.padded

    @property
    def padded_class_dim(self) -> int:
        """``padded_total_output_class_count`` — SIMD+CACHE-aligned class extent.

        Constraints: SIMD (Nodes 5, 8, 11 weight/gradient buffers),
        CACHE@storage (Node 5 logits, Node 7 BCE targets).
        """
        return self._dims.output_classes.padded

    @property
    def padded_module_dim(self) -> int:
        """``padded_total_modules_count`` — CACHE-aligned module extent.

        Constraints: CACHE@storage (Node 13/16 permuted SoA buffer
        dim[1]).
        """
        return self._dims.modules.padded

    # ══════════════════════════════════════════════════════════════
    # Factory classmethods
    #
    # Each factory pre-selects a PrecisionConfig; all other fields
    # (input_dim, hidden_dim, etc.) pass through via **kwargs.
    # Runtime validation in __post_init__ catches invalid values.
    # ══════════════════════════════════════════════════════════════

    # ── Standard IEEE configurations ──────────────────────────────

    @classmethod
    def float32(cls, **kwargs: Any) -> ModelSpec:
        """FP32 storage, FP32 compute, FP32 state.  Default reference."""
        return cls(precision=PrecisionConfig.float32(), **kwargs)

    @classmethod
    def float64(cls, **kwargs: Any) -> ModelSpec:
        """FP64 storage, FP64 compute, FP64 state.  Validation reference."""
        return cls(precision=PrecisionConfig.float64(), **kwargs)

    # ── Mixed IEEE configurations ─────────────────────────────────

    @classmethod
    def mixed_f16_f32(cls, **kwargs: Any) -> ModelSpec:
        """FP16 storage, FP32 compute, FP32 state.  Production mixed-precision."""
        return cls(precision=PrecisionConfig.mixed_f16_f32(), **kwargs)

    @classmethod
    def mixed_f32_f64(cls, **kwargs: Any) -> ModelSpec:
        """FP32 storage, FP64 compute, FP64 state.  High-fidelity scientific."""
        return cls(precision=PrecisionConfig.mixed_f32_f64(), **kwargs)

    @classmethod
    def mixed_f32_f64_state(cls, **kwargs: Any) -> ModelSpec:
        """FP32 storage, FP32 compute, FP64 state.  Extended-stability training."""
        return cls(precision=PrecisionConfig.mixed_f32_f64_state(), **kwargs)

    @classmethod
    def mixed_f16_f64_state(cls, **kwargs: Any) -> ModelSpec:
        """FP16 storage, FP32 compute, FP64 state.  Bandwidth + extended stability."""
        return cls(precision=PrecisionConfig.mixed_f16_f64_state(), **kwargs)

    # ── FP8 configurations ────────────────────────────────────────

    @classmethod
    def fp8_e4m3(cls, **kwargs: Any) -> ModelSpec:
        """E4M3 storage, FP32 compute, FP32 state.  4× compression vs FP32.

        Requires ``ml_dtypes``.
        """
        return cls(precision=PrecisionConfig.fp8_e4m3(), **kwargs)

    @classmethod
    def fp8_e5m2(cls, **kwargs: Any) -> ModelSpec:
        """E5M2 storage, FP32 compute, FP32 state.  Wider dynamic range.

        Requires ``ml_dtypes``.
        """
        return cls(precision=PrecisionConfig.fp8_e5m2(), **kwargs)

    @classmethod
    def fp8_e4m3_f64(cls, **kwargs: Any) -> ModelSpec:
        """E4M3 storage, FP64 compute, FP64 state.  Full FP64 arithmetic fidelity.

        Requires ``ml_dtypes``.
        """
        return cls(precision=PrecisionConfig.fp8_e4m3_f64(), **kwargs)

    @classmethod
    def fp8_e5m2_f64(cls, **kwargs: Any) -> ModelSpec:
        """E5M2 storage, FP64 compute, FP64 state.

        Requires ``ml_dtypes``.
        """
        return cls(precision=PrecisionConfig.fp8_e5m2_f64(), **kwargs)

    # ── Deprecated ────────────────────────────────────────────────

    @classmethod
    def float16(cls, **kwargs: Any) -> ModelSpec:
        """FP16 storage, FP32 compute, FP32 state.

        .. deprecated::
            Use ``ModelSpec.mixed_f16_f32()`` instead.  The name
            ``float16`` misleadingly suggests all-FP16 precision.
        """
        warnings.warn(
            "ModelSpec.float16() is deprecated and will be removed. "
            "Use ModelSpec.mixed_f16_f32() instead "
            "(FP16 storage, FP32 compute, FP32 state).",
            DeprecationWarning,
            stacklevel=2,
        )
        return cls(precision=PrecisionConfig.mixed_f16_f32(), **kwargs)
