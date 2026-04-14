# src/shared/memory_layout.py
"""Padded dimension synthesis (CONCEPT.md §11, CONTRACT §3.3).

Resolves logical model dimensions into padded physical extents by
unifying alignment constraints across all precision roles.

The core problem
────────────────
A single dimension like ``hidden_count`` governs buffers in multiple
precision roles (storage, state, compute), each imposing independent
alignment requirements:

    Storage-role activations:   CACHE → hidden × sizeof(STORAGE) ≡ 0 (mod 128)
    State-role module weights:  CACHE → hidden × sizeof(STATE)   ≡ 0 (mod 128)
    State-role shared weights:  SIMD  → hidden ≡ 0 (mod SIMD_WIDTH)

Each requirement reduces to an element-count divisibility constraint.
The padded extent is the smallest value ≥ logical satisfying all
constraints simultaneously — computed as::

    alignment = lcm(all constraint multiples)
    padded    = ⌈logical / alignment⌉ × alignment

Example: ``hidden_count=200``, ``SIMD_WIDTH=8``,
``PrecisionConfig.fp8_e4m3()`` (1-byte storage, 4-byte state):

    CACHE @ storage (1B elem):  128 / gcd(128, 1) = 128-element multiple
    CACHE @ state   (4B elem):  128 / gcd(128, 4) =  32-element multiple
    SIMD  (width=8):                                   8-element multiple

    alignment = lcm(128, 32, 8) = 128
    padded_hidden = ⌈200 / 128⌉ × 128 = 256

Under ``PrecisionConfig.float32()`` (4-byte storage, 4-byte state):

    CACHE @ storage (4B elem):  32-element multiple
    CACHE @ state   (4B elem):  32-element multiple  (deduplicates)
    SIMD  (width=8):             8-element multiple

    alignment = lcm(32, 8) = 32
    padded_hidden = ⌈200 / 32⌉ × 32 = 224

The FP8 configuration requires wider padding (128 vs 32 elements)
because its 1-byte elements need more elements per cache line — a
direct expression of the Primacy of Memory Strategy's impact on
physical layout.

Organisation
────────────
§1  Pure arithmetic primitives — ``pad_to_multiple`` and LCM
§2  Alignment constraints — typed requirements with diagnostic provenance
§3  Dimension resolution — constraint unification via LCM
§4  Model dimensions — the four canonical padded dimension sets,
    with constraint specifications derived from ``kernels.cl.h``

Authoritative sources
─────────────────────
CONCEPT.md §11        — Padded dimension synthesis mandate
CONTRACT.md §3.3      — Padding Contract specification vocabulary
kernels.cl.h rev. 10  — Per-buffer padding declarations (source of
                         the §4 constraint sets)
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Sequence

from .precision_config import PrecisionConfig

__all__ = [
    "AlignmentConstraint",
    "ModelDimensions",
    "PaddedExtent",
    "cache_constraint",
    "pad_to_multiple",
    "resolve_extent",
    "resolve_model_dimensions",
    "simd_constraint",
]


# ═══════════════════════════════════════════════════════════════════
# §1  Pure Arithmetic Primitives
# ═══════════════════════════════════════════════════════════════════


def _ceildiv(a: int, b: int) -> int:
    """``⌈a / b⌉``.  *b* must be positive."""
    return (a + b - 1) // b


def pad_to_multiple(value: int, multiple: int) -> int:
    """Smallest integer ≥ *value* that is divisible by *multiple*.

    Returns *value* unchanged when ``multiple ≤ 1``.
    """
    if multiple <= 1:
        return value
    return _ceildiv(value, multiple) * multiple


def _lcm(a: int, b: int) -> int:
    """Least common multiple of two positive integers."""
    return a * b // math.gcd(a, b)


# ═══════════════════════════════════════════════════════════════════
# §2  Alignment Constraints
# ═══════════════════════════════════════════════════════════════════


@dataclass(frozen=True)
class AlignmentConstraint:
    """A resolved element-count divisibility requirement.

    Expresses: ``padded_extent ≡ 0 (mod multiple)``.

    This is the irreducible output of converting a contractual padding
    specification (CACHE with a dtype, or SIMD with a lane count) into
    a concrete requirement on a dimension's element count.

    Parameters
    ----------
    source:
        Human-readable provenance for diagnostics and plan inspection
        (e.g. ``"CACHE@storage(2B elem, 128B line)"``).
    multiple:
        The element-count divisor.  The padded extent must be an
        integer multiple of this value.
    """

    source: str
    multiple: int

    def __post_init__(self) -> None:
        if self.multiple < 1:
            raise ValueError(
                f"AlignmentConstraint multiple must be ≥ 1, "
                f"got {self.multiple} (source: {self.source!r})"
            )


def cache_constraint(
    role: str,
    element_size: int,
    cache_line_bytes: int = 128,
) -> AlignmentConstraint:
    """CACHE padding — row byte-width must be a cache-line multiple.

    Converts the byte-alignment requirement from CONTRACT §3.3.1
    to an element-count constraint::

        padded × element_size ≡ 0 (mod cache_line_bytes)
        ⟹  padded ≡ 0 (mod cache_line_bytes / gcd(cache_line_bytes, element_size))

    For the architecture's standard element sizes (1, 2, 4, 8 bytes)
    and 128-byte cache line::

        FP8  (1B): 128 / gcd(128,1) = 128-element multiple
        FP16 (2B): 128 / gcd(128,2) =  64-element multiple
        FP32 (4B): 128 / gcd(128,4) =  32-element multiple
        FP64 (8B): 128 / gcd(128,8) =  16-element multiple

    Parameters
    ----------
    role:
        Precision role name (``"storage"``, ``"compute"``,
        ``"state"``).  For the diagnostic ``source`` string only.
    element_size:
        ``PrecisionConfig.<role>_dtype.itemsize``.
    cache_line_bytes:
        Hardware cache line size.  Defaults to 128 — the
        architecture's standard alignment target.
    """
    g = math.gcd(cache_line_bytes, element_size)
    multiple = cache_line_bytes // g
    return AlignmentConstraint(
        source=f"CACHE@{role}({element_size}B elem, {cache_line_bytes}B line)",
        multiple=multiple,
    )


def simd_constraint(simd_width: int) -> AlignmentConstraint:
    """SIMD padding — element count must be a SIMD-lane multiple.

    Role-independent: the SIMD width is a hardware constant,
    invariant across precision roles.

    Parameters
    ----------
    simd_width:
        ``HardwareProfile.simd_width``.
    """
    return AlignmentConstraint(
        source=f"SIMD(width={simd_width})",
        multiple=simd_width,
    )


# ═══════════════════════════════════════════════════════════════════
# §3  Dimension Resolution
# ═══════════════════════════════════════════════════════════════════


@dataclass(frozen=True)
class PaddedExtent:
    """A resolved dimension carrying both logical and physical extents.

    Produced by :func:`resolve_extent`.  Carries the numeric result
    and the constraint provenance for plan inspection and diagnostics.

    Parameters
    ----------
    name:
        Canonical dimension name (e.g. ``"hidden"``).
    logical:
        Original unpadded element count.
    padded:
        Physical element count satisfying all constraints — the
        value used for buffer allocation and kernel scalar injection.
    alignment:
        LCM of all constraint multiples.  ``padded`` is always a
        multiple of ``alignment``.  1 when no constraints apply.
    constraints:
        The constraints that produced this result (frozen provenance).
    """

    name: str
    logical: int
    padded: int
    alignment: int
    constraints: tuple[AlignmentConstraint, ...]

    @property
    def padding_elements(self) -> int:
        """Number of padding elements: ``padded - logical``."""
        return self.padded - self.logical

    @property
    def is_padded(self) -> bool:
        """Whether any padding was applied."""
        return self.padded != self.logical


def resolve_extent(
    name: str,
    logical: int,
    constraints: Sequence[AlignmentConstraint],
) -> PaddedExtent:
    """Resolve a logical dimension to its padded extent.

    Computes ``alignment = lcm(c.multiple for c in constraints)``
    then ``padded = ⌈logical / alignment⌉ × alignment``.

    When *constraints* is empty, ``alignment = 1`` and
    ``padded = logical`` (the identity case).

    Parameters
    ----------
    name:
        Canonical dimension name for the result.
    logical:
        Unpadded element count (≥ 0).
    constraints:
        Alignment requirements to unify.

    Raises
    ------
    ValueError
        If *logical* is negative.
    """
    if logical < 0:
        raise ValueError(
            f"logical extent must be ≥ 0 for dimension {name!r}, "
            f"got {logical}"
        )

    frozen = tuple(constraints)
    alignment = 1
    for c in frozen:
        alignment = _lcm(alignment, c.multiple)

    return PaddedExtent(
        name=name,
        logical=logical,
        padded=pad_to_multiple(logical, alignment),
        alignment=alignment,
        constraints=frozen,
    )


# ═══════════════════════════════════════════════════════════════════
# §4  Model Dimensions
# ═══════════════════════════════════════════════════════════════════


@dataclass(frozen=True)
class ModelDimensions:
    """Resolved padded extents for all four model dimensions.

    Produced by :func:`resolve_model_dimensions`.  Provides typed
    attribute access for self-documenting code and name-based lookup
    for generic plan-builder iteration.

    Parameters
    ----------
    input:
        ``padded_input_count`` — shared-layer feature dimension.
    hidden:
        ``padded_hidden_count`` — shared-layer activation dimension.
    output_classes:
        ``padded_total_output_class_count`` — classifier output.
    modules:
        ``padded_total_modules_count`` — classifier module count.
    """

    input: PaddedExtent
    hidden: PaddedExtent
    output_classes: PaddedExtent
    modules: PaddedExtent

    def __getitem__(self, name: str) -> PaddedExtent:
        """Look up by canonical dimension name.

        Raises ``KeyError`` for unrecognised names.
        """
        for ext in self._all():
            if ext.name == name:
                return ext
        raise KeyError(
            f"Unknown dimension {name!r}; "
            f"known: {[e.name for e in self._all()]}"
        )

    def __iter__(self):
        """Iterate over all four extents in canonical order."""
        return iter(self._all())

    def _all(self) -> tuple[PaddedExtent, ...]:
        return (self.input, self.hidden, self.output_classes, self.modules)


def resolve_model_dimensions(
    *,
    input_dim: int,
    hidden_dim: int,
    output_classes: int,
    num_modules: int,
    precision: PrecisionConfig,
    simd_width: int,
    cache_line_bytes: int = 128,
) -> ModelDimensions:
    """Resolve all model dimensions to their padded extents.

    Constraint sets are derived from the kernel contracts
    (``kernels.cl.h`` rev. 10).  Each set is annotated with the
    contract provisions — buffer names and node numbers — that
    establish its requirements.

    When a future kernel revision adds or changes a dimension's
    padding requirements, update the constraint tuples below and
    add a comment citing the new provision.

    Parameters
    ----------
    input_dim:
        Logical input feature count.
    hidden_dim:
        Logical hidden-layer unit count.
    output_classes:
        Logical output class count (per module).
    num_modules:
        Number of classifier modules (heads).
    precision:
        Active three-role precision configuration.
    simd_width:
        Hardware SIMD lane count (``HardwareProfile.simd_width``).
    cache_line_bytes:
        Hardware cache line size in bytes.
    """
    s = precision.storage_dtype.itemsize
    st = precision.state_dtype.itemsize
    cl = cache_line_bytes

    # ── padded_input_count ────────────────────────────────────────
    #
    # CACHE @ storage:
    #   Node 4  src_buffer_GLOBAL_input
    #   Node 17 src_buffer_GLOBAL_input
    #   Node 17 dest_buffer_GLOBAL_partial_grad_weights_shared_simd_major
    #
    # CACHE @ state:
    #   Node 4  src_buffer_GLOBAL_CONST_weights_shared_simd_major
    #           (input dim = dim[1])
    #
    input_ext = resolve_extent("input", input_dim, (
        cache_constraint("storage", s, cl),
        cache_constraint("state", st, cl),
    ))

    # ── padded_hidden_count ───────────────────────────────────────
    #
    # CACHE @ storage:
    #   Node 4  dest_buffer_GLOBAL_hidden_activations
    #   Node 4  dest_buffer_GLOBAL_hidden_mask
    #   Node 5  src_buffer_GLOBAL_hidden_activations
    #   Node 8  dest_buffer_GLOBAL_partial_grad_weights_module (dim[3])
    #   Node 9  dest_buffer_GLOBAL_partial_grad_hidden_activations_aos
    #   Node 11 src/dest clipped grad buffers
    #   Node 13 src/dest permuted SoA buffers
    #   Node 18 dest_buffer_GLOBAL_partial_grad_biases_shared
    #
    # CACHE @ state:
    #   Node 5  src_buffer_GLOBAL_CONST_weights_module (dim[1])
    #
    # SIMD (role-independent):
    #   Node 4  src_buffer_GLOBAL_CONST_weights_shared_simd_major
    #           (dim[0] = padded_hidden/SIMD_WIDTH requires divisibility)
    #   Node 4  src_buffer_GLOBAL_CONST_biases_shared
    #
    hidden_ext = resolve_extent("hidden", hidden_dim, (
        cache_constraint("storage", s, cl),
        cache_constraint("state", st, cl),
        simd_constraint(simd_width),
    ))

    # ── padded_total_output_class_count ───────────────────────────
    #
    # SIMD (role-independent):
    #   Node 5  src_buffer_GLOBAL_CONST_weights_module (dim[2])
    #   Node 5  src_buffer_GLOBAL_CONST_biases_module
    #   Node 8  dest_buffer_GLOBAL_partial_grad_weights_module (dim[4])
    #   Node 8  dest_buffer_GLOBAL_partial_grad_biases_module (dim[3])
    #   Node 11 src/dest clipped grad weight/bias buffers
    #
    # CACHE @ storage:
    #   Node 5  dest_buffer_GLOBAL_logits
    #   Node 7  src_buffer_GLOBAL_targets (BCE multi-hot)
    #
    classes_ext = resolve_extent("output_classes", output_classes, (
        simd_constraint(simd_width),
        cache_constraint("storage", s, cl),
    ))

    # ── padded_total_modules_count ────────────────────────────────
    #
    # CACHE @ storage:
    #   Node 13 dest_buffer_GLOBAL_clipped_grad_hidden_activations_
    #           permuted_soa (dim[1])
    #   Node 16 src_buffer_GLOBAL_clipped_grad_hidden_activations_
    #           permuted_soa (dim[1])
    #
    modules_ext = resolve_extent("modules", num_modules, (
        cache_constraint("storage", s, cl),
    ))

    return ModelDimensions(
        input=input_ext,
        hidden=hidden_ext,
        output_classes=classes_ext,
        modules=modules_ext,
    )
