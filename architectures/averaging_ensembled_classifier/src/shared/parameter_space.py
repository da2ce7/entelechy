# src/shared/parameter_space.py
"""Declarative registry of the model's five learnable parameter groups.

This module enumerates the system's learnable parameter groups — their
structural identity, gradient lifecycle, optimizer associations, and
shape geometry — so that the plan builder can drive buffer allocation,
reduction-tree construction, normalisation dispatch, and optimizer
updates through systematic iteration rather than per-group boilerplate.

Every type is a frozen dataclass carrying no device references, no
mutable state, and no active computation beyond host-side arithmetic.

Parameter groups
────────────────
1. **shared_weights**  — W_shared, SIMD-major (SoA) layout, streaming path
2. **shared_biases**   — b_shared, streaming path
3. **module_weights**  — W_module, per-module, module-tiled path
4. **module_biases**   — b_module, per-module, module-tiled path
5. **temperatures**    — τ, per-module, module-tiled path, post-update clamp

Gradient lifecycle paths
────────────────────────
**module_tiled** (Nodes 8/10 → 11 → ModuleChunkGather → Reduction → 21 → 24):
    Gradients produced per-tile by the module-layer backward pass,
    jointly clipped (Node 11), reduced per-module-chunk via
    ``ModuleChunkGather`` (ADR-030).

**streaming** (Nodes 17/18 → 19 → collection → StridedGather → Reduction → 21 → 24):
    Gradients produced per-streaming-chunk by the shared-layer
    backward pass, clipped (Node 19), reduced via ``StridedGather``
    over streaming chunks.

The hidden-gradient path (Nodes 9 → 11 → 13 → 16) is *not* a parameter
group.  It is an intermediate error signal consumed by the streaming
path; it carries no optimizer state and never enters ``adam_update``.

Plan-builder usage sketch
─────────────────────────
::

    geos = resolve_all(model_spec, hardware.simd_width)

    for group in MODULE_GROUPS:
        geo = geos[group.name]
        for mc in range(num_module_chunks):
            epp = modules_per_chunk * geo.elements_per_module
            gather = ModuleChunkGather(tiling, mc, epp)
            ps = ParameterSlice.from_chunk(tiling.modules, mc, geo.elements_per_module)
            # build: ReductionTreeNode  → group.reduce_node_id(mc)
            # build: normalize_gradients → group.normalize_node_id(mc)
            # build: adam_update         → group.adam_node_id(mc)
            # build: post-update         → group.post_update_node_id(mc)

    for group in SHARED_GROUPS:
        geo = geos[group.name]
        gather = StridedGather(num_streaming_chunks, geo.total_flat_elements)
        ps = ParameterSlice.full(geo.optimizer_state_elements)
        # build: ReductionTreeNode  → group.reduce_node_id()
        # build: normalize_gradients → group.normalize_node_id()
        # build: adam_update         → group.adam_node_id()

Organisation
────────────
§1  ``ParameterGroup``   — structural identity and naming conventions
§2  ``ParameterGeometry`` — concrete shapes resolved from a ModelSpec
§3  Registry constants   — the five canonical groups
§4  Resolution functions — geometry computation
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Literal

from .model_spec import ModelSpec

__all__ = [
    "ALL_GROUPS",
    "MODULE_BIASES",
    "MODULE_GROUPS",
    "MODULE_WEIGHTS",
    "ParameterGeometry",
    "ParameterGroup",
    "SHARED_BIASES",
    "SHARED_GROUPS",
    "SHARED_WEIGHTS",
    "TEMPERATURES",
    "resolve_all",
    "resolve_geometry",
]

# ═════════════════════════════════════════════════════════════════════
# Type aliases
# ═════════════════════════════════════════════════════════════════════

Scope = Literal["module", "shared"]
"""Whether the parameter is per-classifier-module or shared across modules."""

GradientPath = Literal["module_tiled", "streaming"]
"""Which gradient lifecycle the parameter follows through the DAG."""


# ═════════════════════════════════════════════════════════════════════
# Internal helpers
# ═════════════════════════════════════════════════════════════════════


def _mc_suffix(base: str, mc: int | None) -> str:
    """Append ``__mc<N>`` for module-chunk-scoped identifiers."""
    return f"{base}__mc{mc}" if mc is not None else base


# ═════════════════════════════════════════════════════════════════════
# §1  ParameterGroup — structural identity
# ═════════════════════════════════════════════════════════════════════


@dataclass(frozen=True)
class ParameterGroup:
    """Immutable descriptor of one learnable parameter group.

    Carries the group's structural identity and provides naming methods
    that generate systematic buffer names and DAG node IDs, replacing
    scattered string formatting in the plan builder.

    Parameters
    ----------
    name:
        Canonical name (CONTRACT Article 8 Canonical Lexicon).
    scope:
        ``"module"`` — per-classifier-module, decomposable into
        module chunks (ADR-030).
        ``"shared"`` — spans all modules, not module-chunked.
    gradient_path:
        ``"module_tiled"`` — gradients produced per-tile (Nodes 8/10),
        reduced via ``ModuleChunkGather``.
        ``"streaming"`` — gradients produced per-streaming-chunk
        (Nodes 17/18), reduced via ``StridedGather``.
    post_update_kernel:
        Kernel name for a transform applied after ``adam_update``
        (e.g. ``"clamp_temperatures"``), or ``None``.
    """

    name: str
    scope: Scope
    gradient_path: GradientPath
    post_update_kernel: str | None = None

    # ── Optimizer state buffer names ──────────────────────────────

    @property
    def m1_name(self) -> str:
        """First moment vector buffer name."""
        return f"m1_{self.name}"

    @property
    def m2_name(self) -> str:
        """Second moment vector buffer name."""
        return f"m2_{self.name}"

    # ── Gradient-flow buffer names ────────────────────────────────

    def summed_grad_name(self, mc: int | None = None) -> str:
        """Reduction output buffer (batch-summed gradient)."""
        return _mc_suffix(f"summed_grad_{self.name}", mc)

    def final_grad_name(self, mc: int | None = None) -> str:
        """Normalised gradient buffer (output of Node 21)."""
        return _mc_suffix(f"final_grad_{self.name}", mc)

    # ── DAG node IDs ──────────────────────────────────────────────

    def reduce_node_id(self, mc: int | None = None) -> str:
        """``ReductionTreeNode`` ID for gradient aggregation."""
        return _mc_suffix(f"reduce_{self.name}_grad", mc)

    def normalize_node_id(self, mc: int | None = None) -> str:
        """``KernelDispatchNode`` ID for ``normalize_gradients``."""
        return _mc_suffix(f"normalize_gradients__{self.name}", mc)

    def adam_node_id(self, mc: int | None = None) -> str:
        """``KernelDispatchNode`` ID for ``adam_update``."""
        return _mc_suffix(f"adam_update__{self.name}", mc)

    def post_update_node_id(self, mc: int | None = None) -> str:
        """``KernelDispatchNode`` ID for the post-adam transform.

        Raises
        ------
        ValueError
            If no ``post_update_kernel`` is defined for this group.
        """
        if self.post_update_kernel is None:
            raise ValueError(
                f"Parameter group {self.name!r} has no post-update kernel"
            )
        return _mc_suffix(self.post_update_kernel, mc)

    @property
    def has_post_update(self) -> bool:
        """Whether this group requires a post-adam dispatch."""
        return self.post_update_kernel is not None


# ═════════════════════════════════════════════════════════════════════
# §2  ParameterGeometry — resolved shapes
# ═════════════════════════════════════════════════════════════════════


@dataclass(frozen=True)
class ParameterGeometry:
    """Concrete shape information for one parameter group.

    Produced by :func:`resolve_geometry` from a :class:`ParameterGroup`,
    a :class:`ModelSpec`, and the hardware SIMD width.  All shape fields
    use **padded** dimensions suitable for direct buffer allocation.

    Parameters
    ----------
    group:
        The parameter group this geometry describes.
    state_shape:
        Padded shape for the MODEL_STATE parameter buffer.
        For ``shared_weights`` this is the SIMD-major (SoA) layout
        ``(padded_hidden // simd, padded_input, simd)``.
    logical_shape:
        Unpadded shape for human-readable annotations, or ``None``
        when the padded shape is self-explanatory.
    elements_per_module:
        Flat element count per single module — the denominator for
        ``ParameterSlice.from_chunk()``.  ``None`` for shared-scope
        groups where the concept is inapplicable.
    optimizer_state_elements:
        Flat element count for the optimizer moment vectors (m1, m2).
        May differ from :attr:`total_flat_elements` when the state
        buffer is padded beyond the logical module count (e.g.
        ``temperatures``: state buffer is ``padded_module_dim``,
        optimizer state is ``num_modules``).
    """

    group: ParameterGroup
    state_shape: tuple[int, ...]
    logical_shape: tuple[int, ...] | None
    elements_per_module: int | None
    optimizer_state_elements: int

    @property
    def total_flat_elements(self) -> int:
        """Product of ``state_shape`` — total elements in the state buffer."""
        return math.prod(self.state_shape)


# ═════════════════════════════════════════════════════════════════════
# §3  Registry — the five canonical parameter groups
# ═════════════════════════════════════════════════════════════════════

SHARED_WEIGHTS: ParameterGroup = ParameterGroup(
    "shared_weights", "shared", "streaming",
)
"""Shared-layer weight matrix W — SIMD-major (SoA) layout."""

SHARED_BIASES: ParameterGroup = ParameterGroup(
    "shared_biases", "shared", "streaming",
)
"""Shared-layer bias vector b."""

MODULE_WEIGHTS: ParameterGroup = ParameterGroup(
    "module_weights", "module", "module_tiled",
)
"""Per-module classifier weight matrices."""

MODULE_BIASES: ParameterGroup = ParameterGroup(
    "module_biases", "module", "module_tiled",
)
"""Per-module classifier bias vectors."""

TEMPERATURES: ParameterGroup = ParameterGroup(
    "temperatures", "module", "module_tiled", "clamp_temperatures",
)
"""Per-module logit temperature scalars — clamped after each update."""

ALL_GROUPS: tuple[ParameterGroup, ...] = (
    SHARED_WEIGHTS,
    SHARED_BIASES,
    MODULE_WEIGHTS,
    MODULE_BIASES,
    TEMPERATURES,
)
"""All five learnable parameter groups in canonical order."""

MODULE_GROUPS: tuple[ParameterGroup, ...] = (
    MODULE_WEIGHTS,
    MODULE_BIASES,
    TEMPERATURES,
)
"""Module-scoped groups (module-tiled gradient path, per-chunk dispatch)."""

SHARED_GROUPS: tuple[ParameterGroup, ...] = (
    SHARED_WEIGHTS,
    SHARED_BIASES,
)
"""Shared groups (streaming gradient path, single dispatch)."""


# ═════════════════════════════════════════════════════════════════════
# §4  Resolution — geometry computation from ModelSpec
# ═════════════════════════════════════════════════════════════════════


def resolve_geometry(
    group: ParameterGroup,
    spec: ModelSpec,
    simd_width: int,
) -> ParameterGeometry:
    """Resolve a parameter group into concrete shapes for a given model.

    Parameters
    ----------
    group:
        The parameter group to resolve.
    spec:
        Model specification providing padded dimension values.
    simd_width:
        Hardware SIMD lane count (``HardwareProfile.simd_width``).
        Governs the shared-weights SIMD-major (SoA) reshape.

    Returns
    -------
    ParameterGeometry
        Concrete shapes and element counts for buffer allocation.

    Raises
    ------
    ValueError
        If ``group.name`` is not one of the five canonical groups, or
        if a dimensional constraint is violated.
    """
    match group.name:

        case "shared_weights":
            if spec.padded_hidden_dim % simd_width != 0:
                raise ValueError(
                    f"padded_hidden_dim ({spec.padded_hidden_dim}) must be "
                    f"divisible by simd_width ({simd_width}) for "
                    f"SIMD-major layout"
                )
            flat = spec.padded_hidden_dim * spec.padded_input_dim
            return ParameterGeometry(
                group=group,
                state_shape=(
                    spec.padded_hidden_dim // simd_width,
                    spec.padded_input_dim,
                    simd_width,
                ),
                logical_shape=(spec.hidden_dim, spec.input_dim),
                elements_per_module=None,
                optimizer_state_elements=flat,
            )

        case "shared_biases":
            return ParameterGeometry(
                group=group,
                state_shape=(spec.padded_hidden_dim,),
                logical_shape=(spec.hidden_dim,),
                elements_per_module=None,
                optimizer_state_elements=spec.padded_hidden_dim,
            )

        case "module_weights":
            epm = spec.padded_hidden_dim * spec.padded_class_dim
            return ParameterGeometry(
                group=group,
                state_shape=(
                    spec.num_modules,
                    spec.padded_hidden_dim,
                    spec.padded_class_dim,
                ),
                logical_shape=(
                    spec.num_modules,
                    spec.hidden_dim,
                    spec.output_classes,
                ),
                elements_per_module=epm,
                optimizer_state_elements=spec.num_modules * epm,
            )

        case "module_biases":
            epm = spec.padded_class_dim
            return ParameterGeometry(
                group=group,
                state_shape=(spec.num_modules, spec.padded_class_dim),
                logical_shape=(spec.num_modules, spec.output_classes),
                elements_per_module=epm,
                optimizer_state_elements=spec.num_modules * epm,
            )

        case "temperatures":
            return ParameterGeometry(
                group=group,
                state_shape=(spec.padded_module_dim,),
                logical_shape=(spec.num_modules,),
                elements_per_module=1,
                optimizer_state_elements=spec.num_modules,
            )

        case _:
            raise ValueError(f"Unknown parameter group: {group.name!r}")


def resolve_all(
    spec: ModelSpec,
    simd_width: int,
) -> dict[str, ParameterGeometry]:
    """Resolve all five parameter groups into a name-keyed geometry dict.

    Convenience wrapper around :func:`resolve_geometry`.

    Example
    -------
    >>> geos = resolve_all(model_spec, hardware.simd_width)
    >>> geos["module_weights"].elements_per_module
    4096
    >>> geos["shared_weights"].total_flat_elements
    32768
    """
    return {
        g.name: resolve_geometry(g, spec, simd_width)
        for g in ALL_GROUPS
    }
