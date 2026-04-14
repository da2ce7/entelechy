# src/shared/problem_type_spec.py
"""Problem-type divergence specification for CCE and BCE operating modes.

This module captures the **complete, closed set** of structural divergences
between CCE (single-label, Categorical Cross-Entropy) and BCE (multi-label,
Binary Cross-Entropy) operating modes.  It is the Policy tier's single
source of truth for problem-type-dependent decisions during plan
construction.

Conceptual role
───────────────
The averaging-ensembled-classifier architecture supports two mutually
exclusive classification modes per module.  These modes share the vast
majority of the computational DAG — the shared-layer forward pass
(Node 4), logit rendering (Node 5), all gradient kernels (Nodes 8–10
via FLAG scalar, CONCEPT.md Principle 3(A)), clipping, reduction,
normalisation, and optimiser update are identical regardless of mode.

The modes diverge in exactly five ways, all captured by
``ProblemTypeSpec``:

1. **Loss kernel** — Node 6 (CCE) vs Node 7 (BCE).  Principle 3(B)
   structural bifurcation: incompatible output shapes and DAG edges.
   Current applicants per CONTRACT Article 8 §7.0.
2. **Target buffer format** — ``int32`` class indices (CCE) vs
   ``STORAGE_TYPE`` multi-hot vectors (BCE).
3. **Loss output topology** — monolithic scatter-write with
   ``ZERO_REQUIRED`` initialisation (CCE) vs tiled partial-renderer
   output requiring downstream reduction (BCE).
4. **Loss buffer binding** — distinct ``dest_`` parameter names
   reflecting the different output semantics.
5. **FLAG value** — ``0`` (CCE) or ``1`` (BCE), injected into the
   shared gradient kernels (Nodes 8, 9, 10) at dispatch time.

What is *not* in this module
──────────────────────────────
Gradient kernel selection (Nodes 8, 9, 10) is mode-invariant.  These
kernels accept a ``src_scalar_FLAG_problem_type`` scalar for internal
path selection — CONCEPT.md Principle 3(A).  The plan builder references
their contracts directly via ``KERNEL_REGISTRY``, not through this
module.

Authoritative sources
─────────────────────
CONCEPT.md §1 Principle 3     — Modular, "Dumb" Kernels (bifurcation
                                 strategies A and B)
CONTRACT.md §4.2.3            — Kernel Bifurcation documentation key
CONTRACT.md Article 8 §7.0    — Bifurcation exception applicants
kernels.cl.h Nodes 6, 7       — Loss kernel contract specifications
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

from .kernel_contracts import (
    KERNEL_REGISTRY,
    KernelContract,
    PROBLEM_TYPE_BCE,
    PROBLEM_TYPE_CCE,
)

__all__ = [
    "BCE",
    "CCE",
    "ProblemTypeSpec",
    "for_mode",
]

# ═════════════════════════════════════════════════════════════════════
# Type aliases
# ═════════════════════════════════════════════════════════════════════

InitContract = Literal["ZERO_REQUIRED", "ZERO_REQUIRED_ADDITIVE", "NOT_REQUIRED"]
PrecisionRole = Literal["storage", "compute", "state"]


# ═════════════════════════════════════════════════════════════════════
# ProblemTypeSpec
# ═════════════════════════════════════════════════════════════════════


@dataclass(frozen=True)
class ProblemTypeSpec:
    """Immutable descriptor of one operating mode's divergence points.

    An instance fully specifies how plan construction must adapt to a
    given operating mode.  There are exactly two canonical instances
    (``CCE`` and ``BCE``), corresponding to the closed bifurcation set
    documented in CONTRACT Article 8 §7.0.

    The plan builder reads static fields for dispatch configuration and
    calls shape factories for runtime-dependent buffer allocation.  No
    field is optional — every divergence point has a defined value for
    both modes.

    Parameters
    ----------
    flag:
        Integer injected as ``src_scalar_FLAG_problem_type`` into
        shared-kernel dispatches (Nodes 8, 9, 10).  ``0`` = CCE,
        ``1`` = BCE.  Matches ``PROBLEM_TYPE_CCE`` /
        ``PROBLEM_TYPE_BCE`` from ``kernels.cl.h``.
    loss_kernel_name:
        Kernel function name for the loss computation — the
        Principle 3(B) bifurcation point.
        ``"compute_probs_loss_cce_chunk"`` (Node 6) or
        ``"compute_probs_loss_bce_chunk"`` (Node 7).
    loss_binding_key:
        The ``dest_`` buffer parameter name in the loss kernel's
        signature that receives the loss output.  CCE uses
        ``"dest_buffer_GLOBAL_final_loss"`` (monolithic,
        scatter-written).  BCE uses
        ``"dest_buffer_GLOBAL_partial_loss"`` (tiled partials).
    loss_init_contract:
        Pre-initialisation requirement for the loss output buffer.
        ``"ZERO_REQUIRED"`` for CCE (conditional writer — only the
        tile containing the true class writes; others rely on
        host-initialised zeros).  ``"NOT_REQUIRED"`` for BCE (every
        tile writes its full slice).
    loss_requires_reduction:
        Whether the loss output must be aggregated by the Recursive
        Reduction Engine (Node 14) before it is host-readable.
        ``False`` for CCE (scatter-write produces the final value).
        ``True`` for BCE (per-class partials require summation).
    targets_logical_name:
        Buffer descriptor name for the ground-truth target buffer.
    targets_precision_role:
        Precision role for the target buffer.  ``None`` for CCE
        (integer-typed class indices, exempt from the precision
        system).  ``"storage"`` for BCE (multi-hot vectors stored
        in ``STORAGE_TYPE``).
    """

    # ── Identity & FLAG injection (Principle 3(A)) ────────────────

    flag: int

    # ── Loss kernel (Principle 3(B) bifurcation) ─────────────────

    loss_kernel_name: str
    loss_binding_key: str
    loss_init_contract: InitContract
    loss_requires_reduction: bool

    # ── Target buffer specification ───────────────────────────────

    targets_logical_name: str
    targets_precision_role: PrecisionRole | None

    # ── Derived properties ────────────────────────────────────────

    @property
    def loss_contract(self) -> KernelContract:
        """The ``KernelContract`` for the loss computation kernel.

        Resolved from ``KERNEL_REGISTRY`` to ensure the canonical
        contract instance is always returned.
        """
        return KERNEL_REGISTRY[self.loss_kernel_name]

    # ── Target buffer factories ───────────────────────────────────

    def targets_element_size(self, storage_itemsize: int) -> int:
        """Element size in bytes for the target buffer.

        Parameters
        ----------
        storage_itemsize:
            ``PrecisionConfig.storage_dtype.itemsize`` — element size
            of the active storage format.  Used for BCE multi-hot
            targets.  Ignored for CCE.

        Returns
        -------
        int
            ``4`` for CCE (``int32`` class indices);
            *storage_itemsize* for BCE (``STORAGE_TYPE`` multi-hot).
        """
        if self.flag == PROBLEM_TYPE_CCE:
            return 4
        return storage_itemsize

    def targets_shape(
        self,
        *,
        batch_size: int,
        padded_class_dim: int,
    ) -> tuple[int, ...]:
        """Padded allocation shape for the target buffer.

        Parameters
        ----------
        batch_size:
            Number of samples in the batch
            (``src_scalar_NATURAL_total_batch_count``).
        padded_class_dim:
            ``ModelSpec.padded_class_dim`` — the padded output-class
            extent (``padded_total_output_class_count``).  Used only
            for BCE.

        Returns
        -------
        tuple[int, ...]
            ``(batch_size,)`` for CCE — 1-D int32 class indices per
            Node 6's ``src_buffer_GLOBAL_targets`` contract.

            ``(batch_size, padded_class_dim)`` for BCE — 2-D
            STORAGE_TYPE multi-hot matrix per Node 7's
            ``src_buffer_GLOBAL_targets`` contract.
        """
        if self.flag == PROBLEM_TYPE_CCE:
            return (batch_size,)
        return (batch_size, padded_class_dim)

    # ── Loss buffer factories ─────────────────────────────────────

    def loss_shape(
        self,
        *,
        num_modules: int,
        batch_size: int,
        tile_count: int,
        modules_per_chunk: int,
    ) -> tuple[int, ...]:
        """Padded allocation shape for the loss output buffer.

        Parameters
        ----------
        num_modules:
            Total classifier modules
            (``src_scalar_NATURAL_total_modules_count``).
        batch_size:
            Samples in the batch
            (``src_scalar_NATURAL_total_batch_count``).
        tile_count:
            ``TilingGeometry.total_tiles``
            (``src_scalar_NATURAL_total_tile_count``).
        modules_per_chunk:
            ``TilingGeometry.modules.chunk_size``
            (``src_scalar_NATURAL_modules_per_chunk``).

        Returns
        -------
        tuple[int, ...]
            ``(num_modules, batch_size)`` for CCE — one scalar loss
            per (module, sample) pair, scatter-written by the
            conditional writer (Node 6's
            ``dest_buffer_GLOBAL_final_loss``).

            ``(tile_count, modules_per_chunk, batch_size)`` for BCE —
            per-tile partial loss requiring downstream aggregation
            by Node 14 (Node 7's ``dest_buffer_GLOBAL_partial_loss``).
        """
        if self.flag == PROBLEM_TYPE_CCE:
            return (num_modules, batch_size)
        return (tile_count, modules_per_chunk, batch_size)


# ═════════════════════════════════════════════════════════════════════
# Canonical Instances
# ═════════════════════════════════════════════════════════════════════

CCE: ProblemTypeSpec = ProblemTypeSpec(
    flag=PROBLEM_TYPE_CCE,
    loss_kernel_name="compute_probs_loss_cce_chunk",
    loss_binding_key="dest_buffer_GLOBAL_final_loss",
    loss_init_contract="ZERO_REQUIRED",
    loss_requires_reduction=False,
    targets_logical_name="targets_cce",
    targets_precision_role=None,
)
"""CCE operating mode — single-label classification (Node 6).

Loss is scatter-written to a monolithic ``(modules, batch)`` buffer
with ``ZERO_REQUIRED`` initialisation.  Only the tile whose class-chunk
range contains the true class index writes a loss value; all other tiles
skip the write, relying on host-initialised zeros.

Targets are ``int32`` class indices with shape ``(batch_size,)``, exempt
from the precision role system.
"""

BCE: ProblemTypeSpec = ProblemTypeSpec(
    flag=PROBLEM_TYPE_BCE,
    loss_kernel_name="compute_probs_loss_bce_chunk",
    loss_binding_key="dest_buffer_GLOBAL_partial_loss",
    loss_init_contract="NOT_REQUIRED",
    loss_requires_reduction=True,
    targets_logical_name="targets_bce",
    targets_precision_role="storage",
)
"""BCE operating mode — multi-label classification (Node 7).

Loss is tiled across ``(tile_count, modules_per_chunk, batch)`` and
requires aggregation by the Recursive Reduction Engine (Node 14).
Each tile writes its full slice (no conditional-writer pattern), so
no pre-initialisation is required.

Targets are ``STORAGE_TYPE`` multi-hot vectors with shape
``(batch_size, padded_class_dim)``, governed by the ``"storage"``
precision role.
"""


# ═════════════════════════════════════════════════════════════════════
# Convenience Lookup
# ═════════════════════════════════════════════════════════════════════

_MODE_LOOKUP: dict[str, ProblemTypeSpec] = {
    "cce": CCE,
    "bce": BCE,
}


def for_mode(mode: str) -> ProblemTypeSpec:
    """Look up the canonical ``ProblemTypeSpec`` by operating mode name.

    Parameters
    ----------
    mode:
        ``"cce"`` or ``"bce"`` (case-sensitive, lowercase per
        CONTRACT Article 8 Canonical Lexicon conventions).

    Returns
    -------
    ProblemTypeSpec
        The corresponding canonical instance.

    Raises
    ------
    KeyError
        If *mode* is not a recognised operating mode.

    Examples
    --------
    >>> from src.shared.problem_type_strategy import for_mode
    >>> spec = for_mode("cce")
    >>> spec.flag
    0
    >>> spec.loss_contract.kernel_name
    'compute_probs_loss_cce_chunk'
    """
    try:
        return _MODE_LOOKUP[mode]
    except KeyError:
        raise KeyError(
            f"Unknown operating mode {mode!r}; "
            f"expected one of {sorted(_MODE_LOOKUP)}"
        ) from None
