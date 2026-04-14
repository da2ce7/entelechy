"""
Backend-neutral kernel contracts — machine-readable mirror of kernels.cl.h

This module reifies every ``@kernel_contract`` block and ``@param``
annotation from *kernels.cl.h* as frozen dataclasses, providing the
Policy tier with a complete, inspectable description of the host-device
interface.

Design decisions
────────────────
* **Option B per-dimension padding** (CONTRACT §3.3.3): each shape
  dimension is a ``Dim(expr, padding)`` pair co-locating the alignment
  constraint with the extent it governs.  This is a lossless encoding
  of the C header's per-dimension Padding Contract dictionary.

* **Shape expressions** strip the ``src_scalar_NATURAL_`` prefix that
  the C parameter names carry; dimension names match the Canonical
  Lexicon (CONTRACT Article 8) directly.

* **Parameter ordering** is identical to the C function signatures —
  the ``params`` tuple can be zipped with a backend's argument list
  without reordering.

* **Shared frozen instances** are used for buffer parameters whose
  contract (name, flow, scope, shape, role, conditional_on) is
  identical across multiple kernels.  This enforces cross-kernel
  consistency; any future divergence requires splitting the constant.

Authoritative source
────────────────────
kernels.cl.h  (Revision 10) — ADR-013 designation.
CONTRACT.md   (Revision 10) — Articles 3, 6, 8.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

# ═════════════════════════════════════════════════════════════════════
# Constants — mirrors of kernels.cl.h §Common Constants
# ═════════════════════════════════════════════════════════════════════

PROBLEM_TYPE_CCE: int = 0
PROBLEM_TYPE_BCE: int = 1
AGG_MODE_SUM: int = 0
AGG_MODE_AVERAGE: int = 1
SENTINEL_ABSENT_PARTIAL: int = 0xFFFF_FFFF

# ═════════════════════════════════════════════════════════════════════
# Type aliases
# ═════════════════════════════════════════════════════════════════════

PaddingType = Literal["CACHE", "SIMD", "UNPADDED"]
PrecisionRole = Literal["storage", "compute", "state", "flag-conditional"]
InitContract = Literal["ZERO_REQUIRED", "ZERO_REQUIRED_ADDITIVE", "NOT_REQUIRED"]

# ═════════════════════════════════════════════════════════════════════
# Core Dataclasses
# ═════════════════════════════════════════════════════════════════════


@dataclass(frozen=True)
class Dim:
    """A single tensor-shape dimension with its alignment constraint.

    Mirrors CONTRACT §3.3.3 per-dimension padding specification.

    Parameters
    ----------
    expr:
        Arithmetic expression for this dimension's extent, using
        Canonical Lexicon terms (CONTRACT Article 8).  References to
        kernel scalar parameters have the ``src_scalar_NATURAL_``
        prefix stripped.
    padding:
        The alignment strategy applied to this dimension.
        Vocabulary follows CONTRACT §3.3.1.
    """

    expr: str
    padding: PaddingType = "UNPADDED"


@dataclass(frozen=True)
class PlacementContract:
    """Mirrors CONTRACT §3.5 — Partial Renderer write-offset strategy.

    Parameters
    ----------
    strategy:
        Canonical strategy name from CONTRACT §3.5.3.
    key_params:
        Full canonical names of the scalar parameters serving as
        unique placement keys.
    context_params:
        Additional scalars required for offset calculation.
    """

    strategy: str
    key_params: tuple[str, ...]
    context_params: tuple[str, ...] = ()


@dataclass(frozen=True)
class BufferParam:
    """Machine-readable encoding of a single buffer parameter's contract.

    Mirrors the ``@param`` commentary block in *kernels.cl.h*,
    capturing every field defined in CONTRACT §3.2 (global-scope) or
    §3.1 (local-scope).

    Parameters
    ----------
    name:
        Full canonical parameter name including flow prefix and
        memory scope (CONTRACT §2.2).
    flow:
        Directional role — CONTRACT §2.1.
    scope:
        Memory address space — CONTRACT §2.2.1.
    shape:
        Tensor dimensions as ``Dim`` instances.  Each dimension
        co-locates its extent expression with its padding type
        (CONTRACT §3.3).  Empty for LOCAL-scope buffers (governed
        by *allocation_formula*) and for buffers whose tensor shape
        is architecturally "Undefined" (e.g. partial-collection
        pools).
    role:
        Precision-role dtype governing element type and allocation
        size (CONTRACT §3.2.1).  ``None`` for integer-typed or
        type-punned buffers exempt from the precision system.
    init:
        Pre-initialization requirement for ``dest_`` buffers
        (CONTRACT §3.4).
    placement:
        Partial Renderer write-offset strategy (CONTRACT §3.5).
    conditional_on:
        Full canonical name of the FLAG scalar that gates access
        (CONTRACT §3.6).
    allocation_formula:
        Constructive byte-count expression for LOCAL-scope buffers
        (CONTRACT §3.1).
    """

    name: str
    flow: Literal["src", "dest", "update", "out", "sync"]
    scope: Literal["GLOBAL", "GLOBAL_CONST", "DEVICE_CONST", "LOCAL"]
    shape: tuple[Dim, ...] = ()
    role: PrecisionRole | None = None
    init: InitContract = "NOT_REQUIRED"
    placement: PlacementContract | None = None
    conditional_on: str | None = None
    allocation_formula: str | None = None


@dataclass(frozen=True)
class ScalarParam:
    """Machine-readable encoding of a single scalar parameter.

    Parameters
    ----------
    name:
        Full canonical parameter name including flow prefix and
        number-type token (CONTRACT §2.3).
    flow:
        Directional role — CONTRACT §2.1.
    number_type:
        Abstract numerical domain — CONTRACT §2.4.
    """

    name: str
    # CONTRACT §2.1 permits "dest" on scalars, but no current kernel
    # uses it (scalars are pass-by-value in OpenCL).  Intentionally excluded.
    flow: Literal["src", "out"]
    number_type: Literal["NATURAL", "INTEGER", "REAL", "FLAG"]


@dataclass(frozen=True)
class KernelContract:
    """Complete interface specification for one kernel.

    Mirrors the ``@kernel_contract`` block and ordered parameter list
    from *kernels.cl.h*.  The ``params`` tuple preserves the C
    function signature's parameter ordering.

    Parameters
    ----------
    kernel_name:
        The C function name.
    idempotency:
        CONTRACT §4.2.1 value.
    sync:
        Synchronization model (CONTRACT §4.4 vocabulary).
    params:
        Ordered sequence of buffer and scalar parameters.
    bifurcation_peer:
        Peer kernel name when this kernel belongs to a
        Principle 3(B) bifurcated pair (CONTRACT §4.2.3).
    precision_variant_of:
        Base kernel name when this is a precision-typed variant
        (CONTRACT §4.2, Precision Variant key).
    """

    kernel_name: str
    idempotency: Literal[
        "Strictly Idempotent",
        "Associatively Non-Idempotent",
        "Fundamentally Non-Idempotent (Stateful)",
    ]
    sync: str
    params: tuple[BufferParam | ScalarParam, ...]
    bifurcation_peer: str | None = None
    precision_variant_of: str | None = None


# ═════════════════════════════════════════════════════════════════════
# Placement Contract helpers
# ═════════════════════════════════════════════════════════════════════

_PLC_MOD_CLS = PlacementContract(
    strategy="grid_mod_cls",
    key_params=("src_scalar_NATURAL_flat_tile_index",),
    context_params=("src_scalar_NATURAL_num_class_chunks",),
)

_PLC_MOD_CLS_BATCH = PlacementContract(
    strategy="grid_mod_cls_batch",
    key_params=(
        "src_scalar_NATURAL_flat_tile_index",
        "src_scalar_NATURAL_batch_chunk_index",
    ),
    context_params=(
        "src_scalar_NATURAL_num_class_chunks",
        "src_scalar_NATURAL_num_batch_chunks",
    ),
)

# ═════════════════════════════════════════════════════════════════════
# Shared frozen BufferParam instances
#
# Each constant is reused in every kernel where the contract
# (name, flow, scope, shape, role, conditional_on) is identical.
# ═════════════════════════════════════════════════════════════════════

# --- Infrastructure -------------------------------------------------

_LOCAL_REDUCE = BufferParam(
    "update_buffer_LOCAL_reduction_tile",
    "update",
    "LOCAL",
    role="compute",
    allocation_formula="get_local_size(0) * sizeof(COMPUTE_TYPE)",
)

_LOCAL_SIMD = BufferParam(
    "update_buffer_LOCAL_simd_tile",
    "update",
    "LOCAL",
    role="compute",
    allocation_formula=(
        "(SIMD_WIDTH + SIMD_WIDTH * SIMD_WIDTH) * sizeof(COMPUTE_TYPE)"
    ),
)

_SAMPLE_MASK = BufferParam(
    "src_buffer_GLOBAL_sample_mask",
    "src",
    "GLOBAL",
    shape=(Dim("(total_batch_count + 31) // 32"),),
)

# --- Forward-pass data flow -----------------------------------------

_INPUT = BufferParam(
    "src_buffer_GLOBAL_input",
    "src",
    "GLOBAL",
    shape=(Dim("total_batch_count"), Dim("padded_input_count", "CACHE")),
    role="storage",
)

_WEIGHTS_SHARED = BufferParam(
    "src_buffer_GLOBAL_CONST_weights_shared_simd_major",
    "src",
    "GLOBAL_CONST",
    shape=(
        Dim("padded_hidden_count // SIMD_WIDTH", "SIMD"),
        Dim("padded_input_count", "CACHE"),
        Dim("SIMD_WIDTH"),
    ),
    role="state",
)

_HIDDEN_ACT = BufferParam(
    "src_buffer_GLOBAL_hidden_activations",
    "src",
    "GLOBAL",
    shape=(Dim("total_batch_count"), Dim("padded_hidden_count", "CACHE")),
    role="storage",
)

_HIDDEN_MASK_SRC = BufferParam(
    "src_buffer_GLOBAL_hidden_mask",
    "src",
    "GLOBAL",
    shape=(Dim("total_batch_count"), Dim("padded_hidden_count", "CACHE")),
    role="storage",
    conditional_on="src_scalar_FLAG_use_explicit_hidden_mask",
)

_LOGITS = BufferParam(
    "src_buffer_GLOBAL_logits",
    "src",
    "GLOBAL",
    shape=(
        Dim("total_modules_count"),
        Dim("total_batch_count"),
        Dim("padded_total_output_class_count", "CACHE"),
    ),
    role="storage",
)

_WEIGHTS_MODULE = BufferParam(
    "src_buffer_GLOBAL_CONST_weights_module",
    "src",
    "GLOBAL_CONST",
    shape=(
        Dim("total_modules_count"),
        Dim("padded_hidden_count", "CACHE"),
        Dim("padded_total_output_class_count", "SIMD"),
    ),
    role="state",
)

# --- Learn-phase shared data ----------------------------------------

_TEMPS = BufferParam(
    "src_buffer_GLOBAL_CONST_temps",
    "src",
    "GLOBAL_CONST",
    shape=(Dim("total_modules_count"),),
    role="state",
)

_PARTIAL_PROBS = BufferParam(
    "src_buffer_GLOBAL_partial_probs",
    "src",
    "GLOBAL",
    shape=(
        Dim("total_tile_count"),
        Dim("modules_per_chunk"),
        Dim("total_batch_count"),
        Dim("classes_per_chunk"),
    ),
    role="storage",
)

_TARGETS_VOID = BufferParam(
    "src_buffer_GLOBAL_targets",
    "src",
    "GLOBAL",
    # Type-punned (void*); shape varies with problem_type FLAG.
    role="flag-conditional",
)

_SUMMED_GRAD_H = BufferParam(
    "src_buffer_GLOBAL_summed_grad_hidden_activations",
    "src",
    "GLOBAL",
    shape=(Dim("final_grad_hidden_activations_total_count"),),
    role="compute",
)


# #####################################################################
# #####################################################################
# ##                                                                 ##
# ##                     KERNEL CONTRACTS                            ##
# ##                                                                 ##
# #####################################################################
# #####################################################################


# =====================================================================
# Act Phase Kernels (Nodes 4–7)
# =====================================================================

# --- Node 4: Shared Layer Forward Pass --------------------------------

_forward_pass = KernelContract(
    kernel_name="forward_pass",
    idempotency="Strictly Idempotent",
    sync="Streamable",
    params=(
        _LOCAL_SIMD,
        _INPUT,
        _SAMPLE_MASK,
        _WEIGHTS_SHARED,
        BufferParam(
            "src_buffer_GLOBAL_CONST_biases_shared",
            "src",
            "GLOBAL_CONST",
            shape=(Dim("padded_hidden_count", "SIMD"),),
            role="state",
        ),
        BufferParam(
            "dest_buffer_GLOBAL_hidden_activations",
            "dest",
            "GLOBAL",
            shape=(
                Dim("total_batch_count"),
                Dim("padded_hidden_count", "CACHE"),
            ),
            role="storage",
        ),
        BufferParam(
            "dest_buffer_GLOBAL_hidden_mask",
            "dest",
            "GLOBAL",
            shape=(
                Dim("total_batch_count"),
                Dim("padded_hidden_count", "CACHE"),
            ),
            role="storage",
            conditional_on="out_scalar_FLAG_produce_hidden_mask",
        ),
        ScalarParam("out_scalar_FLAG_produce_hidden_mask", "out", "FLAG"),
        ScalarParam("src_scalar_NATURAL_batch_chunk_offset", "src", "NATURAL"),
        ScalarParam("src_scalar_NATURAL_batch_chunk_count", "src", "NATURAL"),
        ScalarParam("src_scalar_NATURAL_total_batch_count", "src", "NATURAL"),
        ScalarParam("src_scalar_NATURAL_input_count", "src", "NATURAL"),
        ScalarParam("src_scalar_NATURAL_padded_input_count", "src", "NATURAL"),
        ScalarParam("src_scalar_NATURAL_padded_hidden_count", "src", "NATURAL"),
    ),
)

# --- Node 5: Module Layer Logit Rendering -----------------------------

_render_logits_chunk = KernelContract(
    kernel_name="render_logits_chunk",
    idempotency="Strictly Idempotent",
    sync="Slice Renderer",
    params=(
        _HIDDEN_ACT,
        _HIDDEN_MASK_SRC,
        ScalarParam(
            "src_scalar_FLAG_use_explicit_hidden_mask", "src", "FLAG"
        ),
        _SAMPLE_MASK,
        _WEIGHTS_MODULE,
        BufferParam(
            "src_buffer_GLOBAL_CONST_biases_module",
            "src",
            "GLOBAL_CONST",
            shape=(
                Dim("total_modules_count"),
                Dim("padded_total_output_class_count", "SIMD"),
            ),
            role="state",
        ),
        BufferParam(
            "dest_buffer_GLOBAL_logits",
            "dest",
            "GLOBAL",
            shape=(
                Dim("total_modules_count"),
                Dim("total_batch_count"),
                Dim("padded_total_output_class_count", "CACHE"),
            ),
            role="storage",
        ),
        ScalarParam("src_scalar_NATURAL_batch_chunk_offset", "src", "NATURAL"),
        ScalarParam("src_scalar_NATURAL_batch_chunk_count", "src", "NATURAL"),
        ScalarParam(
            "src_scalar_NATURAL_module_chunk_offset", "src", "NATURAL"
        ),
        ScalarParam(
            "src_scalar_NATURAL_module_chunk_count", "src", "NATURAL"
        ),
        ScalarParam(
            "src_scalar_NATURAL_class_chunk_offset", "src", "NATURAL"
        ),
        ScalarParam(
            "src_scalar_NATURAL_class_chunk_count", "src", "NATURAL"
        ),
        ScalarParam("src_scalar_NATURAL_total_batch_count", "src", "NATURAL"),
        ScalarParam("src_scalar_NATURAL_hidden_count", "src", "NATURAL"),
        ScalarParam(
            "src_scalar_NATURAL_padded_hidden_count", "src", "NATURAL"
        ),
        ScalarParam(
            "src_scalar_NATURAL_total_output_class_count", "src", "NATURAL"
        ),
        ScalarParam(
            "src_scalar_NATURAL_padded_total_output_class_count",
            "src",
            "NATURAL",
        ),
        ScalarParam(
            "src_scalar_NATURAL_total_modules_count", "src", "NATURAL"
        ),
    ),
)

# --- Node 6: CCE Probability & Loss ----------------------------------

_compute_probs_loss_cce_chunk = KernelContract(
    kernel_name="compute_probs_loss_cce_chunk",
    idempotency="Strictly Idempotent",
    sync="Partial Renderer / Conditional Writer",
    bifurcation_peer="compute_probs_loss_bce_chunk",
    params=(
        _LOGITS,
        _TEMPS,
        BufferParam(
            "src_buffer_GLOBAL_targets",
            "src",
            "GLOBAL",
            shape=(Dim("total_batch_count"),),
            # int class indices — no precision role
        ),
        _SAMPLE_MASK,
        BufferParam(
            "dest_buffer_GLOBAL_partial_probs",
            "dest",
            "GLOBAL",
            shape=(
                Dim("total_tile_count"),
                Dim("modules_per_chunk"),
                Dim("total_batch_count"),
                Dim("classes_per_chunk"),
            ),
            role="storage",
            placement=_PLC_MOD_CLS,
        ),
        BufferParam(
            "dest_buffer_GLOBAL_final_loss",
            "dest",
            "GLOBAL",
            shape=(Dim("total_modules_count"), Dim("total_batch_count")),
            role="compute",
            init="ZERO_REQUIRED",
        ),
        ScalarParam("src_scalar_NATURAL_flat_tile_index", "src", "NATURAL"),
        ScalarParam("src_scalar_NATURAL_num_class_chunks", "src", "NATURAL"),
        ScalarParam(
            "src_scalar_NATURAL_classes_per_chunk", "src", "NATURAL"
        ),
        ScalarParam(
            "src_scalar_NATURAL_modules_per_chunk", "src", "NATURAL"
        ),
        ScalarParam("src_scalar_NATURAL_total_batch_count", "src", "NATURAL"),
        ScalarParam(
            "src_scalar_NATURAL_total_output_class_count", "src", "NATURAL"
        ),
        ScalarParam(
            "src_scalar_NATURAL_padded_total_output_class_count",
            "src",
            "NATURAL",
        ),
        ScalarParam(
            "src_scalar_NATURAL_total_modules_count", "src", "NATURAL"
        ),
        ScalarParam(
            "src_scalar_NATURAL_total_tile_count", "src", "NATURAL"
        ),
    ),
)

# --- Node 7: BCE Probability & Loss ----------------------------------

_compute_probs_loss_bce_chunk = KernelContract(
    kernel_name="compute_probs_loss_bce_chunk",
    idempotency="Strictly Idempotent",
    sync="Dual Partial Renderer",
    bifurcation_peer="compute_probs_loss_cce_chunk",
    params=(
        _LOGITS,
        _TEMPS,
        BufferParam(
            "src_buffer_GLOBAL_targets",
            "src",
            "GLOBAL",
            shape=(
                Dim("total_batch_count"),
                Dim("padded_total_output_class_count", "CACHE"),
            ),
            role="storage",
        ),
        _SAMPLE_MASK,
        BufferParam(
            "dest_buffer_GLOBAL_partial_probs",
            "dest",
            "GLOBAL",
            shape=(
                Dim("total_tile_count"),
                Dim("modules_per_chunk"),
                Dim("total_batch_count"),
                Dim("classes_per_chunk"),
            ),
            role="storage",
            placement=_PLC_MOD_CLS,
        ),
        BufferParam(
            "dest_buffer_GLOBAL_partial_loss",
            "dest",
            "GLOBAL",
            shape=(
                Dim("total_tile_count"),
                Dim("modules_per_chunk"),
                Dim("total_batch_count"),
            ),
            role="compute",
            placement=_PLC_MOD_CLS,
        ),
        ScalarParam("src_scalar_NATURAL_flat_tile_index", "src", "NATURAL"),
        ScalarParam("src_scalar_NATURAL_num_class_chunks", "src", "NATURAL"),
        ScalarParam(
            "src_scalar_NATURAL_classes_per_chunk", "src", "NATURAL"
        ),
        ScalarParam(
            "src_scalar_NATURAL_modules_per_chunk", "src", "NATURAL"
        ),
        ScalarParam("src_scalar_NATURAL_total_batch_count", "src", "NATURAL"),
        ScalarParam(
            "src_scalar_NATURAL_total_output_class_count", "src", "NATURAL"
        ),
        ScalarParam(
            "src_scalar_NATURAL_padded_total_output_class_count",
            "src",
            "NATURAL",
        ),
        ScalarParam(
            "src_scalar_NATURAL_total_modules_count", "src", "NATURAL"
        ),
        ScalarParam(
            "src_scalar_NATURAL_total_tile_count", "src", "NATURAL"
        ),
    ),
)


# =====================================================================
# Learn Phase I: Gradient Generation & Clipping (Nodes 8–11)
# =====================================================================

# --- Node 8: Module Parameter Gradients -------------------------------

_calculate_module_param_grads_chunk = KernelContract(
    kernel_name="calculate_module_param_grads_chunk",
    idempotency="Strictly Idempotent",
    sync="Dual Partial Renderer",
    params=(
        _LOCAL_REDUCE,
        _HIDDEN_ACT,
        _PARTIAL_PROBS,
        _TARGETS_VOID,
        _SAMPLE_MASK,
        _TEMPS,
        BufferParam(
            "dest_buffer_GLOBAL_partial_grad_weights_module",
            "dest",
            "GLOBAL",
            shape=(
                Dim("total_tile_count"),
                Dim("num_batch_chunks"),
                Dim("modules_per_chunk"),
                Dim("padded_hidden_count", "CACHE"),
                Dim("padded_total_output_class_count", "SIMD"),
            ),
            role="storage",
            init="ZERO_REQUIRED",
            placement=_PLC_MOD_CLS_BATCH,
        ),
        BufferParam(
            "dest_buffer_GLOBAL_partial_grad_biases_module",
            "dest",
            "GLOBAL",
            shape=(
                Dim("total_tile_count"),
                Dim("num_batch_chunks"),
                Dim("modules_per_chunk"),
                Dim("padded_total_output_class_count", "SIMD"),
            ),
            role="storage",
            init="ZERO_REQUIRED",
            placement=_PLC_MOD_CLS_BATCH,
        ),
        ScalarParam("src_scalar_FLAG_problem_type", "src", "FLAG"),
        ScalarParam("src_scalar_NATURAL_flat_tile_index", "src", "NATURAL"),
        ScalarParam(
            "src_scalar_NATURAL_batch_chunk_index", "src", "NATURAL"
        ),
        ScalarParam(
            "src_scalar_NATURAL_batch_chunk_offset", "src", "NATURAL"
        ),
        ScalarParam(
            "src_scalar_NATURAL_batch_chunk_count", "src", "NATURAL"
        ),
        ScalarParam(
            "src_scalar_NATURAL_num_batch_chunks", "src", "NATURAL"
        ),
        ScalarParam("src_scalar_NATURAL_num_class_chunks", "src", "NATURAL"),
        ScalarParam(
            "src_scalar_NATURAL_classes_per_chunk", "src", "NATURAL"
        ),
        ScalarParam(
            "src_scalar_NATURAL_modules_per_chunk", "src", "NATURAL"
        ),
        ScalarParam("src_scalar_NATURAL_total_batch_count", "src", "NATURAL"),
        ScalarParam("src_scalar_NATURAL_hidden_count", "src", "NATURAL"),
        ScalarParam(
            "src_scalar_NATURAL_padded_hidden_count", "src", "NATURAL"
        ),
        ScalarParam(
            "src_scalar_NATURAL_total_output_class_count", "src", "NATURAL"
        ),
        ScalarParam(
            "src_scalar_NATURAL_padded_total_output_class_count",
            "src",
            "NATURAL",
        ),
        ScalarParam(
            "src_scalar_NATURAL_total_modules_count", "src", "NATURAL"
        ),
        ScalarParam(
            "src_scalar_NATURAL_total_tile_count", "src", "NATURAL"
        ),
    ),
)

# --- Node 9: Hidden Layer Error Backpropagation -----------------------

_backprop_error_to_hidden_chunk = KernelContract(
    kernel_name="backprop_error_to_hidden_chunk",
    idempotency="Strictly Idempotent",
    sync="Partial Renderer",
    params=(
        _PARTIAL_PROBS,
        _TARGETS_VOID,
        _SAMPLE_MASK,
        _WEIGHTS_MODULE,
        _TEMPS,
        BufferParam(
            "dest_buffer_GLOBAL_partial_grad_hidden_activations_aos",
            "dest",
            "GLOBAL",
            shape=(
                Dim("total_tile_count"),
                Dim("modules_per_chunk"),
                Dim("total_batch_count"),
                Dim("padded_hidden_count", "CACHE"),
            ),
            role="storage",
            placement=_PLC_MOD_CLS,
        ),
        ScalarParam("src_scalar_FLAG_problem_type", "src", "FLAG"),
        ScalarParam("src_scalar_NATURAL_flat_tile_index", "src", "NATURAL"),
        ScalarParam("src_scalar_NATURAL_num_class_chunks", "src", "NATURAL"),
        ScalarParam(
            "src_scalar_NATURAL_classes_per_chunk", "src", "NATURAL"
        ),
        ScalarParam(
            "src_scalar_NATURAL_modules_per_chunk", "src", "NATURAL"
        ),
        ScalarParam("src_scalar_NATURAL_total_batch_count", "src", "NATURAL"),
        ScalarParam("src_scalar_NATURAL_hidden_count", "src", "NATURAL"),
        ScalarParam(
            "src_scalar_NATURAL_padded_hidden_count", "src", "NATURAL"
        ),
        ScalarParam(
            "src_scalar_NATURAL_total_output_class_count", "src", "NATURAL"
        ),
        ScalarParam(
            "src_scalar_NATURAL_padded_total_output_class_count",
            "src",
            "NATURAL",
        ),
        ScalarParam(
            "src_scalar_NATURAL_total_modules_count", "src", "NATURAL"
        ),
        ScalarParam(
            "src_scalar_NATURAL_total_tile_count", "src", "NATURAL"
        ),
    ),
)

# --- Node 10: Temperature Gradients -----------------------------------

_calculate_chunk_temp_gradients = KernelContract(
    kernel_name="calculate_chunk_temp_gradients",
    idempotency="Strictly Idempotent",
    sync="Partial Renderer",
    params=(
        _LOCAL_REDUCE,
        _LOGITS,
        _PARTIAL_PROBS,
        _TARGETS_VOID,
        _SAMPLE_MASK,
        _TEMPS,
        BufferParam(
            "dest_buffer_GLOBAL_partial_grad_temps",
            "dest",
            "GLOBAL",
            shape=(Dim("total_tile_count"), Dim("modules_per_chunk")),
            role="storage",
            placement=_PLC_MOD_CLS,
        ),
        ScalarParam("src_scalar_FLAG_problem_type", "src", "FLAG"),
        ScalarParam("src_scalar_NATURAL_flat_tile_index", "src", "NATURAL"),
        ScalarParam("src_scalar_NATURAL_num_class_chunks", "src", "NATURAL"),
        ScalarParam(
            "src_scalar_NATURAL_classes_per_chunk", "src", "NATURAL"
        ),
        ScalarParam(
            "src_scalar_NATURAL_modules_per_chunk", "src", "NATURAL"
        ),
        ScalarParam("src_scalar_NATURAL_total_batch_count", "src", "NATURAL"),
        ScalarParam(
            "src_scalar_NATURAL_total_output_class_count", "src", "NATURAL"
        ),
        ScalarParam(
            "src_scalar_NATURAL_padded_total_output_class_count",
            "src",
            "NATURAL",
        ),
        ScalarParam(
            "src_scalar_NATURAL_total_modules_count", "src", "NATURAL"
        ),
        ScalarParam(
            "src_scalar_NATURAL_total_tile_count", "src", "NATURAL"
        ),
    ),
)

# --- Node 11: Partial Gradient Clipping -------------------------------

_clip_partial_gradients = KernelContract(
    kernel_name="clip_partial_gradients",
    idempotency="Strictly Idempotent",
    sync="Utility / Stability Primitive",
    params=(
        _LOCAL_REDUCE,
        # --- source gradient buffers (post batch-chunk reduction) ---
        BufferParam(
            "src_buffer_GLOBAL_partial_grad_weights_module",
            "src",
            "GLOBAL",
            shape=(
                Dim("total_tile_count"),
                Dim("modules_per_chunk"),
                Dim("padded_hidden_count", "CACHE"),
                Dim("padded_total_output_class_count", "SIMD"),
            ),
            role="storage",
        ),
        BufferParam(
            "src_buffer_GLOBAL_partial_grad_biases_module",
            "src",
            "GLOBAL",
            shape=(
                Dim("total_tile_count"),
                Dim("modules_per_chunk"),
                Dim("padded_total_output_class_count", "SIMD"),
            ),
            role="storage",
        ),
        BufferParam(
            "src_buffer_GLOBAL_partial_grad_temps",
            "src",
            "GLOBAL",
            shape=(Dim("total_tile_count"), Dim("modules_per_chunk")),
            role="storage",
        ),
        BufferParam(
            "src_buffer_GLOBAL_partial_grad_hidden_activations_aos",
            "src",
            "GLOBAL",
            shape=(
                Dim("total_tile_count"),
                Dim("modules_per_chunk"),
                Dim("total_batch_count"),
                Dim("padded_hidden_count", "CACHE"),
            ),
            role="storage",
        ),
        BufferParam(
            "src_buffer_GLOBAL_CONST_clipping_threshold_per_item",
            "src",
            "GLOBAL_CONST",
            shape=(Dim("total_tile_count"),),
            role="compute",
            conditional_on="src_scalar_FLAG_use_per_item_norm",
        ),
        # --- destination (clipped) gradient buffers -----------------
        BufferParam(
            "dest_buffer_GLOBAL_clipped_partial_grad_weights_module",
            "dest",
            "GLOBAL",
            shape=(
                Dim("total_tile_count"),
                Dim("modules_per_chunk"),
                Dim("padded_hidden_count", "CACHE"),
                Dim("padded_total_output_class_count", "SIMD"),
            ),
            role="storage",
            placement=_PLC_MOD_CLS,
        ),
        BufferParam(
            "dest_buffer_GLOBAL_clipped_partial_grad_biases_module",
            "dest",
            "GLOBAL",
            shape=(
                Dim("total_tile_count"),
                Dim("modules_per_chunk"),
                Dim("padded_total_output_class_count", "SIMD"),
            ),
            role="storage",
            placement=_PLC_MOD_CLS,
        ),
        BufferParam(
            "dest_buffer_GLOBAL_clipped_partial_grad_temps",
            "dest",
            "GLOBAL",
            shape=(Dim("total_tile_count"), Dim("modules_per_chunk")),
            role="storage",
            placement=_PLC_MOD_CLS,
        ),
        BufferParam(
            "dest_buffer_GLOBAL_clipped_partial_grad_hidden_activations_aos",
            "dest",
            "GLOBAL",
            shape=(
                Dim("total_tile_count"),
                Dim("modules_per_chunk"),
                Dim("total_batch_count"),
                Dim("padded_hidden_count", "CACHE"),
            ),
            role="storage",
            placement=_PLC_MOD_CLS,
        ),
        # --- scalars ------------------------------------------------
        ScalarParam("src_scalar_FLAG_use_per_item_norm", "src", "FLAG"),
        ScalarParam(
            "src_scalar_REAL_clipping_threshold_t_pre", "src", "REAL"
        ),
        ScalarParam("src_scalar_REAL_epsilon", "src", "REAL"),
        ScalarParam("src_scalar_NATURAL_flat_tile_index", "src", "NATURAL"),
        ScalarParam("src_scalar_NATURAL_num_class_chunks", "src", "NATURAL"),
        ScalarParam(
            "src_scalar_NATURAL_classes_per_chunk", "src", "NATURAL"
        ),
        ScalarParam(
            "src_scalar_NATURAL_modules_per_chunk", "src", "NATURAL"
        ),
        ScalarParam("src_scalar_NATURAL_total_batch_count", "src", "NATURAL"),
        ScalarParam(
            "src_scalar_NATURAL_padded_hidden_count", "src", "NATURAL"
        ),
        ScalarParam(
            "src_scalar_NATURAL_padded_total_output_class_count",
            "src",
            "NATURAL",
        ),
        ScalarParam(
            "src_scalar_NATURAL_total_tile_count", "src", "NATURAL"
        ),
    ),
)


# =====================================================================
# Learn Phase II: Aggregation, Reduction & Specialised Processing
#                 (Nodes 13, 14–15, 16, 20)
# =====================================================================

# --- Node 13: Gradient Gather & Permutation ---------------------------

_gather_and_permute_grad_hidden_activations = KernelContract(
    kernel_name="gather_and_permute_grad_hidden_activations",
    idempotency="Strictly Idempotent",
    sync="Global Barrier",
    params=(
        BufferParam(
            "src_buffer_GLOBAL_clipped_partial_grad_hidden_activations_aos",
            "src",
            "GLOBAL",
            shape=(
                Dim("total_tile_count"),
                Dim("modules_per_chunk"),
                Dim("total_batch_count"),
                Dim("padded_hidden_count", "CACHE"),
            ),
            role="storage",
        ),
        BufferParam(
            "dest_buffer_GLOBAL_clipped_grad_hidden_activations_permuted_soa",
            "dest",
            "GLOBAL",
            shape=(
                Dim("total_batch_count * padded_hidden_count", "CACHE"),
                Dim("padded_total_modules_count", "CACHE"),
            ),
            role="storage",
            init="ZERO_REQUIRED",
        ),
        ScalarParam("src_scalar_NATURAL_total_batch_count", "src", "NATURAL"),
        ScalarParam("src_scalar_NATURAL_hidden_count", "src", "NATURAL"),
        ScalarParam(
            "src_scalar_NATURAL_padded_hidden_count", "src", "NATURAL"
        ),
        ScalarParam(
            "src_scalar_NATURAL_total_modules_count", "src", "NATURAL"
        ),
        ScalarParam(
            "src_scalar_NATURAL_padded_total_modules_count", "src", "NATURAL"
        ),
        ScalarParam(
            "src_scalar_NATURAL_num_module_chunks", "src", "NATURAL"
        ),
        ScalarParam(
            "src_scalar_NATURAL_modules_per_chunk", "src", "NATURAL"
        ),
        ScalarParam("src_scalar_NATURAL_num_class_chunks", "src", "NATURAL"),
        ScalarParam(
            "src_scalar_NATURAL_total_tile_count", "src", "NATURAL"
        ),
    ),
)

# --- Recursive Clip-Aggregation Engine --------------------------------
#
# Single-stage: aggregate_*_reduce  (+  clip_intermediate_grad)
# Multi-stage:  reduce_k_fan_in_and_clip
# Each has storage-entry and compute-entry precision variants.

# -- Register Reduce (Storage-Entry, Tier 1) --

_aggregate_register_reduce = KernelContract(
    kernel_name="aggregate_register_reduce",
    idempotency="Associatively Non-Idempotent",
    sync="Reduction Engine Stage",
    params=(
        BufferParam(
            "src_buffer_GLOBAL_partial_collection",
            "src",
            "GLOBAL",
            role="storage",
        ),
        BufferParam(
            "src_buffer_GLOBAL_CONST_partial_offset_list",
            "src",
            "GLOBAL_CONST",
            shape=(Dim("partial_offset_list_count"),),
        ),
        BufferParam(
            "dest_buffer_GLOBAL_partial",
            "dest",
            "GLOBAL",
            shape=(Dim("partial_width"),),
            role="compute",
        ),
        ScalarParam(
            "src_scalar_NATURAL_partial_offset_list_count", "src", "NATURAL"
        ),
        ScalarParam("src_scalar_NATURAL_partial_width", "src", "NATURAL"),
        ScalarParam("src_scalar_FLAG_operation_type", "src", "FLAG"),
    ),
)

# -- Register Reduce (Compute-Entry, Tier 1) --

_aggregate_register_reduce_from_compute = KernelContract(
    kernel_name="aggregate_register_reduce_from_compute",
    idempotency="Associatively Non-Idempotent",
    sync="Reduction Engine Stage",
    precision_variant_of="aggregate_register_reduce",
    params=(
        BufferParam(
            "src_buffer_GLOBAL_partial_collection",
            "src",
            "GLOBAL",
            role="compute",
        ),
        BufferParam(
            "src_buffer_GLOBAL_CONST_partial_offset_list",
            "src",
            "GLOBAL_CONST",
            shape=(Dim("partial_offset_list_count"),),
        ),
        BufferParam(
            "dest_buffer_GLOBAL_partial",
            "dest",
            "GLOBAL",
            shape=(Dim("partial_width"),),
            role="compute",
        ),
        ScalarParam(
            "src_scalar_NATURAL_partial_offset_list_count", "src", "NATURAL"
        ),
        ScalarParam("src_scalar_NATURAL_partial_width", "src", "NATURAL"),
        ScalarParam("src_scalar_FLAG_operation_type", "src", "FLAG"),
    ),
)

# -- Local Reduce (Storage-Entry, Tier 2) --

_aggregate_local_reduce = KernelContract(
    kernel_name="aggregate_local_reduce",
    idempotency="Associatively Non-Idempotent",
    sync="Reduction Engine Stage / Work-group Parallel",
    params=(
        _LOCAL_REDUCE,
        BufferParam(
            "src_buffer_GLOBAL_partial_collection",
            "src",
            "GLOBAL",
            role="storage",
        ),
        BufferParam(
            "src_buffer_GLOBAL_CONST_partial_offset_list",
            "src",
            "GLOBAL_CONST",
            shape=(Dim("partial_offset_list_count"),),
        ),
        BufferParam(
            "dest_buffer_GLOBAL_partial",
            "dest",
            "GLOBAL",
            shape=(Dim("partial_width"),),
            role="compute",
        ),
        ScalarParam(
            "src_scalar_NATURAL_partial_offset_list_count", "src", "NATURAL"
        ),
        ScalarParam("src_scalar_NATURAL_partial_width", "src", "NATURAL"),
        ScalarParam("src_scalar_FLAG_operation_type", "src", "FLAG"),
    ),
)

# -- Local Reduce (Compute-Entry, Tier 2) --

_aggregate_local_reduce_from_compute = KernelContract(
    kernel_name="aggregate_local_reduce_from_compute",
    idempotency="Associatively Non-Idempotent",
    sync="Reduction Engine Stage / Work-group Parallel",
    precision_variant_of="aggregate_local_reduce",
    params=(
        _LOCAL_REDUCE,
        BufferParam(
            "src_buffer_GLOBAL_partial_collection",
            "src",
            "GLOBAL",
            role="compute",
        ),
        BufferParam(
            "src_buffer_GLOBAL_CONST_partial_offset_list",
            "src",
            "GLOBAL_CONST",
            shape=(Dim("partial_offset_list_count"),),
        ),
        BufferParam(
            "dest_buffer_GLOBAL_partial",
            "dest",
            "GLOBAL",
            shape=(Dim("partial_width"),),
            role="compute",
        ),
        ScalarParam(
            "src_scalar_NATURAL_partial_offset_list_count", "src", "NATURAL"
        ),
        ScalarParam("src_scalar_NATURAL_partial_width", "src", "NATURAL"),
        ScalarParam("src_scalar_FLAG_operation_type", "src", "FLAG"),
    ),
)

# -- Single-Stage Clip --

_clip_intermediate_grad = KernelContract(
    kernel_name="clip_intermediate_grad",
    idempotency="Associatively Non-Idempotent",
    sync="Reduction Engine Stage Clip Primitive",
    params=(
        _LOCAL_REDUCE,
        BufferParam(
            "update_buffer_GLOBAL_intermediate_grad",
            "update",
            "GLOBAL",
            shape=(Dim("parameter_count"),),
            role="compute",
        ),
        ScalarParam(
            "src_scalar_REAL_clipping_threshold_t_j", "src", "REAL"
        ),
        ScalarParam("src_scalar_REAL_epsilon", "src", "REAL"),
        ScalarParam(
            "src_scalar_NATURAL_parameter_count", "src", "NATURAL"
        ),
    ),
)

# -- K-Fan-In Reduce & Clip (Storage-Entry) --

_reduce_k_fan_in_and_clip = KernelContract(
    kernel_name="reduce_k_fan_in_and_clip",
    idempotency="Associatively Non-Idempotent",
    sync="Reduction Engine Stage",
    params=(
        _LOCAL_REDUCE,
        BufferParam(
            "src_buffer_GLOBAL_partial_collection",
            "src",
            "GLOBAL",
            role="storage",
        ),
        BufferParam(
            "src_buffer_GLOBAL_CONST_offset_list_flat",
            "src",
            "GLOBAL_CONST",
            shape=(Dim("node_count * fan_in"),),
        ),
        BufferParam(
            "dest_buffer_GLOBAL_stage_partial",
            "dest",
            "GLOBAL",
            shape=(Dim("node_count * partial_width"),),
            role="compute",
        ),
        ScalarParam("src_scalar_NATURAL_fan_in", "src", "NATURAL"),
        ScalarParam("src_scalar_NATURAL_node_count", "src", "NATURAL"),
        ScalarParam("src_scalar_NATURAL_partial_width", "src", "NATURAL"),
        ScalarParam(
            "src_scalar_REAL_clipping_threshold_t_j", "src", "REAL"
        ),
        ScalarParam("src_scalar_REAL_epsilon", "src", "REAL"),
    ),
)

# -- K-Fan-In Reduce & Clip (Compute-Entry) --

_reduce_k_fan_in_and_clip_from_compute = KernelContract(
    kernel_name="reduce_k_fan_in_and_clip_from_compute",
    idempotency="Associatively Non-Idempotent",
    sync="Reduction Engine Stage",
    precision_variant_of="reduce_k_fan_in_and_clip",
    params=(
        _LOCAL_REDUCE,
        BufferParam(
            "src_buffer_GLOBAL_partial_collection",
            "src",
            "GLOBAL",
            role="compute",
        ),
        BufferParam(
            "src_buffer_GLOBAL_CONST_offset_list_flat",
            "src",
            "GLOBAL_CONST",
            shape=(Dim("node_count * fan_in"),),
        ),
        BufferParam(
            "dest_buffer_GLOBAL_stage_partial",
            "dest",
            "GLOBAL",
            shape=(Dim("node_count * partial_width"),),
            role="compute",
        ),
        ScalarParam("src_scalar_NATURAL_fan_in", "src", "NATURAL"),
        ScalarParam("src_scalar_NATURAL_node_count", "src", "NATURAL"),
        ScalarParam("src_scalar_NATURAL_partial_width", "src", "NATURAL"),
        ScalarParam(
            "src_scalar_REAL_clipping_threshold_t_j", "src", "REAL"
        ),
        ScalarParam("src_scalar_REAL_epsilon", "src", "REAL"),
    ),
)

# --- Precision Bridge Primitive ---------------------------------------

_narrow_to_storage = KernelContract(
    kernel_name="narrow_to_storage",
    idempotency="Strictly Idempotent",
    sync="Utility",
    params=(
        BufferParam(
            "src_buffer_GLOBAL_input",
            "src",
            "GLOBAL",
            shape=(Dim("element_count"),),
            role="compute",
        ),
        BufferParam(
            "dest_buffer_GLOBAL_output",
            "dest",
            "GLOBAL",
            shape=(Dim("element_count"),),
            role="storage",
        ),
        ScalarParam("src_scalar_NATURAL_element_count", "src", "NATURAL"),
    ),
)

# --- Node 16: Specialized Grad_H Reduction ----------------------------

_stabilize_and_reduce_grad_hidden_activations = KernelContract(
    kernel_name="stabilize_and_reduce_grad_hidden_activations",
    idempotency="Associatively Non-Idempotent",
    sync="Specialized Reduction Kernel / Global Barrier",
    params=(
        _LOCAL_REDUCE,
        BufferParam(
            "src_buffer_GLOBAL_clipped_grad_hidden_activations_permuted_soa",
            "src",
            "GLOBAL",
            shape=(
                Dim("total_batch_count * padded_hidden_count", "CACHE"),
                Dim("padded_total_modules_count", "CACHE"),
            ),
            role="storage",
        ),
        BufferParam(
            "dest_buffer_GLOBAL_summed_grad_hidden_activations",
            "dest",
            "GLOBAL",
            shape=(Dim("total_batch_count * padded_hidden_count"),),
            role="compute",
        ),
        BufferParam(
            "src_buffer_GLOBAL_CONST_clipping_threshold_per_stage",
            "src",
            "GLOBAL_CONST",
            shape=(Dim("num_reduction_stages"),),
            role="compute",
        ),
        ScalarParam(
            "src_scalar_NATURAL_num_reduction_stages", "src", "NATURAL"
        ),
        ScalarParam(
            "src_scalar_REAL_clipping_threshold_t_pre", "src", "REAL"
        ),
        ScalarParam("src_scalar_REAL_epsilon", "src", "REAL"),
        ScalarParam("src_scalar_NATURAL_total_batch_count", "src", "NATURAL"),
        ScalarParam(
            "src_scalar_NATURAL_padded_hidden_count", "src", "NATURAL"
        ),
        ScalarParam(
            "src_scalar_NATURAL_total_modules_count", "src", "NATURAL"
        ),
        ScalarParam(
            "src_scalar_NATURAL_padded_total_modules_count", "src", "NATURAL"
        ),
    ),
)


# =====================================================================
# Learn Phase III: Streaming Shared-Layer Backpropagation (Nodes 17–19)
# =====================================================================

# --- Node 17: Shared Weight Gradients ---------------------------------

_backprop_shared_weights_chunk = KernelContract(
    kernel_name="backprop_shared_weights_chunk",
    idempotency="Associatively Non-Idempotent",
    sync="Streamable",
    params=(
        _LOCAL_REDUCE,
        _INPUT,
        _HIDDEN_ACT,
        _HIDDEN_MASK_SRC,
        ScalarParam(
            "src_scalar_FLAG_use_explicit_hidden_mask", "src", "FLAG"
        ),
        _SUMMED_GRAD_H,
        _SAMPLE_MASK,
        BufferParam(
            "dest_buffer_GLOBAL_partial_grad_weights_shared_simd_major",
            "dest",
            "GLOBAL",
            shape=(
                Dim("1"),
                Dim("padded_hidden_count // SIMD_WIDTH", "SIMD"),
                Dim("padded_input_count", "CACHE"),
                Dim("SIMD_WIDTH"),
            ),
            role="storage",
        ),
        ScalarParam(
            "src_scalar_NATURAL_batch_chunk_offset", "src", "NATURAL"
        ),
        ScalarParam(
            "src_scalar_NATURAL_batch_chunk_count", "src", "NATURAL"
        ),
        ScalarParam("src_scalar_NATURAL_total_batch_count", "src", "NATURAL"),
        ScalarParam("src_scalar_NATURAL_input_count", "src", "NATURAL"),
        ScalarParam(
            "src_scalar_NATURAL_padded_input_count", "src", "NATURAL"
        ),
        ScalarParam("src_scalar_NATURAL_hidden_count", "src", "NATURAL"),
        ScalarParam(
            "src_scalar_NATURAL_padded_hidden_count", "src", "NATURAL"
        ),
        ScalarParam(
            "src_scalar_NATURAL_final_grad_hidden_activations_total_count",
            "src",
            "NATURAL",
        ),
    ),
)

# --- Node 18: Shared Bias Gradients -----------------------------------

_backprop_shared_biases_chunk = KernelContract(
    kernel_name="backprop_shared_biases_chunk",
    idempotency="Associatively Non-Idempotent",
    sync="Streamable",
    params=(
        _LOCAL_REDUCE,
        _HIDDEN_ACT,
        _HIDDEN_MASK_SRC,
        ScalarParam(
            "src_scalar_FLAG_use_explicit_hidden_mask", "src", "FLAG"
        ),
        _SUMMED_GRAD_H,
        _SAMPLE_MASK,
        BufferParam(
            "dest_buffer_GLOBAL_partial_grad_biases_shared",
            "dest",
            "GLOBAL",
            shape=(Dim("1"), Dim("padded_hidden_count", "CACHE")),
            role="storage",
        ),
        ScalarParam(
            "src_scalar_NATURAL_batch_chunk_offset", "src", "NATURAL"
        ),
        ScalarParam(
            "src_scalar_NATURAL_batch_chunk_count", "src", "NATURAL"
        ),
        ScalarParam("src_scalar_NATURAL_total_batch_count", "src", "NATURAL"),
        ScalarParam("src_scalar_NATURAL_hidden_count", "src", "NATURAL"),
        ScalarParam(
            "src_scalar_NATURAL_padded_hidden_count", "src", "NATURAL"
        ),
        ScalarParam(
            "src_scalar_NATURAL_final_grad_hidden_activations_total_count",
            "src",
            "NATURAL",
        ),
    ),
)

# --- Node 19: Shared Gradient Clipping --------------------------------

_clip_shared_gradients_chunk = KernelContract(
    kernel_name="clip_shared_gradients_chunk",
    idempotency="Strictly Idempotent",
    sync="Streamable Utility / Stability Primitive",
    params=(
        _LOCAL_REDUCE,
        BufferParam(
            "src_buffer_GLOBAL_partial_grad_weights_shared_simd_major",
            "src",
            "GLOBAL",
            shape=(Dim("weights_parameter_count"),),
            role="storage",
        ),
        BufferParam(
            "src_buffer_GLOBAL_partial_grad_biases_shared",
            "src",
            "GLOBAL",
            shape=(Dim("biases_parameter_count"),),
            role="storage",
        ),
        BufferParam(
            "dest_buffer_GLOBAL_clipped_partial_grad_weights_shared_simd_major",
            "dest",
            "GLOBAL",
            shape=(
                Dim("num_batch_chunks"),
                Dim("weights_parameter_count"),
            ),
            role="storage",
        ),
        BufferParam(
            "dest_buffer_GLOBAL_clipped_partial_grad_biases_shared",
            "dest",
            "GLOBAL",
            shape=(
                Dim("num_batch_chunks"),
                Dim("biases_parameter_count"),
            ),
            role="storage",
        ),
        ScalarParam(
            "src_scalar_REAL_clipping_threshold_t_pre", "src", "REAL"
        ),
        ScalarParam("src_scalar_REAL_epsilon", "src", "REAL"),
        ScalarParam(
            "src_scalar_NATURAL_weights_parameter_count", "src", "NATURAL"
        ),
        ScalarParam(
            "src_scalar_NATURAL_biases_parameter_count", "src", "NATURAL"
        ),
        ScalarParam(
            "out_scalar_NATURAL_weights_write_offset", "out", "NATURAL"
        ),
        ScalarParam(
            "out_scalar_NATURAL_biases_write_offset", "out", "NATURAL"
        ),
        ScalarParam(
            "src_scalar_NATURAL_num_batch_chunks", "src", "NATURAL"
        ),
    ),
)


# =====================================================================
# Learn Phase IV–V: Finalisation & Parameter Updates (Nodes 21, 24–25)
# =====================================================================

# --- Node 21: Gradient Normalization ----------------------------------

_normalize_gradients = KernelContract(
    kernel_name="normalize_gradients",
    idempotency="Strictly Idempotent",
    sync="Finalizer Utility / Batch-wide Normalizer",
    params=(
        BufferParam(
            "src_buffer_GLOBAL_summed_grad",
            "src",
            "GLOBAL",
            shape=(Dim("parameter_count"),),
            role="compute",
        ),
        BufferParam(
            "dest_buffer_GLOBAL_final_grad",
            "dest",
            "GLOBAL",
            shape=(Dim("parameter_count"),),
            role="compute",
        ),
        ScalarParam(
            "src_scalar_REAL_effective_batch_size", "src", "REAL"
        ),
        ScalarParam("src_scalar_REAL_epsilon", "src", "REAL"),
        ScalarParam(
            "src_scalar_NATURAL_parameter_count", "src", "NATURAL"
        ),
    ),
)

# --- Node 24: Adam Optimizer Update -----------------------------------

_adam_update = KernelContract(
    kernel_name="adam_update",
    idempotency="Fundamentally Non-Idempotent (Stateful)",
    sync="Stateful Optimizer Update",
    params=(
        BufferParam(
            "src_buffer_GLOBAL_final_grad",
            "src",
            "GLOBAL",
            shape=(Dim("parameter_count"),),
            role="compute",
        ),
        BufferParam(
            "update_buffer_GLOBAL_parameters",
            "update",
            "GLOBAL",
            shape=(Dim("total_parameter_count"),),
            role="state",
        ),
        BufferParam(
            "update_buffer_GLOBAL_m1",
            "update",
            "GLOBAL",
            shape=(Dim("total_parameter_count"),),
            role="state",
        ),
        BufferParam(
            "update_buffer_GLOBAL_m2",
            "update",
            "GLOBAL",
            shape=(Dim("total_parameter_count"),),
            role="state",
        ),
        ScalarParam("src_scalar_REAL_learning_rate", "src", "REAL"),
        ScalarParam("src_scalar_REAL_beta1_pow_t", "src", "REAL"),
        ScalarParam("src_scalar_REAL_beta2_pow_t", "src", "REAL"),
        ScalarParam("src_scalar_REAL_beta1", "src", "REAL"),
        ScalarParam("src_scalar_REAL_beta2", "src", "REAL"),
        ScalarParam("src_scalar_REAL_epsilon", "src", "REAL"),
        ScalarParam(
            "src_scalar_NATURAL_parameter_offset", "src", "NATURAL"
        ),
        ScalarParam(
            "src_scalar_NATURAL_parameter_count", "src", "NATURAL"
        ),
        ScalarParam(
            "src_scalar_NATURAL_total_parameter_count", "src", "NATURAL"
        ),
    ),
)

# --- Node 25: Temperature Clamping ------------------------------------

_clamp_temperatures = KernelContract(
    kernel_name="clamp_temperatures",
    idempotency="Fundamentally Non-Idempotent (Stateful)",
    sync="Finalizer Utility",
    params=(
        BufferParam(
            "update_buffer_GLOBAL_temps",
            "update",
            "GLOBAL",
            shape=(Dim("total_parameter_count"),),
            role="state",
        ),
        ScalarParam("src_scalar_REAL_min_value", "src", "REAL"),
        ScalarParam("src_scalar_REAL_max_value", "src", "REAL"),
        ScalarParam(
            "src_scalar_NATURAL_parameter_offset", "src", "NATURAL"
        ),
        ScalarParam(
            "src_scalar_NATURAL_parameter_count", "src", "NATURAL"
        ),
        ScalarParam(
            "src_scalar_NATURAL_total_parameter_count", "src", "NATURAL"
        ),
    ),
)


# =====================================================================
# Experimental Kernels
# =====================================================================

# --- Transpose SIMD-Major Matrix-Vector (inverse of Node 4) ----------

_transpose_matvec_masked_simd_major = KernelContract(
    kernel_name="transpose_matvec_masked_simd_major",
    idempotency="Strictly Idempotent",
    sync="Streamable",
    bifurcation_peer="forward_pass",
    params=(
        _LOCAL_SIMD,
        _WEIGHTS_SHARED,
        BufferParam(
            "src_buffer_GLOBAL_grad_hidden_activations",
            "src",
            "GLOBAL",
            shape=(
                Dim("total_batch_count"),
                Dim("padded_hidden_count"),
            ),
            role="compute",
        ),
        _HIDDEN_ACT,
        _HIDDEN_MASK_SRC,
        _SAMPLE_MASK,
        BufferParam(
            "dest_buffer_GLOBAL_grad_input",
            "dest",
            "GLOBAL",
            shape=(
                Dim("total_batch_count"),
                Dim("padded_input_count", "CACHE"),
            ),
            role="storage",
        ),
        ScalarParam(
            "src_scalar_FLAG_use_explicit_hidden_mask", "src", "FLAG"
        ),
        ScalarParam(
            "src_scalar_NATURAL_batch_chunk_offset", "src", "NATURAL"
        ),
        ScalarParam(
            "src_scalar_NATURAL_batch_chunk_count", "src", "NATURAL"
        ),
        ScalarParam("src_scalar_NATURAL_total_batch_count", "src", "NATURAL"),
        ScalarParam("src_scalar_NATURAL_input_count", "src", "NATURAL"),
        ScalarParam(
            "src_scalar_NATURAL_padded_input_count", "src", "NATURAL"
        ),
        ScalarParam("src_scalar_NATURAL_hidden_count", "src", "NATURAL"),
        ScalarParam(
            "src_scalar_NATURAL_padded_hidden_count", "src", "NATURAL"
        ),
    ),
)

# --- Element-Wise Addition (branch-point gradient combination) --------

_elementwise_add = KernelContract(
    kernel_name="elementwise_add",
    idempotency="Strictly Idempotent",
    sync="Utility",
    params=(
        BufferParam(
            "src_buffer_GLOBAL_input_a",
            "src",
            "GLOBAL",
            shape=(Dim("element_count"),),
            role="compute",
        ),
        BufferParam(
            "src_buffer_GLOBAL_input_b",
            "src",
            "GLOBAL",
            shape=(Dim("element_count"),),
            role="compute",
        ),
        BufferParam(
            "dest_buffer_GLOBAL_output",
            "dest",
            "GLOBAL",
            shape=(Dim("element_count"),),
            role="compute",
        ),
        ScalarParam("src_scalar_NATURAL_element_count", "src", "NATURAL"),
    ),
)


# ═════════════════════════════════════════════════════════════════════
# Registry — single look-up point for the Policy tier
# ═════════════════════════════════════════════════════════════════════

KERNEL_REGISTRY: dict[str, KernelContract] = {
    c.kernel_name: c
    for c in (
        _forward_pass,
        _render_logits_chunk,
        _compute_probs_loss_cce_chunk,
        _compute_probs_loss_bce_chunk,
        _calculate_module_param_grads_chunk,
        _backprop_error_to_hidden_chunk,
        _calculate_chunk_temp_gradients,
        _clip_partial_gradients,
        _gather_and_permute_grad_hidden_activations,
        _aggregate_register_reduce,
        _aggregate_register_reduce_from_compute,
        _aggregate_local_reduce,
        _aggregate_local_reduce_from_compute,
        _clip_intermediate_grad,
        _reduce_k_fan_in_and_clip,
        _reduce_k_fan_in_and_clip_from_compute,
        _narrow_to_storage,
        _stabilize_and_reduce_grad_hidden_activations,
        _backprop_shared_weights_chunk,
        _backprop_shared_biases_chunk,
        _clip_shared_gradients_chunk,
        _normalize_gradients,
        _adam_update,
        _clamp_temperatures,
        _transpose_matvec_masked_simd_major,
        _elementwise_add,
    )
}
assert len(KERNEL_REGISTRY) == 26, "Duplicate kernel_name in registry"
