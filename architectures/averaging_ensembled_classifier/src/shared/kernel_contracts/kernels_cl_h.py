"""Backend-neutral kernel contracts — machine-readable mirror of kernels.cl.h.

Each ``KernelContract`` is a frozen dataclass carrying the complete
interface specification for one kernel, transcribed from the authoritative
``kernels.cl.h`` header.  The Policy tier uses these for plan-construction-
time validation without requiring any backend.

Section structure mirrors kernels.cl.h:
  §1  Type infrastructure & constants
  §2  Act Phase (Nodes 4–7)
  §3  Learn Phase I — Gradient Generation & Clipping (Nodes 8–11)
  §4  Learn Phase II — Aggregation, Reduction & Specialised Processing
      (Nodes 13, Recursive Engine, Node 16)
  §5  Learn Phase III — Streaming Shared-Layer Backprop (Nodes 17–19)
  §6  Learn Phase IV–V — Finalisation & Parameter Updates (Nodes 21, 24–25)
  §7  Experimental kernels
  §8  Registry
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal


# =====================================================================
# §1  Type Infrastructure
# =====================================================================

@dataclass(frozen=True)
class Placement:
    """Partial-renderer placement strategy (CONTRACT.md §3.5)."""

    strategy: Literal[
        "grid_mod_cls",
        "grid_mod_cls_batch",
        "linear_batch",
        "linear_generic",
    ]
    keys: tuple[str, ...]


@dataclass(frozen=True)
class BufferParam:
    """Buffer parameter specification (CONTRACT.md §2.2, §3).

    For ``LOCAL`` scope, only *alloc* is meaningful — *shape*, *role*,
    *padding*, *init*, *placement*, *conditional_on* are inapplicable
    (CONTRACT §3.1).
    """

    name: str
    flow: Literal["src", "dest", "update", "sync"]
    scope: Literal["GLOBAL", "LOCAL", "GLOBAL_CONST", "DEVICE_CONST"]
    # ── Global-scope (CONTRACT §3.2) ──────────────────────────────
    shape: tuple[str, ...] = ()
    role: Literal["storage", "compute", "state"] | None = None
    padding: Literal["CACHE", "SIMD", "UNPADDED"] = "UNPADDED"
    init: Literal["ZERO_REQUIRED", "ZERO_REQUIRED_ADDITIVE", "NOT_REQUIRED"] = "NOT_REQUIRED"
    placement: Placement | None = None
    conditional_on: str | None = None
    # ── Local-scope (CONTRACT §3.1) ───────────────────────────────
    alloc: str | None = None


@dataclass(frozen=True)
class ScalarParam:
    """Scalar parameter specification (CONTRACT.md §2.3, §2.4)."""

    name: str
    flow: Literal["src", "out"]
    number_type: Literal["NATURAL", "INTEGER", "REAL", "FLAG"]


Param = BufferParam | ScalarParam


@dataclass(frozen=True)
class KernelContract:
    """Backend-neutral kernel interface contract (CONTRACT.md §4, ADR-007).

    *params* preserves the C signature ordering for backend binding.
    """

    name: str
    idempotency: Literal[
        "Strictly Idempotent",
        "Associatively Non-Idempotent",
        "Fundamentally Non-Idempotent (Stateful)",
    ]
    params: tuple[Param, ...]
    sync: str | None = None
    bifurcation_peer: str | None = None
    precision_variant_of: str | None = None

    # ── Convenience accessors ─────────────────────────────────────

    @property
    def buffers(self) -> tuple[BufferParam, ...]:
        """All buffer parameters in signature order."""
        return tuple(p for p in self.params if isinstance(p, BufferParam))

    @property
    def scalars(self) -> tuple[ScalarParam, ...]:
        """All scalar parameters in signature order."""
        return tuple(p for p in self.params if isinstance(p, ScalarParam))

    @property
    def local_buffers(self) -> tuple[BufferParam, ...]:
        """LOCAL-scope buffer parameters."""
        return tuple(b for b in self.buffers if b.scope == "LOCAL")

    @property
    def global_buffers(self) -> tuple[BufferParam, ...]:
        """Non-LOCAL buffer parameters."""
        return tuple(b for b in self.buffers if b.scope != "LOCAL")

    @property
    def dest_buffers(self) -> tuple[BufferParam, ...]:
        """Write-only global buffer parameters."""
        return tuple(
            b for b in self.global_buffers if b.flow == "dest"
        )


# ── Shared constants (mirrors kernels.cl.h §Common Constants) ────

PROBLEM_TYPE_CCE: int = 0
PROBLEM_TYPE_BCE: int = 1
AGG_MODE_SUM: int = 0
AGG_MODE_AVERAGE: int = 1
SENTINEL_ABSENT_PARTIAL: int = 0xFFFFFFFF

# ── Reusable placement instances ─────────────────────────────────

_PLC_MOD_CLS = Placement(
    "grid_mod_cls",
    ("src_scalar_NATURAL_flat_tile_index",),
)
_PLC_MOD_CLS_BATCH = Placement(
    "grid_mod_cls_batch",
    ("src_scalar_NATURAL_flat_tile_index",
     "src_scalar_NATURAL_batch_chunk_index"),
)


# =====================================================================
# §2  Act Phase — Nodes 4–7
# =====================================================================

# --- Node 4: Shared Layer Forward Pass --------------------------------

forward_pass = KernelContract(
    name="forward_pass",
    idempotency="Strictly Idempotent",
    sync="Streamable",
    params=(
        BufferParam(
            "update_buffer_LOCAL_simd_tile", "update", "LOCAL",
            alloc="(SIMD_WIDTH + SIMD_WIDTH * SIMD_WIDTH)"
                  " * sizeof(COMPUTE_TYPE)"),
        BufferParam(
            "src_buffer_GLOBAL_input", "src", "GLOBAL",
            shape=("total_batch_count", "padded_input_count"),
            role="storage", padding="CACHE"),
        BufferParam(
            "src_buffer_GLOBAL_sample_mask", "src", "GLOBAL",
            shape=("(total_batch_count + 31) // 32",)),
        BufferParam(
            "src_buffer_GLOBAL_CONST_weights_shared_simd_major",
            "src", "GLOBAL_CONST",
            shape=("padded_hidden_count // SIMD_WIDTH",
                   "padded_input_count", "SIMD_WIDTH"),
            role="state", padding="SIMD"),
        BufferParam(
            "src_buffer_GLOBAL_CONST_biases_shared", "src", "GLOBAL_CONST",
            shape=("padded_hidden_count",),
            role="state", padding="SIMD"),
        BufferParam(
            "dest_buffer_GLOBAL_hidden_activations", "dest", "GLOBAL",
            shape=("total_batch_count", "padded_hidden_count"),
            role="storage", padding="CACHE"),
        BufferParam(
            "dest_buffer_GLOBAL_hidden_mask", "dest", "GLOBAL",
            shape=("total_batch_count", "padded_hidden_count"),
            role="storage", padding="CACHE",
            conditional_on="out_scalar_FLAG_produce_hidden_mask"),
        ScalarParam("out_scalar_FLAG_produce_hidden_mask", "out", "FLAG"),
        ScalarParam("src_scalar_NATURAL_batch_chunk_offset", "src", "NATURAL"),
        ScalarParam("src_scalar_NATURAL_batch_chunk_count", "src", "NATURAL"),
        ScalarParam("src_scalar_NATURAL_total_batch_count", "src", "NATURAL"),
        ScalarParam("src_scalar_NATURAL_input_count", "src", "NATURAL"),
        ScalarParam("src_scalar_NATURAL_padded_input_count", "src", "NATURAL"),
        ScalarParam(
            "src_scalar_NATURAL_padded_hidden_count", "src", "NATURAL"),
    ),
)

# --- Node 5: Module Layer Logit Rendering -----------------------------

render_logits_chunk = KernelContract(
    name="render_logits_chunk",
    idempotency="Strictly Idempotent",
    sync="Slice Renderer",
    params=(
        BufferParam(
            "src_buffer_GLOBAL_hidden_activations", "src", "GLOBAL",
            shape=("total_batch_count", "padded_hidden_count"),
            role="storage", padding="CACHE"),
        BufferParam(
            "src_buffer_GLOBAL_hidden_mask", "src", "GLOBAL",
            shape=("total_batch_count", "padded_hidden_count"),
            role="storage", padding="CACHE",
            conditional_on="src_scalar_FLAG_use_explicit_hidden_mask"),
        # NOTE: scalar interleaved between buffers in the C signature.
        ScalarParam(
            "src_scalar_FLAG_use_explicit_hidden_mask", "src", "FLAG"),
        BufferParam(
            "src_buffer_GLOBAL_sample_mask", "src", "GLOBAL",
            shape=("(total_batch_count + 31) // 32",)),
        BufferParam(
            "src_buffer_GLOBAL_CONST_weights_module", "src", "GLOBAL_CONST",
            shape=("total_modules_count", "padded_hidden_count",
                   "padded_total_output_class_count"),
            role="state", padding="CACHE"),
        BufferParam(
            "src_buffer_GLOBAL_CONST_biases_module", "src", "GLOBAL_CONST",
            shape=("total_modules_count",
                   "padded_total_output_class_count"),
            role="state", padding="SIMD"),
        BufferParam(
            "dest_buffer_GLOBAL_logits", "dest", "GLOBAL",
            shape=("total_modules_count", "total_batch_count",
                   "padded_total_output_class_count"),
            role="storage", padding="CACHE"),
        ScalarParam(
            "src_scalar_NATURAL_batch_chunk_offset", "src", "NATURAL"),
        ScalarParam(
            "src_scalar_NATURAL_batch_chunk_count", "src", "NATURAL"),
        ScalarParam(
            "src_scalar_NATURAL_module_chunk_offset", "src", "NATURAL"),
        ScalarParam(
            "src_scalar_NATURAL_module_chunk_count", "src", "NATURAL"),
        ScalarParam(
            "src_scalar_NATURAL_class_chunk_offset", "src", "NATURAL"),
        ScalarParam(
            "src_scalar_NATURAL_class_chunk_count", "src", "NATURAL"),
        ScalarParam(
            "src_scalar_NATURAL_total_batch_count", "src", "NATURAL"),
        ScalarParam("src_scalar_NATURAL_hidden_count", "src", "NATURAL"),
        ScalarParam(
            "src_scalar_NATURAL_padded_hidden_count", "src", "NATURAL"),
        ScalarParam(
            "src_scalar_NATURAL_total_output_class_count", "src", "NATURAL"),
        ScalarParam(
            "src_scalar_NATURAL_padded_total_output_class_count",
            "src", "NATURAL"),
        ScalarParam(
            "src_scalar_NATURAL_total_modules_count", "src", "NATURAL"),
    ),
)

# --- Node 6: CCE Probability & Loss ----------------------------------

compute_probs_loss_cce_chunk = KernelContract(
    name="compute_probs_loss_cce_chunk",
    idempotency="Strictly Idempotent",
    sync="Partial Renderer (probs), Conditional Writer (loss)",
    bifurcation_peer="compute_probs_loss_bce_chunk",
    params=(
        BufferParam(
            "src_buffer_GLOBAL_logits", "src", "GLOBAL",
            shape=("total_modules_count", "total_batch_count",
                   "padded_total_output_class_count"),
            role="storage", padding="CACHE"),
        BufferParam(
            "src_buffer_GLOBAL_CONST_temps", "src", "GLOBAL_CONST",
            shape=("total_modules_count",),
            role="state"),
        BufferParam(
            "src_buffer_GLOBAL_targets", "src", "GLOBAL",
            shape=("total_batch_count",)),  # int class indices
        BufferParam(
            "src_buffer_GLOBAL_sample_mask", "src", "GLOBAL",
            shape=("(total_batch_count + 31) // 32",)),
        BufferParam(
            "dest_buffer_GLOBAL_partial_probs", "dest", "GLOBAL",
            shape=("total_tile_count", "modules_per_chunk",
                   "total_batch_count", "classes_per_chunk"),
            role="storage",
            placement=_PLC_MOD_CLS),
        BufferParam(
            "dest_buffer_GLOBAL_final_loss", "dest", "GLOBAL",
            shape=("total_modules_count", "total_batch_count"),
            role="compute", init="ZERO_REQUIRED"),
        ScalarParam(
            "src_scalar_NATURAL_flat_tile_index", "src", "NATURAL"),
        ScalarParam(
            "src_scalar_NATURAL_num_class_chunks", "src", "NATURAL"),
        ScalarParam(
            "src_scalar_NATURAL_classes_per_chunk", "src", "NATURAL"),
        ScalarParam(
            "src_scalar_NATURAL_modules_per_chunk", "src", "NATURAL"),
        ScalarParam(
            "src_scalar_NATURAL_total_batch_count", "src", "NATURAL"),
        ScalarParam(
            "src_scalar_NATURAL_total_output_class_count", "src", "NATURAL"),
        ScalarParam(
            "src_scalar_NATURAL_padded_total_output_class_count",
            "src", "NATURAL"),
        ScalarParam(
            "src_scalar_NATURAL_total_modules_count", "src", "NATURAL"),
        ScalarParam(
            "src_scalar_NATURAL_total_tile_count", "src", "NATURAL"),
    ),
)

# --- Node 7: BCE Probability & Loss ----------------------------------

compute_probs_loss_bce_chunk = KernelContract(
    name="compute_probs_loss_bce_chunk",
    idempotency="Strictly Idempotent",
    sync="Dual Partial Renderer",
    bifurcation_peer="compute_probs_loss_cce_chunk",
    params=(
        BufferParam(
            "src_buffer_GLOBAL_logits", "src", "GLOBAL",
            shape=("total_modules_count", "total_batch_count",
                   "padded_total_output_class_count"),
            role="storage", padding="CACHE"),
        BufferParam(
            "src_buffer_GLOBAL_CONST_temps", "src", "GLOBAL_CONST",
            shape=("total_modules_count",),
            role="state"),
        BufferParam(
            "src_buffer_GLOBAL_targets", "src", "GLOBAL",
            shape=("total_batch_count",
                   "padded_total_output_class_count"),
            role="storage", padding="CACHE"),  # multi-hot
        BufferParam(
            "src_buffer_GLOBAL_sample_mask", "src", "GLOBAL",
            shape=("(total_batch_count + 31) // 32",)),
        BufferParam(
            "dest_buffer_GLOBAL_partial_probs", "dest", "GLOBAL",
            shape=("total_tile_count", "modules_per_chunk",
                   "total_batch_count", "classes_per_chunk"),
            role="storage",
            placement=_PLC_MOD_CLS),
        BufferParam(
            "dest_buffer_GLOBAL_partial_loss", "dest", "GLOBAL",
            shape=("total_tile_count", "modules_per_chunk",
                   "total_batch_count"),
            role="compute",
            placement=_PLC_MOD_CLS),
        ScalarParam(
            "src_scalar_NATURAL_flat_tile_index", "src", "NATURAL"),
        ScalarParam(
            "src_scalar_NATURAL_num_class_chunks", "src", "NATURAL"),
        ScalarParam(
            "src_scalar_NATURAL_classes_per_chunk", "src", "NATURAL"),
        ScalarParam(
            "src_scalar_NATURAL_modules_per_chunk", "src", "NATURAL"),
        ScalarParam(
            "src_scalar_NATURAL_total_batch_count", "src", "NATURAL"),
        ScalarParam(
            "src_scalar_NATURAL_total_output_class_count", "src", "NATURAL"),
        ScalarParam(
            "src_scalar_NATURAL_padded_total_output_class_count",
            "src", "NATURAL"),
        ScalarParam(
            "src_scalar_NATURAL_total_modules_count", "src", "NATURAL"),
        ScalarParam(
            "src_scalar_NATURAL_total_tile_count", "src", "NATURAL"),
    ),
)


# =====================================================================
# §3  Learn Phase I — Gradient Generation & Clipping (Nodes 8–11)
# =====================================================================

# --- Node 8: Module Parameter Gradients -------------------------------

calculate_module_param_grads_chunk = KernelContract(
    name="calculate_module_param_grads_chunk",
    idempotency="Strictly Idempotent",
    sync="Dual Partial Renderer",
    params=(
        BufferParam(
            "update_buffer_LOCAL_reduction_tile", "update", "LOCAL",
            alloc="get_local_size(0) * sizeof(COMPUTE_TYPE)"),
        BufferParam(
            "src_buffer_GLOBAL_hidden_activations", "src", "GLOBAL",
            shape=("total_batch_count", "padded_hidden_count"),
            role="storage", padding="CACHE"),
        BufferParam(
            "src_buffer_GLOBAL_partial_probs", "src", "GLOBAL",
            shape=("total_tile_count", "modules_per_chunk",
                   "total_batch_count", "classes_per_chunk"),
            role="storage"),
        BufferParam(  # void* — flag-conditional type
            "src_buffer_GLOBAL_targets", "src", "GLOBAL"),
        BufferParam(
            "src_buffer_GLOBAL_sample_mask", "src", "GLOBAL",
            shape=("(total_batch_count + 31) // 32",)),
        BufferParam(
            "src_buffer_GLOBAL_CONST_temps", "src", "GLOBAL_CONST",
            shape=("total_modules_count",),
            role="state"),
        BufferParam(
            "dest_buffer_GLOBAL_partial_grad_weights_module",
            "dest", "GLOBAL",
            shape=("total_tile_count", "num_batch_chunks",
                   "modules_per_chunk", "padded_hidden_count",
                   "padded_total_output_class_count"),
            role="storage", padding="CACHE", init="ZERO_REQUIRED",
            placement=_PLC_MOD_CLS_BATCH),
        BufferParam(
            "dest_buffer_GLOBAL_partial_grad_biases_module",
            "dest", "GLOBAL",
            shape=("total_tile_count", "num_batch_chunks",
                   "modules_per_chunk",
                   "padded_total_output_class_count"),
            role="storage", padding="SIMD", init="ZERO_REQUIRED",
            placement=_PLC_MOD_CLS_BATCH),
        ScalarParam("src_scalar_FLAG_problem_type", "src", "FLAG"),
        ScalarParam(
            "src_scalar_NATURAL_flat_tile_index", "src", "NATURAL"),
        ScalarParam(
            "src_scalar_NATURAL_batch_chunk_index", "src", "NATURAL"),
        ScalarParam(
            "src_scalar_NATURAL_batch_chunk_offset", "src", "NATURAL"),
        ScalarParam(
            "src_scalar_NATURAL_batch_chunk_count", "src", "NATURAL"),
        ScalarParam(
            "src_scalar_NATURAL_num_batch_chunks", "src", "NATURAL"),
        ScalarParam(
            "src_scalar_NATURAL_num_class_chunks", "src", "NATURAL"),
        ScalarParam(
            "src_scalar_NATURAL_classes_per_chunk", "src", "NATURAL"),
        ScalarParam(
            "src_scalar_NATURAL_modules_per_chunk", "src", "NATURAL"),
        ScalarParam(
            "src_scalar_NATURAL_total_batch_count", "src", "NATURAL"),
        ScalarParam("src_scalar_NATURAL_hidden_count", "src", "NATURAL"),
        ScalarParam(
            "src_scalar_NATURAL_padded_hidden_count", "src", "NATURAL"),
        ScalarParam(
            "src_scalar_NATURAL_total_output_class_count", "src", "NATURAL"),
        ScalarParam(
            "src_scalar_NATURAL_padded_total_output_class_count",
            "src", "NATURAL"),
        ScalarParam(
            "src_scalar_NATURAL_total_modules_count", "src", "NATURAL"),
        ScalarParam(
            "src_scalar_NATURAL_total_tile_count", "src", "NATURAL"),
    ),
)

# --- Node 9: Hidden Layer Error Backpropagation -----------------------

backprop_error_to_hidden_chunk = KernelContract(
    name="backprop_error_to_hidden_chunk",
    idempotency="Strictly Idempotent",
    sync="Partial Renderer (monolithic intermediate)",
    params=(
        BufferParam(
            "src_buffer_GLOBAL_partial_probs", "src", "GLOBAL",
            shape=("total_tile_count", "modules_per_chunk",
                   "total_batch_count", "classes_per_chunk"),
            role="storage"),
        BufferParam(  # void* — flag-conditional type
            "src_buffer_GLOBAL_targets", "src", "GLOBAL"),
        BufferParam(
            "src_buffer_GLOBAL_sample_mask", "src", "GLOBAL",
            shape=("(total_batch_count + 31) // 32",)),
        BufferParam(
            "src_buffer_GLOBAL_CONST_weights_module", "src", "GLOBAL_CONST",
            shape=("total_modules_count", "padded_hidden_count",
                   "padded_total_output_class_count"),
            role="state", padding="CACHE"),
        BufferParam(
            "src_buffer_GLOBAL_CONST_temps", "src", "GLOBAL_CONST",
            shape=("total_modules_count",),
            role="state"),
        BufferParam(
            "dest_buffer_GLOBAL_partial_grad_hidden_activations_aos",
            "dest", "GLOBAL",
            shape=("total_tile_count", "modules_per_chunk",
                   "total_batch_count", "padded_hidden_count"),
            role="storage", padding="CACHE",
            placement=_PLC_MOD_CLS),
        ScalarParam("src_scalar_FLAG_problem_type", "src", "FLAG"),
        ScalarParam(
            "src_scalar_NATURAL_flat_tile_index", "src", "NATURAL"),
        ScalarParam(
            "src_scalar_NATURAL_num_class_chunks", "src", "NATURAL"),
        ScalarParam(
            "src_scalar_NATURAL_classes_per_chunk", "src", "NATURAL"),
        ScalarParam(
            "src_scalar_NATURAL_modules_per_chunk", "src", "NATURAL"),
        ScalarParam(
            "src_scalar_NATURAL_total_batch_count", "src", "NATURAL"),
        ScalarParam("src_scalar_NATURAL_hidden_count", "src", "NATURAL"),
        ScalarParam(
            "src_scalar_NATURAL_padded_hidden_count", "src", "NATURAL"),
        ScalarParam(
            "src_scalar_NATURAL_total_output_class_count", "src", "NATURAL"),
        ScalarParam(
            "src_scalar_NATURAL_padded_total_output_class_count",
            "src", "NATURAL"),
        ScalarParam(
            "src_scalar_NATURAL_total_modules_count", "src", "NATURAL"),
        ScalarParam(
            "src_scalar_NATURAL_total_tile_count", "src", "NATURAL"),
    ),
)

# --- Node 10: Temperature Gradients -----------------------------------

calculate_chunk_temp_gradients = KernelContract(
    name="calculate_chunk_temp_gradients",
    idempotency="Strictly Idempotent",
    sync="Partial Renderer (temperature gradients)",
    params=(
        BufferParam(
            "update_buffer_LOCAL_reduction_tile", "update", "LOCAL",
            alloc="get_local_size(0) * sizeof(COMPUTE_TYPE)"),
        BufferParam(
            "src_buffer_GLOBAL_logits", "src", "GLOBAL",
            shape=("total_modules_count", "total_batch_count",
                   "padded_total_output_class_count"),
            role="storage", padding="CACHE"),
        BufferParam(
            "src_buffer_GLOBAL_partial_probs", "src", "GLOBAL",
            shape=("total_tile_count", "modules_per_chunk",
                   "total_batch_count", "classes_per_chunk"),
            role="storage"),
        BufferParam(  # void* — flag-conditional type
            "src_buffer_GLOBAL_targets", "src", "GLOBAL"),
        BufferParam(
            "src_buffer_GLOBAL_sample_mask", "src", "GLOBAL",
            shape=("(total_batch_count + 31) // 32",)),
        BufferParam(
            "src_buffer_GLOBAL_CONST_temps", "src", "GLOBAL_CONST",
            shape=("total_modules_count",),
            role="state"),
        BufferParam(
            "dest_buffer_GLOBAL_partial_grad_temps", "dest", "GLOBAL",
            shape=("total_tile_count", "modules_per_chunk"),
            role="storage",
            placement=_PLC_MOD_CLS),
        ScalarParam("src_scalar_FLAG_problem_type", "src", "FLAG"),
        ScalarParam(
            "src_scalar_NATURAL_flat_tile_index", "src", "NATURAL"),
        ScalarParam(
            "src_scalar_NATURAL_num_class_chunks", "src", "NATURAL"),
        ScalarParam(
            "src_scalar_NATURAL_classes_per_chunk", "src", "NATURAL"),
        ScalarParam(
            "src_scalar_NATURAL_modules_per_chunk", "src", "NATURAL"),
        ScalarParam(
            "src_scalar_NATURAL_total_batch_count", "src", "NATURAL"),
        ScalarParam(
            "src_scalar_NATURAL_total_output_class_count", "src", "NATURAL"),
        ScalarParam(
            "src_scalar_NATURAL_padded_total_output_class_count",
            "src", "NATURAL"),
        ScalarParam(
            "src_scalar_NATURAL_total_modules_count", "src", "NATURAL"),
        ScalarParam(
            "src_scalar_NATURAL_total_tile_count", "src", "NATURAL"),
    ),
)

# --- Node 11: Partial Gradient Clipping -------------------------------

clip_partial_gradients = KernelContract(
    name="clip_partial_gradients",
    idempotency="Strictly Idempotent",
    sync="Utility / Stability Primitive",
    params=(
        BufferParam(
            "update_buffer_LOCAL_reduction_tile", "update", "LOCAL",
            alloc="get_local_size(0) * sizeof(COMPUTE_TYPE)"),
        # --- src (post batch-chunk reduction, so no num_batch_chunks dim) ---
        BufferParam(
            "src_buffer_GLOBAL_partial_grad_weights_module", "src", "GLOBAL",
            shape=("total_tile_count", "modules_per_chunk",
                   "padded_hidden_count",
                   "padded_total_output_class_count"),
            role="storage", padding="CACHE"),
        BufferParam(
            "src_buffer_GLOBAL_partial_grad_biases_module", "src", "GLOBAL",
            shape=("total_tile_count", "modules_per_chunk",
                   "padded_total_output_class_count"),
            role="storage", padding="SIMD"),
        BufferParam(
            "src_buffer_GLOBAL_partial_grad_temps", "src", "GLOBAL",
            shape=("total_tile_count", "modules_per_chunk"),
            role="storage"),
        BufferParam(
            "src_buffer_GLOBAL_partial_grad_hidden_activations_aos",
            "src", "GLOBAL",
            shape=("total_tile_count", "modules_per_chunk",
                   "total_batch_count", "padded_hidden_count"),
            role="storage", padding="CACHE"),
        BufferParam(
            "src_buffer_GLOBAL_CONST_clipping_threshold_per_item",
            "src", "GLOBAL_CONST",
            shape=("total_tile_count",),
            role="compute",
            conditional_on="src_scalar_FLAG_use_per_item_norm"),
        # --- dest ---
        BufferParam(
            "dest_buffer_GLOBAL_clipped_partial_grad_weights_module",
            "dest", "GLOBAL",
            shape=("total_tile_count", "modules_per_chunk",
                   "padded_hidden_count",
                   "padded_total_output_class_count"),
            role="storage", padding="CACHE",
            placement=_PLC_MOD_CLS),
        BufferParam(
            "dest_buffer_GLOBAL_clipped_partial_grad_biases_module",
            "dest", "GLOBAL",
            shape=("total_tile_count", "modules_per_chunk",
                   "padded_total_output_class_count"),
            role="storage", padding="SIMD",
            placement=_PLC_MOD_CLS),
        BufferParam(
            "dest_buffer_GLOBAL_clipped_partial_grad_temps",
            "dest", "GLOBAL",
            shape=("total_tile_count", "modules_per_chunk"),
            role="storage",
            placement=_PLC_MOD_CLS),
        BufferParam(
            "dest_buffer_GLOBAL_clipped_partial_grad_hidden_activations_aos",
            "dest", "GLOBAL",
            shape=("total_tile_count", "modules_per_chunk",
                   "total_batch_count", "padded_hidden_count"),
            role="storage", padding="CACHE",
            placement=_PLC_MOD_CLS),
        # --- scalars ---
        ScalarParam(
            "src_scalar_FLAG_use_per_item_norm", "src", "FLAG"),
        ScalarParam(
            "src_scalar_REAL_clipping_threshold_t_pre", "src", "REAL"),
        ScalarParam("src_scalar_REAL_epsilon", "src", "REAL"),
        ScalarParam(
            "src_scalar_NATURAL_flat_tile_index", "src", "NATURAL"),
        ScalarParam(
            "src_scalar_NATURAL_num_class_chunks", "src", "NATURAL"),
        ScalarParam(
            "src_scalar_NATURAL_classes_per_chunk", "src", "NATURAL"),
        ScalarParam(
            "src_scalar_NATURAL_modules_per_chunk", "src", "NATURAL"),
        ScalarParam(
            "src_scalar_NATURAL_total_batch_count", "src", "NATURAL"),
        ScalarParam(
            "src_scalar_NATURAL_padded_hidden_count", "src", "NATURAL"),
        ScalarParam(
            "src_scalar_NATURAL_padded_total_output_class_count",
            "src", "NATURAL"),
        ScalarParam(
            "src_scalar_NATURAL_total_tile_count", "src", "NATURAL"),
    ),
)


# =====================================================================
# §4  Learn Phase II — Aggregation, Reduction & Specialised Processing
#     (Nodes 13, Recursive Engine, Node 16)
# =====================================================================

# --- Node 13: Gradient Gather & Permutation ---------------------------

gather_and_permute_grad_hidden_activations = KernelContract(
    name="gather_and_permute_grad_hidden_activations",
    idempotency="Strictly Idempotent",
    sync="Global Barrier",
    params=(
        BufferParam(
            "src_buffer_GLOBAL_clipped_partial_grad_hidden_activations_aos",
            "src", "GLOBAL",
            shape=("total_tile_count", "modules_per_chunk",
                   "total_batch_count", "padded_hidden_count"),
            role="storage", padding="CACHE"),
        BufferParam(
            "dest_buffer_GLOBAL_clipped_grad_hidden_activations_permuted_soa",
            "dest", "GLOBAL",
            shape=("total_batch_count * padded_hidden_count",
                   "padded_total_modules_count"),
            role="storage", padding="CACHE", init="ZERO_REQUIRED"),
        ScalarParam(
            "src_scalar_NATURAL_total_batch_count", "src", "NATURAL"),
        ScalarParam("src_scalar_NATURAL_hidden_count", "src", "NATURAL"),
        ScalarParam(
            "src_scalar_NATURAL_padded_hidden_count", "src", "NATURAL"),
        ScalarParam(
            "src_scalar_NATURAL_total_modules_count", "src", "NATURAL"),
        ScalarParam(
            "src_scalar_NATURAL_padded_total_modules_count",
            "src", "NATURAL"),
        ScalarParam(
            "src_scalar_NATURAL_num_module_chunks", "src", "NATURAL"),
        ScalarParam(
            "src_scalar_NATURAL_modules_per_chunk", "src", "NATURAL"),
        ScalarParam(
            "src_scalar_NATURAL_num_class_chunks", "src", "NATURAL"),
        ScalarParam(
            "src_scalar_NATURAL_total_tile_count", "src", "NATURAL"),
    ),
)

# --- Recursive Clip-Aggregation Engine --------------------------------
#
# Single-stage: aggregate_register_reduce / aggregate_local_reduce
#               + clip_intermediate_grad
# Multi-stage:  reduce_k_fan_in_and_clip
# Each has storage-entry and compute-entry precision variants (ADR-026).

aggregate_register_reduce = KernelContract(
    name="aggregate_register_reduce",
    idempotency="Associatively Non-Idempotent",
    sync="Reduction Engine Stage",
    params=(
        BufferParam(
            "src_buffer_GLOBAL_partial_collection", "src", "GLOBAL",
            role="storage"),
        BufferParam(
            "src_buffer_GLOBAL_CONST_partial_offset_list",
            "src", "GLOBAL_CONST",
            shape=("partial_offset_list_count",)),
        BufferParam(
            "dest_buffer_GLOBAL_partial", "dest", "GLOBAL",
            shape=("partial_width",),
            role="compute"),
        ScalarParam(
            "src_scalar_NATURAL_partial_offset_list_count",
            "src", "NATURAL"),
        ScalarParam("src_scalar_NATURAL_partial_width", "src", "NATURAL"),
        ScalarParam("src_scalar_FLAG_operation_type", "src", "FLAG"),
    ),
)

aggregate_register_reduce_from_compute = KernelContract(
    name="aggregate_register_reduce_from_compute",
    idempotency="Associatively Non-Idempotent",
    sync="Reduction Engine Stage",
    precision_variant_of="aggregate_register_reduce",
    params=(
        BufferParam(
            "src_buffer_GLOBAL_partial_collection", "src", "GLOBAL",
            role="compute"),
        BufferParam(
            "src_buffer_GLOBAL_CONST_partial_offset_list",
            "src", "GLOBAL_CONST",
            shape=("partial_offset_list_count",)),
        BufferParam(
            "dest_buffer_GLOBAL_partial", "dest", "GLOBAL",
            shape=("partial_width",),
            role="compute"),
        ScalarParam(
            "src_scalar_NATURAL_partial_offset_list_count",
            "src", "NATURAL"),
        ScalarParam("src_scalar_NATURAL_partial_width", "src", "NATURAL"),
        ScalarParam("src_scalar_FLAG_operation_type", "src", "FLAG"),
    ),
)

aggregate_local_reduce = KernelContract(
    name="aggregate_local_reduce",
    idempotency="Associatively Non-Idempotent",
    sync="Reduction Engine Stage / Work-group Parallel",
    params=(
        BufferParam(
            "update_buffer_LOCAL_reduction_tile", "update", "LOCAL",
            alloc="get_local_size(0) * sizeof(COMPUTE_TYPE)"),
        BufferParam(
            "src_buffer_GLOBAL_partial_collection", "src", "GLOBAL",
            role="storage"),
        BufferParam(
            "src_buffer_GLOBAL_CONST_partial_offset_list",
            "src", "GLOBAL_CONST",
            shape=("partial_offset_list_count",)),
        BufferParam(
            "dest_buffer_GLOBAL_partial", "dest", "GLOBAL",
            shape=("partial_width",),
            role="compute"),
        ScalarParam(
            "src_scalar_NATURAL_partial_offset_list_count",
            "src", "NATURAL"),
        ScalarParam("src_scalar_NATURAL_partial_width", "src", "NATURAL"),
        ScalarParam("src_scalar_FLAG_operation_type", "src", "FLAG"),
    ),
)

aggregate_local_reduce_from_compute = KernelContract(
    name="aggregate_local_reduce_from_compute",
    idempotency="Associatively Non-Idempotent",
    sync="Reduction Engine Stage / Work-group Parallel",
    precision_variant_of="aggregate_local_reduce",
    params=(
        BufferParam(
            "update_buffer_LOCAL_reduction_tile", "update", "LOCAL",
            alloc="get_local_size(0) * sizeof(COMPUTE_TYPE)"),
        BufferParam(
            "src_buffer_GLOBAL_partial_collection", "src", "GLOBAL",
            role="compute"),
        BufferParam(
            "src_buffer_GLOBAL_CONST_partial_offset_list",
            "src", "GLOBAL_CONST",
            shape=("partial_offset_list_count",)),
        BufferParam(
            "dest_buffer_GLOBAL_partial", "dest", "GLOBAL",
            shape=("partial_width",),
            role="compute"),
        ScalarParam(
            "src_scalar_NATURAL_partial_offset_list_count",
            "src", "NATURAL"),
        ScalarParam("src_scalar_NATURAL_partial_width", "src", "NATURAL"),
        ScalarParam("src_scalar_FLAG_operation_type", "src", "FLAG"),
    ),
)

clip_intermediate_grad = KernelContract(
    name="clip_intermediate_grad",
    idempotency="Associatively Non-Idempotent",
    sync="Reduction Engine Stage Clip Primitive",
    params=(
        BufferParam(
            "update_buffer_LOCAL_reduction_tile", "update", "LOCAL",
            alloc="get_local_size(0) * sizeof(COMPUTE_TYPE)"),
        BufferParam(
            "update_buffer_GLOBAL_intermediate_grad", "update", "GLOBAL",
            shape=("parameter_count",),
            role="compute"),
        ScalarParam(
            "src_scalar_REAL_clipping_threshold_t_j", "src", "REAL"),
        ScalarParam("src_scalar_REAL_epsilon", "src", "REAL"),
        ScalarParam(
            "src_scalar_NATURAL_parameter_count", "src", "NATURAL"),
    ),
)

reduce_k_fan_in_and_clip = KernelContract(
    name="reduce_k_fan_in_and_clip",
    idempotency="Associatively Non-Idempotent",
    sync="Reduction Engine Stage",
    params=(
        BufferParam(
            "update_buffer_LOCAL_reduction_tile", "update", "LOCAL",
            alloc="get_local_size(0) * sizeof(COMPUTE_TYPE)"),
        BufferParam(
            "src_buffer_GLOBAL_partial_collection", "src", "GLOBAL",
            role="storage"),
        BufferParam(
            "src_buffer_GLOBAL_CONST_offset_list_flat",
            "src", "GLOBAL_CONST",
            shape=("node_count * fan_in",)),
        BufferParam(
            "dest_buffer_GLOBAL_stage_partial", "dest", "GLOBAL",
            shape=("node_count * partial_width",),
            role="compute"),
        ScalarParam("src_scalar_NATURAL_fan_in", "src", "NATURAL"),
        ScalarParam("src_scalar_NATURAL_node_count", "src", "NATURAL"),
        ScalarParam("src_scalar_NATURAL_partial_width", "src", "NATURAL"),
        ScalarParam(
            "src_scalar_REAL_clipping_threshold_t_j", "src", "REAL"),
        ScalarParam("src_scalar_REAL_epsilon", "src", "REAL"),
    ),
)

reduce_k_fan_in_and_clip_from_compute = KernelContract(
    name="reduce_k_fan_in_and_clip_from_compute",
    idempotency="Associatively Non-Idempotent",
    sync="Reduction Engine Stage",
    precision_variant_of="reduce_k_fan_in_and_clip",
    params=(
        BufferParam(
            "update_buffer_LOCAL_reduction_tile", "update", "LOCAL",
            alloc="get_local_size(0) * sizeof(COMPUTE_TYPE)"),
        BufferParam(
            "src_buffer_GLOBAL_partial_collection", "src", "GLOBAL",
            role="compute"),
        BufferParam(
            "src_buffer_GLOBAL_CONST_offset_list_flat",
            "src", "GLOBAL_CONST",
            shape=("node_count * fan_in",)),
        BufferParam(
            "dest_buffer_GLOBAL_stage_partial", "dest", "GLOBAL",
            shape=("node_count * partial_width",),
            role="compute"),
        ScalarParam("src_scalar_NATURAL_fan_in", "src", "NATURAL"),
        ScalarParam("src_scalar_NATURAL_node_count", "src", "NATURAL"),
        ScalarParam("src_scalar_NATURAL_partial_width", "src", "NATURAL"),
        ScalarParam(
            "src_scalar_REAL_clipping_threshold_t_j", "src", "REAL"),
        ScalarParam("src_scalar_REAL_epsilon", "src", "REAL"),
    ),
)

# --- Precision Bridge -------------------------------------------------

narrow_to_storage = KernelContract(
    name="narrow_to_storage",
    idempotency="Strictly Idempotent",
    sync="Utility",
    params=(
        BufferParam(
            "src_buffer_GLOBAL_input", "src", "GLOBAL",
            shape=("element_count",),
            role="compute"),
        BufferParam(
            "dest_buffer_GLOBAL_output", "dest", "GLOBAL",
            shape=("element_count",),
            role="storage"),
        ScalarParam(
            "src_scalar_NATURAL_element_count", "src", "NATURAL"),
    ),
)

# --- Node 16: Specialised Grad_H Reduction ----------------------------

stabilize_and_reduce_grad_hidden_activations = KernelContract(
    name="stabilize_and_reduce_grad_hidden_activations",
    idempotency="Associatively Non-Idempotent",
    sync="Specialised Reduction Kernel / Global Barrier",
    params=(
        BufferParam(
            "update_buffer_LOCAL_reduction_tile", "update", "LOCAL",
            alloc="get_local_size(0) * sizeof(COMPUTE_TYPE)"),
        BufferParam(
            "src_buffer_GLOBAL_clipped_grad_hidden_activations_permuted_soa",
            "src", "GLOBAL",
            shape=("total_batch_count * padded_hidden_count",
                   "padded_total_modules_count"),
            role="storage", padding="CACHE"),
        BufferParam(
            "dest_buffer_GLOBAL_summed_grad_hidden_activations",
            "dest", "GLOBAL",
            shape=("total_batch_count * padded_hidden_count",),
            role="compute"),
        BufferParam(
            "src_buffer_GLOBAL_CONST_clipping_threshold_per_stage",
            "src", "GLOBAL_CONST",
            shape=("num_reduction_stages",),
            role="compute"),
        ScalarParam(
            "src_scalar_NATURAL_num_reduction_stages", "src", "NATURAL"),
        ScalarParam(
            "src_scalar_REAL_clipping_threshold_t_pre", "src", "REAL"),
        ScalarParam("src_scalar_REAL_epsilon", "src", "REAL"),
        ScalarParam(
            "src_scalar_NATURAL_total_batch_count", "src", "NATURAL"),
        ScalarParam(
            "src_scalar_NATURAL_padded_hidden_count", "src", "NATURAL"),
        ScalarParam(
            "src_scalar_NATURAL_total_modules_count", "src", "NATURAL"),
        ScalarParam(
            "src_scalar_NATURAL_padded_total_modules_count",
            "src", "NATURAL"),
    ),
)


# =====================================================================
# §5  Learn Phase III — Streaming Shared-Layer Backprop (Nodes 17–19)
# =====================================================================

# --- Node 17: Shared Weight Gradients ---------------------------------

backprop_shared_weights_chunk = KernelContract(
    name="backprop_shared_weights_chunk",
    idempotency="Associatively Non-Idempotent",
    sync="Streamable",
    params=(
        BufferParam(
            "update_buffer_LOCAL_reduction_tile", "update", "LOCAL",
            alloc="get_local_size(0) * sizeof(COMPUTE_TYPE)"),
        BufferParam(
            "src_buffer_GLOBAL_input", "src", "GLOBAL",
            shape=("total_batch_count", "padded_input_count"),
            role="storage", padding="CACHE"),
        BufferParam(
            "src_buffer_GLOBAL_hidden_activations", "src", "GLOBAL",
            shape=("total_batch_count", "padded_hidden_count"),
            role="storage", padding="CACHE"),
        BufferParam(
            "src_buffer_GLOBAL_hidden_mask", "src", "GLOBAL",
            shape=("total_batch_count", "padded_hidden_count"),
            role="storage", padding="CACHE",
            conditional_on="src_scalar_FLAG_use_explicit_hidden_mask"),
        # NOTE: scalar interleaved between buffers in the C signature.
        ScalarParam(
            "src_scalar_FLAG_use_explicit_hidden_mask", "src", "FLAG"),
        BufferParam(
            "src_buffer_GLOBAL_summed_grad_hidden_activations",
            "src", "GLOBAL",
            shape=("final_grad_hidden_activations_total_count",),
            role="compute"),
        BufferParam(
            "src_buffer_GLOBAL_sample_mask", "src", "GLOBAL",
            shape=("(total_batch_count + 31) // 32",)),
        BufferParam(
            "dest_buffer_GLOBAL_partial_grad_weights_shared_simd_major",
            "dest", "GLOBAL",
            shape=("padded_hidden_count // SIMD_WIDTH",
                   "padded_input_count", "SIMD_WIDTH"),
            role="storage", padding="SIMD"),
        ScalarParam(
            "src_scalar_NATURAL_batch_chunk_offset", "src", "NATURAL"),
        ScalarParam(
            "src_scalar_NATURAL_batch_chunk_count", "src", "NATURAL"),
        ScalarParam(
            "src_scalar_NATURAL_total_batch_count", "src", "NATURAL"),
        ScalarParam("src_scalar_NATURAL_input_count", "src", "NATURAL"),
        ScalarParam(
            "src_scalar_NATURAL_padded_input_count", "src", "NATURAL"),
        ScalarParam("src_scalar_NATURAL_hidden_count", "src", "NATURAL"),
        ScalarParam(
            "src_scalar_NATURAL_padded_hidden_count", "src", "NATURAL"),
        ScalarParam(
            "src_scalar_NATURAL_final_grad_hidden_activations_total_count",
            "src", "NATURAL"),
    ),
)

# --- Node 18: Shared Bias Gradients -----------------------------------

backprop_shared_biases_chunk = KernelContract(
    name="backprop_shared_biases_chunk",
    idempotency="Associatively Non-Idempotent",
    sync="Streamable",
    params=(
        BufferParam(
            "update_buffer_LOCAL_reduction_tile", "update", "LOCAL",
            alloc="get_local_size(0) * sizeof(COMPUTE_TYPE)"),
        BufferParam(
            "src_buffer_GLOBAL_hidden_activations", "src", "GLOBAL",
            shape=("total_batch_count", "padded_hidden_count"),
            role="storage", padding="CACHE"),
        BufferParam(
            "src_buffer_GLOBAL_hidden_mask", "src", "GLOBAL",
            shape=("total_batch_count", "padded_hidden_count"),
            role="storage", padding="CACHE",
            conditional_on="src_scalar_FLAG_use_explicit_hidden_mask"),
        ScalarParam(
            "src_scalar_FLAG_use_explicit_hidden_mask", "src", "FLAG"),
        BufferParam(
            "src_buffer_GLOBAL_summed_grad_hidden_activations",
            "src", "GLOBAL",
            shape=("final_grad_hidden_activations_total_count",),
            role="compute"),
        BufferParam(
            "src_buffer_GLOBAL_sample_mask", "src", "GLOBAL",
            shape=("(total_batch_count + 31) // 32",)),
        BufferParam(
            "dest_buffer_GLOBAL_partial_grad_biases_shared",
            "dest", "GLOBAL",
            shape=("padded_hidden_count",),
            role="storage", padding="CACHE"),
        ScalarParam(
            "src_scalar_NATURAL_batch_chunk_offset", "src", "NATURAL"),
        ScalarParam(
            "src_scalar_NATURAL_batch_chunk_count", "src", "NATURAL"),
        ScalarParam(
            "src_scalar_NATURAL_total_batch_count", "src", "NATURAL"),
        ScalarParam("src_scalar_NATURAL_hidden_count", "src", "NATURAL"),
        ScalarParam(
            "src_scalar_NATURAL_padded_hidden_count", "src", "NATURAL"),
        ScalarParam(
            "src_scalar_NATURAL_final_grad_hidden_activations_total_count",
            "src", "NATURAL"),
    ),
)

# --- Node 19: Shared Gradient Clipping --------------------------------

clip_shared_gradients_chunk = KernelContract(
    name="clip_shared_gradients_chunk",
    idempotency="Strictly Idempotent",
    sync="Streamable Utility / Stability Primitive",
    params=(
        BufferParam(
            "update_buffer_LOCAL_reduction_tile", "update", "LOCAL",
            alloc="get_local_size(0) * sizeof(COMPUTE_TYPE)"),
        BufferParam(
            "src_buffer_GLOBAL_partial_grad_weights_shared_simd_major",
            "src", "GLOBAL",
            shape=("weights_parameter_count",),
            role="storage"),
        BufferParam(
            "src_buffer_GLOBAL_partial_grad_biases_shared", "src", "GLOBAL",
            shape=("biases_parameter_count",),
            role="storage"),
        BufferParam(
            "dest_buffer_GLOBAL_clipped_partial_grad_weights_shared"
            "_simd_major",
            "dest", "GLOBAL",
            shape=("num_batch_chunks", "weights_parameter_count"),
            role="storage"),
        BufferParam(
            "dest_buffer_GLOBAL_clipped_partial_grad_biases_shared",
            "dest", "GLOBAL",
            shape=("num_batch_chunks", "biases_parameter_count"),
            role="storage"),
        ScalarParam(
            "src_scalar_REAL_clipping_threshold_t_pre", "src", "REAL"),
        ScalarParam("src_scalar_REAL_epsilon", "src", "REAL"),
        ScalarParam(
            "src_scalar_NATURAL_weights_parameter_count", "src", "NATURAL"),
        ScalarParam(
            "src_scalar_NATURAL_biases_parameter_count", "src", "NATURAL"),
        ScalarParam(
            "out_scalar_NATURAL_weights_write_offset", "out", "NATURAL"),
        ScalarParam(
            "out_scalar_NATURAL_biases_write_offset", "out", "NATURAL"),
        ScalarParam(
            "src_scalar_NATURAL_num_batch_chunks", "src", "NATURAL"),
    ),
)


# =====================================================================
# §6  Learn Phase IV–V — Finalisation & Parameter Updates
#     (Nodes 21, 24–25)
# =====================================================================

# --- Node 21: Gradient Normalisation ----------------------------------

normalize_gradients = KernelContract(
    name="normalize_gradients",
    idempotency="Strictly Idempotent",
    sync="Finaliser Utility / Batch-wide Normaliser",
    params=(
        BufferParam(
            "src_buffer_GLOBAL_summed_grad", "src", "GLOBAL",
            shape=("parameter_count",),
            role="compute"),
        BufferParam(
            "dest_buffer_GLOBAL_final_grad", "dest", "GLOBAL",
            shape=("parameter_count",),
            role="compute"),
        ScalarParam(
            "src_scalar_REAL_effective_batch_size", "src", "REAL"),
        ScalarParam("src_scalar_REAL_epsilon", "src", "REAL"),
        ScalarParam(
            "src_scalar_NATURAL_parameter_count", "src", "NATURAL"),
    ),
)

# --- Node 24: Adam Optimizer Update -----------------------------------

adam_update = KernelContract(
    name="adam_update",
    idempotency="Fundamentally Non-Idempotent (Stateful)",
    sync="Stateful Optimizer Update",
    params=(
        BufferParam(
            "src_buffer_GLOBAL_final_grad", "src", "GLOBAL",
            shape=("parameter_count",),
            role="compute"),
        BufferParam(
            "update_buffer_GLOBAL_parameters", "update", "GLOBAL",
            shape=("total_parameter_count",),
            role="state"),
        BufferParam(
            "update_buffer_GLOBAL_m1", "update", "GLOBAL",
            shape=("total_parameter_count",),
            role="state"),
        BufferParam(
            "update_buffer_GLOBAL_m2", "update", "GLOBAL",
            shape=("total_parameter_count",),
            role="state"),
        ScalarParam("src_scalar_REAL_learning_rate", "src", "REAL"),
        ScalarParam("src_scalar_REAL_beta1_pow_t", "src", "REAL"),
        ScalarParam("src_scalar_REAL_beta2_pow_t", "src", "REAL"),
        ScalarParam("src_scalar_REAL_beta1", "src", "REAL"),
        ScalarParam("src_scalar_REAL_beta2", "src", "REAL"),
        ScalarParam("src_scalar_REAL_epsilon", "src", "REAL"),
        ScalarParam(
            "src_scalar_NATURAL_parameter_offset", "src", "NATURAL"),
        ScalarParam(
            "src_scalar_NATURAL_parameter_count", "src", "NATURAL"),
        ScalarParam(
            "src_scalar_NATURAL_total_parameter_count", "src", "NATURAL"),
    ),
)

# --- Node 25: Temperature Clamping ------------------------------------

clamp_temperatures = KernelContract(
    name="clamp_temperatures",
    idempotency="Fundamentally Non-Idempotent (Stateful)",
    sync="Finaliser Utility",
    params=(
        BufferParam(
            "update_buffer_GLOBAL_temps", "update", "GLOBAL",
            shape=("total_parameter_count",),
            role="state"),
        ScalarParam("src_scalar_REAL_min_value", "src", "REAL"),
        ScalarParam("src_scalar_REAL_max_value", "src", "REAL"),
        ScalarParam(
            "src_scalar_NATURAL_parameter_offset", "src", "NATURAL"),
        ScalarParam(
            "src_scalar_NATURAL_parameter_count", "src", "NATURAL"),
        ScalarParam(
            "src_scalar_NATURAL_total_parameter_count", "src", "NATURAL"),
    ),
)


# =====================================================================
# §7  Experimental Kernels
#
# NOT referenced by canonical DAG node numbering. Integration into the
# execution plan requires Policy-tier extensions and new ADR(s).
# =====================================================================

transpose_matvec_masked_simd_major = KernelContract(
    name="transpose_matvec_masked_simd_major",
    idempotency="Strictly Idempotent",
    sync="Streamable",
    bifurcation_peer="forward_pass",
    params=(
        BufferParam(
            "update_buffer_LOCAL_simd_tile", "update", "LOCAL",
            alloc="(SIMD_WIDTH + SIMD_WIDTH * SIMD_WIDTH)"
                  " * sizeof(COMPUTE_TYPE)"),
        BufferParam(
            "src_buffer_GLOBAL_CONST_weights_shared_simd_major",
            "src", "GLOBAL_CONST",
            shape=("padded_hidden_count // SIMD_WIDTH",
                   "padded_input_count", "SIMD_WIDTH"),
            role="state", padding="SIMD"),
        BufferParam(
            "src_buffer_GLOBAL_grad_hidden_activations", "src", "GLOBAL",
            shape=("total_batch_count", "padded_hidden_count"),
            role="compute"),
        BufferParam(
            "src_buffer_GLOBAL_hidden_activations", "src", "GLOBAL",
            shape=("total_batch_count", "padded_hidden_count"),
            role="storage", padding="CACHE"),
        BufferParam(
            "src_buffer_GLOBAL_hidden_mask", "src", "GLOBAL",
            shape=("total_batch_count", "padded_hidden_count"),
            role="storage", padding="CACHE",
            conditional_on="src_scalar_FLAG_use_explicit_hidden_mask"),
        BufferParam(
            "src_buffer_GLOBAL_sample_mask", "src", "GLOBAL",
            shape=("(total_batch_count + 31) // 32",)),
        BufferParam(
            "dest_buffer_GLOBAL_grad_input", "dest", "GLOBAL",
            shape=("total_batch_count", "padded_input_count"),
            role="storage", padding="CACHE"),
        ScalarParam(
            "src_scalar_FLAG_use_explicit_hidden_mask", "src", "FLAG"),
        ScalarParam(
            "src_scalar_NATURAL_batch_chunk_offset", "src", "NATURAL"),
        ScalarParam(
            "src_scalar_NATURAL_batch_chunk_count", "src", "NATURAL"),
        ScalarParam(
            "src_scalar_NATURAL_total_batch_count", "src", "NATURAL"),
        ScalarParam("src_scalar_NATURAL_input_count", "src", "NATURAL"),
        ScalarParam(
            "src_scalar_NATURAL_padded_input_count", "src", "NATURAL"),
        ScalarParam("src_scalar_NATURAL_hidden_count", "src", "NATURAL"),
        ScalarParam(
            "src_scalar_NATURAL_padded_hidden_count", "src", "NATURAL"),
    ),
)

elementwise_add = KernelContract(
    name="elementwise_add",
    idempotency="Strictly Idempotent",
    sync="Utility",
    params=(
        BufferParam(
            "src_buffer_GLOBAL_input_a", "src", "GLOBAL",
            shape=("element_count",),
            role="compute"),
        BufferParam(
            "src_buffer_GLOBAL_input_b", "src", "GLOBAL",
            shape=("element_count",),
            role="compute"),
        BufferParam(
            "dest_buffer_GLOBAL_output", "dest", "GLOBAL",
            shape=("element_count",),
            role="compute"),
        ScalarParam(
            "src_scalar_NATURAL_element_count", "src", "NATURAL"),
    ),
)


# =====================================================================
# §8  Registry
# =====================================================================

_ALL: tuple[KernelContract, ...] = (
    # Act Phase
    forward_pass,
    render_logits_chunk,
    compute_probs_loss_cce_chunk,
    compute_probs_loss_bce_chunk,
    # Learn Phase I
    calculate_module_param_grads_chunk,
    backprop_error_to_hidden_chunk,
    calculate_chunk_temp_gradients,
    clip_partial_gradients,
    # Learn Phase II
    gather_and_permute_grad_hidden_activations,
    aggregate_register_reduce,
    aggregate_register_reduce_from_compute,
    aggregate_local_reduce,
    aggregate_local_reduce_from_compute,
    clip_intermediate_grad,
    reduce_k_fan_in_and_clip,
    reduce_k_fan_in_and_clip_from_compute,
    narrow_to_storage,
    stabilize_and_reduce_grad_hidden_activations,
    # Learn Phase III
    backprop_shared_weights_chunk,
    backprop_shared_biases_chunk,
    clip_shared_gradients_chunk,
    # Learn Phase IV–V
    normalize_gradients,
    adam_update,
    clamp_temperatures,
    # Experimental
    transpose_matvec_masked_simd_major,
    elementwise_add,
)

REGISTRY: dict[str, KernelContract] = {k.name: k for k in _ALL}
"""Kernel-name → contract mapping for all declared kernels."""
