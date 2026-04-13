# src/shared/kernel_contracts/__init__.py
"""Backend-neutral kernel contracts (ADR-007).

Re-exports all types and contract instances from the authoritative
``kernels_cl_h.py`` header mirror.
"""

# ── Core types ────────────────────────────────────────────────────────
from .kernels_cl_h import (
    Placement,
    BufferParam,
    ScalarParam,
    Param,
    KernelContract,
)

# ── Constants ─────────────────────────────────────────────────────────
from .kernels_cl_h import (
    PROBLEM_TYPE_CCE,
    PROBLEM_TYPE_BCE,
    AGG_MODE_SUM,
    AGG_MODE_AVERAGE,
    SENTINEL_ABSENT_PARTIAL,
)

# ── Contract instances — Act Phase (§2) ──────────────────────────────
from .kernels_cl_h import (
    forward_pass,
    render_logits_chunk,
    compute_probs_loss_cce_chunk,
    compute_probs_loss_bce_chunk,
)

# ── Contract instances — Learn Phase I (§3) ──────────────────────────
from .kernels_cl_h import (
    calculate_module_param_grads_chunk,
    backprop_error_to_hidden_chunk,
    calculate_chunk_temp_gradients,
    clip_partial_gradients,
)

# ── Contract instances — Learn Phase II (§4) ─────────────────────────
from .kernels_cl_h import (
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
)

# ── Contract instances — Learn Phase III (§5) ────────────────────────
from .kernels_cl_h import (
    backprop_shared_weights_chunk,
    backprop_shared_biases_chunk,
    clip_shared_gradients_chunk,
)

# ── Contract instances — Learn Phase IV–V (§6) ──────────────────────
from .kernels_cl_h import (
    normalize_gradients,
    adam_update,
    clamp_temperatures,
)

# ── Contract instances — Experimental (§7) ───────────────────────────
from .kernels_cl_h import (
    transpose_matvec_masked_simd_major,
    elementwise_add,
)

# ── Registry ──────────────────────────────────────────────────────────
from .kernels_cl_h import REGISTRY
