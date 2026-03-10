# kernel_signatures/tests/test_header_contract_conformance.py

"""
Cross-cutting contract conformance tests.

These tests parse ``kernels.cl.h`` and verify structural invariants that span
all signature classes:

  1. **Coverage completeness** — every C kernel has ≥1 Python signature.
  2. **No orphaned signatures** — every Python ``kernel_name`` appears in the header.
  3. **Argument count** — ``len(get_args())`` matches the C parameter count.
  4. **Type-category ordering** — each position's type category matches.
"""

from __future__ import annotations

from typing import Dict, List

import numpy as np
import pytest

from .conftest import (
    BATCH,
    BATCH_CHUNKS,
    CLASSES_NATURAL,
    CLASSES_PADDED,
    CLASSES_PER_CHUNK,
    HIDDEN_NATURAL,
    HIDDEN_PADDED,
    INPUT_PADDED,
    MODULES,
    MODULES_PER_CHUNK,
    NUM_CLASS_CHUNKS,
    NUM_MODULE_CHUNKS,
    PADDED_MODULES,
    TILES,
    WG_SIZE_REDUCTION,
    MockArchConsts,
    MockBufferManager,
    ParamCategory,
    classify_python_arg,
    make_tile,
)

from src.kernel_signatures import (
    ForwardPassSignature,
    RenderLogitsChunkSignature,
    ComputeProbsLossCceChunkSignature,
    ComputeProbsLossBceChunkSignature,
    CalculateModuleParamGradsCceSignature,
    CalculateModuleParamGradsBceSignature,
    BackpropErrorToHiddenChunkCceSignature,
    BackpropErrorToHiddenChunkBceSignature,
    CalculateChunkTempGradientsCceSignature,
    CalculateChunkTempGradientsBceSignature,
    ClipPartialGradientsGlobalNormSignature,
    ClipPartialGradientsPerItemNormSignature,
    GradientHandles,
    GatherAndPermuteGradHiddenActivationsSignature,
    AggregateRegisterReduceSignature,
    AggregateLocalReduceSignature,
    ClipIntermediateGradSignature,
    StabilizeAndReduceGradHiddenActivationsSignature,
    BackpropSharedWeightsChunkSignature,
    BackpropSharedBiasesChunkSignature,
    SharedGradientHandles,
    ClipSharedGradientsChunkSignature,
    NormalizeGradientsSignature,
    AdamParameterGroup,
    AdamUpdateSignature,
    ClampTemperaturesSignature,
)


# =========================================================================
# Signature Factory Functions
# =========================================================================

def _make_forward_pass(bm: MockBufferManager, ac: MockArchConsts):
    return ForwardPassSignature(
        _buffer_mgr=bm,
        _arch_consts=ac,
        in_ref=bm.make_handle((BATCH, INPUT_PADDED)),
        mask_ref=bm.make_handle((BATCH,)),
        w_ref=bm.make_handle((HIDDEN_PADDED // ac.simd_width, INPUT_PADDED, ac.simd_width)),
        b_ref=bm.make_handle((HIDDEN_PADDED,)),
        h_out_ref=bm.make_handle((BATCH, HIDDEN_PADDED)),
        h_mask_out_ref=bm.make_handle((BATCH, HIDDEN_PADDED)),
        batch_chunk_offset=np.uint32(0),
        batch_chunk_count=np.uint32(BATCH),
    )


def _make_render_logits_chunk(bm: MockBufferManager, ac: MockArchConsts):
    return RenderLogitsChunkSignature(
        _buffer_mgr=bm,
        _arch_consts=ac,
        h_ref=bm.make_handle((BATCH, HIDDEN_PADDED)),
        h_mask_ref=bm.make_handle((BATCH, HIDDEN_PADDED)),
        w_ref=bm.make_handle((MODULES, HIDDEN_PADDED, CLASSES_PADDED)),
        b_ref=bm.make_handle((MODULES, CLASSES_PADDED)),
        logit_out_ref=bm.make_handle((MODULES, BATCH, CLASSES_PADDED)),
        batch_chunk_offset=np.uint32(0),
        batch_chunk_count=np.uint32(BATCH),
        module_chunk_offset=np.uint32(0),
        module_chunk_count=np.uint32(MODULES_PER_CHUNK),
        class_chunk_offset=np.uint32(0),
        class_chunk_count=np.uint32(CLASSES_PER_CHUNK),
        hidden_count=np.uint32(HIDDEN_NATURAL),
        total_output_class_count=np.uint32(CLASSES_NATURAL),
    )


def _make_compute_probs_loss_cce(bm: MockBufferManager, ac: MockArchConsts):
    return ComputeProbsLossCceChunkSignature(
        _buffer_mgr=bm,
        _arch_consts=ac,
        logit_ref=bm.make_handle((MODULES, BATCH, CLASSES_PADDED)),
        temp_ref=bm.make_handle((MODULES,)),
        target_ref=bm.make_handle((BATCH,)),
        mask_ref=bm.make_handle((BATCH,)),
        prob_out_ref=bm.make_handle(
            (TILES, MODULES_PER_CHUNK, BATCH, CLASSES_PER_CHUNK)
        ),
        loss_out_ref=bm.make_handle((MODULES, BATCH)),
        tile=make_tile(),
        total_output_class_count=np.uint32(CLASSES_NATURAL),
    )


def _make_compute_probs_loss_bce(bm: MockBufferManager, ac: MockArchConsts):
    return ComputeProbsLossBceChunkSignature(
        _buffer_mgr=bm,
        _arch_consts=ac,
        logit_ref=bm.make_handle((MODULES, BATCH, CLASSES_PADDED)),
        temp_ref=bm.make_handle((MODULES,)),
        target_ref=bm.make_handle((BATCH, CLASSES_PADDED)),
        mask_ref=bm.make_handle((BATCH,)),
        prob_out_ref=bm.make_handle(
            (TILES, MODULES_PER_CHUNK, BATCH, CLASSES_PER_CHUNK)
        ),
        partial_loss_out_ref=bm.make_handle(
            (TILES, MODULES_PER_CHUNK, BATCH)
        ),
        tile=make_tile(),
        total_output_class_count=np.uint32(CLASSES_NATURAL),
    )


def _make_calc_module_grads_cce(bm: MockBufferManager, ac: MockArchConsts):
    return CalculateModuleParamGradsCceSignature(
        _buffer_mgr=bm,
        _arch_consts=ac,
        work_group_size_0=WG_SIZE_REDUCTION,
        h_ref=bm.make_handle((BATCH, HIDDEN_PADDED)),
        prob_ref=bm.make_handle(
            (TILES, MODULES_PER_CHUNK, BATCH, CLASSES_PER_CHUNK)
        ),
        mask_ref=bm.make_handle((BATCH,)),
        gw_out_ref=bm.make_handle(
            (TILES, MODULES_PER_CHUNK, HIDDEN_PADDED, CLASSES_PER_CHUNK)
        ),
        gb_out_ref=bm.make_handle(
            (TILES, MODULES_PER_CHUNK, CLASSES_PER_CHUNK)
        ),
        tile=make_tile(),
        batch_chunk_offset=np.uint32(0),
        batch_chunk_count=np.uint32(BATCH),
        hidden_count=np.uint32(HIDDEN_NATURAL),
        total_output_class_count=np.uint32(CLASSES_NATURAL),
        padded_total_output_class_count=np.uint32(CLASSES_PADDED),
        total_modules_count=np.uint32(MODULES),
        targets_cce_ref=bm.make_handle((BATCH,)),
    )


def _make_calc_module_grads_bce(bm: MockBufferManager, ac: MockArchConsts):
    return CalculateModuleParamGradsBceSignature(
        _buffer_mgr=bm,
        _arch_consts=ac,
        work_group_size_0=WG_SIZE_REDUCTION,
        h_ref=bm.make_handle((BATCH, HIDDEN_PADDED)),
        prob_ref=bm.make_handle(
            (TILES, MODULES_PER_CHUNK, BATCH, CLASSES_PER_CHUNK)
        ),
        mask_ref=bm.make_handle((BATCH,)),
        gw_out_ref=bm.make_handle(
            (TILES, MODULES_PER_CHUNK, HIDDEN_PADDED, CLASSES_PER_CHUNK)
        ),
        gb_out_ref=bm.make_handle(
            (TILES, MODULES_PER_CHUNK, CLASSES_PER_CHUNK)
        ),
        tile=make_tile(),
        batch_chunk_offset=np.uint32(0),
        batch_chunk_count=np.uint32(BATCH),
        hidden_count=np.uint32(HIDDEN_NATURAL),
        total_output_class_count=np.uint32(CLASSES_NATURAL),
        padded_total_output_class_count=np.uint32(CLASSES_PADDED),
        total_modules_count=np.uint32(MODULES),
        targets_bce_ref=bm.make_handle((BATCH, CLASSES_PADDED)),
    )


def _make_backprop_hidden_cce(bm: MockBufferManager, ac: MockArchConsts):
    return BackpropErrorToHiddenChunkCceSignature(
        _buffer_mgr=bm,
        _arch_consts=ac,
        prob_ref=bm.make_handle(
            (TILES, MODULES_PER_CHUNK, BATCH, CLASSES_PER_CHUNK)
        ),
        mask_ref=bm.make_handle((BATCH,)),
        w_mod_ref=bm.make_handle((MODULES, HIDDEN_PADDED, CLASSES_PADDED)),
        gh_out_ref=bm.make_handle(
            (TILES, MODULES_PER_CHUNK, BATCH, HIDDEN_PADDED)
        ),
        tile=make_tile(),
        hidden_count=np.uint32(HIDDEN_NATURAL),
        total_output_class_count=np.uint32(CLASSES_NATURAL),
        targets_cce_ref=bm.make_handle((BATCH,)),
    )


def _make_backprop_hidden_bce(bm: MockBufferManager, ac: MockArchConsts):
    return BackpropErrorToHiddenChunkBceSignature(
        _buffer_mgr=bm,
        _arch_consts=ac,
        prob_ref=bm.make_handle(
            (TILES, MODULES_PER_CHUNK, BATCH, CLASSES_PER_CHUNK)
        ),
        mask_ref=bm.make_handle((BATCH,)),
        w_mod_ref=bm.make_handle((MODULES, HIDDEN_PADDED, CLASSES_PADDED)),
        gh_out_ref=bm.make_handle(
            (TILES, MODULES_PER_CHUNK, BATCH, HIDDEN_PADDED)
        ),
        tile=make_tile(),
        hidden_count=np.uint32(HIDDEN_NATURAL),
        total_output_class_count=np.uint32(CLASSES_NATURAL),
        targets_bce_ref=bm.make_handle((BATCH, CLASSES_PADDED)),
    )


def _make_temp_grads_cce(bm: MockBufferManager, ac: MockArchConsts):
    return CalculateChunkTempGradientsCceSignature(
        _buffer_mgr=bm,
        _arch_consts=ac,
        work_group_size_0=WG_SIZE_REDUCTION,
        logit_ref=bm.make_handle((MODULES, BATCH, CLASSES_PADDED)),
        prob_ref=bm.make_handle(
            (TILES, MODULES_PER_CHUNK, BATCH, CLASSES_PER_CHUNK)
        ),
        mask_ref=bm.make_handle((BATCH,)),
        temp_ref=bm.make_handle((MODULES,)),
        gt_out_ref=bm.make_handle((TILES, MODULES_PER_CHUNK)),
        tile=make_tile(),
        total_output_class_count=np.uint32(CLASSES_NATURAL),
        targets_cce_ref=bm.make_handle((BATCH,)),
    )


def _make_temp_grads_bce(bm: MockBufferManager, ac: MockArchConsts):
    return CalculateChunkTempGradientsBceSignature(
        _buffer_mgr=bm,
        _arch_consts=ac,
        work_group_size_0=WG_SIZE_REDUCTION,
        logit_ref=bm.make_handle((MODULES, BATCH, CLASSES_PADDED)),
        prob_ref=bm.make_handle(
            (TILES, MODULES_PER_CHUNK, BATCH, CLASSES_PER_CHUNK)
        ),
        mask_ref=bm.make_handle((BATCH,)),
        temp_ref=bm.make_handle((MODULES,)),
        gt_out_ref=bm.make_handle((TILES, MODULES_PER_CHUNK)),
        tile=make_tile(),
        total_output_class_count=np.uint32(CLASSES_NATURAL),
        targets_bce_ref=bm.make_handle((BATCH, CLASSES_PADDED)),
    )


def _make_gradient_handles(bm: MockBufferManager) -> GradientHandles:
    return GradientHandles(
        grad_weights_module=bm.make_handle(
            (TILES, MODULES_PER_CHUNK, HIDDEN_PADDED, CLASSES_PER_CHUNK)
        ),
        grad_biases_module=bm.make_handle(
            (TILES, MODULES_PER_CHUNK, CLASSES_PER_CHUNK)
        ),
        grad_temps=bm.make_handle((TILES, MODULES_PER_CHUNK)),
        grad_hidden_activations_aos=bm.make_handle(
            (TILES, MODULES_PER_CHUNK, BATCH, HIDDEN_PADDED)
        ),
        clipped_grad_weights_module=bm.make_handle(
            (TILES, MODULES_PER_CHUNK, HIDDEN_PADDED, CLASSES_PER_CHUNK)
        ),
        clipped_grad_biases_module=bm.make_handle(
            (TILES, MODULES_PER_CHUNK, CLASSES_PER_CHUNK)
        ),
        clipped_grad_temps=bm.make_handle((TILES, MODULES_PER_CHUNK)),
        clipped_grad_hidden_activations_aos=bm.make_handle(
            (TILES, MODULES_PER_CHUNK, BATCH, HIDDEN_PADDED)
        ),
    )


def _make_clip_global(bm: MockBufferManager, ac: MockArchConsts):
    return ClipPartialGradientsGlobalNormSignature(
        _buffer_mgr=bm,
        _arch_consts=ac,
        handles=_make_gradient_handles(bm),
        tile=make_tile(),
        epsilon=np.float32(1e-6),
        clipping_threshold_global=np.float32(1.0),
    )


def _make_clip_per_item(bm: MockBufferManager, ac: MockArchConsts):
    return ClipPartialGradientsPerItemNormSignature(
        _buffer_mgr=bm,
        _arch_consts=ac,
        handles=_make_gradient_handles(bm),
        tile=make_tile(),
        epsilon=np.float32(1e-6),
        clipping_threshold_per_item_ref=bm.make_handle((TILES,)),
    )


def _make_gather_permute(bm: MockBufferManager, ac: MockArchConsts):
    return GatherAndPermuteGradHiddenActivationsSignature(
        _buffer_mgr=bm,
        _arch_consts=ac,
        clipped_partials_aos_ref=bm.make_handle(
            (TILES, MODULES_PER_CHUNK, BATCH, HIDDEN_PADDED)
        ),
        permuted_soa_out_ref=bm.make_handle(
            (BATCH * HIDDEN_PADDED, PADDED_MODULES)
        ),
        total_modules_count=np.uint32(MODULES),
        hidden_count=np.uint32(HIDDEN_NATURAL),
        total_batch_count=np.uint32(BATCH),
        num_module_chunks_count=np.uint32(NUM_MODULE_CHUNKS),
        modules_per_chunk_count=np.uint32(MODULES_PER_CHUNK),
        num_class_chunks_count=np.uint32(NUM_CLASS_CHUNKS),
    )


def _make_aggregate_register(bm: MockBufferManager, ac: MockArchConsts):
    return AggregateRegisterReduceSignature(
        _buffer_mgr=bm,
        _arch_consts=ac,
        partial_collection_ref=bm.make_handle((128,)),
        partial_offset_list_ref=bm.make_handle((4,)),
        dest_ref=bm.make_handle((32,)),
        partial_offset_list_count=np.uint32(4),
        partial_width=np.uint32(32),
        operation_type=np.uint32(0),
    )


def _make_aggregate_local(bm: MockBufferManager, ac: MockArchConsts):
    return AggregateLocalReduceSignature(
        _buffer_mgr=bm,
        _arch_consts=ac,
        partial_collection_ref=bm.make_handle((128,)),
        partial_offset_list_ref=bm.make_handle((4,)),
        dest_ref=bm.make_handle((32,)),
        partial_offset_list_count=np.uint32(4),
        partial_width=np.uint32(32),
        operation_type=np.uint32(0),
    )


def _make_clip_intermediate(bm: MockBufferManager, ac: MockArchConsts):
    return ClipIntermediateGradSignature(
        _buffer_mgr=bm,
        _arch_consts=ac,
        intermediate_grad_ref=bm.make_handle((32,)),
        clipping_threshold_t_j=np.float32(1.0),
        epsilon=np.float32(1e-6),
    )


def _make_stabilize_reduce(bm: MockBufferManager, ac: MockArchConsts):
    return StabilizeAndReduceGradHiddenActivationsSignature(
        _buffer_mgr=bm,
        _arch_consts=ac,
        permuted_soa_in_ref=bm.make_handle(
            (BATCH * HIDDEN_PADDED, PADDED_MODULES)
        ),
        final_grad_h_out_ref=bm.make_handle((BATCH, HIDDEN_PADDED)),
        fp_max=np.float32(3.4e38),
        policy_t_algorithmic=np.float32(1.0),
        policy_lambda=np.float32(0.1),
        policy_max_k=np.uint32(16),
        epsilon=np.float32(1e-6),
        total_batch_count=np.uint32(BATCH),
        padded_hidden_count=np.uint32(HIDDEN_PADDED),
        total_modules_count=np.uint32(MODULES),
        padded_total_modules_count=np.uint32(PADDED_MODULES),
    )


def _make_backprop_shared_weights(bm: MockBufferManager, ac: MockArchConsts):
    return BackpropSharedWeightsChunkSignature(
        _buffer_mgr=bm,
        _arch_consts=ac,
        input_ref=bm.make_handle((BATCH, INPUT_PADDED)),
        h_ref=bm.make_handle((BATCH, HIDDEN_PADDED)),
        grad_h_ref=bm.make_handle((BATCH, HIDDEN_PADDED)),
        mask_ref=bm.make_handle((BATCH,)),
        partial_gsw_out_ref=bm.make_handle(
            (BATCH_CHUNKS, INPUT_PADDED, HIDDEN_PADDED)
        ),
        batch_chunk_offset=np.uint32(0),
        batch_chunk_count=np.uint32(BATCH // BATCH_CHUNKS),
        batch_chunk_index=np.uint32(0),
        num_batch_chunks_count=np.uint32(BATCH_CHUNKS),
    )


def _make_backprop_shared_biases(bm: MockBufferManager, ac: MockArchConsts):
    return BackpropSharedBiasesChunkSignature(
        _buffer_mgr=bm,
        _arch_consts=ac,
        h_ref=bm.make_handle((BATCH, HIDDEN_PADDED)),
        grad_h_ref=bm.make_handle((BATCH, HIDDEN_PADDED)),
        mask_ref=bm.make_handle((BATCH,)),
        partial_gsb_out_ref=bm.make_handle((BATCH_CHUNKS, HIDDEN_PADDED)),
        batch_chunk_offset=np.uint32(0),
        batch_chunk_count=np.uint32(BATCH // BATCH_CHUNKS),
        batch_chunk_index=np.uint32(0),
        num_batch_chunks_count=np.uint32(BATCH_CHUNKS),
    )


def _make_clip_shared(bm: MockBufferManager, ac: MockArchConsts):
    return ClipSharedGradientsChunkSignature(
        _buffer_mgr=bm,
        _arch_consts=ac,
        handles=SharedGradientHandles(
            grad_weights_shared_chunk=bm.make_handle((INPUT_PADDED, HIDDEN_PADDED)),
            grad_biases_shared_chunk=bm.make_handle((HIDDEN_PADDED,)),
            clipped_grad_weights_shared_collection=bm.make_handle(
                (BATCH_CHUNKS, INPUT_PADDED, HIDDEN_PADDED)
            ),
            clipped_grad_biases_shared_collection=bm.make_handle(
                (BATCH_CHUNKS, HIDDEN_PADDED)
            ),
        ),
        clipping_threshold_global=np.float32(1.0),
        epsilon=np.float32(1e-6),
        dest_weights_write_offset_elements=np.uint32(0),
        dest_biases_write_offset_elements=np.uint32(0),
        num_batch_chunks=np.uint32(BATCH_CHUNKS),
    )


def _make_normalize(bm: MockBufferManager, ac: MockArchConsts):
    return NormalizeGradientsSignature(
        _buffer_mgr=bm,
        _arch_consts=ac,
        summed_grad_ref=bm.make_handle((32,)),
        final_grad_out_ref=bm.make_handle((32,)),
        effective_batch_size=np.float32(BATCH),
        epsilon=np.float32(1e-6),
    )


def _make_adam_update(bm: MockBufferManager, ac: MockArchConsts):
    return AdamUpdateSignature(
        _buffer_mgr=bm,
        _arch_consts=ac,
        param_group=AdamParameterGroup(
            param_ref=bm.make_handle((32,)),
            grad_ref=bm.make_handle((32,)),
            m1_state_ref=bm.make_handle((32,)),
            m2_state_ref=bm.make_handle((32,)),
        ),
        learning_rate=np.float32(0.001),
        beta1=np.float32(0.9),
        beta2=np.float32(0.999),
        epsilon=np.float32(1e-8),
        beta1_pow_t=np.float32(0.9),
        beta2_pow_t=np.float32(0.999),
    )


def _make_clamp_temps(bm: MockBufferManager, ac: MockArchConsts):
    return ClampTemperaturesSignature(
        _buffer_mgr=bm,
        _arch_consts=ac,
        temps_ref=bm.make_handle((MODULES,)),
        min_val=np.float32(0.1),
        max_val=np.float32(10.0),
    )


# =========================================================================
# The Canonical Registry
# =========================================================================

# Maps a human-readable label to (factory_fn, expected_kernel_name).
# Factory functions accept (MockBufferManager, MockArchConsts) and return
# an instantiated signature.
SIGNATURE_REGISTRY: Dict[str, tuple] = {
    "ForwardPassSignature": (_make_forward_pass, "forward_pass"),
    "RenderLogitsChunkSignature": (_make_render_logits_chunk, "render_logits_chunk"),
    "ComputeProbsLossCceChunkSignature": (_make_compute_probs_loss_cce, "compute_probs_loss_cce_chunk"),
    "ComputeProbsLossBceChunkSignature": (_make_compute_probs_loss_bce, "compute_probs_loss_bce_chunk"),
    "CalculateModuleParamGradsCceSignature": (_make_calc_module_grads_cce, "calculate_module_param_grads_chunk"),
    "CalculateModuleParamGradsBceSignature": (_make_calc_module_grads_bce, "calculate_module_param_grads_chunk"),
    "BackpropErrorToHiddenChunkCceSignature": (_make_backprop_hidden_cce, "backprop_error_to_hidden_chunk"),
    "BackpropErrorToHiddenChunkBceSignature": (_make_backprop_hidden_bce, "backprop_error_to_hidden_chunk"),
    "CalculateChunkTempGradientsCceSignature": (_make_temp_grads_cce, "calculate_chunk_temp_gradients"),
    "CalculateChunkTempGradientsBceSignature": (_make_temp_grads_bce, "calculate_chunk_temp_gradients"),
    "ClipPartialGradientsGlobalNormSignature": (_make_clip_global, "clip_partial_gradients"),
    "ClipPartialGradientsPerItemNormSignature": (_make_clip_per_item, "clip_partial_gradients"),
    "GatherAndPermuteGradHiddenActivationsSignature": (_make_gather_permute, "gather_and_permute_grad_hidden_activations"),
    "AggregateRegisterReduceSignature": (_make_aggregate_register, "aggregate_register_reduce"),
    "AggregateLocalReduceSignature": (_make_aggregate_local, "aggregate_local_reduce"),
    "ClipIntermediateGradSignature": (_make_clip_intermediate, "clip_intermediate_grad"),
    "StabilizeAndReduceGradHiddenActivationsSignature": (_make_stabilize_reduce, "stabilize_and_reduce_grad_hidden_activations"),
    "BackpropSharedWeightsChunkSignature": (_make_backprop_shared_weights, "backprop_shared_weights_chunk"),
    "BackpropSharedBiasesChunkSignature": (_make_backprop_shared_biases, "backprop_shared_biases_chunk"),
    "ClipSharedGradientsChunkSignature": (_make_clip_shared, "clip_shared_gradients_chunk"),
    "NormalizeGradientsSignature": (_make_normalize, "normalize_gradients"),
    "AdamUpdateSignature": (_make_adam_update, "adam_update"),
    "ClampTemperaturesSignature": (_make_clamp_temps, "clamp_temperatures"),
}


# =========================================================================
# Test 1: Coverage — Every C kernel has at least one Python signature
# =========================================================================


class TestCoverageCompleteness:
    """Every ``__kernel void`` in kernels.cl.h must have a Python signature."""

    def test_every_kernel_has_signature(self, parsed_header):
        covered_kernels = {
            kernel_name
            for _, (_, kernel_name) in SIGNATURE_REGISTRY.items()
        }
        header_kernels = set(parsed_header.keys())

        missing = header_kernels - covered_kernels
        assert not missing, (
            f"C kernels with no Python signature class:\n"
            + "\n".join(f"  - {k}" for k in sorted(missing))
        )

    def test_no_orphaned_signatures(self, parsed_header):
        """Every Python signature's kernel_name must exist in the C header."""
        header_kernels = set(parsed_header.keys())

        orphaned = []
        for label, (_, kernel_name) in SIGNATURE_REGISTRY.items():
            if kernel_name not in header_kernels:
                orphaned.append(f"{label} → {kernel_name}")

        assert not orphaned, (
            f"Orphaned Python signatures (kernel_name not in header):\n"
            + "\n".join(f"  - {o}" for o in orphaned)
        )


# =========================================================================
# Test 2: Argument Count — get_args() count matches C parameter count
# =========================================================================


class TestArgCount:
    """``len(get_args())`` must equal the number of C parameters for each kernel."""

    @pytest.mark.parametrize("label", list(SIGNATURE_REGISTRY.keys()))
    def test_arg_count_matches_header(self, label, parsed_header):
        factory, kernel_name = SIGNATURE_REGISTRY[label]
        bm = MockBufferManager()
        ac = MockArchConsts()
        sig = factory(bm, ac)

        python_args = sig.get_args()
        c_params = parsed_header[kernel_name]

        assert len(python_args) == len(c_params), (
            f"[{label}] Arg count mismatch for kernel '{kernel_name}': "
            f"Python get_args() returns {len(python_args)} args, "
            f"C header declares {len(c_params)} params.\n"
            f"  Python types: {[classify_python_arg(a).name for a in python_args]}\n"
            f"  C types:      {[c.name for c in c_params]}"
        )


# =========================================================================
# Test 3: Type Category Ordering — each position matches
# =========================================================================


class TestArgTypeCategoryOrdering:
    """Each position's type category must match between Python and C."""

    @pytest.mark.parametrize("label", list(SIGNATURE_REGISTRY.keys()))
    def test_arg_type_categories_match_header(self, label, parsed_header):
        factory, kernel_name = SIGNATURE_REGISTRY[label]
        bm = MockBufferManager()
        ac = MockArchConsts()
        sig = factory(bm, ac)

        python_args = sig.get_args()
        c_params = parsed_header[kernel_name]
        python_cats = [classify_python_arg(a) for a in python_args]

        mismatches = []
        for i, (py_cat, c_cat) in enumerate(zip(python_cats, c_params)):
            if py_cat != c_cat:
                mismatches.append(
                    f"  pos {i}: Python={py_cat.name}, C={c_cat.name}"
                )

        assert not mismatches, (
            f"[{label}] Type category mismatches for kernel '{kernel_name}':\n"
            + "\n".join(mismatches)
            + f"\n  Full Python: {[c.name for c in python_cats]}"
            + f"\n  Full C:      {[c.name for c in c_params]}"
        )


# =========================================================================
# Test 4: kernel_name property matches expected value
# =========================================================================


class TestKernelNameProperty:
    """Each signature's kernel_name must return the exact C function name."""

    @pytest.mark.parametrize("label", list(SIGNATURE_REGISTRY.keys()))
    def test_kernel_name(self, label):
        factory, expected_kernel_name = SIGNATURE_REGISTRY[label]
        bm = MockBufferManager()
        ac = MockArchConsts()
        sig = factory(bm, ac)

        assert sig.kernel_name == expected_kernel_name, (
            f"[{label}] kernel_name mismatch: "
            f"got '{sig.kernel_name}', expected '{expected_kernel_name}'"
        )


# =========================================================================
# Test 5: Grid well-formedness
# =========================================================================


class TestGridWellFormedness:
    """get_grid() must return (global_size, local_size|None) with positive dims."""

    @pytest.mark.parametrize("label", list(SIGNATURE_REGISTRY.keys()))
    def test_grid_wellformed(self, label):
        factory, _ = SIGNATURE_REGISTRY[label]
        bm = MockBufferManager()
        ac = MockArchConsts()
        sig = factory(bm, ac)

        global_size, local_size = sig.get_grid()

        assert isinstance(global_size, tuple), (
            f"[{label}] global_size must be a tuple, got {type(global_size).__name__}"
        )
        assert len(global_size) >= 1, f"[{label}] global_size must have ≥1 dimension"
        assert all(isinstance(d, int) and d > 0 for d in global_size), (
            f"[{label}] all global_size dims must be positive ints, got {global_size}"
        )

        if local_size is not None:
            assert isinstance(local_size, tuple), (
                f"[{label}] local_size must be a tuple or None, got {type(local_size).__name__}"
            )
            assert len(local_size) == len(global_size), (
                f"[{label}] local_size dims ({len(local_size)}) must match "
                f"global_size dims ({len(global_size)})"
            )
            assert all(isinstance(d, int) and d > 0 for d in local_size), (
                f"[{label}] all local_size dims must be positive ints, got {local_size}"
            )
