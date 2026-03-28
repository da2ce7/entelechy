# src/shared/plan_builder.py
"""Plan construction logic (ADR-002, ADR-003, ADR-004, ADR-009).

Assembles ExecutionPlan DAGs from ModelSpec, HardwareProfile,
StabilizationPolicy, and PlanProblemTypeStrategy inputs.
"""
from __future__ import annotations

from typing import Literal

import numpy as np

from .buffer_lifecycle import BufferDescriptor, BufferHandle, BufferRole
from .hardware_profile import HardwareProfile
from .kernel_contracts import KernelContract
from .kernel_contracts.phase_1_act import (
    forward_pass_contract,
    render_logits_chunk_contract,
)
from .kernel_contracts.phase_2_learn_B_processing import (
    clip_partial_gradients_contract,
)
from .kernel_contracts.phase_2_learn_C_reduction import (
    gather_and_permute_grad_h_contract,
    stabilize_reduce_grad_h_contract,
)
from .kernel_contracts.phase_2_learn_D_backprop import (
    backprop_shared_biases_contract,
    backprop_shared_weights_contract,
    clip_shared_gradients_contract,
)
from .kernel_contracts.phase_3_update import (
    adam_update_contract,
    clamp_temperatures_contract,
    normalize_gradients_contract,
)
from .model_spec import ModelSpec
from .plan_types import (
    BarrierNode,
    ExecutionPlan,
    KernelDispatchNode,
    PlanNode,
    ReductionTreeNode,
    RetrievalNode,
    StreamingLoopNode,
)
from .problem_type_strategy import PlanProblemTypeStrategy
from .reduction_tree_plan import ReductionTreePlan
from .stabilization_policy import StabilizationPolicy
from .streaming_loop_plan import (
    IterationDimension,
    ParameterStride,
    StreamingLoopPlan,
)
from .workload_primitives import TiledGather, TilingScheme


# =========================================================================
# Internal helpers
# =========================================================================


class _BufferAllocator:
    """Monotonically allocates BufferHandles within a single plan."""

    def __init__(self) -> None:
        self._next_id: int = 0
        self._descriptors: dict[BufferHandle, BufferDescriptor] = {}
        self._producers: dict[BufferHandle, str | None] = {}
        self._consumers: dict[BufferHandle, set[str]] = {}

    def allocate(
        self,
        logical_name: str,
        padded_shape: tuple[int, ...],
        element_size_bytes: int,
        role: BufferRole,
    ) -> BufferHandle:
        handle = BufferHandle(self._next_id)
        self._next_id += 1
        size_bytes = int(np.prod(padded_shape)) * element_size_bytes
        self._descriptors[handle] = BufferDescriptor(
            handle=handle,
            logical_name=logical_name,
            padded_shape=padded_shape,
            element_size_bytes=element_size_bytes,
            size_bytes=size_bytes,
            role=role,
            producing_node=None,
            consumers=frozenset(),
            last_consumer=None,
        )
        self._producers[handle] = None
        self._consumers[handle] = set()
        return handle

    def set_producer(self, handle: BufferHandle, node_id: str) -> None:
        self._producers[handle] = node_id

    def add_consumer(self, handle: BufferHandle, node_id: str) -> None:
        self._consumers[handle].add(node_id)

    def finalize(self, topological_order: tuple[str, ...]) -> dict[BufferHandle, BufferDescriptor]:
        order_index = {nid: i for i, nid in enumerate(topological_order)}
        result: dict[BufferHandle, BufferDescriptor] = {}
        for handle, desc in self._descriptors.items():
            consumers = frozenset(self._consumers[handle])
            last = None
            if consumers:
                last = max(consumers, key=lambda c: order_index.get(c, -1))
            # Reconstruct with finalized lifetime info
            result[handle] = BufferDescriptor(
                handle=desc.handle,
                logical_name=desc.logical_name,
                padded_shape=desc.padded_shape,
                element_size_bytes=desc.element_size_bytes,
                size_bytes=desc.size_bytes,
                role=desc.role,
                producing_node=self._producers[handle],
                consumers=consumers,
                last_consumer=last,
            )
        return result


def _make_tiling(spec: ModelSpec) -> TilingScheme:
    """Build a TilingScheme from ModelSpec geometry."""
    chunk_size = 16
    num_mod_chunks = max(1, (spec.num_modules + chunk_size - 1) // chunk_size)
    num_cls_chunks = max(1, (spec.output_classes + chunk_size - 1) // chunk_size)
    return TilingScheme(
        num_module_chunks=num_mod_chunks,
        num_class_chunks=num_cls_chunks,
        total_modules=spec.num_modules,
        total_classes=spec.output_classes,
    )


def _build_reduction_tree(
    policy: StabilizationPolicy,
    hardware: HardwareProfile,
    gather: TiledGather,
    source_buf: BufferHandle,
    dest_buf: BufferHandle,
    tree_variant: Literal["sum", "sum_and_clip"],
) -> ReductionTreePlan:
    """Construct a ReductionTreePlan from a GatherPrimitive and policy."""
    num_partials = gather.num_partials
    if num_partials <= 1:
        fan_in_k = max(2, num_partials)
        num_stages = 1
    else:
        fan_in_k, num_stages = policy.plan_uniform_reduction_tree(
            num_partials, hardware.max_reduce_fan_in
        )
    num_stages = max(1, num_stages)
    offsets = tuple(int(o) for o in gather.get_offsets())

    if tree_variant == "sum_and_clip":
        schedule: tuple[float | None, ...] = tuple(
            policy.get_threshold_for_generic_stage(j, fan_in_k)
            for j in reversed(range(num_stages))  # leaf→root
        )
    else:
        schedule = tuple(None for _ in range(num_stages))

    return ReductionTreePlan(
        num_partials=num_partials,
        fan_in_K=fan_in_k,
        num_stages=num_stages,
        elements_per_partial=gather.elements_per_partial,
        initial_offset_list=offsets,
        tree_variant=tree_variant,
        threshold_schedule=schedule,
        partial_width=gather.elements_per_partial,
        source_buffer=source_buf,
        destination_buffer=dest_buf,
    )


def _dispatch(
    node_id: str,
    depends_on: frozenset[str],
    contract: KernelContract,
    buffer_bindings: dict[str, BufferHandle],
    scalar_params: dict[str, int | float],
    tile_count: int,
    placement_strategy: str | None = None,
) -> KernelDispatchNode:
    return KernelDispatchNode(
        node_id=node_id,
        depends_on=depends_on,
        kernel_name=contract.kernel_name,
        contract=contract,
        buffer_bindings=buffer_bindings,
        scalar_params=scalar_params,
        tile_count=tile_count,
        local_work_size=None,
        placement_strategy=placement_strategy,
    )


# =========================================================================
# Act plan builder
# =========================================================================


def build_act_plan(
    model_spec: ModelSpec,
    hardware: HardwareProfile,
    strategy: PlanProblemTypeStrategy,
    batch_size: int,
) -> ExecutionPlan:
    """Construct an Act-phase (forward pass + inference retrieval) plan."""
    alloc = _BufferAllocator()
    elem = int(np.dtype(model_spec.precision.numpy_dtype).itemsize)
    tiling = _make_tiling(model_spec)
    tile_count = tiling.total_tiles

    # MODEL_STATE buffers
    b_shared_weights = alloc.allocate(
        "shared_weights",
        (model_spec.padded_input_dim, model_spec.padded_hidden_dim),
        elem, BufferRole.MODEL_STATE,
    )
    b_module_weights = alloc.allocate(
        "module_weights",
        (model_spec.num_modules, model_spec.padded_hidden_dim, model_spec.padded_class_dim),
        elem, BufferRole.MODEL_STATE,
    )
    b_module_biases = alloc.allocate(
        "module_biases",
        (model_spec.num_modules, model_spec.padded_class_dim),
        elem, BufferRole.MODEL_STATE,
    )
    b_temperatures = alloc.allocate(
        "temperatures",
        (model_spec.padded_module_dim,),
        elem, BufferRole.MODEL_STATE,
    )

    # BATCH_INPUT buffers
    b_input_data = alloc.allocate(
        "input_data",
        (batch_size, model_spec.padded_input_dim),
        elem, BufferRole.BATCH_INPUT,
    )
    b_sample_mask = alloc.allocate(
        "sample_mask",
        (batch_size,),
        elem, BufferRole.BATCH_INPUT,
    )
    b_targets = alloc.allocate(
        strategy.required_targets_buffer_name,
        (batch_size, model_spec.output_classes),
        elem, BufferRole.BATCH_INPUT,
    )

    # BATCH_INTERMEDIATE
    b_hidden = alloc.allocate(
        "hidden_activations",
        (batch_size, model_spec.padded_hidden_dim),
        elem, BufferRole.BATCH_INTERMEDIATE,
    )
    b_logits = alloc.allocate(
        "full_logits",
        (tile_count, batch_size, model_spec.padded_class_dim),
        elem, BufferRole.BATCH_INTERMEDIATE,
    )
    b_partial_probs = alloc.allocate(
        "partial_probs",
        (tile_count, batch_size, model_spec.padded_class_dim),
        elem, BufferRole.BATCH_INTERMEDIATE,
    )

    # BATCH_OUTPUT
    b_final_probs = alloc.allocate(
        "final_probs",
        (batch_size, model_spec.output_classes),
        elem, BufferRole.BATCH_OUTPUT,
    )

    # --- Nodes ---
    nodes: dict[str, PlanNode] = {}

    # Node 4: forward_pass
    n4 = _dispatch(
        "forward_pass", frozenset(), forward_pass_contract,
        {"shared_weights": b_shared_weights, "input_data": b_input_data,
         "sample_mask": b_sample_mask, "hidden_out": b_hidden},
        {"batch_size": batch_size, "input_dim": model_spec.input_dim,
         "hidden_dim": model_spec.hidden_dim},
        tile_count=1, placement_strategy="linear_batch",
    )
    nodes[n4.node_id] = n4
    alloc.set_producer(b_hidden, n4.node_id)
    alloc.add_consumer(b_shared_weights, n4.node_id)
    alloc.add_consumer(b_input_data, n4.node_id)
    alloc.add_consumer(b_sample_mask, n4.node_id)

    # Node 5: render_logits_chunk
    n5 = _dispatch(
        "render_logits", frozenset({"forward_pass"}), render_logits_chunk_contract,
        {"hidden": b_hidden, "module_weights": b_module_weights,
         "module_biases": b_module_biases, "temperatures": b_temperatures,
         "logits_out": b_logits},
        {"batch_size": batch_size, "num_modules": model_spec.num_modules,
         "output_classes": model_spec.output_classes},
        tile_count=tile_count, placement_strategy="grid_mod_cls",
    )
    nodes[n5.node_id] = n5
    alloc.set_producer(b_logits, n5.node_id)
    alloc.add_consumer(b_hidden, n5.node_id)
    alloc.add_consumer(b_module_weights, n5.node_id)
    alloc.add_consumer(b_module_biases, n5.node_id)
    alloc.add_consumer(b_temperatures, n5.node_id)

    # Node 6/7: loss computation (CCE or BCE)
    loss_contract = strategy.get_loss_contract()
    n_loss = _dispatch(
        "loss_computation", frozenset({"render_logits"}), loss_contract,
        {"logits": b_logits, "targets": b_targets,
         "sample_mask": b_sample_mask, "partial_probs_out": b_partial_probs},
        {"batch_size": batch_size, "output_classes": model_spec.output_classes,
         "num_modules": model_spec.num_modules},
        tile_count=tile_count, placement_strategy="grid_mod_cls",
    )
    nodes[n_loss.node_id] = n_loss
    alloc.set_producer(b_partial_probs, n_loss.node_id)
    alloc.add_consumer(b_logits, n_loss.node_id)
    alloc.add_consumer(b_targets, n_loss.node_id)
    alloc.add_consumer(b_sample_mask, n_loss.node_id)

    # Node 14: Diagnostic reduction (probs)
    probs_gather = TiledGather(
        scheme=tiling,
        _elements_per_partial=batch_size * model_spec.padded_class_dim,
    )
    diag_tree = _build_reduction_tree(
        StabilizationPolicy(
            t_algorithmic=0.0, lambda_=0.0,
            fp_format_max=model_spec.precision.fp_format_max,
        ),
        hardware, probs_gather,
        source_buf=b_partial_probs,
        dest_buf=b_final_probs,
        tree_variant="sum",
    )
    n14 = ReductionTreeNode(
        node_id="diag_reduction",
        depends_on=frozenset({"loss_computation"}),
        reduction_plan=diag_tree,
    )
    nodes[n14.node_id] = n14
    alloc.add_consumer(b_partial_probs, n14.node_id)
    alloc.set_producer(b_final_probs, n14.node_id)

    # Retrieval: inference
    n_ret = RetrievalNode(
        node_id="inference_retrieval",
        depends_on=frozenset({"diag_reduction"}),
        source_buffer=b_final_probs,
        logical_shape=(batch_size, model_spec.output_classes),
        event_name="inference_event",
    )
    nodes[n_ret.node_id] = n_ret
    alloc.add_consumer(b_final_probs, n_ret.node_id)

    topo = ("forward_pass", "render_logits", "loss_computation",
            "diag_reduction", "inference_retrieval")
    buffers = alloc.finalize(topo)

    return ExecutionPlan(
        nodes=nodes,
        buffers=buffers,
        topological_order=topo,
        precision=model_spec.precision,
        hardware=hardware,
    )


# =========================================================================
# Learn plan builder
# =========================================================================


def build_learn_plan(
    model_spec: ModelSpec,
    hardware: HardwareProfile,
    strategy: PlanProblemTypeStrategy,
    batch_size: int,
    policy: StabilizationPolicy,
    activation_lifecycle: Literal["cache", "recompute"] = "recompute",
) -> ExecutionPlan:
    """Construct a Learn-phase (gradient production → update) plan."""
    alloc = _BufferAllocator()
    elem = int(np.dtype(model_spec.precision.numpy_dtype).itemsize)
    tiling = _make_tiling(model_spec)
    tile_count = tiling.total_tiles
    nodes: dict[str, PlanNode] = {}

    # --- MODEL_STATE buffers ---
    b_shared_weights = alloc.allocate(
        "shared_weights",
        (model_spec.padded_input_dim, model_spec.padded_hidden_dim),
        elem, BufferRole.MODEL_STATE,
    )
    b_module_weights = alloc.allocate(
        "module_weights",
        (model_spec.num_modules, model_spec.padded_hidden_dim, model_spec.padded_class_dim),
        elem, BufferRole.MODEL_STATE,
    )
    _b_module_biases = alloc.allocate(
        "module_biases",
        (model_spec.num_modules, model_spec.padded_class_dim),
        elem, BufferRole.MODEL_STATE,
    )
    b_temperatures = alloc.allocate(
        "temperatures", (model_spec.padded_module_dim,),
        elem, BufferRole.MODEL_STATE,
    )
    b_adam_state = alloc.allocate(
        "adam_state", (1,), elem, BufferRole.MODEL_STATE,
    )

    # --- BATCH_INPUT buffers ---
    b_input_data = alloc.allocate(
        "input_data", (batch_size, model_spec.padded_input_dim),
        elem, BufferRole.BATCH_INPUT,
    )
    _b_sample_mask = alloc.allocate(
        "sample_mask", (batch_size,), elem, BufferRole.BATCH_INPUT,
    )
    b_targets = alloc.allocate(
        strategy.required_targets_buffer_name,
        (batch_size, model_spec.output_classes),
        elem, BufferRole.BATCH_INPUT,
    )

    # --- Upstream intermediate buffers (from Act) ---
    b_hidden = alloc.allocate(
        "hidden_activations", (batch_size, model_spec.padded_hidden_dim),
        elem, BufferRole.BATCH_INTERMEDIATE,
    )
    b_logits = alloc.allocate(
        "full_logits",
        (tile_count, batch_size, model_spec.padded_class_dim),
        elem, BufferRole.BATCH_INTERMEDIATE,
    )
    b_partial_probs = alloc.allocate(
        "partial_probs",
        (tile_count, batch_size, model_spec.padded_class_dim),
        elem, BufferRole.BATCH_INTERMEDIATE,
    )

    # --- Phase I intermediate buffers ---
    partial_grad_size = (tile_count,)
    b_partial_grad_mod = alloc.allocate(
        "partial_grad_mod", partial_grad_size, elem, BufferRole.BATCH_INTERMEDIATE,
    )
    b_partial_grad_h = alloc.allocate(
        "partial_grad_h",
        (tile_count, batch_size, model_spec.padded_hidden_dim),
        elem, BufferRole.BATCH_INTERMEDIATE,
    )
    b_partial_grad_temps = alloc.allocate(
        "partial_grad_temps", partial_grad_size, elem, BufferRole.BATCH_INTERMEDIATE,
    )

    # Clipped partials
    b_clipped_grad_mod = alloc.allocate(
        "clipped_partial_grad_mod", partial_grad_size, elem, BufferRole.BATCH_INTERMEDIATE,
    )
    b_clipped_grad_h = alloc.allocate(
        "clipped_partial_grad_h",
        (tile_count, batch_size, model_spec.padded_hidden_dim),
        elem, BufferRole.BATCH_INTERMEDIATE,
    )
    b_clipped_grad_temps = alloc.allocate(
        "clipped_partial_grad_temps", partial_grad_size, elem, BufferRole.BATCH_INTERMEDIATE,
    )

    # Phase II outputs
    b_permuted_grad_h = alloc.allocate(
        "permuted_grad_h",
        (batch_size, model_spec.padded_hidden_dim),
        elem, BufferRole.BATCH_INTERMEDIATE,
    )
    b_summed_grad_h = alloc.allocate(
        "summed_grad_h",
        (model_spec.padded_hidden_dim,),
        elem, BufferRole.BATCH_INTERMEDIATE,
    )
    b_summed_grad_mod = alloc.allocate(
        "summed_grad_mod", (1,), elem, BufferRole.BATCH_INTERMEDIATE,
    )
    b_summed_grad_temps = alloc.allocate(
        "summed_grad_temps", (1,), elem, BufferRole.BATCH_INTERMEDIATE,
    )

    # Phase III intermediates
    b_partial_grad_sw = alloc.allocate(
        "partial_grad_sw",
        (batch_size, model_spec.padded_input_dim, model_spec.padded_hidden_dim),
        elem, BufferRole.BATCH_INTERMEDIATE,
    )
    b_partial_grad_sb = alloc.allocate(
        "partial_grad_sb",
        (batch_size, model_spec.padded_hidden_dim),
        elem, BufferRole.BATCH_INTERMEDIATE,
    )
    b_clipped_grad_s = alloc.allocate(
        "clipped_partial_grad_shared",
        (batch_size,), elem, BufferRole.BATCH_INTERMEDIATE,
    )

    # Phase IV outputs
    b_summed_grad_s = alloc.allocate(
        "summed_grad_shared", (1,), elem, BufferRole.BATCH_INTERMEDIATE,
    )
    b_final_grad_mod = alloc.allocate(
        "final_grad_mod", (1,), elem, BufferRole.BATCH_INTERMEDIATE,
    )
    b_final_grad_temps = alloc.allocate(
        "final_grad_temps", (1,), elem, BufferRole.BATCH_INTERMEDIATE,
    )
    b_final_grad_s = alloc.allocate(
        "final_grad_shared", (1,), elem, BufferRole.BATCH_INTERMEDIATE,
    )

    # BATCH_OUTPUT
    b_final_output = alloc.allocate(
        "final_batch_output", (1,), elem, BufferRole.BATCH_OUTPUT,
    )

    # =====================================================================
    # Phase I: Hierarchical Gradient Generation
    # =====================================================================
    prev_deps: frozenset[str] = frozenset()

    # Node 8: module grad computation
    n8 = _dispatch(
        "calc_module_grads", prev_deps, strategy.get_module_grad_contract(),
        {"partial_probs": b_partial_probs, "hidden": b_hidden,
         "targets": b_targets, "module_weights": b_module_weights,
         "partial_grad_mod_out": b_partial_grad_mod},
        {"batch_size": batch_size, "num_modules": model_spec.num_modules,
         "output_classes": model_spec.output_classes},
        tile_count=tile_count, placement_strategy="grid_mod_cls",
    )
    nodes[n8.node_id] = n8
    alloc.set_producer(b_partial_grad_mod, n8.node_id)
    alloc.add_consumer(b_partial_probs, n8.node_id)
    alloc.add_consumer(b_hidden, n8.node_id)
    alloc.add_consumer(b_targets, n8.node_id)
    alloc.add_consumer(b_module_weights, n8.node_id)

    # Node 9: backprop error to hidden
    n9 = _dispatch(
        "backprop_error_hidden", prev_deps, strategy.get_hidden_grad_contract(),
        {"partial_probs": b_partial_probs, "module_weights": b_module_weights,
         "targets": b_targets, "partial_grad_h_out": b_partial_grad_h},
        {"batch_size": batch_size, "num_modules": model_spec.num_modules,
         "output_classes": model_spec.output_classes,
         "hidden_dim": model_spec.hidden_dim},
        tile_count=tile_count, placement_strategy="grid_mod_cls",
    )
    nodes[n9.node_id] = n9
    alloc.set_producer(b_partial_grad_h, n9.node_id)
    alloc.add_consumer(b_partial_probs, n9.node_id)
    alloc.add_consumer(b_module_weights, n9.node_id)
    alloc.add_consumer(b_targets, n9.node_id)

    # Node 10: temperature gradients
    n10 = _dispatch(
        "calc_temp_grads", prev_deps, strategy.get_temp_grad_contract(),
        {"partial_probs": b_partial_probs, "logits": b_logits,
         "targets": b_targets, "temperatures": b_temperatures,
         "partial_grad_temps_out": b_partial_grad_temps},
        {"batch_size": batch_size, "num_modules": model_spec.num_modules,
         "output_classes": model_spec.output_classes},
        tile_count=tile_count, placement_strategy="grid_mod_cls",
    )
    nodes[n10.node_id] = n10
    alloc.set_producer(b_partial_grad_temps, n10.node_id)
    alloc.add_consumer(b_partial_probs, n10.node_id)
    alloc.add_consumer(b_logits, n10.node_id)
    alloc.add_consumer(b_targets, n10.node_id)
    alloc.add_consumer(b_temperatures, n10.node_id)

    # Node 11: clip partial gradients
    n11 = _dispatch(
        "clip_partial_grads",
        frozenset({"calc_module_grads", "backprop_error_hidden", "calc_temp_grads"}),
        clip_partial_gradients_contract,
        {"partial_grad_mod": b_partial_grad_mod,
         "partial_grad_h": b_partial_grad_h,
         "partial_grad_temps": b_partial_grad_temps,
         "clipped_grad_mod_out": b_clipped_grad_mod,
         "clipped_grad_h_out": b_clipped_grad_h,
         "clipped_grad_temps_out": b_clipped_grad_temps},
        {"clipping_threshold": policy.get_leaf_safety_threshold()},
        tile_count=tile_count, placement_strategy="grid_mod_cls",
    )
    nodes[n11.node_id] = n11
    alloc.set_producer(b_clipped_grad_mod, n11.node_id)
    alloc.set_producer(b_clipped_grad_h, n11.node_id)
    alloc.set_producer(b_clipped_grad_temps, n11.node_id)
    alloc.add_consumer(b_partial_grad_mod, n11.node_id)
    alloc.add_consumer(b_partial_grad_h, n11.node_id)
    alloc.add_consumer(b_partial_grad_temps, n11.node_id)

    # Wrap Phase I in StreamingLoopNode if recompute
    if activation_lifecycle == "recompute":
        streaming_plan = StreamingLoopPlan(
            iteration=IterationDimension(
                total_extent=batch_size,
                chunk_count=max(1, batch_size),
                chunk_size=1,
            ),
            body=("calc_module_grads", "backprop_error_hidden",
                  "calc_temp_grads", "clip_partial_grads"),
            parameter_strides=(
                ParameterStride("batch_offset", base=0, stride=1),
            ),
            scratch_buffers=(),
            constant_scalars={},
        )
        n_loop = StreamingLoopNode(
            node_id="phase_i_recompute_loop",
            depends_on=prev_deps,
            streaming_plan=streaming_plan,
        )
        nodes[n_loop.node_id] = n_loop

    phase_i_done = frozenset({"clip_partial_grads"})

    # =====================================================================
    # Node 13: Item Synchronization — gather_and_permute_grad_h
    # =====================================================================
    n13 = _dispatch(
        "gather_permute_grad_h", phase_i_done,
        gather_and_permute_grad_h_contract,
        {"clipped_grad_h": b_clipped_grad_h,
         "permuted_grad_h_out": b_permuted_grad_h},
        {"batch_size": batch_size, "hidden_dim": model_spec.hidden_dim,
         "tile_count": tile_count},
        tile_count=1, placement_strategy="linear_batch",
    )
    nodes[n13.node_id] = n13
    alloc.set_producer(b_permuted_grad_h, n13.node_id)
    alloc.add_consumer(b_clipped_grad_h, n13.node_id)

    item_sync = BarrierNode(
        node_id="item_sync_barrier",
        depends_on=frozenset({"gather_permute_grad_h"}),
        barrier_name="item_sync",
    )
    nodes[item_sync.node_id] = item_sync

    # =====================================================================
    # Phase II: Specialized & Collective Aggregation
    # =====================================================================

    # Node 16: stabilize_and_reduce_grad_h
    n16 = _dispatch(
        "stabilize_reduce_grad_h",
        frozenset({"item_sync_barrier"}),
        stabilize_reduce_grad_h_contract,
        {"permuted_grad_h": b_permuted_grad_h,
         "summed_grad_h_out": b_summed_grad_h},
        {"batch_size": batch_size, "hidden_dim": model_spec.hidden_dim,
         "policy_max_k": policy.get_specialized_reduction_policy_k(
             batch_size, hardware.max_reduce_fan_in)},
        tile_count=1, placement_strategy="linear_generic",
    )
    nodes[n16.node_id] = n16
    alloc.set_producer(b_summed_grad_h, n16.node_id)
    alloc.add_consumer(b_permuted_grad_h, n16.node_id)

    # Node 15: Module + Temp gradient reduction tree
    mod_gather = TiledGather(scheme=tiling, _elements_per_partial=1)
    mod_tree = _build_reduction_tree(
        policy, hardware, mod_gather,
        source_buf=b_clipped_grad_mod,
        dest_buf=b_summed_grad_mod,
        tree_variant="sum_and_clip",
    )
    n15_mod = ReductionTreeNode(
        node_id="reduce_mod_grads",
        depends_on=frozenset({"item_sync_barrier"}),
        reduction_plan=mod_tree,
    )
    nodes[n15_mod.node_id] = n15_mod
    alloc.add_consumer(b_clipped_grad_mod, n15_mod.node_id)
    alloc.set_producer(b_summed_grad_mod, n15_mod.node_id)

    temps_gather = TiledGather(scheme=tiling, _elements_per_partial=1)
    temps_tree = _build_reduction_tree(
        policy, hardware, temps_gather,
        source_buf=b_clipped_grad_temps,
        dest_buf=b_summed_grad_temps,
        tree_variant="sum_and_clip",
    )
    n15_temps = ReductionTreeNode(
        node_id="reduce_temp_grads",
        depends_on=frozenset({"item_sync_barrier"}),
        reduction_plan=temps_tree,
    )
    nodes[n15_temps.node_id] = n15_temps
    alloc.add_consumer(b_clipped_grad_temps, n15_temps.node_id)
    alloc.set_producer(b_summed_grad_temps, n15_temps.node_id)

    phase_ii_done = frozenset({
        "stabilize_reduce_grad_h", "reduce_mod_grads", "reduce_temp_grads",
    })

    # =====================================================================
    # Phase III: Streaming Backprop
    # =====================================================================

    # Nodes 17, 18, 19 inside a streaming loop
    n17 = _dispatch(
        "backprop_shared_weights", phase_ii_done,
        backprop_shared_weights_contract,
        {"input_data": b_input_data, "hidden": b_hidden,
         "summed_grad_h": b_summed_grad_h,
         "partial_grad_sw_out": b_partial_grad_sw},
        {"batch_size": batch_size, "input_dim": model_spec.input_dim,
         "hidden_dim": model_spec.hidden_dim},
        tile_count=1, placement_strategy="linear_batch",
    )
    nodes[n17.node_id] = n17
    alloc.set_producer(b_partial_grad_sw, n17.node_id)
    alloc.add_consumer(b_input_data, n17.node_id)
    alloc.add_consumer(b_hidden, n17.node_id)
    alloc.add_consumer(b_summed_grad_h, n17.node_id)

    n18 = _dispatch(
        "backprop_shared_biases", phase_ii_done,
        backprop_shared_biases_contract,
        {"hidden": b_hidden, "summed_grad_h": b_summed_grad_h,
         "partial_grad_sb_out": b_partial_grad_sb},
        {"batch_size": batch_size, "hidden_dim": model_spec.hidden_dim},
        tile_count=1, placement_strategy="linear_batch",
    )
    nodes[n18.node_id] = n18
    alloc.set_producer(b_partial_grad_sb, n18.node_id)
    alloc.add_consumer(b_hidden, n18.node_id)
    alloc.add_consumer(b_summed_grad_h, n18.node_id)

    n19 = _dispatch(
        "clip_shared_grads",
        frozenset({"backprop_shared_weights", "backprop_shared_biases"}),
        clip_shared_gradients_contract,
        {"partial_grad_sw": b_partial_grad_sw,
         "partial_grad_sb": b_partial_grad_sb,
         "clipped_grad_s_out": b_clipped_grad_s},
        {"clipping_threshold": policy.get_leaf_safety_threshold()},
        tile_count=1, placement_strategy="linear_batch",
    )
    nodes[n19.node_id] = n19
    alloc.set_producer(b_clipped_grad_s, n19.node_id)
    alloc.add_consumer(b_partial_grad_sw, n19.node_id)
    alloc.add_consumer(b_partial_grad_sb, n19.node_id)

    # Streaming loop wrapping Phase III
    streaming_bp = StreamingLoopPlan(
        iteration=IterationDimension(
            total_extent=batch_size,
            chunk_count=max(1, batch_size),
            chunk_size=1,
        ),
        body=("backprop_shared_weights", "backprop_shared_biases",
              "clip_shared_grads"),
        parameter_strides=(
            ParameterStride("batch_offset", base=0, stride=1),
        ),
        scratch_buffers=(),
        constant_scalars={},
    )
    n_stream_bp = StreamingLoopNode(
        node_id="streaming_backprop_loop",
        depends_on=phase_ii_done,
        streaming_plan=streaming_bp,
    )
    nodes[n_stream_bp.node_id] = n_stream_bp

    phase_iii_done = frozenset({"clip_shared_grads"})

    # =====================================================================
    # Phase IV: Final Aggregation & Normalization
    # =====================================================================

    # Node 20: Shared gradient reduction tree
    shared_gather = TiledGather(
        scheme=TilingScheme(
            num_module_chunks=max(1, batch_size),
            num_class_chunks=1,
            total_modules=batch_size,
            total_classes=1,
        ),
        _elements_per_partial=1,
    )
    shared_tree = _build_reduction_tree(
        policy, hardware, shared_gather,
        source_buf=b_clipped_grad_s,
        dest_buf=b_summed_grad_s,
        tree_variant="sum_and_clip",
    )
    n20 = ReductionTreeNode(
        node_id="reduce_shared_grads",
        depends_on=phase_iii_done,
        reduction_plan=shared_tree,
    )
    nodes[n20.node_id] = n20
    alloc.add_consumer(b_clipped_grad_s, n20.node_id)
    alloc.set_producer(b_summed_grad_s, n20.node_id)

    # Node 21: normalize_gradients
    n21 = _dispatch(
        "normalize_gradients",
        frozenset({"reduce_shared_grads", "reduce_mod_grads", "reduce_temp_grads"}),
        normalize_gradients_contract,
        {"summed_grad_mod": b_summed_grad_mod,
         "summed_grad_temps": b_summed_grad_temps,
         "summed_grad_shared": b_summed_grad_s,
         "final_grad_mod_out": b_final_grad_mod,
         "final_grad_temps_out": b_final_grad_temps,
         "final_grad_shared_out": b_final_grad_s},
        {"batch_size": batch_size},
        tile_count=1, placement_strategy="linear_generic",
    )
    nodes[n21.node_id] = n21
    alloc.set_producer(b_final_grad_mod, n21.node_id)
    alloc.set_producer(b_final_grad_temps, n21.node_id)
    alloc.set_producer(b_final_grad_s, n21.node_id)
    alloc.add_consumer(b_summed_grad_mod, n21.node_id)
    alloc.add_consumer(b_summed_grad_temps, n21.node_id)
    alloc.add_consumer(b_summed_grad_s, n21.node_id)

    # =====================================================================
    # Phase V: Parameter Update
    # =====================================================================

    # Node 22: Batch Synchronization Barrier
    n22 = BarrierNode(
        node_id="batch_sync_barrier",
        depends_on=frozenset({"normalize_gradients"}),
        barrier_name="batch_sync",
    )
    nodes[n22.node_id] = n22

    update_deps = frozenset({"batch_sync_barrier"})

    # Node 24: Adam updates (one per parameter group)
    n24_shared = _dispatch(
        "adam_update_shared", update_deps, adam_update_contract,
        {"params": b_shared_weights, "grads": b_final_grad_s,
         "adam_state": b_adam_state},
        {"batch_size": batch_size},
        tile_count=1, placement_strategy="linear_generic",
    )
    nodes[n24_shared.node_id] = n24_shared
    alloc.add_consumer(b_shared_weights, n24_shared.node_id)
    alloc.add_consumer(b_final_grad_s, n24_shared.node_id)
    alloc.add_consumer(b_adam_state, n24_shared.node_id)

    n24_module = _dispatch(
        "adam_update_module", update_deps, adam_update_contract,
        {"params": b_module_weights, "grads": b_final_grad_mod,
         "adam_state": b_adam_state},
        {"batch_size": batch_size},
        tile_count=1, placement_strategy="linear_generic",
    )
    nodes[n24_module.node_id] = n24_module
    alloc.add_consumer(b_module_weights, n24_module.node_id)
    alloc.add_consumer(b_final_grad_mod, n24_module.node_id)
    alloc.add_consumer(b_adam_state, n24_module.node_id)

    n24_temps = _dispatch(
        "adam_update_temps", update_deps, adam_update_contract,
        {"params": b_temperatures, "grads": b_final_grad_temps,
         "adam_state": b_adam_state},
        {"batch_size": batch_size},
        tile_count=1, placement_strategy="linear_generic",
    )
    nodes[n24_temps.node_id] = n24_temps
    alloc.add_consumer(b_temperatures, n24_temps.node_id)
    alloc.add_consumer(b_final_grad_temps, n24_temps.node_id)
    alloc.add_consumer(b_adam_state, n24_temps.node_id)

    # Node 25: clamp_temperatures
    n25 = _dispatch(
        "clamp_temperatures",
        frozenset({"adam_update_temps"}),
        clamp_temperatures_contract,
        {"temperatures": b_temperatures},
        {},
        tile_count=1, placement_strategy="linear_generic",
    )
    nodes[n25.node_id] = n25
    alloc.add_consumer(b_temperatures, n25.node_id)

    # Final retrieval
    n_final = RetrievalNode(
        node_id="final_batch_retrieval",
        depends_on=frozenset({
            "adam_update_shared", "adam_update_module", "clamp_temperatures",
        }),
        source_buffer=b_final_output,
        logical_shape=(1,),
        event_name="final_batch_event",
    )
    nodes[n_final.node_id] = n_final
    alloc.add_consumer(b_final_output, n_final.node_id)

    # Build topological order
    topo_list = [
        "calc_module_grads", "backprop_error_hidden", "calc_temp_grads",
        "clip_partial_grads",
    ]
    if activation_lifecycle == "recompute":
        topo_list.append("phase_i_recompute_loop")
    topo_list.extend([
        "gather_permute_grad_h", "item_sync_barrier",
        "stabilize_reduce_grad_h", "reduce_mod_grads", "reduce_temp_grads",
        "backprop_shared_weights", "backprop_shared_biases", "clip_shared_grads",
        "streaming_backprop_loop",
        "reduce_shared_grads", "normalize_gradients",
        "batch_sync_barrier",
        "adam_update_shared", "adam_update_module", "adam_update_temps",
        "clamp_temperatures", "final_batch_retrieval",
    ])
    topo = tuple(topo_list)
    buffers = alloc.finalize(topo)

    return ExecutionPlan(
        nodes=nodes,
        buffers=buffers,
        topological_order=topo,
        precision=model_spec.precision,
        hardware=hardware,
    )
