# src/shared/plan_builder.py
"""Plan construction logic (ADR-002, ADR-003, ADR-004, ADR-009).

Assembles ExecutionPlan DAGs from ModelSpec, HardwareProfile,
StabilizationPolicy, and PlanProblemTypeStrategy inputs.
"""
from __future__ import annotations

from typing import Literal, Optional

import numpy as np

from .buffer_lifecycle import BufferDescriptor, BufferHandle, BufferRole
from .hardware_profile import HardwareProfile
from .kernel_contracts import KernelContract
from .optimizer_config import OptimizerConfig
from .precision_config import PrecisionConfig
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
from .workload_primitives import GatherPrimitive, LinearlyChunkedGather, ModuleBufferKind, ModuleChunkGather, TiledGather, TilingScheme


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
        precision_role: Optional[Literal["storage", "compute", "state"]] = "compute",
        logical_shape: tuple[int, ...] | None = None,
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
            precision_role=precision_role,
            logical_shape=logical_shape,
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
                precision_role=desc.precision_role,
                logical_shape=desc.logical_shape,
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
        # ADR-030: Module-chunk geometry for ModuleChunkGather
        padded_hidden_count=spec.padded_hidden_dim,
        padded_total_output_class_count=spec.padded_class_dim,
    )


def _build_reduction_tree(
    policy: StabilizationPolicy,
    hardware: HardwareProfile,
    gather: GatherPrimitive,
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
        fan_in=fan_in_k,
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
    elem_storage = model_spec.precision.storage_dtype.itemsize
    elem_compute = model_spec.precision.compute_dtype.itemsize
    elem_state = model_spec.precision.state_dtype.itemsize
    tiling = _make_tiling(model_spec)
    tile_count = tiling.total_tiles

    # Tiling geometry (needed before buffer allocations)
    modules_per_chunk = (
        (model_spec.num_modules + tiling.num_module_chunks - 1)
        // tiling.num_module_chunks
    )
    classes_per_chunk = (
        (model_spec.output_classes + tiling.num_class_chunks - 1)
        // tiling.num_class_chunks
    )

    # MODEL_STATE buffers
    b_shared_weights = alloc.allocate(
        "shared_weights",
        (model_spec.padded_input_dim, model_spec.padded_hidden_dim),
        elem_state, BufferRole.MODEL_STATE, "state",
        logical_shape=(model_spec.input_dim, model_spec.hidden_dim),
    )
    b_module_weights = alloc.allocate(
        "module_weights",
        (model_spec.num_modules, model_spec.padded_hidden_dim, model_spec.padded_class_dim),
        elem_state, BufferRole.MODEL_STATE, "state",
        logical_shape=(model_spec.num_modules, model_spec.hidden_dim, model_spec.output_classes),
    )
    b_module_biases = alloc.allocate(
        "module_biases",
        (model_spec.num_modules, model_spec.padded_class_dim),
        elem_state, BufferRole.MODEL_STATE, "state",
    )
    b_temperatures = alloc.allocate(
        "temperatures",
        (model_spec.padded_module_dim,),
        elem_state, BufferRole.MODEL_STATE, "state",
    )
    b_biases_shared = alloc.allocate(
        "biases_shared",
        (model_spec.padded_hidden_dim,),
        elem_state, BufferRole.MODEL_STATE, "state",
    )

    # BATCH_INPUT buffers
    b_input_data = alloc.allocate(
        "input_data",
        (batch_size, model_spec.padded_input_dim),
        elem_storage, BufferRole.BATCH_INPUT, "storage",
    )
    mask_words = (batch_size + 31) // 32
    b_sample_mask = alloc.allocate(
        "sample_mask",
        (mask_words,),
        4, BufferRole.BATCH_INPUT, None,
    )
    b_targets = alloc.allocate(
        strategy.required_targets_buffer_name,
        (batch_size, model_spec.padded_class_dim),
        4, BufferRole.BATCH_INPUT, "compute",
    )

    # Mask strategy flags (ADR-031)
    flag_explicit = 1 if model_spec.precision.mask_strategy.mode == "explicit" else 0

    # BATCH_INTERMEDIATE
    b_hidden = alloc.allocate(
        "hidden_activations",
        (batch_size, model_spec.padded_hidden_dim),
        elem_storage, BufferRole.BATCH_INTERMEDIATE, "storage",
    )
    # ADR-031 §1.4: stub when recompute, full buffer when explicit
    if flag_explicit:
        b_hidden_mask = alloc.allocate(
            "hidden_mask",
            (batch_size, model_spec.padded_hidden_dim),
            elem_storage, BufferRole.BATCH_INTERMEDIATE, "storage",
        )
    else:
        b_hidden_mask = alloc.allocate(
            "hidden_mask", (1,),
            elem_storage, BufferRole.BATCH_INTERMEDIATE, "storage",
        )
    b_logits = alloc.allocate(
        "full_logits",
        (model_spec.num_modules, batch_size, model_spec.padded_class_dim),
        elem_storage, BufferRole.BATCH_INTERMEDIATE, "storage",
    )
    b_partial_probs = alloc.allocate(
        "partial_probs",
        (tile_count, modules_per_chunk, batch_size, classes_per_chunk),
        elem_storage, BufferRole.BATCH_INTERMEDIATE, "storage",
    )
    b_loss_output = alloc.allocate(
        "loss_output",
        (model_spec.num_modules, batch_size),
        elem_compute, BufferRole.BATCH_INTERMEDIATE, "compute",
    )

    # BATCH_OUTPUT — reduction tree outputs float* (compute precision)
    b_final_probs = alloc.allocate(
        "final_probs",
        (modules_per_chunk, batch_size, classes_per_chunk),
        elem_compute, BufferRole.BATCH_OUTPUT, "compute",
    )

    # --- Nodes ---
    nodes: dict[str, PlanNode] = {}

    # Node 4: forward_pass
    n4 = _dispatch(
        "forward_pass", frozenset(), forward_pass_contract,
        {"input": b_input_data, "sample_mask": b_sample_mask,
         "weights_shared_simd_major": b_shared_weights,
         "biases_shared": b_biases_shared,
         "hidden_activations": b_hidden, "hidden_mask": b_hidden_mask},
        {"batch_chunk_offset": 0, "batch_chunk_count": batch_size,
         "FLAG_produce_hidden_mask": flag_explicit,
         "total_batch_count": batch_size,
         "padded_input_count": model_spec.padded_input_dim,
         "padded_hidden_count": model_spec.padded_hidden_dim},
        tile_count=1, placement_strategy="linear_batch",
    )
    nodes[n4.node_id] = n4
    alloc.set_producer(b_hidden, n4.node_id)
    alloc.set_producer(b_hidden_mask, n4.node_id)
    alloc.add_consumer(b_shared_weights, n4.node_id)
    alloc.add_consumer(b_biases_shared, n4.node_id)
    alloc.add_consumer(b_input_data, n4.node_id)
    alloc.add_consumer(b_sample_mask, n4.node_id)

    # Node 5: render_logits_chunk
    n5 = _dispatch(
        "render_logits", frozenset({"forward_pass"}), render_logits_chunk_contract,
        {"hidden_activations": b_hidden, "hidden_mask": b_hidden_mask,
         "sample_mask": b_sample_mask,
         "weights_module": b_module_weights, "biases_module": b_module_biases,
         "logits": b_logits},
        {"batch_chunk_offset": 0, "batch_chunk_count": batch_size,
         "FLAG_use_explicit_hidden_mask": flag_explicit,
         "module_chunk_offset": 0,
         "module_chunk_count": model_spec.num_modules,
         "class_chunk_offset": 0,
         "class_chunk_count": model_spec.output_classes,
         "total_batch_count": batch_size,
         "hidden_count": model_spec.hidden_dim,
         "padded_hidden_count": model_spec.padded_hidden_dim,
         "total_output_class_count": model_spec.output_classes,
         "padded_total_output_class_count": model_spec.padded_class_dim,
         "total_modules_count": model_spec.num_modules},
        tile_count=tile_count, placement_strategy="grid_mod_cls",
    )
    nodes[n5.node_id] = n5
    alloc.set_producer(b_logits, n5.node_id)
    alloc.add_consumer(b_hidden, n5.node_id)
    alloc.add_consumer(b_hidden_mask, n5.node_id)
    alloc.add_consumer(b_sample_mask, n5.node_id)
    alloc.add_consumer(b_module_weights, n5.node_id)
    alloc.add_consumer(b_module_biases, n5.node_id)

    # Node 6/7: loss computation (CCE or BCE)
    loss_contract = strategy.get_loss_contract()
    loss_buf_key = "final_loss" if "cce" in loss_contract.kernel_name else "partial_loss"
    n_loss = _dispatch(
        "loss_computation", frozenset({"render_logits"}), loss_contract,
        {"logits": b_logits, "temps": b_temperatures,
         "targets": b_targets, "sample_mask": b_sample_mask,
         "partial_probs": b_partial_probs, loss_buf_key: b_loss_output},
        {"flat_tile_index": 0,
         "num_class_chunks": tiling.num_class_chunks,
         "classes_per_chunk": classes_per_chunk,
         "modules_per_chunk": modules_per_chunk,
         "total_batch_count": batch_size,
         "total_output_class_count": model_spec.output_classes,
         "padded_total_output_class_count": model_spec.padded_class_dim,
         "total_modules_count": model_spec.num_modules,
         "total_tile_count": tile_count},
        tile_count=tile_count, placement_strategy="grid_mod_cls",
    )
    nodes[n_loss.node_id] = n_loss
    alloc.set_producer(b_partial_probs, n_loss.node_id)
    alloc.set_producer(b_loss_output, n_loss.node_id)
    alloc.add_consumer(b_logits, n_loss.node_id)
    alloc.add_consumer(b_temperatures, n_loss.node_id)
    alloc.add_consumer(b_targets, n_loss.node_id)
    alloc.add_consumer(b_sample_mask, n_loss.node_id)

    # Node 14: Diagnostic reduction (probs)
    probs_gather = TiledGather(
        scheme=tiling,
        _elements_per_partial=modules_per_chunk * batch_size * classes_per_chunk,
    )
    diag_tree = _build_reduction_tree(
        StabilizationPolicy(
            t_algorithmic=0.0, lambda_=0.0,
            compute_fp_format_max=model_spec.precision.compute_fp_format_max,
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

    # Retrieval: inference probabilities for ensemble averaging by the host.
    # For single-tile geometries (the common case), final_probs contains all
    # per-module probabilities.  Multi-tile (multi class-chunk) scenarios
    # require a cross-tile gather that is not yet implemented; the retrieval
    # exposes the raw (mpc, batch, cpc) buffer so shapes stay consistent.
    _retrieval_logical: tuple[int, ...]
    if tiling.num_class_chunks == 1:
        # Single class chunk: trim padding in the class dimension
        _retrieval_logical = (modules_per_chunk, batch_size, model_spec.output_classes)
    else:
        # Multi class-chunk: expose raw per-chunk shape (cross-chunk
        # concatenation is not yet implemented, so this is the honest
        # representation of what the reduction tree produces).
        _retrieval_logical = (modules_per_chunk, batch_size, classes_per_chunk)
    n_ret = RetrievalNode(
        node_id="inference_retrieval",
        depends_on=frozenset({"diag_reduction"}),
        source_buffer=b_final_probs,
        logical_shape=_retrieval_logical,
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
    optimizer: OptimizerConfig | None = None,
    adam_step: int = 1,
) -> ExecutionPlan:
    """Construct a Learn-phase (gradient production → update) plan."""
    alloc = _BufferAllocator()
    elem_storage = model_spec.precision.storage_dtype.itemsize
    elem_compute = model_spec.precision.compute_dtype.itemsize
    elem_state = model_spec.precision.state_dtype.itemsize
    tiling = _make_tiling(model_spec)
    tile_count = tiling.total_tiles
    nodes: dict[str, PlanNode] = {}

    # Tiling geometry (mirrors Act plan)
    modules_per_chunk = (
        (model_spec.num_modules + tiling.num_module_chunks - 1)
        // tiling.num_module_chunks
    )
    classes_per_chunk = (
        (model_spec.output_classes + tiling.num_class_chunks - 1)
        // tiling.num_class_chunks
    )

    # Elements-per-partial for reduction trees
    epp_mod_w = modules_per_chunk * model_spec.padded_hidden_dim * model_spec.padded_class_dim
    epp_mod_b = modules_per_chunk * model_spec.padded_class_dim
    epp_temps = modules_per_chunk
    shared_w_param_count = model_spec.padded_input_dim * model_spec.padded_hidden_dim
    shared_b_param_count = model_spec.padded_hidden_dim
    grad_h_total = batch_size * model_spec.padded_hidden_dim

    # --- MODEL_STATE buffers ---
    b_shared_weights = alloc.allocate(
        "shared_weights",
        (model_spec.padded_input_dim, model_spec.padded_hidden_dim),
        elem_state, BufferRole.MODEL_STATE, "state",
        logical_shape=(model_spec.input_dim, model_spec.hidden_dim),
    )
    b_module_weights = alloc.allocate(
        "module_weights",
        (model_spec.num_modules, model_spec.padded_hidden_dim,
         model_spec.padded_class_dim),
        elem_state, BufferRole.MODEL_STATE, "state",
        logical_shape=(model_spec.num_modules, model_spec.hidden_dim,
                       model_spec.output_classes),
    )
    b_module_biases = alloc.allocate(
        "module_biases",
        (model_spec.num_modules, model_spec.padded_class_dim),
        elem_state, BufferRole.MODEL_STATE, "state",
    )
    b_temperatures = alloc.allocate(
        "temperatures", (model_spec.padded_module_dim,),
        elem_state, BufferRole.MODEL_STATE, "state",
    )

    # Adam optimizer state (m1, m2 per parameter group)
    # ADR-030 Step 13.6: Optimizer state at FULL MODEL SIZE, not per-chunk
    full_mod_w_param_count = model_spec.num_modules * model_spec.padded_hidden_dim * model_spec.padded_class_dim
    full_mod_b_param_count = model_spec.num_modules * model_spec.padded_class_dim
    full_temps_param_count = model_spec.num_modules

    b_m1_module = alloc.allocate(
        "m1_module", (full_mod_w_param_count,), elem_state, BufferRole.MODEL_STATE, "state")
    b_m2_module = alloc.allocate(
        "m2_module", (full_mod_w_param_count,), elem_state, BufferRole.MODEL_STATE, "state")
    b_m1_module_biases = alloc.allocate(
        "m1_module_biases", (full_mod_b_param_count,), elem_state, BufferRole.MODEL_STATE, "state")
    b_m2_module_biases = alloc.allocate(
        "m2_module_biases", (full_mod_b_param_count,), elem_state, BufferRole.MODEL_STATE, "state")
    b_m1_temps = alloc.allocate(
        "m1_temps", (full_temps_param_count,), elem_state, BufferRole.MODEL_STATE, "state")
    b_m2_temps = alloc.allocate(
        "m2_temps", (full_temps_param_count,), elem_state, BufferRole.MODEL_STATE, "state")
    b_m1_shared = alloc.allocate(
        "m1_shared", (shared_w_param_count,), elem_state, BufferRole.MODEL_STATE, "state")
    b_m2_shared = alloc.allocate(
        "m2_shared", (shared_w_param_count,), elem_state, BufferRole.MODEL_STATE, "state")
    b_m1_shared_biases = alloc.allocate(
        "m1_shared_biases", (shared_b_param_count,), elem_state, BufferRole.MODEL_STATE, "state")
    b_m2_shared_biases = alloc.allocate(
        "m2_shared_biases", (shared_b_param_count,), elem_state, BufferRole.MODEL_STATE, "state")

    # --- BATCH_INPUT buffers ---
    b_input_data = alloc.allocate(
        "input_data", (batch_size, model_spec.padded_input_dim),
        elem_storage, BufferRole.BATCH_INPUT, "storage",
    )
    mask_words = (batch_size + 31) // 32
    b_sample_mask = alloc.allocate(
        "sample_mask", (mask_words,), 4, BufferRole.BATCH_INPUT, None,
    )
    b_targets = alloc.allocate(
        strategy.required_targets_buffer_name,
        (batch_size, model_spec.padded_class_dim),
        4, BufferRole.BATCH_INPUT, "compute",
    )

    # Mask strategy flags (ADR-031)
    flag_explicit = 1 if model_spec.precision.mask_strategy.mode == "explicit" else 0

    # --- Upstream intermediate buffers (recomputed in Learn phase) ---
    b_hidden = alloc.allocate(
        "hidden_activations", (batch_size, model_spec.padded_hidden_dim),
        elem_storage, BufferRole.BATCH_INTERMEDIATE, "storage",
    )
    # ADR-031 §1.4: stub when recompute, full buffer when explicit
    if flag_explicit:
        b_hidden_mask = alloc.allocate(
            "hidden_mask",
            (batch_size, model_spec.padded_hidden_dim),
            elem_storage, BufferRole.BATCH_INTERMEDIATE, "storage",
        )
    else:
        b_hidden_mask = alloc.allocate(
            "hidden_mask", (1,),
            elem_storage, BufferRole.BATCH_INTERMEDIATE, "storage",
        )
    b_logits = alloc.allocate(
        "full_logits",
        (model_spec.num_modules, batch_size, model_spec.padded_class_dim),
        elem_storage, BufferRole.BATCH_INTERMEDIATE, "storage",
    )
    b_partial_probs = alloc.allocate(
        "partial_probs",
        (tile_count, modules_per_chunk, batch_size, classes_per_chunk),
        elem_storage, BufferRole.BATCH_INTERMEDIATE, "storage",
    )
    b_loss_output = alloc.allocate(
        "loss_output",
        (model_spec.num_modules, batch_size),
        elem_compute, BufferRole.BATCH_INTERMEDIATE, "compute",
    )
    b_biases_shared = alloc.allocate(
        "biases_shared",
        (model_spec.padded_hidden_dim,),
        elem_state, BufferRole.MODEL_STATE, "state",
    )

    # --- Phase I intermediate buffers ---
    b_partial_grad_weights_module = alloc.allocate(
        "partial_grad_weights_module",
        (tile_count, modules_per_chunk,
         model_spec.padded_hidden_dim, model_spec.padded_class_dim),
        elem_storage, BufferRole.BATCH_INTERMEDIATE, "storage",
    )
    b_partial_grad_biases_module = alloc.allocate(
        "partial_grad_biases_module",
        (tile_count, modules_per_chunk, model_spec.padded_class_dim),
        elem_storage, BufferRole.BATCH_INTERMEDIATE, "storage",
    )
    b_partial_grad_hidden = alloc.allocate(
        "partial_grad_hidden_activations_aos",
        (tile_count, modules_per_chunk,
         batch_size, model_spec.padded_hidden_dim),
        elem_storage, BufferRole.BATCH_INTERMEDIATE, "storage",
    )
    b_partial_grad_temps = alloc.allocate(
        "partial_grad_temps",
        (tile_count, modules_per_chunk),
        elem_storage, BufferRole.BATCH_INTERMEDIATE, "storage",
    )

    # Clipped partials
    b_clipped_grad_weights_module = alloc.allocate(
        "clipped_partial_grad_weights_module",
        (tile_count, modules_per_chunk,
         model_spec.padded_hidden_dim, model_spec.padded_class_dim),
        elem_storage, BufferRole.BATCH_INTERMEDIATE, "storage",
    )
    b_clipped_grad_biases_module = alloc.allocate(
        "clipped_partial_grad_biases_module",
        (tile_count, modules_per_chunk, model_spec.padded_class_dim),
        elem_storage, BufferRole.BATCH_INTERMEDIATE, "storage",
    )
    b_clipped_grad_hidden = alloc.allocate(
        "clipped_partial_grad_hidden_activations_aos",
        (tile_count, modules_per_chunk,
         batch_size, model_spec.padded_hidden_dim),
        elem_storage, BufferRole.BATCH_INTERMEDIATE, "storage",
    )
    b_clipped_grad_temps = alloc.allocate(
        "clipped_partial_grad_temps",
        (tile_count, modules_per_chunk),
        elem_storage, BufferRole.BATCH_INTERMEDIATE, "storage",
    )

    # Phase II outputs
    b_permuted_grad_h = alloc.allocate(
        "clipped_grad_hidden_activations_permuted_soa",
        (batch_size * model_spec.padded_hidden_dim,
         model_spec.padded_module_dim),
        elem_storage, BufferRole.BATCH_INTERMEDIATE, "storage",
    )
    b_summed_grad_h = alloc.allocate(
        "summed_grad_hidden_activations",
        (grad_h_total,),
        elem_compute, BufferRole.BATCH_INTERMEDIATE, "compute",
    )
    b_summed_grad_mod = alloc.allocate(
        "summed_grad_mod", (epp_mod_w,),
        elem_compute, BufferRole.BATCH_INTERMEDIATE, "compute",
    )
    b_summed_grad_mod_biases = alloc.allocate(
        "summed_grad_mod_biases", (epp_mod_b,),
        elem_compute, BufferRole.BATCH_INTERMEDIATE, "compute",
    )
    b_summed_grad_temps = alloc.allocate(
        "summed_grad_temps", (epp_temps,),
        elem_compute, BufferRole.BATCH_INTERMEDIATE, "compute",
    )

    # Phase III intermediates
    b_partial_grad_sw = alloc.allocate(
        "partial_grad_weights_shared",
        (batch_size, model_spec.padded_input_dim,
         model_spec.padded_hidden_dim),
        elem_storage, BufferRole.BATCH_INTERMEDIATE, "storage",
    )
    b_partial_grad_sb = alloc.allocate(
        "partial_grad_biases_shared",
        (batch_size, model_spec.padded_hidden_dim),
        elem_storage, BufferRole.BATCH_INTERMEDIATE, "storage",
    )
    b_clipped_grad_sw = alloc.allocate(
        "clipped_partial_grad_weights_shared",
        (batch_size, shared_w_param_count),
        elem_storage, BufferRole.BATCH_INTERMEDIATE, "storage",
    )
    b_clipped_grad_sb = alloc.allocate(
        "clipped_partial_grad_biases_shared",
        (batch_size, shared_b_param_count),
        elem_storage, BufferRole.BATCH_INTERMEDIATE, "storage",
    )

    # Phase IV outputs
    b_summed_grad_shared = alloc.allocate(
        "summed_grad_shared", (shared_w_param_count,),
        elem_compute, BufferRole.BATCH_INTERMEDIATE, "compute",
    )
    b_summed_grad_shared_biases = alloc.allocate(
        "summed_grad_shared_biases", (shared_b_param_count,),
        elem_compute, BufferRole.BATCH_INTERMEDIATE, "compute",
    )
    b_final_grad_mod = alloc.allocate(
        "final_grad_mod", (epp_mod_w,),
        elem_compute, BufferRole.BATCH_INTERMEDIATE, "compute",
    )
    b_final_grad_mod_biases = alloc.allocate(
        "final_grad_mod_biases", (epp_mod_b,),
        elem_compute, BufferRole.BATCH_INTERMEDIATE, "compute",
    )
    b_final_grad_temps = alloc.allocate(
        "final_grad_temps", (epp_temps,),
        elem_compute, BufferRole.BATCH_INTERMEDIATE, "compute",
    )
    b_final_grad_shared = alloc.allocate(
        "final_grad_shared", (shared_w_param_count,),
        elem_compute, BufferRole.BATCH_INTERMEDIATE, "compute",
    )
    b_final_grad_shared_biases = alloc.allocate(
        "final_grad_shared_biases", (shared_b_param_count,),
        elem_compute, BufferRole.BATCH_INTERMEDIATE, "compute",
    )

    # BATCH_OUTPUT
    b_final_output = alloc.allocate(
        "final_batch_output", (1,), elem_compute, BufferRole.BATCH_OUTPUT, "compute",
    )

    # =====================================================================
    # Phase 0: Forward recomputation (activation_lifecycle="recompute")
    # =====================================================================

    # Node 4: forward_pass (recompute hidden activations)
    n4 = _dispatch(
        "forward_pass", frozenset(), forward_pass_contract,
        {"input": b_input_data, "sample_mask": b_sample_mask,
         "weights_shared_simd_major": b_shared_weights,
         "biases_shared": b_biases_shared,
         "hidden_activations": b_hidden, "hidden_mask": b_hidden_mask},
        {"batch_chunk_offset": 0, "batch_chunk_count": batch_size,
         "FLAG_produce_hidden_mask": flag_explicit,
         "total_batch_count": batch_size,
         "padded_input_count": model_spec.padded_input_dim,
         "padded_hidden_count": model_spec.padded_hidden_dim},
        tile_count=1, placement_strategy="linear_batch",
    )
    nodes[n4.node_id] = n4
    alloc.set_producer(b_hidden, n4.node_id)
    alloc.set_producer(b_hidden_mask, n4.node_id)
    alloc.add_consumer(b_shared_weights, n4.node_id)
    alloc.add_consumer(b_biases_shared, n4.node_id)
    alloc.add_consumer(b_input_data, n4.node_id)
    alloc.add_consumer(b_sample_mask, n4.node_id)

    # Node 5: render_logits_chunk (recompute logits)
    n5 = _dispatch(
        "render_logits", frozenset({"forward_pass"}), render_logits_chunk_contract,
        {"hidden_activations": b_hidden, "hidden_mask": b_hidden_mask,
         "sample_mask": b_sample_mask,
         "weights_module": b_module_weights, "biases_module": b_module_biases,
         "logits": b_logits},
        {"batch_chunk_offset": 0, "batch_chunk_count": batch_size,
         "FLAG_use_explicit_hidden_mask": flag_explicit,
         "module_chunk_offset": 0,
         "module_chunk_count": model_spec.num_modules,
         "class_chunk_offset": 0,
         "class_chunk_count": model_spec.output_classes,
         "total_batch_count": batch_size,
         "hidden_count": model_spec.hidden_dim,
         "padded_hidden_count": model_spec.padded_hidden_dim,
         "total_output_class_count": model_spec.output_classes,
         "padded_total_output_class_count": model_spec.padded_class_dim,
         "total_modules_count": model_spec.num_modules},
        tile_count=tile_count, placement_strategy="grid_mod_cls",
    )
    nodes[n5.node_id] = n5
    alloc.set_producer(b_logits, n5.node_id)
    alloc.add_consumer(b_hidden, n5.node_id)
    alloc.add_consumer(b_hidden_mask, n5.node_id)
    alloc.add_consumer(b_sample_mask, n5.node_id)
    alloc.add_consumer(b_module_weights, n5.node_id)
    alloc.add_consumer(b_module_biases, n5.node_id)

    # Node 6/7: loss computation (recompute partial_probs)
    loss_contract = strategy.get_loss_contract()
    loss_buf_key = "final_loss" if "cce" in loss_contract.kernel_name else "partial_loss"
    n_loss = _dispatch(
        "loss_computation", frozenset({"render_logits"}), loss_contract,
        {"logits": b_logits, "temps": b_temperatures,
         "targets": b_targets, "sample_mask": b_sample_mask,
         "partial_probs": b_partial_probs, loss_buf_key: b_loss_output},
        {"flat_tile_index": 0,
         "num_class_chunks": tiling.num_class_chunks,
         "classes_per_chunk": classes_per_chunk,
         "modules_per_chunk": modules_per_chunk,
         "total_batch_count": batch_size,
         "total_output_class_count": model_spec.output_classes,
         "padded_total_output_class_count": model_spec.padded_class_dim,
         "total_modules_count": model_spec.num_modules,
         "total_tile_count": tile_count},
        tile_count=tile_count, placement_strategy="grid_mod_cls",
    )
    nodes[n_loss.node_id] = n_loss
    alloc.set_producer(b_partial_probs, n_loss.node_id)
    alloc.set_producer(b_loss_output, n_loss.node_id)
    alloc.add_consumer(b_logits, n_loss.node_id)
    alloc.add_consumer(b_temperatures, n_loss.node_id)
    alloc.add_consumer(b_targets, n_loss.node_id)
    alloc.add_consumer(b_sample_mask, n_loss.node_id)

    # =====================================================================
    # Phase I: Hierarchical Gradient Generation
    # =====================================================================
    prev_deps: frozenset[str] = frozenset({"loss_computation"})

    # Node 8: module grad computation (calculate_module_param_grads_chunk)
    n8 = _dispatch(
        "calc_module_grads", prev_deps, strategy.get_module_grad_contract(),
        {"hidden_activations": b_hidden, "partial_probs": b_partial_probs,
         "targets": b_targets, "sample_mask": b_sample_mask,
         "temps": b_temperatures,
         "partial_grad_weights_module": b_partial_grad_weights_module,
         "partial_grad_biases_module": b_partial_grad_biases_module},
        {"problem_type": strategy.problem_type_flag,
         "flat_tile_index": 0,
         "batch_chunk_offset": 0, "batch_chunk_count": batch_size,
         "num_class_chunks": tiling.num_class_chunks,
         "classes_per_chunk": classes_per_chunk,
         "modules_per_chunk": modules_per_chunk,
         "total_batch_count": batch_size,
         "hidden_count": model_spec.hidden_dim,
         "padded_hidden_count": model_spec.padded_hidden_dim,
         "total_output_class_count": model_spec.output_classes,
         "padded_total_output_class_count": model_spec.padded_class_dim,
         "total_modules_count": model_spec.num_modules,
         "total_tile_count": tile_count},
        tile_count=tile_count, placement_strategy="grid_mod_cls",
    )
    nodes[n8.node_id] = n8
    alloc.set_producer(b_partial_grad_weights_module, n8.node_id)
    alloc.set_producer(b_partial_grad_biases_module, n8.node_id)
    alloc.add_consumer(b_partial_probs, n8.node_id)
    alloc.add_consumer(b_hidden, n8.node_id)
    alloc.add_consumer(b_targets, n8.node_id)
    alloc.add_consumer(b_sample_mask, n8.node_id)
    alloc.add_consumer(b_temperatures, n8.node_id)

    # Node 9: backprop error to hidden (backprop_error_to_hidden_chunk)
    n9 = _dispatch(
        "backprop_error_hidden", prev_deps, strategy.get_hidden_grad_contract(),
        {"partial_probs": b_partial_probs, "targets": b_targets,
         "sample_mask": b_sample_mask,
         "weights_module": b_module_weights,
         "temps": b_temperatures,
         "partial_grad_hidden_activations_aos": b_partial_grad_hidden},
        {"problem_type": strategy.problem_type_flag,
         "flat_tile_index": 0,
         "num_class_chunks": tiling.num_class_chunks,
         "classes_per_chunk": classes_per_chunk,
         "modules_per_chunk": modules_per_chunk,
         "total_batch_count": batch_size,
         "hidden_count": model_spec.hidden_dim,
         "padded_hidden_count": model_spec.padded_hidden_dim,
         "total_output_class_count": model_spec.output_classes,
         "padded_total_output_class_count": model_spec.padded_class_dim,
         "total_modules_count": model_spec.num_modules,
         "total_tile_count": tile_count},
        tile_count=tile_count, placement_strategy="grid_mod_cls",
    )
    nodes[n9.node_id] = n9
    alloc.set_producer(b_partial_grad_hidden, n9.node_id)
    alloc.add_consumer(b_partial_probs, n9.node_id)
    alloc.add_consumer(b_module_weights, n9.node_id)
    alloc.add_consumer(b_targets, n9.node_id)
    alloc.add_consumer(b_sample_mask, n9.node_id)
    alloc.add_consumer(b_temperatures, n9.node_id)

    # Node 10: temperature gradients (calculate_chunk_temp_gradients)
    n10 = _dispatch(
        "calc_temp_grads", prev_deps, strategy.get_temp_grad_contract(),
        {"logits": b_logits, "partial_probs": b_partial_probs,
         "targets": b_targets, "sample_mask": b_sample_mask,
         "temps": b_temperatures,
         "partial_grad_temps": b_partial_grad_temps},
        {"problem_type": strategy.problem_type_flag,
         "flat_tile_index": 0,
         "num_class_chunks": tiling.num_class_chunks,
         "classes_per_chunk": classes_per_chunk,
         "modules_per_chunk": modules_per_chunk,
         "total_batch_count": batch_size,
         "total_output_class_count": model_spec.output_classes,
         "padded_total_output_class_count": model_spec.padded_class_dim,
         "total_modules_count": model_spec.num_modules,
         "total_tile_count": tile_count},
        tile_count=tile_count, placement_strategy="grid_mod_cls",
    )
    nodes[n10.node_id] = n10
    alloc.set_producer(b_partial_grad_temps, n10.node_id)
    alloc.add_consumer(b_partial_probs, n10.node_id)
    alloc.add_consumer(b_logits, n10.node_id)
    alloc.add_consumer(b_targets, n10.node_id)
    alloc.add_consumer(b_sample_mask, n10.node_id)
    alloc.add_consumer(b_temperatures, n10.node_id)

    # Node 11: clip partial gradients (clip_partial_gradients)
    n11 = _dispatch(
        "clip_partial_grads",
        frozenset({"calc_module_grads", "backprop_error_hidden",
                    "calc_temp_grads"}),
        clip_partial_gradients_contract,
        {"partial_grad_weights_module": b_partial_grad_weights_module,
         "partial_grad_biases_module": b_partial_grad_biases_module,
         "partial_grad_temps": b_partial_grad_temps,
         "partial_grad_hidden_activations_aos": b_partial_grad_hidden,
         "clipped_partial_grad_weights_module": b_clipped_grad_weights_module,
         "clipped_partial_grad_biases_module": b_clipped_grad_biases_module,
         "clipped_partial_grad_temps": b_clipped_grad_temps,
         "clipped_partial_grad_hidden_activations_aos": b_clipped_grad_hidden},
        {"use_per_item_norm": 0,
         # Pre-summation amplification factor (CONCEPT §3.4): Node 13 sums across
         # num_class_chunks, so the safety ceiling must be divided by this factor
         # to prevent overflow at Node 16's first stage.
         "clipping_threshold_t_pre": policy.get_leaf_safety_threshold() / tiling.num_class_chunks,
         "epsilon": model_spec.precision.compute_epsilon,
         "flat_tile_index": 0,
         "num_class_chunks": tiling.num_class_chunks,
         "classes_per_chunk": classes_per_chunk,
         "modules_per_chunk": modules_per_chunk,
         "total_batch_count": batch_size,
         "padded_hidden_count": model_spec.padded_hidden_dim,
         "padded_total_output_class_count": model_spec.padded_class_dim,
         "total_tile_count": tile_count},
        tile_count=tile_count, placement_strategy="grid_mod_cls",
    )
    nodes[n11.node_id] = n11
    alloc.set_producer(b_clipped_grad_weights_module, n11.node_id)
    alloc.set_producer(b_clipped_grad_biases_module, n11.node_id)
    alloc.set_producer(b_clipped_grad_hidden, n11.node_id)
    alloc.set_producer(b_clipped_grad_temps, n11.node_id)
    alloc.add_consumer(b_partial_grad_weights_module, n11.node_id)
    alloc.add_consumer(b_partial_grad_biases_module, n11.node_id)
    alloc.add_consumer(b_partial_grad_hidden, n11.node_id)
    alloc.add_consumer(b_partial_grad_temps, n11.node_id)

    # Phase I recompute loop is not needed when full-batch forward
    # computation (Phase 0) produces hidden/logits/probs for the
    # entire batch before gradient nodes execute.

    phase_i_done = frozenset({"clip_partial_grads"})

    # =====================================================================
    # Node 13: gather_and_permute_grad_hidden_activations
    # =====================================================================
    n13 = _dispatch(
        "gather_permute_grad_h", phase_i_done,
        gather_and_permute_grad_h_contract,
        {"clipped_partial_grad_hidden_activations_aos": b_clipped_grad_hidden,
         "clipped_grad_hidden_activations_permuted_soa": b_permuted_grad_h},
        {"total_batch_count": batch_size,
         "hidden_count": model_spec.hidden_dim,
         "padded_hidden_count": model_spec.padded_hidden_dim,
         "total_modules_count": model_spec.num_modules,
         "padded_total_modules_count": model_spec.padded_module_dim,
         "num_module_chunks": tiling.num_module_chunks,
         "modules_per_chunk": modules_per_chunk,
         "num_class_chunks": tiling.num_class_chunks,
         "total_tile_count": tile_count},
        tile_count=1, placement_strategy="linear_batch",
    )
    nodes[n13.node_id] = n13
    alloc.set_producer(b_permuted_grad_h, n13.node_id)
    alloc.add_consumer(b_clipped_grad_hidden, n13.node_id)

    item_sync = BarrierNode(
        node_id="item_sync_barrier",
        depends_on=frozenset({"gather_permute_grad_h"}),
        barrier_name="item_sync",
    )
    nodes[item_sync.node_id] = item_sync

    # =====================================================================
    # Phase II: Specialized & Collective Aggregation
    # =====================================================================

    # Node 16: stabilize_and_reduce_grad_hidden_activations
    n16 = _dispatch(
        "stabilize_reduce_grad_h",
        frozenset({"item_sync_barrier"}),
        stabilize_reduce_grad_h_contract,
        {"grad_hidden_activations_permuted_soa": b_permuted_grad_h,
         "summed_grad_hidden_activations": b_summed_grad_h},
        {"fp_max": model_spec.precision.compute_fp_format_max,
         "policy_t_algorithmic": policy.t_algorithmic,
         "policy_lambda": policy.lambda_,
         "policy_max_k": policy.get_specialized_reduction_policy_k(
             batch_size, hardware.max_reduce_fan_in),
         "epsilon": model_spec.precision.compute_epsilon,
         "total_batch_count": batch_size,
         "padded_hidden_count": model_spec.padded_hidden_dim,
         "total_modules_count": model_spec.num_modules,
         "padded_total_modules_count": model_spec.padded_module_dim},
        tile_count=1, placement_strategy="linear_generic",
    )
    nodes[n16.node_id] = n16
    alloc.set_producer(b_summed_grad_h, n16.node_id)
    alloc.add_consumer(b_permuted_grad_h, n16.node_id)

    # Module weight gradient reduction tree
    mod_gather = TiledGather(
        scheme=tiling, _elements_per_partial=epp_mod_w)
    mod_tree = _build_reduction_tree(
        policy, hardware, mod_gather,
        source_buf=b_clipped_grad_weights_module,
        dest_buf=b_summed_grad_mod,
        tree_variant="sum_and_clip",
    )
    n15_mod = ReductionTreeNode(
        node_id="reduce_mod_grads",
        depends_on=frozenset({"item_sync_barrier"}),
        reduction_plan=mod_tree,
    )
    nodes[n15_mod.node_id] = n15_mod
    alloc.add_consumer(b_clipped_grad_weights_module, n15_mod.node_id)
    alloc.set_producer(b_summed_grad_mod, n15_mod.node_id)

    # Module bias gradient reduction tree
    mod_b_gather = TiledGather(
        scheme=tiling, _elements_per_partial=epp_mod_b)
    mod_b_tree = _build_reduction_tree(
        policy, hardware, mod_b_gather,
        source_buf=b_clipped_grad_biases_module,
        dest_buf=b_summed_grad_mod_biases,
        tree_variant="sum_and_clip",
    )
    n15_mod_b = ReductionTreeNode(
        node_id="reduce_mod_bias_grads",
        depends_on=frozenset({"item_sync_barrier"}),
        reduction_plan=mod_b_tree,
    )
    nodes[n15_mod_b.node_id] = n15_mod_b
    alloc.add_consumer(b_clipped_grad_biases_module, n15_mod_b.node_id)
    alloc.set_producer(b_summed_grad_mod_biases, n15_mod_b.node_id)

    # Temperature gradient reduction tree
    temps_gather = TiledGather(
        scheme=tiling, _elements_per_partial=epp_temps)
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

    # Node 17: backprop_shared_weights_chunk
    n17 = _dispatch(
        "backprop_shared_weights", phase_ii_done,
        backprop_shared_weights_contract,
        {"input": b_input_data,
         "hidden_activations": b_hidden,
         "hidden_mask": b_hidden_mask,
         "summed_grad_hidden_activations": b_summed_grad_h,
         "sample_mask": b_sample_mask,
         "partial_grad_weights_shared": b_partial_grad_sw},
        {"batch_chunk_offset": 0, "batch_chunk_count": 1,
         "FLAG_use_explicit_hidden_mask": flag_explicit,
         "batch_chunk_index": 0,
         "total_batch_count": batch_size,
         "num_batch_chunks": batch_size,
         "padded_input_count": model_spec.padded_input_dim,
         "padded_hidden_count": model_spec.padded_hidden_dim,
         "final_grad_hidden_activations_total_count": grad_h_total},
        tile_count=1, placement_strategy="linear_batch",
    )
    nodes[n17.node_id] = n17
    alloc.set_producer(b_partial_grad_sw, n17.node_id)
    alloc.add_consumer(b_input_data, n17.node_id)
    alloc.add_consumer(b_hidden, n17.node_id)
    alloc.add_consumer(b_hidden_mask, n17.node_id)
    alloc.add_consumer(b_summed_grad_h, n17.node_id)
    alloc.add_consumer(b_sample_mask, n17.node_id)

    # Node 18: backprop_shared_biases_chunk
    n18 = _dispatch(
        "backprop_shared_biases", phase_ii_done,
        backprop_shared_biases_contract,
        {"hidden_activations": b_hidden,
         "hidden_mask": b_hidden_mask,
         "summed_grad_hidden_activations": b_summed_grad_h,
         "sample_mask": b_sample_mask,
         "partial_grad_biases_shared": b_partial_grad_sb},
        {"batch_chunk_offset": 0, "batch_chunk_count": 1,
         "FLAG_use_explicit_hidden_mask": flag_explicit,
         "batch_chunk_index": 0,
         "total_batch_count": batch_size,
         "num_batch_chunks": batch_size,
         "padded_hidden_count": model_spec.padded_hidden_dim,
         "final_grad_hidden_activations_total_count": grad_h_total},
        tile_count=1, placement_strategy="linear_batch",
    )
    nodes[n18.node_id] = n18
    alloc.set_producer(b_partial_grad_sb, n18.node_id)
    alloc.add_consumer(b_hidden, n18.node_id)
    alloc.add_consumer(b_hidden_mask, n18.node_id)
    alloc.add_consumer(b_summed_grad_h, n18.node_id)
    alloc.add_consumer(b_sample_mask, n18.node_id)

    # Node 19: clip_shared_gradients_chunk
    n19 = _dispatch(
        "clip_shared_grads",
        frozenset({"backprop_shared_weights", "backprop_shared_biases"}),
        clip_shared_gradients_contract,
        {"partial_grad_weights_shared": b_partial_grad_sw,
         "partial_grad_biases_shared": b_partial_grad_sb,
         "clipped_partial_grad_weights_shared": b_clipped_grad_sw,
         "clipped_partial_grad_biases_shared": b_clipped_grad_sb},
        {"clipping_threshold_t_pre": policy.get_leaf_safety_threshold(),
         "epsilon": model_spec.precision.compute_epsilon,
         "weights_parameter_count": shared_w_param_count,
         "biases_parameter_count": shared_b_param_count,
         "weights_write_offset": 0,
         "biases_write_offset": 0,
         "num_batch_chunks": batch_size},
        tile_count=1, placement_strategy="linear_batch",
    )
    nodes[n19.node_id] = n19
    alloc.set_producer(b_clipped_grad_sw, n19.node_id)
    alloc.set_producer(b_clipped_grad_sb, n19.node_id)
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
            ParameterStride("batch_chunk_offset", base=0, stride=1),
            ParameterStride("batch_chunk_index", base=0, stride=1),
            ParameterStride("weights_write_offset",
                            base=0, stride=shared_w_param_count),
            ParameterStride("biases_write_offset",
                            base=0, stride=shared_b_param_count),
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

    # Shared weight gradient reduction tree
    shared_w_gather = LinearlyChunkedGather(
        num_chunks=batch_size,
        elements_per_chunk=shared_w_param_count,
    )
    shared_tree = _build_reduction_tree(
        policy, hardware, shared_w_gather,
        source_buf=b_clipped_grad_sw,
        dest_buf=b_summed_grad_shared,
        tree_variant="sum_and_clip",
    )
    n20 = ReductionTreeNode(
        node_id="reduce_shared_grads",
        depends_on=phase_iii_done,
        reduction_plan=shared_tree,
    )
    nodes[n20.node_id] = n20
    alloc.add_consumer(b_clipped_grad_sw, n20.node_id)
    alloc.set_producer(b_summed_grad_shared, n20.node_id)

    # Shared bias gradient reduction tree
    shared_b_gather = LinearlyChunkedGather(
        num_chunks=batch_size,
        elements_per_chunk=shared_b_param_count,
    )
    shared_b_tree = _build_reduction_tree(
        policy, hardware, shared_b_gather,
        source_buf=b_clipped_grad_sb,
        dest_buf=b_summed_grad_shared_biases,
        tree_variant="sum_and_clip",
    )
    n20_b = ReductionTreeNode(
        node_id="reduce_shared_bias_grads",
        depends_on=phase_iii_done,
        reduction_plan=shared_b_tree,
    )
    nodes[n20_b.node_id] = n20_b
    alloc.add_consumer(b_clipped_grad_sb, n20_b.node_id)
    alloc.set_producer(b_summed_grad_shared_biases, n20_b.node_id)

    # Normalize: one call per parameter group
    norm_deps = frozenset({
        "reduce_shared_grads", "reduce_shared_bias_grads",
        "reduce_mod_grads", "reduce_mod_bias_grads",
        "reduce_temp_grads"})

    n21_mod = _dispatch(
        "normalize_gradients_module", norm_deps,
        normalize_gradients_contract,
        {"summed_grad": b_summed_grad_mod,
         "final_grad": b_final_grad_mod},
        {"effective_batch_size": float(batch_size),
         "epsilon": model_spec.precision.compute_epsilon,
         "parameter_count": epp_mod_w},
        tile_count=1, placement_strategy="linear_generic",
    )
    nodes[n21_mod.node_id] = n21_mod
    alloc.set_producer(b_final_grad_mod, n21_mod.node_id)
    alloc.add_consumer(b_summed_grad_mod, n21_mod.node_id)

    n21_mod_b = _dispatch(
        "normalize_gradients_module_biases", norm_deps,
        normalize_gradients_contract,
        {"summed_grad": b_summed_grad_mod_biases,
         "final_grad": b_final_grad_mod_biases},
        {"effective_batch_size": float(batch_size),
         "epsilon": model_spec.precision.compute_epsilon,
         "parameter_count": epp_mod_b},
        tile_count=1, placement_strategy="linear_generic",
    )
    nodes[n21_mod_b.node_id] = n21_mod_b
    alloc.set_producer(b_final_grad_mod_biases, n21_mod_b.node_id)
    alloc.add_consumer(b_summed_grad_mod_biases, n21_mod_b.node_id)

    n21_temps = _dispatch(
        "normalize_gradients_temps", norm_deps,
        normalize_gradients_contract,
        {"summed_grad": b_summed_grad_temps,
         "final_grad": b_final_grad_temps},
        {"effective_batch_size": float(batch_size),
         "epsilon": model_spec.precision.compute_epsilon,
         "parameter_count": epp_temps},
        tile_count=1, placement_strategy="linear_generic",
    )
    nodes[n21_temps.node_id] = n21_temps
    alloc.set_producer(b_final_grad_temps, n21_temps.node_id)
    alloc.add_consumer(b_summed_grad_temps, n21_temps.node_id)

    n21_shared = _dispatch(
        "normalize_gradients_shared", norm_deps,
        normalize_gradients_contract,
        {"summed_grad": b_summed_grad_shared,
         "final_grad": b_final_grad_shared},
        {"effective_batch_size": float(batch_size),
         "epsilon": model_spec.precision.compute_epsilon,
         "parameter_count": shared_w_param_count},
        tile_count=1, placement_strategy="linear_generic",
    )
    nodes[n21_shared.node_id] = n21_shared
    alloc.set_producer(b_final_grad_shared, n21_shared.node_id)
    alloc.add_consumer(b_summed_grad_shared, n21_shared.node_id)

    n21_shared_b = _dispatch(
        "normalize_gradients_shared_biases", norm_deps,
        normalize_gradients_contract,
        {"summed_grad": b_summed_grad_shared_biases,
         "final_grad": b_final_grad_shared_biases},
        {"effective_batch_size": float(batch_size),
         "epsilon": model_spec.precision.compute_epsilon,
         "parameter_count": shared_b_param_count},
        tile_count=1, placement_strategy="linear_generic",
    )
    nodes[n21_shared_b.node_id] = n21_shared_b
    alloc.set_producer(b_final_grad_shared_biases, n21_shared_b.node_id)
    alloc.add_consumer(b_summed_grad_shared_biases, n21_shared_b.node_id)

    # =====================================================================
    # Phase V: Parameter Update
    # =====================================================================

    # Batch Synchronization Barrier
    n22 = BarrierNode(
        node_id="batch_sync_barrier",
        depends_on=frozenset({
            "normalize_gradients_module",
            "normalize_gradients_module_biases",
            "normalize_gradients_temps",
            "normalize_gradients_shared",
            "normalize_gradients_shared_biases",
        }),
        barrier_name="batch_sync",
    )
    nodes[n22.node_id] = n22

    update_deps = frozenset({"batch_sync_barrier"})

    # Optimizer hyperparameters (ADR-029)
    _opt = optimizer if optimizer is not None else OptimizerConfig()
    _adam_scalars = {
        "learning_rate": _opt.learning_rate,
        "beta1_pow_t": _opt.beta1 ** adam_step,
        "beta2_pow_t": _opt.beta2 ** adam_step,
        "beta1": _opt.beta1,
        "beta2": _opt.beta2,
        "epsilon": _opt.resolve_epsilon(model_spec.precision),
    }

    # Adam update: shared weights
    # ADR-030: parameter_offset=0, total_parameter_count=parameter_count (degeneration)
    n24_shared = _dispatch(
        "adam_update_shared", update_deps, adam_update_contract,
        {"final_grad": b_final_grad_shared,
         "parameters": b_shared_weights,
         "m1": b_m1_shared, "m2": b_m2_shared},
        {**_adam_scalars,
         "parameter_offset": 0,
         "parameter_count": shared_w_param_count,
         "total_parameter_count": shared_w_param_count},
        tile_count=1, placement_strategy="linear_generic",
    )
    nodes[n24_shared.node_id] = n24_shared
    alloc.add_consumer(b_shared_weights, n24_shared.node_id)
    alloc.add_consumer(b_final_grad_shared, n24_shared.node_id)
    alloc.add_consumer(b_m1_shared, n24_shared.node_id)
    alloc.add_consumer(b_m2_shared, n24_shared.node_id)

    # Adam update: shared biases
    n24_shared_b = _dispatch(
        "adam_update_shared_biases", update_deps, adam_update_contract,
        {"final_grad": b_final_grad_shared_biases,
         "parameters": b_biases_shared,
         "m1": b_m1_shared_biases, "m2": b_m2_shared_biases},
        {**_adam_scalars,
         "parameter_offset": 0,
         "parameter_count": shared_b_param_count,
         "total_parameter_count": shared_b_param_count},
        tile_count=1, placement_strategy="linear_generic",
    )
    nodes[n24_shared_b.node_id] = n24_shared_b
    alloc.add_consumer(b_biases_shared, n24_shared_b.node_id)
    alloc.add_consumer(b_final_grad_shared_biases, n24_shared_b.node_id)
    alloc.add_consumer(b_m1_shared_biases, n24_shared_b.node_id)
    alloc.add_consumer(b_m2_shared_biases, n24_shared_b.node_id)

    # Adam update: module weights
    # ADR-030: Full model size for module params (currently single-chunk degenerate case)
    full_mod_w_size = model_spec.num_modules * model_spec.padded_hidden_dim * model_spec.padded_class_dim
    n24_module = _dispatch(
        "adam_update_module", update_deps, adam_update_contract,
        {"final_grad": b_final_grad_mod,
         "parameters": b_module_weights,
         "m1": b_m1_module, "m2": b_m2_module},
        {**_adam_scalars,
         "parameter_offset": 0,
         "parameter_count": epp_mod_w,
         "total_parameter_count": full_mod_w_size},
        tile_count=1, placement_strategy="linear_generic",
    )
    nodes[n24_module.node_id] = n24_module
    alloc.add_consumer(b_module_weights, n24_module.node_id)
    alloc.add_consumer(b_final_grad_mod, n24_module.node_id)
    alloc.add_consumer(b_m1_module, n24_module.node_id)
    alloc.add_consumer(b_m2_module, n24_module.node_id)

    # Adam update: module biases
    full_mod_b_size = model_spec.num_modules * model_spec.padded_class_dim
    n24_module_b = _dispatch(
        "adam_update_module_biases", update_deps, adam_update_contract,
        {"final_grad": b_final_grad_mod_biases,
         "parameters": b_module_biases,
         "m1": b_m1_module_biases, "m2": b_m2_module_biases},
        {**_adam_scalars,
         "parameter_offset": 0,
         "parameter_count": epp_mod_b,
         "total_parameter_count": full_mod_b_size},
        tile_count=1, placement_strategy="linear_generic",
    )
    nodes[n24_module_b.node_id] = n24_module_b
    alloc.add_consumer(b_module_biases, n24_module_b.node_id)
    alloc.add_consumer(b_final_grad_mod_biases, n24_module_b.node_id)
    alloc.add_consumer(b_m1_module_biases, n24_module_b.node_id)
    alloc.add_consumer(b_m2_module_biases, n24_module_b.node_id)

    # Adam update: temperatures
    n24_temps = _dispatch(
        "adam_update_temps", update_deps, adam_update_contract,
        {"final_grad": b_final_grad_temps,
         "parameters": b_temperatures,
         "m1": b_m1_temps, "m2": b_m2_temps},
        {**_adam_scalars,
         "parameter_offset": 0,
         "parameter_count": epp_temps,
         "total_parameter_count": model_spec.num_modules},
        tile_count=1, placement_strategy="linear_generic",
    )
    nodes[n24_temps.node_id] = n24_temps
    alloc.add_consumer(b_temperatures, n24_temps.node_id)
    alloc.add_consumer(b_final_grad_temps, n24_temps.node_id)
    alloc.add_consumer(b_m1_temps, n24_temps.node_id)
    alloc.add_consumer(b_m2_temps, n24_temps.node_id)

    # Clamp temperatures
    # ADR-030: parameter_count replaces total_modules_count
    n25 = _dispatch(
        "clamp_temperatures",
        frozenset({"adam_update_temps"}),
        clamp_temperatures_contract,
        {"temperatures": b_temperatures},
        {"min_value": 0.01, "max_value": 100.0,
         "parameter_offset": 0,
         "parameter_count": model_spec.num_modules,
         "total_parameter_count": model_spec.num_modules},
        tile_count=1, placement_strategy="linear_generic",
    )
    nodes[n25.node_id] = n25
    alloc.add_consumer(b_temperatures, n25.node_id)

    # Final retrieval
    n_final = RetrievalNode(
        node_id="final_batch_retrieval",
        depends_on=frozenset({
            "adam_update_shared", "adam_update_shared_biases",
            "adam_update_module", "adam_update_module_biases",
            "clamp_temperatures",
        }),
        source_buffer=b_final_output,
        logical_shape=(1,),
        event_name="final_batch_event",
    )
    nodes[n_final.node_id] = n_final
    alloc.add_consumer(b_final_output, n_final.node_id)

    # Build topological order
    topo_list = [
        "forward_pass", "render_logits", "loss_computation",
        "calc_module_grads", "backprop_error_hidden", "calc_temp_grads",
        "clip_partial_grads",
    ]
    # Full-batch forward recomputation above provides hidden/logits/probs,
    # so the per-sample Phase I recompute loop is not needed.
    topo_list.extend([
        "gather_permute_grad_h", "item_sync_barrier",
        "stabilize_reduce_grad_h",
        "reduce_mod_grads", "reduce_mod_bias_grads", "reduce_temp_grads",
        "backprop_shared_weights", "backprop_shared_biases",
        "clip_shared_grads", "streaming_backprop_loop",
        "reduce_shared_grads", "reduce_shared_bias_grads",
        "normalize_gradients_module", "normalize_gradients_module_biases",
        "normalize_gradients_temps",
        "normalize_gradients_shared", "normalize_gradients_shared_biases",
        "batch_sync_barrier",
        "adam_update_shared", "adam_update_shared_biases",
        "adam_update_module", "adam_update_module_biases",
        "adam_update_temps",
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
