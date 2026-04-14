# src/shared/plan_builder.py
"""Plan construction logic (ADR-002, ADR-003, ADR-004, ADR-009, ADR-030).

Assembles ``ExecutionPlan`` DAGs from ``ModelSpec``, ``HardwareProfile``,
``StabilizationPolicy``, and ``ProblemTypeSpec`` inputs.

Design
------
* **Forward subgraph extraction:** ``_build_forward_subgraph()`` eliminates
  duplication between Act and Learn plans.
* **Parameter-space-driven chains:** ``_build_gradient_chain()`` constructs
  reduce → normalise → adam → post-update for any ``ParameterGroup``,
  replacing per-group boilerplate with systematic iteration over
  ``MODULE_GROUPS`` and ``SHARED_GROUPS``.
* **Structured buffer groups:** ``_ModelStateBuffers``, ``_BatchInputBuffers``
  bundle related handles and provide group-keyed lookup.
* **Streaming loop body isolation:** Body nodes (Nodes 17, 18, 19) are
  constructed as local ``KernelDispatchNode`` instances owned exclusively
  by their parent ``StreamingLoopNode``.  They are *not* registered in
  the top-level ``nodes`` dict, and their ``depends_on`` edges reference
  only sibling body-node IDs.  External gating (e.g. on Node 16) is
  expressed via the loop node's own ``depends_on``.
* **Consistent padding:** Moment vectors and ``total_parameter_count``
  scalars use ``ParameterGeometry.total_flat_elements`` — the physical
  buffer extent including padding — ensuring agreement with the
  MODEL_STATE parameter buffer size.
* **Dispatch validation:** ``_check_dispatch_params`` verifies that every
  ``KernelDispatchNode``'s buffer bindings and scalar parameters match
  its ``KernelContract`` at plan-construction time (Axiom 1.4).

Key invariants
--------------
* **Module-Chunk Isolation (ADR-030):** Parameter-gradient reduction
  trees use ``ModuleChunkGather`` per module chunk.
* **Probs assembly:** Act-plan currently requires ``total_tiles == 1``.
* **Batch chunking:** ``num_batch_chunks`` is fixed at 1 throughout.
* **Leaf safety ceiling:** Pre-reduction clipping thresholds (Nodes 11,
  19) account for the downstream fan-in of the first reduction stage
  to prevent summation-induced overflow (CONCEPT.md §3.4).
* **Identity tier:** When a reduction tree has a single partial (N=1),
  the plan emits a ``ReductionTreePlan`` with ``num_stages=0``,
  signalling the backend to bypass reduction dispatch (CONCEPT.md §2).
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

import numpy as np

from .buffer_lifecycle import BufferDescriptor, BufferHandle, BufferRole
from .hardware_profile import HardwareProfile
from .kernel_contracts import (
    KERNEL_REGISTRY,
    BufferParam,
    KernelContract,
    ScalarParam,
)
from .model_spec import ModelSpec
from .optimizer_config import OptimizerConfig
from .parameter_space import (
    ALL_GROUPS,
    MODULE_GROUPS,
    SHARED_GROUPS,
    SHARED_BIASES,
    SHARED_WEIGHTS,
    ParameterGeometry,
    ParameterGroup,
    resolve_all,
)
from .plan_types import (
    BarrierNode,
    ExecutionPlan,
    KernelDispatchNode,
    PlanNode,
    ReductionTreeNode,
    RetrievalNode,
    StreamingLoopNode,
)
from .precision_config import PrecisionConfig
from .problem_type_spec import ProblemTypeSpec
from .reduction_tree_plan import ReductionTreePlan
from .stabilization_policy import StabilizationPolicy
from .streaming_loop_plan import (
    IterationDimension,
    ParameterStride,
    StreamingLoopPlan,
)
from .workload_primitives import (
    ChunkDecomposition,
    GatherDescriptor,
    ModuleChunkGather,
    ParameterSlice,
    StridedGather,
    TilingGeometry,
)


# =====================================================================
# Internal helpers
# =====================================================================


def _ceildiv(a: int, b: int) -> int:
    """Integer ceiling division.  *b* must be positive."""
    return (a + b - 1) // b


def _compute_leaf_safety_ceiling(
    policy: StabilizationPolicy,
    hardware: HardwareProfile,
    num_partials: int,
    *,
    amplification: int = 1,
) -> float:
    """Safety ceiling for pre-reduction leaf clipping (CONCEPT §3.4).

    Computes ``compute_fp_format_max / (K × A)`` where *K* is the
    fan-in of the first reduction stage and *A* is the pre-summation
    amplification factor (defaults to 1 when there is no implicit
    upstream summation).

    When *num_partials* ≤ 1, no reduction occurs (Identity tier
    bypass) and the ceiling is ``compute_fp_format_max / A``.
    """
    if num_partials > 1:
        fan_in, _ = policy.plan_uniform_reduction_tree(
            num_partials, hardware.max_reduce_fan_in,
        )
    else:
        fan_in = 1
    return policy.compute_fp_format_max / max(1, fan_in * amplification)


def _check_dispatch_params(
    contract: KernelContract,
    buffer_bindings: dict[str, BufferHandle],
    scalar_params: dict[str, int | float],
) -> None:
    """Validate buffer bindings and scalar params against a contract.

    Ensures every non-LOCAL buffer parameter and every scalar parameter
    declared in the ``KernelContract`` has a corresponding entry in the
    provided dicts, and that no extraneous entries are present.

    This is a Policy-tier pre-dispatch check fulfilling Axiom 1.4
    (Collaborative Interface Verifiability).

    Raises
    ------
    ValueError
        On any mismatch between the contract and the provided params.
    """
    expected_buffers = {
        p.name
        for p in contract.params
        if isinstance(p, BufferParam) and p.scope != "LOCAL"
    }
    provided_buffers = set(buffer_bindings)
    if missing := expected_buffers - provided_buffers:
        raise ValueError(
            f"{contract.kernel_name}: missing buffer bindings {missing}"
        )
    if extra := provided_buffers - expected_buffers:
        raise ValueError(
            f"{contract.kernel_name}: unexpected buffer bindings {extra}"
        )

    expected_scalars = {
        p.name for p in contract.params if isinstance(p, ScalarParam)
    }
    provided_scalars = set(scalar_params)
    if missing := expected_scalars - provided_scalars:
        raise ValueError(
            f"{contract.kernel_name}: missing scalars {missing}"
        )
    if extra := provided_scalars - expected_scalars:
        raise ValueError(
            f"{contract.kernel_name}: unexpected scalars {extra}"
        )


# =====================================================================
# Buffer Allocator
# =====================================================================


class _BufferAllocator:
    """Monotonic buffer-handle allocator with lifecycle annotation.

    MODEL_STATE and BATCH_INPUT buffers are never plan-produced — their
    ``producing_node`` remains ``None``, reflecting their origin outside
    the execution plan (host upload or persistent state).
    """

    def __init__(self) -> None:
        self._next_id: int = 0
        self._descs: dict[BufferHandle, BufferDescriptor] = {}
        self._producers: dict[BufferHandle, str | None] = {}
        self._consumers: dict[BufferHandle, set[str]] = {}

    def allocate(
        self,
        logical_name: str,
        padded_shape: tuple[int, ...],
        element_size_bytes: int,
        role: BufferRole,
        precision_role: Literal["storage", "compute", "state"] | None = "compute",
        *,
        init_contract: Literal[
            "ZERO_REQUIRED", "ZERO_REQUIRED_ADDITIVE", "NOT_REQUIRED",
        ] = "NOT_REQUIRED",
        logical_shape: tuple[int, ...] | None = None,
    ) -> BufferHandle:
        handle = BufferHandle(self._next_id)
        self._next_id += 1
        size_bytes = int(np.prod(padded_shape)) * element_size_bytes
        self._descs[handle] = BufferDescriptor(
            handle=handle,
            logical_name=logical_name,
            padded_shape=padded_shape,
            element_size_bytes=element_size_bytes,
            size_bytes=size_bytes,
            role=role,
            precision_role=precision_role,
            init_contract=init_contract,
            logical_shape=logical_shape,
            producing_node=None,
            consumers=frozenset(),
            last_consumer=None,
        )
        self._producers[handle] = None
        self._consumers[handle] = set()
        return handle

    def set_producer(self, handle: BufferHandle, node_id: str) -> None:
        prior = self._producers.get(handle)
        if prior is not None:
            raise AssertionError(
                f"Buffer {self._descs[handle].logical_name!r} already has "
                f"producer {prior!r}; cannot reassign to {node_id!r}"
            )
        self._producers[handle] = node_id

    def add_consumer(self, handle: BufferHandle, node_id: str) -> None:
        self._consumers[handle].add(node_id)

    def finalize(
        self, execution_order: tuple[str, ...],
    ) -> dict[BufferHandle, BufferDescriptor]:
        """Resolve lifecycle annotations into final ``BufferDescriptor`` s.

        Consumer node IDs are resolved against *execution_order* to
        determine ``last_consumer``.  A ``KeyError`` is raised if any
        consumer ID is missing from the execution order — this catches
        body-node IDs that were erroneously registered as top-level
        consumers.
        """
        order_index = {nid: i for i, nid in enumerate(execution_order)}
        result: dict[BufferHandle, BufferDescriptor] = {}
        for handle, desc in self._descs.items():
            consumers = frozenset(self._consumers[handle])
            last: str | None = None
            if consumers:
                # Hard lookup — KeyError exposes misregistered consumer IDs.
                last = max(consumers, key=lambda c: order_index[c])
            result[handle] = BufferDescriptor(
                handle=desc.handle,
                logical_name=desc.logical_name,
                padded_shape=desc.padded_shape,
                element_size_bytes=desc.element_size_bytes,
                size_bytes=desc.size_bytes,
                role=desc.role,
                precision_role=desc.precision_role,
                init_contract=desc.init_contract,
                logical_shape=desc.logical_shape,
                producing_node=self._producers[handle],
                consumers=consumers,
                last_consumer=last,
            )
        return result


# =====================================================================
# Structured Buffer Groups
# =====================================================================


@dataclass(frozen=True)
class _ModelStateBuffers:
    """Handles for the five MODEL_STATE parameter buffers."""

    shared_weights: BufferHandle  # SoA layout
    biases_shared: BufferHandle
    module_weights: BufferHandle
    module_biases: BufferHandle
    temperatures: BufferHandle

    def for_group(self, group: ParameterGroup) -> BufferHandle:
        """Look up the MODEL_STATE buffer for *group*."""
        return {
            "shared_weights": self.shared_weights,
            "shared_biases": self.biases_shared,
            "module_weights": self.module_weights,
            "module_biases": self.module_biases,
            "temperatures": self.temperatures,
        }[group.name]


@dataclass(frozen=True)
class _BatchInputBuffers:
    """Handles for per-batch host-uploaded input buffers."""

    input_data: BufferHandle
    sample_mask: BufferHandle
    targets: BufferHandle


@dataclass(frozen=True)
class _ForwardSubgraph:
    """Captured result of ``_build_forward_subgraph``.

    ``loss_output`` is retained for structural completeness — the loss
    buffer is allocated and written by the fused loss kernel (Node 6/7)
    as a side effect of probability production.  No downstream node in
    either the Act or Learn plan reads from it; it is architecturally
    write-only.  The allocation is necessary because the kernel requires
    a valid destination buffer regardless of whether its output is
    consumed.
    """

    node_ids: tuple[str, ...]
    hidden: BufferHandle
    hidden_mask: BufferHandle
    logits: BufferHandle
    partial_probs: BufferHandle
    loss_output: BufferHandle
    loss_node_id: str


# =====================================================================
# Allocation Helpers
# =====================================================================


def _allocate_model_state(
    alloc: _BufferAllocator,
    spec: ModelSpec,
    simd_w: int,
) -> _ModelStateBuffers:
    """Allocate MODEL_STATE parameter buffers for all five groups."""
    st = spec.precision.state_dtype.itemsize
    return _ModelStateBuffers(
        shared_weights=alloc.allocate(
            "shared_weights",
            (spec.padded_hidden_dim // simd_w, spec.padded_input_dim, simd_w),
            st, BufferRole.MODEL_STATE, "state",
            logical_shape=(spec.hidden_dim, spec.input_dim),
        ),
        biases_shared=alloc.allocate(
            "biases_shared", (spec.padded_hidden_dim,),
            st, BufferRole.MODEL_STATE, "state",
        ),
        module_weights=alloc.allocate(
            "module_weights",
            (spec.num_modules, spec.padded_hidden_dim, spec.padded_class_dim),
            st, BufferRole.MODEL_STATE, "state",
        ),
        module_biases=alloc.allocate(
            "module_biases",
            (spec.num_modules, spec.padded_class_dim),
            st, BufferRole.MODEL_STATE, "state",
        ),
        temperatures=alloc.allocate(
            "temperatures", (spec.padded_module_dim,),
            st, BufferRole.MODEL_STATE, "state",
        ),
    )


def _allocate_batch_inputs(
    alloc: _BufferAllocator,
    spec: ModelSpec,
    strategy: ProblemTypeSpec,
    batch_size: int,
) -> _BatchInputBuffers:
    """Allocate BATCH_INPUT buffers for inputs, mask, and targets.

    Target buffer shape and element size are determined by the
    ``ProblemTypeSpec``, which encapsulates the CCE/BCE divergence
    (CONTRACT Article 8 §7.0).
    """
    es = spec.precision.storage_dtype.itemsize
    return _BatchInputBuffers(
        input_data=alloc.allocate(
            "input_data", (batch_size, spec.padded_input_dim),
            es, BufferRole.BATCH_INPUT, "storage",
        ),
        sample_mask=alloc.allocate(
            "sample_mask", (_ceildiv(batch_size, 32),),
            4, BufferRole.BATCH_INPUT, None,
        ),
        targets=alloc.allocate(
            strategy.targets_logical_name,
            strategy.targets_shape(
                batch_size=batch_size,
                padded_class_dim=spec.padded_class_dim,
            ),
            strategy.targets_element_size(es),
            BufferRole.BATCH_INPUT,
            strategy.targets_precision_role,
        ),
    )


def _allocate_moments(
    alloc: _BufferAllocator,
    geos: dict[str, ParameterGeometry],
    elem_st: int,
) -> dict[str, BufferHandle]:
    """Allocate Adam m1/m2 moment vectors for all parameter groups.

    Uses ``geo.total_flat_elements`` (the physical buffer extent
    **including padding**) so that moment-vector allocation matches
    the parameter buffer size.  Padding positions stay zero by the
    inductive invariant (CONCEPT.md §3.6, Node 24 Padding
    Zero-Preservation); the adam kernel processes the full padded
    range ``[parameter_offset, parameter_offset + parameter_count)``
    without distinguishing padding from logical positions.

    NOTE: ``ParameterGeometry.optimizer_state_elements`` for the
    ``temperatures`` group is ``num_modules`` (the *logical* extent),
    which is narrower than ``total_flat_elements`` (the *padded*
    extent ``padded_module_dim``).  This allocator uses the latter
    so that ``total_parameter_count`` passed to the adam kernel
    equals the state buffer extent.  ``parameter_space.py`` should
    be updated to align ``optimizer_state_elements`` with the
    padded allocation.
    """
    result: dict[str, BufferHandle] = {}
    for group in ALL_GROUPS:
        n = geos[group.name].total_flat_elements
        result[group.m1_name] = alloc.allocate(
            group.m1_name, (n,), elem_st, BufferRole.MODEL_STATE, "state",
        )
        result[group.m2_name] = alloc.allocate(
            group.m2_name, (n,), elem_st, BufferRole.MODEL_STATE, "state",
        )
    return result


# =====================================================================
# Tiling
# =====================================================================


def _derive_tiling(
    spec: ModelSpec,
    hardware: HardwareProfile,
) -> TilingGeometry:
    """Build a ``TilingGeometry`` from model and hardware constraints.

    The heuristic selects chunk sizes up to ``simd_width``, capped by
    the dimension's extent.  This can produce multi-tile geometries
    when ``num_modules`` *and* ``output_classes`` both exceed their
    respective chunk sizes.  Multi-tile Act plans are not yet
    supported (per CONCEPT.md §1, Architectural Elegance Feedback);
    ``build_act_plan`` rejects them at construction time.
    """
    base = max(1, hardware.simd_width)
    mc = min(base, max(1, spec.num_modules))
    cc = min(base, max(1, spec.output_classes))
    return TilingGeometry(
        modules=ChunkDecomposition(
            total=spec.num_modules,
            num_chunks=max(1, _ceildiv(spec.num_modules, mc)),
        ),
        classes=ChunkDecomposition(
            total=spec.output_classes,
            num_chunks=max(1, _ceildiv(spec.output_classes, cc)),
        ),
    )


# =====================================================================
# Dispatch Helper
# =====================================================================


def _dispatch(
    node_id: str,
    depends_on: frozenset[str],
    contract: KernelContract,
    buffer_bindings: dict[str, BufferHandle],
    scalar_params: dict[str, int | float],
    tile_count: int,
    placement_strategy: str | None = None,
) -> KernelDispatchNode:
    """Create a ``KernelDispatchNode`` with pre-dispatch validation.

    Validates buffer bindings and scalar parameters against the
    contract before constructing the node (Axiom 1.4).
    """
    _check_dispatch_params(contract, buffer_bindings, scalar_params)
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


# =====================================================================
# Forward Subgraph Builder (shared between Act and Learn plans)
# =====================================================================


def _build_forward_subgraph(
    alloc: _BufferAllocator,
    nodes: dict[str, PlanNode],
    spec: ModelSpec,
    strategy: ProblemTypeSpec,
    batch_size: int,
    tiling: TilingGeometry,
    state: _ModelStateBuffers,
    inputs: _BatchInputBuffers,
    upstream_deps: frozenset[str] = frozenset(),
) -> _ForwardSubgraph:
    """Build Nodes 4 → 5 → 6/7 and return intermediate buffer handles.

    The fused loss kernel (Node 6/7) co-produces probabilities and loss.
    Loss output is architecturally write-only — no downstream kernel in
    either the Act or Learn plan consumes it.  The allocation and
    dispatch are necessary because the kernel requires a valid
    destination buffer regardless of consumption.

    All CCE/BCE structural divergence (loss shape, init contract,
    binding key) is delegated to ``ProblemTypeSpec``.
    """
    prec = spec.precision
    es, ec = prec.storage_dtype.itemsize, prec.compute_dtype.itemsize
    tc = tiling.total_tiles
    mpc = tiling.modules.chunk_size
    cpc = tiling.classes.chunk_size
    flag_mask = 1 if prec.mask_strategy == "explicit" else 0

    # ── Intermediate buffers ──────────────────────────────────────

    b_hidden = alloc.allocate(
        "hidden_activations", (batch_size, spec.padded_hidden_dim),
        es, BufferRole.BATCH_INTERMEDIATE, "storage",
    )
    b_mask = alloc.allocate(
        "hidden_mask",
        (batch_size, spec.padded_hidden_dim) if flag_mask else (1,),
        es, BufferRole.BATCH_INTERMEDIATE, "storage",
    )
    b_logits = alloc.allocate(
        "full_logits",
        (spec.num_modules, batch_size, spec.padded_class_dim),
        es, BufferRole.BATCH_INTERMEDIATE, "storage",
    )
    b_probs = alloc.allocate(
        "partial_probs", (tc, mpc, batch_size, cpc),
        es, BufferRole.BATCH_INTERMEDIATE, "storage",
    )

    # Loss buffer shape, init contract, and binding key are all
    # determined by the operating mode via ProblemTypeSpec.
    b_loss = alloc.allocate(
        "loss_output",
        strategy.loss_shape(
            num_modules=spec.num_modules,
            batch_size=batch_size,
            tile_count=tc,
            modules_per_chunk=mpc,
        ),
        ec, BufferRole.BATCH_INTERMEDIATE, "compute",
        init_contract=strategy.loss_init_contract,
    )

    # ── Node 4: forward_pass ─────────────────────────────────────

    n4 = "forward_pass"
    nodes[n4] = _dispatch(
        n4, upstream_deps, KERNEL_REGISTRY[n4],
        {
            "src_buffer_GLOBAL_input": inputs.input_data,
            "src_buffer_GLOBAL_sample_mask": inputs.sample_mask,
            "src_buffer_GLOBAL_CONST_weights_shared_simd_major": state.shared_weights,
            "src_buffer_GLOBAL_CONST_biases_shared": state.biases_shared,
            "dest_buffer_GLOBAL_hidden_activations": b_hidden,
            "dest_buffer_GLOBAL_hidden_mask": b_mask,
        },
        {
            "src_scalar_NATURAL_batch_chunk_offset": 0,
            "src_scalar_NATURAL_batch_chunk_count": batch_size,
            "out_scalar_FLAG_produce_hidden_mask": flag_mask,
            "src_scalar_NATURAL_total_batch_count": batch_size,
            "src_scalar_NATURAL_input_count": spec.input_dim,
            "src_scalar_NATURAL_padded_input_count": spec.padded_input_dim,
            "src_scalar_NATURAL_padded_hidden_count": spec.padded_hidden_dim,
        },
        tile_count=1,
    )
    alloc.set_producer(b_hidden, n4)
    alloc.set_producer(b_mask, n4)
    for h in (state.shared_weights, state.biases_shared,
              inputs.input_data, inputs.sample_mask):
        alloc.add_consumer(h, n4)

    # ── Node 5: render_logits_chunk ──────────────────────────────

    n5 = "render_logits_chunk"
    nodes[n5] = _dispatch(
        n5, frozenset({n4}), KERNEL_REGISTRY[n5],
        {
            "src_buffer_GLOBAL_hidden_activations": b_hidden,
            "src_buffer_GLOBAL_hidden_mask": b_mask,
            "src_buffer_GLOBAL_sample_mask": inputs.sample_mask,
            "src_buffer_GLOBAL_CONST_weights_module": state.module_weights,
            "src_buffer_GLOBAL_CONST_biases_module": state.module_biases,
            "dest_buffer_GLOBAL_logits": b_logits,
        },
        {
            "src_scalar_NATURAL_batch_chunk_offset": 0,
            "src_scalar_NATURAL_batch_chunk_count": batch_size,
            "src_scalar_FLAG_use_explicit_hidden_mask": flag_mask,
            "src_scalar_NATURAL_module_chunk_offset": 0,
            "src_scalar_NATURAL_module_chunk_count": spec.num_modules,
            "src_scalar_NATURAL_class_chunk_offset": 0,
            "src_scalar_NATURAL_class_chunk_count": spec.output_classes,
            "src_scalar_NATURAL_total_batch_count": batch_size,
            "src_scalar_NATURAL_hidden_count": spec.hidden_dim,
            "src_scalar_NATURAL_padded_hidden_count": spec.padded_hidden_dim,
            "src_scalar_NATURAL_total_output_class_count": spec.output_classes,
            "src_scalar_NATURAL_padded_total_output_class_count": spec.padded_class_dim,
            "src_scalar_NATURAL_total_modules_count": spec.num_modules,
        },
        tile_count=tc,
    )
    alloc.set_producer(b_logits, n5)
    for h in (b_hidden, b_mask, inputs.sample_mask,
              state.module_weights, state.module_biases):
        alloc.add_consumer(h, n5)

    # ── Node 6/7: loss computation ───────────────────────────────
    #
    # The loss kernel identity and binding key are determined by the
    # operating mode (ProblemTypeSpec).  Nodes 6 (CCE) and 7 (BCE)
    # are the Principle 3(B) bifurcation point — separate kernels
    # with incompatible output topologies.

    loss_c = strategy.loss_contract
    loss_id = loss_c.kernel_name
    nodes[loss_id] = _dispatch(
        loss_id, frozenset({n5}), loss_c,
        {
            "src_buffer_GLOBAL_logits": b_logits,
            "src_buffer_GLOBAL_CONST_temps": state.temperatures,
            "src_buffer_GLOBAL_targets": inputs.targets,
            "src_buffer_GLOBAL_sample_mask": inputs.sample_mask,
            "dest_buffer_GLOBAL_partial_probs": b_probs,
            strategy.loss_binding_key: b_loss,
        },
        {
            "src_scalar_NATURAL_flat_tile_index": 0,
            "src_scalar_NATURAL_num_class_chunks": tiling.classes.num_chunks,
            "src_scalar_NATURAL_classes_per_chunk": cpc,
            "src_scalar_NATURAL_modules_per_chunk": mpc,
            "src_scalar_NATURAL_total_batch_count": batch_size,
            "src_scalar_NATURAL_total_output_class_count": spec.output_classes,
            "src_scalar_NATURAL_padded_total_output_class_count": spec.padded_class_dim,
            "src_scalar_NATURAL_total_modules_count": spec.num_modules,
            "src_scalar_NATURAL_total_tile_count": tc,
        },
        tile_count=tc,
    )
    alloc.set_producer(b_probs, loss_id)
    alloc.set_producer(b_loss, loss_id)
    for h in (b_logits, state.temperatures, inputs.targets, inputs.sample_mask):
        alloc.add_consumer(h, loss_id)

    return _ForwardSubgraph(
        node_ids=(n4, n5, loss_id),
        hidden=b_hidden, hidden_mask=b_mask, logits=b_logits,
        partial_probs=b_probs, loss_output=b_loss,
        loss_node_id=loss_id,
    )


# =====================================================================
# Reduction Tree Builder
# =====================================================================


def _build_reduction_plan(
    policy: StabilizationPolicy,
    hardware: HardwareProfile,
    gather: GatherDescriptor,
    source_buf: BufferHandle,
    dest_buf: BufferHandle,
    tree_variant: Literal["sum", "sum_and_clip"],
) -> ReductionTreePlan:
    """Construct a ``ReductionTreePlan`` from a gather descriptor.

    When ``num_partials == 1`` (Identity tier, CONCEPT.md §2), the
    plan is emitted with ``num_stages=0`` and ``fan_in=1``, signalling
    the backend to bypass reduction dispatch and reference the single
    partial directly (with a precision-bridge copy when source and
    destination roles differ).

    Raises ``ValueError`` when ``num_partials < 1`` — a reduction tree
    over zero partials is structurally nonsensical.
    """
    n = gather.num_partials
    if n < 1:
        raise ValueError(
            f"Cannot build reduction plan for {n} partials (need ≥ 1)"
        )

    offsets = tuple(int(o) for o in gather.offsets())

    # Identity tier: single partial, no reduction needed.
    if n == 1:
        return ReductionTreePlan(
            num_partials=1,
            fan_in=1,
            num_stages=0,
            initial_offset_list=offsets,
            tree_variant=tree_variant,
            threshold_schedule=(),
            partial_width=gather.elements_per_partial,
            source_buffer=source_buf,
            destination_buffer=dest_buf,
        )

    fan_in_k, num_stages = policy.plan_uniform_reduction_tree(
        n, hardware.max_reduce_fan_in,
    )
    # Defensive: plan_uniform_reduction_tree should return ≥ 1 for N ≥ 2.
    num_stages = max(1, num_stages)

    schedule: tuple[float, ...] = ()
    if tree_variant == "sum_and_clip":
        schedule = tuple(
            policy.get_threshold_for_generic_stage(j, fan_in_k)
            for j in reversed(range(num_stages))
        )

    return ReductionTreePlan(
        num_partials=n,
        fan_in=fan_in_k,
        num_stages=num_stages,
        initial_offset_list=offsets,
        tree_variant=tree_variant,
        threshold_schedule=schedule,
        partial_width=gather.elements_per_partial,
        source_buffer=source_buf,
        destination_buffer=dest_buf,
    )


# =====================================================================
# Generic Per-Group Gradient Chain Builder
# =====================================================================


@dataclass(frozen=True)
class _GradientChainSpec:
    """Specifies one parameter group's reduce → normalise → adam chain.

    ``buf_elems`` (derived from ``gather.elements_per_partial``) is the
    uniform tile extent — the full per-tile allocation including padding
    for the last module chunk.  The reduction tree, normalisation, and
    gradient buffer are all sized to this extent.  The adam kernel
    operates on a potentially smaller ``param_slice.count`` sub-range;
    trailing zeros at padding positions are harmless (Zero-Propagation
    Theorem inductive step).

    For the last module chunk the gradient buffer may be over-allocated
    relative to the adam kernel's ``parameter_count``.  The kernel reads
    only ``[0, parameter_count)``; trailing padding elements are
    architecturally dead.
    """

    group: ParameterGroup
    geo: ParameterGeometry

    # Reduction input
    clipped_buf: BufferHandle
    gather: GatherDescriptor

    # Optimizer targets
    param_buf: BufferHandle
    m1_buf: BufferHandle
    m2_buf: BufferHandle
    param_slice: ParameterSlice

    # Scoping (None for shared groups)
    mc: int | None = None


def _build_gradient_chain(
    alloc: _BufferAllocator,
    nodes: dict[str, PlanNode],
    spec: _GradientChainSpec,
    policy: StabilizationPolicy,
    hardware: HardwareProfile,
    prec: PrecisionConfig,
    effective_batch_size: int,
    adam_scalars: dict[str, float],
    upstream_deps: frozenset[str],
    temp_bounds: tuple[float, float] | None = None,
) -> tuple[tuple[str, ...], frozenset[str]]:
    """Build reduce → normalise → adam [→ post-update] for one group.

    Returns ``(topo_ordered_node_ids, terminal_node_ids)``.
    """
    group = spec.group
    ec = prec.compute_dtype.itemsize

    # The uniform tile extent (includes padding positions for the
    # last module chunk; trailing zeros are harmless).
    buf_elems = spec.gather.elements_per_partial

    # ── Reduction tree ────────────────────────────────────────────

    rid = group.reduce_node_id(spec.mc)
    b_summed = alloc.allocate(
        group.summed_grad_name(spec.mc), (buf_elems,),
        ec, BufferRole.BATCH_INTERMEDIATE, "compute",
    )
    nodes[rid] = ReductionTreeNode(
        node_id=rid,
        depends_on=upstream_deps,
        reduction_plan=_build_reduction_plan(
            policy, hardware, spec.gather,
            spec.clipped_buf, b_summed, "sum_and_clip",
        ),
    )
    alloc.add_consumer(spec.clipped_buf, rid)
    alloc.set_producer(b_summed, rid)

    # ── Normalisation ─────────────────────────────────────────────

    nid = group.normalize_node_id(spec.mc)
    b_final = alloc.allocate(
        group.final_grad_name(spec.mc), (buf_elems,),
        ec, BufferRole.BATCH_INTERMEDIATE, "compute",
    )
    nodes[nid] = _dispatch(
        nid, frozenset({rid}),
        KERNEL_REGISTRY["normalize_gradients"],
        {
            "src_buffer_GLOBAL_summed_grad": b_summed,
            "dest_buffer_GLOBAL_final_grad": b_final,
        },
        {
            "src_scalar_REAL_effective_batch_size": float(effective_batch_size),
            "src_scalar_REAL_epsilon": prec.compute_epsilon,
            "src_scalar_NATURAL_parameter_count": buf_elems,
        },
        tile_count=1,
    )
    alloc.set_producer(b_final, nid)
    alloc.add_consumer(b_summed, nid)

    # ── Adam update ───────────────────────────────────────────────

    aid = group.adam_node_id(spec.mc)
    nodes[aid] = _dispatch(
        aid, frozenset({nid}),
        KERNEL_REGISTRY["adam_update"],
        {
            "src_buffer_GLOBAL_final_grad": b_final,
            "update_buffer_GLOBAL_parameters": spec.param_buf,
            "update_buffer_GLOBAL_m1": spec.m1_buf,
            "update_buffer_GLOBAL_m2": spec.m2_buf,
        },
        {
            **adam_scalars,
            "src_scalar_NATURAL_parameter_offset": spec.param_slice.offset,
            "src_scalar_NATURAL_parameter_count": spec.param_slice.count,
            "src_scalar_NATURAL_total_parameter_count": spec.param_slice.total,
        },
        tile_count=1,
    )
    alloc.add_consumer(b_final, aid)
    for h in (spec.param_buf, spec.m1_buf, spec.m2_buf):
        alloc.add_consumer(h, aid)

    topo: list[str] = [rid, nid, aid]
    terminals: set[str] = set()

    # ── Optional post-update ──────────────────────────────────────

    if group.has_post_update:
        pid = group.post_update_node_id(spec.mc)

        if group.post_update_kernel == "clamp_temperatures":
            assert temp_bounds is not None, (
                "temp_bounds required for clamp_temperatures"
            )
            nodes[pid] = _dispatch(
                pid, frozenset({aid}),
                KERNEL_REGISTRY["clamp_temperatures"],
                {"update_buffer_GLOBAL_temps": spec.param_buf},
                {
                    "src_scalar_REAL_min_value": temp_bounds[0],
                    "src_scalar_REAL_max_value": temp_bounds[1],
                    "src_scalar_NATURAL_parameter_offset": spec.param_slice.offset,
                    "src_scalar_NATURAL_parameter_count": spec.param_slice.count,
                    "src_scalar_NATURAL_total_parameter_count": spec.param_slice.total,
                },
                tile_count=1,
            )
            alloc.add_consumer(spec.param_buf, pid)
            topo.append(pid)
            terminals.add(pid)
        else:
            raise ValueError(
                f"Unknown post-update kernel: {group.post_update_kernel!r}"
            )
    else:
        terminals.add(aid)

    return tuple(topo), frozenset(terminals)


# =====================================================================
# build_act_plan
# =====================================================================


def build_act_plan(
    model_spec: ModelSpec,
    hardware: HardwareProfile,
    strategy: ProblemTypeSpec,
    batch_size: int,
) -> ExecutionPlan:
    """Construct an Act-phase (forward pass + inference retrieval) plan.

    Currently requires a single-tile configuration because probability
    tiles from different ``(module_chunk, class_chunk)`` regions are
    disjoint, not additive — assembly requires a dedicated scatter
    kernel.  Multi-tile assembly is deferred per CONCEPT.md §1
    (Architectural Elegance Feedback).

    For model configurations that produce a multi-tile geometry,
    either reduce the model dimensions or adjust the hardware
    profile's ``simd_width`` to constrain the tiling.
    """
    alloc = _BufferAllocator()
    tiling = _derive_tiling(model_spec, hardware)

    if tiling.total_tiles > 1:
        raise NotImplementedError(
            f"Multi-tile probability assembly requires a dedicated "
            f"scatter kernel (total_tiles={tiling.total_tiles}).  "
            f"Per CONCEPT.md §1: suspend, formalise, then implement."
        )

    nodes: dict[str, PlanNode] = {}
    state = _allocate_model_state(alloc, model_spec, hardware.simd_width)
    inputs = _allocate_batch_inputs(alloc, model_spec, strategy, batch_size)

    fwd = _build_forward_subgraph(
        alloc, nodes, model_spec, strategy, batch_size,
        tiling, state, inputs,
    )

    # ── Inference retrieval (single-tile identity) ────────────────

    ret_id = "inference_retrieval"
    nodes[ret_id] = RetrievalNode(
        node_id=ret_id,
        depends_on=frozenset({fwd.loss_node_id}),
        source_buffer=fwd.partial_probs,
        logical_shape=(
            model_spec.num_modules, batch_size, model_spec.output_classes,
        ),
        event_name="inference_event",
    )
    alloc.add_consumer(fwd.partial_probs, ret_id)

    topo = (*fwd.node_ids, ret_id)
    return ExecutionPlan(
        nodes=nodes,
        buffers=alloc.finalize(topo),
        execution_order=topo,
        precision=model_spec.precision,
        hardware=hardware,
    )


# =====================================================================
# build_learn_plan
# =====================================================================


def build_learn_plan(
    model_spec: ModelSpec,
    hardware: HardwareProfile,
    strategy: ProblemTypeSpec,
    batch_size: int,
    policy: StabilizationPolicy,
    activation_lifecycle: Literal["cache", "recompute"] = "recompute",
    optimizer: OptimizerConfig | None = None,
    adam_step: int = 1,
    effective_batch_size: int | None = None,
    temp_min: float = 0.01,
    temp_max: float = 100.0,
    streaming_chunk_size: int = 1,
) -> ExecutionPlan:
    """Construct a Learn-phase (gradient production → parameter update) plan.

    Parameters
    ----------
    effective_batch_size:
        Valid (unmasked) sample count for gradient normalisation.
        Defaults to *batch_size* (all valid).
    activation_lifecycle:
        ``"recompute"`` re-runs Nodes 4–7 from stored inputs.
    temp_min, temp_max:
        Temperature clamping bounds for Node 25.
    streaming_chunk_size:
        Number of samples per streaming-loop iteration (Nodes 17–19).
        Defaults to 1 (True Streaming: one sample per iteration).
        Must evenly divide *batch_size* when > 1, because the
        ``StreamingLoopPlan`` uses fixed per-iteration strides and
        cannot express a variable-size tail chunk.  Future revisions
        may lift this restriction via batch-dimension padding or
        variable stride support.
    """
    if activation_lifecycle == "cache":
        raise NotImplementedError(
            "activation_lifecycle='cache' is not yet implemented."
        )

    if streaming_chunk_size < 1:
        raise ValueError(
            f"streaming_chunk_size must be ≥ 1, got {streaming_chunk_size}"
        )
    if streaming_chunk_size > 1 and batch_size % streaming_chunk_size != 0:
        raise NotImplementedError(
            f"streaming_chunk_size={streaming_chunk_size} does not evenly "
            f"divide batch_size={batch_size}.  Either use chunk_size=1 "
            f"(default) or ensure divisibility.  Variable-size tail "
            f"chunks require StreamingLoopPlan extensions."
        )

    alloc = _BufferAllocator()
    spec = model_spec
    prec = spec.precision
    es = prec.storage_dtype.itemsize
    ec = prec.compute_dtype.itemsize
    est = prec.state_dtype.itemsize
    simd_w = hardware.simd_width

    tiling = _derive_tiling(spec, hardware)
    tc = tiling.total_tiles
    num_mc = tiling.modules.num_chunks
    mpc = tiling.modules.chunk_size
    cpc = tiling.classes.chunk_size
    flag_mask = 1 if prec.mask_strategy == "explicit" else 0

    # Batch chunking is fixed at 1.  When > 1 is eventually
    # supported, Node 8's output gains a non-trivial batch-chunk
    # axis and a ReductionTreeNode + optional Precision Bridge
    # collapse it before Node 11.
    num_batch_chunks = 1

    geos = resolve_all(spec, simd_w)
    eff_batch = effective_batch_size if effective_batch_size is not None else batch_size
    grad_h_total = batch_size * spec.padded_hidden_dim

    stream_decomp = ChunkDecomposition.from_chunk_size(
        batch_size, streaming_chunk_size,
    )
    n_stream = stream_decomp.num_chunks
    stream_chunk_sz = stream_decomp.chunk_size

    nodes: dict[str, PlanNode] = {}
    precomputed: dict[BufferHandle, np.ndarray] = {}
    topo: list[str] = []

    # ═══════════════════════════════════════════════════════════════
    # Buffer Allocation
    # ═══════════════════════════════════════════════════════════════

    state = _allocate_model_state(alloc, spec, simd_w)
    inputs = _allocate_batch_inputs(alloc, spec, strategy, batch_size)
    moments = _allocate_moments(alloc, geos, est)

    # ═══════════════════════════════════════════════════════════════
    # Phase 0: Forward Recomputation (Nodes 4 → 5 → 6/7)
    # ═══════════════════════════════════════════════════════════════

    fwd = _build_forward_subgraph(
        alloc, nodes, spec, strategy, batch_size,
        tiling, state, inputs,
    )
    topo.extend(fwd.node_ids)

    # ═══════════════════════════════════════════════════════════════
    # Phase I: Gradient Generation & Joint Clipping (Nodes 8–11)
    # ═══════════════════════════════════════════════════════════════

    grad_gen_deps = frozenset({fwd.loss_node_id})

    # Common scalar dicts for tiled gradient kernels (Nodes 8–10).
    # These kernels are mode-invariant (Principle 3(A)); the FLAG
    # value for internal path selection comes from the strategy.
    _tile_scalars: dict[str, int | float] = {
        "src_scalar_FLAG_problem_type": strategy.flag,
        "src_scalar_NATURAL_flat_tile_index": 0,
        "src_scalar_NATURAL_num_class_chunks": tiling.classes.num_chunks,
        "src_scalar_NATURAL_classes_per_chunk": cpc,
        "src_scalar_NATURAL_modules_per_chunk": mpc,
        "src_scalar_NATURAL_total_batch_count": batch_size,
        "src_scalar_NATURAL_total_output_class_count": spec.output_classes,
        "src_scalar_NATURAL_padded_total_output_class_count": spec.padded_class_dim,
        "src_scalar_NATURAL_total_modules_count": spec.num_modules,
        "src_scalar_NATURAL_total_tile_count": tc,
    }
    _hidden_dims: dict[str, int] = {
        "src_scalar_NATURAL_hidden_count": spec.hidden_dim,
        "src_scalar_NATURAL_padded_hidden_count": spec.padded_hidden_dim,
    }

    # ── Partial gradient buffers (Node 8, 9, 10 outputs) ─────────
    #
    # Node 8's kernel contract declares a 5D shape with a
    # num_batch_chunks axis.  Since num_batch_chunks is fixed at 1,
    # the shape here omits the trivial dimension so the flat buffer
    # matches Node 11's 4D input contract directly.

    assert num_batch_chunks == 1, (
        "num_batch_chunks > 1 requires reshaping Node 8 output to 5D "
        "and inserting a batch-chunk ReductionTreeNode + optional "
        "Precision Bridge before Node 11"
    )

    b_partial_w = alloc.allocate(
        "partial_grad_weights_module",
        (tc, mpc, spec.padded_hidden_dim, spec.padded_class_dim),
        es, BufferRole.BATCH_INTERMEDIATE, "storage",
        init_contract="ZERO_REQUIRED",
    )
    b_partial_b = alloc.allocate(
        "partial_grad_biases_module",
        (tc, mpc, spec.padded_class_dim),
        es, BufferRole.BATCH_INTERMEDIATE, "storage",
        init_contract="ZERO_REQUIRED",
    )
    b_partial_h = alloc.allocate(
        "partial_grad_hidden_activations_aos",
        (tc, mpc, batch_size, spec.padded_hidden_dim),
        es, BufferRole.BATCH_INTERMEDIATE, "storage",
    )
    b_partial_t = alloc.allocate(
        "partial_grad_temps", (tc, mpc),
        es, BufferRole.BATCH_INTERMEDIATE, "storage",
    )

    # ── Clipped partial buffers (Node 11 outputs) ────────────────

    b_clipped_w = alloc.allocate(
        "clipped_partial_grad_weights_module",
        (tc, mpc, spec.padded_hidden_dim, spec.padded_class_dim),
        es, BufferRole.BATCH_INTERMEDIATE, "storage",
    )
    b_clipped_b = alloc.allocate(
        "clipped_partial_grad_biases_module",
        (tc, mpc, spec.padded_class_dim),
        es, BufferRole.BATCH_INTERMEDIATE, "storage",
    )
    b_clipped_h = alloc.allocate(
        "clipped_partial_grad_hidden_activations_aos",
        (tc, mpc, batch_size, spec.padded_hidden_dim),
        es, BufferRole.BATCH_INTERMEDIATE, "storage",
    )
    b_clipped_t = alloc.allocate(
        "clipped_partial_grad_temps", (tc, mpc),
        es, BufferRole.BATCH_INTERMEDIATE, "storage",
    )

    # Stub buffer for the per-item clipping threshold (FLAG=0 ⟹
    # unused).  Registered as precomputed so the Orchestration tier
    # knows to upload valid data; the value is architecturally dead
    # when the per-item flag is not set.
    b_clip_stub = alloc.allocate(
        "clipping_threshold_per_item_stub", (1,),
        ec, BufferRole.BATCH_INTERMEDIATE, "compute",
    )
    precomputed[b_clip_stub] = np.zeros(1, dtype=prec.compute_dtype)

    # Group name → clipped buffer (used by gradient chain builder).
    module_clipped: dict[str, BufferHandle] = {
        "module_weights": b_clipped_w,
        "module_biases": b_clipped_b,
        "temperatures": b_clipped_t,
    }

    # ── Node 11 leaf safety ceiling (CONCEPT §3.4) ───────────────
    #
    # The binding constraint is Node 13's class-chunk amplification:
    # after clipping to T_pre, Node 13 sums C class-chunk tiles per
    # element, amplifying to C × T_pre.  For non-overflow:
    #     C × T_pre ≤ compute_fp_format_max
    #     T_pre ≤ compute_fp_format_max / C
    #
    # The module gradient path (clipped_partial → Node 15 reduction)
    # is independently safe because its reduction fan-in K ≤ C (each
    # module chunk has one tile per class chunk, so the reduction
    # sums at most C tiles): K × T_pre ≤ C × T_pre ≤ cfm.
    num_class_chunks = tiling.classes.num_chunks
    node11_leaf_threshold = (
        policy.compute_fp_format_max / max(1, num_class_chunks)
    )

    # ── Node 8: module parameter gradients ────────────────────────

    n8 = "calculate_module_param_grads_chunk"
    nodes[n8] = _dispatch(
        n8, grad_gen_deps, KERNEL_REGISTRY[n8],
        {
            "src_buffer_GLOBAL_hidden_activations": fwd.hidden,
            "src_buffer_GLOBAL_partial_probs": fwd.partial_probs,
            "src_buffer_GLOBAL_targets": inputs.targets,
            "src_buffer_GLOBAL_sample_mask": inputs.sample_mask,
            "src_buffer_GLOBAL_CONST_temps": state.temperatures,
            "dest_buffer_GLOBAL_partial_grad_weights_module": b_partial_w,
            "dest_buffer_GLOBAL_partial_grad_biases_module": b_partial_b,
        },
        {
            **_tile_scalars, **_hidden_dims,
            "src_scalar_NATURAL_batch_chunk_index": 0,
            "src_scalar_NATURAL_batch_chunk_offset": 0,
            "src_scalar_NATURAL_batch_chunk_count": batch_size,
            "src_scalar_NATURAL_num_batch_chunks": num_batch_chunks,
        },
        tile_count=tc,
    )
    alloc.set_producer(b_partial_w, n8)
    alloc.set_producer(b_partial_b, n8)
    for h in (fwd.hidden, fwd.partial_probs, inputs.targets,
              inputs.sample_mask, state.temperatures):
        alloc.add_consumer(h, n8)

    # ── Node 9: hidden-layer error backpropagation ────────────────

    n9 = "backprop_error_to_hidden_chunk"
    nodes[n9] = _dispatch(
        n9, grad_gen_deps, KERNEL_REGISTRY[n9],
        {
            "src_buffer_GLOBAL_partial_probs": fwd.partial_probs,
            "src_buffer_GLOBAL_targets": inputs.targets,
            "src_buffer_GLOBAL_sample_mask": inputs.sample_mask,
            "src_buffer_GLOBAL_CONST_weights_module": state.module_weights,
            "src_buffer_GLOBAL_CONST_temps": state.temperatures,
            "dest_buffer_GLOBAL_partial_grad_hidden_activations_aos": b_partial_h,
        },
        {**_tile_scalars, **_hidden_dims},
        tile_count=tc,
    )
    alloc.set_producer(b_partial_h, n9)
    for h in (fwd.partial_probs, inputs.targets, inputs.sample_mask,
              state.module_weights, state.temperatures):
        alloc.add_consumer(h, n9)

    # ── Node 10: temperature gradients ────────────────────────────

    n10 = "calculate_chunk_temp_gradients"
    nodes[n10] = _dispatch(
        n10, grad_gen_deps, KERNEL_REGISTRY[n10],
        {
            "src_buffer_GLOBAL_logits": fwd.logits,
            "src_buffer_GLOBAL_partial_probs": fwd.partial_probs,
            "src_buffer_GLOBAL_targets": inputs.targets,
            "src_buffer_GLOBAL_sample_mask": inputs.sample_mask,
            "src_buffer_GLOBAL_CONST_temps": state.temperatures,
            "dest_buffer_GLOBAL_partial_grad_temps": b_partial_t,
        },
        {**_tile_scalars},
        tile_count=tc,
    )
    alloc.set_producer(b_partial_t, n10)
    for h in (fwd.logits, fwd.partial_probs, inputs.targets,
              inputs.sample_mask, state.temperatures):
        alloc.add_consumer(h, n10)

    # ── Node 11: joint clip partial gradients ─────────────────────

    n11 = "clip_partial_gradients"
    nodes[n11] = _dispatch(
        n11, frozenset({n8, n9, n10}), KERNEL_REGISTRY[n11],
        {
            "src_buffer_GLOBAL_partial_grad_weights_module": b_partial_w,
            "src_buffer_GLOBAL_partial_grad_biases_module": b_partial_b,
            "src_buffer_GLOBAL_partial_grad_temps": b_partial_t,
            "src_buffer_GLOBAL_partial_grad_hidden_activations_aos": b_partial_h,
            "src_buffer_GLOBAL_CONST_clipping_threshold_per_item": b_clip_stub,
            "dest_buffer_GLOBAL_clipped_partial_grad_weights_module": b_clipped_w,
            "dest_buffer_GLOBAL_clipped_partial_grad_biases_module": b_clipped_b,
            "dest_buffer_GLOBAL_clipped_partial_grad_temps": b_clipped_t,
            "dest_buffer_GLOBAL_clipped_partial_grad_hidden_activations_aos": b_clipped_h,
        },
        {
            "src_scalar_FLAG_use_per_item_norm": 0,
            "src_scalar_REAL_clipping_threshold_t_pre": node11_leaf_threshold,
            "src_scalar_REAL_epsilon": prec.compute_epsilon,
            "src_scalar_NATURAL_flat_tile_index": 0,
            "src_scalar_NATURAL_num_class_chunks": num_class_chunks,
            "src_scalar_NATURAL_classes_per_chunk": cpc,
            "src_scalar_NATURAL_modules_per_chunk": mpc,
            "src_scalar_NATURAL_total_batch_count": batch_size,
            "src_scalar_NATURAL_padded_hidden_count": spec.padded_hidden_dim,
            "src_scalar_NATURAL_padded_total_output_class_count": spec.padded_class_dim,
            "src_scalar_NATURAL_total_tile_count": tc,
        },
        tile_count=tc,
    )
    for b in (b_clipped_w, b_clipped_b, b_clipped_h, b_clipped_t):
        alloc.set_producer(b, n11)
    for h in (b_partial_w, b_partial_b, b_partial_h,
              b_partial_t, b_clip_stub):
        alloc.add_consumer(h, n11)

    topo.extend([n8, n9, n10, n11])

    # ═══════════════════════════════════════════════════════════════
    # Item Synchronisation & Grad_H Path (Nodes 13 → barrier → 16)
    # ═══════════════════════════════════════════════════════════════

    b_permuted_h = alloc.allocate(
        "clipped_grad_hidden_activations_permuted_soa",
        (batch_size * spec.padded_hidden_dim, spec.padded_module_dim),
        es, BufferRole.BATCH_INTERMEDIATE, "storage",
        init_contract="ZERO_REQUIRED",
    )
    b_summed_h = alloc.allocate(
        "summed_grad_hidden_activations", (grad_h_total,),
        ec, BufferRole.BATCH_INTERMEDIATE, "compute",
    )

    # Node 13
    n13 = "gather_and_permute_grad_hidden_activations"
    nodes[n13] = _dispatch(
        n13, frozenset({n11}), KERNEL_REGISTRY[n13],
        {
            "src_buffer_GLOBAL_clipped_partial_grad_hidden_activations_aos": b_clipped_h,
            "dest_buffer_GLOBAL_clipped_grad_hidden_activations_permuted_soa": b_permuted_h,
        },
        {
            "src_scalar_NATURAL_total_batch_count": batch_size,
            "src_scalar_NATURAL_hidden_count": spec.hidden_dim,
            "src_scalar_NATURAL_padded_hidden_count": spec.padded_hidden_dim,
            "src_scalar_NATURAL_total_modules_count": spec.num_modules,
            "src_scalar_NATURAL_padded_total_modules_count": spec.padded_module_dim,
            "src_scalar_NATURAL_num_module_chunks": num_mc,
            "src_scalar_NATURAL_modules_per_chunk": mpc,
            "src_scalar_NATURAL_num_class_chunks": num_class_chunks,
            "src_scalar_NATURAL_total_tile_count": tc,
        },
        tile_count=1,
    )
    alloc.set_producer(b_permuted_h, n13)
    alloc.add_consumer(b_clipped_h, n13)

    barrier_id = "item_sync_barrier"
    nodes[barrier_id] = BarrierNode(
        node_id=barrier_id, depends_on=frozenset({n13}),
        barrier_name="item_sync",
    )

    # ── Node 16 schedule ──────────────────────────────────────────
    #
    # TIER BOUNDARY NOTE: The threshold schedule is computed here
    # (Policy tier) assuming CPU-style rendering where
    # workgroup_size == total_modules (no pre-accumulation).  For
    # GPU backends with workgroup-limited pre-accumulation, the
    # Orchestration tier must re-render the schedule from the
    # abstract policy parameters (CONCEPT.md §5, §11 item 4).
    # This is acceptable for the initial CPU-backend implementation;
    # a backend-neutral schedule representation should be introduced
    # when a GPU backend is integrated.

    n16_max_k = policy.get_specialized_reduction_policy_k(
        spec.num_modules, hardware.max_reduce_fan_in,
    )
    n16_stages, n16_t_pre, n16_schedule = policy.render_node16_schedule(
        spec.num_modules, spec.num_modules, n16_max_k,
    )
    b_n16_sched = alloc.allocate(
        "clipping_threshold_per_stage", (max(1, n16_stages),),
        ec, BufferRole.BATCH_INTERMEDIATE, "compute",
    )
    precomputed[b_n16_sched] = np.array(
        n16_schedule if n16_stages > 0 else [0.0],
        dtype=prec.compute_dtype,
    )

    n16 = "stabilize_and_reduce_grad_hidden_activations"
    nodes[n16] = _dispatch(
        n16, frozenset({barrier_id}), KERNEL_REGISTRY[n16],
        {
            "src_buffer_GLOBAL_clipped_grad_hidden_activations_permuted_soa": b_permuted_h,
            "dest_buffer_GLOBAL_summed_grad_hidden_activations": b_summed_h,
            "src_buffer_GLOBAL_CONST_clipping_threshold_per_stage": b_n16_sched,
        },
        {
            "src_scalar_NATURAL_num_reduction_stages": n16_stages,
            "src_scalar_REAL_clipping_threshold_t_pre": n16_t_pre,
            "src_scalar_REAL_epsilon": prec.compute_epsilon,
            "src_scalar_NATURAL_total_batch_count": batch_size,
            "src_scalar_NATURAL_padded_hidden_count": spec.padded_hidden_dim,
            "src_scalar_NATURAL_total_modules_count": spec.num_modules,
            "src_scalar_NATURAL_padded_total_modules_count": spec.padded_module_dim,
        },
        tile_count=1,
    )
    alloc.set_producer(b_summed_h, n16)
    alloc.add_consumer(b_permuted_h, n16)
    alloc.add_consumer(b_n16_sched, n16)

    topo.extend([n13, barrier_id, n16])

    # ═══════════════════════════════════════════════════════════════
    # Phase II: Per-Module-Chunk Gradient Chains (ADR-030)
    #
    # Depends on clip_partial_gradients — NOT item_sync_barrier.
    # Module gradient path is independent of the grad_h
    # gather/reduce and can execute concurrently with it.
    # ═══════════════════════════════════════════════════════════════

    all_terminals: set[str] = set()
    clip_done = frozenset({n11})

    _opt = optimizer if optimizer is not None else OptimizerConfig()
    _adam: dict[str, float] = {
        "src_scalar_REAL_learning_rate": _opt.learning_rate,
        "src_scalar_REAL_beta1_pow_t": _opt.beta1 ** adam_step,
        "src_scalar_REAL_beta2_pow_t": _opt.beta2 ** adam_step,
        "src_scalar_REAL_beta1": _opt.beta1,
        "src_scalar_REAL_beta2": _opt.beta2,
        "src_scalar_REAL_epsilon": _opt.resolve_epsilon(prec),
    }

    for group in MODULE_GROUPS:
        geo = geos[group.name]
        epp = mpc * geo.elements_per_module

        for mc in range(num_mc):
            gather = ModuleChunkGather(
                geometry=tiling, module_chunk=mc,
                elements_per_tile=epp,
            )
            ps = ParameterSlice(
                offset=tiling.modules.offset_for(mc) * geo.elements_per_module,
                count=tiling.modules.count_for(mc) * geo.elements_per_module,
                total=geo.total_flat_elements,
            )
            chain_topo, chain_terms = _build_gradient_chain(
                alloc, nodes,
                _GradientChainSpec(
                    group=group, geo=geo,
                    clipped_buf=module_clipped[group.name],
                    gather=gather,
                    param_buf=state.for_group(group),
                    m1_buf=moments[group.m1_name],
                    m2_buf=moments[group.m2_name],
                    param_slice=ps,
                    mc=mc,
                ),
                policy, hardware, prec, eff_batch, _adam, clip_done,
                temp_bounds=(temp_min, temp_max),
            )
            topo.extend(chain_topo)
            all_terminals.update(chain_terms)

    # ═══════════════════════════════════════════════════════════════
    # Phase III: Streaming Shared-Layer Backpropagation (17→18→19)
    #
    # Body nodes are owned by the StreamingLoopNode and are NOT
    # registered in the top-level `nodes` dict.  Their depends_on
    # edges reference only sibling body-node IDs (D2 invariant).
    # The loop node's own depends_on gates when the first
    # iteration may begin.
    # ═══════════════════════════════════════════════════════════════

    sw_geo = geos[SHARED_WEIGHTS.name]
    sb_geo = geos[SHARED_BIASES.name]
    sw_count = sw_geo.total_flat_elements
    sb_count = sb_geo.total_flat_elements

    # Scratch buffers (single-slot, overwritten each iteration).
    b_scratch_sw = alloc.allocate(
        "scratch_grad_shared_weights",
        (1, spec.padded_hidden_dim // simd_w, spec.padded_input_dim, simd_w),
        es, BufferRole.BATCH_INTERMEDIATE, "storage",
    )
    b_scratch_sb = alloc.allocate(
        "scratch_grad_shared_biases", (1, spec.padded_hidden_dim),
        es, BufferRole.BATCH_INTERMEDIATE, "storage",
    )

    # Collection buffers (populated across all iterations).
    b_coll_sw = alloc.allocate(
        "clipped_partial_grad_shared_weights",
        (n_stream, sw_count),
        es, BufferRole.BATCH_INTERMEDIATE, "storage",
    )
    b_coll_sb = alloc.allocate(
        "clipped_partial_grad_shared_biases",
        (n_stream, sb_count),
        es, BufferRole.BATCH_INTERMEDIATE, "storage",
    )

    shared_clipped: dict[str, BufferHandle] = {
        SHARED_WEIGHTS.name: b_coll_sw,
        SHARED_BIASES.name: b_coll_sb,
    }

    # ── Node 19 leaf safety ceiling (CONCEPT §3.4) ───────────────
    #
    # Node 19 clips each streaming chunk's concatenated (weights,
    # biases) gradient vector before it enters the collection
    # buffer.  The reduction tree (Node 20) sums K of these clipped
    # partials at its first stage.  For the stage-0 summation not
    # to overflow: K × T_pre ≤ cfm.
    #
    # The algorithmic component of the threshold (Quadratic Scaling
    # Policy) is applied within the Node 20 reduction tree, not
    # here — Node 19's threshold is purely a safety ceiling.
    node19_leaf_threshold = _compute_leaf_safety_ceiling(
        policy, hardware, n_stream,
    )

    # ── Streaming loop body ───────────────────────────────────────

    LOOP_ID = "streaming_backprop_loop"
    grad_h_done = frozenset({n16})

    n17_id = "backprop_shared_weights_chunk"
    n18_id = "backprop_shared_biases_chunk"
    n19_id = "clip_shared_gradients_chunk"

    # Body Node 17 — no intra-body predecessors.
    body_n17 = _dispatch(
        n17_id, frozenset(), KERNEL_REGISTRY[n17_id],
        {
            "src_buffer_GLOBAL_input": inputs.input_data,
            "src_buffer_GLOBAL_hidden_activations": fwd.hidden,
            "src_buffer_GLOBAL_hidden_mask": fwd.hidden_mask,
            "src_buffer_GLOBAL_summed_grad_hidden_activations": b_summed_h,
            "src_buffer_GLOBAL_sample_mask": inputs.sample_mask,
            "dest_buffer_GLOBAL_partial_grad_weights_shared_simd_major": b_scratch_sw,
        },
        {
            "src_scalar_NATURAL_batch_chunk_offset": 0,
            "src_scalar_NATURAL_batch_chunk_count": stream_chunk_sz,
            "src_scalar_FLAG_use_explicit_hidden_mask": flag_mask,
            "src_scalar_NATURAL_total_batch_count": batch_size,
            "src_scalar_NATURAL_input_count": spec.input_dim,
            "src_scalar_NATURAL_padded_input_count": spec.padded_input_dim,
            "src_scalar_NATURAL_hidden_count": spec.hidden_dim,
            "src_scalar_NATURAL_padded_hidden_count": spec.padded_hidden_dim,
            "src_scalar_NATURAL_final_grad_hidden_activations_total_count": grad_h_total,
        },
        tile_count=1,
    )

    # Body Node 18 — no intra-body predecessors.
    body_n18 = _dispatch(
        n18_id, frozenset(), KERNEL_REGISTRY[n18_id],
        {
            "src_buffer_GLOBAL_hidden_activations": fwd.hidden,
            "src_buffer_GLOBAL_hidden_mask": fwd.hidden_mask,
            "src_buffer_GLOBAL_summed_grad_hidden_activations": b_summed_h,
            "src_buffer_GLOBAL_sample_mask": inputs.sample_mask,
            "dest_buffer_GLOBAL_partial_grad_biases_shared": b_scratch_sb,
        },
        {
            "src_scalar_NATURAL_batch_chunk_offset": 0,
            "src_scalar_NATURAL_batch_chunk_count": stream_chunk_sz,
            "src_scalar_FLAG_use_explicit_hidden_mask": flag_mask,
            "src_scalar_NATURAL_total_batch_count": batch_size,
            "src_scalar_NATURAL_hidden_count": spec.hidden_dim,
            "src_scalar_NATURAL_padded_hidden_count": spec.padded_hidden_dim,
            "src_scalar_NATURAL_final_grad_hidden_activations_total_count": grad_h_total,
        },
        tile_count=1,
    )

    # Body Node 19 — waits for both siblings within each iteration.
    body_n19 = _dispatch(
        n19_id, frozenset({n17_id, n18_id}), KERNEL_REGISTRY[n19_id],
        {
            "src_buffer_GLOBAL_partial_grad_weights_shared_simd_major": b_scratch_sw,
            "src_buffer_GLOBAL_partial_grad_biases_shared": b_scratch_sb,
            "dest_buffer_GLOBAL_clipped_partial_grad_weights_shared_simd_major": b_coll_sw,
            "dest_buffer_GLOBAL_clipped_partial_grad_biases_shared": b_coll_sb,
        },
        {
            "src_scalar_REAL_clipping_threshold_t_pre": node19_leaf_threshold,
            "src_scalar_REAL_epsilon": prec.compute_epsilon,
            "src_scalar_NATURAL_weights_parameter_count": sw_count,
            "src_scalar_NATURAL_biases_parameter_count": sb_count,
            "out_scalar_NATURAL_weights_write_offset": 0,
            "out_scalar_NATURAL_biases_write_offset": 0,
            "src_scalar_NATURAL_num_batch_chunks": n_stream,
        },
        tile_count=1,
    )

    # ── Lifecycle: attribute loop-managed buffers to LOOP_ID ──────
    #
    # From the top-level DAG perspective the loop node is the sole
    # producer/consumer.  Body-internal read-after-write on scratch
    # buffers is reflected by setting the loop as both producer and
    # consumer.  Collection buffers are produced by the loop and
    # consumed by the downstream reduction trees.

    for h in (b_scratch_sw, b_scratch_sb, b_coll_sw, b_coll_sb):
        alloc.set_producer(h, LOOP_ID)
    for h in (inputs.input_data, fwd.hidden, fwd.hidden_mask,
              b_summed_h, inputs.sample_mask,
              b_scratch_sw, b_scratch_sb):
        alloc.add_consumer(h, LOOP_ID)

    # ── Assemble the streaming loop node ──────────────────────────

    streaming_plan = StreamingLoopPlan(
        iteration=IterationDimension(
            total_extent=batch_size,
            chunk_count=n_stream,
            chunk_size=stream_chunk_sz,
        ),
        body=(n17_id, n18_id, n19_id),
        parameter_strides=(
            ParameterStride(
                "src_scalar_NATURAL_batch_chunk_offset",
                base=0, stride=stream_chunk_sz,
                target_nodes=frozenset({n17_id, n18_id}),
            ),
            ParameterStride(
                "out_scalar_NATURAL_weights_write_offset",
                base=0, stride=sw_count,
                target_nodes=frozenset({n19_id}),
            ),
            ParameterStride(
                "out_scalar_NATURAL_biases_write_offset",
                base=0, stride=sb_count,
                target_nodes=frozenset({n19_id}),
            ),
        ),
        scratch_buffers=(),
        constant_scalars={},
    )
    nodes[LOOP_ID] = StreamingLoopNode(
        node_id=LOOP_ID,
        depends_on=grad_h_done,
        streaming_plan=streaming_plan,
        body_nodes=(body_n17, body_n18, body_n19),
    )

    topo.append(LOOP_ID)

    # ═══════════════════════════════════════════════════════════════
    # Phase IV: Shared-Layer Gradient Chains
    # ═══════════════════════════════════════════════════════════════

    loop_done = frozenset({LOOP_ID})

    for group in SHARED_GROUPS:
        geo = geos[group.name]
        gather = StridedGather(
            count=n_stream, partial_width=geo.total_flat_elements,
        )
        ps = ParameterSlice.full(geo.total_flat_elements)
        chain_topo, chain_terms = _build_gradient_chain(
            alloc, nodes,
            _GradientChainSpec(
                group=group, geo=geo,
                clipped_buf=shared_clipped[group.name],
                gather=gather,
                param_buf=state.for_group(group),
                m1_buf=moments[group.m1_name],
                m2_buf=moments[group.m2_name],
                param_slice=ps,
            ),
            policy, hardware, prec, eff_batch, _adam, loop_done,
        )
        topo.extend(chain_topo)
        all_terminals.update(chain_terms)

    # ═══════════════════════════════════════════════════════════════
    # Final Retrieval
    # ═══════════════════════════════════════════════════════════════

    final_id = "final_batch_retrieval"
    nodes[final_id] = RetrievalNode(
        node_id=final_id,
        depends_on=frozenset(all_terminals),
        source_buffer=b_summed_h,  # Sentinel; real signal is the event.
        logical_shape=(1,),
        event_name="final_batch_event",
    )
    alloc.add_consumer(b_summed_h, final_id)

    topo.append(final_id)

    return ExecutionPlan(
        nodes=nodes,
        buffers=alloc.finalize(tuple(topo)),
        execution_order=tuple(topo),
        precision=prec,
        hardware=hardware,
        precomputed_buffers=precomputed,
    )
