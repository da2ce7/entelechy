"""VulkanPlanRenderer — PlanRenderer implementation for the Vulkan backend (ADR-001).

Records ExecutionPlan DAG nodes into Vulkan command buffers and submits
them for GPU execution. Supports all five node types: KernelDispatchNode,
ReductionTreeNode, StreamingLoopNode, BarrierNode, and RetrievalNode.
"""
# pyright: reportUnknownVariableType=false, reportUnknownMemberType=false, reportUnknownArgumentType=false
from __future__ import annotations

import ctypes
import logging
from dataclasses import replace
from typing import TYPE_CHECKING, Any, Callable

import numpy as np
from numpy.typing import NDArray

from ...shared.buffer_lifecycle import BufferHandle, BufferRole
from ...shared.plan_types import (
    BarrierNode,
    ExecutionPlan,
    KernelDispatchNode,
    ReductionTreeNode,
    RetrievalNode,
    StreamingLoopNode,
)
from ...shared.retrieval_future import RetrievalFuture
from ...shared.stabilization_policy import render_node16_threshold_schedule
from ...shared.streaming_loop_plan import StreamingLoopPlan
from ._descriptor_manager import VulkanDescriptorManager
from ._pipeline_cache import ComputePipeline, SpecConstants, VulkanPipelineCache
from ._push_constants import (
    BUFFER_BINDING_ORDER,
    DESCRIPTOR_BINDING_COUNTS,
    PUSH_CONSTANT_STRUCTS,
    marshal_push_constants,
)
from .buffer_allocator import VulkanBuffer, VulkanBufferAllocator
from .context import VulkanContext
from .discovery import discover_hardware
from .retrieval import VulkanRetrievalFuture

if TYPE_CHECKING:
    import vulkan as vk  # type: ignore[import-untyped]
    from vulkan._vulkan import ffi as _ffi  # type: ignore[import-untyped]
else:
    try:
        import vulkan as vk
        from vulkan._vulkan import ffi as _ffi  # pyright: ignore[reportMissingTypeStubs]
    except ImportError:
        vk = None  # type: ignore[assignment]
        _ffi = None  # type: ignore[assignment]

logger = logging.getLogger(__name__)

# Shaders whose dispatch uses 2D workgroups (x, y, 1)
_2D_DISPATCH_KERNELS = frozenset({"forward_pass", "backprop_shared_weights_chunk"})

# Reduction engine shaders using push descriptors
_REDUCTION_SHADERS = frozenset({
    "aggregate_partials",
    "aggregate_partials_from_compute",  # ADR-026
    "clip_intermediate_grad",
    "reduce_k_fan_in_and_clip",  # ADR-019
    "reduce_k_fan_in_and_clip_from_compute",  # ADR-019/ADR-026
})


class VulkanPlanRenderer:
    """Plan renderer for the Vulkan backend (ADR-001, ADR-015).

    Records an ExecutionPlan DAG into Vulkan command buffers and submits
    them for GPU execution. Returns VulkanRetrievalFutures for each
    RetrievalNode.
    """

    def __init__(
        self,
        context: VulkanContext | None = None,
        *,
        problem_type: int = 0,
    ) -> None:
        if context is None:
            context = VulkanContext()
            self._owns_context = True
        else:
            self._owns_context = False

        self._ctx = context
        self._problem_type = problem_type
        self._hardware = discover_hardware(context)

        self._allocator = VulkanBufferAllocator(context)
        self._pipeline_cache = VulkanPipelineCache(context)
        self._descriptor_mgr = VulkanDescriptorManager(context)

        # Persistent MODEL_STATE buffer store (ADR-009).
        # Keyed by (logical_name, padded_shape, dtype_name) so that
        # MODEL_STATE buffers survive across render() calls, enabling
        # multi-batch training where parameter updates accumulate.
        self._persistent_model_state: dict[
            tuple[str, tuple[int, ...], str], VulkanBuffer
        ] = {}

        # Per-render state (set during render())
        self._pipelines: dict[str, ComputePipeline] = {}
        self._descriptor_sets: dict[str, Any] = {}
        self._descriptor_layouts: dict[str, Any] = {}

    def render(
        self,
        plan: ExecutionPlan,
        data_injections: dict[str, NDArray[Any]] | None = None,
    ) -> dict[str, RetrievalFuture]:
        """Record and submit the plan, returning futures for retrieval nodes."""
        # 0. Wait for any in-flight work before releasing batch buffers
        vk.vkDeviceWaitIdle(self._ctx.device)

        # 1. Release non-MODEL_STATE buffers from prior render; keep persistent
        self._allocator.release_batch_buffers()

        # 2. Resolve MODEL_STATE persistence: reuse existing device buffers or
        #    allocate fresh ones.  Allocate non-MODEL_STATE buffers normally.
        role_dtypes = {
            "storage": plan.precision.storage_dtype,
            "compute": plan.precision.compute_dtype,
            "state": plan.precision.state_dtype,
        }
        upload_cmd = self._ctx.allocate_command_buffer()
        upload_fence = self._ctx.create_fence()

        newly_allocated_model_state: set[str] = set()
        for descriptor in plan.buffers.values():
            if descriptor.role == BufferRole.MODEL_STATE:
                prole = descriptor.precision_role
                dtype = role_dtypes.get(prole) if prole else None  # type: ignore[arg-type]
                dtype_str = dtype.str if dtype is not None else ""
                key = (descriptor.logical_name, descriptor.padded_shape, dtype_str)
                if key in self._persistent_model_state:
                    # Re-register the existing VulkanBuffer under this plan's handle
                    self._allocator._buffers[descriptor.handle] = (
                        self._persistent_model_state[key]
                    )
                else:
                    # First time: allocate, zero-fill, mark for init
                    self._allocator.allocate(descriptor)
                    self._allocator.zero_fill(
                        descriptor.handle, upload_cmd,
                        self._ctx.compute_queue, upload_fence,
                    )
                    newly_allocated_model_state.add(descriptor.logical_name)
                    self._persistent_model_state[key] = (
                        self._allocator.get_buffer(descriptor.handle)
                    )
            else:
                self._allocator.allocate(descriptor)
                self._allocator.zero_fill(
                    descriptor.handle, upload_cmd,
                    self._ctx.compute_queue, upload_fence,
                )

        # 3. Initialize only newly-allocated MODEL_STATE buffers
        self._init_model_state_buffers(
            plan, upload_cmd, self._ctx.compute_queue, upload_fence,
            only_names=newly_allocated_model_state or None,
        )

        # 4. Host-to-device data injection
        if data_injections:
            name_to_desc = {
                desc.logical_name: desc
                for desc in plan.buffers.values()
            }
            for logical_name, host_data in data_injections.items():
                desc = name_to_desc.get(logical_name)
                if desc is None:
                    continue
                total_elems = int(np.prod(desc.padded_shape))
                prole = desc.precision_role
                buf_dtype = role_dtypes.get(prole) if prole else np.dtype(np.uint32)  # type: ignore[arg-type]
                if (
                    logical_name == "targets_cce"
                    and np.issubdtype(host_data.dtype, np.integer)
                    and not np.issubdtype(buf_dtype, np.integer)
                ):
                    # CCE class indices: raw byte copy preserves int32 bit
                    # pattern so the shader can read the buffer as int*.
                    padded = np.zeros(total_elems, dtype=buf_dtype)
                    src = host_data.astype(np.int32)
                    padded_bytes = padded.view(np.uint8)
                    src_bytes = src.tobytes()
                    n = min(len(src_bytes), len(padded_bytes))
                    padded_bytes[:n] = np.frombuffer(src_bytes[:n], dtype=np.uint8)
                elif host_data.ndim <= 1 or len(desc.padded_shape) <= 1:
                    if len(desc.padded_shape) > 1 and host_data.ndim == 1:
                        padded = np.zeros(total_elems, dtype=buf_dtype)
                        reshaped = padded.reshape(desc.padded_shape)
                        n = min(len(host_data), desc.padded_shape[0])
                        reshaped[:n, 0] = host_data[:n].astype(buf_dtype)
                    else:
                        padded = np.zeros(total_elems, dtype=buf_dtype)
                        flat = host_data.astype(buf_dtype).flatten()
                        n = min(len(flat), len(padded))
                        padded[:n] = flat[:n]
                else:
                    padded = np.zeros(desc.padded_shape, dtype=buf_dtype)
                    host = host_data.astype(buf_dtype)
                    slices = tuple(
                        slice(0, min(host.shape[d], desc.padded_shape[d]))
                        for d in range(min(host.ndim, len(desc.padded_shape)))
                    )
                    padded[slices] = host[slices]
                    padded = padded.ravel()
                self._allocator.upload_to_device(
                    desc.handle, padded, upload_cmd,
                    self._ctx.compute_queue, upload_fence,
                )

        # 5. Build specialization constants
        spec = SpecConstants(
            simd_width=plan.hardware.simd_width,
            bank_padding=1,
            tile_size=8,
            problem_type=self._problem_type,
        )

        # 6. Create pipelines and descriptor layouts for all kernels in the plan
        self._prepare_pipelines(plan, spec)

        # 7. Update descriptor sets with buffer bindings
        self._update_descriptor_sets(plan)

        # 8. Record command buffer
        cmd = self._ctx.allocate_command_buffer()
        fence = self._ctx.create_fence()

        begin_info = vk.VkCommandBufferBeginInfo(
            flags=vk.VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT,
        )
        vk.vkBeginCommandBuffer(cmd, begin_info)

        futures: dict[str, RetrievalFuture] = {}

        for node_id in plan.topological_order:
            node = plan.nodes[node_id]

            if isinstance(node, KernelDispatchNode):
                self._record_kernel_dispatch(cmd, node, plan)
            elif isinstance(node, ReductionTreeNode):
                self._record_reduction_tree(cmd, node, plan, spec)
            elif isinstance(node, StreamingLoopNode):
                self._record_streaming_loop(cmd, node, plan)
            elif isinstance(node, BarrierNode):
                self._record_barrier(cmd)
            else:  # RetrievalNode
                staging = self._record_retrieval(cmd, node, plan)
                future = VulkanRetrievalFuture(
                    context=self._ctx,
                    fence=fence,
                    staging=staging,
                    node_id=node.node_id,
                    logical_shape=node.logical_shape,
                    padded_size_bytes=plan.buffers[
                        node.source_buffer
                    ].size_bytes,
                    dtype=plan.precision.storage_dtype,
                )
                futures[node.event_name] = future

        vk.vkEndCommandBuffer(cmd)

        # 9. Submit
        submit = vk.VkSubmitInfo(
            commandBufferCount=1,
            pCommandBuffers=[cmd],
        )
        vk.vkQueueSubmit(self._ctx.compute_queue, 1, [submit], fence)

        # Clean up upload resources
        vk.vkDestroyFence(self._ctx.device, upload_fence, None)

        return futures

    def destroy(self) -> None:
        """Clean up all Vulkan resources."""
        vk.vkDeviceWaitIdle(self._ctx.device)
        self._pipeline_cache.destroy()
        self._descriptor_mgr.destroy()
        self._allocator.destroy()
        self._persistent_model_state.clear()
        self._pipelines.clear()
        self._descriptor_sets.clear()
        self._descriptor_layouts.clear()
        if self._owns_context:
            self._ctx.destroy()

    # ── MODEL_STATE initialization (must match CPU renderer for parity) ──

    _WEIGHT_SUBSTRINGS = ("weight",)
    _UNIT_INIT_SUBSTRINGS = ("temperature",)

    @staticmethod
    def _buffer_rng(logical_name: str, padded_shape: tuple[int, ...]) -> np.random.Generator:
        """Deterministic per-buffer RNG seeded by buffer identity."""
        import hashlib
        h = hashlib.sha256(f"{logical_name}{padded_shape}".encode()).hexdigest()
        return np.random.default_rng(int(h[:16], 16))

    def _init_model_state_buffers(
        self,
        plan: ExecutionPlan,
        cmd: Any,
        queue: Any,
        fence: Any,
        *,
        only_names: set[str] | None = None,
    ) -> None:
        """Upload Xavier/unit-initialized values for MODEL_STATE buffers.

        When *only_names* is provided, only buffers whose logical_name is
        in the set are initialized (used to skip already-persistent buffers).
        """
        role_dtypes = {
            "storage": plan.precision.storage_dtype,
            "compute": plan.precision.compute_dtype,
            "state": plan.precision.state_dtype,
        }
        for descriptor in plan.buffers.values():
            if descriptor.role != BufferRole.MODEL_STATE:
                continue
            name = descriptor.logical_name
            if only_names is not None and name not in only_names:
                continue
            total = int(np.prod(descriptor.padded_shape))
            prole = descriptor.precision_role
            if prole is None:
                continue
            dtype = role_dtypes[prole]
            host: np.ndarray | None = None
            if any(s in name for s in self._WEIGHT_SUBSTRINGS):
                shape_for_fan = (
                    descriptor.logical_shape
                    if descriptor.logical_shape and len(descriptor.logical_shape) >= 2
                    else descriptor.padded_shape
                )
                if len(shape_for_fan) >= 2:
                    fan_in_plus_out = shape_for_fan[-2] + shape_for_fan[-1]
                else:
                    fan_in_plus_out = max(
                        shape_for_fan[0] if shape_for_fan else total, 2,
                    )
                limit = float(np.sqrt(6.0 / fan_in_plus_out))
                rng = self._buffer_rng(name, descriptor.padded_shape)
                host = rng.uniform(
                    -limit, limit, size=total,
                ).astype(dtype)
            elif any(s in name for s in self._UNIT_INIT_SUBSTRINGS):
                host = np.ones(total, dtype=dtype)
            if host is not None:
                self._allocator.upload_to_device(
                    descriptor.handle, host, cmd, queue, fence,
                )

    # ── Pipeline & descriptor preparation ──

    def _prepare_pipelines(
        self, plan: ExecutionPlan, spec: SpecConstants
    ) -> None:
        """Create compute pipelines for all kernel names in the plan."""
        kernel_names: set[str] = set()
        for node in plan.nodes.values():
            if isinstance(node, KernelDispatchNode):
                kernel_names.add(node.kernel_name)
            elif isinstance(node, ReductionTreeNode):
                rtp = node.reduction_plan
                source_desc = plan.buffers[rtp.source_buffer]
                use_compute_entry = source_desc.precision_role == "compute"

                if rtp.num_stages > 1 or rtp.tree_variant == "sum_and_clip":
                    # ADR-019: Multi-stage trees use K-fan-in kernels
                    kernel_names.add("reduce_k_fan_in_and_clip")
                    if use_compute_entry or rtp.num_stages > 1:
                        kernel_names.add("reduce_k_fan_in_and_clip_from_compute")
                else:
                    # Single-stage diagnostic trees use aggregate kernels
                    kernel_names.add("aggregate_partials")
                    if use_compute_entry:
                        kernel_names.add("aggregate_partials_from_compute")

        for name in kernel_names:
            binding_count = DESCRIPTOR_BINDING_COUNTS.get(name)
            if binding_count is None:
                # ADR-026: Compute-entry variant shares binding count with storage-entry
                if name == "aggregate_partials_from_compute":
                    binding_count = DESCRIPTOR_BINDING_COUNTS["aggregate_partials"]
                else:
                    raise KeyError(f"Unknown kernel {name}")

            layout = self._descriptor_mgr.create_layout(binding_count)
            self._descriptor_layouts[name] = layout

            push_struct = PUSH_CONSTANT_STRUCTS.get(name)
            if push_struct is None:
                # ADR-026: Compute-entry variant shares push constant struct
                if name == "aggregate_partials_from_compute":
                    push_struct = PUSH_CONSTANT_STRUCTS["aggregate_partials"]
                else:
                    raise KeyError(f"Unknown kernel {name}")

            push_size = ctypes.sizeof(push_struct)
            pipeline = self._pipeline_cache.create_pipeline(
                name, layout, push_size, spec, plan.precision
            )
            self._pipelines[name] = pipeline

            # Pre-allocate descriptor set for fixed-binding kernels
            if name not in _REDUCTION_SHADERS:
                desc_set = self._descriptor_mgr.allocate_set(layout)
                self._descriptor_sets[name] = desc_set

    def _update_descriptor_sets(self, plan: ExecutionPlan) -> None:
        """Bind allocated VkBuffers to pre-allocated descriptor sets."""
        # Collect all kernel nodes and their buffer bindings
        kernel_bindings: dict[str, dict[str, BufferHandle]] = {}
        for node in plan.nodes.values():
            if isinstance(node, KernelDispatchNode):
                if node.kernel_name not in kernel_bindings:
                    kernel_bindings[node.kernel_name] = node.buffer_bindings

        for kernel_name, bindings in kernel_bindings.items():
            if kernel_name in _REDUCTION_SHADERS:
                continue  # Dynamic push descriptors
            desc_set = self._descriptor_sets.get(kernel_name)
            if desc_set is None:
                continue

            binding_list: list[tuple[int, VulkanBuffer]] = []
            order = BUFFER_BINDING_ORDER.get(kernel_name)
            if order is not None:
                for binding_idx, key in enumerate(order):
                    if key is not None and key in bindings:
                        vk_buf = self._allocator.get_buffer(bindings[key])
                        binding_list.append((binding_idx, vk_buf))
            else:
                for binding_idx, (_, handle) in enumerate(bindings.items()):
                    vk_buf = self._allocator.get_buffer(handle)
                    binding_list.append((binding_idx, vk_buf))

            self._descriptor_mgr.update_set(desc_set, binding_list)

    # ── Node type recording ──

    # Per-element kernels (placement_strategy="linear_generic") need
    # workgroup count derived from scalar params rather than tile_count.
    # Maps kernel_name → callable(scalar_params, simd_width) → workgroup_count.
    _WORKGROUP_COUNT_RESOLVERS: dict[str, Callable[[dict[str, Any], int], int]] = {
        # Element-wise: ceil(N / simd_width) workgroups
        "normalize_gradients": lambda s, w: (
            (int(s["parameter_count"]) + w - 1) // w
        ),
        "adam_update": lambda s, w: (
            (int(s["parameter_count"]) + w - 1) // w
        ),
        "clamp_temperatures": lambda s, w: (
            (int(s["parameter_count"]) + w - 1) // w
        ),
        # Workgroup-per-row: one workgroup per (batch, hidden) row
        "stabilize_and_reduce_grad_hidden_activations": lambda s, w: (
            int(s["total_batch_count"]) * int(s["padded_hidden_count"])
        ),
        # Phase 1 — Act
        "render_logits_chunk": lambda s, w: (
            (int(s["module_chunk_count"])
             * int(s["batch_chunk_count"])
             * int(s["class_chunk_count"])
             + w - 1) // w
        ),
        # Phase 2A — Production
        # Total items = total_tile_count * modules_per_chunk * total_batch_count
        # Workgroups = ceil(total_items / SIMD_WIDTH)
        "compute_probs_loss_cce_chunk": lambda s, w: (
            (int(s["total_tile_count"]) * int(s["modules_per_chunk"]) * int(s["total_batch_count"]) + w - 1) // w
        ),
        "compute_probs_loss_bce_chunk": lambda s, w: (
            (int(s["total_tile_count"]) * int(s["modules_per_chunk"]) * int(s["total_batch_count"]) + w - 1) // w
        ),
        # Phase 2B
        "calculate_module_param_grads_chunk": lambda s, w: (
            int(s["modules_per_chunk"]) * int(s["hidden_count"])
        ),
        "backprop_error_to_hidden_chunk": lambda s, w: (
            int(s["modules_per_chunk"])
            * int(s["total_batch_count"])
            * int(s["padded_hidden_count"])
        ),
        "calculate_chunk_temp_gradients": lambda s, w: int(s["modules_per_chunk"]),
        # Phase 2C — Reduction
        "gather_and_permute_grad_hidden_activations": lambda s, w: (
            (int(s["total_batch_count"]) * int(s["padded_hidden_count"]) + w - 1) // w
        ),
        # Phase 2D — Backprop
        "backprop_shared_biases_chunk": lambda s, w: int(s["padded_hidden_count"]),
        # clip_shared_gradients_chunk: Dispatch(1,1,1)
        "clip_shared_gradients_chunk": lambda s, w: 1,
        # clip_partial_gradients: ignores task_index → tile_count is correct
    }

    # 2D dispatch kernels: returns (x_groups, y_groups) tuple.
    # The comment in forward_pass.comp says:
    #   Dispatch: vkCmdDispatch(batch_chunk_count, padded_hidden/SIMD_WIDTH, 1)
    _2D_DISPATCH_RESOLVERS: dict[
        str, Callable[[dict[str, Any], int], tuple[int, int]]
    ] = {
        "forward_pass": lambda s, w: (
            int(s["batch_chunk_count"]),
            int(s["padded_hidden_count"]) // w,
        ),
        "backprop_shared_weights_chunk": lambda s, w: (
            int(s["padded_input_count"]),
            int(s["padded_hidden_count"]),
        ),
        "calculate_chunk_temp_gradients_2d": lambda s, w: (
            int(s["total_tile_count"]),
            int(s["modules_per_chunk"]),
        ),
        "gather_and_permute_grad_hidden_activations_2d": lambda s, w: (
            (int(s["total_batch_count"]) * int(s["padded_hidden_count"]) + w - 1) // w,
            int(s["total_modules_count"]),
        ),
    }

    def _record_kernel_dispatch(
        self,
        cmd: Any,
        node: KernelDispatchNode,
        plan: ExecutionPlan,
    ) -> None:
        """Record a single kernel dispatch into the command buffer."""
        kernel = node.kernel_name
        pipeline = self._pipelines[kernel]

        # Node 16: compute threshold schedule and upload to device buffer.
        scalar_params = node.scalar_params
        schedule_data: list[float] = []
        if kernel == "stabilize_and_reduce_grad_hidden_activations":
            scalar_params, schedule_data = self._render_node16_schedule(
                node, plan,
            )

        vk.vkCmdBindPipeline(
            cmd, vk.VK_PIPELINE_BIND_POINT_COMPUTE, pipeline.pipeline
        )

        # Bind descriptor set
        desc_set = self._descriptor_sets.get(kernel)
        if desc_set is not None:
            vk.vkCmdBindDescriptorSets(
                cmd,
                vk.VK_PIPELINE_BIND_POINT_COMPUTE,
                pipeline.layout,
                0,
                1,
                [desc_set],
                0,
                None,
            )

        # Node 16: upload schedule to the bound buffer inline.
        if kernel == "stabilize_and_reduce_grad_hidden_activations":
            schedule_handle = node.buffer_bindings["clipping_threshold_per_stage"]
            vk_buf = self._allocator.get_buffer(schedule_handle)
            if len(schedule_data) > 0:
                host_bytes = np.array(schedule_data, dtype=np.float32).tobytes()
                vk.vkCmdUpdateBuffer(
                    cmd, vk_buf.buffer, 0, len(host_bytes),
                    _ffi.from_buffer(host_bytes),
                )
                # Pipeline barrier: transfer → compute read.
                barrier = vk.VkBufferMemoryBarrier(
                    srcAccessMask=vk.VK_ACCESS_TRANSFER_WRITE_BIT,
                    dstAccessMask=vk.VK_ACCESS_SHADER_READ_BIT,
                    buffer=vk_buf.buffer,
                    offset=0,
                    size=len(host_bytes),
                )
                vk.vkCmdPipelineBarrier(
                    cmd,
                    vk.VK_PIPELINE_STAGE_TRANSFER_BIT,
                    vk.VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,
                    0, 0, None, 1, [barrier], 0, None,
                )

        # Push constants
        pc_bytes = marshal_push_constants(
            kernel, scalar_params, plan.precision.compute_dtype
        )
        pc_buf = _ffi.from_buffer(pc_bytes)
        vk.vkCmdPushConstants(
            cmd,
            pipeline.layout,
            vk.VK_SHADER_STAGE_COMPUTE_BIT,
            0,
            len(pc_bytes),
            pc_buf,
        )

        # Dispatch
        if kernel in _2D_DISPATCH_KERNELS:
            resolver_2d = self._2D_DISPATCH_RESOLVERS.get(kernel)
            if resolver_2d is not None:
                x_groups, y_groups = resolver_2d(
                    scalar_params, plan.hardware.simd_width,
                )
            else:
                # Fallback for 2D kernels without resolvers
                x_groups = node.tile_count
                y_groups = 1
                if node.local_work_size is not None and node.local_work_size > 0:
                    y_groups = node.local_work_size
            vk.vkCmdDispatch(cmd, x_groups, y_groups, 1)
        else:
            resolver = self._WORKGROUP_COUNT_RESOLVERS.get(node.kernel_name)
            if resolver is not None:
                wg_count = resolver(
                    scalar_params, plan.hardware.simd_width,
                )
                vk.vkCmdDispatch(cmd, wg_count, 1, 1)
            else:
                vk.vkCmdDispatch(cmd, node.tile_count, 1, 1)

        # Implicit barrier: Vulkan requires explicit pipeline barriers between
        # compute dispatches with read-after-write dependencies. The plan model
        # doesn't include per-dispatch barrier nodes, so we add one after every
        # kernel dispatch to ensure correctness.
        self._record_barrier(cmd)

    def _render_node16_schedule(
        self,
        node: KernelDispatchNode,
        plan: ExecutionPlan,
    ) -> tuple[dict[str, int | float], list[float]]:
        """Compute the Node 16 threshold schedule for the GPU workgroup size.

        Returns modified scalar_params dict with kernel-interface keys and
        the schedule data to upload.
        """
        sp = node.scalar_params
        W = plan.hardware.simd_width
        M = int(sp["total_modules_count"])
        max_k = int(sp["policy_max_k"])

        num_stages, t_pre, schedule = render_node16_threshold_schedule(
            t_algorithmic=float(sp["policy_t_algorithmic"]),
            lambda_=float(sp["policy_lambda"]),
            compute_fp_format_max=float(sp["compute_fp_format_max"]),
            total_modules=M,
            workgroup_size=W,
            max_fan_in=max_k,
        )

        kernel_params: dict[str, int | float] = {
            "num_reduction_stages": num_stages,
            "clipping_threshold_t_pre": t_pre,
            "epsilon": sp["epsilon"],
            "total_batch_count": sp["total_batch_count"],
            "padded_hidden_count": sp["padded_hidden_count"],
            "total_modules_count": sp["total_modules_count"],
            "padded_total_modules_count": sp["padded_total_modules_count"],
        }
        return kernel_params, schedule

    def _record_reduction_tree(
        self,
        cmd: Any,
        node: ReductionTreeNode,
        plan: ExecutionPlan,
        spec: SpecConstants,
    ) -> None:
        """Record a staged reduction tree into the command buffer.

        ADR-019: Multi-stage trees or sum_and_clip variant use the fused
        reduce_k_fan_in_and_clip kernels for per-node independent L2 clip.

        ADR-026: Selects between storage-entry and compute-entry kernel
        variants based on the source buffer's precision_role and stage.
        """
        rtp = node.reduction_plan

        # ADR-026 §3: Determine if source buffer is compute-role
        source_desc = plan.buffers[rtp.source_buffer]
        use_compute_entry = source_desc.precision_role == "compute"

        # Get source/destination device buffers
        src_device_buf = self._allocator.get_buffer(rtp.source_buffer)
        dst_device_buf = self._allocator.get_buffer(rtp.destination_buffer)

        pw = rtp.partial_width
        fan_in = rtp.fan_in

        # ADR-019: Use K-fan-in kernels for multi-stage or sum_and_clip
        use_k_fan_in = rtp.num_stages > 1 or rtp.tree_variant == "sum_and_clip"

        if not use_k_fan_in:
            # Single-stage diagnostic: use aggregate kernels (no clipping)
            self._record_single_stage_aggregate(
                cmd, node, plan, spec, use_compute_entry,
                src_device_buf, dst_device_buf,
            )
            return

        # --- ADR-019: Multi-stage K-fan-in reduction with per-node clip ---

        # Allocate ping-pong scratch buffers for intermediate stages
        max_intermediates = rtp.num_partials
        compute_elem = plan.precision.compute_dtype.itemsize
        ping_size = max_intermediates * pw * compute_elem
        pong_size = ping_size

        from ...shared.buffer_lifecycle import BufferDescriptor

        # Create temporary scratch buffer descriptors
        ping_handle = BufferHandle(9990)
        pong_handle = BufferHandle(9991)
        offset_handle = BufferHandle(9992)

        ping_desc = BufferDescriptor(
            handle=ping_handle,
            logical_name="_reduction_ping",
            padded_shape=(max_intermediates * pw,),
            element_size_bytes=compute_elem,
            size_bytes=ping_size,
            role=BufferRole.BATCH_INTERMEDIATE,
            precision_role="compute",
            producing_node=None,
            consumers=frozenset(),
            last_consumer=None,
        )
        pong_desc = BufferDescriptor(
            handle=pong_handle,
            logical_name="_reduction_pong",
            padded_shape=(max_intermediates * pw,),
            element_size_bytes=compute_elem,
            size_bytes=pong_size,
            role=BufferRole.BATCH_INTERMEDIATE,
            precision_role="compute",
            producing_node=None,
            consumers=frozenset(),
            last_consumer=None,
        )
        self._allocator.allocate(ping_desc)
        self._allocator.allocate(pong_desc)

        # Build offset lists for each stage
        # Stage 0: uses initial_offset_list (K entries per node)
        # Stage N: uses contiguous offsets into prior stage's output
        current_n = rtp.num_partials
        stage_offset_lists: list[np.ndarray] = []

        for stage in range(rtp.num_stages):
            nodes_at_stage = (current_n + fan_in - 1) // fan_in

            if stage == 0:
                # Use the plan's initial offset list
                offsets = np.array(rtp.initial_offset_list, dtype=np.uint32)
            else:
                # Build contiguous offsets for prior stage's output
                # Each node reads K consecutive output vectors from prior stage
                offsets = np.zeros(nodes_at_stage * fan_in, dtype=np.uint32)
                prev_nodes = current_n
                for n in range(nodes_at_stage):
                    for k in range(fan_in):
                        partial_idx = n * fan_in + k
                        if partial_idx < prev_nodes:
                            # Offset into prior stage's output buffer
                            offsets[n * fan_in + k] = partial_idx * pw
                        else:
                            # Sentinel for absent partial in tail node
                            offsets[n * fan_in + k] = 0xFFFFFFFF

            stage_offset_lists.append(offsets)
            current_n = nodes_at_stage

        # Upload all offset lists
        offset_handles: list[BufferHandle] = []
        for i, offsets in enumerate(stage_offset_lists):
            handle = BufferHandle(9992 + i)
            desc = BufferDescriptor(
                handle=handle,
                logical_name=f"_reduction_offsets_{i}",
                padded_shape=(len(offsets),),
                element_size_bytes=4,
                size_bytes=offsets.nbytes,
                role=BufferRole.BATCH_INTERMEDIATE,
                precision_role="compute",
                producing_node=None,
                consumers=frozenset(),
                last_consumer=None,
            )
            self._allocator.allocate(desc)
            upload_cmd = self._ctx.allocate_command_buffer()
            upload_fence = self._ctx.create_fence()
            self._allocator.upload_to_device(
                handle, offsets, upload_cmd, self._ctx.compute_queue,
                upload_fence,
            )
            vk.vkDestroyFence(self._ctx.device, upload_fence, None)
            offset_handles.append(handle)

        # Select pipelines
        storage_pipeline = self._pipelines["reduce_k_fan_in_and_clip"]
        compute_pipeline = self._pipelines.get(
            "reduce_k_fan_in_and_clip_from_compute", storage_pipeline
        )

        current_src = src_device_buf
        ping_buf = self._allocator.get_buffer(ping_handle)
        pong_buf = self._allocator.get_buffer(pong_handle)
        current_dst = ping_buf
        current_n = rtp.num_partials

        for stage in range(rtp.num_stages):
            nodes_at_stage = (current_n + fan_in - 1) // fan_in

            # ADR-026 §3: Stage 0 uses variant based on source role;
            # stages ≥ 1 always use compute-entry (prior output is COMPUTE_TYPE)
            if stage == 0 and not use_compute_entry:
                pipeline = storage_pipeline
                shader_name = "reduce_k_fan_in_and_clip"
            else:
                pipeline = compute_pipeline
                shader_name = "reduce_k_fan_in_and_clip_from_compute"

            # If last stage, write to destination
            if stage == rtp.num_stages - 1:
                current_dst = dst_device_buf

            # Get offset buffer for this stage
            offset_buf = self._allocator.get_buffer(offset_handles[stage])

            # Push descriptors: [src_collection, offset_list, dest]
            bindings = [
                (0, current_src),
                (1, offset_buf),
                (2, current_dst),
            ]

            if self._ctx.has_push_descriptors:
                self._descriptor_mgr.push_descriptor_set(
                    cmd, pipeline.layout, bindings
                )
            else:
                layout = self._descriptor_layouts[shader_name]
                tmp_set = self._descriptor_mgr.allocate_set(layout)
                self._descriptor_mgr.update_set(tmp_set, bindings)
                vk.vkCmdBindDescriptorSets(
                    cmd,
                    vk.VK_PIPELINE_BIND_POINT_COMPUTE,
                    pipeline.layout,
                    0, 1, [tmp_set], 0, None,
                )

            vk.vkCmdBindPipeline(
                cmd, vk.VK_PIPELINE_BIND_POINT_COMPUTE,
                pipeline.pipeline,
            )

            # Get threshold for this stage (negative bypasses clip)
            threshold = -1.0  # Diagnostic mode by default
            if (
                rtp.tree_variant == "sum_and_clip"
                and rtp.threshold_schedule
                and stage < len(rtp.threshold_schedule)
                and rtp.threshold_schedule[stage] is not None
            ):
                threshold = float(rtp.threshold_schedule[stage])

            # Push constants for reduce_k_fan_in_and_clip
            params = {
                "src_scalar_NATURAL_fan_in": fan_in,
                "src_scalar_NATURAL_node_count": nodes_at_stage,
                "src_scalar_NATURAL_partial_width": pw,
                "src_scalar_REAL_clipping_threshold": threshold,
                "src_scalar_REAL_epsilon": plan.precision.compute_epsilon,
            }
            pc_bytes = marshal_push_constants(
                shader_name, params,
                compute_dtype=plan.precision.compute_dtype,
            )
            pc_buf = _ffi.from_buffer(pc_bytes)
            vk.vkCmdPushConstants(
                cmd, pipeline.layout,
                vk.VK_SHADER_STAGE_COMPUTE_BIT,
                0, len(pc_bytes), pc_buf,
            )

            # Dispatch: one work-group per reduction node
            vk.vkCmdDispatch(cmd, nodes_at_stage, 1, 1)

            # Barrier between stages
            if stage < rtp.num_stages - 1:
                self._record_barrier(cmd)

            # Ping-pong for next stage
            current_src = current_dst
            current_dst = (
                pong_buf if current_src is ping_buf else ping_buf
            )
            current_n = nodes_at_stage

    def _record_single_stage_aggregate(
        self,
        cmd: Any,
        node: ReductionTreeNode,
        plan: ExecutionPlan,
        spec: SpecConstants,
        use_compute_entry: bool,
        src_device_buf: VulkanBuffer,
        dst_device_buf: VulkanBuffer,
    ) -> None:
        """Record a single-stage diagnostic aggregate (no clipping).

        Used for single-stage diagnostic trees where per-node clip is not needed.
        """
        rtp = node.reduction_plan
        pw = rtp.partial_width
        simd_w = plan.hardware.simd_width
        fan_in = rtp.fan_in
        use_local = fan_in > simd_w

        # Upload offset list
        from ...shared.buffer_lifecycle import BufferDescriptor

        offset_handle = BufferHandle(9992)
        offsets = np.array(rtp.initial_offset_list, dtype=np.uint32)
        offset_desc = BufferDescriptor(
            handle=offset_handle,
            logical_name="_reduction_offsets",
            padded_shape=(len(offsets),),
            element_size_bytes=4,
            size_bytes=offsets.nbytes,
            role=BufferRole.BATCH_INTERMEDIATE,
            precision_role="compute",
            producing_node=None,
            consumers=frozenset(),
            last_consumer=None,
        )
        self._allocator.allocate(offset_desc)
        upload_cmd = self._ctx.allocate_command_buffer()
        upload_fence = self._ctx.create_fence()
        self._allocator.upload_to_device(
            offset_handle, offsets, upload_cmd, self._ctx.compute_queue,
            upload_fence,
        )
        vk.vkDestroyFence(self._ctx.device, upload_fence, None)

        # Select pipeline
        if use_compute_entry:
            pipeline = self._pipelines["aggregate_partials_from_compute"]
            shader_name = "aggregate_partials_from_compute"
        else:
            pipeline = self._pipelines["aggregate_partials"]
            shader_name = "aggregate_partials"

        offset_buf = self._allocator.get_buffer(offset_handle)

        # Push descriptors: [src, offset_list, dst]
        bindings = [
            (0, src_device_buf),
            (1, offset_buf),
            (2, dst_device_buf),
        ]

        if self._ctx.has_push_descriptors:
            self._descriptor_mgr.push_descriptor_set(
                cmd, pipeline.layout, bindings
            )
        else:
            layout = self._descriptor_layouts[shader_name]
            tmp_set = self._descriptor_mgr.allocate_set(layout)
            self._descriptor_mgr.update_set(tmp_set, bindings)
            vk.vkCmdBindDescriptorSets(
                cmd,
                vk.VK_PIPELINE_BIND_POINT_COMPUTE,
                pipeline.layout,
                0, 1, [tmp_set], 0, None,
            )

        vk.vkCmdBindPipeline(
            cmd, vk.VK_PIPELINE_BIND_POINT_COMPUTE,
            pipeline.pipeline,
        )

        # Push constants for aggregate
        params = {
            "src_scalar_NATURAL_partial_offset_list_count": len(
                rtp.initial_offset_list
            ),
            "src_scalar_NATURAL_partial_width": pw,
            "src_scalar_NATURAL_operation_type": 0,  # SUM
            "src_scalar_FLAG_use_local_reduce": int(use_local),
        }
        pc_bytes = marshal_push_constants("aggregate_partials", params)
        pc_buf = _ffi.from_buffer(pc_bytes)
        vk.vkCmdPushConstants(
            cmd, pipeline.layout,
            vk.VK_SHADER_STAGE_COMPUTE_BIT,
            0, len(pc_bytes), pc_buf,
        )

        # Dispatch based on tier
        if use_local:
            dispatch_x = pw
        else:
            dispatch_x = (pw + simd_w - 1) // simd_w
        vk.vkCmdDispatch(cmd, dispatch_x, 1, 1)

    def _record_streaming_loop(
        self,
        cmd: Any,
        node: StreamingLoopNode,
        plan: ExecutionPlan,
    ) -> None:
        """Record a streaming loop with per-chunk push constant updates."""
        sp = node.streaming_plan

        for chunk_index in range(sp.iteration.chunk_count):
            for body_node_id in sp.body:
                body_node = plan.nodes[body_node_id]
                if isinstance(body_node, KernelDispatchNode):
                    adjusted = self._apply_strides(
                        body_node, sp, chunk_index
                    )
                    self._record_kernel_dispatch(cmd, adjusted, plan)

            # Barrier between chunks
            if chunk_index < sp.iteration.chunk_count - 1:
                self._record_barrier(cmd)

    def _record_barrier(self, cmd: Any) -> None:
        """Record a compute→compute pipeline barrier."""
        mem_barrier = vk.VkMemoryBarrier(
            srcAccessMask=vk.VK_ACCESS_SHADER_WRITE_BIT,
            dstAccessMask=vk.VK_ACCESS_SHADER_READ_BIT,
        )
        vk.vkCmdPipelineBarrier(
            cmd,
            vk.VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,
            vk.VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,
            0,
            1,
            [mem_barrier],
            0,
            None,
            0,
            None,
        )

    def _record_retrieval(
        self,
        cmd: Any,
        node: RetrievalNode,
        plan: ExecutionPlan,
    ) -> Any:
        """Record a compute→transfer barrier + copy to staging buffer.

        Returns the VulkanStagingBuffer for future readback.
        """
        buf_desc = plan.buffers[node.source_buffer]
        vk_buf = self._allocator.get_buffer(node.source_buffer)

        # Allocate staging buffer for readback
        staging = self._allocator.allocate_staging(buf_desc.size_bytes)

        # Compute → transfer barrier
        buf_barrier = vk.VkBufferMemoryBarrier(
            srcAccessMask=vk.VK_ACCESS_SHADER_WRITE_BIT,
            dstAccessMask=vk.VK_ACCESS_TRANSFER_READ_BIT,
            srcQueueFamilyIndex=vk.VK_QUEUE_FAMILY_IGNORED,
            dstQueueFamilyIndex=vk.VK_QUEUE_FAMILY_IGNORED,
            buffer=vk_buf.buffer,
            offset=0,
            size=buf_desc.size_bytes,
        )
        vk.vkCmdPipelineBarrier(
            cmd,
            vk.VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,
            vk.VK_PIPELINE_STAGE_TRANSFER_BIT,
            0,
            0,
            None,
            1,
            [buf_barrier],
            0,
            None,
        )

        # Copy device buffer to staging
        region = vk.VkBufferCopy(
            srcOffset=0, dstOffset=0, size=buf_desc.size_bytes
        )
        vk.vkCmdCopyBuffer(cmd, vk_buf.buffer, staging.buffer, 1, [region])

        return staging

    # ── Helpers ──

    @staticmethod
    def _apply_strides(
        node: KernelDispatchNode,
        sp: StreamingLoopPlan,
        chunk_index: int,
    ) -> KernelDispatchNode:
        """Create a copy of a KernelDispatchNode with per-chunk strides applied."""
        adjusted_params = dict(node.scalar_params)
        for stride in sp.parameter_strides:
            adjusted_params[stride.param_name] = (
                stride.base + chunk_index * stride.stride
            )
        for name, value in sp.constant_scalars.items():
            adjusted_params[name] = value
        return replace(node, scalar_params=adjusted_params)
