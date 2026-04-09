"""VulkanPlanRenderer — PlanRenderer implementation for the Vulkan backend (ADR-001).

Records ExecutionPlan DAG nodes into Vulkan command buffers and submits
them for GPU execution. Supports all five node types: KernelDispatchNode,
ReductionTreeNode, StreamingLoopNode, BarrierNode, and RetrievalNode.
"""
from __future__ import annotations

import ctypes
import logging
from dataclasses import replace
from typing import TYPE_CHECKING, Any

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
from ...shared.reduction_tree_plan import ReductionTreePlan
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
        from vulkan._vulkan import ffi as _ffi
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


        # Per-render state (set during render())
        self._pipelines: dict[str, ComputePipeline] = {}
        self._descriptor_sets: dict[str, Any] = {}
        self._descriptor_layouts: dict[str, Any] = {}

    def render(
        self,
        plan: ExecutionPlan,
        data_injections: dict[str, NDArray] | None = None,
    ) -> dict[str, RetrievalFuture]:
        """Record and submit the plan, returning futures for retrieval nodes."""
        # 1. Allocate device-local buffers and zero-fill them
        upload_cmd = self._ctx.allocate_command_buffer()
        upload_fence = self._ctx.create_fence()
        for descriptor in plan.buffers.values():
            self._allocator.allocate(descriptor)
            self._allocator.zero_fill(
                descriptor.handle, upload_cmd,
                self._ctx.compute_queue, upload_fence,
            )

        # 2. Initialize MODEL_STATE buffers (Xavier/unit) and upload
        self._init_model_state_buffers(
            plan, upload_cmd, self._ctx.compute_queue, upload_fence,
        )

        # 3. Build specialization constants
        spec = SpecConstants(
            simd_width=plan.hardware.simd_width,
            bank_padding=1,
            tile_size=8,
            problem_type=self._problem_type,
        )

        # 4. Create pipelines and descriptor layouts for all kernels in the plan
        self._prepare_pipelines(plan, spec)

        # 5. Update descriptor sets with buffer bindings
        self._update_descriptor_sets(plan)

        # 6. Record command buffer
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
            elif isinstance(node, RetrievalNode):
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

        # 7. Submit
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
    ) -> None:
        """Upload Xavier/unit-initialized values for MODEL_STATE buffers."""
        role_dtypes = {
            "storage": plan.precision.storage_dtype,
            "compute": plan.precision.compute_dtype,
            "state": plan.precision.state_dtype,
        }
        for descriptor in plan.buffers.values():
            if descriptor.role != BufferRole.MODEL_STATE:
                continue
            name = descriptor.logical_name
            total = int(np.prod(descriptor.padded_shape))
            dtype = role_dtypes[descriptor.precision_role]
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
                kernel_names.add("aggregate_partials")
                # ADR-026: Compute-entry variant for compute-role sources
                # or interior stages of multi-stage trees
                source_desc = plan.buffers[node.reduction_plan.source_buffer]
                if source_desc.precision_role == "compute" or node.reduction_plan.num_stages > 1:
                    kernel_names.add("aggregate_partials_from_compute")
                if node.reduction_plan.tree_variant == "sum_and_clip":
                    kernel_names.add("clip_intermediate_grad")

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
    _WORKGROUP_COUNT_RESOLVERS: dict[str, Any] = {
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
            kernel, scalar_params
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
            x_groups = node.tile_count
            y_groups = 1
            if node.local_work_size is not None and node.local_work_size > 0:
                y_groups = node.local_work_size
                x_groups = node.tile_count
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

        ADR-026: Selects between storage-entry and compute-entry kernel
        variants based on the source buffer's precision_role and stage.

        Uses aggregate_partials with ping-pong buffers and optional
        clip_intermediate_grad between stages.
        """
        rtp = node.reduction_plan

        # ADR-026 §3: Determine if source buffer is compute-role
        source_desc = plan.buffers[rtp.source_buffer]
        use_compute_entry = source_desc.precision_role == "compute"

        # Get source/destination device buffers
        src_device_buf = self._allocator.get_buffer(rtp.source_buffer)
        dst_device_buf = self._allocator.get_buffer(rtp.destination_buffer)

        # Allocate ping-pong scratch buffers for intermediate stages
        pw = rtp.partial_width
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

        # Upload offset list to device
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

        # Upload offset list via separate command buffer
        upload_cmd = self._ctx.allocate_command_buffer()
        upload_fence = self._ctx.create_fence()
        self._allocator.upload_to_device(
            offset_handle, offsets, upload_cmd, self._ctx.compute_queue,
            upload_fence,
        )
        vk.vkDestroyFence(self._ctx.device, upload_fence, None)

        # ADR-026 §3: Select pipeline variants based on source role and stage
        storage_pipeline = self._pipelines["aggregate_partials"]
        compute_pipeline = self._pipelines.get(
            "aggregate_partials_from_compute", storage_pipeline
        )
        clip_pipeline = (
            self._pipelines.get("clip_intermediate_grad")
            if rtp.tree_variant == "sum_and_clip"
            else None
        )

        # Determine tier per stage
        simd_w = plan.hardware.simd_width
        fan_in = rtp.fan_in

        current_src = src_device_buf
        ping_buf = self._allocator.get_buffer(ping_handle)
        pong_buf = self._allocator.get_buffer(pong_handle)
        offset_buf = self._allocator.get_buffer(offset_handle)

        current_n = rtp.num_partials
        current_dst = ping_buf

        for stage in range(rtp.num_stages):
            nodes_at_stage = (current_n + fan_in - 1) // fan_in
            use_local = fan_in > simd_w

            # ADR-026 §3: Stage 0 uses variant based on source role;
            # stages ≥ 1 always use compute-entry (prior output is COMPUTE_TYPE)
            if stage == 0 and not use_compute_entry:
                agg_pipeline = storage_pipeline
                agg_shader_name = "aggregate_partials"
            else:
                agg_pipeline = compute_pipeline
                agg_shader_name = "aggregate_partials_from_compute"

            # If last stage, write to destination
            if stage == rtp.num_stages - 1:
                current_dst = dst_device_buf

            # Push descriptors for aggregate: [src, offset_list, dst]
            agg_bindings = [
                (0, current_src),
                (1, offset_buf),
                (2, current_dst),
            ]

            if self._ctx.has_push_descriptors:
                self._descriptor_mgr.push_descriptor_set(
                    cmd, agg_pipeline.layout, agg_bindings
                )
            else:
                # Fallback: allocate a temporary descriptor set
                agg_layout = self._descriptor_layouts[agg_shader_name]
                tmp_set = self._descriptor_mgr.allocate_set(agg_layout)
                self._descriptor_mgr.update_set(tmp_set, agg_bindings)
                vk.vkCmdBindDescriptorSets(
                    cmd,
                    vk.VK_PIPELINE_BIND_POINT_COMPUTE,
                    agg_pipeline.layout,
                    0, 1, [tmp_set], 0, None,
                )

            vk.vkCmdBindPipeline(
                cmd, vk.VK_PIPELINE_BIND_POINT_COMPUTE,
                agg_pipeline.pipeline,
            )

            # Push constants for aggregate
            agg_params = {
                "src_scalar_NATURAL_partial_offset_list_count": len(
                    rtp.initial_offset_list
                ),
                "src_scalar_NATURAL_partial_width": pw,
                "src_scalar_NATURAL_operation_type": 0,  # SUM
                "src_scalar_FLAG_use_local_reduce": int(use_local),
            }
            pc_bytes = marshal_push_constants(
                "aggregate_partials", agg_params
            )
            pc_buf = _ffi.from_buffer(pc_bytes)
            vk.vkCmdPushConstants(
                cmd, agg_pipeline.layout,
                vk.VK_SHADER_STAGE_COMPUTE_BIT,
                0, len(pc_bytes), pc_buf,
            )

            vk.vkCmdDispatch(cmd, nodes_at_stage, 1, 1)

            # Barrier between aggregate and clip (or next stage)
            self._record_barrier(cmd)

            # Clip stage (if sum_and_clip variant)
            if (
                clip_pipeline is not None
                and rtp.threshold_schedule
                and stage < len(rtp.threshold_schedule)
                and rtp.threshold_schedule[stage] is not None
            ):
                threshold = rtp.threshold_schedule[stage]
                assert threshold is not None

                # clip_intermediate_grad operates in-place on current_dst
                clip_bindings = [(0, current_dst)]
                if self._ctx.has_push_descriptors:
                    self._descriptor_mgr.push_descriptor_set(
                        cmd, clip_pipeline.layout, clip_bindings
                    )
                else:
                    clip_layout = self._descriptor_layouts[
                        "clip_intermediate_grad"
                    ]
                    tmp_clip_set = self._descriptor_mgr.allocate_set(
                        clip_layout
                    )
                    self._descriptor_mgr.update_set(
                        tmp_clip_set, clip_bindings
                    )
                    vk.vkCmdBindDescriptorSets(
                        cmd,
                        vk.VK_PIPELINE_BIND_POINT_COMPUTE,
                        clip_pipeline.layout,
                        0, 1, [tmp_clip_set], 0, None,
                    )

                vk.vkCmdBindPipeline(
                    cmd, vk.VK_PIPELINE_BIND_POINT_COMPUTE,
                    clip_pipeline.pipeline,
                )

                clip_params = {
                    "src_scalar_REAL_clipping_threshold": float(threshold),
                    "src_scalar_REAL_epsilon": plan.precision.compute_epsilon,
                    "src_scalar_NATURAL_parameter_count": nodes_at_stage * pw,
                }
                clip_pc = marshal_push_constants(
                    "clip_intermediate_grad", clip_params
                )
                clip_buf = _ffi.from_buffer(clip_pc)
                vk.vkCmdPushConstants(
                    cmd, clip_pipeline.layout,
                    vk.VK_SHADER_STAGE_COMPUTE_BIT,
                    0, len(clip_pc), clip_buf,
                )

                clip_groups = (nodes_at_stage * pw + 255) // 256
                vk.vkCmdDispatch(cmd, clip_groups, 1, 1)

                self._record_barrier(cmd)

            # Ping-pong for next stage
            current_src = current_dst
            current_dst = (
                pong_buf if current_src is ping_buf else ping_buf
            )
            current_n = nodes_at_stage

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
