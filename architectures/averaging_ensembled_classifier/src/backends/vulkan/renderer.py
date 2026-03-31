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
from ...shared.streaming_loop_plan import StreamingLoopPlan
from ._descriptor_manager import VulkanDescriptorManager
from ._pipeline_cache import ComputePipeline, SpecConstants, VulkanPipelineCache
from ._push_constants import (
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
_REDUCTION_SHADERS = frozenset({"aggregate_partials", "clip_intermediate_grad"})


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
        self, plan: ExecutionPlan
    ) -> dict[str, RetrievalFuture]:
        """Record and submit the plan, returning futures for retrieval nodes."""
        # 1. Allocate device-local buffers
        for descriptor in plan.buffers.values():
            self._allocator.allocate(descriptor)

        # 2. Upload input data (BATCH_INPUT buffers with initial data)
        upload_cmd = self._ctx.allocate_command_buffer()
        upload_fence = self._ctx.create_fence()
        for descriptor in plan.buffers.values():
            if descriptor.role == BufferRole.BATCH_INPUT:
                # Input data is uploaded by the caller before render;
                # the plan model does not carry raw data arrays.
                pass

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
                    dtype=plan.precision.numpy_dtype,
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
                if node.reduction_plan.tree_variant == "sum_and_clip":
                    kernel_names.add("clip_intermediate_grad")

        for name in kernel_names:
            binding_count = DESCRIPTOR_BINDING_COUNTS[name]
            layout = self._descriptor_mgr.create_layout(binding_count)
            self._descriptor_layouts[name] = layout

            push_size = ctypes.sizeof(PUSH_CONSTANT_STRUCTS[name])
            pipeline = self._pipeline_cache.create_pipeline(
                name, layout, push_size, spec
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
            for binding_idx, (_, handle) in enumerate(
                sorted(bindings.items())
            ):
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
            (int(s["total_modules_count"]) + w - 1) // w
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

        # Push constants
        pc_bytes = marshal_push_constants(
            kernel, node.scalar_params
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
            # 2D dispatch: tile_count encodes (x * y), local_work_size
            # provides the second dimension hint
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
                    node.scalar_params, plan.hardware.simd_width,
                )
                vk.vkCmdDispatch(cmd, wg_count, 1, 1)
            else:
                vk.vkCmdDispatch(cmd, node.tile_count, 1, 1)

    def _record_reduction_tree(
        self,
        cmd: Any,
        node: ReductionTreeNode,
        plan: ExecutionPlan,
        spec: SpecConstants,
    ) -> None:
        """Record a staged reduction tree into the command buffer.

        Uses aggregate_partials with ping-pong buffers and optional
        clip_intermediate_grad between stages.
        """
        rtp = node.reduction_plan

        # Get source/destination device buffers
        src_device_buf = self._allocator.get_buffer(rtp.source_buffer)
        dst_device_buf = self._allocator.get_buffer(rtp.destination_buffer)

        # Allocate ping-pong scratch buffers for intermediate stages
        pw = rtp.partial_width
        max_intermediates = rtp.num_partials
        ping_size = max_intermediates * pw * 4  # float32
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
            element_size_bytes=4,
            size_bytes=ping_size,
            role=BufferRole.BATCH_INTERMEDIATE,
            producing_node=None,
            consumers=frozenset(),
            last_consumer=None,
        )
        pong_desc = BufferDescriptor(
            handle=pong_handle,
            logical_name="_reduction_pong",
            padded_shape=(max_intermediates * pw,),
            element_size_bytes=4,
            size_bytes=pong_size,
            role=BufferRole.BATCH_INTERMEDIATE,
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

        agg_pipeline = self._pipelines["aggregate_partials"]
        clip_pipeline = (
            self._pipelines.get("clip_intermediate_grad")
            if rtp.tree_variant == "sum_and_clip"
            else None
        )

        # Determine tier per stage
        simd_w = plan.hardware.simd_width
        fan_in = rtp.fan_in_K

        current_src = src_device_buf
        ping_buf = self._allocator.get_buffer(ping_handle)
        pong_buf = self._allocator.get_buffer(pong_handle)
        offset_buf = self._allocator.get_buffer(offset_handle)

        current_n = rtp.num_partials
        current_dst = ping_buf

        for stage in range(rtp.num_stages):
            nodes_at_stage = (current_n + fan_in - 1) // fan_in
            use_local = fan_in > simd_w

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
                agg_layout = self._descriptor_layouts["aggregate_partials"]
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
                    "src_scalar_REAL_epsilon": plan.precision.epsilon,
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
