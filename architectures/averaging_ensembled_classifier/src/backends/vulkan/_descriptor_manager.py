"""Descriptor set and layout management for the Vulkan backend.

Two-track strategy (VULKAN_BACKEND.md §Descriptor Set Strategy):
  - Fixed pipelines: pre-allocated descriptor sets, updated once per batch.
  - Dynamic pipelines (reduction engine): vkCmdPushDescriptorSetKHR or
    fallback to pre-allocated sets when push descriptors are unavailable.
"""
from __future__ import annotations

from typing import TYPE_CHECKING, Any

from .buffer_allocator import VulkanBuffer
from .context import VulkanContext

if TYPE_CHECKING:
    import vulkan as vk  # type: ignore[import-untyped]
else:
    try:
        import vulkan as vk
    except ImportError:
        vk = None  # type: ignore[assignment]


class VulkanDescriptorManager:
    """Manages descriptor set layouts, pool, and set updates."""

    def __init__(self, context: VulkanContext) -> None:
        self._ctx = context
        self._device = context.device
        self._pool: Any = None
        self._layouts: list[Any] = []
        self._sets: list[Any] = []
        self._max_sets = 64
        self._max_bindings = 640  # 64 sets × 10 max bindings

        self._create_pool()

    def create_layout(self, binding_count: int) -> Any:
        """Create a VkDescriptorSetLayout with N storage buffer bindings."""
        bindings = [
            vk.VkDescriptorSetLayoutBinding(
                binding=i,
                descriptorType=vk.VK_DESCRIPTOR_TYPE_STORAGE_BUFFER,
                descriptorCount=1,
                stageFlags=vk.VK_SHADER_STAGE_COMPUTE_BIT,
            )
            for i in range(binding_count)
        ]
        layout_info = vk.VkDescriptorSetLayoutCreateInfo(
            bindingCount=binding_count,
            pBindings=bindings,
        )
        layout = vk.vkCreateDescriptorSetLayout(
            self._device, layout_info, None
        )
        self._layouts.append(layout)
        return layout

    def allocate_set(self, layout: Any) -> Any:
        """Allocate a descriptor set from the pool."""
        alloc_info = vk.VkDescriptorSetAllocateInfo(
            descriptorPool=self._pool,
            descriptorSetCount=1,
            pSetLayouts=[layout],
        )
        desc_set = vk.vkAllocateDescriptorSets(self._device, alloc_info)[0]
        self._sets.append(desc_set)
        return desc_set

    def update_set(
        self,
        desc_set: Any,
        bindings: list[tuple[int, VulkanBuffer]],
    ) -> None:
        """Update descriptor set bindings with buffer references."""
        writes = []
        for binding_idx, vk_buf in bindings:
            buf_info = vk.VkDescriptorBufferInfo(
                buffer=vk_buf.buffer,
                offset=0,
                range=vk_buf.size_bytes,
            )
            write = vk.VkWriteDescriptorSet(
                dstSet=desc_set,
                dstBinding=binding_idx,
                dstArrayElement=0,
                descriptorCount=1,
                descriptorType=vk.VK_DESCRIPTOR_TYPE_STORAGE_BUFFER,
                pBufferInfo=[buf_info],
            )
            writes.append(write)
        if writes:
            vk.vkUpdateDescriptorSets(
                self._device, len(writes), writes, 0, None
            )

    def push_descriptor_set(
        self,
        cmd: Any,
        layout: Any,
        bindings: list[tuple[int, VulkanBuffer]],
    ) -> None:
        """Record push descriptor set updates into a command buffer.

        Falls back to regular descriptor set update if push descriptors
        are unavailable (the caller must provide a pre-allocated set).
        """
        writes = []
        for binding_idx, vk_buf in bindings:
            buf_info = vk.VkDescriptorBufferInfo(
                buffer=vk_buf.buffer,
                offset=0,
                range=vk_buf.size_bytes,
            )
            write = vk.VkWriteDescriptorSet(
                dstSet=None,
                dstBinding=binding_idx,
                dstArrayElement=0,
                descriptorCount=1,
                descriptorType=vk.VK_DESCRIPTOR_TYPE_STORAGE_BUFFER,
                pBufferInfo=[buf_info],
            )
            writes.append(write)
        if writes:
            push_fn = self._ctx.cmd_push_descriptor_set_khr
            push_fn(  # type: ignore[reportCallIssue]
                cmd,
                vk.VK_PIPELINE_BIND_POINT_COMPUTE,
                layout,
                0,
                len(writes),
                writes,
            )

    def destroy(self) -> None:
        """Destroy all descriptor sets (via pool), layouts, and pool."""
        if self._pool is not None:
            vk.vkDestroyDescriptorPool(self._device, self._pool, None)
            self._pool = None
        self._sets.clear()

        for layout in self._layouts:
            vk.vkDestroyDescriptorSetLayout(self._device, layout, None)
        self._layouts.clear()

    # ── Internal ──

    def _create_pool(self) -> None:
        pool_size = vk.VkDescriptorPoolSize(
            type=vk.VK_DESCRIPTOR_TYPE_STORAGE_BUFFER,
            descriptorCount=self._max_bindings,
        )
        pool_info = vk.VkDescriptorPoolCreateInfo(
            maxSets=self._max_sets,
            poolSizeCount=1,
            pPoolSizes=[pool_size],
        )
        self._pool = vk.vkCreateDescriptorPool(
            self._device, pool_info, None
        )
