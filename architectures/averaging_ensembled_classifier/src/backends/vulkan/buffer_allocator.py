"""Vulkan buffer allocator — device-local buffer management (ADR-009).

Translates BufferDescriptor plan-level declarations into physical
VkBuffer + VkDeviceMemory allocations. Manages staging buffers for
host-device data transfer.
"""
# pyright: reportUnknownVariableType=false, reportUnknownMemberType=false, reportUnknownArgumentType=false
from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

import numpy as np

from ...shared.buffer_lifecycle import BufferDescriptor, BufferHandle, BufferRole
from .context import VulkanContext

if TYPE_CHECKING:
    import vulkan as vk  # type: ignore[import-untyped]
else:
    try:
        import vulkan as vk
    except ImportError:
        vk = None  # type: ignore[assignment]


@dataclass
class VulkanBuffer:
    """A device-side buffer with its backing memory."""

    buffer: Any  # VkBuffer
    memory: Any  # VkDeviceMemory
    size_bytes: int
    descriptor: BufferDescriptor


@dataclass
class VulkanStagingBuffer:
    """A host-visible staging buffer for D2H / H2D transfers."""

    buffer: Any  # VkBuffer
    memory: Any  # VkDeviceMemory
    mapped_buf: Any  # cffi buffer from vkMapMemory
    size_bytes: int


class VulkanBufferAllocator:
    """Allocates Vulkan buffers from BufferDescriptors."""

    def __init__(self, context: VulkanContext) -> None:
        self._ctx = context
        self._device = context.device
        self._physical_device = context.physical_device
        self._mem_props = vk.vkGetPhysicalDeviceMemoryProperties(
            self._physical_device
        )
        self._buffers: dict[BufferHandle, VulkanBuffer] = {}
        self._staging_buffers: list[VulkanStagingBuffer] = []

    def allocate(self, descriptor: BufferDescriptor) -> VulkanBuffer:
        """Allocate a device-local storage buffer for the given descriptor."""
        usage = vk.VK_BUFFER_USAGE_STORAGE_BUFFER_BIT
        # Buffers that may be copied to host need transfer_src
        if descriptor.role == BufferRole.BATCH_OUTPUT:
            usage |= vk.VK_BUFFER_USAGE_TRANSFER_SRC_BIT
        # All buffers may need to receive uploaded data or be zero-filled
        usage |= vk.VK_BUFFER_USAGE_TRANSFER_DST_BIT

        buf = self._create_buffer(
            descriptor.size_bytes,
            usage,
            vk.VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT,
        )
        vk_buf = VulkanBuffer(
            buffer=buf[0],
            memory=buf[1],
            size_bytes=descriptor.size_bytes,
            descriptor=descriptor,
        )
        self._buffers[descriptor.handle] = vk_buf
        return vk_buf

    def zero_fill(
        self,
        handle: BufferHandle,
        command_buffer: Any,
        queue: Any,
        fence: Any,
    ) -> None:
        """Zero-fill a device-local buffer via staging upload."""
        vk_buf = self._buffers[handle]
        size = vk_buf.size_bytes
        staging = self.allocate_staging(size)
        from vulkan._vulkan import ffi as _ffi  # noqa: PLC0415  # pyright: ignore[reportMissingTypeStubs]
        _ffi.memmove(staging.mapped_buf, bytes(size), size)

        begin_info = vk.VkCommandBufferBeginInfo(
            flags=vk.VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT,
        )
        vk.vkBeginCommandBuffer(command_buffer, begin_info)
        region = vk.VkBufferCopy(srcOffset=0, dstOffset=0, size=size)
        vk.vkCmdCopyBuffer(
            command_buffer, staging.buffer, vk_buf.buffer, 1, [region],
        )
        vk.vkEndCommandBuffer(command_buffer)

        submit = vk.VkSubmitInfo(
            commandBufferCount=1,
            pCommandBuffers=[command_buffer],
        )
        vk.vkQueueSubmit(queue, 1, [submit], fence)
        vk.vkWaitForFences(self._device, 1, [fence], True, int(5e9))
        vk.vkResetFences(self._device, 1, [fence])
        vk.vkResetCommandBuffer(command_buffer, 0)

    def get_buffer(self, handle: BufferHandle) -> VulkanBuffer:
        """Retrieve the VulkanBuffer for a given buffer handle."""
        return self._buffers[handle]

    def allocate_staging(self, size_bytes: int) -> VulkanStagingBuffer:
        """Allocate a host-visible, host-coherent staging buffer."""
        buf, mem = self._create_buffer(
            size_bytes,
            vk.VK_BUFFER_USAGE_TRANSFER_SRC_BIT
            | vk.VK_BUFFER_USAGE_TRANSFER_DST_BIT,
            vk.VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT
            | vk.VK_MEMORY_PROPERTY_HOST_COHERENT_BIT,
        )
        # Persistently map
        mapped = vk.vkMapMemory(self._device, mem, 0, size_bytes, 0)
        staging = VulkanStagingBuffer(
            buffer=buf,
            memory=mem,
            mapped_buf=mapped,
            size_bytes=size_bytes,
        )
        self._staging_buffers.append(staging)
        return staging

    def upload_to_device(
        self,
        handle: BufferHandle,
        data: np.ndarray,
        command_buffer: Any,
        queue: Any,
        fence: Any,
    ) -> None:
        """Upload numpy array data to a device-local buffer via staging."""
        vk_buf = self._buffers[handle]
        size = data.nbytes

        # Create transient staging buffer
        staging = self.allocate_staging(size)
        # Copy data to mapped staging memory via cffi
        from vulkan._vulkan import ffi as _ffi  # noqa: PLC0415  # pyright: ignore[reportMissingTypeStubs]
        _ffi.memmove(staging.mapped_buf, bytes(data.data), size)

        # Record copy command
        begin_info = vk.VkCommandBufferBeginInfo(
            flags=vk.VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT,
        )
        vk.vkBeginCommandBuffer(command_buffer, begin_info)
        region = vk.VkBufferCopy(srcOffset=0, dstOffset=0, size=size)
        vk.vkCmdCopyBuffer(
            command_buffer, staging.buffer, vk_buf.buffer, 1, [region]
        )
        vk.vkEndCommandBuffer(command_buffer)

        # Submit and wait
        submit = vk.VkSubmitInfo(
            commandBufferCount=1,
            pCommandBuffers=[command_buffer],
        )
        vk.vkQueueSubmit(queue, 1, [submit], fence)
        vk.vkWaitForFences(self._device, 1, [fence], vk.VK_TRUE, 2**64 - 1)
        vk.vkResetFences(self._device, 1, [fence])
        vk.vkResetCommandBuffer(command_buffer, 0)

        # Free transient staging
        self._destroy_staging(staging)

    def release_batch_buffers(self) -> None:
        """Release per-batch buffers, keeping model state."""
        to_remove = [
            h
            for h, b in self._buffers.items()
            if b.descriptor.role != BufferRole.MODEL_STATE
        ]
        for handle in to_remove:
            vk_buf = self._buffers.pop(handle)
            vk.vkDestroyBuffer(self._device, vk_buf.buffer, None)
            vk.vkFreeMemory(self._device, vk_buf.memory, None)

    def destroy(self) -> None:
        """Free all remaining buffers and staging memory."""
        for vk_buf in self._buffers.values():
            vk.vkDestroyBuffer(self._device, vk_buf.buffer, None)
            vk.vkFreeMemory(self._device, vk_buf.memory, None)
        self._buffers.clear()

        for staging in self._staging_buffers:
            self._destroy_staging(staging)
        self._staging_buffers.clear()

    # ── Internal helpers ──

    def _create_buffer(
        self,
        size_bytes: int,
        usage: int,
        memory_properties: int,
    ) -> tuple[Any, Any]:
        """Create a VkBuffer and allocate backing memory."""
        buf_info = vk.VkBufferCreateInfo(
            size=size_bytes,
            usage=usage,
            sharingMode=vk.VK_SHARING_MODE_EXCLUSIVE,
        )
        buffer = vk.vkCreateBuffer(self._device, buf_info, None)

        mem_req = vk.vkGetBufferMemoryRequirements(self._device, buffer)
        mem_type_idx = self._find_memory_type(
            mem_req.memoryTypeBits, memory_properties  # type: ignore[reportAttributeAccessIssue]
        )

        alloc_info = vk.VkMemoryAllocateInfo(
            allocationSize=mem_req.size,  # type: ignore[reportAttributeAccessIssue]
            memoryTypeIndex=mem_type_idx,
        )
        memory = vk.vkAllocateMemory(self._device, alloc_info, None)
        vk.vkBindBufferMemory(self._device, buffer, memory, 0)

        return buffer, memory

    def _find_memory_type(
        self, type_filter: int, properties: int
    ) -> int:
        """Find a memory type index satisfying the filter and property flags."""
        for i in range(self._mem_props.memoryTypeCount):  # type: ignore[reportAttributeAccessIssue]
            if (type_filter & (1 << i)) and (
                self._mem_props.memoryTypes[i].propertyFlags & properties  # type: ignore[reportAttributeAccessIssue]
            ) == properties:
                return i
        raise RuntimeError(
            f"No suitable memory type found for properties {properties:#x}. "
            f"Available types: {[self._mem_props.memoryTypes[i].propertyFlags for i in range(self._mem_props.memoryTypeCount)]}"  # type: ignore[reportAttributeAccessIssue]
        )

    def _destroy_staging(self, staging: VulkanStagingBuffer) -> None:
        """Unmap and destroy a staging buffer."""
        vk.vkUnmapMemory(self._device, staging.memory)
        vk.vkDestroyBuffer(self._device, staging.buffer, None)
        vk.vkFreeMemory(self._device, staging.memory, None)
        if staging in self._staging_buffers:
            self._staging_buffers.remove(staging)
