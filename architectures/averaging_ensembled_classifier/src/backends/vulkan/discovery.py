"""Vulkan hardware discovery — HardwareProfile population (ADR-006)."""
from __future__ import annotations

from typing import TYPE_CHECKING

from ...shared.hardware_profile import HardwareProfile
from .context import VulkanContext

if TYPE_CHECKING:
    import vulkan as vk  # type: ignore[import-untyped]
else:
    try:
        import vulkan as vk
    except ImportError:
        vk = None  # type: ignore[assignment]


def discover_hardware(context: VulkanContext) -> HardwareProfile:
    """Populate HardwareProfile from Vulkan device property queries."""
    dev = context.physical_device

    # Base properties
    props = vk.vkGetPhysicalDeviceProperties(dev)
    limits = props.limits  # type: ignore[reportAttributeAccessIssue]

    # Subgroup properties (core since Vulkan 1.1)
    subgroup_props = vk.VkPhysicalDeviceSubgroupProperties(
        sType=vk.VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_SUBGROUP_PROPERTIES,
    )
    props2 = vk.VkPhysicalDeviceProperties2(
        sType=vk.VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_PROPERTIES_2,
        pNext=subgroup_props,
    )
    vk.vkGetPhysicalDeviceProperties2(dev, props2)
    subgroup_size = subgroup_props.subgroupSize

    # Memory properties — find largest device-local heap
    mem_props = vk.vkGetPhysicalDeviceMemoryProperties(dev)
    device_local_bytes = 0
    for i in range(mem_props.memoryHeapCount):  # type: ignore[reportAttributeAccessIssue]
        heap = mem_props.memoryHeaps[i]  # type: ignore[reportAttributeAccessIssue]
        if heap.flags & vk.VK_MEMORY_HEAP_DEVICE_LOCAL_BIT:
            device_local_bytes = max(device_local_bytes, heap.size)

    # Shared memory limit
    max_shared_mem = limits.maxComputeSharedMemorySize  # type: ignore[reportAttributeAccessIssue]

    # Cache line estimate: subgroup_size * 4 bytes (FP32)
    cache_line_bytes = subgroup_size * 4

    # max_reduce_fan_in: how many elements we can reduce in shared memory
    # Conservative: shared_mem / sizeof(float), capped by max workgroup size
    max_workgroup_x = limits.maxComputeWorkGroupSize[0]  # type: ignore[reportAttributeAccessIssue]
    fan_in_from_mem = max_shared_mem // 4  # floats that fit in shared mem
    max_reduce_fan_in = min(max_workgroup_x, fan_in_from_mem)

    return HardwareProfile(
        simd_width=subgroup_size,
        cache_line_bytes=cache_line_bytes,
        max_reduce_fan_in=max_reduce_fan_in,
        max_local_mem_bytes=max_shared_mem,
        global_mem_bytes=device_local_bytes,
    )
