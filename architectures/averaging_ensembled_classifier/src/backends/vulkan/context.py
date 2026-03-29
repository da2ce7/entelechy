"""Vulkan device/instance lifecycle management (ADR-015).

Encapsulates VkInstance, VkDevice, VkQueue, VkCommandPool creation
and deterministic teardown. Uses vulkan-python for all Vulkan API calls.
"""
from __future__ import annotations

import logging

logger = logging.getLogger(__name__)

try:
    import vulkan as vk  # type: ignore[import-untyped]

    _VULKAN_AVAILABLE = True
except ImportError:
    _VULKAN_AVAILABLE = False


def _check_vulkan_available() -> None:
    if not _VULKAN_AVAILABLE:
        raise ImportError(
            "vulkan is required for the Vulkan backend. "
            "Install it with: pip install vulkan"
        )


class VulkanContext:
    """Manages the Vulkan device lifecycle for compute workloads.

    Creates an instance, selects a compute-capable physical device
    (preferring discrete GPUs), creates a logical device with a single
    compute queue, and provides command pool / command buffer management.
    """

    def __init__(self, *, enable_validation: bool = False) -> None:
        _check_vulkan_available()

        self._enable_validation = enable_validation
        self._instance: object = None
        self._physical_device: object = None
        self._device: object = None
        self._compute_queue: object = None
        self._command_pool: object = None
        self._queue_family_index: int = -1
        self._vkCmdPushDescriptorSetKHR: object = None

        self._init_instance()
        self._select_physical_device()
        self._create_logical_device()
        self._create_command_pool()

    # ── Properties ──

    @property
    def instance(self) -> object:
        return self._instance

    @property
    def device(self) -> object:
        return self._device

    @property
    def physical_device(self) -> object:
        return self._physical_device

    @property
    def compute_queue(self) -> object:
        return self._compute_queue

    @property
    def queue_family_index(self) -> int:
        return self._queue_family_index

    @property
    def command_pool(self) -> object:
        return self._command_pool

    # ── Initialization ──

    def _init_instance(self) -> None:
        app_info = vk.VkApplicationInfo(
            pApplicationName="AEC-Vulkan-Backend",
            applicationVersion=vk.VK_MAKE_VERSION(0, 1, 0),
            pEngineName="entelechy",
            engineVersion=vk.VK_MAKE_VERSION(0, 1, 0),
            apiVersion=vk.VK_MAKE_VERSION(1, 1, 0),
        )

        layers: list[str] = []
        if self._enable_validation:
            available = vk.vkEnumerateInstanceLayerProperties()
            available_names = {lp.layerName for lp in available}
            if "VK_LAYER_KHRONOS_validation" in available_names:
                layers.append("VK_LAYER_KHRONOS_validation")
            else:
                logger.warning("Validation layer requested but not available")

        create_info = vk.VkInstanceCreateInfo(
            pApplicationInfo=app_info,
            enabledLayerCount=len(layers),
            ppEnabledLayerNames=layers,
        )
        self._instance = vk.vkCreateInstance(create_info, None)

    def _select_physical_device(self) -> None:
        devices = vk.vkEnumeratePhysicalDevices(self._instance)
        if not devices:
            raise RuntimeError("No Vulkan-capable physical devices found")

        # Prefer discrete GPU, fall back to any compute-capable device
        best = None
        best_type = -1
        for dev in devices:
            props = vk.vkGetPhysicalDeviceProperties(dev)
            qf_idx = self._find_compute_queue_family(dev)
            if qf_idx < 0:
                continue

            device_type = props.deviceType
            # Priority: discrete (2) > virtual (3) > integrated (1) > other
            priority = {
                vk.VK_PHYSICAL_DEVICE_TYPE_DISCRETE_GPU: 4,
                vk.VK_PHYSICAL_DEVICE_TYPE_VIRTUAL_GPU: 3,
                vk.VK_PHYSICAL_DEVICE_TYPE_INTEGRATED_GPU: 2,
            }.get(device_type, 1)

            if priority > best_type:
                best = dev
                best_type = priority
                self._queue_family_index = qf_idx

        if best is None:
            raise RuntimeError("No compute-capable Vulkan device found")

        self._physical_device = best
        props = vk.vkGetPhysicalDeviceProperties(best)
        logger.info("Selected Vulkan device: %s", props.deviceName)

    @staticmethod
    def _find_compute_queue_family(device: object) -> int:
        queue_families = vk.vkGetPhysicalDeviceQueueFamilyProperties(device)
        for i, qf in enumerate(queue_families):
            if qf.queueFlags & vk.VK_QUEUE_COMPUTE_BIT:
                return i
        return -1

    def _create_logical_device(self) -> None:
        queue_create = vk.VkDeviceQueueCreateInfo(
            queueFamilyIndex=self._queue_family_index,
            queueCount=1,
            pQueuePriorities=[1.0],
        )

        # Request push descriptor extension if available
        extensions: list[str] = []
        available_exts = vk.vkEnumerateDeviceExtensionProperties(
            self._physical_device, None
        )
        available_ext_names = {e.extensionName for e in available_exts}
        self._has_push_descriptors = (
            "VK_KHR_push_descriptor" in available_ext_names
        )
        if self._has_push_descriptors:
            extensions.append("VK_KHR_push_descriptor")

        device_create = vk.VkDeviceCreateInfo(
            queueCreateInfoCount=1,
            pQueueCreateInfos=[queue_create],
            enabledExtensionCount=len(extensions),
            ppEnabledExtensionNames=extensions,
        )
        self._device = vk.vkCreateDevice(
            self._physical_device, device_create, None
        )
        self._compute_queue = vk.vkGetDeviceQueue(
            self._device, self._queue_family_index, 0
        )

        # Load extension function pointers
        if self._has_push_descriptors:
            self._vkCmdPushDescriptorSetKHR = vk.vkGetDeviceProcAddr(
                self._device, "vkCmdPushDescriptorSetKHR"
            )

    def _create_command_pool(self) -> None:
        pool_info = vk.VkCommandPoolCreateInfo(
            flags=vk.VK_COMMAND_POOL_CREATE_RESET_COMMAND_BUFFER_BIT,
            queueFamilyIndex=self._queue_family_index,
        )
        self._command_pool = vk.vkCreateCommandPool(
            self._device, pool_info, None
        )

    # ── Resource creation helpers ──

    @property
    def has_push_descriptors(self) -> bool:
        return self._has_push_descriptors

    @property
    def cmd_push_descriptor_set_khr(self) -> object:
        """Return the loaded vkCmdPushDescriptorSetKHR function, or None."""
        return self._vkCmdPushDescriptorSetKHR

    def allocate_command_buffer(self) -> object:
        alloc_info = vk.VkCommandBufferAllocateInfo(
            commandPool=self._command_pool,
            level=vk.VK_COMMAND_BUFFER_LEVEL_PRIMARY,
            commandBufferCount=1,
        )
        return vk.vkAllocateCommandBuffers(self._device, alloc_info)[0]

    def create_fence(self, *, signaled: bool = False) -> object:
        flags = vk.VK_FENCE_CREATE_SIGNALED_BIT if signaled else 0
        fence_info = vk.VkFenceCreateInfo(flags=flags)
        return vk.vkCreateFence(self._device, fence_info, None)

    # ── Teardown ──

    def destroy(self) -> None:
        """Destroy all Vulkan objects in reverse creation order."""
        if self._device is None:
            return

        vk.vkDeviceWaitIdle(self._device)

        if self._command_pool is not None:
            vk.vkDestroyCommandPool(self._device, self._command_pool, None)
            self._command_pool = None

        vk.vkDestroyDevice(self._device, None)
        self._device = None

        if self._instance is not None:
            vk.vkDestroyInstance(self._instance, None)
            self._instance = None
