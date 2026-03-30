"""SPIR-V shader loading and Vulkan compute pipeline creation (ADR-014).

Loads compiled .spv modules via importlib.resources, creates VkShaderModule
objects, and builds compute pipelines with specialization constants.
Caches pipelines keyed by (shader_name, problem_type).
"""
from __future__ import annotations

import ctypes
import importlib.resources
import pathlib
import struct
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

from .context import VulkanContext

if TYPE_CHECKING:
    import vulkan as vk  # type: ignore[import-untyped]
else:
    try:
        import vulkan as vk
    except ImportError:
        vk = None  # type: ignore[assignment]


@dataclass(frozen=True)
class SpecConstants:
    """Specialization constant values for pipeline creation."""

    simd_width: int
    bank_padding: int = 1
    tile_size: int = 8
    problem_type: int = 0  # 0=CCE, 1=BCE


@dataclass
class ComputePipeline:
    """A Vulkan compute pipeline with its layout."""

    pipeline: Any  # VkPipeline
    layout: Any  # VkPipelineLayout


class VulkanPipelineCache:
    """Loads SPIR-V modules and creates/caches compute pipelines."""

    def __init__(self, context: VulkanContext) -> None:
        self._ctx = context
        self._device = context.device
        self._pipelines: dict[tuple[str, int], ComputePipeline] = {}
        self._layouts: list[Any] = []  # Track for cleanup

    def create_pipeline(
        self,
        shader_name: str,
        descriptor_layout: Any,  # VkDescriptorSetLayout
        push_constant_size: int,
        spec: SpecConstants,
    ) -> ComputePipeline:
        """Create a compute pipeline from a compiled SPIR-V module.

        Returns a cached pipeline if one already exists for the given
        (shader_name, problem_type) key.
        """
        cache_key = (shader_name, spec.problem_type)
        if cache_key in self._pipelines:
            return self._pipelines[cache_key]

        # Load SPIR-V bytecode
        spv_bytes = self._load_spirv(shader_name)

        # Create shader module (transient — destroyed after pipeline creation)
        module_info = vk.VkShaderModuleCreateInfo(
            codeSize=len(spv_bytes),
            pCode=spv_bytes,
        )
        shader_module = vk.vkCreateShaderModule(
            self._device, module_info, None
        )

        # Pipeline layout: descriptor set layout + push constant range
        push_range = vk.VkPushConstantRange(
            stageFlags=vk.VK_SHADER_STAGE_COMPUTE_BIT,
            offset=0,
            size=push_constant_size,
        )
        layout_info = vk.VkPipelineLayoutCreateInfo(
            setLayoutCount=1,
            pSetLayouts=[descriptor_layout],
            pushConstantRangeCount=1,
            pPushConstantRanges=[push_range],
        )
        pipeline_layout = vk.vkCreatePipelineLayout(
            self._device, layout_info, None
        )
        self._layouts.append(pipeline_layout)

        # Specialization constants
        spec_data = self._pack_spec_constants(spec)
        entries = [
            vk.VkSpecializationMapEntry(
                constantID=0, offset=0, size=4
            ),
            vk.VkSpecializationMapEntry(
                constantID=1, offset=4, size=4
            ),
            vk.VkSpecializationMapEntry(
                constantID=2, offset=8, size=4
            ),
            vk.VkSpecializationMapEntry(
                constantID=3, offset=12, size=4
            ),
        ]
        # vulkan cffi binding needs pData as a cffi pointer
        from vulkan._vulkan import ffi as _ffi  # noqa: PLC0415
        spec_buf = _ffi.new("char[]", spec_data)
        spec_info = vk.VkSpecializationInfo(
            mapEntryCount=4,
            pMapEntries=entries,
            dataSize=len(spec_data),
            pData=spec_buf,
        )

        # Compute pipeline
        stage_info = vk.VkPipelineShaderStageCreateInfo(
            stage=vk.VK_SHADER_STAGE_COMPUTE_BIT,
            module=shader_module,
            pName="main",
            pSpecializationInfo=spec_info,
        )
        pipeline_info = vk.VkComputePipelineCreateInfo(
            stage=stage_info,
            layout=pipeline_layout,
        )
        pipeline = vk.vkCreateComputePipelines(
            self._device, vk.VK_NULL_HANDLE, 1, [pipeline_info], None
        )[0]

        # Destroy shader module (no longer needed after pipeline creation)
        vk.vkDestroyShaderModule(self._device, shader_module, None)

        result = ComputePipeline(pipeline=pipeline, layout=pipeline_layout)
        self._pipelines[cache_key] = result
        return result

    def get_pipeline(
        self, shader_name: str, problem_type: int = 0
    ) -> ComputePipeline:
        """Retrieve a cached pipeline."""
        return self._pipelines[(shader_name, problem_type)]

    def destroy(self) -> None:
        """Destroy all pipelines and layouts."""
        for cp in self._pipelines.values():
            vk.vkDestroyPipeline(self._device, cp.pipeline, None)
        self._pipelines.clear()

        for layout in self._layouts:
            vk.vkDestroyPipelineLayout(self._device, layout, None)
        self._layouts.clear()

    # ── Internal helpers ──

    @staticmethod
    def _load_spirv(shader_name: str) -> bytes:
        """Load a .spv file from the kernel_sources package.

        Discovery order:
        1. importlib.resources (installed package)
        2. Meson builddir adjacent to the source tree (development)
        """
        spv_name = f"{shader_name}.spv"

        # Strategy 1: importlib.resources (installed package)
        try:
            pkg = importlib.resources.files(
                "averaging_ensembled_classifier.backends.vulkan.kernel_sources"
            )
            spv_file = pkg / spv_name
            return spv_file.read_bytes()
        except Exception:
            pass

        # Strategy 2: Meson builddir discovery (development)
        this_dir = pathlib.Path(__file__).resolve().parent
        arch_root = this_dir.parent.parent.parent  # src/backends/vulkan -> arch root
        candidates = [
            arch_root / "builddir" / "src" / "backends" / "vulkan" / "kernel_sources" / spv_name,
        ]
        build_dir = arch_root / "build"
        if build_dir.is_dir():
            for child in build_dir.iterdir():
                if child.is_dir() and child.name.startswith("cp"):
                    candidates.append(
                        child / "src" / "backends" / "vulkan" / "kernel_sources" / spv_name
                    )

        for candidate in candidates:
            if candidate.exists():
                return candidate.read_bytes()

        searched = [str(c) for c in candidates]
        raise FileNotFoundError(
            f"Could not find {spv_name}. Searched:\n"
            + "\n".join(f"  - {p}" for p in searched)
        )

    @staticmethod
    def _pack_spec_constants(spec: SpecConstants) -> bytes:
        """Pack specialization constants into bytes (4 × uint32)."""
        return struct.pack("IIII",
            spec.simd_width,
            spec.bank_padding,
            spec.tile_size,
            spec.problem_type,
        )
