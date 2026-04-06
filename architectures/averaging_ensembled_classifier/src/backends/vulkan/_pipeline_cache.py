"""SPIR-V shader loading and Vulkan compute pipeline creation (ADR-014, ADR-023 §4.1).

Loads compiled .spv modules via importlib.resources, creates VkShaderModule
objects, and builds compute pipelines with specialization constants.
Caches pipelines keyed by (shader_name, problem_type, variant_suffix).
"""
from __future__ import annotations

import ctypes
import importlib.resources
import pathlib
import struct
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

import numpy as np

from ...shared.precision_config import FP8_DTYPES, FP8_E4M3, FP8_E5M2, PrecisionConfig
from ...shared.precision_suffix import (
    compute_only_suffix as _compute_only_suffix_bare,
    precision_to_suffix,
)
from .context import VulkanContext

# Shaders with only compute-role buffers — use compute-only suffix.
# ADR-026: aggregate_partials_from_compute reads/writes COMPUTE_TYPE only.
_COMPUTE_ONLY_SHADERS = frozenset({
    "normalize_gradients",
    "aggregate_partials_from_compute",
})


def _spv_variant_suffix(precision: PrecisionConfig) -> str:
    """Map a PrecisionConfig to the SPIR-V variant suffix (ADR-024 §5.5, ADR-025 §7).

    Returns the canonical suffix with a leading underscore, e.g. ``"_s32c32x32"``.
    """
    return "_" + precision_to_suffix(
        precision.storage_dtype, precision.compute_dtype, precision.state_dtype,
    )


def _compute_only_suffix(precision: PrecisionConfig) -> str:
    """Map a PrecisionConfig to the compute-only SPIR-V variant suffix."""
    return "_" + _compute_only_suffix_bare(precision.compute_dtype)


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
        self._pipelines: dict[tuple[str, int, str], ComputePipeline] = {}
        self._layouts: list[Any] = []  # Track for cleanup

    def create_pipeline(
        self,
        shader_name: str,
        descriptor_layout: Any,  # VkDescriptorSetLayout
        push_constant_size: int,
        spec: SpecConstants,
        precision: PrecisionConfig | None = None,
    ) -> ComputePipeline:
        """Create a compute pipeline from a compiled SPIR-V module.

        Returns a cached pipeline if one already exists for the given
        (shader_name, problem_type, variant_suffix) key.
        """
        if shader_name in _COMPUTE_ONLY_SHADERS:
            variant_suffix = _compute_only_suffix(precision) if precision is not None else "_c32"
        elif precision is None:
            variant_suffix = "_s32c32x32"
        else:
            variant_suffix = _spv_variant_suffix(precision)

        cache_key = (shader_name, spec.problem_type, variant_suffix)
        if cache_key in self._pipelines:
            return self._pipelines[cache_key]

        # Load SPIR-V bytecode
        spv_bytes = self._load_spirv(shader_name, variant_suffix)

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
        self, shader_name: str, problem_type: int = 0, variant_suffix: str = "_fp32",
    ) -> ComputePipeline:
        """Retrieve a cached pipeline."""
        return self._pipelines[(shader_name, problem_type, variant_suffix)]

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
    def _load_spirv(shader_name: str, variant_suffix: str = "_s32c32x32") -> bytes:
        """Load a .spv file from the kernel_sources package.

        Discovery order:
        1. importlib.resources (installed package)
        2. Meson builddir adjacent to the source tree (development)
        """
        spv_name = f"{shader_name}{variant_suffix}.spv"

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
            arch_root / "builddir-vulkan" / "src" / "backends" / "vulkan" / "kernel_sources" / spv_name,
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
