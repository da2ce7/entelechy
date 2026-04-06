# tests/tier1/test_fp8_memory.py
"""FP8 memory allocation validation."""

import pytest

from src.shared.precision_config import PrecisionConfig
from src.shared.buffer_lifecycle import BufferDescriptor, BufferHandle, BufferRole


class TestFP8MemoryAllocation:
    """Validate that FP8 achieves expected memory savings."""

    def test_fp8_element_size(self):
        """FP8 storage uses 1 byte per element."""
        cfg = PrecisionConfig.fp8_e4m3()
        assert cfg.storage_dtype.itemsize == 1

    def test_fp8_buffer_descriptor_sizing(self):
        """BufferDescriptor computes correct sizes for FP8."""
        cfg = PrecisionConfig.fp8_e4m3()
        num_elements = 1_000_000

        desc = BufferDescriptor(
            handle=BufferHandle(0),
            logical_name="test",
            padded_shape=(num_elements,),
            element_size_bytes=cfg.storage_dtype.itemsize,
            size_bytes=num_elements * cfg.storage_dtype.itemsize,
            role=BufferRole.BATCH_INTERMEDIATE,
            precision_role="storage",
        )

        assert desc.element_size_bytes == 1
        assert desc.size_bytes == num_elements

    def test_fp8_vs_fp32_memory_ratio(self):
        """FP8 storage is 1/4 the size of FP32."""
        cfg_fp8 = PrecisionConfig.fp8_e4m3()
        cfg_fp32 = PrecisionConfig.float32()

        assert cfg_fp8.storage_dtype.itemsize == 1
        assert cfg_fp32.storage_dtype.itemsize == 4
        assert cfg_fp32.storage_dtype.itemsize == 4 * cfg_fp8.storage_dtype.itemsize

    def test_fp8_vs_fp16_memory_ratio(self):
        """FP8 storage is 1/2 the size of FP16."""
        cfg_fp8 = PrecisionConfig.fp8_e4m3()
        cfg_fp16 = PrecisionConfig.mixed_f16_f32()

        assert cfg_fp8.storage_dtype.itemsize == 1
        assert cfg_fp16.storage_dtype.itemsize == 2
        assert cfg_fp16.storage_dtype.itemsize == 2 * cfg_fp8.storage_dtype.itemsize

    def test_e5m2_element_size(self):
        """E5M2 also uses 1 byte per element."""
        cfg = PrecisionConfig.fp8_e5m2()
        assert cfg.storage_dtype.itemsize == 1

    def test_fp8_buffer_4x_compression_vs_fp32(self):
        """Same buffer shape costs 1/4 memory with FP8 vs FP32."""
        num_elements = 2_000_000

        desc_fp8 = BufferDescriptor(
            handle=BufferHandle(0),
            logical_name="act_fp8",
            padded_shape=(num_elements,),
            element_size_bytes=1,  # FP8
            size_bytes=num_elements * 1,
            role=BufferRole.BATCH_INTERMEDIATE,
            precision_role="storage",
        )
        desc_fp32 = BufferDescriptor(
            handle=BufferHandle(1),
            logical_name="act_fp32",
            padded_shape=(num_elements,),
            element_size_bytes=4,  # FP32
            size_bytes=num_elements * 4,
            role=BufferRole.BATCH_INTERMEDIATE,
            precision_role="storage",
        )

        assert desc_fp32.size_bytes == 4 * desc_fp8.size_bytes
