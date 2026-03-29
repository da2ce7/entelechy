"""CPU backend for the averaging ensembled classifier.

Provides CPUPlanRenderer — a PlanRenderer implementation that dispatches
execution plans via a native C kernel library using ctypes FFI and a
persistent thread pool.
"""
from .discovery import detect_thread_count, discover_hardware
from .renderer import CPUPlanRenderer

__all__ = ["CPUPlanRenderer", "discover_hardware", "detect_thread_count"]
