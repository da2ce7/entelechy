# utility_kernels.py

"""
A Toolbox of Host-Orchestrated, Driver-Level Utility Patterns.

This module provides implementations for non-domain-specific utility operations
that are composed of standard, built-in driver commands (e.g., a sequence of
`cl.enqueue_copy_buffer` calls).

This module is distinct from `kernel_signatures/utility_signatures.py`. That
file defines the contracts for custom OpenCL kernels, whereas this module
provides higher-level Python functions that *orchestrate* lower-level
primitives without necessarily launching custom C code.

By isolating these host-side patterns, we keep the main orchestrator logic
clean and focused on the primary training algorithm.
"""
