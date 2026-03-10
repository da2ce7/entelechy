# Kernel Signature Contract Conformance Tests — Design Document

## 1. Problem Statement

The `kernel_signatures/` package contains 22 Python `KernelSignature` subclasses
that must be **exact mirrors** of the 19 `__kernel` function declarations in
`kernels.cl.h`. Today, no automated verification exists to guarantee that:

1. Every C kernel has a corresponding Python signature (coverage completeness).
2. Each signature's `kernel_name` property returns the exact C function name.
3. Each signature's `get_args()` returns arguments in the **exact positional order**
   mandated by the C parameter list.
4. The argument **count** from `get_args()` matches the C parameter count.
5. The argument **type category** (local memory · global buffer · uint scalar ·
   float scalar) at each position matches the C declaration.

A single positional mismatch silently corrupts data at runtime on the GPU.
These tests make such errors impossible to ship.

## 2. Scope

### In Scope

| Verification Axis | What is checked |
|:---|:---|
| **Coverage** | Every `__kernel void` in `kernels.cl.h` has ≥1 Python signature with matching `kernel_name`. |
| **Argument Count** | `len(get_args())` == number of C parameters for that kernel. |
| **Argument Order & Type Category** | Each position in the arg list has the correct type category: `LocalMemory`, `cl.Buffer`-like, `np.uint32`, or `SCALAR_TYPE`-like float. |
| **Derived Field Consistency** | Derived fields (`field(init=False)`) are correctly populated from buffer specs. |
| **Grid Well-formedness** | `get_grid()` returns `(tuple, tuple|None)` with positive integer dimensions. |

### Out of Scope

- Device-side kernel **correctness** (tested by integration tests).
- Host orchestration / graph recipe logic.
- Buffer allocation sizing (tested by `test_buffer_plumbing.py`).

## 3. Architectural Constraints

### 3.1 No OpenCL Device Required

All tests in this suite run without an OpenCL runtime. The `BufferManager` and
`DiscoveredArchConstants` dependencies are replaced with lightweight mock objects
that provide the minimum contract needed by signature `__post_init__` and
`get_args()` methods:

- **`MockBufferManager`**: Tracks `BufferHandle → (shape, dtype)` specs and
  returns sentinel objects from `get_cl_buffer()`.
- **`MockArchConsts`**: Provides canonical constants (`simd_width`, `SCALAR_NP_TYPE`,
  `optimal_workgroup_size_1d_reduction`, etc.).

### 3.2 Source of Truth

`kernels.cl.h` is the **sole** source of truth. The test suite includes a
**header parser** that extracts structured kernel definitions directly from the
header file. This prevents the tests themselves from becoming stale copies of
the contract.

### 3.3 Conformance to Architectural Elegance Feedback

Per `AGENTS.md` / `CONCEPT.md`: if a test reveals a structural mismatch that
cannot be resolved by fixing a bug in signature code, implementation must be
suspended and the pattern formalized as an architectural primitive. These tests
are the mechanism by which such signals are produced.

## 4. C Header Parsing Strategy

The parser extracts kernel metadata from `kernels.cl.h` using regex:

```
Step 1: Find all `__kernel void <name>(` blocks.
Step 2: Collect everything between `(` and the closing `);`.
Step 3: Strip doxygen comments (`/** ... */`) from the parameter block.
Step 4: Split on top-level commas to get individual parameter declarations.
Step 5: Classify each parameter into a type category:
        - __local ... *    → LOCAL_MEM
        - __global ... *   → GLOBAL_BUFFER
        - uint             → UINT_SCALAR
        - SCALAR_TYPE      → FLOAT_SCALAR
        - int              → INT_SCALAR (treated as UINT_SCALAR in OpenCL uint)
```

The parser produces a dictionary: `{kernel_name: [ParamCategory, ...]}`.

## 5. Type Category Mapping (C → Python)

| C Declaration Pattern | Category Tag | Expected Python `get_args()` Element |
|:---|:---|:---|
| `__local SCALAR_TYPE *` | `LOCAL_MEM` | `pyopencl.LocalMemory` instance |
| `__global [const] SCALAR_TYPE *` | `GLOBAL_BUFFER` | Object from `BufferManager.get_cl_buffer()` |
| `__global [const] void *` | `GLOBAL_BUFFER` | Object from `BufferManager.get_cl_buffer()` |
| `__global [const] int *` | `GLOBAL_BUFFER` | Object from `BufferManager.get_cl_buffer()` |
| `__global [const] uint *` | `GLOBAL_BUFFER` | Object from `BufferManager.get_cl_buffer()` |
| `uint` | `UINT_SCALAR` | `numpy.uint32` |
| `SCALAR_TYPE` (non-pointer) | `FLOAT_SCALAR` | `numpy.float32` (or half) |
| `int` (non-pointer, non-buffer) | `UINT_SCALAR` | `numpy.uint32` |

## 6. Kernel ↔ Signature Class Mapping

Some C kernels are represented by multiple Python classes (CCE/BCE variants,
Global/PerItem variants). The mapping is:

| C Kernel Name | Python Signature Class(es) | Notes |
|:---|:---|:---|
| `forward_pass` | `ForwardPassSignature` | |
| `render_logits_chunk` | `RenderLogitsChunkSignature` | |
| `compute_probs_loss_cce_chunk` | `ComputeProbsLossCceChunkSignature` | |
| `compute_probs_loss_bce_chunk` | `ComputeProbsLossBceChunkSignature` | |
| `calculate_module_param_grads_chunk` | `CalculateModuleParamGradsCceSignature`, `CalculateModuleParamGradsBceSignature` | Flag-differentiated |
| `backprop_error_to_hidden_chunk` | `BackpropErrorToHiddenChunkCceSignature`, `BackpropErrorToHiddenChunkBceSignature` | Flag-differentiated |
| `calculate_chunk_temp_gradients` | `CalculateChunkTempGradientsCceSignature`, `CalculateChunkTempGradientsBceSignature` | Flag-differentiated |
| `clip_partial_gradients` | `ClipPartialGradientsGlobalNormSignature`, `ClipPartialGradientsPerItemNormSignature` | Flag-differentiated |
| `gather_and_permute_grad_hidden_activations` | `GatherAndPermuteGradHiddenActivationsSignature` | |
| `aggregate_register_reduce` | `AggregateRegisterReduceSignature` | |
| `aggregate_local_reduce` | `AggregateLocalReduceSignature` | |
| `clip_intermediate_grad` | `ClipIntermediateGradSignature` | |
| `stabilize_and_reduce_grad_hidden_activations` | `StabilizeAndReduceGradHiddenActivationsSignature` | |
| `backprop_shared_weights_chunk` | `BackpropSharedWeightsChunkSignature` | |
| `backprop_shared_biases_chunk` | `BackpropSharedBiasesChunkSignature` | |
| `clip_shared_gradients_chunk` | `ClipSharedGradientsChunkSignature` | |
| `normalize_gradients` | `NormalizeGradientsSignature` | |
| `adam_update` | `AdamUpdateSignature` | |
| `clamp_temperatures` | `ClampTemperaturesSignature` | |

## 7. Test File Layout

```
kernel_signatures/tests/
├── DESIGN.md                           ← This document
├── __init__.py
├── conftest.py                         ← Mock infrastructure (MockBufferManager,
│                                          MockArchConsts, header parser, fixtures)
├── test_header_contract_conformance.py ← Cross-cutting: coverage, arg counts,
│                                          type category ordering per position
├── test_phase_1_act.py                 ← Per-signature unit tests for Phase 1
├── test_phase_2_learn_A_production.py  ← Per-signature unit tests for Phase 2A
├── test_phase_2_learn_B_processing.py  ← Per-signature unit tests for Phase 2B
├── test_phase_2_learn_C_reduction.py   ← Per-signature unit tests for Phase 2C
├── test_phase_2_learn_D_backprop.py    ← Per-signature unit tests for Phase 2D
└── test_phase_3_update.py              ← Per-signature unit tests for Phase 3
```

## 8. Test Categories

### 8.1 Cross-Cutting Contract Conformance (`test_header_contract_conformance.py`)

These tests parse `kernels.cl.h` once per session (cached fixture) and verify
structural invariants across **all** signature classes:

| Test | Assertion |
|:---|:---|
| `test_every_kernel_has_signature` | Every `__kernel void` name in the header has ≥1 signature class whose `kernel_name` matches. |
| `test_no_orphaned_signatures` | Every signature class's `kernel_name` appears in the header (no stale signatures). |
| `test_arg_count_matches_header[SigClass]` | `len(sig.get_args())` == C header parameter count for that kernel. |
| `test_arg_type_categories_match_header[SigClass]` | For each position `i`, the Python arg's type category matches the C param's category. |

### 8.2 Per-Phase Unit Tests (`test_phase_*.py`)

Each file tests the signature classes defined in its corresponding
`kernel_signatures/phase_*.py` module:

| Test | Assertion |
|:---|:---|
| `test_kernel_name` | `signature.kernel_name == "expected_c_name"` |
| `test_get_args_count` | `len(signature.get_args()) == EXPECTED_COUNT` |
| `test_get_args_order` | Each arg at position `i` has the correct type category. |
| `test_derived_fields` | Fields with `init=False` are correctly populated from buffer specs. |
| `test_get_grid_wellformed` | `get_grid()` returns `(tuple[int,...], tuple[int,...]|None)` with all dims > 0. |
| `test_flag_differentiated_variants` | CCE/BCE or Global/PerItem variants inject the correct flag value at the correct position. |

## 9. Mock Infrastructure Design

### 9.1 `MockBufferManager`

```python
class MockBufferManager:
    """A device-free stand-in for BufferManager."""

    def register(self, handle, shape, dtype=np.float32):
        """Register a handle with its spec and a sentinel cl.Buffer."""
        ...

    def get_spec(self, ref):
        """Returns (shape, dtype) — the only method signatures call in __post_init__."""
        ...

    def get_cl_buffer(self, ref):
        """Returns a sentinel object (not a real cl.Buffer) for arg assembly."""
        ...
```

### 9.2 `MockArchConsts`

```python
class MockArchConsts:
    """Provides minimum constants required by signature classes."""
    simd_width = 4
    SCALAR_NP_TYPE = np.float32
    optimal_workgroup_size_1d_reduction = 64
    optimal_rectangular_tile_dim1 = 8
```

### 9.3 Header Parser Fixture

```python
@pytest.fixture(scope="session")
def parsed_header():
    """Parses kernels.cl.h once and returns {name: [ParamCategory, ...]}."""
    ...
```

## 10. Rationale for "Parse, Don't Duplicate"

A common anti-pattern in contract tests is to hard-code expected values. This
creates a **second source of truth** that can silently diverge from the header.

By parsing `kernels.cl.h` at test time, we guarantee that:
- Adding a parameter to a C kernel immediately fails the Python test.
- Removing a parameter from a C kernel immediately fails the Python test.
- Reordering parameters in the C header immediately fails the Python test.

The parser is intentionally simple (regex-based, ~60 lines) because the header
follows a rigid, machine-readable format mandated by the System Contract.

## 11. Maintenance Contract

When modifying `kernels.cl.h`:
1. Run `pytest kernel_signatures/tests/` — failures indicate which Python
   signatures need to be updated.
2. Update the Python signature class(es).
3. Run again — all tests pass.

When adding a new kernel:
1. Add the `__kernel void` declaration to `kernels.cl.h`.
2. `test_every_kernel_has_signature` fails.
3. Create the Python signature class in the appropriate `phase_*.py` file.
4. Add per-phase unit tests in the corresponding `test_phase_*.py`.
5. All tests pass.
