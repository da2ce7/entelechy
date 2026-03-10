# ADR-007: KernelSignature Contract/Binding Split

**Status:** ACCEPTED  
**Date:** 2026-03-10  
**Deciders:** —  
**Supersedes:** —  
**Blocked by:** ADR-001, ADR-002, ADR-003 (constrains), ADR-004 (constrains)  
**Blocks:** ADR-011, ADR-012, ADR-013

---

## Context

The `KernelSignature` class hierarchy is the current mechanism by which the host validates and dispatches every kernel in the system. Each concrete signature (e.g., `ForwardPassSignature`, `ClipPartialGradientsGlobalNormSignature`, `AggregateLocalReduceSignature`) is a frozen dataclass that:

1. Accepts `BufferManager` and `DiscoveredArchConstants` as injected dependencies.
2. Derives all physical dimensions from buffer specifications in `__post_init__`.
3. Computes the dispatch grid (`get_grid() → (global_size, local_size)`).
4. Marshals ordered argument lists for OpenCL dispatch (`get_args() → List`).
5. Is executed by `KernelExecutor.launch(queue, signature, wait_for)`, which calls `kernel(queue, global_size, local_size, *kernel_args, wait_for=wait_for)`.

This design fulfils CONTRACT.md Article 1.4 (Collaborative Interface Verifiability): every signature validates buffer shapes, padding contracts, and calculability proofs before dispatch. But it does so by fusing two orthogonal responsibilities into a single class hierarchy:

- **Validation** — verifying that buffer shapes, scalar constraints, padding contracts, and calculability proofs are satisfied. This is inherently backend-neutral. The same dimensional constraints apply regardless of whether the kernel is dispatched via `clEnqueueNDRange`, `vkCmdDispatch`, or `pool_dispatch_and_wait`.

- **Argument marshalling** — translating validated parameters into the backend's native dispatch format. This is inherently backend-specific. OpenCL needs positional `cl.Buffer` objects and `cl.LocalMemory` size specifications. Vulkan needs push constant structs and descriptor set bindings. CPU needs typed C function argument structs.

### The coupling inventory

Every concrete `KernelSignature` imports:

| Import | Backend-specific? | Used for |
| :--- | :--- | :--- |
| `pyopencl as cl` | Yes | `cl.LocalMemory()` in `get_args()` |
| `BufferManager` | Yes | `get_cl_buffer()` in `get_args()`, `get_spec()` in `__post_init__` |
| `DiscoveredArchConstants` | Yes (inherits from `PrecisionContext`) | `simd_width` for `get_grid()`, `SCALAR_NP_TYPE` for local memory sizing |
| `numpy as np` | Shared (but `np.uint32()` wrapping is OpenCL-specific) | Scalar argument type coercion |

The `__post_init__` validation logic (shape derivation, dimension checks) depends only on abstract buffer shapes — never on `cl.Buffer` handles. The `get_args()` method depends entirely on concrete `cl.Buffer` handles and OpenCL-specific memory objects. These two parts are interleaved in every signature class, making extraction non-trivial but structurally clean.

### The three backends' dispatch interfaces

| Dispatch aspect | OpenCL | Vulkan | CPU |
| :--- | :--- | :--- | :--- |
| Buffer arguments | Positional `cl.Buffer` via `clSetKernelArg` | `VkDescriptorSet` bindings (set/binding indices) | Direct `void*` pointers |
| Scalar arguments | Positional `np.uint32`/`np.float32` via `clSetKernelArg` | Push constant struct fields (byte offsets) | Typed C function parameters |
| Local memory | `cl.LocalMemory(size_bytes)` as positional arg | GLSL `shared` qualifier (compile-time sized) | Stack allocation or `alloca` |
| Grid specification | `(global_size, local_size)` tuples | `vkCmdDispatch(groupCountX, groupCountY, groupCountZ)` | `pool_dispatch_and_wait(fn, args, task_count)` |
| Tile index delivery | Host scalar `src_scalar_NATURAL_flat_tile_index` per dispatch | `gl_WorkGroupID.x` (implicit, hardware-provided) | `task_index` function parameter |

The most structurally significant divergence is **tile index delivery**. In the current OpenCL implementation, the host loops over tiles and passes `flat_tile_index` as an explicit scalar argument to each dispatch. In Vulkan, a single `vkCmdDispatch(N, 1, 1)` dispatches all N tiles simultaneously — `gl_WorkGroupID.x` serves as the tile index, and no host scalar is needed. In the CPU backend, the thread pool's `task_index` parameter serves the same role. This means the `flat_tile_index` parameter exists in the OpenCL kernel interface but has no equivalent in the Vulkan or CPU kernel interfaces.

### What ADR-002 already requires

ADR-002 establishes that each `KernelDispatchNode` carries a `contract: KernelContract` reference. This field is defined but not yet specified — ADR-002 states: "The `KernelContract` (defined in ADR-007) encodes the kernel's validated shapes, padding requirements, calculability proofs, and validation preconditions from CONTRACT.md Article 1.4." This ADR delivers that specification.

ADR-002 also establishes the `placement_strategy` field on `KernelDispatchNode`: "The `placement_strategy` field (e.g., `"grid_mod_cls"`, `"linear_batch"`) specifies the abstract strategy by which each tile determines its position in the output buffer. Per ADR-007 (KernelSignature Contract/Binding Split), the mechanism — host-provided `flat_tile_index` scalar (OpenCL), `gl_WorkGroupID.x` (Vulkan), or `task_index` parameter (CPU) — is a binding-level concern, not a plan-level concern."

---

## Decision Drivers

1. **ADR-001 (Plan boundary — shared layer produces backend-neutral data).** The plan layer must validate kernel contracts without touching backend types. The `KernelContract` is consumed at plan-construction time; the `KernelBinding` is consumed at render time. The plan boundary separates these two phases exactly.

2. **CONTRACT.md Article 1.4 (Collaborative Interface Verifiability).** All validation occurs pre-dispatch. The "Host Proof Obligation" (Article 1.4a) requires that buffer dimensions, scalar constraints, and calculability proofs are verified before any dispatch mechanism is invoked. This validation is inherently shared — it must not be duplicated in each backend's binding code.

3. **CONTRACT.md Article 3.2 (Placement Contract).** Placement strategies (`grid_mod_cls`, `linear_batch`, `linear_generic`) are abstractly specified in CONTRACT.md with a `strategy_name(key_parameter)` grammar. The contract specifies the *strategy and key domain*; the binding specifies *how the key is communicated*. This existing abstraction in CONTRACT.md is the natural model for the contract/binding split.

4. **ADR-001 (Three-tier jurisdictional model).** The Policy tier (shared) computes constraints; the Orchestration tier (backend) handles dispatch mechanics. The `KernelContract` carries Policy-tier output; the `KernelBinding` implements Orchestration-tier dispatch. This maps precisely to the jurisdictional boundary.

5. **ADR-004 (Abstract parameter names in stride tables).** `StreamingLoopPlan`'s `ParameterStride.param_name` fields reference parameter names that appear in `KernelDispatchNode.scalar_params`. These names must be abstract — they cannot reference backend-specific parameters like `flat_tile_index` (which exists only in OpenCL). The contract must define a backend-neutral parameter vocabulary.

6. **CONCEPT.md §1 (Architectural Elegance Feedback).** The contract/binding split is a first-class architectural primitive. If a new backend reveals a parameter or dispatch pattern that creates tension with this split, the response is to extend the contract/binding vocabulary — not to leak backend specifics into the shared layer.

---

## Options Considered

### Option A: Extract shared validation into `KernelContract`; keep `KernelSignature` as OpenCL-specific binding

Split each existing `KernelSignature` subclass into two parts:

- A `KernelContract` frozen dataclass that carries the kernel's name, abstract parameter manifest (names, types, roles), buffer shape expectations, calculability proofs, validation preconditions, and placement strategy.
- The existing `KernelSignature` subclass is retained as the OpenCL binding — it consumes a validated `KernelContract` and adds `BufferManager`, `DiscoveredArchConstants`, `get_grid()`, and `get_args()`. Future backends implement their own binding classes.

The `KernelContract` uses **abstract placement keys**: the contract specifies the placement strategy (e.g., `grid_mod_cls`) and its key domain. The mechanism by which the tile index is communicated — host scalar `flat_tile_index` (OpenCL), `gl_WorkGroupID.x` (Vulkan), `task_index` (CPU) — is a binding concern. The tile index parameter does not appear in the contract's parameter manifest.

**Advantages:**
- Minimal disruption to the existing `KernelSignature` hierarchy. Contracts are extracted alongside; signatures become backend-specific bindings that consume the contract.
- Each backend implements `KernelBinding` in its own idiom: OpenCL keeps positional arg lists; Vulkan uses push constant structs + descriptor sets; CPU uses typed C argument structs.
- The contract is a frozen dataclass — pure data, no methods, no backend types. It satisfies ADR-001's plan-as-data-structure requirement.
- The abstract placement key resolves the `flat_tile_index` divergence cleanly: the contract says "this kernel uses `grid_mod_cls` placement with a key domain of `[0, total_tile_count)`"; each binding implements the key delivery mechanism natively.
- ADR-004's `ParameterStride.param_name` fields naturally reference abstract contract-level parameter names, which are the same across all backends.

**Disadvantages:**
- The contract's abstract parameter manifest omits backend-specific parameters (like `flat_tile_index`). A reader examining the contract alone cannot see the full parameter list for any specific backend — they must also consult the binding.
- Local memory parameters (OpenCL's `cl.LocalMemory`, Vulkan's compile-time `shared`, CPU's stack allocation) are binding concerns, but their *size requirements* are contract concerns (they depend on shapes and padding from CONTRACT.md Article 5). The contract must carry local memory size specifications without prescribing the allocation mechanism.
- Extracting contracts from 20+ existing signature classes is a non-trivial mechanical effort.

### Option B: Superset parameter list with backend annotations

The `KernelContract` carries the complete superset of all parameters across all backends, with each parameter annotated by backend applicability: `[OpenCL-only]`, `[Vulkan-only]`, `[CPU-only]`, `[all]`.

```python
@dataclass(frozen=True)
class ParameterSpec:
    name: str
    number_type: str                    # "NATURAL", "INTEGER", "REAL", "FLAG"
    role: str                           # "scalar", "buffer", "local_memory"
    backend_scope: FrozenSet[str]       # {"opencl"}, {"vulkan"}, {"opencl", "vulkan", "cpu"}, etc.
```

**Advantages:**
- Complete parameter visibility. The contract contains every parameter for every backend.
- No information is hidden — a reader sees the full interface in one place.

**Disadvantages:**
- **Pollutes the shared contract with backend concerns.** The `flat_tile_index` parameter is an OpenCL dispatch artefact. Including it in the shared contract — even with an `[OpenCL-only]` annotation — forces every shared-layer consumer to either ignore or handle the annotation. This contradicts ADR-001's principle that the shared layer contains no backend-specific types or logic.
- **Must be extended for every new backend.** Adding a SYCL or Metal backend would require updating the contract's parameter manifests to add `[SYCL-only]` or `[Metal-only]` annotations — a shared-layer modification for what is inherently a backend concern. This violates the encapsulation that ADR-001 establishes: new backends should require only a new `PlanRenderer`, not shared-layer changes.
- **Calculability proofs become ambiguous.** CONTRACT.md Article 1.4.1 requires that all terms in a calculability proof exist in the interface. If `flat_tile_index` is in the contract but annotated `[OpenCL-only]`, a calculability proof referencing it is valid for OpenCL but meaningless for Vulkan. The "closed logical system" is no longer closed — it is conditional on backend selection.
- **Confounds validation and dispatch.** The contract should validate what is universally true about the kernel (shapes, constraints, algorithmic preconditions). Backend-specific parameters are dispatch artefacts — they do not express properties of the computation.

### Option C: Abstract binding interface with embedded contracts

Define an abstract `KernelBinding` base class with a `validate()` method and abstract `dispatch()` method. Each backend implements the full class, embedding both validation and binding. The `KernelContract` is not a separate entity — it is the validation portion of the binding.

```python
class KernelBinding(abc.ABC):
    @abc.abstractmethod
    def validate(self, node: KernelDispatchNode) -> None: ...

    @abc.abstractmethod
    def dispatch(self, context: BackendContext) -> None: ...
```

**Advantages:**
- Simple class hierarchy. Each backend provides a single class per kernel.
- Validation and dispatch are co-located — no need to cross-reference separate contract and binding definitions.

**Disadvantages:**
- **Validation logic is duplicated per backend.** Buffer shape checks, calculability proofs, and scalar constraint validation are identical across backends. Requiring each backend's `validate()` to reimplement them either creates code duplication or requires factoring out a shared validation module — which converges on the `KernelContract` from Option A.
- **Contract is not a data structure.** An abstract base class with methods is not inspectable data. The plan cannot carry an abstract `KernelBinding` as its contract reference without depending on executable backend code. This violates ADR-001's plan-as-data-structure principle, repeating the error identified in ADR-003 Option B (embedding a live `StabilizationPolicy` object in the plan).
- **Plan-time validation requires backend instantiation.** To validate a `KernelDispatchNode` at plan-construction time, the shared layer would need to instantiate a backend-specific `KernelBinding` — coupling the plan builder to a specific backend, which is the exact coupling ADR-001 eliminates.

---

## Analysis

### Eliminating Option B

Option B is eliminated for the same reason ADR-006 eliminated its Option A (raw hardware measurements with backend-category discriminator): it introduces backend-specific information into the shared layer. The `backend_scope` annotation on each parameter is a discriminator that shared-layer consumers must interpret — either by filtering ("ignore OpenCL-only params") or by branching ("if OpenCL, include `flat_tile_index`"). Both are forms of backend-discriminated logic in the shared layer.

The `flat_tile_index` divergence illustrates why the superset model fails. This parameter is:

- **OpenCL**: An explicit host scalar passed to each of N `clEnqueueNDRange` calls. The host loops over tiles and sets `flat_tile_index = i` for each dispatch.
- **Vulkan**: Non-existent. A single `vkCmdDispatch(N, 1, 1)` call dispatches all N tiles. Each shader invocation reads `gl_WorkGroupID.x` — a hardware-provided implicit value that never appears in the host API.
- **CPU**: The `task_index` parameter of the thread pool's task function, passed by the `pool_dispatch_and_wait` infrastructure. It is not a kernel parameter in the CONTRACT.md sense — it is a threading primitive.

These are three structurally distinct mechanisms serving the same *architectural* purpose: identifying which tile a given invocation processes. A superset parameter list conflates the architectural purpose with the dispatch mechanism, forcing the shared contract to enumerate dispatch artefacts from every backend.

CONTRACT.md Article 3.2 already provides the correct abstraction. A Placement Contract is specified as `strategy_name(key_parameter)` — e.g., `grid_mod_cls(src_scalar_NATURAL_flat_tile_index)`. The strategy name is universal; the key parameter name is the OpenCL-specific mechanism. The contract should specify the *strategy* (what the kernel's placement semantics are), not the *key delivery mechanism* (how the tile index reaches the kernel). Each binding implements the key delivery in its native idiom.

### Eliminating Option C

Option C is eliminated for the same reason ADR-003 eliminated its Option B (embedding a live `StabilizationPolicy` in the plan): it embeds executable methods in what ADR-001 requires to be a data structure. ADR-002 specifies that each `KernelDispatchNode` carries `contract: KernelContract` — this is a data reference inspectable at plan-construction time. An abstract `KernelBinding` with `validate()` and `dispatch()` methods is an API surface, not inspectable data.

Furthermore, Option C's co-location of validation and dispatch inverts the jurisdictional model. Validation is a Policy-tier concern (ADR-001 §Three-tier jurisdictional model) — it must be performed once, in the shared layer, at plan-construction time. Dispatch is an Orchestration-tier concern — it is performed per-backend, at render time. Co-locating them in a single backend-specific class would either push validation into the backend (duplicating it three times) or require the shared layer to instantiate backend-specific classes (coupling the plan builder to a backend).

### Choosing Option A

Option A separates shared from backend concerns at the Policy/Orchestration boundary — the same boundary established in ADR-001, applied in ADR-003 (threshold schedules), ADR-004 (stride tables), ADR-005 (Node 16 policy parameters), and ADR-006 (hardware profile constants).

**Policy tier pre-computes (carried in the `KernelContract`):**
- Abstract parameter manifest — the backend-neutral names, types, and roles of all parameters the kernel consumes. This excludes dispatch artefacts like `flat_tile_index` and includes only parameters with architectural significance (buffer dimensions, padding parameters, scalar constraints, placement strategy identifiers).
- Buffer shape expectations — the logical tensor shapes (as symbolic expressions referencing scalar parameters) for each buffer parameter, per CONTRACT.md §3 `Tensor Shape`.
- Padding contracts — the padding strategy and formula for each buffer, per CONTRACT.md §3.1.
- Calculability proofs — the derivation expressions for buffer dimensions, per CONTRACT.md §3 `Calculability Proof`. All terms reference abstract parameter names.
- Validation preconditions — the host's mandatory pre-dispatch assertions, per CONTRACT.md §3 `Validation Preconditions`.
- Placement strategy — the abstract strategy name (e.g., `"grid_mod_cls"`) and key domain, per CONTRACT.md §3.2.
- Local memory requirements — the size specification (as a symbolic expression over scalar parameters) for any `__local`/`shared`/stack-allocated scratch buffers, without prescribing the allocation mechanism.
- Kernel contract block data — idempotency classification, synchronization model, and behavioral invariants from CONTRACT.md Article 4.

**Orchestration tier derives (at render time, in each `KernelBinding`):**
- Native buffer argument translation — `cl.Buffer` handles (OpenCL), descriptor set bindings (Vulkan), `void*` pointers (CPU).
- Native scalar argument translation — positional `np.uint32` (OpenCL), push constant struct fields (Vulkan), typed C parameters (CPU).
- Tile index delivery — host scalar `flat_tile_index` per dispatch (OpenCL), `gl_WorkGroupID.x` implicit (Vulkan), `task_index` from thread pool (CPU).
- Local memory allocation — `cl.LocalMemory(size_bytes)` (OpenCL), compile-time `shared` qualifier (Vulkan), stack/`alloca` (CPU).
- Grid specification — `(global_size, local_size)` (OpenCL), `vkCmdDispatch(gX, gY, gZ)` (Vulkan), `pool_dispatch_and_wait(fn, args, N)` (CPU).

### The abstract placement key model

CONTRACT.md Article 3.2 defines placement strategies with a `strategy_name(key_parameter)` grammar. In the current implementation, `key_parameter` is always an explicit scalar parameter name: `src_scalar_NATURAL_flat_tile_index` for `grid_mod_cls`, `src_scalar_NATURAL_batch_chunk_index` for `linear_batch`.

Under the contract/binding split, the `KernelContract` specifies placement at the strategy level:

```
placement: PlacementContract(
    strategy="grid_mod_cls",
    key_domain=(0, total_tile_count),
    context_params={"num_class_chunks": "src_scalar_NATURAL_num_class_chunks"}
)
```

The binding translates this:

- **OpenCL binding**: Adds `src_scalar_NATURAL_flat_tile_index` as an explicit scalar argument. The host loop sets it to `i` for each of N dispatches.
- **Vulkan binding**: Does not add a tile index argument. The renderer dispatches with `vkCmdDispatch(total_tile_count, 1, 1)`. The shader reads `gl_WorkGroupID.x`. Context parameters (`num_class_chunks`) are delivered via push constants.
- **CPU binding**: The `task_index` parameter is implicit in the `pool_dispatch_and_wait` infrastructure. Context parameters are fields of the task argument struct.

This model aligns with the precedent from ADR-003 and ADR-005: the plan carries *what* (placement strategy and key domain) as Policy-tier data; each backend implements *how* (tile index delivery mechanism) as an Orchestration-tier concern.

### The local memory question

Local memory occupies a unique position in the split. Its *algebraic* size is a contract concern: CONTRACT.md Article 5 defines `LOCAL_MEM_BANK_PADDING = 1`, and the local tile size for `forward_pass` is `SIMD_WIDTH × (SIMD_WIDTH + LOCAL_MEM_BANK_PADDING) × scalar_bytes` — this is a formula composed of shared constants and hardware profile values. The *allocation mechanism* is a binding concern: OpenCL materializes `cl.LocalMemory(size_bytes)` as a positional kernel argument; Vulkan declares `shared` arrays at compile time with specialization-constant-controlled sizes; CPU allocates on the stack.

The contract carries local memory as a specification: for each local buffer parameter, it records the size formula and the CONTRACT.md Article 3.1 padding contract reference. The binding translates this specification into native allocation. This parallels how the contract carries buffer shape specifications (symbolic expressions) without prescribing the allocation of global buffers (which is ADR-009's concern).

### Consistency with upstream ADRs

**ADR-002:** The `KernelDispatchNode` carries `contract: KernelContract` and `placement_strategy: Optional[str]`. The contract is a pure frozen dataclass, validated at plan-construction time. The `placement_strategy` field on the node is a convenience for the renderer — it duplicates context from the embedded `KernelContract` for efficient dispatch without deep contract inspection.

**ADR-003:** `ReductionTreePlan` bodies are rendered atomically by the backend. The renderer must bind and dispatch aggregation and clip kernels at each stage. Each of these kernel dispatches has its own binding — the `ReductionTreePlan` carries the Policy-tier parameters (thresholds, fan-in, offsets); the renderer's binding code translates them to native dispatch. The contract/binding split applies at the individual kernel level within the tree.

**ADR-004:** `StreamingLoopPlan`'s `ParameterStride.param_name` references abstract parameter names from the `KernelContract`'s parameter manifest — never backend-specific dispatch artefacts. When the renderer instantiates per-chunk parameters by applying strides, the resulting names map to the binding's native parameter slots. The stride table is a Policy-tier data structure; the parameter-to-native-slot mapping is an Orchestration-tier concern.

**ADR-005:** Node 16's `KernelContract` validates buffer shapes and scalar parameter ranges (`policy_max_k >= 1`, `T_algorithmic >= 0`, etc.) at plan-construction time. The binding for Node 16 is trivial — a single dispatch with the policy scalars. The contract is rich (reflecting the kernel's complex requirements); the binding is thin (reflecting the collapsed Orchestration tier).

**ADR-006:** `HardwareProfile` provides `simd_width` and `cache_line_bytes` that the `KernelContract`'s validation preconditions reference for padding alignment checks. The profile is the data source; the contract is the consumer. No backend types flow through this dependency chain.

---

## Decision

**Option A: Extract shared `KernelContract` with abstract placement keys; per-backend `KernelBinding`.**

Split each `KernelSignature` into a shared `KernelContract` (backend-neutral validation data, carried in the plan) and per-backend `KernelBinding` implementations (native dispatch argument marshalling, used at render time).

### `KernelContract` data structure

```python
@dataclass(frozen=True)
class BufferParamSpec:
    """Specification for a single buffer parameter in the kernel contract."""
    param_name: str
    """Full canonical name per CONTRACT.md Article 2.1 (e.g.,
    'src_buffer_GLOBAL_input_stream')."""

    flow: str
    """'src', 'dest', 'update', or 'sync'."""

    memory_scope: str
    """'GLOBAL', 'LOCAL', 'GLOBAL_CONST', or 'DEVICE_CONST'."""

    tensor_shape: Tuple[str, ...]
    """Symbolic shape expression. Each element is either an integer literal
    (str of int) or a scalar parameter name from this contract's manifest.
    Example: ('src_scalar_NATURAL_total_batch_count', 'padded_hidden_count')."""

    padding_contract: Dict[str, str]
    """Padding specification per CONTRACT.md Article 3.1.
    Keys: 'Type' (mandatory), 'Formula' (if Type != 'NONE').
    Example: {'Type': 'CACHE', 'Formula': 'Post-pad to 128-byte alignment'}."""

    calculability_proof: Tuple[str, ...]
    """References to scalar parameters or compile-time constants that
    determine this buffer's dimensions. Per CONTRACT.md Article 1.4.1,
    all terms must exist in the contract's parameter manifest or as
    recognized compile-time constants (e.g., 'Compile-time constant: LUT_CAPACITY')."""

    validation_preconditions: Tuple[str, ...]
    """Host-enforceable precondition expressions.
    Example: ('Host shall zero-initialize this buffer prior to dispatch.',)."""


@dataclass(frozen=True)
class ScalarParamSpec:
    """Specification for a single scalar parameter in the kernel contract."""
    param_name: str
    """Full canonical name per CONTRACT.md Article 2.2 (e.g.,
    'src_scalar_NATURAL_total_batch_count')."""

    flow: str
    """'src' or 'dest'."""

    number_type: str
    """'NATURAL', 'INTEGER', 'REAL', or 'FLAG' per CONTRACT.md Article 2.3."""


@dataclass(frozen=True)
class LocalMemorySpec:
    """Specification for a local memory buffer in the kernel contract."""
    param_name: str
    """Full canonical name per CONTRACT.md Article 2.1 (e.g.,
    'update_buffer_LOCAL_reduction_tile')."""

    size_formula: str
    """Symbolic size expression in bytes, composed of scalar parameter names,
    HardwareProfile constants, and compile-time constants.
    Example: 'SIMD_WIDTH * (SIMD_WIDTH + LOCAL_MEM_BANK_PADDING) * scalar_bytes'."""

    padding_contract: Dict[str, str]
    """Padding specification per CONTRACT.md Article 3.1."""


@dataclass(frozen=True)
class PlacementContract:
    """Abstract placement specification per CONTRACT.md Article 3.2."""
    strategy: Optional[str]
    """Canonical placement strategy name: 'grid_mod_cls', 'linear_batch',
    'linear_generic', or None for kernels without placement semantics."""

    key_domain: Optional[Tuple[str, str]]
    """(inclusive_lower_bound, exclusive_upper_bound) expressed as scalar
    parameter names or integer literals. Defines the range of the abstract
    placement key. Example: ('0', 'total_tile_count').
    None when strategy is None."""

    context_params: Dict[str, str]
    """Mapping from CONTRACT.md Article 3.2.3 'Required Context Parameters'
    to scalar parameter names in this contract's manifest.
    Example: {'num_class_chunks': 'src_scalar_NATURAL_num_class_chunks'}.
    Empty dict when no context parameters are required."""


@dataclass(frozen=True)
class KernelContractBlock:
    """Kernel-level contract metadata per CONTRACT.md Article 4."""
    idempotency: str
    """One of: 'Strictly Idempotent', 'Associatively Non-Idempotent',
    'Fundamentally Non-Idempotent (Stateful)'."""

    synchronization_model: Optional[str]
    """Canonical behavioral vocabulary term from CONTRACT.md Article 4.3,
    or None if not applicable.
    Example: 'Partial Renderer', 'Global Barrier', 'Streamable'."""

    holistic_constraints: str
    """Holistic constraint text, or the canonical no-constraint string:
    'All constraints are defined by the parameter commentary blocks.'"""

    behavioral_invariants: Optional[str]
    """Optional text describing strict implementation invariants.
    For Node 16, this references the two-phase execution model.
    None for kernels with no behavioral invariants beyond the parameter contracts."""


@dataclass(frozen=True)
class KernelContract:
    """
    The backend-neutral validation contract for a single kernel.

    This is a pure, frozen data structure carrying the kernel's complete
    pre-dispatch verification specification from CONTRACT.md. It is
    constructed once by the shared layer's plan builder and embedded in
    each KernelDispatchNode (ADR-002). The plan builder validates all
    contracts at plan-construction time; backend renderers trust the
    validated contracts and proceed directly to binding and dispatch.

    Carries no backend types, no executable methods, no backend-specific
    parameters (e.g., no flat_tile_index — see PlacementContract).
    """
    kernel_name: str
    """The kernel function's canonical name as defined in kernels.cl.h.
    Example: 'forward_pass', 'compute_probs_loss_cce_chunk'."""

    contract_block: KernelContractBlock
    """Kernel-level metadata per CONTRACT.md Article 4."""

    buffer_params: Tuple[BufferParamSpec, ...]
    """All buffer parameters in contractual argument order.
    Excludes local memory buffers (those are in local_memory_specs)."""

    scalar_params: Tuple[ScalarParamSpec, ...]
    """All shared scalar parameters in contractual argument order.
    These are the abstract, backend-neutral parameters. Backend-specific
    dispatch artefacts (flat_tile_index for OpenCL, gl_WorkGroupID for
    Vulkan) are NOT included — they are binding concerns."""

    local_memory_specs: Tuple[LocalMemorySpec, ...]
    """Local/shared memory requirements. The contract specifies sizes;
    each backend's binding allocates natively. May be empty."""

    placement: PlacementContract
    """Abstract placement specification. Strategy is None for kernels
    without partial-rendering semantics."""
```

### `KernelBinding` interface

The binding is a backend-specific translating layer. Each backend provides a binding implementation for every kernel. The binding's responsibilities are:

1. **Accept policy data.** Receive the validated `KernelContract`, the `KernelDispatchNode`'s buffer bindings (logical name → physical buffer handle), scalar parameters, and tile count.
2. **Add backend-specific parameters.** For OpenCL: inject `flat_tile_index` as a positional scalar. For Vulkan: no injection — the tile index is implicit. For CPU: pass `task_index` through the thread pool infrastructure.
3. **Translate to native format.** Produce the backend's native dispatch arguments: positional arg list (OpenCL), push constant struct + descriptor set (Vulkan), C function arg struct (CPU).
4. **Allocate local memory.** Evaluate the `LocalMemorySpec.size_formula` from the contract and allocate natively.
5. **Compute the dispatch grid.** From the contract's parameter manifest and the `HardwareProfile`, derive the backend's native grid specification.

The binding is **not** a data structure — it is Orchestration-tier code that lives in the backend package. It is not carried in the plan. The plan carries the `KernelContract` (data); the renderer instantiates the binding (code) at render time:

```
Plan construction:
    KernelContract ← shared layer validates shapes, proofs, preconditions
    KernelDispatchNode.contract = KernelContract  (pure data in the plan)

Render time:
    KernelBinding ← backend renderer creates from KernelContract + physical buffers
    KernelBinding.dispatch(backend_context)  (backend-native execution)
```

Each backend defines its binding classes in its own package (ADR-012: `backends/<name>/kernel_bindings/`). The binding class hierarchy is backend-internal — the shared layer never imports or references binding classes.

### The `flat_tile_index` resolution

The `flat_tile_index` parameter is a dispatch artefact specific to OpenCL's per-tile enqueue model. Under the contract/binding split, it is handled as follows:

- **In the `KernelContract`:** Not present. The contract's `PlacementContract` specifies the strategy and key domain abstractly. The calculability proofs that currently reference `flat_tile_index` are rewritten to reference the abstract placement key domain — e.g., "the write offset is determined by the placement strategy `grid_mod_cls` with key domain `[0, total_tile_count)`."

- **In the OpenCL `KernelBinding`:** Injected as a positional scalar argument. The OpenCL renderer loops over tiles, setting `flat_tile_index = i` for each dispatch. This is OpenCL's native mechanism for the abstract placement key.

- **In the Vulkan `KernelBinding`:** Not injected. The renderer dispatches with `vkCmdDispatch(total_tile_count, 1, 1)`. The GLSL shader reads `gl_WorkGroupID.x` as its native placement key. Context parameters (`num_class_chunks`) are delivered via push constants.

- **In the CPU `KernelBinding`:** Not injected as a kernel parameter. The `task_index` argument of the `pool_dispatch_and_wait` callback serves as the native placement key. Context parameters are fields of the task argument struct.

This resolution requires a CONTRACT.md amendment to Article 3.2.3: acknowledging that the `Mandatory Key Parameter` column specifies the key's semantic role and domain, while the delivery mechanism is backend-specific. The article's current phrasing (`key_parameter: The full, canonical name of the scalar parameter...`) must be updated to distinguish the abstract key concept from its OpenCL-specific realization. The table's `Mandatory Key Parameter` column becomes an abstract key identifier, and a new "Backend Binding" note explains that the mechanism varies.

### Validation flow

Plan-time validation (shared layer, once):

1. **Buffer shape consistency.** For each `BufferParamSpec`, resolve the `tensor_shape` symbolic expression using the `KernelDispatchNode`'s `scalar_params`. Verify that the resolved shape matches the logical buffer's declared shape in the plan's buffer namespace (ADR-009).

2. **Calculability proofs.** For each `BufferParamSpec.calculability_proof`, verify that every referenced scalar parameter or compile-time constant exists in the contract's `scalar_params` manifest or the plan's global constants.

3. **Validation preconditions.** Evaluate the host-enforceable preconditions (e.g., buffer initialization requirements, offset bounds) using the `scalar_params` values.

4. **Placement contract consistency.** If the contract's `PlacementContract.strategy` is set, verify that the `KernelDispatchNode`'s `tile_count` matches the key domain extent, and that all `context_params` are present in `scalar_params`.

5. **Local memory feasibility.** Evaluate each `LocalMemorySpec.size_formula` and verify that the total local memory requirement does not exceed `HardwareProfile.max_local_mem_bytes` (if the profile provides this value — it is `Optional` per ADR-006).

Render-time binding (per-backend, per-dispatch):

1. The renderer retrieves the pre-validated `KernelContract` from the `KernelDispatchNode`.
2. The renderer creates its backend-specific `KernelBinding` instance, translating abstract buffer names to physical buffer handles, abstract scalar names to native argument positions, and evaluating local memory size formulas.
3. For kernels with a placement strategy, the renderer implements the key delivery mechanism.
4. The renderer dispatches using the binding's native format.

### Concrete example: `forward_pass` (Node 4) split

**Before (current single `ForwardPassSignature`):**
A frozen dataclass coupling validation (`__post_init__` derives shapes) and dispatch (`get_args()` builds `cl.Buffer` arg list, `get_grid()` computes OpenCL execution dimensions).

**After — `KernelContract`:**
```python
forward_pass_contract = KernelContract(
    kernel_name="forward_pass",
    contract_block=KernelContractBlock(
        idempotency="Strictly Idempotent",
        synchronization_model="Streamable",
        holistic_constraints="All constraints are defined by the parameter commentary blocks.",
        behavioral_invariants=None,
    ),
    buffer_params=(
        BufferParamSpec(
            param_name="src_buffer_GLOBAL_input",
            flow="src", memory_scope="GLOBAL",
            tensor_shape=("src_scalar_NATURAL_total_batch_count",
                          "src_scalar_NATURAL_padded_input_count"),
            padding_contract={"Type": "CACHE",
                              "Formula": "Post-pad row stride to cache-line alignment"},
            calculability_proof=("src_scalar_NATURAL_total_batch_count",
                                 "src_scalar_NATURAL_padded_input_count"),
            validation_preconditions=(),
        ),
        # ... remaining buffer params ...
    ),
    scalar_params=(
        ScalarParamSpec("src_scalar_NATURAL_batch_chunk_offset", "src", "NATURAL"),
        ScalarParamSpec("src_scalar_NATURAL_batch_chunk_count", "src", "NATURAL"),
        ScalarParamSpec("src_scalar_NATURAL_total_batch_count", "src", "NATURAL"),
        ScalarParamSpec("src_scalar_NATURAL_padded_input_count", "src", "NATURAL"),
        ScalarParamSpec("src_scalar_NATURAL_padded_hidden_count", "src", "NATURAL"),
    ),
    local_memory_specs=(
        LocalMemorySpec(
            param_name="update_buffer_LOCAL_simd_tile",
            size_formula="SIMD_WIDTH * (SIMD_WIDTH + LOCAL_MEM_BANK_PADDING) * scalar_bytes",
            padding_contract={"Type": "BANK_CONFLICT_AVOIDANCE",
                              "Formula": "Pad row stride to (SIMD_WIDTH + LOCAL_MEM_BANK_PADDING) elements"},
        ),
    ),
    placement=PlacementContract(strategy=None, key_domain=None, context_params={}),
)
```

**After — OpenCL `KernelBinding` (in `backends/opencl/kernel_bindings/`):**
The OpenCL binding for `forward_pass` consumes the validated contract and produces the native arg list. It imports `pyopencl`, `BufferManager`, and `DiscoveredArchConstants` — OpenCL-specific types that the shared layer never sees. It computes `get_grid()` using `simd_width` from the backend's hardware context, and builds the positional arg list via `get_cl_buffer()` calls.

**After — CPU `KernelBinding` (in `backends/cpu/kernel_bindings/`):**
The CPU binding translates the same contract into a C function call with typed parameters: `forward_pass_cpu(input_ptr, mask_ptr, weights_ptr, biases_ptr, h_out_ptr, h_mask_out_ptr, batch_offset, batch_count, total_batch, padded_input, padded_hidden, simd_width)`. No `cl.Buffer`, no `cl.LocalMemory`, no OpenCL grid.

### Concrete example: `compute_probs_loss_cce_chunk` (Node 6) — partial renderer

```python
probs_loss_cce_contract = KernelContract(
    kernel_name="compute_probs_loss_cce_chunk",
    contract_block=KernelContractBlock(
        idempotency="Strictly Idempotent",
        synchronization_model="Partial Renderer",
        holistic_constraints="All constraints are defined by the parameter commentary blocks.",
        behavioral_invariants=None,
    ),
    buffer_params=( ... ),
    scalar_params=(
        # Note: flat_tile_index is NOT here — it is an OpenCL dispatch artefact.
        ScalarParamSpec("src_scalar_NATURAL_num_class_chunks", "src", "NATURAL"),
        ScalarParamSpec("src_scalar_NATURAL_classes_per_chunk", "src", "NATURAL"),
        ScalarParamSpec("src_scalar_NATURAL_modules_per_chunk", "src", "NATURAL"),
        ScalarParamSpec("src_scalar_NATURAL_total_batch_count", "src", "NATURAL"),
        ScalarParamSpec("src_scalar_NATURAL_total_output_class_count", "src", "NATURAL"),
        ScalarParamSpec("src_scalar_NATURAL_padded_total_output_class_count", "src", "NATURAL"),
        ScalarParamSpec("src_scalar_NATURAL_total_modules_count", "src", "NATURAL"),
        ScalarParamSpec("src_scalar_NATURAL_total_tile_count", "src", "NATURAL"),
    ),
    local_memory_specs=(),
    placement=PlacementContract(
        strategy="grid_mod_cls",
        key_domain=("0", "src_scalar_NATURAL_total_tile_count"),
        context_params={"num_class_chunks": "src_scalar_NATURAL_num_class_chunks"},
    ),
)
```

The OpenCL binding injects `flat_tile_index` as a positional scalar and loops over tiles. The Vulkan binding dispatches `vkCmdDispatch(total_tile_count, 1, 1)`. The CPU binding passes `task_index` through `pool_dispatch_and_wait`. The contract is identical across all three — it specifies the placement *strategy*, not the *mechanism*.

---

## Consequences

### Positive

- **Single validation, zero duplication.** The `KernelContract` is validated once at plan-construction time in the shared layer. No backend reimplements buffer shape checks, calculability proof verification, or scalar constraint validation. CONTRACT.md Article 1.4a (Host Proof Obligation) is satisfied exactly once.

- **Backend rendering freedom.** Each backend's `KernelBinding` translates the validated contract into its native dispatch format without constraint. OpenCL keeps its positional arg lists. Vulkan uses push constants and descriptor sets. CPU uses typed C structs. No backend is forced into another's dispatch idiom.

- **Abstract placement keys resolve the `flat_tile_index` divergence.** The contract specifies placement semantics abstractly (strategy + key domain). The mechanism is a binding concern. This is consistent with CONTRACT.md Article 3.2's existing abstraction and with ADR-002's `placement_strategy` field on `KernelDispatchNode`.

- **Plan-as-data-structure fidelity.** The `KernelContract` is a frozen dataclass — pure data, no methods, no backend types. It satisfies ADR-001's plan-as-data-structure requirement and can be serialized, inspected, and compared across plan instances.

- **ADR-004 stride table compatibility.** `ParameterStride.param_name` references abstract parameter names from the `KernelContract`'s `scalar_params` manifest. No backend-specific parameter names leak into the stride table.

- **Incremental extraction.** The existing `KernelSignature` hierarchy can be split incrementally: extract the validation logic into `KernelContract` dataclasses while the existing signature classes become the OpenCL bindings. No big-bang refactoring required. This maps to ADR-017 Phase 0 (extract `KernelContract`) and Phase 2 (implement OpenCL `KernelBinding`).

- **Extensibility for new backends.** Adding a new backend requires implementing `KernelBinding` classes in its package. No shared-layer changes. The contract's abstract parameter manifest and placement model accommodate backends that have not yet been conceived.

### Negative

- **Contract + binding = more artefacts per kernel.** Each of the 20+ kernels now has a shared `KernelContract` definition and at least one `KernelBinding` implementation. This increases the artefact count but reduces coupling — each artefact has a single, well-defined responsibility.

- **CONTRACT.md amendment required.** Article 3.2.3's `Mandatory Key Parameter` column currently names a specific OpenCL scalar parameter. The amendment must distinguish the abstract key concept from its backend realization. This is a documentation change, not a behavioral change.

- **Local memory size formulas are symbolic.** The `LocalMemorySpec.size_formula` is a string expression that must be evaluated at bind time. This requires each binding to parse or interpret the formula — a minor complexity cost. An alternative (pre-computed integer sizes) would require the contract to know `scalar_bytes` at contract-definition time, coupling the contract to precision configuration (ADR-008). The symbolic representation avoids this coupling.

- **Partial loss of parameter-list completeness in the contract.** A reader examining only the `KernelContract` for a partial-rendering kernel will not see `flat_tile_index` — that parameter is an OpenCL binding artefact. The reader must understand the placement model to know that the tile index is delivered through the abstract placement key. This is a conceptual overhead but a correct separation of concerns.

### CONTRACT.md amendment

Article 3.2.3's table must be amended to distinguish abstract key semantics from backend-specific key delivery:

- The current `Mandatory Key Parameter` column (e.g., `src_scalar_NATURAL_flat_tile_index`) is replaced with an `Abstract Placement Key` column specifying the key's semantic name and domain.
- A new note beneath the table states: "The abstract placement key identifies which logical tile a kernel invocation processes. The delivery mechanism is backend-specific: an explicit host scalar per dispatch (OpenCL), a hardware-provided work-group index (Vulkan), or a thread-pool task index (CPU). Backend bindings (ADR-007) are responsible for translating the abstract key into the native mechanism."
- The `Required Context Parameters` column remains unchanged — context parameters are shared-layer scalar parameters that exist in all backends.

### Migration implications

Per ADR-017:

**Phase 0 (Foundation):**
- Define the `KernelContract` dataclass hierarchy in `shared/kernel_contracts/__init__.py` (or `shared/plan_types.py`).
- Extract contracts from existing `KernelSignature` subclasses. Each contract is a frozen dataclass literal — no executable code. The existing signatures remain functional and unchanged.
- Validate that every existing `KernelSignature`'s `__post_init__` validation logic is expressible as `KernelContract` fields.

**Phase 2 (OpenCL Renderer):**
- Implement OpenCL `KernelBinding` classes in `backends/opencl/kernel_bindings/`. Each binding wraps the logic currently in `get_args()` and `get_grid()`.
- The OpenCL renderer creates bindings from plan nodes: `binding = OpenCLForwardPassBinding(contract, physical_buffers, arch_consts)`.
- Validation gate: the new contract + binding path produces bit-identical dispatch arguments to the old `KernelSignature` path.

**Phase 3+ (CPU, Vulkan):**
- Each backend implements its own `KernelBinding` classes.
- Cross-backend tests (ADR-016) verify that contracts produce correct bindings for each backend.

---

## References

- [ADR-001: Backend Abstraction Boundary](ADR-001-backend-abstraction-boundary.md) — plan-as-data-structure principle; three-tier jurisdictional model (Policy / Orchestration / Execution); plan boundary stops at the kernel's public interface
- [ADR-002: Plan Node Types & Synchronization Structure](ADR-002-plan-node-types-and-synchronization-structure.md) — `KernelDispatchNode.contract: KernelContract`; `placement_strategy` field; dispatch granularity independence
- [ADR-003: Reduction Tree Plan Representation](ADR-003-reduction-tree-plan-representation.md) — jurisdictional split precedent (Policy tier precomputes thresholds; Orchestration tier renders); kernel tier selection is an Orchestration concern
- [ADR-004: Streaming Loop Plan Representation](ADR-004-streaming-loop-plan-representation.md) — `ParameterStride.param_name` references abstract parameter names; body node contract validation
- [ADR-005: Node 16 Opacity in the Plan](ADR-005-node-16-opacity-in-the-plan.md) — plan-time contract validation for opaque specialized kernels; Policy-tier scalar parameters
- [ADR-006: Hardware Profile](ADR-006-hardware-profile.md) — `HardwareProfile` as shared-layer frozen dataclass; backend-constructed, shared-consumed
- [CONTRACT.md](../CONTRACT.md) — Article 1.4 Collaborative Interface Verifiability; Article 2 Parameter Lexical Mandate; Article 3 Parameter Commentary Contract; Article 3.2 Placement Contract; Article 4 Kernel Contract Block
- [CONCEPT.md](../CONCEPT.md) — §1 Architectural Elegance Feedback; §3 Modular Dumb Kernels
- [VULKAN_BACKEND.md](../VULKAN_BACKEND.md) — push constants for scalars; descriptor set bindings for buffers; `gl_WorkGroupID.x` for tile index; specialization constants for build-time symbols
- [CPU_BACKEND.md](../CPU_BACKEND.md) — `pool_dispatch_and_wait` with `task_index`; typed C function parameters; stack-allocated local memory
