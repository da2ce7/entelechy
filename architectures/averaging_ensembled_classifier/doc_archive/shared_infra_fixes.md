# Infrastructure Evolution Recommendations

The infrastructure is well-factored — the design intent is clear and the type contracts are precise. The friction points I identified are mostly places where the abstractions *almost* connect but force the plan builder to bridge a small gap with manual arithmetic or redundant knowledge. Each recommendation below targets a specific hand-stitching pattern in `plan_builder.py` that could be absorbed into the infrastructure.

---

## 1. `ParameterSlice` Cannot Be Constructed From `ParameterGeometry`

**The problem.** The plan builder needs a slice describing the adam kernel's view into a state buffer for module chunk `mc`. The information is split across three objects:

```python
# plan_builder.py — manual stitching
ps = ParameterSlice(
    offset=tiling.modules.offset_for(mc) * geo.elements_per_module,
    count=tiling.modules.count_for(mc) * geo.elements_per_module,
    total=geo.total_flat_elements,
)
```

`ParameterSlice.from_chunk` exists but computes `total = decomp.total × elements_per_item`, which is wrong when padding inflates the physical buffer beyond the logical extent (temperatures: `num_modules × 1 ≠ padded_module_dim`).

**Root cause.** `from_chunk` doesn't know about the geometry's physical total. It derives `total` from the decomposition's logical count, which is the *module count*, not the *buffer element count*.

**Suggested fix.** Add a factory on `ParameterGeometry` that produces the correct slice directly:

```python
# parameter_space.py
@dataclass(frozen=True)
class ParameterGeometry:
    # ...existing fields...

    def slice_for_chunk(
        self,
        module_decomp: ChunkDecomposition,
        chunk_index: int,
    ) -> ParameterSlice:
        """Optimizer dispatch slice for one module chunk.

        Uses ``elements_per_module`` for offset/count arithmetic
        and ``total_flat_elements`` (the physical buffer extent
        including padding) for ``total``.

        Raises ``ValueError`` for shared-scope groups where
        ``elements_per_module`` is ``None``.
        """
        if self.elements_per_module is None:
            raise ValueError(
                f"slice_for_chunk is not applicable to shared-scope "
                f"group {self.group.name!r} (elements_per_module is None)"
            )
        epm = self.elements_per_module
        return ParameterSlice(
            offset=module_decomp.offset_for(chunk_index) * epm,
            count=module_decomp.count_for(chunk_index) * epm,
            total=self.total_flat_elements,
        )
```

The plan builder then becomes:

```python
ps = geo.slice_for_chunk(tiling.modules, mc)
```

One call, no raw arithmetic, and the invariant `total = physical buffer extent` is structurally guaranteed.

For shared groups, `ParameterSlice.full(geo.total_flat_elements)` already works correctly.

---

## 2. `ModuleChunkGather` Requires External `elements_per_tile` Computation

**The problem.** Every module-chunk reduction tree in the plan builder computes elements-per-tile identically:

```python
epp = mpc * geo.elements_per_module
gather = ModuleChunkGather(
    geometry=tiling, module_chunk=mc,
    elements_per_tile=epp,
)
```

The caller must know that "tile element count = modules_per_chunk × elements_per_module" — a formula that's inherent to the relationship between `TilingGeometry` and `ParameterGeometry`.

**Root cause.** `TilingGeometry.gather_module_chunk` requires `elements_per_tile` as a raw integer. It has no way to derive this from the parameter geometry because it doesn't know about parameter shapes.

**Suggested fix.** Add a convenience method that accepts the geometry:

```python
# workload_primitives.py
@dataclass(frozen=True)
class TilingGeometry:
    # ...existing...

    def gather_module_chunk_for(
        self,
        module_chunk: int,
        geo: ParameterGeometry,
    ) -> ModuleChunkGather:
        """Gather scoped to one module chunk, sized from geometry.

        ``elements_per_tile = modules.chunk_size × geo.elements_per_module``.

        Raises ``ValueError`` if ``geo.elements_per_module`` is ``None``
        (shared-scope groups are not module-chunked).
        """
        if geo.elements_per_module is None:
            raise ValueError(
                f"Module-chunk gather is not applicable to shared-scope "
                f"group {geo.group.name!r}"
            )
        return ModuleChunkGather(
            geometry=self,
            module_chunk=module_chunk,
            elements_per_tile=self.modules.chunk_size * geo.elements_per_module,
        )
```

Plan builder reduces to:

```python
gather = tiling.gather_module_chunk_for(mc, geo)
```

This eliminates the `epp` local and the `mpc × geo.elements_per_module` formula from every module-chunk loop iteration.

---

## 3. `ScratchBufferSpec` — Retire or Integrate

**The problem.** `StreamingLoopPlan` defines `scratch_buffers: tuple[ScratchBufferSpec, ...]` and the docstring says this is for *"renderer-internal scratch buffers ... NOT plan-level buffers"*. The plan builder ignores this entirely and allocates scratch as plan-level `BATCH_INTERMEDIATE` buffers.

This creates three inconsistencies:

1. `streaming_plan = StreamingLoopPlan(..., scratch_buffers=(), ...)` — always empty.
2. Scratch buffers appear in `ExecutionPlan.buffers` with full lifecycle annotations, contradicting the "NOT plan-level" intent.
3. Body nodes reference plan-level handles directly, weakening the two-level namespace isolation.

**Root cause.** The infrastructure provides two mechanisms (plan-level buffers and scratch specs) but neither is fully suitable. Plan-level handles are needed because body nodes' `buffer_bindings` is `dict[str, BufferHandle]`. `ScratchBufferSpec` has no `BufferHandle` — it's a pure size annotation with no way to connect to body node bindings.

**Recommendation.** This is a genuine design choice, not a simple fix. Two paths:

**(A) Promote scratch to plan-level (codify current practice):**

Remove `ScratchBufferSpec` (or mark it as renderer-only, never Policy-tier). Document that body-node buffer bindings reference plan-level handles, and the loop node is the sole top-level producer/consumer. This is what the plan builder already does — it just needs architectural acknowledgment.

Add a comment or `BufferRole` variant:

```python
class BufferRole(Enum):
    MODEL_STATE = "model_state"
    BATCH_INPUT = "batch_input"
    BATCH_INTERMEDIATE = "batch_intermediate"
    BATCH_OUTPUT = "batch_output"
    LOOP_SCRATCH = "loop_scratch"  # Owned by a StreamingLoopNode
```

**(B) Give `ScratchBufferSpec` buffer handles:**

Add a `handle: BufferHandle` field to `ScratchBufferSpec`. The plan builder allocates scratch handles through a loop-scoped mechanism, body nodes bind to these handles, and the plan's `buffers` dict includes them with a `LOOP_SCRATCH` role. This preserves the two-tier intent while making scratch trackable.

**My recommendation:** Path (A). The current practice works, `ScratchBufferSpec` adds complexity without a consumer, and the two-level namespace is already weakened by plan-level handle references in body nodes. Clean documentation is worth more than an unused abstraction.

---

## 4. `ReductionTreePlan` Should Carry Abstract Policy Parameters

**The problem.** The plan bakes a concrete threshold schedule for Node 16:

```python
n16_sched = policy.render_node16_schedule(
    workgroup_size=spec.num_modules,  # CPU-style
    ...
)
precomputed[b_n16_sched] = np.array(n16_schedule, ...)
```

GPU backends need to re-render with their actual workgroup size. The plan carries only the pre-rendered schedule — not the policy parameters needed to re-render.

**Root cause.** `ReductionTreePlan` and Node 16's dispatch carry final numeric values. The `StabilizationPolicy` instance is not attached to the plan.

**Suggested fix.** Add an optional policy-parameter passthrough to the plan:

```python
# stabilization_policy.py — new lightweight carrier
@dataclass(frozen=True)
class PolicyParameters:
    """Abstract policy parameters for backend re-rendering."""
    t_algorithmic: float
    lambda_: float
    compute_fp_format_max: float
    fan_in: int  # resolved K
```

```python
# plan_types.py — extend ExecutionPlan
@dataclass(frozen=True)
class ExecutionPlan:
    # ...existing fields...
    stabilization_params: PolicyParameters | None = None
```

The plan builder attaches it:

```python
plan = ExecutionPlan(
    ...,
    stabilization_params=PolicyParameters(
        t_algorithmic=policy.t_algorithmic,
        lambda_=policy.lambda_,
        compute_fp_format_max=policy.compute_fp_format_max,
        fan_in=policy.resolve_fan_in(hardware.max_reduce_fan_in),
    ),
)
```

GPU Orchestration tiers can then call `policy.render_node16_schedule(total_modules, actual_workgroup_size, ...)` using the abstract parameters rather than reverse-engineering the precomputed array.

---

## 5. `_BufferAllocator` Should Model Producer/Consumer Roles Accurately

**The problem.** Buffers with `update_` semantics (adam's `parameters`, `m1`, `m2`) are registered as consumers:

```python
for h in (chain.param_buf, chain.m1_buf, chain.m2_buf):
    alloc.add_consumer(h, aid)
```

But they're also *written*. `MODEL_STATE` buffers have `producing_node=None` (correct — they originate outside the plan), yet the allocator has no way to express "this node mutates the buffer in-place."

**Root cause.** `_BufferAllocator` models two roles (producer, consumer). `update_` is a third role (mutator) that the allocator conflates with consumer.

**Suggested fix.** Add a `mutators` set to complement `producers` and `consumers`:

```python
class _BufferAllocator:
    def __init__(self):
        # ...existing...
        self._mutators: dict[BufferHandle, set[str]] = {}

    def add_mutator(self, handle: BufferHandle, node_id: str) -> None:
        """Register a node that reads AND writes this buffer in-place."""
        self._mutators[handle].add(node_id)
        self._consumers[handle].add(node_id)  # still needs to be alive
```

And extend `BufferDescriptor`:

```python
@dataclass(frozen=True)
class BufferDescriptor:
    # ...existing...
    mutators: frozenset[str] = frozenset()
```

This gives future backends the information they need for correct fence/barrier insertion (a read-after-write barrier is different from a write-after-read barrier) without changing any existing semantics.

---

## 6. Common Scalar Dictionaries Should Be Infrastructure

**The problem.** The plan builder constructs several boilerplate scalar dictionaries:

```python
_tile_scalars: dict[str, int | float] = {
    "src_scalar_FLAG_problem_type": strategy.flag,
    "src_scalar_NATURAL_flat_tile_index": 0,
    "src_scalar_NATURAL_num_class_chunks": tiling.classes.num_chunks,
    "src_scalar_NATURAL_classes_per_chunk": cpc,
    "src_scalar_NATURAL_modules_per_chunk": mpc,
    "src_scalar_NATURAL_total_batch_count": batch_size,
    "src_scalar_NATURAL_total_output_class_count": spec.output_classes,
    "src_scalar_NATURAL_padded_total_output_class_count": spec.padded_class_dim,
    "src_scalar_NATURAL_total_modules_count": spec.num_modules,
    "src_scalar_NATURAL_total_tile_count": tc,
}
```

These dictionaries encode relationships between `ModelSpec`, `TilingGeometry`, and `ProblemTypeSpec` that are invariant across many kernels. Every time a new kernel or scalar is added to the contracts, the plan builder must be updated in multiple places.

**Suggested fix.** Add a method to `TilingGeometry` (or a standalone helper) that produces the common scalar dict:

```python
# workload_primitives.py
@dataclass(frozen=True)
class TilingGeometry:
    # ...existing...

    def common_tile_scalars(
        self,
        spec: ModelSpec,
        batch_size: int,
    ) -> dict[str, int]:
        """Scalars shared across all tiled gradient kernels."""
        return {
            "src_scalar_NATURAL_flat_tile_index": 0,
            "src_scalar_NATURAL_num_class_chunks": self.classes.num_chunks,
            "src_scalar_NATURAL_classes_per_chunk": self.classes.chunk_size,
            "src_scalar_NATURAL_modules_per_chunk": self.modules.chunk_size,
            "src_scalar_NATURAL_total_batch_count": batch_size,
            "src_scalar_NATURAL_total_output_class_count": spec.output_classes,
            "src_scalar_NATURAL_padded_total_output_class_count": spec.padded_class_dim,
            "src_scalar_NATURAL_total_modules_count": spec.num_modules,
            "src_scalar_NATURAL_total_tile_count": self.total_tiles,
        }
```

The plan builder then does:

```python
_tile_scalars = {
    **tiling.common_tile_scalars(spec, batch_size),
    "src_scalar_FLAG_problem_type": strategy.flag,
}
```

This keeps the `ProblemTypeSpec` injection explicit (it's mode-dependent) while centralising the invariant geometry scalars.

However, this introduces a dependency from `workload_primitives` → `ModelSpec`, which may be undesirable. An alternative is a free function in the plan builder module itself, or a new `plan_scalars.py` module.

---

## 7. `StridedGather` for Shared Groups — Manual Width Computation

**The problem.** Each shared group's reduction tree does:

```python
gather = StridedGather(
    count=n_stream, partial_width=geo.total_flat_elements,
)
```

The `partial_width` is always `geo.total_flat_elements` for shared groups. This is correct but forces the plan builder to know the mapping.

**Suggested fix.** Add a factory on `ParameterGeometry`:

```python
@dataclass(frozen=True)
class ParameterGeometry:
    # ...existing...

    def streaming_gather(self, num_chunks: int) -> StridedGather:
        """Gather for streaming-path reduction over ``num_chunks``
        iteration chunks.

        ``partial_width = total_flat_elements`` — each streaming
        chunk produces one complete gradient for the entire parameter
        group.
        """
        return StridedGather(
            count=num_chunks,
            partial_width=self.total_flat_elements,
        )
```

Plan builder becomes:

```python
gather = geo.streaming_gather(n_stream)
```

---

## 8. `OptimizerConfig` — Missing From Provided Files But Referenced

`plan_builder.py` imports `from .optimizer_config import OptimizerConfig` but this module wasn't provided. The plan builder uses it to construct adam scalars:

```python
_opt = optimizer if optimizer is not None else OptimizerConfig()
_adam = {
    "src_scalar_REAL_learning_rate": _opt.learning_rate,
    "src_scalar_REAL_beta1_pow_t": _opt.beta1 ** adam_step,
    ...
}
```

**Observation.** The bias-correction computation (`_opt.beta1 ** adam_step`) is done in-line in the plan builder. CONCEPT.md §11 states: *"the host computes beta1**t and beta2**t bias correction terms in high precision (FP64)"*. Python's `float.__pow__` uses C double arithmetic, so this is correct — but it's not verified. An `OptimizerConfig.bias_correction_terms(step: int) -> tuple[float, float]` method would make the FP64 guarantee structural:

```python
# optimizer_config.py (hypothetical)
def bias_correction_terms(self, step: int) -> tuple[float, float]:
    """Compute (beta1^t, beta2^t) in FP64."""
    return (float(self.beta1 ** step), float(self.beta2 ** step))
```

Similarly, `resolve_epsilon(prec)` is called but its semantics aren't visible. If it selects between `prec.compute_epsilon` and a user-provided epsilon, that logic should be in `OptimizerConfig`, not the plan builder.

---

## 9. `_GradientChainSpec` — Should Be Infrastructure, Not Plan-Builder-Internal

**The problem.** `_GradientChainSpec` encapsulates the reduce→normalise→adam chain's inputs and is the cornerstone of the parameter-space-driven approach. But it's a private dataclass inside `plan_builder.py`.

Any future plan builder variant (e.g. a multi-layer plan builder for the experimental kernels) would need to duplicate it. The spec is really a Policy-tier concept: "everything needed to build one parameter group's gradient chain."

**Suggested fix.** Promote to `parameter_space.py` or a new `gradient_chain.py`:

```python
@dataclass(frozen=True)
class GradientChainInputs:
    """Inputs for building one parameter group's
    reduce → normalise → adam [→ post-update] chain."""
    group: ParameterGroup
    geo: ParameterGeometry
    clipped_buf: BufferHandle
    gather: GatherDescriptor
    param_buf: BufferHandle
    m1_buf: BufferHandle
    m2_buf: BufferHandle
    param_slice: ParameterSlice
    mc: int | None = None
```

This also makes it testable independently — Tier 1 tests can verify that the chain construction logic produces correct plans from known inputs without needing a backend.

---

## 10. `_ModelStateBuffers.for_group` — Fragile String Matching

```python
def for_group(self, group: ParameterGroup) -> BufferHandle:
    return {
        "shared_weights": self.shared_weights,
        "shared_biases": self.biases_shared,
        ...
    }[group.name]
```

This uses a string dictionary keyed on `group.name`. If a `ParameterGroup` is ever renamed, this silently produces a `KeyError` at plan-construction time rather than a type error at import time.

**Suggested fix.** Use identity-based lookup:

```python
def for_group(self, group: ParameterGroup) -> BufferHandle:
    _MAP = {
        SHARED_WEIGHTS: self.shared_weights,
        SHARED_BIASES: self.biases_shared,
        MODULE_WEIGHTS: self.module_weights,
        MODULE_BIASES: self.module_biases,
        TEMPERATURES: self.temperatures,
    }
    try:
        return _MAP[group]
    except KeyError:
        raise ValueError(f"Unknown parameter group: {group!r}") from None
```

Since `ParameterGroup` is frozen, identity and equality coincide for the five singleton instances. This catches mismatches at the type level rather than via string fragility.

---

## Summary: Priority-Ordered Recommendations

| Priority | Item | Nature | Effort |
|:---------|:-----|:-------|:-------|
| **High** | §1 `ParameterGeometry.slice_for_chunk` | Correctness gap + ergonomics | Small |
| **High** | §3 Retire or integrate `ScratchBufferSpec` | Design inconsistency | ADR decision |
| **Medium** | §2 `gather_module_chunk_for` on `TilingGeometry` | Ergonomics | Small |
| **Medium** | §4 Abstract policy parameters in plan | GPU backend enablement | Small |
| **Medium** | §9 Promote `_GradientChainSpec` to infrastructure | Reusability | Small |
| **Medium** | §5 `_BufferAllocator` mutator tracking | Semantic precision | Small |
| **Low** | §7 `ParameterGeometry.streaming_gather` | Ergonomics | Trivial |
| **Low** | §6 Common scalar dict helper | Boilerplate reduction | Small |
| **Low** | §8 `OptimizerConfig.bias_correction_terms` | Contract enforcement | Trivial |
| **Low** | §10 Identity-based `for_group` lookup | Robustness | Trivial |

The highest-impact changes are §1 and §3. The first eliminates a correctness gap where the provided factory produces wrong values for padded groups. The second resolves a design inconsistency where infrastructure provides an abstraction that the sole consumer ignores. Together they would align the infrastructure with how plans are actually built, making the plan builder both shorter and more self-evidently correct.
