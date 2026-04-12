# ADR-032: Oracle Testing

**Status:** PROPOSED
**Date:** 2026-04-11
**Deciders:** —
**Triggered by:** Oracle hierarchy (A/B/C/D) fully implemented in `tests/oracle/` without a unifying ADR; differential triangulation strategy documented only in `doc_archive/Oracle.md`
**Depends on:** ADR-016 (Test Strategy), ADR-008 (Precision Configuration), ADR-028 (Convergence Testing Standard Problems)
**Constrains:** `tests/oracle/` test package structure, oracle implementation contracts, cross-oracle validation axes
**Extends:** ADR-016 Oracle hierarchy, `doc_archive/Oracle.md` design

---

## Decision Summary

| Decision Axis | Selected Option | Rationale |
|:--------------|:----------------|:----------|
| Oracle hierarchy | **Four oracles (A/B/C/D)** with orthogonal authority domains | Each oracle is authoritative over a distinct concern; disagreements are always diagnostic |
| Validation strategy | **Differential triangulation** — cross-oracle comparison axes | No single oracle is trusted in isolation; cross-comparison proves correctness |
| Oracle D dependency isolation | **NumPy-only (zero torch)** | Independent implementation eliminates shared-bug risk with A/B/C |
| Test package organization | **Unified `tests/oracle/` package** | All oracle implementations and cross-validation tests colocated |

---

## Context

### The Correctness Problem

The engine implements a multi-backend training pipeline (OpenCL, Vulkan, CPU) with tiled decomposition, per-tile clipping, multi-hop gradient pipelines, and three-role precision (storage/compute/state).  Any single reference implementation could share bugs with the engine.  The solution is multiple independent oracles, each authoritative over a different concern, with cross-oracle comparison as the primary validation mechanism.

### Prior Art

`doc_archive/Oracle.md` designed the original dual-oracle strategy (A + B) and the five comparison axes.  ADR-016 established the three-tier test framework (plan correctness, kernel correctness, cross-backend parity) with CPU as the Tier 3 reference oracle.  Neither document codified the full four-oracle hierarchy or the test package structure as architectural decisions.

### The Oracle Package Today

The `tests/oracle/` package implements four oracles and three test modules:

| File | Role |
|:-----|:-----|
| `faithful_oracle.py` | Oracle A — manual FP64 gradient computation with tiling and clipping |
| `autograd_oracle.py` | Oracle B — `torch.autograd` FP64 (B1=flat, B2=tiled) |
| `convergence_oracle.py` | Oracle C — standard PyTorch FP64 multi-step training |
| `precision_oracle.py` | Oracle D — NumPy precision-aware training (zero torch dependency) |
| `oracle_config.py` | Shared `OracleConfig` dataclass for A/B/C |
| `reference_utils.py` | Shared numerical utilities (`adam_update_fp64`, `group_wise_clip`; also contains unused canonical `temperature_scaled_softmax`/`temperature_scaled_sigmoid` implementations) |
| `conftest.py` | Fixtures, config adapters, state synchronization helpers |
| `__init__.py` | Package marker |
| `test_differential.py` | A vs B cross-validation (5 comparison axes) |
| `test_convergence.py` | C cross-validation (C vs A, C vs B, C self-convergence, marathon) |
| `test_precision_oracle.py` | D cross-validation (D vs C, D precision sweeps, mask strategies) |

---

## Decision Drivers

1. **CONCEPT.md §1 (Architectural Elegance Feedback).**  If an oracle test reveals a pattern that cannot be expressed within the current oracle hierarchy — e.g., a validation concern not covered by any oracle's authority domain — the response is to extend the hierarchy, not to add ad-hoc tests outside it.

2. **ADR-016 (Test Strategy).**  The oracle hierarchy is the arbitration backbone for Tier 2 and Tier 3 tests.  Oracle test correctness is therefore a prerequisite for the entire test framework's trustworthiness.

3. **ADR-008 (Precision Configuration).**  Oracles A/B/C operate at FP64 throughout and validate mathematical correctness.  Oracle D extends the hierarchy with precision-parameterized validation, bridging the gap between FP64 truth and mixed-precision engine behavior.

4. **Independence maximization.**  Each oracle must share minimal code with the others and with the engine.  Shared bugs between oracle and engine are the most dangerous failure mode — they produce false passes.

5. **Diagnostic clarity.**  When a test fails, the comparison axis must identify *which concern* is violated.  "A disagrees with B1 at Axis 1" immediately diagnoses a gradient calculus bug in A.  Generic "oracle disagrees with engine" provides no diagnostic signal.

---

## Decisions

### 1. Oracle Hierarchy

Four oracles with orthogonal authority domains:

| Oracle | Implementation | Dependency | Authority |
|:-------|:--------------|:-----------|:----------|
| **A** (Faithful) | Manual FP64 PyTorch | torch | Pipeline fidelity: tiling, clipping, reduction, gather/permute |
| **B** (Autograd) | `torch.autograd` FP64 | torch | Gradient calculus: ∂L/∂W correctness (B1=flat, B2=tiled) |
| **C** (Convergence) | Standard PyTorch FP64 | torch | Multi-step convergence: optimizer state management, trajectory stability |
| **D** (Precision) | NumPy precision-aware | numpy | Precision-limited convergence: storage narrowing, compute-precision arithmetic, state-precision accumulation |

**Authority boundaries are strict:**  Oracle A is *not* authoritative for gradient calculus (B is the arbiter).  Oracle B is *not* authoritative for pipeline fidelity (A is the arbiter).  Oracle C is *not* authoritative for precision effects (D is the arbiter).  Oracle D is *not* authoritative for clipping or tiled decomposition (A is the arbiter).

### 2. Differential Triangulation

Cross-oracle comparison is the primary validation mechanism.  The comparison axes, derived from `doc_archive/Oracle.md`:

| Axis | Comparison | Purpose | Failure Diagnosis |
|:-----|:-----------|:--------|:------------------|
| 1 | A vs B1 (clipping disabled) | Gradient calculus correctness | Manual formula bug in A; autograd is arbiter |
| 2 | A vs B2 (clipping disabled) | Tile decomposition correctness | Per-tile loss decomposition bug |
| 3 | A vs B2 (clipping enabled) | Full pipeline parity | Clip / reduction interaction bug |
| 4 | B1 vs B2 (clipping disabled) | Autograd self-consistency | Tiling wrapper bug in B2 |
| 5 | A vs B2 (multi-step) | Convergence trajectory parity | Accumulation or state management bug |
| 6 | C vs B1 (single-step, no clipping) | C's gradient correctness | C has an independent implementation bug |
| 7 | C vs A (multi-step, no clipping) | Multi-step pipeline parity | Moment/optimizer state divergence |
| 7a | A (clipped) vs C (unclipped) loss curves | Convergence under clipping | Clipping causes divergence but not instability |
| 7b | C moment parity vs A after N steps | Optimizer state management | Adam m1/m2 drift across steps |
| 7c | C normalization invariance | Batch-size independence | Node 21 normalization correctness |
| 7d | C long-run stability (marathon) | No NaN/Inf at large t | Adam bias correction at very large step counts |
| 8 | D vs C (FP64) | D's implementation correctness | NumPy vs PyTorch implementation bug |
| 9 | D precision sweeps | Precision modeling non-triviality | Precision boundaries must produce measurable trajectory differences |
| 10 | D mask strategy | `explicit` vs `recompute` consistency | Strategy agreement at FP32/FP64 (FP8 divergence assertion is future work) |

Axes 1–5 are implemented in `test_differential.py`.  Axes 6–7d in `test_convergence.py`.  Axes 8–10 in `test_precision_oracle.py`.

### 3. Independence Contracts

| Contract | Mechanism |
|:---------|:----------|
| A, B, C share no gradient code | A uses manual formulas; B uses `torch.autograd`; C uses its own forward/backward |
| A, B share `adam_update_fp64` + `group_wise_clip` | Two shared utilities; C has no clipping and shares only `adam_update_fp64` |
| D shares no code with A/B/C | Zero torch dependency; pure NumPy |
| D shares no code with the engine | No imports from `src/` |
| All oracles use `OracleConfig` (A/B/C) or `OracleDConfig` (D) | Config adapters in `conftest.py` bridge the two; note `OracleDConfig` has a separate `normalize_epsilon` field (Node 21) not present in `OracleConfig`, which uses its `epsilon` field for both Adam and normalization |

### 4. State Synchronization

Cross-oracle tests require oracles to start from identical state.  Two mechanisms:

- **`init_with_seed(oracle, seed=123)`** — Initializes an oracle's weights from a deterministic seed (default 123).  Used for A/B comparisons and C self-convergence tests.
- **`sync_oracle_d_from_torch(oracle_d, torch_state)`** — Converts a torch-based oracle's exported state dict (from `export_state()`) to numpy arrays for Oracle D.  Callers pass `oracle_c.export_state()`, not the oracle instance itself.  The adapter handles dtype conversion and layout differences.

### 5. Test Execution Model

Oracle tests are part of the standard `pytest` suite with no special markers beyond `tests/oracle/` directory scoping.  All oracle-internal tests (A/B/C cross-validation) require PyTorch — the package-level `conftest.py` auto-skips when torch is unavailable.  Oracle D's self-tests (precision sweeps, mask strategy) require only numpy, but cross-validation against C requires torch.

---

## Consequences

### Positive

- **Every oracle is validated by at least two independent comparison axes.**  No oracle is trusted in isolation.
- **Failure diagnosis is immediate.**  Each comparison axis maps to a specific concern; the failing axis identifies the bug category.
- **Oracle D's torch-independence provides a completely separate verification path.**  A shared bug between torch-based oracles and the engine cannot infect Oracle D's results.
- **The hierarchy is extensible.**  A future Oracle E (e.g., for quantized integer arithmetic) would follow the same pattern: define authority domain, implement independently, add comparison axes.

### Negative

- **Four oracle implementations to maintain.**  Each must track the engine's mathematical model.  When the model changes (e.g., new loss function, new optimizer), all relevant oracles must be updated.
- **Cross-oracle state synchronization is fragile.**  The `sync_oracle_d_from_torch` adapter must handle every parameter shape and dtype correctly; a bug in the adapter produces false parity failures.
- **Torch dependency for most oracle tests.**  Only Oracle D's self-tests are torch-free; the full differential triangulation matrix requires PyTorch.

### Risks

- **Shared mathematical model errors.**  All oracles implement the same mathematical model (CONCEPT.md's averaged ensembled classifier).  If the model specification itself is wrong, all oracles agree on the wrong answer.  Mitigated by Oracle B's use of `torch.autograd` (autograd verifies the calculus independently of any manual derivation).
- **~~Config adapter drift (`normalize_epsilon`).~~**  RESOLVED — `oracle_config_to_d_config` now explicitly sets `normalize_epsilon=cfg.epsilon`, preserving `OracleConfig`'s single-epsilon semantic.  The adapter functions must still be updated whenever either config changes.
- **~~`conftest.py` numpy fixtures diverge from torch fixtures.~~**  RESOLVED — `small_cce_np_data` and `small_bce_np_data` now carry prominent docstring warnings that they produce different values from the torch fixtures despite using the same seed (different RNG algorithms).  D-vs-C tests convert torch tensors via `_torch_to_numpy()`.
- **~~`_make_unclipped_config` duplicated across test modules.~~**  RESOLVED — extracted to `conftest.py` as `make_unclipped_config`, imported by all three test modules.
- **`OracleConfig` / `OracleDConfig` default divergences.**  Several defaults differ between the two config classes: `epsilon` (1e-7 vs 1e-8), `temp_min` (0.01 vs 0.1), `temp_max` (100.0 vs 10.0).  The adapter transfers actual values for cross-oracle tests, so the divergence only matters for standalone Oracle D tests that construct `OracleDConfig` with defaults.  These defaults model different operating conditions (D's narrower temperature range reflects precision-limited stability), but the divergence is a footgun for anyone expecting config-class defaults to match.
- **B2 reduction pipeline shares code with A.**  `_staged_reduce_node16`, `_reduce_tiles_sum_and_clip`, and `_plan_reduction_tree` are copied verbatim between Oracle A and Oracle B2 (~250 lines).  A bug in the reduction tree logic would infect both oracles, causing them to agree on incorrect clipped gradients.  Oracle C (no reduction tree) is the safeguard — C-vs-A divergence under no-clipping conditions exposes shared-code bugs.

---

## Related ADRs

- **ADR-008** — Defines the precision model; oracles A/B/C validate at FP64, oracle D validates across precisions.
- **ADR-016** — Establishes the test framework; oracle tests are the correctness foundation for Tiers 2 and 3.
- **ADR-028** — Defines convergence test infrastructure that oracle C and D plug into.
- **ADR-031** — Defines mask strategies validated by Oracle D's explicit/recompute comparison.
- **ADR-033** — Defines Oracle D's specific integration into `tests/oracle/` and `tests/convergence/`.
