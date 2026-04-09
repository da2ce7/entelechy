# The Dual-Oracle Strategy: Option A + Option B (B1 & B2)

## Why This Combination Is Powerful

The deep insight is that A and B aren't redundant — they validate *each other* through **differential triangulation**. Each oracle is authoritative over a different concern, and disagreements between them are *always diagnostic*.

```
                    ┌─────────────────────────────┐
                    │     Your Engine (SUT)        │
                    │  OpenCL / Vulkan / CPU       │
                    └──────────────┬──────────────┘
                                   │ compare
                    ┌──────────────┴──────────────┐
                    │      Option A                │
                    │  "Faithful Oracle"            │
                    │  Manual math, per-tile clip   │
                    │  Authority: pipeline fidelity │
                    └──────────────┬──────────────┘
                          compare  │  compare
                 ┌─────────────────┴──────────────────┐
                 │                                     │
     ┌───────────┴───────────┐          ┌──────────────┴──────────────┐
     │     Option B1         │          │       Option B2             │
     │  "Autograd Flat"      │          │  "Autograd Tiled"           │
     │  Standard backward    │          │  Per-tile autograd          │
     │  No per-tile clip     │          │  With per-tile clip         │
     │  Authority: calculus  │          │  Authority: both            │
     └───────────────────────┘          └─────────────────────────────┘
```

---

## What Each Oracle Is Authoritative Over

| Concern | A (Faithful) | B1 (Flat) | B2 (Tiled) |
|---|:---:|:---:|:---:|
| Gradient calculus (∂L/∂W, etc.) | Manual — could have a typo | **Autograd — ground truth** | **Autograd — ground truth** |
| Per-tile decomposition correctness | Manual — matches engine | N/A (no tiles) | Autograd per tile |
| Node 11 group-wise clip interaction | **Exact replica** | Absent | **Exact replica** |
| Quadratic Scaling Policy during reduction | **Exact replica** | Absent | **Exact replica** |
| Node 13 gather/permute | **Exact replica** | Absent | Depends on impl |
| Adam EMA precision | **Manual FP64** | `torch.optim.Adam` or manual | **Manual FP64** |
| "Is the engine correct?" | Primary oracle | Sanity bound | Cross-check oracle |

---

## The Five Comparison Axes

### Axis 1: A vs B1 (clipping disabled in A)

**What you learn:** Are your manually-derived gradient formulas correct?

```python
# Disable clipping in A
oracle_a.clip_threshold = float('inf')

# Both compute gradients for same input
grads_a  = oracle_a.backward(X, y)   # manual math
grads_b1 = oracle_b1.backward(X, y)  # autograd

# These MUST agree to FP64 tolerance
assert torch.allclose(grads_a.dW_module, grads_b1.dW_module, atol=1e-12)
assert torch.allclose(grads_a.dW_shared, grads_b1.dW_shared, atol=1e-12)
assert torch.allclose(grads_a.d_temps,   grads_b1.d_temps,   atol=1e-12)
```

**Failure diagnosis:** You have a calculus bug in A's manual backward. Autograd is the arbiter.

### Axis 2: A vs B2 (clipping enabled, tile-level comparison)

**What you learn:** Does your per-tile clip implementation match autograd's per-tile gradients *before and after* the clip?

```python
# For each tile, B2 computes per-tile gradients via autograd
for tile in tile_decomposition:
    grad_tile_b2 = torch.autograd.grad(loss_tile, params, retain_graph=True)
    grad_tile_a  = oracle_a.compute_tile_gradient(tile)

    # Pre-clip: MUST agree (this is Axis 1 but per-tile)
    assert torch.allclose(grad_tile_a.pre_clip, grad_tile_b2, atol=1e-12)

    # Post-clip: MUST agree (both apply the same Node 11 formula)
    clipped_a  = oracle_a.clip_tile(grad_tile_a)
    clipped_b2 = oracle_b2.clip_tile(grad_tile_b2)  # same clip function
    assert torch.allclose(clipped_a, clipped_b2, atol=1e-12)
```

**Failure diagnosis:**
- Pre-clip disagrees → per-tile loss decomposition bug (how you slice the problem)
- Post-clip disagrees → clip function inconsistency (one oracle has a different formula)

### Axis 3: B1 vs B2 (quantify clipping effect)

**What you learn:** How much does per-tile clipping alter the training trajectory?

```python
# Run both for N steps, same data, same init
for step in range(N):
    oracle_b1.step(X, y)  # clip after sum
    oracle_b2.step(X, y)  # clip before sum, then reduce

    metrics.record(
        param_divergence = (oracle_b1.W - oracle_b2.W).norm(),
        loss_divergence  = abs(oracle_b1.loss - oracle_b2.loss),
        clip_activation_rate = oracle_b2.tiles_clipped / oracle_b2.total_tiles
    )
```

**This is pure research data.** It tells you:
- When `clip_activation_rate ≈ 0`: B1 and B2 converge identically (clipping isn't changing anything)
- When divergence grows: the per-tile strategy is materially shaping training — your architecture's clipping design *matters*
- Whether the Quadratic Scaling Policy's λ parameter actually produces measurably different long-term dynamics

### Axis 4: A vs Engine (the actual validation)

**What you learn:** Does your engine produce the same result as the Faithful Oracle?

```python
# Initialize engine and oracle_a with identical parameters
engine.load_state(oracle_a.export_state())

for step in range(N):
    engine_result  = engine.train_batch(X, y)
    oracle_result  = oracle_a.step(X, y)

    # Compare at precision-appropriate tolerance
    # FP32 engine vs FP64 oracle: atol ~1e-5 to 1e-6
    # FP16 storage engine vs FP64 oracle: atol ~1e-2 to 1e-3
    assert torch.allclose(engine_result.loss, oracle_result.loss, atol=tol)
    assert torch.allclose(engine_params, oracle_params, atol=tol)
```

### Axis 5: B2 vs Engine (cross-check without shared code)

**What you learn:** Does the engine match an *independently-derived* oracle (B2) that shares zero code with A?

This is the most valuable axis for catching systematic bugs. If A and the engine share a formula error (e.g., both transpose a Jacobian the same wrong way), B2 catches it because autograd computes the correct derivative independently.

---

## Implementation Architecture

### Shared Infrastructure

Both oracles share certain utility code without compromising independence:

```python
# shared/reference_utils.py — pure functions, no state

def temperature_scaled_softmax(logits, temps):
    """Numerically stable softmax with temperature scaling. FP64."""
    scaled = logits / temps
    shifted = scaled - scaled.max(dim=-1, keepdim=True).values
    exp_shifted = torch.exp(shifted)
    return exp_shifted / exp_shifted.sum(dim=-1, keepdim=True)

def temperature_scaled_sigmoid(logits, temps):
    """Numerically stable sigmoid with temperature. FP64."""
    scaled = logits / temps
    return torch.sigmoid(scaled)

def group_wise_clip(grad_list, threshold, eps=1e-12):
    """Node 11: single norm over concatenated gradients, single scale."""
    concat = torch.cat([g.flatten() for g in grad_list])
    norm = concat.norm(2)
    if norm > threshold:
        scale = threshold / (norm + eps)
        return [g * scale for g in grad_list]
    return grad_list

def quadratic_threshold(t_algo, lam, j):
    """Quadratic Scaling Policy threshold for layer j."""
    return t_algo + lam * j * j

def adam_update_fp64(param, grad, m1, m2, lr, beta1, beta2, eps, t):
    """Manual Adam with FP64 bias correction throughout."""
    beta1_pow_t = beta1 ** t  # FP64 naturally
    beta2_pow_t = beta2 ** t
    m1_new = beta1 * m1 + (1 - beta1) * grad
    m2_new = beta2 * m2 + (1 - beta2) * grad * grad
    m1_hat = m1_new / (1 - beta1_pow_t)
    m2_hat = m2_new / (1 - beta2_pow_t)
    param_new = param - lr * m1_hat / (torch.sqrt(m2_hat) + eps)
    return param_new, m1_new, m2_new
```

### Option A: Faithful Oracle

```python
class FaithfulOracle:
    """
    Manual forward + manual backward + per-tile clipping + staged reduction.
    Zero autograd. All torch.no_grad().

    Authority: pipeline fidelity (matches engine's exact node sequence).
    Weakness: manual gradient formulas could contain errors.
    """

    def __init__(self, config, tile_decomposition):
        self.dtype = torch.float64
        self.tile_decomp = tile_decomposition  # mirrors engine's chunking
        # State parameters
        self.W_shared = torch.randn(config.hidden_dim, config.input_dim, dtype=self.dtype)
        self.b_shared = torch.zeros(config.hidden_dim, dtype=self.dtype)
        self.W_module = torch.randn(config.num_modules, config.hidden_dim,
                                     config.num_classes, dtype=self.dtype)
        self.b_module = torch.zeros(config.num_modules, config.num_classes, dtype=self.dtype)
        self.temps    = torch.ones(config.num_modules, dtype=self.dtype)
        # Adam state
        self.m1 = {k: torch.zeros_like(v) for k, v in self.named_params()}
        self.m2 = {k: torch.zeros_like(v) for k, v in self.named_params()}
        self.t = 0

    @torch.no_grad()
    def forward(self, X):
        """Nodes 4 → 5 → 6/7. Caches intermediates for backward."""
        self.X = X
        self.H = F.relu(X @ self.W_shared.T + self.b_shared)        # Node 4
        self.logits = torch.einsum('bh,mhc->bmc', self.H, self.W_module) + self.b_module  # Node 5
        if self.mode == 'CCE':
            self.probs = temperature_scaled_softmax(self.logits, self.temps[:, None])  # Node 6
            loss = -torch.log(self.probs.gather(-1, targets.unsqueeze(-1)) + 1e-30)
        else:
            self.probs = temperature_scaled_sigmoid(self.logits, self.temps[:, None, None])  # Node 7
            # stable BCE ...
        return self.probs, loss

    @torch.no_grad()
    def backward_tiled(self, targets, sample_mask):
        """Nodes 8-11, per-tile with group-wise clip."""
        clipped_grad_Wmod_tiles = []
        clipped_grad_bmod_tiles = []
        clipped_grad_temp_tiles = []
        clipped_grad_H_tiles    = []

        for tile in self.tile_decomp.tiles():
            m_slice, c_slice, b_slice = tile.module_slice, tile.class_slice, tile.batch_slice

            # --- Node 8: dL/dW_module, dL/db_module for this tile ---
            # d_logits = (probs - one_hot_or_targets) / temp  [for this tile's slice]
            d_logits_tile = self._compute_d_logits(tile, targets)
            grad_Wmod = torch.einsum('bh,bc->hc', self.H[b_slice], d_logits_tile)
            grad_bmod = d_logits_tile.sum(dim=0)

            # --- Node 9: dL/dH for this tile ---
            grad_H = torch.einsum('bc,hc->bh', d_logits_tile, self.W_module[m_slice, :, c_slice])

            # --- Node 10: dL/d_temp for this tile ---
            grad_temp = self._compute_d_temp(tile, targets)

            # --- Node 11: group-wise clip over all 4 ---
            threshold = self._get_tile_threshold(tile)
            clipped = group_wise_clip(
                [grad_Wmod, grad_bmod, grad_temp, grad_H],
                threshold
            )
            clipped_grad_Wmod_tiles.append(clipped[0])
            clipped_grad_bmod_tiles.append(clipped[1])
            clipped_grad_temp_tiles.append(clipped[2])
            clipped_grad_H_tiles.append(clipped[3])

        # --- Node 13: gather + permute grad_H ---
        # Sum across class chunks (implicit reduction), produce SoA layout
        summed_grad_H_per_module = self._gather_and_permute(clipped_grad_H_tiles)

        # --- Node 16: staged reduction of grad_H with Quadratic Scaling Policy ---
        summed_grad_H = self._staged_reduce_grad_h(
            summed_grad_H_per_module, self.policy
        )

        # --- Nodes 15: reduce module grads with Quadratic Scaling Policy ---
        summed_grad_Wmod = self._staged_reduce(clipped_grad_Wmod_tiles, self.policy)
        summed_grad_bmod = self._staged_reduce(clipped_grad_bmod_tiles, self.policy)
        summed_grad_temp = self._staged_reduce(clipped_grad_temp_tiles, self.policy)

        # --- Nodes 17-19: streaming shared backprop ---
        clipped_grad_Wshared_chunks, clipped_grad_bshared_chunks = \
            self._streaming_shared_backprop(summed_grad_H, sample_mask)

        # --- Node 20: reduce shared grads ---
        summed_grad_Wshared = self._staged_reduce(clipped_grad_Wshared_chunks, self.policy)
        summed_grad_bshared = self._staged_reduce(clipped_grad_bshared_chunks, self.policy)

        # --- Node 21: normalize ---
        N_eff = sample_mask.sum()
        self.final_grads = {
            'W_module': summed_grad_Wmod / (N_eff + 1e-12),
            'b_module': summed_grad_bmod / (N_eff + 1e-12),
            'temps':    summed_grad_temp / (N_eff + 1e-12),
            'W_shared': summed_grad_Wshared / (N_eff + 1e-12),
            'b_shared': summed_grad_bshared / (N_eff + 1e-12),
        }

    @torch.no_grad()
    def optimizer_step(self):
        """Node 24 + 25: Manual Adam + temperature clamp."""
        self.t += 1
        for name, param in self.named_params():
            param_new, self.m1[name], self.m2[name] = adam_update_fp64(
                param, self.final_grads[name],
                self.m1[name], self.m2[name],
                self.lr, self.beta1, self.beta2, self.eps, self.t
            )
            param.copy_(param_new)
        self.temps.clamp_(self.temp_min, self.temp_max)  # Node 25
```

### Option B: Autograd Oracle (Both Variants)

```python
class AutogradOracle:
    """
    PyTorch autograd for gradient computation.
    Two modes: flat (B1) and tiled (B2).

    Authority: gradient calculus (autograd is the arbiter of ∂L/∂W).
    B1 weakness: no per-tile clipping — diverges when clipping activates.
    B2 strength: per-tile autograd gradients + same clip pipeline as A.
    """

    def __init__(self, config, tile_decomposition=None):
        self.dtype = torch.float64
        self.tile_decomp = tile_decomposition  # None for B1, required for B2

        # nn.Parameters for autograd (but we manage optimizer manually)
        self.W_shared = nn.Parameter(torch.randn(..., dtype=self.dtype))
        self.b_shared = nn.Parameter(torch.zeros(..., dtype=self.dtype))
        self.W_module = nn.Parameter(torch.randn(..., dtype=self.dtype))
        self.b_module = nn.Parameter(torch.zeros(..., dtype=self.dtype))
        self.temps    = nn.Parameter(torch.ones(..., dtype=self.dtype))
        self.all_params = [self.W_shared, self.b_shared,
                           self.W_module, self.b_module, self.temps]

        # Manual Adam state (NOT torch.optim — we need FP64 bias correction)
        self.m1 = {id(p): torch.zeros_like(p) for p in self.all_params}
        self.m2 = {id(p): torch.zeros_like(p) for p in self.all_params}
        self.t = 0

    def forward(self, X):
        """Standard autograd-tracked forward pass. Nodes 4 → 5 → 6/7."""
        H = F.relu(X @ self.W_shared.T + self.b_shared)
        logits = torch.einsum('bh,mhc->bmc', H, self.W_module) + self.b_module
        scaled_logits = logits / self.temps[None, :, None]

        if self.mode == 'CCE':
            probs = F.softmax(scaled_logits, dim=-1)
        else:
            probs = torch.sigmoid(scaled_logits)
        return probs, scaled_logits

    # ─────────────────────────────────────────────────────
    # B1: Flat (post-summation clip)
    # ─────────────────────────────────────────────────────
    def backward_flat(self, loss):
        """
        Standard autograd backward. Clipping applied AFTER summation.
        Training trajectory diverges from engine when clipping activates.
        """
        loss.backward()
        # Gradients are now in .grad attributes — already summed across batch
        # Optional: apply a single global clip (NOT per-tile)
        if self.clip_threshold < float('inf'):
            all_grads = [p.grad for p in self.all_params]
            total_norm = torch.cat([g.flatten() for g in all_grads]).norm(2)
            if total_norm > self.clip_threshold:
                scale = self.clip_threshold / (total_norm + 1e-12)
                for g in all_grads:
                    g.mul_(scale)

    # ─────────────────────────────────────────────────────
    # B2: Tiled (per-tile autograd + clip)
    # ─────────────────────────────────────────────────────
    def backward_tiled(self, X, targets, sample_mask):
        """
        Per-tile gradient computation using torch.autograd.grad().
        Reproduces engine's exact Node 8-11 pipeline via autograd.
        """
        clipped_tiles = {name: [] for name in ['W_module', 'b_module', 'temps', 'grad_H']}

        for tile in self.tile_decomp.tiles():
            # Recompute forward for this tile's (module, class) scope
            # to isolate the tile's contribution to the loss
            tile_loss = self._compute_tile_loss(X, targets, tile)

            # Autograd gives us exact per-tile gradients
            # retain_graph=True because tiles share the forward graph
            tile_grads = torch.autograd.grad(
                tile_loss, self.all_params,
                retain_graph=True, create_graph=False
            )

            # Map autograd output to named gradient buffers
            grad_Wmod  = tile_grads[2]  # dL_tile/dW_module
            grad_bmod  = tile_grads[3]  # dL_tile/db_module
            grad_temps = tile_grads[4]  # dL_tile/d_temps
            grad_H     = tile_grads[0] @ self.W_shared  # chain into hidden

            # --- Node 11: same group-wise clip as Oracle A ---
            threshold = self._get_tile_threshold(tile)
            clipped = group_wise_clip(
                [grad_Wmod, grad_bmod, grad_temps, grad_H],
                threshold
            )
            for name, val in zip(clipped_tiles.keys(), clipped):
                clipped_tiles[name].append(val)

        # Reduction + normalization: identical to Oracle A
        # (same staged_reduce, same Quadratic Scaling Policy)
        ...

    def _compute_tile_loss(self, X, targets, tile):
        """
        Compute loss restricted to this tile's (module_chunk, class_chunk)
        scope. This is the key B2 trick: partial-scope loss enables
        per-tile autograd.
        """
        m_start, m_end = tile.module_range
        c_start, c_end = tile.class_range

        H = F.relu(X @ self.W_shared.T + self.b_shared)
        logits_tile = (torch.einsum('bh,mhc->bmc',
                        H,
                        self.W_module[m_start:m_end, :, c_start:c_end])
                       + self.b_module[m_start:m_end, c_start:c_end])

        scaled = logits_tile / self.temps[m_start:m_end, None, None]

        if self.mode == 'CCE':
            # NOTE: softmax needs full class dimension for correctness
            # This is where B2 gets nuanced — see discussion below
            ...
        else:
            probs = torch.sigmoid(scaled)
            targets_slice = targets[:, c_start:c_end]
            # stable BCE per-element
            loss = F.binary_cross_entropy_with_logits(
                scaled, targets_slice, reduction='sum'
            )
        return loss

    @torch.no_grad()
    def optimizer_step(self):
        """Same manual Adam as Oracle A — shared utility."""
        self.t += 1
        for param in self.all_params:
            param_new, self.m1[id(param)], self.m2[id(param)] = adam_update_fp64(
                param.data, self.final_grads[id(param)],
                self.m1[id(param)], self.m2[id(param)],
                self.lr, self.beta1, self.beta2, self.eps, self.t
            )
            param.data.copy_(param_new)
        self.temps.data.clamp_(self.temp_min, self.temp_max)
```

---

## The Critical B2 Subtlety: CCE Tile Decomposition

There's a genuine mathematical subtlety with B2 under CCE (Softmax) that doesn't exist under BCE (Sigmoid).

**BCE is separable by class:** Each class's loss is independent, so `_compute_tile_loss` can slice the class dimension freely. Per-tile autograd gives the exact same gradient as computing over all classes and then slicing.

**CCE is NOT separable by class:** Softmax's denominator couples all classes. Your engine handles this by computing softmax over the *full* class range (Node 6 sees all classes for a module), then the *gradient computation* (Node 8) operates on per-tile probability slices.

This means B2's `_compute_tile_loss` for CCE must:

```python
def _compute_tile_loss_cce(self, X, targets, tile):
    # Softmax MUST be computed over all classes (full logits)
    H = F.relu(X @ self.W_shared.T + self.b_shared)
    full_logits = torch.einsum('bh,mhc->bmc', H,
                                self.W_module[m_start:m_end]) + self.b_module[m_start:m_end]
    full_scaled = full_logits / self.temps[m_start:m_end, None, None]
    full_probs = F.softmax(full_scaled, dim=-1)  # over ALL classes

    # But the "tile loss" is only the cross-entropy contribution
    # from classes in this tile's range
    # This is subtle: the gradient of log(softmax) w.r.t. logits
    # for class c involves ALL probabilities, not just class c
    #
    # Solution: use the full-class loss but mask the gradient contribution
    # Actually, the correct approach is: the ENGINE computes per-tile
    # d_logits = (p_c - 1_{c=target}) / temp, which decomposes additively.
    # So we CAN do per-tile gradient computation from the FULL probs.

    # Compute d_logits for this tile's class range only
    d_logits_tile = full_probs[..., c_start:c_end].clone()
    # Subtract 1 at target position if target falls in this chunk
    for b in range(X.shape[0]):
        t = targets[b]
        if c_start <= t < c_end:
            d_logits_tile[b, :, t - c_start] -= 1.0
    d_logits_tile /= self.temps[m_start:m_end, None, None]

    # Now use autograd to propagate THIS tile's d_logits back
    # through the network to get per-tile parameter gradients
    tile_pseudo_loss = (full_scaled[..., c_start:c_end] * d_logits_tile).sum()
    return tile_pseudo_loss
```

This is where B2 gets more complex than B1 but still tractable. The key insight is that your engine's Node 8 computes `d_logits = (probs - targets) / temp` per tile, which *is* a valid per-tile decomposition of the full softmax gradient — the `probs` already incorporate the full-class coupling. B2 must replicate this decomposition faithfully to match A and the engine.

---

## The Differential Test Matrix

```
┌─────────────┬─────────────┬───────────────────────────────────────────────────┐
│ Comparison   │ Clip State  │ What Disagreement Means                          │
├─────────────┼─────────────┼───────────────────────────────────────────────────┤
│ A vs B1      │ DISABLED    │ Manual gradient formula error in A               │
│ A vs B1      │ ENABLED     │ Expected divergence (different clip timing)      │
│ A vs B2      │ DISABLED    │ Per-tile decomposition bug in A or B2            │
│ A vs B2      │ ENABLED     │ Clip implementation difference (shared code → 0) │
│ B1 vs B2     │ ENABLED     │ Measures clipping strategy's net effect          │
│ B1 vs B2     │ DISABLED    │ Must be zero (same autograd, no clip)            │
│ A vs Engine  │ ENABLED     │ Engine implementation bug                        │
│ B2 vs Engine │ ENABLED     │ Engine bug (independent cross-validator)         │
│ A vs B2 vs   │ ENABLED     │ Three-way: if A≠Engine but B2=Engine → A is     │
│   Engine     │             │ wrong; if A=B2≠Engine → Engine is wrong          │
└─────────────┴─────────────┴───────────────────────────────────────────────────┘
```

The three-way comparison at the bottom is the real payoff. **Two independent oracles agreeing against the system under test is a near-certain bug localization.**

---

## Recommended Test Progression

### Stage 1: Foundation (no clipping, no tiling)

```python
def test_gradient_calculus():
    """A vs B1 with clipping disabled. Pure math validation."""
    oracle_a  = FaithfulOracle(config, clip=inf)
    oracle_b1 = AutogradOracle(config, mode='flat')

    # Identical init
    sync_params(oracle_a, oracle_b1)

    probs_a, loss_a   = oracle_a.forward(X)
    probs_b1, loss_b1 = oracle_b1.forward(X)

    assert allclose(probs_a, probs_b1)   # forward parity
    assert allclose(loss_a, loss_b1)

    oracle_a.backward_tiled(targets, sample_mask)    # no clip
    oracle_b1.backward_flat(loss_b1)

    # Sum of A's per-tile grads must equal B1's autograd grads
    assert allclose(sum(oracle_a.tile_grads['W_module']),
                    oracle_b1.W_module.grad)
```

### Stage 2: Tiled decomposition (no clipping)

```python
def test_tile_decomposition():
    """A vs B2 with clipping disabled. Tile-level gradient parity."""
    for tile in tiles:
        grad_a  = oracle_a.compute_tile_gradient(tile)   # manual
        grad_b2 = oracle_b2.compute_tile_gradient(tile)  # autograd
        assert allclose(grad_a, grad_b2)  # per-tile, per-parameter
```

### Stage 3: Clipping pipeline

```python
def test_clipping_pipeline():
    """A vs B2 with clipping enabled. Full pipeline parity."""
    oracle_a.clip_threshold = 1.0
    oracle_b2.clip_threshold = 1.0

    # After full backward + clip + reduce + normalize
    assert allclose(oracle_a.final_grads, oracle_b2.final_grads)
```

### Stage 4: Multi-step convergence

```python
def test_convergence_trajectory():
    """A vs B2 over 100 steps. Trajectory parity."""
    for step in range(100):
        oracle_a.step(X, y)
        oracle_b2.step(X, y)

        # Parameters should remain close (FP64, deterministic)
        for name in param_names:
            assert allclose(oracle_a.params[name], oracle_b2.params[name], atol=1e-10)
```

### Stage 5: Engine validation

```python
def test_engine_vs_oracles():
    """Three-way: A vs B2 vs Engine."""
    engine.load_state(oracle_a.export_state())
    oracle_b2.load_state(oracle_a.export_state())

    for step in range(10):
        engine.train_batch(X, y)
        oracle_a.step(X, y)
        oracle_b2.step(X, y)

        # A and B2 must agree (FP64 vs FP64)
        for name in param_names:
            assert allclose(oracle_a.params[name], oracle_b2.params[name], atol=1e-10)

        # Engine should be close (FP32/FP16 vs FP64)
        for name in param_names:
            assert allclose(engine.params[name], oracle_a.params[name],
                          atol=precision_tolerance[engine.precision_config])
```

### Stage 6: Clipping effect quantification

```python
def test_clipping_strategy_effect():
    """B1 vs B2: measures how much per-tile clipping alters dynamics."""
    divergence_log = []
    for step in range(1000):
        oracle_b1.step(X, y)
        oracle_b2.step(X, y)
        divergence_log.append({
            'step': step,
            'param_divergence': param_distance(oracle_b1, oracle_b2),
            'loss_b1': oracle_b1.last_loss,
            'loss_b2': oracle_b2.last_loss,
            'tiles_actively_clipped': oracle_b2.clip_stats.num_active,
        })
    # This is research output, not a pass/fail test
    save_analysis(divergence_log)
```

---

## Summary

| Aspect | A alone | B alone | A + B (both) |
|---|---|---|---|
| Gradient math correctness | Trust manual derivation | **Autograd guarantee** | **Cross-validated** |
| Per-tile clip correctness | Self-consistent only | B2 only | **Independently verified** |
| Bug localization power | Can't distinguish own bugs | Can't match engine pipeline | **Three-way triangulation** |
| Clipping strategy research | Can compare with/without | B1 vs B2 comparison | **Full experimental apparatus** |
| Effort | ~400 lines | ~350 lines (both variants) | ~600 lines (shared utilities) |
| Independence | Single implementation risk | Autograd is independent | **Two fully independent derivations** |

The combined strategy gives you something neither oracle provides alone: **confidence that both are correct**, which is the precondition for trusting either one as a reference for validating the engine.

# Convergence Rate Tracking: What "Similar Time" Actually Means

## The Short Answer

Yes — but with an important nuance. You have **three distinct comparison pairs**, and "similar convergence time" means something fundamentally different for each:

| Pair | Expected Relationship | What Divergence Means |
|---|---|---|
| A vs B2 | **Identical** trajectory | Bug in one of them |
| A/B2 vs B1 | **Similar** envelope, not identical | Measures per-tile clipping's cost/benefit |
| Engine vs A/B2 | **Tracking** within precision tolerance | Validates engine correctness |

---

## Pair 1: A vs B2 — This Is NOT a Convergence Comparison

A and B2 implement the **same algorithm** (per-tile group-wise clip, same Quadratic Scaling Policy, same staged reduction, same manual Adam with FP64 bias correction). Both are FP64. Both see the same data.

Their loss curves must be **identical to floating-point tolerance**:

```python
for step in range(10_000):
    oracle_a.step(X, y)
    oracle_b2.step(X, y)

    # This is a CORRECTNESS check, not a convergence comparison.
    # Tolerance is ~1e-10 (FP64 accumulation order differences)
    assert abs(oracle_a.loss - oracle_b2.loss) < 1e-10
    assert param_distance(oracle_a, oracle_b2) < 1e-10
```

If they diverge at step 47, that's not a "convergence rate difference" — it's a **bug**. The triangulation catches it:

- If A's gradient formula has an error → A drifts from B2 (autograd is the arbiter)
- If B2's tile decomposition is wrong → B2 drifts from A (A's manual decomposition mirrors the engine)
- If the shared `group_wise_clip` utility has a bug → both drift identically, and you catch it when neither matches the engine

**This pair is your correctness foundation.** Once A ≡ B2, you have one verified oracle with two independent derivations.

---

## Pair 2: A/B2 vs B1 — The Genuine Convergence Rate Comparison

This is where "similar time" becomes a real, empirically interesting question. B1 clips **after** summation; A/B2 clip **before** summation per tile. These are mathematically different algorithms that produce different gradient updates whenever clipping activates.

### What Determines Divergence

The two strategies produce identical updates when **no tile is clipped** (the threshold is never exceeded). They diverge proportionally to:

1. **Clip activation frequency** — how often any tile exceeds the threshold
2. **Clip severity** — how much the norm exceeds the threshold (a tile at 1.01× threshold barely differs; a tile at 10× threshold is heavily rescaled)
3. **Tile heterogeneity** — if all tiles have similar norms, per-tile clip ≈ post-sum clip; if one tile has a huge norm while others are small, per-tile clip selectively dampens the outlier while post-sum clip dampens everything

### The Three Convergence Regimes

```
Loss
  │
  │╲                        Regime 1: Early training (high gradients)
  │ ╲  B1 ──╮               Clipping frequently active.
  │  ╲      ╰── A/B2        Per-tile clip may converge FASTER (outlier dampening)
  │   ╲         │            or SLOWER (over-dampening small but valid gradients)
  │    ╲─ ─ ─ ─             depending on data distribution.
  │     ╲
  │      ╲                   Regime 2: Mid training (moderate gradients)
  │       ╲                  Clipping occasionally active.
  │        ╲                 Curves track closely but aren't identical.
  │         ╲
  │          ╲╌╌╌╌╌          Regime 3: Late training (small gradients)
  │           ╲              Clipping rarely/never active.
  │            ╲             Curves converge to same minimum (or not —
  │             ╲            different paths may find different local minima!)
  └──────────────────── Steps
```

### What to Track

```python
class ConvergenceTracker:
    def __init__(self):
        self.metrics = []

    def record(self, step, oracle_a, oracle_b1, oracle_b2):
        # B2 must match A (correctness invariant)
        assert abs(oracle_a.loss - oracle_b2.loss) < 1e-10

        self.metrics.append({
            'step': step,

            # --- Loss tracking ---
            'loss_a':  oracle_a.loss,
            'loss_b1': oracle_b1.loss,
            'loss_delta': abs(oracle_a.loss - oracle_b1.loss),
            'loss_ratio': oracle_a.loss / (oracle_b1.loss + 1e-30),

            # --- Convergence rate ---
            'loss_a_derivative':  oracle_a.loss - self.prev_loss_a,
            'loss_b1_derivative': oracle_b1.loss - self.prev_loss_b1,

            # --- Parameter trajectory ---
            'param_distance': param_l2_distance(oracle_a, oracle_b1),
            'param_cosine_sim': param_cosine_similarity(oracle_a, oracle_b1),

            # --- Clipping diagnostics (the causal explanation) ---
            'tiles_clipped_a': oracle_a.clip_stats.num_clipped,
            'total_tiles':     oracle_a.clip_stats.total_tiles,
            'clip_rate':       oracle_a.clip_stats.num_clipped / oracle_a.clip_stats.total_tiles,
            'mean_clip_ratio': oracle_a.clip_stats.mean_norm_over_threshold,
            'max_clip_ratio':  oracle_a.clip_stats.max_norm_over_threshold,

            # --- Gradient direction comparison ---
            # After all clipping+reduction+normalization,
            # how different are the final gradient directions?
            'final_grad_cosine_sim': cosine_similarity(
                oracle_a.final_grads_flat, oracle_b1.final_grads_flat
            ),
        })
```

### The Key Metrics and Their Interpretation

| Metric | Healthy Range | Concern Threshold | Meaning |
|---|---|---|---|
| `loss_ratio` | 0.95 – 1.05 | < 0.8 or > 1.2 | Losses within 5% |
| `final_grad_cosine_sim` | > 0.95 | < 0.8 | Clipping barely changes direction |
| `clip_rate` | 0 – 0.3 | > 0.5 | Majority of tiles unclipped |
| `param_cosine_sim` | > 0.99 | < 0.95 | Parameters still in same "neighborhood" |

### Steps-to-Target Comparison

```python
def measure_convergence_rate(oracle, X_batches, y_batches, target_loss):
    """Count steps to reach a target loss."""
    steps = 0
    for X, y in cycle(zip(X_batches, y_batches)):
        oracle.step(X, y)
        steps += 1
        if oracle.loss <= target_loss:
            return steps
        if steps > MAX_STEPS:
            return None  # did not converge

# Usage:
steps_a  = measure_convergence_rate(oracle_a,  data, target=0.1)
steps_b1 = measure_convergence_rate(oracle_b1, data, target=0.1)

# "Similar time" means:
ratio = steps_a / steps_b1
# ratio ∈ [0.8, 1.2]  → broadly similar convergence rate
# ratio < 0.8         → per-tile clip converges FASTER (outlier dampening helps)
# ratio > 1.2         → per-tile clip converges SLOWER (over-dampening hurts)
```

### Important: This Comparison Is Parameterized by λ and T_algorithmic

The convergence rate difference between A/B2 and B1 is **not a fixed property** of the architecture — it's a function of the Quadratic Scaling Policy hyperparameters:

```python
def sweep_policy_parameters(data):
    """How does the clipping policy affect convergence rate relative to flat?"""
    results = []
    for t_algo in [0.1, 1.0, 10.0, 100.0]:
        for lam in [0.0, 0.01, 0.1, 1.0, 10.0]:
            oracle_a  = FaithfulOracle(config, t_algorithmic=t_algo, lam=lam)
            oracle_b1 = AutogradOracle(config, mode='flat', clip_threshold=t_algo)

            sync_params(oracle_a, oracle_b1)

            steps_a  = measure_convergence_rate(oracle_a, data, target=0.1)
            steps_b1 = measure_convergence_rate(oracle_b1, data, target=0.1)

            results.append({
                't_algo': t_algo, 'lambda': lam,
                'steps_a': steps_a, 'steps_b1': steps_b1,
                'ratio': steps_a / steps_b1 if steps_b1 else None,
                'mean_clip_rate': oracle_a.mean_clip_rate,
            })

    return results
```

When `t_algo` is very high (rarely clips), the ratio → 1.0.
When `t_algo` is very low (always clips), the ratio diverges.
When `λ` is high (permissive funnel), early stages clip less → closer to B1.
When `λ = 0` (fixed ceiling), all stages clip at the same threshold → maximum divergence from B1.

---

## Pair 3: Engine vs A/B2 — The Actual Product Validation

Here "similar convergence time" means: does the finite-precision engine (FP32, FP16, FP8) track the FP64 oracle closely enough that it reaches the same quality in approximately the same number of steps?

### The Precision-Induced Drift Model

```
Loss
  │
  │╲  FP64 oracle (A/B2)
  │ ╲─────────────╮
  │  ╲             ╰─── FP32 engine (tight tracking)
  │   ╲                │
  │    ╲─ ─ ─ ─ ─ ─ ─ ╯
  │     ╲                    FP16 engine (wider envelope, same general shape)
  │      ╲    ╭─╮
  │       ╲──╯  ╰──╮
  │        ╲        ╰────
  │         ╲                FP8 engine (noisy, but convergent)
  │          ╲  ╭╮╭╮
  │           ╲╯╰╯ ╰──────
  └──────────────────────── Steps
```

### Per-Precision Tolerances

```python
CONVERGENCE_TOLERANCES = {
    # (loss_ratio_band, param_cosine_min, max_step_ratio)
    'float32':          (0.02,  0.999,  1.05),  # 2% loss, 0.1% direction, 5% more steps
    'mixed_f16_f32':    (0.05,  0.99,   1.10),  # 5% loss, 1% direction, 10% more steps
    'fp8_e4m3':         (0.15,  0.95,   1.25),  # 15% loss, 5% direction, 25% more steps
    'mixed_f32_f64_st': (0.005, 0.9999, 1.02),  # FP64 state → very tight tracking
}
```

```python
def test_engine_convergence_parity(engine, oracle_a, precision_config):
    tol = CONVERGENCE_TOLERANCES[precision_config.name]
    loss_band, cos_min, step_ratio_max = tol

    target_loss = oracle_a.converged_loss * 1.5  # generous target

    steps_oracle = measure_convergence_rate(oracle_a, data, target_loss)
    steps_engine = measure_convergence_rate(engine,   data, target_loss)

    assert steps_engine is not None, "Engine failed to converge"
    assert steps_engine / steps_oracle < step_ratio_max, (
        f"Engine took {steps_engine} steps vs oracle's {steps_oracle} "
        f"(ratio {steps_engine/steps_oracle:.2f}, max {step_ratio_max})"
    )

    # Also check that every N steps, the loss curves are in the same band
    for checkpoint in range(0, steps_oracle, steps_oracle // 10):
        loss_ratio = engine.loss_at[checkpoint] / oracle_a.loss_at[checkpoint]
        assert abs(1.0 - loss_ratio) < loss_band
```

### What Precision-Induced Convergence Slowdown Tells You

| Observation | Likely Cause | Action |
|---|---|---|
| FP32 tracks FP64 perfectly | Everything works | Ship it |
| FP16 converges but 20% slower | Gradient quantization noise | Expected; validate acceptable for use case |
| FP16 converges but to a higher loss floor | Accumulation precision loss | Check if problem disappears with `mixed_f16_f64_state()` |
| FP8 oscillates but trends downward | Quantization floor zeroing small gradients | Expected; verify final loss is acceptable |
| FP8 diverges | Clipping thresholds exceed FP8 max | Bug — safety ceiling should prevent this |
| Engine matches oracle for 100 steps then diverges | Adam moment precision erosion | Validates the FP64 state role — try `mixed_f32_f64_state()` |

---

## Putting It All Together: The Convergence Dashboard

```python
class ConvergenceDashboard:
    """Runs all three comparison pairs and produces a unified report."""

    def __init__(self, config, data, engine_configs):
        self.oracle_a  = FaithfulOracle(config)
        self.oracle_b1 = AutogradOracle(config, mode='flat')
        self.oracle_b2 = AutogradOracle(config, mode='tiled')
        self.engines   = {name: Engine(cfg) for name, cfg in engine_configs.items()}

        # Synchronize all initial parameters
        state = self.oracle_a.export_state()
        self.oracle_b1.load_state(state)
        self.oracle_b2.load_state(state)
        for engine in self.engines.values():
            engine.load_state(state)

    def run(self, num_steps, data_iter):
        history = []

        for step, (X, y) in enumerate(islice(cycle(data_iter), num_steps)):
            # Step all oracles and engines
            self.oracle_a.step(X, y)
            self.oracle_b1.step(X, y)
            self.oracle_b2.step(X, y)
            for engine in self.engines.values():
                engine.train_batch(X, y)

            history.append({
                'step': step,

                # ── Pair 1: A vs B2 (correctness invariant) ──
                'a_b2_loss_delta': abs(self.oracle_a.loss - self.oracle_b2.loss),
                'a_b2_param_dist': param_distance(self.oracle_a, self.oracle_b2),
                # INVARIANT: both must be < 1e-10

                # ── Pair 2: A/B2 vs B1 (clipping strategy effect) ──
                'loss_a':  self.oracle_a.loss,
                'loss_b1': self.oracle_b1.loss,
                'clip_rate': self.oracle_a.clip_stats.clip_rate,
                'grad_cosine_a_b1': cosine_sim(
                    self.oracle_a.final_grads_flat,
                    self.oracle_b1.final_grads_flat
                ),

                # ── Pair 3: Engine vs Oracle (implementation validation) ──
                **{
                    f'loss_{name}': engine.last_loss
                    for name, engine in self.engines.items()
                },
                **{
                    f'param_cos_{name}': param_cosine_similarity(
                        self.oracle_a, engine
                    )
                    for name, engine in self.engines.items()
                },
            })

        return ConvergenceReport(history)


class ConvergenceReport:
    """Analyzes and summarizes convergence metrics."""

    def correctness_check(self):
        """Pair 1: A ≡ B2 invariant. Any violation is a bug."""
        max_delta = max(h['a_b2_loss_delta'] for h in self.history)
        assert max_delta < 1e-10, f"A/B2 diverged at step {worst_step}: Δ={max_delta}"
        print(f"✓ A ≡ B2 invariant holds (max Δ = {max_delta:.2e})")

    def clipping_effect_summary(self):
        """Pair 2: How much does per-tile clipping change dynamics?"""
        mean_clip_rate = mean(h['clip_rate'] for h in self.history)
        final_loss_a  = self.history[-1]['loss_a']
        final_loss_b1 = self.history[-1]['loss_b1']
        mean_grad_cos = mean(h['grad_cosine_a_b1'] for h in self.history)

        print(f"Clipping activation rate: {mean_clip_rate:.1%}")
        print(f"Final loss: A={final_loss_a:.6f}, B1={final_loss_b1:.6f} "
              f"(ratio={final_loss_a/final_loss_b1:.4f})")
        print(f"Mean gradient cosine similarity: {mean_grad_cos:.6f}")

        if mean_clip_rate < 0.01:
            print("→ Clipping rarely activates. A ≈ B1. "
                  "Convergence rate comparison is moot.")
        elif mean_grad_cos > 0.99:
            print("→ Clipping active but gradient direction preserved. "
                  "Expect similar convergence rate.")
        else:
            print("→ Clipping significantly alters gradient direction. "
                  "Convergence rates may differ materially.")

    def engine_parity_summary(self):
        """Pair 3: Per-engine convergence tracking."""
        for name in self.engine_names:
            losses = [h[f'loss_{name}'] for h in self.history]
            oracle_losses = [h['loss_a'] for h in self.history]

            # Steps to reach oracle's median loss
            oracle_median = sorted(oracle_losses)[len(oracle_losses)//2]
            try:
                engine_steps = next(i for i, l in enumerate(losses)
                                   if l <= oracle_median)
                oracle_steps = next(i for i, l in enumerate(oracle_losses)
                                   if l <= oracle_median)
                ratio = engine_steps / oracle_steps
            except StopIteration:
                ratio = float('inf')

            final_cos = self.history[-1][f'param_cos_{name}']

            print(f"\n{name}:")
            print(f"  Steps-to-median ratio: {ratio:.2f}")
            print(f"  Final param cosine:    {final_cos:.6f}")
            print(f"  Final loss:            {losses[-1]:.6f} "
                  f"(oracle: {oracle_losses[-1]:.6f})")
```

### Example Output

```
═══════════════════════════════════════════════════
           CONVERGENCE REPORT (1000 steps)
═══════════════════════════════════════════════════

[Pair 1: Correctness Invariant]
✓ A ≡ B2 invariant holds (max Δ = 2.31e-14)

[Pair 2: Clipping Strategy Effect]
Clipping activation rate: 23.4%
Final loss: A=0.041523, B1=0.039871 (ratio=1.0414)
Mean gradient cosine similarity: 0.987342
→ Clipping active but gradient direction preserved.
  Expect similar convergence rate.

Steps-to-target (loss < 0.1):
  Oracle A (per-tile clip):     142 steps
  Oracle B1 (post-sum clip):    137 steps
  Ratio: 1.036 (3.6% slower with per-tile clip)

[Pair 3: Engine Parity]

float32:
  Steps-to-median ratio: 1.02
  Final param cosine:    0.999987
  Final loss:            0.041529 (oracle: 0.041523)

mixed_f16_f32:
  Steps-to-median ratio: 1.07
  Final param cosine:    0.999412
  Final loss:            0.042103 (oracle: 0.041523)

fp8_e4m3:
  Steps-to-median ratio: 1.18
  Final param cosine:    0.996854
  Final loss:            0.044271 (oracle: 0.041523)

mixed_f32_f64_state:
  Steps-to-median ratio: 1.01
  Final param cosine:    0.999994
  Final loss:            0.041524 (oracle: 0.041523)
```

---

## The Honest Caveat

"Similar convergence time" is only rigorously meaningful when you control for:

1. **Identical data order** — shuffle once, feed the same sequence to all
2. **Identical initialization** — `export_state()`/`load_state()` synchronization
3. **Identical effective hyperparameters** — B1's post-sum clip threshold is *not* directly comparable to A/B2's per-tile threshold; you need to define what "same" means
4. **Statistical significance** — one run isn't enough; run multiple seeds

The comparison between A/B2 and B1 is inherently an **apples-to-oranges** comparison because they implement different algorithms. The report tells you *how different the oranges are from the apples*, and whether the difference matters for your use case. It does not — and cannot — tell you one is "better" without defining what "better" means for your specific problem.

What it *can* definitively tell you is that your engine, at a given precision, converges to the same solution as the FP64 oracle that faithfully reproduces its algorithm. That is the validation that matters.
