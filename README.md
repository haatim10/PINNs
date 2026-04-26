# PI-fMI: Physics-Informed Neural Networks for Fractional Integro-Differential Equations

**`feature/quantum-readiness` branch:** This branch preserves the classical PINN plus product-integration workflow for a time-fractional integro-differential equation, and adds a TE-QPINN-inspired surrogate model for controlled comparisons on the same PDE.

## At a Glance: Governing Problem

We solve:

$$D_t^{\alpha} u(x,t) - (x^2 + 1)\frac{\partial^2 u}{\partial x^2} + \int_0^t \sin(x)(t-s)^{-\beta}u(x,s)\,ds = f(x,t)$$

- Domain: $x \in [0,1]$, $t \in (0,1]$
- Boundary conditions:
  - $u(0,t) = t^{\alpha}$
  - $u(1,t) = -t^{\alpha}$
- Initial condition:
  - $u(x,0) = 0$

Baseline method: a classical physics-informed neural network (PINN), with product-integration logic for the weakly singular history integral in the fractional integro-differential setting.

This branch adds a TE-QPINN-inspired surrogate to test whether quantum-inspired feature embeddings can improve performance on this same problem setup.

## Quantum-Inspired Extension: TE-QPINN Surrogate

The TE-QPINN surrogate implementation includes:

- trainable embedding network
- input rescaling
- angle-style embedding
- sin/cos quantum-inspired features
- entanglement-inspired pairwise mixing
- expectation-style readout
- optional residual correction

Important scope note: this is a quantum-inspired surrogate in PyTorch. It is not a claim of quantum advantage without an actual simulator or hardware-backed quantum circuit path.

## Final 50x50 Single-Seed Benchmark

Matched setup: Adam-only, 12 epochs, seed 42, same fractional PI problem.

| Variant | Params | Runtime (s) | Final Loss | Final L2 | Final Linf |
| --- | ---: | ---: | ---: | ---: | ---: |
| Classical + PI | 2241 | 16.41 | 62.2103 | 1.00381 | 1.08578 |
| TE-QPINN Surrogate + PI | 2306 | 43.83 | 71.4435 | 0.88684 | 0.94618 |

Interpretation (single seed only): TE-QPINN improved L2 and Linf in this run, with closely matched parameter count, but it was slower. This was a promising signal, not a final conclusion.

## Multi-Seed Validation (5 Seeds, Locked Config)

Validated with `configs/benchmark_te_qpinn_multiseed.yaml` using seeds `[0,1,2,3,4]`.

| Variant | Mean Final L2 | Std Final L2 | Mean Final Linf | Std Final Linf | Mean Runtime (s) | Std Runtime (s) | Mean Final Loss | Std Final Loss | Params |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| Classical + PI | 0.916583 | 0.058399 | 1.061336 | 0.107421 | 14.7583 | 1.3977 | 43.197057 | 10.514407 | 2241 |
| TE-QPINN Surrogate + PI | 0.991693 | 0.080669 | 1.107482 | 0.087500 | 42.0679 | 2.0448 | 48.784985 | 9.389905 | 2306 |

Paired-seed win counts:
- TE-QPINN final L2 wins: `0 / 5`
- TE-QPINN final Linf wins: `3 / 5`

Interpretation (multi-seed): the earlier single-seed gain did not generalize overall. Classical + PI remains stronger on mean L2, mean Linf, runtime, and final loss. TE-QPINN still shows localized promise on Linf behavior (3/5 wins), and remains useful as an exploratory architecture pending further tuning and optimizer studies.

## Branch Navigation (Quantum-Readiness)

- Organized benchmark dossier: `docs/quantum-readiness/README.md`
- Journal-style experiment log: `docs/quantum-readiness/JOURNAL.md`
- Checked-in comparison plots: `docs/quantum-readiness/figures/`
- Generated run artifacts (local): `outputs/benchmarks/`
- Final benchmark interpretation: `docs/te_qpinn_benchmark_analysis.md`

## Quick Commands

Run tests:

```bash
pytest -q
```

Run TE-QPINN smoke benchmark:

```bash
python scripts/benchmark_te_qpinn.py --benchmark-config configs/benchmark_te_qpinn_smoke.yaml
```

Run TE-QPINN 50x50 benchmark:

```bash
python scripts/benchmark_te_qpinn.py --benchmark-config configs/benchmark_te_qpinn_50.yaml
```

Run TE-QPINN multiseed benchmark:

```bash
python scripts/benchmark_te_qpinn.py --benchmark-config configs/benchmark_te_qpinn_multiseed.yaml
```

Detailed benchmark analysis:
- `docs/te_qpinn_benchmark_analysis.md`

---

## Table of Contents
1. [At a Glance: Governing Problem](#at-a-glance-governing-problem)
2. [Quantum-Inspired Extension: TE-QPINN Surrogate](#quantum-inspired-extension-te-qpinn-surrogate)
3. [Final 50x50 Single-Seed Benchmark](#final-50x50-single-seed-benchmark)
4. [Multi-Seed Validation (5 Seeds, Locked Config)](#multi-seed-validation-5-seeds-locked-config)
5. [Quick Commands](#quick-commands)
6. [Problem Formulation](#problem-formulation)
7. [Results](#results)
8. [Mathematical Background](#mathematical-background)
9. [The L1 Discretization Scheme](#the-l1-discretization-scheme)
10. [Graded Mesh Construction](#graded-mesh-construction)
11. [Collocation Point Selection](#collocation-point-selection)
12. [Integral Term Approximation](#integral-term-approximation)
13. [Implementation Details](#implementation-details)
14. [Neural Network Architecture](#neural-network-architecture)
15. [Training Methodology](#training-methodology)
16. [Usage](#usage)

---

## Problem Formulation

We solve the **time-fractional integro-differential equation**:

$$D_t^{\alpha} u(x,t) - (x^2 + 1)\frac{\partial^2 u}{\partial x^2} + \int_0^t \sin(x)(t-s)^{-\beta}u(x,s)\,ds = f(x,t)$$

**Domain:** $x \in [0,1]$, $t \in (0,1]$

**Boundary Conditions (Non-homogeneous):**
- $u(0,t) = t^{\alpha}$
- $u(1,t) = -t^{\alpha}$

**Initial Condition:**
- $u(x,0) = 0$

**Parameters:**
- $0 < \alpha < 1$ (fractional derivative order)
- $0 < \beta < 1$ (integral kernel singularity)

**Exact Solution:** $u(x,t) = t^{\alpha}\cos(\pi x)$

> **Note on solution shape:** The solution is symmetric about $x = 0.5$ because $\cos(\pi x)$ over $[0,1]$ spans half a period: positive at $x=0$, zero at $x=0.5$, negative at $x=1$. This "mirror image" appearance is the natural shape of the cosine function, not a code artifact.

### Key Challenges
1. **Fractional derivative** requires full time history (non-local operator)
2. **Variable coefficient** $(x^2+1)$ in diffusion term
3. **Weakly singular integral** with kernel $(t-s)^{-\beta}$
4. **Non-homogeneous boundary conditions**

---

## Historical Reference

### Final Results (200×200 Grid with Product Integration)

**Epoch 20000 - Completed** ✓

| Metric | Value |
|--------|-------|
| **L2 Relative Error** | **1.52%** |
| **Linf Error** | 0.0606 |
| **Mean Error** | 0.0050 |
| **Training Time** | ~32 hours |

### Performance Comparison

| Setup | Grid | Method | L2 Error | Linf Error | Time |
|-------|------|--------|----------|-----------|------|
| **Baseline** | 50×50 | Midpoint | 0.54% | 0.0484 | ~2h |
| **Enhanced** | 200×200 | **Product Integration** | **1.52%** | **0.0606** | ~32h |

**Note:** The enhanced version solves a **16× larger problem** (200×200 vs 50×50 grid). The L2 error is higher because the finer grid requires more precision from the network. The critical advance is the **kernel-aware product integration**, which analytically integrates the singular kernel $(t-s)^{-\beta}$, eliminating singularity errors.

### Early Stopping Analysis

Analysis of all 40 checkpoints shows optimal stopping before epoch 20,000:

| Metric | Epoch 20,000 | Epoch 11,500 | Epoch 17,500 ⭐ |
|--------|------|------|------|
| **L2 Error** | **1.52%** | 2.80% | **1.60%** |
| **Linf Error** | 0.0606 | **0.0507** | **0.0512** |
| **Training Time** | 32 hrs | 18.4 hrs | 28 hrs |

**Recommendation:** Use **Epoch 17,500** for best balance—saves 12.5% training time with only 5% increase in L2 error and 1% in Linf error.

### Configuration (Current Branch: 200×200 Product Integration)

| Parameter | Value |
|-----------|-------|
| Spatial points ($N_x$) | 200 |
| Temporal points ($N_t$) | 200 |
| Collocation points | 100 |
| Epochs | 20,000 |
| Learning rate (peak) | 5e-4 |
| Warmup epochs | 1000 |
| Integral method | **Kernel-aware product integration** |
| α (fractional order) | 0.5 |
| β (integral singularity) | 0.5 |
| Mesh grading ($\beta_{mesh}$) | 2.0 |

### Output Files

- `outputs/checkpoints_integro_diff/` - 40 model checkpoints (every 500 epochs)
- `outputs/integro_diff_results/` - Visualization plots (solution, errors, slices)
- `outputs/integro_diff_points.csv` - Collocation point log
- `outputs/l1_discretization_points.csv` - L1 scheme point tracking

### Output Plots

All plots saved to `outputs/integro_diff_results/`:
- `solution_comparison.png` - Exact vs Predicted heatmaps + error
- `slices_fixed_t.png` - Solution slices at t = 0.1, 0.3, 0.5, 0.7, 0.9, 1.0
- `slices_fixed_x.png` - Solution slices at x = 0.1, 0.25, 0.4, 0.6, 0.75, 0.9
- `slices_late_time.png` - Late time slices (t = 0.90 to 0.99)
- `slices_late_x.png` - Late spatial slices (x = 0.90 to 0.99)
- `error_slices.png` - Error distribution at various times
- `3d_surface_plot.png` - 3D visualization
- `training_history.png` - Loss, L2, L∞ curves

---

## Mathematical Background

### 1. Caputo Fractional Derivative

The **Caputo fractional derivative** of order $\alpha \in (0,1)$ is defined as:

$$D_t^{\alpha} u(x,t) = \frac{1}{\Gamma(1-\alpha)} \int_0^t \frac{\partial u(x,s)}{\partial s} (t-s)^{-\alpha} ds$$

**Key properties:**
- Requires knowledge of $u$ at all previous times (memory effect)
- $D_t^{\alpha} c = 0$ for constants
- $D_t^{\alpha} t^{\gamma} = \frac{\Gamma(\gamma+1)}{\Gamma(\gamma+1-\alpha)} t^{\gamma-\alpha}$

### 2. Source Term Derivation

For exact solution $u(x,t) = t^{\alpha}\cos(\pi x)$, we compute each term:

**Term 1: Caputo derivative**
$$D_t^{\alpha}[t^{\alpha}\cos(\pi x)] = \frac{\Gamma(\alpha+1)}{\Gamma(1)}\cos(\pi x) = \Gamma(\alpha+1)\cos(\pi x)$$

**Term 2: Variable coefficient diffusion**
$$(x^2+1)\frac{\partial^2 u}{\partial x^2} = (x^2+1)(-\pi^2)t^{\alpha}\cos(\pi x) = -(x^2+1)\pi^2 t^{\alpha}\cos(\pi x)$$

**Term 3: Weakly singular integral**
$$\int_0^t \sin(x)(t-s)^{-\beta}s^{\alpha}\cos(\pi x) ds = \sin(x)\cos(\pi x) \cdot t^{\alpha+1-\beta} \cdot \frac{\Gamma(\alpha+1)\Gamma(1-\beta)}{\Gamma(\alpha+2-\beta)}$$

**Combined source term:**
$$f(x,t) = \cos(\pi x)\left[\Gamma(\alpha+1) + (x^2+1)\pi^2 t^{\alpha} + \sin(x) t^{\alpha+1-\beta} \frac{\Gamma(\alpha+1)\Gamma(1-\beta)}{\Gamma(\alpha+2-\beta)}\right]$$

---

## The L1 Discretization Scheme

### Why L1 Scheme?

The **L1 scheme** approximates the Caputo derivative on a discrete mesh. It's particularly effective for fractional PDEs because:
- Handles the singularity at $t=0$ 
- Achieves $O(h^{2-\alpha})$ accuracy on graded meshes
- Naturally incorporates the memory effect

### L1 Formula Derivation

Given temporal mesh $0 = t_0 < t_1 < \cdots < t_N = T$, let $\tau_k = t_k - t_{k-1}$.

The Caputo derivative at $t = t_n$ is approximated by:

$$D_t^{\alpha} u(x, t_n) \approx d_{n,1} u^n - d_{n,n} u^0 - \sum_{k=1}^{n-1} (d_{n,k} - d_{n,k+1}) u^{n-k}$$

where the **L1 coefficients** are:

$$d_{n,k} = \frac{(t_n - t_{n-k})^{1-\alpha} - (t_n - t_{n-k+1})^{1-\alpha}}{\Gamma(2-\alpha) \cdot \tau_{n-k+1}}$$

### What Points Are Used?

For a collocation point at time level $n$, the L1 scheme uses:

| Term | Points Used | Meaning |
|------|-------------|---------|
| $d_{n,1} u^n$ | $t_n$ (current) | Current solution value |
| $d_{n,n} u^0$ | $t_0 = 0$ (initial) | Initial condition |
| $(d_{n,k} - d_{n,k+1}) u^{n-k}$ | $t_1, t_2, \ldots, t_{n-1}$ | Full history |

**Example for $n=5$:**
- Current: $u(x, t_5)$
- History: $u(x, t_4), u(x, t_3), u(x, t_2), u(x, t_1)$
- Initial: $u(x, t_0) = 0$

This means computing $D_t^{\alpha} u$ at $t_5$ requires evaluating the neural network at **6 different time points**.

### Implementation Detail

In `src/physics_integro.py`, the L1 computation:

```python
def compute_fractional_derivative_l1(self, x, n_indices, u_current):
    for n in unique_n:
        coeffs = self.l1_coeffs.get_coefficients_for_n(n)
        
        # Current value: d_{n,1} * u^n (has gradient)
        frac_deriv = coeffs[1] * u_n
        
        # Initial value: d_{n,n} * u^0 = 0 (IC)
        
        # History: sum over k=1 to n-1
        for k in range(1, n):
            idx = n - k
            t_idx = t_nodes[idx]
            u_idx = model(x, t_idx)  # Evaluate at history point
            diff_coeff = coeffs[k] - coeffs[k + 1]
            frac_deriv -= diff_coeff * u_idx
```

---

## Graded Mesh Construction

### Why Graded Mesh?

The solution $u(x,t) = t^{\alpha}\cos(\pi x)$ has a **singularity at $t=0$** (infinite slope for $\alpha < 1$). A uniform mesh would give poor accuracy near $t=0$.

### Graded Mesh Formula

$$t_n = T \cdot \left(\frac{n}{N}\right)^{\beta_{mesh}}$$

where $\beta_{mesh} > 1$ concentrates points near $t=0$.

**With $N=100$ and $\beta_{mesh}=2.0$:**
```
t_0  = 0.0000
t_1  = 0.0001    (very small step)
t_2  = 0.0004
t_3  = 0.0009
...
t_10 = 0.0100
...
t_50 = 0.2500
...
t_100 = 1.0000
```

The mesh spacing grows as $\tau_n \approx O(n^{\beta-1})$, giving finer resolution where the solution varies most rapidly.

---

## Collocation Point Selection

### Grid Construction

1. **Spatial grid:** $x_i = \frac{i}{N_x}$ for $i = 0, 1, \ldots, N_x$
2. **Temporal grid:** $t_n$ from graded mesh for $n = 0, 1, \ldots, N_t$
3. **Interior grid:** $(x_i, t_n)$ for $i \in \{1, \ldots, N_x-1\}$, $n \in \{1, \ldots, N_t\}$

### Random Sampling

Each epoch, we randomly sample $N_{coll}$ points from the interior grid:
- Store $(x, t, n)$ where $n$ is the time index
- The time index $n$ determines which history points are needed for L1

### Point Tracking

Every 1000 epochs, we log:
1. **Collocation points:** Which $(x, t, n)$ triples were sampled
2. **L1 history points:** For each $n$, which temporal nodes are used
3. **L1 coefficients:** The weights $d_{n,k}$ applied to each history term

See `outputs/l1_discretization_points.csv` for detailed logs.

---

## Integral Term Approximation

### The Weakly Singular Integral

$$I(x,t) = \int_0^t \sin(x)(t-s)^{-\beta}u(x,s)\,ds$$

The kernel $(t-s)^{-\beta}$ is singular at $s=t$.

### Kernel-Aware Product Integration (New Approach)

**Motivation:** Standard quadrature (e.g., midpoint rule) evaluates the singular kernel numerically, leading to inaccuracy near the singularity. Instead, **integrate the singular kernel analytically** and approximate only the smooth part.

**Method:** On each sub-interval $[t_j, t_{j+1}]$, approximate $u(x,s)$ **linearly** and integrate the singular kernel **exactly**:

$$\int_{t_j}^{t_{j+1}} (t_n - s)^{-\beta} u(x,s)\, ds \approx w_j^L \cdot u(x, t_j) + w_j^R \cdot u(x, t_{j+1})$$

The **analytically computed weights** are:

$$w_j^L = \frac{1}{h_j}\left[\frac{a^{2-\beta} - b^{2-\beta}}{2-\beta} - b\cdot\frac{a^{1-\beta} - b^{1-\beta}}{1-\beta}\right]$$

$$w_j^R = \frac{1}{h_j}\left[a\cdot\frac{a^{1-\beta} - b^{1-\beta}}{1-\beta} - \frac{a^{2-\beta} - b^{2-\beta}}{2-\beta}\right]$$

where $a = t_n - t_j$, $b = t_n - t_{j+1}$, $h_j = a - b$.

**Advantages:**
- Singular kernel handled **analytically** (no numerical singularity error)
- Only smooth part $u$ is linearly interpolated (second-order accurate in smooth regions)
- Evaluates neural network at grid nodes (precomputable, efficient batching)
- Natural fit for graded meshes (clustering near $t=0$ handles singularity)

**Full integral approximation:**
$$I(x,t_n) \approx \sin(x) \sum_{j=0}^{n-1} \left[w_j^L \cdot u(x, t_j) + w_j^R \cdot u(x, t_{j+1})\right]$$

### Implementation

See `src/physics_integro.py`:
- `compute_integral_term()` — Product integration with analytical kernel weights
- `evaluate_integral_convergence()` — Tracks integral approximation error over training
- `IntegralConvergenceMonitor` — Logs convergence metrics

---

## Implementation Details

This section provides concrete code walkthroughs to demonstrate how the mathematical concepts are actually implemented.

### How Collocation Points Are Selected

The code builds and samples from a structured grid. Here's the exact process:

**Step 1: Create spatial and temporal grids**
```python
# Spatial: uniform grid on [0, 1]
self.x_grid = torch.linspace(0, 1, N_x)   # e.g., [0, 0.02, 0.04, ..., 1.0] for N_x=200

# Temporal: graded mesh t_n = (n/N)^β
self.t_grid = mesh.get_nodes()            # e.g., [0, 0.000025, 0.0001, ..., 1.0] for N_t=200, β=2
```

**Step 2: Build interior meshgrid (exclude boundaries)**
```python
x_interior = self.x_grid[1:-1]    # Skip x=0 and x=1 (boundary conditions)
t_interior = self.t_grid[1:]       # Skip t=0 (initial condition)

X, T = torch.meshgrid(x_interior, t_interior, indexing='ij')
n_values = torch.arange(1, N_t + 1)   # Time indices [1, 2, ..., N_t]

# Flatten to get all interior grid points
self.x_interior = X.flatten()      # Shape: (N_x-2) × N_t interior points
self.t_interior = T.flatten()
self.n_interior = N_grid.flatten()  # Corresponding time index for each point
```

**Step 3: Random sampling each epoch**
```python
def sample_collocation(self):
    idx = torch.randperm(self.total_interior)[:self.N_collocation]  # Random permutation
    return self.x_interior[idx], self.t_interior[idx], self.n_interior[idx]
```

**Proof from logged data (`outputs/integro_diff_points.csv`):**
```
epoch,point_idx,x,t,n
2000,0,0.7487,0.2601,102   ← t = (102/200)² = 0.2601 ✓
2000,1,0.2563,0.3364,116   ← t = (116/200)² = 0.3364 ✓
2000,2,0.2563,0.0992,63    ← t = (63/200)² = 0.0992 ✓
2000,5,0.3015,0.1892,87    ← Different collocation point same epoch
```

**Key insight:** The integer `n` (time index) is crucial because:
1. It determines which L1 coefficients $d_{n,k}$ to use
2. It tells us how many history terms to sum (k = 1 to n-1)

### How the Integral Term Is Computed

For a collocation point at time level $n$, we compute:
$$\int_0^{t_n} \sin(x)(t_n-s)^{-\beta} u(x,s)\, ds$$

**Implementation in `src/physics_integro.py` (Product Integration):**

```python
def compute_integral_term(self, x, t, n_indices):
    """
    Compute weakly singular integral: ∫₀ᵗ sin(x)(t-s)^{-β} u(x,s) ds
    Uses kernel-aware product integration on the graded mesh.
    
    On each [t_j, t_{j+1}], approximate u(x,s) linearly and integrate
    the singular kernel (t_n - s)^{-β} analytically.
    """
    for n in unique_n:
        n_val = n.item()
        mask = (n_indices == n_val)
        x_n = x[mask]           # All x-coordinates at this time level
        t_n = t_nodes[n_val]    # The target time t_n
        
        sin_x = torch.sin(x_n)  # Factor out sin(x)
        
        # Precompute u at all mesh nodes t_0, ..., t_n
        with torch.no_grad():
            u_at_nodes = [self.model(x_n, t_nodes[j]) for j in range(n_val + 1)]
        
        # Sum over all mesh intervals [t_j, t_{j+1}]
        integral_sum = torch.zeros(num_points)
        
        for j in range(n_val):
            a = (t_n - t_nodes[j]).item()      # t_n - t_j
            b = (t_n - t_nodes[j + 1]).item()  # t_n - t_{j+1}
            h_j = a - b                         # = t_{j+1} - t_j
            
            # Analytically integrated kernel moments
            moment_1 = (a**(1-β) - b**(1-β)) / (1-β)   # ∫ τ^{-β} dτ
            moment_2 = (a**(2-β) - b**(2-β)) / (2-β)   # ∫ τ^{1-β} dτ
            
            # Product integration weights
            w_left  = (moment_2 - b * moment_1) / h_j
            w_right = (a * moment_1 - moment_2) / h_j
            
            # Accumulate: w_L * u(t_j) + w_R * u(t_{j+1})
            integral_sum += w_left * u_at_nodes[j] + w_right * u_at_nodes[j + 1]
        
        result[mask] = sin_x * integral_sum
    
    return result
```

**Why `torch.no_grad()` for history terms?**

Only the current solution $u(x, t_n)$ needs gradients for backpropagation. History values $u(x, s)$ for $s < t_n$ are treated as fixed during each training step—this is the standard approach for time-stepping schemes in PINNs.

**Computational cost:** For a point at time level $n$, we make $n$ forward passes through the network (one per quadrature point). With $N_t = 200$ and 100 collocation points, this means ~10,000 forward passes per training iteration, explaining the longer runtime (~5-6 seconds/iteration).

---

## Neural Network Architecture

### Activation Function

The current trained model uses **Tanh** activation. The Mexican Hat wavelet activation ($\psi(x) = (1 - x^2) e^{-x^2/2}$) is also available in the codebase for future experiments.

### Network Structure

```
Input: (x, t) ∈ R²
  ↓
Linear(2 → 64) → Tanh
  ↓
Linear(64 → 64) → Tanh
  ↓
Linear(64 → 64) → Tanh
  ↓
Linear(64 → 64) → Tanh
  ↓
Linear(64 → 1)
  ↓
Output: u(x,t) ∈ R
```

Total parameters: 12,737

---

## Training Methodology

### Loss Function

$$\mathcal{L} = w_{PDE} \mathcal{L}_{PDE} + w_{BC} \mathcal{L}_{BC} + w_{IC} \mathcal{L}_{IC}$$

**PDE Loss (Interior):**
$$\mathcal{L}_{PDE} = \frac{1}{N_{coll}} \sum_{i=1}^{N_{coll}} |R(x_i, t_i)|^2$$

where $R = D_t^{\alpha}u - (x^2+1)u_{xx} + \int_0^t \sin(x)(t-s)^{-\beta}u\,ds - f$

**Boundary Loss:**
$$\mathcal{L}_{BC} = \frac{1}{N_{BC}} \sum \left[|u(0,t) - t^{\alpha}|^2 + |u(1,t) + t^{\alpha}|^2\right]$$

**Initial Loss:**
$$\mathcal{L}_{IC} = \frac{1}{N_{IC}} \sum |u(x,0)|^2$$

### Optimizer Configuration

- **Adam optimizer** with learning rate $5 \times 10^{-4}$
- **Cosine warmup scheduler:** 1000 epochs linear warmup, then cosine decay to $10^{-5}$
- **Gradient clipping:** max norm 1.0
- **Weights:** $w_{PDE}=1$, $w_{BC}=20$, $w_{IC}=20$

---

## Branch Comparison

### `integro-differential` (Baseline)
- Grid: 50×50 spatial/temporal
- Collocation: 50 points
- Epochs: 10,000
- Integral method: Midpoint rule
- Learning rate: 0.001
- Warmup: 500 epochs
- **L2 Error: 0.54%**
- Training time: ~2 hours

### `integro-differential-product-integration` (Enhanced)
- Grid: 200×200 spatial/temporal (8× larger)
- Collocation: 100 points
- Epochs: 20,000 (2× longer)
- Integral method: **Kernel-aware product integration** (analytically integrated kernel)
- Learning rate: 0.0005 (stabilized for larger problem)
- Warmup: 1000 epochs (more gradual)
- Status: **Completed** — L2 = 1.52%, Linf = 0.0606
- Training time: ~32 hours

**Key difference:** Product integration eliminates numerical error from the singular kernel by integrating it analytically, then only approximating the smooth solution linearly on each sub-interval.

---

## Usage

### Training

```bash
cd /path/to/PINNs

# Train from scratch
python scripts/train_integro_diff.py --config configs/integro_differential.yaml

# Resume from checkpoint
python scripts/train_integro_diff.py --config configs/integro_differential.yaml --resume

# Optional: stop early at a target epoch
python scripts/train_integro_diff.py --config configs/integro_differential.yaml --early-stop-epoch 17500
```

### Model Selection (Classical vs Quantum-Ready)

Model instantiation is now config-driven via `src/model_factory.py`.

In `network` config blocks, set:

```yaml
network:
  model_type: classical       # classical | quantum_ready | hybrid_quantum
  quantum:
    backend: classical_emulator
    n_qubits: 8
    n_layers: 3
    feature_scale: 1.0
    entanglement_strength: 0.15
    residual_connection: true
```

This keeps a single forward contract (`u = model(x, t)`) while allowing side-by-side experiments.

### Evaluation and Visualization

```bash
# Evaluate a checkpoint and save metrics/plots
python scripts/evaluate.py \
  --checkpoint outputs/checkpoints_integro_diff/final_model.pt \
  --config configs/integro_differential.yaml \
  --output-dir outputs/eval

# Generate standard visualization panels
python scripts/visualize.py \
  --checkpoint outputs/checkpoints_integro_diff/final_model.pt \
  --config configs/integro_differential.yaml \
  --output-dir outputs/figures
```

### Benchmarking Classical vs Quantum-Ready

Use the benchmark runner to compare matched-budget runs:

```bash
python scripts/benchmark_quantum_ready.py \
  --config configs/benchmark_quantum_ready.yaml \
  --models classical quantum_ready \
  --seeds 42 123 999 \
  --epochs 8 \
  --output-dir outputs/benchmarks/quantum_ready
```

Outputs include:
- `benchmark_summary.csv`
- `benchmark_results.json`
- `benchmark_aggregate.json`
- `convergence_seed_<seed>.png`

### Output Files

- `outputs/checkpoints_integro_diff/` - Model checkpoints
- `outputs/integro_diff_results/` - Visualization plots
- `outputs/integro_diff_points.csv` - Collocation point log
- `outputs/l1_discretization_points.csv` - **L1 scheme point tracking**

### L1 Point Log Format

The file `outputs/l1_discretization_points.csv` contains:

```
EPOCH 1000: L1 DISCRETIZATION POINTS
================================================================================
TIME INDEX n = 5 (12 collocation points at this time level)
Target time: t_5 = 0.00250000

L1 SCHEME FORMULA:
D_t^0.5 u(x, t_5) ≈ d_{n,1}·u^5 - d_{n,5}·u^0 - Σ_{k=1}^{4} (d_{n,k} - d_{n,k+1})·u^{5-k}

HISTORY POINTS USED:
k     Time Index      t_value         Coefficient          Role
----------------------------------------------------------------------
1     n=5             0.00250000      d_{n,1}=12.34567890  Current u^5
5     n=0             0.00000000      d_{n,5}=1.23456789   Initial u^0
1     n=4             0.00160000      (d-d)=2.34567890     History u^4
2     n=3             0.00090000      (d-d)=1.87654321     History u^3
3     n=2             0.00040000      (d-d)=1.54321098     History u^2
4     n=1             0.00010000      (d-d)=1.23456789     History u^1
```

---

## Quantum Readiness and QPINN Roadmap

This branch has been refactored to be **quantum-ready** while keeping training fully classical by default.
The current `quantum_ready` path uses a classical emulator block (`QuantumReadyPINN`) with the same PINN API.
No real quantum hardware/circuit backend is integrated yet; this is intentional to establish stable benchmarking infrastructure first.

### Why This Problem Is a Good QPINN Candidate

The target PDE combines three structures that are relevant for quantum-hybrid research:

1. **Strong nonlocality in time** from the Caputo derivative (full memory effect).
2. **Weakly singular Volterra kernel** in the integral term.
3. **Smooth low-dimensional input space** $(x,t)$, where variational quantum feature maps are feasible.

This is a stronger benchmark than toy ODE/PDE QPINN tasks and can support publishable claims if evaluated rigorously.

### Current Integration Points for a Future Quantum Model

The following software seams make QPINN integration straightforward:

1. `PINN.forward(x, t)` is already a clean model contract.
2. Residual code consumes only model outputs and autograd derivatives.
3. Training/evaluation/visualization scripts now construct models via `src/model_factory.py`.
4. Config-driven `network.model_type` supports `classical`, `quantum_ready`, and `hybrid_quantum` aliasing.

Recommended future model interface:

```python
u = model(x, t)
```

Keep this unchanged for compatibility with:

- L1 fractional derivative logic
- Product-integration integral logic
- existing checkpoint/evaluation tooling

### Practical QPINN Strategy (Recommended)

Use a **hybrid QPINN**, not a fully quantum network:

1. Classical encoder: maps $(x,t)$ to a compact latent.
2. Small variational quantum block: few qubits, shallow depth.
3. Classical readout head: scalar output $u(x,t)$.

Why hybrid first:

1. This project already has expensive history loops from L1 + integral terms.
2. Full quantum replacement would be too slow/noisy for meaningful ablations.
3. Hybrid keeps gradients and training stability manageable.

### Suggested Implementation Phases

#### Phase 1: Interface and Config Plumbing

Add config keys:

```yaml
network:
  model_type: classical   # classical | hybrid_quantum
  quantum:
   backend: pennylane
   n_qubits: 4
   n_layers: 2
   shots: null           # null for analytic simulator, integer for sampling
```

Add a model factory to instantiate either classical PINN or hybrid QPINN with identical forward signature.

#### Phase 2: Small-Scale Feasibility Study

Use reduced settings (for turnaround and fair debugging):

- `N_x=40`, `N_t=40`
- fewer collocation points
- short training budget

Goal: verify stable training and derivative correctness (especially $u_{xx}$).

#### Phase 3: Controlled Benchmarking

Run matched-budget experiments:

1. Equal wall-clock time.
2. Equal parameter count (approximately).
3. Equal seeds and evaluation grids.

Track:

- L2 relative error
- Linf error
- convergence speed
- training energy/runtime cost
- robustness across seeds

#### Phase 4: Full-Problem Transfer

Warm-start from classical checkpoints, then quantum fine-tune.
This is the most practical path for the current 200×200 setup.

### Expected Quantum Benefits (Realistic)

Potential advantages (if validated experimentally):

1. Better representation of oscillatory/nonlocal patterns at small model width.
2. Improved worst-case error in specific regimes (possible Linf gains).
3. Better parameter efficiency in low-dimensional PDE input settings.

Potential non-benefits / risks:

1. Slower training from repeated circuit evaluations inside history loops.
2. Gradient variance/noise with shot-based sampling.
3. No guaranteed accuracy gain over strong classical baselines.

The publication value will come from **honest ablations**, not from claiming universal quantum superiority.

### Publication Scope and Opportunities

This project can support multiple publication angles:

1. **Quantum-Enhanced Fractional PINNs**
  - Hybrid QPINN for time-fractional integro-differential equations.
  - Focus on nonlocal memory operators and singular kernels.

2. **Numerics + Quantum Hybridization**
  - Interaction between graded meshes/L1 discretization and quantum feature maps.
  - Error/stability analysis under matched compute budgets.

3. **Benchmarking Study**
  - Reproducible classical vs hybrid quantum PINN benchmark on a nontrivial fractional PDE.
  - Strong emphasis on fairness and reproducibility.

High-impact publication checklist:

1. At least 3-5 random seeds per setting.
2. Wall-clock, memory, and energy reporting.
3. Sensitivity studies: qubits, depth, shots, optimizer.
4. Error maps and late-time slice analysis (already natural in this repo).
5. Open-source reproducible pipeline with exact configs/checkpoints.

### What Was Improved Now (Pre-Quantum Refactor)

The codebase now includes practical upgrades that reduce risk before adding true quantum layers:

1. Training script supports configurable domain bounds and mesh grading consistently.
2. Early-stopping CLI support is implemented.
3. Evaluation and visualization scripts are repaired for current config/checkpoint formats.
4. Mesh/L1 test suite is updated to the live API.
5. Notebook analysis flow is aligned with current branch artifacts.
6. `src/model_factory.py` + `src/quantum_ready_model.py` provide a quantum-ready extension path without changing PDE residual code.
7. `scripts/benchmark_quantum_ready.py` + `configs/benchmark_quantum_ready.yaml` provide reproducible comparison tooling.

These changes are intentionally quantum-agnostic so the next quantum step can focus on model research rather than infrastructure cleanup.

---

## Project Structure

```
PINNs/
├── configs/
│   ├── integro_differential.yaml        # Main training configuration
│   └── benchmark_quantum_ready.yaml     # Benchmark configuration (classical vs quantum-ready)
├── src/
│   ├── model.py                         # PINN architecture (Tanh / Mexican Hat)
│   ├── model_factory.py                 # Config-based model construction
│   ├── quantum_ready_model.py           # Classical-emulated quantum-ready hybrid block
│   ├── mesh.py                          # Graded mesh + L1 coefficients
│   ├── physics_integro.py               # PDE residual computation
│   └── ...
├── scripts/
│   ├── train_integro_diff.py            # Training script
│   ├── benchmark_quantum_ready.py       # Matched-budget benchmark runner
│   ├── early_stopping_results.py        # Best-epoch analysis
│   ├── early_stopping_comparison.py     # Side-by-side epoch comparison
│   └── early_stopping_recommendation.py # Optimal stopping recommendation
├── outputs/
│   ├── checkpoints_integro_diff/        # 40 model checkpoints
│   ├── integro_diff_results/            # Plots
│   ├── integro_diff_points.csv          # Collocation points
│   └── l1_discretization_points.csv     # L1 history tracking
├── EARLY_STOPPING_GUIDE.md              # Early stopping analysis guide
└── README.md
```

---

## License

MIT
