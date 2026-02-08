# PI-fMI: Physics-Informed Neural Networks for Fractional Integro-Differential Equations

This branch implements a PINN solver for **time-fractional integro-differential equations** with variable coefficients and weakly singular Volterra integrals.

---

## Table of Contents
1. [Problem Formulation](#problem-formulation)
2. [Results](#results)
3. [Mathematical Background](#mathematical-background)
4. [The L1 Discretization Scheme](#the-l1-discretization-scheme)
5. [Graded Mesh Construction](#graded-mesh-construction)
6. [Collocation Point Selection](#collocation-point-selection)
7. [Integral Term Approximation](#integral-term-approximation)
8. [Implementation Details](#implementation-details)
9. [Neural Network Architecture](#neural-network-architecture)
10. [Training Methodology](#training-methodology)
11. [Usage](#usage)

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

## Results

### Final Accuracy

| Metric | Value |
|--------|-------|
| **L2 Relative Error** | **0.54%** |
| **L∞ Error** | 4.84% |
| **Training Time** | ~2 hours |

### Configuration

| Parameter | Value |
|-----------|-------|
| Spatial points ($N_x$) | 50 |
| Temporal points ($N_t$) | 50 |
| Collocation points | 50 |
| Epochs | 10,000 |
| α (fractional order) | 0.5 |
| β (integral singularity) | 0.5 |
| Mesh grading ($\beta_{mesh}$) | 2.0 |

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

### Quadrature on Graded Mesh

We use composite quadrature on the graded mesh intervals:

$$I(x,t_n) \approx \sin(x) \sum_{j=0}^{n-1} \int_{t_j}^{t_{j+1}} (t_n-s)^{-\beta} u(x,s) ds$$

For each interval, use midpoint rule:
$$\int_{t_j}^{t_{j+1}} (t_n-s)^{-\beta} u(x,s) ds \approx (t_n - t_{mid})^{-\beta} \cdot u(x, t_{mid}) \cdot h_j$$

where $t_{mid} = (t_j + t_{j+1})/2$ and $h_j = t_{j+1} - t_j$.

---

## Implementation Details

This section provides concrete code walkthroughs to demonstrate how the mathematical concepts are actually implemented.

### How Collocation Points Are Selected

The code builds and samples from a structured grid. Here's the exact process:

**Step 1: Create spatial and temporal grids**
```python
# Spatial: uniform grid on [0, 1]
self.x_grid = torch.linspace(0, 1, N_x)   # e.g., [0, 0.02, 0.04, ..., 1.0] for N_x=50

# Temporal: graded mesh t_n = (n/N)^β
self.t_grid = mesh.get_nodes()            # e.g., [0, 0.0004, 0.0016, ..., 1.0] for N_t=50, β=2
```

**Step 2: Build interior meshgrid (exclude boundaries)**
```python
x_interior = self.x_grid[1:-1]    # Skip x=0 and x=1 (boundary conditions)
t_interior = self.t_grid[1:]       # Skip t=0 (initial condition)

X, T = torch.meshgrid(x_interior, t_interior, indexing='ij')
n_values = torch.arange(1, N_t + 1)   # Time indices [1, 2, ..., N_t]

# Flatten to get all interior grid points
self.x_interior = X.flatten()      # Shape: (48 × 50) = 2400 points
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
1,0,0.7755,0.0196,7      ← x from spatial grid, t = (7/50)² = 0.0196 ✓
1,1,0.1633,0.2500,25     ← t = (25/50)² = 0.25 ✓
1,2,0.8980,0.0016,2      ← t = (2/50)² = 0.0016 ✓
1000,0,0.5306,0.3136,28  ← Different random sample at epoch 1000
```

**Key insight:** The integer `n` (time index) is crucial because:
1. It determines which L1 coefficients $d_{n,k}$ to use
2. It tells us how many history terms to sum (k = 1 to n-1)

### How the Integral Term Is Computed

For a collocation point at time level $n$, we compute:
$$\int_0^{t_n} \sin(x)(t_n-s)^{-\beta} u(x,s)\, ds$$

**Implementation in `src/physics_integro.py`:**

```python
def compute_integral_term(self, x, t, n_indices):
    """
    Compute weakly singular integral: ∫₀ᵗ sin(x)(t-s)^{-β} u(x,s) ds
    Uses composite midpoint rule on graded mesh.
    """
    for n in unique_n:
        n_val = n.item()
        mask = (n_indices == n_val)
        x_n = x[mask]           # All x-coordinates at this time level
        t_n = t_nodes[n_val]    # The target time t_n
        
        sin_x = torch.sin(x_n)  # Factor out sin(x)
        
        # Sum over all mesh intervals [t_j, t_{j+1}] for j = 0, ..., n-1
        integral_sum = torch.zeros(num_points)
        
        for j in range(n_val):
            t_j = t_nodes[j]
            t_j_plus_1 = t_nodes[j + 1]
            h_j = t_j_plus_1 - t_j                    # Interval width
            t_mid = (t_j + t_j_plus_1) / 2            # Midpoint
            
            # Kernel: (t_n - t_mid)^{-β}
            kernel = (t_n - t_mid) ** (-self.beta)
            
            # Evaluate neural network at midpoint (no gradient needed for history)
            with torch.no_grad():
                u_mid = self.model(x_n, t_mid)
            
            # Accumulate: kernel × u × width
            integral_sum += kernel * u_mid * h_j
        
        result[mask] = sin_x * integral_sum
    
    return result
```

**Why `torch.no_grad()` for history terms?**

Only the current solution $u(x, t_n)$ needs gradients for backpropagation. History values $u(x, s)$ for $s < t_n$ are treated as fixed during each training step—this is the standard approach for time-stepping schemes in PINNs.

**Computational cost:** For a point at time level $n$, we make $n$ forward passes through the network (one per quadrature point). With $N_t = 50$ and 50 collocation points, this means ~1250 forward passes per training iteration, explaining the ~1 second/iteration runtime.

---

## Neural Network Architecture

### Mexican Hat Wavelet Activation

$$\psi(x) = (1 - x^2) e^{-x^2/2}$$

This wavelet-based activation provides:
- Localized response (good for capturing solution structure)
- Non-monotonic shape (better than tanh/ReLU for oscillatory solutions)
- Bounded output (numerical stability)

### Network Structure

```
Input: (x, t) ∈ R²
  ↓
Linear(2 → 64) → Mexican Hat
  ↓
Linear(64 → 64) → Mexican Hat
  ↓
Linear(64 → 64) → Mexican Hat
  ↓
Linear(64 → 64) → Mexican Hat
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

- **Adam optimizer** with learning rate $10^{-3}$
- **Cosine warmup scheduler:** 500 epochs linear warmup, then cosine decay to $10^{-5}$
- **Gradient clipping:** max norm 1.0
- **Weights:** $w_{PDE}=1$, $w_{BC}=20$, $w_{IC}=20$

---

## Usage

### Training

```bash
cd /workspace/f20220519/PINNs

# Train from scratch
python scripts/train_integro_diff.py --config configs/integro_differential.yaml

# Resume from checkpoint
python scripts/train_integro_diff.py --config configs/integro_differential.yaml --resume
```

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

## Project Structure

```
PINNs/
├── configs/
│   └── integro_differential.yaml    # Training configuration
├── src/
│   ├── model.py                     # PINN architecture + Mexican Hat
│   ├── mesh.py                      # Graded mesh + L1 coefficients
│   ├── physics_integro.py           # PDE residual computation
│   └── ...
├── scripts/
│   └── train_integro_diff.py        # Training script
├── outputs/
│   ├── checkpoints_integro_diff/    # Model checkpoints
│   ├── integro_diff_results/        # Plots
│   ├── integro_diff_points.csv      # Collocation points
│   └── l1_discretization_points.csv # L1 history tracking
└── README.md
```

---

## License

MIT
