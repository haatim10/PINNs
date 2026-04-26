# TE-QPINN Implementation Notes (Codex Ready)

Reference paper:
Trainable Embedding Quantum Physics-Informed Neural Networks (TE-QPINN)

---

## Core Idea

TE-QPINN introduces a **trainable embedding** for quantum feature maps.

Instead of fixed encodings, a small neural network learns how to map inputs into quantum features.

---

## Architecture Overview

TE-QPINN consists of:

1. Trainable embedding network (FNN)
2. Angle embedding (quantum-style encoding)
3. Variational / quantum-inspired layer
4. Readout layer
5. Standard PINN loss

---

## Step 1: Input Rescaling

Inputs must be normalized:

x → [-0.95, 0.95]

For multi-input:
(t, x) → normalized separately

---

## Step 2: Trainable Embedding

A small neural network learns:

phi(x) = [phi_1(x), ..., phi_n(x)]

Where:
- n = number of quantum features (qubits equivalent)
- phi_i(x) are learned scaling factors

---

## Step 3: Angle Embedding

Quantum-style encoding:

theta_i(x) = phi_i(x) * x

For multi-dimensional inputs:

theta = [
  phi_1(t,x)*t,
  phi_2(t,x)*x,
  phi_3(t,x)*t,
  phi_4(t,x)*x,
  ...
]

---

## Step 4: Quantum-Inspired Layer

Since real quantum hardware may not be available, implement a surrogate:

Features:
- sin(theta_i)
- cos(theta_i)

Add interactions:
- sin(theta_i) * cos(theta_j)

Then pass through:
- Linear / MLP layer

This mimics:
- rotation gates
- entanglement
- measurement expectation

---

## Step 5: Hybrid Model

Recommended structure:

Option A:
u(x) = quantum_branch(x)

Option B (better):
u(x) = classical_PINN(x) + λ * quantum_branch(x)

---

## Step 6: Loss Function

DO NOT CHANGE PINN LOSS

Use:

Loss = PDE residual + boundary + initial condition + product-integration (if present)

---

## Step 7: Training

Optimizers:
- Adam (default)
- optional LBFGS

Add:
- early stopping
- gradient clipping
- checkpointing

---

## Step 8: Benchmark Strategy

Compare:

1. Classical PINN
2. Product Integration PINN
3. Quantum-ready PINN
4. Quantum + Product Integration

---

## Grid sizes (FAST)

Use:

- 50×50 (quick)
- 100×100 (main)
- optional 150×150

Avoid 200×200 (too slow)

---

## Step 9: Metrics

Track:

- L2 error
- Linf error
- mean error
- runtime
- GPU memory
- parameter count
- best epoch

---

## Step 10: Plots (REQUIRED)

Generate:

- loss vs epoch
- L2 vs epoch
- Linf vs epoch
- runtime comparison
- error bar charts
- Pareto (error vs runtime)
- side-by-side predictions
- error heatmaps
- embedding visualization phi_i(x)

---

## Important Notes

- This is **quantum-inspired**, not true quantum advantage
- Do NOT claim quantum superiority unless proven
- Focus on:
  - representation power
  - convergence behavior
  - efficiency

---

## Codex Instructions

- Keep classical PINN unchanged
- Add quantum module cleanly
- Use config-driven model selection
- Keep code modular
- Do NOT rewrite entire repo
- Implement incrementally