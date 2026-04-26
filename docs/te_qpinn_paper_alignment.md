# TE-QPINN Paper Alignment Note

## 1. Overview

This document clarifies how the current repository implementation aligns with the core ideas of the TE-QPINN paper ("Trainable Embedding Quantum Physics-Informed Neural Networks for Solving Nonlinear PDEs") and where it intentionally diverges.

The goal in this branch is practical research integration: preserve the existing classical PINN + product-integration workflow for the time-fractional integro-differential equation, while adding a TE-QPINN-inspired surrogate model that can be trained and benchmarked in the same pipeline.

This note is intended to prevent overclaiming and to make the current scope explicit for thesis/report use.

## 2. What is implemented from the TE-QPINN idea

The current surrogate captures the main structural concepts behind TE-QPINN-style hybrid modeling:

- trainable embedding network
  - a small classical network learns embedding factors from input coordinates.
- input rescaling
  - coordinates are rescaled to a bounded range (configured around `[-0.95, 0.95]`) before embedding.
- angle-style embedding
  - learned embedding outputs are multiplied with cycled coordinate components to produce angle-like latent variables.
- sin/cos feature representation
  - sinusoidal transforms are used to emulate angle-encoded feature behavior.
- entanglement-inspired pairwise feature mixing
  - pairwise interaction terms are included as an entanglement-like surrogate mechanism.
- expectation-style readout
  - latent features are mapped through an expectation-like head before final scalar output.
- optional residual connection
  - a configurable residual branch can be added to stabilize or enrich function approximation.

At training level, this surrogate is integrated into the same PINN residual framework used by classical models, including PDE, IC/BC, and product-integration terms.

## 3. What is simplified or adapted

To keep the implementation practical and reproducible in a PyTorch PINN workflow, several design choices are adapted:

- quantum circuit behavior is approximated with differentiable surrogate feature transforms (sin/cos and pairwise terms), not explicit gate operations.
- variational layers are modeled as classical neural transformations over angle-derived features.
- readout is implemented as a classical expectation-style head rather than direct measurement of quantum observables.
- optimization is handled with standard deep-learning optimizers (Adam, optional LBFGS fine-tuning path) inside the existing PINN training loop.

These adaptations allow direct comparison against classical and product-integration baselines under matched data generation and training protocols.

## 4. What is not implemented yet

The current implementation is **not**:

- a true quantum circuit simulator,
- a hardware-executable quantum model,
- a full reproduction of the original TE-QPINN paper,
- validated across multiple seeds/problems yet.

Additional unimplemented items include:

- explicit parameterized quantum circuit construction and simulator-backed gradient flow,
- hardware backend execution pathway,
- full paper-level benchmark matrix across multiple PDE families and larger statistical evaluation.

## 5. Why the surrogate is useful for this fractional integro-differential PINN problem

This repository targets a difficult time-fractional integro-differential equation with nonlocal temporal memory and weakly singular history terms. The surrogate is useful here because:

- it introduces richer nonlinear feature geometry without changing physics constraints,
- it can be swapped into the existing PINN residual formulation with minimal workflow disruption,
- it supports controlled fairness studies (same collocation strategy, same seed policy, same PDE setup),
- it provides a quantum-ready experimentation path before committing to simulator/hardware dependencies.

In short, it is a low-friction bridge between classical PINN baselines and future quantum-native variants.

## 6. Current limitations

Current limitations should be treated explicitly in any report:

- evidence is currently limited to early benchmark settings; broader statistical validation remains pending.
- single-run improvements should not be interpreted as universal superiority.
- wall-clock runtime can increase despite accuracy gains, depending on surrogate configuration.
- conclusions are currently about a quantum-inspired surrogate, not quantum advantage.

## 7. Future work toward closer paper reproduction

Recommended next steps for stronger alignment with TE-QPINN literature:

1. Add an optional simulator-backed quantum path (e.g., PennyLane/Qiskit) with the same trainable embedding interface.
2. Preserve apples-to-apples fairness constraints across model families (parameter budget, collocation points, seeds, epochs).
3. Run multi-seed evaluation and report uncertainty (mean/std or confidence intervals).
4. Expand to additional PDE families (oscillatory, multiscale, sharper-gradient regimes) under consistent protocols.
5. Compare Adam-only and Adam+LBFGS under matched optimization budgets.
6. Separate claims clearly into:
   - surrogate performance observations,
   - simulator-backed quantum observations (if added),
   - hardware-backed observations (if added).

This staged path keeps the work technically honest while moving toward a closer and more defensible TE-QPINN reproduction.
