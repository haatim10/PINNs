# Exact PennyLane TE-QPINN Notes (Phase 9B Milestone A)

## Purpose

This phase adds a new optional model family that is closer to the uploaded TE-QPINN paper design:

- `model_type: te_qpinn_pennylane`
- aliases:
  - `exact_te_qpinn_pennylane`
  - `exact_pqc_te_qpinn`

The existing TE surrogate model remains unchanged and fully supported.

## What This Exact PQC Model Implements

Implemented in `src/exact_te_qpinn_pennylane.py` as `ExactTEQPINNPennyLane`:

1. Input scaling (default target range `[-1, 1]`)
2. Classical trainable embedding FNN:
   - input: `(x, t)`
   - output: `phi_i(x, t)` for each qubit
3. Angle construction:
   - cycled coordinates across qubits (`x, t, x, t, ...`)
   - `angle_i = phi_i * cycled_coordinate_i`
4. PennyLane quantum circuit:
   - `qml.RY(angle_i)` embedding
   - variational layers with `RX/RY/RZ` on each wire
   - CNOT entanglement (`chain` or optional `ring`)
5. Expectation-style readout:
   - `readout_type: z_sum` (per-wire Z expectations -> linear head)
   - `readout_type: z_tensor` (tensor-product Z expectation -> linear head)

The model uses:

- `qml.device("default.qubit", wires=num_qubits)`
- Torch interface
- `diff_method="backprop"`

## How It Differs from the Existing Surrogate

Existing surrogate (`te_qpinn_surrogate`) is a classical differentiable approximation of quantum-style operations.

The exact PQC model executes an actual parameterized quantum circuit in PennyLane simulator (still simulator-based, not hardware quantum).

## Paper Alignment Status

### Aligned

- FNN embedding network for trainable angles
- Angle-style quantum embedding
- Parameterized quantum ansatz with single-qubit rotations
- Entanglement via CNOT pattern
- Expectation-value readout
- Hybrid classical + quantum trainable path

### Adapted

- Same benchmark PDE framework from this repo (fractional integro-differential problem)
- Batch handling via per-sample QNode loop for stability
- Small smoke-budget training for Milestone A
- Configurable readout (`z_sum` / `z_tensor`)

### Not Yet Included in Milestone A

- Large-scale multi-seed evaluation
- Hardware backend experiments
- Memory-aware exact PQC extension
- Full optimizer sensitivity for exact PQC (Adam+LBFGS)

## Dependency Behavior

PennyLane is optional for the whole repo but required for exact PQC model usage.

- If PennyLane is missing:
  - existing models continue to work
  - exact PQC tests skip or raise clear `ImportError` on instantiation

## Derivative and Autograd Status (Milestone A target)

Milestone A validates:

- forward pass
- batching via sample loop
- gradient flow to:
  - embedding FNN parameters
  - quantum variational parameters (`theta`)
- autograd for `u_x` and `u_xx` on tiny test batches

If second derivatives become unstable/too slow in a future environment, the fallback policy is:

- optional finite-difference derivative fallback for smoke diagnostics only
- disabled by default
- clearly documented and not used for final claims without explicit justification

## Smoke Benchmark Scope

Added smoke plan:

- `configs/benchmark_te_qpinn_exact_pqc_smoke.yaml`

Compares:

1. Classical + PI (tiny)
2. Exact TE-QPINN PennyLane + PI (tiny)

Output dirs:

- `outputs/benchmarks/te_qpinn_exact_pqc_smoke/`
- `outputs/plots/te_qpinn_exact_pqc_smoke/`

Milestone A is intentionally tiny and diagnostic; it is not a final performance claim benchmark.

## Milestone A Execution Notes

- `u_x` and `u_xx` autograd checks passed in unit tests on tiny batches.
- Gradients were verified to flow to both:
  - embedding FNN parameters
  - quantum variational parameters (`theta`)
- Smoke training used the existing full fractional/product-integration residual path.
- For practical runtime, smoke benchmark field-level plotting was disabled in
  `configs/benchmark_te_qpinn_exact_pqc_smoke.yaml` via:
  - `field_eval_enabled: false`

This keeps Milestone A focused on model correctness and trainability, while avoiding extremely expensive residual heatmap post-processing for the exact PQC variant.

## Milestone A Feasibility Result

Milestone A should be interpreted as a feasibility and bottleneck study.

What worked technically:

- Exact PennyLane PQC forward path works in the existing PINN pipeline.
- Autograd through the exact PQC path works for both first and second spatial derivatives (`u_x`, `u_xx`).
- Gradients reach both:
  - classical embedding FNN parameters
  - quantum variational parameters (`theta`)
- Tiny smoke benchmark completed end-to-end with recorded metrics.

Tiny smoke benchmark outcome (`seed=42`, tiny budget):

| Variant | Params | Runtime (s) | Final Loss | Final L2 | Final Linf |
| --- | ---: | ---: | ---: | ---: | ---: |
| Classical + PI (tiny) | 337 | 1.6793 | 23.8083 | 0.930595 | 0.968080 |
| Exact TE-QPINN PennyLane + PI (tiny) | 251 | 1348.6111 | 206.3259 | 3.918995 | 2.718204 |

Key bottlenecks observed:

- Sample-wise QNode batch handling is very slow.
- Higher-order autograd through the quantum circuit is expensive.
- Fractional/product-integration residual further increases runtime cost.
- Field/residual heatmap evaluation is currently too expensive for exact PQC smoke, so `field_eval_enabled: false` was used in the smoke benchmark config.

Interpretation:

- The exact PQC model is technically integrated and differentiable.
- In this naive tiny setup it is currently computationally impractical and less accurate than the classical tiny baseline.
- This phase is **not** a performance win; it is a feasibility confirmation plus bottleneck identification step.
