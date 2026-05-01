# Quantum-Readiness Benchmark Report

- Config: `configs/benchmark_quantum_ready.yaml`
- Seeds: 42
- Models: classical PINN, quantum-ready PINN
- Device request: `cuda`
- Epoch override: `2`
- Early stopping: patience `None` on `l2` with min delta `0.0001`

## Summary

| Model | Runs | Runtime (s) | Final L2 | Final Linf | Peak GPU MB |
| --- | ---: | ---: | ---: | ---: | ---: |
| classical PINN | 1 | 10.0199 ± 0.0000 | 1.2192 ± 0.0000 | 1.4614 ± 0.0000 | 22.6050 |
| quantum-ready PINN | 1 | 2.0055 ± 0.0000 | 2.5094 ± 0.0000 | 2.1212 ± 0.0000 | 21.5542 |

## Ratios

- Runtime ratio (quantum-ready / classical): 0.2002
- Final L2 ratio (quantum-ready / classical): 2.0582
- Final Linf ratio (quantum-ready / classical): 1.4515

## Pairwise Wins

- Runtime wins: 1/1
- L2 wins: 0/1
- Linf wins: 0/1

## Plots

- [Summary panels](summary_panels.png)
- [Seed-wise ratios](ratio_panel.png)
- [Convergence seed 42](convergence_seed_42.png)

## Interpretation

The quantum-ready path is still a classical emulator, so these results should be read as an architectural comparison rather than a quantum hardware claim.
This benchmark is useful when the oscillatory or multiscale structure in the solution family makes the feature-mixing block competitive.

## Raw Artifacts

- `benchmark_results.json`
- `benchmark_summary.csv`
- `benchmark_aggregate.json`
