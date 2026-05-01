# Quantum-Readiness Benchmark Report

- Config: `configs/benchmark_quantum_ready_harder_tuned.yaml`
- Seeds: 42
- Models: classical PINN, quantum-ready PINN
- Device request: `cuda`
- Epoch override: `20`
- Early stopping: patience `None` on `l2` with min delta `0.0001`

## Summary

| Model | Runs | Runtime (s) | Final L2 | Final Linf | Peak GPU MB |
| --- | ---: | ---: | ---: | ---: | ---: |
| classical PINN | 1 | 90.9617 ± 0.0000 | 1.1721 ± 0.0000 | 1.3487 ± 0.0000 | 37.2559 |
| quantum-ready PINN | 1 | 72.2919 ± 0.0000 | 1.0281 ± 0.0000 | 1.3140 ± 0.0000 | 26.6602 |

## Ratios

- Runtime ratio (quantum-ready / classical): 0.7948
- Final L2 ratio (quantum-ready / classical): 0.8772
- Final Linf ratio (quantum-ready / classical): 0.9743

## Pairwise Wins

- Runtime wins: 1/1
- L2 wins: 1/1
- Linf wins: 1/1

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
