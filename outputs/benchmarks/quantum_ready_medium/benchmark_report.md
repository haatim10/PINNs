# Quantum-Readiness Benchmark Report

- Config: `configs/benchmark_quantum_ready_medium.yaml`
- Seeds: 42, 123, 999
- Models: classical PINN, quantum-ready PINN
- Device request: `cuda`
- Epoch override: `20`
- Early stopping: patience `None` on `l2` with min delta `0.0001`

## Summary

| Model | Runs | Runtime (s) | Final L2 | Final Linf | Peak GPU MB |
| --- | ---: | ---: | ---: | ---: | ---: |
| classical PINN | 3 | 43.6595 ± 1.4115 | 0.8193 ± 0.0219 | 0.9940 ± 0.0499 | 25.4316 |
| quantum-ready PINN | 3 | 84.2197 ± 0.5642 | 2.5431 ± 3.1266 | 2.0376 ± 1.8494 | 24.3066 |

## Ratios

- Runtime ratio (quantum-ready / classical): 1.9290
- Final L2 ratio (quantum-ready / classical): 3.1042
- Final Linf ratio (quantum-ready / classical): 2.0499

## Pairwise Wins

- Runtime wins: 0/3
- L2 wins: 2/3
- Linf wins: 1/3

## Plots

- [Summary panels](summary_panels.png)
- [Seed-wise ratios](ratio_panel.png)
- [Convergence seed 42](convergence_seed_42.png)
- [Convergence seed 123](convergence_seed_123.png)
- [Convergence seed 999](convergence_seed_999.png)

## Interpretation

The quantum-ready path is still a classical emulator, so these results should be read as an architectural comparison rather than a quantum hardware claim.
This benchmark is useful when the oscillatory or multiscale structure in the solution family makes the feature-mixing block competitive.

## Raw Artifacts

- `benchmark_results.json`
- `benchmark_summary.csv`
- `benchmark_aggregate.json`
