# Quantum-Readiness Benchmark Report

- Config: `configs/benchmark_quantum_ready_harder.yaml`
- Seeds: 42, 123, 999
- Models: classical PINN, quantum-ready PINN
- Device request: `cuda`
- Epoch override: `20`
- Early stopping: patience `None` on `l2` with min delta `0.0001`

## Summary

| Model | Runs | Runtime (s) | Final L2 | Final Linf | Peak GPU MB |
| --- | ---: | ---: | ---: | ---: | ---: |
| classical PINN | 3 | 95.6251 ± 2.0647 | 1.1096 ± 0.0839 | 1.3778 ± 0.2098 | 37.2559 |
| quantum-ready PINN | 3 | 101.2677 ± 0.8837 | 2.5526 ± 2.3408 | 2.2807 ± 1.3685 | 26.6680 |

## Ratios

- Runtime ratio (quantum-ready / classical): 1.0590
- Final L2 ratio (quantum-ready / classical): 2.3004
- Final Linf ratio (quantum-ready / classical): 1.6553

## Pairwise Wins

- Runtime wins: 0/3
- L2 wins: 1/3
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
