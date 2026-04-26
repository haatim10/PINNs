# TE-QPINN Benchmark Report

- Benchmark config: `configs/benchmark_te_qpinn_smoke.yaml`
- Seeds: 42

## Summary

| Variant | Model Type | Runs | Runtime (s) | Params | Final L2 | Final Linf |
| --- | --- | ---: | ---: | ---: | ---: | ---: |
| Classical + PI | classical | 1 | 3.3818 ± 0.0000 | 1297.0 | 1.0280 ± 0.0000 | 1.1961 ± 0.0000 |
| TE-QPINN Surrogate + PI | te_qpinn_surrogate | 1 | 3.2493 ± 0.0000 | 1102.0 | 1.7684 ± 0.0000 | 1.9079 ± 0.0000 |

## Artifacts

- `summary.csv`
- `summary.json`
- `summary_panels.png`
- `convergence_seed_<seed>.png`
