# TE-QPINN Benchmark Report

- Benchmark config: `configs/benchmark_te_qpinn_multiseed.yaml`
- Seeds: 0, 1, 2, 3, 4

## Summary

| Variant | Model Type | Runs | Runtime (s) | Params | Final L2 | Final Linf |
| --- | --- | ---: | ---: | ---: | ---: | ---: |
| Classical + PI | classical | 5 | 14.7583 ± 1.3977 | 2241.0 | 0.9166 ± 0.0584 | 1.0613 ± 0.1074 |
| TE-QPINN Surrogate + PI | te_qpinn_surrogate | 5 | 42.0679 ± 2.0448 | 2306.0 | 0.9917 ± 0.0807 | 1.1075 ± 0.0875 |

## Artifacts

- `summary.csv`
- `summary.json`
- `summary_panels.png`
- `convergence_seed_<seed>.png`
- `seed_<seed>_<run>_u_pred_heatmap.png`
- `seed_<seed>_<run>_u_exact_heatmap.png`
- `seed_<seed>_<run>_abs_error_heatmap.png`
- `seed_<seed>_<run>_pde_residual_heatmap.png` (if residual evaluation succeeds)
- `seed_<seed>_<run>_line_slices.png`
- `error_heatmaps_seed_<seed>_classical_vs_te_qpinn.png`
