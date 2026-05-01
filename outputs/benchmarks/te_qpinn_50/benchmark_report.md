# TE-QPINN Benchmark Report

- Benchmark config: `configs/benchmark_te_qpinn_50.yaml`
- Seeds: 42

## Summary

| Variant | Model Type | Runs | Runtime (s) | Params | Final L2 | Final Linf |
| --- | --- | ---: | ---: | ---: | ---: | ---: |
| Classical + PI | classical | 1 | 18.2386 ± 0.0000 | 2241.0 | 1.0038 ± 0.0000 | 1.0858 ± 0.0000 |
| TE-QPINN Surrogate + PI | te_qpinn_surrogate | 1 | 43.5479 ± 0.0000 | 2306.0 | 0.8868 ± 0.0000 | 0.9462 ± 0.0000 |

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
