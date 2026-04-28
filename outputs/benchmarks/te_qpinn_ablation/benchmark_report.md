# TE-QPINN Benchmark Report

- Benchmark config: `configs/benchmark_te_qpinn_ablation.yaml`
- Seeds: 42

## Summary

| Variant | Model Type | Runs | Runtime (s) | Params | Final Loss | Final L2 | Final Linf |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Classical + PI | classical | 1 | 17.2412 ± 0.0000 | 2241.0 | 62.2103 ± 0.0000 | 1.0038 ± 0.0000 | 1.0858 ± 0.0000 |
| Full TE-QPINN + PI | te_qpinn_surrogate | 1 | 46.3441 ± 0.0000 | 2306.0 | 71.4435 ± 0.0000 | 0.8868 ± 0.0000 | 0.9462 ± 0.0000 |
| TE no residual + PI | te_qpinn_surrogate | 1 | 41.7829 ± 0.0000 | 2273.0 | 49.1515 ± 0.0000 | 0.8756 ± 0.0000 | 1.1938 ± 0.0000 |
| TE residual scale 0.05 + PI | te_qpinn_surrogate | 1 | 48.4825 ± 0.0000 | 2306.0 | 72.2925 ± 0.0000 | 0.9166 ± 0.0000 | 0.9853 ± 0.0000 |
| TE residual scale 0.20 + PI | te_qpinn_surrogate | 1 | 44.3508 ± 0.0000 | 2306.0 | 70.0524 ± 0.0000 | 0.8275 ± 0.0000 | 0.8692 ± 0.0000 |
| TE variational hidden 16 + PI | te_qpinn_surrogate | 1 | 44.6118 ± 0.0000 | 1794.0 | 59.6877 ± 0.0000 | 1.0394 ± 0.0000 | 1.1220 ± 0.0000 |
| TE variational hidden 64 + PI | te_qpinn_surrogate | 1 | 44.5367 ± 0.0000 | 3330.0 | 57.8231 ± 0.0000 | 1.0006 ± 0.0000 | 1.1887 ± 0.0000 |
| TE small embedding + PI | te_qpinn_surrogate | 1 | 41.6095 ± 0.0000 | 1502.0 | 51.6313 ± 0.0000 | 1.0041 ± 0.0000 | 1.1238 ± 0.0000 |
| TE large embedding + PI | te_qpinn_surrogate | 1 | 43.9399 ± 0.0000 | 3062.0 | 53.5456 ± 0.0000 | 0.9242 ± 0.0000 | 1.0567 ± 0.0000 |

## Ablation Comparison

| Variant | L2 vs Full TE | Linf vs Full TE | Loss vs Full TE | Runtime vs Full TE | L2 vs Classical | Linf vs Classical | Loss vs Classical | Runtime vs Classical |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Classical + PI | worsens | worsens | improves | improves | neutral | neutral | neutral | neutral |
| Full TE-QPINN + PI | neutral | neutral | neutral | neutral | improves | improves | worsens | worsens |
| TE no residual + PI | improves | worsens | improves | improves | improves | worsens | improves | worsens |
| TE residual scale 0.05 + PI | worsens | worsens | worsens | worsens | improves | improves | worsens | worsens |
| TE residual scale 0.20 + PI | improves | improves | improves | improves | improves | improves | worsens | worsens |
| TE variational hidden 16 + PI | worsens | worsens | improves | improves | worsens | worsens | improves | worsens |
| TE variational hidden 64 + PI | worsens | worsens | improves | improves | improves | worsens | improves | worsens |
| TE small embedding + PI | worsens | worsens | improves | improves | worsens | worsens | improves | worsens |
| TE large embedding + PI | worsens | worsens | improves | improves | improves | improves | improves | worsens |

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
- `ablation_stats.json`
- `ablation_stats.md`
