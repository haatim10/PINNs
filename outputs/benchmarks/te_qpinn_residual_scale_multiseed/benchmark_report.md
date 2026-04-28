# TE-QPINN Benchmark Report

- Benchmark config: `configs/benchmark_te_qpinn_residual_scale_multiseed.yaml`
- Seeds: 0, 1, 2, 3, 4

## Summary

| Variant | Model Type | Runs | Runtime (s) | Params | Final Loss | Final L2 | Final Linf |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Classical + PI | classical | 5 | 14.4189 ± 1.4946 | 2241.0 | 43.1971 ± 10.5144 | 0.9166 ± 0.0584 | 1.0613 ± 0.1074 |
| Full TE-QPINN + PI (residual 0.10) | te_qpinn_surrogate | 5 | 39.8334 ± 1.1720 | 2306.0 | 48.7850 ± 9.3899 | 0.9917 ± 0.0807 | 1.1075 ± 0.0875 |
| TE-QPINN + PI (residual 0.20) | te_qpinn_surrogate | 5 | 39.6660 ± 1.2766 | 2306.0 | 50.1568 ± 12.4579 | 1.0225 ± 0.1514 | 1.2102 ± 0.1482 |

## Ablation Comparison

| Variant | L2 vs Full TE | Linf vs Full TE | Loss vs Full TE | Runtime vs Full TE | L2 vs Classical | Linf vs Classical | Loss vs Classical | Runtime vs Classical |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Classical + PI | improves | improves | improves | improves | neutral | neutral | neutral | neutral |
| Full TE-QPINN + PI (residual 0.10) | neutral | neutral | neutral | neutral | worsens | worsens | worsens | worsens |
| TE-QPINN + PI (residual 0.20) | worsens | worsens | worsens | improves | worsens | worsens | worsens | worsens |

## Multi-Seed Win Counts

| Variant | L2 wins vs Classical | Linf wins vs Classical | L2 wins vs Full TE | Linf wins vs Full TE |
| --- | ---: | ---: | ---: | ---: |
| Classical + PI | n/a | n/a | 5 / 5 | 2 / 5 |
| Full TE-QPINN + PI (residual 0.10) | 0 / 5 | 3 / 5 | n/a | n/a |
| TE-QPINN + PI (residual 0.20) | 1 / 5 | 0 / 5 | 1 / 5 | 0 / 5 |

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
- `multiseed_stats.json`
- `multiseed_stats.md`
