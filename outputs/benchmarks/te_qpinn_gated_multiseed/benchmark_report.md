# TE-QPINN Benchmark Report

- Benchmark config: `configs/benchmark_te_qpinn_gated_multiseed.yaml`
- Seeds: 0, 1, 2, 3, 4

## Summary

| Variant | Model Type | Runs | Runtime (s) | Params | Final Loss | Final L2 | Final Linf |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Classical + PI | classical | 5 | 14.3691 ± 0.9529 | 2241.0 | 43.1971 ± 10.5144 | 0.9166 ± 0.0584 | 1.0613 ± 0.1074 |
| TE-QPINN Fixed Residual 0.10 + PI | te_qpinn_surrogate | 5 | 40.0081 ± 0.7726 | 2306.0 | 48.7850 ± 9.3899 | 0.9917 ± 0.0807 | 1.1075 ± 0.0875 |
| TE-QPINN Gated Residual + PI | te_qpinn_surrogate | 5 | 43.0674 ± 1.5391 | 2307.0 | 50.6770 ± 16.6908 | 1.2005 ± 0.3493 | 1.5917 ± 0.4856 |

## Ablation Comparison

| Variant | L2 vs Full TE | Linf vs Full TE | Loss vs Full TE | Runtime vs Full TE | L2 vs Classical | Linf vs Classical | Loss vs Classical | Runtime vs Classical |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Classical + PI | improves | improves | improves | improves | neutral | neutral | neutral | neutral |
| TE-QPINN Fixed Residual 0.10 + PI | neutral | neutral | neutral | neutral | worsens | worsens | worsens | worsens |
| TE-QPINN Gated Residual + PI | worsens | worsens | worsens | worsens | worsens | worsens | worsens | worsens |

## Multi-Seed Win Counts

| Variant | L2 wins vs Classical | Linf wins vs Classical | L2 wins vs Full TE | Linf wins vs Full TE |
| --- | ---: | ---: | ---: | ---: |
| Classical + PI | n/a | n/a | 5 / 5 | 2 / 5 |
| TE-QPINN Fixed Residual 0.10 + PI | 0 / 5 | 3 / 5 | n/a | n/a |
| TE-QPINN Gated Residual + PI | 1 / 5 | 0 / 5 | 1 / 5 | 0 / 5 |

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
