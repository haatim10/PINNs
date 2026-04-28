# TE-QPINN Benchmark Report

- Benchmark config: `configs/benchmark_te_qpinn_layernorm_multiseed.yaml`
- Seeds: 0, 1, 2, 3, 4

## Summary

| Variant | Model Type | Runs | Runtime (s) | Params | Final Loss | Final L2 | Final Linf |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Classical + PI | classical | 5 | 18.2714 ± 1.5318 | 2241.0 | 43.1971 ± 10.5144 | 0.9166 ± 0.0584 | 1.0613 ± 0.1074 |
| TE fixed residual 0.10 + PI | te_qpinn_surrogate | 5 | 51.0464 ± 3.1593 | 2306.0 | 48.7850 ± 9.3899 | 0.9917 ± 0.0807 | 1.1075 ± 0.0875 |
| TE LayerNorm post_quantum + PI | te_qpinn_surrogate | 5 | 53.9656 ± 3.7063 | 2338.0 | 43.5986 ± 8.9194 | 0.9622 ± 0.0850 | 1.1045 ± 0.1168 |

## Ablation Comparison

| Variant | L2 vs Full TE | Linf vs Full TE | Loss vs Full TE | Runtime vs Full TE | L2 vs Classical | Linf vs Classical | Loss vs Classical | Runtime vs Classical |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Classical + PI | improves | improves | improves | improves | neutral | neutral | neutral | neutral |
| TE fixed residual 0.10 + PI | neutral | neutral | neutral | neutral | worsens | worsens | worsens | worsens |
| TE LayerNorm post_quantum + PI | improves | improves | improves | worsens | worsens | worsens | worsens | worsens |

## Multi-Seed Win Counts

| Variant | L2 wins vs Classical | Linf wins vs Classical | L2 wins vs Full TE | Linf wins vs Full TE |
| --- | ---: | ---: | ---: | ---: |
| Classical + PI | n/a | n/a | 5 / 5 | 2 / 5 |
| TE fixed residual 0.10 + PI | 0 / 5 | 3 / 5 | n/a | n/a |
| TE LayerNorm post_quantum + PI | 1 / 5 | 3 / 5 | 5 / 5 | 2 / 5 |

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
