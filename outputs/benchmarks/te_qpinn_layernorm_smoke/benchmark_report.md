# TE-QPINN Benchmark Report

- Benchmark config: `configs/benchmark_te_qpinn_layernorm_smoke.yaml`
- Seeds: 42

## Summary

| Variant | Model Type | Runs | Runtime (s) | Params | Final Loss | Final L2 | Final Linf |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Classical + PI | classical | 1 | 15.3493 ± 0.0000 | 2241.0 | 62.2103 ± 0.0000 | 1.0038 ± 0.0000 | 1.0858 ± 0.0000 |
| TE fixed residual 0.10 + PI | te_qpinn_surrogate | 1 | 38.9796 ± 0.0000 | 2306.0 | 71.4435 ± 0.0000 | 0.8868 ± 0.0000 | 0.9462 ± 0.0000 |
| TE LayerNorm post_quantum + PI | te_qpinn_surrogate | 1 | 40.9595 ± 0.0000 | 2338.0 | 54.0574 ± 0.0000 | 0.8239 ± 0.0000 | 0.9576 ± 0.0000 |
| TE LayerNorm post_entanglement + PI | te_qpinn_surrogate | 1 | 40.8774 ± 0.0000 | 2352.0 | 75.8186 ± 0.0000 | 0.9350 ± 0.0000 | 0.9624 ± 0.0000 |

## Ablation Comparison

| Variant | L2 vs Full TE | Linf vs Full TE | Loss vs Full TE | Runtime vs Full TE | L2 vs Classical | Linf vs Classical | Loss vs Classical | Runtime vs Classical |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Classical + PI | worsens | worsens | improves | improves | neutral | neutral | neutral | neutral |
| TE fixed residual 0.10 + PI | neutral | neutral | neutral | neutral | improves | improves | worsens | worsens |
| TE LayerNorm post_quantum + PI | improves | worsens | improves | worsens | improves | improves | improves | worsens |
| TE LayerNorm post_entanglement + PI | worsens | worsens | worsens | worsens | improves | improves | worsens | worsens |

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
