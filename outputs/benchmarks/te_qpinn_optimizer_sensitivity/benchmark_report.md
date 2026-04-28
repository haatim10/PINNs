# TE-QPINN Benchmark Report

- Benchmark config: `configs/benchmark_te_qpinn_optimizer_sensitivity.yaml`
- Seeds: 42

## Summary

| Variant | Model Type | Runs | Runtime (s) | Params | Final Loss | Final L2 | Final Linf |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Classical + PI (Adam) | classical | 1 | 15.7019 ± 0.0000 | 2241.0 | 62.2103 ± 0.0000 | 1.0038 ± 0.0000 | 1.0858 ± 0.0000 |
| Classical + PI (Adam+LBFGS, extended) | classical | 1 | 549.7721 ± 0.0000 | 2241.0 | 0.1611 ± 0.0000 | 0.0851 ± 0.0000 | 0.2604 ± 0.0000 |
| TE fixed residual 0.10 + PI (Adam) | te_qpinn_surrogate | 1 | 40.6095 ± 0.0000 | 2306.0 | 71.4435 ± 0.0000 | 0.8868 ± 0.0000 | 0.9462 ± 0.0000 |
| TE fixed residual 0.10 + PI (Adam+LBFGS, extended) | te_qpinn_surrogate | 1 | 1589.7988 ± 0.0000 | 2306.0 | 0.0817 ± 0.0000 | 0.0590 ± 0.0000 | 0.2091 ± 0.0000 |
| TE LayerNorm post_quantum + PI (Adam) | te_qpinn_surrogate | 1 | 41.8441 ± 0.0000 | 2338.0 | 54.0574 ± 0.0000 | 0.8239 ± 0.0000 | 0.9576 ± 0.0000 |
| TE LayerNorm post_quantum + PI (Adam+LBFGS, extended) | te_qpinn_surrogate | 1 | 1697.6655 ± 0.0000 | 2338.0 | 0.1299 ± 0.0000 | 0.0354 ± 0.0000 | 0.1074 ± 0.0000 |

## Ablation Comparison

| Variant | L2 vs Full TE | Linf vs Full TE | Loss vs Full TE | Runtime vs Full TE | L2 vs Classical | Linf vs Classical | Loss vs Classical | Runtime vs Classical |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Classical + PI (Adam) | worsens | worsens | improves | improves | neutral | neutral | neutral | neutral |
| Classical + PI (Adam+LBFGS, extended) | improves | improves | improves | worsens | improves | improves | improves | worsens |
| TE fixed residual 0.10 + PI (Adam) | neutral | neutral | neutral | neutral | improves | improves | worsens | worsens |
| TE fixed residual 0.10 + PI (Adam+LBFGS, extended) | improves | improves | improves | worsens | improves | improves | improves | worsens |
| TE LayerNorm post_quantum + PI (Adam) | improves | worsens | improves | worsens | improves | improves | improves | worsens |
| TE LayerNorm post_quantum + PI (Adam+LBFGS, extended) | improves | improves | improves | worsens | improves | improves | improves | worsens |

## Optimizer Sensitivity (Adam vs Adam+LBFGS)

- Note: Adam+LBFGS includes an extended optimization budget (Adam stage plus LBFGS fine-tuning) and is not an equal-budget comparison with Adam-only runs.

| Variant | Adam L2 | Adam+LBFGS L2 | Adam Linf | Adam+LBFGS Linf | Adam Loss | Adam+LBFGS Loss | Adam Runtime (s) | Adam+LBFGS Runtime (s) |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| Classical + PI | 1.0038 | 0.0851 | 1.0858 | 0.2604 | 62.2103 | 0.1611 | 15.7019 | 549.7721 |
| TE fixed residual 0.10 + PI | 0.8868 | 0.0590 | 0.9462 | 0.2091 | 71.4435 | 0.0817 | 40.6095 | 1589.7988 |
| TE LayerNorm post_quantum + PI | 0.8239 | 0.0354 | 0.9576 | 0.1074 | 54.0574 | 0.1299 | 41.8441 | 1697.6655 |

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
- `optimizer_sensitivity_stats.json`
- `optimizer_sensitivity_stats.md`
