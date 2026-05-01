# TE-QPINN Benchmark Report

- Benchmark config: `configs/benchmark_te_qpinn_exact_pqc_smoke.yaml`
- Seeds: 42

## Summary

| Variant | Model Type | Runs | Runtime (s) | Params | Final Loss | Final L2 | Final Linf |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Classical + PI (tiny) | classical | 1 | 1.6793 ± 0.0000 | 337.0 | 23.8083 ± 0.0000 | 0.9306 ± 0.0000 | 0.9681 ± 0.0000 |
| Exact TE-QPINN PennyLane + PI (tiny) | te_qpinn_pennylane | 1 | 1348.6111 ± 0.0000 | 251.0 | 206.3259 ± 0.0000 | 3.9190 ± 0.0000 | 2.7182 ± 0.0000 |

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
