# TE-QPINN Benchmark Report

- Benchmark config: `configs/benchmark_te_qpinn_memory_confirmatory_10seed.yaml`
- Seeds: 0, 1, 2, 3, 4, 42, 123, 999, 2024, 2025

## Summary

| Variant | Model Type | Runs | Runtime (s) | Params | Final Loss | Final L2 | Final Linf |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Classical + PI | classical | 10 | 14.7715 ± 1.1693 | 2241.0 | 44.0129 ± 12.6671 | 0.9228 ± 0.0770 | 1.0435 ± 0.1221 |
| Classical + PI + analytic memory | classical | 10 | 17.7416 ± 1.6117 | 2433.0 | 37.5752 ± 16.6954 | 0.7529 ± 0.1890 | 0.8108 ± 0.2179 |
| TE fixed residual 0.10 + PI (local-coordinate baseline) | te_qpinn_surrogate | 10 | 42.4348 ± 2.8408 | 2306.0 | 47.4863 ± 14.1418 | 0.9555 ± 0.0701 | 1.0528 ± 0.0920 |
| TE LayerNorm post_quantum + PI | te_qpinn_surrogate | 10 | 44.6977 ± 3.0572 | 2338.0 | 41.2059 ± 10.4170 | 0.9182 ± 0.0764 | 1.0254 ± 0.1170 |
| TE memory-aware analytic + PI | te_qpinn_surrogate | 10 | 45.0615 ± 3.4201 | 2498.0 | 26.6434 ± 13.3161 | 0.8095 ± 0.1764 | 1.0086 ± 0.2326 |

## Multi-Seed Win Counts

| Variant | L2 wins vs Classical | Linf wins vs Classical | L2 wins vs Full TE | Linf wins vs Full TE |
| --- | ---: | ---: | ---: | ---: |
| Classical + PI | n/a | n/a | 7 / 10 | 4 / 10 |
| Classical + PI + analytic memory | 9 / 10 | 9 / 10 | 8 / 10 | 9 / 10 |
| TE fixed residual 0.10 + PI (local-coordinate baseline) | 3 / 10 | 6 / 10 | n/a | n/a |
| TE LayerNorm post_quantum + PI | 4 / 10 | 6 / 10 | 10 / 10 | 5 / 10 |
| TE memory-aware analytic + PI | 7 / 10 | 6 / 10 | 6 / 10 | 4 / 10 |

## Memory Smoke Comparison

- Scope: multi-seed ([0, 1, 2, 3, 4, 42, 123, 999, 2024, 2025])
- Best non-memory TE reference: **TE LayerNorm post_quantum + PI** (L2=0.9182, Linf=1.0254, Loss=41.2059)
- Classical memory-feature control reference: **Classical + PI + analytic memory**

| Memory Variant | Final L2 | Final Linf | Final Loss | Runtime (s) | Params | ΔL2 vs Best Non-memory TE | ΔLinf vs Best Non-memory TE | ΔL2 vs Classical Memory | ΔLinf vs Classical Memory |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| TE memory-aware analytic + PI | 0.8095 | 1.0086 | 26.6434 | 45.0615 | 2498.0 | -0.1087 | -0.0168 | 0.0566 | 0.1979 |

> Note: this memory section is smoke-level and should not be interpreted as multi-seed evidence.

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
- `multiseed_stats.json`
- `multiseed_stats.md`
