# TE-QPINN Benchmark Report

- Benchmark config: `configs/benchmark_te_qpinn_memory_alpha07_beta03_5seed.yaml`
- Seeds: 0, 1, 2, 3, 4

## Summary

| Variant | Model Type | Runs | Runtime (s) | Params | Final Loss | Final L2 | Final Linf |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Classical + PI (alpha=0.7, beta=0.3) | classical | 5 | 15.7726 ± 2.0796 | 2241.0 | 40.9279 ± 12.2040 | 0.9351 ± 0.0532 | 1.0559 ± 0.1218 |
| Classical + PI + analytic memory (alpha=0.7, beta=0.3) | classical | 5 | 18.1548 ± 0.6639 | 2433.0 | 38.5732 ± 13.8282 | 0.8859 ± 0.1835 | 0.9551 ± 0.2364 |
| TE fixed residual 0.10 + PI (alpha=0.7, beta=0.3) | te_qpinn_surrogate | 5 | 43.5231 ± 1.3878 | 2306.0 | 102.6627 ± 143.9600 | 1.0254 ± 0.1148 | 1.1215 ± 0.1350 |
| TE LayerNorm post_quantum + PI (alpha=0.7, beta=0.3) | te_qpinn_surrogate | 5 | 45.9477 ± 1.1748 | 2338.0 | 57.1170 ± 46.8121 | 0.9852 ± 0.1130 | 1.0709 ± 0.1070 |
| TE memory-aware analytic + PI (alpha=0.7, beta=0.3) | te_qpinn_surrogate | 5 | 46.4354 ± 1.3434 | 2498.0 | 231.4382 ± 441.6007 | 1.0879 ± 0.1671 | 1.2182 ± 0.2277 |

## Multi-Seed Win Counts

| Variant | L2 wins vs Classical | Linf wins vs Classical | L2 wins vs Full TE | Linf wins vs Full TE |
| --- | ---: | ---: | ---: | ---: |
| Classical + PI (alpha=0.7, beta=0.3) | n/a | n/a | 4 / 5 | 3 / 5 |
| Classical + PI + analytic memory (alpha=0.7, beta=0.3) | 3 / 5 | 4 / 5 | 3 / 5 | 4 / 5 |
| TE fixed residual 0.10 + PI (alpha=0.7, beta=0.3) | 1 / 5 | 2 / 5 | n/a | n/a |
| TE LayerNorm post_quantum + PI (alpha=0.7, beta=0.3) | 2 / 5 | 2 / 5 | 5 / 5 | 3 / 5 |
| TE memory-aware analytic + PI (alpha=0.7, beta=0.3) | 1 / 5 | 1 / 5 | 2 / 5 | 2 / 5 |

## Memory Smoke Comparison

- Scope: multi-seed ([0, 1, 2, 3, 4])
- Best non-memory TE reference: **TE LayerNorm post_quantum + PI (alpha=0.7, beta=0.3)** (L2=0.9852, Linf=1.0709, Loss=57.1170)
- Classical memory-feature control reference: **Classical + PI + analytic memory (alpha=0.7, beta=0.3)**

| Memory Variant | Final L2 | Final Linf | Final Loss | Runtime (s) | Params | ΔL2 vs Best Non-memory TE | ΔLinf vs Best Non-memory TE | ΔL2 vs Classical Memory | ΔLinf vs Classical Memory |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| TE memory-aware analytic + PI (alpha=0.7, beta=0.3) | 1.0879 | 1.2182 | 231.4382 | 46.4354 | 2498.0 | 0.1027 | 0.1473 | 0.2020 | 0.2631 |

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
