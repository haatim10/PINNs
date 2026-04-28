# TE-QPINN Benchmark Report

- Benchmark config: `configs/benchmark_te_qpinn_memory_multiseed.yaml`
- Seeds: 0, 1, 2, 3, 4

## Summary

| Variant | Model Type | Runs | Runtime (s) | Params | Final Loss | Final L2 | Final Linf |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Classical + PI | classical | 5 | 14.8572 ± 1.8966 | 2241.0 | 43.1971 ± 10.5144 | 0.9166 ± 0.0584 | 1.0613 ± 0.1074 |
| Classical + PI + analytic memory | classical | 5 | 17.0129 ± 0.9201 | 2433.0 | 37.4756 ± 16.9827 | 0.7619 ± 0.2447 | 0.8483 ± 0.3010 |
| TE fixed residual 0.10 + PI (local-coordinate baseline) | te_qpinn_surrogate | 5 | 42.8147 ± 3.1725 | 2306.0 | 48.7850 ± 9.3899 | 0.9917 ± 0.0807 | 1.1075 ± 0.0875 |
| TE LayerNorm post_quantum + PI | te_qpinn_surrogate | 5 | 44.0333 ± 3.2133 | 2338.0 | 43.5986 ± 8.9194 | 0.9622 ± 0.0850 | 1.1045 ± 0.1168 |
| TE memory-aware analytic + PI | te_qpinn_surrogate | 5 | 44.0522 ± 3.2268 | 2498.0 | 35.7055 ± 8.5926 | 0.8892 ± 0.1879 | 1.0659 ± 0.2690 |
| TE memory-aware analytic + LayerNorm post_quantum + PI | te_qpinn_surrogate | 5 | 46.3956 ± 3.3134 | 2530.0 | 49.0706 ± 43.5803 | 1.0497 ± 0.5060 | 1.2826 ± 0.3747 |

## Multi-Seed Win Counts

| Variant | L2 wins vs Classical | Linf wins vs Classical | L2 wins vs Full TE | Linf wins vs Full TE |
| --- | ---: | ---: | ---: | ---: |
| Classical + PI | n/a | n/a | 5 / 5 | 2 / 5 |
| Classical + PI + analytic memory | 4 / 5 | 4 / 5 | 4 / 5 | 4 / 5 |
| TE fixed residual 0.10 + PI (local-coordinate baseline) | 0 / 5 | 3 / 5 | n/a | n/a |
| TE LayerNorm post_quantum + PI | 1 / 5 | 3 / 5 | 5 / 5 | 2 / 5 |
| TE memory-aware analytic + PI | 2 / 5 | 2 / 5 | 2 / 5 | 2 / 5 |
| TE memory-aware analytic + LayerNorm post_quantum + PI | 3 / 5 | 2 / 5 | 4 / 5 | 2 / 5 |

## Memory Smoke Comparison

- Scope: multi-seed ([0, 1, 2, 3, 4])
- Best non-memory TE reference: **TE LayerNorm post_quantum + PI** (L2=0.9622, Linf=1.1045, Loss=43.5986)
- Classical memory-feature control reference: **Classical + PI + analytic memory**

| Memory Variant | Final L2 | Final Linf | Final Loss | Runtime (s) | Params | ΔL2 vs Best Non-memory TE | ΔLinf vs Best Non-memory TE | ΔL2 vs Classical Memory | ΔLinf vs Classical Memory |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| TE memory-aware analytic + PI | 0.8892 | 1.0659 | 35.7055 | 44.0522 | 2498.0 | -0.0730 | -0.0386 | 0.1272 | 0.2176 |
| TE memory-aware analytic + LayerNorm post_quantum + PI | 1.0497 | 1.2826 | 49.0706 | 46.3956 | 2530.0 | 0.0875 | 0.1780 | 0.2878 | 0.4343 |

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
