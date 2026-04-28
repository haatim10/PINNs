# TE-QPINN Benchmark Report

- Benchmark config: `configs/benchmark_te_qpinn_memory_smoke.yaml`
- Seeds: 42

## Summary

| Variant | Model Type | Runs | Runtime (s) | Params | Final Loss | Final L2 | Final Linf |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Classical + PI | classical | 1 | 17.3504 ± 0.0000 | 2241.0 | 62.2103 ± 0.0000 | 1.0038 ± 0.0000 | 1.0858 ± 0.0000 |
| Classical + PI + analytic memory | classical | 1 | 18.3400 ± 0.0000 | 2433.0 | 62.3695 ± 0.0000 | 0.9249 ± 0.0000 | 0.8390 ± 0.0000 |
| TE fixed residual 0.10 + PI | te_qpinn_surrogate | 1 | 44.6783 ± 0.0000 | 2306.0 | 71.4435 ± 0.0000 | 0.8868 ± 0.0000 | 0.9462 ± 0.0000 |
| TE LayerNorm post_quantum + PI | te_qpinn_surrogate | 1 | 47.2960 ± 0.0000 | 2338.0 | 54.0574 ± 0.0000 | 0.8239 ± 0.0000 | 0.9576 ± 0.0000 |
| TE memory-aware analytic + PI | te_qpinn_surrogate | 1 | 47.3314 ± 0.0000 | 2498.0 | 37.0430 ± 0.0000 | 0.6815 ± 0.0000 | 0.9554 ± 0.0000 |
| TE memory-aware analytic + LayerNorm post_quantum + PI | te_qpinn_surrogate | 1 | 49.8322 ± 0.0000 | 2530.0 | 45.7305 ± 0.0000 | 0.6990 ± 0.0000 | 0.8109 ± 0.0000 |

## Memory Smoke Comparison

- Scope: single-seed ([42])
- Best non-memory TE reference: **TE LayerNorm post_quantum + PI** (L2=0.8239, Linf=0.9576, Loss=54.0574)
- Classical memory-feature control reference: **Classical + PI + analytic memory**

| Memory Variant | Final L2 | Final Linf | Final Loss | Runtime (s) | Params | ΔL2 vs Best Non-memory TE | ΔLinf vs Best Non-memory TE | ΔL2 vs Classical Memory | ΔLinf vs Classical Memory |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| TE memory-aware analytic + PI | 0.6815 | 0.9554 | 37.0430 | 47.3314 | 2498.0 | -0.1424 | -0.0022 | -0.2434 | 0.1164 |
| TE memory-aware analytic + LayerNorm post_quantum + PI | 0.6990 | 0.8109 | 45.7305 | 49.8322 | 2530.0 | -0.1249 | -0.1467 | -0.2259 | -0.0281 |

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
