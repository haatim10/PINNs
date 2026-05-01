# TE-QPINN Multi-Seed Stats

- Benchmark config: `configs/benchmark_te_qpinn_memory_confirmatory_10seed.yaml`
- Seeds: [0, 1, 2, 3, 4, 42, 123, 999, 2024, 2025]
- Reference classical run: `classical_pi`
- Reference full TE run: `te_fixed_pi`

## Aggregate Metrics

| Variant | Final L2 (mean±std) | Final Linf (mean±std) | Final Loss (mean±std) | Runtime (s, mean±std) | Params |
| --- | ---: | ---: | ---: | ---: | ---: |
| Classical + PI | 0.922830 ± 0.076989 | 1.043477 ± 0.122118 | 44.012908 ± 12.667057 | 14.7715 ± 1.1693 | 2241.0 |
| Classical + PI + analytic memory | 0.752910 ± 0.188977 | 0.810768 ± 0.217877 | 37.575163 ± 16.695402 | 17.7416 ± 1.6117 | 2433.0 |
| TE fixed residual 0.10 + PI (local-coordinate baseline) | 0.955529 ± 0.070070 | 1.052766 ± 0.092046 | 47.486260 ± 14.141845 | 42.4348 ± 2.8408 | 2306.0 |
| TE LayerNorm post_quantum + PI | 0.918204 ± 0.076402 | 1.025404 ± 0.117049 | 41.205865 ± 10.416957 | 44.6977 ± 3.0572 | 2338.0 |
| TE memory-aware analytic + PI | 0.809505 ± 0.176401 | 1.008622 ± 0.232580 | 26.643364 ± 13.316055 | 45.0615 ± 3.4201 | 2498.0 |

## Win Counts

| Variant | L2 wins vs Classical | Linf wins vs Classical | L2 wins vs Full TE | Linf wins vs Full TE |
| --- | ---: | ---: | ---: | ---: |
| Classical + PI | n/a | n/a | 7 / 10 | 4 / 10 |
| Classical + PI + analytic memory | 9 / 10 | 9 / 10 | 8 / 10 | 9 / 10 |
| TE fixed residual 0.10 + PI (local-coordinate baseline) | 3 / 10 | 6 / 10 | n/a | n/a |
| TE LayerNorm post_quantum + PI | 4 / 10 | 6 / 10 | 10 / 10 | 5 / 10 |
| TE memory-aware analytic + PI | 7 / 10 | 6 / 10 | 6 / 10 | 4 / 10 |
