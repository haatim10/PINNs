# TE-QPINN Multi-Seed Stats

- Benchmark config: `configs/benchmark_te_qpinn_memory_alpha07_beta03_5seed.yaml`
- Seeds: [0, 1, 2, 3, 4]
- Reference classical run: `classical_pi`
- Reference full TE run: `te_fixed_pi`

## Aggregate Metrics

| Variant | Final L2 (mean±std) | Final Linf (mean±std) | Final Loss (mean±std) | Runtime (s, mean±std) | Params |
| --- | ---: | ---: | ---: | ---: | ---: |
| Classical + PI (alpha=0.7, beta=0.3) | 0.935119 ± 0.053208 | 1.055873 ± 0.121808 | 40.927904 ± 12.203974 | 15.7726 ± 2.0796 | 2241.0 |
| Classical + PI + analytic memory (alpha=0.7, beta=0.3) | 0.885903 ± 0.183472 | 0.955070 ± 0.236414 | 38.573208 ± 13.828171 | 18.1548 ± 0.6639 | 2433.0 |
| TE fixed residual 0.10 + PI (alpha=0.7, beta=0.3) | 1.025442 ± 0.114842 | 1.121506 ± 0.134984 | 102.662733 ± 143.959989 | 43.5231 ± 1.3878 | 2306.0 |
| TE LayerNorm post_quantum + PI (alpha=0.7, beta=0.3) | 0.985235 ± 0.113020 | 1.070939 ± 0.107036 | 57.117036 ± 46.812099 | 45.9477 ± 1.1748 | 2338.0 |
| TE memory-aware analytic + PI (alpha=0.7, beta=0.3) | 1.087911 ± 0.167052 | 1.218205 ± 0.227703 | 231.438177 ± 441.600714 | 46.4354 ± 1.3434 | 2498.0 |

## Win Counts

| Variant | L2 wins vs Classical | Linf wins vs Classical | L2 wins vs Full TE | Linf wins vs Full TE |
| --- | ---: | ---: | ---: | ---: |
| Classical + PI (alpha=0.7, beta=0.3) | n/a | n/a | 4 / 5 | 3 / 5 |
| Classical + PI + analytic memory (alpha=0.7, beta=0.3) | 3 / 5 | 4 / 5 | 3 / 5 | 4 / 5 |
| TE fixed residual 0.10 + PI (alpha=0.7, beta=0.3) | 1 / 5 | 2 / 5 | n/a | n/a |
| TE LayerNorm post_quantum + PI (alpha=0.7, beta=0.3) | 2 / 5 | 2 / 5 | 5 / 5 | 3 / 5 |
| TE memory-aware analytic + PI (alpha=0.7, beta=0.3) | 1 / 5 | 1 / 5 | 2 / 5 | 2 / 5 |
