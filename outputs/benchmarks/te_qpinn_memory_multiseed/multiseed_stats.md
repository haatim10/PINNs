# TE-QPINN Multi-Seed Stats

- Benchmark config: `configs/benchmark_te_qpinn_memory_multiseed.yaml`
- Seeds: [0, 1, 2, 3, 4]
- Reference classical run: `classical_pi`
- Reference full TE run: `te_fixed_pi`

## Aggregate Metrics

| Variant | Final L2 (mean±std) | Final Linf (mean±std) | Final Loss (mean±std) | Runtime (s, mean±std) | Params |
| --- | ---: | ---: | ---: | ---: | ---: |
| Classical + PI | 0.916583 ± 0.058399 | 1.061336 ± 0.107421 | 43.197057 ± 10.514407 | 14.8572 ± 1.8966 | 2241.0 |
| Classical + PI + analytic memory | 0.761948 ± 0.244735 | 0.848289 ± 0.300985 | 37.475642 ± 16.982665 | 17.0129 ± 0.9201 | 2433.0 |
| TE fixed residual 0.10 + PI (local-coordinate baseline) | 0.991693 ± 0.080669 | 1.107482 ± 0.087500 | 48.784985 ± 9.389905 | 42.8147 ± 3.1725 | 2306.0 |
| TE LayerNorm post_quantum + PI | 0.962204 ± 0.084977 | 1.104512 ± 0.116844 | 43.598615 ± 8.919407 | 44.0333 ± 3.2133 | 2338.0 |
| TE memory-aware analytic + PI | 0.889161 ± 0.187862 | 1.065903 ± 0.269021 | 35.705541 ± 8.592595 | 44.0522 ± 3.2268 | 2498.0 |
| TE memory-aware analytic + LayerNorm post_quantum + PI | 1.049739 ± 0.505995 | 1.282558 ± 0.374665 | 49.070580 ± 43.580305 | 46.3956 ± 3.3134 | 2530.0 |

## Win Counts

| Variant | L2 wins vs Classical | Linf wins vs Classical | L2 wins vs Full TE | Linf wins vs Full TE |
| --- | ---: | ---: | ---: | ---: |
| Classical + PI | n/a | n/a | 5 / 5 | 2 / 5 |
| Classical + PI + analytic memory | 4 / 5 | 4 / 5 | 4 / 5 | 4 / 5 |
| TE fixed residual 0.10 + PI (local-coordinate baseline) | 0 / 5 | 3 / 5 | n/a | n/a |
| TE LayerNorm post_quantum + PI | 1 / 5 | 3 / 5 | 5 / 5 | 2 / 5 |
| TE memory-aware analytic + PI | 2 / 5 | 2 / 5 | 2 / 5 | 2 / 5 |
| TE memory-aware analytic + LayerNorm post_quantum + PI | 3 / 5 | 2 / 5 | 4 / 5 | 2 / 5 |
