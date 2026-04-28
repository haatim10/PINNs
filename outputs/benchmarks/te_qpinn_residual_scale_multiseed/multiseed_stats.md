# TE-QPINN Multi-Seed Stats

- Benchmark config: `configs/benchmark_te_qpinn_residual_scale_multiseed.yaml`
- Seeds: [0, 1, 2, 3, 4]
- Reference classical run: `classical_pi`
- Reference full TE run: `full_te_pi`

## Aggregate Metrics

| Variant | Final L2 (mean±std) | Final Linf (mean±std) | Final Loss (mean±std) | Runtime (s, mean±std) | Params |
| --- | ---: | ---: | ---: | ---: | ---: |
| Classical + PI | 0.916583 ± 0.058399 | 1.061336 ± 0.107421 | 43.197057 ± 10.514407 | 14.4189 ± 1.4946 | 2241.0 |
| Full TE-QPINN + PI (residual 0.10) | 0.991693 ± 0.080669 | 1.107482 ± 0.087500 | 48.784985 ± 9.389905 | 39.8334 ± 1.1720 | 2306.0 |
| TE-QPINN + PI (residual 0.20) | 1.022494 ± 0.151428 | 1.210228 ± 0.148215 | 50.156834 ± 12.457935 | 39.6660 ± 1.2766 | 2306.0 |

## Win Counts

| Variant | L2 wins vs Classical | Linf wins vs Classical | L2 wins vs Full TE | Linf wins vs Full TE |
| --- | ---: | ---: | ---: | ---: |
| Classical + PI | n/a | n/a | 5 / 5 | 2 / 5 |
| Full TE-QPINN + PI (residual 0.10) | 0 / 5 | 3 / 5 | n/a | n/a |
| TE-QPINN + PI (residual 0.20) | 1 / 5 | 0 / 5 | 1 / 5 | 0 / 5 |
