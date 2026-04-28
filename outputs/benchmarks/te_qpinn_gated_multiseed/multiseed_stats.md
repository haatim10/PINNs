# TE-QPINN Multi-Seed Stats

- Benchmark config: `configs/benchmark_te_qpinn_gated_multiseed.yaml`
- Seeds: [0, 1, 2, 3, 4]
- Reference classical run: `classical_pi`
- Reference full TE run: `full_te_pi`

## Aggregate Metrics

| Variant | Final L2 (mean±std) | Final Linf (mean±std) | Final Loss (mean±std) | Runtime (s, mean±std) | Params |
| --- | ---: | ---: | ---: | ---: | ---: |
| Classical + PI | 0.916583 ± 0.058399 | 1.061336 ± 0.107421 | 43.197057 ± 10.514407 | 14.3691 ± 0.9529 | 2241.0 |
| TE-QPINN Fixed Residual 0.10 + PI | 0.991693 ± 0.080669 | 1.107482 ± 0.087500 | 48.784985 ± 9.389905 | 40.0081 ± 0.7726 | 2306.0 |
| TE-QPINN Gated Residual + PI | 1.200541 ± 0.349286 | 1.591720 ± 0.485644 | 50.676954 ± 16.690771 | 43.0674 ± 1.5391 | 2307.0 |

## Win Counts

| Variant | L2 wins vs Classical | Linf wins vs Classical | L2 wins vs Full TE | Linf wins vs Full TE |
| --- | ---: | ---: | ---: | ---: |
| Classical + PI | n/a | n/a | 5 / 5 | 2 / 5 |
| TE-QPINN Fixed Residual 0.10 + PI | 0 / 5 | 3 / 5 | n/a | n/a |
| TE-QPINN Gated Residual + PI | 1 / 5 | 0 / 5 | 1 / 5 | 0 / 5 |
