# TE-QPINN Multi-Seed Stats

- Benchmark config: `configs/benchmark_te_qpinn_layernorm_multiseed.yaml`
- Seeds: [0, 1, 2, 3, 4]
- Reference classical run: `classical_pi`
- Reference full TE run: `full_te_pi`

## Aggregate Metrics

| Variant | Final L2 (mean±std) | Final Linf (mean±std) | Final Loss (mean±std) | Runtime (s, mean±std) | Params |
| --- | ---: | ---: | ---: | ---: | ---: |
| Classical + PI | 0.916583 ± 0.058399 | 1.061336 ± 0.107421 | 43.197057 ± 10.514407 | 18.2714 ± 1.5318 | 2241.0 |
| TE fixed residual 0.10 + PI | 0.991693 ± 0.080669 | 1.107482 ± 0.087500 | 48.784985 ± 9.389905 | 51.0464 ± 3.1593 | 2306.0 |
| TE LayerNorm post_quantum + PI | 0.962204 ± 0.084977 | 1.104512 ± 0.116844 | 43.598615 ± 8.919407 | 53.9656 ± 3.7063 | 2338.0 |

## Win Counts

| Variant | L2 wins vs Classical | Linf wins vs Classical | L2 wins vs Full TE | Linf wins vs Full TE |
| --- | ---: | ---: | ---: | ---: |
| Classical + PI | n/a | n/a | 5 / 5 | 2 / 5 |
| TE fixed residual 0.10 + PI | 0 / 5 | 3 / 5 | n/a | n/a |
| TE LayerNorm post_quantum + PI | 1 / 5 | 3 / 5 | 5 / 5 | 2 / 5 |
