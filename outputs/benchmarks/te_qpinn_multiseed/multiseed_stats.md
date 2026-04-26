# TE-QPINN Multiseed Validation Summary

- Benchmark config: `configs/benchmark_te_qpinn_multiseed.yaml`
- Seeds: 0, 1, 2, 3, 4

## Aggregate Metrics

| Variant | Mean Final L2 | Std Final L2 | Mean Final Linf | Std Final Linf | Mean Runtime (s) | Std Runtime (s) | Mean Final Loss | Std Final Loss | Params |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| Classical + PI | 0.916583 | 0.058399 | 1.061336 | 0.107421 | 14.7583 | 1.3977 | 43.197057 | 10.514407 | 2241 |
| TE-QPINN Surrogate + PI | 0.991693 | 0.080669 | 1.107482 | 0.087500 | 42.0679 | 2.0448 | 48.784985 | 9.389905 | 2306 |

## Win Counts (TE-QPINN vs Classical)

- Paired seeds: 5
- TE-QPINN wins by Final L2: 0
- TE-QPINN wins by Final Linf: 3
