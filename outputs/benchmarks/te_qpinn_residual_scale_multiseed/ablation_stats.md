# TE-QPINN Ablation Stats

- Benchmark config: `configs/benchmark_te_qpinn_residual_scale_multiseed.yaml`
- Reference full TE run: `full_te_pi`
- Reference classical run: `classical_pi`

## Metrics

| Variant | Final L2 | Final Linf | Final Loss | Runtime (s) | Params |
| --- | ---: | ---: | ---: | ---: | ---: |
| Classical + PI | 0.916583 | 1.061336 | 43.197057 | 14.4189 | 2241.0 |
| Full TE-QPINN + PI (residual 0.10) | 0.991693 | 1.107482 | 48.784985 | 39.8334 | 2306.0 |
| TE-QPINN + PI (residual 0.20) | 1.022494 | 1.210228 | 50.156834 | 39.6660 | 2306.0 |

## Comparison Status

| Variant | L2 vs Full TE | Linf vs Full TE | Loss vs Full TE | Runtime vs Full TE | L2 vs Classical | Linf vs Classical | Loss vs Classical | Runtime vs Classical |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Classical + PI | improves | improves | improves | improves | neutral | neutral | neutral | neutral |
| Full TE-QPINN + PI (residual 0.10) | neutral | neutral | neutral | neutral | worsens | worsens | worsens | worsens |
| TE-QPINN + PI (residual 0.20) | worsens | worsens | worsens | improves | worsens | worsens | worsens | worsens |
