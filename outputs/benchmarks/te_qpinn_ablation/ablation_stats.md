# TE-QPINN Ablation Stats

- Benchmark config: `configs/benchmark_te_qpinn_ablation.yaml`
- Reference full TE run: `full_te_pi`
- Reference classical run: `classical_pi`

## Metrics

| Variant | Final L2 | Final Linf | Final Loss | Runtime (s) | Params |
| --- | ---: | ---: | ---: | ---: | ---: |
| Classical + PI | 1.003810 | 1.085777 | 62.210301 | 17.2412 | 2241.0 |
| Full TE-QPINN + PI | 0.886842 | 0.946181 | 71.443514 | 46.3441 | 2306.0 |
| TE no residual + PI | 0.875587 | 1.193847 | 49.151477 | 41.7829 | 2273.0 |
| TE residual scale 0.05 + PI | 0.916612 | 0.985341 | 72.292546 | 48.4825 | 2306.0 |
| TE residual scale 0.20 + PI | 0.827545 | 0.869191 | 70.052411 | 44.3508 | 2306.0 |
| TE variational hidden 16 + PI | 1.039427 | 1.121980 | 59.687656 | 44.6118 | 1794.0 |
| TE variational hidden 64 + PI | 1.000603 | 1.188654 | 57.823115 | 44.5367 | 3330.0 |
| TE small embedding + PI | 1.004052 | 1.123759 | 51.631271 | 41.6095 | 1502.0 |
| TE large embedding + PI | 0.924175 | 1.056717 | 53.545564 | 43.9399 | 3062.0 |

## Comparison Status

| Variant | L2 vs Full TE | Linf vs Full TE | Loss vs Full TE | Runtime vs Full TE | L2 vs Classical | Linf vs Classical | Loss vs Classical | Runtime vs Classical |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Classical + PI | worsens | worsens | improves | improves | neutral | neutral | neutral | neutral |
| Full TE-QPINN + PI | neutral | neutral | neutral | neutral | improves | improves | worsens | worsens |
| TE no residual + PI | improves | worsens | improves | improves | improves | worsens | improves | worsens |
| TE residual scale 0.05 + PI | worsens | worsens | worsens | worsens | improves | improves | worsens | worsens |
| TE residual scale 0.20 + PI | improves | improves | improves | improves | improves | improves | worsens | worsens |
| TE variational hidden 16 + PI | worsens | worsens | improves | improves | worsens | worsens | improves | worsens |
| TE variational hidden 64 + PI | worsens | worsens | improves | improves | improves | worsens | improves | worsens |
| TE small embedding + PI | worsens | worsens | improves | improves | worsens | worsens | improves | worsens |
| TE large embedding + PI | worsens | worsens | improves | improves | improves | improves | improves | worsens |
