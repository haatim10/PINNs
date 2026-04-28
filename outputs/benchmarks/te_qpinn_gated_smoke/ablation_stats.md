# TE-QPINN Ablation Stats

- Benchmark config: `configs/benchmark_te_qpinn_gated_smoke.yaml`
- Reference full TE run: `full_te_pi`
- Reference classical run: `classical_pi`

## Metrics

| Variant | Final L2 | Final Linf | Final Loss | Runtime (s) | Params |
| --- | ---: | ---: | ---: | ---: | ---: |
| Classical + PI | 1.003810 | 1.085777 | 62.210301 | 15.7234 | 2241.0 |
| TE-QPINN Fixed Residual 0.10 + PI | 0.886842 | 0.946181 | 71.443514 | 39.0831 | 2306.0 |
| TE-QPINN Gated Residual + PI | 0.674444 | 0.694570 | 57.514838 | 42.9364 | 2307.0 |

## Comparison Status

| Variant | L2 vs Full TE | Linf vs Full TE | Loss vs Full TE | Runtime vs Full TE | L2 vs Classical | Linf vs Classical | Loss vs Classical | Runtime vs Classical |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Classical + PI | worsens | worsens | improves | improves | neutral | neutral | neutral | neutral |
| TE-QPINN Fixed Residual 0.10 + PI | neutral | neutral | neutral | neutral | improves | improves | worsens | worsens |
| TE-QPINN Gated Residual + PI | improves | improves | improves | worsens | improves | improves | improves | worsens |
