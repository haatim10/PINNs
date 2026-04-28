# TE-QPINN Ablation Stats

- Benchmark config: `configs/benchmark_te_qpinn_gated_multiseed.yaml`
- Reference full TE run: `full_te_pi`
- Reference classical run: `classical_pi`

## Metrics

| Variant | Final L2 | Final Linf | Final Loss | Runtime (s) | Params |
| --- | ---: | ---: | ---: | ---: | ---: |
| Classical + PI | 0.916583 | 1.061336 | 43.197057 | 14.3691 | 2241.0 |
| TE-QPINN Fixed Residual 0.10 + PI | 0.991693 | 1.107482 | 48.784985 | 40.0081 | 2306.0 |
| TE-QPINN Gated Residual + PI | 1.200541 | 1.591720 | 50.676954 | 43.0674 | 2307.0 |

## Comparison Status

| Variant | L2 vs Full TE | Linf vs Full TE | Loss vs Full TE | Runtime vs Full TE | L2 vs Classical | Linf vs Classical | Loss vs Classical | Runtime vs Classical |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Classical + PI | improves | improves | improves | improves | neutral | neutral | neutral | neutral |
| TE-QPINN Fixed Residual 0.10 + PI | neutral | neutral | neutral | neutral | worsens | worsens | worsens | worsens |
| TE-QPINN Gated Residual + PI | worsens | worsens | worsens | worsens | worsens | worsens | worsens | worsens |
