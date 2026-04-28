# TE-QPINN Ablation Stats

- Benchmark config: `configs/benchmark_te_qpinn_layernorm_multiseed.yaml`
- Reference full TE run: `full_te_pi`
- Reference classical run: `classical_pi`

## Metrics

| Variant | Final L2 | Final Linf | Final Loss | Runtime (s) | Params |
| --- | ---: | ---: | ---: | ---: | ---: |
| Classical + PI | 0.916583 | 1.061336 | 43.197057 | 18.2714 | 2241.0 |
| TE fixed residual 0.10 + PI | 0.991693 | 1.107482 | 48.784985 | 51.0464 | 2306.0 |
| TE LayerNorm post_quantum + PI | 0.962204 | 1.104512 | 43.598615 | 53.9656 | 2338.0 |

## Comparison Status

| Variant | L2 vs Full TE | Linf vs Full TE | Loss vs Full TE | Runtime vs Full TE | L2 vs Classical | Linf vs Classical | Loss vs Classical | Runtime vs Classical |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Classical + PI | improves | improves | improves | improves | neutral | neutral | neutral | neutral |
| TE fixed residual 0.10 + PI | neutral | neutral | neutral | neutral | worsens | worsens | worsens | worsens |
| TE LayerNorm post_quantum + PI | improves | improves | improves | worsens | worsens | worsens | worsens | worsens |
