# TE-QPINN Ablation Stats

- Benchmark config: `configs/benchmark_te_qpinn_layernorm_smoke.yaml`
- Reference full TE run: `full_te_pi`
- Reference classical run: `classical_pi`

## Metrics

| Variant | Final L2 | Final Linf | Final Loss | Runtime (s) | Params |
| --- | ---: | ---: | ---: | ---: | ---: |
| Classical + PI | 1.003810 | 1.085777 | 62.210301 | 15.3493 | 2241.0 |
| TE fixed residual 0.10 + PI | 0.886842 | 0.946181 | 71.443514 | 38.9796 | 2306.0 |
| TE LayerNorm post_quantum + PI | 0.823901 | 0.957646 | 54.057441 | 40.9595 | 2338.0 |
| TE LayerNorm post_entanglement + PI | 0.934959 | 0.962399 | 75.818580 | 40.8774 | 2352.0 |

## Comparison Status

| Variant | L2 vs Full TE | Linf vs Full TE | Loss vs Full TE | Runtime vs Full TE | L2 vs Classical | Linf vs Classical | Loss vs Classical | Runtime vs Classical |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Classical + PI | worsens | worsens | improves | improves | neutral | neutral | neutral | neutral |
| TE fixed residual 0.10 + PI | neutral | neutral | neutral | neutral | improves | improves | worsens | worsens |
| TE LayerNorm post_quantum + PI | improves | worsens | improves | worsens | improves | improves | improves | worsens |
| TE LayerNorm post_entanglement + PI | worsens | worsens | worsens | worsens | improves | improves | worsens | worsens |
