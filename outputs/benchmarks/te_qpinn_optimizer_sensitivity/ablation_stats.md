# TE-QPINN Ablation Stats

- Benchmark config: `configs/benchmark_te_qpinn_optimizer_sensitivity.yaml`
- Reference full TE run: `te_fixed_adam_pi`
- Reference classical run: `classical_adam_pi`

## Metrics

| Variant | Final L2 | Final Linf | Final Loss | Runtime (s) | Params |
| --- | ---: | ---: | ---: | ---: | ---: |
| Classical + PI (Adam) | 1.003810 | 1.085777 | 62.210301 | 15.7019 | 2241.0 |
| Classical + PI (Adam+LBFGS, extended) | 0.085135 | 0.260378 | 0.161102 | 549.7721 | 2241.0 |
| TE fixed residual 0.10 + PI (Adam) | 0.886842 | 0.946181 | 71.443514 | 40.6095 | 2306.0 |
| TE fixed residual 0.10 + PI (Adam+LBFGS, extended) | 0.059035 | 0.209135 | 0.081750 | 1589.7988 | 2306.0 |
| TE LayerNorm post_quantum + PI (Adam) | 0.823901 | 0.957646 | 54.057441 | 41.8441 | 2338.0 |
| TE LayerNorm post_quantum + PI (Adam+LBFGS, extended) | 0.035359 | 0.107435 | 0.129883 | 1697.6655 | 2338.0 |

## Comparison Status

| Variant | L2 vs Full TE | Linf vs Full TE | Loss vs Full TE | Runtime vs Full TE | L2 vs Classical | Linf vs Classical | Loss vs Classical | Runtime vs Classical |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Classical + PI (Adam) | worsens | worsens | improves | improves | neutral | neutral | neutral | neutral |
| Classical + PI (Adam+LBFGS, extended) | improves | improves | improves | worsens | improves | improves | improves | worsens |
| TE fixed residual 0.10 + PI (Adam) | neutral | neutral | neutral | neutral | improves | improves | worsens | worsens |
| TE fixed residual 0.10 + PI (Adam+LBFGS, extended) | improves | improves | improves | worsens | improves | improves | improves | worsens |
| TE LayerNorm post_quantum + PI (Adam) | improves | worsens | improves | worsens | improves | improves | improves | worsens |
| TE LayerNorm post_quantum + PI (Adam+LBFGS, extended) | improves | improves | improves | worsens | improves | improves | improves | worsens |
