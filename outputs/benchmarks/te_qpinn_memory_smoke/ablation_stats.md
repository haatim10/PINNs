# TE-QPINN Ablation Stats

- Benchmark config: `configs/benchmark_te_qpinn_memory_smoke.yaml`
- Reference full TE run: `te_fixed_pi`
- Reference classical run: `classical_pi`

## Metrics

| Variant | Final L2 | Final Linf | Final Loss | Runtime (s) | Params |
| --- | ---: | ---: | ---: | ---: | ---: |
| Classical + PI | 1.003810 | 1.085777 | 62.210301 | 17.3504 | 2241.0 |
| Classical + PI + analytic memory | 0.924867 | 0.839024 | 62.369484 | 18.3400 | 2433.0 |
| TE fixed residual 0.10 + PI | 0.886842 | 0.946181 | 71.443514 | 44.6783 | 2306.0 |
| TE LayerNorm post_quantum + PI | 0.823901 | 0.957646 | 54.057441 | 47.2960 | 2338.0 |
| TE memory-aware analytic + PI | 0.681474 | 0.955431 | 37.043000 | 47.3314 | 2498.0 |
| TE memory-aware analytic + LayerNorm post_quantum + PI | 0.698971 | 0.810930 | 45.730465 | 49.8322 | 2530.0 |

## Comparison Status

| Variant | L2 vs Full TE | Linf vs Full TE | Loss vs Full TE | Runtime vs Full TE | L2 vs Classical | Linf vs Classical | Loss vs Classical | Runtime vs Classical |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Classical + PI | worsens | worsens | improves | improves | neutral | neutral | neutral | neutral |
| Classical + PI + analytic memory | worsens | improves | improves | improves | improves | improves | worsens | worsens |
| TE fixed residual 0.10 + PI | neutral | neutral | neutral | neutral | improves | improves | worsens | worsens |
| TE LayerNorm post_quantum + PI | improves | worsens | improves | worsens | improves | improves | improves | worsens |
| TE memory-aware analytic + PI | improves | worsens | improves | worsens | improves | improves | improves | worsens |
| TE memory-aware analytic + LayerNorm post_quantum + PI | improves | improves | improves | worsens | improves | improves | improves | worsens |
