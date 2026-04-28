# Optimizer Sensitivity Stats

- Benchmark config: `configs/benchmark_te_qpinn_optimizer_sensitivity.yaml`
- Note: Adam+LBFGS includes an extended optimization budget (Adam stage plus LBFGS fine-tuning) and is not an equal-budget comparison with Adam-only runs.

## Adam vs Adam+LBFGS

| Variant | Adam L2 | Adam+LBFGS L2 | ΔL2 | Adam Linf | Adam+LBFGS Linf | ΔLinf | Adam Loss | Adam+LBFGS Loss | ΔLoss | Adam Runtime (s) | Adam+LBFGS Runtime (s) | ΔRuntime (s) | Adam Params | Adam+LBFGS Params |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| Classical + PI | 1.003810 | 0.085135 | -0.918674 | 1.085777 | 0.260378 | -0.825399 | 62.210301 | 0.161102 | -62.049198 | 15.7019 | 549.7721 | 534.0702 | 2241.0 | 2241.0 |
| TE fixed residual 0.10 + PI | 0.886842 | 0.059035 | -0.827807 | 0.946181 | 0.209135 | -0.737046 | 71.443514 | 0.081750 | -71.361764 | 40.6095 | 1589.7988 | 1549.1893 | 2306.0 | 2306.0 |
| TE LayerNorm post_quantum + PI | 0.823901 | 0.035359 | -0.788542 | 0.957646 | 0.107435 | -0.850211 | 54.057441 | 0.129883 | -53.927558 | 41.8441 | 1697.6655 | 1655.8214 | 2338.0 | 2338.0 |

## Delta Status (Adam+LBFGS - Adam)

| Variant | L2 | Linf | Loss | Runtime | Params |
| --- | --- | --- | --- | --- | --- |
| Classical + PI | improves | improves | improves | worsens | neutral |
| TE fixed residual 0.10 + PI | improves | improves | improves | worsens | neutral |
| TE LayerNorm post_quantum + PI | improves | improves | improves | worsens | neutral |
