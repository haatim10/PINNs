# Confirmatory 10-Seed Stats

- Summary source: `/root/projects/PINNs/outputs/benchmarks/te_qpinn_memory_confirmatory_10seed/summary.csv`
- Seeds: [0, 1, 2, 3, 4, 42, 123, 999, 2024, 2025]
- Primary endpoint: `mean final L2 under Adam-only 10-seed validation`

## Per-Variant Metrics

| Variant | L2 mean±std | L2 median | L2 best | L2 worst | Linf mean±std | Linf median | Linf best | Linf worst | Loss mean±std | Runtime mean±std (s) | Params | Accuracy/Runtime (L2*Runtime) |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| Classical + PI | 0.922830 ± 0.076989 | 0.901796 | 0.782010 | 1.018611 | 1.043477 ± 0.122118 | 1.063469 | 0.840332 | 1.234247 | 44.012908 ± 12.667057 | 14.7715 ± 1.1693 | 2241 | 13.631599 |
| Classical + PI + analytic memory | 0.752910 ± 0.188977 | 0.745080 | 0.454774 | 1.124376 | 0.810768 ± 0.217877 | 0.791522 | 0.579770 | 1.357430 | 37.575163 ± 16.695402 | 17.7416 ± 1.6117 | 2433 | 13.357856 |
| TE fixed residual 0.10 + PI (local-coordinate baseline) | 0.955529 ± 0.070070 | 0.954154 | 0.878501 | 1.104665 | 1.052766 ± 0.092046 | 1.051337 | 0.946181 | 1.233170 | 47.486260 ± 14.141845 | 42.4348 ± 2.8408 | 2306 | 40.547657 |
| TE LayerNorm post_quantum + PI | 0.918204 ± 0.076402 | 0.901254 | 0.823901 | 1.076966 | 1.025404 ± 0.117049 | 0.981166 | 0.878312 | 1.235743 | 41.205865 ± 10.416957 | 44.6977 ± 3.0572 | 2338 | 41.041594 |
| TE memory-aware analytic + PI | 0.809505 ± 0.176401 | 0.803911 | 0.577227 | 1.082807 | 1.008622 ± 0.232580 | 1.029323 | 0.709439 | 1.326101 | 26.643364 ± 13.316055 | 45.0615 ± 3.4201 | 2498 | 36.477499 |

## Requested Win Counts + Paired Tests

### TE memory-aware analytic + PI vs TE fixed residual 0.10 + PI (local-coordinate baseline)
- L2 wins: 6 / 10 (ties 0, losses 4)
- Linf wins: 4 / 10 (ties 0, losses 6)
- Wilcoxon L2 p(two-sided)=0.130859, p(a<b)=0.0654297; mean diff(a-b)=-0.146024, 95% CI [-0.267724, -0.027365], paired d=-0.7181016267606243

### TE memory-aware analytic + PI vs TE LayerNorm post_quantum + PI
- L2 wins: 6 / 10 (ties 0, losses 4)
- Linf wins: 5 / 10 (ties 0, losses 5)
- Wilcoxon L2 p(two-sided)=0.160156, p(a<b)=0.0800781; mean diff(a-b)=-0.108698, 95% CI [-0.228488, 0.006927], paired d=-0.5493017964596704

### TE memory-aware analytic + PI vs Classical + PI
- L2 wins: 7 / 10 (ties 0, losses 3)
- Linf wins: 6 / 10 (ties 0, losses 4)
- Wilcoxon L2 p(two-sided)=0.105469, p(a<b)=0.0527344; mean diff(a-b)=-0.113324, 95% CI [-0.223541, 0.003999], paired d=-0.5862769277891039

### TE memory-aware analytic + PI vs Classical + PI + analytic memory
- L2 wins: 3 / 10 (ties 0, losses 7)
- Linf wins: 2 / 10 (ties 0, losses 8)
- Wilcoxon L2 p(two-sided)=0.695312, p(a<b)=0.6875; mean diff(a-b)=0.056595, 95% CI [-0.092770, 0.193072], paired d=0.23012973030469594

### Classical + PI + analytic memory vs Classical + PI
- L2 wins: 9 / 10 (ties 0, losses 1)
- Linf wins: 9 / 10 (ties 0, losses 1)
- Wilcoxon L2 p(two-sided)=0.0371094, p(a<b)=0.0185547; mean diff(a-b)=-0.169920, 95% CI [-0.292057, -0.049450], paired d=-0.8320595583920033

