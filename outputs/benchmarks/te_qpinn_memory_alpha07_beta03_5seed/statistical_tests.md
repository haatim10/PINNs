# Confirmatory 10-Seed Stats

- Summary source: `/root/projects/PINNs/outputs/benchmarks/te_qpinn_memory_alpha07_beta03_5seed/summary.csv`
- Seeds: [0, 1, 2, 3, 4]
- Primary endpoint: `mean final L2 under Adam-only 10-seed validation`

## Per-Variant Metrics

| Variant | L2 mean±std | L2 median | L2 best | L2 worst | Linf mean±std | Linf median | Linf best | Linf worst | Loss mean±std | Runtime mean±std (s) | Params | Accuracy/Runtime (L2*Runtime) |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| Classical + PI (alpha=0.7, beta=0.3) | 0.935119 ± 0.053208 | 0.906633 | 0.893514 | 1.017869 | 1.055873 ± 0.121808 | 1.024814 | 0.937080 | 1.210795 | 40.927904 ± 12.203974 | 15.7726 ± 2.0796 | 2241 | 14.749299 |
| Classical + PI + analytic memory (alpha=0.7, beta=0.3) | 0.885903 ± 0.183472 | 0.833454 | 0.715898 | 1.189433 | 0.955070 ± 0.236414 | 0.922753 | 0.688759 | 1.337447 | 38.573208 ± 13.828171 | 18.1548 ± 0.6639 | 2433 | 16.083413 |
| TE fixed residual 0.10 + PI (alpha=0.7, beta=0.3) | 1.025442 ± 0.114842 | 1.007409 | 0.897531 | 1.212530 | 1.121506 ± 0.134984 | 1.106146 | 0.957400 | 1.315696 | 102.662733 ± 143.959989 | 43.5231 ± 1.3878 | 2306 | 44.630481 |
| TE LayerNorm post_quantum + PI (alpha=0.7, beta=0.3) | 0.985235 ± 0.113020 | 0.991360 | 0.847100 | 1.157593 | 1.070939 ± 0.107036 | 1.018903 | 0.977633 | 1.246790 | 57.117036 ± 46.812099 | 45.9477 ± 1.1748 | 2338 | 45.269276 |
| TE memory-aware analytic + PI (alpha=0.7, beta=0.3) | 1.087911 ± 0.167052 | 1.126237 | 0.901557 | 1.303846 | 1.218205 ± 0.227703 | 1.220189 | 0.949280 | 1.456272 | 231.438177 ± 441.600714 | 46.4354 ± 1.3434 | 2498 | 50.517624 |

## Requested Win Counts + Paired Tests

### TE memory-aware analytic + PI (alpha=0.7, beta=0.3) vs TE fixed residual 0.10 + PI (alpha=0.7, beta=0.3)
- L2 wins: 2 / 5 (ties 0, losses 3)
- Linf wins: 2 / 5 (ties 0, losses 3)
- Wilcoxon L2 p(two-sided)=0.8125, p(a<b)=0.6875; mean diff(a-b)=0.062469, 95% CI [-0.143381, 0.247767], paired d=0.2492094677615303

### TE memory-aware analytic + PI (alpha=0.7, beta=0.3) vs TE LayerNorm post_quantum + PI (alpha=0.7, beta=0.3)
- L2 wins: 2 / 5 (ties 0, losses 3)
- Linf wins: 2 / 5 (ties 0, losses 3)
- Wilcoxon L2 p(two-sided)=0.4375, p(a<b)=0.84375; mean diff(a-b)=0.102676, 95% CI [-0.099863, 0.302056], paired d=0.3942635343109756

### TE memory-aware analytic + PI (alpha=0.7, beta=0.3) vs Classical + PI (alpha=0.7, beta=0.3)
- L2 wins: 1 / 5 (ties 0, losses 4)
- Linf wins: 1 / 5 (ties 0, losses 4)
- Wilcoxon L2 p(two-sided)=0.1875, p(a<b)=0.9375; mean diff(a-b)=0.152793, 95% CI [-0.009199, 0.289357], paired d=0.8160601762208971

### TE memory-aware analytic + PI (alpha=0.7, beta=0.3) vs Classical + PI + analytic memory (alpha=0.7, beta=0.3)
- L2 wins: 1 / 5 (ties 0, losses 4)
- Linf wins: 1 / 5 (ties 0, losses 4)
- Wilcoxon L2 p(two-sided)=0.1875, p(a<b)=0.9375; mean diff(a-b)=0.202009, 95% CI [-0.044725, 0.387089], paired d=0.7370686040111977

### Classical + PI + analytic memory (alpha=0.7, beta=0.3) vs Classical + PI (alpha=0.7, beta=0.3)
- L2 wins: 3 / 5 (ties 0, losses 2)
- Linf wins: 4 / 5 (ties 0, losses 1)
- Wilcoxon L2 p(two-sided)=0.625, p(a<b)=0.3125; mean diff(a-b)=-0.049216, 95% CI [-0.203067, 0.149557], paired d=-0.22483359723772853

