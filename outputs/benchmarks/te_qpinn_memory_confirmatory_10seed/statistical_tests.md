# Statistical Tests: Memory-Aware 10-Seed Confirmatory Results

- Source: `outputs/benchmarks/te_qpinn_memory_confirmatory_10seed/summary.csv`
- Primary endpoint: mean final L2 under Adam-only 10-seed validation
- Metric analyzed: final L2 (lower is better)

| Comparison | Mean diff (A-B) | Median diff (A-B) | Wilcoxon p (two-sided) | Wilcoxon p (A<B) | Bootstrap 95% CI (mean diff) | Paired d | Cliff's δ | Wins A | Ties | Losses A |
| --- | ---: | ---: | ---: | ---: | --- | ---: | ---: | ---: | ---: | ---: |
| Classical + memory vs Classical | -0.169920 | -0.183730 | 0.0371094 | 0.0185547 | [-0.292771, -0.049636] | -0.8320595583920034 | -0.640000 | 9 | 0 | 1 |
| TE memory analytic vs TE LayerNorm non-memory | -0.108698 | -0.089811 | 0.160156 | 0.0800781 | [-0.227282, 0.004445] | -0.5493017964596704 | -0.340000 | 6 | 0 | 4 |
| TE memory analytic vs TE fixed | -0.146024 | -0.136400 | 0.130859 | 0.0654297 | [-0.266543, -0.030561] | -0.7181016267606241 | -0.480000 | 6 | 0 | 4 |
| TE memory analytic vs Classical + memory | 0.056595 | 0.116844 | 0.695312 | 0.6875 | [-0.092482, 0.195866] | 0.23012973030469594 | 0.120000 | 3 | 0 | 7 |

Notes:
- Differences are computed as A - B, so negative values favor A for final L2.
- Wilcoxon p(A<B) tests whether A tends to have lower final L2 than B.
- Cliff's delta is reported using the standard independent-groups definition over pairwise comparisons.
