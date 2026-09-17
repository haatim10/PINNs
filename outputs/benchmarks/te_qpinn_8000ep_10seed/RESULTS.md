# Adam-Only 10-Seed Confirmatory Results at 8000 Epochs

Seeds: `[0, 1, 2, 3, 4, 42, 123, 999, 2024, 2025]`  
Budget: 8000 epochs, cosine warmup (500). Primary endpoint: mean final L2.

Supersedes the 12-epoch tables, where every variant scored at or above the
zero-predictor level and the ranking measured initialization noise.

## Per-Variant Metrics

| Variant | Final L2 (mean ± std) | Final Linf (mean) | Published (12 ep) | Improvement | Runtime (s) |
| --- | ---: | ---: | ---: | ---: | ---: |
| TE memory-aware analytic + PI | 0.00262 ± 0.00081 | 0.00752 | 0.8095 | 308x | 96 |
| Classical + PI + analytic memory | 0.00415 ± 0.00173 | 0.01070 | 0.7529 | 181x | 50 |
| TE LayerNorm post_quantum + PI | 0.01759 ± 0.00479 | 0.05831 | 0.9182 | 52x | 106 |
| TE fixed residual 0.10 + PI | 0.01834 ± 0.00380 | 0.06057 | 0.9555 | 52x | 95 |
| Classical + PI | 0.03595 ± 0.00740 | 0.08800 | 0.9228 | 26x | 44 |
| **Zero predictor (u == 0)** | 1.00000 | 1.00000 | - | - | 0 |

> Reference baseline: predicting `u == 0` scores relative L2 = 1.0 by definition.
> Every variant is now 27x-380x below that line; in the 12-epoch table none were.

## Paired Tests on Final L2 (same seeds)

| Comparison | Mean diff | Ratio | Wins | Wilcoxon p | Bootstrap 95% CI | Paired d |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| TE memory-aware analytic + PI vs Classical + PI + analytic memory | -0.00153 | 1.58x | 9/10 | 0.0488 | [-0.00266, -0.00041] | -0.80 |
| Classical + PI + analytic memory vs Classical + PI | -0.03180 | 8.66x | 10/10 | 0.0020 | [-0.03612, -0.02774] | -4.44 |
| TE memory-aware analytic + PI vs TE fixed residual 0.10 + PI | -0.01572 | 6.99x | 10/10 | 0.0020 | [-0.01761, -0.01349] | -4.41 |
| TE fixed residual 0.10 + PI vs Classical + PI | -0.01761 | 1.96x | 10/10 | 0.0020 | [-0.02329, -0.01241] | -1.87 |
| TE LayerNorm post_quantum + PI vs TE fixed residual 0.10 + PI | -0.00075 | 1.04x | 7/10 | 0.4922 | [-0.00307, +0.00177] | -0.18 |

> Bonferroni threshold for 5 comparisons: 0.05/5 = 0.0100.
> All rows clear it except TE-memory vs classical-memory (p = 0.0488) and
> LayerNorm vs fixed TE (p = 0.4922).

