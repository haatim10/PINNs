# TE-QPINN Benchmark Analysis (Phase 6D)

## 1. Purpose

This document updates the TE-QPINN benchmark interpretation using both:

- the earlier single-seed `50x50` run (seed 42), and
- the completed multi-seed validation (seeds `0..4`) from:
  - `outputs/benchmarks/te_qpinn_multiseed`
  - `outputs/plots/te_qpinn_multiseed`

The objective is to report results honestly and determine whether the earlier single-seed gain generalizes.

## 2. Benchmark protocol

- PDE: time-fractional integro-differential equation (same PI formulation for all runs)
- Grid: `50 x 50`
- Training budget: Adam-only, `12` epochs
- Compared models:
  1. Classical + PI (`configs/benchmark_te_qpinn_50_classical_pi.yaml`)
  2. TE-QPINN Surrogate + PI (`configs/benchmark_te_qpinn_50_surrogate_pi.yaml`)
- Multi-seed plan: `configs/benchmark_te_qpinn_multiseed.yaml` with seeds `[0,1,2,3,4]`

## 3. Single-seed recap (seed 42)

Earlier seed-42 results showed TE-QPINN improvement:

| Variant | Params | Runtime (s) | Final Loss | Final L2 | Final Linf |
| --- | ---: | ---: | ---: | ---: | ---: |
| Classical + PI | 2241 | 16.41 | 62.2103 | 1.00381 | 1.08578 |
| TE-QPINN Surrogate + PI | 2306 | 43.83 | 71.4435 | 0.88684 | 0.94618 |

This looked promising but required multi-seed validation before drawing stronger conclusions.

## 4. Multi-seed validation results (5 seeds)

### Classical + PI
- mean final L2: `0.916583`
- std final L2: `0.058399`
- mean final Linf: `1.061336`
- std final Linf: `0.107421`
- mean runtime: `14.7583 s`
- std runtime: `1.3977 s`
- mean final loss: `43.197057`
- std final loss: `10.514407`
- parameter count: `2241`

### TE-QPINN Surrogate + PI
- mean final L2: `0.991693`
- std final L2: `0.080669`
- mean final Linf: `1.107482`
- std final Linf: `0.087500`
- mean runtime: `42.0679 s`
- std runtime: `2.0448 s`
- mean final loss: `48.784985`
- std final loss: `9.389905`
- parameter count: `2306`

### Win counts (paired seeds)
- TE-QPINN final L2 wins: `0 / 5`
- TE-QPINN final Linf wins: `3 / 5`

## 5. Interpretation

1. The earlier single-seed improvement did **not** generalize across 5 seeds.
2. TE-QPINN should **not** be claimed as better overall at the current stage.
3. Classical + PI remains stronger on:
   - mean final L2,
   - mean final Linf,
   - mean runtime,
   - mean final loss.
4. TE-QPINN shows localized promise in Linf behavior, with `3/5` paired-seed Linf wins.
5. The current TE-QPINN surrogate remains useful as an exploratory architecture, but it needs further tuning and/or optimizer strategy changes before it can be considered consistently competitive.

## 6. Fairness and scope notes

- Comparison is fair in core setup (same PDE, grid, seed list, and Adam-only epoch budget).
- This is still a quantum-inspired surrogate evaluation, not a hardware/simulator quantum-advantage claim.
- Multi-seed evidence now provides stronger reliability than single-seed interpretation.

## 7. Recommended next steps

1. Ablation study:
   - embedding depth/width,
   - pairwise mixing on/off,
   - residual branch on/off.
2. Adam+LBFGS comparison as exploratory future work (clearly marked as non-budget-matched unless controlled).
3. Equal-runtime comparison (instead of equal-epoch only) to evaluate accuracy-efficiency tradeoffs.
4. Optimizer sensitivity study (learning rate, scheduler, clipping, optimizer type).
5. Architecture tuning only after locking the evaluation protocol to prevent moving-target comparisons.

## 8. Phase 6E ablation snapshot (seed 42, diagnostic)

An ablation smoke study was run with the locked `50x50`, Adam-only 12-epoch setup (`configs/benchmark_te_qpinn_ablation.yaml`) to isolate component effects.

- Baselines in this ablation run:
  - Classical + PI: final L2 `1.0038`, final Linf `1.0858`, final loss `62.2103`, runtime `17.24s`, params `2241`.
  - Full TE-QPINN + PI: final L2 `0.8868`, final Linf `0.9462`, final loss `71.4435`, runtime `46.34s`, params `2306`.
- Strongest ablation in this seed:
  - `residual_scale_020_pi`: final L2 `0.8275`, final Linf `0.8692`, final loss `70.0524`, runtime `44.35s`, params `2306`.
- Notable tradeoff example:
  - `no_residual_pi` improved L2 (`0.8756`) and loss (`49.1515`) relative to full TE, but Linf worsened (`1.1938`).

Interpretation: the ablation is useful diagnostically and suggests residual scaling and embedding/variational choices materially affect TE behavior, but this is still single-seed evidence. We should treat it as directional guidance for the next locked-protocol experiments, not as a generalized performance claim.

## 9. Residual Scale Multi-Seed Validation

A targeted 5-seed validation was run for:

1. Classical + PI
2. Full TE-QPINN + PI (`residual_scale = 0.10`)
3. TE-QPINN + PI (`residual_scale = 0.20`)

Setup was locked to the same `50x50` time-fractional integro-differential problem and the same Adam-only 12-epoch budget (`configs/benchmark_te_qpinn_residual_scale_multiseed.yaml`).

### Aggregate results (seeds 0..4)

| Variant | Mean Final L2 | Std L2 | Mean Final Linf | Std Linf | Mean Final Loss | Std Loss | Mean Runtime (s) | Std Runtime (s) | Params |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| Classical + PI | 0.916583 | 0.058399 | 1.061336 | 0.107421 | 43.197057 | 10.514407 | 14.4189 | 1.4946 | 2241 |
| Full TE (0.10) + PI | 0.991693 | 0.080669 | 1.107482 | 0.087500 | 48.784985 | 9.389905 | 39.8334 | 1.1720 | 2306 |
| TE (0.20) + PI | 1.022494 | 0.151428 | 1.210228 | 0.148215 | 50.156834 | 12.457935 | 39.6660 | 1.2766 | 2306 |

### Win counts

- Full TE (0.10) vs Classical, L2 wins: `0 / 5`
- Full TE (0.10) vs Classical, Linf wins: `3 / 5`
- TE (0.20) vs Classical, L2 wins: `1 / 5`
- TE (0.20) vs Classical, Linf wins: `0 / 5`
- TE (0.20) vs Full TE (0.10), L2 wins: `1 / 5`
- TE (0.20) vs Full TE (0.10), Linf wins: `0 / 5`

### Interpretation

The seed-42 ablation gain for `residual_scale = 0.20` did not generalize to the 5-seed setting. In this locked comparison, `residual_scale = 0.20` is weaker than `0.10` on mean L2, mean Linf, mean final loss, and win counts (with only `1/5` L2 wins vs `0.10` and `0/5` Linf wins). Classical + PI remains strongest overall on mean L2, mean Linf, mean final loss, and runtime.

## 10. Learned Gated Residual Smoke Test

A seed-42 smoke test was run to evaluate a minimal architecture upgrade: learned gated residual blending (`residual_blend_mode = gated`, `residual_gate_init = 0.5`) using the same locked `50x50` setup and Adam-only 12-epoch budget (`configs/benchmark_te_qpinn_gated_smoke.yaml`).

| Variant | Params | Runtime (s) | Final Loss | Final L2 | Final Linf |
| --- | ---: | ---: | ---: | ---: | ---: |
| Classical + PI | 2241 | 15.7234 | 62.2103 | 1.00381 | 1.08578 |
| TE fixed residual 0.10 + PI | 2306 | 39.0831 | 71.4435 | 0.88684 | 0.94618 |
| TE gated residual + PI | 2307 | 42.9364 | 57.5148 | 0.67444 | 0.69457 |

Interpretation: in this seed-42 smoke test, gated residual blending substantially improved L2/Linf and final loss relative to both classical and fixed-residual TE, at a small parameter increase (+1) and slightly higher runtime than fixed TE. This is promising but still preliminary because it is single-seed evidence only. The next step should be a locked 5-seed validation of fixed TE vs gated TE before making any broader claim.

## 11. Learned Gated Residual Multi-Seed Validation

A locked 5-seed validation was run for:

1. Classical + PI
2. TE-QPINN fixed residual `0.10` + PI
3. TE-QPINN learned gated residual + PI

using the same `50x50` problem and Adam-only 12-epoch budget (`configs/benchmark_te_qpinn_gated_multiseed.yaml`).

### Aggregate results (seeds 0..4)

| Variant | Mean Final L2 | Std L2 | Mean Final Linf | Std Linf | Mean Final Loss | Std Loss | Mean Runtime (s) | Std Runtime (s) | Params |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| Classical + PI | 0.916583 | 0.058399 | 1.061336 | 0.107421 | 43.197057 | 10.514407 | 14.3691 | 0.9529 | 2241 |
| TE fixed 0.10 + PI | 0.991693 | 0.080669 | 1.107482 | 0.087500 | 48.784985 | 9.389905 | 40.0081 | 0.7726 | 2306 |
| TE gated + PI | 1.200541 | 0.349286 | 1.591720 | 0.485644 | 50.676954 | 16.690771 | 43.0674 | 1.5391 | 2307 |

### Gated win counts

- Gated TE vs Classical, L2 wins: `1 / 5`
- Gated TE vs Classical, Linf wins: `0 / 5`
- Gated TE vs Fixed TE, L2 wins: `1 / 5`
- Gated TE vs Fixed TE, Linf wins: `0 / 5`

### Interpretation

The strong seed-42 gated smoke result did not generalize across 5 seeds. Under the locked protocol, gated residual blending is worse on mean L2, mean Linf, mean final loss, and win counts versus both classical and fixed TE. It is also slower on average than fixed TE (about `43.07s` vs `40.01s`). Based on current evidence, gated residual blending is not yet a stable architecture improvement for this benchmark and should not be claimed as superior.

## 12. Feature Normalization Smoke Test

A seed-42 smoke benchmark evaluated optional LayerNorm variants while keeping fixed residual blending (`residual_blend_mode = fixed`, `residual_scale = 0.10`) and all other TE settings locked (`configs/benchmark_te_qpinn_layernorm_smoke.yaml`).

| Variant | Params | Runtime (s) | Final Loss | Final L2 | Final Linf |
| --- | ---: | ---: | ---: | ---: | ---: |
| Classical + PI | 2241 | 15.3493 | 62.2103 | 1.00381 | 1.08578 |
| TE fixed residual 0.10 + PI | 2306 | 38.9796 | 71.4435 | 0.88684 | 0.94618 |
| TE LayerNorm post_quantum + PI | 2338 | 40.9595 | 54.0574 | 0.82390 | 0.95765 |
| TE LayerNorm post_entanglement + PI | 2352 | 40.8774 | 75.8186 | 0.93496 | 0.96240 |

Interpretation (seed 42 only): LayerNorm `post_quantum` improved final L2 and final loss versus fixed TE, but had slightly worse Linf and slightly higher runtime. LayerNorm `post_entanglement` did not improve over fixed TE on L2/Linf/loss and was also slower. Because this is single-seed evidence, no general claim should be made yet. The next step is a locked 5-seed validation for the `post_quantum` LayerNorm variant.

## 13. Post-Quantum LayerNorm Multi-Seed Validation

A locked 5-seed validation was run for:

1. Classical + PI
2. TE fixed residual `0.10` + PI
3. TE LayerNorm `post_quantum` + PI

using the same `50x50` problem and Adam-only 12-epoch budget (`configs/benchmark_te_qpinn_layernorm_multiseed.yaml`).

### Aggregate results (seeds 0..4)

| Variant | Mean Final L2 | Std L2 | Mean Final Linf | Std Linf | Mean Final Loss | Std Loss | Mean Runtime (s) | Std Runtime (s) | Params |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| Classical + PI | 0.916583 | 0.058399 | 1.061336 | 0.107421 | 43.197057 | 10.514407 | 18.2714 | 1.5318 | 2241 |
| TE fixed 0.10 + PI | 0.991693 | 0.080669 | 1.107482 | 0.087500 | 48.784985 | 9.389905 | 51.0464 | 3.1593 | 2306 |
| TE LayerNorm post_quantum + PI | 0.962204 | 0.084977 | 1.104512 | 0.116844 | 43.598615 | 8.919407 | 53.9656 | 3.7063 | 2338 |

### LayerNorm win counts

- LayerNorm TE vs Classical, L2 wins: `1 / 5`
- LayerNorm TE vs Classical, Linf wins: `3 / 5`
- LayerNorm TE vs Fixed TE, L2 wins: `5 / 5`
- LayerNorm TE vs Fixed TE, Linf wins: `2 / 5`

### Interpretation

Compared with fixed TE, post-quantum LayerNorm generalized as a useful stabilization change for L2 and loss: it reduced mean final L2 (`0.9622` vs `0.9917`) and mean final loss (`43.60` vs `48.78`), with a small reduction in mean Linf (`1.1045` vs `1.1075`). It also improved L2 on every paired seed (`5/5`) versus fixed TE.

However, this does not establish overall superiority versus Classical + PI. Classical still has better mean L2 (`0.9166`), better mean Linf (`1.0613`), slightly lower mean final loss (`43.20`), and much faster runtime (`18.27s` vs `53.97s`). LayerNorm TE also uses more parameters (`2338` vs `2241` classical, `2306` fixed TE). So the current takeaway is: post-quantum LayerNorm is a promising TE-side improvement over fixed TE, but it remains slower and not yet stronger than classical on mean metrics under the locked budget.

## 14. Optimizer Sensitivity Study

A controlled seed-42 optimizer sensitivity run was completed for:

1. Classical + PI
2. TE fixed residual `0.10` + PI
3. TE LayerNorm `post_quantum` + PI

Each variant was evaluated with:

- Adam-only (locked 12-epoch budget), and
- Adam+LBFGS (extended budget: Adam stage + LBFGS fine-tuning)

using `configs/benchmark_te_qpinn_optimizer_sensitivity.yaml`.

### Per-variant results

| Variant | Optimizer | Final L2 | Final Linf | Final Loss | Runtime (s) | Params |
| --- | --- | ---: | ---: | ---: | ---: | ---: |
| Classical + PI | Adam | 1.003810 | 1.085777 | 62.210301 | 15.7019 | 2241 |
| Classical + PI | Adam+LBFGS (extended) | 0.085135 | 0.260378 | 0.161102 | 549.7721 | 2241 |
| TE fixed 0.10 + PI | Adam | 0.886842 | 0.946181 | 71.443514 | 40.6095 | 2306 |
| TE fixed 0.10 + PI | Adam+LBFGS (extended) | 0.059035 | 0.209135 | 0.081750 | 1589.7988 | 2306 |
| TE LayerNorm post_quantum + PI | Adam | 0.823901 | 0.957646 | 54.057441 | 41.8441 | 2338 |
| TE LayerNorm post_quantum + PI | Adam+LBFGS (extended) | 0.035359 | 0.107435 | 0.129883 | 1697.6655 | 2338 |

### Adam -> Adam+LBFGS change (extended-budget deltas)

- Classical + PI:
  - L2: `1.003810 -> 0.085135` (`-91.52%`)
  - Linf: `1.085777 -> 0.260378` (`-76.02%`)
  - Runtime: `15.70s -> 549.77s` (`+3401%`)
- TE fixed + PI:
  - L2: `0.886842 -> 0.059035` (`-93.34%`)
  - Linf: `0.946181 -> 0.209135` (`-77.90%`)
  - Runtime: `40.61s -> 1589.80s` (`+3815%`)
- TE LayerNorm + PI:
  - L2: `0.823901 -> 0.035359` (`-95.71%`)
  - Linf: `0.957646 -> 0.107435` (`-88.78%`)
  - Runtime: `41.84s -> 1697.67s` (`+3957%`)

### Interpretation

1. Adam+LBFGS substantially improves accuracy and final loss for all three models, but at very large runtime cost.
2. This is not an equal-budget comparison. The Adam+LBFGS results should be interpreted strictly as extended-optimization outcomes.
3. TE variants, especially TE LayerNorm post-quantum, show larger relative gains than classical in this seed-42 study, which suggests stronger optimizer sensitivity.
4. These results are promising for optimizer-aware TE training strategy, but they are still single-seed evidence and should not be used to claim final superiority.

## 15. Memory-Aware TE-QPINN Smoke Test (Phase 8C)

A new seed-42 smoke benchmark was run (`configs/benchmark_te_qpinn_memory_smoke.yaml`) to test the first low-risk analytic memory-feature extension.

Compared variants:

1. Classical + PI
2. Classical + PI + analytic memory features (fairness control)
3. TE fixed residual 0.10 + PI
4. TE LayerNorm post_quantum + PI
5. TE memory-aware analytic + PI
6. TE memory-aware analytic + LayerNorm post_quantum + PI

### Seed-42 smoke results

| Variant | Params | Runtime (s) | Final Loss | Final L2 | Final Linf |
| --- | ---: | ---: | ---: | ---: | ---: |
| Classical + PI | 2241 | 17.3504 | 62.2103 | 1.00381 | 1.08578 |
| Classical + PI + analytic memory | 2433 | 18.3400 | 62.3695 | 0.92487 | 0.83902 |
| TE fixed residual 0.10 + PI | 2306 | 44.6783 | 71.4435 | 0.88684 | 0.94618 |
| TE LayerNorm post_quantum + PI | 2338 | 47.2960 | 54.0574 | 0.82390 | 0.95765 |
| TE memory-aware analytic + PI | 2498 | 47.3314 | 37.0430 | 0.68147 | 0.95543 |
| TE memory-aware analytic + LayerNorm post_quantum + PI | 2530 | 49.8322 | 45.7305 | 0.69897 | 0.81093 |

### Interpretation (smoke only)

1. Using the best non-memory TE reference by L2 (TE LayerNorm post_quantum, L2 `0.82390`), both memory-aware TE variants improved final L2 by more than 2%:
   - TE memory analytic: `0.68147` (about `17.3%` lower).
   - TE memory analytic + LayerNorm: `0.69897` (about `15.2%` lower).
2. TE memory analytic + LayerNorm also improved final Linf strongly (`0.81093`), while TE memory analytic had similar Linf to non-memory TE (`0.95543` vs `0.95765` in this reference).
3. However, the Classical + memory-feature control also improved markedly versus Classical baseline (notably L2 and Linf), which means part of the gain may come from the analytic memory features themselves, not uniquely from TE architecture.
4. Memory-aware TE variants use more parameters and similar/slightly higher runtime than non-memory TE, so fairness interpretation must continue to include parameter and runtime context.
5. This is single-seed evidence only and should not be treated as generalized performance.

### Go / no-go outcome

Under the Phase 8B smoke gate, memory-aware TE **passes** the threshold for proceeding to 5-seed validation (it exceeds the required >=2% improvement criterion on final L2 versus best non-memory TE in seed 42).  
Next step should be locked multi-seed validation before any stronger claim.

## 16. Memory-Aware TE-QPINN Five-Seed Validation (Phase 8D)

A locked 5-seed validation was completed with no retuning (`configs/benchmark_te_qpinn_memory_multiseed.yaml`, seeds `[0,1,2,3,4]`) using:

1. Classical + PI
2. Classical + PI + analytic memory (fairness control)
3. TE fixed residual 0.10 + PI (local-coordinate baseline)
4. TE LayerNorm post_quantum + PI (best non-memory TE reference)
5. TE memory-aware analytic + PI
6. TE memory-aware analytic + LayerNorm post_quantum + PI

### Aggregate results

| Variant | Mean Final L2 | Std L2 | Mean Final Linf | Std Linf | Mean Final Loss | Std Loss | Mean Runtime (s) | Std Runtime (s) | Params |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| Classical + PI | 0.916583 | 0.058399 | 1.061336 | 0.107421 | 43.197057 | 10.514407 | 14.8572 | 1.8966 | 2241 |
| Classical + PI + analytic memory | 0.761948 | 0.244735 | 0.848289 | 0.300985 | 37.475642 | 16.982665 | 17.0129 | 0.9201 | 2433 |
| TE fixed residual 0.10 + PI | 0.991693 | 0.080669 | 1.107482 | 0.087500 | 48.784985 | 9.389905 | 42.8147 | 3.1725 | 2306 |
| TE LayerNorm post_quantum + PI | 0.962204 | 0.084977 | 1.104512 | 0.116844 | 43.598615 | 8.919407 | 44.0333 | 3.2133 | 2338 |
| TE memory-aware analytic + PI | 0.889161 | 0.187862 | 1.065903 | 0.269021 | 35.705541 | 8.592595 | 44.0522 | 3.2268 | 2498 |
| TE memory-aware analytic + LayerNorm post_quantum + PI | 1.049739 | 0.505995 | 1.282558 | 0.374665 | 49.070580 | 43.580305 | 46.3956 | 3.3134 | 2530 |

### Requested win counts

- TE memory vs best non-memory TE (TE LayerNorm): L2 `2/5`, Linf `2/5`
- TE memory + LayerNorm vs TE LayerNorm non-memory: L2 `2/5`, Linf `2/5`
- TE memory vs Classical: L2 `2/5`, Linf `2/5`
- TE memory vs Classical + memory: L2 `1/5`, Linf `1/5`
- Classical + memory vs Classical: L2 `4/5`, Linf `4/5`

### Accuracy-per-runtime context

Using `mean(final L2) * mean(runtime)` (lower is better):

- Classical + PI: `13.62`
- Classical + memory: `12.96`
- TE LayerNorm (non-memory): `42.37`
- TE memory analytic: `39.17`

This indicates memory-aware TE improves efficiency relative to non-memory TE, but remains far slower for a given error level than classical variants.

### Stage-2 go/no-go check (memory TE vs best non-memory TE)

Criteria from the Phase 8B plan:

- (a) mean final L2 improves by >=2%
- (b) >=3/5 L2 wins
- (c) mean final loss improves by >=5% with runtime increase <=35%

Observed for TE memory analytic vs TE LayerNorm non-memory:

- L2 improvement: `+7.59%` (pass a)
- L2 wins: `2/5` (fail b)
- Loss improvement: `+18.10%` with runtime increase `+0.04%` (pass c)

Result: **Stage-2 gate PASS** (criteria a and c passed).

### Interpretation

1. Memory-aware TE (analytic) improves over the best non-memory TE on mean L2 and mean loss, with essentially unchanged runtime.
2. However, the Classical + memory control is stronger than memory-aware TE on both mean L2 and mean Linf, and wins `4/5` against classical baseline.
3. This means current gains appear substantially driven by analytic memory features themselves; a TE-specific advantage is not yet established.
4. TE memory + LayerNorm is unstable in this run (high variance, weaker means), so it should not be treated as a finalist without further stabilization.

Bottom line: memory-aware analytic features are promising and worth continuing, but the current evidence supports a feature-engineering benefit more strongly than a unique TE-architecture benefit.

## 17. Memory-Aware TE-QPINN Ten-Seed Confirmatory Validation (Phase 8E)

A locked Stage 2b confirmatory run was completed with seeds:

`[0, 1, 2, 3, 4, 42, 123, 2024, 2025, 999]`

using `configs/benchmark_te_qpinn_memory_confirmatory_10seed.yaml` and `--resume-incomplete`.

Compared variants:

1. Classical + PI
2. Classical + PI + analytic memory
3. TE fixed residual 0.10 + PI
4. TE LayerNorm post_quantum + PI
5. TE memory-aware analytic + PI

Primary endpoint: mean final L2 (Adam-only, 10 seeds).

### Aggregate means/std (official benchmark summary)

| Variant | Mean Final L2 | Std L2 | Mean Final Linf | Std Linf | Mean Final Loss | Std Loss | Mean Runtime (s) | Std Runtime (s) | Params |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| Classical + PI | 0.922830 | 0.076989 | 1.043477 | 0.122118 | 44.012908 | 12.667057 | 14.7715 | 1.1693 | 2241 |
| Classical + PI + analytic memory | 0.752910 | 0.188977 | 0.810768 | 0.217877 | 37.575163 | 16.695402 | 17.7416 | 1.6117 | 2433 |
| TE fixed 0.10 + PI | 0.955529 | 0.070070 | 1.052766 | 0.092046 | 47.486260 | 14.141845 | 42.4348 | 2.8408 | 2306 |
| TE LayerNorm post_quantum + PI | 0.918204 | 0.076402 | 1.025404 | 0.117049 | 41.205865 | 10.416957 | 44.6977 | 3.0572 | 2338 |
| TE memory-aware analytic + PI | 0.809505 | 0.176401 | 1.008622 | 0.232580 | 26.643364 | 13.316055 | 45.0615 | 3.4201 | 2498 |

### Extended per-variant metrics (median/best/worst + accuracy-per-runtime)

Detailed tables are saved in:

- `outputs/benchmarks/te_qpinn_memory_confirmatory_10seed/confirmatory_stats.md`
- `outputs/benchmarks/te_qpinn_memory_confirmatory_10seed/confirmatory_stats.json`

Key values:

- Classical + PI: L2 median `0.901796`, best `0.782010`, worst `1.018611`, accuracy-per-runtime `13.6316`
- Classical + memory: L2 median `0.745080`, best `0.454774`, worst `1.124376`, accuracy-per-runtime `13.3579`
- TE fixed: L2 median `0.954154`, best `0.878501`, worst `1.104665`, accuracy-per-runtime `40.5477`
- TE LayerNorm: L2 median `0.901254`, best `0.823901`, worst `1.076966`, accuracy-per-runtime `41.0416`
- TE memory analytic: L2 median `0.803911`, best `0.577227`, worst `1.082807`, accuracy-per-runtime `36.4775`

### Requested win counts

- TE memory vs TE fixed: L2 wins `6/10`, Linf wins `4/10`
- TE memory vs TE LayerNorm non-memory: L2 wins `6/10`, Linf wins `5/10`
- TE memory vs Classical: L2 wins `7/10`, Linf wins `6/10`
- TE memory vs Classical + memory: L2 wins `3/10`, Linf wins `2/10`
- Classical + memory vs Classical: L2 wins `9/10`, Linf wins `9/10`

### Statistical tests (paired, from confirmatory_stats)

- Classical + memory vs Classical (L2):
  - Wilcoxon `p(two-sided)=0.0371`, `p(a<b)=0.0186`
  - Mean L2 diff (a-b) `-0.1699`, 95% bootstrap CI `[-0.2921, -0.0494]`
  - Paired Cohen’s d `-0.8321`
- TE memory vs TE LayerNorm non-memory (L2):
  - Wilcoxon `p(two-sided)=0.1602`, `p(a<b)=0.0801`
  - Mean L2 diff `-0.1087`, 95% bootstrap CI `[-0.2285, 0.0069]`
- TE memory vs Classical + memory (L2):
  - Wilcoxon `p(two-sided)=0.6953`, `p(a<b)=0.6875`
  - Mean L2 diff `+0.0566`, 95% bootstrap CI `[-0.0928, 0.1931]`

### Interpretation

1. Memory-aware TE improves over non-memory TE baselines on mean L2 and mean loss in this 10-seed run.
2. However, Classical + analytic memory remains stronger on mean L2, mean Linf, and efficiency; it also wins `9/10` on both L2 and Linf versus Classical baseline.
3. This supports memory-feature usefulness for this benchmark, but does not establish TE-specific superiority.
4. TE memory-aware Linf variance remains relatively high, so stability concerns are still present.
5. No quantum-advantage claim is warranted; this remains a quantum-inspired surrogate study.

## 18. Exact PennyLane TE-QPINN Feasibility Check

An exact paper-aligned PennyLane TE-QPINN model was integrated as an optional model family and validated with a tiny smoke benchmark (`configs/benchmark_te_qpinn_exact_pqc_smoke.yaml`, seed `42`).

### Tiny smoke results

| Variant | Params | Runtime (s) | Final Loss | Final L2 | Final Linf |
| --- | ---: | ---: | ---: | ---: | ---: |
| Classical + PI (tiny) | 337 | 1.6793 | 23.8083 | 0.930595 | 0.968080 |
| Exact TE-QPINN PennyLane + PI (tiny) | 251 | 1348.6111 | 206.3259 | 3.918995 | 2.718204 |

### Interpretation

1. The exact PQC path is differentiable and integrated (forward, autograd, and second derivatives were validated in tests).
2. The current exact PQC path is computationally expensive, primarily due to sample-wise QNode execution and higher-order autograd cost under the fractional/product-integration residual.
3. In this tiny smoke setup, exact PQC is much slower and less accurate than the classical tiny baseline.
4. This is a feasibility/bottleneck result, not a performance-success result.
5. Practical benchmark progress should continue on the memory-aware classical/surrogate tracks while exact PQC runtime bottlenecks are addressed.
