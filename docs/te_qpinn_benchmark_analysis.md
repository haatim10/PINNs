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
