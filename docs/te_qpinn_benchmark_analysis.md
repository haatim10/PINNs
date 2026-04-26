# TE-QPINN Benchmark Analysis (Phase 5)

## 1. Purpose

This document summarizes the finalized Phase 5 benchmark interpretation for the TE-QPINN-inspired surrogate on the `50x50` single-seed run.

Goals:
- record exact final metrics,
- compare Classical + PI vs TE-QPINN Surrogate + PI under the same Adam-only budget,
- document what is promising vs still preliminary,
- define next experiments.

## 2. Benchmark setup

- Benchmark plan: `configs/benchmark_te_qpinn_50.yaml`
- Seed(s): `42` (single-seed run)
- Problem type: time-fractional integro-differential (PI path active)
- Grid: `50 x 50`
- Training budget: Adam-only, `12` epochs for both compared models
- Deterministic sampling: enabled
- Artifacts:
  - `outputs/benchmarks/te_qpinn_50/summary.csv`
  - `outputs/benchmarks/te_qpinn_50/summary.json`
  - `outputs/benchmarks/te_qpinn_50/benchmark_report.md`
  - `outputs/plots/te_qpinn_50/convergence_seed_42.png`
  - `outputs/plots/te_qpinn_50/summary_panels.png`

## 3. Models compared

1. Classical + PI
   - Model type: `classical`
   - Config: `configs/benchmark_te_qpinn_50_classical_pi.yaml`

2. TE-QPINN Surrogate + PI (tuned)
   - Model type: `te_qpinn_surrogate`
   - Config: `configs/benchmark_te_qpinn_50_surrogate_pi.yaml`
   - Tuned for near-matched parameter count and improved convergence.

## 4. Final 50x50 single-seed results

| Variant | Params | Runtime (s) | Final Loss | Final L2 | Final Linf |
| --- | ---: | ---: | ---: | ---: | ---: |
| Classical + PI | 2241 | 16.41 | 62.2103 | 1.00381 | 1.08578 |
| TE-QPINN Surrogate + PI | 2306 | 43.83 | 71.4435 | 0.88684 | 0.94618 |

## 5. Accuracy comparison

Under the same Adam-only 12-epoch budget:
- TE-QPINN Surrogate + PI achieved lower error than Classical + PI on both metrics:
  - L2: `0.88684` vs `1.00381`
  - Linf: `0.94618` vs `1.08578`
- Final training loss is higher for TE (`71.4435` vs `62.2103`), so lower weighted training loss did not align with lower evaluation error in this run.

## 6. Runtime and parameter comparison

- Parameter count is very close:
  - Classical: `2241`
  - TE-QPINN: `2306` (slightly higher)
- Runtime:
  - Classical is faster (`16.41s`)
  - TE-QPINN is slower (`43.83s`)

Interpretation: in this run, TE benefit is accuracy, not runtime.

## 7. Improvement over earlier TE smoke baseline

Earlier TE smoke baseline (`outputs/benchmarks/te_qpinn/summary.csv`):
- L2: `1.76838`
- Linf: `1.90787`

Tuned TE on finalized `50x50` run:
- L2: `0.88684` (about `49.9%` lower)
- Linf: `0.94618` (about `50.4%` lower)

This confirms substantial improvement over the earlier TE smoke baseline.

## 8. Adam vs Adam+LBFGS exploratory note

Exploratory TE-only trial (not part of the fair Adam-only comparison):
- Adam + LBFGS achieved:
  - L2: `0.07948`
  - Linf: `0.18600`
  - Runtime: about `634s`

This run used a much heavier training budget and should be treated as exploratory only, not as a fair head-to-head result against the Adam-only benchmark.

## 9. Fairness limitations

- Single-seed result (`seed=42`) only.
- No statistical confidence estimate yet (no variance across seeds).
- Adam+LBFGS result is not budget-matched to the Adam-only comparison.
- This repo currently uses a quantum-inspired surrogate (not a real quantum simulator/hardware claim).

## 10. Key findings

- Promising result: with matched Adam-only budget, TE-QPINN Surrogate + PI outperformed Classical + PI on L2 and Linf in this `50x50` run.
- Parameter matching was achieved closely (`2306` vs `2241`).
- TE-QPINN is slower in wall-clock time in this configuration.
- Result is preliminary and should not be framed as statistical superiority yet.

## 11. Recommended next steps

1. Lock this tuned TE config as the current candidate for fair follow-up.
2. Run multi-seed validation (e.g., `3-5` seeds) using the same fixed setup.
3. Keep parameter budget tightly matched during all comparisons.
4. Add explicit reporting of best/final metrics by seed and confidence intervals.
5. Evaluate whether TE runtime can be reduced without losing the observed error advantage.
6. Keep Adam+LBFGS as a separate exploratory track until a fair budget-matched protocol is defined.
