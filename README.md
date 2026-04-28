# Quantum-Inspired PINNs for Time-Fractional Integro-Differential Equations

> **Quantum-Readiness Branch**
>
> This branch extends a classical PINN + product-integration solver with a **TE-QPINN-inspired surrogate model**.
> The implemented TE module is **quantum-inspired** (classical PyTorch surrogate), not a hardware quantum circuit.
> The objective is to test whether quantum-style embeddings, entanglement-inspired mixing, and expectation-style readout improve PINN behavior on the same fractional integro-differential equation.

## Main Research Question

Can a TE-QPINN-inspired surrogate improve accuracy, stability, or optimizer sensitivity compared with a classical PINN baseline for a time-fractional integro-differential equation?

## Problem Formulation

We solve:

$$
D_t^\alpha u(x,t) - (x^2 + 1)u_{xx} + \int_0^t \sin(x)(t-s)^{-\beta}u(x,s)\,ds = f(x,t)
$$

- Domain: $x \in [0,1]$, $t \in (0,1]$
- Boundary conditions:
  - $u(0,t) = t^\alpha$
  - $u(1,t) = -t^\alpha$
- Initial condition:
  - $u(x,0) = 0$

## TE-QPINN-Inspired Surrogate Architecture

Implemented components in this branch:

- trainable embedding network
- input rescaling
- angle-style encoding
- sin/cos quantum-inspired feature map
- entanglement-inspired pairwise feature mixing
- expectation-style readout
- residual correction branch
- optional learned gated residual blending
- optional post-quantum/post-entanglement LayerNorm

Scope note:

- This is **not** a full quantum circuit simulator.
- This is **not** hardware-executable quantum computing.
- This is a classical surrogate inspired by TE-QPINN design ideas.

## Classical Baseline (Secondary Reference)

Classical PINN + PI remains the baseline for fair comparison:

- same PDE/problem setup
- same physics-informed residual objective structure
- same boundary/initial conditioning setup
- used to evaluate whether TE-inspired architecture adds value

## Results Snapshot

### A) Adam-Only 5-Seed Comparison (Locked 50x50, Seeds 0..4)

| Variant | Final L2 (mean ± std) | Final Linf (mean ± std) | Final Loss (mean ± std) |
| --- | ---: | ---: | ---: |
| Classical + PI | 0.916583 ± 0.058399 | 1.061336 ± 0.107421 | 43.197057 ± 10.514407 |
| TE fixed residual 0.10 + PI | 0.991693 ± 0.080669 | 1.107482 ± 0.087500 | 48.784985 ± 9.389905 |
| TE LayerNorm post_quantum + PI | 0.962204 ± 0.084977 | 1.104512 ± 0.116844 | 43.598615 ± 8.919407 |

Interpretation:

- Classical remains strongest overall under Adam-only.
- Post-quantum LayerNorm improves TE vs fixed TE in mean L2 and loss.
- TE LayerNorm wins L2 `5/5` against fixed TE, but does not beat Classical overall.

### B) Adam+LBFGS Extended-Budget Seed-42 Comparison

| Variant | Final L2 | Final Linf | Final Loss | Runtime (s) |
| --- | ---: | ---: | ---: | ---: |
| Classical + PI (Adam+LBFGS) | 0.0851 | 0.2604 | 0.1611 | 549.77 |
| TE fixed 0.10 + PI (Adam+LBFGS) | 0.0590 | 0.2091 | 0.0817 | 1589.80 |
| TE LayerNorm post_quantum + PI (Adam+LBFGS) | 0.035359 | 0.107435 | 0.129883 | 1697.67 |

Interpretation:

- Adam+LBFGS is **extended-budget**, not equal-runtime.
- TE LayerNorm achieved the strongest seed-42 accuracy, with high runtime cost.
- This indicates strong optimizer sensitivity and high-accuracy potential on TE variants.

## Experiment Timeline (Completed Phases)

- [x] TE-QPINN surrogate implemented
- [x] model factory support added
- [x] unit tests added
- [x] config-driven benchmark runner added
- [x] smoke benchmark completed
- [x] 50x50 single-seed benchmark completed
- [x] 5-seed validation completed
- [x] ablation study completed
- [x] residual-scale validation completed
- [x] learned gated residual tested
- [x] post-quantum LayerNorm tested
- [x] optimizer sensitivity study completed

## How To Run

Run tests:

```bash
pytest -q
```

Smoke benchmark:

```bash
python scripts/benchmark_te_qpinn.py --benchmark-config configs/benchmark_te_qpinn_smoke.yaml
```

5-seed benchmark:

```bash
python scripts/benchmark_te_qpinn.py --benchmark-config configs/benchmark_te_qpinn_multiseed.yaml
```

LayerNorm multi-seed:

```bash
python scripts/benchmark_te_qpinn.py --benchmark-config configs/benchmark_te_qpinn_layernorm_multiseed.yaml
```

Optimizer sensitivity:

```bash
python scripts/benchmark_te_qpinn.py --benchmark-config configs/benchmark_te_qpinn_optimizer_sensitivity.yaml --resume-incomplete
```

## Output / Artifact Index

- `outputs/benchmarks/te_qpinn_multiseed/`
- `outputs/benchmarks/te_qpinn_layernorm_multiseed/`
- `outputs/benchmarks/te_qpinn_optimizer_sensitivity/`
- `outputs/plots/te_qpinn_multiseed/`
- `outputs/plots/te_qpinn_layernorm_multiseed/`
- `outputs/plots/te_qpinn_optimizer_sensitivity/`
- `docs/te_qpinn_benchmark_analysis.md`

## Current Interpretation

- Classical PINN + PI is still more efficient and stronger under Adam-only averages.
- TE fixed baseline is not sufficient by itself.
- LayerNorm improves TE-side stability and mean TE performance vs fixed TE.
- LBFGS significantly improves accuracy for all variants, especially TE LayerNorm.
- Runtime cost is a major limitation for LBFGS-based high-accuracy runs.
- Additional evidence is needed before any TE-QPINN superiority claim.

## Next Steps

- Stage A: 10-seed Adam-only stability map
- Stage B: targeted Adam+LBFGS runs on finalists
- Stage C: equal-runtime and equal-epoch finalist study
- statistical testing (paired tests + effect sizes)
- bootstrap confidence intervals
- time-to-quality curves
- alpha/beta robustness tests
- hard boundary/initial-condition output transforms

---

For full quantitative detail, phase-by-phase interpretation, and caveats, see:
`docs/te_qpinn_benchmark_analysis.md`.
