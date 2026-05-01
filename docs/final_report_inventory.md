# Final Report Inventory (IEEE-Style, Publication-Focused)

This document is an artifact inventory for preparing the final IEEE two-column report.  
Scope: **results, ablations, exact comparisons, statistical validation, robustness, and application case studies**.  
It is intentionally publication-oriented and claim-constrained.

---

## 1. Recommended Report Title Options

1. **Memory-Aware Quantum-Inspired Physics-Informed Neural Networks for Time-Fractional Integro-Differential Equations**
2. **Benchmarking Classical, Quantum-Inspired, and Memory-Aware PINNs for Fractional Integro-Differential Systems**
3. **Memory-Aware PINN and TE-QPINN Surrogates for Fractional Dynamics with Sparse CSI Forecasting Applications**
4. **A Fairness-Controlled Benchmark of TE-QPINN Surrogates and Memory Features for Fractional PINNs**
5. **From Fractional PINN Baselines to Memory-Aware Quantum-Inspired Surrogates: A Multi-Seed Evaluation**

---

## 2. Recommended Central Research Question

**How much performance gain comes from (i) TE-QPINN-inspired architecture choices, (ii) analytic memory features, (iii) optimizer strategy, and (iv) application-specific domain priors, under fairness-controlled and multi-seed evaluation?**

---

## 3. Recommended Main Contribution List

1. A reproducible benchmark pipeline for time-fractional integro-differential PINNs with controlled comparisons.
2. Implementation of TE-QPINN-inspired surrogate models in a modular config-driven framework.
3. Stabilization studies (LayerNorm, residual scaling, gated residual blending) with ablations and multi-seed validation.
4. Analytic memory-aware feature extension tied to fractional structure (`alpha`, `beta`) with fairness controls.
5. Classical + memory control baseline to separate feature-engineering gains from TE-architecture gains.
6. 10-seed confirmatory validation and paired statistical testing.
7. Alpha/beta robustness test at `(alpha,beta)=(0.7,0.3)`.
8. Optional exact PennyLane TE-QPINN feasibility study (integration + differentiability + bottleneck identification).
9. Application-inspired sparse CSI forecasting demos (scalar and two-channel/MIMO-style synthetic setting).

---

## 4. Recommended IEEE Report Structure

I. Introduction  
II. Fractional Integro-Differential Problem Formulation  
III. Baseline PINN and Numerical Residual Construction  
IV. TE-QPINN-Inspired and Exact PQC Model Variants  
V. Memory-Aware Analytic Feature Design  
VI. Experimental Protocol  
VII. Benchmark and Ablation Results  
VIII. Exact PennyLane TE-QPINN Feasibility Study  
IX. Application Case Study: Sparse CSI Forecasting  
X. Discussion  
XI. Limitations and Future Work  
XII. Conclusion

---

## 4A. MidSem-to-Final Continuity Map

Use the MidSem report as the methodological base and the final report as the evidence-heavy extension:

- **Carry forward from MidSem (Sections II–III of final):**
  - fractional integro-differential formulation,
  - Caputo derivative treatment,
  - L1-style temporal discretization context,
  - graded temporal mesh rationale,
  - weakly singular Volterra/product-integration residual construction,
  - manufactured/exact-solution validation strategy,
  - baseline classical PINN setup.

- **Add in final report (Sections IV–IX):**
  - TE-QPINN surrogate variants and stabilization,
  - memory-aware analytic feature design with fairness controls,
  - multi-seed confirmatory and paired statistics,
  - alpha/beta robustness,
  - exact PennyLane TE-QPINN feasibility appendix,
  - sparse CSI scalar and two-channel application demos.

- **Practical note:** `MidSem_Report.pdf` is not present in the tracked repo tree here, so cite it as an external prior milestone document and map its section numbers explicitly in the final manuscript.

---

## 5. Key Result Tables (Exact Numbers + Sources)

### Table A — Adam-Only 10-Seed Confirmatory (Primary Benchmark)
- **Suggested caption:** *Ten-seed Adam-only confirmatory results on the base fractional setting (primary endpoint: mean final L2).*
- **Source:**  
  - `outputs/benchmarks/te_qpinn_memory_confirmatory_10seed/multiseed_stats.md`  
  - `outputs/benchmarks/te_qpinn_memory_confirmatory_10seed/summary.csv`
- **Exact values:**

| Variant | Final L2 (mean±std) | Final Linf (mean±std) | Final Loss (mean±std) | Runtime (s, mean±std) | Params |
| --- | ---: | ---: | ---: | ---: | ---: |
| Classical + PI | 0.922830 ± 0.076989 | 1.043477 ± 0.122118 | 44.012908 ± 12.667057 | 14.7715 ± 1.1693 | 2241 |
| Classical + PI + analytic memory | 0.752910 ± 0.188977 | 0.810768 ± 0.217877 | 37.575163 ± 16.695402 | 17.7416 ± 1.6117 | 2433 |
| TE fixed residual 0.10 + PI | 0.955529 ± 0.070070 | 1.052766 ± 0.092046 | 47.486260 ± 14.141845 | 42.4348 ± 2.8408 | 2306 |
| TE LayerNorm post_quantum + PI | 0.918204 ± 0.076402 | 1.025404 ± 0.117049 | 41.205865 ± 10.416957 | 44.6977 ± 3.0572 | 2338 |
| TE memory-aware analytic + PI | 0.809505 ± 0.176401 | 1.008622 ± 0.232580 | 26.643364 ± 13.316055 | 45.0615 ± 3.4201 | 2498 |

- **Claim supported:** Analytic memory features are strongly beneficial on this benchmark; TE-memory improves over non-memory TE baselines, but strongest overall performance/efficiency conclusions still favor classical+memory.
- **Layout:** **Double-column**.

---

### Table B — Statistical Test Summary (10-Seed Confirmatory)
- **Suggested caption:** *Paired statistical comparisons for final L2 on the 10-seed confirmatory benchmark.*
- **Source:**  
  - `outputs/benchmarks/te_qpinn_memory_confirmatory_10seed/statistical_tests.md`  
  - `outputs/benchmarks/te_qpinn_memory_confirmatory_10seed/statistical_tests.json`
- **Exact values:**

| Comparison | Mean diff (A-B) | Median diff (A-B) | Wilcoxon p (two-sided) | Wilcoxon p (A<B) | Bootstrap 95% CI | Paired d | Cliff’s δ | Wins A |
| --- | ---: | ---: | ---: | ---: | --- | ---: | ---: | ---: |
| Classical + memory vs Classical | -0.169920 | -0.183730 | 0.0371094 | 0.0185547 | [-0.292771, -0.049636] | -0.8321 | -0.6400 | 9/10 |
| TE memory vs TE LayerNorm non-memory | -0.108698 | -0.089811 | 0.160156 | 0.0800781 | [-0.227282, 0.004445] | -0.5493 | -0.3400 | 6/10 |
| TE memory vs TE fixed | -0.146024 | -0.136400 | 0.130859 | 0.0654297 | [-0.266543, -0.030561] | -0.7181 | -0.4800 | 6/10 |
| TE memory vs Classical + memory | +0.056595 | +0.116844 | 0.695312 | 0.6875 | [-0.092482, 0.195866] | +0.2301 | +0.1200 | 3/10 |

- **Claim supported:** Classical+memory improvement is statistically supported; TE-memory trend vs TE non-memory is directional but weaker; TE-specific superiority over classical+memory is not established.
- **Layout:** **Double-column**.

---

### Table C — Alpha/Beta Robustness (alpha=0.7, beta=0.3)
- **Suggested caption:** *Five-seed robustness results at altered fractional parameters (`alpha=0.7`, `beta=0.3`).*
- **Source:**  
  - `outputs/benchmarks/te_qpinn_memory_alpha07_beta03_5seed/multiseed_stats.md`  
  - `outputs/benchmarks/te_qpinn_memory_alpha07_beta03_5seed/statistical_tests.md`
- **Exact values:**

| Variant | Final L2 (mean±std) | Final Linf (mean±std) | Final Loss (mean±std) | Runtime (s, mean±std) | Params |
| --- | ---: | ---: | ---: | ---: | ---: |
| Classical + PI | 0.935119 ± 0.053208 | 1.055873 ± 0.121808 | 40.927904 ± 12.203974 | 15.7726 ± 2.0796 | 2241 |
| Classical + PI + analytic memory | 0.885903 ± 0.183472 | 0.955070 ± 0.236414 | 38.573208 ± 13.828171 | 18.1548 ± 0.6639 | 2433 |
| TE fixed residual 0.10 + PI | 1.025442 ± 0.114842 | 1.121506 ± 0.134984 | 102.662733 ± 143.959989 | 43.5231 ± 1.3878 | 2306 |
| TE LayerNorm post_quantum + PI | 0.985235 ± 0.113020 | 1.070939 ± 0.107036 | 57.117036 ± 46.812099 | 45.9477 ± 1.1748 | 2338 |
| TE memory-aware analytic + PI | 1.087911 ± 0.167052 | 1.218205 ± 0.227703 | 231.438177 ± 441.600714 | 46.4354 ± 1.3434 | 2498 |

- **Claim supported:** Memory feature benefit transfers more clearly on the classical track here; TE-memory does not generalize strongly at this setting.
- **Layout:** **Double-column**.

---

### Table D — Ablation Summary (Seed-42 Diagnostic)
- **Suggested caption:** *Seed-42 ablation diagnostics for TE-QPINN surrogate components under locked Adam-only budget.*
- **Source:**  
  - `outputs/benchmarks/te_qpinn_ablation/ablation_stats.md`  
  - `outputs/benchmarks/te_qpinn_ablation/ablation_stats.json`
- **Exact values (selected key rows):**

| Variant | Final L2 | Final Linf | Final Loss | Runtime (s) | Params |
| --- | ---: | ---: | ---: | ---: | ---: |
| Classical + PI | 1.003810 | 1.085777 | 62.210301 | 17.2412 | 2241 |
| Full TE-QPINN + PI | 0.886842 | 0.946181 | 71.443514 | 46.3441 | 2306 |
| TE residual scale 0.20 + PI | 0.827545 | 0.869191 | 70.052411 | 44.3508 | 2306 |
| TE no residual + PI | 0.875587 | 1.193847 | 49.151477 | 41.7829 | 2273 |
| TE large embedding + PI | 0.924175 | 1.056717 | 53.545564 | 43.9399 | 3062 |

- **Claim supported:** TE performance is highly component-sensitive; single-seed gains from specific settings require multi-seed confirmation.
- **Layout:** **Double-column** (or split into two single-column tables if needed).

---

### Table E — Optimizer Sensitivity (Seed-42, Extended Budget)
- **Suggested caption:** *Adam vs Adam+LBFGS optimizer sensitivity (seed-42); Adam+LBFGS is extended-budget, not equal-runtime.*
- **Source:**  
  - `outputs/benchmarks/te_qpinn_optimizer_sensitivity/optimizer_sensitivity_stats.md`  
  - `outputs/benchmarks/te_qpinn_optimizer_sensitivity/summary.csv`
- **Exact values:**

| Variant | Adam L2 | Adam+LBFGS L2 | Adam Linf | Adam+LBFGS Linf | Adam Loss | Adam+LBFGS Loss | Adam Runtime (s) | Adam+LBFGS Runtime (s) |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| Classical + PI | 1.003810 | 0.085135 | 1.085777 | 0.260378 | 62.210301 | 0.161102 | 15.7019 | 549.7721 |
| TE fixed + PI | 0.886842 | 0.059035 | 0.946181 | 0.209135 | 71.443514 | 0.081750 | 40.6095 | 1589.7988 |
| TE LayerNorm + PI | 0.823901 | 0.035359 | 0.957646 | 0.107435 | 54.057441 | 0.129883 | 41.8441 | 1697.6655 |

- **Claim supported:** All models improve sharply with Adam+LBFGS, but with very large runtime increase; TE appears optimizer-sensitive.
- **Layout:** **Double-column**.

---

### Table F — Exact PennyLane TE-QPINN Feasibility (Tiny Smoke)
- **Suggested caption:** *Exact PennyLane TE-QPINN feasibility result (tiny smoke): differentiability confirmed, performance/runtime bottleneck observed.*
- **Source:**  
  - `outputs/benchmarks/te_qpinn_exact_pqc_smoke/benchmark_report.md`  
  - `docs/te_qpinn_exact_pqc_notes.md`
- **Exact values:**

| Variant | Params | Runtime (s) | Final Loss | Final L2 | Final Linf |
| --- | ---: | ---: | ---: | ---: | ---: |
| Classical + PI (tiny) | 337 | 1.6793 | 23.8083 | 0.930595 | 0.968080 |
| Exact TE-QPINN PennyLane + PI (tiny) | 251 | 1348.6111 | 206.3259 | 3.918995 | 2.718204 |

- **Claim supported:** Exact PQC path is feasible and differentiable, but currently impractical for large-scale training in this implementation.
- **Layout:** **Single-column**.

---

### Table G — Scalar Sparse CSI Demo Summary
- **Suggested caption:** *Scalar synthetic sparse CSI forecasting summary (interpolation and forecast).*
- **Source:**  
  - `outputs/applications/wireless_channel_demo/application_summary_table.csv`  
  - `outputs/applications/wireless_channel_demo/metrics.json`
- **Exact values:**

| Mode | Best Model | Mean Relative L2 | Mean MSE | Mean Max Error |
| --- | --- | ---: | ---: | ---: |
| Interpolation | MLP + memory + domain | 0.104616 | 0.006813 | 0.225622 |
| Forecast | MLP + memory + domain | 0.249553 | 0.037874 | 0.260614 |

- **Claim supported:** Combined memory+domain priors are strongest for scalar synthetic sparse CSI under the current sweep.
- **Layout:** **Single-column**.

---

### Table H — Two-Channel / MIMO-Style Sparse CSI Demo Summary
- **Suggested caption:** *Two-channel synthetic sparse CSI forecasting summary (aggregate metrics across channels).*
- **Source:**  
  - `outputs/applications/wireless_channel_demo/mimo_application_report.md`  
  - `outputs/applications/wireless_channel_demo/mimo_metrics.json`
- **Exact values (best-by-mode rows):**

| Mode | Best Model | Relative L2 | MSE | Max Error | Ch1 Rel L2 | Ch2 Rel L2 |
| --- | --- | ---: | ---: | ---: | ---: | ---: |
| Interpolation | MLP + domain sin/cos | 0.342656 | 0.069603 | 0.928031 | 0.112648 | 0.514121 |
| Forecast | MLP + memory + domain | 0.563666 | 0.157690 | 1.107839 | 0.223173 | 0.732560 |

- **Claim supported:** Domain priors dominate two-channel interpolation; combined memory+domain is best for two-channel forecast.
- **Layout:** **Single-column**.

---

## 6. Key Figures to Include (Publication-Oriented Selection)

### Figure Inventory (with placement guidance)

1. **`outputs/plots/key_results/memory_confirmatory_10seed_summary.png`**  
   - Paper filename: `fig_confirmatory10_summary.png`  
   - Caption: *Ten-seed confirmatory summary for classical, TE, and memory-aware variants (mean performance, runtime, params).*  
   - Why it matters: Main evidence figure for final claims.  
   - Column: **Double-column**.

2. **`outputs/plots/key_results/alpha07_beta03_summary.png`**  
   - Paper filename: `fig_alpha07_beta03_summary.png`  
   - Caption: *Robustness summary at altered fractional setting (`alpha=0.7`, `beta=0.3`).*  
   - Why it matters: Shows transfer limitations and robustness behavior.  
   - Column: **Double-column**.

3. **`outputs/plots/key_results/optimizer_sensitivity_summary.png`**  
   - Paper filename: `fig_optimizer_sensitivity_summary.png`  
   - Caption: *Seed-42 optimizer sensitivity: Adam vs Adam+LBFGS (extended-budget).*  
   - Why it matters: Demonstrates optimizer dependence and runtime tradeoff.  
   - Column: **Double-column**.

4. **`outputs/plots/key_results/best_te_vs_classical_error_heatmap.png`**  
   - Paper filename: `fig_error_heatmap_classical_vs_te.png`  
   - Caption: *Representative absolute-error field comparison: classical baseline vs selected TE variant.*  
   - Why it matters: Spatial error structure, not only scalar metrics.  
   - Column: **Single-column** (or double if text is dense).

5. **`outputs/applications/wireless_channel_demo/sparse_csi_forecast_showcase.png`**  
   - Paper filename: `fig_sparse_csi_showcase.png`  
   - Caption: *Scalar sparse CSI forecast showcase with train points, forecast split, and model overlays.*  
   - Why it matters: Application narrative anchor figure.  
   - Column: **Double-column**.

6. **`outputs/applications/wireless_channel_demo/mimo_interpolation_predictions.png`**  
   - Paper filename: `fig_mimo_interp_predictions.png`  
   - Caption: *Two-channel interpolation predictions for synthetic MIMO-style link.*  
   - Why it matters: Shows multi-output extension behavior.  
   - Column: **Double-column**.

7. **`outputs/applications/wireless_channel_demo/mimo_forecast_predictions.png`**  
   - Paper filename: `fig_mimo_forecast_predictions.png`  
   - Caption: *Two-channel forecast predictions with future-window split.*  
   - Why it matters: Multi-channel extrapolation difficulty and model behavior.  
   - Column: **Double-column**.

8. **`outputs/applications/wireless_channel_demo/error_comparison.png`**  
   - Paper filename: `fig_scalar_error_time.png`  
   - Caption: *Scalar absolute-error trajectories over interpolation and forecast windows.*  
   - Why it matters: Error dynamics across time.  
   - Column: **Single-column**.

9. **`outputs/applications/wireless_channel_demo/robustness_heatmap_or_bars.png`**  
   - Paper filename: `fig_scalar_robustness_sweep.png`  
   - Caption: *Scalar robustness sweep across noise/sparsity settings.*  
   - Why it matters: Application robustness evidence.  
   - Column: **Double-column**.

10. **Optional appendix figure:** `outputs/plots/te_qpinn_exact_pqc_smoke/summary_panels.png`  
    - Paper filename: `fig_exact_pqc_smoke_summary.png`  
    - Caption: *Exact PennyLane TE-QPINN tiny-smoke feasibility summary (runtime bottleneck).*  
    - Why it matters: Supports feasibility appendix.  
    - Column: **Single-column**.

---

## 7. Results Narrative (Publication-Style Draft Backbone)

1. Under Adam-only confirmatory evaluation, **Classical + analytic memory** is strongest overall (best mean L2/Linf among finalists with much lower runtime than TE variants).
2. **TE memory-aware analytic** improves over non-memory TE baselines in the base setting (10-seed means), especially on L2 and loss, but does not surpass classical+memory.
3. **LayerNorm (post-quantum)** is a useful TE-side stabilization versus fixed TE (especially L2 trend), but not sufficient for overall superiority.
4. **Residual-scale 0.20** and **gated residual** improvements seen in single-seed diagnostics did not generalize reliably.
5. Robustness at `(alpha,beta)=(0.7,0.3)` indicates memory benefits transfer mainly on the classical track; TE-memory does not generalize strongly there.
6. Exact PennyLane TE-QPINN is technically integrated and differentiable (`u_x`, `u_xx`, gradients), but current tiny-smoke runtime/accuracy is poor; this is a feasibility/bottleneck result.
7. In synthetic sparse CSI forecasting, domain priors are strongly effective; combined memory+domain features are best in scalar and forecast-focused settings, with mixed memory-only benefits.

---

## 8. Application Section Inventory (Scalar + Two-Channel)

### Problem framing
- **Scalar task:** synthetic channel response `h(t)` with Doppler-like oscillation, delayed component, drift, and noise.  
- **Two-channel extension:** `h(t)=[h1(t), h2(t)]` with related channel dynamics and optional coupling.
- Modes: **interpolation** and **forecast**.
- Fairness: same split, seed, optimizer, epochs, architecture budget across model variants.

### Models compared
- Linear baseline
- AR(1) baseline
- MLP baseline `[t]`
- MLP + memory-only
- MLP + domain sinusoidal
- MLP + memory + domain

### Scalar best results (current Phase C sweep)
- Interpolation best: **MLP + memory + domain**, rel L2 `0.104616`
- Forecast best: **MLP + memory + domain**, rel L2 `0.249553`
- Source: `outputs/applications/wireless_channel_demo/application_summary_table.csv`

### Two-channel best results
- Interpolation best: **MLP + domain sin/cos**, rel L2 `0.342656`
- Forecast best: **MLP + memory + domain**, rel L2 `0.563666`
- Source: `outputs/applications/wireless_channel_demo/mimo_application_report.md`

### Key application artifacts
- `outputs/applications/wireless_channel_demo/application_report.md`
- `outputs/applications/wireless_channel_demo/mimo_application_report.md`
- `outputs/applications/wireless_channel_demo/sparse_csi_forecast_showcase.png`
- `outputs/applications/wireless_channel_demo/mimo_interpolation_predictions.png`
- `outputs/applications/wireless_channel_demo/mimo_forecast_predictions.png`
- `outputs/applications/wireless_channel_demo/error_comparison.png`
- `outputs/applications/wireless_channel_demo/robustness_heatmap_or_bars.png`

### Limitation phrasing (recommended)
- “This is a synthetic application-inspired benchmark, not a full LEO/MIMO simulator and not a deployment-level channel-prediction claim.”

---

## 9. Exact PennyLane TE-QPINN Feasibility Inventory

### Implementation and status
- Model: `src/exact_te_qpinn_pennylane.py`
- Model types: `te_qpinn_pennylane`, `exact_te_qpinn_pennylane`, `exact_pqc_te_qpinn`
- Dependency spec: `requirements.txt` has `pennylane>=0.42.0`

### Verified capability status
- Forward pass works in benchmark pipeline.
- Autograd supports `u_x` and `u_xx` (tiny-batch verification in tests and notes).
- Gradients flow to embedding FNN and PQC variational parameters.

### Feasibility metrics
- Source: `outputs/benchmarks/te_qpinn_exact_pqc_smoke/benchmark_report.md`
- Classical tiny: L2 `0.930595`, runtime `1.6793s`
- Exact PQC tiny: L2 `3.918995`, runtime `1348.6111s`

### Recommended framing
- Feasibility appendix result: “technically integrated and differentiable, currently computationally impractical and underperforming in naive tiny smoke.”

---

## 10. Claims Supported by Evidence

1. Analytic memory features significantly improve the classical baseline in 10-seed confirmatory evaluation.
2. TE memory-aware analytic improves over TE non-memory baselines in the base setting on mean L2/loss (directionally and in win counts).
3. TE-specific superiority over classical+memory is **not** established.
4. LayerNorm improves TE behavior relative to fixed TE, but classical remains stronger in efficiency and mean accuracy under locked Adam-only budgets.
5. Exact simulated PQC TE-QPINN is feasible and differentiable but currently too slow and less accurate in tiny-smoke conditions.
6. In synthetic sparse CSI tasks, domain-informed and combined priors improve forecasting quality under equal training budgets.

---

## 11. Claims to Avoid

1. “Quantum advantage” or “quantum speedup.”
2. “TE-QPINN universally outperforms classical PINNs.”
3. “Exact PennyLane TE-QPINN is practical at current scale.”
4. “Deployment-ready LEO/MIMO channel predictor.”
5. “Real-world superiority” from synthetic-only application demos.
6. Strong statistical-significance claims where p-values/CIs do not support them.

---

## 12. Recommended Figure Order (Publication Flow)

1. `fig_confirmatory10_summary.png`
2. `fig_alpha07_beta03_summary.png`
3. `fig_optimizer_sensitivity_summary.png`
4. `fig_error_heatmap_classical_vs_te.png`
5. `fig_exact_pqc_smoke_summary.png` (appendix or main Section VIII)
6. `fig_sparse_csi_showcase.png`
7. `fig_mimo_interp_predictions.png`
8. `fig_mimo_forecast_predictions.png`
9. `fig_scalar_error_time.png`
10. `fig_scalar_robustness_sweep.png`

---

## 13. Recommended Table Order (Publication Flow)

1. Table A — 10-seed confirmatory benchmark
2. Table B — statistical tests (confirmatory)
3. Table C — alpha/beta robustness
4. Table D — ablation diagnostics
5. Table E — optimizer sensitivity (extended-budget note)
6. Table F — exact PQC feasibility
7. Table G — scalar CSI demo
8. Table H — two-channel CSI demo

---

## 14. Report Writing Notes (To Look Like a Publication)

1. Keep abstract concise: problem, method families, key outcomes, caveats.
2. Put contribution bullets at end of Introduction.
3. State fairness protocol explicitly (same seeds, points, budgets, metrics).
4. Place tables immediately after corresponding claims.
5. Use captions that contain a finding, not only a description.
6. Separate Adam-only fair comparisons from Adam+LBFGS extended-budget comparisons.
7. Keep exact PQC in a feasibility/bottleneck section, not as a core performance claim.
8. Frame application as transfer/feasibility evidence, not deployment evidence.
9. Use limitations and non-claims explicitly in Discussion.

---

## 15. Suggested References / Citation Placeholders

Use placeholders until bibliography is finalized:

- `[PINNs]` Original Physics-Informed Neural Networks paper.
- `[FracCalc]` Standard fractional calculus text (Caputo derivative definitions).
- `[L1Scheme]` L1 discretization references for Caputo derivatives.
- `[ProductIntegration]` Product-integration methods for weakly singular Volterra terms.
- `[FracPINN]` Fractional PINN references for time-fractional PDE/integro-differential settings.
- `[QNN]` General quantum neural network / variational quantum circuit references.
- `[QPINN]` Quantum PINN references.
- `[TEQPINN]` Trainable Embedding QPINN paper (Scientific Reports 2025).
- `[CSI]` Wireless channel state information prediction references.
- `[MIMO]` MIMO channel modeling/prediction references.
- `[LEO]` LEO satellite channel dynamics / forecasting motivation references.

Note: MidSem content linkage (Caputo/L1/graded mesh/product integration/manufactured solution) should be retained in Sections II–III as foundational methodology, while Sections VII–IX emphasize new final-phase experimental contributions.

---

## Reproducibility Command Index (Quick Copy)

Core tests:
```bash
pytest -q
```

Benchmark runs:
```bash
python scripts/benchmark_te_qpinn.py --benchmark-config configs/benchmark_te_qpinn_multiseed.yaml
python scripts/benchmark_te_qpinn.py --benchmark-config configs/benchmark_te_qpinn_layernorm_multiseed.yaml
python scripts/benchmark_te_qpinn.py --benchmark-config configs/benchmark_te_qpinn_optimizer_sensitivity.yaml --resume-incomplete
python scripts/benchmark_te_qpinn.py --benchmark-config configs/benchmark_te_qpinn_memory_multiseed.yaml --resume-incomplete
python scripts/benchmark_te_qpinn.py --benchmark-config configs/benchmark_te_qpinn_memory_confirmatory_10seed.yaml --resume-incomplete
python scripts/benchmark_te_qpinn.py --benchmark-config configs/benchmark_te_qpinn_memory_alpha07_beta03_5seed.yaml --resume-incomplete
python scripts/benchmark_te_qpinn.py --benchmark-config configs/benchmark_te_qpinn_exact_pqc_smoke.yaml --resume-incomplete
```

Statistical post-processing:
```bash
python scripts/generate_confirmatory_stats.py \
  --summary-csv outputs/benchmarks/te_qpinn_memory_confirmatory_10seed/summary.csv \
  --output-json outputs/benchmarks/te_qpinn_memory_confirmatory_10seed/statistical_tests.json \
  --output-md outputs/benchmarks/te_qpinn_memory_confirmatory_10seed/statistical_tests.md
```

Application demo:
```bash
python scripts/run_wireless_channel_demo.py
```
