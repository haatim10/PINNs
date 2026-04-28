# Key Results (Curated)

This folder contains a compact, presentation-oriented subset of the most informative plots.

Full raw outputs remain under `outputs/plots/` for complete reproducibility.

## Plot Index

- `optimizer_sensitivity_summary.png`  
  Source: `outputs/plots/te_qpinn_optimizer_sensitivity/summary_panels.png`  
  Shows seed-42 Adam vs Adam+LBFGS outcome across Classical/TE variants and highlights the extended-budget runtime tradeoff.

- `optimizer_sensitivity_convergence.png`  
  Source: `outputs/plots/te_qpinn_optimizer_sensitivity/convergence_seed_42.png`  
  Shows convergence behavior for optimizer sensitivity variants on seed 42.

- `layernorm_multiseed_summary.png`  
  Source: `outputs/plots/te_qpinn_layernorm_multiseed/summary_panels.png`  
  Supports the conclusion that post-quantum LayerNorm improved TE vs fixed TE over 5 seeds but did not surpass Classical overall.

- `memory_smoke_summary.png`  
  Source: `outputs/plots/te_qpinn_memory_smoke/summary_panels.png`  
  Summarizes the Phase 8C memory-aware seed-42 smoke comparison.

- `memory_smoke_error_heatmap.png`  
  Source: `outputs/plots/te_qpinn_memory_smoke/seed_42_te_memory_analytic_layernorm_pi_abs_error_heatmap.png`  
  Shows absolute error field for the memory-aware TE + LayerNorm smoke variant.

- `memory_multiseed_summary.png`  
  Source: `outputs/plots/te_qpinn_memory_multiseed/summary_panels.png`  
  Summarizes the locked 5-seed memory-aware validation and enables direct comparison of mean performance, runtime, and parameter count across classical, TE, and memory-augmented variants.

- `memory_multiseed_error_heatmap.png`  
  Source: `outputs/plots/te_qpinn_memory_multiseed/error_heatmaps_seed_0_classical_vs_te_qpinn.png`  
  Representative side-by-side absolute-error heatmap comparison from the memory multi-seed run (seed 0 view).

- `best_te_vs_classical_error_heatmap.png`  
  Source: `outputs/plots/te_qpinn_memory_smoke/error_heatmaps_seed_42_classical_vs_te_qpinn.png`  
  Side-by-side absolute-error heatmaps for Classical + PI versus the best TE variant selected for that seed.

## Important Scope Note

Memory-aware figures now include both:

- **single-seed smoke** diagnostics (seed 42), and
- **locked 5-seed validation** artifacts.

The smoke results remain preliminary by themselves; the five-seed summary should be used for stronger conclusions.
