# Quantum-Readiness Benchmark Dashboard

This page is the fast reviewer view for benchmark outcomes on this branch.

## Scope

- Workload: fractional integro-differential PINN benchmark family
- Compared models: classical PINN vs quantum-ready PINN (classical emulator)
- Budget: matched epoch count within each scenario
- Device: CUDA

## Headline outcome

Across multi-seed medium and harder suites, the classical baseline is still more reliable and more accurate on average. The tuned quantum-ready setup shows a strong single-seed improvement (seed 42), but multi-seed confirmation is still pending.

## Aggregate table

| Scenario | Seeds | Model | Runtime (s) | Final L2 | Final Linf |
| --- | --- | --- | ---: | ---: | ---: |
| Medium | 42, 123, 999 | classical | 43.6595 | 0.8193 | 0.9940 |
| Medium | 42, 123, 999 | quantum-ready | 84.2197 | 2.5431 | 2.0376 |
| Harder | 42, 123, 999 | classical | 95.6251 | 1.1096 | 1.3778 |
| Harder | 42, 123, 999 | quantum-ready | 101.2677 | 2.5526 | 2.2807 |
| Harder tuned | 42 | classical | 90.9617 | 1.1721 | 1.3487 |
| Harder tuned | 42 | quantum-ready | 72.2919 | 1.0281 | 1.3140 |

## Ratio highlights

- Medium runtime ratio (quantum-ready / classical): 1.9290
- Medium final L2 ratio (quantum-ready / classical): 3.1042
- Harder runtime ratio (quantum-ready / classical): 1.0590
- Harder final L2 ratio (quantum-ready / classical): 2.3004
- Harder tuned seed-42 runtime ratio (quantum-ready / classical): 0.7948
- Harder tuned seed-42 final L2 ratio (quantum-ready / classical): 0.8772

## Figure quick links

- Medium summary: [medium_summary_panels.png](figures/medium_summary_panels.png)
- Medium ratios: [medium_ratio_panel.png](figures/medium_ratio_panel.png)
- Harder summary: [harder_summary_panels.png](figures/harder_summary_panels.png)
- Harder ratios: [harder_ratio_panel.png](figures/harder_ratio_panel.png)
- Harder tuned seed-42 summary: [harder_tuned_seed42_summary_panels.png](figures/harder_tuned_seed42_summary_panels.png)
- Harder tuned seed-42 ratios: [harder_tuned_seed42_ratio_panel.png](figures/harder_tuned_seed42_ratio_panel.png)

## Evidence paths

- Medium report: [outputs/benchmarks/quantum_ready_medium/benchmark_report.md](../../outputs/benchmarks/quantum_ready_medium/benchmark_report.md)
- Harder report: [outputs/benchmarks/quantum_ready_harder/benchmark_report.md](../../outputs/benchmarks/quantum_ready_harder/benchmark_report.md)
- Harder tuned seed-42 report: [outputs/benchmarks/quantum_ready_harder_tuned_seed42/benchmark_report.md](../../outputs/benchmarks/quantum_ready_harder_tuned_seed42/benchmark_report.md)
- Journal log: [docs/quantum-readiness/JOURNAL.md](JOURNAL.md)

## Reviewer-ready claim language

This branch delivers a complete quantum-readiness benchmark framework with reproducible configs, seeded comparisons, aggregate metrics, and publication-style plots. Current results do not yet support a general performance win for the quantum-ready architecture; gains appear conditional and currently strongest in the tuned single-seed probe.
