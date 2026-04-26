# Quantum-Readiness Benchmark Dossier

This directory is the organized benchmark package for the `feature/quantum-readiness` branch.

## What is in this folder

- `JOURNAL.md`: experiment log in chronological, lab-notebook style.
- `figures/`: publication-oriented comparison plots copied from completed benchmark runs.

## Benchmark runs captured

1. Medium preset
- Config: `configs/benchmark_quantum_ready_medium.yaml`
- Seeds: 42, 123, 999
- Result: classical baseline wins on runtime and average error.

2. Harder oscillatory preset
- Config: `configs/benchmark_quantum_ready_harder.yaml`
- Seeds: 42, 123, 999
- Result: classical baseline still wins; quantum-ready remains seed-sensitive.

3. Harder tuned preset (single-seed validation)
- Config: `configs/benchmark_quantum_ready_harder_tuned.yaml`
- Seeds: 42
- Result: quantum-ready improves strongly on this seed, but this is not yet a multi-seed win.

## Metrics snapshot

| Scenario | Model | Runtime (s) | Final L2 | Final Linf |
| --- | --- | ---: | ---: | ---: |
| Medium (3 seeds) | classical | 43.6595 | 0.8193 | 0.9940 |
| Medium (3 seeds) | quantum-ready | 84.2197 | 2.5431 | 2.0376 |
| Harder (3 seeds) | classical | 95.6251 | 1.1096 | 1.3778 |
| Harder (3 seeds) | quantum-ready | 101.2677 | 2.5526 | 2.2807 |
| Harder tuned (seed 42) | classical | 90.9617 | 1.1721 | 1.3487 |
| Harder tuned (seed 42) | quantum-ready | 72.2919 | 1.0281 | 1.3140 |

## Figure map

- `figures/medium_summary_panels.png`
- `figures/medium_ratio_panel.png`
- `figures/harder_summary_panels.png`
- `figures/harder_ratio_panel.png`
- `figures/harder_tuned_seed42_summary_panels.png`
- `figures/harder_tuned_seed42_ratio_panel.png`

## Repro commands

```bash
python scripts/benchmark_quantum_ready.py --config configs/benchmark_quantum_ready_medium.yaml --models classical quantum_ready --seeds 42 123 999 --epochs 20 --device cuda --output-dir outputs/benchmarks/quantum_ready_medium
python scripts/benchmark_quantum_ready.py --config configs/benchmark_quantum_ready_harder.yaml --models classical quantum_ready --seeds 42 123 999 --epochs 20 --device cuda --output-dir outputs/benchmarks/quantum_ready_harder
python scripts/benchmark_quantum_ready.py --config configs/benchmark_quantum_ready_harder_tuned.yaml --models classical quantum_ready --seeds 42 --epochs 20 --device cuda --output-dir outputs/benchmarks/quantum_ready_harder_tuned_seed42
```

## Interpretation boundary

The `quantum_ready` model in this branch is a quantum-inspired classical emulator, not a hardware quantum backend. The results are architecture-comparison results under matched training budgets.
