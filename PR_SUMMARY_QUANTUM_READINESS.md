# PR Summary: Quantum-Readiness Infrastructure + Comparative Benchmarks

## Summary

This PR turns the branch into a complete quantum-readiness experimentation track for PINNs. It adds model abstraction, a quantum-inspired hybrid path, configurable oscillatory solution families, benchmark reporting, and reviewer-facing documentation.

The benchmark conclusion is intentionally conservative: the infrastructure is production-ready for experiments, but the current quantum-ready emulator is not yet a consistent multi-seed performance win over the classical baseline.

## What This PR Adds

### Model and integration layer
- `src/model_factory.py` for config-driven model selection (`classical`, `quantum_ready`, alias support).
- `src/quantum_ready_model.py` implementing the hybrid quantum-inspired block while preserving the same forward contract.
- Wiring updates in training and analysis scripts to route through the shared factory:
  - `scripts/train_integro_diff.py`
  - `scripts/evaluate.py`
  - `scripts/visualize.py`

### Physics/solution generalization
- Configurable exact solution support for the integro-differential workflow in `src/physics_integro.py`.
- Shared solution configuration consumed by training/evaluation/visualization paths.

### Benchmark framework
- Extended benchmark runner in `scripts/benchmark_quantum_ready.py` producing:
  - `benchmark_results.json`
  - `benchmark_summary.csv`
  - `benchmark_aggregate.json`
  - convergence plots
  - ratio/summary panels
  - markdown benchmark report

### Configs for reproducible scenarios
- `configs/benchmark_quantum_ready_medium.yaml`
- `configs/benchmark_quantum_ready_harder.yaml`
- `configs/benchmark_quantum_ready_harder_tuned.yaml`
- `configs/benchmark_quantum_ready_large.yaml`

### Tests
- `tests/test_model_factory.py`
- `tests/test_integro_solution.py`

### Organized reviewer documentation
- `docs/quantum-readiness/README.md`
- `docs/quantum-readiness/JOURNAL.md`
- `docs/quantum-readiness/DASHBOARD.md`
- `docs/quantum-readiness/figures/*`

## Benchmark Results (Current Evidence)

### Medium suite (3 seeds)
- Config: `configs/benchmark_quantum_ready_medium.yaml`
- Classical: runtime 43.6595 s, final L2 0.8193, final Linf 0.9940
- Quantum-ready: runtime 84.2197 s, final L2 2.5431, final Linf 2.0376
- Result: classical wins on runtime and average accuracy.

### Harder oscillatory suite (3 seeds)
- Config: `configs/benchmark_quantum_ready_harder.yaml`
- Classical: runtime 95.6251 s, final L2 1.1096, final Linf 1.3778
- Quantum-ready: runtime 101.2677 s, final L2 2.5526, final Linf 2.2807
- Result: classical still wins; quantum-ready remains seed-sensitive.

### Harder tuned probe (single seed)
- Config: `configs/benchmark_quantum_ready_harder_tuned.yaml`
- Seed: 42
- Classical: runtime 90.9617 s, final L2 1.1721, final Linf 1.3487
- Quantum-ready: runtime 72.2919 s, final L2 1.0281, final Linf 1.3140
- Result: tuned quantum-ready improves strongly on this seed; multi-seed confirmation still pending.

## Validation

- `pytest -q tests/test_integro_solution.py tests/test_model_factory.py`
- Result: 11 passed

## Repro Commands

```bash
python scripts/benchmark_quantum_ready.py --config configs/benchmark_quantum_ready_medium.yaml --models classical quantum_ready --seeds 42 123 999 --epochs 20 --device cuda --output-dir outputs/benchmarks/quantum_ready_medium
python scripts/benchmark_quantum_ready.py --config configs/benchmark_quantum_ready_harder.yaml --models classical quantum_ready --seeds 42 123 999 --epochs 20 --device cuda --output-dir outputs/benchmarks/quantum_ready_harder
python scripts/benchmark_quantum_ready.py --config configs/benchmark_quantum_ready_harder_tuned.yaml --models classical quantum_ready --seeds 42 --epochs 20 --device cuda --output-dir outputs/benchmarks/quantum_ready_harder_tuned_seed42
```

## Recommended Follow-up

1. Complete full 3-seed run for the tuned harder config.
2. Add confidence intervals via repeated runs.
3. Run parameter-matched ablations for tighter fairness claims.
4. Keep the model-factory interface stable for a future true quantum backend.
