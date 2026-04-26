# Quantum-Readiness Experiment Journal

This file records benchmark work in chronological order with hypothesis, setup, result, and decision.

## Entry 1: Build a fast but meaningful baseline comparison

Date: 2026-04-26

Hypothesis:
A medium-size run with three seeds should be fast enough for iteration and still reveal stability/performance trends.

Setup:
- Config: `configs/benchmark_quantum_ready_medium.yaml`
- Grid: 100 x 100
- Epochs: 20
- Seeds: 42, 123, 999
- Device: CUDA
- Models: classical, quantum-ready

Observed:
- Classical: runtime 43.6595 s, final L2 0.8193, final Linf 0.9940.
- Quantum-ready: runtime 84.2197 s, final L2 2.5431, final Linf 2.0376.
- One quantum-ready seed produced a large outlier, increasing variance.

Decision:
Keep medium as core comparison case and add a harder oscillatory case to test whether the quantum-inspired mixing block gains relative advantage.

## Entry 2: Stress-test with harder oscillatory solution family

Date: 2026-04-26

Hypothesis:
A harder oscillatory target may better match the intended inductive bias of the quantum-inspired block.

Setup:
- Config: `configs/benchmark_quantum_ready_harder.yaml`
- Solution family: higher spatial frequency and lower time power
- Epochs: 20
- Seeds: 42, 123, 999
- Device: CUDA

Observed:
- Classical: runtime 95.6251 s, final L2 1.1096, final Linf 1.3778.
- Quantum-ready: runtime 101.2677 s, final L2 2.5526, final Linf 2.2807.
- Quantum-ready remained more variable across seeds and did not close the error gap.

Decision:
Tune the quantum-ready block to reduce instability and retest.

## Entry 3: Tune quantum-ready block aggressiveness

Date: 2026-04-26

Hypothesis:
Reducing qubit count, depth, and entanglement strength should improve optimization stability.

Setup:
- Config: `configs/benchmark_quantum_ready_harder_tuned.yaml`
- Changes: reduced quantum block size and interaction strength
- Epochs: 20
- Seed: 42 (single-seed probe)
- Device: CUDA

Observed:
- Classical: runtime 90.9617 s, final L2 1.1721, final Linf 1.3487.
- Quantum-ready: runtime 72.2919 s, final L2 1.0281, final Linf 1.3140.
- This is a clear single-seed improvement versus both classical and untuned quantum-ready for this seed.

Decision:
Treat this as promising but preliminary. Multi-seed confirmation is still required before making a broad claim.

## Entry 4: Branch-level conclusion

Date: 2026-04-26

Conclusion:
The branch now contains a complete quantum-readiness benchmarking workflow with reproducible configs, aggregate metrics, ratio plots, and markdown reports. Current evidence supports the benchmark infrastructure strongly, while model-performance claims remain conservative: the classical baseline still wins on the multi-seed medium and harder suites, and the tuned variant shows targeted promise that needs full multi-seed confirmation.

## Next experiment queue

1. Complete full three-seed run for `benchmark_quantum_ready_harder_tuned.yaml`.
2. Add confidence intervals over multiple reruns per seed.
3. Run controlled ablations on quantum block depth and feature scaling only.
4. Add budget-equalized comparison where classical parameter count is matched to tuned quantum-ready.
