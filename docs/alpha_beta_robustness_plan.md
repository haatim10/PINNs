# Alpha/Beta Robustness Plan (Phase 10B)

This document plans a locked robustness test for the memory-aware feature idea on a second fractional setting, without changing model architecture or tuning.

Scope of this plan:

- No code changes
- No benchmark runs
- No new configs created yet

---

## 1. Purpose

The current benchmark evidence is based primarily on `alpha=0.5, beta=0.5`.  
Because this PDE is fractional and nonlocal in time, behavior can shift meaningfully when `(alpha, beta)` changes.

Testing a second pre-locked pair (`alpha=0.7, beta=0.3`) matters because it checks whether:

- memory-aware analytic features transfer beyond one fractional regime,
- TE improvements vs non-memory TE are stable,
- gains are general memory-physics effects vs pair-specific artifacts.

This is a robustness test, not a re-tuning exercise.

---

## 2. Current alpha/beta source

Current source-of-truth plumbing is already clean:

- Problem config carries `problem.alpha` and `problem.beta`.
- `scripts/train_integro_diff.py` reads them directly and passes them into:
  - mesh/L1 setup,
  - residual operators,
  - exact-solution/forcing evaluation.
- `src/model_factory.py` passes `problem.alpha` and `problem.beta` into `MemoryFeatureBuilder`.
- `scripts/benchmark_te_qpinn.py` uses per-run config files directly (no hidden alpha/beta override path).

Result: changing alpha/beta in run configs will propagate through physics loss and memory features consistently.

---

## 3. Exact solution and forcing function compatibility

`src/physics_integro.py` defines both `exact_solution(...)` and `source_term(...)` from the same solution config and `(alpha, beta)`, so forcing remains manufactured-consistent when alpha/beta changes.

Important compatibility note for this robustness phase:

- Many benchmark configs explicitly set `problem.solution.time_power: 0.5`.
- If we switch to `alpha=0.7, beta=0.3` but leave `time_power=0.5`, we are testing a different time-power regime than the default `time_power=alpha`.

Plan decision:

- For robustness fairness, set `time_power` consistently with the selected benchmark definition and document it explicitly.
- Recommended for this test: set `time_power: 0.7` (or remove the field and rely on default `time_power=alpha`).

---

## 4. MemoryFeatureBuilder compatibility

`src/memory_features.py` is already compatible with alternate alpha/beta settings:

- Uses problem-level `alpha`, `beta` passed from `model_factory`.
- Enforces `memory_features='analytic'` requires alpha/beta (no silent fallback).
- Uses `t_safe = clamp(t, min=memory_epsilon)` for numerical safety near `t=0`.
- Uses deterministic scaling (`memory_feature_normalization: scale`) based on domain bounds and alpha/beta-dependent feature ranges.

Mismatch risk status:

- Low, as long as alpha/beta are changed in the `problem` block of each run config.

---

## 5. Proposed variants

Keep only current key finalists:

1. Classical + PI  
2. Classical + PI + analytic memory  
3. TE LayerNorm post_quantum + PI  
4. TE memory analytic + PI

Optional diagnostic include only if runtime budget allows:

5. TE fixed residual 0.10 + PI (local-coordinate TE baseline)

---

## 6. Proposed seeds

Start with locked 5-seed validation:

`[0, 1, 2, 3, 4]`

Reason: directly comparable with earlier Phase 8D-style multi-seed protocol and still affordable.

---

## 7. Proposed configs

No config creation in this phase (planning only), but next implementation should derive alpha/beta-specific copies from existing locked configs.

Current source configs to copy/modify:

- `configs/benchmark_te_qpinn_50_classical_pi.yaml`
- `configs/benchmark_te_qpinn_50_classical_pi_memory_analytic.yaml`
- `configs/benchmark_te_qpinn_50_surrogate_pi_layernorm_post_quantum.yaml`
- `configs/benchmark_te_qpinn_50_surrogate_pi_memory_analytic.yaml`
- optional: `configs/benchmark_te_qpinn_50_surrogate_pi.yaml`

Required changes in each copied config:

- `problem.alpha: 0.7`
- `problem.beta: 0.3`
- `problem.solution.time_power: 0.7` (or remove to use default `time_power=alpha`)

Benchmark plan config to add later:

- `configs/benchmark_te_qpinn_memory_alpha07_beta03_5seed.yaml`

---

## 8. Output directories

Recommended outputs:

- `outputs/benchmarks/te_qpinn_memory_alpha07_beta03_5seed/`
- `outputs/plots/te_qpinn_memory_alpha07_beta03_5seed/`

Keep prior outputs untouched.

---

## 9. Metrics and stats

Use the same reporting style and tests from Phase 10A/confirmatory stats:

- mean/std final L2
- mean/std final Linf
- mean/std final loss
- mean/std runtime
- parameter count
- paired win counts

Paired statistics (reuse existing helper flow):

- paired mean difference
- paired median difference
- Wilcoxon signed-rank (two-sided and one-sided where applicable)
- bootstrap 95% CI for paired mean difference
- paired Cohen’s d (and Cliff’s delta if included)

Suggested reuse path:

- `scripts/generate_confirmatory_stats.py` on new `summary.csv`

---

## 10. Go/no-go criteria

Memory-feature transfer is considered directionally successful if most of the following hold:

1. Classical + memory improves over Classical on mean L2 (primary).
2. TE memory improves over TE LayerNorm non-memory on mean L2 and/or win counts.
3. Directional memory benefit remains consistent with `alpha=0.5,beta=0.5` results (even if magnitudes differ).

For TE-specific claim progression (not final claim):

- TE memory should beat TE non-memory on mean L2 and at least tie/compete in Linf with acceptable runtime overhead.

If Classical + memory remains strongest, interpret as general memory-feature usefulness, not TE-specific superiority.

---

## 11. Risks

Key risks to manage explicitly:

- **Forcing/exact mismatch risk** if alpha/beta change but `time_power` is left inconsistent.
- **Near-zero-time instability** due to fractional powers (mitigated by `t_safe` clamp).
- **Difficulty-shift risk**: `(0.7, 0.3)` may change conditioning and runtime.
- **Fairness risk**: avoid cherry-picking alpha/beta after viewing results; pre-lock pair and protocol.
- **Interpretation risk**: improvements may come from memory features generally, not TE architecture specifically.

---

## 12. Recommended implementation prompt (Phase 10C, ready to copy)

```text
Phase 10C: Implement and run alpha=0.7, beta=0.3 memory-robustness 5-seed validation.

Context:
Phase 10B planning is complete in docs/alpha_beta_robustness_plan.md.

Constraints:
- Do not retune hyperparameters.
- Do not change model architecture.
- Do not change physics code.
- Keep memory feature definitions unchanged.
- Reuse existing benchmark runner/statistics flow.

Tasks:
1) Create alpha/beta=0.7/0.3 copies of these configs:
   - benchmark_te_qpinn_50_classical_pi.yaml
   - benchmark_te_qpinn_50_classical_pi_memory_analytic.yaml
   - benchmark_te_qpinn_50_surrogate_pi_layernorm_post_quantum.yaml
   - benchmark_te_qpinn_50_surrogate_pi_memory_analytic.yaml
   - optional TE fixed baseline config
   Ensure:
   - problem.alpha = 0.7
   - problem.beta = 0.3
   - solution.time_power = 0.7 (or remove and rely on default time_power=alpha)

2) Create benchmark plan:
   configs/benchmark_te_qpinn_memory_alpha07_beta03_5seed.yaml
   Seeds: [0,1,2,3,4]
   Variants:
   - Classical + PI
   - Classical + PI + analytic memory
   - TE LayerNorm post_quantum + PI
   - TE memory analytic + PI
   - optional TE fixed baseline

3) Run:
   python scripts/benchmark_te_qpinn.py --benchmark-config configs/benchmark_te_qpinn_memory_alpha07_beta03_5seed.yaml --resume-incomplete

4) Generate statistical analysis from summary.csv (same method as Phase 10A):
   - paired mean/median differences
   - Wilcoxon
   - bootstrap 95% CI
   - effect sizes
   - win counts

5) Save outputs:
   - outputs/benchmarks/te_qpinn_memory_alpha07_beta03_5seed/
   - outputs/plots/te_qpinn_memory_alpha07_beta03_5seed/

6) Update docs/te_qpinn_benchmark_analysis.md with:
   - alpha/beta robustness subsection
   - careful interpretation:
     * if Classical+memory remains best, say memory helps generally
     * if TE memory beats TE non-memory, say memory helps TE directionally
     * do not claim TE superiority unless evidence supports it

7) Run pytest -q and report:
   - completion status
   - main metrics table
   - primary endpoint outcome
   - robustness conclusion
```

