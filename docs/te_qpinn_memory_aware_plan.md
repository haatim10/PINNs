# TE-QPINN Memory-Aware Plan (Phase 8B)

## Purpose

Extend TE-QPINN-inspired surrogate models with low-risk analytic memory features tailored to the fractional integro-differential equation, while preserving fair comparison against Classical + PI.

## Motivation

The target PDE is nonlocal in time:

- fractional derivative terms encode history effects,
- integral-kernel terms depend on prior times `s < t`,
- local-coordinate embeddings `[x, t]` may underrepresent this structure.

Adding memory-aware analytic features gives the model explicit access to fractional-history priors during representation learning.

## Proposed v1 Method (Analytic Memory Features)

Base local input:

- `[x, t]`

Analytic memory feature set:

- `[x, t, t^alpha, t^(1-alpha), t^(1-beta), log(1+t), x*t, x*t^alpha]`

Numerical safety:

- `t_safe = clamp(t, min=epsilon)`, with `epsilon = 1e-8`.

Deterministic normalization:

- fixed domain-based scaling (no batch-dependent normalization).

## Fairness Controls

Compare:

1. Classical + PI
2. Classical + PI + same analytic memory features
3. TE fixed
4. TE LayerNorm post_quantum
5. TE memory-aware analytic
6. TE memory-aware analytic + LayerNorm post_quantum

Required reporting:

- final L2, final Linf, final loss,
- runtime,
- parameter count,
- accuracy/runtime interpretation.

## Shared Feature Builder Requirement

Use one shared `MemoryFeatureBuilder` for both:

- TE memory-aware models,
- Classical memory-feature control models.

This ensures identical feature math and avoids fairness drift across implementations.

## Stage Funnel

1. **Stage 1 (seed-42 smoke):** correctness + early signal only.
2. **Stage 2 (5 seeds):** generalization check.
3. **Stage 2b (10 seeds):** confirmatory stability.
4. **Stage 3:** optimizer sensitivity (Adam vs Adam+LBFGS, clearly labeled as extended-budget).
5. **Stage 4:** alpha/beta robustness slice.

## Claim Boundary

If successful, valid claim scope is:

“Memory-aware TE-QPINN improves TE-side stability/accuracy versus local-coordinate TE baselines on this benchmark.”

Out of scope claims:

- no quantum advantage claim,
- no hardware-speedup claim,
- no universal superiority claim over all classical PINNs.
