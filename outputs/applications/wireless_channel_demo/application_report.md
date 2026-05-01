# Application Report: Sparse CSI Forecasting for Time-Varying Wireless Channels

## Purpose

This application-inspired demo evaluates whether memory-style and domain-aware features improve sparse channel-state information (CSI) reconstruction and future-window forecasting on synthetic time-varying wireless channel dynamics.

## Synthetic Channel Model

Noiseless channel response:

`h(t) = a0*A(t)*cos(2*pi*f_d*t + phase0) + a1*A(t)*cos(2*pi*f_d*(t-tau1) + phase1) + a2*exp(-lambda*t_norm) + a3*sin(2*pi*f_slow*t)`

Noisy observations use `h_noisy(t) = h(t) + sigma*N(0,1)` with drift `A(t)=clip(1-drift*t_norm, min=0.25)`.

Component interpretation:

- Doppler-like oscillation: primary cosine term with `f_d`.
- Delayed/multipath-like component: shifted cosine term with delay `tau1`.
- Slow amplitude/path-loss drift: multiplicative envelope `A(t)`.
- Low-frequency trend: sinusoidal component with `f_slow`.
- Observation noise: additive Gaussian noise.

## Task Modes

- **Interpolation**: sparse CSI samples across the full horizon, then dense full-grid evaluation.
- **Forecast**: training on an early-time observation window, then evaluation on an unseen future window.

## Fairness Protocol

- Same train/test split per run across all models.
- Same seed, epochs, optimizer, learning rate, hidden-depth/width, and device for MLP models.
- No model-specific retuning.

## Models Compared

- Linear baseline
- AR(1) baseline
- MLP baseline [t]
- MLP + memory-only
- MLP + domain sin/cos
- MLP + memory + domain

Feature-group interpretation:

- Memory-style priors: `t, t^alpha, t^(1-alpha), log(1+t)`.
- Doppler/domain priors: `sin(2*pi*f_d*t), cos(2*pi*f_d*t)`.
- Combined priors: memory-style + Doppler/domain features.

## Robustness Mini-Sweep Summary

- Noise levels: `0.00, 0.05`
- Interpolation train points: `10, 30`
- Forecast train fractions: `0.5, 0.8`
- Seeds: 0

### Interpolation Metrics (averaged over sweep)

| Model | Rel L2 mean±std | MSE mean±std | Max-Err mean±std | Runtime mean±std (s) |
| --- | ---: | ---: | ---: | ---: |
| Linear baseline | 0.943971 ± 0.171293 | 0.495138 ± 0.175361 | 1.813660 ± 0.093311 | 0.0001 ± 0.0001 |
| AR(1) baseline | 0.976567 ± 0.026263 | 0.517432 ± 0.027815 | 1.314254 ± 0.093784 | 0.0003 ± 0.0001 |
| MLP baseline [t] | 0.988475 ± 0.044983 | 0.530663 ± 0.048223 | 1.403529 ± 0.008258 | 0.5148 ± 0.1512 |
| MLP + memory-only | 1.027520 ± 0.117003 | 0.578093 ± 0.130384 | 1.369818 ± 0.310550 | 0.5008 ± 0.2177 |
| MLP + domain sin/cos | 0.138663 ± 0.054405 | 0.011630 ± 0.008183 | 0.247790 ± 0.107970 | 0.4992 ± 0.2479 |
| MLP + memory + domain | 0.104616 ± 0.046479 | 0.006813 ± 0.005500 | 0.225622 ± 0.076146 | 0.4966 ± 0.2386 |

### Forecast Metrics (averaged over sweep)

| Model | Rel L2 mean±std | MSE mean±std | Max-Err mean±std | Runtime mean±std (s) |
| --- | ---: | ---: | ---: | ---: |
| Linear baseline | 1.047176 ± 0.004772 | 0.468529 ± 0.012275 | 1.154728 ± 0.085010 | 0.0005 ± 0.0002 |
| AR(1) baseline | 1.224581 ± 0.164019 | 0.653187 ± 0.188892 | 1.521850 ± 0.244989 | 0.0003 ± 0.0001 |
| MLP baseline [t] | 1.049113 ± 0.004727 | 0.470325 ± 0.015116 | 1.160896 ± 0.093754 | 0.4420 ± 0.0751 |
| MLP + memory-only | 1.156692 ± 0.121293 | 0.579273 ± 0.138093 | 1.340360 ± 0.306013 | 0.3913 ± 0.0235 |
| MLP + domain sin/cos | 0.361683 ± 0.149029 | 0.064101 ± 0.047983 | 0.374171 ± 0.210664 | 0.4046 ± 0.0351 |
| MLP + memory + domain | 0.249553 ± 0.179789 | 0.037874 ± 0.039424 | 0.260614 ± 0.193645 | 0.3914 ± 0.0213 |

## Key Takeaways

- Best interpolation model (mean relative L2): **MLP + memory + domain**.
- Best forecast model (mean relative L2): **MLP + memory + domain**.
- Memory-only feature effect (relative L2 reduction vs MLP baseline): interpolation `-0.039046`, forecast `-0.107579`.
- Domain sinusoidal feature effect: interpolation `+0.849812`, forecast `+0.687430`.
- Combined memory+domain feature effect: interpolation `+0.883858`, forecast `+0.799560`.
- Domain sinusoidal priors were strongest; memory-only gains were mixed; combined priors were most consistent.

## Two-Channel / MIMO-Style Extension

We additionally evaluate a two-channel synthetic link `h(t)=[h1(t), h2(t)]` where each channel has Doppler-like, delayed/multipath-like, and slow-drift components, with optional coupling from delayed `h1` into `h2`.

This remains an application-inspired synthetic benchmark (not a full MIMO simulator), but it mimics sparse multi-channel CSI forecasting under the same fairness protocol.

### Two-Channel Interpolation (Aggregate Metrics)

| Model | Relative L2 | MSE | Max-Err | Ch1 Rel L2 | Ch2 Rel L2 | Runtime (s) |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Linear baseline | 0.913765 | 0.399035 | 1.832192 | 0.945281 | 0.865840 | 0.0000 |
| AR(1) baseline | 0.975226 | 0.440659 | 1.391531 | 0.975427 | 0.974906 | 0.0004 |
| MLP baseline [t] | 0.989521 | 0.455255 | 1.440633 | 1.007485 | 0.963394 | 0.4035 |
| MLP + memory-only | 1.004689 | 0.470634 | 1.483412 | 1.024969 | 0.975002 | 0.3995 |
| MLP + domain sin/cos | 0.342656 | 0.069603 | 0.928031 | 0.112648 | 0.514121 | 0.3898 |
| MLP + memory + domain | 0.345453 | 0.075578 | 0.951360 | 0.119222 | 0.515633 | 0.4082 |

### Two-Channel Forecast (Aggregate Metrics)

| Model | Relative L2 | MSE | Max-Err | Ch1 Rel L2 | Ch2 Rel L2 | Runtime (s) |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Linear baseline | 1.034141 | 0.502206 | 1.271301 | 1.049472 | 1.022953 | 0.0005 |
| AR(1) baseline | 1.284133 | 0.778462 | 1.746070 | 1.217054 | 1.309350 | 0.0004 |
| MLP baseline [t] | 1.025487 | 0.493761 | 1.240088 | 1.050982 | 1.005469 | 0.4000 |
| MLP + memory-only | 1.109023 | 0.578394 | 1.453839 | 1.214337 | 1.004425 | 0.3942 |
| MLP + domain sin/cos | 0.616854 | 0.189720 | 1.049207 | 0.308393 | 0.775762 | 0.3947 |
| MLP + memory + domain | 0.563666 | 0.157690 | 1.107839 | 0.223173 | 0.732560 | 0.3876 |

- Best two-channel interpolation model: **MLP + domain sin/cos**.
- Best two-channel forecast model: **MLP + memory + domain**.

## Limitations

- This is a synthetic application-inspired sparse-CSI benchmark, not a full LEO/MIMO system simulator.
- It uses scalar channel dynamics and does not model full multi-antenna, orbital, or standards-compliant channel pipelines.
- TE-QPINN surrogate variants were intentionally omitted in this first application phase to keep the demo lightweight and fast; they are future-work extensions for this application track.
- Results indicate method transfer potential only; they are not deployment-level channel-prediction claims.

