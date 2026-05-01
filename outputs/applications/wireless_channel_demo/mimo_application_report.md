# Two-Channel Sparse CSI Forecasting Report

## Purpose

This extension evaluates sparse CSI forecasting for a two-channel synthetic wireless link `h(t)=[h1(t), h2(t)]` under the same fairness protocol as the scalar demo.

## Synthetic Two-Channel Dynamics

- `h1(t)`: Doppler-like + delayed/multipath-like + drift + low-frequency trend.
- `h2(t)`: related but shifted frequencies/phases and delayed terms, with optional coupling from delayed `h1(t)`.
- Independent Gaussian noise is added per channel.
- This is not a full MIMO simulator; it is an application-inspired multi-channel benchmark.

## Sweep Setup

- Noise levels: `0.00, 0.05`
- Interpolation train points: `10, 30`
- Forecast train fractions: `0.5, 0.8`
- Seeds: 0

## Interpolation Results (Aggregate)

| Model | Rel L2 | MSE | Max-Err | Ch1 Rel L2 | Ch2 Rel L2 | Runtime (s) |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Linear baseline | 0.913765 | 0.399035 | 1.832192 | 0.945281 | 0.865840 | 0.0000 |
| AR(1) baseline | 0.975226 | 0.440659 | 1.391531 | 0.975427 | 0.974906 | 0.0004 |
| MLP baseline [t] | 0.989521 | 0.455255 | 1.440633 | 1.007485 | 0.963394 | 0.4035 |
| MLP + memory-only | 1.004689 | 0.470634 | 1.483412 | 1.024969 | 0.975002 | 0.3995 |
| MLP + domain sin/cos | 0.342656 | 0.069603 | 0.928031 | 0.112648 | 0.514121 | 0.3898 |
| MLP + memory + domain | 0.345453 | 0.075578 | 0.951360 | 0.119222 | 0.515633 | 0.4082 |

## Forecast Results (Aggregate)

| Model | Rel L2 | MSE | Max-Err | Ch1 Rel L2 | Ch2 Rel L2 | Runtime (s) |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Linear baseline | 1.034141 | 0.502206 | 1.271301 | 1.049472 | 1.022953 | 0.0005 |
| AR(1) baseline | 1.284133 | 0.778462 | 1.746070 | 1.217054 | 1.309350 | 0.0004 |
| MLP baseline [t] | 1.025487 | 0.493761 | 1.240088 | 1.050982 | 1.005469 | 0.4000 |
| MLP + memory-only | 1.109023 | 0.578394 | 1.453839 | 1.214337 | 1.004425 | 0.3942 |
| MLP + domain sin/cos | 0.616854 | 0.189720 | 1.049207 | 0.308393 | 0.775762 | 0.3947 |
| MLP + memory + domain | 0.563666 | 0.157690 | 1.107839 | 0.223173 | 0.732560 | 0.3876 |

- Best interpolation model: **MLP + domain sin/cos**.
- Best forecast model: **MLP + memory + domain**.

## Interpretation

This two-channel extension tests whether memory/domain priors continue to help under joint multi-output training. It is a synthetic feasibility demo and not a deployment-level claim.

