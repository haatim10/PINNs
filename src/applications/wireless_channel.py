"""Synthetic wireless channel demo utilities.

This module provides:
- application-inspired synthetic time-varying channel generation
- interpolation/forecast split builders
- fair feature builders for model comparison
- lightweight baselines and MLP training helpers
"""

from __future__ import annotations

from dataclasses import dataclass
import time
from typing import Dict, List, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

from src.utils import count_trainable_parameters, resolve_device, set_seed


EPS = 1e-12


@dataclass
class WirelessChannelConfig:
    t_start: float = 0.0
    t_end: float = 1.0
    num_dense_samples: int = 400
    doppler_hz: float = 8.0
    slow_hz: float = 0.7
    delay_tau1: float = 0.045
    noise_std: float = 0.02
    seed: int = 42
    # Channel-shape controls
    a0: float = 0.95
    a1: float = 0.40
    a2: float = 0.30
    a3: float = 0.22
    decay_lambda: float = 2.3
    drift_strength: float = 0.22
    phase0: float = 0.25
    phase1: float = 1.10
    # Channel dimensionality.
    channel_dim: int = 1  # 1 or 2
    # Secondary channel parameters (used when channel_dim=2).
    b0: float = 0.80
    b1: float = 0.30
    b2: float = 0.28
    b3: float = 0.25
    delta_f: float = 0.65
    delay_tau2: float = 0.070
    decay_lambda2: float = 1.9
    phase2: float = 0.75
    phase3: float = 1.85
    phase_shift: float = 0.35
    coupling: float = 0.15
    coupling_delay: float = 0.025
    task_mode: str = "interpolation"  # interpolation | forecast
    num_train_samples: int = 20
    forecast_train_fraction: float = 0.6


@dataclass
class MLPTrainConfig:
    hidden_layers: Tuple[int, ...] = (32, 32)
    activation: str = "tanh"
    epochs: int = 250
    learning_rate: float = 5e-3
    deterministic_torch: bool = True
    device: str = "auto"


class SimpleMLP(nn.Module):
    def __init__(
        self,
        input_dim: int,
        hidden_layers: Tuple[int, ...],
        output_dim: int = 1,
        activation: str = "tanh",
    ):
        super().__init__()
        activations = {
            "tanh": nn.Tanh(),
            "relu": nn.ReLU(),
            "gelu": nn.GELU(),
            "silu": nn.SiLU(),
        }
        act = activations.get(str(activation).lower(), nn.Tanh())
        layers: List[nn.Module] = []
        prev_dim = input_dim
        for width in hidden_layers:
            layers.append(nn.Linear(prev_dim, int(width)))
            layers.append(act)
            prev_dim = int(width)
        layers.append(nn.Linear(prev_dim, output_dim))
        self.network = nn.Sequential(*layers)
        self._initialize_weights()

    def _initialize_weights(self):
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.network(x)


def _normalize_time(t: np.ndarray, t_start: float, t_end: float) -> np.ndarray:
    denom = max(float(t_end) - float(t_start), EPS)
    return (t - float(t_start)) / denom


def generate_wireless_channel(cfg: WirelessChannelConfig) -> Dict[str, np.ndarray]:
    """Generate synthetic channel sequence with memory-like dynamics."""
    rng = np.random.default_rng(int(cfg.seed))
    t = np.linspace(float(cfg.t_start), float(cfg.t_end), int(cfg.num_dense_samples), dtype=float)
    t_norm = _normalize_time(t, cfg.t_start, cfg.t_end)

    # Slow path-loss/amplitude drift.
    amp = 1.0 - float(cfg.drift_strength) * t_norm
    amp = np.clip(amp, 0.25, None)

    # Multipath-like delayed Doppler components + slow term + exponential drift.
    doppler_main = np.cos(2.0 * np.pi * cfg.doppler_hz * t + cfg.phase0)
    doppler_delayed = np.cos(2.0 * np.pi * cfg.doppler_hz * (t - cfg.delay_tau1) + cfg.phase1)
    slow_term = np.sin(2.0 * np.pi * cfg.slow_hz * t)
    decay_term = np.exp(-cfg.decay_lambda * t_norm)

    h1_clean = (
        cfg.a0 * amp * doppler_main
        + cfg.a1 * amp * doppler_delayed
        + cfg.a2 * decay_term
        + cfg.a3 * slow_term
    )

    channel_dim = int(cfg.channel_dim)
    if channel_dim not in {1, 2}:
        raise ValueError("channel_dim must be 1 or 2")

    if channel_dim == 1:
        h_clean = h1_clean.astype(float)
        noise = float(cfg.noise_std) * rng.standard_normal(t.shape[0])
        h_noisy = h_clean + noise
    else:
        doppler_main_2 = np.cos(2.0 * np.pi * (cfg.doppler_hz + cfg.delta_f) * t + cfg.phase2)
        doppler_delayed_2 = np.cos(2.0 * np.pi * cfg.doppler_hz * (t - cfg.delay_tau2) + cfg.phase3)
        slow_term_2 = np.sin(2.0 * np.pi * cfg.slow_hz * t + cfg.phase_shift)
        decay_term_2 = np.exp(-cfg.decay_lambda2 * t_norm)

        h2_base = (
            cfg.b0 * amp * doppler_main_2
            + cfg.b1 * amp * doppler_delayed_2
            + cfg.b2 * decay_term_2
            + cfg.b3 * slow_term_2
        )

        shifted_h1 = np.interp(
            t - cfg.coupling_delay,
            t,
            h1_clean,
            left=float(h1_clean[0]),
            right=float(h1_clean[-1]),
        )
        h2_clean = h2_base + cfg.coupling * shifted_h1

        h_clean = np.stack([h1_clean, h2_clean], axis=1).astype(float)
        noise = float(cfg.noise_std) * rng.standard_normal(h_clean.shape)
        h_noisy = h_clean + noise

    return {
        "t": t,
        "h_clean": h_clean.astype(float),
        "h_noisy": h_noisy.astype(float),
        "noise": noise.astype(float),
    }


def make_channel_split(
    t: np.ndarray,
    h_noisy: np.ndarray,
    h_clean: np.ndarray,
    task_mode: str,
    num_train_samples: int,
    forecast_train_fraction: float,
    seed: int,
) -> Dict[str, np.ndarray]:
    """Create deterministic train/test splits for interpolation or forecast tasks."""
    h_noisy = np.asarray(h_noisy)
    h_clean = np.asarray(h_clean)
    mode = str(task_mode).lower()
    n = int(t.shape[0])
    rng = np.random.default_rng(int(seed))

    if mode == "interpolation":
        n_train = max(4, min(int(num_train_samples), n))
        if n_train >= n:
            train_idx = np.arange(n, dtype=int)
        else:
            interior_pool = np.arange(1, n - 1, dtype=int)
            needed = max(0, n_train - 2)
            interior = (
                rng.choice(interior_pool, size=needed, replace=False)
                if needed > 0 and interior_pool.size > 0
                else np.array([], dtype=int)
            )
            train_idx = np.sort(np.concatenate([np.array([0, n - 1], dtype=int), interior]))
        test_idx = np.arange(n, dtype=int)
    elif mode == "forecast":
        frac = float(forecast_train_fraction)
        cutoff = int(round(frac * n))
        cutoff = max(4, min(cutoff, n - 2))
        train_idx = np.arange(cutoff, dtype=int)
        test_idx = np.arange(cutoff, n, dtype=int)
    else:
        raise ValueError("task_mode must be 'interpolation' or 'forecast'")

    return {
        "task_mode": mode,
        "train_idx": train_idx,
        "test_idx": test_idx,
        "train_t": t[train_idx],
        "train_y_noisy": h_noisy[train_idx],
        "train_y_clean": h_clean[train_idx],
        "test_t": t[test_idx],
        "test_y_clean": h_clean[test_idx],
        "full_t": t,
        "full_y_clean": h_clean,
        "full_y_noisy": h_noisy,
    }


def build_channel_features(
    t: np.ndarray,
    feature_kind: str,
    alpha: float = 0.7,
    doppler_hz: float = 8.0,
    t_start: float = 0.0,
    t_end: float = 1.0,
) -> np.ndarray:
    """Build channel-regression features with clear memory/domain controls."""
    feature_key = str(feature_kind).lower()
    t_norm = _normalize_time(t.astype(float), t_start, t_end)
    t_safe = np.clip(t_norm, 1e-8, None)

    base = [t_norm]
    memory = [t_norm, np.power(t_safe, alpha), np.power(t_safe, 1.0 - alpha), np.log1p(t_norm)]
    domain = [t_norm, np.sin(2.0 * np.pi * doppler_hz * t), np.cos(2.0 * np.pi * doppler_hz * t)]

    if feature_key in {"time", "base", "mlp_t"}:
        feats = base
    elif feature_key in {"memory", "memory_only", "mlp_memory"}:
        feats = memory
    elif feature_key in {"domain", "sinusoidal", "mlp_domain"}:
        feats = domain
    elif feature_key in {"combined", "memory_domain", "mlp_memory_domain"}:
        feats = [
            t_norm,
            np.power(t_safe, alpha),
            np.power(t_safe, 1.0 - alpha),
            np.log1p(t_norm),
            np.sin(2.0 * np.pi * doppler_hz * t),
            np.cos(2.0 * np.pi * doppler_hz * t),
        ]
    else:
        raise ValueError(f"Unsupported feature_kind '{feature_kind}'")

    matrix = np.stack(feats, axis=1).astype(np.float32)
    return matrix


def run_linear_baseline(split: Dict[str, np.ndarray]) -> Tuple[np.ndarray, float]:
    """Linear baseline:
    - interpolation mode: linear interpolation over sparse observations
    - forecast mode: linear extrapolation from observed early window
    """
    start = time.perf_counter()
    mode = split["task_mode"]
    train_t = split["train_t"]
    train_y = split["train_y_noisy"]
    test_t = split["test_t"]

    train_y_arr = np.asarray(train_y)
    if train_y_arr.ndim == 1:
        if mode == "interpolation":
            pred = np.interp(test_t, train_t, train_y_arr)
        else:
            if train_t.size < 2:
                pred = np.full_like(test_t, fill_value=float(train_y_arr[-1]))
            else:
                slope, intercept = np.polyfit(train_t, train_y_arr, deg=1)
                pred = slope * test_t + intercept
    else:
        channels = train_y_arr.shape[1]
        pred_matrix = np.zeros((test_t.shape[0], channels), dtype=float)
        for c in range(channels):
            y_c = train_y_arr[:, c]
            if mode == "interpolation":
                pred_matrix[:, c] = np.interp(test_t, train_t, y_c)
            else:
                if train_t.size < 2:
                    pred_matrix[:, c] = float(y_c[-1])
                else:
                    slope, intercept = np.polyfit(train_t, y_c, deg=1)
                    pred_matrix[:, c] = slope * test_t + intercept
        pred = pred_matrix
    runtime = time.perf_counter() - start
    return pred.astype(float), runtime


def _fit_ar1(train_y: np.ndarray) -> Tuple[float, float]:
    if train_y.size < 2:
        return 1.0, 0.0
    x = train_y[:-1]
    y = train_y[1:]
    design = np.column_stack([x, np.ones_like(x)])
    coeffs, *_ = np.linalg.lstsq(design, y, rcond=None)
    return float(coeffs[0]), float(coeffs[1])


def run_ar1_baseline(split: Dict[str, np.ndarray]) -> Tuple[np.ndarray, float]:
    """AR(1)-style baseline.

    - Forecast: recursive prediction on future window.
    - Interpolation: fit AR(1) on observed sequence, then interpolate AR(1)-generated
      sequence over train timestamps.
    """
    start = time.perf_counter()
    mode = split["task_mode"]
    train_t = split["train_t"]
    train_y = split["train_y_noisy"]
    test_t = split["test_t"]

    train_y_arr = np.asarray(train_y)
    if train_y_arr.ndim == 1:
        a, b = _fit_ar1(train_y_arr)
        if mode == "forecast":
            preds = np.zeros_like(test_t, dtype=float)
            prev = float(train_y_arr[-1])
            for i in range(test_t.shape[0]):
                prev = a * prev + b
                preds[i] = prev
        else:
            ar_series = np.zeros_like(train_y_arr, dtype=float)
            ar_series[0] = float(train_y_arr[0])
            for i in range(1, train_y_arr.shape[0]):
                ar_series[i] = a * ar_series[i - 1] + b
            preds = np.interp(test_t, train_t, ar_series, left=ar_series[0], right=ar_series[-1])
    else:
        channels = train_y_arr.shape[1]
        preds = np.zeros((test_t.shape[0], channels), dtype=float)
        for c in range(channels):
            y_c = train_y_arr[:, c]
            a, b = _fit_ar1(y_c)
            if mode == "forecast":
                prev = float(y_c[-1])
                for i in range(test_t.shape[0]):
                    prev = a * prev + b
                    preds[i, c] = prev
            else:
                ar_series = np.zeros_like(y_c, dtype=float)
                ar_series[0] = float(y_c[0])
                for i in range(1, y_c.shape[0]):
                    ar_series[i] = a * ar_series[i - 1] + b
                preds[:, c] = np.interp(test_t, train_t, ar_series, left=ar_series[0], right=ar_series[-1])

    runtime = time.perf_counter() - start
    return preds.astype(float), runtime


def train_mlp_regressor(
    train_features: np.ndarray,
    train_targets: np.ndarray,
    test_features: np.ndarray,
    train_cfg: MLPTrainConfig,
    seed: int,
) -> Dict:
    """Train a small MLP regressor with strict fairness controls."""
    set_seed(int(seed), deterministic=bool(train_cfg.deterministic_torch))
    device = resolve_device(train_cfg.device)

    x_train = torch.from_numpy(train_features).to(device=device, dtype=torch.float32)
    y_np = np.asarray(train_targets)
    y_train = torch.from_numpy(y_np).to(device=device, dtype=torch.float32)
    if y_train.ndim == 1:
        y_train = y_train.unsqueeze(-1)
    x_test = torch.from_numpy(test_features).to(device=device, dtype=torch.float32)

    output_dim = int(y_train.shape[1])

    model = SimpleMLP(
        input_dim=x_train.shape[1],
        hidden_layers=tuple(train_cfg.hidden_layers),
        output_dim=output_dim,
        activation=train_cfg.activation,
    ).to(device)

    optimizer = optim.Adam(model.parameters(), lr=float(train_cfg.learning_rate))
    criterion = nn.MSELoss()

    losses: List[float] = []
    start = time.perf_counter()
    model.train()
    for _ in range(int(train_cfg.epochs)):
        optimizer.zero_grad(set_to_none=True)
        pred = model(x_train)
        loss = criterion(pred, y_train)
        loss.backward()
        optimizer.step()
        losses.append(float(loss.detach().cpu().item()))
    train_runtime = time.perf_counter() - start

    model.eval()
    with torch.no_grad():
        pred_test_tensor = model(x_test)
        if pred_test_tensor.shape[1] == 1:
            pred_test = pred_test_tensor.squeeze(-1).detach().cpu().numpy().astype(float)
        else:
            pred_test = pred_test_tensor.detach().cpu().numpy().astype(float)

    return {
        "pred_test": pred_test,
        "loss_history": losses,
        "runtime_sec": float(train_runtime),
        "parameter_count": int(count_trainable_parameters(model)),
        "device": device,
    }


def regression_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> Dict[str, float]:
    y_true_arr = np.asarray(y_true)
    y_pred_arr = np.asarray(y_pred)
    err = y_pred_arr - y_true_arr
    mse = float(np.mean(err**2))
    rel_l2 = float(np.linalg.norm(err) / (np.linalg.norm(y_true_arr) + EPS))
    max_abs = float(np.max(np.abs(err)))
    metrics = {
        "mse": mse,
        "relative_l2": rel_l2,
        "max_abs_error": max_abs,
    }
    if y_true_arr.ndim == 2 and y_true_arr.shape[1] > 1:
        for c in range(y_true_arr.shape[1]):
            channel_err = y_pred_arr[:, c] - y_true_arr[:, c]
            channel_l2 = float(np.linalg.norm(channel_err) / (np.linalg.norm(y_true_arr[:, c]) + EPS))
            metrics[f"channel_{c + 1}_relative_l2"] = channel_l2
    return metrics
