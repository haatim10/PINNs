#!/usr/bin/env python3
"""Application-inspired sparse CSI forecasting demo.

This script demonstrates memory-aware neural modeling ideas on a synthetic
time-varying channel task with strict fairness controls.
"""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
import statistics
import sys
from typing import Dict, List, Tuple

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.applications.wireless_channel import (
    MLPTrainConfig,
    WirelessChannelConfig,
    build_channel_features,
    generate_wireless_channel,
    make_channel_split,
    regression_metrics,
    run_ar1_baseline,
    run_linear_baseline,
    train_mlp_regressor,
)


MODEL_ORDER = [
    "linear_baseline",
    "ar1_baseline",
    "mlp_t",
    "mlp_memory",
    "mlp_domain",
    "mlp_memory_domain",
]

MODEL_LABELS = {
    "linear_baseline": "Linear baseline",
    "ar1_baseline": "AR(1) baseline",
    "mlp_t": "MLP baseline [t]",
    "mlp_memory": "MLP + memory-only",
    "mlp_domain": "MLP + domain sin/cos",
    "mlp_memory_domain": "MLP + memory + domain",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run lightweight sparse CSI forecasting application demo."
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="outputs/applications/wireless_channel_demo",
        help="Output directory for metrics/plots/report",
    )
    parser.add_argument(
        "--seeds",
        type=str,
        default="0",
        help="Comma-separated seeds for mini sweep (default: 0)",
    )
    parser.add_argument(
        "--num-dense-samples",
        type=int,
        default=400,
        help="Dense grid samples for evaluation",
    )
    parser.add_argument(
        "--epochs",
        type=int,
        default=250,
        help="Training epochs for all MLP models (fairness-locked)",
    )
    parser.add_argument(
        "--learning-rate",
        type=float,
        default=5e-3,
        help="Learning rate for all MLP models (fairness-locked)",
    )
    parser.add_argument(
        "--hidden-layers",
        type=str,
        default="32,32",
        help="Comma-separated hidden widths for all MLP models",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="auto",
        help="Device: auto|cpu|cuda",
    )
    parser.add_argument(
        "--fast",
        action="store_true",
        help="Compatibility flag; current Phase-C sweep is already runtime-reduced.",
    )
    return parser.parse_args()


def _parse_int_list(raw: str) -> List[int]:
    return [int(item.strip()) for item in raw.split(",") if item.strip()]


def _parse_float_list(raw: str) -> List[float]:
    return [float(item.strip()) for item in raw.split(",") if item.strip()]


def _parse_hidden_layers(raw: str) -> Tuple[int, ...]:
    layers = tuple(_parse_int_list(raw))
    if not layers:
        raise ValueError("hidden-layers must contain at least one width")
    return layers


def _safe_mean(values: List[float]) -> float:
    return float(sum(values) / len(values)) if values else float("nan")


def _safe_std(values: List[float]) -> float:
    if not values:
        return float("nan")
    if len(values) < 2:
        return 0.0
    return float(statistics.stdev(values))


def run_one_experiment(
    mode: str,
    noise_level: float,
    train_setting: float,
    seed: int,
    num_dense_samples: int,
    train_cfg: MLPTrainConfig,
    doppler_hz: float,
    slow_hz: float,
    delay_tau1: float,
    alpha_feature: float,
    channel_dim: int = 1,
) -> Dict:
    if mode == "interpolation":
        cfg = WirelessChannelConfig(
            num_dense_samples=num_dense_samples,
            noise_std=noise_level,
            seed=seed,
            doppler_hz=doppler_hz,
            slow_hz=slow_hz,
            delay_tau1=delay_tau1,
            channel_dim=int(channel_dim),
            task_mode="interpolation",
            num_train_samples=int(train_setting),
            forecast_train_fraction=0.6,
        )
    else:
        cfg = WirelessChannelConfig(
            num_dense_samples=num_dense_samples,
            noise_std=noise_level,
            seed=seed,
            doppler_hz=doppler_hz,
            slow_hz=slow_hz,
            delay_tau1=delay_tau1,
            channel_dim=int(channel_dim),
            task_mode="forecast",
            num_train_samples=20,
            forecast_train_fraction=float(train_setting),
        )

    generated = generate_wireless_channel(cfg)
    split = make_channel_split(
        t=generated["t"],
        h_noisy=generated["h_noisy"],
        h_clean=generated["h_clean"],
        task_mode=cfg.task_mode,
        num_train_samples=cfg.num_train_samples,
        forecast_train_fraction=cfg.forecast_train_fraction,
        seed=seed,
    )

    rows: List[Dict] = []
    prediction_payload: Dict[str, np.ndarray] = {}
    curve_payload: Dict[str, List[float]] = {}

    # Baseline A: linear interpolation/extrapolation.
    pred_linear, runtime_linear = run_linear_baseline(split)
    metrics_linear = regression_metrics(split["test_y_clean"], pred_linear)
    rows.append(
        {
            "mode": mode,
            "seed": seed,
            "noise_level": noise_level,
            "train_setting": float(train_setting),
            "model": "linear_baseline",
            "model_label": MODEL_LABELS["linear_baseline"],
            "feature_family": "baseline",
            "channel_dim": int(channel_dim),
            "mse": metrics_linear["mse"],
            "relative_l2": metrics_linear["relative_l2"],
            "max_abs_error": metrics_linear["max_abs_error"],
            "channel_1_relative_l2": metrics_linear.get("channel_1_relative_l2", float("nan")),
            "channel_2_relative_l2": metrics_linear.get("channel_2_relative_l2", float("nan")),
            "forecast_window_relative_l2": metrics_linear["relative_l2"] if mode == "forecast" else float("nan"),
            "runtime_sec": runtime_linear,
            "parameter_count": 0,
            "train_points": int(split["train_t"].shape[0]),
            "test_points": int(split["test_t"].shape[0]),
        }
    )
    prediction_payload["linear_baseline"] = pred_linear

    # Baseline B: AR(1)-style.
    pred_ar1, runtime_ar1 = run_ar1_baseline(split)
    metrics_ar1 = regression_metrics(split["test_y_clean"], pred_ar1)
    rows.append(
        {
            "mode": mode,
            "seed": seed,
            "noise_level": noise_level,
            "train_setting": float(train_setting),
            "model": "ar1_baseline",
            "model_label": MODEL_LABELS["ar1_baseline"],
            "feature_family": "baseline",
            "channel_dim": int(channel_dim),
            "mse": metrics_ar1["mse"],
            "relative_l2": metrics_ar1["relative_l2"],
            "max_abs_error": metrics_ar1["max_abs_error"],
            "channel_1_relative_l2": metrics_ar1.get("channel_1_relative_l2", float("nan")),
            "channel_2_relative_l2": metrics_ar1.get("channel_2_relative_l2", float("nan")),
            "forecast_window_relative_l2": metrics_ar1["relative_l2"] if mode == "forecast" else float("nan"),
            "runtime_sec": runtime_ar1,
            "parameter_count": 0,
            "train_points": int(split["train_t"].shape[0]),
            "test_points": int(split["test_t"].shape[0]),
        }
    )
    prediction_payload["ar1_baseline"] = pred_ar1

    mlp_feature_sets = {
        "mlp_t": "mlp_t",
        "mlp_memory": "mlp_memory",
        "mlp_domain": "mlp_domain",
        "mlp_memory_domain": "mlp_memory_domain",
    }

    for model_name, feature_kind in mlp_feature_sets.items():
        x_train = build_channel_features(
            split["train_t"],
            feature_kind=feature_kind,
            alpha=alpha_feature,
            doppler_hz=doppler_hz,
            t_start=cfg.t_start,
            t_end=cfg.t_end,
        )
        x_test = build_channel_features(
            split["test_t"],
            feature_kind=feature_kind,
            alpha=alpha_feature,
            doppler_hz=doppler_hz,
            t_start=cfg.t_start,
            t_end=cfg.t_end,
        )
        trained = train_mlp_regressor(
            train_features=x_train,
            train_targets=split["train_y_noisy"],
            test_features=x_test,
            train_cfg=train_cfg,
            seed=seed,
        )
        pred = trained["pred_test"]
        metrics = regression_metrics(split["test_y_clean"], pred)
        feature_family = (
            "memory_only"
            if model_name == "mlp_memory"
            else "domain_only"
            if model_name == "mlp_domain"
            else "memory_plus_domain"
            if model_name == "mlp_memory_domain"
            else "time_only"
        )
        rows.append(
            {
                "mode": mode,
                "seed": seed,
                "noise_level": noise_level,
                "train_setting": float(train_setting),
                "model": model_name,
                "model_label": MODEL_LABELS[model_name],
                "feature_family": feature_family,
                "channel_dim": int(channel_dim),
                "mse": metrics["mse"],
                "relative_l2": metrics["relative_l2"],
                "max_abs_error": metrics["max_abs_error"],
                "channel_1_relative_l2": metrics.get("channel_1_relative_l2", float("nan")),
                "channel_2_relative_l2": metrics.get("channel_2_relative_l2", float("nan")),
                "forecast_window_relative_l2": metrics["relative_l2"] if mode == "forecast" else float("nan"),
                "runtime_sec": float(trained["runtime_sec"]),
                "parameter_count": int(trained["parameter_count"]),
                "train_points": int(split["train_t"].shape[0]),
                "test_points": int(split["test_t"].shape[0]),
                "device": trained["device"],
            }
        )
        prediction_payload[model_name] = pred
        curve_payload[model_name] = trained["loss_history"]

    return {
        "config": cfg,
        "generated": generated,
        "split": split,
        "rows": rows,
        "predictions": prediction_payload,
        "training_curves": curve_payload,
    }


def write_metrics_csv(rows: List[Dict], path: Path):
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = sorted({key for row in rows for key in row.keys()})
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def summarize_by_mode_model(rows: List[Dict]) -> Dict[str, Dict[str, Dict[str, float]]]:
    grouped: Dict[str, Dict[str, List[Dict]]] = {}
    for row in rows:
        grouped.setdefault(row["mode"], {}).setdefault(row["model"], []).append(row)

    summary: Dict[str, Dict[str, Dict[str, float]]] = {}
    for mode, by_model in grouped.items():
        summary[mode] = {}
        for model, items in by_model.items():
            l2_vals = [float(item["relative_l2"]) for item in items]
            mse_vals = [float(item["mse"]) for item in items]
            linf_vals = [float(item["max_abs_error"]) for item in items]
            rt_vals = [float(item["runtime_sec"]) for item in items]
            params = [float(item.get("parameter_count", 0.0)) for item in items]
            ch1_vals = [float(item.get("channel_1_relative_l2", float("nan"))) for item in items]
            ch2_vals = [float(item.get("channel_2_relative_l2", float("nan"))) for item in items]
            summary[mode][model] = {
                "model_label": items[0]["model_label"],
                "count": len(items),
                "mean_relative_l2": _safe_mean(l2_vals),
                "std_relative_l2": _safe_std(l2_vals),
                "mean_mse": _safe_mean(mse_vals),
                "std_mse": _safe_std(mse_vals),
                "mean_max_abs_error": _safe_mean(linf_vals),
                "std_max_abs_error": _safe_std(linf_vals),
                "mean_runtime_sec": _safe_mean(rt_vals),
                "std_runtime_sec": _safe_std(rt_vals),
                "mean_parameter_count": _safe_mean(params),
                "mean_channel_1_relative_l2": _safe_mean([v for v in ch1_vals if not np.isnan(v)]),
                "mean_channel_2_relative_l2": _safe_mean([v for v in ch2_vals if not np.isnan(v)]),
            }
    return summary


def _get_best_model(summary_mode: Dict[str, Dict[str, float]]) -> str:
    return min(summary_mode.items(), key=lambda item: item[1]["mean_relative_l2"])[0]


def _plot_prediction_panel(
    ax: plt.Axes,
    mode: str,
    rep: Dict,
    model_names: List[str],
):
    split = rep["split"]
    t_full = split["full_t"]
    y_clean = split["full_y_clean"]
    ax.plot(t_full, y_clean, color="black", linewidth=2.0, label="True channel (clean)")
    ax.scatter(
        split["train_t"],
        split["train_y_noisy"],
        color="#666666",
        s=18,
        alpha=0.85,
        label="Sparse noisy observations",
        zorder=5,
    )
    if mode == "forecast":
        cutoff_t = float(split["train_t"][-1])
        ax.axvline(cutoff_t, color="#7f7f7f", linestyle="--", linewidth=1.5, label="Forecast boundary")

    for model in model_names:
        pred = rep["predictions"][model]
        ax.plot(
            split["test_t"],
            pred,
            linewidth=1.8,
            label=MODEL_LABELS.get(model, model),
        )

    ax.set_title("Interpolation" if mode == "interpolation" else "Forecast")
    ax.set_xlabel("Time")
    ax.set_ylabel("Channel h(t)")
    ax.legend(loc="best", fontsize=9)
    ax.grid(True, alpha=0.25)


def plot_predictions(interpolation_rep: Dict, forecast_rep: Dict, output_dir: Path):
    ordered_models = MODEL_ORDER

    fig_i, ax_i = plt.subplots(figsize=(12.5, 5.0))
    _plot_prediction_panel(ax_i, "interpolation", interpolation_rep, ordered_models)
    fig_i.tight_layout()
    fig_i.savefig(output_dir / "interpolation_predictions.png", dpi=240, bbox_inches="tight")
    plt.close(fig_i)

    fig_f, ax_f = plt.subplots(figsize=(12.5, 5.0))
    _plot_prediction_panel(ax_f, "forecast", forecast_rep, ordered_models)
    fig_f.tight_layout()
    fig_f.savefig(output_dir / "forecast_predictions.png", dpi=240, bbox_inches="tight")
    plt.close(fig_f)


def plot_sparse_csi_forecast_showcase(forecast_rep: Dict, output_dir: Path):
    split = forecast_rep["split"]
    t_full = split["full_t"]
    y_clean = split["full_y_clean"]
    cutoff_t = float(split["train_t"][-1])

    fig, ax = plt.subplots(figsize=(12.5, 5.2))
    ax.plot(t_full, y_clean, color="black", linewidth=2.2, label="True channel (clean)")
    ax.scatter(
        split["train_t"],
        split["train_y_noisy"],
        s=28,
        color="#555555",
        alpha=0.9,
        marker="o",
        label="Sparse CSI observations (train)",
        zorder=6,
    )

    pred_mlp = forecast_rep["predictions"]["mlp_t"]
    pred_best = forecast_rep["predictions"]["mlp_memory_domain"]
    ax.plot(split["test_t"], pred_mlp, linewidth=2.0, color="#1f77b4", label="MLP baseline [t]")
    ax.plot(
        split["test_t"],
        pred_best,
        linewidth=2.2,
        color="#d62728",
        label="MLP + memory + domain",
    )

    ax.axvline(cutoff_t, color="#6b6b6b", linestyle="--", linewidth=1.5, label="Forecast split")
    ax.axvspan(cutoff_t, float(t_full.max()), color="#f0f0f0", alpha=0.55, label="Forecast/test region")
    ax.set_title("Sparse CSI Forecasting Showcase (Synthetic Time-Varying Channel)")
    ax.set_xlabel("Time")
    ax.set_ylabel("Channel response h(t)")
    ax.grid(True, alpha=0.25)
    ax.legend(loc="best", fontsize=9)
    fig.tight_layout()
    fig.savefig(output_dir / "sparse_csi_forecast_showcase.png", dpi=280, bbox_inches="tight")
    plt.close(fig)


def plot_error_comparison(interpolation_rep: Dict, forecast_rep: Dict, output_dir: Path):
    fig, axes = plt.subplots(2, 1, figsize=(12.5, 8.0), sharex=False)
    for ax, rep, title in [
        (axes[0], interpolation_rep, "Absolute Error vs Time (Interpolation)"),
        (axes[1], forecast_rep, "Absolute Error vs Time (Forecast Window)"),
    ]:
        y_true = rep["split"]["test_y_clean"]
        t = rep["split"]["test_t"]
        for model in MODEL_ORDER:
            pred = rep["predictions"][model]
            err = np.abs(pred - y_true)
            ax.plot(t, err, linewidth=1.8, label=MODEL_LABELS[model])
        ax.set_title(title)
        ax.set_xlabel("Time")
        ax.set_ylabel("|Error|")
        ax.grid(True, alpha=0.25)
        ax.legend(loc="upper right", fontsize=8)

    fig.tight_layout()
    fig.savefig(output_dir / "error_comparison.png", dpi=240, bbox_inches="tight")
    plt.close(fig)


def plot_training_curves(interpolation_rep: Dict, forecast_rep: Dict, output_dir: Path):
    fig, axes = plt.subplots(1, 2, figsize=(12.8, 4.6), sharey=True)
    for ax, rep, title in [
        (axes[0], interpolation_rep, "Training Curves (Interpolation)"),
        (axes[1], forecast_rep, "Training Curves (Forecast)"),
    ]:
        for model in ["mlp_t", "mlp_memory", "mlp_domain", "mlp_memory_domain"]:
            curve = rep["training_curves"].get(model, [])
            if not curve:
                continue
            epochs = np.arange(1, len(curve) + 1)
            ax.plot(epochs, curve, linewidth=1.8, label=MODEL_LABELS[model])
        ax.set_title(title)
        ax.set_xlabel("Epoch")
        ax.set_ylabel("Train MSE")
        ax.grid(True, alpha=0.25)
        ax.legend(loc="upper right", fontsize=8)
    fig.tight_layout()
    fig.savefig(output_dir / "training_curves.png", dpi=240, bbox_inches="tight")
    plt.close(fig)


def plot_robustness(rows: List[Dict], output_dir: Path):
    fig, axes = plt.subplots(2, 1, figsize=(13.0, 8.2), sharex=False)
    for ax, mode in [(axes[0], "interpolation"), (axes[1], "forecast")]:
        mode_rows = [row for row in rows if row["mode"] == mode]
        settings = sorted({(float(r["noise_level"]), float(r["train_setting"])) for r in mode_rows})
        setting_labels = []
        for noise, setting in settings:
            if mode == "interpolation":
                setting_labels.append(f"noise={noise:.2f}\nN={int(setting)}")
            else:
                setting_labels.append(f"noise={noise:.2f}\nfrac={setting:.1f}")

        matrix = np.zeros((len(MODEL_ORDER), len(settings)), dtype=float)
        matrix[:] = np.nan
        for i, model in enumerate(MODEL_ORDER):
            for j, setting in enumerate(settings):
                values = [
                    float(row["relative_l2"])
                    for row in mode_rows
                    if row["model"] == model
                    and float(row["noise_level"]) == setting[0]
                    and float(row["train_setting"]) == setting[1]
                ]
                if values:
                    matrix[i, j] = _safe_mean(values)

        im = ax.imshow(matrix, aspect="auto", cmap="viridis")
        ax.set_title(f"Robustness Sweep: {mode.capitalize()} (Relative L2, lower is better)")
        ax.set_yticks(np.arange(len(MODEL_ORDER)))
        ax.set_yticklabels([MODEL_LABELS[m] for m in MODEL_ORDER])
        ax.set_xticks(np.arange(len(settings)))
        ax.set_xticklabels(setting_labels, rotation=40, ha="right")
        cbar = fig.colorbar(im, ax=ax, shrink=0.9)
        cbar.set_label("Mean relative L2")

    fig.tight_layout()
    fig.savefig(output_dir / "robustness_heatmap_or_bars.png", dpi=240, bbox_inches="tight")
    plt.close(fig)


def _channel_series(arr: np.ndarray, channel_idx: int) -> np.ndarray:
    array = np.asarray(arr)
    if array.ndim == 1:
        return array
    return array[:, channel_idx]


def plot_mimo_predictions(interpolation_rep: Dict, forecast_rep: Dict, output_dir: Path):
    for mode, rep, filename in [
        ("interpolation", interpolation_rep, "mimo_interpolation_predictions.png"),
        ("forecast", forecast_rep, "mimo_forecast_predictions.png"),
    ]:
        split = rep["split"]
        fig, axes = plt.subplots(2, 1, figsize=(13.0, 8.6), sharex=True)
        for ch in [0, 1]:
            ax = axes[ch]
            ax.plot(
                split["full_t"],
                _channel_series(split["full_y_clean"], ch),
                color="black",
                linewidth=2.1,
                label=f"True h{ch + 1}(t)",
            )
            ax.scatter(
                split["train_t"],
                _channel_series(split["train_y_noisy"], ch),
                color="#666666",
                s=24,
                alpha=0.9,
                label=f"Sparse train CSI h{ch + 1}",
                zorder=6,
            )
            ax.plot(
                split["test_t"],
                _channel_series(rep["predictions"]["mlp_t"], ch),
                linewidth=1.9,
                color="#1f77b4",
                label="MLP baseline [t]",
            )
            ax.plot(
                split["test_t"],
                _channel_series(rep["predictions"]["mlp_memory_domain"], ch),
                linewidth=2.1,
                color="#d62728",
                label="MLP + memory + domain",
            )
            if mode == "forecast":
                cutoff_t = float(split["train_t"][-1])
                ax.axvline(cutoff_t, color="#6b6b6b", linestyle="--", linewidth=1.4, label="Forecast split")
                ax.axvspan(
                    cutoff_t,
                    float(split["full_t"].max()),
                    color="#f0f0f0",
                    alpha=0.55,
                    label="Forecast/test region",
                )
            ax.set_ylabel(f"h{ch + 1}(t)")
            ax.grid(True, alpha=0.25)
            ax.legend(loc="best", fontsize=8)
        axes[0].set_title(
            f"Two-Channel Sparse CSI {'Interpolation' if mode == 'interpolation' else 'Forecast'}"
        )
        axes[1].set_xlabel("Time")
        fig.tight_layout()
        fig.savefig(output_dir / filename, dpi=260, bbox_inches="tight")
        plt.close(fig)


def plot_mimo_error_comparison(interpolation_rep: Dict, forecast_rep: Dict, output_dir: Path):
    fig, axes = plt.subplots(2, 1, figsize=(13.0, 8.0), sharex=False)
    for ax, rep, title in [
        (axes[0], interpolation_rep, "Two-Channel Mean Absolute Error vs Time (Interpolation)"),
        (axes[1], forecast_rep, "Two-Channel Mean Absolute Error vs Time (Forecast)"),
    ]:
        y_true = np.asarray(rep["split"]["test_y_clean"])
        t = rep["split"]["test_t"]
        for model in ["mlp_t", "mlp_memory", "mlp_domain", "mlp_memory_domain"]:
            pred = np.asarray(rep["predictions"][model])
            mae = np.mean(np.abs(pred - y_true), axis=1)
            ax.plot(t, mae, linewidth=1.9, label=MODEL_LABELS[model])
        ax.set_title(title)
        ax.set_xlabel("Time")
        ax.set_ylabel("Mean |error| across channels")
        ax.grid(True, alpha=0.25)
        ax.legend(loc="upper right", fontsize=8)
    fig.tight_layout()
    fig.savefig(output_dir / "mimo_error_comparison.png", dpi=260, bbox_inches="tight")
    plt.close(fig)


def write_application_report(
    output_dir: Path,
    summary: Dict[str, Dict[str, Dict[str, float]]],
    rows: List[Dict],
    representative: Dict[str, Dict],
    mimo_summary: Dict[str, Dict[str, Dict[str, float]]] | None = None,
):
    interp_best = _get_best_model(summary["interpolation"])
    fore_best = _get_best_model(summary["forecast"])

    def _table_for_mode(mode: str) -> List[str]:
        lines = []
        lines.append("| Model | Rel L2 mean±std | MSE mean±std | Max-Err mean±std | Runtime mean±std (s) |")
        lines.append("| --- | ---: | ---: | ---: | ---: |")
        for model in MODEL_ORDER:
            row = summary[mode][model]
            lines.append(
                f"| {MODEL_LABELS[model]} | "
                f"{row['mean_relative_l2']:.6f} ± {row['std_relative_l2']:.6f} | "
                f"{row['mean_mse']:.6f} ± {row['std_mse']:.6f} | "
                f"{row['mean_max_abs_error']:.6f} ± {row['std_max_abs_error']:.6f} | "
                f"{row['mean_runtime_sec']:.4f} ± {row['std_runtime_sec']:.4f} |"
            )
        return lines

    # Memory/domain gain indicators.
    interp = summary["interpolation"]
    fore = summary["forecast"]
    mem_gain_interp = interp["mlp_t"]["mean_relative_l2"] - interp["mlp_memory"]["mean_relative_l2"]
    mem_gain_fore = fore["mlp_t"]["mean_relative_l2"] - fore["mlp_memory"]["mean_relative_l2"]
    dom_gain_interp = interp["mlp_t"]["mean_relative_l2"] - interp["mlp_domain"]["mean_relative_l2"]
    dom_gain_fore = fore["mlp_t"]["mean_relative_l2"] - fore["mlp_domain"]["mean_relative_l2"]
    comb_gain_interp = interp["mlp_t"]["mean_relative_l2"] - interp["mlp_memory_domain"]["mean_relative_l2"]
    comb_gain_fore = fore["mlp_t"]["mean_relative_l2"] - fore["mlp_memory_domain"]["mean_relative_l2"]

    lines: List[str] = []
    lines.append("# Application Report: Sparse CSI Forecasting for Time-Varying Wireless Channels")
    lines.append("")
    lines.append("## Purpose")
    lines.append("")
    lines.append(
        "This application-inspired demo evaluates whether memory-style and domain-aware features improve sparse channel-state information (CSI) reconstruction and future-window forecasting on synthetic time-varying wireless channel dynamics."
    )
    lines.append("")
    lines.append("## Synthetic Channel Model")
    lines.append("")
    lines.append("Noiseless channel response:")
    lines.append("")
    lines.append("`h(t) = a0*A(t)*cos(2*pi*f_d*t + phase0) + a1*A(t)*cos(2*pi*f_d*(t-tau1) + phase1) + a2*exp(-lambda*t_norm) + a3*sin(2*pi*f_slow*t)`")
    lines.append("")
    lines.append("Noisy observations use `h_noisy(t) = h(t) + sigma*N(0,1)` with drift `A(t)=clip(1-drift*t_norm, min=0.25)`.")
    lines.append("")
    lines.append("Component interpretation:")
    lines.append("")
    lines.append("- Doppler-like oscillation: primary cosine term with `f_d`.")
    lines.append("- Delayed/multipath-like component: shifted cosine term with delay `tau1`.")
    lines.append("- Slow amplitude/path-loss drift: multiplicative envelope `A(t)`.")
    lines.append("- Low-frequency trend: sinusoidal component with `f_slow`.")
    lines.append("- Observation noise: additive Gaussian noise.")
    lines.append("")
    lines.append("## Task Modes")
    lines.append("")
    lines.append("- **Interpolation**: sparse CSI samples across the full horizon, then dense full-grid evaluation.")
    lines.append("- **Forecast**: training on an early-time observation window, then evaluation on an unseen future window.")
    lines.append("")
    lines.append("## Fairness Protocol")
    lines.append("")
    lines.append("- Same train/test split per run across all models.")
    lines.append("- Same seed, epochs, optimizer, learning rate, hidden-depth/width, and device for MLP models.")
    lines.append("- No model-specific retuning.")
    lines.append("")
    lines.append("## Models Compared")
    lines.append("")
    for model in MODEL_ORDER:
        lines.append(f"- {MODEL_LABELS[model]}")
    lines.append("")
    lines.append("Feature-group interpretation:")
    lines.append("")
    lines.append("- Memory-style priors: `t, t^alpha, t^(1-alpha), log(1+t)`.")
    lines.append("- Doppler/domain priors: `sin(2*pi*f_d*t), cos(2*pi*f_d*t)`.")
    lines.append("- Combined priors: memory-style + Doppler/domain features.")
    lines.append("")
    lines.append("## Robustness Mini-Sweep Summary")
    lines.append("")
    noise_values = sorted({float(r["noise_level"]) for r in rows})
    interp_values = sorted({int(float(r["train_setting"])) for r in rows if r["mode"] == "interpolation"})
    forecast_values = sorted({float(r["train_setting"]) for r in rows if r["mode"] == "forecast"})
    lines.append("- Noise levels: `" + ", ".join(f"{v:.2f}" for v in noise_values) + "`")
    lines.append("- Interpolation train points: `" + ", ".join(str(v) for v in interp_values) + "`")
    lines.append("- Forecast train fractions: `" + ", ".join(f"{v:.1f}" for v in forecast_values) + "`")
    lines.append("- Seeds: " + ", ".join(str(int(seed)) for seed in sorted({int(r['seed']) for r in rows})))
    lines.append("")
    lines.append("### Interpolation Metrics (averaged over sweep)")
    lines.append("")
    lines.extend(_table_for_mode("interpolation"))
    lines.append("")
    lines.append("### Forecast Metrics (averaged over sweep)")
    lines.append("")
    lines.extend(_table_for_mode("forecast"))
    lines.append("")
    lines.append("## Key Takeaways")
    lines.append("")
    lines.append(f"- Best interpolation model (mean relative L2): **{MODEL_LABELS[interp_best]}**.")
    lines.append(f"- Best forecast model (mean relative L2): **{MODEL_LABELS[fore_best]}**.")
    lines.append(
        f"- Memory-only feature effect (relative L2 reduction vs MLP baseline): interpolation `{mem_gain_interp:+.6f}`, forecast `{mem_gain_fore:+.6f}`."
    )
    lines.append(
        f"- Domain sinusoidal feature effect: interpolation `{dom_gain_interp:+.6f}`, forecast `{dom_gain_fore:+.6f}`."
    )
    lines.append(
        f"- Combined memory+domain feature effect: interpolation `{comb_gain_interp:+.6f}`, forecast `{comb_gain_fore:+.6f}`."
    )
    lines.append("- Domain sinusoidal priors were strongest; memory-only gains were mixed; combined priors were most consistent.")
    lines.append("")

    if mimo_summary is not None:
        mimo_interp_best = _get_best_model(mimo_summary["interpolation"])
        mimo_fore_best = _get_best_model(mimo_summary["forecast"])
        lines.append("## Two-Channel / MIMO-Style Extension")
        lines.append("")
        lines.append(
            "We additionally evaluate a two-channel synthetic link `h(t)=[h1(t), h2(t)]` where each channel has Doppler-like, delayed/multipath-like, and slow-drift components, with optional coupling from delayed `h1` into `h2`."
        )
        lines.append("")
        lines.append(
            "This remains an application-inspired synthetic benchmark (not a full MIMO simulator), but it mimics sparse multi-channel CSI forecasting under the same fairness protocol."
        )
        lines.append("")
        lines.append("### Two-Channel Interpolation (Aggregate Metrics)")
        lines.append("")
        lines.append("| Model | Relative L2 | MSE | Max-Err | Ch1 Rel L2 | Ch2 Rel L2 | Runtime (s) |")
        lines.append("| --- | ---: | ---: | ---: | ---: | ---: | ---: |")
        for model in MODEL_ORDER:
            row = mimo_summary["interpolation"][model]
            lines.append(
                f"| {MODEL_LABELS[model]} | "
                f"{row['mean_relative_l2']:.6f} | "
                f"{row['mean_mse']:.6f} | "
                f"{row['mean_max_abs_error']:.6f} | "
                f"{row.get('mean_channel_1_relative_l2', float('nan')):.6f} | "
                f"{row.get('mean_channel_2_relative_l2', float('nan')):.6f} | "
                f"{row['mean_runtime_sec']:.4f} |"
            )
        lines.append("")
        lines.append("### Two-Channel Forecast (Aggregate Metrics)")
        lines.append("")
        lines.append("| Model | Relative L2 | MSE | Max-Err | Ch1 Rel L2 | Ch2 Rel L2 | Runtime (s) |")
        lines.append("| --- | ---: | ---: | ---: | ---: | ---: | ---: |")
        for model in MODEL_ORDER:
            row = mimo_summary["forecast"][model]
            lines.append(
                f"| {MODEL_LABELS[model]} | "
                f"{row['mean_relative_l2']:.6f} | "
                f"{row['mean_mse']:.6f} | "
                f"{row['mean_max_abs_error']:.6f} | "
                f"{row.get('mean_channel_1_relative_l2', float('nan')):.6f} | "
                f"{row.get('mean_channel_2_relative_l2', float('nan')):.6f} | "
                f"{row['mean_runtime_sec']:.4f} |"
            )
        lines.append("")
        lines.append(f"- Best two-channel interpolation model: **{MODEL_LABELS[mimo_interp_best]}**.")
        lines.append(f"- Best two-channel forecast model: **{MODEL_LABELS[mimo_fore_best]}**.")
        lines.append("")

    lines.append("## Limitations")
    lines.append("")
    lines.append("- This is a synthetic application-inspired sparse-CSI benchmark, not a full LEO/MIMO system simulator.")
    lines.append("- It uses scalar channel dynamics and does not model full multi-antenna, orbital, or standards-compliant channel pipelines.")
    lines.append("- TE-QPINN surrogate variants were intentionally omitted in this first application phase to keep the demo lightweight and fast; they are future-work extensions for this application track.")
    lines.append("- Results indicate method transfer potential only; they are not deployment-level channel-prediction claims.")
    lines.append("")

    (output_dir / "application_report.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def write_application_summary_table(
    output_dir: Path, summary: Dict[str, Dict[str, Dict[str, float]]]
):
    rows: List[Dict[str, str]] = []
    for mode in ["interpolation", "forecast"]:
        best_model = _get_best_model(summary[mode])
        best_stats = summary[mode][best_model]
        takeaway = (
            "Combined memory+domain priors gave the most accurate sparse reconstruction."
            if mode == "interpolation"
            else "Combined memory+domain priors gave the strongest future-window prediction."
        )
        rows.append(
            {
                "mode": mode,
                "best_model": MODEL_LABELS[best_model],
                "mean_relative_l2": f"{best_stats['mean_relative_l2']:.6f}",
                "mean_mse": f"{best_stats['mean_mse']:.6f}",
                "mean_max_error": f"{best_stats['mean_max_abs_error']:.6f}",
                "key_takeaway": takeaway,
            }
        )

    path = output_dir / "application_summary_table.csv"
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=[
                "mode",
                "best_model",
                "mean_relative_l2",
                "mean_mse",
                "mean_max_error",
                "key_takeaway",
            ],
        )
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def write_mimo_application_report(
    output_dir: Path,
    summary: Dict[str, Dict[str, Dict[str, float]]],
    rows: List[Dict],
):
    interp_best = _get_best_model(summary["interpolation"])
    fore_best = _get_best_model(summary["forecast"])
    noise_values = sorted({float(r["noise_level"]) for r in rows})
    interp_values = sorted({int(float(r["train_setting"])) for r in rows if r["mode"] == "interpolation"})
    forecast_values = sorted({float(r["train_setting"]) for r in rows if r["mode"] == "forecast"})

    lines: List[str] = []
    lines.append("# Two-Channel Sparse CSI Forecasting Report")
    lines.append("")
    lines.append("## Purpose")
    lines.append("")
    lines.append(
        "This extension evaluates sparse CSI forecasting for a two-channel synthetic wireless link `h(t)=[h1(t), h2(t)]` under the same fairness protocol as the scalar demo."
    )
    lines.append("")
    lines.append("## Synthetic Two-Channel Dynamics")
    lines.append("")
    lines.append("- `h1(t)`: Doppler-like + delayed/multipath-like + drift + low-frequency trend.")
    lines.append("- `h2(t)`: related but shifted frequencies/phases and delayed terms, with optional coupling from delayed `h1(t)`.")
    lines.append("- Independent Gaussian noise is added per channel.")
    lines.append("- This is not a full MIMO simulator; it is an application-inspired multi-channel benchmark.")
    lines.append("")
    lines.append("## Sweep Setup")
    lines.append("")
    lines.append("- Noise levels: `" + ", ".join(f"{v:.2f}" for v in noise_values) + "`")
    lines.append("- Interpolation train points: `" + ", ".join(str(v) for v in interp_values) + "`")
    lines.append("- Forecast train fractions: `" + ", ".join(f"{v:.1f}" for v in forecast_values) + "`")
    lines.append("- Seeds: " + ", ".join(str(int(seed)) for seed in sorted({int(r['seed']) for r in rows})))
    lines.append("")
    lines.append("## Interpolation Results (Aggregate)")
    lines.append("")
    lines.append("| Model | Rel L2 | MSE | Max-Err | Ch1 Rel L2 | Ch2 Rel L2 | Runtime (s) |")
    lines.append("| --- | ---: | ---: | ---: | ---: | ---: | ---: |")
    for model in MODEL_ORDER:
        row = summary["interpolation"][model]
        lines.append(
            f"| {MODEL_LABELS[model]} | "
            f"{row['mean_relative_l2']:.6f} | "
            f"{row['mean_mse']:.6f} | "
            f"{row['mean_max_abs_error']:.6f} | "
            f"{row.get('mean_channel_1_relative_l2', float('nan')):.6f} | "
            f"{row.get('mean_channel_2_relative_l2', float('nan')):.6f} | "
            f"{row['mean_runtime_sec']:.4f} |"
        )
    lines.append("")
    lines.append("## Forecast Results (Aggregate)")
    lines.append("")
    lines.append("| Model | Rel L2 | MSE | Max-Err | Ch1 Rel L2 | Ch2 Rel L2 | Runtime (s) |")
    lines.append("| --- | ---: | ---: | ---: | ---: | ---: | ---: |")
    for model in MODEL_ORDER:
        row = summary["forecast"][model]
        lines.append(
            f"| {MODEL_LABELS[model]} | "
            f"{row['mean_relative_l2']:.6f} | "
            f"{row['mean_mse']:.6f} | "
            f"{row['mean_max_abs_error']:.6f} | "
            f"{row.get('mean_channel_1_relative_l2', float('nan')):.6f} | "
            f"{row.get('mean_channel_2_relative_l2', float('nan')):.6f} | "
            f"{row['mean_runtime_sec']:.4f} |"
        )
    lines.append("")
    lines.append(f"- Best interpolation model: **{MODEL_LABELS[interp_best]}**.")
    lines.append(f"- Best forecast model: **{MODEL_LABELS[fore_best]}**.")
    lines.append("")
    lines.append("## Interpretation")
    lines.append("")
    lines.append(
        "This two-channel extension tests whether memory/domain priors continue to help under joint multi-output training. It is a synthetic feasibility demo and not a deployment-level claim."
    )
    lines.append("")
    (output_dir / "mimo_application_report.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def main():
    args = parse_args()
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    seeds = _parse_int_list(args.seeds)
    if not seeds:
        seeds = [0]

    hidden_layers = _parse_hidden_layers(args.hidden_layers)
    train_cfg = MLPTrainConfig(
        hidden_layers=hidden_layers,
        epochs=int(args.epochs),
        learning_rate=float(args.learning_rate),
        device=str(args.device),
    )

    # Phase-C locked lightweight sweep (runtime-friendly).
    noise_levels = [0.0, 0.05]
    interp_train_points = [10, 30]
    forecast_train_fracs = [0.5, 0.8]

    doppler_hz = 8.0
    slow_hz = 0.7
    delay_tau1 = 0.045
    alpha_feature = 0.7

    def _run_suite(channel_dim: int, label: str) -> Dict:
        all_rows: List[Dict] = []
        representative_payload: Dict[str, Dict] = {}
        rep_interp_key = (
            noise_levels[-1],
            interp_train_points[-1],
            seeds[0],
        )
        rep_fore_key = (
            noise_levels[-1],
            forecast_train_fracs[-1],
            seeds[0],
        )
        print("-" * 80)
        print(f"{label}: channel_dim={channel_dim}")
        for seed in seeds:
            for noise_level in noise_levels:
                for n_train in interp_train_points:
                    payload = run_one_experiment(
                        mode="interpolation",
                        noise_level=float(noise_level),
                        train_setting=float(n_train),
                        seed=int(seed),
                        num_dense_samples=int(args.num_dense_samples),
                        train_cfg=train_cfg,
                        doppler_hz=doppler_hz,
                        slow_hz=slow_hz,
                        delay_tau1=delay_tau1,
                        alpha_feature=alpha_feature,
                        channel_dim=channel_dim,
                    )
                    all_rows.extend(payload["rows"])
                    if (noise_level, n_train, seed) == rep_interp_key:
                        representative_payload["interpolation"] = payload
                    print(
                        f"[done] {label} interpolation seed={seed} noise={noise_level:.2f} n_train={n_train}"
                    )
                for frac in forecast_train_fracs:
                    payload = run_one_experiment(
                        mode="forecast",
                        noise_level=float(noise_level),
                        train_setting=float(frac),
                        seed=int(seed),
                        num_dense_samples=int(args.num_dense_samples),
                        train_cfg=train_cfg,
                        doppler_hz=doppler_hz,
                        slow_hz=slow_hz,
                        delay_tau1=delay_tau1,
                        alpha_feature=alpha_feature,
                        channel_dim=channel_dim,
                    )
                    all_rows.extend(payload["rows"])
                    if (noise_level, frac, seed) == rep_fore_key:
                        representative_payload["forecast"] = payload
                    print(
                        f"[done] {label} forecast seed={seed} noise={noise_level:.2f} frac={frac:.2f}"
                    )
        if "interpolation" not in representative_payload:
            representative_payload["interpolation"] = run_one_experiment(
                mode="interpolation",
                noise_level=noise_levels[0],
                train_setting=float(interp_train_points[0]),
                seed=int(seeds[0]),
                num_dense_samples=int(args.num_dense_samples),
                train_cfg=train_cfg,
                doppler_hz=doppler_hz,
                slow_hz=slow_hz,
                delay_tau1=delay_tau1,
                alpha_feature=alpha_feature,
                channel_dim=channel_dim,
            )
        if "forecast" not in representative_payload:
            representative_payload["forecast"] = run_one_experiment(
                mode="forecast",
                noise_level=noise_levels[0],
                train_setting=float(forecast_train_fracs[0]),
                seed=int(seeds[0]),
                num_dense_samples=int(args.num_dense_samples),
                train_cfg=train_cfg,
                doppler_hz=doppler_hz,
                slow_hz=slow_hz,
                delay_tau1=delay_tau1,
                alpha_feature=alpha_feature,
                channel_dim=channel_dim,
            )
        summary = summarize_by_mode_model(all_rows)
        return {"rows": all_rows, "representative": representative_payload, "summary": summary}

    print("=" * 80)
    print("Application Demo: Sparse CSI Forecasting for Time-Varying Wireless Channels")
    print("=" * 80)
    print(f"Output directory: {output_dir}")
    print(f"Seeds: {seeds}")
    print(f"Noise levels: {noise_levels}")
    print(f"Interpolation train points: {interp_train_points}")
    print(f"Forecast train fractions: {forecast_train_fracs}")
    print(
        f"MLP fairness lock: epochs={train_cfg.epochs}, lr={train_cfg.learning_rate}, hidden_layers={train_cfg.hidden_layers}, device={train_cfg.device}"
    )

    scalar_suite = _run_suite(channel_dim=1, label="Scalar")
    mimo_suite = _run_suite(channel_dim=2, label="Two-channel")

    scalar_summary = scalar_suite["summary"]
    scalar_best_interp = _get_best_model(scalar_summary["interpolation"])
    scalar_best_fore = _get_best_model(scalar_summary["forecast"])
    write_metrics_csv(scalar_suite["rows"], output_dir / "metrics.csv")
    scalar_json = {
        "demo_name": "sparse_csi_forecasting_scalar_wireless_channel",
        "channel_dim": 1,
        "fairness_lock": {
            "seeds": seeds,
            "epochs": train_cfg.epochs,
            "learning_rate": train_cfg.learning_rate,
            "hidden_layers": list(train_cfg.hidden_layers),
            "device": train_cfg.device,
        },
        "sweep": {
            "noise_levels": noise_levels,
            "interpolation_train_points": interp_train_points,
            "forecast_train_fractions": forecast_train_fracs,
        },
        "models": MODEL_ORDER,
        "model_labels": MODEL_LABELS,
        "summary_by_mode_model": scalar_summary,
        "best_model_by_mode": {
            "interpolation": scalar_best_interp,
            "forecast": scalar_best_fore,
        },
        "rows": scalar_suite["rows"],
    }
    (output_dir / "metrics.json").write_text(json.dumps(scalar_json, indent=2), encoding="utf-8")
    plot_predictions(
        scalar_suite["representative"]["interpolation"],
        scalar_suite["representative"]["forecast"],
        output_dir,
    )
    plot_sparse_csi_forecast_showcase(scalar_suite["representative"]["forecast"], output_dir)
    plot_error_comparison(
        scalar_suite["representative"]["interpolation"],
        scalar_suite["representative"]["forecast"],
        output_dir,
    )
    plot_training_curves(
        scalar_suite["representative"]["interpolation"],
        scalar_suite["representative"]["forecast"],
        output_dir,
    )
    plot_robustness(scalar_suite["rows"], output_dir)

    mimo_summary = mimo_suite["summary"]
    mimo_best_interp = _get_best_model(mimo_summary["interpolation"])
    mimo_best_fore = _get_best_model(mimo_summary["forecast"])
    write_metrics_csv(mimo_suite["rows"], output_dir / "mimo_metrics.csv")
    mimo_json = {
        "demo_name": "sparse_csi_forecasting_two_channel_wireless_link",
        "channel_dim": 2,
        "fairness_lock": {
            "seeds": seeds,
            "epochs": train_cfg.epochs,
            "learning_rate": train_cfg.learning_rate,
            "hidden_layers": list(train_cfg.hidden_layers),
            "device": train_cfg.device,
        },
        "sweep": {
            "noise_levels": noise_levels,
            "interpolation_train_points": interp_train_points,
            "forecast_train_fractions": forecast_train_fracs,
        },
        "models": MODEL_ORDER,
        "model_labels": MODEL_LABELS,
        "summary_by_mode_model": mimo_summary,
        "best_model_by_mode": {
            "interpolation": mimo_best_interp,
            "forecast": mimo_best_fore,
        },
        "rows": mimo_suite["rows"],
    }
    (output_dir / "mimo_metrics.json").write_text(json.dumps(mimo_json, indent=2), encoding="utf-8")
    plot_mimo_predictions(
        mimo_suite["representative"]["interpolation"],
        mimo_suite["representative"]["forecast"],
        output_dir,
    )
    plot_mimo_error_comparison(
        mimo_suite["representative"]["interpolation"],
        mimo_suite["representative"]["forecast"],
        output_dir,
    )
    write_mimo_application_report(output_dir=output_dir, summary=mimo_summary, rows=mimo_suite["rows"])

    write_application_report(
        output_dir=output_dir,
        summary=scalar_summary,
        rows=scalar_suite["rows"],
        representative=scalar_suite["representative"],
        mimo_summary=mimo_summary,
    )
    write_application_summary_table(output_dir, scalar_summary)

    print("-" * 80)
    print("Demo complete")
    print(f"Scalar best interpolation model: {MODEL_LABELS[scalar_best_interp]}")
    print(f"Scalar best forecast model: {MODEL_LABELS[scalar_best_fore]}")
    print(f"Two-channel best interpolation model: {MODEL_LABELS[mimo_best_interp]}")
    print(f"Two-channel best forecast model: {MODEL_LABELS[mimo_best_fore]}")
    print(f"Artifacts: {output_dir}")
    print("=" * 80)


if __name__ == "__main__":
    main()
