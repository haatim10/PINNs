#!/usr/bin/env python3
"""Generate publication-style paper figures from existing benchmark artifacts.

This script only reads existing output files and writes polished figures to:
    paper/figures/
"""

from __future__ import annotations

import shutil
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parent.parent
PAPER_FIG_DIR = ROOT / "paper" / "figures"

CONFIRMATORY_SUMMARY = (
    ROOT / "outputs" / "benchmarks" / "te_qpinn_memory_confirmatory_10seed" / "summary.csv"
)
ALPHA_BETA_SUMMARY = (
    ROOT / "outputs" / "benchmarks" / "te_qpinn_memory_alpha07_beta03_5seed" / "summary.csv"
)
OPTIMIZER_SUMMARY = (
    ROOT / "outputs" / "benchmarks" / "te_qpinn_optimizer_sensitivity" / "summary.csv"
)

APP_DIR = ROOT / "outputs" / "applications" / "wireless_channel_demo"
APP_METRICS = APP_DIR / "metrics.csv"
APP_MIMO_METRICS = APP_DIR / "mimo_metrics.csv"
APP_SUMMARY = APP_DIR / "application_summary_table.csv"
APP_SHOWCASE = APP_DIR / "sparse_csi_forecast_showcase.png"
APP_MIMO_FORECAST = APP_DIR / "mimo_forecast_predictions.png"
APP_MIMO_INTERP = APP_DIR / "mimo_interpolation_predictions.png"


plt.style.use("seaborn-v0_8-whitegrid")
plt.rcParams.update(
    {
        "figure.dpi": 180,
        "savefig.dpi": 300,
        "font.size": 11,
        "axes.titlesize": 13,
        "axes.labelsize": 11,
        "legend.fontsize": 10,
        "xtick.labelsize": 10,
        "ytick.labelsize": 10,
        "font.family": "DejaVu Sans",
    }
)


SHORT_LABELS = {
    "classical_pi": "Classical",
    "classical_memory_pi": "Classical+Mem",
    "te_fixed_pi": "TE",
    "te_layernorm_post_quantum_pi": "TE+LN",
    "te_memory_analytic_pi": "TE+Mem",
}

ORDER_MAIN = [
    "classical_pi",
    "classical_memory_pi",
    "te_fixed_pi",
    "te_layernorm_post_quantum_pi",
    "te_memory_analytic_pi",
]


def _mean_std_by_run(csv_path: Path) -> pd.DataFrame:
    df = pd.read_csv(csv_path)
    agg = (
        df.groupby("run_name", as_index=False)
        .agg(
            final_l2_mean=("final_l2", "mean"),
            final_l2_std=("final_l2", "std"),
            runtime_mean=("runtime_sec", "mean"),
            runtime_std=("runtime_sec", "std"),
        )
        .fillna(0.0)
    )
    return agg


def _value_labels(ax: plt.Axes, xs: np.ndarray, ys: np.ndarray, offset: float = 0.015) -> None:
    for x, y in zip(xs, ys):
        ax.text(x, y + offset, f"{y:.3f}", ha="center", va="bottom", fontsize=9)


def fig_main_l2_clean() -> None:
    agg = _mean_std_by_run(CONFIRMATORY_SUMMARY).set_index("run_name")
    runs = [r for r in ORDER_MAIN if r in agg.index]
    means = np.array([agg.loc[r, "final_l2_mean"] for r in runs], dtype=float)
    stds = np.array([agg.loc[r, "final_l2_std"] for r in runs], dtype=float)
    labels = [SHORT_LABELS.get(r, r) for r in runs]

    fig, ax = plt.subplots(figsize=(8.6, 4.6), constrained_layout=True)
    x = np.arange(len(runs))
    colors = ["#4e79a7", "#59a14f", "#f28e2b", "#b07aa1", "#e15759"]

    bars = ax.bar(x, means, yerr=stds, capsize=4, color=colors[: len(runs)], edgecolor="black", linewidth=0.6)
    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.set_ylabel("Final L2 (mean ± std)")
    ax.set_title("10-Seed Confirmatory Performance (Adam-only)")
    ax.set_ylim(0, max(means + stds) * 1.22)
    _value_labels(ax, x, means, offset=max(means) * 0.015)
    for bar in bars:
        bar.set_alpha(0.93)

    fig.savefig(PAPER_FIG_DIR / "fig_main_l2_clean.png")
    plt.close(fig)


def fig_accuracy_runtime_frontier() -> None:
    agg = _mean_std_by_run(CONFIRMATORY_SUMMARY).set_index("run_name")
    runs = [r for r in ORDER_MAIN if r in agg.index]

    fig, ax = plt.subplots(figsize=(7.8, 4.8), constrained_layout=True)
    for r in runs:
        x = float(agg.loc[r, "runtime_mean"])
        y = float(agg.loc[r, "final_l2_mean"])
        ax.scatter(x, y, s=80, zorder=3)
        ax.annotate(
            SHORT_LABELS.get(r, r),
            (x, y),
            textcoords="offset points",
            xytext=(6, 5),
            fontsize=9,
        )

    ax.set_xlabel("Mean Runtime (s)")
    ax.set_ylabel("Mean Final L2 (lower is better)")
    ax.set_title("Accuracy-Runtime Frontier (10-Seed Confirmatory)")
    ax.grid(True, linestyle="--", alpha=0.4)
    fig.savefig(PAPER_FIG_DIR / "fig_accuracy_runtime_frontier.png")
    plt.close(fig)


def fig_memory_effect_clean() -> None:
    agg = _mean_std_by_run(CONFIRMATORY_SUMMARY).set_index("run_name")
    pairs = [
        ("classical_pi", "classical_memory_pi", "Classical"),
        ("te_layernorm_post_quantum_pi", "te_memory_analytic_pi", "TE (LN base)"),
    ]

    left_vals = np.array([agg.loc[a, "final_l2_mean"] for a, _, _ in pairs], dtype=float)
    right_vals = np.array([agg.loc[b, "final_l2_mean"] for _, b, _ in pairs], dtype=float)
    left_err = np.array([agg.loc[a, "final_l2_std"] for a, _, _ in pairs], dtype=float)
    right_err = np.array([agg.loc[b, "final_l2_std"] for _, b, _ in pairs], dtype=float)

    x = np.arange(len(pairs))
    w = 0.32
    fig, ax = plt.subplots(figsize=(7.6, 4.6), constrained_layout=True)
    ax.bar(x - w / 2, left_vals, w, yerr=left_err, capsize=4, label="No memory features", color="#4e79a7")
    ax.bar(x + w / 2, right_vals, w, yerr=right_err, capsize=4, label="With memory features", color="#59a14f")
    ax.set_xticks(x)
    ax.set_xticklabels([name for _, _, name in pairs])
    ax.set_ylabel("Final L2 (mean ± std)")
    ax.set_title("Memory-Feature Effect in Confirmatory Runs")
    ax.legend(loc="upper right", frameon=True)

    for idx, (lv, rv) in enumerate(zip(left_vals, right_vals)):
        delta = rv - lv
        ax.text(x[idx], max(lv, rv) + 0.025, f"Δ={delta:+.3f}", ha="center", fontsize=9)

    fig.savefig(PAPER_FIG_DIR / "fig_memory_effect_clean.png")
    plt.close(fig)


def fig_alpha_beta_clean() -> None:
    agg = _mean_std_by_run(ALPHA_BETA_SUMMARY).set_index("run_name")
    runs = [r for r in ORDER_MAIN if r in agg.index]
    means = np.array([agg.loc[r, "final_l2_mean"] for r in runs], dtype=float)
    stds = np.array([agg.loc[r, "final_l2_std"] for r in runs], dtype=float)
    labels = [SHORT_LABELS.get(r, r) for r in runs]

    fig, ax = plt.subplots(figsize=(8.6, 4.6), constrained_layout=True)
    x = np.arange(len(runs))
    colors = ["#4e79a7", "#59a14f", "#f28e2b", "#b07aa1", "#e15759"]
    ax.bar(x, means, yerr=stds, capsize=4, color=colors[: len(runs)], edgecolor="black", linewidth=0.6)
    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.set_ylabel("Final L2 (mean ± std)")
    ax.set_title(r"Robustness Slice: $\alpha=0.7,\ \beta=0.3$")
    ax.set_ylim(0, max(means + stds) * 1.22)
    _value_labels(ax, x, means, offset=max(means) * 0.015)

    fig.savefig(PAPER_FIG_DIR / "fig_alpha_beta_clean.png")
    plt.close(fig)


def fig_optimizer_clean() -> None:
    df = pd.read_csv(OPTIMIZER_SUMMARY)
    name_map = {
        "classical_adam_pi": ("Classical", "Adam"),
        "classical_adam_lbfgs_pi": ("Classical", "Adam+LBFGS"),
        "te_fixed_adam_pi": ("TE", "Adam"),
        "te_fixed_adam_lbfgs_pi": ("TE", "Adam+LBFGS"),
        "te_layernorm_adam_pi": ("TE+LN", "Adam"),
        "te_layernorm_adam_lbfgs_pi": ("TE+LN", "Adam+LBFGS"),
    }
    rows = []
    for _, row in df.iterrows():
        rn = row["run_name"]
        if rn in name_map:
            fam, opt = name_map[rn]
            rows.append((fam, opt, float(row["final_l2"])))
    opt_df = pd.DataFrame(rows, columns=["family", "optimizer", "final_l2"])

    families = ["Classical", "TE", "TE+LN"]
    ad = np.array(
        [opt_df[(opt_df.family == fam) & (opt_df.optimizer == "Adam")]["final_l2"].iloc[0] for fam in families]
    )
    lb = np.array(
        [
            opt_df[(opt_df.family == fam) & (opt_df.optimizer == "Adam+LBFGS")]["final_l2"].iloc[0]
            for fam in families
        ]
    )
    x = np.arange(len(families))
    w = 0.34

    fig, ax = plt.subplots(figsize=(8.0, 4.6), constrained_layout=True)
    ax.bar(x - w / 2, ad, w, label="Adam", color="#4e79a7")
    ax.bar(x + w / 2, lb, w, label="Adam+LBFGS", color="#e15759")
    ax.set_xticks(x)
    ax.set_xticklabels(families)
    ax.set_ylabel("Final L2 (seed 42)")
    ax.set_title("Optimizer Sensitivity (Extended-Budget Note)")
    ax.legend(loc="upper right")
    ax.text(
        0.01,
        0.98,
        "Adam+LBFGS uses extended optimization budget.\nNot an equal-runtime comparison.",
        transform=ax.transAxes,
        va="top",
        ha="left",
        fontsize=9,
        bbox=dict(boxstyle="round,pad=0.3", facecolor="white", edgecolor="#bbbbbb"),
    )

    fig.savefig(PAPER_FIG_DIR / "fig_optimizer_clean.png")
    plt.close(fig)


def fig_sparse_csi_clean() -> None:
    if APP_SHOWCASE.exists():
        shutil.copy2(APP_SHOWCASE, PAPER_FIG_DIR / "fig_sparse_csi_clean.png")
        return
    # Fallback: summarize from metrics if showcase image is unavailable.
    df = pd.read_csv(APP_METRICS)
    sub = df[df["mode"] == "forecast"].copy()
    summary = sub.groupby("model_label", as_index=False)["relative_l2"].mean().sort_values("relative_l2")
    fig, ax = plt.subplots(figsize=(8.2, 4.8), constrained_layout=True)
    ax.bar(summary["model_label"], summary["relative_l2"], color="#4e79a7")
    ax.set_ylabel("Mean Relative L2")
    ax.set_title("Sparse CSI Forecasting (Fallback Summary)")
    ax.tick_params(axis="x", rotation=25)
    fig.savefig(PAPER_FIG_DIR / "fig_sparse_csi_clean.png")
    plt.close(fig)


def fig_mimo_clean() -> None:
    # Build a clean combined panel from existing interpolation/forecast figures.
    if APP_MIMO_INTERP.exists() and APP_MIMO_FORECAST.exists():
        interp = plt.imread(APP_MIMO_INTERP)
        forecast = plt.imread(APP_MIMO_FORECAST)
        fig, axes = plt.subplots(1, 2, figsize=(11.5, 4.8), constrained_layout=True)
        axes[0].imshow(interp)
        axes[0].axis("off")
        axes[0].set_title("Two-Channel Interpolation")
        axes[1].imshow(forecast)
        axes[1].axis("off")
        axes[1].set_title("Two-Channel Forecast")
        fig.savefig(PAPER_FIG_DIR / "fig_mimo_clean.png")
        plt.close(fig)
        return
    # Fallback summary from metrics.
    df = pd.read_csv(APP_MIMO_METRICS)
    sub = df[df["mode"] == "forecast"].copy()
    summary = sub.groupby("model_label", as_index=False)["relative_l2"].mean().sort_values("relative_l2")
    fig, ax = plt.subplots(figsize=(8.2, 4.8), constrained_layout=True)
    ax.bar(summary["model_label"], summary["relative_l2"], color="#59a14f")
    ax.set_ylabel("Relative L2")
    ax.set_title("Two-Channel Forecast (Fallback Summary)")
    ax.tick_params(axis="x", rotation=25)
    fig.savefig(PAPER_FIG_DIR / "fig_mimo_clean.png")
    plt.close(fig)


def fig_method_pipeline() -> None:
    from matplotlib.patches import FancyBboxPatch

    fig, ax = plt.subplots(figsize=(10.5, 3.8), constrained_layout=True)
    ax.axis("off")

    boxes = [
        (0.03, 0.55, 0.26, 0.33, "Input\n(x,t)"),
        (0.35, 0.55, 0.26, 0.33, "Classical Path\nPINN / PINN+Memory"),
        (0.67, 0.55, 0.30, 0.33, "TE Surrogate Path\nEmbedding → Angle → Sin/Cos\nPairwise Mix → Readout"),
        (0.35, 0.10, 0.26, 0.30, "Residual Engine\nCaputo/L1 + Product Integration"),
        (0.67, 0.10, 0.30, 0.30, "Losses\nPDE + BC + IC\n(+ metrics)"),
    ]

    for x, y, w, h, txt in boxes:
        patch = FancyBboxPatch(
            (x, y),
            w,
            h,
            boxstyle="round,pad=0.02,rounding_size=0.02",
            linewidth=1.0,
            edgecolor="#444444",
            facecolor="#f8f9fb",
        )
        ax.add_patch(patch)
        ax.text(x + w / 2, y + h / 2, txt, ha="center", va="center", fontsize=10)

    arrow = dict(arrowstyle="->", lw=1.2, color="#444444")
    ax.annotate("", xy=(0.35, 0.72), xytext=(0.29, 0.72), arrowprops=arrow)
    ax.annotate("", xy=(0.67, 0.72), xytext=(0.61, 0.72), arrowprops=arrow)
    ax.annotate("", xy=(0.48, 0.55), xytext=(0.48, 0.40), arrowprops=arrow)
    ax.annotate("", xy=(0.82, 0.55), xytext=(0.82, 0.40), arrowprops=arrow)
    ax.annotate("", xy=(0.67, 0.25), xytext=(0.61, 0.25), arrowprops=arrow)

    ax.set_title("Method Pipeline: Classical, TE-Surrogate, and Memory-Aware Paths", fontsize=12, pad=10)
    fig.savefig(PAPER_FIG_DIR / "fig_method_pipeline.png")
    plt.close(fig)


def main() -> None:
    PAPER_FIG_DIR.mkdir(parents=True, exist_ok=True)

    fig_main_l2_clean()
    fig_accuracy_runtime_frontier()
    fig_memory_effect_clean()
    fig_alpha_beta_clean()
    fig_optimizer_clean()
    fig_sparse_csi_clean()
    fig_mimo_clean()
    fig_method_pipeline()

    # Ensure these input files are touched so static analyzers know they are used.
    _ = APP_SUMMARY

    print("Generated paper figures in:", PAPER_FIG_DIR)


if __name__ == "__main__":
    main()
