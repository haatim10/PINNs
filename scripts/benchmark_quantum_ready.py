#!/usr/bin/env python3
"""Benchmark classical PINN vs quantum-ready PINN configurations.

The script runs matched-budget training jobs, records runtime and error
metrics, and writes a markdown report plus publication-style plots.
"""

import argparse
import copy
import csv
import json
import os
from pathlib import Path
import statistics
import sys
import time
from typing import Dict, Iterable, List, Tuple

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
import yaml

sys.path.insert(0, str(Path(__file__).parent.parent))

from scripts.train_integro_diff import train


plt.style.use("seaborn-v0_8-whitegrid")
plt.rcParams.update(
    {
        "figure.dpi": 150,
        "savefig.dpi": 300,
        "axes.titlesize": 13,
        "axes.labelsize": 11,
        "legend.fontsize": 10,
        "xtick.labelsize": 10,
        "ytick.labelsize": 10,
        "font.family": "DejaVu Sans",
    }
)


def parse_args():
    parser = argparse.ArgumentParser(description="Benchmark classical vs quantum-ready PINN workflows")
    parser.add_argument("--config", type=str, default="configs/benchmark_quantum_ready.yaml", help="Base benchmark config")
    parser.add_argument("--output-dir", type=str, default="outputs/benchmarks/quantum_ready", help="Benchmark output directory")
    parser.add_argument("--models", nargs="+", default=["classical", "quantum_ready"], help="Model types to compare")
    parser.add_argument("--seeds", nargs="+", type=int, default=[42], help="Seeds to evaluate")
    parser.add_argument("--epochs", type=int, default=None, help="Optional epoch override")
    parser.add_argument("--device", choices=["auto", "cpu", "cuda"], default="auto", help="Execution device")
    parser.add_argument("--eval-interval", type=int, default=1, help="Evaluation interval during benchmark runs")
    parser.add_argument("--early-stop-patience", type=int, default=None, help="Stop after this many stagnant evals")
    parser.add_argument("--early-stop-min-delta", type=float, default=1e-4, help="Minimum improvement required to reset patience")
    parser.add_argument("--early-stop-metric", choices=["l2", "linf", "loss"], default="l2", help="Metric used for plateau stopping")
    parser.add_argument("--dry-run", action="store_true", help="Print estimated runtime budget and exit")
    parser.add_argument("--report-name", type=str, default="benchmark_report.md", help="Markdown report filename")
    return parser.parse_args()


def get_current_rss_mb() -> float:
    """Return current process RSS in MB using /proc (Linux)."""
    with open("/proc/self/statm", "r", encoding="utf-8") as handle:
        fields = handle.readline().strip().split()
    rss_pages = int(fields[1])
    page_size = os.sysconf("SC_PAGE_SIZE")
    return (rss_pages * page_size) / (1024.0 ** 2)


def normalize_model_name(model_type: str) -> str:
    model_type = str(model_type).lower()
    if model_type in {"hybrid_quantum", "qready"}:
        return "quantum_ready"
    return model_type


def make_variant_label(model_type: str) -> str:
    model_type = normalize_model_name(model_type)
    if model_type == "classical":
        return "classical PINN"
    if model_type == "quantum_ready":
        return "quantum-ready PINN"
    return model_type.replace("_", " ")


def safe_mean(values: Iterable[float | None]) -> float | None:
    filtered = [value for value in values if value is not None]
    if not filtered:
        return None
    return float(sum(filtered) / len(filtered))


def safe_std(values: Iterable[float | None]) -> float | None:
    filtered = [value for value in values if value is not None]
    if len(filtered) < 2:
        return 0.0 if filtered else None
    return float(statistics.stdev(filtered))


def best_history_value(history: Dict, key: str) -> Tuple[float | None, int | None]:
    epochs = history.get("epochs", [])
    values = history.get(key, [])
    if not epochs or not values:
        return None, None
    best_index = min(range(len(values)), key=lambda idx: values[idx])
    return float(values[best_index]), int(epochs[best_index])


def estimate_run_seconds(config: Dict, model_type: str, epochs: int, device_mode: str) -> float:
    disc = config.get("discretization", {})
    network = config.get("network", {})

    collocation = int(disc.get("N_collocation", 0))
    boundary = int(disc.get("N_boundary", 0))
    initial = int(disc.get("N_initial", 0))
    grid_scale = max(1.0, (int(disc.get("N_x", 1)) * int(disc.get("N_t", 1))) / 2500.0)
    sample_scale = max(1.0, (collocation + 2 * boundary + initial) / 100.0)
    hidden_layers = network.get("hidden_layers", [64, 64, 64, 64])
    width_scale = max(1.0, sum(int(layer) for layer in hidden_layers) / 256.0)
    model_scale = 1.0 if normalize_model_name(model_type) == "classical" else 1.18
    device_scale = 1.0 if device_mode == "cuda" else 2.4

    per_epoch = 4.0 * grid_scale * sample_scale * width_scale * model_scale * device_scale
    return float(max(1.0, epochs * per_epoch))


def build_planned_runs(config: Dict, models: List[str], seeds: List[int], epochs: int, device_mode: str) -> List[Dict]:
    planned = []
    for seed in seeds:
        for model_type in models:
            planned.append(
                {
                    "seed": seed,
                    "model_type": normalize_model_name(model_type),
                    "variant": make_variant_label(model_type),
                    "epochs": epochs,
                    "estimated_runtime_sec": estimate_run_seconds(config, model_type, epochs, device_mode),
                }
            )
    return planned


def run_single_benchmark(
    base_config: Dict,
    model_type: str,
    seed: int,
    output_dir: Path,
    epochs_override: int | None = None,
    device_override: str = "auto",
    eval_interval: int = 1,
    early_stop_patience: int | None = None,
    early_stop_min_delta: float = 1e-4,
    early_stop_metric: str = "l2",
) -> Dict:
    config = copy.deepcopy(base_config)
    config.setdefault("network", {})["model_type"] = normalize_model_name(model_type)
    config["seed"] = int(seed)

    if epochs_override is not None:
        config.setdefault("training", {})["epochs"] = int(epochs_override)

    if device_override != "auto":
        config["device"] = device_override

    config.setdefault("logging", {})
    config["logging"]["track_points"] = False
    config["logging"]["track_l1_points"] = False
    config["logging"]["eval_interval"] = max(1, int(eval_interval))
    config["logging"]["checkpoint_interval"] = max(
        int(config["training"]["epochs"]) + 1,
        int(config["logging"].get("checkpoint_interval", 1000)),
    )
    config.setdefault("reproducibility", {})
    config["reproducibility"].setdefault("deterministic_torch", True)
    config["reproducibility"]["deterministic_sampling"] = True
    config["reproducibility"].setdefault("fixed_collocation", False)
    config["reproducibility"].setdefault("data_seed", int(seed))

    run_dir = output_dir / f"seed_{seed}" / normalize_model_name(model_type)
    checkpoint_dir = run_dir / "checkpoints"
    results_dir = run_dir / "artifacts"
    run_dir.mkdir(parents=True, exist_ok=True)

    config.setdefault("paths", {})
    config["paths"]["checkpoint_dir"] = str(checkpoint_dir)
    config["paths"]["results_dir"] = str(results_dir)

    config_path = run_dir / "config.generated.yaml"
    with open(config_path, "w", encoding="utf-8") as handle:
        yaml.safe_dump(config, handle, sort_keys=False)

    use_cuda = config.get("device", "cuda") == "cuda" and torch.cuda.is_available()
    if use_cuda:
        torch.cuda.reset_peak_memory_stats()

    cpu_before_mb = get_current_rss_mb()
    start = time.perf_counter()

    _, history = train(
        str(config_path),
        resume=False,
        early_stop_epoch=None,
        early_stop_patience=early_stop_patience,
        early_stop_min_delta=early_stop_min_delta,
        early_stop_metric=early_stop_metric,
        generate_artifacts=False,
    )

    runtime_sec = time.perf_counter() - start
    cpu_after_mb = get_current_rss_mb()
    cpu_delta_mb = max(0.0, cpu_after_mb - cpu_before_mb)

    peak_gpu_mb = None
    if use_cuda:
        peak_gpu_mb = torch.cuda.max_memory_allocated() / (1024.0 ** 2)

    final_loss = history["loss"][-1] if history.get("loss") else None
    final_l2 = history["l2"][-1] if history.get("l2") else None
    final_linf = history["linf"][-1] if history.get("linf") else None
    best_l2, best_l2_epoch = best_history_value(history, "l2")
    best_linf, best_linf_epoch = best_history_value(history, "linf")
    best_loss, best_loss_epoch = best_history_value(history, "loss")
    final_epoch = history["epochs"][-1] if history.get("epochs") else int(config["training"]["epochs"])
    parameter_count = history.get("parameter_count")
    data_seed = history.get("data_seed", config.get("reproducibility", {}).get("data_seed"))
    deterministic_sampling = history.get("deterministic_sampling", config.get("reproducibility", {}).get("deterministic_sampling"))
    fixed_collocation = history.get("fixed_collocation", config.get("reproducibility", {}).get("fixed_collocation"))

    record = {
        "seed": seed,
        "model_type": normalize_model_name(model_type),
        "variant": make_variant_label(model_type),
        "runtime_sec": runtime_sec,
        "estimated_runtime_sec": estimate_run_seconds(base_config, model_type, int(config["training"]["epochs"]), config.get("device", device_override)),
        "cpu_rss_delta_mb": cpu_delta_mb,
        "peak_gpu_memory_mb": peak_gpu_mb,
        "parameter_count": parameter_count,
        "epochs": int(config["training"]["epochs"]),
        "final_epoch": final_epoch,
        "final_loss": final_loss,
        "final_l2": final_l2,
        "final_linf": final_linf,
        "best_loss": best_loss,
        "best_loss_epoch": best_loss_epoch,
        "best_l2": best_l2,
        "best_l2_epoch": best_l2_epoch,
        "best_linf": best_linf,
        "best_linf_epoch": best_linf_epoch,
        "data_seed": data_seed,
        "deterministic_sampling": deterministic_sampling,
        "fixed_collocation": fixed_collocation,
        "history": history,
        "run_dir": str(run_dir),
        "config_path": str(config_path),
    }

    with open(run_dir / "metrics.json", "w", encoding="utf-8") as handle:
        json.dump(record, handle, indent=2)

    return record


def write_summary_csv(records: List[Dict], csv_path: Path):
    fieldnames = [
        "seed",
        "model_type",
        "variant",
        "epochs",
        "final_epoch",
        "runtime_sec",
        "estimated_runtime_sec",
        "cpu_rss_delta_mb",
        "peak_gpu_memory_mb",
        "parameter_count",
        "final_loss",
        "final_l2",
        "final_linf",
        "best_loss",
        "best_loss_epoch",
        "best_l2",
        "best_l2_epoch",
        "best_linf",
        "best_linf_epoch",
        "data_seed",
        "deterministic_sampling",
        "fixed_collocation",
        "run_dir",
    ]

    with open(csv_path, "w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for record in records:
            row = {key: record.get(key) for key in fieldnames}
            writer.writerow(row)


def summarize_records(records: List[Dict]) -> Dict[str, Dict]:
    grouped: Dict[str, List[Dict]] = {}
    for record in records:
        grouped.setdefault(record["model_type"], []).append(record)

    summary = {}
    for model_type, items in grouped.items():
        summary[model_type] = {
            "num_runs": len(items),
            "avg_runtime_sec": safe_mean(item["runtime_sec"] for item in items),
            "std_runtime_sec": safe_std(item["runtime_sec"] for item in items),
            "avg_estimated_runtime_sec": safe_mean(item["estimated_runtime_sec"] for item in items),
            "avg_cpu_rss_delta_mb": safe_mean(item["cpu_rss_delta_mb"] for item in items),
            "std_cpu_rss_delta_mb": safe_std(item["cpu_rss_delta_mb"] for item in items),
            "avg_peak_gpu_memory_mb": safe_mean(item["peak_gpu_memory_mb"] for item in items),
            "std_peak_gpu_memory_mb": safe_std(item["peak_gpu_memory_mb"] for item in items),
            "avg_parameter_count": safe_mean(item["parameter_count"] for item in items),
            "std_parameter_count": safe_std(item["parameter_count"] for item in items),
            "avg_final_loss": safe_mean(item["final_loss"] for item in items),
            "std_final_loss": safe_std(item["final_loss"] for item in items),
            "avg_final_l2": safe_mean(item["final_l2"] for item in items),
            "std_final_l2": safe_std(item["final_l2"] for item in items),
            "avg_final_linf": safe_mean(item["final_linf"] for item in items),
            "std_final_linf": safe_std(item["final_linf"] for item in items),
            "avg_best_l2": safe_mean(item["best_l2"] for item in items),
            "avg_best_linf": safe_mean(item["best_linf"] for item in items),
            "avg_final_epoch": safe_mean(item["final_epoch"] for item in items),
        }

    return summary


def write_model_aggregate(records: List[Dict], output_path: Path):
    aggregate = summarize_records(records)
    with open(output_path, "w", encoding="utf-8") as handle:
        json.dump(aggregate, handle, indent=2)


def plot_seedwise_convergence(records: List[Dict], output_dir: Path):
    by_seed: Dict[int, List[Dict]] = {}
    for record in records:
        by_seed.setdefault(int(record["seed"]), []).append(record)

    for seed, seed_records in by_seed.items():
        fig, axes = plt.subplots(1, 3, figsize=(16, 4.2), sharex=True)
        colors = {
            "classical": "#1f77b4",
            "quantum_ready": "#d62728",
        }

        for record in seed_records:
            history = record.get("history", {})
            epochs = history.get("epochs", [])
            if not epochs:
                continue

            label = record["variant"]
            color = colors.get(record["model_type"], None)
            losses = history.get("loss", [])
            l2_vals = history.get("l2", [])
            linf_vals = history.get("linf", [])

            if losses:
                axes[0].semilogy(epochs, losses, marker="o", markersize=3, linewidth=1.8, label=label, color=color)
            if l2_vals:
                axes[1].semilogy(epochs, l2_vals, marker="o", markersize=3, linewidth=1.8, label=label, color=color)
            if linf_vals:
                axes[2].semilogy(epochs, linf_vals, marker="o", markersize=3, linewidth=1.8, label=label, color=color)

        axes[0].set_title(f"Seed {seed}: Training Loss")
        axes[1].set_title("Relative L2 Error")
        axes[2].set_title("Linf Error")

        for ax in axes:
            ax.set_xlabel("Epoch")
            ax.grid(True, alpha=0.25)
            ax.legend(frameon=False)

        axes[0].set_ylabel("Value")
        plt.tight_layout()
        fig.savefig(output_dir / f"convergence_seed_{seed}.png", dpi=300, bbox_inches="tight")
        plt.close(fig)


def plot_summary_panels(records: List[Dict], output_dir: Path):
    summary = summarize_records(records)
    model_order = [model for model in ["classical", "quantum_ready"] if model in summary]
    if not model_order:
        return

    metrics = [
        ("runtime_sec", "Runtime (s)", False),
        ("final_l2", "Final L2", True),
        ("final_linf", "Final Linf", True),
        ("peak_gpu_memory_mb", "Peak GPU Memory (MB)", False),
    ]

    fig, axes = plt.subplots(2, 2, figsize=(14, 9))
    axes = axes.flatten()

    for ax, (metric_key, title, use_log) in zip(axes, metrics):
        means = [summary[model].get(f"avg_{metric_key}") for model in model_order]
        stds = [summary[model].get(f"std_{metric_key}") for model in model_order]
        positions = np.arange(len(model_order))
        colors = ["#1f77b4" if model == "classical" else "#d62728" for model in model_order]

        ax.bar(positions, means, yerr=stds, capsize=5, color=colors, alpha=0.85)
        ax.set_xticks(positions, [make_variant_label(model) for model in model_order])
        ax.set_title(title)
        ax.grid(True, axis="y", alpha=0.25)
        if use_log:
            ax.set_yscale("log")

        for idx, model in enumerate(model_order):
            raw_values = [record.get(metric_key) for record in records if record["model_type"] == model and record.get(metric_key) is not None]
            if raw_values:
                jitter = np.linspace(-0.08, 0.08, len(raw_values)) if len(raw_values) > 1 else np.array([0.0])
                ax.scatter(np.full(len(raw_values), idx) + jitter, raw_values, s=28, color="black", alpha=0.7, zorder=3)

    fig.suptitle("Quantum-Readiness Benchmark Summary", fontsize=15, fontweight="bold")
    plt.tight_layout(rect=[0, 0, 1, 0.97])
    fig.savefig(output_dir / "summary_panels.png", dpi=300, bbox_inches="tight")
    plt.close(fig)


def plot_ratio_panel(records: List[Dict], output_dir: Path):
    grouped: Dict[int, Dict[str, Dict]] = {}
    for record in records:
        grouped.setdefault(int(record["seed"]), {})[record["model_type"]] = record

    seeds = []
    runtime_ratios = []
    l2_ratios = []
    linf_ratios = []

    for seed, seed_records in grouped.items():
        classical = seed_records.get("classical")
        quantum = seed_records.get("quantum_ready")
        if not classical or not quantum:
            continue
        if classical.get("runtime_sec") and quantum.get("runtime_sec"):
            seeds.append(seed)
            runtime_ratios.append(quantum["runtime_sec"] / classical["runtime_sec"])
            l2_ratios.append(quantum["final_l2"] / classical["final_l2"])
            linf_ratios.append(quantum["final_linf"] / classical["final_linf"])

    if not seeds:
        return

    fig, ax = plt.subplots(figsize=(10, 4.8))
    x = np.arange(len(seeds))
    width = 0.25
    ax.bar(x - width, runtime_ratios, width, label="Runtime ratio", color="#6baed6")
    ax.bar(x, l2_ratios, width, label="L2 ratio", color="#fdae6b")
    ax.bar(x + width, linf_ratios, width, label="Linf ratio", color="#74c476")
    ax.axhline(1.0, color="black", linewidth=1, linestyle="--", alpha=0.7)
    ax.set_xticks(x, [str(seed) for seed in seeds])
    ax.set_ylabel("Quantum-ready / classical")
    ax.set_xlabel("Seed")
    ax.set_title("Seed-wise Performance Ratios")
    ax.legend(frameon=False, ncol=3)
    ax.grid(True, axis="y", alpha=0.25)
    plt.tight_layout()
    fig.savefig(output_dir / "ratio_panel.png", dpi=300, bbox_inches="tight")
    plt.close(fig)


def write_markdown_report(
    records: List[Dict],
    summary: Dict[str, Dict],
    output_path: Path,
    config_path: str,
    args,
):
    seeds = sorted({int(record["seed"]) for record in records})
    model_order = [model for model in ["classical", "quantum_ready"] if model in summary]
    classical = summary.get("classical", {})
    quantum = summary.get("quantum_ready", {})

    def fmt(value, precision=4):
        if value is None:
            return "n/a"
        if isinstance(value, int):
            return str(value)
        return f"{value:.{precision}f}"

    def ratio(numerator, denominator):
        if numerator is None or denominator in (None, 0):
            return None
        return numerator / denominator

    runtime_ratio = ratio(quantum.get("avg_runtime_sec"), classical.get("avg_runtime_sec"))
    l2_ratio = ratio(quantum.get("avg_final_l2"), classical.get("avg_final_l2"))
    linf_ratio = ratio(quantum.get("avg_final_linf"), classical.get("avg_final_linf"))

    wins = {
        "runtime": 0,
        "l2": 0,
        "linf": 0,
    }
    paired = 0
    for seed in seeds:
        classical_run = next((record for record in records if int(record["seed"]) == seed and record["model_type"] == "classical"), None)
        quantum_run = next((record for record in records if int(record["seed"]) == seed and record["model_type"] == "quantum_ready"), None)
        if not classical_run or not quantum_run:
            continue
        paired += 1
        if quantum_run["runtime_sec"] < classical_run["runtime_sec"]:
            wins["runtime"] += 1
        if quantum_run["final_l2"] < classical_run["final_l2"]:
            wins["l2"] += 1
        if quantum_run["final_linf"] < classical_run["final_linf"]:
            wins["linf"] += 1

    lines = [
        "# Quantum-Readiness Benchmark Report",
        "",
        f"- Config: `{config_path}`",
        f"- Seeds: {', '.join(str(seed) for seed in seeds)}",
        f"- Models: {', '.join(make_variant_label(model) for model in model_order)}",
        f"- Device request: `{args.device}`",
        f"- Epoch override: `{args.epochs if args.epochs is not None else 'config default'}`",
        f"- Early stopping: patience `{args.early_stop_patience}` on `{args.early_stop_metric}` with min delta `{args.early_stop_min_delta}`",
        "",
        "## Summary",
        "",
        "| Model | Runs | Runtime (s) | Params | Final L2 | Final Linf | Peak GPU MB |",
        "| --- | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]

    for model in model_order:
        row = summary[model]
        lines.append(
            f"| {make_variant_label(model)} | {row['num_runs']} | {fmt(row['avg_runtime_sec'])} ± {fmt(row['std_runtime_sec'])} | "
            f"{fmt(row.get('avg_parameter_count'), precision=1)} | "
            f"{fmt(row['avg_final_l2'])} ± {fmt(row['std_final_l2'])} | {fmt(row['avg_final_linf'])} ± {fmt(row['std_final_linf'])} | {fmt(row['avg_peak_gpu_memory_mb'])} |"
        )

    lines += [
        "",
        "## Ratios",
        "",
        f"- Runtime ratio (quantum-ready / classical): {fmt(runtime_ratio)}",
        f"- Final L2 ratio (quantum-ready / classical): {fmt(l2_ratio)}",
        f"- Final Linf ratio (quantum-ready / classical): {fmt(linf_ratio)}",
        "",
        "## Pairwise Wins",
        "",
        f"- Runtime wins: {wins['runtime']}/{paired}",
        f"- L2 wins: {wins['l2']}/{paired}",
        f"- Linf wins: {wins['linf']}/{paired}",
        "",
        "## Plots",
        "",
        "- [Summary panels](summary_panels.png)",
        "- [Seed-wise ratios](ratio_panel.png)",
    ]

    for seed in seeds:
        lines.append(f"- [Convergence seed {seed}](convergence_seed_{seed}.png)")

    lines += [
        "",
        "## Interpretation",
        "",
        "The quantum-ready path is still a classical emulator, so these results should be read as an architectural comparison rather than a quantum hardware claim.",
        "This benchmark is useful when the oscillatory or multiscale structure in the solution family makes the feature-mixing block competitive.",
        "",
        "## Raw Artifacts",
        "",
        "- `benchmark_results.json`",
        "- `benchmark_summary.csv`",
        "- `benchmark_aggregate.json`",
    ]

    output_path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def warmup_device(device_mode: str):
    """Initialize CUDA context outside timed sections to reduce first-run bias."""
    if device_mode in {"auto", "cuda"} and torch.cuda.is_available():
        _ = torch.tensor([0.0], device="cuda")
        torch.cuda.synchronize()


def main():
    args = parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    plots_dir = output_dir

    with open(args.config, "r", encoding="utf-8") as handle:
        base_config = yaml.safe_load(handle)

    epochs = int(args.epochs if args.epochs is not None else base_config.get("training", {}).get("epochs", 1))
    planned_runs = build_planned_runs(base_config, args.models, args.seeds, epochs, args.device)

    print("=" * 80)
    print("Quantum-readiness Benchmark")
    print("=" * 80)
    print(f"Models: {args.models}")
    print(f"Seeds: {args.seeds}")
    print(f"Base config: {args.config}")
    print(f"Output dir: {output_dir}")
    print(f"Dry run: {args.dry_run}")

    if args.dry_run:
        estimate_path = output_dir / "benchmark_estimate.json"
        total_estimate = sum(run["estimated_runtime_sec"] for run in planned_runs)
        estimate_payload = {
            "config": args.config,
            "device": args.device,
            "epochs": epochs,
            "total_estimated_runtime_sec": total_estimate,
            "planned_runs": planned_runs,
        }
        estimate_path.write_text(json.dumps(estimate_payload, indent=2) + "\n", encoding="utf-8")
        print(f"Estimated total runtime: {total_estimate / 60.0:.1f} minutes")
        print(f"Estimate written to: {estimate_path}")
        return

    warmup_device(args.device)

    all_records = []
    for seed in args.seeds:
        for model_type in args.models:
            print("-" * 80)
            print(f"Running seed={seed}, model_type={model_type}")
            record = run_single_benchmark(
                base_config=base_config,
                model_type=model_type,
                seed=seed,
                output_dir=output_dir,
                epochs_override=args.epochs,
                device_override=args.device,
                eval_interval=args.eval_interval,
                early_stop_patience=args.early_stop_patience,
                early_stop_min_delta=args.early_stop_min_delta,
                early_stop_metric=args.early_stop_metric,
            )
            all_records.append(record)
            print(
                f"Completed {record['variant']}: runtime={record['runtime_sec']:.2f}s, "
                f"final_l2={record['final_l2']}, final_linf={record['final_linf']}, "
                f"stopped_epoch={record['final_epoch']}"
            )

    summary_json = output_dir / "benchmark_results.json"
    summary_csv = output_dir / "benchmark_summary.csv"
    aggregate_json = output_dir / "benchmark_aggregate.json"
    report_md = output_dir / args.report_name

    summary = summarize_records(all_records)

    with open(summary_json, "w", encoding="utf-8") as handle:
        json.dump(all_records, handle, indent=2)

    write_summary_csv(all_records, summary_csv)
    write_model_aggregate(all_records, aggregate_json)
    plot_summary_panels(all_records, plots_dir)
    plot_ratio_panel(all_records, plots_dir)
    plot_seedwise_convergence(all_records, plots_dir)
    write_markdown_report(all_records, summary, report_md, args.config, args)

    print("=" * 80)
    print("Benchmark complete")
    print(f"Results JSON: {summary_json}")
    print(f"Summary CSV: {summary_csv}")
    print(f"Aggregate JSON: {aggregate_json}")
    print(f"Report: {report_md}")
    print("=" * 80)


if __name__ == "__main__":
    main()
