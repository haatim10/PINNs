#!/usr/bin/env python3
"""Config-driven TE-QPINN benchmark runner.

This runner compares explicit config files (no CLI overrides passed to training):
- classical + product-integration
- te_qpinn_surrogate + product-integration

Outputs:
- summary.csv
- summary.json
- benchmark_report.md
- plots under configured plots directory
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
from src.model_factory import model_name_from_config
from src.utils import resolve_device


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
    parser = argparse.ArgumentParser(description="Run TE-QPINN benchmarks from explicit config files")
    parser.add_argument(
        "--benchmark-config",
        type=str,
        default="configs/benchmark_te_qpinn_smoke.yaml",
        help="Benchmark plan YAML with run config file paths",
    )
    parser.add_argument("--dry-run", action="store_true", help="Print planned runs and exit")
    return parser.parse_args()


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
    if model_type in {"te_qpinn", "teqpinn", "te-qpinn"}:
        return "te_qpinn_surrogate"
    return model_type


def load_yaml(path: Path) -> Dict:
    with open(path, "r", encoding="utf-8") as handle:
        return yaml.safe_load(handle)


def summarize_records(records: List[Dict]) -> Dict[str, Dict]:
    grouped: Dict[str, List[Dict]] = {}
    for record in records:
        grouped.setdefault(record["run_name"], []).append(record)

    summary = {}
    for run_name, items in grouped.items():
        summary[run_name] = {
            "label": items[0]["label"],
            "model_type": items[0]["model_type"],
            "num_runs": len(items),
            "avg_runtime_sec": safe_mean(item["runtime_sec"] for item in items),
            "std_runtime_sec": safe_std(item["runtime_sec"] for item in items),
            "avg_cpu_rss_delta_mb": safe_mean(item["cpu_rss_delta_mb"] for item in items),
            "avg_peak_gpu_memory_mb": safe_mean(item["peak_gpu_memory_mb"] for item in items),
            "avg_parameter_count": safe_mean(item["parameter_count"] for item in items),
            "avg_final_l2": safe_mean(item["final_l2"] for item in items),
            "std_final_l2": safe_std(item["final_l2"] for item in items),
            "avg_final_linf": safe_mean(item["final_linf"] for item in items),
            "std_final_linf": safe_std(item["final_linf"] for item in items),
            "avg_best_l2": safe_mean(item["best_l2"] for item in items),
            "avg_best_linf": safe_mean(item["best_linf"] for item in items),
            "avg_final_epoch": safe_mean(item["final_epoch"] for item in items),
        }
    return summary


def write_summary_csv(records: List[Dict], output_path: Path):
    fieldnames = [
        "run_name",
        "label",
        "seed",
        "model_type",
        "epochs",
        "final_epoch",
        "runtime_sec",
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
        "config_source",
        "config_path",
        "run_dir",
    ]
    with open(output_path, "w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for record in records:
            writer.writerow({name: record.get(name) for name in fieldnames})


def plot_seedwise_convergence(records: List[Dict], plots_dir: Path):
    by_seed: Dict[int, List[Dict]] = {}
    for record in records:
        by_seed.setdefault(int(record["seed"]), []).append(record)

    for seed, seed_records in by_seed.items():
        fig, axes = plt.subplots(1, 3, figsize=(16, 4.2), sharex=True)
        for record in seed_records:
            history = record.get("history", {})
            epochs = history.get("epochs", [])
            if not epochs:
                continue
            label = record["label"]
            losses = history.get("loss", [])
            l2_vals = history.get("l2", [])
            linf_vals = history.get("linf", [])

            if losses:
                axes[0].semilogy(epochs, losses, marker="o", markersize=3, linewidth=1.8, label=label)
            if l2_vals:
                axes[1].semilogy(epochs, l2_vals, marker="o", markersize=3, linewidth=1.8, label=label)
            if linf_vals:
                axes[2].semilogy(epochs, linf_vals, marker="o", markersize=3, linewidth=1.8, label=label)

        axes[0].set_title(f"Seed {seed}: Training Loss")
        axes[1].set_title("Relative L2 Error")
        axes[2].set_title("Linf Error")
        for ax in axes:
            ax.set_xlabel("Epoch")
            ax.grid(True, alpha=0.25)
            ax.legend(frameon=False)
        axes[0].set_ylabel("Value")
        plt.tight_layout()
        fig.savefig(plots_dir / f"convergence_seed_{seed}.png", dpi=300, bbox_inches="tight")
        plt.close(fig)


def plot_summary_panels(summary: Dict[str, Dict], plots_dir: Path):
    run_names = list(summary.keys())
    if not run_names:
        return

    labels = [summary[name]["label"] for name in run_names]
    metrics = [
        ("avg_runtime_sec", "Runtime (s)", False),
        ("avg_final_l2", "Final L2", True),
        ("avg_final_linf", "Final Linf", True),
        ("avg_parameter_count", "Parameter Count", False),
    ]

    fig, axes = plt.subplots(2, 2, figsize=(14, 9))
    axes = axes.flatten()
    for ax, (metric_key, title, use_log) in zip(axes, metrics):
        means = [summary[name].get(metric_key) for name in run_names]
        std_key = metric_key.replace("avg_", "std_")
        stds = [summary[name].get(std_key, 0.0) for name in run_names]
        x_pos = np.arange(len(run_names))
        ax.bar(x_pos, means, yerr=stds, capsize=5, alpha=0.85)
        ax.set_xticks(x_pos, labels)
        ax.set_title(title)
        ax.grid(True, axis="y", alpha=0.25)
        if use_log:
            ax.set_yscale("log")
    fig.suptitle("TE-QPINN Benchmark Summary", fontsize=15, fontweight="bold")
    plt.tight_layout(rect=[0, 0, 1, 0.97])
    fig.savefig(plots_dir / "summary_panels.png", dpi=300, bbox_inches="tight")
    plt.close(fig)


def write_markdown_report(
    records: List[Dict],
    summary: Dict[str, Dict],
    output_path: Path,
    benchmark_config_path: str,
):
    seeds = sorted({int(record["seed"]) for record in records})
    run_names = list(summary.keys())

    def fmt(value, precision=4):
        if value is None:
            return "n/a"
        if isinstance(value, int):
            return str(value)
        return f"{value:.{precision}f}"

    lines = [
        "# TE-QPINN Benchmark Report",
        "",
        f"- Benchmark config: `{benchmark_config_path}`",
        f"- Seeds: {', '.join(str(seed) for seed in seeds)}",
        "",
        "## Summary",
        "",
        "| Variant | Model Type | Runs | Runtime (s) | Params | Final L2 | Final Linf |",
        "| --- | --- | ---: | ---: | ---: | ---: | ---: |",
    ]
    for run_name in run_names:
        row = summary[run_name]
        lines.append(
            f"| {row['label']} | {row['model_type']} | {row['num_runs']} | "
            f"{fmt(row['avg_runtime_sec'])} ± {fmt(row.get('std_runtime_sec', 0.0))} | "
            f"{fmt(row['avg_parameter_count'], precision=1)} | "
            f"{fmt(row['avg_final_l2'])} ± {fmt(row.get('std_final_l2', 0.0))} | "
            f"{fmt(row['avg_final_linf'])} ± {fmt(row.get('std_final_linf', 0.0))} |"
        )

    lines += [
        "",
        "## Artifacts",
        "",
        "- `summary.csv`",
        "- `summary.json`",
        "- `summary_panels.png`",
        "- `convergence_seed_<seed>.png`",
    ]
    output_path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def run_benchmark_case(
    run_name: str,
    run_label: str,
    config_source: Path,
    seed: int,
    output_dir: Path,
    device_mode: str,
) -> Dict:
    base_config = load_yaml(config_source)
    config = copy.deepcopy(base_config)
    config["seed"] = int(seed)

    resolved_device = resolve_device(device_mode if device_mode != "auto" else config.get("device", "auto"))
    config["device"] = resolved_device

    config.setdefault("reproducibility", {})
    config["reproducibility"].setdefault("deterministic_torch", True)
    config["reproducibility"]["deterministic_sampling"] = True
    config["reproducibility"].setdefault("fixed_collocation", False)
    config["reproducibility"]["data_seed"] = int(seed)

    config.setdefault("logging", {})
    config["logging"]["track_points"] = False
    config["logging"]["track_l1_points"] = False

    run_dir = output_dir / f"seed_{seed}" / run_name
    checkpoint_dir = run_dir / "checkpoints"
    artifacts_dir = run_dir / "artifacts"
    run_dir.mkdir(parents=True, exist_ok=True)

    config.setdefault("paths", {})
    config["paths"]["checkpoint_dir"] = str(checkpoint_dir)
    config["paths"]["results_dir"] = str(artifacts_dir)

    generated_config_path = run_dir / "config.generated.yaml"
    with open(generated_config_path, "w", encoding="utf-8") as handle:
        yaml.safe_dump(config, handle, sort_keys=False)

    use_cuda = resolved_device == "cuda" and torch.cuda.is_available()
    if use_cuda:
        torch.cuda.reset_peak_memory_stats()

    cpu_before_mb = get_current_rss_mb()
    start = time.perf_counter()

    _, history = train(
        str(generated_config_path),
        resume=False,
        early_stop_epoch=None,
        early_stop_patience=None,
        early_stop_min_delta=1e-4,
        early_stop_metric="l2",
        generate_artifacts=False,
    )

    runtime_sec = time.perf_counter() - start
    cpu_after_mb = get_current_rss_mb()
    cpu_delta_mb = max(0.0, cpu_after_mb - cpu_before_mb)
    peak_gpu_mb = torch.cuda.max_memory_allocated() / (1024.0 ** 2) if use_cuda else None

    final_loss = history["loss"][-1] if history.get("loss") else None
    final_l2 = history["l2"][-1] if history.get("l2") else None
    final_linf = history["linf"][-1] if history.get("linf") else None
    best_l2, best_l2_epoch = best_history_value(history, "l2")
    best_linf, best_linf_epoch = best_history_value(history, "linf")
    best_loss, best_loss_epoch = best_history_value(history, "loss")
    final_epoch = history["epochs"][-1] if history.get("epochs") else int(config["training"]["epochs"])

    model_type = model_name_from_config(config.get("network", {}))
    record = {
        "run_name": run_name,
        "label": run_label,
        "seed": seed,
        "model_type": normalize_model_name(model_type),
        "epochs": int(config["training"]["epochs"]),
        "final_epoch": final_epoch,
        "runtime_sec": runtime_sec,
        "cpu_rss_delta_mb": cpu_delta_mb,
        "peak_gpu_memory_mb": peak_gpu_mb,
        "parameter_count": history.get("parameter_count"),
        "final_loss": final_loss,
        "final_l2": final_l2,
        "final_linf": final_linf,
        "best_loss": best_loss,
        "best_loss_epoch": best_loss_epoch,
        "best_l2": best_l2,
        "best_l2_epoch": best_l2_epoch,
        "best_linf": best_linf,
        "best_linf_epoch": best_linf_epoch,
        "config_source": str(config_source),
        "config_path": str(generated_config_path),
        "history": history,
        "run_dir": str(run_dir),
    }

    with open(run_dir / "metrics.json", "w", encoding="utf-8") as handle:
        json.dump(record, handle, indent=2)
    return record


def main():
    args = parse_args()
    benchmark_config_path = Path(args.benchmark_config)
    benchmark_cfg = load_yaml(benchmark_config_path)

    seeds = [int(seed) for seed in benchmark_cfg.get("seeds", [42])]
    runs = benchmark_cfg.get("runs", [])
    if not runs:
        raise ValueError("Benchmark config must include at least one run entry under 'runs'.")

    output_dir = Path(benchmark_cfg.get("output_dir", "outputs/benchmarks/te_qpinn"))
    plots_dir = Path(benchmark_cfg.get("plots_dir", "outputs/plots/te_qpinn"))
    report_name = str(benchmark_cfg.get("report_name", "benchmark_report.md"))
    device_mode = str(benchmark_cfg.get("device", "auto"))

    output_dir.mkdir(parents=True, exist_ok=True)
    plots_dir.mkdir(parents=True, exist_ok=True)

    planned = []
    for seed in seeds:
        for run in runs:
            planned.append(
                {
                    "seed": seed,
                    "run_name": run["name"],
                    "label": run.get("label", run["name"]),
                    "config": run["config"],
                }
            )

    print("=" * 80)
    print("TE-QPINN Config-Driven Benchmark")
    print("=" * 80)
    print(f"Benchmark config: {benchmark_config_path}")
    print(f"Seeds: {seeds}")
    print(f"Output dir: {output_dir}")
    print(f"Plots dir: {plots_dir}")
    if args.dry_run:
        print("\nPlanned runs:")
        for item in planned:
            print(f"  - seed={item['seed']} run={item['run_name']} config={item['config']}")
        return

    records: List[Dict] = []
    root_dir = Path(__file__).parent.parent
    for item in planned:
        run_name = item["run_name"]
        run_label = item["label"]
        source_cfg_path = Path(item["config"])
        if not source_cfg_path.is_absolute():
            source_cfg_path = (root_dir / source_cfg_path).resolve()

        print("-" * 80)
        print(f"Running seed={item['seed']} run={run_name}")
        record = run_benchmark_case(
            run_name=run_name,
            run_label=run_label,
            config_source=source_cfg_path,
            seed=int(item["seed"]),
            output_dir=output_dir,
            device_mode=device_mode,
        )
        records.append(record)
        print(
            f"Completed {run_label}: runtime={record['runtime_sec']:.2f}s, "
            f"final_l2={record['final_l2']}, final_linf={record['final_linf']}, "
            f"epoch={record['final_epoch']}"
        )

    summary = summarize_records(records)
    summary_csv_path = output_dir / "summary.csv"
    summary_json_path = output_dir / "summary.json"
    report_path = output_dir / report_name

    write_summary_csv(records, summary_csv_path)
    summary_payload = {
        "benchmark_config": str(benchmark_config_path),
        "seeds": seeds,
        "records": records,
        "aggregate": summary,
    }
    with open(summary_json_path, "w", encoding="utf-8") as handle:
        json.dump(summary_payload, handle, indent=2)

    plot_seedwise_convergence(records, plots_dir)
    plot_summary_panels(summary, plots_dir)
    write_markdown_report(records, summary, report_path, str(benchmark_config_path))

    print("=" * 80)
    print("Benchmark complete")
    print(f"summary.csv: {summary_csv_path}")
    print(f"summary.json: {summary_json_path}")
    print(f"plots: {plots_dir}")
    print(f"report: {report_path}")
    print("=" * 80)


if __name__ == "__main__":
    main()
