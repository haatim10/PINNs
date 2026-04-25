#!/usr/bin/env python3
"""Benchmark classical PINN vs quantum-ready PINN configurations.

This benchmark runs short, matched-budget training jobs and records:
- runtime
- peak memory (CPU/GPU)
- training loss history
- L2/Linf history (when available)
"""

import argparse
import copy
import csv
import json
import os
from pathlib import Path
import time
from typing import Dict, List

import matplotlib.pyplot as plt
import torch
import yaml

import sys
sys.path.insert(0, str(Path(__file__).parent.parent))

from scripts.train_integro_diff import train


def parse_args():
    parser = argparse.ArgumentParser(description="Benchmark classical vs quantum-ready PINN workflows")
    parser.add_argument("--config", type=str, default="configs/benchmark_quantum_ready.yaml", help="Base benchmark config")
    parser.add_argument("--output-dir", type=str, default="outputs/benchmarks/quantum_ready", help="Benchmark output directory")
    parser.add_argument("--models", nargs="+", default=["classical", "quantum_ready"], help="Model types to compare")
    parser.add_argument("--seeds", nargs="+", type=int, default=[42], help="Seeds to evaluate")
    parser.add_argument("--epochs", type=int, default=None, help="Optional epoch override")
    parser.add_argument("--device", choices=["auto", "cpu", "cuda"], default="auto", help="Execution device")
    return parser.parse_args()


def get_current_rss_mb() -> float:
    """Return current process RSS in MB using /proc (Linux)."""
    with open("/proc/self/statm", "r") as f:
        fields = f.readline().strip().split()
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
        return "integro_diff_pinn"
    if model_type == "quantum_ready":
        return "quantum_ready_pinn"
    return model_type


def run_single_benchmark(
    base_config: Dict,
    model_type: str,
    seed: int,
    output_dir: Path,
    epochs_override: int = None,
    device_override: str = "auto",
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
    config["logging"]["eval_interval"] = int(config["logging"].get("eval_interval", 1))
    config["logging"]["checkpoint_interval"] = max(
        int(config["training"]["epochs"]) + 1,
        int(config["logging"].get("checkpoint_interval", 1000)),
    )

    run_dir = output_dir / f"seed_{seed}" / normalize_model_name(model_type)
    checkpoint_dir = run_dir / "checkpoints"
    results_dir = run_dir / "artifacts"
    run_dir.mkdir(parents=True, exist_ok=True)

    config.setdefault("paths", {})
    config["paths"]["checkpoint_dir"] = str(checkpoint_dir)
    config["paths"]["results_dir"] = str(results_dir)

    config_path = run_dir / "config.generated.yaml"
    with open(config_path, "w") as f:
        yaml.safe_dump(config, f, sort_keys=False)

    use_cuda = config.get("device", "cuda") == "cuda" and torch.cuda.is_available()
    if use_cuda:
        torch.cuda.reset_peak_memory_stats()

    cpu_before_mb = get_current_rss_mb()
    start = time.perf_counter()

    _, history = train(
        str(config_path),
        resume=False,
        early_stop_epoch=None,
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

    record = {
        "seed": seed,
        "model_type": normalize_model_name(model_type),
        "variant": make_variant_label(model_type),
        "runtime_sec": runtime_sec,
        "cpu_rss_delta_mb": cpu_delta_mb,
        "peak_gpu_memory_mb": peak_gpu_mb,
        "epochs": int(config["training"]["epochs"]),
        "final_loss": final_loss,
        "final_l2": final_l2,
        "final_linf": final_linf,
        "history": history,
        "run_dir": str(run_dir),
    }

    with open(run_dir / "metrics.json", "w") as f:
        json.dump(record, f, indent=2)

    return record


def write_summary_csv(records: List[Dict], csv_path: Path):
    fieldnames = [
        "seed",
        "model_type",
        "variant",
        "epochs",
        "runtime_sec",
        "cpu_rss_delta_mb",
        "peak_gpu_memory_mb",
        "final_loss",
        "final_l2",
        "final_linf",
        "run_dir",
    ]

    with open(csv_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for record in records:
            row = {k: record.get(k) for k in fieldnames}
            writer.writerow(row)


def write_model_aggregate(records: List[Dict], output_path: Path):
    grouped = {}
    for record in records:
        grouped.setdefault(record["model_type"], []).append(record)

    aggregate = {}
    for model_type, items in grouped.items():
        def avg(values):
            values = [v for v in values if v is not None]
            return sum(values) / len(values) if values else None

        aggregate[model_type] = {
            "num_runs": len(items),
            "avg_runtime_sec": avg([item["runtime_sec"] for item in items]),
            "avg_cpu_rss_delta_mb": avg([item["cpu_rss_delta_mb"] for item in items]),
            "avg_peak_gpu_memory_mb": avg([item["peak_gpu_memory_mb"] for item in items]),
            "avg_final_loss": avg([item["final_loss"] for item in items]),
            "avg_final_l2": avg([item["final_l2"] for item in items]),
            "avg_final_linf": avg([item["final_linf"] for item in items]),
        }

    with open(output_path, "w") as f:
        json.dump(aggregate, f, indent=2)


def plot_convergence(records: List[Dict], output_dir: Path):
    by_seed = {}
    for record in records:
        by_seed.setdefault(record["seed"], []).append(record)

    for seed, seed_records in by_seed.items():
        fig, axes = plt.subplots(1, 3, figsize=(16, 4))

        for record in seed_records:
            history = record.get("history", {})
            epochs = history.get("epochs", [])
            if not epochs:
                continue

            label = record["variant"]
            losses = history.get("loss", [])
            l2_vals = history.get("l2", [])
            linf_vals = history.get("linf", [])

            if losses:
                axes[0].plot(epochs, losses, marker="o", label=label)
            if l2_vals:
                axes[1].plot(epochs, l2_vals, marker="o", label=label)
            if linf_vals:
                axes[2].plot(epochs, linf_vals, marker="o", label=label)

        axes[0].set_title("Training Loss")
        axes[1].set_title("L2 Relative Error")
        axes[2].set_title("Linf Error")

        for ax in axes:
            ax.set_xlabel("Epoch")
            ax.grid(True, alpha=0.3)
            ax.legend()

        plt.tight_layout()
        plt.savefig(output_dir / f"convergence_seed_{seed}.png", dpi=150)
        plt.close()


def warmup_device(device_mode: str):
    """Initialize CUDA context outside timed sections to reduce first-run bias."""
    if device_mode in {"auto", "cuda"} and torch.cuda.is_available():
        _ = torch.tensor([0.0], device="cuda")
        torch.cuda.synchronize()


def main():
    args = parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    with open(args.config, "r") as f:
        base_config = yaml.safe_load(f)

    all_records = []

    warmup_device(args.device)

    print("=" * 80)
    print("Quantum-readiness Benchmark")
    print("=" * 80)
    print(f"Models: {args.models}")
    print(f"Seeds: {args.seeds}")
    print(f"Base config: {args.config}")
    print(f"Output dir: {output_dir}")

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
            )
            all_records.append(record)
            print(
                f"Completed {record['variant']}: runtime={record['runtime_sec']:.2f}s, "
                f"loss={record['final_loss']}, l2={record['final_l2']}, linf={record['final_linf']}"
            )

    summary_json = output_dir / "benchmark_results.json"
    summary_csv = output_dir / "benchmark_summary.csv"
    aggregate_json = output_dir / "benchmark_aggregate.json"

    with open(summary_json, "w") as f:
        json.dump(all_records, f, indent=2)

    write_summary_csv(all_records, summary_csv)
    write_model_aggregate(all_records, aggregate_json)
    plot_convergence(all_records, output_dir)

    print("=" * 80)
    print("Benchmark complete")
    print(f"Results JSON: {summary_json}")
    print(f"Summary CSV: {summary_csv}")
    print(f"Aggregate JSON: {aggregate_json}")
    print("=" * 80)


if __name__ == "__main__":
    main()
