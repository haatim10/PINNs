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
from typing import Dict, Iterable, List, Optional, Tuple

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
import yaml

sys.path.insert(0, str(Path(__file__).parent.parent))

from scripts.train_integro_diff import train
from src.model_factory import model_name_from_config
from src.mesh import GradedMesh, L1Coefficients
from src.physics_integro import (
    IntegroDifferentialResidual,
    exact_solution as integro_exact_solution,
)
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
    parser.add_argument(
        "--resume-incomplete",
        action="store_true",
        help="Resume benchmark by reusing existing run metrics and executing only missing runs",
    )
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


def _delta_status(current: float | None, reference: float | None, lower_is_better: bool = True) -> Tuple[float | None, str]:
    """Return (delta, status) where status in {improves, worsens, neutral, n/a}."""
    if current is None or reference is None:
        return None, "n/a"
    delta = float(current - reference)
    eps = 1e-12
    if abs(delta) <= eps:
        return delta, "neutral"
    if lower_is_better:
        return delta, "improves" if delta < 0 else "worsens"
    return delta, "improves" if delta > 0 else "worsens"


def _find_reference_run(summary: Dict[str, Dict], target: str) -> str | None:
    if target == "classical":
        if "classical_pi" in summary:
            return "classical_pi"
        for run_name, row in summary.items():
            if str(row.get("model_type")) == "classical":
                return run_name
        return None
    if target == "full_te":
        if "full_te_pi" in summary:
            return "full_te_pi"
        if "te_qpinn_surrogate_pi" in summary:
            return "te_qpinn_surrogate_pi"
        for run_name, row in summary.items():
            if str(row.get("model_type")) == "te_qpinn_surrogate":
                return run_name
        return None
    return None


def build_ablation_stats(summary: Dict[str, Dict], benchmark_config_path: str) -> Dict | None:
    """Build compact ablation deltas/status vs full TE and classical references."""
    if "ablation" not in str(benchmark_config_path).lower():
        return None
    if len(summary) < 3:
        return None

    classical_ref = _find_reference_run(summary, "classical")
    full_te_ref = _find_reference_run(summary, "full_te")
    if classical_ref is None or full_te_ref is None:
        return None

    metrics = [
        ("avg_final_l2", True),
        ("avg_final_linf", True),
        ("avg_final_loss", True),
        ("avg_runtime_sec", True),
    ]

    rows = []
    for run_name, row in summary.items():
        row_payload = {
            "run_name": run_name,
            "label": row.get("label"),
            "model_type": row.get("model_type"),
            "mean_final_l2": row.get("avg_final_l2"),
            "std_final_l2": row.get("std_final_l2"),
            "mean_final_linf": row.get("avg_final_linf"),
            "std_final_linf": row.get("std_final_linf"),
            "mean_final_loss": row.get("avg_final_loss"),
            "std_final_loss": row.get("std_final_loss"),
            "mean_runtime_sec": row.get("avg_runtime_sec"),
            "std_runtime_sec": row.get("std_runtime_sec"),
            "mean_parameter_count": row.get("avg_parameter_count"),
            "vs_full_te": {},
            "vs_classical": {},
        }

        for metric_key, lower_is_better in metrics:
            d_te, s_te = _delta_status(
                row.get(metric_key),
                summary[full_te_ref].get(metric_key),
                lower_is_better=lower_is_better,
            )
            d_cls, s_cls = _delta_status(
                row.get(metric_key),
                summary[classical_ref].get(metric_key),
                lower_is_better=lower_is_better,
            )
            row_payload["vs_full_te"][metric_key] = {"delta": d_te, "status": s_te}
            row_payload["vs_classical"][metric_key] = {"delta": d_cls, "status": s_cls}

        d_param_te, s_param_te = _delta_status(
            row.get("avg_parameter_count"),
            summary[full_te_ref].get("avg_parameter_count"),
            lower_is_better=True,
        )
        d_param_cls, s_param_cls = _delta_status(
            row.get("avg_parameter_count"),
            summary[classical_ref].get("avg_parameter_count"),
            lower_is_better=True,
        )
        row_payload["vs_full_te"]["avg_parameter_count"] = {"delta": d_param_te, "status": s_param_te}
        row_payload["vs_classical"]["avg_parameter_count"] = {"delta": d_param_cls, "status": s_param_cls}

        rows.append(row_payload)

    return {
        "benchmark_config": benchmark_config_path,
        "reference_runs": {
            "classical": classical_ref,
            "full_te": full_te_ref,
        },
        "rows": rows,
    }


def write_ablation_stats(ablation_stats: Dict, output_dir: Path):
    """Write ablation stats to JSON and compact markdown."""
    json_path = output_dir / "ablation_stats.json"
    md_path = output_dir / "ablation_stats.md"
    with open(json_path, "w", encoding="utf-8") as handle:
        json.dump(ablation_stats, handle, indent=2)

    def fmt(value: float | None, precision: int = 6) -> str:
        if value is None:
            return "n/a"
        return f"{float(value):.{precision}f}"

    lines = [
        "# TE-QPINN Ablation Stats",
        "",
        f"- Benchmark config: `{ablation_stats.get('benchmark_config')}`",
        f"- Reference full TE run: `{ablation_stats.get('reference_runs', {}).get('full_te')}`",
        f"- Reference classical run: `{ablation_stats.get('reference_runs', {}).get('classical')}`",
        "",
        "## Metrics",
        "",
        "| Variant | Final L2 | Final Linf | Final Loss | Runtime (s) | Params |",
        "| --- | ---: | ---: | ---: | ---: | ---: |",
    ]
    for row in ablation_stats.get("rows", []):
        lines.append(
            f"| {row.get('label')} | "
            f"{fmt(row.get('mean_final_l2'))} | "
            f"{fmt(row.get('mean_final_linf'))} | "
            f"{fmt(row.get('mean_final_loss'))} | "
            f"{fmt(row.get('mean_runtime_sec'), precision=4)} | "
            f"{fmt(row.get('mean_parameter_count'), precision=1)} |"
        )

    lines += [
        "",
        "## Comparison Status",
        "",
        "| Variant | L2 vs Full TE | Linf vs Full TE | Loss vs Full TE | Runtime vs Full TE | L2 vs Classical | Linf vs Classical | Loss vs Classical | Runtime vs Classical |",
        "| --- | --- | --- | --- | --- | --- | --- | --- | --- |",
    ]
    for row in ablation_stats.get("rows", []):
        full = row.get("vs_full_te", {})
        cls = row.get("vs_classical", {})
        lines.append(
            f"| {row.get('label')} | "
            f"{full.get('avg_final_l2', {}).get('status', 'n/a')} | "
            f"{full.get('avg_final_linf', {}).get('status', 'n/a')} | "
            f"{full.get('avg_final_loss', {}).get('status', 'n/a')} | "
            f"{full.get('avg_runtime_sec', {}).get('status', 'n/a')} | "
            f"{cls.get('avg_final_l2', {}).get('status', 'n/a')} | "
            f"{cls.get('avg_final_linf', {}).get('status', 'n/a')} | "
            f"{cls.get('avg_final_loss', {}).get('status', 'n/a')} | "
            f"{cls.get('avg_runtime_sec', {}).get('status', 'n/a')} |"
        )

    md_path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def _paired_seed_records(records: List[Dict], run_name: str, reference_run: str) -> List[Tuple[Dict, Dict]]:
    """Collect paired records for (run_name, reference_run) over common seeds."""
    by_seed: Dict[int, Dict[str, Dict]] = {}
    for record in records:
        seed = int(record["seed"])
        by_seed.setdefault(seed, {})
        by_seed[seed][str(record["run_name"])] = record

    paired: List[Tuple[Dict, Dict]] = []
    for seed_map in by_seed.values():
        if run_name in seed_map and reference_run in seed_map:
            paired.append((seed_map[run_name], seed_map[reference_run]))
    return paired


def _win_count_for_metric(
    records: List[Dict],
    run_name: str,
    reference_run: str,
    metric_key: str,
) -> Dict[str, int]:
    """Count wins/ties/losses for lower-is-better metrics on paired seeds."""
    paired = _paired_seed_records(records, run_name, reference_run)
    wins = 0
    ties = 0
    losses = 0
    for run_record, ref_record in paired:
        run_value = run_record.get(metric_key)
        ref_value = ref_record.get(metric_key)
        if run_value is None or ref_value is None:
            continue
        run_float = float(run_value)
        ref_float = float(ref_value)
        if abs(run_float - ref_float) <= 1e-12:
            ties += 1
        elif run_float < ref_float:
            wins += 1
        else:
            losses += 1
    return {
        "wins": wins,
        "ties": ties,
        "losses": losses,
        "paired_seeds": len(paired),
    }


def build_multiseed_stats(records: List[Dict], summary: Dict[str, Dict], benchmark_config_path: str) -> Dict | None:
    """Build compact multi-seed statistics including paired win counts."""
    unique_seeds = sorted({int(record["seed"]) for record in records})
    if len(unique_seeds) < 2:
        return None

    classical_ref = _find_reference_run(summary, "classical")
    full_te_ref = _find_reference_run(summary, "full_te")

    run_rows = []
    for run_name, row in summary.items():
        row_payload = {
            "run_name": run_name,
            "label": row.get("label"),
            "model_type": row.get("model_type"),
            "mean_final_l2": row.get("avg_final_l2"),
            "std_final_l2": row.get("std_final_l2"),
            "mean_final_linf": row.get("avg_final_linf"),
            "std_final_linf": row.get("std_final_linf"),
            "mean_final_loss": row.get("avg_final_loss"),
            "std_final_loss": row.get("std_final_loss"),
            "mean_runtime_sec": row.get("avg_runtime_sec"),
            "std_runtime_sec": row.get("std_runtime_sec"),
            "mean_parameter_count": row.get("avg_parameter_count"),
            "wins": {},
        }

        if classical_ref is not None and run_name != classical_ref:
            row_payload["wins"]["vs_classical"] = {
                "final_l2": _win_count_for_metric(records, run_name, classical_ref, "final_l2"),
                "final_linf": _win_count_for_metric(records, run_name, classical_ref, "final_linf"),
            }
        if full_te_ref is not None and run_name != full_te_ref:
            row_payload["wins"]["vs_full_te"] = {
                "final_l2": _win_count_for_metric(records, run_name, full_te_ref, "final_l2"),
                "final_linf": _win_count_for_metric(records, run_name, full_te_ref, "final_linf"),
            }

        run_rows.append(row_payload)

    return {
        "benchmark_config": benchmark_config_path,
        "seeds": unique_seeds,
        "reference_runs": {
            "classical": classical_ref,
            "full_te": full_te_ref,
        },
        "runs": run_rows,
    }


def write_multiseed_stats(multiseed_stats: Dict, output_dir: Path):
    """Write multi-seed stats to JSON and compact markdown."""
    json_path = output_dir / "multiseed_stats.json"
    md_path = output_dir / "multiseed_stats.md"
    with open(json_path, "w", encoding="utf-8") as handle:
        json.dump(multiseed_stats, handle, indent=2)

    def fmt(value: float | None, precision: int = 6) -> str:
        if value is None:
            return "n/a"
        return f"{float(value):.{precision}f}"

    lines = [
        "# TE-QPINN Multi-Seed Stats",
        "",
        f"- Benchmark config: `{multiseed_stats.get('benchmark_config')}`",
        f"- Seeds: {multiseed_stats.get('seeds', [])}",
        f"- Reference classical run: `{multiseed_stats.get('reference_runs', {}).get('classical')}`",
        f"- Reference full TE run: `{multiseed_stats.get('reference_runs', {}).get('full_te')}`",
        "",
        "## Aggregate Metrics",
        "",
        "| Variant | Final L2 (mean±std) | Final Linf (mean±std) | Final Loss (mean±std) | Runtime (s, mean±std) | Params |",
        "| --- | ---: | ---: | ---: | ---: | ---: |",
    ]

    for row in multiseed_stats.get("runs", []):
        lines.append(
            f"| {row.get('label')} | "
            f"{fmt(row.get('mean_final_l2'))} ± {fmt(row.get('std_final_l2'))} | "
            f"{fmt(row.get('mean_final_linf'))} ± {fmt(row.get('std_final_linf'))} | "
            f"{fmt(row.get('mean_final_loss'))} ± {fmt(row.get('std_final_loss'))} | "
            f"{fmt(row.get('mean_runtime_sec'), precision=4)} ± {fmt(row.get('std_runtime_sec'), precision=4)} | "
            f"{fmt(row.get('mean_parameter_count'), precision=1)} |"
        )

    lines += [
        "",
        "## Win Counts",
        "",
        "| Variant | L2 wins vs Classical | Linf wins vs Classical | L2 wins vs Full TE | Linf wins vs Full TE |",
        "| --- | ---: | ---: | ---: | ---: |",
    ]

    for row in multiseed_stats.get("runs", []):
        wins = row.get("wins", {})
        cls_l2 = wins.get("vs_classical", {}).get("final_l2", {})
        cls_linf = wins.get("vs_classical", {}).get("final_linf", {})
        te_l2 = wins.get("vs_full_te", {}).get("final_l2", {})
        te_linf = wins.get("vs_full_te", {}).get("final_linf", {})

        def fmt_wins(payload: Dict) -> str:
            if not payload:
                return "n/a"
            return f"{payload.get('wins', 0)} / {payload.get('paired_seeds', 0)}"

        lines.append(
            f"| {row.get('label')} | "
            f"{fmt_wins(cls_l2)} | "
            f"{fmt_wins(cls_linf)} | "
            f"{fmt_wins(te_l2)} | "
            f"{fmt_wins(te_linf)} |"
        )

    md_path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def _optimizer_metric_delta(
    adam_value: float | None,
    adam_lbfgs_value: float | None,
    lower_is_better: bool = True,
) -> Dict[str, float | str | None]:
    """Return delta payload for Adam+LBFGS minus Adam."""
    if adam_value is None or adam_lbfgs_value is None:
        return {
            "delta": None,
            "relative_percent": None,
            "status": "n/a",
        }

    adam_float = float(adam_value)
    lbfgs_float = float(adam_lbfgs_value)
    delta = lbfgs_float - adam_float
    if abs(delta) <= 1e-12:
        status = "neutral"
    elif lower_is_better:
        status = "improves" if delta < 0 else "worsens"
    else:
        status = "improves" if delta > 0 else "worsens"

    relative = None
    if abs(adam_float) > 1e-12:
        relative = (delta / adam_float) * 100.0
    return {
        "delta": float(delta),
        "relative_percent": None if relative is None else float(relative),
        "status": status,
    }


def build_optimizer_sensitivity_stats(summary: Dict[str, Dict], benchmark_config_path: str) -> Dict | None:
    """Build Adam vs Adam+LBFGS comparison payload for optimizer sensitivity studies."""
    pair_specs = [
        {
            "variant_id": "classical_pi",
            "label": "Classical + PI",
            "adam_run": "classical_adam_pi",
            "adam_lbfgs_run": "classical_adam_lbfgs_pi",
        },
        {
            "variant_id": "te_fixed_pi",
            "label": "TE fixed residual 0.10 + PI",
            "adam_run": "te_fixed_adam_pi",
            "adam_lbfgs_run": "te_fixed_adam_lbfgs_pi",
        },
        {
            "variant_id": "te_layernorm_post_quantum_pi",
            "label": "TE LayerNorm post_quantum + PI",
            "adam_run": "te_layernorm_adam_pi",
            "adam_lbfgs_run": "te_layernorm_adam_lbfgs_pi",
        },
    ]

    rows = []
    for spec in pair_specs:
        adam_run = spec["adam_run"]
        adam_lbfgs_run = spec["adam_lbfgs_run"]
        if adam_run not in summary or adam_lbfgs_run not in summary:
            continue

        adam_row = summary[adam_run]
        lbfgs_row = summary[adam_lbfgs_run]
        rows.append(
            {
                "variant_id": spec["variant_id"],
                "label": spec["label"],
                "adam_run": adam_run,
                "adam_lbfgs_run": adam_lbfgs_run,
                "adam": {
                    "final_l2": adam_row.get("avg_final_l2"),
                    "final_linf": adam_row.get("avg_final_linf"),
                    "final_loss": adam_row.get("avg_final_loss"),
                    "runtime_sec": adam_row.get("avg_runtime_sec"),
                    "parameter_count": adam_row.get("avg_parameter_count"),
                },
                "adam_lbfgs": {
                    "final_l2": lbfgs_row.get("avg_final_l2"),
                    "final_linf": lbfgs_row.get("avg_final_linf"),
                    "final_loss": lbfgs_row.get("avg_final_loss"),
                    "runtime_sec": lbfgs_row.get("avg_runtime_sec"),
                    "parameter_count": lbfgs_row.get("avg_parameter_count"),
                },
                "delta_adam_lbfgs_minus_adam": {
                    "final_l2": _optimizer_metric_delta(
                        adam_row.get("avg_final_l2"),
                        lbfgs_row.get("avg_final_l2"),
                        lower_is_better=True,
                    ),
                    "final_linf": _optimizer_metric_delta(
                        adam_row.get("avg_final_linf"),
                        lbfgs_row.get("avg_final_linf"),
                        lower_is_better=True,
                    ),
                    "final_loss": _optimizer_metric_delta(
                        adam_row.get("avg_final_loss"),
                        lbfgs_row.get("avg_final_loss"),
                        lower_is_better=True,
                    ),
                    "runtime_sec": _optimizer_metric_delta(
                        adam_row.get("avg_runtime_sec"),
                        lbfgs_row.get("avg_runtime_sec"),
                        lower_is_better=True,
                    ),
                    "parameter_count": _optimizer_metric_delta(
                        adam_row.get("avg_parameter_count"),
                        lbfgs_row.get("avg_parameter_count"),
                        lower_is_better=True,
                    ),
                },
            }
        )

    if not rows:
        return None

    return {
        "benchmark_config": benchmark_config_path,
        "budget_note": (
            "Adam+LBFGS includes an extended optimization budget "
            "(Adam stage plus LBFGS fine-tuning) and is not an equal-budget "
            "comparison with Adam-only runs."
        ),
        "rows": rows,
    }


def write_optimizer_sensitivity_stats(optimizer_stats: Dict, output_dir: Path):
    """Write optimizer sensitivity stats to JSON and markdown."""
    json_path = output_dir / "optimizer_sensitivity_stats.json"
    md_path = output_dir / "optimizer_sensitivity_stats.md"

    with open(json_path, "w", encoding="utf-8") as handle:
        json.dump(optimizer_stats, handle, indent=2)

    def fmt(value: float | None, precision: int = 6) -> str:
        if value is None:
            return "n/a"
        return f"{float(value):.{precision}f}"

    lines = [
        "# Optimizer Sensitivity Stats",
        "",
        f"- Benchmark config: `{optimizer_stats.get('benchmark_config')}`",
        f"- Note: {optimizer_stats.get('budget_note')}",
        "",
        "## Adam vs Adam+LBFGS",
        "",
        "| Variant | Adam L2 | Adam+LBFGS L2 | ΔL2 | Adam Linf | Adam+LBFGS Linf | ΔLinf | Adam Loss | Adam+LBFGS Loss | ΔLoss | Adam Runtime (s) | Adam+LBFGS Runtime (s) | ΔRuntime (s) | Adam Params | Adam+LBFGS Params |",
        "| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for row in optimizer_stats.get("rows", []):
        adam = row.get("adam", {})
        lbfgs = row.get("adam_lbfgs", {})
        delta = row.get("delta_adam_lbfgs_minus_adam", {})
        lines.append(
            f"| {row.get('label')} | "
            f"{fmt(adam.get('final_l2'))} | {fmt(lbfgs.get('final_l2'))} | {fmt(delta.get('final_l2', {}).get('delta'))} | "
            f"{fmt(adam.get('final_linf'))} | {fmt(lbfgs.get('final_linf'))} | {fmt(delta.get('final_linf', {}).get('delta'))} | "
            f"{fmt(adam.get('final_loss'))} | {fmt(lbfgs.get('final_loss'))} | {fmt(delta.get('final_loss', {}).get('delta'))} | "
            f"{fmt(adam.get('runtime_sec'), precision=4)} | {fmt(lbfgs.get('runtime_sec'), precision=4)} | {fmt(delta.get('runtime_sec', {}).get('delta'), precision=4)} | "
            f"{fmt(adam.get('parameter_count'), precision=1)} | {fmt(lbfgs.get('parameter_count'), precision=1)} |"
        )

    lines += [
        "",
        "## Delta Status (Adam+LBFGS - Adam)",
        "",
        "| Variant | L2 | Linf | Loss | Runtime | Params |",
        "| --- | --- | --- | --- | --- | --- |",
    ]
    for row in optimizer_stats.get("rows", []):
        delta = row.get("delta_adam_lbfgs_minus_adam", {})
        lines.append(
            f"| {row.get('label')} | "
            f"{delta.get('final_l2', {}).get('status', 'n/a')} | "
            f"{delta.get('final_linf', {}).get('status', 'n/a')} | "
            f"{delta.get('final_loss', {}).get('status', 'n/a')} | "
            f"{delta.get('runtime_sec', {}).get('status', 'n/a')} | "
            f"{delta.get('parameter_count', {}).get('status', 'n/a')} |"
        )

    md_path.write_text("\n".join(lines) + "\n", encoding="utf-8")


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


def short_display_label(label: str) -> str:
    """Compact labels for plot readability."""
    compact = str(label)
    replacements = [
        ("TE-QPINN", "TE"),
        ("Surrogate", "Surr."),
        ("LayerNorm", "LN"),
        ("post_quantum", "postQ"),
        ("post_entanglement", "postEnt"),
        ("Classical", "Cls"),
        ("residual", "res"),
        ("memory-aware", "mem"),
        ("analytic", "ana"),
        (" + PI", "+PI"),
    ]
    for old, new in replacements:
        compact = compact.replace(old, new)
    return compact.strip()


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
            label = short_display_label(record["label"])
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
            ax.legend(frameon=False, fontsize=8)
        axes[0].set_ylabel("Value")
        plt.tight_layout()
        fig.savefig(plots_dir / f"convergence_seed_{seed}.png", dpi=300, bbox_inches="tight")
        plt.close(fig)


def plot_summary_panels(summary: Dict[str, Dict], plots_dir: Path):
    run_names = list(summary.keys())
    if not run_names:
        return

    labels = [summary[name]["label"] for name in run_names]
    short_labels = [short_display_label(label) for label in labels]
    metrics = [
        ("avg_runtime_sec", "Runtime (s)", False),
        ("avg_final_l2", "Final L2", True),
        ("avg_final_linf", "Final Linf", True),
        ("avg_parameter_count", "Parameter Count", False),
    ]

    fig_width = max(14.0, 4.0 + 2.4 * len(run_names))
    fig, axes = plt.subplots(2, 2, figsize=(fig_width, 9), constrained_layout=True)
    axes = axes.flatten()
    for ax, (metric_key, title, use_log) in zip(axes, metrics):
        means = [summary[name].get(metric_key) for name in run_names]
        std_key = metric_key.replace("avg_", "std_")
        stds = [summary[name].get(std_key, 0.0) for name in run_names]
        x_pos = np.arange(len(run_names))
        ax.bar(x_pos, means, yerr=stds, capsize=5, alpha=0.85)
        ax.set_xticks(x_pos)
        ax.set_xticklabels(short_labels, rotation=35, ha="right")
        ax.set_title(title)
        ax.grid(True, axis="y", alpha=0.25)
        if use_log:
            ax.set_yscale("log")
    fig.suptitle("TE-QPINN Benchmark Summary", fontsize=15, fontweight="bold")
    fig.savefig(plots_dir / "summary_panels.png", dpi=300, bbox_inches="tight")
    plt.close(fig)


def _is_memory_variant(run_name: str, label: str) -> bool:
    token = f"{run_name} {label}".lower()
    return "memory" in token or "mem_" in token or "_mem" in token


def build_memory_smoke_stats(records: List[Dict], summary: Dict[str, Dict]) -> Dict | None:
    """Build memory-smoke comparison stats when memory and non-memory TE runs coexist."""
    seeds = sorted({int(record["seed"]) for record in records})
    rows = []
    for run_name, row in summary.items():
        rows.append(
            {
                "run_name": run_name,
                "label": row.get("label", run_name),
                "model_type": str(row.get("model_type")),
                "final_l2": row.get("avg_final_l2"),
                "final_linf": row.get("avg_final_linf"),
                "final_loss": row.get("avg_final_loss"),
                "runtime_sec": row.get("avg_runtime_sec"),
                "parameter_count": row.get("avg_parameter_count"),
            }
        )

    te_rows = [row for row in rows if row["model_type"] == "te_qpinn_surrogate"]
    te_memory = [row for row in te_rows if _is_memory_variant(row["run_name"], row["label"])]
    te_non_memory = [row for row in te_rows if not _is_memory_variant(row["run_name"], row["label"])]
    classical_memory = [
        row
        for row in rows
        if row["model_type"] == "classical" and _is_memory_variant(row["run_name"], row["label"])
    ]
    if not te_memory or not te_non_memory:
        return None

    best_non_memory_te = min(
        te_non_memory,
        key=lambda item: float("inf") if item["final_l2"] is None else float(item["final_l2"]),
    )
    classical_memory_ref = classical_memory[0] if classical_memory else None

    def delta(payload: Dict, ref: Dict, key: str) -> float | None:
        lhs = payload.get(key)
        rhs = ref.get(key)
        if lhs is None or rhs is None:
            return None
        return float(lhs) - float(rhs)

    comparisons = []
    for row in te_memory:
        comparisons.append(
            {
                "run_name": row["run_name"],
                "label": row["label"],
                "vs_best_non_memory_te": {
                    "delta_final_l2": delta(row, best_non_memory_te, "final_l2"),
                    "delta_final_linf": delta(row, best_non_memory_te, "final_linf"),
                    "delta_final_loss": delta(row, best_non_memory_te, "final_loss"),
                    "delta_runtime_sec": delta(row, best_non_memory_te, "runtime_sec"),
                    "delta_parameter_count": delta(row, best_non_memory_te, "parameter_count"),
                },
                "vs_classical_memory": (
                    None
                    if classical_memory_ref is None
                    else {
                        "delta_final_l2": delta(row, classical_memory_ref, "final_l2"),
                        "delta_final_linf": delta(row, classical_memory_ref, "final_linf"),
                        "delta_final_loss": delta(row, classical_memory_ref, "final_loss"),
                        "delta_runtime_sec": delta(row, classical_memory_ref, "runtime_sec"),
                        "delta_parameter_count": delta(row, classical_memory_ref, "parameter_count"),
                    }
                ),
            }
        )

    return {
        "seed_scope": "single-seed" if len(seeds) == 1 else "multi-seed",
        "seed_values": seeds,
        "best_non_memory_te": best_non_memory_te,
        "classical_memory_reference": classical_memory_ref,
        "memory_te_comparisons": comparisons,
    }


def _field_meshgrid(x: np.ndarray, t: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    return np.meshgrid(x, t, indexing="ij")


def plot_field_heatmap(
    x: np.ndarray,
    t: np.ndarray,
    values: np.ndarray,
    title: str,
    output_path: Path,
    cmap: str = "viridis",
    vmin: Optional[float] = None,
    vmax: Optional[float] = None,
):
    x_grid, t_grid = _field_meshgrid(x, t)
    fig, ax = plt.subplots(figsize=(6.0, 4.6))
    im = ax.pcolormesh(x_grid, t_grid, values, cmap=cmap, shading="auto", vmin=vmin, vmax=vmax)
    ax.set_xlabel("x")
    ax.set_ylabel("t")
    ax.set_title(title)
    ax.grid(False)
    plt.colorbar(im, ax=ax)
    plt.tight_layout()
    fig.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close(fig)


def evaluate_field_data(
    model: torch.nn.Module,
    config: Dict,
    device: str,
) -> Optional[Dict]:
    """Evaluate prediction/exact/error fields and optional PDE residual field."""
    problem_cfg = config.get("problem", {})
    if str(problem_cfg.get("type", "integro_differential")).lower() != "integro_differential":
        return None

    disc_cfg = config.get("discretization", {})
    alpha = float(problem_cfg.get("alpha", 0.5))
    beta = float(problem_cfg.get("beta", 0.5))
    x_min = float(problem_cfg.get("x_min", 0.0))
    x_max = float(problem_cfg.get("x_max", 1.0))
    t_max = float(problem_cfg.get("t_max", 1.0))
    mesh_grading = float(problem_cfg.get("mesh_grading", 2.0))
    solution_cfg = problem_cfg.get("solution", {})

    n_x = int(disc_cfg.get("N_x", 100))
    n_t = int(disc_cfg.get("N_t", 100))
    n_quad = int(disc_cfg.get("N_integral_quad", 20))

    x = torch.linspace(x_min, x_max, n_x, dtype=torch.float64, device=device)
    mesh = GradedMesh(N=n_t, t_max=t_max, beta=mesh_grading, device=device)
    t_nodes = mesh.get_nodes()
    if t_nodes.numel() <= 1:
        return None
    # Domain for this benchmark is t in (0, 1], so we skip t=0.
    t = t_nodes[1:]

    x_grid, t_grid = torch.meshgrid(x, t, indexing="ij")
    model.eval()
    with torch.no_grad():
        u_pred = model(x_grid.flatten(), t_grid.flatten()).reshape(x_grid.shape)
        u_exact = integro_exact_solution(x_grid, t_grid, alpha, solution_cfg)
        abs_error = torch.abs(u_pred - u_exact)

    residual = None
    x_res = None
    t_res = None
    try:
        l1_coeffs = L1Coefficients(mesh, alpha, device=device)
        residual_eval = IntegroDifferentialResidual(
            model,
            mesh,
            l1_coeffs,
            alpha=alpha,
            beta=beta,
            solution_cfg=solution_cfg,
            n_quad=n_quad,
            device=device,
            history_gradient_mode=str(problem_cfg.get("history_gradient_mode", "full")),
        )

        # Keep residual plotting affordable for larger grids.
        n_x_res = min(n_x, 40)
        n_t_res = min(n_t, 40)
        x_idx = np.unique(np.linspace(0, n_x - 1, n_x_res, dtype=int))
        n_idx = np.unique(np.linspace(1, n_t, n_t_res, dtype=int))

        x_res_t = x[x_idx]
        n_idx_t = torch.tensor(n_idx, dtype=torch.long, device=device)
        t_res_t = t_nodes[n_idx_t]

        x_res_grid, t_res_grid = torch.meshgrid(x_res_t, t_res_t, indexing="ij")
        n_grid = n_idx_t.unsqueeze(0).expand(len(x_idx), -1).reshape(-1)
        residual = residual_eval.compute(
            x_res_grid.flatten(),
            t_res_grid.flatten(),
            n_grid,
        ).reshape(x_res_grid.shape).detach()
        x_res = x_res_t.detach().cpu().numpy()
        t_res = t_res_t.detach().cpu().numpy()
    except Exception as exc:
        print(f"[warn] Residual field evaluation skipped: {exc}")

    return {
        "x": x.detach().cpu().numpy(),
        "t": t.detach().cpu().numpy(),
        "u_pred": u_pred.detach().cpu().numpy(),
        "u_exact": u_exact.detach().cpu().numpy(),
        "abs_error": abs_error.detach().cpu().numpy(),
        "x_res": x_res,
        "t_res": t_res,
        "residual": None if residual is None else residual.detach().cpu().numpy(),
    }


def plot_time_slices(
    x: np.ndarray,
    t: np.ndarray,
    u_pred: np.ndarray,
    u_exact: np.ndarray,
    output_path: Path,
):
    target_times = [0.25, 0.50, 0.75, 1.00]
    fig, axes = plt.subplots(2, 2, figsize=(10.5, 7.2), sharex=True)
    axes = axes.flatten()

    for ax, target_t in zip(axes, target_times):
        idx = int(np.argmin(np.abs(t - target_t)))
        t_actual = float(t[idx])
        ax.plot(x, u_exact[:, idx], linewidth=2.0, label="Exact")
        ax.plot(x, u_pred[:, idx], linewidth=2.0, linestyle="--", label="Predicted")
        ax.set_title(f"t = {t_actual:.3f}")
        ax.set_xlabel("x")
        ax.set_ylabel("u(x,t)")
        ax.grid(True, alpha=0.25)
    axes[0].legend(frameon=False)
    plt.tight_layout()
    fig.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close(fig)


def save_field_plots_for_record(record: Dict, field_data: Dict, plots_dir: Path) -> Dict[str, str]:
    """Save per-run field plots and return generated artifact paths."""
    seed = int(record["seed"])
    run_name = str(record["run_name"])
    label = str(record["label"])
    prefix = f"seed_{seed}_{run_name}"

    x = field_data["x"]
    t = field_data["t"]
    u_pred = field_data["u_pred"]
    u_exact = field_data["u_exact"]
    abs_error = field_data["abs_error"]

    paths = {}
    pred_path = plots_dir / f"{prefix}_u_pred_heatmap.png"
    exact_path = plots_dir / f"{prefix}_u_exact_heatmap.png"
    abs_err_path = plots_dir / f"{prefix}_abs_error_heatmap.png"
    slice_path = plots_dir / f"{prefix}_line_slices.png"
    npz_path = Path(record["run_dir"]) / "field_eval.npz"

    plot_field_heatmap(
        x=x,
        t=t,
        values=u_pred,
        title=f"{label}: Predicted Field",
        output_path=pred_path,
        cmap="viridis",
    )
    plot_field_heatmap(
        x=x,
        t=t,
        values=u_exact,
        title=f"{label}: Exact Field",
        output_path=exact_path,
        cmap="viridis",
    )
    plot_field_heatmap(
        x=x,
        t=t,
        values=abs_error,
        title=f"{label}: |u_pred - u_exact|",
        output_path=abs_err_path,
        cmap="hot",
    )
    plot_time_slices(x, t, u_pred, u_exact, slice_path)

    paths["u_pred_heatmap"] = str(pred_path)
    paths["u_exact_heatmap"] = str(exact_path)
    paths["abs_error_heatmap"] = str(abs_err_path)
    paths["line_slices"] = str(slice_path)

    residual = field_data.get("residual")
    x_res = field_data.get("x_res")
    t_res = field_data.get("t_res")
    if residual is not None and x_res is not None and t_res is not None:
        residual_path = plots_dir / f"{prefix}_pde_residual_heatmap.png"
        residual_abs_max = float(np.max(np.abs(residual)))
        plot_field_heatmap(
            x=x_res,
            t=t_res,
            values=residual,
            title=f"{label}: PDE Residual",
            output_path=residual_path,
            cmap="RdBu_r",
            vmin=-residual_abs_max,
            vmax=residual_abs_max,
        )
        paths["pde_residual_heatmap"] = str(residual_path)

    np.savez(
        npz_path,
        x=x,
        t=t,
        u_pred=u_pred,
        u_exact=u_exact,
        abs_error=abs_error,
    )
    paths["field_eval_npz"] = str(npz_path)
    return paths


def plot_seedwise_error_heatmaps(records: List[Dict], plots_dir: Path):
    """Plot side-by-side absolute-error heatmaps for classical vs TE surrogate."""
    by_seed: Dict[int, List[Dict]] = {}
    for record in records:
        by_seed.setdefault(int(record["seed"]), []).append(record)

    for seed, seed_records in by_seed.items():
        classical = next((r for r in seed_records if str(r.get("model_type")) == "classical"), None)
        te_candidates = [
            r for r in seed_records if str(r.get("model_type")) == "te_qpinn_surrogate"
        ]
        te_surrogate = None
        if te_candidates:
            te_surrogate = min(
                te_candidates,
                key=lambda row: float("inf")
                if row.get("final_l2") is None
                else float(row.get("final_l2")),
            )
        if not classical or not te_surrogate:
            continue

        classical_npz = Path(str(classical.get("field_eval_npz", "")))
        te_npz = Path(str(te_surrogate.get("field_eval_npz", "")))
        if not classical_npz.exists() or not te_npz.exists():
            continue

        with np.load(classical_npz, allow_pickle=False) as cls_data:
            x_cls = cls_data["x"]
            t_cls = cls_data["t"]
            err_cls = cls_data["abs_error"]
        with np.load(te_npz, allow_pickle=False) as te_data:
            x_te = te_data["x"]
            t_te = te_data["t"]
            err_te = te_data["abs_error"]

        vmax = max(float(np.max(err_cls)), float(np.max(err_te)))
        fig, axes = plt.subplots(1, 2, figsize=(12.6, 4.8), sharey=True)

        x_grid_cls, t_grid_cls = _field_meshgrid(x_cls, t_cls)
        im0 = axes[0].pcolormesh(
            x_grid_cls,
            t_grid_cls,
            err_cls,
            cmap="hot",
            shading="auto",
            vmin=0.0,
            vmax=vmax,
        )
        axes[0].set_title("Classical + PI |Error|")
        axes[0].set_xlabel("x")
        axes[0].set_ylabel("t")

        x_grid_te, t_grid_te = _field_meshgrid(x_te, t_te)
        im1 = axes[1].pcolormesh(
            x_grid_te,
            t_grid_te,
            err_te,
            cmap="hot",
            shading="auto",
            vmin=0.0,
            vmax=vmax,
        )
        axes[1].set_title(f"{te_surrogate.get('label', 'TE-QPINN')} |Error|")
        axes[1].set_xlabel("x")
        axes[1].set_ylabel("t")

        cbar = fig.colorbar(im1, ax=axes.ravel().tolist(), shrink=0.92)
        cbar.set_label("|u_pred - u_exact|")
        fig.subplots_adjust(wspace=0.18, right=0.9)
        fig.savefig(
            plots_dir / f"error_heatmaps_seed_{seed}_classical_vs_te_qpinn.png",
            dpi=300,
            bbox_inches="tight",
        )
        plt.close(fig)


def write_markdown_report(
    records: List[Dict],
    summary: Dict[str, Dict],
    output_path: Path,
    benchmark_config_path: str,
    ablation_stats: Dict | None = None,
    multiseed_stats: Dict | None = None,
    optimizer_sensitivity_stats: Dict | None = None,
    memory_smoke_stats: Dict | None = None,
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
        "| Variant | Model Type | Runs | Runtime (s) | Params | Final Loss | Final L2 | Final Linf |",
        "| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for run_name in run_names:
        row = summary[run_name]
        lines.append(
            f"| {row['label']} | {row['model_type']} | {row['num_runs']} | "
            f"{fmt(row['avg_runtime_sec'])} ± {fmt(row.get('std_runtime_sec', 0.0))} | "
            f"{fmt(row['avg_parameter_count'], precision=1)} | "
            f"{fmt(row.get('avg_final_loss'))} ± {fmt(row.get('std_final_loss', 0.0))} | "
            f"{fmt(row['avg_final_l2'])} ± {fmt(row.get('std_final_l2', 0.0))} | "
            f"{fmt(row['avg_final_linf'])} ± {fmt(row.get('std_final_linf', 0.0))} |"
        )

    if ablation_stats is not None:
        lines += [
            "",
            "## Ablation Comparison",
            "",
            "| Variant | L2 vs Full TE | Linf vs Full TE | Loss vs Full TE | Runtime vs Full TE | L2 vs Classical | Linf vs Classical | Loss vs Classical | Runtime vs Classical |",
            "| --- | --- | --- | --- | --- | --- | --- | --- | --- |",
        ]
        for row in ablation_stats.get("rows", []):
            full = row.get("vs_full_te", {})
            cls = row.get("vs_classical", {})
            lines.append(
                f"| {row.get('label')} | "
                f"{full.get('avg_final_l2', {}).get('status', 'n/a')} | "
                f"{full.get('avg_final_linf', {}).get('status', 'n/a')} | "
                f"{full.get('avg_final_loss', {}).get('status', 'n/a')} | "
                f"{full.get('avg_runtime_sec', {}).get('status', 'n/a')} | "
                f"{cls.get('avg_final_l2', {}).get('status', 'n/a')} | "
                f"{cls.get('avg_final_linf', {}).get('status', 'n/a')} | "
                f"{cls.get('avg_final_loss', {}).get('status', 'n/a')} | "
                f"{cls.get('avg_runtime_sec', {}).get('status', 'n/a')} |"
            )

    if multiseed_stats is not None:
        lines += [
            "",
            "## Multi-Seed Win Counts",
            "",
            "| Variant | L2 wins vs Classical | Linf wins vs Classical | L2 wins vs Full TE | Linf wins vs Full TE |",
            "| --- | ---: | ---: | ---: | ---: |",
        ]

        for row in multiseed_stats.get("runs", []):
            wins = row.get("wins", {})
            cls_l2 = wins.get("vs_classical", {}).get("final_l2", {})
            cls_linf = wins.get("vs_classical", {}).get("final_linf", {})
            te_l2 = wins.get("vs_full_te", {}).get("final_l2", {})
            te_linf = wins.get("vs_full_te", {}).get("final_linf", {})

            def fmt_wins(payload: Dict) -> str:
                if not payload:
                    return "n/a"
                return f"{payload.get('wins', 0)} / {payload.get('paired_seeds', 0)}"

            lines.append(
                f"| {row.get('label')} | "
                f"{fmt_wins(cls_l2)} | "
                f"{fmt_wins(cls_linf)} | "
                f"{fmt_wins(te_l2)} | "
                f"{fmt_wins(te_linf)} |"
            )

    if optimizer_sensitivity_stats is not None:
        lines += [
            "",
            "## Optimizer Sensitivity (Adam vs Adam+LBFGS)",
            "",
            f"- Note: {optimizer_sensitivity_stats.get('budget_note')}",
            "",
            "| Variant | Adam L2 | Adam+LBFGS L2 | Adam Linf | Adam+LBFGS Linf | Adam Loss | Adam+LBFGS Loss | Adam Runtime (s) | Adam+LBFGS Runtime (s) |",
            "| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
        ]
        for row in optimizer_sensitivity_stats.get("rows", []):
            adam = row.get("adam", {})
            lbfgs = row.get("adam_lbfgs", {})
            lines.append(
                f"| {row.get('label')} | "
                f"{fmt(adam.get('final_l2'))} | {fmt(lbfgs.get('final_l2'))} | "
                f"{fmt(adam.get('final_linf'))} | {fmt(lbfgs.get('final_linf'))} | "
                f"{fmt(adam.get('final_loss'))} | {fmt(lbfgs.get('final_loss'))} | "
                f"{fmt(adam.get('runtime_sec'))} | {fmt(lbfgs.get('runtime_sec'))} |"
            )

    if memory_smoke_stats is not None:
        best_non_memory = memory_smoke_stats.get("best_non_memory_te", {})
        classical_memory = memory_smoke_stats.get("classical_memory_reference")
        lines += [
            "",
            "## Memory Smoke Comparison",
            "",
            f"- Scope: {memory_smoke_stats.get('seed_scope')} ({memory_smoke_stats.get('seed_values')})",
            (
                f"- Best non-memory TE reference: **{best_non_memory.get('label')}** "
                f"(L2={fmt(best_non_memory.get('final_l2'))}, "
                f"Linf={fmt(best_non_memory.get('final_linf'))}, "
                f"Loss={fmt(best_non_memory.get('final_loss'))})"
            ),
            (
                "- Classical memory-feature control reference: "
                f"**{classical_memory.get('label')}**" if classical_memory is not None else
                "- Classical memory-feature control reference: n/a"
            ),
            "",
            "| Memory Variant | Final L2 | Final Linf | Final Loss | Runtime (s) | Params | ΔL2 vs Best Non-memory TE | ΔLinf vs Best Non-memory TE | ΔL2 vs Classical Memory | ΔLinf vs Classical Memory |",
            "| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
        ]
        for row in memory_smoke_stats.get("memory_te_comparisons", []):
            summary_row = summary.get(row.get("run_name", ""), {})
            vs_best = row.get("vs_best_non_memory_te", {})
            vs_classical = row.get("vs_classical_memory", {}) or {}
            lines.append(
                f"| {row.get('label')} | "
                f"{fmt(summary_row.get('avg_final_l2'))} | "
                f"{fmt(summary_row.get('avg_final_linf'))} | "
                f"{fmt(summary_row.get('avg_final_loss'))} | "
                f"{fmt(summary_row.get('avg_runtime_sec'))} | "
                f"{fmt(summary_row.get('avg_parameter_count'), precision=1)} | "
                f"{fmt(vs_best.get('delta_final_l2'))} | "
                f"{fmt(vs_best.get('delta_final_linf'))} | "
                f"{fmt(vs_classical.get('delta_final_l2'))} | "
                f"{fmt(vs_classical.get('delta_final_linf'))} |"
            )
        lines += [
            "",
            "> Note: this memory section is smoke-level and should not be interpreted as multi-seed evidence.",
        ]

    lines += [
        "",
        "## Artifacts",
        "",
        "- `summary.csv`",
        "- `summary.json`",
        "- `summary_panels.png`",
        "- `convergence_seed_<seed>.png`",
        "- `seed_<seed>_<run>_u_pred_heatmap.png`",
        "- `seed_<seed>_<run>_u_exact_heatmap.png`",
        "- `seed_<seed>_<run>_abs_error_heatmap.png`",
        "- `seed_<seed>_<run>_pde_residual_heatmap.png` (if residual evaluation succeeds)",
        "- `seed_<seed>_<run>_line_slices.png`",
        "- `error_heatmaps_seed_<seed>_classical_vs_te_qpinn.png`",
    ]
    if ablation_stats is not None:
        lines += [
            "- `ablation_stats.json`",
            "- `ablation_stats.md`",
        ]
    if multiseed_stats is not None:
        lines += [
            "- `multiseed_stats.json`",
            "- `multiseed_stats.md`",
        ]
    if optimizer_sensitivity_stats is not None:
        lines += [
            "- `optimizer_sensitivity_stats.json`",
            "- `optimizer_sensitivity_stats.md`",
        ]
    output_path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def run_benchmark_case(
    run_name: str,
    run_label: str,
    config_source: Path,
    seed: int,
    output_dir: Path,
    plots_dir: Path,
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

    model, history = train(
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

    final_loss = history.get("final_loss")
    if final_loss is None and history.get("loss"):
        final_loss = history["loss"][-1]

    final_l2 = history.get("final_l2")
    if final_l2 is None and history.get("l2"):
        final_l2 = history["l2"][-1]

    final_linf = history.get("final_linf")
    if final_linf is None and history.get("linf"):
        final_linf = history["linf"][-1]

    best_l2, best_l2_epoch = best_history_value(history, "l2")
    best_linf, best_linf_epoch = best_history_value(history, "linf")
    best_loss, best_loss_epoch = best_history_value(history, "loss")
    final_epoch = history.get("final_epoch")
    if final_epoch is None:
        final_epoch = history["epochs"][-1] if history.get("epochs") else int(config["training"]["epochs"])

    if final_l2 is not None and (best_l2 is None or float(final_l2) < float(best_l2)):
        best_l2 = float(final_l2)
        best_l2_epoch = int(final_epoch)
    if final_linf is not None and (best_linf is None or float(final_linf) < float(best_linf)):
        best_linf = float(final_linf)
        best_linf_epoch = int(final_epoch)
    if final_loss is not None and (best_loss is None or float(final_loss) < float(best_loss)):
        best_loss = float(final_loss)
        best_loss_epoch = int(final_epoch)

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

    # Field-level evaluation plots for the configured PDE benchmark.
    try:
        field_data = evaluate_field_data(model=model, config=config, device=resolved_device)
        if field_data is not None:
            field_paths = save_field_plots_for_record(record, field_data, plots_dir)
            record.update(field_paths)
    except Exception as exc:
        print(f"[warn] Field-level plotting skipped for run={run_name}, seed={seed}: {exc}")

    with open(run_dir / "metrics.json", "w", encoding="utf-8") as handle:
        json.dump(record, handle, indent=2)
    return record


def load_existing_record(output_dir: Path, seed: int, run_name: str) -> Dict | None:
    """Load previously saved metrics for a run if available."""
    metrics_path = output_dir / f"seed_{seed}" / run_name / "metrics.json"
    if not metrics_path.exists():
        return None
    with open(metrics_path, "r", encoding="utf-8") as handle:
        record = json.load(handle)
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
        seed = int(item["seed"])

        if args.resume_incomplete:
            existing = load_existing_record(output_dir=output_dir, seed=seed, run_name=run_name)
            if existing is not None:
                print("-" * 80)
                print(f"Reusing existing metrics for seed={seed} run={run_name}")
                records.append(existing)
                continue

        source_cfg_path = Path(item["config"])
        if not source_cfg_path.is_absolute():
            source_cfg_path = (root_dir / source_cfg_path).resolve()

        print("-" * 80)
        print(f"Running seed={seed} run={run_name}")
        record = run_benchmark_case(
            run_name=run_name,
            run_label=run_label,
            config_source=source_cfg_path,
            seed=seed,
            output_dir=output_dir,
            plots_dir=plots_dir,
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
    ablation_stats = build_ablation_stats(summary, str(benchmark_config_path))
    if ablation_stats is not None:
        write_ablation_stats(ablation_stats, output_dir)
    multiseed_stats = build_multiseed_stats(records, summary, str(benchmark_config_path))
    if multiseed_stats is not None:
        write_multiseed_stats(multiseed_stats, output_dir)
    optimizer_sensitivity_stats = build_optimizer_sensitivity_stats(summary, str(benchmark_config_path))
    if optimizer_sensitivity_stats is not None:
        write_optimizer_sensitivity_stats(optimizer_sensitivity_stats, output_dir)
    memory_smoke_stats = build_memory_smoke_stats(records, summary)

    write_summary_csv(records, summary_csv_path)
    summary_payload = {
        "benchmark_config": str(benchmark_config_path),
        "seeds": seeds,
        "records": records,
        "aggregate": summary,
    }
    if ablation_stats is not None:
        summary_payload["ablation_stats"] = ablation_stats
    if multiseed_stats is not None:
        summary_payload["multiseed_stats"] = multiseed_stats
    if optimizer_sensitivity_stats is not None:
        summary_payload["optimizer_sensitivity_stats"] = optimizer_sensitivity_stats
    if memory_smoke_stats is not None:
        summary_payload["memory_smoke_stats"] = memory_smoke_stats
    with open(summary_json_path, "w", encoding="utf-8") as handle:
        json.dump(summary_payload, handle, indent=2)

    plot_seedwise_convergence(records, plots_dir)
    plot_summary_panels(summary, plots_dir)
    plot_seedwise_error_heatmaps(records, plots_dir)
    write_markdown_report(
        records,
        summary,
        report_path,
        str(benchmark_config_path),
        ablation_stats=ablation_stats,
        multiseed_stats=multiseed_stats,
        optimizer_sensitivity_stats=optimizer_sensitivity_stats,
        memory_smoke_stats=memory_smoke_stats,
    )

    print("=" * 80)
    print("Benchmark complete")
    print(f"summary.csv: {summary_csv_path}")
    print(f"summary.json: {summary_json_path}")
    print(f"plots: {plots_dir}")
    print(f"report: {report_path}")
    print("=" * 80)


if __name__ == "__main__":
    main()
