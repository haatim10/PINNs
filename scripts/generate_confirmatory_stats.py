#!/usr/bin/env python3
"""Generate richer confirmatory stats from benchmark summary.csv.

This is a post-processing helper for locked benchmark outputs.
It does not modify training, models, or physics definitions.
"""

from __future__ import annotations

import argparse
import csv
import json
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
from scipy import stats


LOWER_BETTER_METRICS = ("final_l2", "final_linf", "final_loss", "runtime_sec")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate confirmatory stats from summary.csv")
    parser.add_argument("--summary-csv", required=True, help="Path to benchmark summary.csv")
    parser.add_argument("--output-json", required=True, help="Path to write confirmatory stats json")
    parser.add_argument("--output-md", required=True, help="Path to write confirmatory stats markdown")
    parser.add_argument(
        "--bootstrap-samples",
        type=int,
        default=10000,
        help="Bootstrap samples for paired mean-difference CI",
    )
    parser.add_argument("--seed", type=int, default=0, help="Bootstrap RNG seed")
    return parser.parse_args()


def _as_float(value: str) -> float:
    return float(value) if value not in ("", "None", None) else float("nan")


def load_rows(path: Path) -> List[Dict]:
    with path.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        rows = [dict(row) for row in reader]
    if not rows:
        raise ValueError(f"No rows found in {path}")
    return rows


def group_by_run(rows: List[Dict]) -> Dict[str, List[Dict]]:
    grouped: Dict[str, List[Dict]] = defaultdict(list)
    for row in rows:
        grouped[row["run_name"]].append(row)
    return grouped


def summarize_run(rows: List[Dict]) -> Dict:
    def arr(key: str) -> np.ndarray:
        return np.array([_as_float(r[key]) for r in rows], dtype=float)

    l2 = arr("final_l2")
    linf = arr("final_linf")
    loss = arr("final_loss")
    runtime = arr("runtime_sec")
    params = arr("parameter_count")

    return {
        "label": rows[0]["label"],
        "model_type": rows[0]["model_type"],
        "seeds": [int(r["seed"]) for r in rows],
        "n_runs": len(rows),
        "final_l2": {
            "mean": float(np.mean(l2)),
            "std": float(np.std(l2, ddof=1)) if len(l2) > 1 else 0.0,
            "median": float(np.median(l2)),
            "best": float(np.min(l2)),
            "worst": float(np.max(l2)),
        },
        "final_linf": {
            "mean": float(np.mean(linf)),
            "std": float(np.std(linf, ddof=1)) if len(linf) > 1 else 0.0,
            "median": float(np.median(linf)),
            "best": float(np.min(linf)),
            "worst": float(np.max(linf)),
        },
        "final_loss": {
            "mean": float(np.mean(loss)),
            "std": float(np.std(loss, ddof=1)) if len(loss) > 1 else 0.0,
            "median": float(np.median(loss)),
            "best": float(np.min(loss)),
            "worst": float(np.max(loss)),
        },
        "runtime_sec": {
            "mean": float(np.mean(runtime)),
            "std": float(np.std(runtime, ddof=1)) if len(runtime) > 1 else 0.0,
            "median": float(np.median(runtime)),
            "best": float(np.min(runtime)),
            "worst": float(np.max(runtime)),
        },
        "parameter_count": float(np.mean(params)),
        # Lower is better: mean L2 * mean runtime
        "accuracy_per_runtime_l2_runtime_product": float(np.mean(l2) * np.mean(runtime)),
    }


def _paired_dict(
    rows_a: List[Dict],
    rows_b: List[Dict],
    metric: str,
) -> Tuple[np.ndarray, np.ndarray, List[int]]:
    a_by_seed = {int(r["seed"]): _as_float(r[metric]) for r in rows_a}
    b_by_seed = {int(r["seed"]): _as_float(r[metric]) for r in rows_b}
    seeds = sorted(set(a_by_seed) & set(b_by_seed))
    a_vals = np.array([a_by_seed[s] for s in seeds], dtype=float)
    b_vals = np.array([b_by_seed[s] for s in seeds], dtype=float)
    return a_vals, b_vals, seeds


def win_count(rows_a: List[Dict], rows_b: List[Dict], metric: str) -> Dict:
    a_vals, b_vals, seeds = _paired_dict(rows_a, rows_b, metric)
    wins = int(np.sum(a_vals < b_vals))
    ties = int(np.sum(np.isclose(a_vals, b_vals)))
    losses = int(np.sum(a_vals > b_vals))
    return {
        "wins": wins,
        "ties": ties,
        "losses": losses,
        "paired_seeds": len(seeds),
        "seed_ids": seeds,
    }


def paired_wilcoxon(a_vals: np.ndarray, b_vals: np.ndarray) -> Dict:
    # lower is better; H1: run_a < run_b
    diff = a_vals - b_vals
    if len(diff) < 2:
        return {"statistic": None, "pvalue_two_sided": None, "pvalue_less": None}
    two_sided = stats.wilcoxon(a_vals, b_vals, alternative="two-sided", zero_method="wilcox")
    less = stats.wilcoxon(a_vals, b_vals, alternative="less", zero_method="wilcox")
    return {
        "statistic": float(two_sided.statistic),
        "pvalue_two_sided": float(two_sided.pvalue),
        "pvalue_less": float(less.pvalue),
    }


def bootstrap_mean_diff_ci(
    diff: np.ndarray,
    samples: int,
    rng: np.random.Generator,
    alpha: float = 0.05,
) -> Dict:
    if len(diff) == 0:
        return {"mean_diff": None, "ci_low": None, "ci_high": None}
    boot_means = np.empty(samples, dtype=float)
    n = len(diff)
    for i in range(samples):
        idx = rng.integers(0, n, size=n)
        boot_means[i] = float(np.mean(diff[idx]))
    low = float(np.quantile(boot_means, alpha / 2))
    high = float(np.quantile(boot_means, 1 - alpha / 2))
    return {
        "mean_diff": float(np.mean(diff)),
        "ci_low": low,
        "ci_high": high,
    }


def paired_cohens_d(diff: np.ndarray) -> float | None:
    if len(diff) < 2:
        return None
    sd = float(np.std(diff, ddof=1))
    if np.isclose(sd, 0.0):
        return None
    return float(np.mean(diff) / sd)


def build_comparison(
    grouped: Dict[str, List[Dict]],
    run_a: str,
    run_b: str,
    bootstrap_samples: int,
    rng: np.random.Generator,
) -> Dict:
    rows_a = grouped[run_a]
    rows_b = grouped[run_b]
    l2_a, l2_b, seeds = _paired_dict(rows_a, rows_b, "final_l2")
    linf_a, linf_b, _ = _paired_dict(rows_a, rows_b, "final_linf")
    loss_a, loss_b, _ = _paired_dict(rows_a, rows_b, "final_loss")
    rt_a, rt_b, _ = _paired_dict(rows_a, rows_b, "runtime_sec")

    # lower is better, so negative diff means run_a better
    diff_l2 = l2_a - l2_b
    diff_linf = linf_a - linf_b
    diff_loss = loss_a - loss_b
    diff_rt = rt_a - rt_b

    return {
        "run_a": run_a,
        "label_a": rows_a[0]["label"],
        "run_b": run_b,
        "label_b": rows_b[0]["label"],
        "paired_seeds": seeds,
        "win_counts": {
            "final_l2": win_count(rows_a, rows_b, "final_l2"),
            "final_linf": win_count(rows_a, rows_b, "final_linf"),
        },
        "paired_differences": {
            "final_l2_mean_diff_a_minus_b": float(np.mean(diff_l2)),
            "final_linf_mean_diff_a_minus_b": float(np.mean(diff_linf)),
            "final_loss_mean_diff_a_minus_b": float(np.mean(diff_loss)),
            "runtime_sec_mean_diff_a_minus_b": float(np.mean(diff_rt)),
        },
        "wilcoxon": {
            "final_l2": paired_wilcoxon(l2_a, l2_b),
            "final_linf": paired_wilcoxon(linf_a, linf_b),
        },
        "bootstrap_95ci": {
            "final_l2_mean_diff_a_minus_b": bootstrap_mean_diff_ci(diff_l2, bootstrap_samples, rng, alpha=0.05),
            "final_linf_mean_diff_a_minus_b": bootstrap_mean_diff_ci(diff_linf, bootstrap_samples, rng, alpha=0.05),
        },
        "effect_size": {
            "paired_cohens_d_final_l2": paired_cohens_d(diff_l2),
            "paired_cohens_d_final_linf": paired_cohens_d(diff_linf),
        },
    }


def to_md(payload: Dict) -> str:
    lines: List[str] = []
    lines.append("# Confirmatory 10-Seed Stats")
    lines.append("")
    lines.append(f"- Summary source: `{payload['summary_csv']}`")
    lines.append(f"- Seeds: {payload['seeds']}")
    lines.append(f"- Primary endpoint: `{payload['primary_endpoint']}`")
    lines.append("")
    lines.append("## Per-Variant Metrics")
    lines.append("")
    lines.append(
        "| Variant | L2 mean±std | L2 median | L2 best | L2 worst | Linf mean±std | Linf median | Linf best | Linf worst | Loss mean±std | Runtime mean±std (s) | Params | Accuracy/Runtime (L2*Runtime) |"
    )
    lines.append("| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |")
    for run_name, row in payload["runs"].items():
        l2 = row["final_l2"]
        linf = row["final_linf"]
        loss = row["final_loss"]
        rt = row["runtime_sec"]
        lines.append(
            f"| {row['label']} | "
            f"{l2['mean']:.6f} ± {l2['std']:.6f} | {l2['median']:.6f} | {l2['best']:.6f} | {l2['worst']:.6f} | "
            f"{linf['mean']:.6f} ± {linf['std']:.6f} | {linf['median']:.6f} | {linf['best']:.6f} | {linf['worst']:.6f} | "
            f"{loss['mean']:.6f} ± {loss['std']:.6f} | {rt['mean']:.4f} ± {rt['std']:.4f} | {row['parameter_count']:.0f} | "
            f"{row['accuracy_per_runtime_l2_runtime_product']:.6f} |"
        )
    lines.append("")
    lines.append("## Requested Win Counts + Paired Tests")
    lines.append("")
    for comp in payload["comparisons"]:
        lines.append(f"### {comp['label_a']} vs {comp['label_b']}")
        l2w = comp["win_counts"]["final_l2"]
        linfw = comp["win_counts"]["final_linf"]
        lines.append(
            f"- L2 wins: {l2w['wins']} / {l2w['paired_seeds']} (ties {l2w['ties']}, losses {l2w['losses']})"
        )
        lines.append(
            f"- Linf wins: {linfw['wins']} / {linfw['paired_seeds']} (ties {linfw['ties']}, losses {linfw['losses']})"
        )
        w_l2 = comp["wilcoxon"]["final_l2"]
        ci_l2 = comp["bootstrap_95ci"]["final_l2_mean_diff_a_minus_b"]
        d_l2 = comp["effect_size"]["paired_cohens_d_final_l2"]
        lines.append(
            f"- Wilcoxon L2 p(two-sided)={w_l2['pvalue_two_sided']:.6g}, p(a<b)={w_l2['pvalue_less']:.6g}; "
            f"mean diff(a-b)={ci_l2['mean_diff']:.6f}, 95% CI [{ci_l2['ci_low']:.6f}, {ci_l2['ci_high']:.6f}], "
            f"paired d={d_l2 if d_l2 is not None else 'n/a'}"
        )
        lines.append("")
    return "\n".join(lines) + "\n"


def main() -> None:
    args = parse_args()
    summary_csv = Path(args.summary_csv).resolve()
    output_json = Path(args.output_json).resolve()
    output_md = Path(args.output_md).resolve()
    output_json.parent.mkdir(parents=True, exist_ok=True)
    output_md.parent.mkdir(parents=True, exist_ok=True)

    rows = load_rows(summary_csv)
    grouped = group_by_run(rows)

    requested_runs = [
        "classical_pi",
        "classical_memory_pi",
        "te_fixed_pi",
        "te_layernorm_post_quantum_pi",
        "te_memory_analytic_pi",
    ]
    missing = [r for r in requested_runs if r not in grouped]
    if missing:
        raise ValueError(f"Missing required runs in summary.csv: {missing}")

    run_stats = {run_name: summarize_run(grouped[run_name]) for run_name in requested_runs}
    all_seeds = sorted({int(row["seed"]) for row in rows})

    rng = np.random.default_rng(args.seed)
    pair_defs = [
        ("te_memory_analytic_pi", "te_fixed_pi"),
        ("te_memory_analytic_pi", "te_layernorm_post_quantum_pi"),
        ("te_memory_analytic_pi", "classical_pi"),
        ("te_memory_analytic_pi", "classical_memory_pi"),
        ("classical_memory_pi", "classical_pi"),
    ]
    comparisons = [
        build_comparison(grouped, a, b, bootstrap_samples=args.bootstrap_samples, rng=rng)
        for a, b in pair_defs
    ]

    payload = {
        "summary_csv": str(summary_csv),
        "seeds": all_seeds,
        "primary_endpoint": "mean final L2 under Adam-only 10-seed validation",
        "runs": run_stats,
        "comparisons": comparisons,
    }

    with output_json.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2)

    with output_md.open("w", encoding="utf-8") as handle:
        handle.write(to_md(payload))

    print(f"Wrote {output_json}")
    print(f"Wrote {output_md}")


if __name__ == "__main__":
    main()
